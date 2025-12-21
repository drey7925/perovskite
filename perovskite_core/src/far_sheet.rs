//! Shared helpers for geometry of far sheets, encoded as packed grids in the protocol.
//!
//! Primarily includes basis vector, bounds, and similar. Note that substantial logic is
//! kept in `perovskite_client`; this focuses specifically on those things that both sides
//! need to share.
use crate::protocol::map::FarSheetControl as FarSheetControlProto;
use anyhow::{bail, ensure, Context, Result};
use cgmath::{vec3, Vector2, Vector3};
use itertools::Itertools;

pub use crate::protocol::map::TessellationMode;

/// The primitive topology that should be used with an index buffer built by this module.
///
/// This is implemented as a separate enum to avoid pulling a dependency on vulkan for
/// perovskite_core.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexPrimitiveTopology {
    /// Use VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, disable primitive restart.
    VkTriangleList,
    /// Use VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP with primitive restart enabled.
    VkTriangleStrip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// The front face to use for an index buffer. Note that this comes with some caveats -
/// first, it's up to the user to decide whether they care about the top or the bottom,
/// and second, the basis vectors affect the effective winding order in world space.
pub enum FrontFace {
    /// Counter-clockwise
    Ccw,
    /// Clockwise
    Cw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Opaque key that determines an index buffer for a sheet. Suitable as a hash map key -
/// the index buffer is completely determined by the fields of this struct.
pub struct IndexBufferKey {
    tess: TessellationMode,
    m: usize,
    n: usize,
    k: isize,
}
impl IndexBufferKey {
    const U32_RESTART_INDEX: u32 = u32::MAX;

    /// Builds an index buffer for the sheet with this key.
    ///
    /// The current implementation is not highly optimized, good enough for
    /// a handful of builds during startup, but would become a bottleneck if repeatedly
    /// called in the critical path of rendering every sheet.
    ///
    /// Caller should really be caching this based on the key.
    pub fn build(&self) -> Vec<u32> {
        match self.tess {
            TessellationMode::Invalid => panic!("Invalid tessellation mode"),
            TessellationMode::NegDiagonal => {
                // This is really an area problem
                let triangles_est = 2 * self.m as isize * self.n as isize
                    + (self.m as isize * self.k * self.n as isize);
                let triangles_est = triangles_est as usize;
                let num_strips = self.n - 1;
                // Each strip requires a restart flag. We also need one index per triangle
                // but also two more indices per strip to start/end the strip.
                let indices_est = triangles_est + (num_strips * 3);
                let mut indices = Vec::with_capacity(indices_est);

                let m = self.m as isize;
                let n = self.n as isize;
                let k = self.k as isize;

                for i in 0..=(n - 1) {
                    // The start index is a function of the preceding `i` strips
                    let bottom_len = m + k * i; // in vertices, not edges
                    let top_len = m + k * (i + 1);

                    let bottom_vertex_start_index = (m + 1) * i + k * i * (i - 1) / 2;
                    let top_vertex_start_index = bottom_vertex_start_index + bottom_len + 1;

                    let bottom_indices = (0..=bottom_len).map(|j| bottom_vertex_start_index + j);
                    let top_indices = (0..=top_len).map(|j| top_vertex_start_index + j);

                    indices.extend(
                        bottom_indices
                            .interleave_shortest(top_indices)
                            .map(|i| i as u32),
                    );

                    if i != (n - 1) {
                        indices.push(Self::U32_RESTART_INDEX)
                    }
                }

                if indices.len() > indices_est {
                    log::warn!("Index buffer for sheet {:?} is larger than estimated. This can lead to a performance penalty", self);
                }
                indices
            }
        }
    }

    /// Returns the primitive topology that should be used with the index buffer for this sheet.
    pub fn primitive_topology(&self) -> IndexPrimitiveTopology {
        match self.tess {
            TessellationMode::Invalid => panic!("Invalid tessellation mode"),
            TessellationMode::NegDiagonal => IndexPrimitiveTopology::VkTriangleStrip,
        }
    }
    /// Returns the front face that should be used with the index buffer for this sheet.
    ///
    /// Front face depends on the basis vectors, and cannot be determined
    /// from the index buffer alone. This provides an initial value, which must then be
    /// adjusted keeping in mind both the basis vectors and the desired visible face.
    pub fn front_face(&self) -> FrontFace {
        match self.tess {
            TessellationMode::Invalid => panic!("Invalid tessellation mode"),
            TessellationMode::NegDiagonal => FrontFace::Cw,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SheetControl {
    origin: Vector3<f64>,
    basis_u: Vector2<f64>,
    basis_v: Vector2<f64>,
    m: usize,
    n: usize,
    k: isize,
    tess: TessellationMode,
}
impl SheetControl {
    /// The total number of vertices in the sheet.
    pub fn vertex_count(&self) -> usize {
        let m = self.m as isize;
        let n = self.n as isize;
        let k = self.k;
        ((m + 1) * (n + 1) + (k * n * (n + 1) / 2)) as usize
    }

    /// Returns a key that can be used to identify the index buffer for this sheet.
    /// No matter what the basis vectors and origin are, the index buffer will be the same
    /// for the same tessellation mode, m, n, and k. This can be used for memoization.
    ///
    /// Note that this is not guaranteed to be network-stable, it's meant for use in-process.
    ///
    /// This docstring makes no guarantee that the actual use-case will repeat the same
    /// index buffer key - it's likely for the current roadmap, however.
    pub fn index_buffer_key(&self) -> IndexBufferKey {
        IndexBufferKey {
            tess: self.tess,
            m: self.m,
            n: self.n,
            k: self.k,
        }
    }

    /// Converts this sheet control to a protocol buffer.
    pub fn to_proto(&self) -> FarSheetControlProto {
        FarSheetControlProto {
            origin: Some(
                self.origin
                    .try_into()
                    .expect("origin Vec3D protobuf conversion failed"),
            ),
            basis_u: Some(
                self.basis_u
                    .try_into()
                    .expect("basis_u Vec2D protobuf conversion failed"),
            ),
            basis_v: Some(
                self.basis_v
                    .try_into()
                    .expect("basis_v Vec2D protobuf conversion failed"),
            ),
            m: self.m as u32,
            n: self.n as u32,
            k: self.k as i32,
            tess: self.tess.into(),
        }
    }

    /// Returns an iterator over the lattice points in the sheet.
    ///
    /// The points are returned in the order the index buffer will use them.
    #[inline(always)]
    pub fn iter_lattice_points_lattice_space(
        &self,
    ) -> impl Iterator<Item = (isize, isize)> + use<'_> {
        (0..=self.m as isize)
            .flat_map(|i| (0..=self.n as isize + (self.k * i) as isize).map(move |j| (i, j)))
    }

    /// Returns an iterator over the lattice points in the sheet, in local space.
    ///
    /// The basis vectors are respected, but the origin is ignored.
    ///
    /// The points are returned in the order the index buffer will use them.
    #[inline(always)]
    pub fn iter_lattice_points_local_space(&self) -> impl Iterator<Item = Vector3<f64>> + use<'_> {
        self.iter_lattice_points_lattice_space().map(|(i, j)| {
            let i_f = i as f64;
            let j_f = j as f64;
            vec3(
                self.basis_u.x * i_f + self.basis_v.x * j_f,
                0.0,
                self.basis_u.y * i_f + self.basis_v.y * j_f,
            )
        })
    }

    #[inline(always)]
    pub fn origin(&self) -> Vector3<f64> {
        self.origin
    }

    /// Returns an iterator over the lattice points in the sheet, in world space.
    ///
    /// The basis vectors and origin are respected.
    ///
    /// The points are returned in the order the index buffer will use them.
    #[inline(always)]
    pub fn iter_lattice_points_world_space(&self) -> impl Iterator<Item = Vector3<f64>> + use<'_> {
        let origin = self.origin;
        self.iter_lattice_points_local_space()
            .map(move |point| vec3(origin.x + point.x, origin.y + point.y, origin.z + point.z))
    }

    pub fn new(
        origin: Vector3<f64>,
        basis_u: Vector2<f64>,
        basis_v: Vector2<f64>,
        m: usize,
        n: usize,
        k: isize,
        tess: TessellationMode,
    ) -> Result<Self> {
        ensure!(
            m as isize + n as isize * k as isize >= 0,
            "m + n * k must be non-negative"
        );
        if tess == crate::protocol::map::TessellationMode::Invalid {
            bail!("Invalid tessellation mode");
        }
        Ok(Self {
            origin,
            basis_u,
            basis_v,
            m,
            n,
            k,
            tess,
        })
    }
}

impl TryFrom<FarSheetControlProto> for SheetControl {
    type Error = anyhow::Error;

    fn try_from(value: FarSheetControlProto) -> Result<Self> {
        Self::new(
            value
                .origin
                .context("Missing origin")?
                .try_into()
                .context("origin")?,
            value
                .basis_u
                .context("Missing basis_u")?
                .try_into()
                .context("basis_u")?,
            value
                .basis_v
                .context("Missing basis_v")?
                .try_into()
                .context("basis_v")?,
            value.m as usize,
            value.n as usize,
            value.k as isize,
            value.tess(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_array_length_and_iteration() {
        let control = SheetControl {
            origin: Vector3::new(10.0, 5.0, 20.0),
            basis_u: Vector2::new(1.0, 0.5),
            basis_v: Vector2::new(0.0, 1.0),
            m: 2,
            n: 2,
            k: 1,
            tess: TessellationMode::NegDiagonal,
        };
        assert_eq!(control.vertex_count(), 12);

        let control = SheetControl {
            origin: Vector3::new(10.0, 5.0, 20.0),
            basis_u: Vector2::new(1.0, 0.5),
            basis_v: Vector2::new(0.0, 1.0),
            m: 2,
            n: 2,
            k: -1,
            tess: TessellationMode::NegDiagonal,
        };
        assert_eq!(control.vertex_count(), 6);

        let lattice_points = control
            .iter_lattice_points_lattice_space()
            .collect::<Vec<_>>();
        assert_eq!(
            lattice_points,
            vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)]
        );
        let lattice_world = control
            .iter_lattice_points_world_space()
            .collect::<Vec<_>>();
        assert_eq!(
            lattice_world,
            vec![
                vec3(10.0, 5.0, 20.0),
                vec3(10.0, 5.0, 21.0),
                vec3(10.0, 5.0, 22.0),
                vec3(11.0, 5.0, 20.5),
                vec3(11.0, 5.0, 21.5),
                vec3(12.0, 5.0, 21.0),
            ]
        );
    }

    #[test]
    fn test_conversion_ok() {
        let proto = FarSheetControlProto {
            origin: Some(crate::protocol::coordinates::Vec3D {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            }),
            m: 3,
            n: 3,
            k: 1,
            basis_u: Some(crate::protocol::coordinates::Vec2D { x: 1.0, y: 0.0 }),
            basis_v: Some(crate::protocol::coordinates::Vec2D { x: 0.0, y: 1.0 }),
            tess: TessellationMode::NegDiagonal.into(),
        };
        let control = SheetControl::try_from(proto).unwrap();
        assert_eq!(
            control,
            SheetControl {
                origin: Vector3::new(0.0, 0.0, 0.0),
                basis_u: Vector2::new(1.0, 0.0),
                basis_v: Vector2::new(0.0, 1.0),
                m: 3,
                n: 3,
                k: 1,
                tess: TessellationMode::NegDiagonal,
            }
        );
        assert_eq!(control.to_proto(), proto);
    }

    #[test]
    fn test_bad_conversions() {
        let good = FarSheetControlProto {
            origin: Some(crate::protocol::coordinates::Vec3D {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            }),
            m: 3,
            n: 3,
            k: 1,
            basis_u: Some(crate::protocol::coordinates::Vec2D { x: 1.0, y: 0.0 }),
            basis_v: Some(crate::protocol::coordinates::Vec2D { x: 0.0, y: 1.0 }),
            tess: TessellationMode::NegDiagonal.into(),
        };

        assert!(SheetControl::try_from(good.clone()).is_ok());
        assert!(SheetControl::try_from(FarSheetControlProto {
            tess: TessellationMode::Invalid.into(),
            ..good
        })
        .is_err());

        assert!(SheetControl::try_from(FarSheetControlProto { m: 0, ..good }).is_err());

        assert!(SheetControl::try_from(FarSheetControlProto {
            m: 3,
            n: 3,
            k: -5,
            ..good
        })
        .is_err());

        assert!(SheetControl::try_from(FarSheetControlProto {
            basis_u: None,
            ..good
        })
        .is_err());

        assert!(SheetControl::try_from(FarSheetControlProto {
            basis_v: None,
            ..good
        })
        .is_err());
    }
    #[test]
    fn test_index_buffer() {
        let control = SheetControl {
            origin: Vector3::new(0.0, 0.0, 0.0),
            basis_u: Vector2::new(1.0, 0.0),
            basis_v: Vector2::new(0.0, 1.0),
            m: 3,
            n: 3,
            k: -1,
            tess: TessellationMode::NegDiagonal,
        };
        let key = control.index_buffer_key();

        assert_eq!(
            key,
            IndexBufferKey {
                tess: TessellationMode::NegDiagonal,
                m: 3,
                n: 3,
                k: -1,
            }
        );
        let indices = key.build();

        const R: u32 = u32::MAX;

        assert_eq!(
            indices,
            vec![0, 4, 1, 5, 2, 6, 3, R, 4, 7, 5, 8, 6, R, 7, 9, 8]
        );

        assert_eq!(
            key.primitive_topology(),
            IndexPrimitiveTopology::VkTriangleStrip
        );
        assert_eq!(key.front_face(), FrontFace::Cw);
    }

    #[test]
    fn test_index_buffer_square() {
        let key = IndexBufferKey {
            tess: TessellationMode::NegDiagonal,
            m: 2,
            n: 2,
            k: 0,
        };
        let indices = key.build();

        const R: u32 = u32::MAX;

        assert_eq!(indices, vec![0, 3, 1, 4, 2, 5, R, 3, 6, 4, 7, 5, 8]);
    }

    #[test]
    fn test_index_buffer_overhang() {
        let key = IndexBufferKey {
            tess: TessellationMode::NegDiagonal,
            m: 2,
            n: 2,
            k: 1,
        };
        let indices = key.build();

        const R: u32 = u32::MAX;

        assert_eq!(indices, vec![0, 3, 1, 4, 2, 5, R, 3, 7, 4, 8, 5, 9, 6, 10]);
    }
}
