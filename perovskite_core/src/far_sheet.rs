//! Shared helpers for geometry of far sheets, encoded as packed grids in the protocol.
//!
//! Primarily includes basis vector, bounds, and similar. Note that substantial logic is
//! kept in `perovskite_client`; this focuses specifically on those things that both sides
//! need to share.
use crate::protocol::map::FarSheetControl as FarSheetControlProto;
use anyhow::{bail, ensure, Context, Result};
use cgmath::{Vector2, Vector3};

pub use crate::protocol::map::TessellationMode;

#[derive(Debug, Clone, PartialEq)]
pub struct SheetControl {
    origin: Vector3<f64>,
    basis_u: Vector2<f32>,
    basis_v: Vector2<f32>,
    m: usize,
    n: usize,
    k: isize,
    tess: TessellationMode,
}
impl SheetControl {
    pub fn array_length(&self) -> usize {
        let m = self.m as isize;
        let n = self.n as isize;
        let k = self.k;
        (m * n + k * n * (n - 1) / 2) as usize
    }
}

impl TryFrom<FarSheetControlProto> for SheetControl {
    type Error = anyhow::Error;

    fn try_from(value: FarSheetControlProto) -> Result<Self> {
        ensure!(value.m > 0, "m must be positive");
        ensure!(value.n > 0, "n must be positive");
        ensure!(
            value.m as isize + value.n as isize * value.k as isize >= 0,
            "m + n * k must be non-negative"
        );

        let basis_u: Vector3<f32> = value.basis_u.context("Missing basis_u")?.try_into()?;
        let basis_v: Vector3<f32> = value.basis_v.context("Missing basis_v")?.try_into()?;

        let tess = value.tess();
        if tess == crate::protocol::map::TessellationMode::Invalid {
            bail!("Invalid tessellation mode");
        }

        Ok(SheetControl {
            origin: value.origin.context("Missing origin")?.try_into()?,
            basis_u: Vector2::new(basis_u.x, basis_u.z),
            basis_v: Vector2::new(basis_v.x, basis_v.z),
            m: value.m as usize,
            n: value.n as usize,
            k: value.k as isize,
            tess,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_array_length() {
        let control = SheetControl {
            origin: Vector3::new(0.0, 0.0, 0.0),
            basis_u: Vector2::new(1.0, 0.0),
            basis_v: Vector2::new(0.0, 1.0),
            m: 3,
            n: 3,
            k: 1,
            tess: TessellationMode::NegDiagonal,
        };
        assert_eq!(control.array_length(), 12);

        let control = SheetControl {
            origin: Vector3::new(0.0, 0.0, 0.0),
            basis_u: Vector2::new(1.0, 0.0),
            basis_v: Vector2::new(0.0, 1.0),
            m: 3,
            n: 3,
            k: -1,
            tess: TessellationMode::NegDiagonal,
        };
        assert_eq!(control.array_length(), 6);
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
            basis_u: Some(crate::protocol::coordinates::Vec3F {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            }),
            basis_v: Some(crate::protocol::coordinates::Vec3F {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            }),
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
            basis_u: Some(crate::protocol::coordinates::Vec3F {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            }),
            basis_v: Some(crate::protocol::coordinates::Vec3F {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            }),
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
}
