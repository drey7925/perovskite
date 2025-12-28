use std::{collections::hash_map::Entry, sync::Arc};

use anyhow::{Context, Result};
use cgmath::{ElementWise, Matrix4, Vector3};
use perovskite_core::{
    coordinates::ChunkCoordinate,
    far_sheet::{IndexBufferKey, SheetControl},
    protocol::{game_rpc, map as map_rpc},
};
use rustc_hash::{FxHashMap, FxHashSet};
use vulkano::buffer::{BufferUsage, Subbuffer};

use crate::{
    client_state::block_types::ClientBlockTypeManager,
    vulkan::{
        shaders::far_mesh::{FarMeshDrawCall, FarMeshVertex},
        VulkanContext,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum WindingOrder {
    Clockwise,
    CounterClockwise,
}

struct ClientFarSheet {
    index_buffer: Subbuffer<[u32]>,
    vertex_buffer: Subbuffer<[FarMeshVertex]>,
    origin: Vector3<f64>,
    chunk_corners: [(i32, i32); 4],
    winding_order: WindingOrder,
}

fn build_index_buffer(key: IndexBufferKey, vk_ctx: &VulkanContext) -> Result<Subbuffer<[u32]>> {
    vk_ctx.iter_to_device_via_staging(key.build().into_iter(), BufferUsage::INDEX_BUFFER)
}

impl ClientFarSheet {
    fn new(
        sheet: &map_rpc::FarSheet,
        index_buffer_cache: &mut FxHashMap<IndexBufferKey, Subbuffer<[u32]>>,
        vk_ctx: &VulkanContext,
        block_type_manager: &ClientBlockTypeManager,
    ) -> Result<Self> {
        let control =
            SheetControl::try_from(sheet.control.clone().context("Missing far sheet control")?)?;

        let index_buffer_key = control.index_buffer_key();
        let index_buffer = match index_buffer_cache.entry(index_buffer_key) {
            Entry::Occupied(entry) => entry.get().clone(),
            Entry::Vacant(entry) => {
                let index_buffer = build_index_buffer(index_buffer_key, vk_ctx)?;
                entry.insert(index_buffer.clone());
                index_buffer
            }
        };

        // TODO: This should not use CubeGeometryVertex, but rather a far-mesh specific
        // vertex type. This is a hack to quickly test using the existing cube geometry
        // pipeline.
        let vertices = control
            .iter_lattice_points_local_space()
            .zip(sheet.heights.iter().copied())
            .zip(sheet.block_types_no_variant.iter().copied())
            .map(|((point, height), block_index)| FarMeshVertex {
                position: [point.x as f32, -height, point.z as f32],
                color: block_type_manager
                    .lod_color_argb_from_index(block_index as usize)
                    .to_le_bytes(),
            })
            .collect::<Vec<_>>();

        let chunk_corners = control.lattice_corners_world_space().map(|corner| {
            let x = (corner.x.clamp(i32::MIN as f64, i32::MAX as f64) as i32).div_euclid(16);
            let z = (corner.y.clamp(i32::MIN as f64, i32::MAX as f64) as i32).div_euclid(16);
            (x, z)
        });

        let basis_cross = control.basis_u().perp_dot(control.basis_v());
        let winding_order = if basis_cross > 0.0 {
            WindingOrder::Clockwise
        } else {
            WindingOrder::CounterClockwise
        };

        Ok(Self {
            index_buffer,
            vertex_buffer: vk_ctx
                .iter_to_device_via_staging(vertices.into_iter(), BufferUsage::VERTEX_BUFFER)?,
            origin: control.origin(),
            chunk_corners,
            winding_order,
        })
    }

    fn should_render(&self, ignore_chunks: &FxHashSet<(i32, i32)>) -> bool {
        self.chunk_corners
            .iter()
            .any(|corner| !ignore_chunks.contains(corner))
    }
}
pub(crate) struct FarGeometryState {
    sheets: FxHashMap<u64, ClientFarSheet>,
    index_buffer_cache: FxHashMap<IndexBufferKey, Subbuffer<[u32]>>,
    block_type_manager: Arc<ClientBlockTypeManager>,
}
impl FarGeometryState {
    pub(crate) fn new(block_type_manager: Arc<ClientBlockTypeManager>) -> Self {
        Self {
            sheets: FxHashMap::default(),
            index_buffer_cache: FxHashMap::default(),
            block_type_manager,
        }
    }
    pub(crate) fn update(
        &mut self,
        rpc: &game_rpc::FarGeometry,
        vk_ctx: &VulkanContext,
    ) -> Result<()> {
        let remove_set = FxHashSet::from_iter(rpc.remove_ids.iter().copied());
        let add_set = FxHashSet::from_iter(rpc.far_sheet.iter().map(|sheet| sheet.geometry_id));
        let intersection = FxHashSet::from_iter(remove_set.intersection(&add_set));
        if !intersection.is_empty() {
            log::warn!("Far sheet ID collision: {:?}", intersection);
        }

        for id in &rpc.remove_ids {
            if self.sheets.remove(id).is_none() {
                log::warn!("Tried to remove non-existent far sheet {}", id);
            }
        }
        for sheet in &rpc.far_sheet {
            self.sheets.insert(
                sheet.geometry_id,
                ClientFarSheet::new(
                    sheet,
                    &mut self.index_buffer_cache,
                    vk_ctx,
                    &self.block_type_manager,
                )?,
            );
        }
        Ok(())
    }

    pub(crate) fn num_meshes(&self) -> usize {
        self.sheets.len()
    }

    pub(crate) fn draw_calls(
        &self,
        player_position: Vector3<f64>,
        // Will eventually be used for frustum culling
        _view_proj_matrix: Matrix4<f32>,
        ignore_chunks: &FxHashSet<ChunkCoordinate>,
    ) -> Vec<FarMeshDrawCall> {
        let ignore_slices =
            FxHashSet::from_iter(ignore_chunks.iter().map(|coord| (coord.x, coord.z)));

        self.sheets
            .values()
            .filter(|sheet| sheet.should_render(&ignore_slices))
            .map(|sheet| {
                let relative_origin =
                    (sheet.origin - player_position).mul_element_wise(Vector3::new(1., -1., 1.));
                let translation = Matrix4::from_translation(relative_origin.cast().unwrap());

                FarMeshDrawCall {
                    model_matrix: translation,
                    vtx: sheet.vertex_buffer.clone(),
                    idx: sheet.index_buffer.clone(),
                    num_indices: sheet.index_buffer.len() as u32,
                    winding_order: sheet.winding_order,
                }
            })
            .collect()
    }
}
