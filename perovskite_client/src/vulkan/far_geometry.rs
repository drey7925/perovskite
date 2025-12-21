use std::collections::hash_map::Entry;

use anyhow::{Context, Result};
use cgmath::{ElementWise, Matrix4, Vector3};
use perovskite_core::{
    far_sheet::{IndexBufferKey, SheetControl},
    protocol::{game_rpc, map as map_rpc},
};
use rustc_hash::FxHashMap;
use vulkano::buffer::{BufferUsage, Subbuffer};

use crate::vulkan::{
    shaders::far_mesh::{FarMeshDrawCall, FarMeshVertex},
    VulkanContext,
};
struct ClientFarSheet {
    index_buffer: Subbuffer<[u32]>,
    vertex_buffer: Subbuffer<[FarMeshVertex]>,
    origin: Vector3<f64>,
}

fn build_index_buffer(key: IndexBufferKey, vk_ctx: &VulkanContext) -> Result<Subbuffer<[u32]>> {
    vk_ctx.iter_to_device_via_staging(key.build().into_iter(), BufferUsage::INDEX_BUFFER)
}

impl ClientFarSheet {
    fn new(
        sheet: &map_rpc::FarSheet,
        index_buffer_cache: &mut FxHashMap<IndexBufferKey, Subbuffer<[u32]>>,
        vk_ctx: &VulkanContext,
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
            .map(|(point, height)| FarMeshVertex {
                position: [point.x as f32, -height, point.z as f32],
            })
            .collect::<Vec<_>>();

        Ok(Self {
            index_buffer,
            vertex_buffer: vk_ctx
                .iter_to_device_via_staging(vertices.into_iter(), BufferUsage::VERTEX_BUFFER)?,
            origin: control.origin(),
        })
    }
}
pub(crate) struct FarGeometryState {
    sheets: FxHashMap<u64, ClientFarSheet>,
    index_buffer_cache: FxHashMap<IndexBufferKey, Subbuffer<[u32]>>,
}
impl FarGeometryState {
    pub(crate) fn new() -> Self {
        Self {
            sheets: FxHashMap::default(),
            index_buffer_cache: FxHashMap::default(),
        }
    }
    pub(crate) fn update(
        &mut self,
        rpc: &game_rpc::FarGeometry,
        vk_ctx: &VulkanContext,
    ) -> Result<()> {
        for id in &rpc.remove_ids {
            if self.sheets.remove(id).is_none() {
                log::warn!("Tried to remove non-existent far sheet {}", id);
            }
        }
        for sheet in &rpc.far_sheet {
            self.sheets.insert(
                sheet.geometry_id,
                ClientFarSheet::new(sheet, &mut self.index_buffer_cache, vk_ctx)?,
            );
        }
        Ok(())
    }

    pub(crate) fn draw_calls(
        &self,
        player_position: Vector3<f64>,
        // Will eventually be used for frustum culling
        _view_proj_matrix: Matrix4<f32>,
    ) -> Vec<FarMeshDrawCall> {
        self.sheets
            .values()
            .map(|sheet| {
                let relative_origin =
                    (sheet.origin - player_position).mul_element_wise(Vector3::new(1., -1., 1.));
                let translation = Matrix4::from_translation(relative_origin.cast().unwrap());

                FarMeshDrawCall {
                    model_matrix: translation,
                    vtx: sheet.vertex_buffer.clone(),
                    idx: sheet.index_buffer.clone(),
                    num_indices: sheet.index_buffer.len() as u32,
                }
            })
            .collect()
    }
}
