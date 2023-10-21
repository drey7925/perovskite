use crate::{
    block_renderer::{BlockRenderer, CubeExtents},
    vulkan::shaders::cube_geometry::CubeGeometryVertex,
};
use anyhow::Result;
use cgmath::{ElementWise, Vector3, Zero};
use rustc_hash::FxHashMap;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryUsage},
};

pub(crate) struct GameEntity {
    // todo refine
    pub(crate) position: Vector3<f64>,
}
impl GameEntity {
    pub(crate) fn as_transform(&self, base_position: Vector3<f64>) -> cgmath::Matrix4<f32> {
        cgmath::Matrix4::from_translation(
            (self.position - base_position).mul_element_wise(Vector3::new(1., -1., 1.)),
        )
        .cast()
        .unwrap()
    }
}

pub(crate) struct EntityManager {
    // todo properly encapsulate
    pub(crate) entities: FxHashMap<u64, GameEntity>,

    pub(crate) fake_entity_vtx: Subbuffer<[CubeGeometryVertex]>,
    pub(crate) fake_entity_idx: Subbuffer<[u32]>,
}
impl EntityManager {
    pub(crate) fn new(block_renderer: &BlockRenderer) -> Result<Self> {
        let fake_extents = CubeExtents::new((-0.375, 0.375), (-0.2, 1.5), (-0.01, 0.01));
        let tex = [block_renderer.fake_entity_tex_coords(); 6];

        let mut vtx = vec![];
        let mut idx = vec![];

        block_renderer.emit_single_cube_simple(
            fake_extents,
            Vector3::zero(),
            tex,
            &mut vtx,
            &mut idx,
        );

        Ok(Self {
            entities: FxHashMap::default(),
            fake_entity_vtx: Buffer::from_iter(
                block_renderer.allocator(),
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    usage: MemoryUsage::Upload,
                    ..Default::default()
                },
                vtx.into_iter(),
            )?,
            fake_entity_idx: Buffer::from_iter(
                block_renderer.allocator(),
                BufferCreateInfo {
                    usage: BufferUsage::INDEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    usage: MemoryUsage::Upload,
                    ..Default::default()
                },
                idx.into_iter(),
            )?,
        })
    }
}
