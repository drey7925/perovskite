use std::time::{Duration, Instant};

use crate::{
    block_renderer::{BlockRenderer, CubeExtents},
    vulkan::shaders::cube_geometry::CubeGeometryVertex,
};
use anyhow::{Context, Result};
use cgmath::{vec3, ElementWise, Vector3, Zero};
use perovskite_core::protocol::entities as entities_proto;
use rustc_hash::FxHashMap;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryUsage},
};

#[derive(Copy, Clone, Debug)]
pub(crate) struct EntityMove {
    start_pos: Vector3<f64>,
    velocity: Vector3<f32>,
    acceleration: Vector3<f32>,
    total_time_seconds: f32,
    face_direction: f32,
}
impl EntityMove {
    fn stay_forever(position: Vector3<f64>) -> EntityMove {
        EntityMove {
            start_pos: position,
            velocity: vec3(0.0, 0.0, 0.0),
            acceleration: vec3(0.0, 0.0, 0.0),
            total_time_seconds: f32::MAX,
            face_direction: 0.0,
        }
    }
    fn qproj(&self, time: f32) -> Vector3<f64> {
        vec3(
            qproj(self.start_pos.x, self.velocity.x, self.acceleration.x, time),
            qproj(self.start_pos.y, self.velocity.y, self.acceleration.y, time),
            qproj(self.start_pos.z, self.velocity.z, self.acceleration.z, time),
        )
    }
}
impl TryFrom<&entities_proto::EntityMove> for EntityMove {
    type Error = anyhow::Error;

    fn try_from(value: &entities_proto::EntityMove) -> Result<Self, Self::Error> {
        if !value.total_time_seconds.is_finite() {
            return Err(anyhow::anyhow!("total_time_seconds contained NaN or inf"));
        }
        if !value.face_direction.is_finite() {
            return Err(anyhow::anyhow!("face_direction contained NaN or inf"));
        }
        Ok(Self {
            start_pos: value
                .start_position
                .as_ref()
                .context("Missing start position")?
                .try_into()?,
            velocity: value
                .velocity
                .as_ref()
                .context("Missing velocity")?
                .try_into()?,
            acceleration: value
                .acceleration
                .as_ref()
                .context("Missing acceleration")?
                .try_into()?,
            total_time_seconds: value.total_time_seconds,
            face_direction: value.face_direction,
        })
    }
}

pub(crate) struct GameEntity {
    // todo refine
    pub(crate) current_move: EntityMove,
    current_move_started: Instant,
    pub(crate) next_move: EntityMove,
    pub(crate) mod_count: u64,
}
impl GameEntity {
    pub(crate) fn advance_state(&mut self) {
        let now = Instant::now();
        let elapsed = (now - self.current_move_started).as_secs_f32();
        if elapsed > self.current_move.total_time_seconds {
            // total_time_seconds can't be NaN, inf, or overly large since otherwise we wouldn't ever complete the move
            self.current_move_started +=
                Duration::from_secs_f32(self.current_move.total_time_seconds);

            self.current_move = self.next_move;
            // And consume the next move
            self.next_move = EntityMove::stay_forever(
                self.current_move
                    .qproj(self.current_move.total_time_seconds),
            );
        }
    }

    pub(crate) fn update(
        &mut self,
        current_move: EntityMove,
        current_move_elapsed: f32,
        next_move: Option<EntityMove>,
        mod_count: u64,
    ) {
        if self.mod_count != mod_count {
            self.current_move = current_move;
            self.current_move_started =
                Instant::now() - Duration::from_secs_f32(current_move_elapsed);
        }
        if let Some(next_move) = next_move {
            self.next_move = next_move;
        } else {
            self.next_move = EntityMove::stay_forever(
                self.current_move
                    .qproj(self.current_move.total_time_seconds),
            )
        }
        self.mod_count = mod_count;
    }

    pub(crate) fn as_transform(&self, base_position: Vector3<f64>) -> cgmath::Matrix4<f32> {
        cgmath::Matrix4::from_translation(
            (self.position() - base_position).mul_element_wise(Vector3::new(1., -1., 1.)),
        )
        .cast()
        .unwrap()
    }

    fn position(&self) -> Vector3<f64> {
        let time = (Instant::now() - self.current_move_started).as_secs_f32();
        self.current_move.qproj(time)
    }

    pub(crate) fn new(
        _id: u64,
        current_move: EntityMove,
        current_move_elapsed: f32,
        next_move: Option<EntityMove>,
    ) -> Result<GameEntity> {
        Ok(Self {
            next_move: next_move.unwrap_or(EntityMove::stay_forever(
                current_move.qproj(current_move.total_time_seconds),
            )),
            current_move,
            current_move_started: Instant::now()
                - Duration::try_from_secs_f32(current_move_elapsed)?,
            mod_count: u64::MAX,
        })
    }
}

#[inline]
fn qproj(s: f64, v: f32, a: f32, t: f32) -> f64 {
    s + (v * t + 0.5 * a * t * t) as f64
}

// Minimal stub implementation of the entity manager
// Quite barebones, and will need extensive optimization in the future.
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
