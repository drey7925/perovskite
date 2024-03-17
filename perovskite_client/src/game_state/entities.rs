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
    seq: u64,
}
impl EntityMove {
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
            seq: value.sequence,
        })
    }
}

pub(crate) struct GameEntity {
    // todo refine
    // 9 should be enough, server has a queue depth of 8
    pub(crate) move_queue: circular_buffer::CircularBuffer<9, EntityMove>,
    current_move_started: Instant,
    pub(crate) current_move_sequence: u64,
    pub fallback_position: Vector3<f64>,

    // debug only
    created: Instant,
}
impl GameEntity {
    pub(crate) fn advance_state(&mut self) {
        let now = Instant::now();
        let elapsed = (now - self.current_move_started).as_secs_f32();

        if self
            .move_queue
            .front()
            .is_some_and(|m| elapsed > m.total_time_seconds)
        {
            println!("=== @ {} ===", self.created.elapsed().as_secs_f32());
            println!(
                "remaining buffer: {:?}",
                self.move_queue
                    .iter()
                    .map(|m| m.total_time_seconds)
                    .sum::<f32>()
            );
            let popped_move = self.move_queue.pop_front().unwrap();
            self.current_move_started += Duration::from_secs_f32(popped_move.total_time_seconds);
            self.current_move_sequence = self.move_queue.front().map(|m| m.seq).unwrap_or(0);
        }
    }

    pub(crate) fn update(
        &mut self,
        update: &entities_proto::EntityUpdate,
    ) -> Result<(), &'static str> {
        println!(">>> @ {} <<<", self.created.elapsed().as_secs_f32());
        println!(
            "Got {} -> {}, while CMS is {}",
            update
                .planned_move
                .iter()
                .map(|m| m.sequence)
                .min()
                .unwrap(),
            update
                .planned_move
                .iter()
                .map(|m| m.sequence)
                .max()
                .unwrap(),
            self.current_move_sequence
        );

        while self
            .move_queue
            .front()
            .is_some_and(|m| m.seq < update.current_move_sequence)
        {
            self.move_queue.pop_front().unwrap();
        }
        for m in &update.planned_move {
            self.move_queue.push_back(m.try_into().map_err(|_| {
                dbg!(m);

                "invalid planned move"
            })?);
        }

        if update.current_move_progress < 0.0 {
            return Err("invalid current move progress: negative");
        } else if update.current_move_progress.is_nan() {
            return Err("invalid current move progress: NaN");
        }

        if update.current_move_sequence != self.current_move_sequence {
            println!(
                "retiming {} -> {}",
                self.current_move_sequence, update.current_move_sequence
            );
            println!(
                "{:?} -> {:?}",
                self.current_move_started,
                Instant::now() - Duration::from_secs_f32(update.current_move_progress)
            );
            self.current_move_sequence = update.current_move_sequence;
            self.current_move_started =
                Instant::now() - Duration::from_secs_f32(update.current_move_progress);
        }
        self.fallback_position = self
            .move_queue
            .back()
            .map(|m| m.qproj(m.total_time_seconds))
            .unwrap_or(vec3(f64::NAN, f64::NAN, f64::NAN));

        Ok(())
    }

    pub(crate) fn as_transform(&self, base_position: Vector3<f64>) -> cgmath::Matrix4<f32> {
        cgmath::Matrix4::from_translation(
            (self.position() - base_position).mul_element_wise(Vector3::new(1., -1., 1.)),
        )
        .cast()
        .unwrap()
    }

    pub(crate) fn position(&self) -> Vector3<f64> {
        let time = (Instant::now() - self.current_move_started).as_secs_f32();
        self.move_queue
            .front()
            .map(|m| m.qproj(time))
            .unwrap_or(self.fallback_position)
    }

    pub(crate) fn from_proto(update: &entities_proto::EntityUpdate) -> Result<GameEntity, &str> {
        if update.planned_move.is_empty() {
            return Err("Empty move plan");
        }
        let mut queue = circular_buffer::CircularBuffer::new();
        for planned_move in &update.planned_move {
            queue.push_back(planned_move.try_into().map_err(|_| {
                dbg!(planned_move);

                "invalid planned move"
            })?);
        }
        println!("cms: {:?}", update.current_move_progress);
        Ok(GameEntity {
            fallback_position: queue
                .back()
                .map(|m: &EntityMove| m.qproj(m.total_time_seconds))
                .unwrap(),
            move_queue: queue,
            current_move_started: Instant::now()
                - Duration::try_from_secs_f32(update.current_move_progress)
                    .map_err(|_| "invalid current move progress")?,
            current_move_sequence: update.current_move_sequence,
            created: Instant::now(),
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
