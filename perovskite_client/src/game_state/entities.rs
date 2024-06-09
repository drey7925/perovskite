use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

use crate::vulkan::{
    block_renderer::{BlockRenderer, CubeExtents, VkCgvBufferGpu},
    entity_renderer::EntityRenderer,
    shaders::{
        cube_geometry::{CubeGeometryDrawCall, CubeGeometryVertex},
        entity_geometry::EntityGeometryDrawCall,
    },
};
use anyhow::{Context, Result};
use cgmath::{vec3, Deg, ElementWise, InnerSpace, Matrix4, Rad, Vector3, Zero};
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
    start_tick: u64,
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
            total_time_seconds: value.time_ticks as f32 / 1_000_000_000.0,
            face_direction: value.face_direction,
            seq: value.sequence,
            start_tick: value.start_tick,
        })
    }
}

pub(crate) struct GameEntity {
    // todo refine
    // 33 should be enough, server has a queue depth of up to 32
    pub(crate) move_queue: VecDeque<EntityMove>,
    pub fallback_position: Vector3<f64>,
    pub last_face_dir: f32,
    id: u64,
    class: u32,
    // debug only
    created: Instant,
}
impl GameEntity {
    pub(crate) fn advance_state(&mut self, until_tick: u64) {
        let any_popped = self.pop_until(until_tick);
        if any_popped {
            println!(
                "remaining buffer: {:?}",
                self.move_queue.back().map(|m| m.start_tick
                    + (m.total_time_seconds * 1_000_000_000.0) as u64
                    - until_tick)
            );
        }
    }

    fn pop_until(&mut self, until_tick: u64) -> bool {
        let idx = self
            .move_queue
            .iter()
            .rposition(|m| m.start_tick < until_tick);

        if let Some(idx) = idx {
            if idx == 0 {
                // The first move is the only move that started in the past, so we need to keep it
                false
            } else {
                // drain up to *and not including* idx
                // e.g. if idx == 2, move_queue[2] is the last move that started in the past, so we need to drain move_queue[0..2], which is move_queue[0] and move_queue[1]
                self.move_queue.drain(..idx);
                true
            }
        } else {
            false
        }
    }

    pub(crate) fn estimated_buffer(&self, until_tick: u64) -> f32 {
        let ticks = self
            .move_queue
            .back()
            .map(|m| m.start_tick + (m.total_time_seconds * 1_000_000_000.0) as u64 - until_tick)
            .unwrap_or(0);
        (ticks as f32) / 1_000_000_000.0
    }

    pub(crate) fn estimated_buffer_count(&self) -> usize {
        self.move_queue.len()
    }
    pub(crate) fn debug_cms(&self) -> u64 {
        self.move_queue.front().map(|m| m.seq).unwrap_or(0)
    }
    pub(crate) fn debug_cme(&self, time_tick: u64) -> f32 {
        self.move_queue
            .front()
            .map(|m| (time_tick - m.start_tick) as f32 / 1_000_000_000.0)
            .unwrap_or(0.0)
    }

    pub(crate) fn update(
        &mut self,
        update: &entities_proto::EntityUpdate,
        update_tick: u64,
    ) -> Result<(), &'static str> {
        if update.id == 8 {
            println!("{:?}", update);
        }
        for m in &update.planned_move {
            self.move_queue.push_back(m.try_into().map_err(|_| {
                dbg!(m);

                "invalid planned move"
            })?);
        }

        log::info!(
            ">>> tick {update_tick}, queue before pops: {:?}",
            self.move_queue
        );
        if let Some(m) = self.move_queue.back() {
            self.fallback_position = m.qproj(m.total_time_seconds);
        }

        let any_popped = self.pop_until(update_tick);

        log::info!(
            "tick {update_tick}, queue: {:?}, any popped? {}",
            self.move_queue,
            any_popped
        );

        if let Some(m) = self.move_queue.back() {
            self.fallback_position = m.qproj(m.total_time_seconds);
        }
        self.last_face_dir = self
            .move_queue
            .back()
            .map(|m| m.face_direction)
            .unwrap_or(0.0);
        Ok(())
    }

    pub(crate) fn as_transform(
        &self,
        base_position: Vector3<f64>,
        time_tick: u64,
    ) -> cgmath::Matrix4<f32> {
        let (pos, angle) = self.position(time_tick);
        let translation = cgmath::Matrix4::from_translation(
            (pos - base_position).mul_element_wise(Vector3::new(1., -1., 1.)),
        )
        .cast()
        .unwrap();

        translation * Matrix4::from_angle_y(angle)
    }

    pub(crate) fn position(&self, time_tick: u64) -> (Vector3<f64>, Rad<f32>) {
        let pos = self
            .move_queue
            .front()
            .map(|m| {
                let time = self
                    .move_queue
                    .front()
                    .map(|m| {
                        ((time_tick.saturating_sub(m.start_tick) as f32) / 1_000_000_000.0)
                            .clamp(0.0, m.total_time_seconds)
                    })
                    .unwrap_or(0.0);

                m.qproj(time)
            })
            .unwrap_or(self.fallback_position);
        let dir = self
            .move_queue
            .front()
            .map(|m| m.face_direction)
            .unwrap_or(self.last_face_dir);
        if !pos.x.is_finite() || !pos.y.is_finite() || !pos.z.is_finite() {
            log::info!(
                "invalid position, move queue contains {:?}",
                self.move_queue
            );
            return (Vector3::zero(), Rad(0.0));
        }
        (pos, Rad(dir))
    }

    pub(crate) fn debug_speed(&self, time_tick: u64) -> f32 {
        self.move_queue
            .front()
            .map(|m| {
                let time = self
                    .move_queue
                    .front()
                    .map(|m| {
                        ((time_tick.saturating_sub(m.start_tick) as f32) / 1_000_000_000.0)
                            .clamp(0.0, m.total_time_seconds)
                    })
                    .unwrap_or(0.0);

                (m.velocity + m.acceleration * time).magnitude()
            })
            .unwrap_or(0.0)
    }

    pub(crate) fn from_proto(update: &entities_proto::EntityUpdate) -> Result<GameEntity, &str> {
        if update.planned_move.is_empty() {
            return Err("Empty move plan");
        }
        let mut queue = VecDeque::new();
        for planned_move in &update.planned_move {
            queue.push_back(planned_move.try_into().map_err(|_| {
                dbg!(planned_move);

                "invalid planned move"
            })?);
        }
        log::info!("initial queue: {:?}", queue);
        // println!("cms: {:?}", update.current_move_progress);
        Ok(GameEntity {
            fallback_position: queue
                .back()
                .map(|m: &EntityMove| m.qproj(m.total_time_seconds))
                .unwrap(),
            last_face_dir: queue.back().map(|m| m.face_direction).unwrap(),
            move_queue: queue,
            created: Instant::now(),
            class: update.entity_class,
            id: update.id,
        })
    }
}

#[inline]
fn qproj(s: f64, v: f32, a: f32, t: f32) -> f64 {
    s + (v * t + 0.5 * a * t * t) as f64
}

// Minimal stub implementation of the entity manager
// Quite barebones, and will need extensive optimization in the future.
pub(crate) struct EntityState {
    // todo properly encapsulate
    pub(crate) entities: FxHashMap<u64, GameEntity>,

    pub(crate) fallback_entity: VkCgvBufferGpu,

    pub(crate) attached_to_entity: Option<u64>,
}
impl EntityState {
    // TODO - this should not use the block renderer once we're properly using meshes from the server.
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
            fallback_entity: VkCgvBufferGpu::from_buffers(&vtx, &idx, &block_renderer.allocator())?
                .unwrap(),
            attached_to_entity: None,
        })
    }

    pub(crate) fn advance_all_states(&mut self, until_tick: u64) {
        for entity in self.entities.values_mut() {
            entity.advance_state(until_tick);
        }
    }

    pub(crate) fn render_calls(
        &self,
        player_position: Vector3<f64>,
        time_tick: u64,
        entity_renderer: &EntityRenderer,
    ) -> Vec<EntityGeometryDrawCall> {
        // Current implementation will just render each entity in its own draw call.
        // This will obviously not scale well, but is good enough for now.
        // A later impl should eventually:
        // * Provide multiple entities in a single draw call
        // * Find a way to evaluate the movement of each individual entity. This will likely require a change
        //    to the vertex format to include velocity, acceleration, and the necessary timing details.
        self.entities
            .iter()
            .map(|(id, entity)| EntityGeometryDrawCall {
                model_matrix: entity.as_transform(player_position, time_tick),
                model: entity_renderer
                    .get_singleton(entity.class)
                    // class 0 is the fallback
                    .unwrap_or(
                        entity_renderer
                            .get_singleton(0)
                            .unwrap_or(self.fallback_entity.clone()),
                    ),
            })
            .collect()
    }
}
