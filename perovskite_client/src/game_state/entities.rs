use std::time::{Duration, Instant};

use crate::vulkan::{
    block_renderer::{BlockRenderer, CubeExtents, VkCgvBufferGpu},
    entity_renderer::EntityRenderer,
    shaders::{
        cube_geometry::{CubeGeometryDrawCall, CubeGeometryVertex},
        entity_geometry::EntityGeometryDrawCall,
    },
};
use anyhow::{Context, Result};
use cgmath::{vec3, Deg, ElementWise, Matrix4, Rad, Vector3, Zero};
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
    pub last_face_dir: f32,
    id: u64,
    class: u32,
    // debug only
    created: Instant,
}
impl GameEntity {
    pub(crate) fn advance_state(&mut self) {
        let now = Instant::now();
        let mut elapsed = (now - self.current_move_started).as_secs_f32();
        let mut any_popped = false;
        while self
            .move_queue
            .front()
            .is_some_and(|m| elapsed > m.total_time_seconds)
        {
            // println!("=== @ {} ===", self.created.elapsed().as_secs_f32());
            any_popped = true;
            let popped_move = self.move_queue.pop_front().unwrap();
            // println!("popping move {:?}", popped_move);
            self.current_move_started += Duration::from_secs_f32(popped_move.total_time_seconds);
            elapsed -= popped_move.total_time_seconds;
            self.current_move_sequence = self.move_queue.front().map(|m| m.seq).unwrap_or(0);
        }
        if any_popped && self.id == 8 {
            println!(
                "remaining buffer: {:?}",
                self.move_queue
                    .iter()
                    .map(|m| m.total_time_seconds)
                    .sum::<f32>()
                    - elapsed
            );
        }
    }

    pub(crate) fn update(
        &mut self,
        update: &entities_proto::EntityUpdate,
        estimated_send_time: Instant,
    ) -> Result<(), &'static str> {
        if update.id == 8 {
            println!("{:?}", update);
            println!(
                "estimated send was {} sec ago",
                estimated_send_time.elapsed().as_secs_f32()
            );
        }

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
            if update.id == 8 {
                println!(
                    "retiming {} -> {}",
                    self.current_move_sequence, update.current_move_sequence
                );
                println!(
                    "{:?} -> {:?}",
                    self.current_move_started,
                    Instant::now() - Duration::from_secs_f32(update.current_move_progress)
                );
            }

            self.current_move_sequence = update.current_move_sequence;
            self.current_move_started =
                estimated_send_time - Duration::from_secs_f32(update.current_move_progress);
        }
        self.fallback_position = self
            .move_queue
            .back()
            .map(|m| m.qproj(m.total_time_seconds))
            .unwrap_or(vec3(f64::NAN, f64::NAN, f64::NAN));
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
        time: Instant,
    ) -> cgmath::Matrix4<f32> {
        let (pos, angle) = self.position(time);
        let translation = cgmath::Matrix4::from_translation(
            (pos - base_position).mul_element_wise(Vector3::new(1., -1., 1.)),
        )
        .cast()
        .unwrap();

        translation * Matrix4::from_angle_y(angle)
    }

    pub(crate) fn position(&self, time: Instant) -> (Vector3<f64>, Rad<f32>) {
        let time = (time.saturating_duration_since(self.current_move_started)).as_secs_f32();
        let pos = self
            .move_queue
            .front()
            .map(|m| m.qproj(time))
            .unwrap_or(self.fallback_position);
        let dir = self
            .move_queue
            .front()
            .map(|m| m.face_direction)
            .unwrap_or(self.last_face_dir);
        (pos, Rad(dir))
    }

    pub(crate) fn from_proto(
        update: &entities_proto::EntityUpdate,
        estimated_send_time: Instant,
    ) -> Result<GameEntity, &str> {
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
        // println!("cms: {:?}", update.current_move_progress);
        Ok(GameEntity {
            fallback_position: queue
                .back()
                .map(|m: &EntityMove| m.qproj(m.total_time_seconds))
                .unwrap(),
            last_face_dir: queue.back().map(|m| m.face_direction).unwrap(),
            move_queue: queue,
            current_move_started: estimated_send_time
                - Duration::try_from_secs_f32(update.current_move_progress)
                    .map_err(|_| "invalid current move progress")?,
            current_move_sequence: update.current_move_sequence,
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

    pub(crate) fn advance_all_states(&mut self) {
        for entity in self.entities.values_mut() {
            entity.advance_state();
        }
    }

    pub(crate) fn render_calls(
        &self,
        player_position: Vector3<f64>,
        time: Instant,
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
                model_matrix: entity.as_transform(player_position, time),
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
