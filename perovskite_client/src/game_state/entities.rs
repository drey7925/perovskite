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
use cgmath::{vec3, Deg, ElementWise, InnerSpace, Matrix4, Rad, SquareMatrix, Vector3, Zero};
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
    // 33 should be enough, server has a queue depth of up to 32
    pub(crate) move_queue: VecDeque<EntityMove>,
    current_move_started: Instant,
    pub(crate) current_move_sequence: u64,
    pub fallback_position: Vector3<f64>,
    pub last_face_dir: f32,
    id: u64,
    class: u32,
    // debug only
    created: Instant,

    pub(crate) lookback_buffer: VecDeque<EntityMove>,
}
impl GameEntity {
    pub(crate) fn advance_state(&mut self, until: Instant) {
        let mut elapsed = (until - self.current_move_started).as_secs_f32();
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

            self.lookback_buffer.push_front(popped_move);
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

    pub(crate) fn estimated_buffer(&self, until: Instant) -> f32 {
        self.move_queue
            .iter()
            .map(|m| m.total_time_seconds)
            .sum::<f32>()
            - (until - self.current_move_started).as_secs_f32()
    }

    pub(crate) fn estimated_buffer_count(&self) -> usize {
        self.move_queue.len()
    }
    pub(crate) fn debug_cms(&self) -> u64 {
        self.current_move_sequence
    }
    pub(crate) fn debug_cme(&self, time: Instant) -> f32 {
        (time - self.current_move_started).as_secs_f32()
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

        if let Some(m) = self.move_queue.back() {
            self.fallback_position = m.qproj(m.total_time_seconds);
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
        time: Instant,
    ) -> [cgmath::Matrix4<f32>; 8] {
        let positions = self.position_lookback(time);

        let mut results = [Matrix4::zero(); 8];
        for (i, (pos, angle)) in positions.iter().enumerate() {
            let translation = cgmath::Matrix4::from_translation(
                (pos - base_position).mul_element_wise(Vector3::new(1., -1., 1.)),
            )
            .cast()
            .unwrap();

            results[i] = translation * Matrix4::from_angle_y(*angle);
        }
        results
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
        if !pos.x.is_finite() || !pos.y.is_finite() || !pos.z.is_finite() {
            log::info!(
                "invalid position, move queue contains {:?}",
                self.move_queue
            );
            log::info!("CMS {:?}, time {}", self.current_move_started, time);
            return (Vector3::zero(), Rad(0.0));
        }
        (pos, Rad(dir))
    }

    // TODO THIS IS ONLY A PROTOTYPE
    pub(crate) fn position_lookback(&self, time: Instant) -> [(Vector3<f64>, Rad<f32>); 8] {
        if self.move_queue.is_empty() {
            return [(self.fallback_position, Rad(0.0)); 8];
        }
        let time = (time.saturating_duration_since(self.current_move_started)).as_secs_f32();
        let mut current_move = self.move_queue.front().unwrap();

        let accel_sign = (current_move
            .acceleration
            .dot(current_move.velocity.normalize()))
        .signum();
        let mut move_distance = current_move.velocity.magnitude() * time
            + 0.5 * accel_sign * current_move.acceleration.magnitude() * time * time;

        let mut results = [(self.fallback_position, Rad(0.0)); 8];

        let mut lookback_iterator = self.lookback_buffer.iter();
        for i in 0..8 {
            results[i] = (
                current_move.start_pos
                    + (move_distance * current_move.velocity.normalize())
                        .cast()
                        .unwrap(),
                Rad(current_move.face_direction),
            );
            move_distance -= 1.0;
            while move_distance < 0.0 {
                if let Some(m) = lookback_iterator.next() {
                    current_move = m;
                    let move_time = current_move.total_time_seconds;

                    let accel_sign = (current_move
                        .acceleration
                        .dot(current_move.velocity.normalize()))
                    .signum();

                    let new_move_distance = move_time * current_move.velocity.magnitude()
                        + 0.5
                            * accel_sign
                            * current_move.acceleration.magnitude()
                            * move_time
                            * move_time;
                    move_distance += new_move_distance;
                } else {
                    break;
                }
            }
        }

        results

        // let mut lookback = lookback as f32;

        // if self.move_queue.is_empty() {
        //     return (self.fallback_position, Rad(0.0));
        // }

        // let time = (time.saturating_duration_since(self.current_move_started)).as_secs_f32();
        // let mut current_move = self.move_queue.front().unwrap();

        // // The distance from the start of the move to the current point in it
        // let mut distance = current_move.velocity.magnitude() * time + 0.5 * current_move.acceleration.magnitude() * time * time;

        // let mut lookback_iterator = self.lookback_buffer.iter();
        // let (c_move, m_distance) = loop {
        //     if lookback < distance {
        //         break (current_move, distance - lookback);
        //     } else {
        //         lookback -= distance;

        //         if let Some(m) = lookback_iterator.next() {
        //             current_move = m;

        //             let current_move_time = current_move.total_time_seconds;
        //             let current_move_length = current_move_time * current_move.velocity.magnitude() + 0.5 * current_move.acceleration.magnitude() * current_move_time * current_move_time;
        //             distance = current_move_length;
        //         } else {
        //             return (self.fallback_position, Rad(0.0));
        //         }
        //     }
        // };
        // // ASSUMES THAT ACCELERATION AND VELOCITY POINT IN THE SAME DIRECTION
        // (current_move.start_pos + (current_move.velocity.normalize() * m_distance).cast().unwrap(), Rad(c_move.face_direction))
    }

    pub(crate) fn from_proto(
        update: &entities_proto::EntityUpdate,
        estimated_send_time: Instant,
    ) -> Result<GameEntity, &str> {
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
            lookback_buffer: VecDeque::new(),
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

    pub(crate) fn advance_all_states(&mut self, until: Instant) {
        for entity in self.entities.values_mut() {
            entity.advance_state(until);
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
        // * Find a way to evaluate the movement of each individual entity in the shader. This will likely require a change
        //    to the vertex format to include velocity, acceleration, and the necessary timing details.
        self.entities
            .iter()
            .flat_map(|(id, entity)| {
                let ent_transforms = entity.as_transform(player_position, time);

                ent_transforms.map(|mat| EntityGeometryDrawCall {
                    model_matrix: mat,
                    model: entity_renderer
                        .get_singleton(entity.class)
                        // class 0 is the fallback
                        .unwrap_or(
                            entity_renderer
                                .get_singleton(0)
                                .unwrap_or(self.fallback_entity.clone()),
                        ),
                })
            })
            .collect()
    }
}
