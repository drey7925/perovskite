use anyhow::{anyhow, bail, Context, Result};
use cgmath::{vec3, ElementWise, InnerSpace, Matrix4, Rad, Vector3, Vector4, Zero};
use perovskite_core::protocol::audio::SoundSource;
use rustc_hash::FxHashMap;
use std::{collections::VecDeque, time::Instant};

use crate::audio::{EngineHandle, ProceduralEntityToken, SOUND_ENTITY_SPATIAL, SOUND_PRESENT};
use crate::client_state::tool_controller::check_intersection_core;
use crate::client_state::ClientState;
use crate::vulkan::{
    block_renderer::{BlockRenderer, CubeExtents, VkCgvBufferGpu},
    entity_renderer::EntityRenderer,
    shaders::entity_geometry::EntityGeometryDrawCall,
};
use perovskite_core::protocol::entities as entities_proto;
use perovskite_core::protocol::entities::TurbulenceAudioModel;
use perovskite_core::protocol::game_rpc::EntityTarget;

#[derive(Copy, Clone, Debug)]
pub(crate) struct EntityMove {
    start_pos: Vector3<f64>,
    velocity: Vector3<f32>,
    acceleration: Vector3<f32>,
    total_time_seconds: f32,
    face_direction: f32,
    pitch: f32,
    start_tick: u64,
    seq: u64,
}

pub(crate) enum ElapsedOrOverflow {
    ElapsedThisMove(f32),
    OverflowIntoNext(f32),
}

impl EntityMove {
    pub(crate) fn elapsed(&self, tick: u64) -> f32 {
        ((tick.saturating_sub(self.start_tick) as f32) / 1_000_000_000.0)
            .clamp(0.0, self.total_time_seconds)
    }

    pub(crate) fn elapsed_or_overflow(&self, tick: u64) -> ElapsedOrOverflow {
        let time = (tick.saturating_sub(self.start_tick) as f32) / 1_000_000_000.0;
        if time < 0.0 {
            ElapsedOrOverflow::ElapsedThisMove(0.0)
        } else if time <= self.total_time_seconds {
            ElapsedOrOverflow::ElapsedThisMove(time)
        } else {
            ElapsedOrOverflow::OverflowIntoNext(time - self.total_time_seconds)
        }
    }

    #[inline]
    pub(crate) fn qproj(&self, time: f32) -> Vector3<f64> {
        vec3(
            qproj(self.start_pos.x, self.velocity.x, self.acceleration.x, time),
            qproj(self.start_pos.y, self.velocity.y, self.acceleration.y, time),
            qproj(self.start_pos.z, self.velocity.z, self.acceleration.z, time),
        )
    }

    #[inline]
    pub(crate) fn instantaneous_velocity(&self, time: f32) -> Vector3<f32> {
        vec3(
            self.velocity.x + (self.acceleration.x * time),
            self.velocity.y + (self.acceleration.y * time),
            self.velocity.z + (self.acceleration.z * time),
        )
    }

    #[inline]
    pub(crate) fn current_seqnum(&self) -> u64 {
        self.seq
    }

    #[inline]
    pub(crate) fn total_time_sec(&self) -> f32 {
        self.total_time_seconds
    }

    /// An *estimate* of the distance covered in this move, *only* to be used for trailing
    /// entities. This only works when acceleration and velocity are in the same direction,
    /// and when acceleration doesn't cause the entity to reverse, but we only support trailing
    /// entities when that property is true.
    ///
    /// To properly implement this, we would need to implement the correct formulae for arc length
    /// on the appropriate parabolic curve in question.
    fn odometer_distance(&self) -> f64 {
        self.partial_odometer_distance(self.total_time_seconds)
    }

    /// An *estimate* of the distance covered in this move, *only* to be used for trailing
    /// entities.
    fn partial_odometer_distance(&self, time: f32) -> f64 {
        let sign_correction = self.acceleration.dot(self.velocity).signum();
        (self.velocity.magnitude() * time
            + 0.5 * sign_correction * self.acceleration.magnitude() * time * time) as f64
    }

    fn position_along_length(&self, move_progress_distance: f64) -> Vector3<f64> {
        // Avoid a NaN/divide by zero for moves without any distance or initial
        // velocity
        let direction = if self.velocity.magnitude2() > 0.00001 {
            self.velocity.normalize()
        } else if self.acceleration.magnitude2() > 0.00001 {
            self.acceleration.normalize()
        } else {
            return self.start_pos;
        };
        self.start_pos + (move_progress_distance * direction.cast().unwrap())
    }

    pub(crate) const fn zero() -> Self {
        EntityMove {
            start_pos: Vector3::new(0.0, 0.0, 0.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            acceleration: Vector3::new(0.0, 0.0, 0.0),
            total_time_seconds: f32::MAX,
            face_direction: 0.0,
            pitch: 0.0,
            start_tick: 0,
            seq: 0,
        }
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
            pitch: value.pitch,
            seq: value.sequence,
            start_tick: value.start_tick,
        })
    }
}

pub(crate) struct GameEntity {
    // todo refine
    // 33 should be enough, server has a queue depth of up to 32
    move_queue: VecDeque<EntityMove>,
    fallback_position: Vector3<f64>,
    last_face_dir: f32,
    last_pitch: f32,

    back_buffer: VecDeque<EntityMove>,

    id: u64,
    class: u32,
    needed_back_buffer_len: f32,
    trailing_entities: Vec<(u32, f32)>,

    audio_token: Option<ProceduralEntityToken>,
    turbulence_audio_model: Option<TurbulenceAudioModel>,
    // debug only
    created: Instant,
}

impl GameEntity {
    pub(crate) fn buffer_debug(&self, tick: u64) -> String {
        let idx = self.move_queue.iter().rposition(|m| m.start_tick < tick);
        format!("now: {tick}: \n")
            + self
                .move_queue
                .iter()
                .map(|m| {
                    format!(
                        "[{} @ {}: {} in {}], ",
                        m.seq,
                        m.start_tick,
                        m.total_time_seconds,
                        m.start_tick as f64 / 1_000_000_000.0 - tick as f64 / 1_000_000_000.0
                    )
                })
                .collect::<String>()
                .as_str()
            + format!("idx: {:?}", idx).as_str()
    }
}

impl GameEntity {
    pub(crate) fn visit_trailing_entities<T>(
        &self,
        tick: u64,
        builder: impl Fn(Vector3<f64>, Rad<f32>, Rad<f32>) -> T,
    ) -> Vec<(T, u32)> {
        // This would be really nice as a generator expression...
        let mut result = vec![];
        result.push((self.as_transform(tick, &builder), self.class));

        // fast path
        if self.trailing_entities.is_empty() {
            return result;
        }
        let cm_distance = match self.move_queue.front() {
            Some(current_move) => {
                let cme = current_move.elapsed(tick);
                let cm_distance = current_move.partial_odometer_distance(cme) as f32;
                // First add the trailing entities that are in the same move
                for &(class, bb_distance) in self.trailing_entities.iter() {
                    if bb_distance < cm_distance {
                        let move_progress_distance = (cm_distance - bb_distance) as f64;
                        let position = current_move.position_along_length(move_progress_distance);
                        result.push((
                            (&builder)(
                                position,
                                Rad(current_move.face_direction),
                                Rad(current_move.pitch),
                            ),
                            class,
                        ));
                    } else {
                        break;
                    }
                }
                cm_distance
            }
            // fast path - if we don't have a current move, we don't have a backbuffer
            // This is ensured by the implementation of pop_until: it will never pop move 0, so
            // the move buffer will never become empty (unless it started and remained empty), and
            // furthermore, no buffer can end up in the backbuffer without first being in the
            // main move buffer
            None => 0.0,
        };

        // acc_distance represents the distance from the start of the current iterated move, to
        // the current entity position
        let mut acc_distance = cm_distance;
        // Now do the same for the backbuffer, back to front
        for m in self.back_buffer.iter().rev() {
            let move_len = m.odometer_distance() as f32;
            // We now generate moves between acc_end and acc_distance
            let acc_end = acc_distance;
            acc_distance += move_len;
            for &(class, bb_distance) in self.trailing_entities.iter() {
                if bb_distance >= acc_end && bb_distance < acc_distance {
                    let move_progress_distance = (acc_distance - bb_distance) as f64;
                    let position = m.position_along_length(move_progress_distance);
                    result.push((
                        builder(position, Rad(m.face_direction), Rad(m.pitch)),
                        class,
                    ))
                }
            }
        }

        result
    }

    pub(crate) fn advance_state(
        &mut self,
        tick_now: u64,
        audio_handle: &EngineHandle,
        volume: f32,
        player_position: Vector3<f64>,
        is_attached: bool,
    ) {
        let any_popped = self.pop_until(tick_now);
        if any_popped {
            log::debug!(
                "remaining buffer: {:?}",
                self.move_queue.back().map(|m| m.start_tick
                    + ((m.total_time_seconds * 1_000_000_000.0) as u64).saturating_sub(tick_now))
            );
            let front = self.move_queue.front();
            if let Some(front) = front {
                let mut flags = SOUND_PRESENT;
                if !is_attached {
                    flags |= SOUND_ENTITY_SPATIAL;
                }

                if let Some(audio) = &self.turbulence_audio_model {
                    let control = crate::audio::ProceduralEntitySoundControlBlock {
                        flags,
                        entity_id: self.id,
                        leading: *front,
                        second: self.move_queue.get(1).copied(),
                        entity_len: self.trailing_entities.last().map(|x| x.1).unwrap_or(0.0),
                        turbulence: crate::audio::TurbulenceSourceControlBlock {
                            volume: audio.volume,
                            volume_attached: audio.volume_attached,
                            lpf_cutoff_hz: audio.lpf_cutoff_hz,
                        },
                        sound_source: SoundSource::SoundsourceWorld,
                    };
                    self.audio_token = audio_handle.update_entity_state(
                        tick_now,
                        player_position,
                        control,
                        self.audio_token,
                    );
                }
            }
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
                if self.needed_back_buffer_len > 0.0 {
                    self.back_buffer.extend(self.move_queue.drain(..idx));
                    // The back buffer now contains moves, in the original sequence. We need to
                    // pop off its front to keep the total length of moves in the back buffer
                    // not much higher than max_back_buffer_length
                    self.pop_backbuffer();
                } else {
                    drop(self.move_queue.drain(..idx));
                }
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
        for m in &update.planned_move {
            self.move_queue.push_back(m.try_into().map_err(|_| {
                dbg!(m);

                "invalid planned move"
            })?);
        }
        if let Some(m) = self.move_queue.back() {
            self.fallback_position = m.qproj(m.total_time_seconds);
        }
        self.pop_until(update_tick);
        if let Some(m) = self.move_queue.back() {
            self.fallback_position = m.qproj(m.total_time_seconds);
        }
        (self.last_face_dir, self.last_pitch) = self
            .move_queue
            .back()
            .map(|m| (m.face_direction, m.pitch))
            .unwrap_or((0.0, 0.0));
        Ok(())
    }

    pub(crate) fn as_transform<T>(
        &self,
        time_tick: u64,
        builder: impl FnOnce(Vector3<f64>, Rad<f32>, Rad<f32>) -> T,
    ) -> T {
        let (pos, face_dir, pitch) = self.position(time_tick);
        builder(pos, face_dir, pitch)
    }

    pub(crate) fn position(&self, time_tick: u64) -> (Vector3<f64>, Rad<f32>, Rad<f32>) {
        let pos = self
            .move_queue
            .front()
            .map(|m| {
                let time = self
                    .move_queue
                    .front()
                    .map(|m| m.elapsed(time_tick))
                    .unwrap_or(0.0);

                m.qproj(time)
            })
            .unwrap_or(self.fallback_position);
        let (face_dir, pitch) = self
            .move_queue
            .front()
            .map(|m| (m.face_direction, m.pitch))
            .unwrap_or((self.last_face_dir, self.last_pitch));
        if !pos.x.is_finite() || !pos.y.is_finite() || !pos.z.is_finite() {
            log::info!(
                "invalid position, move queue contains {:?}",
                self.move_queue
            );
            return (Vector3::zero(), Rad(0.0), Rad(0.0));
        }
        (pos, Rad(face_dir), Rad(pitch))
    }

    pub(crate) fn attach_position(
        &self,
        time_tick: u64,
        entity_manager: &EntityRenderer,
        trailing_entity_index: u32,
    ) -> Option<Vector3<f64>> {
        if trailing_entity_index == 0 {
            let (position, face_dir, pitch) = self.position(time_tick);
            Some(entity_manager.transform_position(self.class, position, face_dir, pitch))
        } else {
            let (trailing_pos, class) = self
                .visit_trailing_entities(time_tick, |pos, face, pitch| (pos, face, pitch))
                .get(trailing_entity_index as usize)
                .copied()?;
            Some(entity_manager.transform_position(
                class,
                trailing_pos.0,
                trailing_pos.1,
                trailing_pos.2,
            ))
        }
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

    pub(crate) fn from_proto(
        update: &entities_proto::EntityUpdate,
        client_state: &ClientState,
    ) -> Result<GameEntity, &'static str> {
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

        let mut trailing_entities: Vec<(u32, f32)> = update
            .trailing_entity
            .iter()
            .map(|te| (te.class, te.distance))
            .collect();
        trailing_entities.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let turbulence_audio = client_state
            .entity_renderer
            .client_info(update.entity_class)
            .and_then(|x| x.turbulence_audio);

        Ok(GameEntity {
            fallback_position: queue
                .back()
                .map(|m: &EntityMove| m.qproj(m.total_time_seconds))
                .unwrap(),
            last_face_dir: queue.back().map(|m| m.face_direction).unwrap(),
            last_pitch: queue.back().map(|m| m.pitch).unwrap(),
            move_queue: queue,
            back_buffer: VecDeque::new(),
            needed_back_buffer_len: update
                .trailing_entity
                .iter()
                .map(|x| x.distance)
                .fold(0.0, |a, b| a.max(b)),
            trailing_entities,
            created: Instant::now(),
            class: update.entity_class,
            id: update.id,
            audio_token: None,
            turbulence_audio_model: turbulence_audio,
        })
    }
    fn pop_backbuffer(&mut self) {
        let mut idx = self.back_buffer.len() - 1;
        let mut distance_so_far = 0.0;
        loop {
            distance_so_far += self.back_buffer[idx].odometer_distance();
            // An LLM hallucination would try to naively decrement idx here
            // This is a classic way to produce an underflow

            if distance_so_far >= self.needed_back_buffer_len as f64 {
                // we have enough distance
                break;
            } else if idx == 0 {
                // we already checked the earliest move in the back buffer, and we didn't find
                // enough distance
                return;
            }
            idx -= 1;
        }
        // then drain, non-inclusively. idx is the index of the last move in the back buffer that
        // ought to be kept
        self.back_buffer.drain(..idx);
    }
}

#[inline]
fn qproj(s: f64, v: f32, a: f32, t: f32) -> f64 {
    s + (v * t + 0.5 * a * t * t) as f64
}

fn build_transform(
    base_position: Vector3<f64>,
    pos: Vector3<f64>,
    face_dir: Rad<f32>,
    pitch: Rad<f32>,
) -> Matrix4<f32> {
    let translation = Matrix4::from_translation(
        (pos - base_position).mul_element_wise(Vector3::new(1., -1., 1.)),
    )
    .cast()
    .unwrap();

    translation * Matrix4::from_angle_y(face_dir) * Matrix4::from_angle_x(pitch)
}

fn build_inverse_transform(
    base_position: Vector3<f64>,
    pos: Vector3<f64>,
    face_dir: Rad<f32>,
    pitch: Rad<f32>,
) -> Matrix4<f32> {
    let translation = Matrix4::from_translation(
        (base_position - pos).mul_element_wise(Vector3::new(1., -1., 1.)),
    )
    .cast()
    .unwrap();

    Matrix4::from_angle_x(-pitch) * Matrix4::from_angle_y(-face_dir) * translation
}

// Minimal stub implementation of the entity manager
// Quite barebones, and will need extensive optimization in the future.
pub(crate) struct EntityState {
    // todo properly encapsulate
    pub(crate) entities: FxHashMap<u64, GameEntity>,

    pub(crate) fallback_entity: VkCgvBufferGpu,

    pub(crate) attached_to_entity: Option<EntityTarget>,
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
            fallback_entity: VkCgvBufferGpu::from_buffers(
                &vtx,
                &idx,
                block_renderer.clone_vk_allocator(),
            )?
            .unwrap(),
            attached_to_entity: None,
        })
    }

    pub(crate) fn advance_all_states(
        &mut self,
        tick_now: u64,
        audio_handle: &EngineHandle,
        player_position: Vector3<f64>,
    ) {
        for entity in self.entities.values_mut() {
            let is_attached = self
                .attached_to_entity
                .is_some_and(|x| x.entity_id == entity.id);
            let volume = if is_attached { 0.125 } else { 1.0 };
            entity.advance_state(tick_now, audio_handle, volume, player_position, is_attached);
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
            .flat_map(|(_id, entity)| {
                self.entity_transforms(player_position, time_tick, entity_renderer, entity)
            })
            .map(|(model_matrix, class)| {
                EntityGeometryDrawCall {
                    model_matrix,
                    model: entity_renderer
                        .get_singleton(class)
                        // class 0 is the fallback
                        .unwrap_or(
                            entity_renderer
                                .get_singleton(0)
                                .unwrap_or(self.fallback_entity.clone()),
                        ),
                }
            })
            .collect()
    }

    pub(crate) fn transforms_for_entity(
        &self,
        player_position: Vector3<f64>,
        time_tick: u64,
        entity_renderer: &EntityRenderer,
        id: u64,
    ) -> Option<impl Iterator<Item = (Matrix4<f32>, u32)>> {
        self.entities
            .get(&id)
            .map(|x| self.entity_transforms(player_position, time_tick, entity_renderer, x))
    }

    fn entity_transforms(
        &self,
        player_position: Vector3<f64>,
        time_tick: u64,
        entity_renderer: &EntityRenderer,
        entity: &GameEntity,
    ) -> impl Iterator<Item = (Matrix4<f32>, u32)> {
        entity
            .visit_trailing_entities(time_tick, |pos, face, pitch| {
                build_transform(player_position, pos, face, pitch)
            })
            .into_iter()
    }

    pub(crate) fn raycast(
        &self,
        player_position: Vector3<f64>,
        pointing_vector: Vector3<f64>,
        time_tick: u64,
        entity_renderer: &EntityRenderer,
        max_distance: f32,
    ) -> Option<(u64, u32, u32, f32)> {
        // Convert pointing vector to homogeneous coordinates
        let pointing_vector = pointing_vector.normalize().cast().unwrap();
        let pointing_vector =
            Vector4::new(pointing_vector.x, pointing_vector.y, pointing_vector.z, 1.0);

        let mut hit = None;
        let mut best_dist = max_distance;
        for (&id, entity) in &self.entities {
            for (trailing_index, (inverse_matrix, class)) in entity
                .visit_trailing_entities(time_tick, |pos, face, pitch| {
                    build_inverse_transform(player_position, pos, face, pitch)
                })
                .into_iter()
                .enumerate()
            {
                let (min, max) = match entity_renderer.mesh_aabb(class) {
                    None => {
                        continue;
                    }
                    Some((min, max)) => (min, max),
                };
                // these transforms incorporate the translation of the entity from model space to
                // player-centered world space (i.e. translated to be around the player, but without
                // the player's rotation)
                //
                // By inverting it, we have a transform that takes us from player-centered world
                // space to model space. We want to raycast the [zero -> pointing_vector] ray
                // segment against the bounding box of the entity which we know in model space.
                let raycast_start = inverse_matrix * Vector4::new(0.0, 0.0, 0.0, 1.0);
                let raycast_end = inverse_matrix * pointing_vector;

                let raycast_start = raycast_start.truncate();
                let raycast_end = raycast_end.truncate();
                let delta = raycast_end - raycast_start;
                let delta_inv = Vector3::new(1.0 / delta.x, -1.0 / delta.y, 1.0 / delta.z);

                let t = check_intersection_core(raycast_start, delta_inv, min, max);
                if let Some(t) = t {
                    if t < best_dist {
                        hit = Some((id, trailing_index as u32, class, t));
                        best_dist = t;
                    }
                }
            }
        }
        hit
    }

    pub(crate) fn remove_entity(&mut self, id: u64, cs: &ClientState) -> Option<GameEntity> {
        let entity = self.entities.remove(&id);
        if let Some(entity) = entity.as_ref() {
            if let Some(token) = entity.audio_token {
                cs.audio.remove_entity_state(token);
            }
        }
        entity
    }
}
