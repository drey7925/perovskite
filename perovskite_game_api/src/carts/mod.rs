use std::{collections::VecDeque, sync::atomic::AtomicU64, time::Instant};

// This is a temporary implementation used while developing the entity system
use anyhow::Result;
use async_trait::async_trait;
use cgmath::{vec2, vec3, InnerSpace, Vector3};
use perovskite_core::{
    block_id::BlockId,
    chat::{ChatMessage, SERVER_ERROR_COLOR, SERVER_MESSAGE_COLOR},
    constants::items::default_item_interaction_rules,
    coordinates::BlockCoordinate,
    protocol::items::item_def::QuantityType,
};
use perovskite_server::game_state::{
    self,
    chat::commands::ChatCommandHandler,
    entities::{
        ContinuationResult, ContinuationResultValue, CoroutineResult, DeferrableResult, Deferral,
        EntityClassId, EntityCoroutine, EntityCoroutineServices, EntityDef, EntityTypeId, Movement,
        ReenterableResult,
    },
    event::{EventInitiator, HandlerContext},
    items::ItemStack,
    GameStateExtension,
};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::{
    blocks::{variants::rotate_nesw_azimuth_to_variant, BlockBuilder, CubeAppearanceBuilder},
    game_builder::{GameBuilderExtension, StaticBlockName, StaticItemName, StaticTextureName},
    include_texture_bytes,
};

use self::signals::automatic_signal_acquire;

mod signals;
mod tracks;

#[derive(Clone)]
struct CartsGameBuilderExtension {
    rail_block: BlockId,
    rail_sw_right: BlockId,
    rail_sw_left: BlockId,
    speedpost_1: BlockId,
    speedpost_2: BlockId,
    speedpost_3: BlockId,
    automatic_signal: BlockId,
    interlocking_signal: BlockId,

    cart_id: EntityClassId,
}
impl Default for CartsGameBuilderExtension {
    fn default() -> Self {
        CartsGameBuilderExtension {
            rail_block: 0.into(),
            rail_sw_right: 0.into(),
            rail_sw_left: 0.into(),
            speedpost_1: 0.into(),
            speedpost_2: 0.into(),
            speedpost_3: 0.into(),
            automatic_signal: 0.into(),
            interlocking_signal: 0.into(),
            cart_id: EntityClassId::new(0),
        }
    }
}

pub fn register_carts(game_builder: &mut crate::game_builder::GameBuilder) -> Result<()> {
    let rail_tex = StaticTextureName("carts:rail");
    let speedpost1_tex = StaticTextureName("carts:speedpost1");
    let speedpost2_tex = StaticTextureName("carts:speedpost2");
    let speedpost3_tex = StaticTextureName("carts:speedpost3");
    include_texture_bytes!(game_builder, rail_tex, "textures/rail.png")?;
    include_texture_bytes!(game_builder, speedpost1_tex, "textures/speedpost_1.png")?;
    include_texture_bytes!(game_builder, speedpost2_tex, "textures/speedpost_2.png")?;
    include_texture_bytes!(game_builder, speedpost3_tex, "textures/speedpost_3.png")?;

    let cart_tex = StaticTextureName("carts:cart_temp");
    include_texture_bytes!(game_builder, cart_tex, "textures/testonly_cart.png")?;

    // TODO: These are fake stand-ins for real switch rails that differ at each point in the switch
    let rail_sw_right_tex = StaticTextureName("carts:rail_sw_right");
    let rail_sw_left_tex = StaticTextureName("carts:rail_sw_left");
    include_texture_bytes!(
        game_builder,
        rail_sw_right_tex,
        "textures/rail_sw_right.png"
    )?;
    include_texture_bytes!(game_builder, rail_sw_left_tex, "textures/rail_sw_left.png")?;

    let rail = game_builder.add_block(
        BlockBuilder::new(StaticBlockName("carts:rail")).set_cube_appearance(
            CubeAppearanceBuilder::new()
                .set_single_texture(rail_tex)
                .set_rotate_laterally(),
        ),
    )?;

    let rail_sw_right = game_builder.add_block(
        BlockBuilder::new(StaticBlockName("carts:rail_sw_right")).set_cube_appearance(
            CubeAppearanceBuilder::new()
                .set_single_texture(rail_sw_right_tex)
                .set_rotate_laterally(),
        ),
    )?;
    let rail_sw_left = game_builder.add_block(
        BlockBuilder::new(StaticBlockName("carts:rail_sw_left")).set_cube_appearance(
            CubeAppearanceBuilder::new()
                .set_single_texture(rail_sw_left_tex)
                .set_rotate_laterally(),
        ),
    )?;

    let speedpost1 = game_builder.add_block(
        BlockBuilder::new(StaticBlockName("carts:speedpost1"))
            .set_cube_appearance(CubeAppearanceBuilder::new().set_single_texture(speedpost1_tex)),
    )?;
    let speedpost2 = game_builder.add_block(
        BlockBuilder::new(StaticBlockName("carts:speedpost2"))
            .set_cube_appearance(CubeAppearanceBuilder::new().set_single_texture(speedpost2_tex)),
    )?;
    let speedpost3 = game_builder.add_block(
        BlockBuilder::new(StaticBlockName("carts:speedpost3"))
            .set_cube_appearance(CubeAppearanceBuilder::new().set_single_texture(speedpost3_tex)),
    )?;

    let cart_id = game_builder.inner.entities_mut().register(EntityDef {
        move_queue_type: game_state::entities::MoveQueueType::Buffer8,
        class_name: "carts:cart_temp".to_string(),
    })?;

    let (automatic_signal, interlocking_signal) = signals::register_signal_block(game_builder)?;

    let rail_block_id = tracks::register_tracks(game_builder)?;

    let ext = game_builder.builder_extension::<CartsGameBuilderExtension>();
    ext.rail_block = rail_block_id;
    ext.rail_sw_right = rail_sw_right.id;
    ext.rail_sw_left = rail_sw_left.id;
    ext.speedpost_1 = speedpost1.id;
    ext.speedpost_2 = speedpost2.id;
    ext.speedpost_3 = speedpost3.id;
    ext.automatic_signal = automatic_signal;
    ext.interlocking_signal = interlocking_signal;

    ext.cart_id = cart_id;

    let ext_clone = ext.clone();

    game_builder
        .inner
        .items_mut()
        .register_item(game_state::items::Item {
            proto: perovskite_core::perovskite::protocol::items::ItemDef {
                short_name: "carts:cart_temp".to_string(),
                display_name: "High-speed minecart".to_string(),
                inventory_texture: Some(cart_tex.into()),
                groups: vec![],
                block_apperance: "".to_string(),
                interaction_rules: default_item_interaction_rules(),
                quantity_type: Some(QuantityType::Stack(256)),
            },
            dig_handler: None,
            tap_handler: None,
            place_handler: Some(Box::new(move |ctx, _placement_coord, anchor, stack| {
                place_cart(ctx, anchor, stack, ext_clone.clone())
            })),
        })?;

    Ok(())
}

impl GameBuilderExtension for CartsGameBuilderExtension {
    fn pre_run(&mut self, server_builder: &mut perovskite_server::server::ServerBuilder) {
        server_builder.add_extension(self.clone());
    }
}
impl GameStateExtension for CartsGameBuilderExtension {}

fn place_cart(
    ctx: &HandlerContext,
    coord: BlockCoordinate,
    stack: &ItemStack,
    config: CartsGameBuilderExtension,
) -> Result<Option<ItemStack>> {
    let block = ctx.game_map().get_block(coord)?;
    let (rail_pos, variant) = if block.equals_ignore_variant(config.rail_block) {
        (coord, block.variant())
    } else {
        ctx.initiator().send_chat_message(
            ChatMessage::new_server_message("Not on rail").with_color(SERVER_ERROR_COLOR),
        )?;
        return Ok(Some(stack.clone()));
    };

    let player_pos = match ctx.initiator().position() {
        Some(pos) => pos,
        None => {
            ctx.initiator().send_chat_message(
                ChatMessage::new_server_message("No initiator position")
                    .with_color(SERVER_ERROR_COLOR),
            )?;
            return Ok(Some(stack.clone()));
        }
    };
    let variant = rotate_nesw_azimuth_to_variant(player_pos.face_direction.0);

    let initial_state = tracks::ScanState::spawn_at(
        rail_pos,
        (variant as u8 + 2) % 4,
        ctx.block_types().get_by_name("carts:rail_tile").unwrap(),
        ctx.game_map(),
    )?;
    let initial_state = match initial_state {
        Some(x) => x,
        None => {
            ctx.initiator().send_chat_message(
                ChatMessage::new_server_message("Can't spawn").with_color(SERVER_ERROR_COLOR),
            )?;
            return Ok(Some(stack.clone()));
        }
    };

    let id = ctx.entities().new_entity_blocking(
        b2vec(rail_pos.try_delta(0, 1, 0).unwrap()),
        Some(Box::pin(CartCoroutine {
            config: config.clone(),
            scheduled_segments: VecDeque::new(),
            unplanned_segments: VecDeque::new(),
            scan_position: rail_pos,
            // Until we encounter a speed post, proceed at minimal safe speed
            last_speed_post_indication: 0.5,
            last_submitted_move_exit_speed: 0.5,
            spawn_time: Instant::now(),
            rail_scan_state: RailScanState {
                valid: true,
                major_angle: variant as u8,
                horizontal_deflection: 0,
                vertical_deflection: 0,
                horizontal_offset: 0,
                vertical_offset: 0,
            },
            scan_state: initial_state,
            cleared_signals: FxHashMap::default(),
            held_signal: None,
        })),
        EntityTypeId {
            class: config.cart_id,
            data: None,
        },
    );

    ctx.initiator().send_chat_message(
        ChatMessage::new_server_message(format!("Spawned cart with id {}", id))
            .with_color(SERVER_MESSAGE_COLOR),
    )?;

    Ok(stack.decrement())
}

fn b2vec(b: BlockCoordinate) -> cgmath::Vector3<f64> {
    cgmath::Vector3::new(b.x as f64, b.y as f64, b.z as f64)
}
fn vec2b(v: cgmath::Vector3<f64>) -> BlockCoordinate {
    BlockCoordinate::new(v.x as i32, v.y as i32, v.z as i32)
}
type Flt = f64;

/// A segment of track where we know we can run
#[derive(Copy, Clone, Debug)]
struct TrackSegment {
    from: Vector3<f64>,
    to: Vector3<f64>,
    // The maximum speed we can run in this track segment
    max_speed: Flt,
    // test only
    seg_id: u64,

    enter_signal: Option<(BlockCoordinate, BlockId)>,
    exit_signal: Option<(BlockCoordinate, BlockId)>,
}
// test only
static NEXT_SEGMENT_ID: AtomicU64 = AtomicU64::new(0);
impl TrackSegment {
    fn manhattan_dist(&self) -> f64 {
        (self.from.x - self.to.x).abs()
            + (self.from.y - self.to.y).abs()
            + (self.from.z - self.to.z).abs()
    }
    fn distance(&self) -> Flt {
        let dx = self.from.x as Flt - self.to.x as Flt;
        let dy = self.from.y as Flt - self.to.y as Flt;
        let dz = self.from.z as Flt - self.to.z as Flt;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    fn split_at_offset(&self, offset: Flt) -> (Self, Option<Self>) {
        if offset > 0.0 && offset < self.distance() {
            let split_point = self.from + (self.to - self.from).normalize() * offset;
            (
                Self {
                    from: self.from,
                    to: split_point,
                    max_speed: self.max_speed,
                    seg_id: self.seg_id,
                    enter_signal: self.enter_signal,
                    exit_signal: None,
                },
                Some(Self {
                    from: split_point,
                    to: self.to,
                    max_speed: self.max_speed,
                    seg_id: self.seg_id,
                    enter_signal: None,
                    exit_signal: self.exit_signal,
                }),
            )
        } else {
            (self.clone(), None)
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct ScheduledSegment {
    segment: TrackSegment,
    speed: Flt,
    acceleration: Flt,
    move_time: Flt,
}
impl ScheduledSegment {
    fn to_movement(self) -> Option<Movement> {
        let displacement = self.segment.to - self.segment.from;
        if displacement.magnitude() < 0.01 {
            None
        } else {
            let displacement_f32 = displacement.cast().unwrap();
            Some(Movement {
                velocity: self.speed as f32 * displacement_f32.normalize(),
                acceleration: self.acceleration as f32 * displacement_f32.normalize(),
                // TODO check this angle, might be off by pi/2 radians
                face_direction: f32::atan2(displacement_f32.x, displacement_f32.z),
                move_time: self.move_time as f32,
            })
        }
    }
}

const MAX_ACCEL: Flt = 8.0;
struct CartCoroutine {
    config: CartsGameBuilderExtension,
    // Segments where we've already calculated a braking curve. (clearance, starting speed, acceleration, time)
    scheduled_segments: VecDeque<ScheduledSegment>,
    // Segments where we don't yet have a braking curve
    unplanned_segments: VecDeque<TrackSegment>,
    // The current coordinate where we're going to scan next
    scan_position: BlockCoordinate,
    // The rail scan state, indicating which direction we're scanning in
    rail_scan_state: RailScanState,
    // The last speed post we encountered while scanning
    last_speed_post_indication: Flt,
    // The last velocity we already delivered to the entity system
    last_submitted_move_exit_speed: Flt,
    // debug only
    spawn_time: Instant,
    // Track scan state
    scan_state: tracks::ScanState,
    // TODO improve this as needed
    cleared_signals: FxHashMap<BlockCoordinate, BlockId>,
    held_signal: Option<(BlockCoordinate, BlockId)>,
}
impl EntityCoroutine for CartCoroutine {
    fn plan_move(
        mut self: std::pin::Pin<&mut Self>,
        services: &perovskite_server::game_state::entities::EntityCoroutineServices<'_>,
        current_position: cgmath::Vector3<f64>,
        whence: cgmath::Vector3<f64>,
        when: f32,
        queue_space: usize,
    ) -> perovskite_server::game_state::entities::CoroutineResult {
        tracing::debug!(
            "{:?} ========== planning {} seconds in advance",
            self.spawn_time.elapsed(),
            when
        );
        // debug only
        let mut step_count = 0.0;
        for seg in self.scheduled_segments.iter() {
            step_count += seg.segment.distance();
        }
        for seg in self.unplanned_segments.iter() {
            step_count += seg.distance();
        }
        tracing::debug!(
            "cart coro: step_count = {}. {} in braking curve, {} unplanned",
            step_count,
            self.scheduled_segments.len(),
            self.unplanned_segments.len()
        );

        let maybe_deferral = self.scan_tracks_new(services, 1024.0);
        if let Some(deferral) = maybe_deferral {
            return deferral;
        }

        let schedulable_segments = self.promote_schedulable(when, queue_space);

        self.schedule_segments(schedulable_segments);

        if !self.scheduled_segments.iter().any(|x| x.move_time > 0.001) {
            // No schedulable segments
            // Scan every second, trying to start again.
            tracing::debug!("No schedulable segments");
            return perovskite_server::game_state::entities::CoroutineResult::Successful(
                perovskite_server::game_state::entities::EntityMoveDecision::AskAgainLaterFlexible(
                    0.5..1.0,
                ),
            );
        }

        let mut returned_moves = Vec::with_capacity(queue_space);
        let mut returned_moves_time = 0.0;

        while self.scheduled_segments.len() > 0 && returned_moves.len() < queue_space {
            let segment = self.scheduled_segments.pop_front().unwrap();
            self.last_submitted_move_exit_speed =
                segment.speed + (segment.acceleration * segment.move_time);

            tracing::debug!(
                "returning a movement, speed is {} -> {}, accel is {}, time is {}",
                segment.speed,
                self.last_submitted_move_exit_speed,
                segment.acceleration,
                segment.move_time
            );
            tracing::debug!(
                "seg position: {:?}, actually {:?}",
                segment.segment.from,
                whence
            );

            if let Some((enter_signal_coord, enter_signal_block)) = segment.segment.enter_signal {
                tracing::debug!(
                    "entering block in {} + {} seconds at {:?}",
                    returned_moves_time,
                    when,
                    enter_signal_coord
                );
                services.spawn_async(
                    std::time::Duration::from_secs_f64(returned_moves_time + when as f64),
                    move |ctx| {
                        ctx.game_map()
                            .mutate_block_atomically(enter_signal_coord, |b, _ext| {
                                tracing::debug!("entering block");
                                signals::automatic_signal_enter_block(
                                    enter_signal_coord,
                                    b,
                                    enter_signal_block,
                                );
                                Ok(())
                            })
                            .unwrap();
                    },
                );
            }
            returned_moves_time += segment.move_time;
            if let Some((exit_signal_coord, exit_signal_block)) = segment.segment.exit_signal {
                let exit_speed = segment.speed + (segment.acceleration * segment.move_time);
                const SIGNAL_BUFFER_DISTANCE: f64 = 1.0;
                let extra_delay = (SIGNAL_BUFFER_DISTANCE / exit_speed).clamp(0.0, 1.0);
                tracing::debug!(
                    "exiting block in {} + {} + {} seconds at {:?}",
                    returned_moves_time,
                    when,
                    extra_delay,
                    exit_signal_coord
                );

                services.spawn_async(
                    std::time::Duration::from_secs_f64(
                        returned_moves_time + when as f64 + extra_delay,
                    ),
                    move |ctx| {
                        ctx.game_map()
                            .mutate_block_atomically(exit_signal_coord, |b, _ext| {
                                signals::automatic_signal_release(
                                    exit_signal_coord,
                                    b,
                                    exit_signal_block,
                                );
                                Ok(())
                            })
                            .unwrap();
                    },
                );
            }

            if let Some(movement) = segment.to_movement() {
                returned_moves.push(movement);
            } else {
                tracing::warn!("Segment {:?} had no movement", segment);
            }
        }
        return perovskite_server::game_state::entities::CoroutineResult::Successful(
            perovskite_server::game_state::entities::EntityMoveDecision::QueueUpMultiple(
                returned_moves,
            ),
        );
    }
    fn continuation(
        mut self: std::pin::Pin<&mut Self>,
        services: &perovskite_server::game_state::entities::EntityCoroutineServices<'_>,
        current_position: cgmath::Vector3<f64>,
        whence: cgmath::Vector3<f64>,
        when: f32,
        queue_space: usize,
        continuation_result: ContinuationResult,
    ) -> CoroutineResult {
        if continuation_result.tag == CONTINUATION_TAG_SIGNAL {
            match continuation_result.value {
                ContinuationResultValue::GetBlock(block_id, coord) => {
                    self.cleared_signals.insert(coord, block_id);
                }
                _ => {
                    log::error!(
                        "Unhandled continuation value with signal tag {:?}",
                        continuation_result.value
                    );
                }
            };
        }

        self.plan_move(services, current_position, whence, when, queue_space)
    }
}

const CONTINUATION_TAG_SIGNAL: u32 = 0x516ea1;

impl CartCoroutine {
    fn signal_coordinates(
        &self,
        scan_position: BlockCoordinate,
        rail_state: &RailScanState,
    ) -> SmallVec<[BlockCoordinate; 2]> {
        let mut coords = SmallVec::new();

        if let Some((x, y, z)) = rail_state.wayside_signal_delta() {
            if let Some(coord) = scan_position.try_delta(x, y, z) {
                coords.push(coord);
            }
        }
        if let Some(coord) = scan_position.try_delta(0, 3, 0) {
            // overhead signal
            coords.push(coord);
        }

        coords
    }

    fn parse_signal(
        &self,
        services: &EntityCoroutineServices<'_>,
        signal_coord: BlockCoordinate,
        signal_block: BlockId,
    ) -> ReenterableResult<SignalResult> {
        if signal_block == self.config.speedpost_1 {
            SignalResult::SpeedRestriction(3.0).into()
        } else if signal_block == self.config.speedpost_2 {
            SignalResult::SpeedRestriction(30.0).into()
        } else if signal_block == self.config.speedpost_3 {
            SignalResult::SpeedRestriction(90.0).into()
        } else if signal_block.equals_ignore_variant(self.config.automatic_signal) {
            let rotation = signal_block.variant() & 0b11;
            if !self.scan_state.signal_rotation_ok(rotation) {
                println!("Skipping signal at {:?}", signal_coord);
                return SignalResult::Stop.into();
            }

            let outcome = services.mutate_block_atomically(signal_coord, move |block_id, _| {
                tracing::debug!("trying to acquire the signal");
                Ok(automatic_signal_acquire(
                    signal_coord,
                    block_id,
                    signal_block,
                ))
            });
            match outcome {
                DeferrableResult::AvailableNow(Ok(result)) => match result {
                    signals::AutomaticSignalOutcome::Acquired => {
                        tracing::debug!("acquired inline");
                        SignalResult::Permissive.into()
                    }
                    signals::AutomaticSignalOutcome::Contended => SignalResult::Stop.into(),
                    signals::AutomaticSignalOutcome::InvalidSignal => SignalResult::NoSignal.into(),
                },
                DeferrableResult::AvailableNow(Err(e)) => {
                    tracing::error!("Failed to parse signal: {}", e);
                    SignalResult::NoSignal.into()
                }
                DeferrableResult::Deferred(d) => {
                    ReenterableResult::Deferred(d.map(move |x| match x {
                        Ok(x) => match x {
                            signals::AutomaticSignalOutcome::Acquired => {
                                tracing::debug!("acquired deferred");
                                ContinuationResultValue::GetBlock(signal_block, signal_coord).into()
                            }
                            signals::AutomaticSignalOutcome::Contended => {
                                ContinuationResultValue::GetBlock(0.into(), signal_coord).into()
                            }
                            signals::AutomaticSignalOutcome::InvalidSignal => {
                                tracing::debug!("Invalid signal 540");
                                ContinuationResultValue::None
                            }
                        },
                        Err(e) => {
                            tracing::error!("Failed to parse signal: {}", e);
                            ContinuationResultValue::None
                        }
                    }))
                }
            }
        } else {
            SignalResult::NoSignal.into()
        }
    }

    fn scan_tracks_new(
        &mut self,
        services: &EntityCoroutineServices<'_>,
        max_steps_ahead: f64,
    ) -> Option<CoroutineResult> {
        tracing::debug!(
            "starting scan at {:?}: {:?}",
            self.scan_position,
            self.rail_scan_state
        );
        let mut steps = 0.0;
        for seg in self.scheduled_segments.iter() {
            steps += seg.segment.distance();
        }
        for seg in self.unplanned_segments.iter() {
            steps += seg.distance();
        }

        if self.unplanned_segments.is_empty() {
            tracing::debug!("unplanned segments empty, adding new");
            let empty_segment = TrackSegment {
                from: self.scan_state.vec_coord,
                to: self.scan_state.vec_coord,
                max_speed: self
                    .last_speed_post_indication
                    .min(self.scan_state.allowable_speed as f64),
                seg_id: NEXT_SEGMENT_ID.fetch_add(10, std::sync::atomic::Ordering::Relaxed),
                enter_signal: None,
                exit_signal: None,
            };
            self.unplanned_segments.push_back(empty_segment);
        }

        let block_getter = |coord| services.get_block(coord);

        'scan_loop: while steps < max_steps_ahead {
            // Precondition: self.scan_state is valid
            let new_state = match self.scan_state.advance_verbose(false, block_getter) {
                Ok(tracks::ScanOutcome::Success(state)) => state,
                Ok(tracks::ScanOutcome::Failure) => break 'scan_loop,
                Ok(tracks::ScanOutcome::NotOnTrack) => {
                    tracing::warn!("Not on track at {:?}", self.scan_state.vec_coord);
                    break 'scan_loop;
                }
                Ok(tracks::ScanOutcome::Deferral(d)) => {
                    tracing::debug!("track scan deferral");
                    return Some(d.defer_and_reinvoke(1));
                }
                Err(e) => {
                    tracing::error!("Failed to parse track: {}", e);
                    break 'scan_loop;
                }
            };

            // TODO check the facing direction of the signal against data from the track tile table, and also check multiple possible mounting positions.
            let signal_coord = new_state.block_coord.try_delta(0, 2, 0);
            if let Some(signal_coord) = signal_coord {
                if let Some(block_id) = self.cleared_signals.remove(&signal_coord) {
                    tracing::debug!("cached clear signal at {:?}", signal_coord);
                    if block_id.equals_ignore_variant(0.into()) {
                        // contended signal
                        break 'scan_loop;
                    } else {
                        tracing::debug!("Cached permissive signal at {:?}", new_state.vec_coord);
                        self.unplanned_segments.back_mut().unwrap().exit_signal =
                            self.held_signal.take();
                        self.start_new_unplanned_segment();
                        self.unplanned_segments.back_mut().unwrap().enter_signal =
                            Some((signal_coord, block_id));
                        self.held_signal = Some((signal_coord, block_id))
                        // break 'signal_loop,
                    }
                }

                let signal_block = match services.get_block(signal_coord) {
                    DeferrableResult::AvailableNow(x) => x.unwrap(),
                    DeferrableResult::Deferred(d) => {
                        tracing::debug!("signal lookup deferral");
                        return Some(d.defer_and_reinvoke(1));
                    }
                };

                let signal_result = match self.parse_signal(services, signal_coord, signal_block) {
                    ReenterableResult::AvailableNow(x) => x,
                    ReenterableResult::Deferred(d) => {
                        tracing::debug!("signal parse deferral");
                        return Some(d.defer_and_reinvoke(2));
                    }
                };
                match signal_result {
                    SignalResult::Stop => break 'scan_loop,
                    SignalResult::Permissive => {
                        tracing::debug!("Permissive signal at {:?}", new_state.vec_coord);
                        self.unplanned_segments.back_mut().unwrap().exit_signal =
                            self.held_signal.take();
                        self.start_new_unplanned_segment();
                        self.unplanned_segments.back_mut().unwrap().enter_signal =
                            Some((signal_coord, signal_block));
                        self.held_signal = Some((signal_coord, signal_block))
                        // break 'signal_loop,
                    }
                    SignalResult::SpeedRestriction(speed) => {
                        self.last_speed_post_indication = speed as Flt;
                        //break 'signal_loop;
                    }
                    SignalResult::NoSignal => {} // continue 'signal_loop,
                }
            }
            steps += 1.0;

            let last_move = self.unplanned_segments.back().unwrap();
            // We got success, so we're at a new position.
            let new_delta = new_state.vec_coord - last_move.to;
            assert!(new_delta.magnitude() > 0.001);

            let last_move_delta = last_move.to - last_move.from;

            let effective_speed = self
                .last_speed_post_indication
                .min(new_state.allowable_speed as f64);

            if last_move_delta.magnitude() > 256.0 {
                tracing::debug!(
                    "Splitting a segment due to length; prev length was {}",
                    last_move_delta.magnitude()
                );
                self.start_new_unplanned_segment();
            } else if last_move_delta.dot(new_delta)
                / (last_move_delta.magnitude() * new_delta.magnitude())
                < 0.999999
            {
                tracing::debug!(
                    "Splitting a segment due to angle; prev length was {}, cos similiarity was {}",
                    last_move_delta.magnitude(),
                    last_move_delta.dot(new_delta)
                        / (last_move_delta.magnitude() * new_delta.magnitude())
                );
                self.start_new_unplanned_segment();
            } else if effective_speed != last_move.max_speed {
                tracing::debug!("Splitting a segment due to effective speed; prev length was {}, speed changing {} -> {}", last_move_delta.magnitude(), last_move.max_speed, effective_speed);
                self.start_new_unplanned_segment();
            }
            let last_seg_mut = self.unplanned_segments.back_mut().unwrap();
            last_seg_mut.to = new_state.vec_coord;
            last_seg_mut.max_speed = effective_speed;
            self.scan_state = new_state;
        }

        None
    }

    fn promote_schedulable(
        &mut self,
        when: f32,
        queue_space: usize,
    ) -> VecDeque<(TrackSegment, Flt)> {
        // NOTE: While this function iterates in reverse order (end of track -> nearest unscheduled segment),
        // comments and names refer to segments in *track* order. So in reality, the iteration is from the
        // furthest-away segment to the closest segment.

        // Prepare a braking curve for the segments we have, looking for the first moment
        // where we're no longer limited by the braking curve while scanning backwards from
        // stop to start
        //
        // "No longer limited" is defined as "there's at least one upcoming track segment where we're
        // limited by track speed" - even if we're on a braking curve approaching that track segment,
        // it doesn't matter if the track scan detects more tracks or a signal clears - the braking curve
        // up to that limiting track segment is unchanged

        // The maximum speed at which we're allowed to exit the current segment.
        // Updated to be the max entry speed of the following segment.
        // It's 0.0 to start, because there is no further segment detected by track scan (or that segment
        // cannot be entered due to a restrictive signal or a switch set against us)
        let mut max_exit_speed: Flt = 0.0;

        // If this is true, we've hit a segment where we were limited by track speed.
        let mut unconditionally_schedulable = false;
        let mut schedulable_segments = VecDeque::new();

        let estimated_segments_remaining = (9 - queue_space) + self.scheduled_segments.len();

        for (idx, seg) in self.unplanned_segments.iter().enumerate().rev() {
            tracing::debug!("> segment idx {}, {:?}", idx, seg);
            // The entrance speed, if we just consider the max acceleration

            max_exit_speed = max_exit_speed.min(seg.max_speed);

            let mut max_entry_speed =
                (max_exit_speed * max_exit_speed + 2.0 * MAX_ACCEL * seg.distance()).sqrt();
            tracing::debug!(
                "> unplanned segment {:?} w/ max exit speed {}, max entry speed {}",
                seg,
                max_exit_speed,
                max_entry_speed
            );
            let limited_by_seg_speed = max_entry_speed >= seg.max_speed;

            if limited_by_seg_speed {
                max_entry_speed = seg.max_speed;
                // This segment itself is not yet schedulable, so we don't set unconditionally_schedulable yet
            }
            // If we encountered a further segment where we were limited by track speed, we can schedule this one
            if unconditionally_schedulable {
                schedulable_segments.push_front((*seg, max_exit_speed));
                tracing::debug!(
                    "> seg_schedulable = {}, seg = {:?}",
                    unconditionally_schedulable,
                    seg.seg_id
                );
            } else if (estimated_segments_remaining < 2 || when < 1.0) && idx == 0 {
                // We're low on cached moves, so let's schedule this segment
                schedulable_segments.push_front((*seg, max_exit_speed));
                tracing::debug!(
                    "> panic scheduling! seg_schedulable = {}, seg = {:?}",
                    unconditionally_schedulable,
                    seg.seg_id
                );
                // We don't actually need to set this, but this guards against refactorings that remove the idx==0 conditions
                unconditionally_schedulable = true;
            }

            if limited_by_seg_speed {
                unconditionally_schedulable = true;
            }
            // The max exit speed of the preeding segment is the same as the max entry speed of the current segment
            // (after limiting for track speed)
            max_exit_speed = max_entry_speed;
        }

        // We scheduled some prefix of the unscheduled segments, on a 1:1 basis
        // Remove that prefix
        for _ in 0..schedulable_segments.len() {
            self.unplanned_segments.pop_front().unwrap();
        }

        // front is the first remaining unplanned segment
        if let Some(front) = self.unplanned_segments.front() {
            // back is the last segment we planned
            if let Some(back) = schedulable_segments.back() {
                // They ought to match up
                assert!((front.from - back.0.to).magnitude() < 0.01);
            }
        }

        schedulable_segments
    }

    fn start_new_unplanned_segment(&mut self) {
        let prev_segment = self.unplanned_segments.back().unwrap();
        if prev_segment.distance() > 0.01 || prev_segment.exit_signal.is_some() {
            tracing::debug!("new segment starting from {:?}", prev_segment.to);
            let empty_segment = TrackSegment {
                from: prev_segment.to,
                to: prev_segment.to,
                max_speed: self.last_speed_post_indication,
                seg_id: NEXT_SEGMENT_ID.fetch_add(10, std::sync::atomic::Ordering::Relaxed),
                enter_signal: None,
                exit_signal: None,
            };
            self.unplanned_segments.push_back(empty_segment);
        } else {
            tracing::debug!("old segment was empty, not starting new one");
        }
    }

    fn schedule_segments(&mut self, schedulable_segments: VecDeque<(TrackSegment, Flt)>) {
        // Start with the last segment's exit speed. Note that if we sent all of our segments
        // to the entity system, self.scheduled_segments would be empty, so we would use
        // the speed stored in self.last_submitted_move_exit_speed instead.
        let mut last_segment_exit_speed = self
            .scheduled_segments
            .back()
            .map(|x| (x.speed + (x.acceleration * x.move_time)))
            .unwrap_or(self.last_submitted_move_exit_speed);

        // Each segment contributes one or more moves to the internal move queue
        for (seg, brake_curve_exit_speed) in schedulable_segments.into_iter() {
            last_segment_exit_speed =
                self.schedule_single_segment(seg, last_segment_exit_speed, brake_curve_exit_speed);
        }
    }

    /// Schedules a single segment as one or more moves. Returns the actual speed at the exit of the segment
    ///
    /// Args:
    ///     seg: The segment
    ///     brake_curve_exit_speed: The speed at which we must exit this segment to satisfy the brake curve
    ///     enter_speed: The speed at which we enter this segment (i.e. the speed at which we exited the previous segment)
    fn schedule_single_segment(
        &mut self,
        seg: TrackSegment,
        mut enter_speed: f64,
        brake_curve_exit_speed: f64,
    ) -> f64 {
        tracing::debug!(
            "schedule_single_segment({:?}, {} -> {})",
            seg,
            enter_speed,
            brake_curve_exit_speed,
        );
        if seg.distance() < 1e-3 {
            self.scheduled_segments.push_back(ScheduledSegment {
                segment: seg,
                speed: enter_speed,
                acceleration: 0.0,
                move_time: 0.0,
            });
            return enter_speed;
        }
        // See if we need to accelerate at the entrance of the segment, to reach the intended movement speed
        let mut entrance_accel_distance = if (seg.max_speed - enter_speed).abs() < 1e-3 {
            0.0
        } else {
            (seg.max_speed * seg.max_speed - enter_speed * enter_speed) / (2.0 * MAX_ACCEL)
        };
        // and likewise at the exit
        let mut exit_accel_distance = if (seg.max_speed - brake_curve_exit_speed).abs() < 1e-3 {
            0.0
        } else {
            (seg.max_speed * seg.max_speed - brake_curve_exit_speed * brake_curve_exit_speed)
                / (2.0 * MAX_ACCEL)
        };

        // flush acceleration distances to 0 if they're too small
        // Note that these aren't simply checks for ==0 (with floating error tolerance),
        // we're actually checking for small nonzero segments that we don't really care about
        if entrance_accel_distance < 1e-3 {
            entrance_accel_distance = 0.0;
        }
        if exit_accel_distance < 1e-3 {
            exit_accel_distance = 0.0;
        }

        if enter_speed > seg.max_speed {
            tracing::warn!(
                "enter_speed {} > seg.max_speed {}, seg = {:?}",
                enter_speed,
                seg.max_speed,
                seg.seg_id
            );
            // TODO see whether this happens often and the difference is greater than numerical error.
            // This really shouldn't happen, but might possibly start to happen under some complex signalling cases
            enter_speed = seg.max_speed;
            // Note that we need to ensure that enter_speed <= seg.max_speed, otherwise
            // later code might get confused.
        }

        let seg_distance = seg.distance();
        tracing::debug!(
            ">> seg_distance {} seg.max_speed {}",
            seg_distance,
            seg.max_speed
        );
        tracing::debug!(
            ">> accel distances: entrance {} exit {}",
            entrance_accel_distance,
            exit_accel_distance
        );
        let total_accel_distance = entrance_accel_distance + exit_accel_distance;

        // Simple case: no need for any acceleration.
        if entrance_accel_distance == 0.0 && exit_accel_distance == 0.0 {
            tracing::debug!(
                ">> no acceleration required. speed {} time {}",
                seg.max_speed,
                seg_distance
            );
            self.scheduled_segments.push_back(ScheduledSegment {
                segment: seg,
                speed: seg.max_speed,
                acceleration: 0.0,
                move_time: seg_distance / seg.max_speed,
            });
            seg.max_speed
        } else if entrance_accel_distance > 0.0 && exit_accel_distance == 0.0 {
            // Case 2: We're cleared to exit at track speed, but we start at a lower speed.
            // We'll accelerate right away and then spend the rest of the segment running at full speed, if able

            if entrance_accel_distance < seg_distance {
                // 2a - we have enough distance to finish the acceleration

                let remaining_distance = seg_distance - entrance_accel_distance;

                let (split_before, split_after) = seg.split_at_offset(entrance_accel_distance);
                let split_after = split_after.unwrap();

                self.scheduled_segments.push_back(ScheduledSegment {
                    segment: split_before,
                    speed: enter_speed,
                    acceleration: MAX_ACCEL,
                    move_time: (seg.max_speed - enter_speed) / MAX_ACCEL,
                });

                self.scheduled_segments.push_back(ScheduledSegment {
                    segment: split_after,
                    speed: seg.max_speed,
                    acceleration: 0.0,
                    move_time: remaining_distance / seg.max_speed,
                });
                seg.max_speed
            } else {
                // We don't have enough distance to finish the acceleration
                let actual_exit_speed =
                    (enter_speed * enter_speed + 2.0 * MAX_ACCEL * seg_distance).sqrt();

                self.scheduled_segments.push_back(ScheduledSegment {
                    segment: seg,
                    speed: enter_speed,
                    acceleration: MAX_ACCEL,
                    move_time: (actual_exit_speed - enter_speed) / MAX_ACCEL,
                });
                actual_exit_speed
            }
        } else if entrance_accel_distance == 0.0 && exit_accel_distance > 0.0 {
            // Case 3: We entered at track speed, but we need to slow down before exiting
            if exit_accel_distance > seg_distance {
                // This shouldn't happen; we're braking the cart with more acceleration than
                // we used when planning the brake curve

                let new_accel = (enter_speed * enter_speed
                    - brake_curve_exit_speed * brake_curve_exit_speed)
                    / (2.0 * seg_distance);
                assert!(new_accel > 0.0);

                tracing::warn!(
                    "exit_accel_distance {} > seg_distance {}, decelerating {} => {} using accel of {}",
                    exit_accel_distance,
                    seg_distance,
                    enter_speed,
                    brake_curve_exit_speed,
                    new_accel
                );

                self.scheduled_segments.push_back(ScheduledSegment {
                    segment: seg,
                    speed: enter_speed,
                    acceleration: -new_accel,
                    move_time: (enter_speed - brake_curve_exit_speed) / new_accel,
                });
                brake_curve_exit_speed
            } else {
                // We have enough distance to finish the deceleration
                let remaining_distance = seg_distance - exit_accel_distance;
                let (split_before, split_after) = seg.split_at_offset(remaining_distance);

                self.scheduled_segments.push_back(ScheduledSegment {
                    segment: split_before,
                    speed: enter_speed,
                    acceleration: 0.0,
                    move_time: (remaining_distance) / enter_speed,
                });

                self.scheduled_segments.push_back(ScheduledSegment {
                    segment: split_after.unwrap(),
                    speed: enter_speed,
                    acceleration: -MAX_ACCEL,
                    move_time: (seg.max_speed - brake_curve_exit_speed) / MAX_ACCEL,
                });
                brake_curve_exit_speed
            }
        } else if total_accel_distance > seg_distance {
            // We can't get to the max track speed in this segment, so we'll just
            // accelerate as much as we can
            let corrected_acc_distance = (2.0 * MAX_ACCEL * seg_distance
                - enter_speed * enter_speed
                + brake_curve_exit_speed * brake_curve_exit_speed)
                / (4.0 * MAX_ACCEL);
            let decel_distance = seg_distance - corrected_acc_distance;
            tracing::debug!(
                ">> can't reach max speed, accel distance {} > seg distance {}. Speed {} => {}",
                total_accel_distance,
                seg_distance,
                enter_speed,
                brake_curve_exit_speed
            );
            tracing::debug!(
                ">>> corrected_acc_distance {} decel_distance {}",
                corrected_acc_distance,
                decel_distance
            );

            // three sub cases:
            // * Either the max point is within the segment (i.e. both distances are positive)
            // * Or the max point is right of the segment, meaning that we won't hit a speed cap.
            // * Or the max point is left of the segment, meaning that we need to brake harder than
            //     allowed
            if corrected_acc_distance > 0.1 && decel_distance > 0.1 {
                let speed_left =
                    (enter_speed * enter_speed + 2.0 * MAX_ACCEL * corrected_acc_distance).sqrt();
                let speed_right = (brake_curve_exit_speed * brake_curve_exit_speed
                    + 2.0 * MAX_ACCEL * decel_distance)
                    .sqrt();
                assert!(
                    (speed_left - speed_right).abs() < 1e-3,
                    "{} != {}",
                    speed_left,
                    speed_right
                );
                tracing::debug!(
                    ">> can't reach max speed, speed left = {} after {}, speed right = {} after {}",
                    speed_left,
                    corrected_acc_distance,
                    speed_right,
                    decel_distance
                );

                assert!(speed_left - seg.max_speed < 1e-3);
                tracing::debug!(
                    ">> subcase 1, distances {} + {}, reaching {}",
                    corrected_acc_distance,
                    decel_distance,
                    speed_left
                );

                let (split_before, split_after) = seg.split_at_offset(corrected_acc_distance);

                self.scheduled_segments.push_back(ScheduledSegment {
                    segment: split_before,
                    speed: enter_speed,
                    acceleration: MAX_ACCEL,
                    move_time: (speed_left - enter_speed) / MAX_ACCEL,
                });

                self.scheduled_segments.push_back(ScheduledSegment {
                    segment: split_after.unwrap(),
                    speed: speed_right,
                    acceleration: -MAX_ACCEL,
                    move_time: (speed_right - brake_curve_exit_speed) / MAX_ACCEL,
                });
                return brake_curve_exit_speed;
            } else if corrected_acc_distance > 0.1 {
                // decel distance is negligible or negative
                let achieved_speed =
                    (enter_speed * enter_speed + 2.0 * MAX_ACCEL * seg_distance).sqrt();

                tracing::debug!(
                    ">> subcase 2, distances {} + {}, reaching {}",
                    corrected_acc_distance,
                    decel_distance,
                    achieved_speed
                );
                self.scheduled_segments.push_back(ScheduledSegment {
                    segment: seg,
                    speed: enter_speed,
                    acceleration: MAX_ACCEL,
                    move_time: (achieved_speed - enter_speed) / MAX_ACCEL,
                });
                achieved_speed
            } else if decel_distance > 0.1 {
                tracing::debug!(
                    ">> subcase 3 (emergency braking), distances {} + {}, reaching {}",
                    corrected_acc_distance,
                    decel_distance,
                    brake_curve_exit_speed
                );
                assert!(enter_speed > brake_curve_exit_speed);
                let new_accel = (enter_speed * enter_speed
                    - brake_curve_exit_speed * brake_curve_exit_speed)
                    / (2.0 * seg_distance);
                assert!(new_accel > 0.0);

                tracing::warn!(
                    "exit_accel_distance {} > seg_distance {}, decelerating {} => {} using accel of {}",
                    exit_accel_distance,
                    seg_distance,
                    enter_speed,
                    brake_curve_exit_speed,
                    new_accel
                );

                self.scheduled_segments.push_back(ScheduledSegment {
                    segment: seg,
                    speed: enter_speed,
                    acceleration: -new_accel,
                    move_time: (enter_speed - brake_curve_exit_speed) / new_accel,
                });
                brake_curve_exit_speed
            } else {
                panic!(
                    "Invalid subcase, distances {} + {}",
                    corrected_acc_distance, decel_distance
                );
            }
        } else {
            // We have enough distance to accelerate to max speed, cruise, and then decelerate

            let remaining_distance = seg_distance - total_accel_distance;
            tracing::debug!(
                ">> can reach max speed, accel distance {} > seg distance {}",
                total_accel_distance,
                seg_distance
            );
            tracing::debug!(
                "Will accel for {}, cruise for {}, decel for {}",
                (seg.max_speed - enter_speed) / MAX_ACCEL,
                remaining_distance / seg.max_speed,
                (seg.max_speed - brake_curve_exit_speed) / MAX_ACCEL
            );
            // The acceleration segment
            self.scheduled_segments.push_back(ScheduledSegment {
                segment: seg,
                speed: enter_speed,
                acceleration: MAX_ACCEL,
                move_time: (seg.max_speed - enter_speed) / MAX_ACCEL,
            });
            // The cruise segment
            self.scheduled_segments.push_back(ScheduledSegment {
                segment: seg,
                speed: seg.max_speed,
                acceleration: 0.0,
                move_time: remaining_distance / seg.max_speed,
            });
            // The deceleration segment
            self.scheduled_segments.push_back(ScheduledSegment {
                segment: seg,
                speed: seg.max_speed,
                acceleration: -MAX_ACCEL,
                move_time: (seg.max_speed - brake_curve_exit_speed) / MAX_ACCEL,
            });

            brake_curve_exit_speed
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum SignalResult {
    /// Signal/sign instructs all carts to stop
    Stop,
    /// Signal/sign allows all carts to continue, without speed restriction
    Permissive,
    /// Signal/sign allows all carts to continue, with speed restriction
    SpeedRestriction(f32),
    /// No signal is present
    NoSignal,
}
impl SignalResult {
    fn to_f64(self) -> f64 {
        match self {
            SignalResult::Stop => 0.0,
            SignalResult::Permissive => f64::INFINITY,
            SignalResult::SpeedRestriction(x) => x as f64,
            SignalResult::NoSignal => f64::NAN,
        }
    }
    fn from_f64(x: f64) -> Self {
        if x == 0.0 {
            Self::Stop
        } else if x == f64::INFINITY {
            Self::Permissive
        } else if x.is_finite() && x > 0.0 {
            Self::SpeedRestriction(x as f32)
        } else {
            panic!("Invalid signal result: {:?}, {:x}", x, x.to_bits());
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RailScanState {
    /// If true, the rail is valid
    valid: bool,
    /// The angle from the Z+ direction, in 90 degree increments
    major_angle: u8,
    /// Direction in 1/96th-block per step off the major angle
    /// Left of direction (i.e. increasing angle) is positive
    /// e.g. setting to 3 means you'll shift left a full block after
    /// moving 32 blocks forward
    horizontal_deflection: i8,
    /// Direction in 1/96th-block per step off the horizontal
    /// Ascending grades in the scan direction is positive
    /// A value of 3 means a 3.125% grade
    vertical_deflection: i8,
    /// How many sixteenths-steps have actually been taken off the major angle
    horizontal_offset: i8,
    /// How many sixteenths-steps have actually been taken off the horizontal
    vertical_offset: i8,
}
impl RailScanState {
    fn to_u64(&self) -> u64 {
        self.major_angle as u64
            | (self.horizontal_deflection as u64) << 8
            | (self.vertical_deflection as u64) << 16
            | (self.horizontal_offset as u64) << 24
            | (self.vertical_offset as u64) << 32
            | (self.valid as u64) << 40
    }
    fn from_u64(x: u64) -> Self {
        Self {
            valid: (x >> 40) != 0,
            major_angle: (x & 0xFF) as u8,
            horizontal_deflection: ((x >> 8) & 0xFF) as i8,
            vertical_deflection: ((x >> 16) & 0xFF) as i8,
            horizontal_offset: ((x >> 24) & 0xFF) as i8,
            vertical_offset: ((x >> 32) & 0xFF) as i8,
        }
    }

    // Determines the next rail to parse and the resulting scan state, as well as the distance moved
    fn next_scan(&self, current_coord: BlockCoordinate) -> Option<(Self, BlockCoordinate, f64)> {
        if !self.valid {
            panic!("next_scan called on invalid RailScanState: {:?}", self);
        }

        let horz = self.horizontal_offset + self.horizontal_deflection;
        let vert = self.vertical_offset + self.vertical_deflection;
        let (horz_major, horz_minor) = symmetric_mod_rm_96(horz);
        let (vert_major, vert_minor) = symmetric_mod_rm_96(vert);

        let (dx, dy, dz) = match self.major_angle % 4 {
            0 => (-horz_major, vert_major, 1),
            1 => (1, vert_major, horz_major),
            2 => (horz_major, vert_major, -1),
            3 => (-1, vert_major, -horz_major),
            _ => unreachable!(),
        };

        Some((
            Self {
                valid: true,
                major_angle: self.major_angle,
                horizontal_deflection: self.horizontal_deflection,
                vertical_deflection: self.vertical_deflection,
                horizontal_offset: horz_minor,
                vertical_offset: vert_minor,
            },
            current_coord
                .try_delta(dx.into(), dy.into(), dz.into())
                .unwrap(),
            vec2(
                1.0 + (self.horizontal_deflection as f64 / 96.0),
                self.vertical_deflection as f64 / 96.0,
            )
            .magnitude(),
        ))
    }

    fn position(&self, coarse: BlockCoordinate) -> Vector3<f64> {
        let h_f64 = self.horizontal_offset as f64 / 96.0;
        let v_f64 = self.vertical_offset as f64 / 96.0;

        let (dx, dy, dz) = match self.major_angle % 4 {
            0 => (-h_f64, v_f64, 0.0),
            1 => (0.0, v_f64, h_f64),
            2 => (h_f64, v_f64, 0.0),
            3 => (0.0, v_f64, -h_f64),
            _ => unreachable!(),
        };
        b2vec(coarse) + vec3(dx, dy, dz)
    }

    fn wayside_signal_delta(&self) -> Option<(i32, i32, i32)> {
        Some(match self.major_angle % 4 {
            0 => (1, 2, 0),
            1 => (0, 2, -1),
            2 => (-1, 2, 0),
            3 => (0, 2, 1),
            _ => unreachable!(),
        })
    }

    fn same_direction(&self, rail_scan_state: RailScanState) -> bool {
        if !self.valid || !rail_scan_state.valid {
            panic!(
                "same_direction called on invalid RailScanState: {:?}, {:?}",
                self, rail_scan_state
            );
        }

        self.major_angle == rail_scan_state.major_angle
            && self.horizontal_deflection == rail_scan_state.horizontal_deflection
            && self.vertical_deflection == rail_scan_state.vertical_deflection
    }
}

fn symmetric_mod_rm_96(x: i8) -> (i8, i8) {
    match x {
        ..=-96 => (-1, x + 96),
        -95..=95 => (0, x),
        96.. => (1, x - 96),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScanStopReason {
    /// We scanned as far as we're limited to
    ScanDistance,
    /// We encountered a restrictive signal or a switch that's not set for us
    Signal,
    /// We didn't find a rail.
    NoRail,
}
