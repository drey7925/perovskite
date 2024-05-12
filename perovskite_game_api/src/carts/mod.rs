use std::{collections::VecDeque, sync::atomic::AtomicU64, time::Instant};

// This is a temporary implementation used while developing the entity system
use anyhow::Result;

use cgmath::{InnerSpace, Vector3};
use perovskite_core::{
    block_id::BlockId,
    chat::{ChatMessage, SERVER_ERROR_COLOR, SERVER_MESSAGE_COLOR},
    constants::items::default_item_interaction_rules,
    coordinates::BlockCoordinate,
    protocol::{self, items::item_def::QuantityType, render::CustomMesh},
    util::{TraceBuffer, TraceLog},
};
use perovskite_server::game_state::{
    self,
    entities::{
        ContinuationResult, ContinuationResultValue, CoroutineResult, DeferrableResult,
        EntityClassId, EntityCoroutine, EntityCoroutineServices, EntityDef, EntityTypeId, Movement,
        ReenterableResult,
    },
    event::HandlerContext,
    items::ItemStack,
    GameStateExtension,
};
use rustc_hash::FxHashMap;

use crate::{
    blocks::{variants::rotate_nesw_azimuth_to_variant, BlockBuilder, CubeAppearanceBuilder},
    game_builder::{GameBuilderExtension, StaticBlockName, StaticTextureName},
    include_texture_bytes,
};

use self::{interlocking::InterlockingStep, signals::automatic_signal_acquire, tracks::ScanState};

mod interlocking;
mod signals;
mod tracks;

#[derive(Clone)]
struct CartsGameBuilderExtension {
    rail_block: BlockId,
    speedpost_1: BlockId,
    speedpost_2: BlockId,
    speedpost_3: BlockId,
    switch_unset: BlockId,
    switch_straight: BlockId,
    switch_diverging: BlockId,
    automatic_signal: BlockId,
    interlocking_signal: BlockId,

    cart_id: EntityClassId,
}
impl Default for CartsGameBuilderExtension {
    fn default() -> Self {
        CartsGameBuilderExtension {
            rail_block: 0.into(),
            speedpost_1: 0.into(),
            speedpost_2: 0.into(),
            speedpost_3: 0.into(),
            switch_unset: 0.into(),
            switch_straight: 0.into(),
            switch_diverging: 0.into(),
            automatic_signal: 0.into(),
            interlocking_signal: 0.into(),
            cart_id: EntityClassId::new(0),
        }
    }
}

const CART_MESH_BYTES: &[u8] = include_bytes!("minecart.obj");
lazy_static::lazy_static! {
    static ref CART_MESH: CustomMesh = {
        perovskite_server::formats::load_obj_mesh(CART_MESH_BYTES, "carts:minecart_uv").unwrap()
    };
}

pub fn register_carts(game_builder: &mut crate::game_builder::GameBuilder) -> Result<()> {
    let rail_tex = StaticTextureName("carts:rail");
    let speedpost1_tex = StaticTextureName("carts:speedpost1");
    let speedpost2_tex = StaticTextureName("carts:speedpost2");
    let speedpost3_tex = StaticTextureName("carts:speedpost3");

    let switch_unset_tex = StaticTextureName("carts:switch_unset");
    let switch_straight_tex = StaticTextureName("carts:switch_straight");
    let switch_diverging_tex = StaticTextureName("carts:switch_diverging");

    include_texture_bytes!(game_builder, rail_tex, "textures/rail.png")?;
    include_texture_bytes!(game_builder, speedpost1_tex, "textures/speedpost_1.png")?;
    include_texture_bytes!(game_builder, speedpost2_tex, "textures/speedpost_2.png")?;
    include_texture_bytes!(game_builder, speedpost3_tex, "textures/speedpost_3.png")?;

    include_texture_bytes!(game_builder, switch_unset_tex, "textures/switch_unset.png")?;
    include_texture_bytes!(
        game_builder,
        switch_straight_tex,
        "textures/switch_straight.png"
    )?;
    include_texture_bytes!(
        game_builder,
        switch_diverging_tex,
        "textures/switch_diverging.png"
    )?;

    let cart_tex = StaticTextureName("carts:cart_temp");
    include_texture_bytes!(game_builder, cart_tex, "textures/testonly_cart.png")?;

    let cart_uv_tex = StaticTextureName("carts:minecart_uv");
    include_texture_bytes!(game_builder, cart_uv_tex, "textures/cart_uv.png")?;

    let _rail = game_builder.add_block(
        BlockBuilder::new(StaticBlockName("carts:rail")).set_cube_appearance(
            CubeAppearanceBuilder::new()
                .set_single_texture(rail_tex)
                .set_rotate_laterally(),
        ),
    )?;
    // TODO update the speedposts
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

    let switch_unset = game_builder.add_block(
        BlockBuilder::new(StaticBlockName("carts:switch_unset")).set_cube_appearance(
            CubeAppearanceBuilder::new()
                .set_single_texture(switch_unset_tex)
                .set_rotate_laterally(),
        ),
    )?;
    let switch_straight = game_builder.add_block(
        BlockBuilder::new(StaticBlockName("carts:switch_straight")).set_cube_appearance(
            CubeAppearanceBuilder::new()
                .set_single_texture(switch_straight_tex)
                .set_rotate_laterally(),
        ),
    )?;
    let switch_diverging = game_builder.add_block(
        BlockBuilder::new(StaticBlockName("carts:switch_diverging")).set_cube_appearance(
            CubeAppearanceBuilder::new()
                .set_single_texture(switch_diverging_tex)
                .set_rotate_laterally(),
        ),
    )?;

    let cart_id = game_builder.inner.entities_mut().register(EntityDef {
        move_queue_type: game_state::entities::MoveQueueType::Buffer8,
        class_name: "carts:high_speed_minecart".to_string(),
        client_info: protocol::entities::EntityAppearance {
            custom_mesh: vec![CART_MESH.clone()],
        },
    })?;

    let (automatic_signal, interlocking_signal) = signals::register_signal_block(game_builder)?;

    let rail_block_id = tracks::register_tracks(game_builder)?;

    let ext = game_builder.builder_extension::<CartsGameBuilderExtension>();
    ext.rail_block = rail_block_id;
    ext.speedpost_1 = speedpost1.id;
    ext.speedpost_2 = speedpost2.id;
    ext.speedpost_3 = speedpost3.id;
    ext.switch_unset = switch_unset.id;
    ext.switch_straight = switch_straight.id;
    ext.switch_diverging = switch_diverging.id;
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
    let (rail_pos, _variant) = if block.equals_ignore_variant(config.rail_block) {
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

    static ID_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

    let id = ctx.entities().new_entity_blocking(
        // TODO: support an offset when attaching a player to an entity, so the camera position is right
        // and we don't need to do this hackery
        initial_state.vec_coord() + cgmath::Vector3::new(0.0, 1.0, 0.0),
        Some(Box::pin(CartCoroutine {
            config: config.clone(),
            scheduled_segments: VecDeque::new(),
            unplanned_segments: VecDeque::new(),
            // Until we encounter a speed post, proceed at minimal safe speed
            last_speed_post_indication: 0.5,
            last_submitted_move_exit_speed: 0.5,
            spawn_time: Instant::now(),
            scan_state: initial_state,
            cleared_signals: FxHashMap::default(),
            held_signal: None,
            id: ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            precomputed_steps: Vec::new(),
        })),
        EntityTypeId {
            class: dbg!(config.cart_id),
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

    // The signal to acquire when entering this segment
    enter_signal: Option<(BlockCoordinate, BlockId)>,
    // The signal to release when exiting this segment
    exit_signal: Option<(BlockCoordinate, BlockId)>,
    // The switch to reset when exiting this segment
    pass_switch: Option<(BlockCoordinate, BlockId)>,
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

    fn any_content(&self) -> bool {
        self.enter_signal.is_some()
            || self.exit_signal.is_some()
            || self.distance() > 0.0
            || self.pass_switch.is_some()
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
                    pass_switch: None,
                },
                Some(Self {
                    from: split_point,
                    to: self.to,
                    max_speed: self.max_speed,
                    seg_id: self.seg_id,
                    enter_signal: None,
                    exit_signal: self.exit_signal,
                    pass_switch: None,
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
    created: &'static str,
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
    // The last speed post we encountered while scanning
    last_speed_post_indication: Flt,
    // The last velocity we already delivered to the entity system
    last_submitted_move_exit_speed: Flt,
    id: u32,
    // Track scan state
    scan_state: tracks::ScanState,
    // TODO improve this as needed
    cleared_signals: FxHashMap<BlockCoordinate, BlockId>,
    held_signal: Option<(BlockCoordinate, BlockId)>,
    // If we got a planned path from the interlocking system, this is the path
    precomputed_steps: Vec<InterlockingStep>,
    // debug only
    spawn_time: Instant,
}
impl EntityCoroutine for CartCoroutine {
    fn plan_move(
        mut self: std::pin::Pin<&mut Self>,
        services: &perovskite_server::game_state::entities::EntityCoroutineServices<'_>,
        _current_position: cgmath::Vector3<f64>,
        whence: cgmath::Vector3<f64>,
        when: f32,
        queue_space: usize,
    ) -> perovskite_server::game_state::entities::CoroutineResult {
        let trace_buffer = TraceBuffer::new(self.id == 0);
        self.plan_move_impl(services, whence, when, queue_space, trace_buffer)
    }
    fn continuation(
        mut self: std::pin::Pin<&mut Self>,
        services: &perovskite_server::game_state::entities::EntityCoroutineServices<'_>,
        _current_position: cgmath::Vector3<f64>,
        whence: cgmath::Vector3<f64>,
        when: f32,
        queue_space: usize,
        continuation_result: ContinuationResult,
        trace_buffer: TraceBuffer,
    ) -> CoroutineResult {
        trace_buffer.log("In continuation");
        if continuation_result.tag == CONTINUATION_TAG_SIGNAL {
            match continuation_result.value {
                ContinuationResultValue::GetBlock(block_id, coord) => {
                    self.cleared_signals.insert(coord, block_id);
                }
                ContinuationResultValue::HeapResult(result) => {
                    match result.downcast::<Option<Vec<InterlockingStep>>>() {
                        Ok(steps) => {
                            match *steps {
                                Some(steps) => {
                                    if !self.precomputed_steps.is_empty() {
                                        tracing::warn!(
                                            "precomputed steps is not empty: {:?}",
                                            self.precomputed_steps
                                        )
                                    }
                                    self.precomputed_steps = steps;
                                }
                                None => {
                                    // empty route plan, try to scan the interlocking again next time
                                    // For now, just schedule what we can, and we'll get reinvoked to try again
                                    tracing::debug!("Got empty route plan, rescheduling");
                                    trace_buffer.log("Got empty route plan, rescheduling");
                                    return self.finish_schedule(
                                        when,
                                        queue_space,
                                        whence,
                                        services,
                                        trace_buffer,
                                    );
                                }
                            }
                        }
                        Err(_) => {
                            log::error!("Unexpected heap result");
                        }
                    }
                }
                _ => {
                    log::error!(
                        "Unhandled continuation value with signal tag {:?}",
                        continuation_result.value
                    );
                }
            };
        }
        self.plan_move_impl(services, whence, when, queue_space, trace_buffer)
    }
}

const CONTINUATION_TAG_SIGNAL: u32 = 0x516ea1;

impl CartCoroutine {
    fn plan_move_impl(
        &mut self,
        services: &EntityCoroutineServices<'_>,
        whence: cgmath::Vector3<f64>,
        when: f32,
        queue_space: usize,
        trace_buffer: TraceBuffer,
    ) -> CoroutineResult {
        tracing::debug!(
            "{} {:?} ========== planning {} seconds in advance",
            self.id,
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

        let maybe_deferral = self.scan_tracks(services, 1024.0, when, &trace_buffer);
        if let Some(deferral) = maybe_deferral {
            return deferral.with_trace_buffer(trace_buffer);
        }

        self.finish_schedule(when, queue_space, whence, services, trace_buffer)
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
                    signals::SignalLockOutcome::Acquired => {
                        tracing::debug!("acquired inline");
                        SignalResult::Permissive.into()
                    }
                    signals::SignalLockOutcome::Contended => {
                        tracing::debug!("contended inline");
                        SignalResult::Stop.into()
                    }
                    signals::SignalLockOutcome::InvalidSignal => SignalResult::NoSignal.into(),
                },
                DeferrableResult::AvailableNow(Err(e)) => {
                    tracing::error!("Failed to parse signal: {}", e);
                    SignalResult::NoSignal.into()
                }
                DeferrableResult::Deferred(d) => {
                    ReenterableResult::Deferred(d.map(move |x| match x {
                        Ok(x) => match x {
                            signals::SignalLockOutcome::Acquired => {
                                tracing::debug!("acquired deferred");
                                ContinuationResultValue::GetBlock(signal_block, signal_coord).into()
                            }
                            signals::SignalLockOutcome::Contended => {
                                tracing::debug!("contended deferred");
                                ContinuationResultValue::GetBlock(0.into(), signal_coord).into()
                            }
                            signals::SignalLockOutcome::InvalidSignal => {
                                tracing::debug!("Invalid signal");
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
        } else if signal_block.equals_ignore_variant(self.config.interlocking_signal) {
            let state_clone = self.scan_state.clone();
            let config_clone = self.config.clone();
            return ReenterableResult::Deferred(services.spawn_async(move |ctx| async move {
                let state = state_clone;
                let result = interlocking::interlock_cart(ctx, state, 256, config_clone).await;
                // todo better error handling?
                ContinuationResultValue::HeapResult(Box::new(result.unwrap_or_default()))
            }));
        } else {
            SignalResult::NoSignal.into()
        }
    }

    fn scan_tracks(
        &mut self,
        services: &EntityCoroutineServices<'_>,
        max_steps_ahead: f64,
        when: f32,
        trace_buffer: &TraceBuffer,
    ) -> Option<CoroutineResult> {
        if !self.precomputed_steps.is_empty() {
            trace_buffer.log("Using precomputed steps");
        }
        for (i, step) in std::mem::replace(&mut self.precomputed_steps, vec![])
            .into_iter()
            .enumerate()
        {
            tracing::debug!("step {} at {:?}", i, step.scan_state.block_coord);
            let state = step.scan_state;

            if let Some(switch_coord) = state.block_coord.try_delta(0, -1, 0) {
                if step.pass_switch != BlockId::from(0) {
                    self.unplanned_segments.back_mut().unwrap().pass_switch =
                        Some((switch_coord, step.pass_switch));
                }
                self.start_new_unplanned_segment();
            }

            if let Some(signal_coord) = state.block_coord.try_delta(0, 2, 0) {
                if step.enter_signal != BlockId::from(0) {
                    tracing::debug!("Acquiring signal at {:?}", state.block_coord);
                    tracing::debug!(
                        "Releasing signal at {:?}",
                        self.held_signal.as_ref().map(|x| x.0)
                    );
                    self.unplanned_segments.back_mut().unwrap().exit_signal =
                        self.held_signal.take();
                    self.start_new_unplanned_segment();
                    self.unplanned_segments.back_mut().unwrap().enter_signal =
                        Some((signal_coord, step.enter_signal));
                    self.held_signal = Some((signal_coord, step.enter_signal));
                }
            }
            self.apply_step(state);
        }

        trace_buffer.log("Starting scan");
        let mut steps = 0.0;
        for seg in self.scheduled_segments.iter() {
            steps += seg.segment.distance();
        }
        for seg in self.unplanned_segments.iter() {
            steps += seg.distance();
        }

        // this should be an underestimate
        // max_speed is an overestimate of the actual speed, and it's in the denominator
        let unplanned_buffer_estimate = self
            .unplanned_segments
            .iter()
            .map(|seg| (seg.distance() / seg.max_speed) as f32)
            .sum::<f32>();
        let scheduled_buffer_estimate = self
            .scheduled_segments
            .iter()
            .map(|seg| seg.move_time as f32)
            .sum::<f32>();
        let mut buffer_time_estimate = when + unplanned_buffer_estimate + scheduled_buffer_estimate;

        // this should be an overestimate
        // todo: tighten this bound. If we start at a high max speed (even if unachievable) we'll end up with a high estimated max speed
        let mut estimated_max_speed = self
            .unplanned_segments
            .back()
            .map(|seg| seg.max_speed)
            .unwrap_or(
                self.scheduled_segments
                    .back()
                    .map(|seg| seg.speed + (seg.acceleration * seg.move_time))
                    .unwrap_or(self.last_speed_post_indication.max(5.0)),
            ) as f32;
        tracing::debug!("estimated max speed {}", estimated_max_speed);
        if self.unplanned_segments.is_empty() {
            tracing::debug!("unplanned segments empty, adding new");
            let empty_segment = TrackSegment {
                from: self.scan_state.vec_coord(),
                to: self.scan_state.vec_coord(),
                max_speed: self
                    .last_speed_post_indication
                    .min(self.scan_state.allowable_speed as f64),
                seg_id: NEXT_SEGMENT_ID.fetch_add(10, std::sync::atomic::Ordering::Relaxed),
                enter_signal: None,
                exit_signal: None,
                pass_switch: None,
            };
            self.unplanned_segments.push_back(empty_segment);
        }

        let block_getter = |coord| services.get_block(coord);

        'scan_loop: while steps < max_steps_ahead
            && buffer_time_estimate < (2.0 + 2.0 * estimated_max_speed / MAX_ACCEL as f32)
        {
            // Precondition: self.scan_state is valid
            let new_state = match self.scan_state.advance::<false>(block_getter) {
                Ok(tracks::ScanOutcome::Success(state)) => state,
                Ok(tracks::ScanOutcome::CannotAdvance) => break 'scan_loop,
                Ok(tracks::ScanOutcome::NotOnTrack) => {
                    tracing::warn!("Not on track at {:?}", self.scan_state.block_coord);
                    break 'scan_loop;
                }
                Ok(tracks::ScanOutcome::Deferral(d)) => {
                    tracing::debug!("track scan deferral");
                    trace_buffer.log("ScanState::advance deferral");
                    return Some(d.defer_and_reinvoke(1));
                }
                Err(e) => {
                    tracing::error!("Failed to parse track: {}", e);
                    break 'scan_loop;
                }
            };

            let signal_coord = new_state.block_coord.try_delta(0, 2, 0);
            if let Some(signal_coord) = signal_coord {
                if let Some(block_id) = self.cleared_signals.remove(&signal_coord) {
                    tracing::debug!("cached clear signal at {:?}", signal_coord);
                    if block_id.equals_ignore_variant(0.into()) {
                        // contended signal
                        break 'scan_loop;
                    } else {
                        tracing::debug!("Cached permissive signal at {:?}", new_state.block_coord);
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
                        return Some(d.defer_and_reinvoke(CONTINUATION_TAG_SIGNAL));
                    }
                };
                match signal_result {
                    SignalResult::Stop => {
                        tracing::debug!("Stop signal at {:?}", new_state.block_coord);
                        break 'scan_loop;
                    }
                    SignalResult::Permissive => {
                        tracing::debug!("Permissive signal at {:?}", new_state.block_coord);
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

            let movement_len = (new_state.vec_coord() - self.scan_state.vec_coord()).magnitude();
            steps += movement_len;
            // last speed indication is an overestimate of speed
            let speed_upper_bound = self
                .last_speed_post_indication
                .min(new_state.allowable_speed as f64);
            buffer_time_estimate += (movement_len / speed_upper_bound) as f32;
            estimated_max_speed = (estimated_max_speed * estimated_max_speed
                + 2.0 * MAX_ACCEL as f32 * movement_len as f32)
                .sqrt()
                .min(speed_upper_bound as f32);
            self.apply_step(new_state);
        }

        tracing::debug!(
            "Finished with {} steps, {} buffer time estimate, {} max speed, {} time limit",
            steps,
            buffer_time_estimate,
            estimated_max_speed,
            (2.0 * estimated_max_speed / MAX_ACCEL as f32)
        );
        trace_buffer.log("finished scanning");

        None
    }

    fn apply_step(&mut self, new_state: ScanState) {
        let last_move = self.unplanned_segments.back().unwrap();
        // We got success, so we're at a new position.
        let new_delta = new_state.vec_coord() - last_move.to;
        if new_delta.magnitude() <= 0.001 {
            return;
        }

        let last_move_delta = last_move.to - last_move.from;

        let effective_speed = self
            .last_speed_post_indication
            .min(new_state.allowable_speed as f64);
        tracing::debug!("Considering split at {:?}", new_state.vec_coord());
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
            tracing::info!("Splitting a segment due to effective speed; prev length was {}, speed changing {} -> {}", last_move_delta.magnitude(), last_move.max_speed, effective_speed);
            self.start_new_unplanned_segment();
        }
        let last_seg_mut = self.unplanned_segments.back_mut().unwrap();
        last_seg_mut.to = new_state.vec_coord();
        last_seg_mut.max_speed = effective_speed;
        self.scan_state = new_state;
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

        // If this is true, we've hit a segment where we were limited by track speed.
        let mut unconditionally_schedulable = false;
        let mut schedulable_segments = VecDeque::new();

        let available_scheduled_segments = (8usize.checked_sub(queue_space).unwrap())
            + self
                .scheduled_segments
                .iter()
                .filter(|s| s.segment.distance() > 0.001)
                .count();
        tracing::debug!(
            "Available scheduled segments: {:?}",
            self.scheduled_segments
        );

        tracing::debug!(
            "{} segments remaining in track scan",
            available_scheduled_segments
        );

        // split the last segment into a segment as long as the stopping distance and a remainder, if possible
        // This allows us to avoid panic-scheduling the entire last segment if it's long

        let stopping_split_pos = self.unplanned_segments.back().and_then(|last_segment| {
            // 0 = vi^2 + 2ad, d = vi^2 / 2a after correcting for signs
            let stopping_distance =
                (last_segment.max_speed * last_segment.max_speed) / (2.0 * MAX_ACCEL);
            if stopping_distance > 0.0 && (stopping_distance + 0.001) < last_segment.distance() {
                Some(last_segment.distance() - stopping_distance)
            } else {
                None
            }
        });
        let slowing_split_pos = self.unplanned_segments.back().and_then(|last_segment| {
            if last_segment.max_speed < 6.0 {
                return None;
            };
            let target_speed = last_segment.max_speed - 5.0;
            let slowing_distance = (last_segment.max_speed * last_segment.max_speed
                - target_speed * target_speed)
                / (2.0 * MAX_ACCEL);
            if slowing_distance > 0.0 && (slowing_distance + 0.001) < last_segment.distance() {
                Some(slowing_distance)
            } else {
                None
            }
        });

        if let Some(split_pos) = stopping_split_pos {
            tracing::debug!(
                "Splitting last segment {:?} at {} out of {}",
                self.unplanned_segments.back().unwrap(),
                split_pos,
                self.unplanned_segments.back().unwrap().distance()
            );
            let (main, stopping_distance) = self
                .unplanned_segments
                .pop_back()
                .unwrap()
                .split_at_offset(split_pos);
            self.unplanned_segments.push_back(main);
            self.unplanned_segments
                .push_back(stopping_distance.unwrap());
            if let Some(slowing_split_pos) = slowing_split_pos {
                let (slowing_distance, remainder) = self
                    .unplanned_segments
                    .pop_back()
                    .unwrap()
                    .split_at_offset(slowing_split_pos);
                self.unplanned_segments.push_back(slowing_distance);
                self.unplanned_segments.push_back(remainder.unwrap());
            }
        }

        // The maximum speed at which we're allowed to exit the current segment.
        // Updated to be the max entry speed of the following segment.
        // It's 0.0 to start, because there is no further segment detected by track scan (or that segment
        // cannot be entered due to a restrictive signal or a switch set against us)
        let mut max_exit_speed: Flt = 0.0;

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
            let should_panic_schedule =
                available_scheduled_segments < 2 || self.last_submitted_move_exit_speed < 0.001;

            if unconditionally_schedulable {
                schedulable_segments.push_front((*seg, max_exit_speed));
                tracing::debug!(
                    "> seg_schedulable = {}, seg = {:?}",
                    unconditionally_schedulable,
                    seg.seg_id
                );
            } else if should_panic_schedule && idx == 0 && when < 1.0 {
                // We're low on cached moves, so let's schedule this segment
                schedulable_segments.push_front((*seg, max_exit_speed));
                tracing::debug!(
                    "> panic scheduling! seg_schedulable = {}, seg = {:?}",
                    unconditionally_schedulable,
                    seg
                );
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
        tracing::debug!("Scheduling {:?} segments", schedulable_segments.len());
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
        if prev_segment.any_content() {
            tracing::debug!("new segment starting from {:?}", prev_segment.to);
            let empty_segment = TrackSegment {
                from: prev_segment.to,
                to: prev_segment.to,
                max_speed: self.last_speed_post_indication,
                seg_id: NEXT_SEGMENT_ID.fetch_add(10, std::sync::atomic::Ordering::Relaxed),
                enter_signal: None,
                exit_signal: None,
                pass_switch: None,
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
            if seg.any_content() {
                tracing::debug!("Pushing short segment: {:?}", seg);
                self.scheduled_segments.push_back(ScheduledSegment {
                    segment: seg,
                    speed: enter_speed,
                    acceleration: 0.0,
                    move_time: 0.0,
                    created: "short segment",
                });
            } else {
                tracing::debug!("Skipping short segment: {:?}", seg);
            }
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
                created: "no acceleration required",
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
                    created: "case 2a first",
                });

                self.scheduled_segments.push_back(ScheduledSegment {
                    segment: split_after,
                    speed: seg.max_speed,
                    acceleration: 0.0,
                    move_time: remaining_distance / seg.max_speed,
                    created: "case 2a second",
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
                    created: "case 2b",
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
                    created: "case 3",
                });
                brake_curve_exit_speed
            } else {
                // We have enough distance to finish the deceleration
                let remaining_distance = seg_distance - exit_accel_distance;
                let (split_before, split_after) = seg.split_at_offset(remaining_distance);
                tracing::debug!(">> case 3a, remaining_distance {}", remaining_distance);
                self.scheduled_segments.push_back(ScheduledSegment {
                    segment: split_before,
                    speed: enter_speed,
                    acceleration: 0.0,
                    move_time: split_before.distance() / enter_speed,
                    created: "case 3a first",
                });
                if let Some(after) = split_after {
                    self.scheduled_segments.push_back(ScheduledSegment {
                        segment: after,
                        speed: enter_speed,
                        acceleration: -MAX_ACCEL,
                        move_time: (seg.max_speed - brake_curve_exit_speed) / MAX_ACCEL,
                        created: "case 3a second",
                    });
                }
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
                    created: "case 4a first",
                });

                self.scheduled_segments.push_back(ScheduledSegment {
                    segment: split_after.unwrap(),
                    speed: speed_right,
                    acceleration: -MAX_ACCEL,
                    move_time: (speed_right - brake_curve_exit_speed) / MAX_ACCEL,
                    created: "case 4a second",
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
                    created: "case 4b",
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
                    created: "case 4c",
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
                created: "case 5 first",
            });
            // The cruise segment
            self.scheduled_segments.push_back(ScheduledSegment {
                segment: seg,
                speed: seg.max_speed,
                acceleration: 0.0,
                move_time: remaining_distance / seg.max_speed,
                created: "case 5 second",
            });
            // The deceleration segment
            self.scheduled_segments.push_back(ScheduledSegment {
                segment: seg,
                speed: seg.max_speed,
                acceleration: -MAX_ACCEL,
                move_time: (seg.max_speed - brake_curve_exit_speed) / MAX_ACCEL,
                created: "case 5 third",
            });

            brake_curve_exit_speed
        }
    }

    fn finish_schedule(
        &mut self,
        when: f32,
        queue_space: usize,
        whence: Vector3<f64>,
        services: &EntityCoroutineServices<'_>,
        trace_buffer: TraceBuffer,
    ) -> CoroutineResult {
        trace_buffer.log("Promoting schedulable segments");
        let schedulable_segments = self.promote_schedulable(when, queue_space);
        trace_buffer.log("Scheduling segments");
        self.schedule_segments(schedulable_segments);
        if !self.scheduled_segments.iter().any(|x| x.move_time > 0.001) {
            // No scheduled segments
            // Scan every second, trying to start again.
            tracing::debug!("No schedulable segments");
            trace_buffer.log("No schedulable segments");
            return perovskite_server::game_state::entities::CoroutineResult::Successful(
                perovskite_server::game_state::entities::EntityMoveDecision::AskAgainLaterFlexible(
                    0.5..1.0,
                ),
            );
        }
        trace_buffer.log("Building moves");
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
                services.spawn_delayed(
                    std::time::Duration::from_secs_f64(
                        (returned_moves_time + when as f64).max(0.0),
                    ),
                    move |ctx| {
                        ctx.game_map()
                            .mutate_block_atomically(enter_signal_coord, |b, _ext| {
                                tracing::debug!("entering block");
                                signals::signal_enter_block(
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

                services.spawn_delayed(
                    std::time::Duration::from_secs_f64(
                        (returned_moves_time + when as f64 + extra_delay).max(0.0),
                    ),
                    move |ctx| {
                        ctx.game_map()
                            .mutate_block_atomically(exit_signal_coord, |b, _ext| {
                                signals::signal_release(exit_signal_coord, b, exit_signal_block);
                                Ok(())
                            })
                            .unwrap();
                    },
                );
            }
            if let Some((pass_switch_coord, pass_switch_block)) = segment.segment.pass_switch {
                let exit_speed = segment.speed + (segment.acceleration * segment.move_time);
                const SIGNAL_BUFFER_DISTANCE: f64 = 1.0;
                let extra_delay = (SIGNAL_BUFFER_DISTANCE / exit_speed).clamp(0.0, 1.0);
                tracing::info!(
                    "passing switch in {} + {} + {} seconds at {:?}",
                    returned_moves_time,
                    when,
                    extra_delay,
                    pass_switch_coord
                );
                let switch_unset = self.config.switch_unset;
                services.spawn_delayed(
                    std::time::Duration::from_secs_f64(
                        (returned_moves_time + when as f64 + extra_delay).max(0.0),
                    ),
                    move |ctx| match ctx
                        .game_map()
                        .compare_and_set_block(
                            pass_switch_coord,
                            pass_switch_block,
                            switch_unset,
                            None,
                            false,
                        )
                        .unwrap()
                    {
                        (game_state::game_map::CasOutcome::Match, _, _) => {}
                        (game_state::game_map::CasOutcome::Mismatch, id, _) => {
                            tracing::warn!(
                                "Pass switch mismatch: got {:?} but expected {:?} at {:?}",
                                id,
                                pass_switch_block,
                                pass_switch_coord
                            );
                        }
                    },
                );
            }

            if let Some(movement) = segment.to_movement() {
                returned_moves.push(movement);
            } else if !segment.segment.any_content() {
                tracing::warn!("Segment {:?} had no movement", segment);
            }
        }
        tracing::debug!("returning {} moves", returned_moves.len());
        trace_buffer.log("Done!");
        CoroutineResult::Successful(
            perovskite_server::game_state::entities::EntityMoveDecision::QueueUpMultiple(
                returned_moves,
            ),
        )
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
