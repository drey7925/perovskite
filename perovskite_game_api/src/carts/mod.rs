use std::pin::Pin;
use std::time::Duration;
use std::{
    collections::{BinaryHeap, VecDeque},
    time::Instant,
};
// This is a temporary implementation used while developing the entity system
use anyhow::{bail, Context, Result};

use self::{interlocking::InterlockingStep, signals::automatic_signal_acquire, tracks::ScanState};
use crate::carts::util::AsyncRefcount;
use crate::default_game::block_groups::BRITTLE;
use crate::{
    blocks::{variants::rotate_nesw_azimuth_to_variant, BlockBuilder, CubeAppearanceBuilder},
    game_builder::{GameBuilderExtension, StaticBlockName, StaticTextureName},
    include_texture_bytes,
};
use cgmath::{vec3, InnerSpace, Vector3};
use interlocking::{InterlockingResumeState, InterlockingRoute};
use perovskite_core::protocol::game_rpc::EntityTarget;
use perovskite_core::{
    block_id::BlockId,
    chat::{ChatMessage, SERVER_ERROR_COLOR, SERVER_MESSAGE_COLOR},
    constants::items::default_item_interaction_rules,
    coordinates::BlockCoordinate,
    protocol::{self, items::item_def::QuantityType, render::CustomMesh},
    util::{TraceBuffer, TraceLog},
};
use perovskite_server::game_state::client_ui::Popup;
use perovskite_server::game_state::entities::{EntityHandlers, EntityMoveDecision, TrailingEntity};
use perovskite_server::game_state::event::EventInitiator;
use perovskite_server::game_state::items::{Item, ItemInteractionResult};
use perovskite_server::game_state::player::Player;
use perovskite_server::game_state::{
    self,
    client_ui::{PopupAction, PopupResponse, UiElementContainer},
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
use tokio_util::sync::CancellationToken;

mod interlocking;
mod signals;
mod track_tool;
mod tracks;

#[derive(Clone, Debug)]
struct CartsGameBuilderExtension {
    rail_block: BlockId,
    rail_slope_1: BlockId,
    rail_slopes_8: [BlockId; 8],
    speedpost_1: BlockId,
    speedpost_2: BlockId,
    speedpost_3: BlockId,
    switch_unset: BlockId,
    switch_straight: BlockId,
    switch_diverging: BlockId,
    automatic_signal: BlockId,
    interlocking_signal: BlockId,
    starting_signal: BlockId,

    cart_id: EntityClassId,
}
impl CartsGameBuilderExtension {
    fn parse_speedpost(&self, signal_block: BlockId) -> Option<f32> {
        if signal_block.equals_ignore_variant(self.speedpost_1) {
            Some(3.0)
        } else if signal_block.equals_ignore_variant(self.speedpost_2) {
            Some(30.0)
        } else if signal_block.equals_ignore_variant(self.speedpost_3) {
            Some(90.0)
        } else {
            None
        }
    }

    // Returns (numerator, denominator, rotation)
    fn parse_slope(&self, block: BlockId) -> Option<(u8, u8, u16)> {
        if self.rail_slope_1.equals_ignore_variant(block) {
            Some((1, 1, block.variant() & 0b11))
        } else if let Some(idx) = self
            .rail_slopes_8
            .iter()
            .position(|b| b.equals_ignore_variant(block))
        {
            let numerator = idx as u8 + 1;
            Some((numerator, 8, block.variant() & 0b11))
        } else {
            None
        }
    }

    fn is_any_rail_block(&self, block: BlockId) -> bool {
        self.rail_block.equals_ignore_variant(block)
            || self.rail_slope_1.equals_ignore_variant(block)
            || self
                .rail_slopes_8
                .iter()
                .any(|b| b.equals_ignore_variant(block))
    }
}
impl Default for CartsGameBuilderExtension {
    fn default() -> Self {
        CartsGameBuilderExtension {
            rail_block: 0.into(),
            rail_slope_1: 0.into(),
            rail_slopes_8: [0.into(); 8],
            speedpost_1: 0.into(),
            speedpost_2: 0.into(),
            speedpost_3: 0.into(),
            switch_unset: 0.into(),
            switch_straight: 0.into(),
            switch_diverging: 0.into(),
            automatic_signal: 0.into(),
            interlocking_signal: 0.into(),
            starting_signal: 0.into(),
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

struct CartHandlers;

impl EntityHandlers for CartHandlers {
    fn on_interact_key(&self, ctx: &HandlerContext, target: EntityTarget) -> Result<Option<Popup>> {
        let work = |p: &Player| -> Result<()> {
            // Race condition but oh well
            let old_attachment = p.detach_from_entity_blocking()?;
            if old_attachment != Some(target) {
                p.attach_to_entity_blocking(target)?
            }
            Ok(())
        };
        match &ctx.initiator() {
            EventInitiator::Player(p) => {
                work(p.player)?;
            }
            EventInitiator::WeakPlayerRef(p) => {
                p.try_to_run(work).context("WeakPlayerRef gone")??;
            }
            _ => bail!("Only players can interact board minecarts"),
        }
        Ok(None)
    }

    fn on_dig(
        &self,
        ctx: &HandlerContext,
        target: EntityTarget,
        tool: Option<&ItemStack>,
    ) -> Result<ItemInteractionResult> {
        ctx.entities().remove_blocking(target.entity_id);
        Ok(ItemInteractionResult {
            updated_tool: tool.cloned(),
            obtained_items: vec![ctx
                .item_manager()
                .get_item("carts:minecart")
                .context("missing item")?
                .make_stack(1)],
        })
    }
}

pub fn register_carts(game_builder: &mut crate::game_builder::GameBuilder) -> Result<()> {
    crate::circuits::register_circuits(game_builder)?;
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

    let switch_straight_id = switch_straight.id;
    let switch_diverging_id = switch_diverging.id;
    let switch_unset_id = switch_unset.id;

    game_builder
        .inner
        .blocks_mut()
        .register_cold_load_postprocessor(Box::new(move |data| {
            for block in data {
                if block.equals_ignore_variant(switch_straight_id)
                    || block.equals_ignore_variant(switch_diverging_id)
                {
                    *block = switch_unset_id.with_variant_of(*block);
                }
            }
        }));

    let cart_id = game_builder.inner.entities_mut().register(EntityDef {
        move_queue_type: game_state::entities::MoveQueueType::Buffer64,
        class_name: "carts:high_speed_minecart".to_string(),
        client_info: protocol::entities::EntityAppearance {
            custom_mesh: vec![CART_MESH.clone()],
            attachment_offset: Some(vec3(0.0, 1.0, 0.0).try_into()?),
            attachment_offset_in_model_space: false,
            merge_trailing_entities_for_dig: true,
            tool_interaction_groups: vec![BRITTLE.to_string()],
            base_dig_time: 1.0,
        },
        handlers: Box::new(CartHandlers),
    })?;

    let (automatic_signal, interlocking_signal, starting_signal) =
        signals::register_signal_blocks(game_builder)?;

    let (rail_block_id, rail_slope_1, rail_slopes_8) = tracks::register_tracks(game_builder)?;

    let ext = game_builder.builder_extension::<CartsGameBuilderExtension>();
    ext.rail_block = rail_block_id;
    ext.rail_slope_1 = rail_slope_1;
    ext.rail_slopes_8 = rail_slopes_8;
    ext.speedpost_1 = speedpost1.id;
    ext.speedpost_2 = speedpost2.id;
    ext.speedpost_3 = speedpost3.id;
    ext.switch_unset = switch_unset.id;
    ext.switch_straight = switch_straight.id;
    ext.switch_diverging = switch_diverging.id;
    ext.automatic_signal = automatic_signal;
    ext.interlocking_signal = interlocking_signal;
    ext.starting_signal = starting_signal;

    ext.cart_id = cart_id;

    let ext_clone = ext.clone();
    track_tool::register_track_tool(game_builder, &ext_clone)?;
    game_builder
        .inner
        .items_mut()
        .register_item(game_state::items::Item {
            place_on_block_handler: Some(Box::new(move |ctx, _placement_coord, anchor, stack| {
                place_cart(ctx, anchor, stack, ext_clone.clone())
            })),
            ..Item::default_with_proto(protocol::items::ItemDef {
                short_name: "carts:minecart".to_string(),
                display_name: "High-speed minecart".to_string(),
                inventory_texture: Some(cart_tex.into()),
                groups: vec![],
                block_apperance: "".to_string(),
                interaction_rules: default_item_interaction_rules(),
                quantity_type: Some(QuantityType::Stack(256)),
                sort_key: "carts:cart".to_string(),
            })
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
    let (rail_pos, _variant) = if config.is_any_rail_block(block) {
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
    let config = config.clone();
    let popup = ctx
        .new_popup()
        .title("Spawn Cart")
        .text_field("cart_name", "Cart name", "", true, false)
        .text_field("cart_length", "Cart length", "8", true, false)
        .checkbox("board", "Board cart?", true, true)
        .button("spawn", "Spawn", true, true)
        .set_button_callback(Box::new(move |response: PopupResponse<'_>| {
            match response.user_action {
                PopupAction::PopupClosed => return Ok(()),
                PopupAction::ButtonClicked(b) => {
                    if b != "spawn" {
                        return Ok(());
                    }
                }
            }
            let cart_name = response
                .textfield_values
                .get("cart_name")
                .context("missing cart_name")?;
            let cart_length = response
                .textfield_values
                .get("cart_length")
                .expect("missing cart_length")
                .parse::<u32>()
                .context("bad cart_length")?;
            let board_cart = response
                .checkbox_values
                .get("board")
                .context("Missing board")?;

            if let Err(e) = actually_spawn_cart(
                config.clone(),
                &response.ctx,
                &cart_name,
                cart_length,
                rail_pos,
                variant,
                *board_cart,
            ) {
                response.ctx.initiator().send_chat_message(
                    ChatMessage::new_server_message(e.to_string()).with_color(SERVER_ERROR_COLOR),
                )?;
            }
            Ok(())
        }));

    match ctx.initiator() {
        EventInitiator::Player(p) => p.player.show_popup_blocking(popup)?,
        EventInitiator::WeakPlayerRef(wp) => {
            match wp.try_to_run(|p| p.show_popup_blocking(popup)) {
                Some(_) => {}
                None => {
                    tracing::warn!("Weak player ref tried to show popup but failed");
                }
            }
        }
        _ => tracing::warn!("Not a player"),
    };

    Ok(stack.decrement())
}

fn actually_spawn_cart(
    config: CartsGameBuilderExtension,
    ctx: &HandlerContext,
    cart_name: &str,
    cart_length: u32,
    rail_pos: BlockCoordinate,
    variant: u16,
    board_cart: bool,
) -> Result<()> {
    let initial_state =
        ScanState::spawn_at(rail_pos, (variant as u8 + 2) % 4, ctx.game_map(), &config)?;
    let initial_state = match initial_state {
        Some(x) => x,
        None => {
            ctx.initiator().send_chat_message(
                ChatMessage::new_server_message("Can't spawn").with_color(SERVER_ERROR_COLOR),
            )?;
            return Ok(());
        }
    };

    anyhow::ensure!(cart_length >= 1, "Cart length must be at least 1");
    anyhow::ensure!(cart_length <= 256, "Cart length must be at most 256");

    static ID_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

    let mut trailing_entities = Vec::new();
    for i in 1..cart_length {
        trailing_entities.push(TrailingEntity {
            class_id: config.cart_id.as_u32(),
            trailing_distance: i as f32,
        });
    }

    let id = ctx.entities().new_entity_blocking(
        initial_state.vec_coord(),
        Some(Box::pin(CartCoroutine {
            config: config.clone(),
            cart_name: cart_name.to_string(),
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
            precomputed_steps: None,
            pending_actions: BinaryHeap::new(),
            cart_length,

            interlocking_resume_state: None,
            cancellation: CancellationToken::new(),
            spawned_task_count: AsyncRefcount::new(),
        })),
        EntityTypeId {
            class: dbg!(config.cart_id),
            data: None,
        },
        Some(trailing_entities.into_boxed_slice()),
    );

    ctx.initiator().send_chat_message(
        ChatMessage::new_server_message(format!("Spawned cart with id {}", id))
            .with_color(SERVER_MESSAGE_COLOR),
    )?;

    if board_cart {
        let attach_target = EntityTarget {
            entity_id: id,
            trailing_entity_index: 0,
        };
        match ctx.initiator() {
            game_state::event::EventInitiator::Player(p) => {
                p.player.attach_to_entity_blocking(attach_target)?
            }
            game_state::event::EventInitiator::WeakPlayerRef(wp) => {
                match wp.try_to_run(|p| p.attach_to_entity_blocking(attach_target)) {
                    Some(_) => {}
                    None => {
                        tracing::warn!("Weak player ref tried to board entity but failed");
                    }
                }
            }
            _ => tracing::warn!("Not a player"),
        }
    }
    ctx.initiator()
        .send_chat_message(ChatMessage::new_server_message(format!(
            "Boarded cart with ID {id}; \
    /detach_from_entity to get off; \
    /attach_to_entity {id} to re-board"
        )))?;

    Ok(())
}

fn b2vec(b: BlockCoordinate) -> Vector3<f64> {
    Vector3::new(b.x as f64, b.y as f64, b.z as f64)
}

/// A segment of track where we know we can run
#[derive(Copy, Clone, Debug)]
struct TrackSegment {
    from: Vector3<f64>,
    to: Vector3<f64>,
    // The maximum speed we can run in this track segment
    max_speed: f64,
    // The odometer value at the start of this segment
    starting_odometer: f64,
}
impl TrackSegment {
    #[allow(unused)]
    fn manhattan_dist(&self) -> f64 {
        (self.from.x - self.to.x).abs()
            + (self.from.y - self.to.y).abs()
            + (self.from.z - self.to.z).abs()
    }
    fn distance(&self) -> f64 {
        let dx = self.from.x - self.to.x;
        let dy = self.from.y - self.to.y;
        let dz = self.from.z - self.to.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    fn any_content(&self) -> bool {
        self.distance() > 0.0
    }

    fn split_at_offset(&self, offset: f64) -> (Self, Option<Self>) {
        if offset > 0.0 && offset < self.distance() {
            let split_point = self.from + (self.to - self.from).normalize() * offset;
            (
                Self {
                    from: self.from,
                    to: split_point,
                    max_speed: self.max_speed,
                    starting_odometer: self.starting_odometer,
                },
                Some(Self {
                    from: split_point,
                    to: self.to,
                    max_speed: self.max_speed,
                    starting_odometer: self.starting_odometer + offset,
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
    speed: f64,
    acceleration: f64,
    move_time: f64,
    // Used for debugging; only costs one pointer so worth keeping for ease of development
    #[allow(unused)]
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
                face_direction: f32::atan2(displacement_f32.x, displacement_f32.z),
                pitch: f32::atan2(
                    displacement_f32.y,
                    f32::hypot(displacement_f32.x, displacement_f32.z),
                ),
                move_time: self.move_time as f32,
            })
        }
    }

    fn event_time(&self, event_odometer: f64) -> f64 {
        let relative_odometer = event_odometer - self.segment.starting_odometer;
        if relative_odometer < -0.0001 {
            panic!(
                "odometer {} is before start of segment {}",
                event_odometer, self.segment.starting_odometer
            );
        } else if relative_odometer > (self.segment.distance() + 0.0001) {
            panic!(
                "odometer {} is after end of segment {}",
                event_odometer, self.segment.starting_odometer
            );
        }
        // solving a quadratic of the form
        // relative_odometer = 0.5 * a0 * t^2 + v0 * t
        // 0 = 0.5 * a0 * t^2 + v0 * t - relative_odometer
        let a = self.acceleration / 2.0;
        let b = self.speed;
        let c = -relative_odometer;
        let t = if a.abs() < 0.01 {
            -c / b
        } else {
            (-b + (b * b - 4.0 * a * c).sqrt()) / (2.0 * a)
        };
        tracing::debug!("a {} b {} c {} t {}", a, b, c, t);
        tracing::debug!("odometer time for segment {:?} is {}", self, t);
        t
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum PendingAction {
    SignalRelease(BlockCoordinate, BlockId),
    SignalEnterBlock(BlockCoordinate, BlockId),
    StartingSignalEnterBlockReverse(BlockCoordinate, BlockId),
    SwitchRelease(BlockCoordinate, BlockId),
}

#[derive(Copy, Clone, Debug)]
struct PendingActionEntry {
    odometer: f64,
    action: PendingAction,
}

impl PartialEq for PendingActionEntry {
    fn eq(&self, other: &Self) -> bool {
        self.odometer == other.odometer
    }
}

impl PartialOrd for PendingActionEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for PendingActionEntry {}

impl Ord for PendingActionEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .odometer
            .partial_cmp(&self.odometer)
            .expect("Invalid comparison of odometers - NaN or infinity?")
            .then_with(|| self.action.cmp(&other.action))
    }
}

mod util {
    use std::future::Future;
    use std::sync::atomic::Ordering;

    struct AsyncRefcountInner {
        count: std::sync::atomic::AtomicU32,
        notification: tokio::sync::Notify,
    }

    pub(super) struct AsyncRefcountHandle {
        inner: std::sync::Arc<AsyncRefcountInner>,
    }
    impl Drop for AsyncRefcountHandle {
        fn drop(&mut self) {
            if self.inner.count.fetch_sub(1, Ordering::Release) <= 1 {
                self.inner.notification.notify_waiters();
            }
        }
    }

    pub(super) struct AsyncRefcount {
        inner: std::sync::Arc<AsyncRefcountInner>,
        waiting: bool,
    }
    impl AsyncRefcount {
        pub(super) fn new() -> AsyncRefcount {
            AsyncRefcount {
                inner: std::sync::Arc::new(AsyncRefcountInner {
                    count: std::sync::atomic::AtomicU32::new(0),
                    notification: tokio::sync::Notify::new(),
                }),
                waiting: false,
            }
        }

        /// Attempts to increment the reference count, returning None if there's already a waiter
        pub(super) fn acquire(&mut self) -> Option<AsyncRefcountHandle> {
            if self.waiting {
                None
            } else {
                self.inner.count.fetch_add(1, Ordering::Relaxed);
                Some(AsyncRefcountHandle {
                    inner: self.inner.clone(),
                })
            }
        }

        pub(super) fn wait(&mut self) -> impl Future<Output = ()> {
            self.waiting = true;
            let inner_clone = self.inner.clone();
            async move {
                while inner_clone.count.load(Ordering::Acquire) > 0 {
                    inner_clone.notification.notified().await;
                }
            }
        }
    }
}

pub(crate) const MAX_ACCEL: f64 = 8.0;
struct CartCoroutine {
    config: CartsGameBuilderExtension,
    // Segments where we've already calculated a braking curve. (clearance, starting speed, acceleration, time)
    scheduled_segments: VecDeque<ScheduledSegment>,
    // Segments where we don't yet have a braking curve
    unplanned_segments: VecDeque<TrackSegment>,
    // The last speed post we encountered while scanning
    last_speed_post_indication: f64,
    // The last velocity we already delivered to the entity system
    last_submitted_move_exit_speed: f64,
    id: u32,
    // Track scan state
    scan_state: ScanState,
    // TODO improve this as needed
    cleared_signals: FxHashMap<BlockCoordinate, BlockId>,
    held_signal: Option<(BlockCoordinate, BlockId)>,
    // If we got a planned path from the interlocking system, this is the path
    precomputed_steps: Option<Vec<InterlockingStep>>,
    // The cart name, used for interlockings
    cart_name: String,
    // Pending actions that need to happen once the cart reaches a specific point
    pending_actions: BinaryHeap<PendingActionEntry>,
    // debug only
    spawn_time: Instant,
    // only 1 supported for now
    cart_length: u32,

    interlocking_resume_state: Option<InterlockingResumeState>,
    cancellation: CancellationToken,
    spawned_task_count: AsyncRefcount,
}
impl EntityCoroutine for CartCoroutine {
    fn plan_move(
        mut self: std::pin::Pin<&mut Self>,
        services: &EntityCoroutineServices<'_>,
        _current_position: Vector3<f64>,
        whence: Vector3<f64>,
        when: f32,
        queue_space: usize,
    ) -> CoroutineResult {
        let trace_buffer = TraceBuffer::new(false);
        self.plan_move_impl(services, whence, when, queue_space, trace_buffer)
    }
    fn continuation(
        mut self: std::pin::Pin<&mut Self>,
        services: &EntityCoroutineServices<'_>,
        _current_position: Vector3<f64>,
        whence: Vector3<f64>,
        when: f32,
        queue_space: usize,
        continuation_result: ContinuationResult,
        trace_buffer: TraceBuffer,
    ) -> CoroutineResult {
        trace_buffer.log("In continuation");
        if continuation_result.tag == CONTINUATION_TAG_SIGNAL {
            match continuation_result.value {
                Ok(ContinuationResultValue::GetBlock(block_id, coord)) => {
                    self.cleared_signals.insert(coord, block_id);
                }
                Ok(ContinuationResultValue::HeapResult(result)) => {
                    match result.downcast::<Option<InterlockingRoute>>() {
                        Ok(steps) => {
                            match *steps {
                                Some(steps) => {
                                    if self.precomputed_steps.is_some() {
                                        tracing::warn!(
                                            "precomputed steps is not empty: {:?}",
                                            self.precomputed_steps
                                        )
                                    }
                                    self.precomputed_steps = Some(steps.steps);
                                    self.interlocking_resume_state = steps.resume;
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
                        // err from downcasting
                        Err(_) => {
                            log::error!("Unexpected heap result");
                        }
                    }
                }
                // Err from the deferred task
                Err(e) => {
                    log::error!("Cart coroutine got an error: {e:?}");
                    // Consider best-effort release of held signals
                    return CoroutineResult::Successful(EntityMoveDecision::ImmediateDespawn);
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

    fn pre_delete(mut self: Pin<&mut Self>, services: &EntityCoroutineServices<'_>) {
        let mut work_items = vec![];
        work_items.extend(self.pending_actions.drain().map(|x| x.action));
        if let Some(signal) = self.held_signal {
            work_items.push(PendingAction::SignalRelease(signal.0, signal.1));
        }
        let config_clone = self.config.clone();
        let wait_for_tasks = self.spawned_task_count.wait();
        self.cancellation.cancel();
        let cart_id = self.id;
        services.spawn_async::<(), _>(move |ctx| async move {
            wait_for_tasks.await;

            for work in work_items.into_iter() {
                if let Err(e) = Self::handle_pending_action(&ctx, &config_clone, work) {
                    tracing::error!("Failed to handle pending action in pre_delete: {e:?}")
                }
            }
            tracing::info!("All held signals of cart {} cleaned up", cart_id);
        });
    }
}

impl std::fmt::Debug for CartCoroutine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CartCoroutine")
    }
}

const CONTINUATION_TAG_SIGNAL: u32 = 0x516ea1;

impl CartCoroutine {
    fn plan_move_impl(
        &mut self,
        services: &EntityCoroutineServices<'_>,
        whence: Vector3<f64>,
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

        let maybe_deferral = self.scan_tracks(services, 2048.0, when, &trace_buffer);
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
        trace_buffer: &TraceBuffer,
        new_state: ScanState,
        buffer_time_estimate: f32,
    ) -> ReenterableResult<SignalResult> {
        if let Some(speed) = self.config.parse_speedpost(signal_block) {
            SignalResult::SpeedRestriction(speed).into()
        } else if signal_block.equals_ignore_variant(self.config.automatic_signal) {
            let rotation = signal_block.variant() & 0b11;
            // Use the new state since we want to check whether we are allowed to enter the block
            // (based on the face direction of that proposed state)
            if !new_state.signal_rotation_ok(rotation) {
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
        } else if signal_block.equals_ignore_variant(self.config.interlocking_signal)
            || signal_block.equals_ignore_variant(self.config.starting_signal)
        {
            // This uses the old state, since the interlocking code will itself validate entry
            let state_clone = self.scan_state.clone();
            let config_clone = self.config.clone();

            tracing::debug!(
                ">>>> interlocking signal start at {:?}",
                state_clone.block_coord
            );
            let trace_buffer = trace_buffer.clone();
            let cart_name_clone = self.cart_name.clone();
            let resume = self.interlocking_resume_state.clone();
            let last_speed_post = self.last_speed_post_indication as f32;

            // If true, we want to add a bit of a delay before actually resuming out of the interlocking
            // This is a fudge factor to avoid path optimizer thrashing as carts come up to speed, run into
            // red signals ahead, and then come to a stop then start again cyclically
            let starting_from_standstill =
                self.last_segment_exit_speed() <= 0.001 || buffer_time_estimate < 0.001;
            let cart_id = self.id;
            return ReenterableResult::Deferred(services.defer_async(move |ctx| async move {
                let state = state_clone;
                trace_buffer.log("Interlocking signal deferred");
                if starting_from_standstill {
                    tracing::debug!("Will delay once the interlocking clears");
                }

                let result = interlocking::interlock_cart(
                    ctx,
                    state,
                    &cart_name_clone,
                    cart_id,
                    1024,
                    config_clone,
                    resume,
                    last_speed_post,
                    starting_from_standstill,
                    buffer_time_estimate,
                    signal_coord,
                )
                .await;
                trace_buffer.log("Interlocking signal done");
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
        self.apply_precomputed_steps(trace_buffer);

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

        tracing::debug!(">>>> steps: {}, buffer time estimate: {}, estimated max speed: {}, max steps ahead: {}", steps, buffer_time_estimate, estimated_max_speed, max_steps_ahead);

        if self.unplanned_segments.is_empty() {
            tracing::debug!("unplanned segments empty, adding new");
            let empty_segment = TrackSegment {
                from: self.scan_state.vec_coord(),
                to: self.scan_state.vec_coord(),
                max_speed: self
                    .last_speed_post_indication
                    .min(self.scan_state.allowable_speed as f64),
                starting_odometer: self.scan_state.odometer,
            };
            self.unplanned_segments.push_back(empty_segment);
        }

        let block_getter = |coord| services.get_block(coord);

        'scan_loop: while steps < max_steps_ahead
            && buffer_time_estimate < (5.0 + 2.0 * estimated_max_speed / MAX_ACCEL as f32)
        {
            // Precondition: self.scan_state is valid
            let new_state = match self.scan_state.advance::<false>(block_getter, &self.config) {
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
                if let Some(signal_block_id) = self.cleared_signals.remove(&signal_coord) {
                    tracing::debug!("cached clear signal at {:?}", signal_coord);
                    if signal_block_id.equals_ignore_variant(0.into()) {
                        // contended signal
                        break 'scan_loop;
                    } else {
                        tracing::debug!("Cached permissive signal at {:?}", new_state.block_coord);
                        self.pending_actions.push(PendingActionEntry {
                            odometer: new_state.odometer,
                            action: PendingAction::SignalEnterBlock(signal_coord, signal_block_id),
                        });
                        if let Some(held_signal) = self.held_signal.take() {
                            self.pending_actions.push(PendingActionEntry {
                                odometer: new_state.odometer + self.cart_length as f64,
                                action: PendingAction::SignalRelease(held_signal.0, held_signal.1),
                            });
                        }
                        self.held_signal = Some((signal_coord, signal_block_id));
                    }
                }

                let signal_block = match services.get_block(signal_coord) {
                    DeferrableResult::AvailableNow(x) => x.unwrap(),
                    DeferrableResult::Deferred(d) => {
                        tracing::debug!("signal lookup deferral");
                        return Some(d.defer_and_reinvoke(1));
                    }
                };

                let signal_result = match self.parse_signal(
                    services,
                    signal_coord,
                    signal_block,
                    &trace_buffer,
                    new_state,
                    buffer_time_estimate,
                ) {
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
                        self.pending_actions.push(PendingActionEntry {
                            odometer: new_state.odometer,
                            action: PendingAction::SignalEnterBlock(signal_coord, signal_block),
                        });
                        if let Some(held_signal) = self.held_signal.take() {
                            self.pending_actions.push(PendingActionEntry {
                                odometer: new_state.odometer + self.cart_length as f64,
                                action: PendingAction::SignalRelease(held_signal.0, held_signal.1),
                            });
                        }
                        self.held_signal = Some((signal_coord, signal_block));
                        // break 'signal_loop,
                    }
                    SignalResult::SpeedRestriction(speed) => {
                        self.last_speed_post_indication = speed as f64;
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

    fn apply_precomputed_steps(&mut self, trace_buffer: &TraceBuffer) {
        if self.precomputed_steps.is_some() {
            trace_buffer.log("Using precomputed steps");

            let steps = self.precomputed_steps.take().unwrap();

            for (i, step) in steps.into_iter().enumerate() {
                tracing::debug!("step {} at {:?}", i, step.scan_state.block_coord);
                let state = step.scan_state;

                if let Some((switch_coord, block_id, deferral)) = step.clear_switch {
                    tracing::debug!("Clearing switch at {:?}", switch_coord);
                    self.pending_actions.push(PendingActionEntry {
                        odometer: state.odometer + deferral as f64 + self.cart_length as f64,
                        action: PendingAction::SwitchRelease(switch_coord, block_id),
                    })
                }

                if let Some(speed) = step.speed_post {
                    self.last_speed_post_indication = speed as f64;
                }

                if let Some(signal_coord) = state.block_coord.try_delta(0, 2, 0) {
                    if step.acquired_signal != BlockId::from(0) {
                        tracing::debug!("Acquiring signal at {:?}", state.block_coord);
                        tracing::debug!(
                            "Releasing signal at {:?}",
                            self.held_signal.as_ref().map(|x| x.0)
                        );

                        let action = if step.acquired_signal_was_reverse_starting {
                            PendingAction::StartingSignalEnterBlockReverse(
                                signal_coord,
                                step.acquired_signal,
                            )
                        } else {
                            PendingAction::SignalEnterBlock(signal_coord, step.acquired_signal)
                        };

                        self.pending_actions.push(PendingActionEntry {
                            odometer: state.odometer,
                            action,
                        });
                        if let Some(held_signal) = self.held_signal.take() {
                            self.pending_actions.push(PendingActionEntry {
                                odometer: state.odometer + self.cart_length as f64,
                                action: PendingAction::SignalRelease(held_signal.0, held_signal.1),
                            });
                        }

                        self.held_signal = Some((signal_coord, step.acquired_signal));
                    }
                }
                self.apply_step(state);
            }
        }
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
            tracing::debug!("Splitting a segment due to effective speed at {:?} (our vec coord {:?}); prev length was {}, speed changing {} -> {}", last_move.to,  new_state.vec_coord(), last_move_delta.magnitude(), last_move.max_speed, effective_speed);
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
    ) -> VecDeque<(TrackSegment, f64)> {
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

        let available_scheduled_segments = (64usize.checked_sub(queue_space).unwrap())
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
        let mut max_exit_speed: f64 = 0.0;

        let mut total_skipped_segments = 0.0;

        let panic_max_index = self.estimate_panic_max_index();
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
            let should_panic_schedule = available_scheduled_segments < 2
                || self.last_submitted_move_exit_speed < 0.001
                || when < 2.0;

            if unconditionally_schedulable {
                schedulable_segments.push_front((*seg, max_exit_speed));
            } else if should_panic_schedule && idx <= panic_max_index && when < 2.0 {
                // We're low on cached moves, so let's schedule this segment
                schedulable_segments.push_front((*seg, max_exit_speed));
                tracing::debug!(
                    "> panic scheduling! seg_schedulable = {}, seg = {:?}",
                    unconditionally_schedulable,
                    seg
                );
                unconditionally_schedulable = true;
            } else {
                total_skipped_segments += seg.distance();
            }

            if limited_by_seg_speed {
                unconditionally_schedulable = true;
            }
            // The max exit speed of the preeding segment is the same as the max entry speed of the current segment
            // (after limiting for track speed)
            max_exit_speed = max_entry_speed;
        }

        tracing::debug!("Skipped {} blocks", total_skipped_segments);

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
            tracing::debug!(
                "odometer check: track state says {}, segments say {}",
                self.scan_state.odometer,
                prev_segment.starting_odometer + prev_segment.distance()
            );
            let empty_segment = TrackSegment {
                from: prev_segment.to,
                to: prev_segment.to,
                max_speed: prev_segment.max_speed,
                starting_odometer: prev_segment.starting_odometer + prev_segment.distance(),
            };
            self.unplanned_segments.push_back(empty_segment);
        } else {
            tracing::debug!("old segment was empty, not starting new one");
        }
    }

    fn schedule_segments(&mut self, schedulable_segments: VecDeque<(TrackSegment, f64)>) {
        // Start with the last segment's exit speed. Note that if we sent all of our segments
        // to the entity system, self.scheduled_segments would be empty, so we would use
        // the speed stored in self.last_submitted_move_exit_speed instead.
        let mut last_segment_exit_speed = self.last_segment_exit_speed();

        // Each segment contributes one or more moves to the internal move queue
        for (seg, brake_curve_exit_speed) in schedulable_segments.into_iter() {
            last_segment_exit_speed =
                self.schedule_single_segment(seg, last_segment_exit_speed, brake_curve_exit_speed);
        }
    }

    fn last_segment_exit_speed(&self) -> f64 {
        self.scheduled_segments
            .back()
            .map(|x| x.speed + (x.acceleration * x.move_time))
            .unwrap_or(self.last_submitted_move_exit_speed)
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
                seg
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
        if entrance_accel_distance < 0.0000001 && exit_accel_distance < 0.000001 {
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
        } else if entrance_accel_distance > 0.0 && exit_accel_distance < 0.0000001 {
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
        } else if entrance_accel_distance < 0.0000001 && exit_accel_distance > 0.000001 {
            // Case 3: We entered at track speed, but we need to slow down before exiting
            if exit_accel_distance > seg_distance {
                // This shouldn't happen; we're braking the cart with more acceleration than
                // we used when planning the brake curve

                let new_accel = (enter_speed * enter_speed
                    - brake_curve_exit_speed * brake_curve_exit_speed)
                    / (2.0 * seg_distance);
                assert!(new_accel > 0.0);
                if new_accel > (MAX_ACCEL + 1e-3) {
                    tracing::warn!(
                    "exit_accel_distance {} > seg_distance {}, decelerating {} => {} using accel of {}",
                    exit_accel_distance,
                    seg_distance,
                    enter_speed,
                    brake_curve_exit_speed,
                    new_accel
                );
                }

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

                if new_accel > (MAX_ACCEL + 1e-3) {
                    tracing::warn!(
                        "exit_accel_distance {} > seg_distance {}, decelerating {} => {} using accel of {}",
                        exit_accel_distance,
                        seg_distance,
                        enter_speed,
                        brake_curve_exit_speed,
                        new_accel
                    );
                }

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
            return CoroutineResult::Successful(
                game_state::entities::EntityMoveDecision::AskAgainLaterFlexible(0.5..1.0),
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

            let odometer_end = segment.segment.starting_odometer + segment.segment.distance();

            loop {
                if self.pending_actions.is_empty() {
                    break;
                }
                let action = self.pending_actions.peek().unwrap();
                if action.odometer > odometer_end {
                    break;
                }

                let action = self.pending_actions.pop().unwrap();
                tracing::debug!("action {:?} at {}", action.action, action.odometer);

                let odometer_time = segment.event_time(action.odometer);
                let action_time = (returned_moves_time + when as f64 + odometer_time).max(0.0);
                let config_clone = self.config.clone();
                let cancel_clone = self.cancellation.clone();
                let ref_clone = self
                    .spawned_task_count
                    .acquire()
                    .expect("self.spawned_task_count.acquire should succeed unless scheduling was invoked after pre_remove");
                services.spawn_delayed(
                    Duration::from_secs_f64(action_time),
                    move |ctx| {
                        Self::handle_pending_action(ctx, &config_clone, action.action).unwrap();
                        drop(ref_clone);
                    },
                    Some(cancel_clone),
                )
            }

            returned_moves_time += segment.move_time;

            if let Some(movement) = segment.to_movement() {
                returned_moves.push(movement);
            } else if !segment.segment.any_content() {
                tracing::warn!("Segment {:?} had no movement", segment);
            }
        }
        tracing::debug!("returning {} moves", returned_moves.len());
        trace_buffer.log("Done!");
        CoroutineResult::Successful(game_state::entities::EntityMoveDecision::QueueUpMultiple(
            returned_moves,
        ))
    }

    fn handle_pending_action(
        ctx: &HandlerContext,
        config: &CartsGameBuilderExtension,
        action: PendingAction,
    ) -> Result<()> {
        match action {
            PendingAction::SignalEnterBlock(enter_signal_coord, enter_signal_block) => ctx
                .game_map()
                .mutate_block_atomically(enter_signal_coord, |b, _ext| {
                    tracing::debug!("entering block");
                    signals::signal_enter_block(enter_signal_coord, b, enter_signal_block);
                    Ok(())
                }),
            PendingAction::StartingSignalEnterBlockReverse(
                enter_signal_coord,
                enter_signal_block,
            ) => ctx
                .game_map()
                .mutate_block_atomically(enter_signal_coord, |b, _ext| {
                    tracing::debug!("entering block");
                    signals::starting_signal_reverse_enter_block(
                        enter_signal_coord,
                        b,
                        enter_signal_block,
                    );
                    Ok(())
                }),
            PendingAction::SignalRelease(exit_signal_coord, exit_signal_block) => ctx
                .game_map()
                .mutate_block_atomically(exit_signal_coord, |b, _ext| {
                    signals::signal_release(exit_signal_coord, b, exit_signal_block);
                    Ok(())
                }),
            PendingAction::SwitchRelease(pass_switch_coord, pass_switch_block) => {
                let switch_unset = config.switch_unset;
                match ctx.game_map().compare_and_set_block(
                    pass_switch_coord,
                    pass_switch_block,
                    switch_unset,
                    None,
                    false,
                )? {
                    (game_state::game_map::CasOutcome::Match, _, _) => {
                        tracing::debug!(
                            "Pass switch match: got {:?} at {:?}",
                            pass_switch_block,
                            pass_switch_coord
                        );
                    }
                    (game_state::game_map::CasOutcome::Mismatch, id, _) => {
                        tracing::warn!(
                            "Pass switch mismatch: got {:?} but expected {:?} at {:?}",
                            id,
                            pass_switch_block,
                            pass_switch_coord
                        );
                    }
                }
                Ok(())
            }
        }
    }

    fn estimate_panic_max_index(&self) -> usize {
        let mut total_time_estimate = 0.0;
        for (idx, seg) in self.unplanned_segments.iter().enumerate() {
            let seg_time_estimate = seg.distance() / seg.max_speed;
            total_time_estimate += seg_time_estimate;
            if total_time_estimate > 2.0 {
                return idx;
            }
        }
        self.unplanned_segments.len() - 1
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
