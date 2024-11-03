use std::num::ParseIntError;
use std::str::FromStr;
use std::sync::{Arc, Weak};
use std::time::Duration;

use anyhow::{Context, Result};
use lazy_static::lazy_static;
use prost::Message;
use rhai::packages::{Package, StandardPackage};
use rhai::{def_package, OptimizationLevel};
use smallvec::SmallVec;
use tokio::time::Instant;

use perovskite_core::block_id::BlockId;
use perovskite_core::chat::ChatMessage;
use perovskite_core::constants::item_groups::HIDDEN_FROM_CREATIVE;
use perovskite_core::coordinates::BlockCoordinate;
use perovskite_server::game_state::blocks::{
    BlockType, ExtDataHandling, ExtendedData, FastBlockName,
};
use perovskite_server::game_state::client_ui::{Popup, PopupAction, UiElementContainer};
use perovskite_server::game_state::event::HandlerContext;

use crate::blocks::{AaBoxProperties, BlockBuilder, RotationMode};
use crate::circuits::events::{make_root_context, transmit_edge, CircuitHandlerContext};
use crate::circuits::gates::{get_side_tex, make_chip_shape};
use crate::circuits::{
    get_incoming_pin_states, BlockConnectivity, CircuitBlockBuilder, CircuitBlockCallbacks,
    CircuitBlockProperties, CircuitGameBuilder, PinState,
};
use crate::game_builder::{GameBuilder, StaticBlockName, StaticTextureName};
use crate::include_texture_bytes;

const UNBROKEN_NAME: StaticBlockName = StaticBlockName("circuits:basic_microcontroller");
const BROKEN_NAME: StaticBlockName = StaticBlockName("circuits:basic_microcontroller_broken");
const UNBROKEN_TEXTURE: StaticTextureName = StaticTextureName("circuits:basic_microcontroller");
const BROKEN_TEXTURE: StaticTextureName =
    StaticTextureName("circuits:basic_microcontroller_broken");

const CONN_PORT: [BlockConnectivity; 4] = [
    BlockConnectivity::unrotated(1, 0, 0, 1),
    BlockConnectivity::unrotated(0, 0, 1, 2),
    BlockConnectivity::unrotated(-1, 0, 0, 4),
    BlockConnectivity::unrotated(0, 0, -1, 8),
];
// Variants indicating what ports the microcontroller is driving high
const VAR_PORT: [u16; 4] = [1, 2, 4, 8];

#[derive(Clone, Copy, Debug)]
struct MicrocontrollerIds {
    ok: BlockId,
    broken: BlockId,
}

impl MicrocontrollerIds {
    fn is_microcontroller(&self, id: BlockId) -> bool {
        self.ok.equals_ignore_variant(id) || self.broken.equals_ignore_variant(id)
    }
}

struct MicrocontrollerCircuitCallbacks {
    ids: MicrocontrollerIds,
}

impl CircuitBlockCallbacks for MicrocontrollerCircuitCallbacks {
    fn on_incoming_edge(
        &self,
        ctx: &CircuitHandlerContext<'_>,
        coord: BlockCoordinate,
        from: BlockCoordinate,
        incoming_state: PinState,
    ) -> Result<()> {
        let (core, selected_interrupt, pin_reg) =
            match ctx.game_map().mutate_block_atomically(coord, |id, ext| {
                if !id.equals_ignore_variant(self.ids.ok) {
                    tracing::info!("failed block id check");
                    return Ok(None);
                }
                let extended_data = match ext.as_mut() {
                    Some(x) => x,
                    None => return Ok(None),
                };
                let state: &mut MicrocontrollerExtendedData = match extended_data
                    .custom_data
                    .as_mut()
                    .and_then(|x| x.downcast_mut())
                {
                    Some(x) => x,
                    None => return Ok(None),
                };
                tracing::info!("downcasted");
                if state.microcontroller_config.program.is_empty() {
                    tracing::info!("no program");
                    return Ok(None);
                }
                tracing::info!("found program {:?}", state.microcontroller_config);

                let self_variant = id.variant();
                let mut pin_reg = 0;
                for (i, (port_mask, connectivity)) in
                    VAR_PORT.into_iter().zip(CONN_PORT).enumerate()
                {
                    tracing::info!(
                        "{}, {}, {:?}, {:?}",
                        port_mask,
                        self_variant,
                        from,
                        connectivity.eval(coord, self_variant)
                    );
                    let driven_state = (port_mask & self_variant) != 0;
                    if connectivity.eval(coord, self_variant) == Some(from) {
                        let effective_state = driven_state | (incoming_state == PinState::High);
                        if effective_state {
                            pin_reg |= 1 << i;
                        }
                    } else {
                        pin_reg |= state.microcontroller_config.external_pin_drives & (1 << i);
                    }
                }
                tracing::info!(
                    "pin_reg　{}, epd: {}",
                    pin_reg,
                    state.microcontroller_config.external_pin_drives
                );
                let rising_interrupts = pin_reg
                    & !state.microcontroller_config.external_pin_drives
                    & state.microcontroller_config.rising_interrupt_mask;
                let falling_interrupts = !pin_reg
                    & state.microcontroller_config.external_pin_drives
                    & state.microcontroller_config.falling_interrupt_mask;
                state.microcontroller_config.pending_interrupts |= rising_interrupts;
                state.microcontroller_config.pending_interrupts |= falling_interrupts;
                dbg!((
                    rising_interrupts,
                    falling_interrupts,
                    state.microcontroller_config.pending_interrupts
                ));
                let selected_interrupt =
                    max_set_bit(state.microcontroller_config.pending_interrupts);
                state.microcontroller_config.external_pin_drives = pin_reg;
                if selected_interrupt == 0 {
                    // No work to do
                    return Ok(None);
                }
                tracing::info!("irq {selected_interrupt}");
                assert_ne!(
                    selected_interrupt & state.microcontroller_config.pending_interrupts,
                    0
                );
                // The selected interrupt is no longer pending.
                state.microcontroller_config.pending_interrupts &= !selected_interrupt;

                // the core has work in progress
                let counter = ctx.startup_counter();
                if state.lock == counter {
                    // pending interrupts is updated, when that core finishes work it'll detect
                    // this and do more work. No work to do until the core finishes work

                    tracing::info!("core locked already");
                    Ok(None)
                } else {
                    // Apply the lock
                    state.lock = counter;
                    let core = Arc::downgrade(&state.core_state);

                    tracing::info!("got the core lock");
                    Ok(Some((core, selected_interrupt, pin_reg)))
                }
            })? {
                Some((core, selected_interrupt, pin_reg)) => (core, selected_interrupt, pin_reg),
                None => return Ok(()),
            };

        // The core itself is responsible for running the code, and then updating the game map
        // (under the advisory lock)
        ctx.run_deferred_async(|ctx2| {
            interrupt_poll_loop(ctx2, coord, core, selected_interrupt, pin_reg, self.ids)
        });
        Ok(())
    }

    fn sample_pin(
        &self,
        ctx: &CircuitHandlerContext<'_>,
        coord: BlockCoordinate,
        destination: BlockCoordinate,
    ) -> PinState {
        let block = match ctx.game_map().try_get_block(coord) {
            Some(block) => block,
            None => return super::PinState::Low,
        };
        let self_variant = block.variant();
        for (mask, connectivity) in VAR_PORT.into_iter().zip(CONN_PORT) {
            if connectivity.eval(coord, self_variant) == Some(destination) {
                return if self_variant & mask != 0 {
                    PinState::High
                } else {
                    PinState::Low
                };
            }
        }
        PinState::Low
    }

    fn on_overheat(
        &self,
        ctx: &HandlerContext,
        coord: perovskite_core::coordinates::BlockCoordinate,
    ) {
        break_microcontroller(ctx, coord, self.ids, "External overheat".to_string()).unwrap();
    }
}

async fn interrupt_poll_loop(
    ctx: HandlerContext<'static>,
    coord: BlockCoordinate,
    core: Weak<tokio::sync::Mutex<CoreState>>,
    mut next_interrupt: u8,
    mut pin_reg: u8,
    ids: MicrocontrollerIds,
) -> Result<()> {
    tracing::info!("IPL starting");
    let engine = build_rhai_engine();
    let final_port = loop {
        let upgraded = match core.upgrade() {
            // Core disappeared, e.g. block was unloaded or replaced in the map
            None => return Ok(()),
            Some(x) => x,
        };
        tracing::info!("upgraded");
        let mut lock = upgraded.lock().await;
        tracing::info!("locked");
        let now = Instant::now();
        lock.last_run_times.push_back(now);
        if lock.last_run_times.is_full()
            && lock
                .last_run_times
                .front()
                .is_some_and(|&x| (now - x) < Duration::from_secs(1))
        {
            // Overheating
            return break_microcontroller(
                &ctx,
                coord,
                ids,
                "Over 16 events in a second".to_string(),
            );
        }

        if lock.ast.is_none() {
            // We can't use get_or_insert_with because we need to bail on error
            let compiled = match tokio::task::block_in_place(|| engine.compile(&lock.program)) {
                Ok(x) => x,
                Err(e) => return break_microcontroller(&ctx, coord, ids, e.to_string()),
            };
            lock.ast = Some(compiled);
            tracing::info!("compiled");
        }
        let mut scope = rhai::Scope::new();
        scope.push("port", lock.port_register as rhai::INT);
        scope.push("pin", pin_reg as rhai::INT);
        scope.push("isrc", next_interrupt as rhai::INT);
        let eval_result = tokio::task::block_in_place(|| {
            engine.eval_ast_with_scope::<()>(&mut scope, lock.ast.as_ref().unwrap())
        });
        tracing::info!("evaluated");
        if let Err(e) = eval_result {
            return break_microcontroller(&ctx, coord, ids, e.to_string());
        }

        lock.port_register = match scope.get_value::<rhai::INT>("port") {
            None => {
                return break_microcontroller(
                    &ctx,
                    coord,
                    ids,
                    "port register corrupted".to_string(),
                );
            }
            Some(x) => x as u8,
        };
        tracing::info!("new port: {}", lock.port_register);
        match ctx.game_map().try_mutate_block_atomically(
            coord,
            |id, ext| {
                if !id.equals_ignore_variant(ids.ok) {
                    return Ok(LoopOutcome::Bail("mismatched id"));
                }
                let extended_data = match ext.as_mut() {
                    Some(x) => x,
                    None => return Ok(LoopOutcome::Bail("no extended data")),
                };
                let state: &mut MicrocontrollerExtendedData = match extended_data
                    .custom_data
                    .as_mut()
                    .and_then(|x| x.downcast_mut())
                {
                    Some(x) => x,
                    None => return Ok(LoopOutcome::Bail("downcast failed")),
                };

                tracing::info!("Downcasted");

                if !Arc::ptr_eq(&state.core_state, &upgraded) {
                    // Did the core get reprogrammed/replaced?
                    return Ok(LoopOutcome::Bail("Arc mismatch"));
                }
                tracing::info!("Core matches");

                let selected_interrupt =
                    max_set_bit(state.microcontroller_config.pending_interrupts);
                if selected_interrupt == 0 {
                    state.lock = 0;
                    return Ok(LoopOutcome::Finish(lock.port_register));
                }

                assert_ne!(
                    selected_interrupt & state.microcontroller_config.pending_interrupts,
                    0
                );
                // The selected interrupt is no longer pending.
                state.microcontroller_config.pending_interrupts &= !selected_interrupt;

                Ok(LoopOutcome::Continue(
                    next_interrupt,
                    state.microcontroller_config.external_pin_drives,
                ))
            },
            true,
        )? {
            Some(LoopOutcome::Continue(int, pin)) => {
                tracing::info!("continue IPL int {} pin {}", int, pin);
                next_interrupt = int;
                pin_reg = pin;
            }
            Some(LoopOutcome::Finish(port)) => break port,
            Some(LoopOutcome::Bail(reason)) => {
                tracing::warn!("Bailing out of microcontroller loop: {:?}", reason);
                return Ok(());
            }
            None => return Ok(()),
        };
    };
    tracing::info!("finished IPL");

    let mut edges: SmallVec<[(BlockCoordinate, BlockCoordinate, PinState); 4]> =
        smallvec::SmallVec::new();
    ctx.game_map().mutate_block_atomically(coord, |id, ext| {
        if !id.equals_ignore_variant(ids.ok) {
            return Ok(());
        }
        let mut variant = id.variant();
        tracing::info!("old variant {}", variant);
        for (i, (port_mask, connectivity)) in VAR_PORT.into_iter().zip(CONN_PORT).enumerate() {
            if (variant & port_mask == 0) ^ (final_port & (1 << i) == 0) {
                let is_rising = (final_port & 1 << i) != 0;
                if is_rising {
                    variant |= port_mask;
                } else {
                    variant &= !port_mask;
                }
                if let Some(target) = connectivity.eval(coord, variant) {
                    log::info!("push edge {:?}, {:?}, {:?}", target, coord, is_rising);
                    edges.push((target, coord, is_rising.into()));
                }
            }
        }
        tracing::info!("new variant: {}", variant);
        *id = id.with_variant_unchecked(variant);
        Ok(())
    })?;

    let cctx = make_root_context(&ctx);
    for (dst, src, edge) in edges {
        transmit_edge(&cctx, dst, src, edge)?;
    }
    Ok(())
}

enum LoopOutcome {
    Finish(u8),
    Continue(u8, u8),
    Bail(&'static str),
}

fn break_microcontroller(
    ctx: &HandlerContext,
    coord: BlockCoordinate,
    ids: MicrocontrollerIds,
    err: String,
) -> Result<()> {
    tokio::task::block_in_place(|| {
        ctx.game_map().mutate_block_atomically(coord, |id, ext| {
            if !ids.is_microcontroller(*id) {
                return Ok(());
            }
            let extended_data = ext.get_or_insert_with(|| ExtendedData::default());
            let downcasted: Option<Box<MicrocontrollerExtendedData>> = extended_data
                .custom_data
                .take()
                // If we have the wrong extended data here, we destroy it. But we're already convinced
                // that this should be a microcontroller based on block ID, and are about to break it
                .and_then(|x| x.downcast().ok());
            let mut custom_data = downcasted.map(|x| *x).unwrap_or_default();
            custom_data.microcontroller_config.last_error = Some(err);
            extended_data.custom_data = Some(Box::new(custom_data));
            *id = ids.broken;
            Ok(())
        })
    })
}

fn program_microcontroller(
    ctx: &HandlerContext,
    coord: BlockCoordinate,
    ids: MicrocontrollerIds,
    program: String,
    falling_mask: u8,
    rising_mask: u8,
) -> Result<()> {
    tokio::task::block_in_place(|| {
        let core = match ctx.game_map().mutate_block_atomically(coord, |id, ext| {
            if !ids.is_microcontroller(*id) {
                return Ok(None);
            }
            let extended_data = ext.get_or_insert_with(|| ExtendedData::default());
            let downcasted: Option<Box<MicrocontrollerExtendedData>> = extended_data
                .custom_data
                .take()
                // If we have the wrong extended data here, we destroy it. But we're already convinced
                // that this should be a microcontroller based on block ID, and are about to break it
                .and_then(|x| x.downcast().ok());
            let mut custom_data = downcasted.map(|x| *x).unwrap_or_default();
            custom_data.microcontroller_config.program = program.clone();
            custom_data.microcontroller_config.falling_interrupt_mask = falling_mask;
            custom_data.microcontroller_config.rising_interrupt_mask = rising_mask;
            custom_data.microcontroller_config.last_error = None;
            custom_data.lock = ctx.startup_counter();
            custom_data.core_state = Arc::new(tokio::sync::Mutex::new(CoreState {
                program,
                ..Default::default()
            }));
            let core = Arc::downgrade(&custom_data.core_state);
            extended_data.custom_data = Some(Box::new(custom_data));
            *id = ids.ok.with_variant_of(*id);
            Ok(Some(core))
        })? {
            Some(c) => c,
            None => {
                ctx.initiator()
                    .send_chat_message(ChatMessage::new_server_message("Not a microcontroller"))?;
                return Ok(());
            }
        };

        let mut pin_reg = 0;
        let states = get_incoming_pin_states(&make_root_context(&ctx), coord);
        log::info!("incoming pin states: {:?}", states);
        for (conn, _coord, state) in states {
            if state == PinState::High {
                pin_reg |= conn.id as u8;
            }
        }

        ctx.run_deferred_async(|ctx2| interrupt_poll_loop(ctx2, coord, core, 8, pin_reg, ids));
        Ok(())
    })
}

// For future use: 8, 16, 32, 64
// Initial program run
// Message packets
// Timer interrupts
// Watchdog
// Note that we can widen u8 -> u16 freely; this doesn't need to go into block ID variants

#[derive(Clone, Debug)]
struct MicrocontrollerConfig {
    program: String,
    last_error: Option<String>,
    external_pin_drives: u8,
    pending_interrupts: u8,
    falling_interrupt_mask: u8,
    rising_interrupt_mask: u8,
}
impl Default for MicrocontrollerConfig {
    fn default() -> Self {
        MicrocontrollerConfig {
            program: String::new(),
            rising_interrupt_mask: 0xf,
            falling_interrupt_mask: 0,
            pending_interrupts: 0,
            external_pin_drives: 0,
            last_error: None,
        }
    }
}

#[derive(Message)]
struct SerializedMicrocontrollerConfig {
    #[prost(string, tag = "1")]
    program: String,
    #[prost(string, tag = "2")]
    last_error: String,
    #[prost(uint32, tag = "3")]
    external_pin_drives: u32,
    #[prost(uint32, tag = "4")]
    pending_interrupts: u32,
    #[prost(uint32, tag = "5")]
    falling_interrupt_mask: u32,
    #[prost(uint32, tag = "6")]
    rising_interrupt_mask: u32,
}
impl TryInto<MicrocontrollerConfig> for SerializedMicrocontrollerConfig {
    type Error = anyhow::Error;

    fn try_into(self) -> std::result::Result<MicrocontrollerConfig, Self::Error> {
        Ok(MicrocontrollerConfig {
            program: self.program,
            last_error: if self.last_error.is_empty() {
                None
            } else {
                Some(self.last_error)
            },
            external_pin_drives: self.external_pin_drives.try_into()?,
            pending_interrupts: self.pending_interrupts.try_into()?,
            falling_interrupt_mask: self.falling_interrupt_mask.try_into()?,
            rising_interrupt_mask: self.rising_interrupt_mask.try_into()?,
        })
    }
}
impl Into<SerializedMicrocontrollerConfig> for &MicrocontrollerConfig {
    fn into(self) -> SerializedMicrocontrollerConfig {
        SerializedMicrocontrollerConfig {
            program: self.program.clone(),
            last_error: self.last_error.clone().unwrap_or_default(),
            external_pin_drives: self.external_pin_drives.into(),
            pending_interrupts: self.pending_interrupts.into(),
            falling_interrupt_mask: self.falling_interrupt_mask.into(),
            rising_interrupt_mask: self.rising_interrupt_mask.into(),
        }
    }
}

/// The configuration of the microcontroller
struct MicrocontrollerExtendedData {
    // 0 if unlocked, the current server startup counter if locked.
    lock: u64,
    microcontroller_config: MicrocontrollerConfig,
    core_state: Arc<tokio::sync::Mutex<CoreState>>,
}

impl Default for MicrocontrollerExtendedData {
    fn default() -> Self {
        MicrocontrollerExtendedData {
            microcontroller_config: Default::default(),
            lock: 0,
            core_state: Arc::new(tokio::sync::Mutex::new(CoreState::default())),
        }
    }
}

/// The actual state of the microcontroller, run on the event handler thread (or possibly on a
/// separate tokio task)
struct CoreState {
    program: String,
    ast: Option<rhai::AST>,
    // Used for overheat protection
    last_run_times: circular_buffer::CircularBuffer<16, Instant>,
    // The core updates this; the peripherals observe it
    port_register: u8,
    // todo: microcontroller memory
}
impl Default for CoreState {
    fn default() -> Self {
        Self {
            program: String::new(),
            ast: None,
            last_run_times: circular_buffer::CircularBuffer::new(),
            port_register: 0,
        }
    }
}

lazy_static! {
    static ref RHAI_INTERPRETER: rhai::Engine = build_rhai_engine();
    static ref SHARED_PACKAGE: RhaiGlobalPackage = RhaiGlobalPackage::new();
}

def_package! {
    pub RhaiGlobalPackage(module) : StandardPackage {
    }
}

fn build_rhai_engine() -> rhai::Engine {
    let mut engine = rhai::Engine::new_raw();
    SHARED_PACKAGE.register_into_engine(&mut engine);
    engine.set_max_variables(16);
    engine.set_fast_operators(true);
    engine.set_max_string_size(256);
    engine.set_max_array_size(256);
    engine.set_max_map_size(256);
    engine.set_max_operations(4096);
    engine.set_max_functions(8);
    engine.set_max_modules(0);
    engine.set_max_call_levels(8);
    engine.set_max_expr_depths(64, 64);
    engine.disable_symbol("print");
    engine.disable_symbol("debug");
    engine.disable_symbol("eval");
    engine.disable_symbol("sleep");
    engine.set_optimization_level(OptimizationLevel::Simple);

    engine
}

pub(super) fn register_microcontroller(builder: &mut GameBuilder) -> Result<()> {
    include_texture_bytes!(
        builder,
        UNBROKEN_TEXTURE,
        "../textures/microcontroller_top.png"
    )?;
    include_texture_bytes!(
        builder,
        BROKEN_TEXTURE,
        "../textures/microcontroller_top_broken.png"
    )?;

    let broken_box_properties = AaBoxProperties::new(
        get_side_tex(true),
        get_side_tex(true),
        BROKEN_TEXTURE,
        super::BOTTOM_TEX,
        get_side_tex(true),
        get_side_tex(true),
        crate::blocks::TextureCropping::AutoCrop,
        RotationMode::None,
    );

    let block_modifier = |block: &mut BlockType| {
        block.extended_data_handling = ExtDataHandling::ServerSide;
        let name_ok = FastBlockName::new(UNBROKEN_NAME.0);
        let name_broken = FastBlockName::new(BROKEN_NAME.0);
        block.interact_key_handler = Some(Box::new(move |ctx, coord| {
            let ids = MicrocontrollerIds {
                ok: ctx.block_types().resolve_name(&name_ok).unwrap(),
                broken: ctx.block_types().resolve_name(&name_broken).unwrap(),
            };
            microcontroller_interaction(&ctx, coord, ids)
        }));
        block.serialize_extended_data_handler = Some(Box::new(move |_ctx, data| {
            let downcast = match data.downcast_ref::<MicrocontrollerExtendedData>() {
                None => return Ok(None),
                Some(x) => x,
            };
            let proto: SerializedMicrocontrollerConfig = (&downcast.microcontroller_config).into();
            Ok(Some(Message::encode_to_vec(&proto)))
        }));
        block.deserialize_extended_data_handler = Some(Box::new(|_ctx, data| {
            let proto = SerializedMicrocontrollerConfig::decode(data)?;
            let program = proto.program.clone();
            let data = MicrocontrollerExtendedData {
                lock: 0,
                microcontroller_config: proto.try_into()?,
                core_state: Arc::new(tokio::sync::Mutex::new(CoreState {
                    program,
                    ..Default::default()
                })),
            };
            Ok(Some(Box::new(data)))
        }));
    };

    // Note no connectivity or circuit handlers; we want a custom block that keeps extended data so
    // players don't lose their programs when their microcontrollers overheat
    let broken_block = builder.add_block(
        BlockBuilder::new(BROKEN_NAME)
            .set_axis_aligned_boxes_appearance(make_chip_shape(broken_box_properties))
            .set_allow_light_propagation(true)
            .set_display_name("Basic microcontroller (broken)")
            .add_item_group(HIDDEN_FROM_CREATIVE)
            .set_inventory_texture(UNBROKEN_TEXTURE)
            .set_dropped_item(UNBROKEN_NAME.0, 1)
            .add_modifier(Box::new(block_modifier.clone())),
    )?;

    let box_properties = AaBoxProperties::new(
        get_side_tex(true),
        get_side_tex(true),
        UNBROKEN_TEXTURE,
        super::BOTTOM_TEX,
        get_side_tex(true),
        get_side_tex(true),
        crate::blocks::TextureCropping::AutoCrop,
        RotationMode::None,
    );
    let microcontroller_block = builder.add_block(
        BlockBuilder::new(UNBROKEN_NAME)
            .set_axis_aligned_boxes_appearance(make_chip_shape(box_properties))
            .set_allow_light_propagation(true)
            .set_display_name("Basic microcontroller")
            .set_inventory_texture(UNBROKEN_TEXTURE)
            .register_circuit_callbacks()
            .add_modifier(Box::new(block_modifier)),
    )?;
    let properties = CircuitBlockProperties {
        connectivity: CONN_PORT.to_vec(),
    };
    builder.define_circuit_callbacks(
        microcontroller_block.id,
        Box::new(MicrocontrollerCircuitCallbacks {
            ids: MicrocontrollerIds {
                ok: microcontroller_block.id,
                broken: broken_block.id,
            },
        }),
        properties,
    )?;
    Ok(())
}

fn max_set_bit(n: u8) -> u8 {
    match n.leading_zeros() {
        8 => 0,
        x @ 0..=7 => {
            let shift = 7u8.checked_sub(x as u8).unwrap();
            1 << shift
        }
        _ => unreachable!(),
    }
}

fn microcontroller_interaction(
    ctx: &HandlerContext,
    coord: BlockCoordinate,
    ids: MicrocontrollerIds,
) -> Result<Option<Popup>> {
    let data = ctx.game_map().mutate_block_atomically(coord, |id, ext| {
        if !ids.is_microcontroller(*id) {
            return Ok(None);
        }
        let extended_data = match ext.as_ref() {
            Some(x) => x,
            None => return Ok(Some(Default::default())),
        };
        match extended_data
            .custom_data
            .as_ref()
            .and_then(|x| x.downcast_ref::<MicrocontrollerExtendedData>())
        {
            Some(x) => Ok(Some(x.microcontroller_config.clone())),
            None => return Ok(Some(Default::default())),
        }
    })?;
    let data = match data {
        Some(x) => x,
        None => {
            ctx.initiator()
                .send_chat_message(ChatMessage::new_server_message(
                    "Not a microcontroller block",
                ))?;
            return Ok(None);
        }
    };
    let popup = ctx
        .new_popup()
        .title("Microcontroller")
        .text_field("program", "Program:", data.program, true, true)
        .text_field(
            "falling_mask",
            "Falling edge interrupt mask:",
            data.falling_interrupt_mask.to_string(),
            true,
            false,
        )
        .text_field(
            "rising_mask",
            "Rising edge interrupt mask:",
            data.falling_interrupt_mask.to_string(),
            true,
            false,
        )
        .text_field(
            "error_msg",
            "Last error:",
            data.last_error.as_deref().unwrap_or("-"),
            false,
            false,
        )
        .button("upload", "Upload", true, true)
        .set_button_callback(move |resp| {
            match resp.user_action {
                PopupAction::PopupClosed => {
                    return Ok(());
                }
                PopupAction::ButtonClicked(btn) => {
                    if btn != "upload" {
                        return Ok(());
                    }
                }
            }
            let program = resp
                .textfield_values
                .get("program")
                .context("Missing program field")?
                .clone();
            let falling_mask = match parse_u8(
                resp.textfield_values
                    .get("falling_mask")
                    .context("Missing falling mask field")?,
            ) {
                Ok(x) => x,
                Err(e) => {
                    return break_microcontroller(
                        &resp.ctx,
                        coord,
                        ids,
                        "Invalid falling interrupt mask".to_string(),
                    )
                }
            };
            let rising_mask = match parse_u8(
                resp.textfield_values
                    .get("rising_mask")
                    .context("Missing rising mask field")?,
            ) {
                Ok(x) => x,
                Err(e) => {
                    return break_microcontroller(
                        &resp.ctx,
                        coord,
                        ids,
                        "Invalid rising interrupt mask".to_string(),
                    )
                }
            };

            program_microcontroller(&resp.ctx, coord, ids, program, falling_mask, rising_mask)
        });

    Ok(Some(popup))
}

fn parse_u8(s: &str) -> std::result::Result<u8, std::num::ParseIntError> {
    if let Some(s) = s.strip_prefix("0x") {
        u8::from_str_radix(s, 16)
    } else if let Some(s) = s.strip_prefix("0o") {
        u8::from_str_radix(s, 8)
    } else if let Some(s) = s.strip_prefix("0b") {
        u8::from_str_radix(s, 2)
    } else {
        u8::from_str_radix(s, 10)
    }
}
