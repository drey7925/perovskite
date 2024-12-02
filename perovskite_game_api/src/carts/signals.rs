//! # Signalling methodology
//!
//! tl;dr a highly overengineered mutex
//!
//! First, one major priority in Perovskite is efficiency. For this reason, we avoid any sort of
//!  1) collision detection with carts
//!  2) scanning all possible carts anywhere in a track segment to figure out whether the segment
//!     is occupied
//!  3) indefinite prescanning
//!
//! At the same time, we have one other consideration: while real-world signals are signalled by professionals,
//! thoroughly tested, etc, we need to be able to handle various error scenarios and avoid manual programming
//! or wiring of interlockings.
//!
//! Also, since this is inherently a chunk-based game, we need to be able to handle the case where
//! a chunk is not loaded, or contains unexpected state (e.g. missing tracks). Unlike real-world rail systems,
//! we cannot inherently assume that the track, or next station, even physically exists.
//!
//! As a result, the signals behave more like mutexes than like track circuits. Each cart is responsible for
//! acquiring any signals that it wants to traverse, and then releasing them once it has cleared the region
//! that they protect.
//!
//! ## Cart state
//!
//! As carts compute a motion plan, they will inevitably acquire and release signals throughout the process.
//! Because the motion plan is computed ahead of time, these signals need to actually be flipped at the time
//! when the cart physically passes the relevant point.
//!
//! Each movement in the motion plan contains a signal to be acquired and/or a signal to be released. Likewise,
//! a cart also has a pending signal, that may be released many motion plan segments later (when the next signal
//! is acquired)
//!
//! ## Automatic signals
//!
//! Automatic signals provide the simplest functionality - they keep carts spaced apart, and avoid having
//! a cart crash into the back of another cart. Track covered by automatic signals is inherently one-directional.
//!
//! During the track scan loop, if a cart encounters an automatic signal, it will attempt to acquire it.
//! Typically, the signal is in `VARIANT_RESTRICTIVE`, meaning that it has not yet permitted a cart to pass.
//! However, in this state, it is *eligible* to be acquired. Track scanning continues, and the cart will
//! compute a motion plan that goes past the signal. As it passes the signal, it will unset `VARIANT_PERMISSIVE`
//! and set `VARIANT_RESTRICTIVE_TRAFFIC | VARIANT_RESTRICTIVE` (two red lights). Finally, once the cart
//! reaches the *next* signal of any kind (or later on, when it's de-spawned for any reason), it will reset
//! the signal back to `VARIANT_RESTRICTIVE`, the idle state.
//!
//! If the acquisition is successful, it will set `VARIANT_PERMISSIVE` on the signal, to indicate that *that*
//! cart has permission to pass.
//!
//! If a cart encounters a signal that has `VARIANT_PERMISSIVE`, something is wrong (a cart was spawned
//! improperly or encountered stale state from a shutdown). If a cart encounters a signal that's
//! tagged `VARIANT_RESTRICTIVE_TRAFFIC`, it will be unable to pass - the signal is in use because a cart
//! just passed it, entered the protected block, and hasn't yet entered the next block.
//!
//! In either case, the cart will just wait until the signal is in a more favorable state.
//!
//! ## Interlocking signals
//!
//! This is essentially a deadlock avoidance algorithm.
//!
//! Interlockings allow a variety of complex movements, including splitting, merging, or even stopping
//! (which is described separately under the section on starting signals). Tracks can be bidirectional.
//!
//! The principle is simple: in order to avoid blocking/deadlocking, a cart needs to be able to find a
//! complete path through the interlocking to either a physical end of the track, an automatic signal, or
//! a starting signal that tells the cart to stop.
//!
//! This is handled in [crate::carts::interlocking]. In short, we use a transaction-like structure, where
//! the track scanner tries to get through the interlocking, following any left/right guidance and setting
//! all signals/switches along the way. As the track scanner encounters a signal/switch, it'll set the
//! `VARIANT_PRELOCKED` flag on the signal and save it to the transaction buffer. If a complete path is found,
//! then the transaction is committed, and all signals+switches flip to the correct states (permissive
//! with left/right indicators as needed for signals, straight/diverging states for switches)
//!
//! The route plan is returned to the cart's main coroutine, which motion plans through the interlocking
//! and releases signals and switches as the cart passes them (signals, as before,
//! are released when the next signal is reached)
//!
//! If a route cannot be found, the transaction is rolled back instead, and the cart tries again after
//! a random backoff.
//!
//! It's worth noting that because a switch also acts like a mutex, any path with at least one switch is
//! fully protected against collisions, because the switches are set before a cart enters, released after
//! the cart leaves them, and no other cart can pass through the same switches in any way until they're
//! released. Switches set for "straight through" and "diverging" are both considered an acquired state,
//! there's a separate idle switch state.
//!
//! ## Starting signals
//!
//! Starting signals are a special case of interlocking signal. While normal interlocking signals are
//! used to set up a complete route through an interlocking to either a track end or to an automatic
//! signal, starting signals allow a cart to selectively stop within the interlocking (e.g. at a platform,
//! or a hopper for loading items once cart-with-chest + hopper is implemented).
//!
//! There are a few particular observations about starting signals:
//! - They need to have a *user-controlled* proceed vs stop state, so that users can actually start and
//!   stop the cart.
//! - When set to proceed, they act like normal interlocking signals.
//!     - The route plan should pass right through the signal, taking a cart from one end of the interlocking
//!       to the other.
//!     - In this case, the normal acquisition of switches and interlocking signals suffices to prevent
//!       collision with other carts.
//! - When set to stop, they almost split the track into two parts, one before and one after.
//!     - A cart approaching from the front that encounters a stop should treat it as end-of-track,
//!       (*not* as an unacquirable signal) and set up a route plan that approaches the signal and
//!       then stops.
//!     - The cart only sets up a route plan that gets it up to the starting signal, and the switches
//!       and interlocking signals it acquired along the way will eventually be cleared.
//!     - As a result, the starting signal is *itself* responsible for ensuring that a conflicting cart
//!       cannot pass through the signal from the back.
//!     - On the other hand, if there is no cart stopped in front of the starting signal (or preparing
//!       to approach and stop), then the cart approaching from the back should pass through and complete
//!       a path through the interlocking (unless it encounters a different starting signal's front while
//!       it's set to stop)
//!
//! This safety is ensured with an additional flag bit, `VARIANT_STARTING_HELD`. When this bit is set,
//! it is unsafe to pass the starting signal because the section track in *front* of the signal is contended.
//! A cart approaching the front and stopping must successfully set this bit and acquire the signal in order
//! to approach it. A cart approaching from the back must successfully set this bit and acquire the signal
//! in order to proceed past it.
//!
//! If a cart is held at the signal while it is stopped, it will continue waiting for the signal to be
//! released by a player or their circuits automations. Once it's back to proceed, the cart will acquire
//! it normally as if it were an interlocking signal, setting the `VARIANT_PERMISSIVE` flag when acquiring,
//! setting `VARIANT_RESTRICTIVE | VARIANT_RESTRICTIVE_TRAFFIC` when passing, and finally restoring it to
//! an idle state when it reaches the next signal.

use perovskite_core::{block_id::BlockId, chat::ChatMessage, coordinates::BlockCoordinate};
use perovskite_server::game_state::{
    blocks::{CustomData, ExtDataHandling, ExtendedData, InlineContext},
    client_ui::{Popup, PopupAction, PopupResponse, UiElementContainer},
    event::HandlerContext,
};
use prost::{DecodeError, Message};
use std::num::ParseIntError;

use crate::circuits::events::CircuitHandlerContext;
use crate::circuits::{
    BlockConnectivity, BusMessage, CircuitBlockBuilder, CircuitBlockCallbacks,
    CircuitBlockProperties, CircuitGameBuilder,
};
use crate::{
    blocks::{AaBoxProperties, AxisAlignedBoxesAppearanceBuilder, BlockBuilder, BuiltBlock},
    game_builder::{GameBuilder, StaticBlockName, StaticTextureName},
    include_texture_bytes,
};
use anyhow::{bail, ensure, Context, Result};
use perovskite_server::game_state::client_ui::TextFieldBuilder;

pub(crate) const SIGNAL_BLOCK: StaticBlockName = StaticBlockName("carts:signal");
pub(crate) const INTERLOCKING_SIGNAL_BLOCK: StaticBlockName =
    StaticBlockName("carts:enhanced_signal");
const SIGNAL_BLOCK_TEX_OFF: StaticTextureName = StaticTextureName("carts:signal_block_off");
const SIGNAL_BLOCK_TEX_OFF_ENHANCED: StaticTextureName =
    StaticTextureName("carts:signal_block_off_enhanced");
const SIGNAL_BLOCK_TEX_ON: StaticTextureName = StaticTextureName("carts:signal_block_on");
const SIGNAL_SIDE_TOP_TEX: StaticTextureName = StaticTextureName("carts:signal_side_top");

// Note that the two low bits of the signal variant are used for the direction it's facing
/// Set when the signal permits traffic
const VARIANT_PERMISSIVE: u16 = 4;
/// Set when the signal cannot clear because there is traffic in the block
const VARIANT_RESTRICTIVE_TRAFFIC: u16 = 8;
/// Set when the signal is not clear for any reason (either conflicting traffic, no cart
/// requested it, external circuit signal is on)
const VARIANT_RESTRICTIVE: u16 = 16;
/// Set when an external circuit signal is on, meaning that this signal will not clear
/// for a player-requested reason
const VARIANT_RESTRICTIVE_EXTERNAL: u16 = 32;
/// Set when the signal indicates a turnout to the right
const VARIANT_RIGHT: u16 = 64;
/// Set when the signal indicates a turnout to the left
const VARIANT_LEFT: u16 = 128;
/// A cart is trying to acquire this signal in an interlocking. It hasn't yet managed
/// to complete, but it wants to reserve the route for now.
const VARIANT_PRELOCKED: u16 = 256;
/// A starting signal is permitting a cart to approach it; reverse moves through it are
/// forbidden.
const VARIANT_STARTING_HELD: u16 = 512;

pub(crate) fn register_signal_blocks(
    game_builder: &mut GameBuilder,
) -> Result<(BlockId, BlockId, BlockId)> {
    include_texture_bytes!(
        game_builder,
        SIGNAL_BLOCK_TEX_OFF,
        "textures/signal_off.png"
    )?;
    include_texture_bytes!(
        game_builder,
        SIGNAL_BLOCK_TEX_OFF_ENHANCED,
        "textures/signal_off_blue.png"
    )?;
    include_texture_bytes!(game_builder, SIGNAL_BLOCK_TEX_ON, "textures/signal_on.png")?;
    include_texture_bytes!(
        game_builder,
        SIGNAL_SIDE_TOP_TEX,
        "textures/signal_side_top.png"
    )?;

    let automatic_signal = register_single_signal(
        game_builder,
        SIGNAL_BLOCK,
        SIGNAL_BLOCK_TEX_OFF,
        "Automatic Signal",
        Box::new(AutomaticSignalCircuitCallbacks),
    )?;
    let interlocking_signal = register_single_signal(
        game_builder,
        INTERLOCKING_SIGNAL_BLOCK,
        SIGNAL_BLOCK_TEX_OFF_ENHANCED,
        "Interlocking Signal",
        Box::new(InterlockingSignalCircuitCallbacks),
    )?;

    let starting_signal = register_starting_signal(game_builder)?;

    let automatic_signal_id = automatic_signal.id;
    let interlocking_signal_id = interlocking_signal.id;
    let starting_signal_id = starting_signal.id;

    game_builder
        .inner
        .blocks_mut()
        .register_cold_load_postprocessor(Box::new(move |data| {
            for block in data {
                if block.equals_ignore_variant(automatic_signal_id)
                    || block.equals_ignore_variant(interlocking_signal_id)
                    || block.equals_ignore_variant(starting_signal_id)
                {
                    // Reset the signal back to the idle state, holding the same rotation
                    *block =
                        block.with_variant_unchecked(block.variant() & 0x3 | VARIANT_RESTRICTIVE)
                }
            }
        }));

    Ok((
        automatic_signal.id,
        interlocking_signal.id,
        starting_signal.id,
    ))
}

struct AutomaticSignalCircuitCallbacks;
impl CircuitBlockCallbacks for AutomaticSignalCircuitCallbacks {
    // All default impls
}
struct InterlockingSignalCircuitCallbacks;
impl CircuitBlockCallbacks for InterlockingSignalCircuitCallbacks {
    // All default impls for now, with one placeholder
    fn on_bus_message(
        &self,
        ctx: &CircuitHandlerContext<'_>,
        coordinate: BlockCoordinate,
        from: BlockCoordinate,
        message: &BusMessage,
    ) -> Result<()> {
        if from == coordinate {
            return Ok(());
        }

        let cart_id = match message.data.get("cart_id") {
            None => return Ok(()),
            Some(x) => match x.parse::<u32>() {
                Ok(i) => i,
                Err(_) => return Ok(()),
            },
        };
        let signal_nickname = match message.data.get("cart_id") {
            None => return Ok(()),
            Some(x) => x,
        };
        let decision = match message.data.get("cart_id") {
            None => return Ok(()),
            Some(x) => match x.to_ascii_lowercase().as_str() {
                "straight" => InterlockingSignalRoute::Straight,
                "left" => InterlockingSignalRoute::DivergingLeft,
                "right" => InterlockingSignalRoute::DivergingRight,
                // Later, add "hold" when we add automation for starting signals
                _ => return Ok(()),
            },
        };
        ctx.game_map()
            .mutate_block_atomically(coordinate, |_block, ext| {
                let ext_inner = ext.get_or_insert_with(|| ExtendedData::default());

                match ext_inner.custom_data.as_mut() {
                    Some(custom_data) => match custom_data.downcast_mut::<SignalConfig>() {
                        Some(config) => {
                            if config.signal_nickname.as_str() == signal_nickname.as_str() {
                                config.pending_manual_route = Some(PendingManualRoute {
                                    startup_counter: ctx.startup_counter(),
                                    cart_id,
                                    decision: decision.into(),
                                })
                            }
                        }
                        _ => {
                            tracing::warn!("expected SignalConfig, got a different type");
                        }
                    },
                    None => {}
                }
                Ok(())
            })
    }
}

fn register_starting_signal(game_builder: &mut GameBuilder) -> Result<BuiltBlock> {
    const FRONT_TEXTURE: StaticTextureName = StaticTextureName("carts:starting_signal_front");
    const BACK_TEXTURE: StaticTextureName = StaticTextureName("carts:starting_signal_back");
    const BACK_ON_TEXTURE: StaticTextureName = StaticTextureName("carts:starting_signal_back_on");
    const BLOCK_NAME: StaticBlockName = StaticBlockName("carts:starting_signal");
    include_texture_bytes!(
        game_builder,
        FRONT_TEXTURE,
        "textures/signal_off_orange.png"
    )?;
    include_texture_bytes!(
        game_builder,
        BACK_TEXTURE,
        "textures/signal_orange_back.png"
    )?;
    include_texture_bytes!(
        game_builder,
        BACK_ON_TEXTURE,
        "textures/signal_orange_back_on.png"
    )?;
    let signal_off_box = AaBoxProperties::new(
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_SIDE_TOP_TEX,
        FRONT_TEXTURE,
        BACK_TEXTURE,
        crate::blocks::TextureCropping::AutoCrop,
        crate::blocks::RotationMode::RotateHorizontally,
    );
    let signal_on_box = AaBoxProperties::new(
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_BLOCK_TEX_ON,
        BACK_ON_TEXTURE,
        crate::blocks::TextureCropping::AutoCrop,
        crate::blocks::RotationMode::RotateHorizontally,
    );
    let block = game_builder.add_block(
        BlockBuilder::new(BLOCK_NAME)
            .set_axis_aligned_boxes_appearance(
                AxisAlignedBoxesAppearanceBuilder::new()
                    .add_box(
                        signal_off_box.clone(),
                        /* x= */ (-0.5, 0.5),
                        /* y= */ (-0.5, 0.5),
                        /* z= */ (-0.125, 0.125),
                    )
                    .add_box_with_variant_mask(
                        signal_on_box.clone(),
                        /* x= */ (-3.0 / 32.0, 3.0 / 32.0),
                        /* y= */ (-11.0 / 32.0, -5.0 / 32.0),
                        /* z= */ (-0.140, -0.125),
                        VARIANT_PERMISSIVE as u32,
                    )
                    .add_box_with_variant_mask(
                        signal_on_box.clone(),
                        /* x= */ (-11.0 / 32.0, -5.0 / 32.0),
                        /* y= */ (-11.0 / 32.0, -5.0 / 32.0),
                        /* z= */ (-0.140, -0.125),
                        VARIANT_LEFT as u32,
                    )
                    .add_box_with_variant_mask(
                        signal_on_box.clone(),
                        /* x= */ (5.0 / 32.0, 11.0 / 32.0),
                        /* y= */ (-11.0 / 32.0, -5.0 / 32.0),
                        /* z= */ (-0.140, -0.125),
                        VARIANT_RIGHT as u32,
                    )
                    .add_box_with_variant_mask(
                        signal_on_box.clone(),
                        /* x= */ (-3.0 / 32.0, 3.0 / 32.0),
                        /* y= */ (-3.0 / 32.0, 3.0 / 32.0),
                        /* z= */ (-0.140, -0.125),
                        VARIANT_RESTRICTIVE_TRAFFIC as u32,
                    )
                    .add_box_with_variant_mask(
                        signal_on_box.clone(),
                        /* x= */ (-3.0 / 32.0, 3.0 / 32.0),
                        /* y= */ (5.0 / 32.0, 11.0 / 32.0),
                        /* z= */ (-0.140, -0.125),
                        VARIANT_RESTRICTIVE as u32,
                    )
                    .add_box_with_variant_mask(
                        signal_on_box.clone(),
                        /* x= */ (5.0 / 32.0, 11.0 / 32.0),
                        /* y= */ (5.0 / 32.0, 11.0 / 32.0),
                        /* z= */ (-0.140, -0.125),
                        (VARIANT_RESTRICTIVE_EXTERNAL | VARIANT_PRELOCKED) as u32,
                    )
                    // lamp on the back of the signal indicating that a cart is allowed to approach it,
                    // so a reverse move past it would be forbidden
                    .add_box_with_variant_mask(
                        signal_on_box.clone(),
                        // This lamp is a long bar
                        /* x= */
                        (-11.0 / 32.0, 11.0 / 32.0),
                        /* y= */ (-3.0 / 32.0, 3.0 / 32.0),
                        /* z= */ (0.125, 0.140),
                        VARIANT_STARTING_HELD as u32,
                    )
                    .add_box_with_variant_mask(
                        signal_on_box.clone(),
                        /* x= */ (-11.0 / 32.0, -5.0 / 32.0),
                        /* y= */ (5.0 / 32.0, 11.0 / 32.0),
                        /* z= */ (0.124, 0.140),
                        (VARIANT_RESTRICTIVE_EXTERNAL | VARIANT_PRELOCKED) as u32,
                    ),
            )
            .set_allow_light_propagation(true)
            .set_light_emission(8)
            .add_modifier(Box::new(|bt| {
                bt.extended_data_handling = ExtDataHandling::ServerSide;
                bt.interact_key_handler = Some(Box::new(spawn_popup));
                bt.deserialize_extended_data_handler = Some(Box::new(signal_config_deserialize));
                bt.serialize_extended_data_handler = Some(Box::new(signal_config_serialize));
            }))
            .add_item_modifier(Box::new(|it| {
                let old_place_handler = it.place_handler.take().unwrap();
                it.place_handler = Some(Box::new(move |ctx, coord, anchor, stack| {
                    let result = old_place_handler(ctx, coord, anchor, stack)?;
                    ctx.game_map().mutate_block_atomically(coord, |b, _ext| {
                        *b = b.with_variant_unchecked(b.variant() | VARIANT_RESTRICTIVE);
                        Ok(())
                    })?;
                    Ok(result)
                }))
            }))
            .set_display_name("Starting Signal"),
    )?;
    Ok(block)
}

pub(super) const SIGNAL_BLOCK_CONNECTIVITY: [BlockConnectivity; 2] = [
    BlockConnectivity::rotated_nesw_with_variant(0, 0, 1, 1),
    BlockConnectivity::rotated_nesw_with_variant(0, 1, 1, 2),
];

fn register_single_signal(
    game_builder: &mut GameBuilder,
    block_name: StaticBlockName,
    face_texture: StaticTextureName,
    display_name: &str,
    circuit_callbacks: Box<dyn CircuitBlockCallbacks>,
) -> Result<BuiltBlock> {
    let signal_off_box = AaBoxProperties::new(
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_SIDE_TOP_TEX,
        face_texture,
        SIGNAL_SIDE_TOP_TEX,
        crate::blocks::TextureCropping::AutoCrop,
        crate::blocks::RotationMode::RotateHorizontally,
    );
    let signal_on_box = AaBoxProperties::new(
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_BLOCK_TEX_ON,
        SIGNAL_SIDE_TOP_TEX,
        crate::blocks::TextureCropping::AutoCrop,
        crate::blocks::RotationMode::RotateHorizontally,
    );
    let block = game_builder.add_block(
        BlockBuilder::new(block_name)
            .set_axis_aligned_boxes_appearance(
                AxisAlignedBoxesAppearanceBuilder::new()
                    .add_box(
                        signal_off_box.clone(),
                        /* x= */ (-0.5, 0.5),
                        /* y= */ (-0.5, 0.5),
                        /* z= */ (0.25, 0.5),
                    )
                    .add_box_with_variant_mask(
                        signal_on_box.clone(),
                        /* x= */ (-3.0 / 32.0, 3.0 / 32.0),
                        /* y= */ (-11.0 / 32.0, -5.0 / 32.0),
                        /* z= */ (0.235, 0.25),
                        VARIANT_PERMISSIVE as u32,
                    )
                    .add_box_with_variant_mask(
                        signal_on_box.clone(),
                        /* x= */ (-11.0 / 32.0, -5.0 / 32.0),
                        /* y= */ (-11.0 / 32.0, -5.0 / 32.0),
                        /* z= */ (0.235, 0.25),
                        VARIANT_LEFT as u32,
                    )
                    .add_box_with_variant_mask(
                        signal_on_box.clone(),
                        /* x= */ (5.0 / 32.0, 11.0 / 32.0),
                        /* y= */ (-11.0 / 32.0, -5.0 / 32.0),
                        /* z= */ (0.235, 0.25),
                        VARIANT_RIGHT as u32,
                    )
                    .add_box_with_variant_mask(
                        signal_on_box.clone(),
                        /* x= */ (-3.0 / 32.0, 3.0 / 32.0),
                        /* y= */ (-3.0 / 32.0, 3.0 / 32.0),
                        /* z= */ (0.235, 0.25),
                        VARIANT_RESTRICTIVE_TRAFFIC as u32,
                    )
                    .add_box_with_variant_mask(
                        signal_on_box.clone(),
                        /* x= */ (-3.0 / 32.0, 3.0 / 32.0),
                        /* y= */ (5.0 / 32.0, 11.0 / 32.0),
                        /* z= */ (0.235, 0.25),
                        VARIANT_RESTRICTIVE as u32,
                    )
                    .add_box_with_variant_mask(
                        signal_on_box.clone(),
                        /* x= */ (5.0 / 32.0, 11.0 / 32.0),
                        /* y= */ (5.0 / 32.0, 11.0 / 32.0),
                        /* z= */ (0.235, 0.25),
                        (VARIANT_RESTRICTIVE_EXTERNAL | VARIANT_PRELOCKED) as u32,
                    ),
            )
            .set_allow_light_propagation(true)
            .set_light_emission(8)
            .add_modifier(Box::new(|bt| {
                bt.extended_data_handling = ExtDataHandling::ServerSide;
                bt.interact_key_handler = Some(Box::new(spawn_popup));
                bt.deserialize_extended_data_handler = Some(Box::new(signal_config_deserialize));
                bt.serialize_extended_data_handler = Some(Box::new(signal_config_serialize));
            }))
            .add_item_modifier(Box::new(|it| {
                let old_place_handler = it.place_handler.take().unwrap();
                it.place_handler = Some(Box::new(move |ctx, coord, anchor, stack| {
                    let result = old_place_handler(ctx, coord, anchor, stack)?;
                    ctx.game_map().mutate_block_atomically(coord, |b, _ext| {
                        *b = b.with_variant_unchecked(b.variant() | VARIANT_RESTRICTIVE);
                        Ok(())
                    })?;
                    Ok(result)
                }))
            }))
            .set_display_name(display_name)
            .register_circuit_callbacks(),
    )?;
    let circuit_properties = CircuitBlockProperties {
        connectivity: SIGNAL_BLOCK_CONNECTIVITY.to_vec(),
    };
    game_builder.define_circuit_callbacks(block.id, circuit_callbacks, circuit_properties)?;
    Ok(block)
}

fn spawn_popup(ctx: HandlerContext, coord: BlockCoordinate) -> Result<Option<Popup>> {
    let (block, ext) =
        ctx.game_map()
            .get_block_with_extended_data(coord, |ext| match ext.custom_data {
                Some(ref custom_data) => match custom_data.downcast_ref::<SignalConfig>() {
                    Some(config) => Ok(Some(config.clone())),
                    _ => {
                        tracing::warn!("expected SignalConfig, got a different type");
                        Ok(None)
                    }
                },
                None => Ok(None),
            })?;
    let variant = block.variant();
    let ext = ext.unwrap_or_default();

    let left_routes = ext.left_routes.join("\n");
    let right_routes = ext.right_routes.join("\n");
    const PATTERN_HOVER_TEXT: &str =
        "Patterns that match cart names, one per line. Supports * and ? as wildcards";
    Ok(Some(
        ctx.new_popup()
            .title("Mrs. Yellow")
            .label("Update signal:")
            .text_field(
                "rotation",
                "Rotation",
                (variant & 3).to_string(),
                true,
                false,
            )
            .checkbox(
                "restrictive",
                "Restrictive",
                (variant & VARIANT_RESTRICTIVE) != 0,
                true,
            )
            .checkbox(
                "restrictive_external",
                "Restrictive External",
                (variant & VARIANT_RESTRICTIVE_EXTERNAL) != 0,
                true,
            )
            .checkbox(
                "restrictive_traffic",
                "Restrictive Traffic",
                (variant & VARIANT_RESTRICTIVE_TRAFFIC) != 0,
                true,
            )
            .checkbox(
                "permissive",
                "Permissive",
                (variant & VARIANT_PERMISSIVE) != 0,
                true,
            )
            .checkbox(
                "starting_held",
                "Starting Held",
                (variant & VARIANT_STARTING_HELD) != 0,
                true,
            )
            .checkbox("left", "Left", (variant & VARIANT_LEFT) != 0, true)
            .checkbox("right", "Right", (variant & VARIANT_RIGHT) != 0, true)
            .text_field_from_builder(
                TextFieldBuilder::new("left_routes")
                    .label("Left routes")
                    .initial(left_routes)
                    .multiline(true)
                    .hover_text(PATTERN_HOVER_TEXT),
            )
            .text_field_from_builder(
                TextFieldBuilder::new("right_routes")
                    .label("Right routes")
                    .initial(right_routes)
                    .multiline(true)
                    .hover_text(PATTERN_HOVER_TEXT),
            )
            .text_field_from_builder(
                TextFieldBuilder::new("signal_nickname")
                    .label("Signal nickname")
                    .initial(ext.signal_nickname.as_str())
                    .multiline(false)
                    .hover_text("Used for digital messages sent over wires to/from the signal"),
            )
            .button("apply", "Apply", true, true)
            .set_button_callback(Box::new(move |response: PopupResponse<'_>| {
                match handle_popup_response(&response, coord) {
                    Ok(_) => {}
                    Err(e) => {
                        response
                            .ctx
                            .initiator()
                            .send_chat_message(ChatMessage::new(
                                "[ERROR]",
                                "Failed to parse popup response: ".to_string()
                                    + e.to_string().as_str(),
                            ))?;
                    }
                }
                Ok(())
            })),
    ))
}

fn handle_popup_response(response: &PopupResponse, coord: BlockCoordinate) -> Result<()> {
    match &response.user_action {
        PopupAction::ButtonClicked(x) => {
            match x.as_str() {
                "apply" => {
                    let rotation = response
                        .textfield_values
                        .get("rotation")
                        .context("missing rotation")?
                        .parse::<u16>()?;
                    let restrictive = *response
                        .checkbox_values
                        .get("restrictive")
                        .context("missing restrictive")?;
                    let restrictive_external = *response
                        .checkbox_values
                        .get("restrictive_external")
                        .context("missing restrictive_external")?;
                    let restrictive_traffic = *response
                        .checkbox_values
                        .get("restrictive_traffic")
                        .context("missing restrictive_traffic")?;
                    let permissive = *response
                        .checkbox_values
                        .get("permissive")
                        .context("missing permissive")?;
                    let left = *response
                        .checkbox_values
                        .get("left")
                        .context("missing left")?;
                    let right = *response
                        .checkbox_values
                        .get("right")
                        .context("missing right")?;
                    let starting_held = *response
                        .checkbox_values
                        .get("starting_held")
                        .context("missing starting_held")?;
                    let left_routes = response
                        .textfield_values
                        .get("left_routes")
                        .context("missing left_routes")?;
                    if left_routes.len() > 1024 {
                        response.ctx.initiator().send_chat_message(
                            ChatMessage::new_server_message("left_routes too long"),
                        )?;
                        return Ok(());
                    }
                    let right_routes = response
                        .textfield_values
                        .get("right_routes")
                        .context("missing right_routes")?;

                    if right_routes.len() > 1024 {
                        response.ctx.initiator().send_chat_message(
                            ChatMessage::new_server_message("right_routes too long"),
                        )?;
                        return Ok(());
                    }
                    let manual_routes = response
                        .textfield_values
                        .get("manual_routes")
                        .context("missing manual_routes")?;

                    if manual_routes.len() > 1024 {
                        response.ctx.initiator().send_chat_message(
                            ChatMessage::new_server_message("manual_routes too long"),
                        )?;
                        return Ok(());
                    }
                    let signal_nickname = response
                        .textfield_values
                        .get("signal_nickname")
                        .context("missing signal_nickname")?
                        .clone();
                    if signal_nickname.len() > 32 {
                        response.ctx.initiator().send_chat_message(
                            ChatMessage::new_server_message("signal_nickname too long"),
                        )?;
                        return Ok(());
                    }
                    let mut variant = rotation & 3;
                    if restrictive {
                        variant |= VARIANT_RESTRICTIVE;
                    }
                    if restrictive_external {
                        variant |= VARIANT_RESTRICTIVE_EXTERNAL;
                    }
                    if restrictive_traffic {
                        variant |= VARIANT_RESTRICTIVE_TRAFFIC;
                    }
                    if permissive {
                        variant |= VARIANT_PERMISSIVE;
                    }
                    if left {
                        variant |= VARIANT_LEFT;
                    }
                    if right {
                        variant |= VARIANT_RIGHT;
                    }
                    if starting_held {
                        variant |= VARIANT_STARTING_HELD;
                    }

                    response
                        .ctx
                        .game_map()
                        .mutate_block_atomically(coord, |b, ext| {
                            *b = b.with_variant_unchecked(variant);
                            let ext_inner = ext.get_or_insert_with(|| ExtendedData::default());
                            let _ = ext_inner.custom_data.insert(Box::new(SignalConfig {
                                left_routes: left_routes
                                    .split('\n')
                                    .map(|s| s.trim().to_string())
                                    .filter(|s| !s.is_empty())
                                    .collect(),
                                right_routes: right_routes
                                    .split('\n')
                                    .map(|s| s.trim().to_string())
                                    .filter(|s| !s.is_empty())
                                    .collect(),
                                manual_routes: manual_routes
                                    .split('\n')
                                    .map(|s| s.trim().to_string())
                                    .filter(|s| !s.is_empty())
                                    .collect(),
                                signal_nickname,
                                pending_manual_route: None,
                            }));
                            ext.set_dirty();
                            Ok(())
                        })?;
                }
                _ => {
                    return Err(anyhow::anyhow!("unknown button {}", x));
                }
            }
        }
        PopupAction::PopupClosed => {}
    }
    Ok(())
}

fn signal_config_deserialize(_ctx: InlineContext, data: &[u8]) -> Result<Option<CustomData>> {
    let signal_config = SignalConfig::decode(data)?;
    Ok(Some(Box::new(signal_config)))
}
fn signal_config_serialize(_ctx: InlineContext, state: &CustomData) -> Result<Option<Vec<u8>>> {
    let signal_config = state
        .downcast_ref::<SignalConfig>()
        .context("FurnaceState downcast failed")?;
    Ok(Some(signal_config.encode_to_vec()))
}

#[derive(Clone, Message)]
pub(crate) struct SignalConfig {
    /// The list of cart names that will diverge left
    #[prost(string, repeated, tag = "1")]
    pub(crate) left_routes: Vec<String>,
    /// The list of cart names that will diverge right
    #[prost(string, repeated, tag = "2")]
    pub(crate) right_routes: Vec<String>,
    /// The list of cart names that will hold at the signal until a circuits BusMessage is sent
    #[prost(string, repeated, tag = "4")]
    pub(crate) manual_routes: Vec<String>,
    /// A nickname for the signal, used in circuit messages
    #[prost(string, tag = "3")]
    pub(crate) signal_nickname: String,
    /// A nickname for the signal, used in circuit messages
    #[prost(message, tag = "5")]
    pub(crate) pending_manual_route: Option<PendingManualRoute>,
}

#[derive(Clone, Message)]
pub(crate) struct PendingManualRoute {
    /// The startup counter value for which this route is valid
    #[prost(uint64, tag = "1")]
    pub(crate) startup_counter: u64,
    /// The cart ID for which this route is valid
    #[prost(uint32, tag = "2")]
    pub(crate) cart_id: u32,
    #[prost(enumeration = "InterlockingSignalRoute", tag = "3")]
    /// The direction where this cart will be sent
    pub(crate) decision: i32,
}

/// The outcome of trying to acquire most signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SignalLockOutcome {
    /// The signal is invalid (either not a signal, or it is not in the right state)
    InvalidSignal,
    /// The signal was acquired
    Acquired,
    /// The signal was not acquired because another cart has already acquired it
    Contended,
}

/// The outcome of trying to acquire an approach signal *from the front*.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StartingSignalLockOutcome {
    /// The signal is invalid (either not a signal, or it is not in the right state)
    InvalidSignal,
    /// The signal was acquired
    Acquired,
    /// The signal was not acquired because another cart has already acquired it
    Contended,
    /// The signal is set to stop, but the cart is permitted to approach it then stop.
    ApproachThenStop,
}

/// Attempts to acquire an automatic signal.
/// This will transition it from the restrictive to the permissive state.
///
/// This should NOT be called on interlocking signals; those require closer coordination.
pub(crate) fn automatic_signal_acquire(
    _block_coord: BlockCoordinate,
    id: &mut BlockId,
    expected_id_with_rotation: BlockId,
) -> SignalLockOutcome {
    if !id.equals_ignore_variant(expected_id_with_rotation) {
        return SignalLockOutcome::InvalidSignal;
    }

    let mut variant = id.variant();
    if variant & 3 != expected_id_with_rotation.variant() & 3 {
        return SignalLockOutcome::InvalidSignal;
    }

    if variant & (VARIANT_RESTRICTIVE_TRAFFIC | VARIANT_RESTRICTIVE_EXTERNAL) != 0 {
        // The signal is already restricted because of traffic or external signal (not implemented yet)
        return SignalLockOutcome::Contended;
    }
    if variant & VARIANT_PERMISSIVE != 0 {
        // The signal is already cleared, but we're trying to clear it. Therefore we're not the ones that cleared it
        return SignalLockOutcome::Contended;
    }

    variant |= VARIANT_PERMISSIVE;
    variant &= !VARIANT_RESTRICTIVE;
    *id = id.with_variant_unchecked(variant);
    SignalLockOutcome::Acquired
}

/// Updates a signal as a cart passes it and enters the following block.
///
/// This transitions it from permissive to restrictive | restrictive_traffic
pub(crate) fn signal_enter_block(
    block_coord: BlockCoordinate,
    id: &mut BlockId,
    expected_id_with_rotation: BlockId,
) {
    if !id.equals_ignore_variant(expected_id_with_rotation) {
        return;
    }
    let mut variant = id.variant();

    if variant & 3 != expected_id_with_rotation.variant() & 3 {
        return;
    }

    if variant & VARIANT_PERMISSIVE == 0 {
        // This signal is not actually clear
        tracing::warn!(
            "Attempting to clear a signal that is already cleared at {:?}. Variant is {:x}",
            block_coord,
            variant
        );
    }

    variant |= VARIANT_RESTRICTIVE;
    variant |= VARIANT_RESTRICTIVE_TRAFFIC;
    variant &= !VARIANT_PERMISSIVE;
    *id = id.with_variant_unchecked(variant);
}

/// Releases a signal as a cart leaves the block, transitioning it from restrictive_traffic | restrictive to just restrictive
pub(crate) fn signal_release(
    block_coord: BlockCoordinate,
    id: &mut BlockId,
    expected_id_with_rotation: BlockId,
) {
    if !id.equals_ignore_variant(expected_id_with_rotation) {
        tracing::warn!(
            "Attempting to release unexpected signal at {:?}. Expected {:?}, got {:?}",
            block_coord,
            expected_id_with_rotation,
            id
        );
        return;
    }

    let mut variant = id.variant();
    if variant & 3 != expected_id_with_rotation.variant() & 3 {
        return;
    }

    if variant & VARIANT_RESTRICTIVE_TRAFFIC == 0 {
        // This signal is not actually clear
        tracing::warn!(
            "Attempting to release a signal that is already cleared at {:?}. Variant is {:x}",
            block_coord,
            variant
        )
    }
    variant &=
        !(VARIANT_RESTRICTIVE_TRAFFIC | VARIANT_LEFT | VARIANT_RIGHT | VARIANT_STARTING_HELD);
    variant |= VARIANT_RESTRICTIVE;
    *id = id.with_variant_unchecked(variant);
}

/// Attempts to acquire an interlocking signal.
/// This will mark it as preacquired, but not transition it to permissive or traffic restricted.
pub(crate) fn interlocking_signal_preacquire(
    _block_coord: BlockCoordinate,
    id: &mut BlockId,
) -> SignalLockOutcome {
    let mut variant = id.variant();

    if variant & (VARIANT_PRELOCKED) != 0 {
        // Another cart is already trying to acquire this signal and beat us to it.
        return SignalLockOutcome::Contended;
    }

    if variant & (VARIANT_RESTRICTIVE_TRAFFIC | VARIANT_RESTRICTIVE_EXTERNAL) != 0 {
        // A cart is still moving through the block protected by this signal
        return SignalLockOutcome::Contended;
    }
    if variant & VARIANT_PERMISSIVE != 0 {
        // The signal is already cleared, so another cart already has a path through the interlocking.
        return SignalLockOutcome::Contended;
    }

    variant |= VARIANT_PRELOCKED;
    *id = id.with_variant_unchecked(variant);
    SignalLockOutcome::Acquired
}

/// Attempts to acquire a starting signal in an interlocking signal when approaching from the front.
/// This will mark it as preacquired, but not transition it to permissive or traffic restricted.
///
/// Unlike interlocking signals, the signal will still be acquired if it's in a RESTRICTIVE_EXTERNAL state;
/// the interlocking resolver will set up a route up to just before this signal then stop.
pub(crate) fn starting_signal_preacquire_front(
    _block_coord: BlockCoordinate,
    id: &mut BlockId,
) -> SignalLockOutcome {
    let mut variant = id.variant();

    if variant & (VARIANT_PRELOCKED) != 0 {
        // Another cart is already trying to acquire this signal and beat us to it.
        return SignalLockOutcome::Contended;
    }
    if variant & (VARIANT_STARTING_HELD) != 0 {
        // Another cart is already allowed to approach it and occupy the section just in front of the signal.
        return SignalLockOutcome::Contended;
    }

    if variant & (VARIANT_RESTRICTIVE_TRAFFIC) != 0 {
        // A cart is still moving through the block protected by this signal
        return SignalLockOutcome::Contended;
    }
    if variant & VARIANT_PERMISSIVE != 0 {
        // The signal is already cleared, so another cart already has a path through the interlocking.
        return SignalLockOutcome::Contended;
    }

    variant |= VARIANT_PRELOCKED;
    *id = id.with_variant_unchecked(variant);
    SignalLockOutcome::Acquired
}

pub(crate) fn starting_signal_depart_forward(
    _block_coord: BlockCoordinate,
    id: &mut BlockId,
) -> SignalLockOutcome {
    let mut variant = id.variant();

    if variant & (VARIANT_PRELOCKED) != 0 {
        return SignalLockOutcome::Contended;
    }
    if variant & (VARIANT_RESTRICTIVE_TRAFFIC | VARIANT_RESTRICTIVE_EXTERNAL) != 0 {
        // A cart is still moving through the block protected by this signal, or we don't want to leave yet
        return SignalLockOutcome::Contended;
    }
    variant |= VARIANT_PRELOCKED;

    *id = id.with_variant_unchecked(variant);
    SignalLockOutcome::Acquired
}

pub(crate) fn starting_signal_acquire_back(
    _block_coord: BlockCoordinate,
    id: &mut BlockId,
) -> SignalLockOutcome {
    let mut variant = id.variant();

    if variant & (VARIANT_PRELOCKED) != 0 {
        // Another cart is already trying to acquire this signal and beat us to it.
        return SignalLockOutcome::Contended;
    }
    if variant & (VARIANT_STARTING_HELD) != 0 {
        // Another cart is already allowed to approach it and occupy the section just in front of the signal.
        return SignalLockOutcome::Contended;
    }
    if variant & (VARIANT_RESTRICTIVE_TRAFFIC) != 0 {
        // A cart is still moving through the block protected by this signal
        // Or, a cart was allowed to approach it from the back by another caller to this function, and still hasn't passed it yet.
        return SignalLockOutcome::Contended;
    }
    // We set restrictive_traffic because we haven't actually gotten the cart past it yet, but we still want to
    // lock out traffic passing through it
    variant |= VARIANT_RESTRICTIVE_TRAFFIC;
    *id = id.with_variant_unchecked(variant);
    SignalLockOutcome::Acquired
}

/// Updates a signal as a cart passes it and enters the following block.
///
/// This transitions it from permissive to restrictive | restrictive_traffic
pub(crate) fn starting_signal_reverse_enter_block(
    block_coord: BlockCoordinate,
    id: &mut BlockId,
    expected_id_with_rotation: BlockId,
) {
    if !id.equals_ignore_variant(expected_id_with_rotation) {
        return;
    }
    let mut variant = id.variant();

    if variant & 3 != expected_id_with_rotation.variant() & 3 {
        return;
    }

    const EXPECTED_STATE: u16 =
        VARIANT_STARTING_HELD | VARIANT_RESTRICTIVE | VARIANT_RESTRICTIVE_TRAFFIC;
    const IMPORTANT_BITS: u16 = VARIANT_STARTING_HELD
        | VARIANT_RESTRICTIVE
        | VARIANT_RESTRICTIVE_TRAFFIC
        | VARIANT_PERMISSIVE;

    if variant & IMPORTANT_BITS == EXPECTED_STATE {
        // This signal is not actually clear
        tracing::warn!(
            "Attempting to clear a signal that is already cleared at {:?}. Variant is {:x}",
            block_coord,
            variant
        );
    }

    variant |= VARIANT_STARTING_HELD;
    variant &= !VARIANT_RESTRICTIVE_TRAFFIC;
    *id = id.with_variant_unchecked(variant);
}

/// Attempt to acquire a starting signal.
/// This will transition it from the restrictive to the permissive state.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SignalParseOutcome {
    /// The signal indicates that the next switch is to be set straight
    Straight,
    /// The signal indicates that the next switch (capable of being set diverging left, i.e. switch tile with flip_x = true) is to be set diverging
    DivergingLeft,
    /// The signal indicates that the next switch (capable of being set diverging right, i.e. switch tile with flip_x = false) is to be set diverging
    DivergingRight,
    /// Not yet supported, may be removed in favor of microcontroller resolutions
    #[deprecated]
    Fork,
    /// The signal does snot permit travel.
    Deny,
    /// This is not a signal (it might be a speedpost or similar)
    NoIndication,
    /// This is an automatic signal, so we're done with the interlocking
    AutomaticSignal,
    /// Starting signal, set to stop, but the cart is permitted to approach it from the front and then stop.
    ///
    /// This is not applicable when the cart is approaching from the back.
    StartingSignalApproachThenStop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub(crate) enum InterlockingSignalRoute {
    Straight = 0,
    DivergingLeft = 1,
    DivergingRight = 2,
    /// Not yet supported, may be removed in favor of microcontroller resolutions
    #[deprecated]
    Fork = 3,
    StartingSignalApproachThenStop = 4,
    /// Turns into a deny
    ManuallySignalledNoDecision = 5,
}
impl InterlockingSignalRoute {
    pub(crate) fn adjust_variant(&self, variant: u16) -> Option<u16> {
        let rotation = variant & 3;
        match self {
            InterlockingSignalRoute::Straight => Some(rotation | VARIANT_PERMISSIVE),
            InterlockingSignalRoute::DivergingLeft => {
                Some(rotation | VARIANT_PERMISSIVE | VARIANT_LEFT)
            }
            InterlockingSignalRoute::DivergingRight => {
                Some(rotation | VARIANT_PERMISSIVE | VARIANT_RIGHT)
            }
            InterlockingSignalRoute::Fork => None,
            InterlockingSignalRoute::StartingSignalApproachThenStop => {
                // If we ended up in this branch, it's because VARIANT_RESTRICTIVE_EXTERNAL was set.
                // Make sure that we keep it set.
                Some(
                    rotation
                        | VARIANT_RESTRICTIVE
                        | VARIANT_RESTRICTIVE_EXTERNAL
                        | VARIANT_STARTING_HELD,
                )
            }
            InterlockingSignalRoute::ManuallySignalledNoDecision => None,
        }
    }

    pub(crate) fn to_parse_outcome(&self) -> SignalParseOutcome {
        match self {
            InterlockingSignalRoute::Straight => SignalParseOutcome::Straight,
            InterlockingSignalRoute::DivergingLeft => SignalParseOutcome::DivergingLeft,
            InterlockingSignalRoute::DivergingRight => SignalParseOutcome::DivergingRight,
            InterlockingSignalRoute::Fork => SignalParseOutcome::Fork,
            InterlockingSignalRoute::StartingSignalApproachThenStop => {
                SignalParseOutcome::StartingSignalApproachThenStop
            }
            InterlockingSignalRoute::ManuallySignalledNoDecision => SignalParseOutcome::Deny,
        }
    }
}

pub(crate) fn query_interlocking_signal(
    ext: Option<&ExtendedData>,
    cart_route_name: &str,
    cart_id: (u64, u32),
) -> Result<InterlockingSignalRoute> {
    if ext.is_none() {
        return Ok(InterlockingSignalRoute::Straight);
    }
    let ext = ext.unwrap();
    let signal_config = match ext.custom_data.as_ref() {
        Some(config) => match config.downcast_ref::<SignalConfig>() {
            Some(config) => Some(config),
            _ => {
                tracing::warn!("expected SignalConfig, got a different type");
                None
            }
        },
        _ => None,
    };

    if let Some(config) = signal_config {
        if config
            .manual_routes
            .iter()
            .any(|x| wildmatch::WildMatch::new(x).matches(cart_route_name))
        {
            return query_manual_interlocking_signal(config, cart_id);
        }

        if config
            .left_routes
            .iter()
            .any(|x| wildmatch::WildMatch::new(x).matches(cart_route_name))
        {
            return Ok(InterlockingSignalRoute::DivergingLeft);
        }
        if config
            .right_routes
            .iter()
            .any(|x| wildmatch::WildMatch::new(x).matches(cart_route_name))
        {
            return Ok(InterlockingSignalRoute::DivergingRight);
        }
    }
    Ok(InterlockingSignalRoute::Straight)
}

fn query_manual_interlocking_signal(
    signal_config: &SignalConfig,
    cart_id: (u64, u32),
) -> Result<InterlockingSignalRoute> {
    if let Some(pending) = signal_config.pending_manual_route.as_ref() {
        if cart_id == (pending.startup_counter, pending.cart_id) {
            match InterlockingSignalRoute::try_from(pending.decision)? {
                InterlockingSignalRoute::Fork => {
                    bail!("Invalid fork decision for manual signal");
                }
                InterlockingSignalRoute::StartingSignalApproachThenStop => {
                    bail!("StartingSignalApproachThenStop but not at a starting signal");
                }
                x => return Ok(x),
            }
        }
    }
    Ok(InterlockingSignalRoute::ManuallySignalledNoDecision)
}

pub(crate) fn query_starting_signal(
    variant: u16,
    ext: Option<&ExtendedData>,
    cart_route_name: &str,
    cart_id: (u64, u32),
) -> Result<InterlockingSignalRoute> {
    if variant & VARIANT_RESTRICTIVE_EXTERNAL != 0 {
        return Ok(InterlockingSignalRoute::StartingSignalApproachThenStop);
    }
    // TODO allow settings in the extension to also hold the cart at the signal until cleared
    query_interlocking_signal(ext, cart_route_name, cart_id)
}
