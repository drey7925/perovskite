use perovskite_core::{block_id::BlockId, chat::ChatMessage, coordinates::BlockCoordinate};
use perovskite_server::game_state::{
    blocks::{CustomData, ExtDataHandling, ExtendedData, InlineContext},
    client_ui::{Popup, PopupAction, PopupResponse, UiElementContainer},
    event::HandlerContext,
};
use prost::Message;

use crate::{
    blocks::{AaBoxProperties, AxisAlignedBoxesAppearanceBuilder, BlockBuilder, BuiltBlock},
    game_builder::{GameBuilder, StaticBlockName, StaticTextureName},
    include_texture_bytes,
};
use anyhow::{Context, Result};

pub(crate) const SIGNAL_BLOCK: StaticBlockName = StaticBlockName("carts:signal");
pub(crate) const ENHANCED_SIGNAL_BLOCK: StaticBlockName = StaticBlockName("carts:enhanced_signal");
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

pub(crate) fn register_signal_block(
    game_builder: &mut crate::game_builder::GameBuilder,
) -> Result<(BlockId, BlockId)> {
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
    )?;
    let interlocking_signal = register_single_signal(
        game_builder,
        ENHANCED_SIGNAL_BLOCK,
        SIGNAL_BLOCK_TEX_OFF_ENHANCED,
        "Interlocking Signal",
    )?;

    Ok((automatic_signal.id, interlocking_signal.id))
}

fn register_single_signal(
    game_builder: &mut GameBuilder,
    block_name: StaticBlockName,
    face_texture: StaticTextureName,
    display_name: &str,
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
                        *b = b.with_variant(b.variant() | VARIANT_RESTRICTIVE)?;
                        Ok(())
                    })?;
                    Ok(result)
                }))
            }))
            .set_display_name(display_name),
    )?;
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
            .checkbox("left", "Left", (variant & VARIANT_LEFT) != 0, true)
            .checkbox("right", "Right", (variant & VARIANT_RIGHT) != 0, true)
            .text_field("left_routes", "Left Routes", left_routes, true, true)
            .text_field("right_routes", "Right Routes", right_routes, true, true)
            .button("apply", "Apply", true)
            .set_button_callback(Box::new(move |response: PopupResponse<'_>| {
                match handle_popup_response(&response, coord) {
                    Ok(_) => {}
                    Err(e) => {
                        response
                            .ctx
                            .initiator()
                            .send_chat_message(ChatMessage::new(
                                "[ERROR]",
                                "Failed to parse popup response: ".to_string() + &e.to_string(),
                            ))
                            .unwrap();
                    }
                }
            })),
    ))
}

fn handle_popup_response(response: &PopupResponse, coord: BlockCoordinate) -> Result<()> {
    match &response.user_action {
        PopupAction::ButtonClicked(x) => match x.as_str() {
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
                let left_routes = response
                    .textfield_values
                    .get("left_routes")
                    .context("missing left_routes")?;
                let right_routes = response
                    .textfield_values
                    .get("right_routes")
                    .context("missing right_routes")?;
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

                response
                    .ctx
                    .game_map()
                    .mutate_block_atomically(coord, |b, ext| {
                        *b = b.with_variant(variant)?;
                        let ext_inner = ext.get_or_insert_with(|| ExtendedData::default());
                        let _ = ext_inner.custom_data.insert(Box::new(SignalConfig {
                            left_routes: left_routes
                                .split('\n')
                                .map(|s| s.trim().to_string())
                                .collect(),
                            right_routes: right_routes
                                .split('\n')
                                .map(|s| s.trim().to_string())
                                .collect(),
                        }));
                        ext.set_dirty();
                        Ok(())
                    })?;
            }
            _ => {
                return Err(anyhow::anyhow!("unknown button {}", x));
            }
        },
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

/// Extended data for a furnace. One tick is 0.25 seconds.
#[derive(Clone, Message)]
pub(crate) struct SignalConfig {
    /// The list of routes that will diverge left
    #[prost(string, repeated, tag = "1")]
    pub(crate) left_routes: Vec<String>,
    /// The list of routes that will diverge right
    #[prost(string, repeated, tag = "2")]
    pub(crate) right_routes: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SignalLockOutcome {
    InvalidSignal,
    Acquired,
    Contended,
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
    *id = id.with_variant(variant).unwrap();
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
    *id = id.with_variant(variant).unwrap();
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
    variant &= !(VARIANT_RESTRICTIVE_TRAFFIC | VARIANT_LEFT | VARIANT_RIGHT);
    variant |= VARIANT_RESTRICTIVE;
    *id = id.with_variant(variant).unwrap();
}

/// Attempts to acquire an automatic signal.
/// This will transition it from the restrictive to the permissive state.
pub(crate) fn interlocking_signal_preacquire(
    _block_coord: BlockCoordinate,
    id: &mut BlockId,
    expected_id_with_rotation: BlockId,
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
    *id = id.with_variant(variant).unwrap();
    SignalLockOutcome::Acquired
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SignalParseOutcome {
    /// The signal indicates that the next switch is to be set straight
    Straight,
    /// The signal indicates that the next switch (capable of being set diverging left, i.e. switch tile with flip_x = true) is to be set diverging
    DivergingLeft,
    /// The signal indicates that the next switch (capable of being set diverging right, i.e. switch tile with flip_x = false) is to be set diverging
    DivergingRight,
    Fork,
    Deny,
    NotASignal,
    /// This is an automatic signal, so we're done with the interlocking
    AutomaticSignal,
}
impl SignalParseOutcome {
    pub(crate) fn adjust_variant(&self, variant: u16) -> Option<u16> {
        let rotation = variant & 3;
        match self {
            SignalParseOutcome::Straight => Some(VARIANT_PERMISSIVE | rotation),
            SignalParseOutcome::DivergingLeft => Some(VARIANT_PERMISSIVE | rotation | VARIANT_LEFT),
            SignalParseOutcome::DivergingRight => {
                Some(VARIANT_PERMISSIVE | rotation | VARIANT_RIGHT)
            }
            SignalParseOutcome::Fork => None,
            SignalParseOutcome::Deny => None,
            SignalParseOutcome::NotASignal => None,
            SignalParseOutcome::AutomaticSignal => None,
        }
    }
}

pub(crate) fn query_interlocking_signal(
    ext: Option<&ExtendedData>,
    cart_route_name: &str,
) -> Result<SignalParseOutcome> {
    if ext.is_none() {
        return Ok(SignalParseOutcome::Straight);
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
        // TODO avoid expensive wildmatch construction every time
        if config
            .left_routes
            .iter()
            .any(|x| wildmatch::WildMatch::new(x).matches(cart_route_name))
        {
            return Ok(SignalParseOutcome::DivergingLeft);
        }
        if config
            .right_routes
            .iter()
            .any(|x| wildmatch::WildMatch::new(x).matches(cart_route_name))
        {
            return Ok(SignalParseOutcome::DivergingRight);
        }
    }
    Ok(SignalParseOutcome::Straight)
}
