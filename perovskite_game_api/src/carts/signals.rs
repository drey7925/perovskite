use perovskite_core::{
    block_id::BlockId, chat::ChatMessage, coordinates::BlockCoordinate,
    protocol::render::DynamicCrop,
};
use perovskite_server::game_state::{
    client_ui::{Popup, PopupAction, PopupResponse, UiElementContainer},
    event::HandlerContext,
};
use rand::Rng;

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
const VARIANT_PERMISSIVE_SIGNAL: u16 = 4;
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
/// Set when the signal has routing rules in its extended data (not yet implemented)
const VARIANT_EXTENDED_ROUTING: u16 = 256;

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
                        VARIANT_PERMISSIVE_SIGNAL as u32,
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
                        VARIANT_RESTRICTIVE_EXTERNAL as u32,
                    ),
            )
            .set_allow_light_propagation(true)
            .set_light_emission(8)
            .add_modifier(Box::new(|bt| {
                bt.interact_key_handler = Some(Box::new(spawn_popup));
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
    let variant = ctx.game_map().get_block(coord)?.variant();
    Ok(Some(
        ctx.new_popup()
            .title("Mrs. Yellow")
            .label("Update signal:")
            .text_field("rotation", "Rotation", (variant & 3).to_string(), true)
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
                (variant & VARIANT_PERMISSIVE_SIGNAL) != 0,
                true,
            )
            .checkbox("left", "Left", (variant & VARIANT_LEFT) != 0, true)
            .checkbox("right", "Right", (variant & VARIANT_RIGHT) != 0, true)
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
                    variant |= VARIANT_PERMISSIVE_SIGNAL;
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
                    .mutate_block_atomically(coord, |b, _ext| {
                        *b = b.with_variant(variant)?;
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

pub(crate) enum AutomaticSignalOutcome {
    InvalidSignal,
    Acquired,
    Contended,
}

/// Attempts to acquire an automatic signal.
/// This will transition it from the restrictive to the permissive state.
pub(crate) fn automatic_signal_acquire(
    block_coord: BlockCoordinate,
    id: &mut BlockId,
    expected_id_with_rotation: BlockId,
) -> AutomaticSignalOutcome {
    if !id.equals_ignore_variant(expected_id_with_rotation) {
        return AutomaticSignalOutcome::InvalidSignal;
    }

    let mut variant = id.variant();
    if variant & 3 != expected_id_with_rotation.variant() & 3 {
        return AutomaticSignalOutcome::InvalidSignal;
    }

    if variant & (VARIANT_RESTRICTIVE_TRAFFIC | VARIANT_RESTRICTIVE_EXTERNAL) != 0 {
        // The signal is already restricted because of traffic or external signal (not implemented yet)
        return AutomaticSignalOutcome::Contended;
    }
    if variant & VARIANT_PERMISSIVE_SIGNAL != 0 {
        // The signal is already cleared, but we're trying to clear it. Therefore we're not the ones that cleared it
        return AutomaticSignalOutcome::Contended;
    }

    variant |= VARIANT_PERMISSIVE_SIGNAL;
    variant &= !VARIANT_RESTRICTIVE;
    *id = id.with_variant(variant).unwrap();
    AutomaticSignalOutcome::Acquired
}

/// Updates an automatic signal as a cart passes it and enters the following block.
///
/// This transitions it from permissive to restrictive | restrictive_traffic
pub(crate) fn automatic_signal_enter_block(
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

    if variant & VARIANT_PERMISSIVE_SIGNAL == 0 {
        // This signal is not actually clear
        tracing::warn!(
            "Attempting to clear a signal that is already cleared at {:?}. Variant is {:x}",
            block_coord,
            variant
        );
    }

    variant |= VARIANT_RESTRICTIVE;
    variant |= VARIANT_RESTRICTIVE_TRAFFIC;
    variant &= !VARIANT_PERMISSIVE_SIGNAL;
    *id = id.with_variant(variant).unwrap();
}

/// Releases an automatic signal as a cart leaves the block
pub(crate) fn automatic_signal_release(
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

    if variant & VARIANT_RESTRICTIVE_TRAFFIC == 0 {
        // This signal is not actually clear
        tracing::warn!(
            "Attempting to release a signal that is already cleared at {:?}. Variant is {:x}",
            block_coord,
            variant
        )
    }
    variant &= !VARIANT_RESTRICTIVE_TRAFFIC;
    variant |= VARIANT_RESTRICTIVE;
    *id = id.with_variant(variant).unwrap();
}
