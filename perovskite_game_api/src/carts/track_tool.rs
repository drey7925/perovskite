use std::ops::ControlFlow;

use crate::carts::tracks::build_block;
use crate::{
    blocks::variants::rotate_nesw_azimuth_to_variant,
    game_builder::{GameBuilder, StaticTextureName},
    include_texture_bytes,
};
use anyhow::{bail, Context, Result};
use itertools::Itertools;
use perovskite_core::{
    chat::{ChatMessage, SERVER_ERROR_COLOR, SERVER_WARNING_COLOR},
    constants::{block_groups::TRIVIALLY_REPLACEABLE, items::default_item_interaction_rules},
    coordinates::BlockCoordinate,
    protocol::items as items_proto,
};
use perovskite_server::game_state::{
    client_ui::{Popup, UiElementContainer},
    event::HandlerContext,
    items::{Item, ItemStack},
    player::Player,
};

use super::{
    tracks::{eval_rotation, TRACK_TEMPLATES},
    CartsGameBuilderExtension,
};

pub(crate) fn register_track_tool(
    game_builder: &mut GameBuilder,
    config: &CartsGameBuilderExtension,
) -> Result<()> {
    const TRACK_TOOL_TEXTURE: StaticTextureName = StaticTextureName("carts:track_tool");
    include_texture_bytes!(game_builder, TRACK_TOOL_TEXTURE, "textures/track_tool.png")?;

    game_builder
        .inner
        .blocks_mut()
        .register_fast_block_group(TRIVIALLY_REPLACEABLE);

    let config_clone = config.clone();
    let item = Item {
        proto: items_proto::ItemDef {
            short_name: "carts:track_tool".to_string(),
            display_name: "Track placement tool".to_string(),
            inventory_texture: Some(TRACK_TOOL_TEXTURE.into()),
            groups: vec![],
            block_apperance: "".to_string(),
            interaction_rules: default_item_interaction_rules(),
            quantity_type: None,
        },
        dig_handler: None,
        place_handler: Some(Box::new(move |ctx, place_coord, anchor_coord, stack| {
            track_tool_interaction(ctx, place_coord, anchor_coord, stack, &config_clone)
        })),
        tap_handler: None,
    };
    game_builder.inner.items_mut().register_item(item)?;

    Ok(())
}

fn track_tool_interaction(
    ctx: &HandlerContext,
    place_coord: BlockCoordinate,
    anchor_coord: BlockCoordinate,
    stack: &ItemStack,
    config: &CartsGameBuilderExtension,
) -> Result<Option<ItemStack>> {
    let work = |p: &Player| -> Result<()> {
        let anchor_block = ctx.game_map().get_block(anchor_coord)?;
        let working_block = if config.is_any_rail_block(anchor_block) {
            anchor_coord
        } else {
            if Some(place_coord) != anchor_coord.try_delta(0, 1, 0) {
                p.send_chat_message(
                    ChatMessage::new_server_message(
                        "Track tool only works on rails or on top of other blocks",
                    )
                    .with_color(SERVER_ERROR_COLOR),
                )?;
                return Ok(());
            }
            place_coord
        };

        let face_dir = rotate_nesw_azimuth_to_variant(p.last_position().face_direction.0);

        p.show_popup_blocking(track_build_popup(ctx, config, working_block, face_dir)?)?;

        Ok(())
    };

    match ctx.initiator() {
        perovskite_server::game_state::event::EventInitiator::Player(p) => {
            work(p.player)?;
        }
        perovskite_server::game_state::event::EventInitiator::WeakPlayerRef(weak) => {
            match weak.try_to_run(|p| work(p)) {
                None => {}
                Some(Ok(_)) => {}
                Some(Err(e)) => return Err(e),
            }
        }
        _ => {}
    };
    Ok(Some(stack.clone()))
}

fn track_build_popup(
    ctx: &HandlerContext,
    config: &CartsGameBuilderExtension,
    initial_coord: BlockCoordinate,
    face_dir_as_variant: u16,
) -> Result<Popup> {
    let mut popup = ctx.new_popup().title("Track tool");
    for (key, chunk) in &TRACK_TEMPLATES.iter().chunk_by(|x| &x.category) {
        popup = popup.side_by_side_layout(key, |mut group| {
            // todo: disable buttons based on inventory, and also disable buttons for switches if there is no switch
            // actuator underneath
            for template in chunk {
                if template.bifurcate {
                    group = group.button(
                        &(template.id.to_string() + "_l"),
                        &("<-".to_string() + &template.name),
                        true,
                        false,
                    );
                    group = group.button(
                        &(template.id.to_string() + "_r"),
                        &(template.name.to_string() + "->"),
                        true,
                        false,
                    );
                } else {
                    group = group.button(
                        // this is a hack: right-handed parses as unflipped, so we just
                        // encode the right-handed template ID here.
                        &(template.id.to_string() + "_r"),
                        &template.name,
                        true,
                        false,
                    );
                }
            }
            Ok(group)
        })?;
    }
    let cloned_config = config.clone();
    popup = popup.set_button_callback(move |resp| match resp.user_action {
        perovskite_server::game_state::client_ui::PopupAction::PopupClosed => {}
        perovskite_server::game_state::client_ui::PopupAction::ButtonClicked(btn) => {
            if let Err(e) = build_track(
                &resp.ctx,
                &cloned_config,
                initial_coord,
                face_dir_as_variant,
                &btn,
            ) {
                resp.ctx
                    .initiator()
                    .send_chat_message(
                        ChatMessage::new_server_message(e.to_string())
                            .with_color(SERVER_ERROR_COLOR),
                    )
                    .unwrap();
            }
        }
    });
    Ok(popup)
}

fn build_track(
    ctx: &HandlerContext,
    config: &CartsGameBuilderExtension,
    initial_coord: BlockCoordinate,
    face_dir_as_variant: u16,
    template_id: &str,
) -> Result<()> {
    let (template, flip) = if let Some(prefix) = template_id.strip_suffix("_l") {
        (
            TRACK_TEMPLATES
                .iter()
                .find(|x| x.id == prefix)
                .with_context(|| format!("No track template with ID {}", prefix))?,
            true,
        )
    } else if let Some(prefix) = template_id.strip_suffix("_r") {
        (
            TRACK_TEMPLATES
                .iter()
                .find(|x| x.id == prefix)
                .with_context(|| format!("No track template with ID {}", prefix))?,
            false,
        )
    } else {
        bail!("Can't parse template ID {}", template_id);
    };

    let trivially_replaceable = ctx
        .block_types()
        .fast_block_group(TRIVIALLY_REPLACEABLE)
        .context("No fast block group TRIVIALLY_REPLACEABLE")?;

    // TODO: Check inventory
    for tile in template.entries.iter() {
        let (cx, cz) = eval_rotation(tile.offset_x, tile.offset_z, flip, face_dir_as_variant);
        let coord = initial_coord
            .try_delta(cx, tile.offset_y, cz)
            .with_context(|| {
                format!(
                    "Out of bounds for {:?} + ({:?}, {:?})",
                    initial_coord, cx, cz
                )
            })?;
        let block = build_block(config, tile.tile_id, face_dir_as_variant, flip)
            .with_context(|| format!("Invalid tile ID: {:?}", tile.tile_id))
            .unwrap();
        let outcome = ctx.game_map().mutate_block_atomically(coord, |b, _| {
            if config.is_any_rail_block(*b) || trivially_replaceable.contains(*b) {
                *b = block;
                Ok(ControlFlow::Continue(()))
            } else {
                Ok(ControlFlow::Break("Not either a rail or blank"))
            }
        })?;

        if let ControlFlow::Break(message) = outcome {
            ctx.initiator()
                .send_chat_message(
                    ChatMessage::new_server_message(format!("Could not place track: {}", message))
                        .with_color(SERVER_WARNING_COLOR),
                )
                .unwrap();
            break;
        }
    }

    Ok(())
}
