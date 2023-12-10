use anyhow::Result;
use perovskite_core::{
    constants::{block_groups::{DEFAULT_SOLID, INSTANT_DIG}, item_groups::TOOL_WEAR},
    protocol::{
        items::{self as items_proto, interaction_rule::DigBehavior, InteractionRule, Empty},
        render::TextureReference,
    },
};
use perovskite_server::game_state::items::{Item, ItemInteractionResult};

use crate::{game_builder::TextureName, include_texture_bytes};

use super::block_groups::BRITTLE;

/// Registers a new pickaxe based on the given materials
/// **This API is subject to change.**
pub(crate) fn register_pickaxe(
    game_builder: &mut super::GameBuilder,
    texture: TextureName,
    name: impl Into<String>,
    display_name: impl Into<String>,
    durability: u32,
) -> Result<()> {
    // TODO: consider implementing this using an ItemBuilder
    let item = Item {
        proto: items_proto::ItemDef {
            short_name: name.into(),
            display_name: display_name.into(),
            inventory_texture: Some(TextureReference {
                texture_name: texture.0.to_string(),
                crop: None,
            }),
            groups: vec![TOOL_WEAR.to_string()],
            interaction_rules: vec![
                InteractionRule {
                    block_group: vec![BRITTLE.to_string()],
                    tool_wear: 1,
                    dig_behavior: Some(DigBehavior::ScaledTime(0.25)),
                },
                InteractionRule {
                    block_group: vec![INSTANT_DIG.to_string()],
                    dig_behavior: Some(DigBehavior::InstantDigOneshot(
                        Empty {},
                    )),
                    tool_wear: 0,
                },
                InteractionRule {
                    block_group: vec![DEFAULT_SOLID.to_string()],
                    tool_wear: 1,
                    dig_behavior: Some(DigBehavior::ScaledTime(2.0)),
                },
            ],
            quantity_type: Some(items_proto::item_def::QuantityType::Wear(durability)),
            block_apperance: "".to_string(),
        },
        dig_handler: None,
        tap_handler: None,
        place_handler: None,
    };
    game_builder.inner.items_mut().register_item(item)
}

fn register_superuser_pickaxe(
    game_builder: &mut super::GameBuilder,
    texture: TextureName,
    name: impl Into<String>,
    display_name: impl Into<String>,
    durability: u32,
) -> Result<()> {
    // TODO: consider implementing this using an ItemBuilder
    
    let air = game_builder.air_block;
    let item = Item {
        proto: items_proto::ItemDef {
            short_name: name.into(),
            display_name: display_name.into(),
            inventory_texture: Some(TextureReference {
                texture_name: texture.0.to_string(),
                crop: None,
            }),
            groups: vec![TOOL_WEAR.to_string()],
            interaction_rules: vec![
                InteractionRule {
                    block_group: vec![BRITTLE.to_string()],
                    tool_wear: 1,
                    dig_behavior: Some(DigBehavior::ScaledTime(0.25)),
                },
                InteractionRule {
                    block_group: vec![INSTANT_DIG.to_string()],
                    dig_behavior: Some(DigBehavior::InstantDigOneshot(
                        Empty {},
                    )),
                    tool_wear: 0,
                },
                InteractionRule {
                    block_group: vec![DEFAULT_SOLID.to_string()],
                    tool_wear: 1,
                    dig_behavior: Some(DigBehavior::ScaledTime(0.25)),
                },
            ],
            quantity_type: Some(items_proto::item_def::QuantityType::Wear(durability)),
            block_apperance: "".to_string(),
        },
        dig_handler: Some(Box::new(move |ctx, coord, tool| {
            let (old_block, _) = ctx.game_map().set_block(coord, air, None)?;
            let (block, variant) = ctx.block_types().get_block(&old_block)?;
            tracing::info!("superuser pickaxe dug {}:{:x}", block.short_name(), variant);
            Ok(ItemInteractionResult { updated_tool: Some(tool.clone()), obtained_items: vec![] })
        })),
        tap_handler: None,
        place_handler: None,
    };
    game_builder.inner.items_mut().register_item(item)
}

/// Registers a new pickaxe based on the given materials
/// **This API is subject to change.**
pub(crate) fn register_shovel() -> Result<()> {
    todo!()
}

pub(crate) fn register_default_tools(game_builder: &mut super::GameBuilder) -> Result<()> {
    let test_pick_texture = TextureName("textures/test_pickaxe.png");
    include_texture_bytes!(
        game_builder,
        test_pick_texture,
        "textures/test_pickaxe.png"
    )?;
    register_pickaxe(
        game_builder,
        test_pick_texture,
        "default:test_pickaxe",
        "Test Pickaxe",
        256,
    )?;
    register_superuser_pickaxe(
        game_builder,
        test_pick_texture,
        "default:superuser_pickaxe",
        "Superuser Pickaxe",
        256,
    )?;
    Ok(())
}
