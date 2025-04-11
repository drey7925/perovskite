use anyhow::Result;
use perovskite_core::protocol::items::item_stack::QuantityType;
use perovskite_core::{
    block_id::special_block_defs::AIR_ID,
    constants::{
        block_groups::{DEFAULT_SOLID, INSTANT_DIG},
        item_groups::TOOL_WEAR,
    },
    protocol::{
        items::{self as items_proto, interaction_rule::DigBehavior, Empty, InteractionRule},
        render::TextureReference,
    },
};
use perovskite_server::game_state::items::{Item, ItemInteractionResult};

use crate::default_game::basic_blocks::ores::{DIAMOND_PIECE, GOLD_INGOT, IRON_INGOT};
use crate::default_game::foliage::STICK_ITEM;
use crate::default_game::recipes::RecipeSlot;
use crate::default_game::{item_groups, DefaultGameBuilder};
use crate::{game_builder::StaticTextureName, include_texture_bytes};

use super::block_groups::{BRITTLE, FIBROUS, GRANULAR};

/// Registers a new pickaxe based on the given materials
/// **This API is subject to change.**
pub(crate) fn register_pickaxe(
    game_builder: &mut super::GameBuilder,
    texture: StaticTextureName,
    name: impl Into<String>,
    display_name: impl Into<String>,
    durability: u32,
    base_dig_time: f64,
    sort_key_component: &str,
    craft_component: Option<RecipeSlot>,
) -> Result<()> {
    let name = name.into();
    register_tool(
        game_builder,
        texture,
        name,
        display_name,
        durability,
        "default:tools:pickaxes:".to_string() + sort_key_component,
        base_dig_time,
        BRITTLE,
        craft_component.map(|craft_component| {
            [
                craft_component.clone(),
                craft_component.clone(),
                craft_component,
                RecipeSlot::Empty,
                RecipeSlot::Exact(STICK_ITEM.0.to_string()),
                RecipeSlot::Empty,
                RecipeSlot::Empty,
                RecipeSlot::Exact(STICK_ITEM.0.to_string()),
                RecipeSlot::Empty,
            ]
        }),
    )
}

pub(crate) fn register_tool(
    game_builder: &mut super::GameBuilder,
    texture: StaticTextureName,
    name: impl Into<String>,
    display_name: impl Into<String>,
    durability: u32,
    sort_key: String,
    base_dig_time: f64,
    strong_block_group: &str,
    craft_recipe: Option<[RecipeSlot; 9]>,
) -> Result<()> {
    let name = name.into();
    // TODO: consider implementing this using an ItemBuilder
    let item = Item::default_with_proto(items_proto::ItemDef {
        short_name: name.clone(),
        display_name: display_name.into(),
        inventory_texture: Some(TextureReference {
            texture_name: texture.0.to_string(),
            crop: None,
        }),
        groups: vec![TOOL_WEAR.to_string()],
        interaction_rules: vec![
            InteractionRule {
                block_group: vec![strong_block_group.to_string()],
                tool_wear: 1,
                dig_behavior: Some(DigBehavior::ScaledTime(base_dig_time)),
            },
            InteractionRule {
                block_group: vec![INSTANT_DIG.to_string()],
                dig_behavior: Some(DigBehavior::InstantDigOneshot(Empty {})),
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
        sort_key,
    });
    game_builder.inner.items_mut().register_item(item)?;

    if let Some(recipe) = craft_recipe {
        game_builder.register_crafting_recipe(
            recipe,
            name,
            durability,
            Some(QuantityType::Wear(durability)),
            false,
        );
    }
    Ok(())
}

fn register_superuser_pickaxe(
    game_builder: &mut super::GameBuilder,
    texture: StaticTextureName,
    name: impl Into<String>,
    display_name: impl Into<String>,
    durability: u32,
) -> Result<()> {
    // TODO: consider implementing this using an ItemBuilder
    let item = Item {
        dig_handler: Some(Box::new(move |ctx, coord, tool| {
            let (old_block, _) = ctx.game_map().set_block(coord, AIR_ID, None)?;
            let (block, variant) = ctx.block_types().get_block(&old_block)?;
            tracing::info!("superuser pickaxe dug {}:{:x}", block.short_name(), variant);
            Ok(ItemInteractionResult {
                updated_tool: Some(tool.clone()),
                obtained_items: vec![],
            })
        })),
        ..Item::default_with_proto(items_proto::ItemDef {
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
                    dig_behavior: Some(DigBehavior::InstantDigOneshot(Empty {})),
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
            sort_key: "default:tools:pickaxes:superuser".to_string(),
        })
    };
    game_builder.inner.items_mut().register_item(item)
}

/// Registers a new shovel based on the given materials
/// **This API is subject to change.**
pub(crate) fn register_shovel(
    game_builder: &mut super::GameBuilder,
    texture: StaticTextureName,
    name: impl Into<String>,
    display_name: impl Into<String>,
    durability: u32,
    base_dig_time: f64,
    sort_key_component: &str,
    craft_component: Option<RecipeSlot>,
) -> Result<()> {
    let name = name.into();
    register_tool(
        game_builder,
        texture,
        name,
        display_name,
        durability,
        "default:tools:shovels:".to_string() + sort_key_component,
        base_dig_time,
        GRANULAR,
        craft_component.map(|craft_component| {
            [
                RecipeSlot::Empty,
                craft_component,
                RecipeSlot::Empty,
                RecipeSlot::Empty,
                RecipeSlot::Exact(STICK_ITEM.0.to_string()),
                RecipeSlot::Empty,
                RecipeSlot::Empty,
                RecipeSlot::Exact(STICK_ITEM.0.to_string()),
                RecipeSlot::Empty,
            ]
        }),
    )
}

/// Registers a new axe based on the given materials
/// **This API is subject to change.**
pub(crate) fn register_axe(
    game_builder: &mut super::GameBuilder,
    texture: StaticTextureName,
    name: impl Into<String>,
    display_name: impl Into<String>,
    durability: u32,
    base_dig_time: f64,
    sort_key_component: &str,
    craft_component: Option<RecipeSlot>,
) -> Result<()> {
    let name = name.into();
    register_tool(
        game_builder,
        texture,
        name,
        display_name,
        durability,
        "default:tools:axes:".to_string() + sort_key_component,
        base_dig_time,
        FIBROUS,
        craft_component.map(|craft_component| {
            [
                craft_component.clone(),
                craft_component.clone(),
                RecipeSlot::Empty,
                craft_component,
                RecipeSlot::Exact(STICK_ITEM.0.to_string()),
                RecipeSlot::Empty,
                RecipeSlot::Empty,
                RecipeSlot::Exact(STICK_ITEM.0.to_string()),
                RecipeSlot::Empty,
            ]
        }),
    )
}

pub(crate) fn register_default_tools(game_builder: &mut super::GameBuilder) -> Result<()> {
    let admin_pick_texture = StaticTextureName("default:admin_pickaxe");
    include_texture_bytes!(
        game_builder,
        admin_pick_texture,
        "textures/admin_pickaxe.png"
    )?;

    let wood_pick_texture = StaticTextureName("default:wood_pickaxe");
    include_texture_bytes!(game_builder, wood_pick_texture, "textures/wood_pickaxe.png")?;
    let iron_pick_texture = StaticTextureName("default:iron_pickaxe");
    include_texture_bytes!(game_builder, iron_pick_texture, "textures/iron_pickaxe.png")?;
    let gold_pick_texture = StaticTextureName("default:gold_pickaxe");
    include_texture_bytes!(game_builder, gold_pick_texture, "textures/gold_pickaxe.png")?;
    let diamond_pick_texture = StaticTextureName("default:diamond_pickaxe");
    include_texture_bytes!(
        game_builder,
        diamond_pick_texture,
        "textures/diamond_pickaxe.png"
    )?;

    register_pickaxe(
        game_builder,
        wood_pick_texture,
        "default:wood_pickaxe",
        "Wood Pickaxe",
        256,
        2.0,
        "wood",
        Some(RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string())),
    )?;
    register_pickaxe(
        game_builder,
        iron_pick_texture,
        "default:iron_pickaxe",
        "Iron Pickaxe",
        256,
        1.0,
        "iron",
        Some(RecipeSlot::Exact(IRON_INGOT.0.to_string())),
    )?;
    register_pickaxe(
        game_builder,
        gold_pick_texture,
        "default:gold_pickaxe",
        "Gold Pickaxe",
        256,
        0.6,
        "gold",
        Some(RecipeSlot::Exact(GOLD_INGOT.0.to_string())),
    )?;
    register_pickaxe(
        game_builder,
        diamond_pick_texture,
        "default:diamond_pickaxe",
        "Diamond Pickaxe",
        256,
        0.3,
        "diamond",
        Some(RecipeSlot::Exact(DIAMOND_PIECE.0.to_string())),
    )?;
    register_superuser_pickaxe(
        game_builder,
        admin_pick_texture,
        "default:superuser_pickaxe",
        "Superuser Pickaxe",
        256,
    )?;
    Ok(())
}
