// Copyright 2023 drey7925
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::{sync::atomic::AtomicU32, time::Duration};

use super::{
    block_groups::{BRITTLE, GRANULAR},
    default_mapgen::OreDefinition,
    recipes::RecipeSlot,
    shaped_blocks::{make_slab, make_stairs},
    DefaultGameBuilder, DefaultGameBuilderExtension,
};
use crate::default_game::chest::register_chest;
use crate::default_game::signs::register_sign;
use crate::game_builder::{GRASS_FOOTSTEP_SOUND_NAME, SNOW_FOOTSTEP_SOUND_NAME};
use crate::{
    blocks::{
        AaBoxProperties, AxisAlignedBoxesAppearanceBuilder, BlockBuilder, CubeAppearanceBuilder,
        MatterType, PlantLikeAppearanceBuilder,
    },
    game_builder::{
        include_texture_bytes, GameBuilder, StaticBlockName, StaticItemName, StaticTextureName,
    },
};
use anyhow::Result;
use perovskite_core::block_id::BlockId;
use perovskite_core::constants::items::default_item_interaction_rules;
use perovskite_core::coordinates::BlockCoordinate;
use perovskite_core::protocol::items::ItemDef;
use perovskite_core::{
    constants::{
        block_groups::{self, DEFAULT_LIQUID, TOOL_REQUIRED},
        item_groups::HIDDEN_FROM_CREATIVE,
    },
    protocol::{
        self,
        blocks::{block_type_def::PhysicsInfo, FluidPhysicsInfo},
        items::{item_stack::QuantityType, InteractionRule},
    },
};
use perovskite_server::game_state::blocks::{ExtendedDataHolder, InlineContext};
use perovskite_server::game_state::game_map::{
    TimerCallback, TimerInlineCallback, TimerSettings, TimerState,
};
use perovskite_server::game_state::{
    blocks::BlockInteractionResult,
    items::{Item, ItemStack},
};

/// Dirt without grass on it.
pub const DIRT: StaticBlockName = StaticBlockName("default:dirt");
/// Dirt with grass on top.
pub const DIRT_WITH_GRASS: StaticBlockName = StaticBlockName("default:dirt_with_grass");

/// Solid grey stone.
pub const STONE: StaticBlockName = StaticBlockName("default:stone");

/// Beach sand.
pub const SAND: StaticBlockName = StaticBlockName("default:sand");
pub const SILT_DRY: StaticBlockName = StaticBlockName("default:silt_dry");
pub const SILT_DAMP: StaticBlockName = StaticBlockName("default:silt_damp");
pub const CLAY: StaticBlockName = StaticBlockName("default:clay");

/// Desert materials.
pub const DESERT_STONE: StaticBlockName = StaticBlockName("default:desert_stone");
pub const DESERT_SAND: StaticBlockName = StaticBlockName("default:desert_sand");

/// Transparent glass.
pub const GLASS: StaticBlockName = StaticBlockName("default:glass");

/// Torch
pub const TORCH: StaticBlockName = StaticBlockName("default:torch");

/// Water
/// Stability note: not stable (liquids are TBD)
pub const WATER: StaticBlockName = StaticBlockName("default:water");

/// test only, no crafting recipe, not finalized
pub const TNT: StaticBlockName = StaticBlockName("default:tnt");

/// Limestone for karst landscapes
pub const LIMESTONE: StaticBlockName = StaticBlockName("default:limestone");
pub const LIMESTONE_LIGHT: StaticBlockName = StaticBlockName("default:limestone_light");
pub const LIMESTONE_DARK: StaticBlockName = StaticBlockName("default:limestone_dark");
pub const DIRT_WITH_SNOW: StaticBlockName = StaticBlockName("default:dirt_with_snow");
/// Snow, not a full block
pub const SNOW: StaticBlockName = StaticBlockName("default:snow");
pub const SNOW_FOOTPRINT: StaticBlockName = StaticBlockName("default:snow_footprint");
/// A full, solid block of snow
pub const SNOW_BLOCK: StaticBlockName = StaticBlockName("default:snow_block");
pub const SNOW_BLOCK_FOOTPRINT: StaticBlockName = StaticBlockName("default:snow_block_footprint");
pub const SNOWBALL: StaticItemName = StaticItemName("default:snowball");

const DIRT_TEXTURE: StaticTextureName = StaticTextureName("default:dirt");
const DIRT_GRASS_SIDE_TEXTURE: StaticTextureName = StaticTextureName("default:dirt_grass_side");
const GRASS_TOP_TEXTURE: StaticTextureName = StaticTextureName("default:grass_top");
const STONE_TEXTURE: StaticTextureName = StaticTextureName("default:stone");
const SAND_TEXTURE: StaticTextureName = StaticTextureName("default:sand");
const SILT_DRY_TEXTURE: StaticTextureName = StaticTextureName("default:silt_dry");
const SILT_DAMP_TEXTURE: StaticTextureName = StaticTextureName("default:silt_damp");
const CLAY_TEXTURE: StaticTextureName = StaticTextureName("default:clay");
const DESERT_STONE_TEXTURE: StaticTextureName = StaticTextureName("default:desert_stone");
const DESERT_SAND_TEXTURE: StaticTextureName = StaticTextureName("default:desert_sand");
const GLASS_TEXTURE: StaticTextureName = StaticTextureName("default:glass");
const WATER_TEXTURE: StaticTextureName = StaticTextureName("default:water");

const TORCH_TEXTURE: StaticTextureName = StaticTextureName("default:torch");
const TNT_TEXTURE: StaticTextureName = StaticTextureName("default:tnt");
const TESTONLY_UNKNOWN_TEX: StaticTextureName = StaticTextureName("default:testonly_unknown");

const LIMESTONE_TEXTURE: StaticTextureName = StaticTextureName("default:limestone");
const LIMESTONE_LIGHT_TEXTURE: StaticTextureName = StaticTextureName("default:limestone_light");
const LIMESTONE_DARK_TEXTURE: StaticTextureName = StaticTextureName("default:limestone_dark");
const SNOW_TEXTURE: StaticTextureName = StaticTextureName("default:snow");
const SNOW_FOOTPRINT_TEXTURE: StaticTextureName = StaticTextureName("default:snow_footprint");
const DIRT_SNOW_SIDE_TEXTURE: StaticTextureName = StaticTextureName("default:dirt_snow_side");

const SNOWBALL_TEXTURE: StaticTextureName = StaticTextureName("default:snowball");

pub mod ores {
    use perovskite_core::constants::block_groups::TOOL_REQUIRED;
    use perovskite_server::game_state::items::ItemStack;
    use rand::Rng;

    use crate::{
        blocks::CubeAppearanceBuilder,
        default_game::{
            item_groups,
            recipes::{RecipeImpl, RecipeSlot},
            DefaultGameBuilderExtension,
        },
        game_builder::StaticItemName,
    };

    use super::*;

    pub const COAL_ORE: StaticBlockName = StaticBlockName("default:coal_ore");
    pub const COAL_PIECE: StaticItemName = StaticItemName("default:coal_piece");
    pub const COAL_ORE_TEXTURE: StaticTextureName = StaticTextureName("default:coal_ore");
    pub const COAL_PIECE_TEXTURE: StaticTextureName = StaticTextureName("default:coal_piece");

    pub const IRON_ORE: StaticBlockName = StaticBlockName("default:iron_ore");
    pub const IRON_PIECE: StaticItemName = StaticItemName("default:iron_piece");
    pub const IRON_INGOT: StaticItemName = StaticItemName("default:iron_ingot");
    pub const IRON_ORE_TEXTURE: StaticTextureName = StaticTextureName("default:iron_ore");
    pub const IRON_PIECE_TEXTURE: StaticTextureName = StaticTextureName("default:iron_piece");
    pub const IRON_INGOT_TEXTURE: StaticTextureName = StaticTextureName("default:iron_ingot");

    pub const GOLD_ORE: StaticBlockName = StaticBlockName("default:gold_ore");
    pub const GOLD_PIECE: StaticItemName = StaticItemName("default:gold_piece");
    pub const GOLD_INGOT: StaticItemName = StaticItemName("default:gold_ingot");
    pub const GOLD_ORE_TEXTURE: StaticTextureName = StaticTextureName("default:gold_ore");
    pub const GOLD_PIECE_TEXTURE: StaticTextureName = StaticTextureName("default:gold_piece");
    pub const GOLD_INGOT_TEXTURE: StaticTextureName = StaticTextureName("default:gold_ingot");

    pub const DIAMOND_ORE: StaticBlockName = StaticBlockName("default:diamond_ore");
    pub const DIAMOND_PIECE: StaticItemName = StaticItemName("default:diamond_piece");
    pub const DIAMOND_ORE_TEXTURE: StaticTextureName = StaticTextureName("default:diamond_ore");
    pub const DIAMOND_PIECE_TEXTURE: StaticTextureName = StaticTextureName("default:diamond_piece");

    pub(crate) fn register_ores(game_builder: &mut GameBuilder) -> Result<()> {
        // todo factor this into a function per-ore
        include_texture_bytes!(game_builder, COAL_ORE_TEXTURE, "textures/coal_ore.png")?;
        include_texture_bytes!(game_builder, COAL_PIECE_TEXTURE, "textures/coal_piece.png")?;
        game_builder.register_basic_item(
            COAL_PIECE,
            "Piece of coal",
            COAL_PIECE_TEXTURE,
            vec![],
            "default:ore_pieces:coal",
        )?;
        let coal_ore = game_builder.add_block(
            BlockBuilder::new(COAL_ORE)
                .set_cube_appearance(
                    CubeAppearanceBuilder::new().set_single_texture(COAL_ORE_TEXTURE),
                )
                .add_block_group(BRITTLE)
                .add_block_group(TOOL_REQUIRED)
                .set_dropped_item_closure(|| (COAL_PIECE, rand::thread_rng().gen_range(1..=2)))
                .set_item_sort_key("default:ore_blocks:coal"),
        )?;

        game_builder
            .builder_extension::<DefaultGameBuilderExtension>()
            .register_ore(OreDefinition {
                block: coal_ore.id,
                noise_cutoff: splines::Spline::from_vec(vec![
                    splines::Key {
                        value: 0.5,
                        t: 0.,
                        interpolation: splines::Interpolation::Linear,
                    },
                    splines::Key {
                        value: 0.6,
                        t: 100.,
                        interpolation: splines::Interpolation::Linear,
                    },
                ]),
                cave_bias_effect: 0.0,
                noise_scale: (4., 0.25, 4.),
            });
        game_builder.register_smelting_fuel(RecipeSlot::Exact(COAL_PIECE.0.to_string()), 16);

        include_texture_bytes!(game_builder, IRON_ORE_TEXTURE, "textures/iron_ore.png")?;
        include_texture_bytes!(game_builder, IRON_PIECE_TEXTURE, "textures/iron_piece.png")?;
        include_texture_bytes!(game_builder, IRON_INGOT_TEXTURE, "textures/iron_ingot.png")?;
        game_builder.register_basic_item(
            IRON_PIECE,
            "Piece of iron",
            IRON_PIECE_TEXTURE,
            vec![item_groups::RAW_ORES.into()],
            "default:ore_pieces:iron",
        )?;
        game_builder.register_basic_item(
            IRON_INGOT,
            "Iron ingot",
            IRON_INGOT_TEXTURE,
            vec![item_groups::METAL_INGOTS.into()],
            "default:ingots:iron",
        )?;
        // todo clean this up when ore APIs are more mature
        game_builder
            .builder_extension::<DefaultGameBuilderExtension>()
            .smelting_recipes
            .register_recipe(RecipeImpl {
                slots: [RecipeSlot::Exact(IRON_PIECE.0.to_string())],
                result: ItemStack {
                    proto: protocol::items::ItemStack {
                        item_name: IRON_INGOT.0.to_string(),
                        quantity: 1,
                        current_wear: 0,
                        quantity_type: Some(QuantityType::Stack(256)),
                    },
                },
                shapeless: false,
                metadata: 8,
            });

        let iron_ore = game_builder.add_block(
            BlockBuilder::new(IRON_ORE)
                .set_cube_appearance(
                    CubeAppearanceBuilder::new().set_single_texture(IRON_ORE_TEXTURE),
                )
                .add_block_group(BRITTLE)
                .add_block_group(TOOL_REQUIRED)
                .set_dropped_item_closure(|| (IRON_PIECE, rand::thread_rng().gen_range(1..=2)))
                .set_item_sort_key("default:ore_blocks:iron"),
        )?;

        game_builder
            .builder_extension::<DefaultGameBuilderExtension>()
            .register_ore(OreDefinition {
                block: iron_ore.id,
                // Use the same schedule as coal
                noise_cutoff: splines::Spline::from_vec(vec![
                    splines::Key {
                        value: 0.5,
                        t: 0.,
                        interpolation: splines::Interpolation::Linear,
                    },
                    splines::Key {
                        value: 0.6,
                        t: 100.,
                        interpolation: splines::Interpolation::Linear,
                    },
                ]),
                cave_bias_effect: 0.5,
                noise_scale: (4., 0.25, 4.),
            });

        include_texture_bytes!(
            game_builder,
            DIAMOND_ORE_TEXTURE,
            "textures/diamond_ore.png"
        )?;
        include_texture_bytes!(
            game_builder,
            DIAMOND_PIECE_TEXTURE,
            "textures/diamond_piece.png"
        )?;
        game_builder.register_basic_item(
            DIAMOND_PIECE,
            "Piece of diamond",
            DIAMOND_PIECE_TEXTURE,
            vec![item_groups::GEMS.into()],
            "default:ore_pieces:diamond",
        )?;
        let diamond_ore = game_builder.add_block(
            BlockBuilder::new(DIAMOND_ORE)
                .set_cube_appearance(
                    CubeAppearanceBuilder::new().set_single_texture(DIAMOND_ORE_TEXTURE),
                )
                .add_block_group(BRITTLE)
                .add_block_group(TOOL_REQUIRED)
                .set_dropped_item_closure(|| (DIAMOND_PIECE, rand::thread_rng().gen_range(1..=2)))
                .set_item_sort_key("default:ore_blocks:diamond"),
        )?;

        game_builder
            .builder_extension::<DefaultGameBuilderExtension>()
            .register_ore(OreDefinition {
                block: diamond_ore.id,
                noise_cutoff: splines::Spline::from_vec(vec![
                    splines::Key {
                        value: 0.9,
                        t: 0.,
                        interpolation: splines::Interpolation::Linear,
                    },
                    splines::Key {
                        value: 0.825,
                        t: 100.,
                        interpolation: splines::Interpolation::Linear,
                    },
                    splines::Key {
                        value: 0.775,
                        t: 400.,
                        interpolation: splines::Interpolation::Linear,
                    },
                ]),
                cave_bias_effect: 0.125,
                noise_scale: (4., 0.25, 4.),
            });

        include_texture_bytes!(game_builder, GOLD_ORE_TEXTURE, "textures/gold_ore.png")?;
        include_texture_bytes!(game_builder, GOLD_PIECE_TEXTURE, "textures/gold_piece.png")?;
        include_texture_bytes!(game_builder, GOLD_INGOT_TEXTURE, "textures/gold_ingot.png")?;
        game_builder.register_basic_item(
            GOLD_PIECE,
            "Piece of gold",
            GOLD_PIECE_TEXTURE,
            vec![item_groups::RAW_ORES.into()],
            "default:ore_pieces:gold",
        )?;
        game_builder.register_basic_item(
            GOLD_INGOT,
            "Gold ingot",
            GOLD_INGOT_TEXTURE,
            vec![item_groups::METAL_INGOTS.into()],
            "default:ingots:gold",
        )?;
        let gold_ore = game_builder.add_block(
            BlockBuilder::new(GOLD_ORE)
                .set_cube_appearance(
                    CubeAppearanceBuilder::new().set_single_texture(GOLD_ORE_TEXTURE),
                )
                .add_block_group(BRITTLE)
                .add_block_group(TOOL_REQUIRED)
                .set_dropped_item_closure(|| (GOLD_PIECE, rand::thread_rng().gen_range(1..=2)))
                .set_item_sort_key("default:ore_blocks:gold"),
        )?;

        game_builder
            .builder_extension::<DefaultGameBuilderExtension>()
            .register_ore(OreDefinition {
                block: gold_ore.id,
                // Use the same schedule as coal
                noise_cutoff: splines::Spline::from_vec(vec![
                    splines::Key {
                        value: 0.9,
                        t: 0.,
                        interpolation: splines::Interpolation::Linear,
                    },
                    splines::Key {
                        value: 0.815,
                        t: 100.,
                        interpolation: splines::Interpolation::Linear,
                    },
                    splines::Key {
                        value: 0.755,
                        t: 200.,
                        interpolation: splines::Interpolation::Linear,
                    },
                ]),
                cave_bias_effect: 0.125,
                noise_scale: (4., 0.25, 4.),
            });

        game_builder
            .builder_extension::<DefaultGameBuilderExtension>()
            .smelting_recipes
            .register_recipe(RecipeImpl {
                slots: [RecipeSlot::Exact(GOLD_PIECE.0.to_string())],
                result: ItemStack {
                    proto: protocol::items::ItemStack {
                        item_name: GOLD_INGOT.0.to_string(),
                        quantity: 1,
                        current_wear: 0,
                        quantity_type: Some(QuantityType::Stack(256)),
                    },
                },
                shapeless: false,
                metadata: 8,
            });

        Ok(())
    }
}

pub(crate) fn register_basic_blocks(game_builder: &mut GameBuilder) -> Result<()> {
    register_core_blocks(game_builder)?;
    register_tnt(game_builder)?;
    ores::register_ores(game_builder)?;

    game_builder
        .builder_extension::<DefaultGameBuilderExtension>()
        .crafting_recipes
        .register_recipe(super::recipes::RecipeImpl {
            slots: [
                RecipeSlot::Exact("default:stick".to_string()),
                RecipeSlot::Empty,
                RecipeSlot::Empty,
                RecipeSlot::Exact("default:coal_piece".to_string()),
                RecipeSlot::Empty,
                RecipeSlot::Empty,
                RecipeSlot::Empty,
                RecipeSlot::Empty,
                RecipeSlot::Empty,
            ],
            result: ItemStack {
                proto: protocol::items::ItemStack {
                    item_name: "default:torch".to_string(),
                    quantity: 4,
                    current_wear: 0,
                    quantity_type: Some(QuantityType::Stack(256)),
                },
            },
            shapeless: true,
            metadata: (),
        });

    Ok(())
}

fn register_tnt(builder: &mut GameBuilder) -> Result<()> {
    builder
        .inner
        .items_mut()
        .register_item(Item::default_with_proto(protocol::items::ItemDef {
            short_name: "default:tnt_actor_tool".to_string(),
            display_name: "TNT actor tool (you should not see this)".to_string(),
            inventory_texture: Some(TNT_TEXTURE.into()),
            groups: vec![HIDDEN_FROM_CREATIVE.into()],
            interaction_rules: vec![InteractionRule {
                block_group: vec![],
                tool_wear: 0,
                dig_behavior: Some(
                    protocol::items::interaction_rule::DigBehavior::InstantDigOneshot(
                        Default::default(),
                    ),
                ),
            }],
            quantity_type: None,
            block_apperance: "".to_string(),
            sort_key: "default:internal:tnt_actor_tool".to_string(),
        }))?;

    let tnt_tool_stack = Some(ItemStack {
        proto: protocol::items::ItemStack {
            item_name: "default:tnt_actor_tool".to_string(),
            quantity: 1,
            ..Default::default()
        },
    });

    include_texture_bytes!(builder, TNT_TEXTURE, "textures/tnt.png")?;
    //let air = builder.air_block;
    builder.add_block(
        BlockBuilder::new(TNT)
            .set_cube_appearance(CubeAppearanceBuilder::new().set_single_texture(TNT_TEXTURE))
            .add_modifier(Box::new(move |block_type| {
                block_type.tap_handler_full = Some(Box::new(move |ctx, coord, tool| {
                    if tool.is_some_and(|tool| tool.proto.item_name == "default:superuser_pickaxe")
                    {
                        return Ok(Default::default());
                    }

                    for i in -3..=3 {
                        for j in -3..=3 {
                            for k in -3..=3 {
                                if let Some(neighbor) = coord.try_delta(i, j, k) {
                                    // discard the block interaction result; the player isn't getting the drops
                                    ctx.game_map().dig_block(
                                        neighbor,
                                        ctx.initiator(),
                                        tnt_tool_stack.as_ref(),
                                    )?;
                                }
                            }
                        }
                    }

                    Ok(BlockInteractionResult {
                        item_stacks: vec![ItemStack {
                            proto: protocol::items::ItemStack {
                                item_name: "default:tnt".to_string(),
                                quantity: 2,
                                current_wear: 0,
                                quantity_type: Some(QuantityType::Stack(256)),
                            },
                        }],
                        tool_wear: 0,
                    })
                }))
            })),
    )?;

    Ok(())
}

static TESTONLY_COUNTER: AtomicU32 = AtomicU32::new(0);

fn register_core_blocks(game_builder: &mut GameBuilder) -> Result<()> {
    include_texture_bytes!(game_builder, DIRT_TEXTURE, "textures/dirt.png")?;
    include_texture_bytes!(
        game_builder,
        DIRT_GRASS_SIDE_TEXTURE,
        "textures/dirt_grass_side.png"
    )?;
    include_texture_bytes!(game_builder, GRASS_TOP_TEXTURE, "textures/grass_top.png")?;
    include_texture_bytes!(game_builder, STONE_TEXTURE, "textures/stone.png")?;
    include_texture_bytes!(game_builder, SAND_TEXTURE, "textures/sand.png")?;
    include_texture_bytes!(game_builder, SILT_DRY_TEXTURE, "textures/silt_dry.png")?;
    include_texture_bytes!(game_builder, SILT_DAMP_TEXTURE, "textures/silt_damp.png")?;
    include_texture_bytes!(game_builder, CLAY_TEXTURE, "textures/clay.png")?;

    include_texture_bytes!(
        game_builder,
        DESERT_STONE_TEXTURE,
        "textures/desert_stone.png"
    )?;
    include_texture_bytes!(
        game_builder,
        DESERT_SAND_TEXTURE,
        "textures/desert_sand.png"
    )?;

    include_texture_bytes!(game_builder, GLASS_TEXTURE, "textures/glass.png")?;

    include_texture_bytes!(game_builder, WATER_TEXTURE, "textures/water.png")?;

    include_texture_bytes!(game_builder, TORCH_TEXTURE, "textures/torch.png")?;

    include_texture_bytes!(
        game_builder,
        TESTONLY_UNKNOWN_TEX,
        "../media/block_unknown.png"
    )?;

    include_texture_bytes!(game_builder, LIMESTONE_TEXTURE, "textures/limestone.png")?;
    include_texture_bytes!(
        game_builder,
        LIMESTONE_DARK_TEXTURE,
        "textures/limestone_dark.png"
    )?;
    include_texture_bytes!(
        game_builder,
        LIMESTONE_LIGHT_TEXTURE,
        "textures/limestone_light.png"
    )?;
    include_texture_bytes!(game_builder, SNOW_TEXTURE, "textures/snow.png")?;
    include_texture_bytes!(
        game_builder,
        SNOW_FOOTPRINT_TEXTURE,
        "textures/snow_footprint.png"
    )?;
    include_texture_bytes!(
        game_builder,
        DIRT_SNOW_SIDE_TEXTURE,
        "textures/dirt_snow_side.png"
    )?;

    include_texture_bytes!(game_builder, SNOWBALL_TEXTURE, "textures/snowball.png")?;

    let dirt = game_builder.add_block(
        BlockBuilder::new(DIRT)
            .add_block_group(GRANULAR)
            .set_cube_single_texture(DIRT_TEXTURE)
            .set_display_name("Dirt block"),
    )?;

    let grass_footstep = game_builder
        .get_sound_id(GRASS_FOOTSTEP_SOUND_NAME)
        .expect("grass footstep sound");

    game_builder.add_block(
        BlockBuilder::new(DIRT_WITH_GRASS)
            .add_block_group(GRANULAR)
            .set_cube_appearance(CubeAppearanceBuilder::new().set_individual_textures(
                DIRT_GRASS_SIDE_TEXTURE,
                DIRT_GRASS_SIDE_TEXTURE,
                GRASS_TOP_TEXTURE,
                DIRT_TEXTURE,
                DIRT_GRASS_SIDE_TEXTURE,
                DIRT_GRASS_SIDE_TEXTURE,
            ))
            .set_dropped_item(DIRT.0, 1)
            .set_footstep_sound(Some(grass_footstep))
            // testonly
            .set_dropped_item(DIRT.0, 1),
    )?;
    let stone = game_builder.add_block(
        BlockBuilder::new(STONE)
            .add_block_group(BRITTLE)
            .add_block_group(TOOL_REQUIRED)
            .set_cube_single_texture(STONE_TEXTURE)
            .set_display_name("Stone"),
    )?;
    let limestone = game_builder.add_block(
        BlockBuilder::new(LIMESTONE)
            .add_block_group(BRITTLE)
            .add_block_group(TOOL_REQUIRED)
            .set_cube_single_texture(LIMESTONE_TEXTURE)
            .set_display_name("Limestone"),
    )?;
    let limestone_light = game_builder.add_block(
        BlockBuilder::new(LIMESTONE_LIGHT)
            .add_block_group(BRITTLE)
            .add_block_group(TOOL_REQUIRED)
            .set_cube_single_texture(LIMESTONE_LIGHT_TEXTURE)
            .set_display_name("Light limestone"),
    )?;
    let limestone_dark = game_builder.add_block(
        BlockBuilder::new(LIMESTONE_DARK)
            .add_block_group(BRITTLE)
            .add_block_group(TOOL_REQUIRED)
            .set_cube_single_texture(LIMESTONE_DARK_TEXTURE)
            .set_display_name("Dark limestone"),
    )?;

    let _sand = game_builder.add_block(
        BlockBuilder::new(SAND)
            .add_block_group(GRANULAR)
            .set_cube_single_texture(SAND_TEXTURE)
            .set_display_name("Sand")
            .set_falls_down(true),
    )?;

    let _silt_dry = game_builder.add_block(
        BlockBuilder::new(SILT_DRY)
            .add_block_group(GRANULAR)
            .set_cube_single_texture(SILT_DRY_TEXTURE)
            .set_display_name("Silt (dry)"),
    )?;
    // TODO: damp silt should dry out when far from water, on a stochastic timer
    let _silt_damp = game_builder.add_block(
        BlockBuilder::new(SILT_DAMP)
            .add_block_group(GRANULAR)
            .set_cube_single_texture(SILT_DAMP_TEXTURE)
            .set_display_name("Silt (damp)")
            .set_dropped_item(SILT_DRY.0, 1)
            .add_modifier(Box::new(|bt| {
                // TODO tune these, TODO move these into the block builder
                bt.client_info.physics_info = Some(PhysicsInfo::Fluid(FluidPhysicsInfo {
                    horizontal_speed: 0.75,
                    vertical_speed: -0.1,
                    jump_speed: 0.2,
                    sink_speed: -0.1,
                    surface_thickness: 0.2,
                    surf_horizontal_speed: 2.,
                    surf_vertical_speed: -0.1,
                    surf_jump_speed: 0.1,
                    surf_sink_speed: -0.1,
                }))
            })),
    )?;
    let _clay = game_builder.add_block(
        BlockBuilder::new(CLAY)
            .add_block_group(GRANULAR)
            .set_cube_single_texture(CLAY_TEXTURE)
            .set_display_name("Clay block"),
    )?;

    let desert_stone = game_builder.add_block(
        BlockBuilder::new(DESERT_STONE)
            .add_block_group(BRITTLE)
            .add_block_group(TOOL_REQUIRED)
            .set_cube_single_texture(DESERT_STONE_TEXTURE)
            .set_display_name("Desert stone"),
    )?;
    let _desert_sand = game_builder.add_block(
        BlockBuilder::new(DESERT_SAND)
            .add_block_group(GRANULAR)
            .set_cube_single_texture(DESERT_SAND_TEXTURE)
            .set_display_name("Desert sand")
            .set_falls_down(true),
    )?;

    let snow_footstep = game_builder
        .get_sound_id(SNOW_FOOTSTEP_SOUND_NAME)
        .expect("snow footstep sound");

    let snow_footprint = game_builder.add_block(
        BlockBuilder::new(SNOW_FOOTPRINT)
            .add_block_group(GRANULAR)
            .set_cube_appearance(
                CubeAppearanceBuilder::new()
                    .set_individual_textures(
                        SNOW_TEXTURE,
                        SNOW_TEXTURE,
                        SNOW_FOOTPRINT_TEXTURE,
                        SNOW_TEXTURE,
                        SNOW_TEXTURE,
                        SNOW_TEXTURE,
                    )
                    .set_variant_from_height(),
            )
            .set_allow_light_propagation(true)
            .add_item_group(HIDDEN_FROM_CREATIVE)
            .set_footstep_sound(Some(snow_footstep))
            .set_trivially_replaceable(true)
            .set_display_name("Snow w/ footprint")
            .set_dropped_item_closure_extended(|param| {
                (SNOWBALL, param.variant.clamp(0, 7) as u32 + 1)
            }),
    )?;
    let snow_footprint_id = snow_footprint.id;

    let snow = game_builder.add_block(
        BlockBuilder::new(SNOW)
            .add_block_group(GRANULAR)
            .set_cube_appearance(
                CubeAppearanceBuilder::new()
                    .set_single_texture(SNOW_TEXTURE)
                    .set_variant_from_height(),
            )
            .set_allow_light_propagation(true)
            .set_footstep_sound(Some(snow_footstep))
            .add_item_group(HIDDEN_FROM_CREATIVE)
            .set_trivially_replaceable(true)
            .set_display_name("Snow")
            .set_dropped_item_closure_extended(|param| {
                (SNOWBALL, param.variant.clamp(0, 7) as u32 + 1)
            })
            .add_modifier(Box::new(move |bt| {
                bt.step_on_handler_inline = Some(Box::new(move |_, block, _, _| {
                    *block = snow_footprint_id.with_variant_of(*block);
                    Ok(BlockInteractionResult::default())
                }))
            })),
    )?;

    let snow_block_footprint = game_builder.add_block(
        BlockBuilder::new(SNOW_BLOCK_FOOTPRINT)
            .add_block_group(GRANULAR)
            .set_cube_appearance(CubeAppearanceBuilder::new().set_individual_textures(
                SNOW_TEXTURE,
                SNOW_TEXTURE,
                SNOW_FOOTPRINT_TEXTURE,
                SNOW_TEXTURE,
                SNOW_TEXTURE,
                SNOW_TEXTURE,
            ))
            .set_allow_light_propagation(false)
            .add_item_group(HIDDEN_FROM_CREATIVE)
            .set_footstep_sound(Some(snow_footstep))
            .set_trivially_replaceable(false)
            .set_display_name("Snow block w/ footprint"),
    )?;
    let snow_block_footprint_id = snow_block_footprint.id;
    let snow_block = game_builder.add_block(
        BlockBuilder::new(SNOW_BLOCK)
            .add_block_group(GRANULAR)
            .set_cube_appearance(CubeAppearanceBuilder::new().set_single_texture(SNOW_TEXTURE))
            .set_allow_light_propagation(false)
            .set_footstep_sound(Some(snow_footstep))
            .set_trivially_replaceable(false)
            .set_display_name("Snow block")
            .add_modifier(Box::new(move |bt| {
                bt.step_on_handler_inline = Some(Box::new(move |_, block, _, _| {
                    *block = snow_block_footprint_id.with_variant_of(*block);
                    Ok(BlockInteractionResult::default())
                }))
            })),
    )?;
    let snow_id = snow.id;
    let snow_block_id = snow_block.id;
    game_builder.inner.items_mut().register_item(Item {
        place_on_block_handler: Some(Box::new(move |ctx, place_coord, anchor_coord, stack| {
            let incremented =
                ctx.game_map()
                    .mutate_block_atomically(anchor_coord, |block, _| {
                        if (block.equals_ignore_variant(snow_id)
                            || block.equals_ignore_variant(snow_footprint_id))
                            && block.variant() < 7
                        {
                            let new_variant = block.variant() + 1;
                            if new_variant < 7 {
                                // Placing snow covers up footprints
                                *block = snow_id.with_variant_unchecked(new_variant);
                            } else {
                                *block = snow_block_id;
                            }
                            Ok(true)
                        } else {
                            Ok(false)
                        }
                    })?;
            if incremented {
                return Ok(stack.decrement());
            }
            // We couldn't increment at the anchor coord, place at the place coord instead...
            let replaced =
                ctx.game_map()
                    .mutate_block_atomically(place_coord, move |block, _| {
                        if ctx.block_types().is_trivially_replaceable(*block) {
                            *block = snow_id.with_variant_unchecked(0);
                            Ok(true)
                        } else {
                            Ok(false)
                        }
                    })?;
            if replaced {
                Ok(stack.decrement())
            } else {
                Ok(Some(stack.clone()))
            }
        })),
        ..Item::default_with_proto(ItemDef {
            short_name: SNOWBALL.0.to_string(),
            display_name: "Snowball".to_string(),
            inventory_texture: Some(SNOWBALL_TEXTURE.into()),
            groups: vec![],
            interaction_rules: default_item_interaction_rules(),
            quantity_type: Some(protocol::items::item_def::QuantityType::Stack(256)),
            block_apperance: String::new(),
            sort_key: "default:snowball".into(),
        })
    })?;
    game_builder.add_block(
        BlockBuilder::new(DIRT_WITH_SNOW)
            .add_block_group(GRANULAR)
            .set_cube_appearance(CubeAppearanceBuilder::new().set_individual_textures(
                DIRT_SNOW_SIDE_TEXTURE,
                DIRT_SNOW_SIDE_TEXTURE,
                SNOW_TEXTURE,
                DIRT_TEXTURE,
                DIRT_SNOW_SIDE_TEXTURE,
                DIRT_SNOW_SIDE_TEXTURE,
            ))
            .set_dropped_item(DIRT.0, 1)
            .set_footstep_sound(Some(snow_footstep))
            .set_display_name("Dirt with snow"),
    )?;

    struct RemoveSnowFootprintsTimerCallback {
        snow_id: BlockId,
        snow_footprint_id: BlockId,
        snow_block_id: BlockId,
        snow_block_footprint_id: BlockId,
    }
    impl TimerInlineCallback for RemoveSnowFootprintsTimerCallback {
        fn inline_callback(
            &self,
            _: BlockCoordinate,
            _: &TimerState,
            block: &mut BlockId,
            _: &mut ExtendedDataHolder,
            _: &InlineContext,
        ) -> Result<()> {
            if block.equals_ignore_variant(self.snow_block_footprint_id) {
                *block = self.snow_block_id;
            } else if block.equals_ignore_variant(self.snow_footprint_id) {
                *block = self.snow_id.with_variant_of(*block);
            }
            Ok(())
        }
    }
    game_builder.inner.add_timer(
        "snow_rm_footprints",
        TimerSettings {
            interval: Duration::from_secs(5),
            shards: 16,
            spreading: 1.0,
            block_types: vec![snow_footprint_id, snow_block_footprint_id],
            ignore_block_type_presence_check: false,
            per_block_probability: 0.1,
            idle_chunk_after_unchanged: false,
            ..Default::default()
        },
        TimerCallback::PerBlockLocked(Box::new(RemoveSnowFootprintsTimerCallback {
            snow_id,
            snow_footprint_id,
            snow_block_id,
            snow_block_footprint_id,
        })),
    );

    let glass = game_builder.add_block(
        BlockBuilder::new(GLASS)
            .add_block_group(BRITTLE)
            .set_cube_appearance(
                CubeAppearanceBuilder::new()
                    .set_single_texture(GLASS_TEXTURE)
                    .set_needs_transparency(),
            )
            .set_allow_light_propagation(true)
            .set_display_name("Glass"),
    )?;
    let mut water_builder = BlockBuilder::new(WATER)
        .add_block_group(DEFAULT_LIQUID)
        .set_trivially_replaceable(true)
        .add_item_group("testonly_wet")
        .set_cube_appearance(
            CubeAppearanceBuilder::new()
                .set_single_texture(WATER_TEXTURE)
                .set_liquid_shape()
                .set_needs_translucency(),
        )
        .set_not_diggable()
        .set_allow_light_propagation(true)
        .set_display_name("Water block")
        .set_liquid_flow(Some(Duration::from_millis(500)))
        .set_matter_type(MatterType::Liquid);
    water_builder.client_info.physics_info = Some(PhysicsInfo::Fluid(FluidPhysicsInfo {
        horizontal_speed: 1.5,
        vertical_speed: -0.5,
        jump_speed: 1.0,
        sink_speed: -1.0,
        surface_thickness: 0.1,
        surf_horizontal_speed: 2.,
        surf_vertical_speed: -0.5,
        surf_jump_speed: 1.0,
        surf_sink_speed: -0.5,
    }));
    game_builder.add_block(water_builder)?;

    game_builder.add_block(
        BlockBuilder::new(TORCH)
            .set_plant_like_appearance(
                // todo more suitable render-mode
                PlantLikeAppearanceBuilder::new()
                    .set_wave_effect_scale(0.0)
                    .set_texture(TORCH_TEXTURE),
            )
            .add_block_group(block_groups::INSTANT_DIG)
            .set_allow_light_propagation(true)
            .set_display_name("Torch")
            .set_light_emission(8),
    )?;

    register_chest(game_builder)?;
    register_sign(game_builder)?;

    game_builder.add_block(
        BlockBuilder::new(StaticBlockName("testonly:aabb_geometry_test"))
            .set_axis_aligned_boxes_appearance(AxisAlignedBoxesAppearanceBuilder::new().add_box(
                AaBoxProperties::new_single_tex(
                    TESTONLY_UNKNOWN_TEX,
                    crate::blocks::TextureCropping::AutoCrop,
                    crate::blocks::RotationMode::None,
                ),
                (-0.3, 0.5),
                (-0.3, 0.5),
                (-0.3, 0.5),
            )),
    )?;
    game_builder.add_block(
        BlockBuilder::new(StaticBlockName("testonly:aabb_geometry_test2")).set_cube_appearance(
            CubeAppearanceBuilder::new().set_single_texture(TESTONLY_UNKNOWN_TEX),
        ),
    )?;
    game_builder.register_crafting_recipe(
        std::array::from_fn(|_| RecipeSlot::Exact(SNOWBALL.0.to_string())),
        snow_block.item_name.0.to_string(),
        1,
        Some(QuantityType::Stack(256)),
        false,
    );

    make_stairs(game_builder, &stone, true)?;
    make_stairs(game_builder, &glass, true)?;
    make_stairs(game_builder, &dirt, false)?;
    make_stairs(game_builder, &desert_stone, true)?;
    make_stairs(game_builder, &limestone, true)?;
    make_stairs(game_builder, &limestone_dark, true)?;
    make_stairs(game_builder, &limestone_light, true)?;

    make_slab(game_builder, &stone, true)?;
    make_slab(game_builder, &glass, true)?;
    make_slab(game_builder, &dirt, false)?;
    make_slab(game_builder, &desert_stone, true)?;
    make_slab(game_builder, &limestone, true)?;
    make_slab(game_builder, &limestone_dark, true)?;
    make_slab(game_builder, &limestone_light, true)?;

    register_test_audio_block(game_builder)?;

    Ok(())
}

fn register_test_audio_block(game_builder: &mut GameBuilder) -> Result<()> {
    game_builder
        .inner
        .media_mut()
        .register_from_memory("default:test_audio", include_bytes!("test_audio.wav"))?;
    let audio_id = game_builder
        .inner
        .media_mut()
        .register_file_for_sampled_audio("default:test_audio")?;
    const TEST_AUDIO_SOURCE: StaticBlockName = StaticBlockName("default:test_audio_source");
    const TEST_AUDIO_TEX: StaticTextureName = StaticTextureName("default:test_audio_source");
    include_texture_bytes!(
        game_builder,
        TEST_AUDIO_TEX,
        "textures/test_audio_source.png"
    )?;

    BlockBuilder::new(TEST_AUDIO_SOURCE)
        .set_display_name("Test audio source")
        .set_cube_single_texture(TEST_AUDIO_TEX)
        .add_modifier(Box::new(move |bt| {
            bt.client_info.sound_id = audio_id.0;
            bt.client_info.sound_volume = 0.25;
        }))
        .build_and_deploy_into(game_builder)?;

    Ok(())
}
