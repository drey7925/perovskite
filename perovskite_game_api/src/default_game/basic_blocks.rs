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

use crate::{
    blocks::{
        AaBoxProperties, AxisAlignedBoxesAppearanceBuilder, BlockBuilder, CubeAppearanceBuilder,
        MatterType, PlantLikeAppearanceBuilder,
    },
    default_game::{basic_blocks::ores::IRON_INGOT, item_groups},
    game_builder::{
        include_texture_bytes, GameBuilder, StaticBlockName, StaticItemName, StaticTextureName,
    },
};
use anyhow::Result;
use perovskite_core::{
    chat::{ChatMessage, SERVER_ERROR_COLOR},
    constants::{
        block_groups::{self, DEFAULT_LIQUID, TOOL_REQUIRED, TRIVIALLY_REPLACEABLE},
        item_groups::HIDDEN_FROM_CREATIVE,
        permissions,
    },
    protocol::{
        self,
        blocks::{block_type_def::PhysicsInfo, FluidPhysicsInfo},
        items::{item_stack::QuantityType, InteractionRule},
    },
};
use perovskite_server::game_state::{
    blocks::{BlockInteractionResult, ExtDataHandling, ExtendedData},
    client_ui::UiElementContainer,
    items::{Item, ItemStack},
};

use super::{
    block_groups::{BRITTLE, GRANULAR},
    mapgen::OreDefinition,
    recipes::RecipeSlot,
    shaped_blocks::{make_slab, make_stairs},
    DefaultGameBuilder, DefaultGameBuilderExtension,
};

/// Dirt without grass on it.
pub const DIRT: StaticBlockName = StaticBlockName("default:dirt");
/// Dirt with grass on top.
pub const DIRT_WITH_GRASS: StaticBlockName = StaticBlockName("default:dirt_with_grass");
/// Solid grey stone.
pub const STONE: StaticBlockName = StaticBlockName("default:stone");

/// Beach sand.
pub const SAND: StaticBlockName = StaticBlockName("default:sand");

/// Desert materials.
pub const DESERT_STONE: StaticBlockName = StaticBlockName("default:desert_stone");
pub const DESERT_SAND: StaticBlockName = StaticBlockName("default:desert_sand");

/// Transparent glass.
pub const GLASS: StaticBlockName = StaticBlockName("default:glass");
/// Unlocked chest.
pub const CHEST: StaticBlockName = StaticBlockName("default:chest");
/// Locked chest.
pub const LOCKED_CHEST: StaticBlockName = StaticBlockName("default:locked_chest");

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

const DIRT_TEXTURE: StaticTextureName = StaticTextureName("default:dirt");
const DIRT_GRASS_SIDE_TEXTURE: StaticTextureName = StaticTextureName("default:dirt_grass_side");
const GRASS_TOP_TEXTURE: StaticTextureName = StaticTextureName("default:grass_top");
const STONE_TEXTURE: StaticTextureName = StaticTextureName("default:stone");
const SAND_TEXTURE: StaticTextureName = StaticTextureName("default:sand");
const DESERT_STONE_TEXTURE: StaticTextureName = StaticTextureName("default:desert_stone");
const DESERT_SAND_TEXTURE: StaticTextureName = StaticTextureName("default:desert_sand");
const GLASS_TEXTURE: StaticTextureName = StaticTextureName("default:glass");
const WATER_TEXTURE: StaticTextureName = StaticTextureName("default:water");

const CHEST_SIDE_TEXTURE: StaticTextureName = StaticTextureName("default:chest_side");
const CHEST_TOP_TEXTURE: StaticTextureName = StaticTextureName("default:chest_top");
const CHEST_FRONT_TEXTURE: StaticTextureName = StaticTextureName("default:chest_front");
const LOCKED_CHEST_FRONT_TEXTURE: StaticTextureName =
    StaticTextureName("default:chest_front_locked");

const TORCH_TEXTURE: StaticTextureName = StaticTextureName("default:torch");
const TNT_TEXTURE: StaticTextureName = StaticTextureName("default:tnt");
const TESTONLY_UNKNOWN_TEX: StaticTextureName = StaticTextureName("default:testonly_unknown");

const LIMESTONE_TEXTURE: StaticTextureName = StaticTextureName("default:limestone");
const LIMESTONE_LIGHT_TEXTURE: StaticTextureName = StaticTextureName("default:limestone_light");
const LIMESTONE_DARK_TEXTURE: StaticTextureName = StaticTextureName("default:limestone_dark");

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
        )?;
        let coal_ore = game_builder.add_block(
            BlockBuilder::new(COAL_ORE)
                .set_cube_appearance(
                    CubeAppearanceBuilder::new().set_single_texture(COAL_ORE_TEXTURE),
                )
                .add_block_group(BRITTLE)
                .add_block_group(TOOL_REQUIRED)
                .set_dropped_item_closure(|| (COAL_PIECE, rand::thread_rng().gen_range(1..=2))),
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
        )?;
        game_builder.register_basic_item(
            IRON_INGOT,
            "Iron ingot",
            IRON_INGOT_TEXTURE,
            vec![item_groups::METAL_INGOTS.into()],
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
                .set_dropped_item_closure(|| (IRON_PIECE, rand::thread_rng().gen_range(1..=2))),
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
        )?;
        let diamond_ore = game_builder.add_block(
            BlockBuilder::new(DIAMOND_ORE)
                .set_cube_appearance(
                    CubeAppearanceBuilder::new().set_single_texture(DIAMOND_ORE_TEXTURE),
                )
                .add_block_group(BRITTLE)
                .add_block_group(TOOL_REQUIRED)
                .set_dropped_item_closure(|| (DIAMOND_PIECE, rand::thread_rng().gen_range(1..=2))),
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
        )?;
        game_builder.register_basic_item(
            GOLD_INGOT,
            "Gold ingot",
            GOLD_INGOT_TEXTURE,
            vec![item_groups::METAL_INGOTS.into()],
        )?;
        let gold_ore = game_builder.add_block(
            BlockBuilder::new(GOLD_ORE)
                .set_cube_appearance(
                    CubeAppearanceBuilder::new().set_single_texture(GOLD_ORE_TEXTURE),
                )
                .add_block_group(BRITTLE)
                .add_block_group(TOOL_REQUIRED)
                .set_dropped_item_closure(|| (GOLD_PIECE, rand::thread_rng().gen_range(1..=2))),
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
    builder.inner.items_mut().register_item(Item {
        proto: protocol::items::ItemDef {
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
        },
        dig_handler: None,
        tap_handler: None,
        place_handler: None,
    })?;

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
    include_texture_bytes!(game_builder, CHEST_SIDE_TEXTURE, "textures/chest_side.png")?;
    include_texture_bytes!(game_builder, CHEST_TOP_TEXTURE, "textures/chest_top.png")?;
    include_texture_bytes!(
        game_builder,
        CHEST_FRONT_TEXTURE,
        "textures/chest_front.png"
    )?;
    include_texture_bytes!(
        game_builder,
        LOCKED_CHEST_FRONT_TEXTURE,
        "textures/chest_front_locked.png"
    )?;

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

    let dirt = game_builder.add_block(
        BlockBuilder::new(DIRT)
            .add_block_group(GRANULAR)
            .set_cube_single_texture(DIRT_TEXTURE)
            .set_display_name("Dirt block"),
    )?;
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
            // testonly
            .set_dropped_item_closure(|| {
                // Test only: There is no way to get glass yet (no sand, no crafting)
                // We need glass to test renderer changes
                if TESTONLY_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst) % 2 == 0 {
                    (StaticItemName("default:dirt"), 1)
                } else {
                    (StaticItemName("default:glass"), 5)
                }
            }),
    )?;
    let stone = game_builder.add_block(
        BlockBuilder::new(STONE)
            // TODO: make not-diggable-by-hand after tools are implemented
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
        .add_block_group(TRIVIALLY_REPLACEABLE)
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

    Ok(())
}

fn register_chest(game_builder: &mut GameBuilder) -> Result<()> {
    game_builder.add_block(
        BlockBuilder::new(CHEST)
            .set_cube_appearance(
                CubeAppearanceBuilder::new()
                    .set_individual_textures(
                        CHEST_SIDE_TEXTURE,
                        CHEST_SIDE_TEXTURE,
                        CHEST_TOP_TEXTURE,
                        CHEST_TOP_TEXTURE,
                        CHEST_FRONT_TEXTURE,
                        CHEST_SIDE_TEXTURE,
                    )
                    .set_rotate_laterally(),
            )
            .set_display_name("Unlocked chest")
            .add_modifier(Box::new(|bt| {
                bt.extended_data_handling = ExtDataHandling::ServerSide;
                bt.interact_key_handler = Some(Box::new(|ctx, coord| match ctx.initiator() {
                    perovskite_server::game_state::event::EventInitiator::Player(p) => {
                        Ok(Some(make_chest_popup(&ctx, coord, p)?))
                    }
                    _ => Ok(None),
                }))
            })),
    )?;

    const LOCKED_CHEST_OWNER: &str = "default:locked_chest:owner";

    game_builder.add_block(
        BlockBuilder::new(LOCKED_CHEST)
            .set_cube_appearance(
                CubeAppearanceBuilder::new()
                    .set_individual_textures(
                        CHEST_SIDE_TEXTURE,
                        CHEST_SIDE_TEXTURE,
                        CHEST_TOP_TEXTURE,
                        CHEST_TOP_TEXTURE,
                        LOCKED_CHEST_FRONT_TEXTURE,
                        CHEST_SIDE_TEXTURE,
                    )
                    .set_rotate_laterally(),
            )
            .set_display_name("Locked chest")
            .add_modifier(Box::new(|bt| {
                bt.extended_data_handling = ExtDataHandling::ServerSide;
                bt.interact_key_handler = Some(Box::new(|ctx, coord| match ctx.initiator() {
                    perovskite_server::game_state::event::EventInitiator::Player(p) => {
                        let (_, owner) =
                            ctx.game_map().get_block_with_extended_data(coord, |data| {
                                Ok(data.simple_data.get(LOCKED_CHEST_OWNER).cloned())
                            })?;

                        let owner_matches = if let Some(owner) = &owner {
                            owner == p.player.name()
                        } else {
                            true
                        };

                        if owner_matches
                            || p.player
                                .has_permission(permissions::BYPASS_INVENTORY_CHECKS)
                        {
                            Ok(Some(make_chest_popup(&ctx, coord, p)?))
                        } else {
                            // unwrap is safe - if owner were none, we would have given access to the chest
                            p.player.send_chat_message(
                                ChatMessage::new_server_message(format!(
                                    "Only {} can open this chest",
                                    owner.unwrap()
                                ))
                                .with_color(SERVER_ERROR_COLOR),
                            )?;
                            Ok(None)
                        }
                    }
                    _ => Ok(None),
                }))
            }))
            .add_item_modifier(Box::new(|it| {
                let old_place_handler = it.place_handler.take().unwrap();
                it.place_handler = Some(Box::new(move |ctx, coord, anchor, tool_stack| {
                    let result = old_place_handler(ctx, coord, anchor, tool_stack)?;
                    if let Some(player) = ctx.initiator().player_name() {
                        ctx.game_map()
                            .mutate_block_atomically(coord, |_, extended_data| {
                                let data = extended_data.get_or_insert_with(ExtendedData::default);
                                data.simple_data
                                    .insert(LOCKED_CHEST_OWNER.to_string(), player.to_string());
                                Ok(())
                            })?;
                    }
                    Ok(result)
                }));
            })),
    )?;
    game_builder
        .builder_extension::<DefaultGameBuilderExtension>()
        .crafting_recipes
        .register_recipe(super::recipes::RecipeImpl {
            slots: [
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Empty,
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
            ],
            result: ItemStack {
                proto: protocol::items::ItemStack {
                    item_name: CHEST.0.to_string(),
                    quantity: 1,
                    current_wear: 0,
                    quantity_type: Some(QuantityType::Stack(256)),
                },
            },
            shapeless: false,
            metadata: (),
        });
    game_builder
        .builder_extension::<DefaultGameBuilderExtension>()
        .crafting_recipes
        .register_recipe(super::recipes::RecipeImpl {
            slots: [
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Exact(IRON_INGOT.0.to_string()),
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
            ],
            result: ItemStack {
                proto: protocol::items::ItemStack {
                    item_name: LOCKED_CHEST.0.to_string(),
                    quantity: 1,
                    current_wear: 0,
                    quantity_type: Some(QuantityType::Stack(256)),
                },
            },
            shapeless: false,
            metadata: (),
        });
    Ok(())
}

fn make_chest_popup(
    ctx: &perovskite_server::game_state::event::HandlerContext<'_>,
    coord: perovskite_core::coordinates::BlockCoordinate,
    p: &perovskite_server::game_state::event::PlayerInitiator<'_>,
) -> Result<perovskite_server::game_state::client_ui::Popup, anyhow::Error> {
    Ok((ctx
        .new_popup()
        .title("Chest")
        .inventory_view_block(
            "chest_inv",
            "Chest contents:",
            (4, 8),
            coord,
            "chest_inv".to_string(),
            true,
            true,
            false,
        )?
        .inventory_view_stored(
            "player_inv",
            "Player inventory:",
            p.player.main_inventory(),
            true,
            true,
        ))?)
}
