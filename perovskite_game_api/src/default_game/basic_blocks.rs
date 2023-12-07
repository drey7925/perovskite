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
    game_builder::{
        include_texture_bytes, GameBuilder, StaticBlockName, StaticItemName, TextureName,
    },
};
use anyhow::Result;
use perovskite_core::{
    constants::{
        block_groups::{self, DEFAULT_LIQUID, TOOL_REQUIRED},
        item_groups::HIDDEN_FROM_CREATIVE,
    },
    coordinates::ChunkOffset,
    protocol::{
        self,
        blocks::{block_type_def::PhysicsInfo, FluidPhysicsInfo},
        items::{item_stack::QuantityType, InteractionRule},
    },
};
use perovskite_server::game_state::{
    blocks::{BlockInteractionResult, BlockTypeHandle, ExtDataHandling},
    game_map::{BulkUpdateCallback, TimerState},
    items::{Item, ItemStack}, client_ui::UiElementContainer,
};

use super::{
    block_groups::{BRITTLE, GRANULAR},
    mapgen::OreDefinition,
    recipes::RecipeSlot,
    shaped_blocks::{make_slab, make_stairs},
    DefaultGameBuilder,
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

/// Torch
pub const TORCH: StaticBlockName = StaticBlockName("default:torch");

/// Water
/// Stability note: not stable (liquids are TBD)
pub const WATER: StaticBlockName = StaticBlockName("default:water");

/// test only, no crafting recipe, not finalized
pub const TNT: StaticBlockName = StaticBlockName("default:tnt");

const DIRT_TEXTURE: TextureName = TextureName("default:dirt");
const DIRT_GRASS_SIDE_TEXTURE: TextureName = TextureName("default:dirt_grass_side");
const GRASS_TOP_TEXTURE: TextureName = TextureName("default:grass_top");
const STONE_TEXTURE: TextureName = TextureName("default:stone");
const SAND_TEXTURE: TextureName = TextureName("default:sand");
const DESERT_STONE_TEXTURE: TextureName = TextureName("default:desert_stone");
const DESERT_SAND_TEXTURE: TextureName = TextureName("default:desert_sand");
const GLASS_TEXTURE: TextureName = TextureName("default:glass");
const WATER_TEXTURE: TextureName = TextureName("default:water");
// TODO real chest texture
const CHEST_TEXTURE: TextureName = TextureName("default:chest");
const TORCH_TEXTURE: TextureName = TextureName("default:torch");
const TNT_TEXTURE: TextureName = TextureName("default:tnt");
const TESTONLY_UNKNOWN_TEX: TextureName = TextureName("default:testonly_unknown");

pub mod ores {
    use perovskite_core::constants::block_groups::TOOL_REQUIRED;
    use perovskite_server::game_state::items::ItemStack;
    use rand::Rng;

    use crate::{
        blocks::CubeAppearanceBuilder,
        default_game::{recipes::{RecipeImpl, RecipeSlot}, item_groups},
        game_builder::StaticItemName,
    };

    use super::*;

    const COAL_ORE: StaticBlockName = StaticBlockName("default:coal_ore");
    const COAL_PIECE: StaticItemName = StaticItemName("default:coal_piece");
    const COAL_ORE_TEXTURE: TextureName = TextureName("default:coal_ore");
    const COAL_PIECE_TEXTURE: TextureName = TextureName("default:coal_piece");

    const IRON_ORE: StaticBlockName = StaticBlockName("default:iron_ore");
    const IRON_PIECE: StaticItemName = StaticItemName("default:iron_piece");
    const IRON_INGOT: StaticItemName = StaticItemName("default:iron_ingot");
    const IRON_ORE_TEXTURE: TextureName = TextureName("default:iron_ore");
    const IRON_PIECE_TEXTURE: TextureName = TextureName("default:iron_piece");
    const IRON_INGOT_TEXTURE: TextureName = TextureName("default:iron_ingot");

    const GOLD_ORE: StaticBlockName = StaticBlockName("default:gold_ore");
    const GOLD_PIECE: StaticItemName = StaticItemName("default:gold_piece");
    const GOLD_INGOT: StaticItemName = StaticItemName("default:gold_ingot");
    const GOLD_ORE_TEXTURE: TextureName = TextureName("default:gold_ore");
    const GOLD_PIECE_TEXTURE: TextureName = TextureName("default:gold_piece");
    const GOLD_INGOT_TEXTURE: TextureName = TextureName("default:gold_ingot");

    const DIAMOND_ORE: StaticBlockName = StaticBlockName("default:diamond_ore");
    const DIAMOND_PIECE: StaticItemName = StaticItemName("default:diamond_piece");
    const DIAMOND_ORE_TEXTURE: TextureName = TextureName("default:diamond_ore");
    const DIAMOND_PIECE_TEXTURE: TextureName = TextureName("default:diamond_piece");

    pub(crate) fn register_ores(game_builder: &mut DefaultGameBuilder) -> Result<()> {
        // todo factor this into a function per-ore
        include_texture_bytes!(
            &mut game_builder.inner,
            COAL_ORE_TEXTURE,
            "textures/coal_ore.png"
        )?;
        include_texture_bytes!(
            &mut game_builder.inner,
            COAL_PIECE_TEXTURE,
            "textures/coal_piece.png"
        )?;
        game_builder.game_builder().register_basic_item(
            COAL_PIECE,
            "Piece of coal",
            COAL_PIECE_TEXTURE,
            vec![],
        )?;
        let coal_ore = game_builder.game_builder().add_block(
            BlockBuilder::new(COAL_ORE)
                .set_cube_appearance(
                    CubeAppearanceBuilder::new().set_single_texture(COAL_ORE_TEXTURE),
                )
                .add_block_group(BRITTLE)
                .add_block_group(TOOL_REQUIRED)
                .set_dropped_item_closure(|| (COAL_PIECE, rand::thread_rng().gen_range(1..=2))),
        )?;

        game_builder.register_ore(OreDefinition {
            block: coal_ore.handle,
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
        game_builder.smelting_fuels.register_recipe(RecipeImpl {
            slots: [RecipeSlot::Exact(COAL_PIECE.0.to_string())],
            result: ItemStack {
                proto: Default::default(),
            },
            shapeless: false,
            metadata: 16,
        });

        include_texture_bytes!(
            &mut game_builder.inner,
            IRON_ORE_TEXTURE,
            "textures/iron_ore.png"
        )?;
        include_texture_bytes!(
            &mut game_builder.inner,
            IRON_PIECE_TEXTURE,
            "textures/iron_piece.png"
        )?;
        include_texture_bytes!(
            &mut game_builder.inner,
            IRON_INGOT_TEXTURE,
            "textures/iron_ingot.png"
        )?;
        game_builder.game_builder().register_basic_item(
            IRON_PIECE,
            "Piece of iron",
            IRON_PIECE_TEXTURE,
            vec![item_groups::RAW_ORES.into()],
        )?;
        game_builder.game_builder().register_basic_item(
            IRON_INGOT,
            "Iron ingot",
            IRON_INGOT_TEXTURE,
            vec![item_groups::METAL_INGOTS.into()],
        )?;
        game_builder.smelting_recipes.register_recipe(RecipeImpl {
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

        let iron_ore = game_builder.game_builder().add_block(
            BlockBuilder::new(IRON_ORE)
                .set_cube_appearance(
                    CubeAppearanceBuilder::new().set_single_texture(IRON_ORE_TEXTURE),
                )
                .add_block_group(BRITTLE)
                .add_block_group(TOOL_REQUIRED)
                .set_dropped_item_closure(|| (IRON_PIECE, rand::thread_rng().gen_range(1..=2))),
        )?;

        game_builder.register_ore(OreDefinition {
            block: iron_ore.handle,
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
            &mut game_builder.inner,
            DIAMOND_ORE_TEXTURE,
            "textures/diamond_ore.png"
        )?;
        include_texture_bytes!(
            &mut game_builder.inner,
            DIAMOND_PIECE_TEXTURE,
            "textures/diamond_piece.png"
        )?;
        game_builder.game_builder().register_basic_item(
            DIAMOND_PIECE,
            "Piece of diamond",
            DIAMOND_PIECE_TEXTURE,
            vec![item_groups::GEMS.into()],
        )?;
        let diamond_ore = game_builder.game_builder().add_block(
            BlockBuilder::new(DIAMOND_ORE)
                .set_cube_appearance(
                    CubeAppearanceBuilder::new().set_single_texture(DIAMOND_ORE_TEXTURE),
                )
                .add_block_group(BRITTLE)
                .add_block_group(TOOL_REQUIRED)
                .set_dropped_item_closure(|| (DIAMOND_PIECE, rand::thread_rng().gen_range(1..=2))),
        )?;

        game_builder.register_ore(OreDefinition {
            block: diamond_ore.handle,
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

        include_texture_bytes!(
            &mut game_builder.inner,
            GOLD_ORE_TEXTURE,
            "textures/gold_ore.png"
        )?;
        include_texture_bytes!(
            &mut game_builder.inner,
            GOLD_PIECE_TEXTURE,
            "textures/gold_piece.png"
        )?;
        include_texture_bytes!(
            &mut game_builder.inner,
            GOLD_INGOT_TEXTURE,
            "textures/gold_ingot.png"
        )?;
        game_builder.game_builder().register_basic_item(
            GOLD_PIECE,
            "Piece of gold",
            GOLD_PIECE_TEXTURE,
            vec![item_groups::RAW_ORES.into()],
        )?;
        game_builder.game_builder().register_basic_item(
            GOLD_INGOT,
            "Gold ingot",
            GOLD_INGOT_TEXTURE,
            vec![item_groups::METAL_INGOTS.into()],
        )?;
        let gold_ore = game_builder.game_builder().add_block(
            BlockBuilder::new(GOLD_ORE)
                .set_cube_appearance(
                    CubeAppearanceBuilder::new().set_single_texture(GOLD_ORE_TEXTURE),
                )
                .add_block_group(BRITTLE)
                .add_block_group(TOOL_REQUIRED)
                .set_dropped_item_closure(|| (GOLD_PIECE, rand::thread_rng().gen_range(1..=2))),
        )?;

        game_builder.register_ore(OreDefinition {
            block: gold_ore.handle,
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

        game_builder.smelting_recipes.register_recipe(RecipeImpl {
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

pub(crate) fn register_basic_blocks(game_builder: &mut DefaultGameBuilder) -> Result<()> {
    register_core_blocks(game_builder)?;
    register_tnt(&mut game_builder.inner)?;
    ores::register_ores(game_builder)?;

    game_builder
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
            shapeless: false,
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
            .set_modifier(Box::new(move |block_type| {
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

fn register_core_blocks(game_builder: &mut DefaultGameBuilder) -> Result<()> {
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
    include_texture_bytes!(game_builder, CHEST_TEXTURE, "textures/chest_side.png")?;
    include_texture_bytes!(game_builder, TORCH_TEXTURE, "textures/torch.png")?;

    include_texture_bytes!(
        game_builder,
        TESTONLY_UNKNOWN_TEX,
        "../media/block_unknown.png"
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

    let _sand = game_builder.add_block(
        BlockBuilder::new(SAND)
            .add_block_group(GRANULAR)
            .set_cube_single_texture(SAND_TEXTURE)
            .set_display_name("Sand")
            .set_falls_down(true),
    )?;

    let _desert_stone = game_builder.add_block(
        BlockBuilder::new(DESERT_STONE)
            .add_block_group(BRITTLE)
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

    // testonly
    game_builder.add_block(
        BlockBuilder::new(CHEST)
            .set_cube_single_texture(CHEST_TEXTURE)
            .set_display_name("Unlocked chest")
            .set_modifier(Box::new(|bt| {
                bt.extended_data_handling = ExtDataHandling::ServerSide;
                bt.interact_key_handler = Some(Box::new(|ctx, coord| match ctx.initiator() {
                    perovskite_server::game_state::event::EventInitiator::Player(p) => Ok(Some(
                        (ctx.new_popup()
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
                            ))?,
                    )),
                    _ => Ok(None),
                }))
            })),
    )?;

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
    make_slab(game_builder, &stone, true)?;
    make_slab(game_builder, &glass, true)?;
    make_slab(game_builder, &dirt, false)?;

    Ok(())
}

struct TestSpamDirtStoneCallback {
    dirt: BlockTypeHandle,
    stone: BlockTypeHandle,
}
impl BulkUpdateCallback for TestSpamDirtStoneCallback {
    fn bulk_update_callback(
        &self,
        _chunk_coordinate: perovskite_core::coordinates::ChunkCoordinate,
        _staet: &TimerState,
        _game_state: &std::sync::Arc<perovskite_server::game_state::GameState>,
        chunk: &mut perovskite_server::game_state::game_map::MapChunk,
        _neighbors: Option<&perovskite_server::game_state::game_map::ChunkNeighbors>,
    ) -> Result<()> {
        for x in 0..16 {
            for z in 0..16 {
                for y in 0..16 {
                    if chunk.get_block(ChunkOffset { x, y, z }) == self.dirt.id() {
                        chunk.set_block(ChunkOffset { x, y, z }, self.stone, None);
                    } else if chunk.get_block(ChunkOffset { x, y, z }) == self.stone.id() {
                        chunk.set_block(ChunkOffset { x, y, z }, self.dirt, None);
                    }
                }
            }
        }
        Ok(())
    }
}
