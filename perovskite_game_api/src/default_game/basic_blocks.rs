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
    blocks::{BlockBuilder, CubeAppearanceBuilder, MatterType, PlantLikeAppearanceBuilder},
    game_builder::{include_texture_bytes, BlockName, GameBuilder, ItemName, TextureName},
};
use anyhow::Result;
use perovskite_core::{
    constants::block_groups::{self, DEFAULT_LIQUID, TOOL_REQUIRED},
    coordinates::ChunkOffset,
    protocol::{
        self,
        blocks::{
            block_type_def::PhysicsInfo, AxisAlignedBox, AxisAlignedBoxRotation, AxisAlignedBoxes,
            FluidPhysicsInfo,
        },
        items::item_stack::QuantityType,
    },
};
use perovskite_server::game_state::{
    blocks::{BlockInteractionResult, BlockTypeHandle, ExtDataHandling},
    game_map::{BulkUpdateCallback, TimerCallback, TimerSettings, TimerState},
    items::{BlockInteractionHandler, ItemStack},
};

use super::{
    block_groups::{BRITTLE, GRANULAR},
    mapgen::OreDefinition,
    recipes::RecipeSlot,
    DefaultGameBuilder,
};

/// Dirt without grass on it.
pub const DIRT: BlockName = BlockName("default:dirt");
/// Dirt with grass on top.
pub const DIRT_WITH_GRASS: BlockName = BlockName("default:dirt_with_grass");
/// Solid grey stone.
pub const STONE: BlockName = BlockName("default:stone");
/// Transparent glass.
pub const GLASS: BlockName = BlockName("default:glass");
/// Unlocked chest.
pub const CHEST: BlockName = BlockName("default:chest");

/// Torch
pub const TORCH: BlockName = BlockName("default:torch");

/// Water
/// Stability note: not stable (liquids are TBD)
pub const WATER: BlockName = BlockName("default:water");

/// test only, no crafting recipe, not finalized
pub const TNT: BlockName = BlockName("default:tnt");

const DIRT_TEXTURE: TextureName = TextureName("default:dirt");
const DIRT_GRASS_SIDE_TEXTURE: TextureName = TextureName("default:dirt_grass_side");
const GRASS_TOP_TEXTURE: TextureName = TextureName("default:grass_top");
const STONE_TEXTURE: TextureName = TextureName("default:stone");
const GLASS_TEXTURE: TextureName = TextureName("default:glass");
const WATER_TEXTURE: TextureName = TextureName("default:water");
// TODO real chest texture
const CHEST_TEXTURE: TextureName = TextureName("default:chest");
const TORCH_TEXTURE: TextureName = TextureName("default:torch");
const TNT_TEXTURE: TextureName = TextureName("default:tnt");

pub mod ores {
    use perovskite_core::constants::block_groups::TOOL_REQUIRED;
    use perovskite_server::game_state::items::ItemStack;
    use rand::Rng;

    use crate::{
        blocks::CubeAppearanceBuilder,
        default_game::recipes::{RecipeImpl, RecipeSlot},
        game_builder::ItemName,
    };

    use super::*;

    const COAL_ORE: BlockName = BlockName("default:coal_ore");
    const COAL_PIECE: ItemName = ItemName("default:coal_piece");
    const COAL_ORE_TEXTURE: TextureName = TextureName("default:coal_ore");
    const COAL_PIECE_TEXTURE: TextureName = TextureName("default:coal_piece");

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
        let coal_ore_builder = BlockBuilder::new(COAL_ORE)
            .set_cube_appearance(CubeAppearanceBuilder::new().set_single_texture(COAL_ORE_TEXTURE))
            .set_inventory_texture(COAL_ORE_TEXTURE)
            .add_block_group(BRITTLE)
            .add_block_group(TOOL_REQUIRED)
            .set_dropped_item_closure(|| (COAL_PIECE, rand::thread_rng().gen_range(1..=2)));
        let coal_ore = coal_ore_builder.build_and_deploy_into(game_builder.game_builder())?;
        game_builder.register_ore(OreDefinition {
            block: coal_ore,
            noise_cutoff: splines::Spline::from_vec(vec![
                splines::Key {
                    value: -0.5,
                    t: 0.,
                    interpolation: splines::Interpolation::Linear,
                },
                splines::Key {
                    value: -0.6,
                    t: 100.,
                    interpolation: splines::Interpolation::Linear,
                },
            ]),

            noise_scale: (4., 0.25, 4.),
        });
        game_builder.smelting_fuels.register_recipe(RecipeImpl {
            slots: [RecipeSlot::Exact(COAL_PIECE.0.to_string())],
            result: ItemStack {
                proto: Default::default(),
            },
            shapeless: false,
            metadata: 8,
        });
        Ok(())
    }
}

pub(crate) fn register_basic_blocks(game_builder: &mut DefaultGameBuilder) -> Result<()> {
    register_core_blocks(&mut game_builder.inner)?;
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
    include_texture_bytes!(builder, TNT_TEXTURE, "textures/tnt.png")?;
    let air = builder.air_block;
    BlockBuilder::new(TNT)
        .set_cube_appearance(CubeAppearanceBuilder::new().set_single_texture(TNT_TEXTURE))
        .set_inventory_texture(TNT_TEXTURE)
        .set_modifier(Box::new(move |block_type| {
            block_type.tap_handler_full = Some(Box::new(move |ctx, coord, tool| {
                if tool.is_some_and(|tool| tool.proto.item_name == "default:superuser_pickaxe") {
                    return Ok(Default::default());
                }

                for i in -3..=3 {
                    for j in -3..=3 {
                        for k in -3..=3 {
                            if let Some(neighbor) = coord.try_delta(i, j, k) {
                                ctx.game_map().set_block(neighbor, air, None)?;
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
        }))
        .build_and_deploy_into(builder)?;

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
    include_texture_bytes!(game_builder, GLASS_TEXTURE, "textures/glass.png")?;

    include_texture_bytes!(game_builder, WATER_TEXTURE, "textures/water.png")?;
    include_texture_bytes!(game_builder, CHEST_TEXTURE, "textures/chest_side.png")?;
    include_texture_bytes!(game_builder, TORCH_TEXTURE, "textures/torch.png")?;
    let dirt = game_builder.add_block(
        BlockBuilder::new(DIRT)
            .add_block_group(GRANULAR)
            .set_cube_single_texture(DIRT_TEXTURE)
            .set_inventory_texture(DIRT_TEXTURE)
            .set_inventory_display_name("Dirt block"),
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
            .set_inventory_texture(DIRT_GRASS_SIDE_TEXTURE)
            .set_dropped_item(DIRT.0, 1)
            // testonly
            .set_dropped_item_closure(|| {
                // Test only: There is no way to get glass yet (no sand, no crafting)
                // We need glass to test renderer changes
                if TESTONLY_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst) % 2 == 0 {
                    (ItemName("default:dirt"), 1)
                } else {
                    (ItemName("default:glass"), 5)
                }
            }),
    )?;
    let stone = game_builder.add_block(
        BlockBuilder::new(STONE)
            // TODO: make not-diggable-by-hand after tools are implemented
            .add_block_group(BRITTLE)
            .add_block_group(TOOL_REQUIRED)
            .set_cube_single_texture(STONE_TEXTURE)
            .set_inventory_texture(STONE_TEXTURE)
            .set_inventory_display_name("Stone block"),
    )?;

    // testonly - add a handler to generate slabs, and extra builders to allow other
    // blocks to set custom geometry in a fine grained manner
    let stone_slab = game_builder.add_block(
        BlockBuilder::new(BlockName("default:stone_slab"))
            // TODO: make not-diggable-by-hand after tools are implemented
            .add_block_group(BRITTLE)
            .add_block_group(TOOL_REQUIRED)
            .set_cube_single_texture(STONE_TEXTURE)
            .set_inventory_texture(STONE_TEXTURE)
            .set_inventory_display_name("Stone slab")
            .set_allow_light_propagation(true)
            .set_modifier(Box::new(|bt| {
                let the_box = AxisAlignedBox {
                    x_min: -0.5,
                    y_min: -0.5,
                    z_min: -0.5,
                    x_max: 0.5,
                    y_max: 0.0,
                    z_max: 0.5,
                    tex_left: Some(STONE_TEXTURE.into()),
                    tex_right: Some(STONE_TEXTURE.into()),
                    tex_top: Some(STONE_TEXTURE.into()),
                    tex_bottom: Some(STONE_TEXTURE.into()),
                    tex_front: Some(STONE_TEXTURE.into()),
                    tex_back: Some(STONE_TEXTURE.into()),
                    rotation: AxisAlignedBoxRotation::None.into(),
                    variant_mask: 0,
                };

                let aabbs = AxisAlignedBoxes {
                    boxes: vec![the_box],
                };
                bt.client_info.physics_info =
                    Some(PhysicsInfo::SolidCustomCollisionboxes(aabbs.clone()));
                bt.client_info.render_info =
                    Some(protocol::blocks::block_type_def::RenderInfo::AxisAlignedBoxes(aabbs))
            })),
    )?;

    let stone_stair = game_builder.add_block(
        BlockBuilder::new(BlockName("default:stone_stair"))
            // TODO: make not-diggable-by-hand after tools are implemented
            .add_block_group(BRITTLE)
            .add_block_group(TOOL_REQUIRED)
            .set_cube_single_texture(STONE_TEXTURE)
            .set_inventory_texture(STONE_TEXTURE)
            .set_inventory_display_name("Stone stair")
            .set_allow_light_propagation(true)
            .set_cube_appearance(CubeAppearanceBuilder::new().set_rotate_laterally())
            .set_modifier(Box::new(|bt| {
                let bottom_box = AxisAlignedBox {
                    x_min: -0.5,
                    y_min: -0.5,
                    z_min: -0.5,
                    x_max: 0.5,
                    y_max: 0.0,
                    z_max: 0.5,
                    tex_left: Some(STONE_TEXTURE.into()),
                    tex_right: Some(STONE_TEXTURE.into()),
                    tex_top: Some(STONE_TEXTURE.into()),
                    tex_bottom: Some(STONE_TEXTURE.into()),
                    tex_front: Some(STONE_TEXTURE.into()),
                    tex_back: Some(STONE_TEXTURE.into()),
                    rotation: AxisAlignedBoxRotation::Nesw.into(),
                    variant_mask: 0,
                };

                let back_box = AxisAlignedBox {
                    x_min: -0.5,
                    y_min: -0.5,
                    z_min: 0.0,
                    x_max: 0.5,
                    y_max: 0.5,
                    z_max: 0.5,
                    tex_left: Some(STONE_TEXTURE.into()),
                    tex_right: Some(STONE_TEXTURE.into()),
                    tex_top: Some(STONE_TEXTURE.into()),
                    tex_bottom: Some(STONE_TEXTURE.into()),
                    tex_front: Some(STONE_TEXTURE.into()),
                    tex_back: Some(STONE_TEXTURE.into()),
                    rotation: AxisAlignedBoxRotation::Nesw.into(),
                    variant_mask: 0,
                };

                let aabbs = AxisAlignedBoxes {
                    boxes: vec![bottom_box, back_box],
                };
                bt.client_info.physics_info =
                    Some(PhysicsInfo::SolidCustomCollisionboxes(aabbs.clone()));
                bt.client_info.render_info =
                    Some(protocol::blocks::block_type_def::RenderInfo::AxisAlignedBoxes(aabbs))
            })),
    )?;

    game_builder.add_block(
        BlockBuilder::new(GLASS)
            .add_block_group(BRITTLE)
            .set_cube_appearance(
                CubeAppearanceBuilder::new()
                    .set_single_texture(GLASS_TEXTURE)
                    .set_needs_transparency(),
            )
            .set_allow_light_propagation(true)
            .set_inventory_texture(GLASS_TEXTURE)
            .set_inventory_display_name("Glass block"),
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
        .set_inventory_texture(WATER_TEXTURE)
        .set_inventory_display_name("Water block")
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
            .set_inventory_texture(TORCH_TEXTURE)
            .set_inventory_display_name("Torch")
            .set_light_emission(8),
    )?;

    // testonly
    game_builder.add_block(
        BlockBuilder::new(CHEST)
            .set_cube_single_texture(CHEST_TEXTURE)
            .set_inventory_texture(CHEST_TEXTURE)
            .set_inventory_display_name("Unlocked chest")
            .set_modifier(Box::new(|bt| {
                bt.extended_data_handling = ExtDataHandling::ServerSide;
                bt.interact_key_handler = Some(Box::new(|ctx, coord| match ctx.initiator() {
                    perovskite_server::game_state::event::EventInitiator::Player(p) => Ok(Some(
                        (ctx.new_popup()
                            .title("Chest")
                            .label("Chest contents:")
                            .inventory_view_block(
                                "chest_inv",
                                (4, 8),
                                coord,
                                "chest_inv".to_string(),
                                true,
                                true,
                                false,
                            )?
                            .label("Player inventory:")
                            .inventory_view_stored(
                                "player_inv",
                                p.player.main_inventory(),
                                true,
                                true,
                            ))?,
                    )),
                    _ => Ok(None),
                }))
            })),
    )?;

    // game_builder.inner.add_timer(
    //     "testonly_spam",
    //     TimerSettings {
    //         interval: Duration::from_secs(5),
    //         shards: 16,
    //         spreading: 1.0,
    //         block_types: vec![dirt.0, stone.0],
    //         per_block_probability: 1.0,
    //         ..Default::default()
    //     },
    //     TimerCallback::BulkUpdate(Box::new(TestSpamDirtStoneCallback {
    //         dirt: dirt.0,
    //         stone: stone.0,
    //     })),
    // );

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
