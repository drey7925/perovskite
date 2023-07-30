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

use std::sync::atomic::AtomicU32;

use crate::{
    blocks::BlockBuilder,
    game_builder::{include_texture_bytes, BlockName, GameBuilder, ItemName, TextureName},
};
use anyhow::Result;
use cuberef_core::{protocol::blocks::{block_type_def::PhysicsInfo, FluidPhysicsInfo}, constants::block_groups::TOOL_REQUIRED};
use cuberef_server::game_state::blocks::ExtDataHandling;
use splines::Spline;

use super::{
    block_groups::{BRITTLE, GRANULAR},
    mapgen::OreDefinition,
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

/// Water
/// Stability note: not stable (liquids are TBD)
pub const WATER: BlockName = BlockName("default:water");

const DIRT_TEXTURE: TextureName = TextureName("default:dirt");
const DIRT_GRASS_SIDE_TEXTURE: TextureName = TextureName("default:dirt_grass_side");
const GRASS_TOP_TEXTURE: TextureName = TextureName("default:grass_top");
const STONE_TEXTURE: TextureName = TextureName("default:stone");
const GLASS_TEXTURE: TextureName = TextureName("default:glass");
const WATER_TEXTURE: TextureName = TextureName("default:water");
// TODO real chest texture
const CHEST_TEXTURE: TextureName = TextureName("default:chest");

pub mod ores {
    use cuberef_core::constants::block_groups::TOOL_REQUIRED;
    use cuberef_server::game_state::items::ItemStack;
    use rand::Rng;

    use crate::{game_builder::ItemName, default_game::recipes::{RecipeImpl, RecipeSlot}};

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
            .set_texture_all(COAL_ORE_TEXTURE)
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
            slots: [
                RecipeSlot::Exact(COAL_PIECE.0.to_string()),
            ],
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
    ores::register_ores(game_builder)?;
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
    game_builder.add_block(
        BlockBuilder::new(DIRT)
            .add_block_group(GRANULAR)
            .set_texture_all(DIRT_TEXTURE)
            .set_inventory_display_name("Dirt block"),
    )?;
    game_builder.add_block(
        BlockBuilder::new(DIRT_WITH_GRASS)
            .add_block_group(GRANULAR)
            .set_individual_textures(
                DIRT_GRASS_SIDE_TEXTURE,
                DIRT_GRASS_SIDE_TEXTURE,
                GRASS_TOP_TEXTURE,
                DIRT_TEXTURE,
                DIRT_GRASS_SIDE_TEXTURE,
                DIRT_GRASS_SIDE_TEXTURE,
                DIRT_GRASS_SIDE_TEXTURE,
            )
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
    game_builder.add_block(
        BlockBuilder::new(STONE)
            // TODO: make not-diggable-by-hand after tools are implemented
            .add_block_group(BRITTLE)
            .add_block_group(TOOL_REQUIRED)
            .set_texture_all(STONE_TEXTURE)
            .set_inventory_display_name("Stone block"),
    )?;
    game_builder.add_block(
        BlockBuilder::new(GLASS)
            .add_block_group(BRITTLE)
            .set_texture_all(GLASS_TEXTURE)
            .set_needs_transparency()
            .set_inventory_display_name("Glass block"),
    )?;
    // TODO: implement actual water
    let mut water_builder = BlockBuilder::new(WATER)
        .add_block_group(BRITTLE)
        .add_item_group("testonly_wet")
        .set_texture_all(WATER_TEXTURE)
        .set_inventory_display_name("Water block")
        .set_needs_translucency();
    water_builder.physics_info = PhysicsInfo::Fluid(FluidPhysicsInfo {
        horizontal_speed: 1.5,
        vertical_speed: -0.5,
        jump_speed: 1.0,
        sink_speed: -1.0,
        surface_thickness: 0.1,
        surf_horizontal_speed: 2.,
        surf_vertical_speed: -0.5,
        surf_jump_speed: 1.0,
        surf_sink_speed: -0.5,
    });
    game_builder.add_block(water_builder)?;

    // testonly
    game_builder.add_block(
        BlockBuilder::new(CHEST)
            .set_texture_all(CHEST_TEXTURE)
            .set_inventory_display_name("Unlocked chest")
            .set_modifier(Box::new(|bt| {
                bt.extended_data_handling = ExtDataHandling::ServerSide;
                bt.interact_key_handler = Some(Box::new(|ctx, coord| match ctx.initiator() {
                    cuberef_server::game_state::event::EventInitiator::Engine => Ok(None),
                    cuberef_server::game_state::event::EventInitiator::Player(p) => Ok(Some(
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
                            .inventory_view_stored("player_inv", p.main_inventory(), true, true))?,
                    )),
                }))
            })),
    )?;

    Ok(())
}
