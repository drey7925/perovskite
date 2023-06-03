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

use crate::{
    blocks::BlockBuilder,
    game_builder::{include_texture_bytes, Block, GameBuilder, Tex},
};
use anyhow::Result;

use super::block_groups::{GRANULAR, BRITTLE};

/// Dirt without grass on it.
pub const DIRT: Block = Block("default:dirt");
/// Dirt with grass on top.
pub const DIRT_WITH_GRASS: Block = Block("default:dirt_with_grass");
/// Solid grey stone.
pub const STONE: Block = Block("default:stone");



/// Water
/// Stability note: not stable (liquids are TBD)
pub const WATER: Block = Block("default:water");

const DIRT_TEXTURE: Tex = Tex("default:dirt");
const DIRT_GRASS_SIDE_TEXTURE: Tex = Tex("default:dirt_grass_side");
const GRASS_TOP_TEXTURE: Tex = Tex("default:grass_top");
const STONE_TEXTURE: Tex = Tex("default:stone");


const WATER_TEXTURE: Tex = Tex("default:water");

pub(crate) fn register_basic_blocks(game_builder: &mut GameBuilder) -> Result<()> {
    register_core_blocks(game_builder)?;
    Ok(())
}

fn register_core_blocks(game_builder: &mut GameBuilder) -> Result<()> {
    include_texture_bytes!(game_builder, DIRT_TEXTURE, "textures/dirt.png")?;
    include_texture_bytes!(
        game_builder,
        DIRT_GRASS_SIDE_TEXTURE,
        "textures/dirt_grass_side.png"
    )?;
    include_texture_bytes!(game_builder, STONE_TEXTURE, "textures/stone.png")?;
    include_texture_bytes!(game_builder, GRASS_TOP_TEXTURE, "textures/grass_top.png")?;
    include_texture_bytes!(game_builder, WATER_TEXTURE, "textures/water.png")?;
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
                DIRT_TEXTURE,
            )
            .set_dropped_item(DIRT.0, 1),
    )?;
    game_builder.add_block(
        BlockBuilder::new(STONE)
        // TODO: make not-diggable-by-hand after tools are implemented
            .add_block_group(BRITTLE)
            .set_texture_all(STONE_TEXTURE)
            .set_inventory_display_name("Dirt block"),
    )?;
    // TODO: implement actual water
    game_builder.add_block(
        BlockBuilder::new(WATER)
       
            .add_block_group(BRITTLE)
            .set_texture_all(WATER_TEXTURE)
            .set_inventory_display_name("Dirt block"),
    )?;
    Ok(())
}
