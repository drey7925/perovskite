use anyhow::Result;

use crate::{
    blocks::{BlockBuilder, CubeAppearanceBuilder, PlantLikeAppearanceBuilder},
    game_builder::{GameBuilder, StaticBlockName, TextureName},
    include_texture_bytes,
};

use super::{block_groups::FIBROUS, DefaultGameBuilder};

pub const MAPLE_TREE: StaticBlockName = StaticBlockName("default:maple_tree");
pub const MAPLE_TREE_TOP_TEX: TextureName = TextureName("default:maple_tree_top");
pub const MAPLE_TREE_SIDE_TEX: TextureName = TextureName("default:maple_tree_side");

pub const MAPLE_LEAVES: StaticBlockName = StaticBlockName("default:maple_leaves");
pub const MAPLE_LEAVES_TEX: TextureName = TextureName("default:maple_leaves");

pub const TALL_GRASS: StaticBlockName = StaticBlockName("default:tall_grass");
pub const TALL_GRASS_TEX: TextureName = TextureName("default:tall_grass");

pub const CACTUS: StaticBlockName = StaticBlockName("default:cactus");
pub const CACTUS_TOP_TEX: TextureName = TextureName("default:cactus_top");
pub const CACTUS_SIDE_TEX: TextureName = TextureName("default:cactus_side");

pub(crate) fn register_foliage(builder: &mut GameBuilder) -> Result<()> {
    include_texture_bytes!(builder, MAPLE_TREE_TOP_TEX, "textures/maple_tree_top.png")?;
    include_texture_bytes!(builder, MAPLE_TREE_SIDE_TEX, "textures/maple_tree_side.png")?;
    include_texture_bytes!(builder, MAPLE_LEAVES_TEX, "textures/maple_leaves.png")?;
    include_texture_bytes!(builder, TALL_GRASS_TEX, "textures/tall_grass.png")?;
    include_texture_bytes!(builder, CACTUS_TOP_TEX, "textures/cactus_top.png")?;
    include_texture_bytes!(builder, CACTUS_SIDE_TEX, "textures/cactus_side.png")?;
    builder.add_block(
        BlockBuilder::new(MAPLE_TREE)
            .add_block_group(FIBROUS)
            .set_cube_appearance(CubeAppearanceBuilder::new().set_individual_textures(
                MAPLE_TREE_SIDE_TEX,
                MAPLE_TREE_SIDE_TEX,
                MAPLE_TREE_TOP_TEX,
                MAPLE_TREE_TOP_TEX,
                MAPLE_TREE_SIDE_TEX,
                MAPLE_TREE_SIDE_TEX,
            )),
    )?;
    builder.add_block(
        BlockBuilder::new(TALL_GRASS)
            .set_plant_like_appearance(
                PlantLikeAppearanceBuilder::new().set_texture(TALL_GRASS_TEX),
            )
            .set_display_name("Tall grass")
            .set_inventory_texture(TALL_GRASS_TEX)
            .set_allow_light_propagation(true)
            .set_no_drops(),
    )?;
    builder.register_smelting_fuel(MAPLE_TREE.0, 8);
    builder.register_smelting_fuel(MAPLE_TREE.0, 1);
    builder.add_block(
        BlockBuilder::new(MAPLE_LEAVES)
            .add_block_group(FIBROUS)
            .set_cube_appearance(
                CubeAppearanceBuilder::new()
                    .set_single_texture(MAPLE_LEAVES_TEX)
                    .set_needs_transparency(),
            )
            .set_allow_light_propagation(true),
    )?;
    builder.add_block(
        BlockBuilder::new(CACTUS)
            .add_block_group(FIBROUS)
            .set_cube_appearance(CubeAppearanceBuilder::new().set_individual_textures(
                CACTUS_SIDE_TEX,
                CACTUS_SIDE_TEX,
                CACTUS_TOP_TEX,
                CACTUS_TOP_TEX,
                CACTUS_SIDE_TEX,
                CACTUS_SIDE_TEX,
            )),
    )?;
    Ok(())
}
