use anyhow::Result;
use perovskite_server::game_state::items::ItemStack;

use crate::{
    blocks::{BlockBuilder, CubeAppearanceBuilder, PlantLikeAppearanceBuilder},
    game_builder::{StaticBlockName, TextureName},
    include_texture_bytes,
};

use super::{
    block_groups::FIBROUS,
    recipes::{self, RecipeSlot},
    DefaultGameBuilder,
};

pub const MAPLE_TREE: StaticBlockName = StaticBlockName("default:maple_tree");
pub const MAPLE_TREE_TOP_TEX: TextureName = TextureName("default:maple_tree_top");
pub const MAPLE_TREE_SIDE_TEX: TextureName = TextureName("default:maple_tree_side");

pub const MAPLE_LEAVES: StaticBlockName = StaticBlockName("default:maple_leaves");
pub const MAPLE_LEAVES_TEX: TextureName = TextureName("default:maple_leaves");

pub const TALL_GRASS: StaticBlockName = StaticBlockName("default:tall_grass");
pub const TALL_GRASS_TEX: TextureName = TextureName("default:tall_grass");

pub(crate) fn register_foliage(builder: &mut DefaultGameBuilder) -> Result<()> {
    include_texture_bytes!(
        builder.inner,
        MAPLE_TREE_TOP_TEX,
        "textures/maple_tree_top.png"
    )?;
    include_texture_bytes!(
        builder.inner,
        MAPLE_TREE_SIDE_TEX,
        "textures/maple_tree_side.png"
    )?;
    include_texture_bytes!(builder.inner, MAPLE_LEAVES_TEX, "textures/maple_leaves.png")?;
    include_texture_bytes!(builder.inner, TALL_GRASS_TEX, "textures/tall_grass.png")?;
    builder.game_builder().add_block(
        BlockBuilder::new(MAPLE_TREE)
            .add_block_group(FIBROUS)
            .set_cube_appearance(CubeAppearanceBuilder::new().set_individual_textures(
                MAPLE_TREE_SIDE_TEX,
                MAPLE_TREE_SIDE_TEX,
                MAPLE_TREE_TOP_TEX,
                MAPLE_TREE_TOP_TEX,
                MAPLE_TREE_SIDE_TEX,
                MAPLE_TREE_SIDE_TEX,
            ))
            .set_inventory_texture(MAPLE_TREE_TOP_TEX),
    )?;
    builder.game_builder().add_block(
        BlockBuilder::new(TALL_GRASS)
            .set_plant_like_appearance(
                PlantLikeAppearanceBuilder::new().set_texture(TALL_GRASS_TEX),
            )
            .set_display_name("Tall grass")
            .set_inventory_texture(TALL_GRASS_TEX)
            .set_allow_light_propagation(true)
            .set_no_drops(),
    )?;
    builder.smelting_fuels.register_recipe(recipes::Recipe {
        slots: [RecipeSlot::Exact(MAPLE_TREE.0.to_string())],
        result: ItemStack {
            proto: Default::default(),
        },
        shapeless: false,
        metadata: 8,
    });
    builder.smelting_fuels.register_recipe(recipes::Recipe {
        slots: [RecipeSlot::Exact(MAPLE_LEAVES.0.to_string())],
        result: ItemStack {
            proto: Default::default(),
        },
        shapeless: false,
        metadata: 1,
    });
    BlockBuilder::new(MAPLE_LEAVES)
        .add_block_group(FIBROUS)
        .set_cube_appearance(
            CubeAppearanceBuilder::new()
                .set_single_texture(MAPLE_LEAVES_TEX)
                .set_needs_transparency(),
        )
        .set_allow_light_propagation(true)
        .set_inventory_texture(MAPLE_LEAVES_TEX)
        .build_and_deploy_into(builder.game_builder())?;
    Ok(())
}