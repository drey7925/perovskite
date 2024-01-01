use anyhow::Result;
use perovskite_core::protocol::{
    blocks::{block_type_def::PhysicsInfo, Empty},
    items::item_stack::QuantityType,
};
use perovskite_server::game_state::{blocks::BlockType, items::ItemStack};

use crate::{
    blocks::{BlockBuilder, CubeAppearanceBuilder, PlantLikeAppearanceBuilder},
    game_builder::{GameBuilder, StaticBlockName, StaticItemName, StaticTextureName, TextureName},
    include_texture_bytes,
};

use super::{block_groups, item_groups, recipes::RecipeSlot, shaped_blocks, DefaultGameBuilder};

pub mod foliage_groups {
    pub const MAPLE: &str = "maple";
    pub const FLOWERS: &str = "flowers";
}

pub const MAPLE_TREE: StaticBlockName = StaticBlockName("default:maple_tree");
pub const MAPLE_TREE_TOP_TEX: StaticTextureName = StaticTextureName("default:maple_tree_top");
pub const MAPLE_TREE_SIDE_TEX: StaticTextureName = StaticTextureName("default:maple_tree_side");

pub const MAPLE_LEAVES: StaticBlockName = StaticBlockName("default:maple_leaves");
pub const MAPLE_LEAVES_TEX: StaticTextureName = StaticTextureName("default:maple_leaves");

pub const MAPLE_PLANKS: StaticBlockName = StaticBlockName("default:maple_planks");
pub const MAPLE_PLANKS_TEX: StaticTextureName = StaticTextureName("default:maple_planks");

pub const TALL_GRASS: StaticBlockName = StaticBlockName("default:tall_grass");
pub const TALL_GRASS_TEX: StaticTextureName = StaticTextureName("default:tall_grass");

pub const CACTUS: StaticBlockName = StaticBlockName("default:cactus");
pub const CACTUS_TOP_TEX: StaticTextureName = StaticTextureName("default:cactus_top");
pub const CACTUS_SIDE_TEX: StaticTextureName = StaticTextureName("default:cactus_side");

pub const STICK_ITEM: StaticItemName = StaticItemName("default:stick");
pub const STICK_TEX: StaticTextureName = StaticTextureName("default:stick");

pub(crate) fn register_foliage(builder: &mut GameBuilder) -> Result<()> {
    include_texture_bytes!(builder, MAPLE_TREE_TOP_TEX, "textures/maple_tree_top.png")?;
    include_texture_bytes!(builder, MAPLE_TREE_SIDE_TEX, "textures/maple_tree_side.png")?;
    include_texture_bytes!(builder, MAPLE_LEAVES_TEX, "textures/maple_leaves.png")?;
    include_texture_bytes!(builder, MAPLE_PLANKS_TEX, "textures/maple_planks.png")?;
    include_texture_bytes!(builder, TALL_GRASS_TEX, "textures/tall_grass.png")?;
    include_texture_bytes!(builder, CACTUS_TOP_TEX, "textures/cactus_top.png")?;
    include_texture_bytes!(builder, CACTUS_SIDE_TEX, "textures/cactus_side.png")?;
    include_texture_bytes!(builder, STICK_TEX, "textures/stick.png")?;
    let maple_trunk = builder.add_block(
        BlockBuilder::new(MAPLE_TREE)
            .add_block_group(block_groups::TREE_TRUNK)
            .add_block_group(block_groups::FIBROUS)
            .add_block_group(foliage_groups::MAPLE)
            .add_item_group(item_groups::TREE_TRUNK)
            .add_item_group(foliage_groups::MAPLE)
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
    builder.register_smelting_fuel(RecipeSlot::Group(item_groups::TREE_TRUNK.to_string()), 8);
    builder.register_smelting_fuel(RecipeSlot::Group(item_groups::TREE_LEAVES.to_string()), 1);

    builder.register_smelting_fuel(RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()), 2);

    builder.register_basic_item(STICK_ITEM, "Wooden stick", STICK_TEX, vec![])?;

    let maple_planks = builder.add_block(
        BlockBuilder::new(MAPLE_PLANKS)
            .add_block_group(block_groups::WOOD_PLANKS)
            .add_block_group(block_groups::FIBROUS)
            .add_block_group(foliage_groups::MAPLE)
            .add_item_group(item_groups::WOOD_PLANKS)
            .add_item_group(foliage_groups::MAPLE)
            .set_cube_appearance(CubeAppearanceBuilder::new().set_single_texture(MAPLE_PLANKS_TEX))
            .set_display_name("Maple planks"),
    )?;

    builder.register_crafting_recipe(
        [
            RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
            RecipeSlot::Empty,
            RecipeSlot::Empty,
            RecipeSlot::Empty,
            RecipeSlot::Empty,
            RecipeSlot::Empty,
            RecipeSlot::Empty,
            RecipeSlot::Empty,
            RecipeSlot::Empty,
        ],
        STICK_ITEM.0.to_string(),
        4,
        Some(QuantityType::Stack(256)),
        true,
    );

    builder.register_crafting_recipe(
        [
            RecipeSlot::Exact(MAPLE_TREE.0.to_string()),
            RecipeSlot::Empty,
            RecipeSlot::Empty,
            RecipeSlot::Empty,
            RecipeSlot::Empty,
            RecipeSlot::Empty,
            RecipeSlot::Empty,
            RecipeSlot::Empty,
            RecipeSlot::Empty,
        ],
        MAPLE_PLANKS.0.to_string(),
        4,
        Some(QuantityType::Stack(256)),
        true,
    );

    builder.add_block(
        BlockBuilder::new(MAPLE_LEAVES)
            .add_block_group(block_groups::FIBROUS)
            .add_block_group(block_groups::TREE_LEAVES)
            .add_item_group(item_groups::TREE_LEAVES)
            .set_cube_appearance(
                CubeAppearanceBuilder::new()
                    .set_single_texture(MAPLE_LEAVES_TEX)
                    .set_needs_transparency(),
            )
            .set_allow_light_propagation(true),
    )?;
    builder.add_block(
        BlockBuilder::new(CACTUS)
            .add_block_group(block_groups::FIBROUS)
            .set_cube_appearance(CubeAppearanceBuilder::new().set_individual_textures(
                CACTUS_SIDE_TEX,
                CACTUS_SIDE_TEX,
                CACTUS_TOP_TEX,
                CACTUS_TOP_TEX,
                CACTUS_SIDE_TEX,
                CACTUS_SIDE_TEX,
            )),
    )?;
    shaped_blocks::make_slab(builder, &maple_planks, true)?;
    shaped_blocks::make_stairs(builder, &maple_planks, true)?;
    shaped_blocks::make_slab(builder, &maple_trunk, true)?;
    shaped_blocks::make_stairs(builder, &maple_trunk, true)?;

    register_flowers(builder)?;

    Ok(())
}

fn register_flowers(builder: &mut GameBuilder) -> Result<()> {
    for &(block, texture, display_name, color, tex_bytes) in TERRESTRIAL_FLOWERS
        .iter()
        .chain([WHITE_LOTUS_DEF, PURPLE_LOTUS_DEF].iter())
    {
        builder.register_texture_bytes(texture, tex_bytes)?;

        builder.add_block(
            BlockBuilder::new(block)
                .set_plant_like_appearance(PlantLikeAppearanceBuilder::new().set_texture(texture))
                .set_display_name(display_name)
                .set_inventory_texture(texture)
                .set_allow_light_propagation(true)
                .add_modifier(Box::new(|block: &mut BlockType| {
                    block.client_info.physics_info = Some(PhysicsInfo::Air(Empty {}));
                })),
        )?;
        builder.register_crafting_recipe(
            [
                RecipeSlot::Exact(block.0.to_string()),
                RecipeSlot::Empty,
                RecipeSlot::Empty,
                RecipeSlot::Empty,
                RecipeSlot::Empty,
                RecipeSlot::Empty,
                RecipeSlot::Empty,
                RecipeSlot::Empty,
                RecipeSlot::Empty,
            ],
            color.dye_item_name(),
            1,
            Some(QuantityType::Stack(256)),
            true,
        );
    }
    Ok(())
}

type FlowerDef = (
    StaticBlockName,
    StaticTextureName,
    &'static str,
    crate::colors::Color,
    &'static [u8],
);

pub(crate) const WHITE_LOTUS_DEF: FlowerDef = (
    StaticBlockName("default:white_lotus"),
    StaticTextureName("default:white_lotus"),
    "White lotus",
    crate::colors::Color::White,
    include_bytes!("textures/tall_grass.png"),
);
pub(crate) const PURPLE_LOTUS_DEF: FlowerDef = (
    StaticBlockName("default:purple_lotus"),
    StaticTextureName("default:purple_lotus"),
    "Purple lotus",
    crate::colors::Color::Purple,
    include_bytes!("textures/tall_grass.png"),
);

pub(crate) const TERRESTRIAL_FLOWERS: &[FlowerDef] = &[
    (
        StaticBlockName("default:rose"),
        StaticTextureName("default:rose"),
        "Rose",
        crate::colors::Color::Pink,
        include_bytes!("textures/flowers/rose.png"),
    ),
    (
        StaticBlockName("default:carnation"),
        StaticTextureName("default:carnation"),
        "Carnation",
        crate::colors::Color::Red,
        include_bytes!("textures/flowers/carnation.png"),
    ),
    (
        StaticBlockName("default:dandelion"),
        StaticTextureName("default:dandelion"),
        "Dandelion",
        crate::colors::Color::Yellow,
        include_bytes!("textures/flowers/dandelion.png"),
    ),
    (
        StaticBlockName("default:calendula"),
        StaticTextureName("default:calendula"),
        "Calendula",
        crate::colors::Color::Orange,
        include_bytes!("textures/flowers/calendula.png"),
    ),
    (
        StaticBlockName("default:cornflower"),
        StaticTextureName("default:cornflower"),
        "Cornflower",
        crate::colors::Color::Blue,
        include_bytes!("textures/flowers/cornflower.png"),
    ),
    (
        StaticBlockName("default:lavender"),
        StaticTextureName("default:lavender"),
        "Lavender",
        crate::colors::Color::Purple,
        include_bytes!("textures/flowers/lavender.png"),
    ),
];
