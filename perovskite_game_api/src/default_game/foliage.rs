//! Foliage and plant products. Texture names are public to allow other plugins to derive textures
//! and use them in their own block definitions

use anyhow::Result;
use perovskite_core::protocol::{
    blocks::{block_type_def::PhysicsInfo, Empty},
    items::item_stack::QuantityType,
};
use perovskite_server::game_state::blocks::BlockType;

use crate::{
    blocks::{BlockBuilder, CubeAppearanceBuilder, PlantLikeAppearanceBuilder},
    game_builder::{GameBuilder, StaticBlockName, StaticItemName, StaticTextureName},
    include_texture_bytes,
};

use super::{block_groups, item_groups, recipes::RecipeSlot, shaped_blocks, DefaultGameBuilder};

pub mod foliage_groups {
    pub const MAPLE: &str = "maple";
    pub const PINE: &str = "pine";
    pub const FLOWERS: &str = "flowers";
}

pub const MAPLE_TREE: StaticBlockName = StaticBlockName("default:maple_tree");
pub const MAPLE_TREE_TOP_TEX: StaticTextureName = StaticTextureName("default:maple_tree_top");
pub const MAPLE_TREE_SIDE_TEX: StaticTextureName = StaticTextureName("default:maple_tree_side");

pub const MAPLE_LEAVES: StaticBlockName = StaticBlockName("default:maple_leaves");
pub const MAPLE_LEAVES_TEX: StaticTextureName = StaticTextureName("default:maple_leaves");

pub const MAPLE_PLANKS: StaticBlockName = StaticBlockName("default:maple_planks");
pub const MAPLE_PLANKS_TEX: StaticTextureName = StaticTextureName("default:maple_planks");

pub const PINE_TREE: StaticBlockName = StaticBlockName("default:pine_tree");
pub const PINE_TREE_TOP_TEX: StaticTextureName = StaticTextureName("default:pine_tree_top");
pub const PINE_TREE_SIDE_TEX: StaticTextureName = StaticTextureName("default:pine_tree_side");

pub const PINE_NEEDLES: StaticBlockName = StaticBlockName("default:pine_needles");
pub const PINE_NEEDLES_TEX: StaticTextureName = StaticTextureName("default:pine_needles");

pub const PINE_PLANKS: StaticBlockName = StaticBlockName("default:pine_planks");
pub const PINE_PLANKS_TEX: StaticTextureName = StaticTextureName("default:pine_planks");

pub const TALL_GRASS: StaticBlockName = StaticBlockName("default:tall_grass");
pub const TALL_GRASS_TEX: StaticTextureName = StaticTextureName("default:tall_grass");

pub const MARSH_GRASS: StaticBlockName = StaticBlockName("default:marsh_grass");
pub const MARSH_GRASS_TEX: StaticTextureName = StaticTextureName("default:marsh_grass_tex");

pub const TALL_REED: StaticBlockName = StaticBlockName("default:tall_reed");
pub const TALL_REED_TEX: StaticTextureName = StaticTextureName("default:tall_reed");

pub const CACTUS: StaticBlockName = StaticBlockName("default:cactus");
pub const CACTUS_TOP_TEX: StaticTextureName = StaticTextureName("default:cactus_top");
pub const CACTUS_SIDE_TEX: StaticTextureName = StaticTextureName("default:cactus_side");

pub const STICK_ITEM: StaticItemName = StaticItemName("default:stick");
pub const STICK_TEX: StaticTextureName = StaticTextureName("default:stick");

struct TreeDef {
    name: &'static str,
    trunk: StaticBlockName,
    trunk_top_tex: StaticTextureName,
    trunk_side_tex: StaticTextureName,
    planks: StaticBlockName,
    planks_tex: StaticTextureName,
    leaves: Option<(StaticBlockName, StaticTextureName)>,
    group: &'static str,
}

fn register_tree(builder: &mut GameBuilder, tree: &TreeDef) -> Result<()> {
    let trunk = builder.add_block(
        BlockBuilder::new(tree.trunk)
            .add_block_group(block_groups::TREE_TRUNK)
            .add_block_group(block_groups::FIBROUS)
            .add_block_group(tree.group)
            .add_item_group(item_groups::TREE_TRUNK)
            .add_item_group(tree.group)
            .set_item_sort_key(format!("default:trees:tree_trunk:{}", tree.name))
            .set_cube_appearance(CubeAppearanceBuilder::new().set_individual_textures(
                tree.trunk_side_tex,
                tree.trunk_side_tex,
                tree.trunk_top_tex,
                tree.trunk_top_tex,
                tree.trunk_side_tex,
                tree.trunk_side_tex,
            )),
    )?;

    let planks = builder.add_block(
        BlockBuilder::new(tree.planks)
            .add_block_group(block_groups::WOOD_PLANKS)
            .add_block_group(block_groups::FIBROUS)
            .add_block_group(tree.group)
            .add_item_group(item_groups::WOOD_PLANKS)
            .add_item_group(tree.group)
            .set_cube_appearance(CubeAppearanceBuilder::new().set_single_texture(tree.planks_tex))
            .set_item_sort_key(format!("default:trees:wood_planks:{}", tree.name))
            .set_display_name(format!(
                "{} planks",
                tree.name
                    .chars()
                    .next()
                    .unwrap()
                    .to_uppercase()
                    .collect::<String>()
                    + &tree.name[1..]
            )),
    )?;

    builder.register_crafting_recipe(
        [
            RecipeSlot::Exact(tree.trunk.0.to_string()),
            RecipeSlot::Empty,
            RecipeSlot::Empty,
            RecipeSlot::Empty,
            RecipeSlot::Empty,
            RecipeSlot::Empty,
            RecipeSlot::Empty,
            RecipeSlot::Empty,
            RecipeSlot::Empty,
        ],
        tree.planks.0.to_string(),
        4,
        Some(QuantityType::Stack(256)),
        true,
    );

    // Register leaves if provided
    if let Some((leaves_name, leaves_tex)) = tree.leaves {
        builder.add_block(
            BlockBuilder::new(leaves_name)
                .add_block_group(block_groups::FIBROUS)
                .add_block_group(block_groups::TREE_LEAVES)
                .add_item_group(item_groups::TREE_LEAVES)
                .set_cube_appearance(
                    CubeAppearanceBuilder::new()
                        .set_single_texture(leaves_tex)
                        .set_needs_transparency(),
                )
                .set_item_sort_key(format!("default:trees:tree_leaves:{}", tree.name))
                .set_allow_light_propagation(true),
        )?;
    }

    shaped_blocks::make_slab(builder, &planks, true)?;
    shaped_blocks::make_stairs(builder, &planks, true)?;
    shaped_blocks::make_slab(builder, &trunk, true)?;
    shaped_blocks::make_stairs(builder, &trunk, true)?;

    Ok(())
}

pub(crate) fn register_foliage(builder: &mut GameBuilder) -> Result<()> {
    // Register textures
    include_texture_bytes!(builder, MAPLE_TREE_TOP_TEX, "textures/maple_tree_top.png")?;
    include_texture_bytes!(builder, MAPLE_TREE_SIDE_TEX, "textures/maple_tree_side.png")?;
    include_texture_bytes!(builder, MAPLE_LEAVES_TEX, "textures/maple_leaves.png")?;
    include_texture_bytes!(builder, MAPLE_PLANKS_TEX, "textures/maple_planks.png")?;
    include_texture_bytes!(builder, PINE_TREE_TOP_TEX, "textures/pine_tree_top.png")?;
    include_texture_bytes!(builder, PINE_TREE_SIDE_TEX, "textures/pine_tree_side.png")?;
    include_texture_bytes!(builder, PINE_PLANKS_TEX, "textures/pine_planks.png")?;
    include_texture_bytes!(builder, PINE_NEEDLES_TEX, "textures/pine_needles.png")?;
    include_texture_bytes!(builder, TALL_GRASS_TEX, "textures/tall_grass.png")?;
    include_texture_bytes!(builder, MARSH_GRASS_TEX, "textures/marsh_grass.png")?;
    include_texture_bytes!(builder, TALL_REED_TEX, "textures/tall_reed.png")?;
    include_texture_bytes!(builder, CACTUS_TOP_TEX, "textures/cactus_top.png")?;
    include_texture_bytes!(builder, CACTUS_SIDE_TEX, "textures/cactus_side.png")?;
    include_texture_bytes!(builder, STICK_TEX, "textures/stick.png")?;

    // Register trees
    let maple_tree = TreeDef {
        name: "maple",
        trunk: MAPLE_TREE,
        trunk_top_tex: MAPLE_TREE_TOP_TEX,
        trunk_side_tex: MAPLE_TREE_SIDE_TEX,
        planks: MAPLE_PLANKS,
        planks_tex: MAPLE_PLANKS_TEX,
        leaves: Some((MAPLE_LEAVES, MAPLE_LEAVES_TEX)),
        group: foliage_groups::MAPLE,
    };

    let pine_tree = TreeDef {
        name: "pine",
        trunk: PINE_TREE,
        trunk_top_tex: PINE_TREE_TOP_TEX,
        trunk_side_tex: PINE_TREE_SIDE_TEX,
        planks: PINE_PLANKS,
        planks_tex: PINE_PLANKS_TEX,
        leaves: Some((PINE_NEEDLES, PINE_NEEDLES_TEX)),
        group: foliage_groups::PINE,
    };

    register_tree(builder, &maple_tree)?;
    register_tree(builder, &pine_tree)?;

    // Register other foliage items
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

    builder.add_block(
        BlockBuilder::new(MARSH_GRASS)
            .set_plant_like_appearance(
                PlantLikeAppearanceBuilder::new().set_texture(MARSH_GRASS_TEX),
            )
            .set_display_name("Marsh grass")
            .set_inventory_texture(MARSH_GRASS_TEX)
            .set_allow_light_propagation(true)
            .set_no_drops(),
    )?;

    builder.add_block(
        BlockBuilder::new(TALL_REED)
            .set_plant_like_appearance(
                PlantLikeAppearanceBuilder::new()
                    .set_texture(TALL_REED_TEX)
                    .set_wave_effect_scale(0.0)
                    .set_is_solid(true),
            )
            .set_display_name("Marsh grass")
            .set_inventory_texture(TALL_REED_TEX)
            .set_allow_light_propagation(true)
            .set_no_drops(),
    )?;

    builder.register_smelting_fuel(RecipeSlot::Group(item_groups::TREE_TRUNK.to_string()), 8);
    builder.register_smelting_fuel(RecipeSlot::Group(item_groups::TREE_LEAVES.to_string()), 1);
    builder.register_smelting_fuel(RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()), 2);

    builder.register_basic_item(
        STICK_ITEM,
        "Wooden stick",
        STICK_TEX,
        vec![],
        "default:stick",
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
                .add_block_group(foliage_groups::FLOWERS)
                .set_allow_light_propagation(true)
                .add_modifier(Box::new(|block: &mut BlockType| {
                    block.client_info.physics_info = Some(PhysicsInfo::Air(Empty {}));
                }))
                .set_item_sort_key(format!("default:flowers:{}", block.0)),
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
