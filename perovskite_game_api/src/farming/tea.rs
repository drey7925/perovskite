//! A module allowing players to grow, process, and enjoy various varieties of tea. Eventually, a
//! possible testbed for additional item functionality (e.g. in teapots and rare varieties of leaves
//! obtained by chance)
//!
//! Roadmap:
//! * [ ] Basic tea growth
//! * [ ] Basic tea processing (e.g., withering, oxidation, fixing, drying)
//! * [ ] ...

use crate::blocks::{
    AaBoxProperties, AxisAlignedBoxesAppearanceBuilder, BlockAppearanceBuilder, BlockBuilder,
    PlantLikeAppearanceBuilder, RotationMode, TextureCropping,
};
use crate::default_game::basic_blocks::{DIRT, DIRT_WITH_GRASS};
use crate::default_game::foliage::{MAPLE_PLANKS_TEX, STICK_ITEM};
use crate::default_game::recipes::RecipeSlot;
use crate::default_game::DefaultGameBuilder;
use crate::farming::crops::{
    define_crop, ConstantGrowProbability, CropDefinition, DefaultGrowInLight, GrowthStage,
    InteractionAccumulator, InteractionEffect, InteractionTransition, InteractionTransitionTarget,
    NeverGrow,
};
use crate::farming::FarmingGameStateExtension;
use crate::game_builder::{
    BlockName, GameBuilder, OwnedTextureName, StaticBlockName, StaticItemName, StaticTextureName,
    FALLBACK_UNKNOWN_TEXTURE_NAME,
};
use crate::include_texture_bytes;
use crate::items::{ItemAction, ItemActionTarget, ItemBuilder, ItemHandler, StackDecrement};
use anyhow::Result;
use perovskite_core::protocol::items::item_stack::QuantityType;
use perovskite_core::protocol::render::TextureReference;
use perovskite_server::game_state::blocks::FastBlockName;
use std::ops::RangeInclusive;
use std::time::Duration;

/// Fresh tea leaves from the plant
const TEA_LEAVES_FRESH_ITEM: StaticItemName = StaticItemName("farming:tea_leaves_fresh");
/// Heat-fixed tea leaves, could be rolled and dried to make green tea
const TEA_LEAVES_STEAMED_ITEM: StaticItemName = StaticItemName("farming:tea_leaves_steamed");
/// Set out on a basket, withered, and can now be bruised to continue to the next stage
const TEA_LEAVES_WITHERED_ITEM: StaticItemName = StaticItemName("farming:tea_leaves_withered");
/// Withered leaves that were mechanically agitated to damage cell walls
const TEA_LEAVES_WITHERED_BRUISED: StaticItemName =
    StaticItemName("farming:tea_leaves_withered_bruised");
/// Partially oxidized tea leaves, still need drying and rolling to become oolong
const TEA_LEAVES_PARTIALLY_OXIDIZED: StaticItemName =
    StaticItemName("farming:tea_leaves_partially_oxidized");
/// Fully oxidized tea leaves, still need drying to become red tea
const TEA_LEAVES_FULLY_OXIDIZED: StaticItemName =
    StaticItemName("farming:tea_leaves_fully_oxidized");

const TEA_PER_BASKET: u32 = 4;

pub(crate) fn register_tea(builder: &mut GameBuilder) -> Result<()> {
    ItemBuilder::new(TEA_LEAVES_FRESH_ITEM)
        .set_display_name("Fresh tea leaves")
        .set_max_stack(256)
        .set_sort_key("farming:tea:leaves:00fresh")
        .add_right_click_handler(ItemHandler {
            target: ItemActionTarget::Block(FastBlockName::from(TEA_BASKET_EMPTY)),
            action: ItemAction::PlaceBlock {
                block: FastBlockName::from(TEA_BASKET_WITHERING),
                rotation_mode: RotationMode::None,
                ignore_trivially_replaceable: true,
                place_onto: false,
            },
            stack_decrement: Some(StackDecrement::FixedRequireSufficient(TEA_PER_BASKET)),
            ..Default::default()
        })
        .add_default_dig_handler()
        .build_into(builder)?;

    let steamed_leaves = builder.register_basic_item(
        TEA_LEAVES_STEAMED_ITEM,
        "Steamed tea leaves",
        FALLBACK_UNKNOWN_TEXTURE_NAME,
        vec![],
        "farming:tea:leaves:01steamed",
    )?;
    let steamed_leaves_stack = steamed_leaves.make_stack(1);
    // Borrow checker: need to pre-create the stack since `steamed_leaves` borrows from `builder`.
    // (this is ugly, maybe we could get non-borrowing item handles holding string names????)

    // Steaming fresh leaves gets us oxidation-fixed leaves, which then become green tea when rolled
    // and then dried
    builder.register_smelting_recipe(TEA_LEAVES_FRESH_ITEM.into(), steamed_leaves_stack);

    let _withered_leaves = builder.register_basic_item(
        TEA_LEAVES_WITHERED_ITEM,
        "Withered tea leaves",
        FALLBACK_UNKNOWN_TEXTURE_NAME,
        vec![],
        "farming:tea:leaves:02withered",
    )?;
    let _partial_ox_leaves = builder.register_basic_item(
        TEA_LEAVES_PARTIALLY_OXIDIZED,
        "Partially oxidized tea leaves",
        FALLBACK_UNKNOWN_TEXTURE_NAME,
        vec![],
        "farming:tea:leaves:03partial_ox",
    )?;
    let _full_ox_leaves = builder.register_basic_item(
        TEA_LEAVES_FULLY_OXIDIZED,
        "Fully oxidized tea leaves",
        FALLBACK_UNKNOWN_TEXTURE_NAME,
        vec![],
        "farming:tea:leaves:04full_ox",
    )?;
    ItemBuilder::new(TEA_LEAVES_WITHERED_BRUISED)
        .set_display_name("Withered+Bruised tea leaves")
        .set_max_stack(256)
        .set_sort_key("farming:tea:leaves:03bruised")
        .add_right_click_handler(ItemHandler {
            target: ItemActionTarget::Block(FastBlockName::from(TEA_BASKET_EMPTY)),
            action: ItemAction::PlaceBlock {
                block: FastBlockName::from(TEA_BASKET_UNOXIDIZED),
                rotation_mode: RotationMode::None,
                ignore_trivially_replaceable: true,
                place_onto: false,
            },
            stack_decrement: Some(StackDecrement::FixedRequireSufficient(TEA_PER_BASKET)),
            ..Default::default()
        })
        .build_into(builder)?;

    use RecipeSlot::Empty;
    builder.register_crafting_recipe(
        [
            TEA_LEAVES_WITHERED_ITEM.into(),
            TEA_LEAVES_WITHERED_ITEM.into(),
            TEA_LEAVES_WITHERED_ITEM.into(),
            TEA_LEAVES_WITHERED_ITEM.into(),
            Empty,
            Empty,
            Empty,
            Empty,
            Empty,
        ],
        TEA_LEAVES_WITHERED_BRUISED.0.to_string(),
        1,
        Some(QuantityType::Stack(256)),
        true, // shapeless
    );

    // let steamed_rolled_leaves = builder.register_basic_item(
    //     TEA_LEAVES_STEAMED_ITEM,
    // );

    register_tea_plant_stages(builder)?;

    register_tea_basket_stages(builder)?;

    Ok(())
}

/// An empty tea basket
const TEA_BASKET_EMPTY: StaticBlockName = StaticBlockName("farming:tea_basket_empty");
/// Tea which didn't undergo heat-fixation was placed on the basket, and is now starting to wither
const TEA_BASKET_WITHERING: StaticBlockName = StaticBlockName("farming:tea_basket_withering");
/// Some time has passed, and tea has withered. It can now be taken and bruised/rolled.
const TEA_BASKET_WITHERED: StaticBlockName = StaticBlockName("farming:tea_basket_withered");
/// Bruised/rolled tea has been placed back on the basket to oxidize.
const TEA_BASKET_UNOXIDIZED: StaticBlockName = StaticBlockName("farming:tea_basket_unoxidized");
/// Some time has passed, tea is partially oxidized. If harvested now and heat-treated, we get oolong
const TEA_BASKET_PARTIALLY_OXIDIZED: StaticBlockName =
    StaticBlockName("farming:tea_basket_partially_oxidized");
/// More time has passed, tea is fully oxidized. If harvested now and dried, we get black tea
const TEA_BASKET_FULLY_OXIDIZED: StaticBlockName =
    StaticBlockName("farming:tea_basket_fully_oxidized");

fn make_basket_aabbs(
    contents_texture: impl Into<TextureReference>,
    contents_z: f32,
) -> AxisAlignedBoxesAppearanceBuilder {
    let basket_properties = AaBoxProperties::new_single_tex(
        MAPLE_PLANKS_TEX,
        TextureCropping::AutoCrop,
        RotationMode::None,
    );
    let contents_properties = AaBoxProperties::new(
        MAPLE_PLANKS_TEX,
        MAPLE_PLANKS_TEX,
        contents_texture,
        MAPLE_PLANKS_TEX,
        MAPLE_PLANKS_TEX,
        MAPLE_PLANKS_TEX,
        TextureCropping::AutoCrop,
        RotationMode::None,
    );
    AxisAlignedBoxesAppearanceBuilder::new()
        .add_box(
            basket_properties.clone(),
            (-0.5, 0.5),
            (-0.5, -0.25),
            (-0.5, -0.4),
        )
        .add_box(
            basket_properties.clone(),
            (-0.5, 0.5),
            (-0.5, -0.25),
            (0.4, 0.5),
        )
        .add_box(
            basket_properties.clone(),
            (-0.5, -0.4),
            (-0.5, -0.25),
            (-0.5, 0.5),
        )
        .add_box(
            basket_properties.clone(),
            (0.4, 0.5),
            (-0.5, -0.25),
            (-0.5, 0.5),
        )
        .add_box(
            contents_properties,
            (-0.5, 0.5),
            (-0.5, contents_z),
            (-0.5, 0.5),
        )
}

fn register_tea_basket_stages(game_builder: &mut GameBuilder) -> Result<()> {
    let empty_basket = BlockBuilder::new(TEA_BASKET_EMPTY)
        .set_display_name("Empty tea basket")
        .set_axis_aligned_boxes_appearance(make_basket_aabbs(MAPLE_PLANKS_TEX, -0.4))
        .set_allow_light_propagation(true)
        .build_and_deploy_into(game_builder)?;
    use RecipeSlot::Empty;
    game_builder.register_crafting_recipe(
        [
            Empty,
            Empty,
            Empty,
            STICK_ITEM.into(),
            Empty,
            STICK_ITEM.into(),
            Empty,
            Empty,
            Empty,
        ],
        empty_basket.item_name.0.clone(),
        4,
        Some(QuantityType::Stack(256)),
        false,
    );

    const TEA_PER_BASKET_RANGE: RangeInclusive<u32> = TEA_PER_BASKET..=TEA_PER_BASKET;

    let wither_stages = vec![
        GrowthStage {
            tap_effect: Some(InteractionEffect {
                item_drops: vec![(TEA_LEAVES_FRESH_ITEM.into(), TEA_PER_BASKET_RANGE)],
                transition: InteractionTransitionTarget::ChangeBlockType(empty_basket.id).into(),
                transition_probability: 1.0,
                ..Default::default()
            }),
            dig_effect: Some(InteractionEffect {
                item_drops: vec![
                    (empty_basket.item_name.clone(), 1..=1),
                    (TEA_LEAVES_FRESH_ITEM.into(), TEA_PER_BASKET_RANGE),
                ],
                transition: InteractionTransitionTarget::Remove.into(),
                transition_probability: 1.0,
                ..Default::default()
            }),
            block_name: Some(TEA_BASKET_WITHERING.into()),
            grow_probability: Box::new(ConstantGrowProbability(
                0.5,
                InteractionTransition {
                    target: InteractionTransitionTarget::NextStage,
                    accumulator: Some(InteractionAccumulator { add: 1, trip: 10 }),
                },
            )),
            appearance: make_basket_aabbs(OwnedTextureName::from_css_color("#00ff00"), -0.3).into(),
            ..Default::default()
        },
        GrowthStage {
            tap_effect: Some(InteractionEffect {
                item_drops: vec![(TEA_LEAVES_WITHERED_ITEM.into(), TEA_PER_BASKET_RANGE)],
                transition: InteractionTransitionTarget::ChangeBlockType(empty_basket.id).into(),
                transition_probability: 1.0,
                ..Default::default()
            }),
            dig_effect: Some(InteractionEffect {
                item_drops: vec![
                    (empty_basket.item_name.clone(), 1..=1),
                    (TEA_LEAVES_WITHERED_ITEM.into(), TEA_PER_BASKET_RANGE),
                ],
                transition: InteractionTransitionTarget::Remove.into(),
                transition_probability: 1.0,
                ..Default::default()
            }),
            block_name: Some(TEA_BASKET_WITHERED.into()),
            grow_probability: Box::new(NeverGrow),
            appearance: make_basket_aabbs(OwnedTextureName::from_css_color("#44ff00"), -0.3).into(),
            ..Default::default()
        },
    ];
    define_crop(
        game_builder,
        CropDefinition {
            base_name: "default:tea_baskets_wither".to_string(),
            stages: wither_stages,
            eligible_soil_blocks: vec![],
            grow_on_any_block: true,
            timer_period: Duration::from_secs(5),
            ..Default::default()
        },
    )?;

    let oxxidize_stages = vec![
        GrowthStage {
            tap_effect: Some(InteractionEffect {
                item_drops: vec![(TEA_LEAVES_WITHERED_BRUISED.into(), TEA_PER_BASKET_RANGE)],
                transition: InteractionTransitionTarget::ChangeBlockType(empty_basket.id).into(),
                transition_probability: 1.0,
                ..Default::default()
            }),
            dig_effect: Some(InteractionEffect {
                item_drops: vec![
                    (empty_basket.item_name.clone(), 1..=1),
                    (TEA_LEAVES_WITHERED_BRUISED.into(), TEA_PER_BASKET_RANGE),
                ],
                transition: InteractionTransitionTarget::Remove.into(),
                transition_probability: 1.0,
                ..Default::default()
            }),
            block_name: Some(TEA_BASKET_UNOXIDIZED.into()),
            grow_probability: Box::new(ConstantGrowProbability(
                0.5,
                InteractionTransition {
                    target: InteractionTransitionTarget::NextStage,
                    accumulator: Some(InteractionAccumulator { add: 1, trip: 10 }),
                },
            )),
            appearance: make_basket_aabbs(OwnedTextureName::from_css_color("#33cc00"), -0.3).into(),
            ..Default::default()
        },
        GrowthStage {
            tap_effect: Some(InteractionEffect {
                item_drops: vec![(TEA_LEAVES_PARTIALLY_OXIDIZED.into(), TEA_PER_BASKET_RANGE)],
                transition: InteractionTransitionTarget::ChangeBlockType(empty_basket.id).into(),
                transition_probability: 1.0,
                ..Default::default()
            }),
            dig_effect: Some(InteractionEffect {
                item_drops: vec![
                    (empty_basket.item_name.clone(), TEA_PER_BASKET_RANGE),
                    (TEA_LEAVES_PARTIALLY_OXIDIZED.into(), 1..=1),
                ],
                transition: InteractionTransitionTarget::Remove.into(),
                transition_probability: 1.0,
                ..Default::default()
            }),
            block_name: Some(TEA_BASKET_PARTIALLY_OXIDIZED.into()),
            grow_probability: Box::new(ConstantGrowProbability(
                0.5,
                InteractionTransition {
                    target: InteractionTransitionTarget::NextStage,
                    accumulator: Some(InteractionAccumulator { add: 1, trip: 10 }),
                },
            )),
            appearance: make_basket_aabbs(OwnedTextureName::from_css_color("#448000"), -0.3).into(),
            ..Default::default()
        },
        GrowthStage {
            tap_effect: Some(InteractionEffect {
                item_drops: vec![(TEA_LEAVES_FULLY_OXIDIZED.into(), TEA_PER_BASKET_RANGE)],
                transition: InteractionTransitionTarget::ChangeBlockType(empty_basket.id).into(),
                transition_probability: 1.0,
                ..Default::default()
            }),
            dig_effect: Some(InteractionEffect {
                item_drops: vec![
                    (empty_basket.item_name.clone(), 1..=1),
                    (TEA_LEAVES_FULLY_OXIDIZED.into(), TEA_PER_BASKET_RANGE),
                ],
                transition: InteractionTransitionTarget::Remove.into(),
                transition_probability: 1.0,
                ..Default::default()
            }),
            block_name: Some(TEA_BASKET_FULLY_OXIDIZED.into()),
            grow_probability: Box::new(NeverGrow),
            appearance: make_basket_aabbs(OwnedTextureName::from_css_color("#804400"), -0.3).into(),
            ..Default::default()
        },
    ];
    define_crop(
        game_builder,
        CropDefinition {
            base_name: "default:tea_baskets_oxidize".to_string(),
            stages: oxxidize_stages,
            eligible_soil_blocks: vec![],
            grow_on_any_block: true,
            timer_period: Duration::from_secs(5),
            ..Default::default()
        },
    )?;

    Ok(())
}

fn register_tea_plant_stages(builder: &mut GameBuilder) -> Result<()> {
    const TEX_SPROUT: StaticTextureName = StaticTextureName("farming:tea_sprout");
    const TEX_SEEDLING: StaticTextureName = StaticTextureName("farming:tea_seedling");
    const TEX_NO_LEAVES: StaticTextureName = StaticTextureName("farming:tea_plant_no_leaves");
    const TEX_SPARSE_LEAVES: StaticTextureName =
        StaticTextureName("farming:tea_plant_sparse_leaves");
    const TEX_DENSE_LEAVES: StaticTextureName = StaticTextureName("farming:tea_plant_dense_leaves");

    include_texture_bytes!(builder, TEX_SPROUT, "textures/tea_sprout.png")?;
    include_texture_bytes!(builder, TEX_SEEDLING, "textures/tea_seedling.png")?;
    include_texture_bytes!(builder, TEX_NO_LEAVES, "textures/tea_plant_no_leaves.png")?;
    include_texture_bytes!(
        builder,
        TEX_SPARSE_LEAVES,
        "textures/tea_plant_sparse_leaves.png"
    )?;
    include_texture_bytes!(
        builder,
        TEX_DENSE_LEAVES,
        "textures/tea_plant_dense_leaves.png"
    )?;

    let stages = vec![
        GrowthStage {
            dig_effect: Some(InteractionEffect {
                item_drops: vec![
                    // TODO: tea tree sapling?
                ],
                transition: InteractionTransitionTarget::Remove.into(),
                transition_probability: 1.0,
                ..Default::default()
            }),
            block_name: Some(BlockName::from("farming:tea_sprout")),
            grow_probability: Box::new(DefaultGrowInLight(InteractionTransition {
                target: InteractionTransitionTarget::NextStage,
                accumulator: Some(InteractionAccumulator { add: 1, trip: 20 }),
            })),
            appearance: BlockAppearanceBuilder::Plantlike(PlantLikeAppearanceBuilder::from_tex(
                TEX_SPROUT,
            )),
            ..Default::default()
        },
        GrowthStage {
            dig_effect: Some(InteractionEffect {
                item_drops: vec![
                    // TODO: tea tree sapling?
                ],
                transition: InteractionTransitionTarget::Remove.into(),
                transition_probability: 1.0,
                ..Default::default()
            }),
            block_name: Some(BlockName::from("farming:tea_seedling")),
            grow_probability: Box::new(DefaultGrowInLight(InteractionTransition {
                target: InteractionTransitionTarget::NextStage,
                accumulator: Some(InteractionAccumulator { add: 1, trip: 20 }),
            })),
            appearance: BlockAppearanceBuilder::Plantlike(PlantLikeAppearanceBuilder::from_tex(
                TEX_SEEDLING,
            )),
            ..Default::default()
        },
        GrowthStage {
            dig_effect: Some(InteractionEffect {
                item_drops: vec![
                    // TODO: tea tree sapling?
                ],
                transition: InteractionTransitionTarget::Remove.into(),
                transition_probability: 1.0,
                ..Default::default()
            }),
            block_name: Some(BlockName::from("farming:tea_plant_no_leaves")),
            grow_probability: Box::new(DefaultGrowInLight(InteractionTransition {
                target: InteractionTransitionTarget::NextStage,
                accumulator: Some(InteractionAccumulator { add: 1, trip: 20 }),
            })),
            appearance: BlockAppearanceBuilder::Plantlike(PlantLikeAppearanceBuilder::from_tex(
                TEX_NO_LEAVES,
            )),
            ..Default::default()
        },
        GrowthStage {
            dig_effect: Some(InteractionEffect {
                item_drops: vec![
                    // TODO: tea tree sapling?
                ],
                transition: InteractionTransitionTarget::Remove.into(),
                transition_probability: 1.0,
                ..Default::default()
            }),
            tap_effect: Some(InteractionEffect {
                item_drops: vec![(TEA_LEAVES_FRESH_ITEM.into(), 1..=3)],
                transition: InteractionTransitionTarget::JumpToStage(2).into(),
                transition_probability: 1.0,
                ..Default::default()
            }),
            block_name: Some(BlockName::from("farming:tea_plant_sparse_leaves")),
            grow_probability: Box::new(DefaultGrowInLight(InteractionTransition {
                target: InteractionTransitionTarget::NextStage,
                accumulator: Some(InteractionAccumulator { add: 1, trip: 20 }),
            })),
            appearance: BlockAppearanceBuilder::Plantlike(PlantLikeAppearanceBuilder::from_tex(
                TEX_SPARSE_LEAVES,
            )),
            ..Default::default()
        },
        GrowthStage {
            dig_effect: Some(InteractionEffect {
                item_drops: vec![
                    // TODO: tea tree sapling?
                ],
                transition: InteractionTransitionTarget::Remove.into(),
                transition_probability: 1.0,
                ..Default::default()
            }),
            tap_effect: Some(InteractionEffect {
                item_drops: vec![(TEA_LEAVES_FRESH_ITEM.into(), 4..=6)],
                transition: InteractionTransitionTarget::JumpToStage(2).into(),
                transition_probability: 1.0,
                ..Default::default()
            }),
            block_name: Some(BlockName::from("farming:tea_plant_dense_leaves")),
            grow_probability: Box::new(NeverGrow),
            appearance: BlockAppearanceBuilder::Plantlike(PlantLikeAppearanceBuilder::from_tex(
                TEX_DENSE_LEAVES,
            )),
            ..Default::default()
        },
    ];
    let ext = builder.builder_extension_mut::<FarmingGameStateExtension>();
    let soil_blocks = vec![
        ext.soil_dry.clone(),
        ext.soil_wet.clone(),
        // Tea will grow on unkept dirt, or dirt-with-grass
        FastBlockName::from(DIRT),
        FastBlockName::from(DIRT_WITH_GRASS),
        // We don't allow growing tea on snow. However, there's no temperature check - players could
        // grow a shan tuyết on a snowy mountain as long as they prepare dirt without snow or tilled
        // soil
    ];
    define_crop(
        builder,
        CropDefinition {
            base_name: "farming:tea_plant".to_string(),
            stages,
            eligible_soil_blocks: soil_blocks.clone(),
            timer_period: Duration::from_secs(5),
            ..Default::default()
        },
    )?;

    Ok(())
}
