//! A module allowing players to grow, process, and enjoy various varieties of tea. Eventually, a
//! possible testbed for additional item functionality (e.g. in teapots and rare varieties of leaves
//! obtained by chance)
//!
//! Roadmap:
//! * [ ] Basic tea growth
//! * [ ] Basic tea processing (e.g., withering, oxidation, fixing, drying)
//! * [ ] ...

use crate::blocks::{
    AxisAlignedBoxesAppearanceBuilder, BlockAppearanceBuilder, PlantLikeAppearanceBuilder,
};
use crate::default_game::basic_blocks::{DIRT, DIRT_WITH_GRASS};
use crate::default_game::recipes::RecipeSlot;
use crate::default_game::DefaultGameBuilder;
use crate::farming::crops::{
    define_crop, CropDefinition, DefaultGrowInLight, GrowthStage, InteractionAccumulator,
    InteractionEffect, InteractionTransition, InteractionTransitionTarget,
};
use crate::farming::FarmingGameStateExtension;
use crate::game_builder::{
    GameBuilder, StaticBlockName, StaticTextureName, FALLBACK_UNKNOWN_TEXTURE_NAME,
};
use anyhow::Result;
use perovskite_server::game_state::blocks::FastBlockName;
use std::time::Duration;

const TEA_LEAVES_FRESH_ITEM: &'static str = "farming:tea_leaves_fresh";
const TEA_LEAVES_STEAMED_ITEM: &'static str = "farming:tea_leaves_steamed";

pub(crate) fn register_tea(builder: &mut GameBuilder) -> Result<()> {
    builder.register_basic_item(
        TEA_LEAVES_FRESH_ITEM,
        "Fresh tea leaves",
        FALLBACK_UNKNOWN_TEXTURE_NAME,
        vec![],
        "farming:tea:leaves:fresh",
    )?;

    let steamed_leaves = builder.register_basic_item(
        TEA_LEAVES_STEAMED_ITEM,
        "Steamed tea leaves",
        FALLBACK_UNKNOWN_TEXTURE_NAME,
        vec![],
        "farming:tea:leaves:steamed",
    )?;
    let steamed_leaves_stack = steamed_leaves.make_stack(1);
    // Borrow checker: need to pre-create the stack since `steamed_leaves` borrows from `builder`.
    // (this is ugly, maybe we could get non-borrowing item handles holding string names????)

    // Steaming fresh leaves gets us oxidation-fixed leaves, which then become green tea when rolled
    // and then dried
    builder.register_smelting_recipe(
        RecipeSlot::Exact(TEA_LEAVES_FRESH_ITEM.to_string()),
        steamed_leaves_stack,
    );

    // let steamed_rolled_leaves = builder.register_basic_item(
    //     TEA_LEAVES_STEAMED_ITEM,
    // );

    register_tea_plant_stages(builder)?;

    register_tea_basket_stages(builder)?;

    Ok(())
}

const EMPTY_TEA_BASKET: StaticBlockName = StaticBlockName("farming:tea_basket_empty");

fn make_basket_aabbs(contents_texture: StaticTextureName) -> AxisAlignedBoxesAppearanceBuilder {
    todo!()
}

fn register_tea_basket_stages(game_builder: &mut GameBuilder) -> Result<()> {
    // TODO: build the graph of open-air processing stages, potentially using something like
    // https://en.wikipedia.org/wiki/Tea_processing#/media/File:Teaprocessing.svg

    //
    // let stages = vec![
    //     // empty basket should be a basic block via the block builder, so it can get its own item.
    //     // etc
    //     // GrowthStage {
    //     //     dig_effect: Some(InteractionEffect {
    //     //         item_drops: vec![
    //     //             // TODO: tea tree sapling?
    //     //         ],
    //     //         transition: InteractionTransitionTarget::Remove.into(),
    //     //         transition_probability: 1.0,
    //     //         ..Default::default()
    //     //     }),
    //     //     block_name: Some(String::from("farming:tea_basket_empty")),
    //     //     grow_probability: Box::new(DefaultGrowInLight(InteractionTransition {
    //     //         target: InteractionTransitionTarget::NextStage,
    //     //         accumulator: Some(InteractionAccumulator { add: 1, trip: 20 }),
    //     //     })),
    //     //     appearance: BlockAppearanceBuilder::Plantlike(PlantLikeAppearanceBuilder::default()),
    //     //     ..Default::default()
    //     // },
    //
    // ];

    Ok(())
}

fn register_tea_plant_stages(builder: &mut GameBuilder) -> Result<()> {
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
            block_name: Some(String::from("farming:tea_sprout")),
            grow_probability: Box::new(DefaultGrowInLight(InteractionTransition {
                target: InteractionTransitionTarget::NextStage,
                accumulator: Some(InteractionAccumulator { add: 1, trip: 20 }),
            })),
            appearance: BlockAppearanceBuilder::Plantlike(PlantLikeAppearanceBuilder::default()),
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
            block_name: Some(String::from("farming:tea_seedling")),
            grow_probability: Box::new(DefaultGrowInLight(InteractionTransition {
                target: InteractionTransitionTarget::NextStage,
                accumulator: Some(InteractionAccumulator { add: 1, trip: 20 }),
            })),
            appearance: BlockAppearanceBuilder::Plantlike(PlantLikeAppearanceBuilder::default()),
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
            block_name: Some(String::from("farming:tea_plant_no_leaves")),
            grow_probability: Box::new(DefaultGrowInLight(InteractionTransition {
                target: InteractionTransitionTarget::NextStage,
                accumulator: Some(InteractionAccumulator { add: 1, trip: 20 }),
            })),
            appearance: BlockAppearanceBuilder::Plantlike(PlantLikeAppearanceBuilder::default()),
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
                item_drops: vec![(String::from(TEA_LEAVES_FRESH_ITEM), 1..=3)],
                transition: InteractionTransitionTarget::JumpToStage(2).into(),
                transition_probability: 1.0,
                ..Default::default()
            }),
            block_name: Some(String::from("farming:tea_plant_sparse_leaves")),
            grow_probability: Box::new(DefaultGrowInLight(InteractionTransition {
                target: InteractionTransitionTarget::NextStage,
                accumulator: Some(InteractionAccumulator { add: 1, trip: 20 }),
            })),
            appearance: BlockAppearanceBuilder::Plantlike(PlantLikeAppearanceBuilder::default()),
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
                item_drops: vec![(String::from(TEA_LEAVES_FRESH_ITEM), 1..=3)],
                transition: InteractionTransitionTarget::JumpToStage(2).into(),
                transition_probability: 1.0,
                ..Default::default()
            }),
            block_name: Some(String::from("farming:tea_plant_dense_leaves")),
            grow_probability: Box::new(DefaultGrowInLight(InteractionTransition {
                target: InteractionTransitionTarget::NextStage,
                accumulator: Some(InteractionAccumulator { add: 1, trip: 20 }),
            })),
            appearance: BlockAppearanceBuilder::Plantlike(PlantLikeAppearanceBuilder::default()),
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
