/// Utilities to define plants that can be grown
pub mod crops;
mod soil;
mod tea;

use crate::blocks::{DroppedItem, PlantLikeAppearanceBuilder, RotationMode};
use crate::default_game::basic_blocks::ores::IRON_INGOT;
use crate::default_game::block_groups::SOILS;
use crate::default_game::foliage::STICK_ITEM;
use crate::default_game::recipes::RecipeSlot;
use crate::default_game::DefaultGameBuilder;
use crate::farming::crops::{
    CropDefinition, DefaultGrowInLight, GrowthStage, InteractionTransitionTarget,
};
use crate::game_builder::{
    GameBuilder, GameBuilderExtension, OwnedTextureName, FALLBACK_UNKNOWN_TEXTURE_NAME,
};
use crate::items::{ItemAction, ItemActionTarget, ItemBuilder, ItemHandler, StackDecrement};
use anyhow::Result;
use perovskite_core::protocol::items::item_stack::QuantityType;
use perovskite_core::protocol::render::TextureReference;
use perovskite_server::game_state::blocks::FastBlockName;
use perovskite_server::game_state::GameStateExtension;
use perovskite_server::server::ServerBuilder;
use rand::{thread_rng, Rng};
use std::time::Duration;

#[derive(Clone)]
pub struct FarmingGameStateExtension {
    /// Paddy, successfully flooded by the player
    pub paddy_dry: FastBlockName,
    /// Paddy, still requiring water to be supplied
    pub paddy_wet: FastBlockName,
    /// Soil, tilled and irrigated
    pub soil_wet: FastBlockName,
    /// Soil, tilled but not yet irrigated
    pub soil_dry: FastBlockName,
}
impl GameStateExtension for FarmingGameStateExtension {}

pub fn initialize_farming(builder: &mut GameBuilder) -> Result<()> {
    soil::register_soil_blocks(builder)?;
    let ext = builder.builder_extension_mut::<FarmingGameStateExtension>();
    let wet_soil_blocks = vec![ext.paddy_wet.clone(), ext.soil_wet.clone()];
    let soil_dry = ext.soil_dry.clone();
    let mut stages = vec![];
    let mut rng = thread_rng();
    for _ in 0..32 {
        let tex = OwnedTextureName::from_css_color(&format!(
            "rgb({} {} {})",
            rng.gen_range(0..255),
            rng.gen_range(0..255),
            rng.gen_range(0..255)
        ));
        stages.push(GrowthStage {
            dig_effect: None,
            tap_effect: None,
            interaction_effects: Default::default(),
            extra_block_groups: vec![],
            grow_probability: Box::new(DefaultGrowInLight(
                InteractionTransitionTarget::NextStage.into(),
            )),
            appearance: PlantLikeAppearanceBuilder::from_tex(tex).into(),
            ..Default::default()
        })
    }
    crops::define_crop(
        builder,
        CropDefinition {
            base_name: "farming:test_crop".to_string(),
            stages,
            eligible_soil_blocks: wet_soil_blocks,
            timer_period: Duration::from_secs(1),
            ..CropDefinition::default()
        },
    )?;
    tea::register_tea(builder)?;
    register_hoe(
        builder,
        FALLBACK_UNKNOWN_TEXTURE_NAME,
        "farming:iron_hoe",
        "Iron hoe",
        120,
        "iron",
        Some(IRON_INGOT.into()),
        soil_dry,
    )?;
    Ok(())
}

impl Default for FarmingGameStateExtension {
    fn default() -> Self {
        Self {
            paddy_dry: FastBlockName::new("farming:paddy_dry"),
            paddy_wet: FastBlockName::new("farming:paddy_wet"),
            soil_wet: FastBlockName::new("farming:soil_wet"),
            soil_dry: FastBlockName::new("farming:soil_dry"),
        }
    }
}

impl GameBuilderExtension for FarmingGameStateExtension {
    fn pre_run(&mut self, server_builder: &mut ServerBuilder) {
        server_builder.add_extension(self.clone())
    }
}

pub(crate) fn register_hoe(
    game_builder: &mut GameBuilder,
    texture: impl Into<TextureReference>,
    name: impl Into<String>,
    display_name: impl Into<String>,
    durability: u32,
    sort_key_component: &str,
    craft_component: Option<RecipeSlot>,
    produced_soil: FastBlockName,
) -> Result<()> {
    let name = name.into();
    ItemBuilder::new(name.clone())
        .set_display_name(display_name)
        .add_tap_handler(ItemHandler {
            target: ItemActionTarget::BlockGroup(SOILS.into()),
            action: ItemAction::PlaceBlock {
                block: produced_soil,
                rotation_mode: RotationMode::None,
                ignore_trivially_replaceable: true,
                place_onto: false,
            },
            stack_decrement: Some(StackDecrement::FixedDestroyIfInsufficient(1)),
            dropped_item: DroppedItem::None,
            ..Default::default()
        })
        .set_inventory_texture(texture)
        .set_max_wear(durability)
        .set_sort_key("farming:tools:hoe:".to_string() + sort_key_component)
        .build_into(game_builder)?;
    if let Some(component) = craft_component {
        game_builder.register_crafting_recipe(
            [
                component.clone(),
                component,
                RecipeSlot::Empty,
                RecipeSlot::Empty,
                RecipeSlot::Exact(STICK_ITEM.0.to_string()),
                RecipeSlot::Empty,
                RecipeSlot::Empty,
                RecipeSlot::Exact(STICK_ITEM.0.to_string()),
                RecipeSlot::Empty,
            ],
            name,
            durability,
            Some(QuantityType::Wear(durability)),
            false,
        );
    }
    Ok(())
}
