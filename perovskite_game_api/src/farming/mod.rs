/// Utilities to define plants that can be grown
pub mod crops;
mod soil;
mod tea;

use crate::blocks::PlantLikeAppearanceBuilder;
use crate::farming::crops::{CropDefinition, DefaultGrowInLight, GrowthStage};
use crate::game_builder::{GameBuilder, GameBuilderExtension, OwnedTextureName};
use anyhow::Result;
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
    let soil_blocks = vec![ext.paddy_wet.clone(), ext.soil_wet.clone()];
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
            grow_probability: Box::new(DefaultGrowInLight),
            appearance: PlantLikeAppearanceBuilder::from_tex(tex).into(),
            ..Default::default()
        })
    }
    crops::define_crop(
        builder,
        CropDefinition {
            base_name: "farming:test_crop".to_string(),
            stages,
            eligible_soil_blocks: soil_blocks,
            timer_period: Duration::from_secs(1),
            ..CropDefinition::default()
        },
    )?;
    tea::register_tea(builder)?;
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
