mod soil;

use crate::game_builder::{GameBuilder, GameBuilderExtension};
use anyhow::Result;
use perovskite_server::game_state::blocks::FastBlockName;
use perovskite_server::game_state::GameStateExtension;
use perovskite_server::server::ServerBuilder;

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
    soil::register_soil_blocks(builder)
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
    fn pre_run(&mut self, server_builder: &mut ServerBuilder) {}
}
