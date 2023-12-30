use perovskite_server::game_state::GameStateExtension;

use crate::game_builder::{GameBuilder, StaticBlockName};

pub const PADDY_DRY: StaticBlockName = StaticBlockName("farming:paddy_dry");
pub const PADDY_WET: StaticBlockName = StaticBlockName("farming:paddy_wet");

struct FarmingGameStateExtension {}
impl GameStateExtension for FarmingGameStateExtension {}

pub trait FarmingGameBuilder {
    fn initialize_farming(&mut self);
}

impl FarmingGameBuilder for GameBuilder {
    fn initialize_farming(&mut self) {
        todo!("initialize farming")
    }
}
