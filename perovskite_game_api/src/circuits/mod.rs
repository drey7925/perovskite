// placeholders for circuits

use crate::{default_game::DefaultGameBuilder, game_builder::GameBuilder};
use anyhow::Result;

mod wire;

pub mod constants {
    pub const CIRCUIT_BLOCK_GROUP: &str = "circuits";
}


pub fn register_circuits(builder: &mut GameBuilder) -> Result<()> {
    wire::register_wire(builder)?;
    Ok(())
}