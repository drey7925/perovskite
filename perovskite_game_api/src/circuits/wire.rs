use anyhow::Result;

use crate::{blocks::BlockBuilder, game_builder::{StaticBlockName, GameBuilder}};

pub const WIRE_BLOCK: StaticBlockName = StaticBlockName("circuits:wire");

pub(crate) fn register_wire(builder: &mut GameBuilder) -> Result<()> {
    
    let wire_block = builder.add_block(BlockBuilder::new(WIRE_BLOCK));
    todo!();
}