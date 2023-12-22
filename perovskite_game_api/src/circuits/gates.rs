use crate::{game_builder::{TextureName, GameBuilder}, include_texture_bytes};

use anyhow::Result;


const GATE_SIDE_TEXTURE: TextureName = TextureName("circuits:gate_side");
const GATE_SIDE_PIN_TEXTURE: TextureName = TextureName("circuits:gate_side_pin");


pub(crate) fn register_gates(builder: &mut GameBuilder) -> Result<()> {
    include_texture_bytes!(builder, GATE_SIDE_TEXTURE, "../textures/gate_side.png")?;
    include_texture_bytes!(builder, GATE_SIDE_PIN_TEXTURE, "../textures/gate_side_pin.png")?;

    Ok(())
}