use anyhow::Result;
use perovskite_core::block_id::BlockId;

use crate::{
    blocks::{
        AaBoxProperties, AxisAlignedBoxesAppearanceBuilder, BlockBuilder, CubeAppearanceBuilder,
    },
    game_builder::{StaticBlockName, TextureName},
    include_texture_bytes,
};

use super::{
    get_pin_state, BlockConnectivity, CircuitBlockBuilder, CircuitBlockCallbacks,
    CircuitGameBuilder,
};

const CIRCUITS_SOURCE_BLOCK: StaticBlockName = StaticBlockName("circuits:source");
const LAMP_OFF_BLOCK: StaticBlockName = StaticBlockName("circuits:simple_lamp_off");
const LAMP_ON_BLOCK: StaticBlockName = StaticBlockName("circuits:simple_lamp_on");

const CIRCUITS_ON_TEXTURE: TextureName = TextureName("circuits:source");
const CIRCUITS_OFF_TEXTURE: TextureName = TextureName("circuits:lamp_off");

const CIRCUITS_SOURCE_CONNECTIVITIES: [BlockConnectivity; 4] = [
    BlockConnectivity::unrotated(1, 0, 0, 0),
    BlockConnectivity::unrotated(0, 0, 1, 0),
    BlockConnectivity::unrotated(-1, 0, 0, 0),
    BlockConnectivity::unrotated(0, 0, -1, 0),
];

struct SourceBlockCallbacks;
impl CircuitBlockCallbacks for SourceBlockCallbacks {
    fn update_connectivity(
        &self,
        _ctx: &super::events::CircuitHandlerContext<'_>,
        _coord: perovskite_core::coordinates::BlockCoordinate,
    ) -> Result<()> {
        // No need to do anything when connections change.
        // This could have used the default impl, but we want to
        // be explicit as this is a useful example for users of the API.
        Ok(())
    }

    fn on_incoming_edge(
        &self,
        _ctx: &super::events::CircuitHandlerContext<'_>,
        _coordinate: perovskite_core::coordinates::BlockCoordinate,
        _from: perovskite_core::coordinates::BlockCoordinate,
        _state: super::PinState,
    ) -> Result<()> {
        // Likewise, no need to do anything.
        Ok(())
    }

    fn sample_pin(
        &self,
        _ctx: &super::events::CircuitHandlerContext<'_>,
        _coord: perovskite_core::coordinates::BlockCoordinate,
        _destination: perovskite_core::coordinates::BlockCoordinate,
    ) -> super::PinState {
        println!("sample");
        super::PinState::High
    }
}

struct SimpleLampCallbacks {
    lamp_off: BlockId,
    lamp_on: BlockId,
}
impl CircuitBlockCallbacks for SimpleLampCallbacks {
    fn update_connectivity(
        &self,
        _ctx: &super::events::CircuitHandlerContext<'_>,
        _coord: perovskite_core::coordinates::BlockCoordinate,
    ) -> Result<()> {
        // No need to do anything when connections change.
        // This could have used the default impl, but we want to
        // be explicit as this is a useful example for users of the API.
        Ok(())
    }

    fn on_incoming_edge(
        &self,
        ctx: &super::events::CircuitHandlerContext<'_>,
        coord: perovskite_core::coordinates::BlockCoordinate,
        from: perovskite_core::coordinates::BlockCoordinate,
        state: super::PinState,
    ) -> Result<()> {
        let mut any_high = false;
        for connection in CIRCUITS_SOURCE_CONNECTIVITIES {
            let neighbor = match connection.eval(coord, 0) {
                Some(neighbor) => neighbor,
                None => continue,
            };
            if get_pin_state(ctx, neighbor, coord) == super::PinState::High {
                any_high = true;
                break;
            }
        }
        let desired = if any_high {
            self.lamp_on
        } else {
            self.lamp_off
        };
        ctx.game_map().mutate_block_atomically(coord, |block, _| {
            if block.base_id() == self.lamp_off.base_id()
                || block.base_id() == self.lamp_on.base_id()
            {
                *block = desired;
            }
            Ok(())
        })?;
        Ok(())
    }

    fn sample_pin(
        &self,
        _ctx: &super::events::CircuitHandlerContext<'_>,
        _coord: perovskite_core::coordinates::BlockCoordinate,
        _destination: perovskite_core::coordinates::BlockCoordinate,
    ) -> super::PinState {
        super::PinState::Low
    }
}

pub(crate) fn register_simple_blocks(builder: &mut crate::game_builder::GameBuilder) -> Result<()> {
    include_texture_bytes!(builder, CIRCUITS_ON_TEXTURE, "textures/wire_on.png")?;
    include_texture_bytes!(builder, CIRCUITS_OFF_TEXTURE, "textures/wire_off.png")?;
    let source_block = builder.add_block(
        BlockBuilder::new(CIRCUITS_SOURCE_BLOCK)
            .set_light_emission(4)
            .set_display_name("Digital source (always on)")
            .set_axis_aligned_boxes_appearance(AxisAlignedBoxesAppearanceBuilder::new().add_box(
                AaBoxProperties::new_single_tex(
                    CIRCUITS_ON_TEXTURE,
                    crate::blocks::TextureCropping::AutoCrop,
                    crate::blocks::RotationMode::None,
                ),
                (-0.2, 0.2),
                (-0.5, -0.1),
                (-0.2, 0.2),
            ))
            .register_circuit_callbacks(),
    )?;
    builder.define_circuit_callbacks(
        source_block.id,
        Box::new(SourceBlockCallbacks),
        super::CircuitBlockProperties {
            connectivity: CIRCUITS_SOURCE_CONNECTIVITIES.to_vec(),
        },
    )?;

    let lamp_off_block = builder.add_block(
        BlockBuilder::new(LAMP_OFF_BLOCK)
            .set_light_emission(0)
            .set_display_name("Lamp (off)")
            .set_cube_appearance(
                CubeAppearanceBuilder::new().set_single_texture(CIRCUITS_OFF_TEXTURE),
            )
            .register_circuit_callbacks(),
    )?;
    let lamp_on_block = builder.add_block(
        BlockBuilder::new(LAMP_ON_BLOCK)
            .set_light_emission(15)
            .set_display_name("Lamp (on)")
            .set_cube_appearance(
                CubeAppearanceBuilder::new().set_single_texture(CIRCUITS_ON_TEXTURE),
            )
            .register_circuit_callbacks(),
    )?;
    for id in [lamp_off_block.id, lamp_on_block.id] {
        builder.define_circuit_callbacks(
            id,
            Box::new(SimpleLampCallbacks {
                lamp_off: lamp_off_block.id,
                lamp_on: lamp_on_block.id,
            }),
            super::CircuitBlockProperties {
                connectivity: CIRCUITS_SOURCE_CONNECTIVITIES.to_vec(),
            },
        )?;
    }

    Ok(())
}
