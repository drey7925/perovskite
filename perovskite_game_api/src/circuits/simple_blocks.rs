use std::time::Duration;

use anyhow::Result;
use perovskite_core::{block_id::BlockId, coordinates::ChunkOffset};
use perovskite_server::game_state::{game_map::{BulkUpdateCallback, TimerCallback, TimerSettings}, event::HandlerContext};

use crate::{
    blocks::{
        AaBoxProperties, AxisAlignedBoxesAppearanceBuilder, BlockBuilder, CubeAppearanceBuilder,
    },
    game_builder::{StaticBlockName, TextureName},
    include_texture_bytes,
};

use super::{
    get_incoming_pin_states, get_pin_state, BlockConnectivity, CircuitBlockBuilder,
    CircuitBlockCallbacks, CircuitGameBuilder, PinState, events::{transmit_edge, CircuitHandlerContext, make_root_context},
};

const CIRCUITS_SOURCE_BLOCK: StaticBlockName = StaticBlockName("circuits:source");
const LAMP_OFF_BLOCK: StaticBlockName = StaticBlockName("circuits:simple_lamp_off");
const LAMP_ON_BLOCK: StaticBlockName = StaticBlockName("circuits:simple_lamp_on");
const OSCILLATOR_ON_BLOCK: StaticBlockName = StaticBlockName("circuits:oscillator_on");
const OSCILLATOR_OFF_BLOCK: StaticBlockName = StaticBlockName("circuits:oscillator_off");

const CIRCUITS_ON_TEXTURE: TextureName = TextureName("circuits:source");
const CIRCUITS_OFF_TEXTURE: TextureName = TextureName("circuits:lamp_off");

const CIRCUITS_SOURCE_CONNECTIVITIES: [BlockConnectivity; 4] = [
    BlockConnectivity::unrotated(1, 0, 0, 0),
    BlockConnectivity::unrotated(0, 0, 1, 0),
    BlockConnectivity::unrotated(-1, 0, 0, 0),
    BlockConnectivity::unrotated(0, 0, -1, 0),
];

struct SourceBlockCallbacks(PinState);
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
        self.0
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
        _from: perovskite_core::coordinates::BlockCoordinate,
        _state: super::PinState,
    ) -> Result<()> {
        let desired = if get_incoming_pin_states(ctx, coord)
            .iter()
            .any(|(_, _, state)| state == &super::PinState::High)
        {
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
        Box::new(SourceBlockCallbacks(PinState::High)),
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
            .set_dropped_item(LAMP_OFF_BLOCK.0, 1)
            .register_circuit_callbacks(),
    )?;
    let oscillator_off_block = builder.add_block(
        BlockBuilder::new(OSCILLATOR_OFF_BLOCK)
            .set_light_emission(0)
            .set_axis_aligned_boxes_appearance(AxisAlignedBoxesAppearanceBuilder::new().add_box(
                AaBoxProperties::new_single_tex(
                    CIRCUITS_OFF_TEXTURE,
                    crate::blocks::TextureCropping::AutoCrop,
                    crate::blocks::RotationMode::None,
                ),
                (-0.2, 0.2),
                (-0.5, -0.1),
                (-0.2, 0.2),
            ))
            .set_dropped_item(LAMP_ON_BLOCK.0, 1)
            .set_display_name("Oscillator")
            .register_circuit_callbacks(),
    )?;
    let oscillator_on_block = builder.add_block(
        BlockBuilder::new(OSCILLATOR_ON_BLOCK)
            .set_light_emission(4)
            .set_axis_aligned_boxes_appearance(AxisAlignedBoxesAppearanceBuilder::new().add_box(
                AaBoxProperties::new_single_tex(
                    CIRCUITS_ON_TEXTURE,
                    crate::blocks::TextureCropping::AutoCrop,
                    crate::blocks::RotationMode::None,
                ),
                (-0.2, 0.2),
                (-0.5, 0.1),
                (-0.2, 0.2),
            ))
            .set_display_name("Oscillator (on)")
            .set_dropped_item(OSCILLATOR_OFF_BLOCK.0, 1)
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
    builder.define_circuit_callbacks(
        oscillator_off_block.id,
        Box::new(SourceBlockCallbacks(PinState::Low)),
        super::CircuitBlockProperties {
            connectivity: CIRCUITS_SOURCE_CONNECTIVITIES.to_vec(),
        },
    )?;
    builder.define_circuit_callbacks(
        oscillator_on_block.id,
        Box::new(SourceBlockCallbacks(PinState::High)),
        super::CircuitBlockProperties {
            connectivity: CIRCUITS_SOURCE_CONNECTIVITIES.to_vec(),
        },
    )?;
    builder.inner.add_timer(
        "circuits_osc_1Hz",
        TimerSettings {
            interval: Duration::from_secs(1),
            shards: 16,
            spreading: 0.1,
            block_types: vec![oscillator_off_block.id, oscillator_on_block.id],
            ..Default::default()
        },
        TimerCallback::BulkUpdate(Box::new(OscillatorTimerHandler {
            on: oscillator_on_block.id,
            off: oscillator_off_block.id,
        })),
    );

    Ok(())
}

struct OscillatorTimerHandler {
    on: BlockId,
    off: BlockId,
}
impl BulkUpdateCallback for OscillatorTimerHandler {
    fn bulk_update_callback(
        &self,
        ctx: &HandlerContext<'_>,
        chunk_coordinate: perovskite_core::coordinates::ChunkCoordinate,
        _timer_state: &perovskite_server::game_state::game_map::TimerState,
        chunk: &mut perovskite_server::game_state::game_map::MapChunk,
        _neighbors: Option<&perovskite_server::game_state::game_map::ChunkNeighbors>,
    ) -> Result<()> {
        let ctx = make_root_context(ctx);
        let mut transitions = vec![];

        for dx in 0..16 {
            for dz in 0..16 {
                for dy in 0..16 {
                    let offset = ChunkOffset::new(dx, dy, dz);
                    let block = chunk.get_block(offset);
                    let coord = chunk_coordinate.with_offset(offset);
                    if block.equals_ignore_variant(self.off) {
                        chunk.set_block(offset, self.on, None);
                        for connectivity in CIRCUITS_SOURCE_CONNECTIVITIES {
                            if let Some(dest_coord) = connectivity.eval(coord, 0) {
                                transitions.push((dest_coord, coord, PinState::High));
                            }
                        }
                    } else if block.equals_ignore_variant(self.on) {
                        chunk.set_block(offset, self.off, None);
                        for connectivity in CIRCUITS_SOURCE_CONNECTIVITIES {
                            if let Some(dest_coord) = connectivity.eval(coord, 0) {
                                transitions.push((dest_coord, coord, PinState::Low));
                            }
                        }
                    }
                }
            }
        }

        // TODO rate limiting and pushback
        ctx.run_deferred(|ctx| {
            let ctx = make_root_context(ctx);
            for (dest_coord, src_coord, state) in transitions {
                transmit_edge(&ctx, dest_coord, src_coord, state)?;
            }
            Ok(())
        });

        Ok(())
    }
}
