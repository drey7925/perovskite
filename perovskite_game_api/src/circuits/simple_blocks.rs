use std::time::Duration;

use anyhow::Result;
use perovskite_core::{
    block_id::BlockId, constants::item_groups::HIDDEN_FROM_CREATIVE, coordinates::ChunkOffset,
};
use perovskite_server::game_state::{
    event::HandlerContext,
    game_map::{BulkUpdateCallback, TimerCallback, TimerSettings},
};

use crate::{
    blocks::{
        AaBoxProperties, AxisAlignedBoxesAppearanceBuilder, BlockBuilder, CubeAppearanceBuilder,
    },
    default_game::basic_blocks::DIRT,
    game_builder::{StaticBlockName, StaticTextureName},
    include_texture_bytes,
};

use super::{
    events::{make_root_context, transmit_edge},
    get_incoming_pin_states, BlockConnectivity, CircuitBlockBuilder, CircuitBlockCallbacks,
    CircuitGameBuilder, PinState,
};

const CIRCUITS_SOURCE_BLOCK: StaticBlockName = StaticBlockName("circuits:source");
const LAMP_OFF_BLOCK: StaticBlockName = StaticBlockName("circuits:simple_lamp_off");
const LAMP_ON_BLOCK: StaticBlockName = StaticBlockName("circuits:simple_lamp_on");
const OSCILLATOR_ON_BLOCK: StaticBlockName = StaticBlockName("circuits:oscillator_on");
const OSCILLATOR_OFF_BLOCK: StaticBlockName = StaticBlockName("circuits:oscillator_off");

const CIRCUITS_ON_TEXTURE: StaticTextureName = StaticTextureName("circuits:source");
const CIRCUITS_OFF_TEXTURE: StaticTextureName = StaticTextureName("circuits:lamp_off");

const CIRCUITS_SOURCE_CONNECTIVITIES: [BlockConnectivity; 4] = [
    BlockConnectivity::unrotated(1, 0, 0, 0),
    BlockConnectivity::unrotated(0, 0, 1, 0),
    BlockConnectivity::unrotated(-1, 0, 0, 0),
    BlockConnectivity::unrotated(0, 0, -1, 0),
];

pub(super) struct SourceBlockCallbacks(pub(crate) PinState);
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
        _state: PinState,
    ) -> Result<()> {
        // Likewise, no need to do anything.
        Ok(())
    }

    fn sample_pin(
        &self,
        _ctx: &super::events::CircuitHandlerContext<'_>,
        _coord: perovskite_core::coordinates::BlockCoordinate,
        _destination: perovskite_core::coordinates::BlockCoordinate,
    ) -> PinState {
        self.0
    }

    fn on_overheat(
        &self,
        ctx: &HandlerContext,
        coord: perovskite_core::coordinates::BlockCoordinate,
    ) {
        // todo use something other than dirt
        if let Err(e) =
            ctx.game_map()
                .set_block(coord, ctx.block_types().get_by_name(DIRT.0).unwrap(), None)
        {
            tracing::error!("Error setting block: {}", e);
        }
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
        _state: PinState,
    ) -> Result<()> {
        let desired = if get_incoming_pin_states(ctx, coord)
            .iter()
            .any(|(_, _, state)| state == &PinState::High)
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
    ) -> PinState {
        PinState::Low
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
            .add_item_group(HIDDEN_FROM_CREATIVE)
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
            .set_dropped_item(OSCILLATOR_OFF_BLOCK.0, 1)
            .set_display_name("Oscillator")
            .set_allow_light_propagation(true)
            .register_circuit_callbacks(),
    )?;
    let oscillator_on_block = builder.add_block(
        BlockBuilder::new(OSCILLATOR_ON_BLOCK)
            .set_light_emission(4)
            .set_allow_light_propagation(true)
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
            .add_item_group(HIDDEN_FROM_CREATIVE)
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
    register_colored_lamps(builder)?;
    Ok(())
}

fn register_colored_lamps(builder: &mut crate::game_builder::GameBuilder) -> Result<()> {
    use crate::game_builder::{BlockName, TextureName};

    let base_on_image = image::load_from_memory(include_bytes!("textures/lamp_on_base.png"))?;
    let base_off_image = image::load_from_memory(include_bytes!("textures/lamp_off_base.png"))?;
    let mask_image =
        image::load_from_memory(include_bytes!("textures/lamp_color_mask.png"))?.to_luma8();
    for color in crate::colors::ALL_COLORS {
        let colorized_on = color.colorize_to_png_with_mask(&base_on_image, &mask_image)?;
        let colorized_off = color.colorize_to_png_with_mask(&base_off_image, &mask_image)?;

        let on_texture = TextureName(format!("circuits:lamp_{}_on", color.as_string()));
        let off_texture = TextureName(format!("circuits:lamp_{}_off", color.as_string()));
        builder.register_texture_bytes(&on_texture, &colorized_on)?;
        builder.register_texture_bytes(&off_texture, &colorized_off)?;

        let on_block_name = BlockName(format!("circuits:lamp_{}_on", color.as_string()));
        let off_block_name = BlockName(format!("circuits:lamp_{}_off", color.as_string()));

        let off_block = builder.add_block(
            BlockBuilder::new(off_block_name.clone())
                .set_light_emission(0)
                .set_display_name(format!("{} lamp", color.as_display_string()))
                .set_cube_appearance(CubeAppearanceBuilder::new().set_single_texture(&off_texture))
                .set_item_sort_key(format!("circuits:lamp:{:03}", color.sort_key()))
                .register_circuit_callbacks(),
        )?;
        let on_block = builder.add_block(
            BlockBuilder::new(on_block_name.clone())
                .set_light_emission(15)
                .set_display_name(format!("{} lamp (on)", color.as_display_string()))
                .set_cube_appearance(CubeAppearanceBuilder::new().set_single_texture(&on_texture))
                .add_item_group(HIDDEN_FROM_CREATIVE)
                .set_dropped_item(&off_block_name.0, 1)
                .register_circuit_callbacks(),
        )?;
        for id in [off_block.id, on_block.id] {
            builder.define_circuit_callbacks(
                id,
                Box::new(SimpleLampCallbacks {
                    lamp_off: off_block.id,
                    lamp_on: on_block.id,
                }),
                super::CircuitBlockProperties {
                    connectivity: CIRCUITS_SOURCE_CONNECTIVITIES.to_vec(),
                },
            )?;
        }
    }

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
