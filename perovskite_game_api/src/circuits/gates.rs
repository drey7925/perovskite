use std::time::Duration;

use crate::{
    blocks::{AaBoxProperties, AxisAlignedBoxesAppearanceBuilder, BlockBuilder},
    game_builder::{GameBuilder, StaticBlockName, TextureName},
    include_texture_bytes,
};

use anyhow::Result;
use perovskite_core::{block_id::BlockId, constants::item_groups::HIDDEN_FROM_CREATIVE};
use perovskite_server::game_state::{blocks::FastBlockName, event::HandlerContext};
use smallvec::SmallVec;

use super::{
    events::{make_root_context, transmit_edge},
    get_incoming_pin_states, get_pin_state, BlockConnectivity, CircuitBlockBuilder,
    CircuitBlockCallbacks, CircuitBlockProperties, CircuitGameBuilder,
};

const SIDE_TEX: TextureName = TextureName("circuits:gate_side");
const SIDE_PIN_TEX: TextureName = TextureName("circuits:gate_side_pin");
const BOTTOM_TEX: TextureName = TextureName("circuits:gate_bottom");

const BROKEN_GATE: StaticBlockName = StaticBlockName("circuits:broken_gate");
const BROKEN_TOP_TEX: TextureName = TextureName("circuits:broken_gate_top");

fn get_side_tex(connects: bool) -> TextureName {
    if connects {
        SIDE_PIN_TEX
    } else {
        SIDE_TEX
    }
}

/// Configuration for a single combinational gate, taking up to three inputs and producing a single output.
#[derive(Clone, Debug)]
pub struct CombinationalGate {
    description: &'static str,
    off_name: StaticBlockName,
    on_name: StaticBlockName,
    off_texture: TextureName,
    on_texture: TextureName,
    /// The truth table for the gate. Each bit is 0 or 1, eight elements indexed by three bits of input state.
    /// e.g. left on, front off, right off is bit 4 (0b100)
    /// Output is always on the back.
    truth_table: u8,
    connects_left: bool,
    connects_front: bool,
    connects_right: bool,
}

const AND_GATE_CONFIG: CombinationalGate = CombinationalGate {
    description: "And gate",
    off_name: StaticBlockName("circuits:and_gate_off"),
    on_name: StaticBlockName("circuits:and_gate_on"),
    off_texture: TextureName("circuits:and_gate_off"),
    on_texture: TextureName("circuits:and_gate_on"),
    // 101 and 111: left and right on, front is dont care
    truth_table: (1 << 0b101) | (1 << 0b111),
    connects_left: true,
    connects_front: false,
    connects_right: true,
};

const NOT_GATE_CONFIG: CombinationalGate = CombinationalGate {
    description: "Not gate",
    off_name: StaticBlockName("circuits:not_gate_off"),
    on_name: StaticBlockName("circuits:not_gate_on"),
    off_texture: TextureName("circuits:not_gate_off"),
    on_texture: TextureName("circuits:not_gate_on"),
    // front should be low, otherwise don't care
    truth_table: (1 << 0b000) | (1 << 0b001) | (1 << 0b100) | (1 << 0b101),
    connects_left: false,
    connects_front: true,
    connects_right: false,
};

const XOR_GATE_CONFIG: CombinationalGate = CombinationalGate {
    description: "Xor gate",
    off_name: StaticBlockName("circuits:xor_gate_off"),
    on_name: StaticBlockName("circuits:xor_gate_on"),
    off_texture: TextureName("circuits:xor_gate_off"),
    on_texture: TextureName("circuits:xor_gate_on"),
    // 001, 011, 100, 110: left xor right, front is dont care
    truth_table: (1 << 0b001) | (1 << 0b011) | (1 << 0b100) | (1 << 0b110),
    connects_left: true,
    connects_front: false,
    connects_right: true,
};

struct CombinationalGateImpl {
    config: CombinationalGate,
    on: BlockId,
    off: BlockId,
    broken: FastBlockName,
}
impl CircuitBlockCallbacks for CombinationalGateImpl {
    fn on_incoming_edge(
        &self,
        ctx: &super::events::CircuitHandlerContext<'_>,
        coord: perovskite_core::coordinates::BlockCoordinate,
        _from: perovskite_core::coordinates::BlockCoordinate,
        _state: super::PinState,
    ) -> Result<()> {
        let inbound = get_incoming_pin_states(ctx, coord);
        let mut enabled = 0u8;
        for (connectivity, _coordinate, state) in inbound {
            if state == super::PinState::High {
                enabled |= connectivity.id as u8;
            }
        }
        let result_is_high = self.config.truth_table & (1 << enabled) != 0;
        let pin_state = if result_is_high {
            super::PinState::High
        } else {
            super::PinState::Low
        };

        let (should_signal, variant) =
            ctx.game_map().mutate_block_atomically(coord, |block, _| {
                let variant = block.variant();
                if result_is_high && block.equals_ignore_variant(self.off) {
                    *block = self.on.with_variant(variant)?;
                    return Ok((true, variant));
                } else if !result_is_high && block.equals_ignore_variant(self.on) {
                    *block = self.off.with_variant(variant)?;
                    return Ok((true, variant));
                } else {
                    return Ok((false, 0));
                }
            })?;
        if should_signal {
            transmit_edge(
                ctx,
                OUTPUT_CONNECTVIITY.eval(coord, variant).unwrap(),
                coord,
                pin_state,
            )?;
        }
        Ok(())
    }

    fn sample_pin(
        &self,
        ctx: &super::events::CircuitHandlerContext<'_>,
        coord: perovskite_core::coordinates::BlockCoordinate,
        destination: perovskite_core::coordinates::BlockCoordinate,
    ) -> super::PinState {
        let block = match ctx.game_map().try_get_block(coord) {
            Some(block) => block,
            None => return super::PinState::Low,
        };
        let self_variant = block.variant();
        if OUTPUT_CONNECTVIITY.eval(coord, self_variant) != Some(destination) {
            return super::PinState::Low;
        }

        if block.equals_ignore_variant(self.on) {
            super::PinState::High
        } else {
            super::PinState::Low
        }
    }

    fn on_overheat(
        &self,
        ctx: &perovskite_server::game_state::event::HandlerContext,
        coord: perovskite_core::coordinates::BlockCoordinate,
    ) {
        ctx.game_map().set_block(coord, &self.broken, None).unwrap();
    }
}

const OUTPUT_CONNECTVIITY: BlockConnectivity =
    BlockConnectivity::rotated_nesw_with_variant(0, 0, -1, 0);
const LEFT_CONNECTIVITY: BlockConnectivity =
    BlockConnectivity::rotated_nesw_with_variant(1, 0, 0, 0b100);
const FRONT_CONNECTIVITY: BlockConnectivity =
    BlockConnectivity::rotated_nesw_with_variant(0, 0, 1, 0b010);
const RIGHT_CONNECTIVITY: BlockConnectivity =
    BlockConnectivity::rotated_nesw_with_variant(-1, 0, 0, 0b001);

/// Registers a new gate. Its on and off textures must already be registered.
pub fn register_combinational_gate(
    builder: &mut GameBuilder,
    gate: CombinationalGate,
) -> Result<()> {
    register_gate(builder, gate.clone(), move |off_block, on_block| {
        Box::new(CombinationalGateImpl {
            config: gate.clone(),
            on: on_block,
            off: off_block,
            broken: FastBlockName::new(BROKEN_GATE.0.to_string()),
        })
    })?;
    Ok(())
}

fn register_gate(
    builder: &mut GameBuilder,
    gate: CombinationalGate,
    mut to_callbacks: impl FnMut(BlockId, BlockId) -> Box<dyn CircuitBlockCallbacks>,
) -> Result<()> {
    // Precondition: the first connectivity must be the output
    let mut connectivity = SmallVec::<[BlockConnectivity; 4]>::new();
    connectivity.push(OUTPUT_CONNECTVIITY);
    if gate.connects_left {
        connectivity.push(LEFT_CONNECTIVITY);
    }
    if gate.connects_front {
        connectivity.push(FRONT_CONNECTIVITY);
    }
    if gate.connects_right {
        connectivity.push(RIGHT_CONNECTIVITY);
    }
    let box_properties_off = AaBoxProperties::new(
        get_side_tex(gate.connects_left),
        get_side_tex(gate.connects_right),
        gate.off_texture,
        BOTTOM_TEX,
        get_side_tex(true),
        get_side_tex(gate.connects_front),
        crate::blocks::TextureCropping::AutoCrop,
        crate::blocks::RotationMode::RotateHorizontally,
    );
    let box_properties_on = AaBoxProperties::new(
        get_side_tex(gate.connects_left),
        get_side_tex(gate.connects_right),
        gate.on_texture,
        BOTTOM_TEX,
        get_side_tex(true),
        get_side_tex(gate.connects_front),
        crate::blocks::TextureCropping::AutoCrop,
        crate::blocks::RotationMode::RotateHorizontally,
    );
    let off_block = builder.add_block(
        BlockBuilder::new(gate.off_name)
            .set_axis_aligned_boxes_appearance(make_chip_shape(box_properties_off))
            .set_allow_light_propagation(true)
            .set_display_name(gate.description)
            .set_inventory_texture(gate.off_texture)
            .register_circuit_callbacks(),
    )?;
    let on_block = builder.add_block(
        BlockBuilder::new(gate.on_name)
            .set_axis_aligned_boxes_appearance(make_chip_shape(box_properties_on))
            .set_allow_light_propagation(true)
            .set_light_emission(4)
            .set_display_name(gate.description)
            .set_inventory_texture(gate.on_texture)
            .add_item_group(HIDDEN_FROM_CREATIVE)
            .set_dropped_item(gate.off_name.0, 1)
            .register_circuit_callbacks(),
    )?;
    for block_id in [off_block.id, on_block.id] {
        let properties = CircuitBlockProperties {
            connectivity: connectivity.to_vec(),
        };
        builder.define_circuit_callbacks(
            block_id,
            to_callbacks(off_block.id, on_block.id),
            properties,
        )?;
    }

    Ok(())
}

pub(crate) fn register_base_gates(builder: &mut GameBuilder) -> Result<()> {
    include_texture_bytes!(builder, SIDE_TEX, "textures/gate_side.png")?;
    include_texture_bytes!(builder, SIDE_PIN_TEX, "textures/gate_side_pin.png")?;
    include_texture_bytes!(builder, BROKEN_TOP_TEX, "textures/broken_gate_top.png")?;
    include_texture_bytes!(builder, BOTTOM_TEX, "textures/gate_bottom.png")?;

    let box_properties_broken = AaBoxProperties::new(
        get_side_tex(false),
        get_side_tex(false),
        BROKEN_TOP_TEX,
        BOTTOM_TEX,
        get_side_tex(false),
        get_side_tex(false),
        crate::blocks::TextureCropping::AutoCrop,
        crate::blocks::RotationMode::RotateHorizontally,
    );

    include_texture_bytes!(
        builder,
        AND_GATE_CONFIG.off_texture,
        "textures/and_gate_top.png"
    )?;
    include_texture_bytes!(
        builder,
        AND_GATE_CONFIG.on_texture,
        "textures/and_gate_top_on.png"
    )?;

    include_texture_bytes!(
        builder,
        NOT_GATE_CONFIG.off_texture,
        "textures/not_gate_top.png"
    )?;
    include_texture_bytes!(
        builder,
        NOT_GATE_CONFIG.on_texture,
        "textures/not_gate_top_on.png"
    )?;

    include_texture_bytes!(
        builder,
        XOR_GATE_CONFIG.off_texture,
        "textures/xor_gate_top.png"
    )?;
    include_texture_bytes!(
        builder,
        XOR_GATE_CONFIG.on_texture,
        "textures/xor_gate_top_on.png"
    )?;

    register_delay_gate(builder)?;
    register_dff(builder)?;

    builder.add_block(
        BlockBuilder::new(BROKEN_GATE)
            .set_axis_aligned_boxes_appearance(make_chip_shape(box_properties_broken))
            .set_allow_light_propagation(true)
            .set_no_drops()
            .set_inventory_texture(BROKEN_TOP_TEX)
            .add_item_group(HIDDEN_FROM_CREATIVE),
    )?;

    register_combinational_gate(builder, AND_GATE_CONFIG)?;
    register_combinational_gate(builder, NOT_GATE_CONFIG)?;
    register_combinational_gate(builder, XOR_GATE_CONFIG)?;
    Ok(())
}

fn make_chip_shape(box_properties: AaBoxProperties) -> AxisAlignedBoxesAppearanceBuilder {
    // Main body
    AxisAlignedBoxesAppearanceBuilder::new()
        .add_box(
            box_properties.clone(),
            (-0.5, 0.5),
            (-0.5, -0.375),
            (-0.5, 0.5),
        )
        // Chip
        .add_box(
            box_properties,
            (-0.25, 0.25),
            (-0.375, -0.3125),
            (-0.25, 0.25),
        )
}

// This isn't strictly a combinational gate, but we'll provide our own callbacks implementation
const DELAY_GATE_PROPERTIES: CombinationalGate = CombinationalGate {
    description: "Delay gate",
    off_name: StaticBlockName("circuits:delay_gate_off"),
    on_name: StaticBlockName("circuits:delay_gate_on"),
    off_texture: TextureName("circuits:delay_gate_off"),
    on_texture: TextureName("circuits:delay_gate_on"),
    // 001, 011, 100, 110: left xor right, front is dont care
    truth_table: 0,
    connects_left: false,
    connects_front: true,
    connects_right: false,
};
const DFF_PROPERTIES: CombinationalGate = CombinationalGate {
    description: "D Flip-Flop",
    off_name: StaticBlockName("circuits:dff_off"),
    on_name: StaticBlockName("circuits:dff_on"),
    off_texture: TextureName("circuits:dff_off"),
    on_texture: TextureName("circuits:dff_on"),
    // 001, 011, 100, 110: left xor right, front is dont care
    truth_table: 0,
    connects_left: false,
    connects_front: true,
    connects_right: true,
};

const DELAY_GATE_INPUT_WAS_HIGH_VARIANT_BIT: u16 = 4;

struct DelayGateImpl {
    on: BlockId,
    off: BlockId,
    broken: FastBlockName,
}
impl CircuitBlockCallbacks for DelayGateImpl {
    fn on_incoming_edge(
        &self,
        ctx: &super::events::CircuitHandlerContext<'_>,
        coord: perovskite_core::coordinates::BlockCoordinate,
        from: perovskite_core::coordinates::BlockCoordinate,
        state: super::PinState,
    ) -> Result<()> {
        let block = match ctx.game_map().try_get_block(coord) {
            Some(block) => block,
            None => return Ok(()),
        };
        let self_variant = block.variant();
        if FRONT_CONNECTIVITY.eval(coord, self_variant) != Some(from) {
            return Ok(());
        }

        let should_signal = ctx.game_map().mutate_block_atomically(coord, |block, _| {
            let variant = block.variant();
            let pending_count = (variant >> 3) + 1;

            let new_variant = if state == super::PinState::High {
                variant | DELAY_GATE_INPUT_WAS_HIGH_VARIANT_BIT
            } else {
                variant & !DELAY_GATE_INPUT_WAS_HIGH_VARIANT_BIT
            };
            if block.equals_ignore_variant(self.off) || block.equals_ignore_variant(self.on) {
                if pending_count > 8 {
                    *block = ctx.block_types().resolve_name(&self.broken).unwrap();
                    return Ok(false);
                }

                if variant != new_variant {
                    let new_variant_with_counter = (new_variant & 0x7) | (pending_count << 3);
                    *block = block.with_variant(new_variant_with_counter)?;
                    return Ok(true);
                }
                return Ok(false);
            } else {
                // race condition, wrong block is there
                return Ok(false);
            }
        })?;

        if should_signal {
            let on = self.on;
            let off = self.off;
            ctx.run_deferred_delayed(Duration::from_millis(250), move |ctx| {
                delayed_edge(ctx, coord, on, off, state)
            })
        }

        Ok(())
    }

    fn sample_pin(
        &self,
        ctx: &super::events::CircuitHandlerContext<'_>,
        coord: perovskite_core::coordinates::BlockCoordinate,
        destination: perovskite_core::coordinates::BlockCoordinate,
    ) -> super::PinState {
        let block = match ctx.game_map().try_get_block(coord) {
            Some(block) => block,
            None => return super::PinState::Low,
        };
        let self_variant = block.variant();
        if OUTPUT_CONNECTVIITY.eval(coord, self_variant) != Some(destination) {
            return super::PinState::Low;
        }

        if block.equals_ignore_variant(self.on) {
            super::PinState::High
        } else {
            super::PinState::Low
        }
    }

    fn on_overheat(
        &self,
        ctx: &perovskite_server::game_state::event::HandlerContext,
        coord: perovskite_core::coordinates::BlockCoordinate,
    ) {
        ctx.game_map().set_block(coord, &self.broken, None).unwrap();
    }
}

fn delayed_edge(
    ctx: &HandlerContext,
    coord: perovskite_core::coordinates::BlockCoordinate,
    on: BlockId,
    off: BlockId,
    state: super::PinState,
) -> Result<()> {
    let (edge_happened, deferred_variant) =
        ctx.game_map().mutate_block_atomically(coord, |block, _| {
            let variant = block.variant();

            let variant_count = variant >> 3;
            let new_variant = (variant & 0x7) | ((variant_count.saturating_sub(1)) << 3);

            let is_off = block.equals_ignore_variant(off);
            let is_on = block.equals_ignore_variant(on);

            if is_off || is_on {
                if state == super::PinState::High {
                    if variant_count == 0 {
                        tracing::warn!("deferred edge but no pending edges");
                    }
                    *block = on.with_variant(new_variant)?;
                    return Ok((is_off, new_variant));
                } else if state == super::PinState::Low {
                    if variant_count == 0 {
                        tracing::warn!("deferred edge but no pending edges");
                    }
                    *block = off.with_variant(new_variant)?;
                    return Ok((is_on, new_variant));
                }
            }

            Ok((false, 0))
        })?;
    if edge_happened {
        transmit_edge(
            &make_root_context(ctx),
            OUTPUT_CONNECTVIITY.eval(coord, deferred_variant).unwrap(),
            coord,
            state,
        )?;
    }

    Ok(())
}

fn register_delay_gate(builder: &mut GameBuilder) -> Result<()> {
    include_texture_bytes!(
        builder,
        DELAY_GATE_PROPERTIES.off_texture,
        "textures/delay_gate_top.png"
    )?;
    include_texture_bytes!(
        builder,
        DELAY_GATE_PROPERTIES.on_texture,
        "textures/delay_gate_top_on.png"
    )?;

    register_gate(builder, DELAY_GATE_PROPERTIES, |off, on| {
        Box::new(DelayGateImpl {
            on,
            off,
            broken: FastBlockName::new(BROKEN_GATE.0.to_string()),
        })
    })?;

    Ok(())
}

const DFF_GATE_INPUT_CLOCK_WAS_HIGH_VARIANT_BIT: u16 = 4;
struct DffImpl {
    on: BlockId,
    off: BlockId,
    broken: FastBlockName,
}
impl CircuitBlockCallbacks for DffImpl {
    fn on_incoming_edge(
        &self,
        ctx: &super::events::CircuitHandlerContext<'_>,
        coord: perovskite_core::coordinates::BlockCoordinate,
        from: perovskite_core::coordinates::BlockCoordinate,
        clock_state: super::PinState,
    ) -> Result<()> {
        let block = match ctx.game_map().try_get_block(coord) {
            Some(block) => block,
            None => return Ok(()),
        };
        let self_variant = block.variant();
        if RIGHT_CONNECTIVITY.eval(coord, self_variant) != Some(from) {
            return Ok(());
        }
        // at this point, we've verified that the incoming edge is on the clock pin.
        // Let's sample the data pin
        let data_pin_neighbor = match FRONT_CONNECTIVITY.eval(coord, self_variant) {
            Some(neighbor) => neighbor,
            None => {
                return Ok(());
            }
        };

        let data_pin_state = match get_pin_state(ctx, data_pin_neighbor, coord) {
            super::PinReading::Valid(x) => x,
            _ => super::PinState::Low,
        };

        let should_signal = ctx.game_map().mutate_block_atomically(coord, |block, _| {
            let variant = block.variant();
            let pending_count = variant >> 3;

            let new_variant = if clock_state == super::PinState::High {
                variant | DFF_GATE_INPUT_CLOCK_WAS_HIGH_VARIANT_BIT
            } else {
                variant & !DFF_GATE_INPUT_CLOCK_WAS_HIGH_VARIANT_BIT
            };
            if block.equals_ignore_variant(self.off) || block.equals_ignore_variant(self.on) {
                if pending_count > 8 {
                    *block = ctx.block_types().resolve_name(&self.broken).unwrap();
                    return Ok(false);
                }
                if variant != new_variant {
                    let new_pending_count = if clock_state == super::PinState::High {
                        pending_count + 1
                    } else {
                        pending_count
                    };
                    let new_variant_with_counter = (new_variant & 0x7) | (new_pending_count << 3);
                    *block = block.with_variant(new_variant_with_counter)?;
                    return Ok(clock_state == super::PinState::High);
                }
                return Ok(false);
            } else {
                // race condition, wrong block is there
                return Ok(false);
            }
        })?;

        if should_signal {
            let on = self.on;
            let off = self.off;
            ctx.run_deferred_delayed(Duration::from_millis(100), move |ctx| {
                delayed_edge(ctx, coord, on, off, data_pin_state)
            })
        }

        Ok(())
    }

    fn sample_pin(
        &self,
        ctx: &super::events::CircuitHandlerContext<'_>,
        coord: perovskite_core::coordinates::BlockCoordinate,
        destination: perovskite_core::coordinates::BlockCoordinate,
    ) -> super::PinState {
        let block = match ctx.game_map().try_get_block(coord) {
            Some(block) => block,
            None => return super::PinState::Low,
        };
        let self_variant = block.variant();
        if OUTPUT_CONNECTVIITY.eval(coord, self_variant) != Some(destination) {
            return super::PinState::Low;
        }

        if block.equals_ignore_variant(self.on) {
            super::PinState::High
        } else {
            super::PinState::Low
        }
    }

    fn on_overheat(
        &self,
        ctx: &perovskite_server::game_state::event::HandlerContext,
        coord: perovskite_core::coordinates::BlockCoordinate,
    ) {
        ctx.game_map().set_block(coord, &self.broken, None).unwrap();
    }
}

fn register_dff(builder: &mut GameBuilder) -> Result<()> {
    include_texture_bytes!(builder, DFF_PROPERTIES.off_texture, "textures/dff_top.png")?;
    include_texture_bytes!(
        builder,
        DFF_PROPERTIES.on_texture,
        "textures/dff_top_on.png"
    )?;

    register_gate(builder, DFF_PROPERTIES, |off, on| {
        Box::new(DffImpl {
            on,
            off,
            broken: FastBlockName::new(BROKEN_GATE.0.to_string()),
        })
    })?;
    Ok(())
}
