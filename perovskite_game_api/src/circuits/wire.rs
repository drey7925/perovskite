// Unlike most circuits-related blocks which either initiate effects or react to effects
// wires need to actually search through their topology to propagate effects near-instantaneously.
// Therefore there's a lot of special logic in here that won't apply to other blocks.

use std::collections::VecDeque;

use anyhow::{Context, Result};
use perovskite_core::{block_id::BlockId, coordinates::BlockCoordinate};
use perovskite_server::game_state::event::HandlerContext;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    blocks::{AaBoxProperties, AxisAlignedBoxesAppearanceBuilder, BlockBuilder},
    game_builder::{GameBuilder, StaticBlockName, TextureName},
    include_texture_bytes,
};

use super::{
    get_live_connectivities, BlockConnectivity, CircuitBlockBuilder, CircuitBlockCallbacks,
    CircuitBlockProperties, CircuitGameBuilerPrivate, CircuitGameStateExtension,
    CircuitHandlerContext, PinState,
};

pub const WIRE_BLOCK_OFF: StaticBlockName = StaticBlockName("circuits:wire_off");
pub const WIRE_BLOCK_ON: StaticBlockName = StaticBlockName("circuits:wire_on");
pub const WIRE_TEXTURE_OFF: TextureName = TextureName("circuits:wire_off");
pub const WIRE_TEXTURE_ON: TextureName = TextureName("circuits:wire_on");

const VARIANT_XPLUS: u32 = 1;
const VARIANT_XMINUS: u32 = 2;
const VARIANT_ZPLUS: u32 = 4;
const VARIANT_ZMINUS: u32 = 8;
const VARIANT_XPLUS_ABOVE: u32 = 16;
const VARIANT_XMINUS_ABOVE: u32 = 32;
const VARIANT_ZPLUS_ABOVE: u32 = 64;
const VARIANT_ZMINUS_ABOVE: u32 = 128;

// IDs are the variants that should be enabled when this connection is made
const WIRE_CONNECTIVITY_RULES: [BlockConnectivity; 12] = [
    // Connections on the same level
    BlockConnectivity::unrotated(1, 0, 0, VARIANT_XPLUS),
    BlockConnectivity::unrotated(-1, 0, 0, VARIANT_XMINUS),
    BlockConnectivity::unrotated(0, 0, 1, VARIANT_ZPLUS),
    BlockConnectivity::unrotated(0, 0, -1, VARIANT_ZMINUS),
    // Connections to one above
    BlockConnectivity::unrotated(1, 1, 0, VARIANT_XPLUS_ABOVE),
    BlockConnectivity::unrotated(-1, 1, 0, VARIANT_XMINUS_ABOVE),
    BlockConnectivity::unrotated(0, 1, 1, VARIANT_ZPLUS_ABOVE),
    BlockConnectivity::unrotated(0, 1, -1, VARIANT_ZMINUS_ABOVE),
    // Connections to one below
    // We don't need another variant because these don't draw anything different
    // from the normal same-height variants, but we do need to allow connectivity to
    // one spot down
    BlockConnectivity::unrotated(1, -1, 0, VARIANT_XPLUS),
    BlockConnectivity::unrotated(-1, -1, 0, VARIANT_XMINUS),
    BlockConnectivity::unrotated(0, -1, 1, VARIANT_ZPLUS),
    BlockConnectivity::unrotated(0, -1, -1, VARIANT_ZMINUS),
];

fn build_wire_aabox(texture: TextureName) -> AxisAlignedBoxesAppearanceBuilder {
    let prototype = AaBoxProperties::new_single_tex(
        texture,
        crate::blocks::TextureCropping::AutoCrop,
        crate::blocks::RotationMode::None,
    );
    AxisAlignedBoxesAppearanceBuilder::new()
        .add_box(prototype.clone(), (-0.1, 0.1), (-0.5, -0.4), (-0.1, 0.1))
        .add_box_with_variant_mask(
            prototype.clone(),
            (0.1, 0.5),
            (-0.5, -0.4),
            (-0.1, 0.1),
            VARIANT_XPLUS | VARIANT_XPLUS_ABOVE,
        )
        .add_box_with_variant_mask(
            prototype.clone(),
            (-0.5, -0.1),
            (-0.5, -0.4),
            (-0.1, 0.1),
            VARIANT_XMINUS | VARIANT_XMINUS_ABOVE,
        )
        .add_box_with_variant_mask(
            prototype.clone(),
            (-0.1, 0.1),
            (-0.5, -0.4),
            (0.1, 0.5),
            VARIANT_ZPLUS | VARIANT_ZPLUS_ABOVE,
        )
        .add_box_with_variant_mask(
            prototype.clone(),
            (-0.1, 0.1),
            (-0.5, -0.4),
            (-0.5, -0.1),
            VARIANT_ZMINUS | VARIANT_ZMINUS_ABOVE,
        )
        .add_box_with_variant_mask(
            prototype.clone(),
            (0.4, 0.5),
            (-0.5, 0.5),
            (-0.1, 0.1),
            VARIANT_XPLUS_ABOVE,
        )
        .add_box_with_variant_mask(
            prototype.clone(),
            (-0.5, -0.4),
            (-0.5, 0.5),
            (-0.1, 0.1),
            VARIANT_XMINUS_ABOVE,
        )
        .add_box_with_variant_mask(
            prototype.clone(),
            (-0.1, 0.1),
            (-0.5, 0.5),
            (0.4, 0.5),
            VARIANT_ZPLUS_ABOVE,
        )
        .add_box_with_variant_mask(
            prototype,
            (-0.1, 0.1),
            (-0.5, 0.5),
            (-0.5, -0.4),
            VARIANT_ZMINUS_ABOVE,
        )
}

pub(crate) fn register_wire(builder: &mut GameBuilder) -> Result<()> {
    include_texture_bytes!(builder, WIRE_TEXTURE_OFF, "textures/wire_off.png")?;
    include_texture_bytes!(builder, WIRE_TEXTURE_ON, "textures/wire_on.png")?;

    // todo bigger hitbox?
    let wire_off = builder.add_block(
        BlockBuilder::new(WIRE_BLOCK_OFF)
            .set_allow_light_propagation(true)
            .set_axis_aligned_boxes_appearance(build_wire_aabox(WIRE_TEXTURE_OFF))
            .set_display_name("Digital wire")
            .set_inventory_texture(WIRE_TEXTURE_OFF)
            .register_circuit_callbacks(),
    )?;
    let wire_on = builder.add_block(
        BlockBuilder::new(WIRE_BLOCK_ON)
            .set_allow_light_propagation(true)
            .set_axis_aligned_boxes_appearance(build_wire_aabox(WIRE_TEXTURE_ON))
            .set_display_name("Digital wire")
            .set_inventory_texture(WIRE_TEXTURE_ON)
            .set_dropped_item(WIRE_BLOCK_OFF.0, 1)
            .set_light_emission(2)
            .register_circuit_callbacks(),
    )?;

    builder.register_wire_private(
        wire_on,
        wire_off,
        CircuitBlockProperties {
            connectivity: WIRE_CONNECTIVITY_RULES.to_vec(),
        },
    )?;

    Ok(())
}

pub(crate) struct WireCallbacksImpl {
    pub(crate) base_id: BlockId,
    pub(crate) state: PinState,
}
impl CircuitBlockCallbacks for WireCallbacksImpl {
    fn update_connectivity(
        &self,
        ctx: &CircuitHandlerContext<'_>,
        coord: BlockCoordinate,
    ) -> Result<()> {
        let mut resulting_variant = 0;

        for (connectivity, _) in get_live_connectivities(ctx, coord) {
            resulting_variant |= connectivity.id;
        }
        
        let extension = ctx.circuits_ext();
        let on_base_id = extension.basic_wire_on.0;
        let off_base_id = extension.basic_wire_off.0;
        ctx.game_map().mutate_block_atomically(coord, |block, _| {
            // It's important to check against both the on and off base ids
            // A state transition might have happened from another callback
            if block.base_id() == off_base_id || block.base_id() == on_base_id {
                *block = block
                    .with_variant(resulting_variant as u16)
                    .context("Invalid variant generated")?;
            }

            Ok(())
        })?;
        Ok(())
    }

    fn sample_pin(
        &self,
        _ctx: &CircuitHandlerContext<'_>,
        _coord: BlockCoordinate,
        _destination: BlockCoordinate,
    ) -> super::PinState {
        self.state
    }
}

pub(crate) fn recalculate_wire(
    ctx: &CircuitHandlerContext<'_>,
    // The first block in the wire that got signalled
    first_wire: BlockCoordinate,
    // The coordinate of the block that signalled us
    who_signalled: BlockCoordinate,
    new_state: super::PinState,
) -> Result<()> {
    // TODO: use the edge type as an optimization hint
    // Essentially, do a breadth-first search of the wire, starting at first_wire. Signal all
    // blocks that are attached and undergo a transition, with the exception of the one that
    // signalled us on the same exact port.

    let circuits_ext = ctx.circuits_ext();

    // Visited blocks that we need to signal: (dest, from) -> callback to signal
    let mut need_transition_signals: FxHashMap<
        (BlockCoordinate, BlockCoordinate),
        &Box<dyn CircuitBlockCallbacks>,
    > = FxHashMap::default();
    // Visited wires
    let mut visited: FxHashSet<BlockCoordinate> = FxHashSet::default();
    // (dest, prev)
    // For a wire, we just explore dest
    // For a non-wire block, we sample that block's signal driven into dest
    let mut queue: VecDeque<(BlockCoordinate, BlockCoordinate)> = VecDeque::new();

    queue.push_back((first_wire, who_signalled));

    let mut any_driven_high = false;
    while let Some((coord, prev)) = queue.pop_front() {
        let block = match ctx.game_map().try_get_block(coord) {
            Some(block) => block,
            None => {
                continue;
            }
        };

        if block.base_id() == circuits_ext.basic_wire_off.0
            || block.base_id() == circuits_ext.basic_wire_on.0
        {
            if visited.contains(&coord) {
                continue;
            }
            visited.insert(coord);
            for (_, neighbor) in get_live_connectivities(ctx, coord) {
                queue.push_back((neighbor, coord));
            }
        } else {
            // Not a wire.
            let callbacks = match circuits_ext.callbacks.get(&block.base_id()) {
                Some(callbacks) => callbacks,
                None => {
                    continue;
                }
            };
            // prev is the wire we just explored
            let drive = callbacks.sample_pin(ctx, coord, prev);
            if drive == super::PinState::High {
                any_driven_high = true;
            }

            // todo optimize based on the actual transition
            need_transition_signals.insert((coord, prev), callbacks);
        }
    }
    for coord in visited {
        ctx.game_map().mutate_block_atomically(coord, |block, _| {
            if any_driven_high && block.base_id() == circuits_ext.basic_wire_off.0 {
                *block = circuits_ext.basic_wire_on.with_variant(block.variant())?;
            }
            if (!any_driven_high) && block.base_id() == circuits_ext.basic_wire_on.0 {
                *block = circuits_ext.basic_wire_off.with_variant(block.variant())?;
            }
            Ok(())
        })?;
    }

    let pin_state = if any_driven_high {
        super::PinState::High
    } else {
        super::PinState::Low
    };

    for ((coord, prev), callbacks) in need_transition_signals {
        callbacks.on_incoming_edge(ctx, coord, prev, pin_state)?;
    }

    Ok(())
}
