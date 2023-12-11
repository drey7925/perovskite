// Unlike most circuits-related blocks which either initiate effects or react to effects
// wires need to actually search through their topology to propagate effects near-instantaneously.
// Therefore there's a lot of special logic in here that won't apply to other blocks.

use anyhow::{Context, Result};
use perovskite_core::{block_id::BlockId, coordinates::BlockCoordinate};
use perovskite_server::game_state::event::HandlerContext;

use crate::{
    blocks::{AaBoxProperties, AxisAlignedBoxesAppearanceBuilder, BlockBuilder},
    game_builder::{GameBuilder, StaticBlockName, TextureName},
    include_texture_bytes,
};

use super::{
    BlockConnectivity, CircuitBlockBuilder, CircuitBlockCallbacks, CircuitBlockProperties,
    CircuitGameBuilerPrivate, CircuitGameStateExtension, CircuitHandlerContext,
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

const WIRE_CONNECTIVITY_RULES: [(BlockConnectivity, u32); 12] = [
    // Connections on the same level
    (BlockConnectivity::unrotated(1, 0, 0), VARIANT_XPLUS),
    (BlockConnectivity::unrotated(-1, 0, 0), VARIANT_XMINUS),
    (BlockConnectivity::unrotated(0, 0, 1), VARIANT_ZPLUS),
    (BlockConnectivity::unrotated(0, 0, -1), VARIANT_ZMINUS),
    // Connections to one above
    (BlockConnectivity::unrotated(1, 1, 0), VARIANT_XPLUS_ABOVE),
    (BlockConnectivity::unrotated(-1, 1, 0), VARIANT_XMINUS_ABOVE),
    (BlockConnectivity::unrotated(0, 1, 1), VARIANT_ZPLUS_ABOVE),
    (BlockConnectivity::unrotated(0, 1, -1), VARIANT_ZMINUS_ABOVE),
    // Connections to one below
    // We don't need another variant because these don't draw anything different
    // from the normal same-height variants, but we do need to allow connectivity to
    // one spot down
    (BlockConnectivity::unrotated(1, -1, 0), VARIANT_XPLUS),
    (BlockConnectivity::unrotated(-1, -1, 0), VARIANT_XMINUS),
    (BlockConnectivity::unrotated(0, -1, 1), VARIANT_ZPLUS),
    (BlockConnectivity::unrotated(0, -1, -1), VARIANT_ZMINUS),
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
        wire_off,
        wire_on,
        CircuitBlockProperties {
            connectivity: WIRE_CONNECTIVITY_RULES.iter().map(|(c, _)| *c).collect(),
        },
    )?;

    Ok(())
}

pub(crate) fn update_wire_connectivity(
    ctx: &CircuitHandlerContext<'_>,
    coord: BlockCoordinate,
    base_id: BlockId,
) -> Result<()> {
    let mut resulting_variant = 0;
    let circuit_extension = ctx.inner.extension::<CircuitGameStateExtension>().unwrap();

    for (connectivity, self_variant) in WIRE_CONNECTIVITY_RULES {
        // The variant is irrelevant for wires
        let neighbor_coord = match connectivity.eval(coord, 0) {
            Some(c) => c,
            None => continue,
        };
        let neighbor_block = match ctx.inner.game_map().try_get_block(neighbor_coord) {
            Some(b) => b,
            None => continue,
        };
        let neighbor_properties = match circuit_extension
            .basic_properties
            .get(&neighbor_block.base_id())
        {
            Some(p) => p,
            None => continue,
        };
        if neighbor_properties
            .connectivity
            .iter()
            .any(|x| x.eval(neighbor_coord, neighbor_block.variant()) == Some(coord))
        {
            resulting_variant |= self_variant;
        }
    }
    ctx.game_map().compare_and_set_block_predicate(
        coord,
        |block, _, _| Ok(block.base_id() == base_id.base_id()),
        base_id
            .with_variant(resulting_variant as u16)
            .context("Invalid variant generated")?,
        None,
    )?;
    Ok(())
}

pub(crate) struct WireCallbacksImpl {
    pub(crate) base_id: BlockId,
}
impl CircuitBlockCallbacks for WireCallbacksImpl {
    fn update_connectivity(
        &self,
        ctx: &CircuitHandlerContext<'_>,
        coord: BlockCoordinate,
    ) -> Result<()> {
        update_wire_connectivity(ctx, coord, self.base_id)
    }
}
