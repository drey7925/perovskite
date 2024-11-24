use crate::blocks::{
    AaBoxProperties, AxisAlignedBoxesAppearanceBuilder, BlockBuilder, RotationMode, TextureCropping,
};
use crate::circuits::events::{make_root_context, transmit_edge};
use crate::circuits::simple_blocks::SourceBlockCallbacks;
use crate::circuits::{
    gates, BlockConnectivity, CircuitBlockBuilder, CircuitBlockProperties, CircuitGameBuilder,
    PinState,
};
use crate::game_builder::{StaticBlockName, StaticTextureName};
use crate::include_texture_bytes;
use perovskite_core::constants::item_groups::HIDDEN_FROM_CREATIVE;
use std::time::Duration;

const SWITCH_OFF: StaticBlockName = StaticBlockName("circuits:switch_off");
const SWITCH_ON: StaticBlockName = StaticBlockName("circuits:switch_on");
const BUTTON_OFF: StaticBlockName = StaticBlockName("circuits:button_off");
const BUTTON_ON: StaticBlockName = StaticBlockName("circuits:button_on");
const SWITCH_OFF_TEXTURE: StaticTextureName = StaticTextureName("circuits:switch_off");
const SWITCH_ON_TEXTURE: StaticTextureName = StaticTextureName("circuits:switch_on");

const SWITCH_CONNECTIVITIES: [BlockConnectivity; 2] = [
    BlockConnectivity::rotated_nesw_with_variant(0, 0, 1, 0),
    BlockConnectivity::rotated_nesw_with_variant(0, -1, 1, 0),
];

pub(crate) fn register_switches(
    builder: &mut crate::game_builder::GameBuilder,
) -> anyhow::Result<()> {
    include_texture_bytes!(builder, SWITCH_OFF_TEXTURE, "textures/switch_off.png")?;
    include_texture_bytes!(builder, SWITCH_ON_TEXTURE, "textures/switch_on.png")?;

    let switch_off_name = builder.inner.blocks().make_block_name(SWITCH_OFF.0);
    let switch_on_name = builder.inner.blocks().make_block_name(SWITCH_ON.0);

    // TODO: The in-game appearance is visually ugly
    let switch_off_block = builder.add_block(
        BlockBuilder::new(SWITCH_OFF)
            .set_light_emission(0)
            .set_axis_aligned_boxes_appearance(
                AxisAlignedBoxesAppearanceBuilder::new()
                    .add_box(
                        AaBoxProperties::new(
                            gates::BOTTOM_TEX,
                            gates::BOTTOM_TEX,
                            gates::BOTTOM_TEX,
                            gates::BOTTOM_TEX,
                            SWITCH_OFF_TEXTURE,
                            gates::BOTTOM_TEX,
                            TextureCropping::AutoCrop,
                            RotationMode::RotateHorizontally,
                        ),
                        (-0.5, 0.5),
                        (-0.5, 0.5),
                        (0.25, 0.5),
                    )
                    .add_box(
                        AaBoxProperties::new_single_tex(
                            gates::BOTTOM_TEX,
                            TextureCropping::AutoCrop,
                            RotationMode::RotateHorizontally,
                        ),
                        (-0.125, 0.125),
                        (-0.375, -0.125),
                        (0.0, 0.25),
                    ),
            )
            .set_allow_light_propagation(true)
            .set_display_name("Switch")
            .add_modifier(Box::new(|block| {
                block.interact_key_handler = Some(Box::new(move |ctx, coord| {
                    let switch_off = ctx
                        .block_types()
                        .resolve_name(&switch_off_name)
                        .expect("Turned-off switch should be registered");
                    let switch_on = ctx
                        .block_types()
                        .resolve_name(&switch_on_name)
                        .expect("Turned-on switch should be registered");

                    let mut transitions: smallvec::SmallVec<[_; 2]> = smallvec::SmallVec::new();
                    ctx.game_map().mutate_block_atomically(coord, |id, _| {
                        let variant = id.variant();
                        if id.equals_ignore_variant(switch_off) {
                            *id = switch_on.with_variant_unchecked(variant);
                            for connectivity in SWITCH_CONNECTIVITIES {
                                if let Some(dest_coord) = connectivity.eval(coord, variant) {
                                    transitions.push((dest_coord, coord));
                                }
                            }
                        }
                        Ok(())
                    })?;

                    let cctx = make_root_context(&ctx);
                    for (dest_coord, src_coord) in transitions {
                        transmit_edge(&cctx, dest_coord, src_coord, PinState::High)?;
                    }
                    Ok(None)
                }))
            }))
            .register_circuit_callbacks(),
    )?;

    let switch_off_name = builder.inner.blocks().make_block_name(SWITCH_OFF.0);
    let switch_on_name = builder.inner.blocks().make_block_name(SWITCH_ON.0);
    let switch_on_block = builder.add_block(
        BlockBuilder::new(SWITCH_ON)
            .set_light_emission(0)
            .set_axis_aligned_boxes_appearance(
                AxisAlignedBoxesAppearanceBuilder::new()
                    .add_box(
                        AaBoxProperties::new(
                            gates::BOTTOM_TEX,
                            gates::BOTTOM_TEX,
                            gates::BOTTOM_TEX,
                            gates::BOTTOM_TEX,
                            SWITCH_ON_TEXTURE,
                            gates::BOTTOM_TEX,
                            TextureCropping::AutoCrop,
                            RotationMode::RotateHorizontally,
                        ),
                        (-0.5, 0.5),
                        (-0.5, 0.5),
                        (0.25, 0.5),
                    )
                    .add_box(
                        AaBoxProperties::new_single_tex(
                            gates::BOTTOM_TEX,
                            TextureCropping::AutoCrop,
                            RotationMode::RotateHorizontally,
                        ),
                        (-0.125, 0.125),
                        (0.125, 0.375),
                        (0.0, 0.25),
                    ),
            )
            .set_allow_light_propagation(true)
            .set_dropped_item(SWITCH_OFF.0, 1)
            .set_display_name("Switch")
            .add_item_group(HIDDEN_FROM_CREATIVE)
            .add_modifier(Box::new(|block| {
                block.interact_key_handler = Some(Box::new(move |ctx, coord| {
                    let switch_off = ctx
                        .block_types()
                        .resolve_name(&switch_off_name)
                        .expect("Turned-off switch should be registered");
                    let switch_on = ctx
                        .block_types()
                        .resolve_name(&switch_on_name)
                        .expect("Turned-on switch should be registered");
                    let mut transitions: smallvec::SmallVec<[_; 2]> = smallvec::SmallVec::new();
                    ctx.game_map().mutate_block_atomically(coord, |id, _| {
                        let variant = id.variant();
                        if id.equals_ignore_variant(switch_on) {
                            *id = switch_off.with_variant_unchecked(variant);
                            for connectivity in SWITCH_CONNECTIVITIES {
                                if let Some(dest_coord) = connectivity.eval(coord, variant) {
                                    transitions.push((dest_coord, coord));
                                }
                            }
                        }
                        Ok(())
                    })?;
                    let cctx = make_root_context(&ctx);
                    for (dest_coord, src_coord) in transitions {
                        transmit_edge(&cctx, dest_coord, src_coord, PinState::Low)?;
                    }
                    Ok(None)
                }))
            }))
            .register_circuit_callbacks(),
    )?;

    builder.define_circuit_callbacks(
        switch_off_block.id,
        Box::new(SourceBlockCallbacks(PinState::Low)),
        CircuitBlockProperties {
            connectivity: SWITCH_CONNECTIVITIES.to_vec(),
        },
    )?;

    builder.define_circuit_callbacks(
        switch_on_block.id,
        Box::new(SourceBlockCallbacks(PinState::High)),
        CircuitBlockProperties {
            connectivity: SWITCH_CONNECTIVITIES.to_vec(),
        },
    )?;

    let button_off_name = builder.inner.blocks().make_block_name(BUTTON_OFF.0);
    let button_on_name = builder.inner.blocks().make_block_name(BUTTON_ON.0);

    // TODO: The in-game appearance is visually ugly
    let button_off_block = builder.add_block(
        BlockBuilder::new(BUTTON_OFF)
            .set_light_emission(0)
            .set_axis_aligned_boxes_appearance(
                // TODO make a nicer appearance
                AxisAlignedBoxesAppearanceBuilder::new()
                    .add_box(
                        AaBoxProperties::new(
                            gates::BOTTOM_TEX,
                            gates::BOTTOM_TEX,
                            gates::BOTTOM_TEX,
                            gates::BOTTOM_TEX,
                            SWITCH_OFF_TEXTURE,
                            gates::BOTTOM_TEX,
                            TextureCropping::AutoCrop,
                            RotationMode::RotateHorizontally,
                        ),
                        (-0.5, 0.5),
                        (-0.5, 0.5),
                        (0.25, 0.5),
                    )
                    .add_box(
                        AaBoxProperties::new(
                            gates::BOTTOM_TEX,
                            gates::BOTTOM_TEX,
                            gates::BOTTOM_TEX,
                            gates::BOTTOM_TEX,
                            SWITCH_OFF_TEXTURE,
                            gates::BOTTOM_TEX,
                            TextureCropping::AutoCrop,
                            RotationMode::RotateHorizontally,
                        ),
                        (-0.125, 0.125),
                        (-0.375, 0.375),
                        (0.0, 0.25),
                    ),
            )
            .set_allow_light_propagation(true)
            .set_display_name("Button")
            .register_circuit_callbacks()
            .add_modifier(Box::new(|block| {
                block.interact_key_handler = Some(Box::new(move |ctx, coord| {
                    let button_off = ctx
                        .block_types()
                        .resolve_name(&button_off_name)
                        .expect("Turned-off button should be registered");
                    let button_on = ctx
                        .block_types()
                        .resolve_name(&button_on_name)
                        .expect("Turned-on button should be registered");

                    let mut transitions: smallvec::SmallVec<[_; 2]> = smallvec::SmallVec::new();
                    ctx.game_map().mutate_block_atomically(coord, |id, _| {
                        let variant = id.variant();
                        if id.equals_ignore_variant(button_off) {
                            *id = button_on.with_variant_unchecked(variant);
                            for connectivity in SWITCH_CONNECTIVITIES {
                                if let Some(dest_coord) = connectivity.eval(coord, variant) {
                                    transitions.push((dest_coord, coord));
                                }
                            }
                        }
                        Ok(())
                    })?;

                    let cctx = make_root_context(&ctx);
                    for (dest_coord, src_coord) in transitions {
                        transmit_edge(&cctx, dest_coord, src_coord, PinState::High)?;
                    }
                    ctx.run_deferred_delayed(Duration::from_millis(250), move |ctx| {
                        let mut transitions: smallvec::SmallVec<[_; 2]> = smallvec::SmallVec::new();
                        ctx.game_map().mutate_block_atomically(coord, |id, _| {
                            let variant = id.variant();
                            if id.equals_ignore_variant(button_on) {
                                *id = button_off.with_variant_unchecked(variant);
                                for connectivity in SWITCH_CONNECTIVITIES {
                                    if let Some(dest_coord) = connectivity.eval(coord, variant) {
                                        transitions.push((dest_coord, coord));
                                    }
                                }
                            }

                            Ok(())
                        })?;

                        let cctx = make_root_context(&ctx);
                        for (dest_coord, src_coord) in transitions {
                            transmit_edge(&cctx, dest_coord, src_coord, PinState::Low)?;
                        }

                        Ok(())
                    });
                    Ok(None)
                }))
            })),
    )?;

    let button_off_name = builder.inner.blocks().make_block_name(BUTTON_OFF.0);
    let button_on_name = builder.inner.blocks().make_block_name(BUTTON_ON.0);
    let button_on_block = builder.add_block(
        BlockBuilder::new(BUTTON_ON)
            .set_light_emission(0)
            .set_axis_aligned_boxes_appearance(
                AxisAlignedBoxesAppearanceBuilder::new()
                    .add_box(
                        AaBoxProperties::new(
                            gates::BOTTOM_TEX,
                            gates::BOTTOM_TEX,
                            gates::BOTTOM_TEX,
                            gates::BOTTOM_TEX,
                            SWITCH_ON_TEXTURE,
                            gates::BOTTOM_TEX,
                            TextureCropping::AutoCrop,
                            RotationMode::RotateHorizontally,
                        ),
                        (-0.5, 0.5),
                        (-0.5, 0.5),
                        (0.25, 0.5),
                    )
                    .add_box(
                        AaBoxProperties::new(
                            gates::BOTTOM_TEX,
                            gates::BOTTOM_TEX,
                            gates::BOTTOM_TEX,
                            gates::BOTTOM_TEX,
                            SWITCH_ON_TEXTURE,
                            gates::BOTTOM_TEX,
                            TextureCropping::AutoCrop,
                            RotationMode::RotateHorizontally,
                        ),
                        (-0.125, 0.125),
                        (-0.375, 0.375),
                        (0.125, 0.25),
                    ),
            )
            .set_allow_light_propagation(true)
            .set_dropped_item(BUTTON_OFF.0, 1)
            .set_display_name("Button")
            .add_item_group(HIDDEN_FROM_CREATIVE)
            .add_modifier(Box::new(|block| {
                block.interact_key_handler = Some(Box::new(move |ctx, coord| {
                    let button_off = ctx
                        .block_types()
                        .resolve_name(&button_off_name)
                        .expect("Turned-off button should be registered");
                    let button_on = ctx
                        .block_types()
                        .resolve_name(&button_on_name)
                        .expect("Turned-on button should be registered");
                    let mut transitions: smallvec::SmallVec<[_; 2]> = smallvec::SmallVec::new();
                    ctx.game_map().mutate_block_atomically(coord, |id, _| {
                        let variant = id.variant();
                        if id.equals_ignore_variant(button_on) {
                            *id = button_off.with_variant_unchecked(variant);
                            for connectivity in SWITCH_CONNECTIVITIES {
                                if let Some(dest_coord) = connectivity.eval(coord, variant) {
                                    transitions.push((dest_coord, coord));
                                }
                            }
                        }
                        Ok(())
                    })?;
                    let cctx = make_root_context(&ctx);
                    for (dest_coord, src_coord) in transitions {
                        transmit_edge(&cctx, dest_coord, src_coord, PinState::Low)?;
                    }
                    Ok(None)
                }))
            }))
            .register_circuit_callbacks(),
    )?;

    builder.define_circuit_callbacks(
        button_off_block.id,
        Box::new(SourceBlockCallbacks(PinState::Low)),
        CircuitBlockProperties {
            connectivity: SWITCH_CONNECTIVITIES.to_vec(),
        },
    )?;

    builder.define_circuit_callbacks(
        button_on_block.id,
        Box::new(SourceBlockCallbacks(PinState::High)),
        CircuitBlockProperties {
            connectivity: SWITCH_CONNECTIVITIES.to_vec(),
        },
    )?;

    Ok(())
}
