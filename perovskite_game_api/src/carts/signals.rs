use perovskite_core::{block_id::BlockId, chat::ChatMessage, protocol::render::DynamicCrop};
use rand::Rng;

use crate::{
    blocks::{AaBoxProperties, AxisAlignedBoxesAppearanceBuilder, BlockBuilder},
    game_builder::{StaticBlockName, StaticTextureName},
    include_texture_bytes,
};
use anyhow::Result;

pub(crate) const SIGNAL_BLOCK: StaticBlockName = StaticBlockName("carts:signal_block");
const SIGNAL_BLOCK_TEX_OFF: StaticTextureName = StaticTextureName("carts:signal_block_off");
const SIGNAL_BLOCK_TEX_ON: StaticTextureName = StaticTextureName("carts:signal_block_on");
const SIGNAL_SIDE_TOP_TEX: StaticTextureName = StaticTextureName("carts:signal_side_top");
const RAIL_TEST_TEX: StaticTextureName = StaticTextureName("carts:rail_test");

// Note that the two low bits of the signal variant are used for the direction it's facing
/// Set when the signal permits traffic
const VARIANT_PERMISSIVE_SIGNAL: u16 = 4;
/// Set when the signal cannot clear because there is traffic in the block
const VARIANT_RESTRICTIVE_TRAFFIC: u16 = 8;
/// Set when the signal is not clear for any reason (either conflicting traffic, no cart
/// requested it, external circuit signal is on)
const VARIANT_RESTRICTIVE: u16 = 16;
/// Set when an external circuit signal is on, meaning that this signal will not clear
/// for a player-requested reason
const VARIANT_RESTRICTIVE_EXTERNAL: u16 = 32;
/// Set when the signal indicates a turnout to the right
const VARIANT_RIGHT: u16 = 64;
/// Set when the signal indicates a turnout to the left
const VARIANT_LEFT: u16 = 128;
/// Set when the signal has routing rules in its extended data (not yet implemented)
const VARIANT_EXTENDED_ROUTING: u16 = 256;

pub(crate) fn register_signal_block(
    game_builder: &mut crate::game_builder::GameBuilder,
) -> Result<BlockId> {
    include_texture_bytes!(
        game_builder,
        SIGNAL_BLOCK_TEX_OFF,
        "textures/signal_off.png"
    )?;
    include_texture_bytes!(game_builder, SIGNAL_BLOCK_TEX_ON, "textures/signal_on.png")?;
    include_texture_bytes!(
        game_builder,
        SIGNAL_SIDE_TOP_TEX,
        "textures/signal_side_top.png"
    )?;

    include_texture_bytes!(game_builder, RAIL_TEST_TEX, "textures/rail_atlas_test.png")?;

    let signal_off_box = AaBoxProperties::new(
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_BLOCK_TEX_OFF,
        SIGNAL_SIDE_TOP_TEX,
        crate::blocks::TextureCropping::AutoCrop,
        crate::blocks::RotationMode::RotateHorizontally,
    );
    let signal_on_box = AaBoxProperties::new(
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_BLOCK_TEX_ON,
        SIGNAL_SIDE_TOP_TEX,
        crate::blocks::TextureCropping::AutoCrop,
        crate::blocks::RotationMode::RotateHorizontally,
    );

    let block = game_builder.add_block(
        BlockBuilder::new(SIGNAL_BLOCK)
            .set_axis_aligned_boxes_appearance(
                AxisAlignedBoxesAppearanceBuilder::new()
                    .add_box(
                        signal_off_box.clone(),
                        /* x= */ (-0.5, 0.5),
                        /* y= */ (-0.5, 0.5),
                        /* z= */ (0.0, 0.5),
                    )
                    .add_box_with_variant_mask(
                        signal_on_box.clone(),
                        /* x= */ (-3.0 / 32.0, 3.0 / 32.0),
                        /* y= */ (-11.0 / 32.0, -5.0 / 32.0),
                        /* z= */ (-0.025, 0.0),
                        VARIANT_PERMISSIVE_SIGNAL as u32,
                    )
                    .add_box_with_variant_mask(
                        signal_on_box.clone(),
                        /* x= */ (-11.0 / 32.0, -5.0 / 32.0),
                        /* y= */ (-11.0 / 32.0, -5.0 / 32.0),
                        /* z= */ (-0.025, 0.0),
                        VARIANT_LEFT as u32,
                    )
                    .add_box_with_variant_mask(
                        signal_on_box.clone(),
                        /* x= */ (5.0 / 32.0, 11.0 / 32.0),
                        /* y= */ (-11.0 / 32.0, -5.0 / 32.0),
                        /* z= */ (-0.025, 0.0),
                        VARIANT_RIGHT as u32,
                    )
                    .add_box_with_variant_mask(
                        signal_on_box.clone(),
                        /* x= */ (-3.0 / 32.0, 3.0 / 32.0),
                        /* y= */ (-3.0 / 32.0, 3.0 / 32.0),
                        /* z= */ (-0.025, 0.0),
                        VARIANT_RESTRICTIVE_TRAFFIC as u32,
                    )
                    .add_box_with_variant_mask(
                        signal_on_box.clone(),
                        /* x= */ (-3.0 / 32.0, 3.0 / 32.0),
                        /* y= */ (5.0 / 32.0, 11.0 / 32.0),
                        /* z= */ (-0.025, 0.0),
                        VARIANT_RESTRICTIVE as u32,
                    )
                    .add_box_with_variant_mask(
                        signal_on_box.clone(),
                        /* x= */ (5.0 / 32.0, 11.0 / 32.0),
                        /* y= */ (5.0 / 32.0, 11.0 / 32.0),
                        /* z= */ (-0.025, 0.0),
                        VARIANT_RESTRICTIVE_EXTERNAL as u32,
                    ),
            )
            .set_allow_light_propagation(true)
            .set_light_emission(8)
            .add_modifier(Box::new(|bt| {
                bt.interact_key_handler = Some(Box::new(|ctx, coord| {
                    let mut rng = rand::thread_rng();
                    let variant = rng.gen_range(0..4096);
                    ctx.game_map().mutate_block_atomically(coord, |b, _ext| {
                        // TODO this should check equals_ignore_variant
                        let old_variant = b.variant();
                        let new_variant = (variant & !3) | (old_variant & 3);
                        *b = b.with_variant(new_variant)?;
                        Ok(())
                    })?;
                    ctx.initiator()
                        .send_chat_message(ChatMessage::new("[RNG]", format!("{:b}", variant)))?;
                    Ok(None)
                }));
            }))
            .add_item_modifier(Box::new(|it| {
                let old_place_handler = it.place_handler.take().unwrap();
                it.place_handler = Some(Box::new(move |ctx, coord, anchor, stack| {
                    let result = old_place_handler(ctx, coord, anchor, stack)?;
                    ctx.game_map().mutate_block_atomically(coord, |b, _ext| {
                        *b = b.with_variant(b.variant() | VARIANT_RESTRICTIVE)?;
                        Ok(())
                    })?;
                    Ok(result)
                }))
            })),
    )?;
    let rail_test_box = AaBoxProperties::new(
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_SIDE_TOP_TEX,
        RAIL_TEST_TEX,
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_SIDE_TOP_TEX,
        crate::blocks::TextureCropping::AutoCrop,
        crate::blocks::RotationMode::RotateHorizontally,
    );
    game_builder.add_block(
        BlockBuilder::new(StaticBlockName("carts:rail_test"))
            .set_axis_aligned_boxes_appearance(AxisAlignedBoxesAppearanceBuilder::new().add_box(
                rail_test_box,
                (-0.5, 0.5),
                (-0.5, -0.4),
                (-0.5, 0.5),
            ))
            .add_modifier(Box::new(|bt| {
                bt.interact_key_handler = Some(Box::new(|ctx, coord| {
                    let mut rng = rand::thread_rng();
                    let variant = rng.gen_range(0..4096);
                    ctx.game_map().mutate_block_atomically(coord, |b, _ext| {
                        // TODO this should check equals_ignore_variant
                        let old_variant = b.variant();
                        let new_variant = (variant & !3) | (old_variant & 3);
                        *b = b.with_variant(new_variant)?;

                        ctx.initiator().send_chat_message(ChatMessage::new("[RNG]", format!("{:b}", variant)))?;

                        Ok(())
                    })?;
                    Ok(None)
                }));
                let ri = bt.client_info.render_info.as_mut().unwrap();
                match ri {
                    perovskite_core::protocol::blocks::block_type_def::RenderInfo::AxisAlignedBoxes(aabb) => {
                        aabb.boxes.iter_mut().for_each(|b| {
                            b.tex_top.as_mut().unwrap().crop.as_mut().unwrap().dynamic = Some(
                                DynamicCrop {
                                    x_selector_bits: 0b0000_0000_0011_1100,
                                    y_selector_bits: 0b0000_0011_1100_0000,
                                    x_cells: 16,
                                    y_cells: 11,
                                    flip_x_bit: 0b0000_0100_0000_0000,
                                    flip_y_bit: 0,
                                }
                            )
                        })
                    },
                    _ => unreachable!()
                }
            })),
    )?;

    Ok(block.id)
}
