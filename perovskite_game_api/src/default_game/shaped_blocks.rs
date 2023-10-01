use crate::{
    blocks::{
        AaBoxProperties, AxisAlignedBoxesAppearanceBuilder, BlockBuilder, BuiltBlock, RotationMode, TextureCropping,
    },
    game_builder::{BlockName, FALLBACK_UNKNOWN_TEXTURE_NAME},
};
use anyhow::{Context, Result};
use perovskite_core::{
    protocol::{
        blocks::{block_type_def::RenderInfo, CubeRenderInfo},
        items::item_stack::QuantityType,
        render::TextureReference,
    },
};

use super::{recipes::RecipeSlot, DefaultGameBuilder};

fn convert_or_fallback(tex: &Option<TextureReference>) -> TextureReference {
    tex.clone()
        .map(Into::into)
        .unwrap_or(FALLBACK_UNKNOWN_TEXTURE_NAME.into())
}

/// Registers a new block that acts like stairs of the given block
pub fn make_stairs(
    game_builder: &mut DefaultGameBuilder,
    base: &BuiltBlock,
    craftable: bool,
) -> Result<BuiltBlock> {
    let base_item = base.item_name.0.clone();
    let built_block =
        make_derived_block_core(game_builder, base, "_stair", " Stair", |cube_appearance| {
            let aa_box_textures = AaBoxProperties::new(
                convert_or_fallback(&cube_appearance.tex_left),
                convert_or_fallback(&cube_appearance.tex_right),
                convert_or_fallback(&cube_appearance.tex_top),
                convert_or_fallback(&cube_appearance.tex_bottom),
                convert_or_fallback(&cube_appearance.tex_front),
                convert_or_fallback(&cube_appearance.tex_back),
                TextureCropping::AutoCrop,
                RotationMode::RotateHorizontally,
            );

            AxisAlignedBoxesAppearanceBuilder::new()
                .add_box(
                    aa_box_textures.clone(),
                    (-0.5, 0.5),
                    (-0.5, 0.0),
                    (-0.5, 0.5),
                )
                .add_box(aa_box_textures, (-0.5, 0.5), (-0.5, 0.5), (0.0, 0.5))
        })?;
    if craftable {
        let base_item = RecipeSlot::Exact(base_item);
        game_builder.register_crafting_recipe(
            [
                base_item.clone(),
                RecipeSlot::Empty,
                RecipeSlot::Empty,
                base_item.clone(),
                base_item.clone(),
                RecipeSlot::Empty,
                base_item.clone(),
                base_item.clone(),
                base_item.clone(),
            ],
            built_block.item_name.0.clone(),
            4,
            Some(QuantityType::Stack(256)),
        );
    }
    Ok(built_block)
}

/// Registers a new block that acts like a slab of the given block
pub fn make_slab(
    game_builder: &mut DefaultGameBuilder,
    base: &BuiltBlock,
    craftable: bool,
) -> Result<BuiltBlock> {
    let base_item = base.item_name.0.clone();
    let built_block =
        make_derived_block_core(game_builder, base, "_slab", " Slab", |cube_appearance| {
            let aa_box_textures = AaBoxProperties::new(
                convert_or_fallback(&cube_appearance.tex_left),
                convert_or_fallback(&cube_appearance.tex_right),
                convert_or_fallback(&cube_appearance.tex_top),
                convert_or_fallback(&cube_appearance.tex_bottom),
                convert_or_fallback(&cube_appearance.tex_front),
                convert_or_fallback(&cube_appearance.tex_back),
                TextureCropping::AutoCrop,
                RotationMode::RotateHorizontally,
            );

            AxisAlignedBoxesAppearanceBuilder::new().add_box(
                aa_box_textures,
                (-0.5, 0.5),
                (-0.5, 0.0),
                (-0.5, 0.5),
            )
        })?;
    if craftable {
        let base_item = RecipeSlot::Exact(base_item);
        game_builder.register_crafting_recipe(
            [
                RecipeSlot::Empty,
                RecipeSlot::Empty,
                RecipeSlot::Empty,
                RecipeSlot::Empty,
                RecipeSlot::Empty,
                RecipeSlot::Empty,
                base_item.clone(),
                base_item.clone(),
                base_item.clone(),
            ],
            built_block.item_name.0.clone(),
            6,
            Some(QuantityType::Stack(256)),
        );
    }
    Ok(built_block)
}

fn make_derived_block_core(
    game_builder: &mut DefaultGameBuilder,
    base: &BuiltBlock,
    short_suffix: &str,
    display_suffix: &str,
    appearance_builder: impl Fn(&CubeRenderInfo) -> AxisAlignedBoxesAppearanceBuilder,
) -> Result<BuiltBlock> {
    let block_type = game_builder
        .inner
        .inner
        .blocks()
        .get_block(&base.handle.0)?;
    let item = game_builder
        .inner
        .inner
        .items()
        .get_item(&base.item_name.0)
        .context("Base item not found")?;
    let appearance = match &block_type.0.client_info.render_info {
        Some(RenderInfo::Cube(cube_appearance)) => cube_appearance,
        Some(_) => return Err(anyhow::anyhow!("Base block must have a cube appearance.")),
        None => return Err(anyhow::anyhow!("Base block must have a render info.")),
    };
    let appearance = appearance_builder(appearance);
    let block_builder = BlockBuilder::new(BlockName(
        block_type.0.short_name().to_owned() + short_suffix,
    ))
    .set_axis_aligned_boxes_appearance(appearance)
    .set_display_name(item.proto.display_name.clone() + display_suffix)
    .set_allow_light_propagation(true)
    .add_block_groups(block_type.0.client_info.groups.iter().cloned());
    let built_block = game_builder.inner.add_block(block_builder)?;

    Ok(built_block)
}
