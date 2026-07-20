use crate::{
    blocks::{
        AaBoxProperties, AxisAlignedBoxesAppearanceBuilder, BlockBuilder, BuiltBlock, RotationMode,
        TextureCropping,
    },
    default_game::block_groups::{FENCES, SLABS, STAIRS, VARIANT_ENCODES_PLACER},
    game_builder::{BlockName, GameBuilder, FALLBACK_UNKNOWN_TEXTURE_NAME},
};
use anyhow::{Context, Result};
use perovskite_core::protocol::{
    blocks::{block_type_def::RenderInfo, CubeRenderInfo},
    items::item_stack::QuantityType,
    render::TextureReference,
};
use perovskite_core::render_selector;

use super::{recipes::RecipeSlot, DefaultGameBuilder};

fn convert_or_fallback(tex: &Option<TextureReference>) -> TextureReference {
    tex.clone()
        .map(Into::into)
        .unwrap_or(FALLBACK_UNKNOWN_TEXTURE_NAME.into())
}

/// Registers a new block that acts like stairs of the given block
pub fn make_stairs(
    game_builder: &mut GameBuilder,
    base: &BuiltBlock,
    craftable: bool,
) -> Result<BuiltBlock> {
    let base_item = base.item_name.0.clone();
    let built_block = make_derived_block_core(
        game_builder,
        base,
        "_stair",
        " Stair",
        &[STAIRS],
        |cube_appearance| {
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
        },
        false,
    )?;
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
            false,
        );
        game_builder.register_crafting_recipe(
            [
                RecipeSlot::Empty,
                RecipeSlot::Empty,
                base_item.clone(),
                RecipeSlot::Empty,
                base_item.clone(),
                base_item.clone(),
                base_item.clone(),
                base_item.clone(),
                base_item.clone(),
            ],
            built_block.item_name.0.clone(),
            4,
            Some(QuantityType::Stack(256)),
            false,
        );
    }
    Ok(built_block)
}

/// Registers a new block that acts like a slab of the given block
pub fn make_slab(
    game_builder: &mut GameBuilder,
    base: &BuiltBlock,
    craftable: bool,
) -> Result<BuiltBlock> {
    let base_item = base.item_name.0.clone();
    let built_block = make_derived_block_core(
        game_builder,
        base,
        "_slab",
        " Slab",
        &[SLABS],
        |cube_appearance| {
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
        },
        false,
    )?;
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
            false,
        );
    }
    Ok(built_block)
}

/// Registers a new block that acts like a fence made of the given block.
///
/// The fence is a central post plus four horizontal arms. The arms appear
/// automatically when a connecting neighbor (a solid block, or another fence of the
/// same type) is present in the corresponding direction: they are conditioned on the
/// neighbor-presence bits of the client's render selector (see
/// [`render_selector`]), so no server-side connectivity bookkeeping is needed.
pub fn make_fence(
    game_builder: &mut GameBuilder,
    base: &BuiltBlock,
    craftable: bool,
) -> Result<BuiltBlock> {
    let base_item = base.item_name.0.clone();
    let built_block = make_derived_block_core(
        game_builder,
        base,
        "_fence",
        " Fence",
        &[FENCES],
        |cube_appearance| {
            let aa_box_textures = AaBoxProperties::new(
                convert_or_fallback(&cube_appearance.tex_left),
                convert_or_fallback(&cube_appearance.tex_right),
                convert_or_fallback(&cube_appearance.tex_top),
                convert_or_fallback(&cube_appearance.tex_bottom),
                convert_or_fallback(&cube_appearance.tex_front),
                convert_or_fallback(&cube_appearance.tex_back),
                TextureCropping::AutoCrop,
                RotationMode::None,
            );

            let mut builder = AxisAlignedBoxesAppearanceBuilder::new()
                // Central post, always present
                .add_box(
                    aa_box_textures.clone(),
                    (-0.125, 0.125),
                    (-0.5, 0.5),
                    (-0.125, 0.125),
                );
            // Two rails per direction, shown only when the neighbor in that direction
            // connects.
            for (x, z, neighbor_bit) in [
                (
                    (0.125, 0.5),
                    (-0.0625, 0.0625),
                    render_selector::NEIGHBOR_XPLUS,
                ),
                (
                    (-0.5, -0.125),
                    (-0.0625, 0.0625),
                    render_selector::NEIGHBOR_XMINUS,
                ),
                (
                    (-0.0625, 0.0625),
                    (0.125, 0.5),
                    render_selector::NEIGHBOR_ZPLUS,
                ),
                (
                    (-0.0625, 0.0625),
                    (-0.5, -0.125),
                    render_selector::NEIGHBOR_ZMINUS,
                ),
            ] {
                for y in [(0.125, 0.3125), (-0.3125, -0.125)] {
                    builder = builder.add_box_with_variant_mask(
                        aa_box_textures.clone(),
                        x,
                        y,
                        z,
                        neighbor_bit,
                    );
                }
            }
            builder
        },
        true,
    )?;
    if craftable {
        let base_item = RecipeSlot::Exact(base_item);
        let stick = RecipeSlot::Exact(super::foliage::STICK_ITEM.0.to_string());
        game_builder.register_crafting_recipe(
            [
                RecipeSlot::Empty,
                RecipeSlot::Empty,
                RecipeSlot::Empty,
                base_item.clone(),
                stick.clone(),
                base_item.clone(),
                base_item.clone(),
                stick.clone(),
                base_item.clone(),
            ],
            built_block.item_name.0.clone(),
            3,
            Some(QuantityType::Stack(256)),
            false,
        );
    }
    Ok(built_block)
}

fn make_derived_block_core(
    game_builder: &mut GameBuilder,
    base: &BuiltBlock,
    short_suffix: &str,
    display_suffix: &str,
    extra_groups: &[&str],
    appearance_builder: impl Fn(&CubeRenderInfo) -> AxisAlignedBoxesAppearanceBuilder,
    allow_weather_propagation: bool,
) -> Result<BuiltBlock> {
    let block_type = game_builder.inner.blocks().get_block(base.id)?;
    let item = game_builder
        .inner
        .items()
        .get_item(&base.item_name.0)
        .context("Base item not found")?;
    let appearance = match &block_type.0.client_info.render_info {
        Some(RenderInfo::Cube(cube_appearance)) => cube_appearance,
        Some(_) => {
            return Err(anyhow::anyhow!(
                "Base block must have a cube appearance as its render info."
            ))
        }
        None => return Err(anyhow::anyhow!("Base block must have a render info.")),
    };
    let appearance = appearance_builder(appearance);
    let block_builder = BlockBuilder::new(BlockName(
        block_type.0.short_name().to_owned() + short_suffix,
    ))
    .set_axis_aligned_boxes_appearance(appearance)
    .set_display_name(item.proto.display_name.clone() + display_suffix)
    .set_allow_light_propagation(true)
    .set_allow_weather_propagation(allow_weather_propagation)
    .add_block_groups(
        block_type
            .0
            .client_info
            .groups
            .iter()
            .filter(|g| g.as_str() != VARIANT_ENCODES_PLACER)
            .cloned(),
    )
    .add_block_groups(extra_groups.iter().map(|s| s.to_string()))
    .set_track_placer();
    let built_block = game_builder.add_block(block_builder)?;

    Ok(built_block)
}
