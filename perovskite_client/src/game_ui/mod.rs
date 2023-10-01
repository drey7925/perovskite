use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use image::{DynamicImage, RgbImage};
use perovskite_core::constants::textures::FALLBACK_UNKNOWN_TEXTURE;
use texture_packer::{importer::ImageImporter, Rect, TexturePacker};

use anyhow::{Context, Error, Result};

use crate::{
    block_renderer::{AsyncTextureLoader, BlockRenderer},
    game_state::items::ClientItemManager,
    vulkan::{mini_renderer::MiniBlockRenderer, Texture2DHolder, VulkanContext},
};

use self::{egui_ui::EguiUi, hud::GameHud};

pub(crate) mod egui_ui;
pub(crate) mod hud;

pub(crate) async fn make_uis<T>(
    item_defs: Arc<ClientItemManager>,
    texture_loader: T,
    ctx: Arc<VulkanContext>,
    block_renderer: &BlockRenderer,
) -> Result<(hud::GameHud, egui_ui::EguiUi)>
where
    T: AsyncTextureLoader,
{
    let (texture_atlas, texture_coords) =
        build_texture_atlas(&item_defs, texture_loader, ctx, block_renderer).await?;

    let hud = GameHud {
        texture_coords: texture_coords.clone(),
        texture_atlas: texture_atlas.clone(),
        item_defs: item_defs.clone(),
        last_size: (0, 0),
        hotbar_slot: 0,
        crosshair_draw_call: None,
        hotbar_draw_call: None,
        hotbar_view_id: None,
        fps_counter: fps_counter::FPSCounter::new(),
    };

    let egui_ui = EguiUi::new(texture_atlas, texture_coords, item_defs.clone());

    Ok((hud, egui_ui))
}

fn pack_tex(
    texture_packer: &mut TexturePacker<'_, DynamicImage, String>,
    name: &str,
    img: DynamicImage,
) -> Result<()> {
    texture_packer
        .pack_own(name.into(), img)
        .map_err(|x| Error::msg(format!("Texture pack for {} failed: {:?}", name, x)))
}

async fn build_texture_atlas<T>(
    item_defs: &ClientItemManager,
    mut texture_loader: T,
    ctx: Arc<VulkanContext>,
    block_renderer: &BlockRenderer,
) -> Result<(Arc<Texture2DHolder>, HashMap<String, Rect>)>
where
    T: AsyncTextureLoader,
{
    let mut all_texture_names = HashSet::new();
    let mut all_rendered_blocks = HashSet::new();
    for def in item_defs.all_item_defs() {
        if !def.block_apperance.is_empty() {
            all_rendered_blocks.insert(def.block_apperance.clone());
        } else if let Some(tex) = &def.inventory_texture {
            all_texture_names.insert(tex.texture_name.clone());
        }
    }

    let all_texture_names = item_defs
        .all_item_defs()
        .flat_map(|x| &x.inventory_texture)
        .map(|x| x.texture_name.clone())
        .collect::<HashSet<_>>();
    let mut renderer = MiniBlockRenderer::new(
        ctx.clone(),
        [128, 128],
        block_renderer.atlas(),
        block_renderer.block_types().air_block(),
    )?;

    let mut simple_textures = HashMap::new();
    let mut rendered_block_textures = HashMap::new();
    for name in all_texture_names {
        let texture = texture_loader.load_texture(&name).await?;
        simple_textures.insert(name, texture);
    }
    for block_name in all_rendered_blocks {
        let block_tex = renderer.render(&block_name, block_renderer)?.unwrap();
        rendered_block_textures.insert(block_name, block_tex);
    }

    let config = texture_packer::TexturePackerConfig {
        // todo tweak these or make into a setting
        allow_rotation: false,
        max_width: 1024,
        max_height: 1024,
        border_padding: 2,
        texture_padding: 2,
        texture_extrusion: 2,
        trim: false,
        texture_outlines: false,
    };
    let mut texture_packer = texture_packer::TexturePacker::new_skyline(config);

    pack_tex(
        &mut texture_packer,
        CROSSHAIR,
        ImageImporter::import_from_memory(include_bytes!("crosshair.png")).unwrap(),
    )?;
    pack_tex(
        &mut texture_packer,
        UNKNOWN_TEXTURE,
        ImageImporter::import_from_memory(include_bytes!("../block_unknown.png")).unwrap(),
    )?;
    pack_tex(
        &mut texture_packer,
        FRAME_SELECTED,
        ImageImporter::import_from_memory(include_bytes!("frame_selected.png")).unwrap(),
    )?;
    pack_tex(
        &mut texture_packer,
        FRAME_UNSELECTED,
        ImageImporter::import_from_memory(include_bytes!("frame_unselected.png")).unwrap(),
    )?;
    pack_tex(
        &mut texture_packer,
        DIGIT_ATLAS,
        ImageImporter::import_from_memory(include_bytes!("digit_atlas.png")).unwrap(),
    )?;
    pack_tex(
        &mut texture_packer,
        TEST_ITEM,
        ImageImporter::import_from_memory(include_bytes!("testonly_pickaxe.png")).unwrap(),
    )?;

    const COLOR_SCALE_R: [u8; 8] = [255, 255, 255, 255, 191, 128, 64, 0];
    const COLOR_SCALE_G: [u8; 8] = [0, 64, 128, 191, 255, 255, 255, 255];
    for i in 0..8 {
        let mut image = RgbImage::new(1, 1);
        image.put_pixel(0, 0, image::Rgb([COLOR_SCALE_R[i], COLOR_SCALE_G[i], 0]));
        texture_packer
            .pack_own(
                format!("builtin:wear_{}", i),
                DynamicImage::ImageRgb8(image),
            )
            .map_err(|x| Error::msg(format!("Texture pack failed: {:?}", x)))?;
    }

    for (name, texture) in simple_textures {
        pack_tex(
            &mut texture_packer,
            &("simple:".to_string() + &name),
            texture,
        )?;
    }
    for (name, texture) in rendered_block_textures {
        pack_tex(
            &mut texture_packer,
            &("rendered_block:".to_string() + &name),
            texture,
        )?;
    }

    let fallback_rect = texture_packer
        .get_frame(&String::from(FALLBACK_UNKNOWN_TEXTURE))
        .unwrap()
        .frame;

    let texture_atlas = texture_packer::exporter::ImageExporter::export(&texture_packer)
        .map_err(|x| Error::msg(format!("Texture atlas export failed: {:?}", x)))?;

    let mut texture_coords = HashMap::new();

    for item in item_defs.all_item_defs() {
        if !item.block_apperance.is_empty() {
            texture_coords.insert(
                item.short_name.clone(),
                texture_packer
                    .get_frame(&("rendered_block:".to_string() + &item.block_apperance))
                    .map(|x| x.frame)
                    .unwrap_or(fallback_rect),
            );
        } else if let Some(tex) = &item.inventory_texture {
            texture_coords.insert(
                item.short_name.clone(),
                texture_packer
                    .get_frame(&("simple:".to_string() + &tex.texture_name))
                    .map(|x| x.frame)
                    .unwrap_or(fallback_rect),
            );
        } else {
            texture_coords.insert(item.short_name.clone(), fallback_rect);
        };
    }

    for (key, value) in texture_packer.get_frames().iter() {
        if key.starts_with("builtin:") {
            texture_coords.insert(key.clone(), value.frame);
        }
    }

    let texture_atlas = Arc::new(Texture2DHolder::create(&ctx, &texture_atlas)?);
    Ok((texture_atlas, texture_coords))
}

fn get_texture(
    item: &perovskite_core::protocol::items::ItemStack,
    atlas_coords: &HashMap<String, Rect>,
    item_defs: &ClientItemManager,
) -> Rect {
    item_defs
        .get(&item.item_name)
        .and_then(|x| atlas_coords.get(&x.short_name).copied())
        .unwrap_or(*atlas_coords.get(UNKNOWN_TEXTURE).unwrap())
}

const CROSSHAIR: &str = "builtin:crosshair";
const DIGIT_ATLAS: &str = "builtin:digit_atlas";
const FRAME_SELECTED: &str = "builtin:frame_selected";
const FRAME_UNSELECTED: &str = "builtin:frame_unselected";
const TEST_ITEM: &str = "builtin:test_item";
const UNKNOWN_TEXTURE: &str = "builtin:unknown";
