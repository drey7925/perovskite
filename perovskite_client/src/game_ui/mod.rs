use std::{
    collections::{HashMap, HashSet},
    io::Cursor,
    sync::Arc,
};

use image::{DynamicImage, RgbImage};
use parking_lot::Mutex;
use perovskite_core::constants::textures::FALLBACK_UNKNOWN_TEXTURE;
use texture_packer::{importer::ImageImporter, Rect, TexturePacker};

use self::{egui_ui::EguiUi, hud::GameHud};
use crate::client_state::settings::GameSettings;
use crate::media::load_or_generate_image;
use crate::{
    client_state::items::ClientItemManager,
    media::CacheManager,
    vulkan::{
        block_renderer::BlockRenderer, mini_renderer::MiniBlockRenderer, Texture2DHolder,
        VulkanContext,
    },
};
use anyhow::{bail, Error, Result};
use arc_swap::ArcSwap;
use perovskite_core::protocol::items::item_def::Appearance;

pub(crate) mod egui_ui;
pub(crate) mod hud;

pub(crate) async fn make_uis(
    item_defs: Arc<ClientItemManager>,
    cache_manager: &mut CacheManager,
    ctx: Arc<VulkanContext>,
    block_renderer: &BlockRenderer,
    settings: Arc<ArcSwap<GameSettings>>,
) -> Result<(GameHud, EguiUi)> {
    let (texture_atlas, texture_coords) =
        build_texture_atlas(&item_defs, cache_manager, ctx, block_renderer).await?;

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

    let egui_ui = EguiUi::new(texture_atlas, texture_coords, item_defs.clone(), settings);

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

async fn build_texture_atlas(
    item_defs: &ClientItemManager,
    cache_manager: &mut CacheManager,
    ctx: Arc<VulkanContext>,
    block_renderer: &BlockRenderer,
) -> Result<(Arc<Texture2DHolder>, HashMap<String, Rect>)> {
    let mut all_texture_names = HashSet::new();
    let mut all_rendered_blocks = HashSet::new();
    for def in item_defs.all_item_defs() {
        match &def.appearance {
            None => {}
            Some(Appearance::InventoryTexture(inv_tex)) => {
                all_texture_names.insert(inv_tex.clone());
            }
            Some(Appearance::BlockApperance(block_name)) => {
                all_rendered_blocks.insert(block_name.clone());
            }
        }
    }
    let all_rendered_blocks = all_rendered_blocks.into_iter().collect::<Vec<_>>();

    let mut simple_textures = HashMap::new();
    let rendered_block_textures = Arc::new(Mutex::new(HashMap::new()));
    for name in all_texture_names {
        let texture = load_or_generate_image(cache_manager, &name).await?;
        simple_textures.insert(name, texture);
    }

    let cache_insertions = Arc::new(Mutex::new(Vec::new()));

    // We must block in place; the task cannot be made 'static easily.
    tokio::task::block_in_place(|| {
        std::thread::scope(|s| {
            const NUM_MINI_RENDER_THREADS: usize = 4;
            let mut handles = Vec::with_capacity(NUM_MINI_RENDER_THREADS);
            for i in 0..NUM_MINI_RENDER_THREADS {
                let ctx = ctx.clone();

                let mut tasks = vec![];
                for (j, name) in all_rendered_blocks.iter().enumerate() {
                    if (j % NUM_MINI_RENDER_THREADS) == i {
                        tasks.push(name);
                    }
                }
                handles.push(s.spawn(|| -> Result<()> {
                    let mut renderer =
                        MiniBlockRenderer::new(ctx, [64, 64], block_renderer.atlas())?;
                    for name in tasks {
                        let block_id = match block_renderer.block_types().get_block_by_name(name) {
                            Some(id) => id,
                            None => continue,
                        };
                        let block_def = match block_renderer.block_types().get_blockdef(block_id) {
                            Some(block_type) => block_type,
                            None => continue,
                        };

                        let cached_content = cache_manager.try_get_block_appearance(block_def)?;
                        if let Some(content) = cached_content {
                            let imported = match ImageImporter::import_from_memory(&content) {
                                Ok(x) => x,
                                Err(x) => {
                                    bail!("Failed to import block appearance: {:?}", x);
                                }
                            };
                            rendered_block_textures.lock().insert(name, imported);
                            continue;
                        }

                        let block_tex = renderer.render(block_renderer, block_id, block_def)?;

                        let mut bytes: Vec<u8> = Vec::new();
                        block_tex
                            .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)?;
                        cache_insertions.lock().push((block_def, bytes));
                        rendered_block_textures.lock().insert(name, block_tex);
                    }
                    Ok(())
                }));
            }
            for handle in handles {
                match handle.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(x)) => {
                        bail!("Mini render thread failed: {:?}", x);
                    }
                    Err(x) => {
                        bail!("Mini render thread panic: {:?}", x);
                    }
                };
            }
            Ok(())
        })
    })?;

    for (key, value) in cache_insertions.lock().drain(..) {
        cache_manager.insert_block_appearance(key, value).await?;
    }

    let config = texture_packer::TexturePackerConfig {
        // todo tweak these or make into a setting
        allow_rotation: false,
        max_width: 4096,
        max_height: 4096,
        border_padding: 1,
        texture_padding: 1,
        texture_extrusion: 1,
        trim: false,
        texture_outlines: false,
        force_max_dimensions: false,
    };
    let mut texture_packer = TexturePacker::new_skyline(config);

    pack_tex(
        &mut texture_packer,
        CROSSHAIR,
        ImageImporter::import_from_memory(include_bytes!("crosshair.png")).unwrap(),
    )?;
    pack_tex(
        &mut texture_packer,
        UNKNOWN_TEXTURE,
        ImageImporter::import_from_memory(include_bytes!("../vulkan/block_unknown.png")).unwrap(),
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
    for (name, texture) in rendered_block_textures.lock().drain() {
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

    let texture_atlas = texture_packer::exporter::ImageExporter::export(&texture_packer, None)
        .map_err(|x| Error::msg(format!("Texture atlas export failed: {:?}", x)))?;

    let mut texture_coords = HashMap::new();

    for item in item_defs.all_item_defs() {
        match &item.appearance {
            Some(Appearance::InventoryTexture(inv_tex)) => {
                texture_coords.insert(
                    item.short_name.clone(),
                    texture_packer
                        .get_frame(&("simple:".to_string() + &inv_tex))
                        .map(|x| x.frame)
                        .unwrap_or(fallback_rect),
                );
            }
            Some(Appearance::BlockApperance(block_name)) => {
                texture_coords.insert(
                    item.short_name.clone(),
                    texture_packer
                        .get_frame(&("rendered_block:".to_string() + &block_name))
                        .map(|x| x.frame)
                        .unwrap_or(fallback_rect),
                );
            }
            None => {
                texture_coords.insert(item.short_name.clone(), fallback_rect);
            }
        }
    }

    for (key, value) in texture_packer.get_frames().iter() {
        if key.starts_with("builtin:") {
            texture_coords.insert(key.clone(), value.frame);
        }
    }

    let texture_atlas = Arc::new(Texture2DHolder::from_srgb(&ctx, &texture_atlas)?);
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
