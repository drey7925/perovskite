use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use image::{DynamicImage, RgbImage};
use texture_packer::{importer::ImageImporter, Rect};

use anyhow::{Error, Result};

use crate::{
    block_renderer::AsyncTextureLoader,
    game_state::items::ClientItemManager,
    vulkan::{Texture2DHolder, VulkanContext},
};

use self::{egui_ui::EguiUi, hud::GameHud};

pub(crate) mod egui_ui;
pub(crate) mod hud;

pub(crate) async fn make_uis<T>(
    item_defs: Arc<ClientItemManager>,
    texture_loader: T,
    ctx: &VulkanContext,
) -> Result<(hud::GameHud, egui_ui::EguiUi)>
where
    T: AsyncTextureLoader,
{
    let (texture_atlas, texture_coords) =
        build_texture_atlas(&item_defs, texture_loader, ctx).await?;

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

async fn build_texture_atlas<T>(
    item_defs: &ClientItemManager,
    mut texture_loader: T,
    ctx: &VulkanContext,
) -> Result<(Arc<Texture2DHolder>, HashMap<String, Rect>)>
where
    T: AsyncTextureLoader,
{
    let all_texture_names = item_defs
        .all_item_defs()
        .flat_map(|x| &x.inventory_texture)
        .map(|x| x.texture_name.clone())
        .collect::<HashSet<_>>();

    let mut textures = HashMap::new();
    for name in all_texture_names {
        let texture = texture_loader.load_texture(&name).await?;
        textures.insert(name, texture);
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

    texture_packer
        .pack_own(
            String::from(CROSSHAIR),
            ImageImporter::import_from_memory(include_bytes!("crosshair.png")).unwrap(),
        )
        .map_err(|x| Error::msg(format!("Texture pack failed: {:?}", x)))?;
    texture_packer
        .pack_own(
            String::from(UNKNOWN_TEXTURE),
            ImageImporter::import_from_memory(include_bytes!("../block_unknown.png")).unwrap(),
        )
        .map_err(|x| Error::msg(format!("Texture pack failed: {:?}", x)))?;
    texture_packer
        .pack_own(
            String::from(FRAME_SELECTED),
            ImageImporter::import_from_memory(include_bytes!("frame_selected.png")).unwrap(),
        )
        .map_err(|x| Error::msg(format!("Texture pack failed: {:?}", x)))?;
    texture_packer
        .pack_own(
            String::from(FRAME_UNSELECTED),
            ImageImporter::import_from_memory(include_bytes!("frame_unselected.png")).unwrap(),
        )
        .map_err(|x| Error::msg(format!("Texture pack failed: {:?}", x)))?;
    texture_packer
        .pack_own(
            String::from(DIGIT_ATLAS),
            ImageImporter::import_from_memory(include_bytes!("digit_atlas.png")).unwrap(),
        )
        .map_err(|x| Error::msg(format!("Texture pack failed: {:?}", x)))?;
    texture_packer
        .pack_own(
            String::from(TEST_ITEM),
            ImageImporter::import_from_memory(include_bytes!("testonly_pickaxe.png")).unwrap(),
        )
        .map_err(|x| Error::msg(format!("Texture pack failed: {:?}", x)))?;

    const COLOR_SCALE_R: [u8; 8] = [255, 255, 255, 255, 191, 128, 64, 0];
    const COLOR_SCALE_G: [u8; 8] = [0, 64, 128, 191, 255, 255, 255, 255];
    for i in 0..8 {
        let mut image = RgbImage::new(1, 1);
        image.put_pixel(0, 0, image::Rgb([COLOR_SCALE_R[i], COLOR_SCALE_G[i], 0]));
        texture_packer
            .pack_own(
                String::from(format!("builtin:wear_{}", i)),
                DynamicImage::ImageRgb8(image),
            )
            .map_err(|x| Error::msg(format!("Texture pack failed: {:?}", x)))?;
    }

    for (name, texture) in textures {
        texture_packer
            .pack_own(name, texture)
            .map_err(|x| Error::msg(format!("Texture pack failed: {:?}", x)))?;
    }
    let texture_atlas = texture_packer::exporter::ImageExporter::export(&texture_packer)
        .map_err(|x| Error::msg(format!("Texture atlas export failed: {:?}", x)))?;
    let texture_coords = texture_packer
        .get_frames()
        .iter()
        .map(|(k, v)| (k.clone(), v.frame))
        .collect();

    let texture_atlas = Arc::new(Texture2DHolder::create(ctx, &texture_atlas)?);
    Ok((texture_atlas, texture_coords))
}

fn get_texture(
    item: &cuberef_core::protocol::items::ItemStack,
    atlas_coords: &HashMap<String, Rect>,
    item_defs: &ClientItemManager,
) -> Rect {
    item_defs
        .get(&item.item_name)
        .and_then(|x| x.inventory_texture.as_ref())
        .map(|x| x.texture_name.as_ref())
        .and_then(|x: &str| atlas_coords.get(x).copied())
        .unwrap_or(*atlas_coords.get(UNKNOWN_TEXTURE).unwrap())
}

const CROSSHAIR: &str = "builtin:crosshair";
const DIGIT_ATLAS: &str = "builtin:digit_atlas";
const FRAME_SELECTED: &str = "builtin:frame_selected";
const FRAME_UNSELECTED: &str = "builtin:frame_unselected";
const TEST_ITEM: &str = "builtin:test_item";
const UNKNOWN_TEXTURE: &str = "builtin:unknown";
