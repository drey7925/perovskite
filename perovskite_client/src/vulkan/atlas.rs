use crate::vulkan::{Texture2DHolder, VulkanContext};
use anyhow::{anyhow, bail, Error, Result};
use image::imageops::FilterType;
use image::RgbaImage;
use image::{GenericImage, Rgba};
use lazy_static::lazy_static;
use perovskite_core::protocol::render::TextureReference;
use rustc_hash::FxHashMap;
use std::fmt::{Debug, Formatter};
use texture_packer::texture::Texture;
use texture_packer::{Rect, TexturePacker};

lazy_static! {
    static ref UNKNOWN_TEX: RgbaImage = {
        image::load_from_memory(include_bytes!("block_unknown.png"))
            .unwrap()
            .into_rgba8()
    };
    static ref SELECTION_TEX: RgbaImage = {
        image::load_from_memory(include_bytes!("selection.png"))
            .unwrap()
            .into_rgba8()
    };
}

#[derive(Clone, Hash, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub(crate) enum TextureKey {
    Named(NamedTextureKey),
    FallbackUnknownTex,
    SelectionRectangle,
}
impl TextureKey {
    fn diffuse<'a>(&self, textures: &'a FxHashMap<String, RgbaImage>) -> Option<&'a RgbaImage> {
        match self {
            TextureKey::Named(key) => key
                .diffuse
                .as_ref()
                .map(|x| textures.get(x).unwrap_or(&UNKNOWN_TEX)),
            TextureKey::FallbackUnknownTex => Some(&UNKNOWN_TEX),
            TextureKey::SelectionRectangle => Some(&SELECTION_TEX),
        }
    }
    fn specular<'a>(&self, textures: &'a FxHashMap<String, RgbaImage>) -> Option<&'a RgbaImage> {
        match self {
            TextureKey::Named(key) => key
                .specular
                .as_ref()
                .map(|x| textures.get(x).unwrap_or(&UNKNOWN_TEX)),
            _ => None,
        }
    }

    fn emissive<'a>(&self, textures: &'a FxHashMap<String, RgbaImage>) -> Option<&'a RgbaImage> {
        match self {
            TextureKey::Named(key) => key
                .emissive
                .as_ref()
                .map(|x| textures.get(x).unwrap_or(&UNKNOWN_TEX)),
            _ => None,
        }
    }

    fn size(&self, textures: &FxHashMap<String, RgbaImage>) -> Result<(u32, u32)> {
        let diffuse_size = self
            .diffuse(&textures)
            .map(|x| (x.width(), x.height()))
            .unwrap_or((1, 1));
        let specular_size = self
            .specular(&textures)
            .map(|x| (x.width(), x.height()))
            .unwrap_or((1, 1));
        let emissive_size = self
            .emissive(&textures)
            .map(|x| (x.width(), x.height()))
            .unwrap_or((1, 1));
        let max_x = diffuse_size.0.max(specular_size.0).max(emissive_size.0);
        let max_y = diffuse_size.1.max(specular_size.1).max(emissive_size.1);

        if max_x % diffuse_size.0 != 0 {
            bail!("{self:?}: Target width of {max_x} pixels not divisible by diffuse texture width {}", diffuse_size.0);
        }
        if max_y % diffuse_size.1 != 0 {
            bail!("{self:?}: Target height of {max_x} pixels not divisible by diffuse texture height {}", diffuse_size.0);
        }
        if max_x % specular_size.0 != 0 {
            bail!("{self:?}: Target width of {max_x} pixels not divisible by specular texture width {}", diffuse_size.0);
        }
        if max_y % specular_size.1 != 0 {
            bail!("{self:?}: Target height of {max_y} pixels not divisible by specular texture height {}", diffuse_size.0);
        }
        if max_x % emissive_size.0 != 0 {
            bail!("{self:?}: Target width of {max_x} pixels not divisible by emissive texture width {}", diffuse_size.0);
        }
        if max_y % emissive_size.1 != 0 {
            bail!("{self:?}: Target height of {max_y} pixels not divisible by emissive texture height {}", diffuse_size.0);
        }
        Ok((max_x, max_y))
    }
}

#[derive(Clone, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub(crate) struct NamedTextureKey {
    pub(crate) diffuse: Option<String>,
    pub(crate) specular: Option<String>,
    pub(crate) emissive: Option<String>,
}
impl Debug for NamedTextureKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "[{:?} / {:?} / {:?}]",
            self.diffuse.as_ref().map(String::as_str).unwrap_or(""),
            self.specular.as_ref().map(String::as_str).unwrap_or(""),
            self.emissive.as_ref().map(String::as_str).unwrap_or(""),
        ))
    }
}
impl From<&TextureReference> for TextureKey {
    fn from(value: &TextureReference) -> Self {
        Self::Named(value.into())
    }
}
impl From<&TextureReference> for NamedTextureKey {
    fn from(value: &TextureReference) -> Self {
        fn if_nonempty(s: &str) -> Option<String> {
            if s.is_empty() {
                None
            } else {
                Some(s.to_string())
            }
        }
        Self {
            diffuse: if_nonempty(value.diffuse.as_ref()),
            specular: if_nonempty(value.rt_specular.as_ref()),
            emissive: if_nonempty(value.emissive.as_ref()),
        }
    }
}

pub(crate) struct TextureAtlas {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) diffuse: Texture2DHolder,
    pub(crate) specular: Texture2DHolder,
    pub(crate) emissive: Texture2DHolder,
    pub(crate) texel_coords: FxHashMap<TextureKey, texture_packer::Rect>,
}
impl TextureAtlas {
    pub(crate) fn new(
        ctx: &VulkanContext,
        keys: impl IntoIterator<Item = TextureKey>,
        textures: FxHashMap<String, RgbaImage>,
    ) -> Result<Self> {
        // 4096 listed on https://registry.khronos.org/vulkan/specs/latest/man/html/Required_Limits.html
        // If increased past 4096, hardware detection is needed. If increased past 65535, vertex
        // buffer types must be changed
        const ATLAS_DIM: u32 = 4096;
        let config = texture_packer::TexturePackerConfig {
            // todo break these out into config
            allow_rotation: false,
            max_width: ATLAS_DIM,
            max_height: ATLAS_DIM,
            border_padding: 0,
            texture_padding: 0,
            texture_extrusion: 1,
            trim: false,
            texture_outlines: false,
            force_max_dimensions: false,
        };
        let mut diffuse_packer = TexturePacker::new_skyline(config);

        for key in keys.into_iter().chain([
            TextureKey::FallbackUnknownTex,
            TextureKey::SelectionRectangle,
        ]) {
            let target_size = key.size(&textures)?;
            let diffuse = key
                .diffuse(&textures)
                .map(|x| {
                    image::imageops::resize(x, target_size.0, target_size.1, FilterType::Nearest)
                })
                .unwrap_or(make_blank(target_size.0, target_size.1));

            diffuse_packer
                .pack_own(key, diffuse)
                .map_err(|x| anyhow!("{x:?}"))?;
        }
        let diffuse_image = texture_packer::exporter::ImageExporter::export(&diffuse_packer, None)
            .map_err(|x| Error::msg(format!("Texture atlas export failed: {:?}", x)))?;
        let texel_coords: FxHashMap<TextureKey, Rect> = diffuse_packer
            .get_frames()
            .iter()
            .map(|(k, v)| (k.clone(), v.frame))
            .collect();

        let mut specular_image = RgbaImage::new(diffuse_image.width(), diffuse_image.height());
        let mut emissive_image = RgbaImage::new(diffuse_image.width(), diffuse_image.height());
        for (key, rect) in &texel_coords {
            let target_size = key.size(&textures)?;
            assert_eq!(rect.w, target_size.0);
            assert_eq!(rect.h, target_size.1);
            let specular = key
                .specular(&textures)
                .map(|x| {
                    image::imageops::resize(x, target_size.0, target_size.1, FilterType::Nearest)
                })
                .unwrap_or(make_blank(target_size.0, target_size.1));
            specular_image.copy_from(&specular, rect.x, rect.y)?;

            let emissive = key
                .emissive(&textures)
                .map(|x| {
                    image::imageops::resize(x, target_size.0, target_size.1, FilterType::Nearest)
                })
                .unwrap_or(make_blank(target_size.0, target_size.1));
            emissive_image.copy_from(&emissive, rect.x, rect.y)?;
        }

        Ok(TextureAtlas {
            width: diffuse_image.width(),
            height: diffuse_image.height(),
            diffuse: Texture2DHolder::from_srgb(ctx, diffuse_image.into_rgba8())?,
            specular: Texture2DHolder::from_srgb(ctx, specular_image)?,
            emissive: Texture2DHolder::from_srgb(ctx, emissive_image)?,
            texel_coords,
        })
    }
}

fn make_blank(width: u32, height: u32) -> RgbaImage {
    let mut image = RgbaImage::new(width, height);
    image.pixels_mut().for_each(|(pixel)| {
        *pixel = Rgba([0, 0, 0, 255]);
    });
    image
}
