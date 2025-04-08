#![allow(unused)]

use anyhow::{bail, Context, Error, Result};
use futures::Future;

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    pin::{pin, Pin},
};

use directories::ProjectDirs;
use perovskite_core::protocol::{
    blocks::BlockTypeDef, game_rpc::ListMediaResponse, items::ItemDef, render::TextureReference,
};
use sha2::{Digest, Sha256};

use crate::{
    client_state::settings::{clean_path, project_dirs},
    net_client::AsyncMediaLoader,
};

pub(crate) struct CacheManager {
    file_hashes: HashMap<String, [u8; 32]>,
    cache_dir: Option<PathBuf>,
    loader: Box<dyn AsyncMediaLoader + Send + Sync + 'static>,
}
impl CacheManager {
    pub(crate) fn new(
        media_list: ListMediaResponse,
        loader: Box<dyn AsyncMediaLoader + Send + Sync + 'static>,
    ) -> Result<CacheManager> {
        let cache_dir: Option<PathBuf> = project_dirs()
            .as_ref()
            .map(ProjectDirs::cache_dir)
            .map(Into::into);
        if let Some(cache_dir) = &cache_dir {
            log::info!("Using cache dir: {}", clean_path(cache_dir.clone()));
            if cache_dir.exists() && !cache_dir.is_dir() {
                panic!("Cache dir is not a directory.");
            } else if !cache_dir.exists() {
                log::info!("Creating cache dir...");
                std::fs::create_dir_all(cache_dir).unwrap();
            }
        } else {
            log::warn!("Couldn't find cache dir. Content will not be cached.");
        }

        let mut file_hashes = HashMap::new();
        for media in media_list.media {
            file_hashes.insert(
                media.media_name.clone(),
                media.sha256.try_into().map_err(|_| {
                    Error::msg(format!(
                        "invalid sha256 for {}: not 256 bits",
                        media.media_name
                    ))
                })?,
            );
        }

        Ok(CacheManager {
            file_hashes,
            cache_dir,
            loader,
        })
    }

    pub(crate) async fn load_media_by_name(&mut self, media_name: &str) -> Result<Vec<u8>> {
        if let Some(hash) = self.file_hashes.get(media_name).copied() {
            // For media by name, the ID hash is the same as the content hash
            let data = tokio::task::block_in_place(|| self.try_get_cached(&hash, Some(&hash)))?;
            if let Some(data) = data {
                Ok(data)
            } else {
                let loaded_data = self
                    .loader
                    .load_media(media_name)
                    .await
                    .context("failed to load media")?;
                let actual_hash = sha2::Sha256::digest(&loaded_data).to_vec();
                if actual_hash != hash {
                    bail!(
                        "Hashes don't match for {}, expected {}, got {}",
                        media_name,
                        hex::encode(hash),
                        hex::encode(actual_hash)
                    );
                }

                tokio::task::block_in_place(|| self.insert_into_cache(&hash, &loaded_data))?;

                Ok(loaded_data)
            }
        } else {
            log::warn!(
                "{} not found in media manifest. Will load it but not cache it",
                media_name
            );
            self.loader.load_media(media_name).await
        }
    }

    fn insert_into_cache(&mut self, hash: &[u8; 32], data: &[u8]) -> Result<()> {
        if let Some(cache_dir) = &self.cache_dir {
            let path = get_cache_path(hash, cache_dir);
            if !path.parent().unwrap().exists() {
                std::fs::create_dir_all(path.parent().unwrap())?;
            }
            std::fs::write(&path, data)?;
        }
        Ok(())
    }

    pub(crate) fn try_get_block_appearance(
        &self,
        block_type: &BlockTypeDef,
    ) -> Result<Option<Vec<u8>>> {
        let hash = block_type
            .render_info
            .as_ref()
            .and_then(|x| self.hash_render_info(x));
        if let Some(hash) = hash {
            // For block appearances, the id hash is based on the render info and the content hashes feeding into it.
            // We don't know the content hash.
            self.try_get_cached(&hash, None)
        } else {
            Ok(None)
        }
    }

    pub(crate) fn insert_block_appearance(
        &mut self,
        block_type: &BlockTypeDef,
        data: Vec<u8>,
    ) -> Result<()> {
        let hash = block_type
            .render_info
            .as_ref()
            .and_then(|x| self.hash_render_info(x));
        if let Some(hash) = hash {
            self.insert_into_cache(&hash, &data)?;
        }
        Ok(())
    }

    fn try_get_cached(
        &self,
        id_hash: &[u8; 32],
        content_hash: Option<&[u8; 32]>,
    ) -> Result<Option<Vec<u8>>> {
        if let Some(cache_dir) = &self.cache_dir {
            let path = get_cache_path(id_hash, cache_dir);
            if path.exists() {
                let data = std::fs::read(&path)?;
                if let Some(expected_hash) = content_hash {
                    let actual_hash = sha2::Sha256::digest(&data).to_vec();
                    if actual_hash != expected_hash {
                        log::warn!(
                            "Hashes don't match for {}, got {}",
                            path.display(),
                            hex::encode(actual_hash)
                        );
                        std::fs::rename(&path, path.with_extension("corrupt")).unwrap();
                        return Ok(None);
                    }
                }
                return Ok(Some(data));
            }
        }
        Ok(None)
    }

    fn hash_texture_reference(
        &self,
        tex: Option<&TextureReference>,
        hasher: &mut Sha256,
    ) -> Option<()> {
        if let Some(tex) = tex {
            hasher.update(b"tex_ref:");
            hasher.update(self.get_file_hash(&tex.texture_name)?);
            hasher.update(b":");
            if let Some(crop) = &tex.crop {
                hasher.update(crop.top.to_le_bytes());
                hasher.update(crop.bottom.to_le_bytes());
                hasher.update(crop.left.to_le_bytes());
                hasher.update(crop.right.to_le_bytes());
            }
        } else {
            hasher.update(b"empty_tex_ref:");
        }
        Some(())
    }

    fn hash_cube_render_info(
        &self,
        cube: &perovskite_core::protocol::blocks::CubeRenderInfo,
        hasher: &mut Sha256,
    ) -> Option<()> {
        hasher.update(b"cube:");
        hasher.update(cube.render_mode.to_le_bytes());
        hasher.update(cube.variant_effect.to_le_bytes());
        self.hash_texture_reference(cube.tex_left.as_ref(), hasher)?;
        self.hash_texture_reference(cube.tex_right.as_ref(), hasher)?;
        self.hash_texture_reference(cube.tex_top.as_ref(), hasher)?;
        self.hash_texture_reference(cube.tex_bottom.as_ref(), hasher)?;
        self.hash_texture_reference(cube.tex_front.as_ref(), hasher)?;
        self.hash_texture_reference(cube.tex_back.as_ref(), hasher)?;
        Some(())
    }

    fn hash_aabox_render_info(
        &self,
        aa_boxes: &perovskite_core::protocol::blocks::AxisAlignedBoxes,
        hasher: &mut Sha256,
    ) -> Option<()> {
        hasher.update(b"aa_boxes:");
        for aa_box in aa_boxes.boxes.iter() {
            hasher.update(aa_box.rotation.to_le_bytes());
            hasher.update(aa_box.variant_mask.to_le_bytes());
            hasher.update(aa_box.x_min.to_le_bytes());
            hasher.update(aa_box.y_min.to_le_bytes());
            hasher.update(aa_box.z_min.to_le_bytes());
            hasher.update(aa_box.x_max.to_le_bytes());
            hasher.update(aa_box.y_max.to_le_bytes());
            hasher.update(aa_box.z_max.to_le_bytes());
            self.hash_texture_reference(aa_box.tex_left.as_ref(), hasher)?;
            self.hash_texture_reference(aa_box.tex_right.as_ref(), hasher)?;
            self.hash_texture_reference(aa_box.tex_top.as_ref(), hasher)?;
            self.hash_texture_reference(aa_box.tex_bottom.as_ref(), hasher)?;
            self.hash_texture_reference(aa_box.tex_front.as_ref(), hasher)?;
            self.hash_texture_reference(aa_box.tex_back.as_ref(), hasher)?;
            hasher.update(aa_box.top_slope_x.to_le_bytes());
            hasher.update(aa_box.top_slope_z.to_le_bytes());
        }
        Some(())
    }

    fn hash_plant_render_info(
        &self,
        plant: &perovskite_core::protocol::blocks::PlantLikeRenderInfo,
        hasher: &mut Sha256,
    ) -> Option<()> {
        hasher.update(b"plant_like:");
        hasher.update(plant.wave_effect_scale.to_le_bytes());
        self.hash_texture_reference(plant.tex.as_ref(), hasher)?;
        Some(())
    }

    pub(crate) fn hash_render_info(
        &self,
        render_info: &perovskite_core::protocol::blocks::block_type_def::RenderInfo,
    ) -> Option<[u8; 32]> {
        let mut hasher = sha2::Sha256::new();
        hasher.update(b"render_info:");
        match render_info {
            perovskite_core::protocol::blocks::block_type_def::RenderInfo::Empty(_) => return None,
            perovskite_core::protocol::blocks::block_type_def::RenderInfo::Cube(cube) => {
                self.hash_cube_render_info(cube, &mut hasher)?;
            }
            perovskite_core::protocol::blocks::block_type_def::RenderInfo::AxisAlignedBoxes(
                aa_boxes,
            ) => {
                self.hash_aabox_render_info(aa_boxes, &mut hasher)?;
            }
            perovskite_core::protocol::blocks::block_type_def::RenderInfo::PlantLike(plant) => {
                self.hash_plant_render_info(plant, &mut hasher)?;
            }
        }
        Some(hasher.finalize().into())
    }

    fn get_file_hash(&self, file_name: &str) -> Option<[u8; 32]> {
        self.file_hashes.get(file_name).cloned()
    }
}

fn get_cache_path(expected_hash: &[u8; 32], cache_dir: &Path) -> PathBuf {
    let hash_hex = hex::encode(expected_hash);
    assert_eq!(hash_hex.len(), 64);
    // Security: The hash is untrusted... But we verified its length, and we generate the hex string ourselves.
    let prefix = hash_hex[0..2].to_string();
    let suffix = hash_hex[2..].to_string();

    cache_dir.join(prefix).join(suffix)
}
