use std::sync::Arc;

use crate::{cache::CacheManager, vulkan::RectF32};
use perovskite_core::{constants::textures::FALLBACK_UNKNOWN_TEXTURE, protocol::entities as proto};
use rustc_hash::{FxHashMap, FxHashSet};
use texture_packer::{importer::ImageImporter, Rect};

use super::{
    block_renderer::VkCgvBufferGpu, shaders::cube_geometry::CubeGeometryVertex, Texture2DHolder,
    VkAllocator, VulkanContext,
};
use anyhow::{bail, ensure, Error, Result};
use cgmath::{Matrix3, Rad, Vector3, Zero};

/// Manages the entity definitions and their respective meshes
///
/// Does not hold the actual state of the entities being sent to the display adapter;
/// that state is in EntityState.
pub(crate) struct EntityRenderer {
    texture_atlas: Arc<Texture2DHolder>,
    allocator: Arc<VkAllocator>,
    mesh_definitions: FxHashMap<u32, EntityMesh>,
    /// These include buffers that render a single entity. Later on, as we add various accelerated and
    /// instanced rendering, the renderer may build buffers with multiple entities in them.
    singleton_gpu_buffers: FxHashMap<u32, Option<VkCgvBufferGpu>>,
}
impl EntityRenderer {
    pub(crate) async fn new(
        entity_defs: Vec<proto::EntityDef>,
        cache_manager: &mut CacheManager,
        ctx: &VulkanContext,
    ) -> Result<EntityRenderer> {
        let mut all_texture_names = FxHashSet::default();
        for def in &entity_defs {
            if let Some(appearance) = &def.appearance {
                for mesh in &appearance.custom_mesh {
                    if let Some(tex) = &mesh.texture {
                        all_texture_names.insert(tex.texture_name.clone());
                    }
                }
            }
        }

        let mut all_textures = FxHashMap::default();
        for x in all_texture_names {
            let texture = cache_manager.load_media_by_name(&x).await?;
            all_textures.insert(x, texture);
        }

        let config = texture_packer::TexturePackerConfig {
            // todo break these out into config
            allow_rotation: false,
            max_width: 4096,
            max_height: 4096,
            border_padding: 2,
            texture_padding: 2,
            texture_extrusion: 2,
            trim: false,
            texture_outlines: false,
            force_max_dimensions: false,
        };
        let mut texture_packer = texture_packer::TexturePacker::new_skyline(config);
        // TODO move these files to a sensible location
        texture_packer
            .pack_own(
                String::from(FALLBACK_UNKNOWN_TEXTURE),
                ImageImporter::import_from_memory(include_bytes!("block_unknown.png")).unwrap(),
            )
            .map_err(|x| Error::msg(format!("Texture pack failed: {:?}", x)))?;
        texture_packer
            .pack_own(
                String::from(TESTONLY_ENTITY),
                ImageImporter::import_from_memory(include_bytes!("temporary_player_entity.png"))
                    .unwrap(),
            )
            .map_err(|x| Error::msg(format!("Texture pack failed: {:?}", x)))?;

        for (name, texture) in all_textures {
            let texture = ImageImporter::import_from_memory(&texture)
                .map_err(|e| Error::msg(format!("Texture import failed: {:?}", e)))?;
            texture_packer
                .pack_own(name, texture)
                .map_err(|x| Error::msg(format!("Texture pack failed: {:?}", x)))?;
        }

        let texture_atlas = texture_packer::exporter::ImageExporter::export(&texture_packer, None)
            .map_err(|x| Error::msg(format!("Texture atlas export failed: {:?}", x)))?;
        let texture_coords: FxHashMap<String, Rect> = texture_packer
            .get_frames()
            .iter()
            .map(|(k, v)| (k.clone(), v.frame))
            .collect();

        // Entity classes may not be contgious or ordered; hence a hashmap for now
        let mut all_meshes = FxHashMap::default();
        let mut singleton_gpu_buffers = FxHashMap::default();
        let texture_atlas = Arc::new(Texture2DHolder::create(ctx, &texture_atlas)?);
        for def in entity_defs {
            if let Some(appearance) = def.appearance {
                let mesh = EntityRenderer::pre_render(
                    &appearance,
                    &texture_coords,
                    texture_atlas.dimensions(),
                )?;
                let singleton_buffer =
                    VkCgvBufferGpu::from_buffers(&mesh.vtx, &mesh.idx, ctx.clone_allocator())?;
                singleton_gpu_buffers.insert(def.entity_class, singleton_buffer);

                all_meshes.insert(def.entity_class, mesh);
            }
        }
        Ok(EntityRenderer {
            texture_atlas,
            allocator: ctx.clone_allocator(),
            mesh_definitions: all_meshes,
            singleton_gpu_buffers,
        })
    }

    pub(crate) fn atlas(&self) -> &Texture2DHolder {
        &self.texture_atlas
    }

    /// Convert from the network mesh to a vulkan renderable mesh
    fn pre_render(
        appearance: &proto::EntityAppearance,
        texture_coords: &FxHashMap<String, Rect>,
        atlas_dims: (u32, u32),
    ) -> Result<EntityMesh> {
        let meshes = &appearance.custom_mesh;
        let vertex_count = meshes.iter().map(|x| x.x.len()).sum();
        if vertex_count >= u32::MAX as usize {
            bail!("Too many vertices");
        }
        let mut vertices = Vec::with_capacity(vertex_count);
        let mut indices = Vec::with_capacity(meshes.iter().map(|x| x.indices.len()).sum());
        for mesh in meshes {
            let tex_rectangle: RectF32 = mesh
                .texture
                .as_ref()
                .and_then(|x| texture_coords.get(&x.texture_name))
                .unwrap_or_else(|| texture_coords.get(FALLBACK_UNKNOWN_TEXTURE).unwrap())
                .into();
            let tex_rectangle = tex_rectangle.div(atlas_dims);
            let vertices_len = mesh.x.len();
            ensure!(mesh.y.len() == vertices_len);
            ensure!(mesh.z.len() == vertices_len);
            ensure!(mesh.u.len() == vertices_len);
            ensure!(mesh.v.len() == vertices_len);
            ensure!(mesh.nx.len() == vertices_len);
            ensure!(mesh.ny.len() == vertices_len);
            ensure!(mesh.nz.len() == vertices_len);
            for i in 0..vertices_len {
                let u = mesh.u[i] * tex_rectangle.w + tex_rectangle.l;
                let v = mesh.v[i] * tex_rectangle.h + tex_rectangle.t;
                vertices.push(CubeGeometryVertex {
                    position: [mesh.x[i], mesh.y[i], mesh.z[i]],
                    normal: [mesh.nx[i], mesh.ny[i], mesh.nz[i]],
                    uv_texcoord: [u, v],
                    brightness: 1.0,
                    global_brightness_contribution: 0.0,
                    wave_horizontal: 0.0,
                });
            }
            ensure!(mesh.indices.len() % 3 == 0);
            ensure!(mesh.indices.iter().all(|x| *x < vertices_len as u32));
            let offset = indices.len();
            for index in &mesh.indices {
                indices.push(index + offset as u32);
            }
        }
        Ok(EntityMesh {
            vtx: vertices,
            idx: indices,
            attach_offset: match &appearance.attachment_offset {
                Some(x) => x.try_into().unwrap(),
                None => Vector3::zero(),
            },
            attach_in_model_space: appearance.attachment_offset_in_model_space,
        })
    }

    pub(crate) fn transform_position(
        &self,
        class: u32,
        pos: Vector3<f64>,
        face_dir: Rad<f32>,
        pitch: Rad<f32>,
    ) -> Vector3<f64> {
        let def = match self.mesh_definitions.get(&class) {
            Some(x) => x,
            None => return pos,
        };
        if def.attach_in_model_space {
            let rotation: Matrix3<f64> = (Matrix3::from_angle_y(face_dir)
                * Matrix3::from_angle_x(pitch))
            .cast()
            .unwrap();
            pos + (rotation * def.attach_offset)
        } else {
            def.attach_offset + pos
        }
    }

    pub(crate) fn get_singleton(&self, class: u32) -> Option<VkCgvBufferGpu> {
        self.singleton_gpu_buffers.get(&class).cloned().flatten()
    }
}

pub(crate) struct EntityMesh {
    // Note: As the shaders evolve, we will stop using CubeGeometryVertex and instead will use
    // a more capable vertex type that can express entity-specific details.
    pub(crate) vtx: Vec<CubeGeometryVertex>,
    pub(crate) idx: Vec<u32>,
    pub(crate) attach_offset: Vector3<f64>,
    pub(crate) attach_in_model_space: bool,
}

const TESTONLY_ENTITY: &str = "builtin:temporary_player_entity";

struct EntityRendererState {}
