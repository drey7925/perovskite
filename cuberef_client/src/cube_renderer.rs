// Copyright 2023 drey7925
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use std::sync::Arc;

use cgmath::{vec3, ElementWise, Matrix4, Vector2, Vector3, Zero};

use cuberef_core::coordinates::{BlockCoordinate, ChunkCoordinate};
use cuberef_core::protocol::blocks::block_type_def::RenderInfo;
use cuberef_core::protocol::blocks::{self as blocks_proto, BlockTypeDef, CubeRenderInfo};
use cuberef_core::protocol::render::TextureReference;
use cuberef_core::{block_id::BlockId, coordinates::ChunkOffset};

use anyhow::{ensure, Context, Error, Result};
use image::DynamicImage;
use rustc_hash::FxHashMap;
use texture_packer::importer::ImageImporter;
use texture_packer::Rect;
use tonic::async_trait;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::memory::allocator::{
    AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryUsage,
};

use crate::game_state::chunk::ClientChunk;
use crate::vulkan::shaders::cube_geometry::{CubeGeometryDrawCall, CubeGeometryVertex};
use crate::vulkan::{Texture2DHolder, VulkanContext};

const FALLBACK_UNKNOWN_TEXTURE: &str = "builtin:unknown";
const SELECTION_RECTANGLE: &str = "builtin:selection_rectangle";

// Given in game world coordinates (Y is up)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd)]
pub(crate) enum CubeFace {
    XPlus,
    XMinus,
    YPlus,
    YMinus,
    ZPlus,
    ZMinus,
}

pub(crate) struct ClientBlockTypeManager {
    block_defs: Vec<Option<blocks_proto::BlockTypeDef>>,
    fallback_block_def: blocks_proto::BlockTypeDef,
}
impl ClientBlockTypeManager {
    pub(crate) fn new(
        server_defs: Vec<blocks_proto::BlockTypeDef>,
    ) -> Result<ClientBlockTypeManager> {
        let max_id = server_defs
            .iter()
            .map(|x| x.id)
            .max()
            .with_context(|| "Server defs were empty")?;

        let mut block_defs = Vec::new();
        block_defs.resize_with(BlockId(max_id).index() + 1, || None);
        for def in server_defs {
            let id = BlockId(def.id);
            ensure!(id.variant() == 0);
            ensure!(block_defs[id.index()].is_none());
            block_defs[id.index()] = Some(def);
        }

        Ok(ClientBlockTypeManager {
            block_defs,
            fallback_block_def: make_fallback(),
        })
    }

    pub(crate) fn all_block_defs(&self) -> impl Iterator<Item = &blocks_proto::BlockTypeDef> {
        self.block_defs.iter().flatten()
    }

    fn get_fallback_blockdef(&self) -> &blocks_proto::BlockTypeDef {
        &self.fallback_block_def
    }
    pub(crate) fn get_blockdef(&self, id: BlockId) -> Option<&blocks_proto::BlockTypeDef> {
        match self.block_defs.get(id.index()) {
            // none if get() failed due to bounds check
            None => None,
            Some(x) => {
                // Still an option, since we might be missing block defs from the server
                x.as_ref()
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CubeExtents {
    x: (f32, f32),
    y: (f32, f32),
    z: (f32, f32),
}

/// Manages the block type definitions, and their underlying textures,
/// for the game.
pub(crate) struct BlockRenderer {
    block_defs: Arc<ClientBlockTypeManager>,
    texture_coords: HashMap<String, Rect>,
    texture_atlas: Arc<Texture2DHolder>,
    allocator: Arc<GenericMemoryAllocator<Arc<FreeListAllocator>>>,
}
impl BlockRenderer {
    fn get_texture(&self, id: &str) -> Result<Rect> {
        self.texture_coords
            .get(&id.to_string())
            .copied()
            .with_context(|| "Texture not in atlas")
    }
    pub(crate) async fn new<T>(
        block_defs: Arc<ClientBlockTypeManager>,
        texture_loader: T,
        ctx: &VulkanContext,
    ) -> Result<BlockRenderer>
    where
        T: AsyncTextureLoader,
    {
        let (texture_atlas, texture_coords) =
            build_texture_atlas(&block_defs, texture_loader).await?;
        let texture_atlas = Arc::new(Texture2DHolder::create(ctx, &texture_atlas)?);
        Ok(BlockRenderer {
            block_defs,
            texture_coords,
            texture_atlas,
            allocator: ctx.allocator(),
        })
    }

    pub(crate) fn texture(&self) -> Arc<Texture2DHolder> {
        self.texture_atlas.clone()
    }

    pub(crate) fn allocator(&self) -> &GenericMemoryAllocator<Arc<FreeListAllocator>> {
        &self.allocator
    }

    pub(crate) fn emit_cube_vk(
        &self,
        chunk_data: &FxHashMap<ChunkCoordinate, ClientChunk>,
        current_chunk: &ClientChunk,
        block_coord: BlockCoordinate,
        vtx: &mut Vec<CubeGeometryVertex>,
        idx: &mut Vec<u32>,
    ) {
        let block = self.get_block(current_chunk.block_ids(), block_coord.offset());
        match &block.render_info {
            Some(RenderInfo::Empty(_)) => {}
            Some(RenderInfo::Cube(cube)) => self.emit_solid_cube(
                chunk_data,
                block_coord,
                current_chunk.block_ids(),
                vtx,
                idx,
                cube,
            ),
            Some(RenderInfo::CubeEx(_)) => {
                log::warn!("Meshing a CubeEx which we can't mesh yet");
            }
            None => {}
        }
    }

    fn emit_solid_cube(
        &self,
        chunk_data: &FxHashMap<ChunkCoordinate, ClientChunk>,
        coord: BlockCoordinate,
        own_ids: &[BlockId; 4096],
        vtx: &mut Vec<CubeGeometryVertex>,
        idx: &mut Vec<u32>,
        render_info: &CubeRenderInfo,
    ) {
        const FULL_CUBE_EXTENTS: CubeExtents = CubeExtents {
            x: (-0.5, 0.5),
            y: (-0.5, 0.5),
            z: (-0.5, 0.5),
        };
        let e = FULL_CUBE_EXTENTS;

        let chunk = coord.chunk();
        let offset = coord.offset();
        let pos = vec3(offset.x.into(), offset.y.into(), offset.z.into());

        if !coord.try_delta(-1, 0, 0).map_or(false, |x| {
            is_cube(self.get_block_maybe_neighbor(chunk_data, own_ids, chunk, x))
        }) {
            let frame = self.get_texture(render_info.tex_left.tex_name()).unwrap();
            emit_cube_face_vk(
                pos,
                frame,
                self.texture_atlas.dimensions(),
                CubeFace::XMinus,
                vtx,
                idx,
                e,
            );
        }

        if !coord.try_delta(1, 0, 0).map_or(false, |x| {
            is_cube(self.get_block_maybe_neighbor(chunk_data, own_ids, chunk, x))
        }) {
            let frame = self.get_texture(render_info.tex_left.tex_name()).unwrap();
            emit_cube_face_vk(
                pos,
                frame,
                self.texture_atlas.dimensions(),
                CubeFace::XPlus,
                vtx,
                idx,
                e,
            );
        }
        if !coord.try_delta(0, -1, 0).map_or(false, |x| {
            is_cube(self.get_block_maybe_neighbor(chunk_data, own_ids, chunk, x))
        }) {
            let frame = self.get_texture(render_info.tex_bottom.tex_name()).unwrap();
            emit_cube_face_vk(
                pos,
                frame,
                self.texture_atlas.dimensions(),
                CubeFace::YMinus,
                vtx,
                idx,
                e,
            );
        }
        if !coord.try_delta(0, 1, 0).map_or(false, |x| {
            is_cube(self.get_block_maybe_neighbor(chunk_data, own_ids, chunk, x))
        }) {
            let frame = self.get_texture(render_info.tex_top.tex_name()).unwrap();
            emit_cube_face_vk(
                pos,
                frame,
                self.texture_atlas.dimensions(),
                CubeFace::YPlus,
                vtx,
                idx,
                e,
            );
        }
        if !coord.try_delta(0, 0, -1).map_or(false, |x| {
            is_cube(self.get_block_maybe_neighbor(chunk_data, own_ids, chunk, x))
        }) {
            let frame = self.get_texture(render_info.tex_front.tex_name()).unwrap();
            emit_cube_face_vk(
                pos,
                frame,
                self.texture_atlas.dimensions(),
                CubeFace::ZMinus,
                vtx,
                idx,
                e,
            );
        }
        if !coord.try_delta(0, 0, 1).map_or(false, |x| {
            is_cube(self.get_block_maybe_neighbor(chunk_data, own_ids, chunk, x))
        }) {
            let frame = self.get_texture(render_info.tex_back.tex_name()).unwrap();
            emit_cube_face_vk(
                pos,
                frame,
                self.texture_atlas.dimensions(),
                CubeFace::ZPlus,
                vtx,
                idx,
                e,
            );
        }
    }

    fn get_block(&self, ids: &[BlockId; 4096], coord: ChunkOffset) -> &BlockTypeDef {
        let block_id = ids[coord.as_index()];

        let def = self
            .block_defs
            .get_blockdef(block_id)
            .unwrap_or_else(|| self.block_defs.get_fallback_blockdef());
        def
    }

    fn get_block_maybe_neighbor(
        &self,
        all_chunks: &FxHashMap<ChunkCoordinate, ClientChunk>,
        own_ids: &[BlockId; 4096],
        own_chunk: ChunkCoordinate,
        target: BlockCoordinate,
    ) -> Option<&BlockTypeDef> {
        let target_chunk = target.chunk();
        if target_chunk == own_chunk {
            Some(self.get_block(own_ids, target.offset()))
        } else {
            all_chunks
                .get(&target_chunk)
                .map(|x| self.get_block(x.block_ids(), target.offset()))
        }
    }

    pub(crate) fn make_pointee_cube(
        &self,
        player_position: cgmath::Vector3<f64>,
        pointee: cuberef_core::coordinates::BlockCoordinate,
    ) -> Result<CubeGeometryDrawCall> {
        let mut vtx = vec![];
        let mut idx = vec![];
        let frame = self.get_texture(SELECTION_RECTANGLE).unwrap();
        const POINTEE_SELECTION_EXTENTS: CubeExtents = CubeExtents {
            x: (-0.51, 0.51),
            y: (-0.51, 0.51),
            z: (-0.51, 0.51),
        };
        let e = POINTEE_SELECTION_EXTENTS;
        let vk_pos = Vector3::zero();
        emit_cube_face_vk(
            vk_pos,
            frame,
            self.texture_atlas.dimensions(),
            CubeFace::XMinus,
            &mut vtx,
            &mut idx,
            e,
        );
        emit_cube_face_vk(
            vk_pos,
            frame,
            self.texture_atlas.dimensions(),
            CubeFace::XPlus,
            &mut vtx,
            &mut idx,
            e,
        );
        emit_cube_face_vk(
            vk_pos,
            frame,
            self.texture_atlas.dimensions(),
            CubeFace::XMinus,
            &mut vtx,
            &mut idx,
            e,
        );
        emit_cube_face_vk(
            vk_pos,
            frame,
            self.texture_atlas.dimensions(),
            CubeFace::XPlus,
            &mut vtx,
            &mut idx,
            e,
        );
        emit_cube_face_vk(
            vk_pos,
            frame,
            self.texture_atlas.dimensions(),
            CubeFace::YMinus,
            &mut vtx,
            &mut idx,
            e,
        );
        emit_cube_face_vk(
            vk_pos,
            frame,
            self.texture_atlas.dimensions(),
            CubeFace::YPlus,
            &mut vtx,
            &mut idx,
            e,
        );
        emit_cube_face_vk(
            vk_pos,
            frame,
            self.texture_atlas.dimensions(),
            CubeFace::ZMinus,
            &mut vtx,
            &mut idx,
            e,
        );
        emit_cube_face_vk(
            vk_pos,
            frame,
            self.texture_atlas.dimensions(),
            CubeFace::ZPlus,
            &mut vtx,
            &mut idx,
            e,
        );
        let vtx = Buffer::from_iter(
            self.allocator(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            vtx.into_iter(),
        )?;
        let idx = Buffer::from_iter(
            self.allocator(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            idx.into_iter(),
        )?;
        let offset = (vec3(pointee.x as f64, pointee.y as f64, pointee.z as f64) - player_position)
            .mul_element_wise(Vector3::new(1., -1., 1.));
        Ok(CubeGeometryDrawCall {
            vertex: vtx,
            index: idx,
            model_matrix: Matrix4::from_translation(offset.cast().unwrap()),
        })
    }
}

async fn build_texture_atlas<T: AsyncTextureLoader>(
    block_defs: &ClientBlockTypeManager,
    mut texture_loader: T,
) -> Result<(DynamicImage, HashMap<String, Rect>), Error> {
    let mut all_texture_names = HashSet::new();
    for def in block_defs.all_block_defs() {
        match &def.render_info {
            Some(RenderInfo::Cube(cube)) => {
                if let Some(tex) = &cube.tex_back {
                    all_texture_names.insert(tex.texture_name.clone());
                }
                if let Some(tex) = &cube.tex_front {
                    all_texture_names.insert(tex.texture_name.clone());
                }
                if let Some(tex) = &cube.tex_left {
                    all_texture_names.insert(tex.texture_name.clone());
                }
                if let Some(tex) = &cube.tex_right {
                    all_texture_names.insert(tex.texture_name.clone());
                }
                if let Some(tex) = &cube.tex_top {
                    all_texture_names.insert(tex.texture_name.clone());
                }
                if let Some(tex) = &cube.tex_bottom {
                    all_texture_names.insert(tex.texture_name.clone());
                }
            }
            Some(RenderInfo::CubeEx(_)) => {
                log::warn!(
                    "Got a CubeEx block {} which we can't render or put in the atlas",
                    def.short_name
                );
            }
            Some(RenderInfo::Empty(_)) => {}
            None => {
                log::warn!("Got a block without renderinfo: {}", def.short_name)
            }
        }
    }

    let mut all_textures = HashMap::new();
    for x in all_texture_names {
        let texture = texture_loader.load_texture(&x).await?;
        all_textures.insert(x, texture);
    }

    let config = texture_packer::TexturePackerConfig {
        // todo break these out into config
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
    // TODO move these files to a sensible location
    texture_packer
        .pack_own(
            String::from(FALLBACK_UNKNOWN_TEXTURE),
            ImageImporter::import_from_memory(include_bytes!("block_unknown.png")).unwrap(),
        )
        .map_err(|x| Error::msg(format!("Texture pack failed: {:?}", x)))?;

    texture_packer
        .pack_own(
            String::from(SELECTION_RECTANGLE),
            ImageImporter::import_from_memory(include_bytes!("selection.png")).unwrap(),
        )
        .map_err(|x| Error::msg(format!("Texture pack failed: {:?}", x)))?;

    for (name, texture) in all_textures {
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
    Ok((texture_atlas, texture_coords))
}

fn is_cube(def: Option<&BlockTypeDef>) -> bool {
    if let Some(def) = def {
        match def.render_info {
            Some(blocks_proto::block_type_def::RenderInfo::Cube(..)) => true,
            _ => false,
        }
    } else {
        false
    }
}

fn fallback_texture() -> Option<TextureReference> {
    Some(TextureReference {
        texture_name: FALLBACK_UNKNOWN_TEXTURE.to_string(),
    })
}

fn make_fallback() -> blocks_proto::BlockTypeDef {
    blocks_proto::BlockTypeDef {
        render_info: Some(blocks_proto::block_type_def::RenderInfo::Cube(
            blocks_proto::CubeRenderInfo {
                tex_left: fallback_texture(),
                tex_right: fallback_texture(),
                tex_top: fallback_texture(),
                tex_bottom: fallback_texture(),
                tex_front: fallback_texture(),
                tex_back: fallback_texture(),
            },
        )),

        ..Default::default()
    }
}

enum TextureRotation {
    Zero,
    Ccw90,
    Flip180,
    Cw90,
}

#[inline]
fn make_cgv(
    coord: Vector3<f32>,
    x: f32,
    y: f32,
    z: f32,
    tex_uv: cgmath::Vector2<f32>,
) -> CubeGeometryVertex {
    CubeGeometryVertex {
        position: [coord.x + x, -(coord.y) + y, coord.z + z],
        uv_texcoord: tex_uv.into(),
        brightness: 1.0,
    }
}

pub(crate) fn emit_cube_face_vk(
    coord: Vector3<f32>,
    frame: Rect,
    tex_dimension: (u32, u32),
    face: CubeFace,
    vert_buf: &mut Vec<CubeGeometryVertex>,
    idx_buf: &mut Vec<u32>,
    e: CubeExtents,
) {
    let width = (tex_dimension.0) as f32;
    let height = (tex_dimension.1) as f32;
    let l = frame.left() as f32 / width;
    let r = (frame.right() + 1) as f32 / width;
    let t = frame.top() as f32 / height;
    let b = (frame.bottom() + 1) as f32 / height;
    // todo handle rotating textures (both in the texture ref and here)
    // Possibly at the caller?
    let tl = Vector2::new(l, t);
    let bl = Vector2::new(l, b);
    let tr = Vector2::new(r, t);
    let br = Vector2::new(r, b);

    let mut vertices = match face {
        CubeFace::ZMinus => vec![
            make_cgv(coord, e.x.1, e.y.0, e.z.0, tl),
            make_cgv(coord, e.x.1, e.y.1, e.z.0, bl),
            make_cgv(coord, e.x.0, e.y.1, -e.z.1, br),
            make_cgv(coord, e.x.0, e.y.0, -e.z.1, tr),
        ],
        CubeFace::ZPlus => vec![
            make_cgv(coord, e.x.0, e.y.0, e.z.1, tl),
            make_cgv(coord, e.x.0, e.y.1, e.z.1, bl),
            make_cgv(coord, e.x.1, e.y.1, e.z.1, br),
            make_cgv(coord, e.x.1, e.y.0, e.z.1, tr),
        ],
        CubeFace::XPlus => vec![
            make_cgv(coord, e.x.1, e.y.0, e.z.1, tl),
            make_cgv(coord, e.x.1, e.y.1, e.z.1, bl),
            make_cgv(coord, e.x.1, e.y.1, -e.z.1, br),
            make_cgv(coord, e.x.1, e.y.0, -e.z.1, tr),
        ],
        CubeFace::XMinus => vec![
            make_cgv(coord, e.x.0, e.y.0, e.z.0, tl),
            make_cgv(coord, e.x.0, e.y.1, e.z.0, bl),
            make_cgv(coord, e.x.0, e.y.1, e.z.1, br),
            make_cgv(coord, e.x.0, e.y.0, e.z.1, tr),
        ],
        CubeFace::YPlus => vec![
            make_cgv(coord, e.x.0, e.y.0, e.z.0, tl),
            make_cgv(coord, e.x.0, e.y.0, e.z.1, bl),
            make_cgv(coord, e.x.1, e.y.0, e.z.1, br),
            make_cgv(coord, e.x.1, e.y.0, -e.z.1, tr),
        ],
        CubeFace::YMinus => vec![
            make_cgv(coord, e.x.0, e.y.1, e.z.0, tl),
            make_cgv(coord, e.x.1, e.y.1, e.z.0, bl),
            make_cgv(coord, e.x.1, e.y.1, e.z.1, br),
            make_cgv(coord, e.x.0, e.y.1, e.z.1, tr),
        ],
    };
    let si: u32 = vert_buf.len().try_into().unwrap();
    if si > (u32::MAX - 8) {
        panic!("vertex buffer got too big");
    }
    let mut indices = vec![si, si + 1, si + 2, si, si + 2, si + 3];
    vert_buf.append(&mut vertices);
    idx_buf.append(&mut indices);
}

trait TexRefHelper {
    fn tex_name(&self) -> &str;
}
impl TexRefHelper for Option<TextureReference> {
    fn tex_name(&self) -> &str {
        match self {
            Some(x) => &x.texture_name,
            None => FALLBACK_UNKNOWN_TEXTURE,
        }
    }
}

#[async_trait]
pub(crate) trait AsyncTextureLoader {
    async fn load_texture(&mut self, tex_name: &str) -> Result<DynamicImage>;
}
