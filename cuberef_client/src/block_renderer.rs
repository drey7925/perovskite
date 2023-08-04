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

use cgmath::num_traits::Num;
use cgmath::{vec3, ElementWise, Matrix4, Vector2, Vector3, Zero};

use cuberef_core::constants::textures::FALLBACK_UNKNOWN_TEXTURE;
use cuberef_core::coordinates::{BlockCoordinate, ChunkCoordinate};
use cuberef_core::protocol::blocks::block_type_def::RenderInfo;
use cuberef_core::protocol::blocks::{
    self as blocks_proto, BlockTypeDef, CubeRenderInfo, CubeRenderMode,
};
use cuberef_core::protocol::render::TextureReference;
use cuberef_core::{block_id::BlockId, coordinates::ChunkOffset};

use anyhow::{ensure, Context, Error, Result};
use image::DynamicImage;

use texture_packer::importer::ImageImporter;
use texture_packer::Rect;
use tonic::async_trait;
use tracy_client::span;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{
    AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryUsage,
    StandardMemoryAllocator,
};

use crate::game_state::chunk::ClientChunk;
use crate::game_state::{make_fallback_blockdef, ChunkManagerView, ChunkManagerClonedView};
use crate::vulkan::shaders::cube_geometry::{CubeGeometryDrawCall, CubeGeometryVertex};
use crate::vulkan::{Texture2DHolder, VulkanContext};

const SELECTION_RECTANGLE: &str = "builtin:selection_rectangle";

/// Given in game world coordinates (Y is up)
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
            fallback_block_def: make_fallback_blockdef(),
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

/// The extents of a single axis-aligned cube.
///
/// ```ignore
///    1      5
///    +------+
///   /|     /|
///  / |    / |
/// +------+4 |
/// |0 |   |  |
/// |  +---|--+
/// | /3   | /7
/// |/     |/
/// +------+
/// 2      6
///
///    Z
///   /
///  /
/// +---X
/// |
/// |
/// Y
/// ```
#[derive(Clone, Copy, Debug)]
pub struct CubeExtents {
    v: [Vector3<f32>; 8],
    /// Given in the same order as [`CubeFace`].
    neighbor: [(i8, i8, i8); 6],
}
impl CubeExtents {
    pub const fn new(x: (f32, f32), y: (f32, f32), z: (f32, f32)) -> Self {
        Self {
            v: [
                vec3(x.0, y.0, z.0),
                vec3(x.0, y.0, z.1),
                vec3(x.0, y.1, z.0),
                vec3(x.0, y.1, z.1),
                vec3(x.1, y.0, z.0),
                vec3(x.1, y.0, z.1),
                vec3(x.1, y.1, z.0),
                vec3(x.1, y.1, z.1),
            ],
            neighbor: [
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            ],
        }
    }

    pub fn rotate_y(self, variant: u16) -> CubeExtents {
        let mut v = self.v;
        let mut neighbor = self.neighbor;
        for i in 0..8 {
            v[i] = Vector3::from(rotate_y((v[i].x, v[i].y, v[i].z), variant));
        }
        for i in 0..6 {
            neighbor[i] = rotate_y(neighbor[i], variant);
        }
        Self { v, neighbor }
    }
}

#[inline]
fn rotate_y<T: Num + std::ops::Neg<Output = T>>(
    c: (T, T, T),
    angle_90_deg_units: u16,
) -> (T, T, T) {
    match angle_90_deg_units % 4 {
        0 => c,
        1 => (c.2, c.1, -c.0),
        2 => (-c.0, c.1, -c.2),
        3 => (-c.2, c.1, c.0),
        _ => unreachable!(),
    }
}

const CUBE_EXTENTS_FACE_ORDER: [CubeFace; 6] = [
    CubeFace::XPlus,
    CubeFace::XMinus,
    CubeFace::YPlus,
    CubeFace::YMinus,
    CubeFace::ZPlus,
    CubeFace::ZMinus,
];

#[derive(Clone)]
pub(crate) struct VkChunkPass {
    pub(crate) vtx: Subbuffer<[CubeGeometryVertex]>,
    pub(crate) idx: Subbuffer<[u32]>,
}
impl VkChunkPass {
    pub(crate) fn from_buffers(
        vtx: Vec<CubeGeometryVertex>,
        idx: Vec<u32>,
        allocator: &StandardMemoryAllocator,
    ) -> Result<Option<VkChunkPass>> {
        if vtx.is_empty() {
            Ok(None)
        } else {
            Ok(Some(VkChunkPass {
                vtx: Buffer::from_iter(
                    allocator,
                    BufferCreateInfo {
                        usage: BufferUsage::VERTEX_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        usage: MemoryUsage::Upload,
                        ..Default::default()
                    },
                    vtx.into_iter(),
                )?,
                idx: Buffer::from_iter(
                    allocator,
                    BufferCreateInfo {
                        usage: BufferUsage::INDEX_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        usage: MemoryUsage::Upload,
                        ..Default::default()
                    },
                    idx.into_iter(),
                )?,
            }))
        }
    }
}

#[derive(Clone)]
pub(crate) struct VkChunkVertexData {
    pub(crate) solid_opaque: Option<VkChunkPass>,
    pub(crate) transparent: Option<VkChunkPass>,
    pub(crate) translucent: Option<VkChunkPass>,
}
impl VkChunkVertexData {
    pub(crate) fn clone_if_nonempty(&self) -> Option<Self> {
        if self.solid_opaque.is_some() || self.transparent.is_some() || self.translucent.is_some() {
            Some(self.clone())
        } else {
            None
        }
    }
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
    fn get_texture(&self, id: &str) -> Rect {
        self.texture_coords
            .get(&id.to_string())
            .copied()
            .unwrap_or_else(|| *self.texture_coords.get(FALLBACK_UNKNOWN_TEXTURE).unwrap())
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

    pub(crate) fn atlas(&self) -> &Texture2DHolder {
        &self.texture_atlas
    }

    pub(crate) fn allocator(&self) -> &GenericMemoryAllocator<Arc<FreeListAllocator>> {
        &self.allocator
    }

    pub(crate) fn mesh_chunk(
        &self,
        chunk_data: &ChunkManagerClonedView,
        current_chunk: &ClientChunk,
    ) -> Result<VkChunkVertexData> {
        Ok(VkChunkVertexData {
            solid_opaque: self.mesh_chunk_subpass(
                chunk_data,
                current_chunk,
                |block| match block.render_info {
                    Some(RenderInfo::Cube(CubeRenderInfo { render_mode: x, .. })) => {
                        x == CubeRenderMode::SolidOpaque.into()
                    }
                    Some(_) | None => false,
                },
                |_block, neighbor| {
                    neighbor.is_some_and(|neighbor| match neighbor.render_info {
                        Some(RenderInfo::Cube(CubeRenderInfo { render_mode: x, .. })) => {
                            x == CubeRenderMode::SolidOpaque.into()
                        }
                        Some(_) | None => false,
                    })
                },
            )?,
            transparent: self.mesh_chunk_subpass(
                chunk_data,
                current_chunk,
                |block| match block.render_info {
                    Some(RenderInfo::Cube(CubeRenderInfo { render_mode: x, .. })) => {
                        x == CubeRenderMode::Transparent.into()
                    }
                    Some(_) | None => false,
                },
                |block, neighbor| {
                    neighbor.is_some_and(|neighbor| {
                        BlockId(block.id).equals_ignore_variant(BlockId(neighbor.id))
                    })
                },
            )?,
            translucent: self.mesh_chunk_subpass(
                chunk_data,
                current_chunk,
                |block| match block.render_info {
                    Some(RenderInfo::Cube(CubeRenderInfo { render_mode: x, .. })) => {
                        x == CubeRenderMode::Translucent.into()
                    }
                    Some(_) | None => false,
                },
                |block, neighbor| {
                    neighbor.is_some_and(|neighbor| {
                        if BlockId(block.id).equals_ignore_variant(BlockId(neighbor.id)) {
                            return true;
                        }
                        match &neighbor.render_info {
                            Some(RenderInfo::Cube(x)) => {
                                x.render_mode() == CubeRenderMode::SolidOpaque
                            }
                            Some(_) => false,
                            None => false,
                        }
                    })
                },
            )?,
        })
    }

    pub(crate) fn mesh_chunk_subpass<F, G>(
        &self,
        chunk_data: &ChunkManagerClonedView,
        current_chunk: &ClientChunk,
        // closure taking a block and returning whether this subpass should render it
        include_block_when: F,
        // closure taking a block and its neighbor, and returning whether we should render the face of our block that faces the given neighbor
        suppress_face_when: G,
    ) -> Result<Option<VkChunkPass>>
    where
        F: Fn(&BlockTypeDef) -> bool,
        G: Fn(&BlockTypeDef, Option<&BlockTypeDef>) -> bool,
    {
        let _span = span!("mesh subpass");
        let mut vtx = Vec::new();
        let mut idx = Vec::new();
        for x in 0..16 {
            for y in 0..16 {
                for z in 0..16 {
                    let offset = ChunkOffset { x, y, z };

                    let block_ids = current_chunk.block_ids();
                    let (block, variant) = self.get_block(&block_ids, offset);
                    if let Some(RenderInfo::Cube(cube_render_info)) = &block.render_info {
                        if include_block_when(block) {
                            self.emit_full_cube(
                                block,
                                variant,
                                chunk_data,
                                current_chunk.coord().with_offset(offset),
                                &current_chunk.block_ids(),
                                &mut vtx,
                                &mut idx,
                                cube_render_info,
                                &suppress_face_when,
                            );
                        }
                    }
                }
            }
        }
        VkChunkPass::from_buffers(vtx, idx, self.allocator())
    }

    fn emit_full_cube<F>(
        &self,
        block: &BlockTypeDef,
        variant: u16,
        chunk_data: &ChunkManagerClonedView,
        coord: BlockCoordinate,
        own_ids: &[BlockId; 4096],
        vtx: &mut Vec<CubeGeometryVertex>,
        idx: &mut Vec<u32>,
        render_info: &CubeRenderInfo,
        suppress_face_when: F,
    ) where
        F: Fn(&BlockTypeDef, Option<&BlockTypeDef>) -> bool,
    {
        let e = get_cube_extents(render_info, variant);

        let chunk = coord.chunk();
        let offset = coord.offset();
        let pos = vec3(offset.x.into(), offset.y.into(), offset.z.into());

        let textures = [
            self.get_texture(render_info.tex_right.tex_name()),
            self.get_texture(render_info.tex_left.tex_name()),
            self.get_texture(render_info.tex_top.tex_name()),
            self.get_texture(render_info.tex_bottom.tex_name()),
            self.get_texture(render_info.tex_back.tex_name()),
            self.get_texture(render_info.tex_front.tex_name()),
        ];
        for i in 0..6 {
            let (n_x, n_y, n_z) = e.neighbor[i];
            if suppress_face_when(
                block,
                coord.try_delta(n_x.into(), n_y.into(), n_z.into()).and_then(|neighbor| {
                    self.get_block_maybe_neighbor(chunk_data, own_ids, chunk, neighbor)
                }),
            ) {
                continue;
            };
            
            emit_cube_face_vk(
                pos,
                textures[i],
                self.texture_atlas.dimensions(),
                CUBE_EXTENTS_FACE_ORDER[i],
                vtx,
                idx,
                e,
            );
        }

        // if !suppress_face_when(
        //     block,
        //     coord.try_delta(-1, 0, 0).and_then(|neighbor| {
        //         self.get_block_maybe_neighbor(chunk_data, own_ids, chunk, neighbor)
        //     }),
        // ) {
        //     let frame = self.get_texture(render_info.tex_left.tex_name());
        //     emit_cube_face_vk(
        //         pos,
        //         frame,
        //         self.texture_atlas.dimensions(),
        //         CubeFace::XMinus,
        //         vtx,
        //         idx,
        //         e,
        //     );
        // }

        // if !suppress_face_when(
        //     block,
        //     coord.try_delta(1, 0, 0).and_then(|neighbor| {
        //         self.get_block_maybe_neighbor(chunk_data, own_ids, chunk, neighbor)
        //     }),
        // ) {
        //     let frame = self.get_texture(render_info.tex_left.tex_name());
        //     emit_cube_face_vk(
        //         pos,
        //         frame,
        //         self.texture_atlas.dimensions(),
        //         CubeFace::XPlus,
        //         vtx,
        //         idx,
        //         e,
        //     );
        // }
        // if !suppress_face_when(
        //     block,
        //     coord.try_delta(0, -1, 0).and_then(|neighbor| {
        //         self.get_block_maybe_neighbor(chunk_data, own_ids, chunk, neighbor)
        //     }),
        // ) {
        //     let frame = self.get_texture(render_info.tex_bottom.tex_name());
        //     emit_cube_face_vk(
        //         pos,
        //         frame,
        //         self.texture_atlas.dimensions(),
        //         CubeFace::YMinus,
        //         vtx,
        //         idx,
        //         e,
        //     );
        // }
        // if !suppress_face_when(
        //     block,
        //     coord.try_delta(0, 1, 0).and_then(|neighbor| {
        //         self.get_block_maybe_neighbor(chunk_data, own_ids, chunk, neighbor)
        //     }),
        // ) {
        //     let frame = self.get_texture(render_info.tex_top.tex_name());
        //     emit_cube_face_vk(
        //         pos,
        //         frame,
        //         self.texture_atlas.dimensions(),
        //         CubeFace::YPlus,
        //         vtx,
        //         idx,
        //         e,
        //     );
        // }
        // if !suppress_face_when(
        //     block,
        //     coord.try_delta(0, 0, -1).and_then(|neighbor| {
        //         self.get_block_maybe_neighbor(chunk_data, own_ids, chunk, neighbor)
        //     }),
        // ) {
        //     let frame = self.get_texture(render_info.tex_front.tex_name());
        //     emit_cube_face_vk(
        //         pos,
        //         frame,
        //         self.texture_atlas.dimensions(),
        //         CubeFace::ZMinus,
        //         vtx,
        //         idx,
        //         e,
        //     );
        // }
        // if !suppress_face_when(
        //     block,
        //     coord.try_delta(0, 0, 1).and_then(|neighbor| {
        //         self.get_block_maybe_neighbor(chunk_data, own_ids, chunk, neighbor)
        //     }),
        // ) {
        //     let frame = self.get_texture(render_info.tex_back.tex_name());
        //     emit_cube_face_vk(
        //         pos,
        //         frame,
        //         self.texture_atlas.dimensions(),
        //         CubeFace::ZPlus,
        //         vtx,
        //         idx,
        //         e,
        //     );
        // }
    }

    fn get_block(&self, ids: &[BlockId; 4096], coord: ChunkOffset) -> (&BlockTypeDef, u16) {
        let block_id = ids[coord.as_index()];

        let def = self
            .block_defs
            .get_blockdef(block_id)
            .unwrap_or_else(|| self.block_defs.get_fallback_blockdef());
        (def, block_id.variant())
    }

    fn get_block_maybe_neighbor(
        &self,
        all_chunks: &ChunkManagerClonedView,
        own_ids: &[BlockId; 4096],
        own_chunk: ChunkCoordinate,
        target: BlockCoordinate,
    ) -> Option<&BlockTypeDef> {
        let target_chunk = target.chunk();
        if target_chunk == own_chunk {
            Some(self.get_block(own_ids, target.offset()).0)
        } else {
            all_chunks
                .get(&target_chunk)
                .map(|x| self.get_block(&x.block_ids(), target.offset()).0)
        }
    }

    pub(crate) fn make_pointee_cube(
        &self,
        player_position: cgmath::Vector3<f64>,
        pointee: cuberef_core::coordinates::BlockCoordinate,
    ) -> Result<CubeGeometryDrawCall> {
        let mut vtx = vec![];
        let mut idx = vec![];
        let frame = *self.texture_coords.get(SELECTION_RECTANGLE).unwrap();
        const POINTEE_SELECTION_EXTENTS: CubeExtents =
            CubeExtents::new((-0.51, 0.51), (-0.51, 0.51), (-0.51, 0.51));
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
            model_matrix: Matrix4::from_translation(offset.cast().unwrap()),
            models: VkChunkVertexData {
                solid_opaque: None,
                transparent: Some(VkChunkPass { vtx, idx }),
                translucent: None,
            },
        })
    }
}

fn get_cube_extents(render_info: &CubeRenderInfo, variant: u16) -> CubeExtents {
    const FULL_CUBE_EXTENTS: CubeExtents = CubeExtents::new((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5));
    match render_info.variant_effect() {
        blocks_proto::CubeVariantEffect::None => FULL_CUBE_EXTENTS,
        blocks_proto::CubeVariantEffect::RotateNesw => FULL_CUBE_EXTENTS.rotate_y(variant),
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

pub(crate) fn fallback_texture() -> Option<TextureReference> {
    Some(TextureReference {
        texture_name: FALLBACK_UNKNOWN_TEXTURE.to_string(),
    })
}

#[inline]
fn make_cgv(coord: Vector3<f32>, tex_uv: cgmath::Vector2<f32>) -> CubeGeometryVertex {
    CubeGeometryVertex {
        position: [coord.x, coord.y, coord.z],
        uv_texcoord: tex_uv.into(),
        brightness: 1.0,
        global_brightness_contribution: 0.0,
    }
}

/// Emits a single face of a cube into the given buffers.
/// Arguments:
///    coord: The center of the overall cube, in world space, with y-axis up (opposite Vulkan)
///     This is the center of the cube body, not the current face.
///    frame: The texture for the face.
pub(crate) fn emit_cube_face_vk(
    coord: Vector3<f32>,
    frame: Rect,
    tex_dimension: (u32, u32),
    face: CubeFace,
    vert_buf: &mut Vec<CubeGeometryVertex>,
    idx_buf: &mut Vec<u32>,
    e: CubeExtents,
) {
    // Flip the coordinate system to Vulkan
    let coord = vec3(coord.x, -coord.y, coord.z);
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
            make_cgv(coord + e.v[0], tl),
            make_cgv(coord + e.v[2], bl),
            make_cgv(coord + e.v[6], br),
            make_cgv(coord + e.v[4], tr),
        ],
        CubeFace::ZPlus => vec![
            make_cgv(coord + e.v[5], tl),
            make_cgv(coord + e.v[7], bl),
            make_cgv(coord + e.v[3], br),
            make_cgv(coord + e.v[1], tr),
        ],
        CubeFace::XMinus => vec![
            make_cgv(coord + e.v[1], tl),
            make_cgv(coord + e.v[3], bl),
            make_cgv(coord + e.v[2], br),
            make_cgv(coord + e.v[0], tr),
        ],
        CubeFace::XPlus => vec![
            make_cgv(coord + e.v[4], tl),
            make_cgv(coord + e.v[6], bl),
            make_cgv(coord + e.v[7], br),
            make_cgv(coord + e.v[5], tr),
        ],
        // For Y+ and Y-, the top of the texture faces the front of the cube (prior to any rotations)
        // Y- is the bottom face (opposite Vulkan), so it takes Y+ vulkan coordinates
        CubeFace::YMinus => vec![
            make_cgv(coord + e.v[2], tl),
            make_cgv(coord + e.v[3], bl),
            make_cgv(coord + e.v[7], br),
            make_cgv(coord + e.v[6], tr),
        ],
        CubeFace::YPlus => vec![
            make_cgv(coord + e.v[4], tl),
            make_cgv(coord + e.v[5], bl),
            make_cgv(coord + e.v[1], br),
            make_cgv(coord + e.v[0], tr),
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
