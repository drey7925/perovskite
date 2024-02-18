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

use std::collections::HashSet;

use std::sync::Arc;

use cgmath::num_traits::Num;
use cgmath::{vec3, ElementWise, InnerSpace, Matrix4, Vector2, Vector3, Zero};

use perovskite_core::constants::blocks::AIR;
use perovskite_core::constants::textures::FALLBACK_UNKNOWN_TEXTURE;

use perovskite_core::protocol::blocks::block_type_def::RenderInfo;
use perovskite_core::protocol::blocks::{
    self as blocks_proto, AxisAlignedBoxes, BlockTypeDef, CubeRenderInfo, CubeRenderMode,
};
use perovskite_core::protocol::render::{TextureCrop, TextureReference};
use perovskite_core::{block_id::BlockId, coordinates::ChunkOffset};

use anyhow::{ensure, Context, Error, Result};

use rustc_hash::FxHashMap;
use texture_packer::importer::ImageImporter;
use texture_packer::Rect;
use tonic::async_trait;
use tracy_client::span;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{
    AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryUsage,
    StandardMemoryAllocator,
};

use crate::cache::CacheManager;
use crate::game_state::chunk::{ChunkDataView, ChunkOffsetExt, LockedChunkDataView};
use crate::game_state::make_fallback_blockdef;
use crate::vulkan::shaders::cube_geometry::{CubeGeometryDrawCall, CubeGeometryVertex};
use crate::vulkan::{Texture2DHolder, VulkanContext};

use bitvec::prelude as bv;

const SELECTION_RECTANGLE: &str = "builtin:selection_rectangle";
const TESTONLY_ENTITY: &str = "builtin:testonly_entity";

/// Given in game world coordinates (Y is up)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd)]
#[repr(u8)]
pub(crate) enum CubeFace {
    XPlus,
    XMinus,
    YPlus,
    YMinus,
    ZPlus,
    ZMinus,
    PlantXPlusZPlus,
    PlantXPlusZMinus,
    PlantXMinusZPlus,
    PlantXMinusZMinus,
}
impl CubeFace {
    #[inline(always)]
    fn index(&self) -> usize {
        *self as usize
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RectF32 {
    l: f32,
    t: f32,
    w: f32,
    h: f32,
}
impl RectF32 {
    fn new(l: f32, t: f32, w: f32, h: f32) -> Self {
        Self { l, t, w, h }
    }
    fn top(&self) -> f32 {
        self.t
    }
    fn bottom(&self) -> f32 {
        self.t + self.h
    }
    fn left(&self) -> f32 {
        self.l
    }
    fn right(&self) -> f32 {
        self.l + self.w
    }
}
impl From<Rect> for RectF32 {
    fn from(rect: Rect) -> Self {
        Self::new(rect.x as f32, rect.y as f32, rect.w as f32, rect.h as f32)
    }
}

pub(crate) struct ClientBlockTypeManager {
    block_defs: Vec<Option<blocks_proto::BlockTypeDef>>,
    fallback_block_def: blocks_proto::BlockTypeDef,
    air_block: BlockId,
    light_propagators: bv::BitVec,
    light_emitters: Vec<u8>,
    solid_opaque_blocks: bv::BitVec,
    transparent_render_blocks: bv::BitVec,
    translucent_render_blocks: bv::BitVec,
    name_to_id: FxHashMap<String, BlockId>,
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

        let mut light_propagators = bv::BitVec::new();
        light_propagators.resize(BlockId(max_id).index() + 1, false);

        let mut solid_opaque_blocks = bv::BitVec::new();
        solid_opaque_blocks.resize(BlockId(max_id).index() + 1, false);

        let mut transparent_render_blocks = bv::BitVec::new();
        transparent_render_blocks.resize(BlockId(max_id).index() + 1, false);

        let mut translucent_render_blocks = bv::BitVec::new();
        translucent_render_blocks.resize(BlockId(max_id).index() + 1, false);

        let mut light_emitters = Vec::new();
        light_emitters.resize(BlockId(max_id).index() + 1, 0);

        let mut name_to_id = FxHashMap::default();

        let mut air_block = BlockId::from(u32::MAX);
        for def in server_defs {
            let id = BlockId(def.id);
            ensure!(id.variant() == 0);
            ensure!(block_defs[id.index()].is_none());
            if def.short_name == AIR {
                air_block = id;
            }
            name_to_id.insert(def.short_name.clone(), id);
            if def.allow_light_propagation {
                light_propagators.set(id.index(), true);
            }
            let light_emission = if def.light_emission > 15 {
                log::warn!(
                    "Clamping light emission of {} from {} to 15",
                    def.short_name,
                    def.light_emission
                );
                15
            } else {
                def.light_emission as u8
            };
            if let Some(RenderInfo::Cube(render_info)) = &def.render_info {
                if render_info.render_mode() == CubeRenderMode::SolidOpaque {
                    solid_opaque_blocks.set(id.index(), true);
                }
            }

            let is_transparent = match &def.render_info {
                Some(RenderInfo::Cube(CubeRenderInfo { render_mode: x, .. })) => {
                    *x == CubeRenderMode::Transparent.into()
                }
                // plant-like blocks always go into the transparent pass
                Some(RenderInfo::PlantLike(_)) => true,
                // axis-aligned boxes always go into the transparent pass
                Some(RenderInfo::AxisAlignedBoxes(_)) => true,
                Some(_) | None => false,
            };

            if is_transparent {
                transparent_render_blocks.set(id.index(), true);
            }

            let is_translucent = def.render_info.as_ref().is_some_and(|x| match x {
                RenderInfo::Cube(CubeRenderInfo { render_mode: x, .. }) => {
                    *x == CubeRenderMode::Translucent.into()
                }
                _ => false,
            });

            if is_translucent {
                translucent_render_blocks.set(id.index(), true);
            }

            light_emitters[id.index()] = light_emission;
            block_defs[id.index()] = Some(def);
        }
        if air_block.0 == u32::MAX {
            log::warn!("Server didn't send an air block definition");
        }

        Ok(ClientBlockTypeManager {
            block_defs,
            fallback_block_def: make_fallback_blockdef(),
            air_block,
            light_propagators,
            light_emitters,
            solid_opaque_blocks,
            transparent_render_blocks,
            translucent_render_blocks,
            name_to_id,
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

    pub(crate) fn get_block_by_name(&self, name: &str) -> Option<BlockId> {
        self.name_to_id.get(name).copied()
    }

    pub(crate) fn air_block(&self) -> BlockId {
        self.air_block
    }

    /// Determines whether this block propagates light
    #[inline]
    pub(crate) fn propagates_light(&self, id: BlockId) -> bool {
        // Optimization note: This is called from the neighbor propagation code, which is single-threaded for
        // consistency. Unlike mesh generation, which can be parallelized to as many cores as are available, this
        // is *much* more important to speed up. Therefore, these functions get special implementations rather than
        // going through the usual get_blockdef() path. In particular, going through get_blockdef() involves large structs, hence cache pressure
        //
        // Todo: actually benchmark and optimize this. (this comment only provides rationale, but the code hasn't been optimized yet)
        // A typical cacheline on x86 has a size of 64 bytes == 512 bits. A typical l1d cache size is on the order of double-digit KiB -> triple digit kilobits
        // This is plenty of space to fit the bitvecs, although there will still be some cache misses since the chunk won't fit in a typical L1$.
        // Hence, a bitvec is the initial, and likely the fastest, approach. I can't imagine that bloom filtering would be cheap compared to bit reads, even if
        // we do have some caches misses along the way.
        //
        // n.b. the working set for the light propagation algorithm is at least 110 KiB.
        if id.index() < self.light_propagators.len() {
            self.light_propagators[id.index()]
        } else {
            // unknown blocks don't propagate light
            false
        }
    }
    #[inline]
    pub(crate) fn light_emission(&self, id: BlockId) -> u8 {
        if id.index() < self.light_emitters.len() {
            self.light_emitters[id.index()]
        } else {
            0
        }
    }
    #[inline]
    pub(crate) fn is_solid_opaque(&self, id: BlockId) -> bool {
        if id.index() < self.solid_opaque_blocks.len() {
            self.solid_opaque_blocks[id.index()]
        } else {
            // unknown blocks are solid opaque
            true
        }
    }
    #[inline]
    pub(crate) fn is_transparent_render(&self, id: BlockId) -> bool {
        if id.index() < self.transparent_render_blocks.len() {
            self.transparent_render_blocks[id.index()]
        } else {
            // unknown blocks are solid opaque
            false
        }
    }

    #[inline]
    pub(crate) fn is_translucent_render(&self, id: BlockId) -> bool {
        if id.index() < self.translucent_render_blocks.len() {
            self.translucent_render_blocks[id.index()]
        } else {
            // unknown blocks are solid opaque
            false
        }
    }
}

const DEFAULT_FACE_NORMALS: [(i8, i8, i8); 10] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
    (1, 0, 1),
    (1, 0, -1),
    (-1, 0, 1),
    (-1, 0, -1),
];

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
    pub vertices: [Vector3<f32>; 8],
    /// Given in the same order as the first six members of [`CubeFace`].
    pub normals: [(i8, i8, i8); 10],
    pub force: [bool; 6],
}
impl CubeExtents {
    pub const fn new(x: (f32, f32), y: (f32, f32), z: (f32, f32)) -> Self {
        Self {
            vertices: [
                vec3(x.0, y.0, z.0),
                vec3(x.0, y.0, z.1),
                vec3(x.0, y.1, z.0),
                vec3(x.0, y.1, z.1),
                vec3(x.1, y.0, z.0),
                vec3(x.1, y.0, z.1),
                vec3(x.1, y.1, z.0),
                vec3(x.1, y.1, z.1),
            ],
            normals: DEFAULT_FACE_NORMALS,
            force: [false; 6],
        }
    }

    pub fn rotate_y(self, variant: u16) -> CubeExtents {
        let mut vertices = self.vertices;
        let mut normals = self.normals;
        for vtx in &mut vertices {
            *vtx = Vector3::from(rotate_y((vtx.x, vtx.y, vtx.z), variant));
        }
        for neighbor in &mut normals {
            *neighbor = rotate_y(*neighbor, variant);
        }
        /*    XPlus,
        XMinus,
        YPlus,
        YMinus,
        ZPlus,
        ZMinus, */
        let force_swizzled = match variant % 4 {
            0 => self.force,
            // todo double-check these
            // also look into how to encourage this to be vectorized as an AVX/SSE shuffle on x86
            1 => [
                self.force[5],
                self.force[4],
                self.force[2],
                self.force[3],
                self.force[0],
                self.force[1],
            ],
            2 => [
                self.force[1],
                self.force[0],
                self.force[2],
                self.force[3],
                self.force[5],
                self.force[4],
            ],
            3 => [
                self.force[4],
                self.force[5],
                self.force[2],
                self.force[3],
                self.force[1],
                self.force[0],
            ],
            _ => unreachable!(),
        };

        Self {
            vertices,
            normals,
            force: force_swizzled,
        }
    }

    #[inline]
    fn force_face(&self, i: usize) -> bool {
        self.force[i]
    }
}

#[inline]
pub fn rotate_y<T: Num + std::ops::Neg<Output = T>>(
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

pub fn rotate_x<T: Num + std::ops::Neg<Output = T>>(
    c: (T, T, T),
    angle_90_deg_units: u16,
) -> (T, T, T) {
    match angle_90_deg_units % 4 {
        0 => c,
        1 => (c.0, -c.2, c.1),
        2 => (c.0, -c.1, -c.2),
        3 => (c.0, c.2, -c.1),
        _ => unreachable!(),
    }
}

pub fn rotate_z<T: Num + std::ops::Neg<Output = T>>(
    c: (T, T, T),
    angle_90_deg_units: u16,
) -> (T, T, T) {
    match angle_90_deg_units % 4 {
        0 => c,
        1 => (c.1, -c.0, c.2),
        2 => (-c.0, -c.1, c.2),
        3 => (-c.1, c.0, c.2),
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
const PLANTLIKE_FACE_ORDER: [CubeFace; 4] = [
    CubeFace::PlantXPlusZPlus,
    CubeFace::PlantXPlusZMinus,
    CubeFace::PlantXMinusZPlus,
    CubeFace::PlantXMinusZMinus,
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
    pub(crate) bottom_all_solid: bool,
}
impl VkChunkVertexData {
    pub(crate) fn clone_if_nonempty(&self) -> Option<Self> {
        if self.solid_opaque.is_some() || self.transparent.is_some() || self.translucent.is_some() {
            Some(self.clone())
        } else {
            None
        }
    }

    fn empty(bottom_all_solid: bool) -> VkChunkVertexData {
        VkChunkVertexData {
            solid_opaque: None,
            transparent: None,
            translucent: None,
            bottom_all_solid,
        }
    }
}

fn get_texture(
    texture_coords: &FxHashMap<String, Rect>,
    tex: Option<&TextureReference>,
) -> RectF32 {
    let rect = tex
        .and_then(|tex| texture_coords.get(&tex.texture_name).copied())
        .unwrap_or_else(|| *texture_coords.get(FALLBACK_UNKNOWN_TEXTURE).unwrap());
    let mut rect_f = RectF32::new(rect.x as f32, rect.y as f32, rect.w as f32, rect.h as f32);
    if let Some(crop) = tex.and_then(|tex| tex.crop.as_ref()) {
        rect_f = crop_texture(crop, rect_f);
    }
    rect_f
}

fn crop_texture(crop: &TextureCrop, r: RectF32) -> RectF32 {
    RectF32::new(
        r.l + (crop.left * r.w),
        r.t + (crop.top * r.h),
        r.w * (crop.right - crop.left),
        r.h * (crop.bottom - crop.top),
    )
}

/// Manages the block type definitions, and their underlying textures,
/// for the game.
pub(crate) struct BlockRenderer {
    block_defs: Arc<ClientBlockTypeManager>,
    texture_atlas: Arc<Texture2DHolder>,
    selection_box_tex_coord: RectF32,
    fallback_tex_coord: RectF32,
    simple_block_tex_coords: SimpleTexCoordCache,
    axis_aligned_box_blocks: AxisAlignedBoxBlocksCache,
    fake_entity_tex_coords: RectF32,
    allocator: Arc<GenericMemoryAllocator<Arc<FreeListAllocator>>>,
}
impl BlockRenderer {
    pub(crate) async fn new(
        block_defs: Arc<ClientBlockTypeManager>,
        mut cache_manager: parking_lot::MutexGuard<'_, CacheManager>,
        ctx: &VulkanContext,
    ) -> Result<BlockRenderer> {
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
                Some(RenderInfo::PlantLike(plant_like)) => {
                    if let Some(tex) = &plant_like.tex {
                        all_texture_names.insert(tex.texture_name.clone());
                    }
                }
                Some(RenderInfo::AxisAlignedBoxes(aa_boxes)) => {
                    for aa_box in &aa_boxes.boxes {
                        if let Some(tex) = &aa_box.tex_back {
                            all_texture_names.insert(tex.texture_name.clone());
                        }
                        if let Some(tex) = &aa_box.tex_front {
                            all_texture_names.insert(tex.texture_name.clone());
                        }
                        if let Some(tex) = &aa_box.tex_left {
                            all_texture_names.insert(tex.texture_name.clone());
                        }
                        if let Some(tex) = &aa_box.tex_right {
                            all_texture_names.insert(tex.texture_name.clone());
                        }
                        if let Some(tex) = &aa_box.tex_top {
                            all_texture_names.insert(tex.texture_name.clone());
                        }
                        if let Some(tex) = &aa_box.tex_bottom {
                            all_texture_names.insert(tex.texture_name.clone());
                        }
                    }
                }
                Some(RenderInfo::Empty(_)) => {}
                None => {
                    log::warn!("Got a block without renderinfo: {}", def.short_name)
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

        let texture_atlas = texture_packer::exporter::ImageExporter::export(&texture_packer)
            .map_err(|x| Error::msg(format!("Texture atlas export failed: {:?}", x)))?;
        let texture_coords: FxHashMap<String, Rect> = texture_packer
            .get_frames()
            .iter()
            .map(|(k, v)| (k.clone(), v.frame))
            .collect();

        let simple_block_tex_coords = SimpleTexCoordCache {
            blocks: block_defs
                .block_defs
                .iter()
                .map(|x| {
                    x.as_ref()
                        .and_then(|x| build_simple_cache_entry(x, &texture_coords))
                })
                .collect(),
        };

        let axis_aligned_box_blocks = AxisAlignedBoxBlocksCache {
            blocks: block_defs
                .block_defs
                .iter()
                .map(|x| {
                    x.as_ref()
                        .and_then(|x| build_axis_aligned_box_cache_entry(x, &texture_coords))
                })
                .collect(),
        };

        let texture_atlas = Arc::new(Texture2DHolder::create(ctx, &texture_atlas)?);
        Ok(BlockRenderer {
            block_defs,
            texture_atlas,
            selection_box_tex_coord: (*texture_coords.get(SELECTION_RECTANGLE).unwrap()).into(),
            fallback_tex_coord: (*texture_coords.get(FALLBACK_UNKNOWN_TEXTURE).unwrap()).into(),
            fake_entity_tex_coords: (*texture_coords.get(TESTONLY_ENTITY).unwrap()).into(),
            simple_block_tex_coords,
            axis_aligned_box_blocks,
            allocator: ctx.clone_allocator(),
        })
    }

    pub(crate) fn block_types(&self) -> &ClientBlockTypeManager {
        &self.block_defs
    }

    pub(crate) fn atlas(&self) -> &Texture2DHolder {
        &self.texture_atlas
    }

    pub(crate) fn allocator(&self) -> &GenericMemoryAllocator<Arc<FreeListAllocator>> {
        &self.allocator
    }

    pub(crate) fn mesh_chunk(&self, chunk_data: &LockedChunkDataView) -> Result<VkChunkVertexData> {
        let _span = span!("meshing");

        let mut bottom_all_solid = true;
        {
            let _span = span!("shell precheck");
            let mut all_solid = true;
            // if all edges have solid blocks, all interior geometry is hidden
            for x in -1..17 {
                for z in -1..17 {
                    for y in -1..17 {
                        if (x == -1 || x == 16) || (y == -1 || y == 16) || (z == -1 || z == 16) {
                            let id = chunk_data.block_ids()[(x, y, z).as_extended_index()];
                            if !self.block_types().is_solid_opaque(id) {
                                all_solid = false;
                            }
                        }
                    }
                    if x != -1 && x != 16  && z != -1 && z != 16 {
                        let id = chunk_data.block_ids()[(x, 0, z).as_extended_index()];
                        if !self.block_types().is_solid_opaque(id) {
                            bottom_all_solid = false;
                        }
                    }

                }
            }
            if all_solid {
                //println!("discarding all_solid");
                return Ok(VkChunkVertexData::empty(true));
            }
        }

        Ok(VkChunkVertexData {
            solid_opaque: self.mesh_chunk_subpass(
                chunk_data,
                |id| self.block_types().is_solid_opaque(id),
                |_block, neighbor| self.block_defs.is_solid_opaque(neighbor),
            )?,
            transparent: self.mesh_chunk_subpass(
                chunk_data,
                |block| self.block_defs.is_transparent_render(block),
                |block, neighbor| {
                    block.equals_ignore_variant(neighbor)
                        || self.block_defs.is_solid_opaque(neighbor)
                },
            )?,
            translucent: self.mesh_chunk_subpass(
                chunk_data,
                |block| self.block_defs.is_translucent_render(block),
                |block, neighbor| {
                    block.equals_ignore_variant(neighbor)
                        || self.block_defs.is_solid_opaque(neighbor)
                },
            )?,
            bottom_all_solid,
        })
    }

    pub(crate) fn mesh_chunk_subpass<F, G>(
        &self,
        chunk_data: &LockedChunkDataView,
        // closure taking a block and returning whether this subpass should render it
        include_block_when: F,
        // closure taking a block and its neighbor, and returning whether we should render the face of our block that faces the given neighbor
        // This only applies to cube blocks.
        suppress_face_when: G,
    ) -> Result<Option<VkChunkPass>>
    where
        F: Fn(BlockId) -> bool,
        G: Fn(BlockId, BlockId) -> bool,
    {
        let _span = span!("mesh subpass");
        let mut vtx = Vec::new();
        let mut idx = Vec::new();
        for x in 0..16 {
            for z in 0..16 {
                for y in 0..16 {
                    let offset = ChunkOffset { x, y, z };
                    let id = self.get_block_id(chunk_data.block_ids(), offset);

                    if include_block_when(id) {
                        self.render_single_block(
                            self.block_types()
                                .get_blockdef(id)
                                .unwrap_or_else(|| self.block_defs.get_fallback_blockdef()),
                            id,
                            offset,
                            chunk_data,
                            &mut vtx,
                            &mut idx,
                            &suppress_face_when,
                        );
                    }
                }
            }
        }
        VkChunkPass::from_buffers(vtx, idx, self.allocator())
    }

    pub(crate) fn render_single_block<G>(
        &self,
        block: &BlockTypeDef,
        id: BlockId,
        offset: ChunkOffset,
        chunk_data: &impl ChunkDataView,
        vtx: &mut Vec<CubeGeometryVertex>,
        idx: &mut Vec<u32>,
        suppress_face_when: &G,
    ) where
        G: Fn(BlockId, BlockId) -> bool,
    {
        match &block.render_info {
            Some(RenderInfo::Cube(cube_render_info)) => {
                self.emit_full_cube(
                    block,
                    id,
                    offset,
                    chunk_data,
                    vtx,
                    idx,
                    cube_render_info,
                    suppress_face_when,
                );
            }
            Some(RenderInfo::PlantLike(plantlike_render_info)) => self.emit_plantlike(
                block,
                id,
                offset,
                chunk_data,
                vtx,
                idx,
                plantlike_render_info,
            ),
            Some(RenderInfo::AxisAlignedBoxes(render_info)) => {
                self.emit_axis_aligned_boxes(block, id, offset, chunk_data, vtx, idx, render_info)
            }
            _ => (),
        }
    }

    fn emit_full_cube<F>(
        &self,
        block: &BlockTypeDef,
        id: BlockId,
        offset: ChunkOffset,
        chunk_data: &impl ChunkDataView,
        vtx: &mut Vec<CubeGeometryVertex>,
        idx: &mut Vec<u32>,
        render_info: &CubeRenderInfo,
        suppress_face_when: F,
    ) where
        F: Fn(BlockId, BlockId) -> bool,
    {
        let e = get_cube_extents(render_info, id, chunk_data, offset);

        let pos = vec3(offset.x.into(), offset.y.into(), offset.z.into());

        let textures = match self.simple_block_tex_coords.get(id) {
            Some(x) => *x,
            None => [self.fallback_tex_coord; 6],
        };
        self.emit_single_cube_impl(
            e,
            offset,
            suppress_face_when,
            block,
            chunk_data,
            pos,
            textures,
            vtx,
            idx,
        );
    }

    // made crate-visible for the sake of entities
    pub(crate) fn emit_single_cube_impl<F>(
        &self,
        e: CubeExtents,
        offset: ChunkOffset,
        suppress_face_when: F,
        block: &BlockTypeDef,
        chunk_data: &impl ChunkDataView,
        pos: Vector3<f32>,
        textures: [RectF32; 6],
        vtx: &mut Vec<CubeGeometryVertex>,
        idx: &mut Vec<u32>,
    ) where
        F: Fn(BlockId, BlockId) -> bool,
    {
        for i in 0..6 {
            let (n_x, n_y, n_z) = e.normals[i];
            let neighbor_index = (
                offset.x as i8 + n_x,
                offset.y as i8 + n_y,
                offset.z as i8 + n_z,
            )
                .as_extended_index();
            if e.force_face(i)
                || !suppress_face_when(BlockId(block.id), chunk_data.block_ids()[neighbor_index])
            {
                emit_cube_face_vk(
                    pos,
                    textures[i],
                    self.texture_atlas.dimensions(),
                    CUBE_EXTENTS_FACE_ORDER[i],
                    vtx,
                    idx,
                    e,
                    0,
                    chunk_data.lightmap()[neighbor_index],
                    0.0,
                    CUBE_FACE_BRIGHTNESS_BIASES[i],
                );
            }
        }
    }

    // made crate-visible for the sake of entities
    pub(crate) fn emit_single_cube_simple(
        &self,
        e: CubeExtents,
        pos: Vector3<f32>,
        textures: [RectF32; 6],
        vtx: &mut Vec<CubeGeometryVertex>,
        idx: &mut Vec<u32>,
    ) {
        for i in 0..6 {
            emit_cube_face_vk(
                pos,
                textures[i],
                self.texture_atlas.dimensions(),
                CUBE_EXTENTS_FACE_ORDER[i],
                vtx,
                idx,
                e,
                // TODO proper brightness for entities
                0xf0,
                0x00,
                0.0,
                CUBE_FACE_BRIGHTNESS_BIASES[i],
            );
        }
    }

    fn emit_plantlike(
        &self,
        _block: &BlockTypeDef,
        id: BlockId,
        offset: ChunkOffset,
        chunk_data: &impl ChunkDataView,
        vtx: &mut Vec<CubeGeometryVertex>,
        idx: &mut Vec<u32>,
        plantlike_render_info: &blocks_proto::PlantLikeRenderInfo,
    ) {
        let e = FULL_CUBE_EXTENTS;
        let pos = vec3(offset.x.into(), offset.y.into(), offset.z.into());
        let tex = match self.simple_block_tex_coords.get(id) {
            Some(x) => x[0],
            None => self.fallback_tex_coord,
        };
        vtx.reserve(8);
        idx.reserve(24);
        for face in PLANTLIKE_FACE_ORDER {
            emit_cube_face_vk(
                pos,
                tex,
                self.texture_atlas.dimensions(),
                face,
                vtx,
                idx,
                e,
                chunk_data.lightmap()[offset.as_extended_index()],
                0x00,
                plantlike_render_info.wave_effect_scale,
                1.0,
            );
        }
    }

    fn emit_axis_aligned_boxes(
        &self,
        _block: &BlockTypeDef,
        id: BlockId,
        offset: ChunkOffset,
        chunk_data: &impl ChunkDataView,
        vtx: &mut Vec<CubeGeometryVertex>,
        idx: &mut Vec<u32>,
        _render_info: &AxisAlignedBoxes,
    ) {
        let aabb_data = self.axis_aligned_box_blocks.get(id);
        if aabb_data.is_none() {
            // todo handle this case properly
            return;
        }

        let pos = vec3(offset.x.into(), offset.y.into(), offset.z.into());

        let aabb_data = aabb_data.unwrap();
        for aabb in aabb_data {
            if aabb.mask != 0 && (aabb.mask & id.variant() == 0) {
                continue;
            }
            let mut e = aabb.extents;
            match aabb.rotation {
                AabbRotation::None => (),
                AabbRotation::Nesw => e = e.rotate_y(id.variant() % 4),
            }
            for i in 0..6 {
                let (n_x, n_y, n_z) = e.normals[i];
                let neighbor_index = (
                    offset.x as i8 + n_x,
                    offset.y as i8 + n_y,
                    offset.z as i8 + n_z,
                )
                    .as_extended_index();

                emit_cube_face_vk(
                    pos,
                    aabb.textures[i],
                    self.texture_atlas.dimensions(),
                    CUBE_EXTENTS_FACE_ORDER[i],
                    vtx,
                    idx,
                    e,
                    chunk_data.lightmap()[neighbor_index],
                    chunk_data.lightmap()[offset.as_extended_index()],
                    0.0,
                    CUBE_FACE_BRIGHTNESS_BIASES[i],
                );
            }
        }
    }

    fn get_block(
        &self,
        ids: &[BlockId; 18 * 18 * 18],
        coord: ChunkOffset,
    ) -> (BlockId, &BlockTypeDef) {
        let block_id = ids[coord.as_extended_index()];

        let def = self
            .block_defs
            .get_blockdef(block_id)
            .unwrap_or_else(|| self.block_defs.get_fallback_blockdef());
        (block_id, def)
    }

    fn get_block_id(&self, ids: &[BlockId; 18 * 18 * 18], coord: ChunkOffset) -> BlockId {
        ids[coord.as_extended_index()]
    }

    pub(crate) fn make_pointee_cube(
        &self,
        player_position: cgmath::Vector3<f64>,
        pointee: perovskite_core::coordinates::BlockCoordinate,
    ) -> Result<CubeGeometryDrawCall> {
        let mut vtx = vec![];
        let mut idx = vec![];
        let frame = self.selection_box_tex_coord;
        const POINTEE_SELECTION_EXTENTS: CubeExtents =
            CubeExtents::new((-0.51, 0.51), (-0.51, 0.51), (-0.51, 0.51));
        let e = POINTEE_SELECTION_EXTENTS;
        let vk_pos = Vector3::zero();

        for &face in &CUBE_EXTENTS_FACE_ORDER {
            emit_cube_face_vk(
                vk_pos,
                frame,
                self.texture_atlas.dimensions(),
                face,
                &mut vtx,
                &mut idx,
                e,
                0xff,
                0x00,
                0.0,
                1.0,
            );
        }

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
                bottom_all_solid: false,
            },
        })
    }

    // TODO stop relying on the block renderer to render entities, or clean up
    // responsibility
    pub(crate) fn fake_entity_tex_coords(&self) -> RectF32 {
        self.fake_entity_tex_coords
    }
}

fn build_axis_aligned_box_cache_entry(
    x: &BlockTypeDef,
    texture_coords: &FxHashMap<String, Rect>,
) -> Option<Box<[CachedAxisAlignedBox]>> {
    if let Some(RenderInfo::AxisAlignedBoxes(aa_boxes)) = &x.render_info {
        let mut result = Vec::new();
        for (i, aa_box) in aa_boxes.boxes.iter().enumerate() {
            let extents = CubeExtents::new(
                (aa_box.x_min, aa_box.x_max),
                (-aa_box.y_max, -aa_box.y_min),
                (aa_box.z_min, aa_box.z_max),
            );
            let textures = [
                get_texture(texture_coords, aa_box.tex_right.as_ref()),
                get_texture(texture_coords, aa_box.tex_left.as_ref()),
                get_texture(texture_coords, aa_box.tex_top.as_ref()),
                get_texture(texture_coords, aa_box.tex_bottom.as_ref()),
                get_texture(texture_coords, aa_box.tex_back.as_ref()),
                get_texture(texture_coords, aa_box.tex_front.as_ref()),
            ];
            if aa_box.variant_mask & 0xfff != aa_box.variant_mask {
                log::warn!(
                    "Block {} box {} had bad variant mask: {:x}",
                    x.short_name,
                    i,
                    aa_box.variant_mask
                );
            }
            result.push(CachedAxisAlignedBox {
                extents,
                textures,
                rotation: match aa_box.rotation() {
                    blocks_proto::AxisAlignedBoxRotation::None => AabbRotation::None,
                    blocks_proto::AxisAlignedBoxRotation::Nesw => AabbRotation::Nesw,
                },
                mask: (aa_box.variant_mask & 0xfff) as u16,
            });
        }
        Some(result.into_boxed_slice())
    } else {
        None
    }
}

fn build_simple_cache_entry(
    block_def: &BlockTypeDef,
    texture_coords: &FxHashMap<String, Rect>,
) -> Option<[RectF32; 6]> {
    match &block_def.render_info {
        Some(RenderInfo::Cube(render_info)) => Some([
            get_texture(texture_coords, render_info.tex_right.as_ref()),
            get_texture(texture_coords, render_info.tex_left.as_ref()),
            get_texture(texture_coords, render_info.tex_top.as_ref()),
            get_texture(texture_coords, render_info.tex_bottom.as_ref()),
            get_texture(texture_coords, render_info.tex_back.as_ref()),
            get_texture(texture_coords, render_info.tex_front.as_ref()),
        ]),
        Some(RenderInfo::PlantLike(render_info)) => {
            let coords = get_texture(texture_coords, render_info.tex.as_ref());
            Some([coords; 6])
        }
        _ => None,
    }
}

const FULL_CUBE_EXTENTS: CubeExtents = CubeExtents::new((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5));
#[inline]
fn get_cube_extents(
    render_info: &CubeRenderInfo,
    id: BlockId,
    chunk_data: &impl ChunkDataView,
    offset: ChunkOffset,
) -> CubeExtents {
    match render_info.variant_effect() {
        blocks_proto::CubeVariantEffect::None => FULL_CUBE_EXTENTS,
        blocks_proto::CubeVariantEffect::RotateNesw => FULL_CUBE_EXTENTS.rotate_y(id.variant()),
        blocks_proto::CubeVariantEffect::Liquid => {
            build_liquid_cube_extents(chunk_data, offset, id)
        }
    }
}

fn build_liquid_cube_extents(
    chunk_data: &impl ChunkDataView,
    offset: ChunkOffset,
    id: BlockId,
) -> CubeExtents {
    let variant = id.variant();
    let neighbor_variant = |offset: ChunkOffset, dx: i8, dz: i8| -> u16 {
        let (x, z) = (offset.x as i8 + dx, offset.z as i8 + dz);
        let neighbor = chunk_data.block_ids()[(x, offset.y as i8, z).as_extended_index()];
        if neighbor.equals_ignore_variant(id) {
            neighbor.variant()
        } else {
            0
        }
    };

    fn variant_to_height(variant: u16) -> f32 {
        let height = ((variant as f32) / 7.0).clamp(0.025, 1.0);
        0.5 - height
    }

    let y_xn_zn = variant_to_height(
        variant
            .max(neighbor_variant(offset, -1, 0))
            .max(neighbor_variant(offset, 0, -1))
            .max(neighbor_variant(offset, -1, -1)),
    );
    let y_xn_zp = variant_to_height(
        variant
            .max(neighbor_variant(offset, -1, 0))
            .max(neighbor_variant(offset, 0, 1))
            .max(neighbor_variant(offset, -1, 1)),
    );
    let y_xp_zn = variant_to_height(
        variant
            .max(neighbor_variant(offset, 1, 0))
            .max(neighbor_variant(offset, 0, -1))
            .max(neighbor_variant(offset, 1, -1)),
    );
    let y_xp_zp = variant_to_height(
        variant
            .max(neighbor_variant(offset, 1, 0))
            .max(neighbor_variant(offset, 0, 1))
            .max(neighbor_variant(offset, 1, 1)),
    );

    CubeExtents {
        vertices: [
            vec3(-0.5, y_xn_zn, -0.5),
            vec3(-0.5, y_xn_zp, 0.5),
            vec3(-0.5, 0.5, -0.5),
            vec3(-0.5, 0.5, 0.5),
            vec3(0.5, y_xp_zn, -0.5),
            vec3(0.5, y_xp_zp, 0.5),
            vec3(0.5, 0.5, -0.5),
            vec3(0.5, 0.5, 0.5),
        ],
        normals: DEFAULT_FACE_NORMALS,
        // top face should be forced if it's not flush with the bottom of the next block
        force: [false, false, variant < 7, false, false, false],
    }
}

struct SimpleTexCoordCache {
    blocks: Vec<Option<[RectF32; 6]>>,
}
impl SimpleTexCoordCache {
    fn get(&self, block_id: BlockId) -> Option<&[RectF32; 6]> {
        self.blocks.get(block_id.index()).and_then(|x| x.as_ref())
    }
}

enum AabbRotation {
    None,
    Nesw,
}

struct CachedAxisAlignedBox {
    extents: CubeExtents,
    textures: [RectF32; 6],
    rotation: AabbRotation,
    mask: u16,
}

struct AxisAlignedBoxBlocksCache {
    blocks: Vec<Option<Box<[CachedAxisAlignedBox]>>>,
}
impl AxisAlignedBoxBlocksCache {
    fn get(&self, block_id: BlockId) -> Option<&[CachedAxisAlignedBox]> {
        self.blocks.get(block_id.index()).and_then(|x| x.as_deref())
    }
}

pub(crate) fn fallback_texture() -> Option<TextureReference> {
    Some(TextureReference {
        texture_name: FALLBACK_UNKNOWN_TEXTURE.to_string(),
        crop: None,
    })
}

const GLOBAL_BRIGHTNESS_TABLE_RAW: [f32; 16] = [
    0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75,
    0.8125, 0.875, 0.9375,
];
// TODO enable global brightness
const BRIGHTNESS_TABLE_RAW: [f32; 16] = [
    0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125,
    0.875, 0.9375, 1.00,
];
const CUBE_FACE_BRIGHTNESS_BIASES: [f32; 6] = [0.975, 0.975, 1.05, 0.95, 0.975, 0.975];

lazy_static::lazy_static! {
    static ref GLOBAL_BRIGHTNESS_TABLE: [f32; 16] = {
        GLOBAL_BRIGHTNESS_TABLE_RAW.iter().map(|x| x.powf(1.5)).collect::<Vec<f32>>().try_into().unwrap()
    };
    static ref BRIGHTNESS_TABLE: [f32; 16] = {
        BRIGHTNESS_TABLE_RAW.iter().map(|x| x.powf(1.25)).collect::<Vec<f32>>().try_into().unwrap()
    };
}

#[inline]
fn make_cgv(
    coord: Vector3<f32>,
    normal: Vector3<f32>,
    tex_uv: cgmath::Vector2<f32>,
    brightness: f32,
    global_brightness: f32,
    wave_horizontal: f32,
) -> CubeGeometryVertex {
    CubeGeometryVertex {
        position: [coord.x, coord.y, coord.z],
        normal: [normal.x, normal.y, normal.z],
        uv_texcoord: tex_uv.into(),
        brightness,
        global_brightness_contribution: global_brightness,
        wave_horizontal,
    }
}

/// Emits a single face of a cube into the given buffers.
/// Arguments:
///    coord: The center of the overall cube, in world space, with y-axis up (opposite Vulkan)
///     This is the center of the cube body, not the current face.
///    frame: The texture for the face.
#[inline]
pub(crate) fn emit_cube_face_vk(
    coord: Vector3<f32>,
    frame: RectF32,
    tex_dimension: (u32, u32),
    face: CubeFace,
    vert_buf: &mut Vec<CubeGeometryVertex>,
    idx_buf: &mut Vec<u32>,
    e: CubeExtents,
    encoded_brightness: u8,
    encoded_brightness_2: u8,
    horizontal_wave: f32,
    brightness_bias: f32,
) {
    // Flip the coordinate system to Vulkan
    let coord = vec3(coord.x, -coord.y, coord.z);
    let width = (tex_dimension.0) as f32;
    let height = (tex_dimension.1) as f32;
    let l = frame.left() / width;
    let r = frame.right() / width;
    let t = frame.top() / height;
    let b = frame.bottom() / height;
    // todo handle rotating textures (both in the texture ref and here)
    // Possibly at the caller?
    let tl = Vector2::new(l, t);
    let bl = Vector2::new(l, b);
    let tr = Vector2::new(r, t);
    let br = Vector2::new(r, b);

    let (global_brightness_1, brightness_1) = get_brightnesses(encoded_brightness, brightness_bias);
    let (global_brightness_2, brightness_2) =
        get_brightnesses(encoded_brightness_2, brightness_bias);

    let brightness = brightness_1.max(brightness_2);
    let global_brightness = global_brightness_1.max(global_brightness_2);

    let normal = e.normals[face.index()];
    let normal = vec3(normal.0 as f32, -normal.1 as f32, normal.2 as f32).normalize();
    let mut vertices = match face {
        CubeFace::ZMinus => vec![
            make_cgv(
                coord + e.vertices[0],
                normal,
                tl,
                brightness,
                global_brightness.max(global_brightness_2),
                0.0,
            ),
            make_cgv(
                coord + e.vertices[2],
                normal,
                bl,
                brightness,
                global_brightness.max(global_brightness_2),
                0.0,
            ),
            make_cgv(
                coord + e.vertices[6],
                normal,
                br,
                brightness,
                global_brightness,
                0.0,
            ),
            make_cgv(
                coord + e.vertices[4],
                normal,
                tr,
                brightness,
                global_brightness,
                0.0,
            ),
        ],
        CubeFace::ZPlus => vec![
            make_cgv(
                coord + e.vertices[5],
                normal,
                tl,
                brightness,
                global_brightness,
                0.0,
            ),
            make_cgv(
                coord + e.vertices[7],
                normal,
                bl,
                brightness,
                global_brightness,
                0.0,
            ),
            make_cgv(
                coord + e.vertices[3],
                normal,
                br,
                brightness,
                global_brightness,
                0.0,
            ),
            make_cgv(
                coord + e.vertices[1],
                normal,
                tr,
                brightness,
                global_brightness,
                0.0,
            ),
        ],
        CubeFace::XMinus => vec![
            make_cgv(
                coord + e.vertices[1],
                normal,
                tl,
                brightness,
                global_brightness,
                0.0,
            ),
            make_cgv(
                coord + e.vertices[3],
                normal,
                bl,
                brightness,
                global_brightness,
                0.0,
            ),
            make_cgv(
                coord + e.vertices[2],
                normal,
                br,
                brightness,
                global_brightness,
                0.0,
            ),
            make_cgv(
                coord + e.vertices[0],
                normal,
                tr,
                brightness,
                global_brightness,
                0.0,
            ),
        ],
        CubeFace::XPlus => vec![
            make_cgv(
                coord + e.vertices[4],
                normal,
                tl,
                brightness,
                global_brightness,
                0.0,
            ),
            make_cgv(
                coord + e.vertices[6],
                normal,
                bl,
                brightness,
                global_brightness,
                0.0,
            ),
            make_cgv(
                coord + e.vertices[7],
                normal,
                br,
                brightness,
                global_brightness,
                0.0,
            ),
            make_cgv(
                coord + e.vertices[5],
                normal,
                tr,
                brightness,
                global_brightness,
                0.0,
            ),
        ],
        // For Y+ and Y-, the top of the texturehe front of the cube (prior to any rotations)
        // Y- is the bottom face (opposite Vulkat takes Y+ vulkan coordinates
        CubeFace::YMinus => vec![
            make_cgv(
                coord + e.vertices[2],
                normal,
                tl,
                brightness,
                global_brightness,
                0.0,
            ),
            make_cgv(
                coord + e.vertices[3],
                normal,
                bl,
                brightness,
                global_brightness,
                0.0,
            ),
            make_cgv(
                coord + e.vertices[7],
                normal,
                br,
                brightness,
                global_brightness,
                0.0,
            ),
            make_cgv(
                coord + e.vertices[6],
                normal,
                tr,
                brightness,
                global_brightness,
                0.0,
            ),
        ],
        CubeFace::YPlus => vec![
            make_cgv(
                coord + e.vertices[4],
                normal,
                tl,
                brightness,
                global_brightness,
                0.0,
            ),
            make_cgv(
                coord + e.vertices[5],
                normal,
                bl,
                brightness,
                global_brightness,
                0.0,
            ),
            make_cgv(
                coord + e.vertices[1],
                normal,
                br,
                brightness,
                global_brightness,
                0.0,
            ),
            make_cgv(
                coord + e.vertices[0],
                normal,
                tr,
                brightness,
                global_brightness,
                0.0,
            ),
        ],
        CubeFace::PlantXMinusZMinus => vec![
            make_cgv(
                coord + e.vertices[1],
                normal,
                tl,
                brightness,
                global_brightness,
                horizontal_wave,
            ),
            make_cgv(
                coord + e.vertices[3],
                normal,
                bl,
                brightness,
                global_brightness,
                0.0,
            ),
            make_cgv(
                coord + e.vertices[6],
                normal,
                br,
                brightness,
                global_brightness,
                0.0,
            ),
            make_cgv(
                coord + e.vertices[4],
                normal,
                tr,
                brightness,
                global_brightness,
                horizontal_wave,
            ),
        ],
        CubeFace::PlantXPlusZPlus => vec![
            make_cgv(
                coord + e.vertices[4],
                normal,
                tl,
                brightness,
                global_brightness,
                horizontal_wave,
            ),
            make_cgv(
                coord + e.vertices[6],
                normal,
                bl,
                brightness,
                global_brightness,
                0.0,
            ),
            make_cgv(
                coord + e.vertices[3],
                normal,
                br,
                brightness,
                global_brightness,
                0.0,
            ),
            make_cgv(
                coord + e.vertices[1],
                normal,
                tr,
                brightness,
                global_brightness,
                horizontal_wave,
            ),
        ],
        CubeFace::PlantXMinusZPlus => vec![
            make_cgv(
                coord + e.vertices[5],
                normal,
                tl,
                brightness,
                global_brightness,
                horizontal_wave,
            ),
            make_cgv(
                coord + e.vertices[7],
                normal,
                bl,
                brightness,
                global_brightness,
                0.0,
            ),
            make_cgv(
                coord + e.vertices[2],
                normal,
                br,
                brightness,
                global_brightness,
                0.0,
            ),
            make_cgv(
                coord + e.vertices[0],
                normal,
                tr,
                brightness,
                global_brightness,
                horizontal_wave,
            ),
        ],
        CubeFace::PlantXPlusZMinus => vec![
            make_cgv(
                coord + e.vertices[0],
                normal,
                tl,
                brightness,
                global_brightness,
                horizontal_wave,
            ),
            make_cgv(
                coord + e.vertices[2],
                normal,
                bl,
                brightness,
                global_brightness,
                0.0,
            ),
            make_cgv(
                coord + e.vertices[7],
                normal,
                br,
                brightness,
                global_brightness,
                0.0,
            ),
            make_cgv(
                coord + e.vertices[5],
                normal,
                tr,
                brightness,
                global_brightness,
                horizontal_wave,
            ),
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

#[inline]
fn get_brightnesses(encoded_brightness: u8, brightness_bias: f32) -> (f32, f32) {
    let brightness_upper = (encoded_brightness >> 4) as usize;
    let brightness_lower = (encoded_brightness & 0x0F) as usize;

    let global_brightness = GLOBAL_BRIGHTNESS_TABLE[brightness_upper];
    let brightness = BRIGHTNESS_TABLE[brightness_lower] * brightness_bias;
    (global_brightness, brightness)
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
pub(crate) trait AsyncMediaLoader {
    async fn load_media(&mut self, tex_name: &str) -> Result<Vec<u8>>;
}
