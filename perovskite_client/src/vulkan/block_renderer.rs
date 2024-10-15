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
use std::ops::Deref;

use std::sync::Arc;

use cgmath::num_traits::Num;
use cgmath::{vec3, ElementWise, InnerSpace, Matrix4, Vector2, Vector3, Zero};

use perovskite_core::constants::textures::FALLBACK_UNKNOWN_TEXTURE;

use perovskite_core::protocol::blocks::block_type_def::RenderInfo;
use perovskite_core::protocol::blocks::{
    self as blocks_proto, AxisAlignedBoxes, BlockTypeDef, CubeRenderInfo,
};
use perovskite_core::protocol::render::{TextureCrop, TextureReference};
use perovskite_core::{block_id::BlockId, coordinates::ChunkOffset};

use anyhow::{Error, Result};

use rustc_hash::FxHashMap;
use texture_packer::importer::ImageImporter;
use texture_packer::Rect;

use tracy_client::span;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};

use crate::cache::CacheManager;
use crate::clog;
use crate::game_state::block_types::ClientBlockTypeManager;
use crate::game_state::chunk::{
    ChunkDataView, ChunkOffsetExt, LockedChunkDataView, MeshVectorReclaim, SOLID_RECLAIMER,
    TRANSLUCENT_RECLAIMER, TRANSPARENT_RECLAIMER,
};
use crate::vulkan::shaders::cube_geometry::{CubeGeometryDrawCall, CubeGeometryVertex};
use crate::vulkan::{Texture2DHolder, VulkanContext};

use super::{RectF32, VkAllocator};

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
struct DynamicRect {
    base: RectF32,
    x_selector: u16,
    y_selector: u16,
    // The size of a cell
    x_cell_stride: f32,
    y_cell_stride: f32,
    // How much distance/offset corresponds to a value of (1) in selector AND variant
    x_selector_factor: f32,
    y_selector_factor: f32,
    flip_x_bit: u16,
    flip_y_bit: u16,
    extra_flip_x: bool,
    extra_flip_y: bool,
}
impl DynamicRect {
    #[inline]
    fn resolve(&self, variant: u16) -> RectF32 {
        let x_min = self.base.l + (variant & self.x_selector) as f32 * self.x_selector_factor;
        let y_min = self.base.t + (variant & self.y_selector) as f32 * self.y_selector_factor;

        let (real_xmin, real_xstride) = if ((variant & self.flip_x_bit) != 0) ^ self.extra_flip_x {
            (x_min + self.x_cell_stride, -self.x_cell_stride)
        } else {
            (x_min, self.x_cell_stride)
        };
        let (real_ymin, real_ystride) = if ((variant & self.flip_y_bit) != 0) ^ self.extra_flip_y {
            (y_min + self.y_cell_stride, -self.y_cell_stride)
        } else {
            (y_min, self.y_cell_stride)
        };

        RectF32::new(real_xmin, real_ymin, real_xstride, real_ystride)
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

    fn warp_top_inplace(&mut self, top_slope_x: f32, top_slope_z: f32) {
        for idx in [0, 1, 4, 5] {
            self.vertices[idx].y +=
                top_slope_x * self.vertices[idx].x + top_slope_z * self.vertices[idx].z;
        }
    }
    fn warp_bottom_inplace(&mut self, bottom_slope_x: f32, bottom_slope_z: f32) {
        for idx in [2, 3, 6, 7] {
            self.vertices[idx].y +=
                bottom_slope_x * self.vertices[idx].x + bottom_slope_z * self.vertices[idx].z;
        }
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

#[derive(Clone, PartialEq)]
pub(crate) struct VkCgvBufferCpu {
    pub(crate) vtx: Vec<CubeGeometryVertex>,
    pub(crate) idx: Vec<u32>,
}
impl VkCgvBufferCpu {
    fn to_gpu(&self, allocator: Arc<VkAllocator>) -> Result<Option<VkCgvBufferGpu>> {
        VkCgvBufferGpu::from_buffers(&self.vtx, &self.idx, allocator)
    }
}

#[derive(Clone)]
pub(crate) struct VkCgvBufferGpu {
    pub(crate) vtx: Subbuffer<[CubeGeometryVertex]>,
    pub(crate) idx: Subbuffer<[u32]>,
}
impl VkCgvBufferGpu {
    pub(crate) fn from_buffers(
        vtx: &[CubeGeometryVertex],
        idx: &[u32],
        allocator: Arc<VkAllocator>,
    ) -> Result<Option<VkCgvBufferGpu>> {
        if vtx.is_empty() {
            Ok(None)
        } else {
            Ok(Some(VkCgvBufferGpu {
                vtx: Buffer::from_iter(
                    allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::VERTEX_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        // TODO: Consider whether we should manage our own staging buffer,
                        // and copy into VRAM, or just use one buffer that's host sequential write
                        // plus device local
                        memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                            | MemoryTypeFilter::PREFER_DEVICE,
                        ..Default::default()
                    },
                    vtx.iter().copied(),
                )?,
                idx: Buffer::from_iter(
                    allocator,
                    BufferCreateInfo {
                        usage: BufferUsage::INDEX_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                            | MemoryTypeFilter::PREFER_DEVICE,
                        ..Default::default()
                    },
                    idx.iter().copied(),
                )?,
            }))
        }
    }
}

#[derive(Clone)]
pub(crate) struct VkChunkVertexDataGpu {
    pub(crate) solid_opaque: Option<VkCgvBufferGpu>,
    pub(crate) transparent: Option<VkCgvBufferGpu>,
    pub(crate) translucent: Option<VkCgvBufferGpu>,
}
impl VkChunkVertexDataGpu {
    pub(crate) fn clone_if_nonempty(&self) -> Option<Self> {
        if self.solid_opaque.is_some() || self.transparent.is_some() || self.translucent.is_some() {
            Some(self.clone())
        } else {
            None
        }
    }

    pub(crate) fn empty() -> VkChunkVertexDataGpu {
        VkChunkVertexDataGpu {
            solid_opaque: None,
            transparent: None,
            translucent: None,
        }
    }
}

#[derive(Clone, PartialEq)]
pub(crate) struct VkChunkVertexDataCpu {
    pub(crate) solid_opaque: Option<VkCgvBufferCpu>,
    pub(crate) transparent: Option<VkCgvBufferCpu>,
    pub(crate) translucent: Option<VkCgvBufferCpu>,
}
impl VkChunkVertexDataCpu {
    fn empty() -> VkChunkVertexDataCpu {
        VkChunkVertexDataCpu {
            solid_opaque: None,
            transparent: None,
            translucent: None,
        }
    }

    pub(crate) fn to_gpu(&self, allocator: Arc<VkAllocator>) -> Result<VkChunkVertexDataGpu> {
        Ok(VkChunkVertexDataGpu {
            solid_opaque: match &self.solid_opaque {
                Some(x) => x.to_gpu(allocator.clone())?,
                None => None,
            },
            transparent: match &self.transparent {
                Some(x) => x.to_gpu(allocator.clone())?,
                None => None,
            },
            translucent: match &self.translucent {
                Some(x) => x.to_gpu(allocator)?,
                None => None,
            },
        })
    }
}

fn get_selector_shift(bits: u32) -> u32 {
    let leading_zeros = bits.leading_zeros();
    let trailing_zeros = bits.trailing_zeros();
    let ones = bits.count_ones();
    if (leading_zeros + trailing_zeros + ones) != 32 {
        log::warn!("Selector with non-contiguous bits: {:b}", bits);
    }

    dbg!(1 << trailing_zeros)
}

fn get_texture(
    texture_coords: &FxHashMap<String, Rect>,
    tex: Option<&TextureReference>,
    width: f32,
    height: f32,
) -> MaybeDynamicRect {
    let rect = tex
        .and_then(|tex| texture_coords.get(&tex.texture_name).copied())
        .unwrap_or_else(|| *texture_coords.get(FALLBACK_UNKNOWN_TEXTURE).unwrap());
    let mut rect_f = RectF32::new(
        rect.x as f32 / width,
        rect.y as f32 / height,
        rect.w as f32 / width,
        rect.h as f32 / height,
    );
    if let Some(crop) = tex.and_then(|tex| tex.crop.as_ref()) {
        rect_f = crop_texture(crop, rect_f);
        if let Some(dynamic) = crop.dynamic.as_ref() {
            let x_selector_shift_factor = get_selector_shift(dynamic.x_selector_bits);
            let y_selector_shift_factor = get_selector_shift(dynamic.y_selector_bits);

            return MaybeDynamicRect::Dynamic(DynamicRect {
                base: rect_f,
                x_selector: dynamic.x_selector_bits as u16,
                y_selector: dynamic.y_selector_bits as u16,
                // these are given in texture coordinates
                x_cell_stride: rect_f.w / (dynamic.x_cells as f32),
                y_cell_stride: rect_f.h / (dynamic.y_cells as f32),
                x_selector_factor: rect_f.w
                    / (dynamic.x_cells as f32 * x_selector_shift_factor as f32),
                y_selector_factor: rect_f.h
                    / (dynamic.y_cells as f32 * y_selector_shift_factor as f32),
                flip_x_bit: dynamic.flip_x_bit as u16,
                flip_y_bit: dynamic.flip_y_bit as u16,
                extra_flip_x: dynamic.extra_flip_x,
                extra_flip_y: dynamic.extra_flip_y,
            });
        }
    }
    MaybeDynamicRect::Static(RectF32 {
        l: rect_f.l,
        t: rect_f.t,
        w: rect_f.w,
        h: rect_f.h,
    })
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
    vk_ctx: Arc<VulkanContext>,
}
impl BlockRenderer {
    pub(crate) async fn new(
        block_defs: Arc<ClientBlockTypeManager>,
        mut cache_manager: parking_lot::MutexGuard<'_, CacheManager>,
        ctx: Arc<VulkanContext>,
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
        log::info!("all media loaded");

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
        log::info!("Static textures packed");

        let texture_atlas = texture_packer::exporter::ImageExporter::export(&texture_packer, None)
            .map_err(|x| Error::msg(format!("Texture atlas export failed: {:?}", x)))?;
        let texture_coords: FxHashMap<String, Rect> = texture_packer
            .get_frames()
            .iter()
            .map(|(k, v)| (k.clone(), v.frame))
            .collect();

        let simple_block_tex_coords = SimpleTexCoordCache {
            blocks: block_defs
                .block_defs()
                .iter()
                .map(|x| {
                    x.as_ref().and_then(|x| {
                        build_simple_cache_entry(
                            x,
                            &texture_coords,
                            texture_atlas.width() as f32,
                            texture_atlas.height() as f32,
                        )
                    })
                })
                .collect(),
        };
        log::info!("Simple block tex coords done");

        let axis_aligned_box_blocks = AxisAlignedBoxBlocksCache {
            blocks: block_defs
                .block_defs()
                .iter()
                .map(|x| {
                    x.as_ref().and_then(|x| {
                        build_axis_aligned_box_cache_entry(
                            x,
                            &texture_coords,
                            texture_atlas.width() as f32,
                            texture_atlas.height() as f32,
                        )
                    })
                })
                .collect(),
        };
        log::info!("aabb blocks done");

        let texture_atlas = Arc::new(Texture2DHolder::create(ctx.deref(), &texture_atlas)?);
        log::info!("atlas created (GPU)");

        let selection_rect: RectF32 = (*texture_coords.get(SELECTION_RECTANGLE).unwrap()).into();
        let fallback_rect: RectF32 =
            (*texture_coords.get(FALLBACK_UNKNOWN_TEXTURE).unwrap()).into();
        let fake_entity_rect: RectF32 = (*texture_coords.get(TESTONLY_ENTITY).unwrap()).into();
        let atlas_dims = texture_atlas.dimensions();
        log::info!("BlockRenderer ready");
        Ok(BlockRenderer {
            block_defs,
            texture_atlas,
            selection_box_tex_coord: selection_rect.div(atlas_dims),
            fallback_tex_coord: fallback_rect.div(atlas_dims),
            fake_entity_tex_coords: fake_entity_rect.div(atlas_dims),
            simple_block_tex_coords,
            axis_aligned_box_blocks,
            vk_ctx: ctx,
        })
    }

    pub(crate) fn block_types(&self) -> &ClientBlockTypeManager {
        &self.block_defs
    }

    pub(crate) fn atlas(&self) -> &Texture2DHolder {
        &self.texture_atlas
    }

    pub(crate) fn allocator(&self) -> &VkAllocator {
        &self.vk_ctx.allocator()
    }

    pub(crate) fn clone_vk_allocator(&self) -> Arc<VkAllocator> {
        self.vk_ctx.clone_allocator()
    }

    pub(crate) fn vk_ctx(&self) -> &VulkanContext {
        &self.vk_ctx
    }

    pub(crate) fn mesh_chunk(
        &self,
        chunk_data: &LockedChunkDataView,
    ) -> Result<VkChunkVertexDataCpu> {
        let _span = span!("meshing");

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
                }
            }
            if all_solid {
                return Ok(VkChunkVertexDataCpu::empty());
            }
        }

        Ok(VkChunkVertexDataCpu {
            solid_opaque: self.mesh_chunk_subpass(
                chunk_data,
                |id| self.block_types().is_solid_opaque(id),
                |_block, neighbor| self.block_defs.is_solid_opaque(neighbor),
                &SOLID_RECLAIMER,
            ),
            transparent: self.mesh_chunk_subpass(
                chunk_data,
                |block| self.block_defs.is_transparent_render(block),
                |block, neighbor| {
                    block.equals_ignore_variant(neighbor)
                        || self.block_defs.is_solid_opaque(neighbor)
                },
                &TRANSPARENT_RECLAIMER,
            ),
            translucent: self.mesh_chunk_subpass(
                chunk_data,
                |block| self.block_defs.is_translucent_render(block),
                |block, neighbor| {
                    block.equals_ignore_variant(neighbor)
                        || self.block_defs.is_solid_opaque(neighbor)
                },
                &TRANSLUCENT_RECLAIMER,
            ),
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
        reclaimer: &MeshVectorReclaim,
    ) -> Option<VkCgvBufferCpu>
    where
        F: Fn(BlockId) -> bool,
        G: Fn(BlockId, BlockId) -> bool,
    {
        let _span = span!("mesh subpass");

        let (mut idx, mut vtx) = match reclaimer.take() {
            Some((idx, vtx)) => (idx, vtx),
            None => (Vec::new(), Vec::new()),
        };
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
        if vtx.is_empty() {
            // Don't put these into the reclaimer. They're empty and there's no memory
            // allocation to reclaim.
            None
        } else {
            Some(VkCgvBufferCpu { vtx, idx })
        }
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

        let textures = self
            .simple_block_tex_coords
            .get(id)
            .unwrap_or([self.fallback_tex_coord; 6]);
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
        let tex = match self.simple_block_tex_coords.get_zero(id) {
            Some(x) => x,
            None => self.fallback_tex_coord,
        };
        vtx.reserve(8);
        idx.reserve(24);
        for face in PLANTLIKE_FACE_ORDER {
            emit_cube_face_vk(
                pos,
                tex,
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
                    aabb.textures[i].rect(id.variant()),
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

    fn get_block_id(&self, ids: &[BlockId; 18 * 18 * 18], coord: ChunkOffset) -> BlockId {
        ids[coord.as_extended_index()]
    }

    pub(crate) fn make_pointee_cube(
        &self,
        player_position: Vector3<f64>,
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
                vk_pos, frame, face, &mut vtx, &mut idx, e, 0xff, 0x00, 0.0, 1.0,
            );
        }

        let vtx = Buffer::from_iter(
            self.vk_ctx.clone_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                    | MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            vtx.into_iter(),
        )?;
        let idx = Buffer::from_iter(
            self.vk_ctx.clone_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                    | MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            idx.into_iter(),
        )?;

        clog!("pointee buffers allocated");
        let offset = (vec3(pointee.x as f64, pointee.y as f64, pointee.z as f64) - player_position)
            .mul_element_wise(Vector3::new(1., -1., 1.));
        Ok(CubeGeometryDrawCall {
            model_matrix: Matrix4::from_translation(offset.cast().unwrap()),
            models: VkChunkVertexDataGpu {
                solid_opaque: None,
                transparent: Some(VkCgvBufferGpu { vtx, idx }),
                translucent: None,
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
    w: f32,
    h: f32,
) -> Option<Box<[CachedAxisAlignedBox]>> {
    if let Some(RenderInfo::AxisAlignedBoxes(aa_boxes)) = &x.render_info {
        let mut result = Vec::new();
        for (i, aa_box) in aa_boxes.boxes.iter().enumerate() {
            let mut extents = CubeExtents::new(
                (aa_box.x_min, aa_box.x_max),
                (-aa_box.y_max, -aa_box.y_min),
                (aa_box.z_min, aa_box.z_max),
            );
            // Negate to go from API coordinates to Vulkan coordinates
            extents.warp_top_inplace(-aa_box.top_slope_x, -aa_box.top_slope_z);
            extents.warp_bottom_inplace(-aa_box.bottom_slope_x, -aa_box.bottom_slope_z);
            let textures = [
                get_texture(texture_coords, aa_box.tex_right.as_ref(), w, h),
                get_texture(texture_coords, aa_box.tex_left.as_ref(), w, h),
                get_texture(texture_coords, aa_box.tex_top.as_ref(), w, h),
                get_texture(texture_coords, aa_box.tex_bottom.as_ref(), w, h),
                get_texture(texture_coords, aa_box.tex_back.as_ref(), w, h),
                get_texture(texture_coords, aa_box.tex_front.as_ref(), w, h),
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
    w: f32,
    h: f32,
) -> Option<[MaybeDynamicRect; 6]> {
    match &block_def.render_info {
        Some(RenderInfo::Cube(render_info)) => Some([
            get_texture(texture_coords, render_info.tex_right.as_ref(), w, h),
            get_texture(texture_coords, render_info.tex_left.as_ref(), w, h),
            get_texture(texture_coords, render_info.tex_top.as_ref(), w, h),
            get_texture(texture_coords, render_info.tex_bottom.as_ref(), w, h),
            get_texture(texture_coords, render_info.tex_back.as_ref(), w, h),
            get_texture(texture_coords, render_info.tex_front.as_ref(), w, h),
        ]),
        Some(RenderInfo::PlantLike(render_info)) => {
            let coords = get_texture(texture_coords, render_info.tex.as_ref(), w, h);
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

#[derive(Clone, Copy, Debug)]
enum MaybeDynamicRect {
    Static(RectF32),
    Dynamic(DynamicRect),
}
impl MaybeDynamicRect {
    fn rect(&self, variant: u16) -> RectF32 {
        match self {
            Self::Static(rect) => *rect,
            Self::Dynamic(rect) => rect.resolve(variant),
        }
    }
}

struct SimpleTexCoordCache {
    blocks: Vec<Option<[MaybeDynamicRect; 6]>>,
}
impl SimpleTexCoordCache {
    fn get(&self, block_id: BlockId) -> Option<[RectF32; 6]> {
        self.blocks
            .get(block_id.index())
            .and_then(|x| x.as_ref())
            .and_then(|entry| {
                Some([
                    entry[0].rect(block_id.variant()),
                    entry[1].rect(block_id.variant()),
                    entry[2].rect(block_id.variant()),
                    entry[3].rect(block_id.variant()),
                    entry[4].rect(block_id.variant()),
                    entry[5].rect(block_id.variant()),
                ])
            })
    }

    fn get_zero(&self, block_id: BlockId) -> Option<RectF32> {
        self.blocks
            .get(block_id.index())
            .and_then(|x| x.as_ref())
            .and_then(|entry| Some(entry[0].rect(block_id.variant())))
    }
}

enum AabbRotation {
    None,
    Nesw,
}

struct CachedAxisAlignedBox {
    extents: CubeExtents,
    textures: [MaybeDynamicRect; 6],
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
    tex_uv: Vector2<f32>,
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
    let l = frame.left();
    let r = frame.right();
    let t = frame.top();
    let b = frame.bottom();
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
