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
use std::fmt::{Debug, Formatter};
use std::ops::Deref;

use cgmath::num_traits::Num;
use cgmath::{vec3, ElementWise, Matrix4, Vector2, Vector3, Zero};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use perovskite_core::protocol::blocks::block_type_def::RenderInfo;
use perovskite_core::protocol::blocks::{
    self as blocks_proto, AxisAlignedBoxes, BlockTypeDef, CubeRenderInfo, CubeVariantEffect,
};
use perovskite_core::protocol::render::{TextureCrop, TextureReference};
use perovskite_core::{block_id::BlockId, coordinates::ChunkOffset};

use anyhow::{Error, Result};
use image::{DynamicImage, RgbImage, RgbaImage};
use rustc_hash::FxHashMap;
use texture_packer::importer::ImageImporter;
use texture_packer::Rect;

use super::{RectF32, VkAllocator};
use crate::client_state::block_types::ClientBlockTypeManager;
use crate::client_state::chunk::{
    ChunkDataView, ChunkOffsetExt, LockedChunkDataView, MeshVectorReclaim,
    RAYTRACE_FALLBACK_RECLAIMER, SOLID_RECLAIMER, TRANSLUCENT_RECLAIMER, TRANSPARENT_RECLAIMER,
    TRANSPARENT_WITH_SPECULAR_RECLAIMER,
};
use crate::client_state::ClientState;
use crate::media::{load_or_generate_image, CacheManager};
use crate::vulkan::atlas::{NamedTextureKey, TextureAtlas, TextureKey};
use crate::vulkan::gpu_chunk_table::ht_consts::{FLAG_HASHTABLE_HEAVY, FLAG_HASHTABLE_PRESENT};
use crate::vulkan::shaders::cube_geometry::{CubeGeometryDrawCall, CubeGeometryVertex};
use crate::vulkan::shaders::raytracer::{SimpleCubeInfo, TexRef};
use crate::vulkan::shaders::{VkBufferCpu, VkDrawBufferGpu};
use crate::vulkan::{Texture2DHolder, VulkanContext};
use perovskite_core::game_actions::ToolTarget;
use perovskite_core::protocol::game_rpc::EntityTarget;
use tracy_client::span;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};

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

    #[inline(always)]
    fn repr(&self) -> u8 {
        *self as u8
    }

    #[inline(always)]
    fn rotate_y(&self, variant: u16) -> CubeFace {
        let idx = (variant % 4) as usize;

        const LUT: [[usize; 4]; 10] = [
            [0, 4, 1, 5],
            [1, 5, 0, 4],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
            [4, 1, 5, 0],
            [5, 0, 4, 1],
            [6, 8, 9, 7],
            [7, 6, 8, 9],
            [8, 9, 7, 6],
            [9, 7, 6, 8],
        ];
        CUBE_EXTENTS_FACE_ORDER[LUT[self.index()][idx]]
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
    pub adjacency: [(i8, i8, i8); 10],
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
            adjacency: DEFAULT_FACE_NORMALS,
            force: [false; 6],
        }
    }

    pub fn rotate_y(self, variant: u16) -> CubeExtents {
        let mut vertices = self.vertices;
        let mut adjacency = self.adjacency;
        for vtx in &mut vertices {
            *vtx = Vector3::from(rotate_y((vtx.x, vtx.y, vtx.z), variant));
        }
        for neighbor in &mut adjacency {
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
            adjacency,
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

#[derive(Clone)]
pub(crate) struct VkChunkVertexDataGpu {
    pub(crate) opaque: Option<VkDrawBufferGpu<CubeGeometryVertex>>,
    pub(crate) transparent: Option<VkDrawBufferGpu<CubeGeometryVertex>>,
    pub(crate) translucent: Option<VkDrawBufferGpu<CubeGeometryVertex>>,
    pub(crate) raytrace_fallback: Option<VkDrawBufferGpu<CubeGeometryVertex>>,
    pub(crate) transparent_with_specular: Option<VkDrawBufferGpu<CubeGeometryVertex>>,
}
impl VkChunkVertexDataGpu {
    pub(crate) fn clone_if_nonempty(&self) -> Option<Self> {
        if self.opaque.is_some()
            || self.transparent.is_some()
            || self.translucent.is_some()
            || self.raytrace_fallback.is_some()
        {
            Some(self.clone())
        } else {
            None
        }
    }

    pub(crate) fn empty() -> VkChunkVertexDataGpu {
        VkChunkVertexDataGpu {
            opaque: None,
            transparent: None,
            translucent: None,
            raytrace_fallback: None,
            transparent_with_specular: None,
        }
    }
}

static RAYTRACE_CHUNK_VERSION_COUNTER: AtomicUsize = AtomicUsize::new(1);

#[derive(Clone)]
pub(crate) struct VkChunkRaytraceData {
    pub(crate) flags: u32,
    // Only Some if we overrode anything
    pub(crate) blocks: Option<Box<[u32; 5832]>>,
    // May only be used for debugging, still TBD. Small enough to keep for now
    pub(crate) version: usize,
}
#[derive(Clone, PartialEq)]
pub(crate) struct VkChunkVertexDataCpu {
    pub(crate) opaque: Option<VkBufferCpu<CubeGeometryVertex>>,
    pub(crate) transparent: Option<VkBufferCpu<CubeGeometryVertex>>,
    pub(crate) translucent: Option<VkBufferCpu<CubeGeometryVertex>>,
    pub(crate) raytrace_fallback: Option<VkBufferCpu<CubeGeometryVertex>>,
    pub(crate) transparent_with_specular: Option<VkBufferCpu<CubeGeometryVertex>>,
}
impl VkChunkVertexDataCpu {
    fn empty() -> VkChunkVertexDataCpu {
        VkChunkVertexDataCpu {
            opaque: None,
            transparent: None,
            translucent: None,
            raytrace_fallback: None,
            transparent_with_specular: None,
        }
    }

    pub(crate) fn to_gpu(&self, allocator: Arc<VkAllocator>) -> Result<VkChunkVertexDataGpu> {
        let work = move |x: &VkBufferCpu<CubeGeometryVertex>| x.to_gpu(allocator.clone());
        Ok(VkChunkVertexDataGpu {
            opaque: self.opaque.as_ref().map(&work).transpose()?.flatten(),
            transparent: self.transparent.as_ref().map(&work).transpose()?.flatten(),
            translucent: self.translucent.as_ref().map(&work).transpose()?.flatten(),
            raytrace_fallback: self
                .raytrace_fallback
                .as_ref()
                .map(&work)
                .transpose()?
                .flatten(),
            transparent_with_specular: self
                .transparent_with_specular
                .as_ref()
                .map(&work)
                .transpose()?
                .flatten(),
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

    1 << trailing_zeros
}

#[inline]
fn get_texture(
    texture_coords: &FxHashMap<TextureKey, Rect>,
    tex: Option<&TextureReference>,
    width: f32,
    height: f32,
) -> MaybeDynamicRect {
    let crop = tex.and_then(|tex| tex.crop.as_ref());

    make_maybe_dynamic(
        tex.and_then(|tex| texture_coords.get(&TextureKey::from(tex)).copied())
            .unwrap_or_else(|| *texture_coords.get(&TextureKey::FallbackUnknownTex).unwrap()),
        crop,
        width,
        height,
    )
}

fn make_maybe_dynamic(
    mut rect: Rect,
    crop: Option<&TextureCrop>,
    width: f32,
    height: f32,
) -> MaybeDynamicRect {
    let mut rect_f = RectF32::new(
        rect.x as f32 / width,
        rect.y as f32 / height,
        rect.w as f32 / width,
        rect.h as f32 / height,
    );
    if let Some(crop) = crop {
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
    texture_atlas: TextureAtlas,
    selection_box_tex_coord: RectF32,
    fallback_tex_coord: RectF32,
    simple_block_tex_coords: SimpleTexCoordCache,
    axis_aligned_box_blocks: AxisAlignedBoxBlocksCache,
    vk_ctx: Arc<VulkanContext>,
    raytrace_control_ssbo: Subbuffer<[SimpleCubeInfo]>,
}

impl BlockRenderer {
    pub(crate) fn raytrace_control_ssbo(&self) -> Subbuffer<[SimpleCubeInfo]> {
        self.raytrace_control_ssbo.clone()
    }
}

const BLACK_PIXEL: &'static str = "builtin:black_pixel";

impl BlockRenderer {
    pub(crate) async fn new(
        block_type_manager: Arc<ClientBlockTypeManager>,
        cache_manager: &mut CacheManager,
        ctx: Arc<VulkanContext>,
    ) -> Result<BlockRenderer> {
        let mut fetch_textures = HashSet::new();
        let mut pack_textures: HashSet<TextureKey> = HashSet::new();
        for def in block_type_manager.all_block_defs() {
            let mut insert_if_present = |tex: &Option<TextureReference>| {
                if let Some(tex) = tex {
                    fetch_textures.insert(tex.diffuse.clone());
                    if !tex.rt_specular.is_empty() {
                        fetch_textures.insert(tex.rt_specular.clone());
                    }
                    if !tex.emissive.is_empty() {
                        fetch_textures.insert(tex.emissive.clone());
                    }
                    pack_textures.insert(tex.into());
                }
            };
            match &def.render_info {
                Some(RenderInfo::Cube(cube)) => {
                    insert_if_present(&cube.tex_back);
                    insert_if_present(&cube.tex_front);
                    insert_if_present(&cube.tex_left);
                    insert_if_present(&cube.tex_right);
                    insert_if_present(&cube.tex_top);
                    insert_if_present(&cube.tex_bottom);
                }
                Some(RenderInfo::PlantLike(plant_like)) => insert_if_present(&plant_like.tex),
                Some(RenderInfo::AxisAlignedBoxes(aa_boxes)) => {
                    for aa_box in &aa_boxes.boxes {
                        insert_if_present(&aa_box.tex_back);
                        insert_if_present(&aa_box.tex_front);
                        insert_if_present(&aa_box.tex_left);
                        insert_if_present(&aa_box.tex_right);
                        insert_if_present(&aa_box.tex_top);
                        insert_if_present(&aa_box.tex_bottom);
                        insert_if_present(&aa_box.plant_like_tex);
                    }
                }
                Some(RenderInfo::Empty(_)) => {}
                None => {
                    log::warn!("Got a block without renderinfo: {}", def.short_name)
                }
            }
        }

        let mut fetched_textures = FxHashMap::default();
        for texture_name in fetch_textures {
            let texture = load_or_generate_image(cache_manager, &texture_name).await?;
            fetched_textures.insert(texture_name, texture);
        }

        let texture_atlas = TextureAtlas::new(&ctx, pack_textures, fetched_textures)?;
        let simple_block_tex_coords = SimpleTexCoordCache {
            blocks: block_type_manager
                .block_defs()
                .iter()
                .map(|x| {
                    x.as_ref().and_then(|x| {
                        build_simple_cache_entry(
                            x,
                            &texture_atlas.texel_coords,
                            &block_type_manager,
                            texture_atlas.width as f32,
                            texture_atlas.height as f32,
                        )
                    })
                })
                .collect(),
        };
        let iterator = block_type_manager
            .block_defs()
            .iter()
            .map(|x| match x.as_ref() {
                None => SimpleCubeInfo {
                    flags: 0.into(),
                    tex: [TexRef::default(); 6],
                },
                Some(x) => build_ssbo_entry(
                    &x,
                    &texture_atlas.texel_coords,
                    &block_type_manager,
                    texture_atlas.width as f32,
                    texture_atlas.height as f32,
                ),
            });
        let raytrace_control_ssbo =
            ctx.iter_to_device_via_staging(iterator, BufferUsage::STORAGE_BUFFER)?;

        let axis_aligned_box_blocks = AxisAlignedBoxBlocksCache {
            blocks: block_type_manager
                .block_defs()
                .iter()
                .map(|x| {
                    x.as_ref().and_then(|x| {
                        build_axis_aligned_box_cache_entry(
                            x,
                            &texture_atlas.texel_coords,
                            texture_atlas.width as f32,
                            texture_atlas.height as f32,
                        )
                    })
                })
                .collect(),
        };

        let selection_rect: RectF32 = texture_atlas
            .texel_coords
            .get(&TextureKey::SelectionRectangle)
            .unwrap()
            .into();
        let fallback_rect: RectF32 = texture_atlas
            .texel_coords
            .get(&TextureKey::SelectionRectangle)
            .unwrap()
            .into();
        let atlas_dims = (texture_atlas.width, texture_atlas.height);
        Ok(BlockRenderer {
            block_defs: block_type_manager,
            texture_atlas,
            selection_box_tex_coord: selection_rect.div(atlas_dims),
            fallback_tex_coord: fallback_rect.div(atlas_dims),
            simple_block_tex_coords,
            axis_aligned_box_blocks,
            raytrace_control_ssbo,
            vk_ctx: ctx,
        })
    }

    pub(crate) fn block_types(&self) -> &ClientBlockTypeManager {
        &self.block_defs
    }

    pub(crate) fn atlas(&self) -> &TextureAtlas {
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

    pub(crate) fn build_raytrace_data(
        &self,
        block_ids: &[BlockId; 18 * 18 * 18],
    ) -> Option<VkChunkRaytraceData> {
        let _span = span!("build_raytrace_data");

        if !block_ids
            .iter()
            .any(|x| self.block_defs.is_raytrace_present(*x))
        {
            return None;
        }

        let max_block_id = ((self.raytrace_control_ssbo.len() as u32) << 12) - 1;
        let mut flags = FLAG_HASHTABLE_PRESENT;

        let blocks = if block_ids
            .iter()
            .any(|x| x.0 > max_block_id || !self.block_defs.is_raytrace_present(*x))
        {
            Some(Box::new(block_ids.map(|x| {
                if x.0 > max_block_id {
                    0
                } else if !self.block_defs.is_raytrace_present(x) {
                    0
                } else {
                    x.0
                }
            })))
        } else {
            None
        };

        if block_ids
            .iter()
            .any(|x| self.block_defs.is_raytrace_heavy(*x))
        {
            flags |= FLAG_HASHTABLE_HEAVY;
        }

        Some(VkChunkRaytraceData {
            flags,
            blocks,
            version: RAYTRACE_CHUNK_VERSION_COUNTER.fetch_add(1, Ordering::Relaxed),
        })
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
            opaque: self.mesh_chunk_subpass(
                chunk_data,
                |id| self.block_types().is_opaque(id),
                |block, neighbor| {
                    if self.block_defs.is_solid_opaque(neighbor) {
                        return true;
                    }
                    // Need to be careful here, since the neighbor block isn't a full block.
                    // If we're OK with suppressing on exact matches and the neighbor matches,
                    // we're good. Likewise, if we can suppress based on base block, and it matches
                    (self
                        .block_defs
                        .allow_face_suppress_on_same_base_block(block)
                        && block.equals_ignore_variant(neighbor))
                        || (self.block_defs.allow_face_suppress_on_exact_match(block)
                            && block == neighbor)
                },
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
            raytrace_fallback: self.mesh_chunk_subpass(
                chunk_data,
                |block| self.block_defs.is_raytrace_fallback_render(block),
                |_, _| false,
                &RAYTRACE_FALLBACK_RECLAIMER,
            ),
            transparent_with_specular: self.mesh_chunk_subpass(
                chunk_data,
                |block| self.block_defs.is_transparent_with_specular(block),
                |block, neighbor| {
                    block.equals_ignore_variant(neighbor)
                        || self.block_defs.is_solid_opaque(neighbor)
                },
                &TRANSPARENT_WITH_SPECULAR_RECLAIMER,
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
    ) -> Option<VkBufferCpu<CubeGeometryVertex>>
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
            Some(VkBufferCpu { vtx, idx })
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
        let (e, rotator) = match render_info.variant_effect() {
            CubeVariantEffect::None => (FULL_CUBE_EXTENTS, 0),
            CubeVariantEffect::RotateNesw => {
                (FULL_CUBE_EXTENTS.rotate_y(id.variant()), id.variant())
            }
            CubeVariantEffect::Liquid => (build_liquid_cube_extents(chunk_data, offset, id), 0),
            CubeVariantEffect::CubeVariantHeight => {
                (cube_variant_height_unblended(id.variant()), 0)
            }
        };

        let pos = vec3(offset.x.into(), offset.y.into(), offset.z.into());

        let textures = self
            .simple_block_tex_coords
            .get(id)
            .unwrap_or([self.fallback_tex_coord; 6]);
        self.emit_single_cube_impl(
            e,
            rotator,
            offset,
            suppress_face_when,
            id,
            chunk_data,
            pos,
            textures,
            vtx,
            idx,
        );
    }

    #[inline]
    fn emit_single_cube_impl<F>(
        &self,
        e: CubeExtents,
        rotator: u16,
        offset: ChunkOffset,
        suppress_face_when: F,
        id: BlockId,
        chunk_data: &impl ChunkDataView,
        pos: Vector3<f32>,
        textures: [RectF32; 6],
        vtx: &mut Vec<CubeGeometryVertex>,
        idx: &mut Vec<u32>,
    ) where
        F: Fn(BlockId, BlockId) -> bool,
    {
        for i in 0..6 {
            let (n_x, n_y, n_z) = e.adjacency[i];
            let neighbor_index = (
                offset.x as i8 + n_x,
                offset.y as i8 + n_y,
                offset.z as i8 + n_z,
            )
                .as_extended_index();
            if e.force_face(i) || !suppress_face_when(id, chunk_data.block_ids()[neighbor_index]) {
                emit_cube_face_vk(
                    pos,
                    textures[i],
                    CUBE_EXTENTS_FACE_ORDER[i],
                    CUBE_EXTENTS_FACE_ORDER[i].rotate_y(4 - (rotator & 3)),
                    vtx,
                    idx,
                    e,
                    0,
                    chunk_data.lightmap()[neighbor_index],
                    0,
                );
            }
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
                face,
                vtx,
                idx,
                e,
                chunk_data.lightmap()[offset.as_extended_index()],
                0x00,
                (plantlike_render_info.wave_effect_scale * 255.0).clamp(0.0, 255.0) as u8,
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

            let rotation = match aabb.rotation {
                AabbRotation::None => 0,
                AabbRotation::Nesw => id.variant() % 4,
            };
            e = e.rotate_y(rotation);
            match aabb.textures {
                CachedAabbTextures::Prism(textures) => {
                    for i in 0..6 {
                        let (n_x, n_y, n_z) = e.adjacency[i];
                        let neighbor_index = (
                            offset.x as i8 + n_x,
                            offset.y as i8 + n_y,
                            offset.z as i8 + n_z,
                        )
                            .as_extended_index();

                        emit_cube_face_vk(
                            pos,
                            textures[i].rect(id.variant()),
                            CUBE_EXTENTS_FACE_ORDER[i],
                            CUBE_EXTENTS_FACE_ORDER[i].rotate_y(rotation),
                            vtx,
                            idx,
                            e,
                            chunk_data.lightmap()[neighbor_index],
                            chunk_data.lightmap()[offset.as_extended_index()],
                            0,
                        );
                    }
                }
                CachedAabbTextures::Plantlike(plantlike) => {
                    for i in 0..4 {
                        emit_cube_face_vk(
                            pos,
                            plantlike.rect(id.variant()),
                            PLANTLIKE_FACE_ORDER[i],
                            PLANTLIKE_FACE_ORDER[i].rotate_y(rotation),
                            vtx,
                            idx,
                            e,
                            chunk_data.lightmap()[offset.as_extended_index()],
                            0x00,
                            0,
                        );
                    }
                }
            }
        }
    }

    fn get_block_id(&self, ids: &[BlockId; 18 * 18 * 18], coord: ChunkOffset) -> BlockId {
        ids[coord.as_extended_index()]
    }

    pub(crate) fn make_pointee_cube(
        &self,
        player_position: Vector3<f64>,
        pointee: ToolTarget,
        state: &ClientState,
        tick: u64,
    ) -> Result<Option<CubeGeometryDrawCall>> {
        let mut vtx = vec![];
        let mut idx = vec![];
        let frame = self.selection_box_tex_coord;
        const POINTEE_SELECTION_EXTENTS: CubeExtents =
            CubeExtents::new((-0.51, 0.51), (-0.51, 0.51), (-0.51, 0.51));
        let e = POINTEE_SELECTION_EXTENTS;
        let vk_pos = Vector3::zero();

        for &face in &CUBE_EXTENTS_FACE_ORDER {
            emit_cube_face_vk(
                vk_pos, frame, face, face, &mut vtx, &mut idx, e, 0x0f, 0x00, 0,
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

        let model_matrix = match pointee {
            ToolTarget::Block(pointee) => {
                let translation = (vec3(pointee.x as f64, pointee.y as f64, pointee.z as f64)
                    - player_position)
                    .mul_element_wise(Vector3::new(1., -1., 1.));
                Matrix4::from_translation(translation.cast().unwrap())
            }
            ToolTarget::Entity(EntityTarget {
                entity_id,
                trailing_entity_index,
            }) => {
                let task = || -> Option<Matrix4<f32>> {
                    let (transformation, class) = state
                        .entities
                        .lock()
                        .transforms_for_entity(
                            player_position,
                            tick,
                            &state.entity_renderer,
                            entity_id,
                        )
                        .map(|x| x.skip(trailing_entity_index as usize).next())
                        .flatten()?;
                    let (min, max) = state.entity_renderer.mesh_aabb(class)?;
                    let range = max - min;
                    let center = (min + max) / 2.0;
                    let translation = vec3(center.x, center.y, center.z);
                    let prescale = Matrix4::from_nonuniform_scale(range.x, range.y, range.z);
                    Some(transformation * Matrix4::from_translation(translation) * prescale)
                };

                match task() {
                    Some(x) => x,
                    None => return Ok(None),
                }
            }
        };
        let buffer = VkDrawBufferGpu::<CubeGeometryVertex> {
            num_indices: idx.len() as u32,
            vtx,
            idx,
        };
        Ok(Some(CubeGeometryDrawCall {
            model_matrix,
            models: VkChunkVertexDataGpu {
                opaque: None,
                transparent: Some(buffer.clone()),
                translucent: None,
                raytrace_fallback: Some(buffer),
                transparent_with_specular: None,
            },
        }))
    }
}

fn build_axis_aligned_box_cache_entry(
    x: &BlockTypeDef,
    texture_coords: &FxHashMap<TextureKey, Rect>,
    atlas_w: f32,
    atlas_h: f32,
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

            // plantlike overrides box-like
            let textures = if let Some(plantlike) = aa_box.plant_like_tex.as_ref() {
                CachedAabbTextures::Plantlike(get_texture(
                    texture_coords,
                    Some(plantlike),
                    atlas_w,
                    atlas_h,
                ))
            } else {
                CachedAabbTextures::Prism([
                    get_texture(texture_coords, aa_box.tex_right.as_ref(), atlas_w, atlas_h),
                    get_texture(texture_coords, aa_box.tex_left.as_ref(), atlas_w, atlas_h),
                    get_texture(texture_coords, aa_box.tex_top.as_ref(), atlas_w, atlas_h),
                    get_texture(texture_coords, aa_box.tex_bottom.as_ref(), atlas_w, atlas_h),
                    get_texture(texture_coords, aa_box.tex_back.as_ref(), atlas_w, atlas_h),
                    get_texture(texture_coords, aa_box.tex_front.as_ref(), atlas_w, atlas_h),
                ])
            };
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

pub(crate) mod rt_flags {
    pub(crate) const FLAG_BLOCK_PRESENT: u32 = 1;
    pub(crate) const FLAG_BLOCK_ROTATE_NESW: u32 = 2;
    pub(crate) const FLAG_BLOCK_FALLBACK: u32 = 4;
}

fn build_simple_cache_entry(
    block_def: &BlockTypeDef,
    texture_coords: &FxHashMap<TextureKey, Rect>,
    block_type_manager: &ClientBlockTypeManager,
    w: f32,
    h: f32,
) -> Option<SimpleTexCoordEntry> {
    use rt_flags::*;
    match &block_def.render_info {
        Some(RenderInfo::Cube(render_info)) => {
            let mut flags = FLAG_BLOCK_PRESENT;
            if render_info.variant_effect() == CubeVariantEffect::RotateNesw {
                flags |= FLAG_BLOCK_ROTATE_NESW;
            }
            if block_type_manager.is_raytrace_fallback_render(BlockId(block_def.id)) {
                flags |= FLAG_BLOCK_FALLBACK;
            }
            Some(SimpleTexCoordEntry {
                coords: get_cube_coords(texture_coords, w, h, render_info),
            })
        }
        Some(RenderInfo::PlantLike(render_info)) => Some(SimpleTexCoordEntry {
            coords: [get_texture(texture_coords, render_info.tex.as_ref(), w, h); 6],
        }),
        _ => None,
    }
}

#[inline]
fn get_cube_coords(
    tex_coords: &FxHashMap<TextureKey, Rect>,
    w: f32,
    h: f32,
    render_info: &CubeRenderInfo,
) -> [MaybeDynamicRect; 6] {
    [
        get_texture(tex_coords, render_info.tex_right.as_ref(), w, h),
        get_texture(tex_coords, render_info.tex_left.as_ref(), w, h),
        get_texture(tex_coords, render_info.tex_top.as_ref(), w, h),
        get_texture(tex_coords, render_info.tex_bottom.as_ref(), w, h),
        get_texture(tex_coords, render_info.tex_back.as_ref(), w, h),
        get_texture(tex_coords, render_info.tex_front.as_ref(), w, h),
    ]
}

fn build_ssbo_entry(
    block_def: &BlockTypeDef,
    texture_coords: &FxHashMap<TextureKey, Rect>,
    block_type_manager: &ClientBlockTypeManager,
    w: f32,
    h: f32,
) -> SimpleCubeInfo {
    use rt_flags::*;
    match &block_def.render_info {
        Some(RenderInfo::Cube(render_info)) => {
            let mut flags = FLAG_BLOCK_PRESENT;
            if render_info.variant_effect() == CubeVariantEffect::RotateNesw {
                flags |= FLAG_BLOCK_ROTATE_NESW;
            }
            if block_type_manager.is_raytrace_fallback_render(BlockId(block_def.id)) {
                flags |= FLAG_BLOCK_FALLBACK;
            }
            let coords = get_cube_coords(texture_coords, w, h, render_info);
            SimpleCubeInfo {
                flags: flags.into(),
                tex: coords.map(|x| x.rect(0).into()),
            }
        }
        Some(RenderInfo::PlantLike(render_info)) => {
            let coords = get_texture(texture_coords, render_info.tex.as_ref(), w, h);
            SimpleCubeInfo {
                flags: (1 | 4).into(),
                tex: [coords.rect(0).into(); 6],
            }
        }
        _ => SimpleCubeInfo {
            // This is the default, but written explicitly to better highlight the exact value
            flags: 0.into(),
            tex: [TexRef::default(); 6],
        },
    }
}

const FULL_CUBE_EXTENTS: CubeExtents = CubeExtents::new((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5));

fn cube_variant_height_unblended(variant: u16) -> CubeExtents {
    let y_max = variant_to_height(variant);
    CubeExtents {
        vertices: [
            vec3(-0.5, y_max, -0.5),
            vec3(-0.5, y_max, 0.5),
            vec3(-0.5, 0.5, -0.5),
            vec3(-0.5, 0.5, 0.5),
            vec3(0.5, y_max, -0.5),
            vec3(0.5, y_max, 0.5),
            vec3(0.5, 0.5, -0.5),
            vec3(0.5, 0.5, 0.5),
        ],
        adjacency: DEFAULT_FACE_NORMALS,
        // top face should be forced if it's not flush with the bottom of the next block
        force: [false, false, variant < 7, false, false, false],
    }
}

fn variant_to_height(variant: u16) -> f32 {
    let height = ((variant as f32) / 7.0).clamp(0.025, 1.0);
    0.5 - height
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
        adjacency: DEFAULT_FACE_NORMALS,
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

struct SimpleTexCoordEntry {
    coords: [MaybeDynamicRect; 6],
}

struct SimpleTexCoordCache {
    blocks: Vec<Option<SimpleTexCoordEntry>>,
}

impl SimpleTexCoordCache {
    fn get(&self, block_id: BlockId) -> Option<[RectF32; 6]> {
        self.blocks
            .get(block_id.index())
            .and_then(|x| x.as_ref())
            .map(|entry| entry.coords.map(|x| x.rect(block_id.variant())))
    }

    fn get_zero(&self, block_id: BlockId) -> Option<RectF32> {
        self.blocks
            .get(block_id.index())
            .and_then(|x| x.as_ref())
            .and_then(|entry| Some(entry.coords[0].rect(block_id.variant())))
    }
}

enum AabbRotation {
    None,
    Nesw,
}

#[derive(Clone, Debug)]
enum CachedAabbTextures {
    Prism([MaybeDynamicRect; 6]),
    Plantlike(MaybeDynamicRect),
}

struct CachedAxisAlignedBox {
    extents: CubeExtents,
    textures: CachedAabbTextures,
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

const GLOBAL_BRIGHTNESS_TABLE_RAW: [f32; 16] = [
    0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75,
    0.8125, 0.875, 0.9375,
];
const BRIGHTNESS_TABLE_RAW: [f32; 16] = [
    0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125,
    0.875, 0.9375, 1.00,
];
const CUBE_FACE_BRIGHTNESS_BIASES: [f32; 10] = [
    0.975, 0.975, 1.05, 0.95, 0.975, 0.975, 0.975, 0.975, 0.975, 0.975,
];

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
    normal: u8,
    tex_uv: Vector2<f32>,
    brightness: u8,
    global_brightness: u8,
    wave_horizontal: u8,
) -> CubeGeometryVertex {
    CubeGeometryVertex {
        position: [coord.x, coord.y, coord.z],
        normal,
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
///    frame: The texture of the face.
///    source_face: The face, as referencing into CubeExtents
///    dest_face: The physical direction this face will be oriented. Currently only used for normals
///      and some lighting tweaks
///    vert_buf, idx_buf: The buffers to append to
///    e: The actual cube extents
///    encoded_brightness(_2): The brightnesses that will contribute to this face's effective
///         brightness; these two are max'd elementwise
///    horizontal_wave: The strength of how much this face will wave at the top, 0-255
#[inline]
pub(crate) fn emit_cube_face_vk(
    coord: Vector3<f32>,
    frame: RectF32,
    source_face: CubeFace,
    dest_face: CubeFace,
    vert_buf: &mut Vec<CubeGeometryVertex>,
    idx_buf: &mut Vec<u32>,
    e: CubeExtents,
    encoded_brightness: u8,
    encoded_brightness_2: u8,
    horizontal_wave: u8,
) {
    // Flip the coordinate system to Vulkan
    let coord = vec3(coord.x, -coord.y, coord.z);
    let l = frame.left();
    let r = frame.right();
    let t = frame.top();
    let b = frame.bottom();
    let tl = Vector2::new(l, t);
    let bl = Vector2::new(l, b);
    let tr = Vector2::new(r, t);
    let br = Vector2::new(r, b);

    let brightness_bias = CUBE_FACE_BRIGHTNESS_BIASES[dest_face.index()];
    let (global_brightness_1, brightness_1) = get_brightnesses(encoded_brightness, brightness_bias);
    let (global_brightness_2, brightness_2) =
        get_brightnesses(encoded_brightness_2, brightness_bias);

    let brightness = brightness_1.max(brightness_2);
    let global_brightness = global_brightness_1.max(global_brightness_2);

    let normal = dest_face.repr();
    let vertices = match source_face {
        CubeFace::ZMinus => [
            make_cgv(
                coord + e.vertices[0],
                normal,
                tl,
                brightness,
                global_brightness.max(global_brightness_2),
                0,
            ),
            make_cgv(
                coord + e.vertices[2],
                normal,
                bl,
                brightness,
                global_brightness.max(global_brightness_2),
                0,
            ),
            make_cgv(
                coord + e.vertices[6],
                normal,
                br,
                brightness,
                global_brightness,
                0,
            ),
            make_cgv(
                coord + e.vertices[4],
                normal,
                tr,
                brightness,
                global_brightness,
                0,
            ),
        ],
        CubeFace::ZPlus => [
            make_cgv(
                coord + e.vertices[5],
                normal,
                tl,
                brightness,
                global_brightness,
                0,
            ),
            make_cgv(
                coord + e.vertices[7],
                normal,
                bl,
                brightness,
                global_brightness,
                0,
            ),
            make_cgv(
                coord + e.vertices[3],
                normal,
                br,
                brightness,
                global_brightness,
                0,
            ),
            make_cgv(
                coord + e.vertices[1],
                normal,
                tr,
                brightness,
                global_brightness,
                0,
            ),
        ],
        CubeFace::XMinus => [
            make_cgv(
                coord + e.vertices[1],
                normal,
                tl,
                brightness,
                global_brightness,
                0,
            ),
            make_cgv(
                coord + e.vertices[3],
                normal,
                bl,
                brightness,
                global_brightness,
                0,
            ),
            make_cgv(
                coord + e.vertices[2],
                normal,
                br,
                brightness,
                global_brightness,
                0,
            ),
            make_cgv(
                coord + e.vertices[0],
                normal,
                tr,
                brightness,
                global_brightness,
                0,
            ),
        ],
        CubeFace::XPlus => [
            make_cgv(
                coord + e.vertices[4],
                normal,
                tl,
                brightness,
                global_brightness,
                0,
            ),
            make_cgv(
                coord + e.vertices[6],
                normal,
                bl,
                brightness,
                global_brightness,
                0,
            ),
            make_cgv(
                coord + e.vertices[7],
                normal,
                br,
                brightness,
                global_brightness,
                0,
            ),
            make_cgv(
                coord + e.vertices[5],
                normal,
                tr,
                brightness,
                global_brightness,
                0,
            ),
        ],
        // For Y+ and Y-, the top of the texture is front of the cube (prior to any rotations)
        // Y- is the bottom face (opposite Vulkat takes Y+ vulkan coordinates
        CubeFace::YMinus => [
            make_cgv(
                coord + e.vertices[2],
                normal,
                tl,
                brightness,
                global_brightness,
                0,
            ),
            make_cgv(
                coord + e.vertices[3],
                normal,
                bl,
                brightness,
                global_brightness,
                0,
            ),
            make_cgv(
                coord + e.vertices[7],
                normal,
                br,
                brightness,
                global_brightness,
                0,
            ),
            make_cgv(
                coord + e.vertices[6],
                normal,
                tr,
                brightness,
                global_brightness,
                0,
            ),
        ],
        CubeFace::YPlus => [
            make_cgv(
                coord + e.vertices[4],
                normal,
                tl,
                brightness,
                global_brightness,
                0,
            ),
            make_cgv(
                coord + e.vertices[5],
                normal,
                bl,
                brightness,
                global_brightness,
                0,
            ),
            make_cgv(
                coord + e.vertices[1],
                normal,
                br,
                brightness,
                global_brightness,
                0,
            ),
            make_cgv(
                coord + e.vertices[0],
                normal,
                tr,
                brightness,
                global_brightness,
                0,
            ),
        ],
        CubeFace::PlantXMinusZMinus => [
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
                0,
            ),
            make_cgv(
                coord + e.vertices[6],
                normal,
                br,
                brightness,
                global_brightness,
                0,
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
        CubeFace::PlantXPlusZPlus => [
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
                0,
            ),
            make_cgv(
                coord + e.vertices[3],
                normal,
                br,
                brightness,
                global_brightness,
                0,
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
        CubeFace::PlantXMinusZPlus => [
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
                0,
            ),
            make_cgv(
                coord + e.vertices[2],
                normal,
                br,
                brightness,
                global_brightness,
                0,
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
        CubeFace::PlantXPlusZMinus => [
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
                0,
            ),
            make_cgv(
                coord + e.vertices[7],
                normal,
                br,
                brightness,
                global_brightness,
                0,
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
    vert_buf.extend_from_slice(&vertices);
    idx_buf.extend_from_slice(&[si, si + 1, si + 2, si, si + 2, si + 3]);
}

#[inline]
fn get_brightnesses(encoded_brightness: u8, brightness_bias: f32) -> (u8, u8) {
    let brightness_upper = (encoded_brightness >> 4) as usize;
    let brightness_lower = (encoded_brightness & 0x0F) as usize;

    let global_brightness = GLOBAL_BRIGHTNESS_TABLE[brightness_upper];
    let brightness = BRIGHTNESS_TABLE[brightness_lower] * brightness_bias;
    (
        (global_brightness * 255.0) as u8,
        (brightness * 255.0) as u8,
    )
}
