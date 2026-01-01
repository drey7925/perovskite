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

use super::{SelectedFormats, VkAllocator};
use crate::client_state::settings::Supersampling;
use crate::vulkan::{Texture2DHolder, VulkanContext};
use anyhow::Result;
use cgmath::{Matrix4, Vector3};
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};

pub(crate) mod cube_geometry;
pub(crate) mod egui_adapter;
pub(crate) mod entity_geometry;
pub(crate) mod far_mesh;
pub(crate) mod flat_texture;
pub(crate) mod raytracer;

/// Shaders that render 3D
/// If we need more advanced texturing, this might be subdivided later on
pub(crate) mod vert_3d {
    vulkano_shaders::shader! {
        shaders: {
            cube_geometry: {
                ty: "vertex",
                path: "src/vulkan/shaders/cube_geometry.vert"
            },
            // WIP entity shader - essentially just a simple transform-and-shade pipeline. At some
            // point, this should be re-structured to better scale entity rendering (e.g. doing move
            // calcs on the GPU for batches of entities)
            entity_tentative: {
                ty: "vertex",
                path: "src/vulkan/shaders/entity_vert.glsl"
            },
        }
    }
}

pub(crate) mod frag_lighting_basic_color {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/vulkan/shaders/raster_frag_lighting.glsl",
        // https://youtrack.jetbrains.com/issue/RUST-18217/
        define: [("ENABLE_BASIC_COLOR", "2")]
    }
}
pub(crate) mod frag_lighting_unified_specular {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/vulkan/shaders/raster_frag_lighting.glsl",
        define: [("ENABLE_UNIFIED_SPECULAR", "1")]
    }
}
pub(crate) mod frag_specular_only {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/vulkan/shaders/raster_frag_lighting.glsl",
        define: [("ENABLE_SPECULAR_ONLY", "1")]
    }
}

// Simple vertex shader for drawing flat textures to the screen
// This shader should be run without a depth test
pub(crate) mod vert_2d {
    vulkano_shaders::shader! {
        shaders: {
            flat_tex: {
            ty: "vertex",
            src: r"
            #version 460
                layout(location = 0) in vec2 position;
                layout(location = 1) in vec2 uv_texcoord;

                layout(set = 1, binding = 0) uniform UniformData { 
                    vec2 device_w_h;
                };

                layout(location = 0) out vec2 uv_texcoord_out;

                void main() {
                    // pixel -> normalized device coordinates
                    vec2 ndc = (2.0 * position + 1) / device_w_h - 1.0;
                    // Z shouldn't matter (no depth test)
                    gl_Position = vec4(ndc, 0.5, 1.0);
                    uv_texcoord_out = uv_texcoord;
                }
            "
            }
        }
    }
}

pub(crate) mod post_process;
pub(crate) mod sky;

// Fragment shader(s) that simply render colors directly
pub(crate) mod frag_simple {
    vulkano_shaders::shader! {
    ty: "fragment",
    src: r"
    #version 460

    layout(location = 0) in vec2 uv_texcoord;

    layout(location = 0) out vec4 f_color;
    layout(set = 0, binding = 0) uniform sampler2D tex;

    void main() {
        f_color = texture(tex, uv_texcoord);
    }
    "
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LiveRenderConfig {
    pub(crate) supersampling: Supersampling,
    pub(crate) hdr: bool,
    pub(crate) raytracing: bool,
    pub(crate) hybrid_rt: bool,
    pub(crate) render_distance: u32,
    pub(crate) raytracer_debug: bool,
    pub(crate) raytracing_specular_downsampling: u32,
    pub(crate) blur_steps: usize,
    pub(crate) bloom_strength: f32,
    pub(crate) lens_flare_strength: f32,
    pub(crate) approx_gaussian_blit: bool,

    pub(crate) formats: SelectedFormats,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SceneState {
    pub(crate) vp_matrix: Matrix4<f32>,
    pub(crate) global_light_color: [f32; 3],
    pub(crate) sun_direction: Vector3<f32>,
    pub(crate) player_pos_block: u32,
}

#[derive(Clone, PartialEq)]
pub(crate) struct VkBufferCpu<T: BufferContents + Copy> {
    pub(crate) vtx: Vec<T>,
    pub(crate) idx: Vec<u32>,
}
impl<T: BufferContents + Copy> VkBufferCpu<T> {
    pub(crate) fn to_gpu(&self, allocator: Arc<VkAllocator>) -> Result<Option<VkDrawBufferGpu<T>>> {
        VkDrawBufferGpu::from_buffers(&self.vtx, &self.idx, allocator)
    }
}

#[derive(Clone)]
pub(crate) struct VkDrawBufferGpu<T: BufferContents + Copy> {
    pub(crate) vtx: Subbuffer<[T]>,
    pub(crate) idx: Subbuffer<[u32]>,
    pub(crate) num_indices: u32,
}
impl<T: BufferContents + Copy> VkDrawBufferGpu<T> {
    pub(crate) fn from_buffers(
        vtx: &[T],
        idx: &[u32],
        allocator: Arc<VkAllocator>,
    ) -> Result<Option<VkDrawBufferGpu<T>>> {
        if vtx.is_empty() {
            Ok(None)
        } else {
            Ok(Some(VkDrawBufferGpu {
                num_indices: idx.len() as u32,
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

pub(crate) const fn x5y5z5_pack16(x: i8, y: i8, z: i8) -> u16 {
    const fn encode(i: i8) -> u16 {
        (i & 0x1f) as u16
    }
    (encode(x) << 10) | (encode(y) << 5) | (encode(z))
}
pub(crate) fn x5y5z5_pack_vec(v: Vector3<f32>) -> u16 {
    let max_component = v.x.abs().max(v.y.abs()).max(v.z.abs());
    let v = v * 15.0 / max_component;
    x5y5z5_pack16(v.x as i8, v.y as i8, v.z as i8)
}

#[test]
fn test_x5y5z5_pack_vec() {
    assert_eq!(x5y5z5_pack_vec(Vector3::new(1.0, 0.0, 0.0)), 0xf << 10);
    assert_eq!(
        x5y5z5_pack_vec(Vector3::new(1.0, -1.0, 0.0)),
        0xf << 10 | 0x11 << 5
    );
}

// Blue noise texture for dithering. Made by running
// blue-noise generate --size 64 --output blue-noise.png --seed 28273
// with https://crates.io/crates/blue-noise version 0.2.1, Windows 11 x64,
// rustc 1.92.0
const _BLUE_NOISE_PNG_BYTES: &[u8] = include_bytes!("blue_noise.png");
fn make_blue_noise_image(ctx: &VulkanContext) -> Result<Texture2DHolder> {
    let image = image::load_from_memory(_BLUE_NOISE_PNG_BYTES)?.to_luma8();
    Ok(Texture2DHolder::from_image(
        ctx,
        image,
        vulkano::format::Format::R8_UNORM,
    )?)
}
