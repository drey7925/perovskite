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

use super::VkAllocator;
use crate::client_state::settings::Supersampling;
use anyhow::Result;
use cgmath::{Matrix4, Vector3};
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};

pub(crate) mod cube_geometry;
pub(crate) mod egui_adapter;
pub(crate) mod entity_geometry;
pub(crate) mod flat_texture;
pub(crate) mod raytracer;

/// Shaders that render 3D
/// If we need more advanced texturing, this might be subdivided later on
pub(crate) mod vert_3d {
    vulkano_shaders::shader! {
        shaders: {
            cube_geometry: {
            ty: "vertex",
            src: r"
            #version 460
                layout(location = 0) in vec3 position;
                layout(location = 2) in vec2 uv_texcoord;
                layout(location = 1) in uint normal;
                layout(location = 3) in float brightness;
                layout(location = 4) in float global_brightness_contribution;
                layout(location = 5) in float wave_horizontal;

                layout(set = 1, binding = 0) uniform UniformData { 
                    mat4 vp_matrix;
                    vec2 plant_wave_vector;
                    vec3 global_brightness_color;
                    vec3 global_light_direction;
                };
                // 64 bytes of push constants :(
                layout(push_constant) uniform ModelMatrix {
                    mat4 model_matrix;
                };

                layout(location = 0) out vec2 uv_texcoord_out;
                layout(location = 1) flat out float brightness_out;
                layout(location = 2) flat out vec3 global_brightness_out;
                layout(location = 3) flat out vec3 world_normal_out;
                layout(location = 4) out vec3 world_pos_out;


                vec3 decode_normal(uint index) {
                     const float sqrt_half = sqrt(0.5);
                     // Matches CubeFace in BlockRenderer
                     const vec3 normals[10] = vec3[](
                        vec3(1.0, 0.0, 0.0),
                        vec3(-1.0, 0.0, 0.0),
                        // Warning: CubeFace Y+ then Y- is in world coords, not Vk coords
                        vec3(0.0, -1.0, 0.0),
                        vec3(0.0, 1.0, 0.0),
                        vec3(0.0, 0.0, 1.0),
                        vec3(0.0, 0.0, -1.0),
                        vec3(sqrt_half, 0.0, sqrt_half),
                        vec3(sqrt_half, 0.0, -sqrt_half),
                        vec3(-sqrt_half, 0.0, sqrt_half),
                        vec3(-sqrt_half, 0.0, -sqrt_half)
                    );
                    return normals[index];
                }
                void main() {
                    float wave_x = wave_horizontal * plant_wave_vector.x;
                    float wave_z = wave_horizontal * plant_wave_vector.y;
                    vec3 position_with_wave = vec3(position.x + wave_x, position.y, position.z + wave_z);
                    vec4 world_pos = model_matrix * vec4(position_with_wave, 1.0);
                    world_pos_out = world_pos.xyz / world_pos.w;
                    gl_Position = vp_matrix * world_pos;

                    uv_texcoord_out = uv_texcoord;
                    brightness_out = brightness;

                    world_normal_out = decode_normal(normal);
                    float gbc_adjustment = 0.5 + 0.5 * max(0, dot(global_light_direction, world_normal_out));
                    global_brightness_out = global_brightness_color * global_brightness_contribution * gbc_adjustment;
                }
            "
            },
            // WIP entity shader - essentially just a simple transform-and-shade pipeline. At some
            // point, this should be re-structured to better scale entity rendering (e.g. doing move
            // calcs on the GPU for batches of entities)
            entity_tentative: {
            ty: "vertex",
            src: r"#version 460
                layout(location = 0) in vec3 position;
                layout(location = 1) in vec3 normal;
                layout(location = 2) in vec2 uv_texcoord;
                // TODO: bring back entity lighting at some later point

                layout(set = 1, binding = 0) uniform EntityUniformData {
                    mat4 vp_matrix;
                };
                // 64 bytes of push constants :(
                layout(push_constant) uniform ModelMatrix {
                    mat4 model_matrix;
                };

                layout(location = 0) out vec2 uv_texcoord_out;
                layout(location = 1) flat out float brightness_out;
                layout(location = 2) flat out vec3 global_brightness_out;
                layout(location = 3) flat out vec3 world_normal_out;
                layout(location = 4) out vec3 world_pos_out;

                void main() {
                    vec4 world_pos = model_matrix * vec4(position, 1.0);
                    world_pos_out = world_pos.xyz / world_pos.w;
                    gl_Position = vp_matrix * world_pos;
                    uv_texcoord_out = uv_texcoord;
                    brightness_out = 1.0;
                    global_brightness_out = vec3(0.0, 0.0, 0.0);
                }
            "
            },
        }
    }

    impl From<cgmath::Matrix4<f32>> for ModelMatrix {
        fn from(value: cgmath::Matrix4<f32>) -> Self {
            ModelMatrix {
                model_matrix: value.into(),
            }
        }
    }
}

pub(crate) mod frag_lighting {
    vulkano_shaders::shader! {
    ty: "fragment",
    src: r"
    #version 460

    layout(location = 0) in vec2 uv_texcoord;
    layout(location = 1) flat in float brightness;
    layout(location = 2) flat in vec3 global_brightness;
    layout(location = 3) flat in vec3 world_normal;
    layout(location = 4) in vec3 world_pos;

    layout(location = 0) out vec4 f_color;
    layout(set = 0, binding = 0) uniform sampler2D diffuse_tex;
    layout(set = 0, binding = 1) uniform sampler2D emissive_tex;

    layout (constant_id = 0) const bool SPARSE = true;
    layout (constant_id = 1) const bool DEBUG_INVERT_RASTER_TRANSPARENT = true;

    void main() {
        vec2 pix = gl_FragCoord.xy / 8.0;
        vec4 diffuse = texture(diffuse_tex, uv_texcoord);
        vec4 emissive = texture(emissive_tex, uv_texcoord);

        float normal_pos_dot = abs(dot(normalize(world_normal), normalize(world_pos)));

        f_color = vec4((brightness + global_brightness) * diffuse.rgb, diffuse.a);
        float emissive_anisotropy_power = mix(10.0, 0.001, emissive.a);
        f_color.rgb += pow(normal_pos_dot, emissive_anisotropy_power) * emissive.rgb;

        if (DEBUG_INVERT_RASTER_TRANSPARENT) {
            f_color.rgb = 1.0 - f_color.rgb;
        }

        if (SPARSE) {
            if (f_color.a < 0.5) {
                discard;
            } else {
                f_color.a = 1.0;
            }
        }
    }
    "
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

pub(crate) struct LiveRenderConfig {
    pub(crate) supersampling: Supersampling,
    pub(crate) raytracing: bool,
    pub(crate) raytracing_reflections: bool,
    pub(crate) render_distance: u32,
    pub(crate) raytracer_debug: bool,
    pub(crate) raytracing_specular_downsampling: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SceneState {
    pub(crate) vp_matrix: Matrix4<f32>,
    pub(crate) global_light_color: [f32; 3],
    // TODO: This doesn't do anything because the background is rendered by the sky shader
    pub(crate) clear_color: [f32; 4],
    pub(crate) sun_direction: Vector3<f32>,
    pub(crate) player_pos_block: u32,
}

#[derive(Clone, PartialEq)]
pub(crate) struct VkBufferCpu<T: BufferContents + Copy> {
    pub(crate) vtx: Vec<T>,
    pub(crate) idx: Vec<u32>,
}
impl<T: BufferContents + Copy> VkBufferCpu<T> {
    pub(crate) fn to_gpu(&self, allocator: Arc<VkAllocator>) -> Result<Option<VkBufferGpu<T>>> {
        VkBufferGpu::from_buffers(&self.vtx, &self.idx, allocator)
    }
}

#[derive(Clone)]
pub(crate) struct VkBufferGpu<T: BufferContents + Copy> {
    pub(crate) vtx: Subbuffer<[T]>,
    pub(crate) idx: Subbuffer<[u32]>,
}
impl<T: BufferContents + Copy> VkBufferGpu<T> {
    pub(crate) fn from_buffers(
        vtx: &[T],
        idx: &[u32],
        allocator: Arc<VkAllocator>,
    ) -> Result<Option<VkBufferGpu<T>>> {
        if vtx.is_empty() {
            Ok(None)
        } else {
            Ok(Some(VkBufferGpu {
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
