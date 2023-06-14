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

use anyhow::Result;

use super::{CommandBufferBuilder, VulkanContext};

pub(crate) mod cube_geometry;
pub(crate) mod egui_adapter;
pub(crate) mod flat_texture;

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
                layout(location = 1) in vec2 uv_texcoord;
                layout(location = 2) in float brightness;

                layout(set = 1, binding = 0) uniform UniformData { 
                    mat4 vp_matrix;
                };
                // 64 bytes of push constants :(
                layout(push_constant) uniform ModelMatrix {
                    mat4 model_matrix;
                };

                layout(location = 0) out vec2 uv_texcoord_out;
                layout(location = 1) out float brightness_out;

                void main() {
                    gl_Position = vp_matrix * model_matrix * vec4(position, 1.0);
                    uv_texcoord_out = uv_texcoord;
                    brightness_out = brightness;
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

// Fragment shader(s) that use lighting data
pub(crate) mod frag_lighting {
    vulkano_shaders::shader! {
    ty: "fragment",
    src: r"
    #version 460

    layout(location = 0) in vec2 uv_texcoord;
    layout(location = 1) in float brightness;

    layout(location = 0) out vec4 f_color;
    layout(set = 0, binding = 0) uniform sampler2D tex;

    void main() {
        // todo use brightness data
        f_color = texture(tex, uv_texcoord);
    }
    "
    }
}
// Fragment shader(s) that use lighting data and does a discard based on alpha
// Alpha=0 and alpha=1 are supported, behavior is undefined for other values.
pub(crate) mod frag_lighting_sparse {
    vulkano_shaders::shader! {
    ty: "fragment",
    src: r"
    #version 460

    layout(location = 0) in vec2 uv_texcoord;
    layout(location = 1) in float brightness;

    layout(location = 0) out vec4 f_color;
    layout(set = 0, binding = 0) uniform sampler2D tex;

    void main() {
        // todo brightness
        f_color = texture(tex, uv_texcoord);
        if (f_color.a < 0.5) {
            discard;
        } else {
            f_color.a = 1.0;
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
                layout(location = 1) out float brightness_out;

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

// Fragment shader(s) that simply render colors directly
pub(crate) mod frag_simple {
    vulkano_shaders::shader! {
    ty: "fragment",
    src: r"
    #version 460

    layout(location = 0) in vec2 uv_texcoord;

    layout(location = 0) out vec4 f_color;
    layout(set = 0, binding = 1) uniform sampler2D tex;

    void main() {
        f_color = texture(tex, uv_texcoord);
    }
    "
    }
}

pub(crate) trait PipelineWrapper<T, U> {
    type PassIdentifier;
    /// Actually draw. The pipeline must have been bound using bind.
    fn draw(
        &mut self,
        builder: &mut CommandBufferBuilder,
        draw_calls: &[T],
        pass: Self::PassIdentifier,
    ) -> Result<()>;
    /// Bind the pipeline. Must be called each frame before draw
    fn bind(
        &mut self,
        ctx: &VulkanContext,
        per_frame_config: U,
        command_buf_builder: &mut CommandBufferBuilder,
        pass: Self::PassIdentifier,
    ) -> Result<()>;
}

pub(crate) trait PipelineProvider
where
    Self: Sized,
{
    type DrawCall;
    type PerPipelineConfig;
    type PerFrameConfig;
    type PipelineWrapperImpl: PipelineWrapper<Self::DrawCall, Self::PerFrameConfig>;
    fn make_pipeline(
        &self,
        ctx: &VulkanContext,
        config: Self::PerPipelineConfig,
    ) -> Result<Self::PipelineWrapperImpl>;
}
