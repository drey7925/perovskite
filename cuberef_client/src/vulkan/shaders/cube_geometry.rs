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

use anyhow::{Context, Result};
use cgmath::{vec3, vec4, Matrix4, Vector4, Zero};
use std::sync::Arc;
use tracy_client::{plot, span};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::Device,
    memory::allocator::{AllocationCreateInfo, MemoryUsage},
    pipeline::{
        graphics::{
            color_blend::ColorBlendState,
            depth_stencil::{CompareOp, DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            rasterization::RasterizationState,
            vertex_input::Vertex,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, StateMode,
    },
    render_pass::Subpass,
    shader::ShaderModule,
};

use crate::{
    cube_renderer::VkChunkVertexData,
    vulkan::{
        shaders::{frag_lighting, vert_3d::ModelMatrix},
        CommandBufferBuilder, Texture2DHolder, VulkanContext,
    },
};

use crate::vulkan::shaders::{
    vert_3d::{self, UniformData},
    PipelineProvider, PipelineWrapper,
};

use super::frag_lighting_sparse;

#[derive(BufferContents, Vertex, Copy, Clone, Debug)]
#[repr(C)]
pub(crate) struct CubeGeometryVertex {
    #[format(R32G32B32_SFLOAT)]
    /// Position, given relative to the origin of the chunk.
    /// Not transformed into camera space via a view matrix yet
    pub(crate) position: [f32; 3],
    // Texture coordinate in tex space (0-1)
    #[format(R32G32_SFLOAT)]
    pub(crate) uv_texcoord: [f32; 2],
    // TODO design lighting, and add it to the shader
    #[format(R32_SFLOAT)]
    pub(crate) brightness: f32,
}
pub(crate) struct CubeGeometryDrawCall {
    pub(crate) models: VkChunkVertexData,
    pub(crate) model_matrix: Matrix4<f32>,
}

pub(crate) struct CubePipelineWrapper {
    solid_pipeline: Arc<GraphicsPipeline>,
    sparse_pipeline: Arc<GraphicsPipeline>,
    translucent_pipeline: Arc<GraphicsPipeline>,
    solid_descriptor: Arc<PersistentDescriptorSet>,
    sparse_descriptor: Arc<PersistentDescriptorSet>,
    translucent_descriptor: Arc<PersistentDescriptorSet>,
    bound_projection: Matrix4<f32>,
}
impl CubePipelineWrapper {
    fn check_frustum(&self, pass_data: &CubeGeometryDrawCall) -> bool {
        let transformation = self.bound_projection * pass_data.model_matrix;
        const CORNERS: [Vector4<f32>; 8] = [
            vec4(0., 0., 0., 1.),
            vec4(16., 0., 0., 1.),
            vec4(0., -16., 0., 1.),
            vec4(16., -16., 0., 1.),
            vec4(0., 0., 16., 1.),
            vec4(16., 0., 16., 1.),
            vec4(0., -16., 16., 1.),
            vec4(16., -16., 16., 1.),
        ];
        let mut ndc_min = vec3(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut ndc_max = vec3(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        for corner in CORNERS {
            let mut ndc = transformation * corner;
            ndc /= ndc.w;
            ndc_min = vec3(
                ndc_min.x.min(ndc.x),
                ndc_min.y.min(ndc.y),
                ndc_min.z.min(ndc.z),
            );
            ndc_max = vec3(
                ndc_max.x.max(ndc.x),
                ndc_max.y.max(ndc.y),
                ndc_max.z.max(ndc.z),
            );
        }
        overlaps(ndc_min.x, ndc_max.x, -1., 1.)
            && overlaps(ndc_min.y, ndc_max.y, -1., 1.)
            && overlaps(ndc_min.z, ndc_max.z, 0., 1.)
    }
}

#[inline]
fn overlaps(min1: f32, max1: f32, min2: f32, max2: f32) -> bool {
    (min1 <= min2 && max1 >= max2)
        || (min2 <= min1 && max2 >= max1)
        || (min1 <= max2 && min1 >= min2)
        || (min2 <= max1 && min2 >= min1)
}

/// Which render step we are rendering in this renderer.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockRenderPass {
    Opaque,
    Transparent,
    Translucent,
}

impl PipelineWrapper<CubeGeometryDrawCall, Matrix4<f32>> for CubePipelineWrapper {
    type PassIdentifier = BlockRenderPass;
    fn draw<L>(
        &mut self,
        builder: &mut CommandBufferBuilder<L>,
        draw_calls: &[CubeGeometryDrawCall],
        pass: BlockRenderPass,
    ) -> Result<()> {
        let _span = match pass {
            BlockRenderPass::Opaque => span!("draw opaque"),
            BlockRenderPass::Transparent => span!("draw transparent"),
            BlockRenderPass::Translucent => span!("draw translucent"),
        };
        let pipeline = match pass {
            BlockRenderPass::Opaque => self.solid_pipeline.clone(),
            BlockRenderPass::Transparent => self.sparse_pipeline.clone(),
            BlockRenderPass::Translucent => self.translucent_pipeline.clone(),
        };
        let layout = pipeline.layout().clone();
        builder.bind_pipeline_graphics(pipeline);
        let mut pre_frustum = 0;
        let mut post_frustum = 0;
        for call in draw_calls.iter() {
            let pass_data = match pass {
                BlockRenderPass::Opaque => &call.models.solid_opaque,
                BlockRenderPass::Transparent => &call.models.transparent,
                BlockRenderPass::Translucent => &call.models.translucent,
            };
            if let Some(pass_data) = pass_data {
                pre_frustum += 1;
                if self.check_frustum(call) {
                    post_frustum += 1;

                    let push_data: ModelMatrix = call.model_matrix.into();
                    builder
                        .push_constants(layout.clone(), 0, push_data)
                        .bind_vertex_buffers(0, pass_data.vtx.clone())
                        .bind_index_buffer(pass_data.idx.clone())
                        .draw_indexed(pass_data.idx.len().try_into()?, 1, 0, 0, 0)?;
                }
            }
        }
        plot!("total_calls", draw_calls.len() as f64);
        let draw_rate = pre_frustum as f64 / (draw_calls.len() as f64);
        let draw_rate_frustum = post_frustum as f64 / pre_frustum as f64;
        match pass {
            BlockRenderPass::Opaque => {
                plot!("opaque_rate", draw_rate);
                plot!("opaque_rate_frustum", draw_rate_frustum);
            }
            BlockRenderPass::Transparent => {
                plot!("transparent_rate", draw_rate);
            }
            BlockRenderPass::Translucent => {
                plot!("translucent_rate", draw_rate);
            }
        };
        Ok(())
    }

    fn bind<L>(
        &mut self,
        ctx: &VulkanContext,
        per_frame_config: Matrix4<f32>,
        command_buf_builder: &mut CommandBufferBuilder<L>,
        pass: BlockRenderPass,
    ) -> Result<()> {
        let _span = match pass {
            BlockRenderPass::Opaque => span!("bind opaque"),
            BlockRenderPass::Transparent => span!("bind transparent"),
            BlockRenderPass::Translucent => span!("bind translucent"),
        };
        self.bound_projection = per_frame_config;
        let layout = match pass {
            BlockRenderPass::Opaque => self.solid_pipeline.layout().clone(),
            BlockRenderPass::Transparent => self.sparse_pipeline.layout().clone(),
            BlockRenderPass::Translucent => self.translucent_pipeline.layout().clone(),
        };

        let per_frame_set_layout = layout
            .set_layouts()
            .get(1)
            .with_context(|| "Layout missing set 1")?;

        let uniform_buffer = Buffer::from_data(
            &ctx.memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            UniformData {
                vp_matrix: per_frame_config.into(),
            },
        )?;

        let per_frame_set = PersistentDescriptorSet::new(
            &ctx.descriptor_set_allocator,
            per_frame_set_layout.clone(),
            [WriteDescriptorSet::buffer(0, uniform_buffer)],
        )?;

        let descriptor = match pass {
            BlockRenderPass::Opaque => self.solid_descriptor.clone(),
            BlockRenderPass::Transparent => self.sparse_descriptor.clone(),
            BlockRenderPass::Translucent => self.translucent_descriptor.clone(),
        };
        command_buf_builder.bind_descriptor_sets(
            vulkano::pipeline::PipelineBindPoint::Graphics,
            layout,
            0,
            vec![descriptor, per_frame_set],
        );

        Ok(())
    }
}

pub(crate) struct CubePipelineProvider {
    device: Arc<Device>,
    vs_cube: Arc<ShaderModule>,
    fs_solid: Arc<ShaderModule>,
    fs_sparse: Arc<ShaderModule>,
}
impl CubePipelineProvider {
    pub(crate) fn new(device: Arc<Device>) -> Result<CubePipelineProvider> {
        let vs_cube = vert_3d::load_cube_geometry(device.clone())?;
        let fs_solid = frag_lighting::load(device.clone())?;
        let fs_sparse = frag_lighting_sparse::load(device.clone())?;
        Ok(CubePipelineProvider {
            device,
            vs_cube,
            fs_solid,
            fs_sparse,
        })
    }
}
impl PipelineProvider for CubePipelineProvider {
    fn make_pipeline(
        &self,
        ctx: &VulkanContext,
        config: &Texture2DHolder,
    ) -> Result<CubePipelineWrapper> {
        let supersampling = ctx.supersampling;
        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [
                ctx.viewport.dimensions[0] * supersampling.get() as f32,
                ctx.viewport.dimensions[1] * supersampling.get() as f32,
            ],
            depth_range: 0.0..1.0,
        };
        let default_pipeline = GraphicsPipeline::start()
            .vertex_input_state(CubeGeometryVertex::per_vertex())
            .vertex_shader(
                self.vs_cube
                    .entry_point("main")
                    .context("Could not find vertex shader entry point")?,
                (),
            )
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
            .rasterization_state(
                RasterizationState::default()
                    .front_face(
                        vulkano::pipeline::graphics::rasterization::FrontFace::CounterClockwise,
                    )
                    .cull_mode(vulkano::pipeline::graphics::rasterization::CullMode::Back),
            )
            .color_blend_state(ColorBlendState::default().blend_alpha())
            .depth_stencil_state(DepthStencilState::simple_depth_test())
            .render_pass(
                Subpass::from(ctx.pre_blit_pass.clone(), 0).context("Could not find subpass 0")?,
            );

        let solid_pipeline = default_pipeline
            .clone()
            .fragment_shader(
                self.fs_solid
                    .entry_point("main")
                    .with_context(|| "Couldn't find dense fragment shader entry point")?,
                (),
            )
            .build(self.device.clone())?;

        let sparse_pipeline = default_pipeline
            .clone()
            .fragment_shader(
                self.fs_sparse
                    .entry_point("main")
                    .with_context(|| "Couldn't find sparse fragment shader entry point")?,
                (),
            )
            .build(self.device.clone())?;

        let translucent_pipeline = default_pipeline
            .clone()
            .fragment_shader(
                self.fs_solid
                    .entry_point("main")
                    .with_context(|| "Couldn't find dense fragment shader entry point")?,
                (),
            )
            .depth_stencil_state(DepthStencilState {
                depth: Some(DepthState {
                    enable_dynamic: false,
                    compare_op: StateMode::Fixed(CompareOp::Less),
                    write_enable: StateMode::Fixed(false),
                }),
                depth_bounds: Default::default(),
                stencil: Default::default(),
            })
            .build(self.device.clone())?;

        let solid_descriptor = config.descriptor_set(&solid_pipeline, 0, 0)?;
        let sparse_descriptor = config.descriptor_set(&sparse_pipeline, 0, 0)?;
        let translucent_descriptor = config.descriptor_set(&translucent_pipeline, 0, 0)?;
        Ok(CubePipelineWrapper {
            solid_pipeline,
            sparse_pipeline,
            translucent_pipeline,
            solid_descriptor,
            sparse_descriptor,
            translucent_descriptor,
            bound_projection: Matrix4::zero(),
        })
    }

    type DrawCall = CubeGeometryDrawCall;
    type PerFrameConfig = Matrix4<f32>;
    type PipelineWrapperImpl = CubePipelineWrapper;
    type PerPipelineConfig<'a> = &'a Texture2DHolder;
}
