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
use cgmath::{Matrix4, Rad, Angle};
use std::{sync::Arc, time::Instant};
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
            viewport::ViewportState,
        },
        GraphicsPipeline, Pipeline, StateMode,
    },
    render_pass::Subpass,
    shader::ShaderModule,
};

use crate::{
    block_renderer::VkChunkVertexData,
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
    /// Position, given relative to the origin of the chunk.
    /// Not transformed into camera space via a view matrix yet
    #[format(R32G32B32_SFLOAT)]
    pub(crate) position: [f32; 3],

    // Texture coordinate in tex space (0-1)
    #[format(R32G32_SFLOAT)]
    pub(crate) uv_texcoord: [f32; 2],

    // The local brightness (from nearby sources, unchanging as the global lighting varies)
    #[format(R32_SFLOAT)]
    pub(crate) brightness: f32,

    // How much the global brightness should affect the brightness of this vertex
    #[format(R32_SFLOAT)]
    pub(crate) global_brightness_contribution: f32,

    // How much this vertex should wave with wavy input
    #[format(R32_SFLOAT)]
    pub(crate) wave_horizontal: f32,
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
    start_time: Instant
}
const PLANT_WAVE_FREQUENCY_HZ: f64 = 0.25;
impl CubePipelineWrapper {
    fn get_plant_wave_vector(&self) -> [f32; 2] {
        // f64 has enough bits of precision that we shouldn't worry about floating-point error here.
        let time = self.start_time.elapsed().as_secs_f64();
        let sin_cos = Rad(PLANT_WAVE_FREQUENCY_HZ * time * 2.0 * std::f64::consts::PI).sin_cos();
        [sin_cos.0 as f32, sin_cos.1 as f32]
    }
}

/// Which render step we are rendering in this renderer.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockRenderPass {
    Opaque,
    Transparent,
    Translucent,
}

impl PipelineWrapper<&mut [CubeGeometryDrawCall], Matrix4<f32>> for CubePipelineWrapper {
    type PassIdentifier = BlockRenderPass;
    fn draw<L>(
        &mut self,
        builder: &mut CommandBufferBuilder<L>,
        draw_calls: &mut [CubeGeometryDrawCall],
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
        let mut effective_calls = 0;
        for call in draw_calls.iter_mut() {
            let pass_data = match pass {
                BlockRenderPass::Opaque => call.models.solid_opaque.take(),
                BlockRenderPass::Transparent => call.models.transparent.take(),
                BlockRenderPass::Translucent => call.models.translucent.take(),
            };
            if let Some(pass_data) = pass_data {
                effective_calls += 1;

                let push_data: ModelMatrix = call.model_matrix.into();
                builder
                    .push_constants(layout.clone(), 0, push_data)
                    .bind_vertex_buffers(0, pass_data.vtx.clone())
                    .bind_index_buffer(pass_data.idx.clone())
                    .draw_indexed(pass_data.idx.len().try_into()?, 1, 0, 0, 0)?;
            }
        }
        plot!("total_calls", draw_calls.len() as f64);
        let draw_rate = effective_calls as f64 / (draw_calls.len() as f64);
        match pass {
            BlockRenderPass::Opaque => {
                plot!("opaque_rate", draw_rate);
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
                plant_wave_vector: self.get_plant_wave_vector(),
                // TODO set this
                global_brightness: 1.0,
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
        let default_pipeline = GraphicsPipeline::start()
            .vertex_input_state(CubeGeometryVertex::per_vertex())
            .vertex_shader(
                self.vs_cube
                    .entry_point("main")
                    .context("Could not find vertex shader entry point")?,
                (),
            )
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([ctx
                .viewport
                .clone()]))
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
                Subpass::from(ctx.render_pass.clone(), 0).context("Could not find subpass 0")?,
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
            start_time: Instant::now(),
        })
    }

    type DrawCall<'a> = &'a mut [CubeGeometryDrawCall];
    type PerFrameConfig = Matrix4<f32>;
    type PipelineWrapperImpl = CubePipelineWrapper;
    type PerPipelineConfig<'a> = &'a Texture2DHolder;
}
