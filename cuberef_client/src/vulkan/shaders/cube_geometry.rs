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
use cgmath::{Matrix4, Zero};
use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::Device,
    memory::allocator::{AllocationCreateInfo, MemoryUsage},
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            rasterization::RasterizationState,
            vertex_input::Vertex,
            viewport::{ViewportState}, color_blend::ColorBlendState,
        },
        GraphicsPipeline, Pipeline,
    },
    render_pass::{Subpass},
    shader::ShaderModule,
};

use crate::vulkan::{
    shaders::{frag_lighting, vert_3d::ModelMatrix},
    Texture2DHolder, VulkanContext, CommandBufferBuilder,
};

use crate::vulkan::shaders::{
    vert_3d::{self, UniformData},
    PipelineProvider, PipelineWrapper,
};

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
    pub(crate) vertex: Subbuffer<[CubeGeometryVertex]>,
    pub(crate) index: Subbuffer<[u32]>,
    pub(crate) model_matrix: Matrix4<f32>,
}

pub(crate) struct CubePipelineWrapper {
    pipeline: Arc<GraphicsPipeline>,
    texture_descriptor: Arc<PersistentDescriptorSet>,
    bound_projection: Matrix4<f32>
}

impl PipelineWrapper<CubeGeometryDrawCall, Matrix4<f32>> for CubePipelineWrapper {
    fn pipeline(&self) -> &GraphicsPipeline {
        self.pipeline.as_ref()
    }
    fn pipeline_arc(&self) -> Arc<GraphicsPipeline> {
        self.pipeline.clone()
    }

    fn draw(
        &self,
        builder: &mut CommandBufferBuilder,
        draw_calls: &[CubeGeometryDrawCall],
    ) -> Result<()> {
        let layout = self.pipeline.layout();

        builder.bind_pipeline_graphics(self.pipeline.clone());

        for call in draw_calls.iter() {
            let push_data: ModelMatrix = call.model_matrix.into();
            builder
                .push_constants(layout.clone(), 0, push_data)
                .bind_vertex_buffers(0, call.vertex.clone())
                .bind_index_buffer(call.index.clone())
                .draw_indexed(call.index.len().try_into().unwrap(), 1, 0, 0, 0)?;
        }

        Ok(())
    }

    fn bind(
        &mut self,
        ctx: &VulkanContext,
        per_frame_config: Matrix4<f32>,
        command_buf_builder: &mut CommandBufferBuilder,
    ) -> Result<()> {
        self.bound_projection = per_frame_config;
        let layout = self.pipeline.layout().clone();
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

        command_buf_builder.bind_descriptor_sets(
            vulkano::pipeline::PipelineBindPoint::Graphics,
            layout,
            0,
            vec![self.texture_descriptor.clone(), per_frame_set],
        );

        Ok(())
    }
}

pub(crate) struct CubePipelineProvider {
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
}
impl PipelineProvider for CubePipelineProvider {
    fn new(device: Arc<Device>) -> Result<CubePipelineProvider> {
        let vs = vert_3d::load_cube_geometry(device.clone())?;
        let fs = frag_lighting::load(device.clone())?;
        Ok(CubePipelineProvider { device, vs, fs })
    }
    fn make_pipeline(
        &self,
        ctx: &VulkanContext,
        config: Arc<Texture2DHolder>,
    ) -> Result<CubePipelineWrapper> {
        let pipeline = GraphicsPipeline::start()
            .vertex_input_state(CubeGeometryVertex::per_vertex())
            .vertex_shader(self.vs.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
                ctx.viewport.clone()
            ]))
            .fragment_shader(self.fs.entry_point("main").unwrap(), ())
            .depth_stencil_state(DepthStencilState::simple_depth_test())
            .rasterization_state(
                RasterizationState::default()
                    .front_face(
                        vulkano::pipeline::graphics::rasterization::FrontFace::CounterClockwise,
                    )
                    .cull_mode(vulkano::pipeline::graphics::rasterization::CullMode::Back),
            )
            .color_blend_state(ColorBlendState::default().blend_alpha())
            .render_pass(Subpass::from(ctx.render_pass.clone(), 0).unwrap())
            .build(self.device.clone())?;
        let texture_descriptor = config.descriptor_set(&pipeline, 0, 0)?;
        Ok(CubePipelineWrapper {
            pipeline,
            texture_descriptor,
            bound_projection: Matrix4::zero(),
        })
    }

    type DrawCall = CubeGeometryDrawCall;
    type PerFrameConfig = Matrix4<f32>;
    type PipelineWrapperImpl = CubePipelineWrapper;
    type PerPipelineConfig = Arc<Texture2DHolder>;
}
