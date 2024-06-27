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
use cgmath::Matrix4;
use std::sync::Arc;
use tracy_client::span;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
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
    render_pass::{RenderPass, Subpass},
    shader::ShaderModule,
};

use crate::vulkan::{
    block_renderer::VkCgvBufferGpu, shaders::vert_3d::ModelMatrix, CommandBufferBuilder,
    Texture2DHolder, VulkanContext, VulkanWindow,
};

use crate::vulkan::shaders::{
    vert_3d::{self, UniformData},
    PipelineProvider, PipelineWrapper,
};

use super::{cube_geometry::CubeGeometryVertex, frag_lighting_sparse, SceneState};

pub(crate) struct EntityGeometryDrawCall {
    pub(crate) model: VkCgvBufferGpu,
    pub(crate) model_matrix: Matrix4<f32>,
}

pub(crate) struct EntityPipelineWrapper {
    pipeline: Arc<GraphicsPipeline>,
    descriptor: Arc<PersistentDescriptorSet>,
}

impl PipelineWrapper<Vec<EntityGeometryDrawCall>, SceneState> for EntityPipelineWrapper {
    type PassIdentifier = ();
    fn draw<L>(
        &mut self,
        builder: &mut CommandBufferBuilder<L>,
        draw_calls: Vec<EntityGeometryDrawCall>,
        pass: (),
    ) -> Result<()> {
        let _span = span!("draw entities");
        let pipeline = self.pipeline.clone();
        let layout = pipeline.layout().clone();
        builder.bind_pipeline_graphics(pipeline);
        for call in draw_calls.into_iter() {
            let push_data: ModelMatrix = call.model_matrix.into();
            builder
                .push_constants(layout.clone(), 0, push_data)
                .bind_vertex_buffers(0, call.model.vtx.clone())
                .bind_index_buffer(call.model.idx.clone())
                .draw_indexed(call.model.idx.len().try_into()?, 1, 0, 0, 0)?;
        }

        Ok(())
    }

    fn bind<L>(
        &mut self,
        ctx: &VulkanContext,
        per_frame_config: SceneState,
        command_buf_builder: &mut CommandBufferBuilder<L>,
        _pass: (),
    ) -> Result<()> {
        let _span = span!("bind entities");
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
                vp_matrix: per_frame_config.vp_matrix.into(),
                plant_wave_vector: [0.0, 0.0].into(),
                global_brightness_color: per_frame_config.global_light_color.into(),
                global_light_direction: per_frame_config.global_light_direction.into(),
            },
        )?;

        let per_frame_set = PersistentDescriptorSet::new(
            &ctx.descriptor_set_allocator,
            per_frame_set_layout.clone(),
            [WriteDescriptorSet::buffer(0, uniform_buffer)],
        )?;

        let descriptor = self.descriptor.clone();
        command_buf_builder.bind_descriptor_sets(
            vulkano::pipeline::PipelineBindPoint::Graphics,
            layout,
            0,
            vec![descriptor, per_frame_set],
        );

        Ok(())
    }
}

pub(crate) struct EntityPipelineProvider {
    device: Arc<Device>,
    vs_entity: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
}
impl EntityPipelineProvider {
    pub(crate) fn new(device: Arc<Device>) -> Result<EntityPipelineProvider> {
        let vs_entity = vert_3d::load_cube_geometry(device.clone())?;
        let fs = frag_lighting_sparse::load(device.clone())?;
        Ok(EntityPipelineProvider {
            device,
            vs_entity,
            fs,
        })
    }

    pub(crate) fn build_pipeline(
        &self,
        _ctx: &VulkanContext,
        viewport: Viewport,
        render_pass: Arc<RenderPass>,
        tex: &Texture2DHolder,
    ) -> Result<EntityPipelineWrapper> {
        let default_pipeline = GraphicsPipeline::start()
            .vertex_input_state(CubeGeometryVertex::per_vertex())
            .vertex_shader(
                self.vs_entity
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
                Subpass::from(render_pass.clone(), 0).context("Could not find subpass 0")?,
            );

        let solid_pipeline = default_pipeline
            .clone()
            .fragment_shader(
                self.fs
                    .entry_point("main")
                    .with_context(|| "Couldn't find dense fragment shader entry point")?,
                (),
            )
            .build(self.device.clone())?;

        let sparse_pipeline = default_pipeline
            .clone()
            .fragment_shader(
                self.fs
                    .entry_point("main")
                    .with_context(|| "Couldn't find sparse fragment shader entry point")?,
                (),
            )
            .build(self.device.clone())?;

        let translucent_pipeline = default_pipeline
            .clone()
            .fragment_shader(
                self.fs
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

        let solid_descriptor = tex.descriptor_set(&solid_pipeline, 0, 0)?;
        let sparse_descriptor = tex.descriptor_set(&sparse_pipeline, 0, 0)?;
        let translucent_descriptor = tex.descriptor_set(&translucent_pipeline, 0, 0)?;
        Ok(EntityPipelineWrapper {
            pipeline: solid_pipeline,
            descriptor: solid_descriptor,
        })
    }
}
impl PipelineProvider for EntityPipelineProvider {
    fn make_pipeline(
        &self,
        wnd: &VulkanWindow,
        config: &Texture2DHolder,
    ) -> Result<EntityPipelineWrapper> {
        self.build_pipeline(
            &wnd.vk_ctx,
            wnd.viewport.clone(),
            wnd.render_pass.clone(),
            config,
        )
    }

    type DrawCall<'a> = Vec<EntityGeometryDrawCall>;
    type PerFrameConfig = SceneState;
    type PipelineWrapperImpl = EntityPipelineWrapper;
    type PerPipelineConfig<'a> = &'a Texture2DHolder;
}
