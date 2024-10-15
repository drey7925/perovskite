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

use crate::game_state::settings::Supersampling;
use anyhow::{Context, Result};
use cgmath::Matrix4;
use smallvec::smallvec;
use std::sync::Arc;
use tracy_client::span;
use vulkano::memory::allocator::MemoryTypeFilter;
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, ColorBlendAttachmentState, ColorComponents,
};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, FrontFace};
use vulkano::pipeline::graphics::subpass::PipelineSubpassType;
use vulkano::pipeline::graphics::vertex_input::VertexDefinition;
use vulkano::pipeline::graphics::viewport::Scissor;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{PipelineCreateFlags, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::Device,
    memory::allocator::AllocationCreateInfo,
    pipeline::{
        graphics::{
            color_blend::ColorBlendState,
            depth_stencil::{CompareOp, DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            rasterization::RasterizationState,
            vertex_input::Vertex,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline,
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
    LiveRenderConfig, PipelineProvider, PipelineWrapper,
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
        _pass: (),
    ) -> Result<()> {
        let _span = span!("draw entities");
        clog!("draw  entities");
        let pipeline = self.pipeline.clone();
        let layout = pipeline.layout().clone();
        builder.bind_pipeline_graphics(pipeline)?;
        for call in draw_calls.into_iter() {
            let push_data: ModelMatrix = call.model_matrix.into();
            builder
                .push_constants(layout.clone(), 0, push_data)?
                .bind_vertex_buffers(0, call.model.vtx.clone())?
                .bind_index_buffer(call.model.idx.clone())?
                .draw_indexed(call.model.idx.len().try_into()?, 1, 0, 0, 0)?;
        }
        clog!("draw entities done");
        Ok(())
    }

    fn bind<L>(
        &mut self,
        ctx: &VulkanContext,
        per_frame_config: SceneState,
        command_buf_builder: &mut CommandBufferBuilder<L>,
        _pass: (),
    ) -> Result<()> {
        clog!("bind entities");
        let _span = span!("bind entities");
        let layout = self.pipeline.layout().clone();

        let per_frame_set_layout = layout
            .set_layouts()
            .get(1)
            .with_context(|| "Layout missing set 1")?;

        let uniform_buffer = Buffer::from_data(
            ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                    | MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            UniformData {
                vp_matrix: per_frame_config.vp_matrix.into(),
                plant_wave_vector: [0.0, 0.0].into(),
                global_brightness_color: per_frame_config.global_light_color.into(),
                global_light_direction: per_frame_config.global_light_direction.into(),
            },
        )?;
        clog!("uniform uploaded");
        let per_frame_set = PersistentDescriptorSet::new(
            &ctx.descriptor_set_allocator,
            per_frame_set_layout.clone(),
            [WriteDescriptorSet::buffer(0, uniform_buffer)],
            [],
        )?;
        clog!("pfs");
        let descriptor = self.descriptor.clone();
        command_buf_builder.bind_descriptor_sets(
            vulkano::pipeline::PipelineBindPoint::Graphics,
            layout,
            0,
            vec![descriptor, per_frame_set],
        )?;
        clog!("bind entities done");
        Ok(())
    }
}

pub(crate) struct EntityPipelineProvider {
    device: Arc<Device>,
    vs_entity: Arc<ShaderModule>,
    fs_sparse: Arc<ShaderModule>,
}
impl EntityPipelineProvider {
    pub(crate) fn new(device: Arc<Device>) -> Result<EntityPipelineProvider> {
        let vs_entity = vert_3d::load_cube_geometry(device.clone())?;
        let fs = frag_lighting_sparse::load(device.clone())?;
        Ok(EntityPipelineProvider {
            device,
            vs_entity,
            fs_sparse: fs,
        })
    }

    pub(crate) fn build_pipeline(
        &self,
        ctx: &VulkanContext,
        viewport: Viewport,
        render_pass: Arc<RenderPass>,
        tex: &Texture2DHolder,
        supersampling: Supersampling,
    ) -> Result<EntityPipelineWrapper> {
        let vs = self
            .vs_entity
            .entry_point("main")
            .context("Missing vertex shader")?;
        let fs_sparse = self
            .fs_sparse
            .entry_point("main")
            .context("Missing fragment shader")?;
        let vertex_input_state =
            CubeGeometryVertex::per_vertex().definition(&vs.info().input_interface)?;
        let stages_sparse = smallvec![
            PipelineShaderStageCreateInfo::new(vs.clone()),
            PipelineShaderStageCreateInfo::new(fs_sparse),
        ];
        let layout_eparse = PipelineLayout::new(
            self.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages_sparse)
                .into_pipeline_layout_create_info(self.device.clone())?,
        )?;

        let sparse_pipeline_info = GraphicsPipelineCreateInfo {
            flags: PipelineCreateFlags::ALLOW_DERIVATIVES,
            stages: stages_sparse,
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            rasterization_state: Some(RasterizationState {
                cull_mode: CullMode::Back,
                front_face: FrontFace::CounterClockwise,
                ..Default::default()
            }),
            // TODO multisample state later when we have MSAA
            multisample_state: Some(MultisampleState::default()),
            viewport_state: Some(ViewportState {
                viewports: smallvec![Viewport {
                    offset: [0.0, 0.0],
                    depth_range: 0.0..=1.0,
                    extent: [
                        viewport.extent[0] * supersampling.to_float(),
                        viewport.extent[1] * supersampling.to_float()
                    ],
                }],
                scissors: smallvec![Scissor {
                    offset: [0, 0],
                    extent: [
                        viewport.extent[0] as u32 * supersampling.to_int(),
                        viewport.extent[1] as u32 * supersampling.to_int()
                    ],
                }],
                ..Default::default()
            }),
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState::simple()),
                ..Default::default()
            }),
            color_blend_state: Some(ColorBlendState {
                attachments: vec![ColorBlendAttachmentState {
                    blend: Some(AttachmentBlend::alpha()),
                    color_write_mask: ColorComponents::all(),
                    color_write_enable: true,
                }],
                ..Default::default()
            }),
            subpass: Some(PipelineSubpassType::BeginRenderPass(
                Subpass::from(render_pass.clone(), 0).context("Missing subpass")?,
            )),
            ..GraphicsPipelineCreateInfo::layout(layout_eparse.clone())
        };
        let pipeline = GraphicsPipeline::new(self.device.clone(), None, sparse_pipeline_info)?;

        let solid_descriptor = tex.descriptor_set(&pipeline, 0, 0)?;
        Ok(EntityPipelineWrapper {
            pipeline,
            descriptor: solid_descriptor,
        })
    }
}
impl PipelineProvider for EntityPipelineProvider {
    fn make_pipeline(
        &self,
        wnd: &VulkanWindow,
        config: &Texture2DHolder,
        global_config: &LiveRenderConfig,
    ) -> Result<EntityPipelineWrapper> {
        self.build_pipeline(
            &wnd.vk_ctx,
            wnd.viewport.clone(),
            wnd.ssaa_render_pass.clone(),
            config,
            global_config.supersampling,
        )
    }

    type DrawCall<'a> = Vec<EntityGeometryDrawCall>;
    type PerFrameConfig = SceneState;
    type PipelineWrapperImpl = EntityPipelineWrapper;
    type PerPipelineConfig<'a> = &'a Texture2DHolder;
}
