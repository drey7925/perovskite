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

use crate::client_state::settings::Supersampling;
use crate::vulkan::{
    shaders::vert_3d::ModelMatrix, CommandBufferBuilder, Texture2DHolder, VulkanContext,
    VulkanWindow,
};
use anyhow::{Context, Result};
use cgmath::Matrix4;
use smallvec::smallvec;
use std::collections::HashMap;
use std::sync::Arc;
use tracy_client::span;
use vulkano::buffer::BufferContents;
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
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    device::Device,
    memory::allocator::AllocationCreateInfo,
    pipeline::{
        graphics::{
            color_blend::ColorBlendState,
            depth_stencil::{DepthState, DepthStencilState},
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

use crate::vulkan::shaders::{
    frag_lighting,
    vert_3d::{self, UniformData},
    LiveRenderConfig, VkBufferGpu,
};

use super::SceneState;

#[derive(BufferContents, Vertex, Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub(crate) struct EntityVertex {
    /// Position, given relative to the origin of the chunk in world space.
    /// Not transformed into camera space via a view matrix yet
    #[format(R32G32B32_SFLOAT)]
    pub(crate) position: [f32; 3],

    /// Normal, given in world space
    #[format(R32G32B32_SFLOAT)]
    pub(crate) normal: [f32; 3],

    // Texture coordinate in tex space (0-1)
    #[format(R32G32_SFLOAT)]
    pub(crate) uv_texcoord: [f32; 2],
}

pub(crate) struct EntityGeometryDrawCall {
    pub(crate) model: VkBufferGpu<EntityVertex>,
    pub(crate) model_matrix: Matrix4<f32>,
}

pub(crate) struct EntityPipelineWrapper {
    pipeline: Arc<GraphicsPipeline>,
    descriptor: Arc<DescriptorSet>,
}

impl EntityPipelineWrapper {
    pub(crate) fn draw<L>(
        &mut self,
        ctx: &VulkanContext,
        builder: &mut CommandBufferBuilder<L>,
        per_frame_config: SceneState,
        draw_calls: Vec<EntityGeometryDrawCall>,
    ) -> Result<()> {
        let _span = span!("draw entities");

        let pipeline = self.pipeline.clone();
        let layout = pipeline.layout().clone();

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
                global_light_direction: per_frame_config.sun_direction.into(),
            },
        )?;

        let per_frame_set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            per_frame_set_layout.clone(),
            [WriteDescriptorSet::buffer(0, uniform_buffer)],
            [],
        )?;

        let descriptor = self.descriptor.clone();
        builder.bind_descriptor_sets(
            vulkano::pipeline::PipelineBindPoint::Graphics,
            layout.clone(),
            0,
            vec![descriptor, per_frame_set],
        )?;

        builder.bind_pipeline_graphics(pipeline)?;
        for call in draw_calls.into_iter() {
            let push_data: ModelMatrix = call.model_matrix.into();
            builder
                .push_constants(layout.clone(), 0, push_data)?
                .bind_vertex_buffers(0, call.model.vtx.clone())?
                .bind_index_buffer(call.model.idx.clone())?;
            unsafe {
                // TODO this needs validation
                builder.draw_indexed(call.model.idx.len().try_into()?, 1, 0, 0, 0)?;
            }
        }

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
        let vs_entity = vert_3d::load_entity_tentative(device.clone())?;
        let fs = frag_lighting::load(device.clone())?;
        Ok(EntityPipelineProvider {
            device,
            vs_entity,
            fs_sparse: fs,
        })
    }

    pub(crate) fn build_pipeline(
        &self,
        _ctx: &VulkanContext,
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
            .specialize(HashMap::from_iter([(0, false.into()), (1, false.into())]))?
            .entry_point("main")
            .context("Missing fragment shader")?;
        let vertex_input_state = EntityVertex::per_vertex().definition(&vs)?;
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
impl EntityPipelineProvider {
    pub(crate) fn make_pipeline(
        &self,
        wnd: &VulkanWindow,
        config: &Texture2DHolder,
        global_config: &LiveRenderConfig,
    ) -> Result<EntityPipelineWrapper> {
        self.build_pipeline(
            &wnd.vk_ctx,
            wnd.viewport.clone(),
            wnd.renderpasses.color_depth.clone(),
            config,
            global_config.supersampling,
        )
    }
}
