// Copyright 2025 drey7925
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
use smallvec::smallvec;
use std::collections::HashMap;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, ColorBlendAttachmentState, ColorBlendState, ColorComponents,
};
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::PrimitiveTopology;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, FrontFace, RasterizationState};
use vulkano::pipeline::graphics::subpass::PipelineSubpassType;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::shader::SpecializationConstant;
use vulkano::{
    device::Device,
    pipeline::{graphics::input_assembly::InputAssemblyState, GraphicsPipeline, Pipeline},
    render_pass::Subpass,
    shader::ShaderModule,
};

use crate::vulkan::far_geometry::WindingOrder;
use crate::vulkan::shaders::SceneState;
use crate::vulkan::{
    shaders::LiveRenderConfig, FramebufferAndLoadOpId, ImageId, LoadOp, VulkanWindow,
};
use crate::vulkan::{CommandBufferBuilder, Texture2DHolder, VulkanContext};

vulkano_shaders::shader! {
    shaders: {
        vert: {
            ty: "vertex",
            path: "src/vulkan/shaders/far_mesh.vert"
        },
        frag: {
            ty: "fragment",
            path: "src/vulkan/shaders/far_mesh.frag"
        }
    }
}

#[derive(BufferContents, Vertex, Clone, Copy, Debug)]
#[repr(C)]
pub(crate) struct FarMeshVertex {
    #[format(R32G32B32_SFLOAT)]
    pub(crate) position: [f32; 3],
    #[format(B8G8R8A8_UNORM)]
    pub(crate) top_color: [u8; 4],
    #[format(B8G8R8A8_UNORM)]
    pub(crate) side_color: [u8; 4],
    /// Normal, encoded in 15 bits
    #[format(R16_SINT)]
    pub(crate) normal: u16,
    #[format(R8_SNORM)]
    pub(crate) lod_orientation_bias: i8,
}

pub(crate) struct FarMeshDrawCall {
    pub(crate) model_matrix: Matrix4<f32>,
    pub(crate) vtx: Subbuffer<[FarMeshVertex]>,
    pub(crate) idx: Subbuffer<[u32]>,
    pub(crate) num_indices: u32,
    pub(crate) winding_order: WindingOrder,
    pub(crate) alpha: bool,
}

pub(crate) struct FarMeshPipelineWrapper {
    pub(crate) pipeline_cw_alpha: Arc<GraphicsPipeline>,
    pub(crate) pipeline_ccw_alpha: Arc<GraphicsPipeline>,
    pub(crate) pipeline_cw_no_alpha: Arc<GraphicsPipeline>,
    pub(crate) pipeline_ccw_no_alpha: Arc<GraphicsPipeline>,
    pub(crate) global_descriptor_set: Arc<DescriptorSet>,
}
impl FarMeshPipelineWrapper {
    pub(crate) fn draw<L>(
        &self,
        ctx: &VulkanContext,
        builder: &mut CommandBufferBuilder<L>,
        draw_calls: &[FarMeshDrawCall],
        scene_state: SceneState,
    ) -> Result<()> {
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
            FarMeshUniforms {
                global_brightness: scene_state.global_light_color.into(),
                global_light_dir: scene_state.sun_direction.into(),
                vp_matrix: scene_state.vp_matrix.into(),
            },
        )?;

        let layout = self.pipeline_cw_alpha.layout().clone();
        let set_layout = layout
            .set_layouts()
            .get(1)
            .context("far_mesh layout missing set 1")
            .unwrap();
        let per_frame_set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [WriteDescriptorSet::buffer(0, uniform_buffer)],
            [],
        )?;
        builder.bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            layout.clone(),
            0,
            vec![self.global_descriptor_set.clone(), per_frame_set],
        )?;

        for (winding_order, alpha, pipeline) in [
            // no alpha must be drawn first, so alpha can overdraw it.
            (WindingOrder::Clockwise, false, &self.pipeline_ccw_no_alpha),
            (
                WindingOrder::CounterClockwise,
                false,
                &self.pipeline_cw_no_alpha,
            ),
            (WindingOrder::Clockwise, true, &self.pipeline_ccw_alpha),
            (
                WindingOrder::CounterClockwise,
                true,
                &self.pipeline_cw_alpha,
            ),
        ] {
            builder.bind_pipeline_graphics(pipeline.clone())?;
            for draw_call in draw_calls {
                if (draw_call.winding_order, draw_call.alpha) != (winding_order, alpha) {
                    continue;
                }
                let push_data = FarMeshPushConstants {
                    model_matrix: draw_call.model_matrix.into(),
                };
                builder.push_constants(layout.clone(), 0, push_data)?;
                builder.bind_vertex_buffers(0, draw_call.vtx.clone())?;
                builder.bind_index_buffer(draw_call.idx.clone())?;

                unsafe { builder.draw_indexed(draw_call.num_indices, 1, 0, 0, 0)? };
            }
        }

        Ok(())
    }
}

pub(crate) struct FarMeshPipelineProvider {
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    blue_noise: Texture2DHolder,
}

impl FarMeshPipelineProvider {
    pub(crate) fn new(window: &VulkanWindow) -> Result<Self> {
        let vs = load_vert(window.vk_device.clone())?;
        let fs = load_frag(window.vk_device.clone())?;
        let blue_noise = super::make_blue_noise_image(&window)?;
        Ok(FarMeshPipelineProvider {
            device: window.vk_device.clone(),
            vs,
            fs,
            blue_noise,
        })
    }

    pub(crate) fn make_pipeline(
        &self,
        ctx: &VulkanWindow,
        global_config: &LiveRenderConfig,
    ) -> Result<FarMeshPipelineWrapper> {
        // Use ImageId::MainColor for far mesh rendering
        let subpass = Subpass::from(
            ctx.renderpasses
                .get_by_framebuffer_id(FramebufferAndLoadOpId {
                    color_attachments: [(ImageId::MainColor, LoadOp::Load)],
                    depth_stencil_attachment: Some((ImageId::MainDepthStencil, LoadOp::Load)),
                    input_attachments: [],
                })?,
            0,
        )
        .context("subpass 0 missing")?;

        let vs = self
            .vs
            .entry_point("main")
            .context("Missing vertex shader")?;
        let fs = self
            .fs
            .specialize(HashMap::from_iter([(
                0,
                SpecializationConstant::U32(global_config.supersampling.to_int()),
            )]))?
            .entry_point("main")
            .context("Missing fragment shader")?;

        let vertex_input_state = FarMeshVertex::per_vertex().definition(&vs)?;
        let stages = smallvec![
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        let layout = PipelineLayout::new(
            self.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(self.device.clone())?,
        )?;

        let make_pipeline = |front_face: FrontFace, attachment_blend: Option<AttachmentBlend>| {
            GraphicsPipelineCreateInfo {
                stages: stages.clone(),
                vertex_input_state: Some(vertex_input_state.clone()),
                input_assembly_state: Some(InputAssemblyState {
                    topology: PrimitiveTopology::TriangleStrip,
                    primitive_restart_enable: true,
                    ..Default::default()
                }),
                rasterization_state: Some(RasterizationState {
                    cull_mode: CullMode::Back,
                    front_face,
                    ..Default::default()
                }),
                multisample_state: Some(MultisampleState::default()),
                viewport_state: Some(
                    ImageId::MainColor.viewport_state(&ctx.viewport, *global_config),
                ),
                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(DepthState::simple()),
                    ..Default::default()
                }),
                color_blend_state: Some(ColorBlendState {
                    attachments: vec![ColorBlendAttachmentState {
                        blend: attachment_blend,
                        color_write_mask: ColorComponents::all(),
                        color_write_enable: true,
                    }],
                    ..Default::default()
                }),
                subpass: Some(PipelineSubpassType::BeginRenderPass(subpass.clone())),
                ..GraphicsPipelineCreateInfo::layout(layout.clone())
            }
        };

        let create_info_cw_alpha =
            make_pipeline(FrontFace::Clockwise, Some(AttachmentBlend::alpha()));
        let create_info_ccw_alpha =
            make_pipeline(FrontFace::CounterClockwise, Some(AttachmentBlend::alpha()));
        let create_info_cw_no_alpha = make_pipeline(FrontFace::Clockwise, None);
        let create_info_ccw_no_alpha = make_pipeline(FrontFace::CounterClockwise, None);

        let pipeline_cw_alpha =
            GraphicsPipeline::new(self.device.clone(), None, create_info_cw_alpha)?;
        let pipeline_ccw_alpha =
            GraphicsPipeline::new(self.device.clone(), None, create_info_ccw_alpha)?;
        let pipeline_cw_no_alpha =
            GraphicsPipeline::new(self.device.clone(), None, create_info_cw_no_alpha)?;
        let pipeline_ccw_no_alpha =
            GraphicsPipeline::new(self.device.clone(), None, create_info_ccw_no_alpha)?;

        let layout = pipeline_cw_alpha.layout().clone();
        let set_layout = layout
            .set_layouts()
            .get(0)
            .context("far_mesh layout missing set 0")
            .unwrap();
        let global_descriptor_set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [WriteDescriptorSet::image_view_sampler(
                0,
                self.blue_noise.image_view.clone(),
                self.blue_noise.sampler.clone(),
            )],
            [],
        )?;
        Ok(FarMeshPipelineWrapper {
            pipeline_cw_alpha,
            pipeline_ccw_alpha,
            pipeline_cw_no_alpha,
            pipeline_ccw_no_alpha,
            global_descriptor_set,
        })
    }
}
