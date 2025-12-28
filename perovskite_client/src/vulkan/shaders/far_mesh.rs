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
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::graphics::color_blend::{
    ColorBlendAttachmentState, ColorBlendState, ColorComponents,
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
use crate::vulkan::{CommandBufferBuilder, VulkanContext};

vulkano_shaders::shader! {
    shaders: {
        vert: {
            ty: "vertex",
            path: "src/vulkan/shaders/far_mesh_vert.glsl"
        },
        frag: {
            ty: "fragment",
            path: "src/vulkan/shaders/far_mesh_frag.glsl"
        }
    }
}

#[derive(BufferContents, Vertex, Clone, Copy, Debug)]
#[repr(C)]
pub(crate) struct FarMeshVertex {
    #[format(R32G32B32_SFLOAT)]
    pub(crate) position: [f32; 3],
    #[format(B8G8R8A8_UNORM)]
    pub(crate) color: [u8; 4],
}

pub(crate) struct FarMeshDrawCall {
    pub(crate) model_matrix: Matrix4<f32>,
    pub(crate) vtx: Subbuffer<[FarMeshVertex]>,
    pub(crate) idx: Subbuffer<[u32]>,
    pub(crate) num_indices: u32,
    pub(crate) winding_order: WindingOrder,
}

pub(crate) struct FarMeshPipelineWrapper {
    pub(crate) pipeline_cw: Arc<GraphicsPipeline>,
    pub(crate) pipeline_ccw: Arc<GraphicsPipeline>,
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
                vp_matrix: scene_state.vp_matrix.into(),
            },
        )?;

        let layout = self.pipeline_cw.layout().clone();
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
            1,
            vec![per_frame_set.clone()],
        )?;
        for (winding_order, pipeline) in [
            (WindingOrder::Clockwise, &self.pipeline_ccw),
            (WindingOrder::CounterClockwise, &self.pipeline_cw),
        ] {
            builder.bind_pipeline_graphics(pipeline.clone())?;
            for draw_call in draw_calls {
                if draw_call.winding_order != winding_order {
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
}

impl FarMeshPipelineProvider {
    pub(crate) fn new(device: Arc<Device>) -> Result<Self> {
        let vs = load_vert(device.clone())?;
        let fs = load_frag(device.clone())?;
        Ok(FarMeshPipelineProvider { device, vs, fs })
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

        let create_info_cw = GraphicsPipelineCreateInfo {
            stages,
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState {
                topology: PrimitiveTopology::TriangleStrip,
                primitive_restart_enable: true,
                ..Default::default()
            }),
            rasterization_state: Some(RasterizationState {
                cull_mode: CullMode::Back,
                front_face: FrontFace::Clockwise,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            viewport_state: Some(ImageId::MainColor.viewport_state(&ctx.viewport, *global_config)),
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState::simple()),
                ..Default::default()
            }),
            color_blend_state: Some(ColorBlendState {
                attachments: vec![ColorBlendAttachmentState {
                    blend: None,
                    color_write_mask: ColorComponents::all(),
                    color_write_enable: true,
                }],
                ..Default::default()
            }),
            subpass: Some(PipelineSubpassType::BeginRenderPass(subpass)),
            ..GraphicsPipelineCreateInfo::layout(layout)
        };

        let mut create_info_ccw = create_info_cw.clone();
        create_info_ccw
            .rasterization_state
            .as_mut()
            .unwrap()
            .front_face = FrontFace::CounterClockwise;

        let pipeline_cw = GraphicsPipeline::new(self.device.clone(), None, create_info_cw)?;
        let pipeline_ccw = GraphicsPipeline::new(self.device.clone(), None, create_info_ccw)?;

        Ok(FarMeshPipelineWrapper {
            pipeline_cw,
            pipeline_ccw,
        })
    }
}
