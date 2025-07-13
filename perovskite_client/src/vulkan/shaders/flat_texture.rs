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

use std::sync::Arc;

use crate::client_state::settings::Supersampling;
use anyhow::{Context, Result};
use smallvec::smallvec;
use texture_packer::Rect;
use tinyvec::array_vec;
use tracy_client::span;
use vulkano::memory::allocator::MemoryTypeFilter;
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, ColorBlendAttachmentState, ColorBlendState, ColorComponents,
};
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, FrontFace};
use vulkano::pipeline::graphics::subpass::PipelineSubpassType;
use vulkano::pipeline::graphics::vertex_input::VertexDefinition;
use vulkano::pipeline::graphics::viewport::{Scissor, Viewport};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    device::Device,
    memory::allocator::AllocationCreateInfo,
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState, input_assembly::InputAssemblyState,
            rasterization::RasterizationState, vertex_input::Vertex, viewport::ViewportState,
        },
        GraphicsPipeline, Pipeline,
    },
    render_pass::Subpass,
    shader::ShaderModule,
};

use crate::vulkan::{
    shaders::{frag_simple, vert_2d, vert_2d::UniformData, LiveRenderConfig},
    CommandBufferBuilder, FramebufferAndLoadOpId, ImageId, LoadOp, Texture2DHolder, VulkanContext,
    VulkanWindow,
};

#[derive(BufferContents, Vertex, Copy, Clone, Debug)]
#[repr(C)]
pub(crate) struct FlatTextureVertex {
    #[format(R32G32_SFLOAT)]
    /// Position in pixel space
    pub(crate) position: [f32; 2],
    // Texture coordinate in tex space (0-1)
    #[format(R32G32_SFLOAT)]
    pub(crate) uv_texcoord: [f32; 2],
}

#[derive(Clone)]
pub(crate) struct FlatTextureDrawCall {
    vertex_buffer: Subbuffer<[FlatTextureVertex]>,
}

impl FlatTextureDrawCall {
    pub(crate) fn vertex_buffer(&self) -> &Subbuffer<[FlatTextureVertex]> {
        &self.vertex_buffer
    }
}

pub(crate) struct FlatTextureDrawBuilder {
    vertices: Vec<FlatTextureVertex>,
}
impl FlatTextureDrawBuilder {
    pub(crate) fn new() -> FlatTextureDrawBuilder {
        FlatTextureDrawBuilder {
            vertices: Vec::new(),
        }
    }
    pub(crate) fn rect(&mut self, screen_coord: Rect, tex_coord: Rect, tex_dimension: (u32, u32)) {
        let width = tex_dimension.0 as f32;
        let height = tex_dimension.1 as f32;

        let bl = FlatTextureVertex {
            position: [screen_coord.left() as f32, screen_coord.bottom() as f32],
            uv_texcoord: [
                tex_coord.left() as f32 / width,
                (tex_coord.bottom() + 1) as f32 / height,
            ],
        };
        let br = FlatTextureVertex {
            position: [screen_coord.right() as f32, screen_coord.bottom() as f32],
            uv_texcoord: [
                (tex_coord.right() + 1) as f32 / width,
                (tex_coord.bottom() + 1) as f32 / height,
            ],
        };
        let tl = FlatTextureVertex {
            position: [screen_coord.left() as f32, screen_coord.top() as f32],
            uv_texcoord: [
                tex_coord.left() as f32 / width,
                tex_coord.top() as f32 / height,
            ],
        };
        let tr = FlatTextureVertex {
            position: [screen_coord.right() as f32, screen_coord.top() as f32],
            uv_texcoord: [
                (tex_coord.right() + 1) as f32 / width,
                tex_coord.top() as f32 / height,
            ],
        };
        self.vertices.append(&mut vec![bl, br, tr, bl, tr, tl]);
    }
    pub(crate) fn centered_rect(
        &mut self,
        pos: (u32, u32),
        tex_coord: Rect,
        tex_dimension: (u32, u32),
        scale: u32,
    ) {
        let w = tex_coord.w * scale;
        let h = tex_coord.h * scale;
        self.rect(
            Rect {
                x: pos.0.saturating_sub(w / 2),
                y: pos.1.saturating_sub(h / 2),
                w,
                h,
            },
            tex_coord,
            tex_dimension,
        );
    }

    pub(crate) fn build(self, ctx: &VulkanContext) -> Result<FlatTextureDrawCall> {
        Ok(FlatTextureDrawCall {
            vertex_buffer: Buffer::from_iter(
                ctx.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                        | MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                self.vertices.into_iter(),
            )?,
        })
    }
}

pub(crate) struct FlatTexPipelineWrapper {
    pipeline: Arc<GraphicsPipeline>,
    buffer: Subbuffer<UniformData>,
    texture_descriptor_set: Arc<DescriptorSet>,
}
impl FlatTexPipelineWrapper {
    pub(crate) fn draw<L>(
        &mut self,
        builder: &mut CommandBufferBuilder<L>,
        calls: &[FlatTextureDrawCall],
    ) -> Result<()> {
        let _span = span!("Draw flat graphics");
        builder.bind_pipeline_graphics(self.pipeline.clone())?;
        for call in calls {
            builder.bind_vertex_buffers(0, call.vertex_buffer.clone())?;
            unsafe {
                builder.draw(call.vertex_buffer.len() as u32, 1, 0, 0)?;
            }
        }
        Ok(())
    }

    pub(crate) fn bind<L>(
        &mut self,
        ctx: &VulkanContext,
        command_buf_builder: &mut CommandBufferBuilder<L>,
    ) -> Result<()> {
        let layout = self.pipeline.layout().clone();
        let per_frame_set_layout = layout
            .set_layouts()
            .get(1)
            .with_context(|| "Layout missing set 1")?;
        let per_frame_set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            per_frame_set_layout.clone(),
            [WriteDescriptorSet::buffer(0, self.buffer.clone())],
            [],
        )?;
        command_buf_builder.bind_descriptor_sets(
            vulkano::pipeline::PipelineBindPoint::Graphics,
            layout,
            0,
            vec![self.texture_descriptor_set.clone(), per_frame_set],
        )?;
        Ok(())
    }
}

pub(crate) struct FlatTexPipelineProvider {
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
}
impl FlatTexPipelineProvider {
    pub(crate) fn new(device: Arc<Device>) -> Result<Self> {
        let vs = vert_2d::load_flat_tex(device.clone())?;
        let fs = frag_simple::load(device.clone())?;
        Ok(FlatTexPipelineProvider { device, vs, fs })
    }
}
pub(crate) struct FlatPipelineConfig<'a> {
    pub(crate) atlas: &'a Texture2DHolder,
    pub(crate) image_id: ImageId,
}

impl FlatTexPipelineProvider {
    pub(crate) fn make_pipeline(
        &self,
        ctx: &VulkanWindow,
        config: FlatPipelineConfig<'_>,
        global_config: &LiveRenderConfig,
    ) -> Result<FlatTexPipelineWrapper> {
        let FlatPipelineConfig { atlas, image_id } = config;

        let subpass = Subpass::from(
            ctx.renderpasses
                .get_by_framebuffer_id(FramebufferAndLoadOpId {
                    color_attachments: [(image_id, LoadOp::Load)],
                    depth_stencil_attachment: None,
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
        let vertex_input_state = FlatTextureVertex::per_vertex().definition(&vs)?;
        let stages = smallvec![
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];
        let layout = PipelineLayout::new(
            self.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(self.device.clone())?,
        )?;

        let pipeline = GraphicsPipeline::new(
            self.device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages,
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                rasterization_state: Some(RasterizationState {
                    cull_mode: CullMode::Back,
                    front_face: FrontFace::CounterClockwise,
                    ..Default::default()
                }),
                multisample_state: Some(MultisampleState::default()),
                viewport_state: Some(image_id.viewport_state(&ctx.viewport, *global_config)),
                depth_stencil_state: None,
                color_blend_state: Some(ColorBlendState {
                    attachments: vec![ColorBlendAttachmentState {
                        blend: Some(AttachmentBlend::alpha()),
                        color_write_mask: ColorComponents::all(),
                        color_write_enable: true,
                    }],
                    ..Default::default()
                }),
                subpass: Some(PipelineSubpassType::BeginRenderPass(subpass)),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )?;
        // TODO turn this into push constants
        let buffer = Buffer::from_data(
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
                device_w_h: ctx.viewport.extent,
            },
        )
        .with_context(|| "Failed to make buffer for uniforms")?;

        let texture_descriptor_set = atlas.descriptor_set(&pipeline, 0, 0)?;
        Ok(FlatTexPipelineWrapper {
            pipeline,
            buffer,
            texture_descriptor_set,
        })
    }
}
