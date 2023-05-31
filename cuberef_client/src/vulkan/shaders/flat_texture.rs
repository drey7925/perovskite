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

use anyhow::{Context, Result};
use texture_packer::Rect;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::Device,
    memory::allocator::{AllocationCreateInfo, MemoryUsage},
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
    shaders::{frag_lighting, vert_2d, PipelineProvider, PipelineWrapper},
    Texture2DHolder, VulkanContext, CommandBufferBuilder,
};

use super::vert_2d::UniformData;

#[derive(BufferContents, Vertex, Copy, Clone, Debug)]
#[repr(C)]
struct FlatTextureVertex {
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
        let width = (tex_dimension.0) as f32;
        let height = (tex_dimension.1) as f32;

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
        scale: u32
    ) {
        let w = tex_coord.w * scale;
        let h = tex_coord.h * scale;
        self.rect(
            Rect {
                x: pos.0 - (w / 2),
                y: pos.1 - (h / 2),
                w,
                h,
            },
            tex_coord,
            tex_dimension
        );
    }

    pub(crate) fn build(self, ctx: &VulkanContext) -> Result<FlatTextureDrawCall> {
        Ok(FlatTextureDrawCall {
            vertex_buffer: Buffer::from_iter(
                &ctx.memory_allocator,
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    usage: MemoryUsage::Upload,
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
    texture_descriptor_set: Arc<PersistentDescriptorSet>,
}
impl PipelineWrapper<FlatTextureDrawCall, ()> for FlatTexPipelineWrapper {
    fn pipeline(&self) -> &GraphicsPipeline {
        self.pipeline.as_ref()
    }
    fn pipeline_arc(&self) -> Arc<GraphicsPipeline> {
        self.pipeline.clone()
    }

    fn draw(
        &self,
        builder: &mut CommandBufferBuilder,
        draw_calls: &[FlatTextureDrawCall],
    ) -> anyhow::Result<()> {
        builder.bind_pipeline_graphics(self.pipeline.clone());
        for call in draw_calls.iter() {
            builder
                .bind_vertex_buffers(0, call.vertex_buffer.clone())
                .draw(call.vertex_buffer.len() as u32, 1, 0, 0)?;
        }
        Ok(())
    }

    fn bind(
        &mut self,
        ctx: &crate::vulkan::VulkanContext,
        _per_frame_config: (),
        command_buf_builder: &mut CommandBufferBuilder,
    ) -> anyhow::Result<()> {
        let layout = self.pipeline.layout().clone();
        let per_frame_set_layout = layout
            .set_layouts()
            .get(1)
            .with_context(|| "Layout missing set 1")?;
        let per_frame_set = PersistentDescriptorSet::new(
            &ctx.descriptor_set_allocator,
            per_frame_set_layout.clone(),
            [WriteDescriptorSet::buffer(0, self.buffer.clone())],
        )?;
        command_buf_builder.bind_descriptor_sets(
            vulkano::pipeline::PipelineBindPoint::Graphics,
            layout,
            0,
            vec![self.texture_descriptor_set.clone(), per_frame_set],
        );
        Ok(())
    }
}

pub(crate) struct FlatTexPipelineProvider {
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
}
impl PipelineProvider for FlatTexPipelineProvider {
    type DrawCall = FlatTextureDrawCall;
    type PerFrameConfig = ();
    type PipelineWrapperImpl = FlatTexPipelineWrapper;
    type PerPipelineConfig = Arc<Texture2DHolder>;

    fn new(device: Arc<Device>) -> Result<Self> {
        let vs = vert_2d::load_flat_tex(device.clone())?;
        let fs = frag_lighting::load(device.clone())?;
        Ok(FlatTexPipelineProvider { device, vs, fs })
    }

    fn make_pipeline(
        &self,
        ctx: &VulkanContext,
        config: Arc<Texture2DHolder>,
    ) -> Result<Self::PipelineWrapperImpl> {
        let pipeline = GraphicsPipeline::start()
            .vertex_input_state(FlatTextureVertex::per_vertex())
            .vertex_shader(self.vs.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([ctx
                .viewport
                .clone()]))
            .fragment_shader(self.fs.entry_point("main").unwrap(), ())
            .depth_stencil_state(DepthStencilState::disabled())
            .rasterization_state(
                RasterizationState::default()
                    .front_face(
                        vulkano::pipeline::graphics::rasterization::FrontFace::CounterClockwise,
                    )
                    .cull_mode(vulkano::pipeline::graphics::rasterization::CullMode::None),
            )
            .color_blend_state(
                vulkano::pipeline::graphics::color_blend::ColorBlendState::default().blend_alpha(),
            )
            .render_pass(Subpass::from(ctx.render_pass.clone(), 0).unwrap())
            .build(self.device.clone())?;

        let buffer = Buffer::from_data(
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
                device_w_h: ctx.viewport.dimensions,
            },
        )
        .with_context(|| "Failed to make buffer for uniforms")?;

        let texture_descriptor_set = config.descriptor_set(&pipeline, 0, 0)?;
        Ok(FlatTexPipelineWrapper {
            pipeline,
            buffer,
            texture_descriptor_set,
        })
    }
}
