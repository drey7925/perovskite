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
    block_renderer::VkChunkVertexDataGpu,
    shaders::{frag_lighting, vert_3d::ModelMatrix},
    CommandBufferBuilder, Texture2DHolder, VulkanContext, VulkanWindow,
};
use anyhow::{ensure, Context, Result};
use cgmath::{Angle, Matrix4, Rad};
use smallvec::smallvec;
use std::{sync::Arc, time::Instant};
use tracy_client::{plot, span};
use vulkano::descriptor_set::DescriptorSet;
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
use vulkano::pipeline::{PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    descriptor_set::WriteDescriptorSet,
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
    DeviceSize,
};

use crate::vulkan::shaders::{
    vert_3d::{self, UniformData},
    LiveRenderConfig, PipelineProvider, PipelineWrapper,
};

use super::{frag_lighting_sparse, SceneState};

#[derive(BufferContents, Vertex, Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub(crate) struct CubeGeometryVertex {
    /// Position, given relative to the origin of the chunk in world space.
    /// Not transformed into camera space via a view matrix yet
    #[format(R32G32B32_SFLOAT)]
    pub(crate) position: [f32; 3],

    // Texture coordinate in tex space (0-1)
    #[format(R32G32_SFLOAT)]
    pub(crate) uv_texcoord: [f32; 2],

    /// Normal, given in world space
    #[format(R8_UINT)]
    pub(crate) normal: u8,

    // The local brightness (from nearby sources, unchanging as the global lighting varies)
    #[format(R8_UNORM)]
    pub(crate) brightness: u8,

    // How much the global brightness should affect the brightness of this vertex
    #[format(R8_UNORM)]
    pub(crate) global_brightness_contribution: u8,

    // How much this vertex should wave with wavy input
    #[format(R8_UNORM)]
    pub(crate) wave_horizontal: u8,
}
pub(crate) struct CubeGeometryDrawCall {
    pub(crate) models: VkChunkVertexDataGpu,
    pub(crate) model_matrix: Matrix4<f32>,
}

pub(crate) struct CubePipelineWrapper {
    solid_pipeline: Arc<GraphicsPipeline>,
    sparse_pipeline: Arc<GraphicsPipeline>,
    translucent_pipeline: Arc<GraphicsPipeline>,
    solid_descriptor: Arc<DescriptorSet>,
    sparse_descriptor: Arc<DescriptorSet>,
    translucent_descriptor: Arc<DescriptorSet>,
    start_time: Instant,
    max_draw_indexed_index_value: u32,
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

/// Which render step we are rendering now.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockRenderPass {
    Opaque,
    Transparent,
    Translucent,
}

impl PipelineWrapper<&mut [CubeGeometryDrawCall], SceneState> for CubePipelineWrapper {
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
        builder.bind_pipeline_graphics(pipeline)?;
        let mut effective_calls = 0;
        for call in draw_calls.iter_mut() {
            let pass_data = match pass {
                BlockRenderPass::Opaque => call.models.opaque.take(),
                BlockRenderPass::Transparent => call.models.transparent.take(),
                BlockRenderPass::Translucent => call.models.translucent.take(),
            };
            if let Some(pass_data) = pass_data {
                effective_calls += 1;

                let push_data: ModelMatrix = call.model_matrix.into();
                builder
                    .push_constants(layout.clone(), 0, push_data)?
                    .bind_vertex_buffers(0, pass_data.vtx.clone())?
                    .bind_index_buffer(pass_data.idx.clone())?;
                ensure!(pass_data.vtx.len() < self.max_draw_indexed_index_value as DeviceSize);
                unsafe {
                    // Safety:
                    // Every vertex number that is retrieved from the index buffer must fall within the range of the bound vertex-rate vertex buffers.
                    //   - Assured by block_renderer, which generates vertices and corresponding indices one-triangle-at-a-time.
                    // Every vertex number that is retrieved from the index buffer, if it is not the special primitive restart value, must be no greater than the max_draw_indexed_index_value device limit.
                    //   - Verified above
                    // If a descriptor set binding was created with DescriptorBindingFlags::PARTIALLY_BOUND, then if the shader accesses a descriptor in that binding, the descriptor must be initialized and contain a valid resource.
                    //   - N/A, we don't set PARTIALLY_BOUND
                    // Shader safety: We rely on "Vulkano will validate many of these requirements, but it is only able to do so when the resources involved are statically known."
                    //   - TODO: validate these more closely.
                    builder.draw_indexed(pass_data.idx.len().try_into()?, 1, 0, 0, 0)?;
                }
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
        per_frame_config: SceneState,
        command_buf_builder: &mut CommandBufferBuilder<L>,
        pass: BlockRenderPass,
    ) -> Result<()> {
        let _span = match pass {
            BlockRenderPass::Opaque => span!("bind opaque"),
            BlockRenderPass::Transparent => span!("bind transparent"),
            BlockRenderPass::Translucent => span!("bind translucent"),
        };
        let pipeline = match pass {
            BlockRenderPass::Opaque => self.solid_pipeline.clone(),
            BlockRenderPass::Transparent => self.sparse_pipeline.clone(),
            BlockRenderPass::Translucent => self.translucent_pipeline.clone(),
        };
        let layout = pipeline.layout().clone();
        command_buf_builder.bind_pipeline_graphics(pipeline)?;
        let per_frame_set_layout = layout
            .set_layouts()
            .get(1)
            .with_context(|| "Sky layout missing set 1")?;

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
                plant_wave_vector: self.get_plant_wave_vector().into(),
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
        )?;

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

    pub(crate) fn build_pipeline(
        &self,
        viewport: Viewport,
        render_pass: Arc<RenderPass>,
        tex: &Texture2DHolder,
        supersampling: Supersampling,
    ) -> Result<CubePipelineWrapper> {
        let vs = self
            .vs_cube
            .entry_point("main")
            .context("Missing vertex shader")?;
        let fs_solid = self
            .fs_solid
            .entry_point("main")
            .context("Missing fragment shader")?;
        let fs_sparse = self
            .fs_sparse
            .entry_point("main")
            .context("Missing fragment shader")?;
        let vertex_input_state = CubeGeometryVertex::per_vertex().definition(&vs)?;
        let stages_solid = smallvec![
            PipelineShaderStageCreateInfo::new(vs.clone()),
            PipelineShaderStageCreateInfo::new(fs_solid),
        ];
        let stages_sparse = smallvec![
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs_sparse),
        ];
        let layout_solid = PipelineLayout::new(
            self.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages_solid)
                .into_pipeline_layout_create_info(self.device.clone())?,
        )?;
        let layout_sparse = PipelineLayout::new(
            self.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages_sparse)
                .into_pipeline_layout_create_info(self.device.clone())?,
        )?;

        let solid_pipeline_info = GraphicsPipelineCreateInfo {
            stages: stages_solid,
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
            ..GraphicsPipelineCreateInfo::layout(layout_solid.clone())
        };

        // This should use the derivative pipeline mechanism, but vulkano doesn't seem to be
        // validating it properly
        // See https://stackoverflow.com/a/59312390/1424875 as well
        let sparse_pipeline_info = GraphicsPipelineCreateInfo {
            // This pipeline just needs a shader swap, but can use the same depth test
            stages: stages_sparse,
            layout: layout_sparse,
            ..solid_pipeline_info.clone()
        };
        let translucent_pipeline_info = GraphicsPipelineCreateInfo {
            // This pipeline uses the same solid shader (since we don't want discards as performed
            // by the sparse pipeline), but it needs to adjust the depth test to read but not write
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState {
                    compare_op: CompareOp::Less,
                    write_enable: false,
                }),
                depth_bounds: Default::default(),
                stencil: Default::default(),
                ..Default::default()
            }),
            ..solid_pipeline_info.clone()
        };
        let solid_pipeline = GraphicsPipeline::new(self.device.clone(), None, solid_pipeline_info)?;
        let sparse_pipeline =
            GraphicsPipeline::new(self.device.clone(), None, sparse_pipeline_info)?;
        let translucent_pipeline =
            GraphicsPipeline::new(self.device.clone(), None, translucent_pipeline_info)?;

        let solid_descriptor = tex.descriptor_set(&solid_pipeline, 0, 0)?;
        let sparse_descriptor = tex.descriptor_set(&sparse_pipeline, 0, 0)?;
        let translucent_descriptor = tex.descriptor_set(&translucent_pipeline, 0, 0)?;
        Ok(CubePipelineWrapper {
            solid_pipeline,
            sparse_pipeline,
            translucent_pipeline,
            solid_descriptor,
            sparse_descriptor,
            translucent_descriptor,
            start_time: Instant::now(),
            max_draw_indexed_index_value: self
                .device
                .physical_device()
                .properties()
                .max_draw_indexed_index_value,
        })
    }
}
impl PipelineProvider for CubePipelineProvider {
    type DrawCall<'a> = &'a mut [CubeGeometryDrawCall];

    type PerPipelineConfig<'a> = &'a Texture2DHolder;
    type PerFrameConfig = SceneState;
    type PipelineWrapperImpl = CubePipelineWrapper;
    fn make_pipeline(
        &self,
        wnd: &VulkanWindow,
        config: &Texture2DHolder,
        global_config: &LiveRenderConfig,
    ) -> Result<CubePipelineWrapper> {
        self.build_pipeline(
            wnd.viewport.clone(),
            wnd.ssaa_render_pass.clone(),
            config,
            global_config.supersampling,
        )
    }
}
