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

use super::{frag_lighting_unified_specular, frag_specular_only, SceneState};
use crate::client_state::settings::Supersampling;
use crate::vulkan::atlas::TextureAtlas;
use crate::vulkan::shaders::{
    vert_3d::{self, UniformData},
    LiveRenderConfig,
};
use crate::vulkan::{
    block_renderer::VkChunkVertexDataGpu,
    shaders::{frag_lighting_basic_color, vert_3d::ModelMatrix},
    CommandBufferBuilder, FramebufferAndLoadOpId, ImageId, LoadOp, Texture2DHolder, VulkanContext,
    VulkanWindow,
};
use anyhow::{ensure, Context, Result};
use cgmath::{Angle, Matrix4, Rad};
use smallvec::smallvec;
use std::collections::HashMap;
use std::{sync::Arc, time::Instant};
use tinyvec::array_vec;
use tracy_client::{plot, span};
use vulkano::buffer::Subbuffer;
use vulkano::descriptor_set::DescriptorSet;
use vulkano::memory::allocator::MemoryTypeFilter;
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, ColorBlendAttachmentState, ColorComponents,
};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, FrontFace};
use vulkano::pipeline::graphics::subpass::{PipelineRenderingCreateInfo, PipelineSubpassType};
use vulkano::pipeline::graphics::vertex_input::VertexDefinition;
use vulkano::pipeline::graphics::viewport::Scissor;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::shader::SpecializationConstant;
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

#[derive(BufferContents, Vertex, Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub(crate) struct CubeGeometryVertex {
    /// Position, given relative to the origin of the chunk in world space.
    /// Not transformed into camera space via a view matrix yet
    #[format(R32G32B32_SFLOAT)]
    pub(crate) position: [f32; 3],

    // Texture coordinate in tex space (0-1)
    #[format(R16G16_UINT)]
    pub(crate) uv_texcoord: [u16; 2],

    /// Normal, encoded in 15 bits
    #[format(R16_SINT)]
    pub(crate) normal: u16,

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
#[derive(Clone, Copy, PartialEq, Eq, enum_map::Enum)]
pub(crate) enum CubeDrawStep {
    /// Simple blocks with no discard, no specular, no heavy ubershader (in future), etc
    OpaqueSimple,
    /// Opaquw blocks that require additional treatment (e.g. specular)
    OpaqueSpecular,
    /// The color component of transparent drawing
    Transparent,
    /// The specular component of transparent drawing
    TransparentSpecular,
    Translucent,
    // There is no TranslucentSpecular; translucent is mostly used for water, and its weirdness
    // w.r.t. stacked transparency makes it rarely used. Translucent just gets the unified shader
    // for now, in a single pass.
    // Transparent specular *actually* requires two passes - it needs to commit reflection and
    // color buffers in different patterns, but VUlkan doesn't allow us to commit to a subset of
    // render targets, and moreover, we have different depth commits.
    RaytraceFallback,
}

impl CubePipelineWrapper {
    /// Binds and draws the specified cube drawing pass. Drawing commands are *consumed* from the
    /// provided CubeGeometryDrawCalls (i.e., after calling this with BlockRenderPass::Opaque, all
    /// the elements in CubeGeometryDrawCall will have their opaque draw calls taken away (if
    /// they were present to begin with)
    pub(crate) fn draw_single_step<L>(
        &mut self,
        ctx: &VulkanContext,
        builder: &mut CommandBufferBuilder<L>,
        uniform_buffer: Subbuffer<UniformData>,
        draw_calls: &mut [CubeGeometryDrawCall],
        pass: CubeDrawStep,
    ) -> Result<()> {
        let _span = match pass {
            CubeDrawStep::OpaqueSimple => span!("draw opaque"),
            CubeDrawStep::OpaqueSpecular => span!("draw opaque w/ specular"),
            CubeDrawStep::Transparent => span!("draw transparent"),
            CubeDrawStep::Translucent => span!("draw translucent"),
            CubeDrawStep::TransparentSpecular => span!("draw transparent w/ specular"),
            CubeDrawStep::RaytraceFallback => span!("draw raytrace_fallback"),
        };
        let pipeline = match pass {
            CubeDrawStep::OpaqueSimple => self.solid_pipeline.clone(),
            CubeDrawStep::OpaqueSpecular => self.solid_pipeline_specular.as_ref().unwrap_or(&self.solid_pipeline).clone(),
            CubeDrawStep::Transparent => self.transparent_pipeline.clone(),
            CubeDrawStep::TransparentSpecular => self.transparent_specular_pipeline.as_ref().context("Missing transparent specular pipeline but trying to render transparent specular")?.clone(),
            CubeDrawStep::Translucent => self.translucent_pipeline.clone(),
            CubeDrawStep::RaytraceFallback => self.transparent_pipeline.clone(),
        };
        let atlas_descriptor_set = match pass {
            CubeDrawStep::TransparentSpecular => self.specular_atlas_descriptor_set.as_ref().context("Missing transparent specular descriptor set but trying to render transparent specular")?.clone(),
            CubeDrawStep::Translucent | CubeDrawStep::OpaqueSpecular => self.unified_atlas_descriptor_set.clone(),
            _ => self.atlas_descriptor_set.clone()
        };
        let layout = pipeline.layout().clone();
        builder.bind_pipeline_graphics(pipeline)?;
        let per_frame_set_layout = layout
            .set_layouts()
            .get(1)
            .with_context(|| "Cube pipeline layout missing set 1")?;

        let per_frame_set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            per_frame_set_layout.clone(),
            [WriteDescriptorSet::buffer(0, uniform_buffer)],
            [],
        )?;

        builder.bind_descriptor_sets(
            vulkano::pipeline::PipelineBindPoint::Graphics,
            layout.clone(),
            0,
            vec![atlas_descriptor_set, per_frame_set],
        )?;

        let mut effective_calls = 0;
        for call in draw_calls.iter_mut() {
            let pass_data = call.models.draw_buffers[pass].take();
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
                    builder.draw_indexed(pass_data.num_indices.try_into()?, 1, 0, 0, 0)?;
                }
            }
        }
        plot!("total_calls", draw_calls.len() as f64);
        let draw_rate = effective_calls as f64 / (draw_calls.len() as f64);
        match pass {
            CubeDrawStep::OpaqueSimple => {
                plot!("opaque_rate", draw_rate);
            }
            CubeDrawStep::OpaqueSpecular => {
                plot!("opaque_heavy_rate", draw_rate);
            }
            CubeDrawStep::Transparent => {
                plot!("transparent_rate", draw_rate);
            }
            CubeDrawStep::TransparentSpecular => {
                plot!("transparent_specular_rate", draw_rate);
            }
            CubeDrawStep::Translucent => {
                plot!("translucent_rate", draw_rate);
            }
            CubeDrawStep::RaytraceFallback => {
                plot!("raytrace_fallback_rate", draw_rate);
            }
        };
        Ok(())
    }

    pub(crate) fn make_uniform_buffer(
        &self,
        ctx: &VulkanContext,
        per_frame_config: SceneState,
    ) -> Result<Subbuffer<UniformData>> {
        Buffer::from_data(
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
        )
        .context("Failed to create uniform buffer")
    }
}

pub(crate) const MAIN_FRAMEBUFFER: FramebufferAndLoadOpId<1, 0> = FramebufferAndLoadOpId {
    color_attachments: [(ImageId::MainColor, LoadOp::Load)],
    depth_stencil_attachment: Some((ImageId::MainDepthStencil, LoadOp::Load)),
    input_attachments: [],
};

pub(crate) const SPECULAR_FRAMEBUFFER: FramebufferAndLoadOpId<2, 0> = FramebufferAndLoadOpId {
    color_attachments: [
        (ImageId::RtSpecStrength, LoadOp::Load),
        (ImageId::RtSpecRayDir, LoadOp::Load),
    ],
    depth_stencil_attachment: Some((ImageId::TransparentWithSpecularDepth, LoadOp::Load)),
    input_attachments: [],
};

pub(crate) const SPECULAR_FRAMEBUFFER_CLEAR_OUTPUT: FramebufferAndLoadOpId<2, 0> =
    FramebufferAndLoadOpId {
        color_attachments: [
            (ImageId::RtSpecStrength, LoadOp::Clear),
            (ImageId::RtSpecRayDir, LoadOp::Clear),
        ],
        depth_stencil_attachment: Some((ImageId::TransparentWithSpecularDepth, LoadOp::Load)),
        input_attachments: [],
    };

pub(crate) const UNIFIED_FRAMEBUFFER: FramebufferAndLoadOpId<3, 0> = FramebufferAndLoadOpId {
    color_attachments: [
        (ImageId::MainColor, LoadOp::Load),
        (ImageId::RtSpecStrength, LoadOp::Load),
        (ImageId::RtSpecRayDir, LoadOp::Load),
    ],
    depth_stencil_attachment: Some((ImageId::MainDepthStencil, LoadOp::Load)),
    input_attachments: [],
};

pub(crate) const UNIFIED_FRAMEBUFFER_CLEAR_SPECULAR: FramebufferAndLoadOpId<3, 0> =
    FramebufferAndLoadOpId {
        color_attachments: [
            (ImageId::MainColor, LoadOp::Load),
            (ImageId::RtSpecStrength, LoadOp::Clear),
            (ImageId::RtSpecRayDir, LoadOp::Clear),
        ],
        depth_stencil_attachment: Some((ImageId::MainDepthStencil, LoadOp::Load)),
        input_attachments: [],
    };

pub(crate) struct CubePipelineProvider {
    device: Arc<Device>,
    vs_cube: Arc<ShaderModule>,
    fs_basic_color: Arc<ShaderModule>,
    fs_unified_specular: Arc<ShaderModule>,
    fs_specular_only: Arc<ShaderModule>,
}

pub(crate) struct CubePipelineWrapper {
    solid_pipeline: Arc<GraphicsPipeline>,
    solid_pipeline_specular: Option<Arc<GraphicsPipeline>>,
    transparent_pipeline: Arc<GraphicsPipeline>,
    transparent_specular_pipeline: Option<Arc<GraphicsPipeline>>,
    translucent_pipeline: Arc<GraphicsPipeline>,
    atlas_descriptor_set: Arc<DescriptorSet>,
    specular_atlas_descriptor_set: Option<Arc<DescriptorSet>>,
    unified_atlas_descriptor_set: Arc<DescriptorSet>,
    start_time: Instant,
    max_draw_indexed_index_value: u32,
}

impl CubePipelineProvider {
    pub(crate) fn new(device: Arc<Device>) -> Result<CubePipelineProvider> {
        let vs_cube = vert_3d::load_cube_geometry(device.clone())?;
        let fs_basic_color = frag_lighting_basic_color::load(device.clone())?;
        let fs_unified_specular = frag_lighting_unified_specular::load(device.clone())?;
        let fs_specular_only = frag_specular_only::load(device.clone())?;
        Ok(CubePipelineProvider {
            device,
            vs_cube,
            fs_basic_color,
            fs_unified_specular,
            fs_specular_only,
        })
    }

    pub(crate) fn build_pipeline(
        &self,
        ctx: &VulkanContext,
        viewport_state: ViewportState,
        render_passes: RenderPasses,
        tex: &TextureAtlas,
        config: &LiveRenderConfig,
    ) -> Result<CubePipelineWrapper> {
        let vs = self
            .vs_cube
            .entry_point("main")
            .context("Missing vertex shader")?;
        let fs_solid = self
            .fs_basic_color
            .specialize(HashMap::from_iter([
                (0, false.into()),
                (1, config.raytracer_debug.into()),
            ]))
            .context("Specializing basic color shader for non-sparse failed")?
            .entry_point("main")
            .context("Missing fragment shader (basic color)")?;
        let fs_sparse = self
            .fs_basic_color
            .specialize(HashMap::from_iter([
                (0, true.into()),
                (1, config.raytracer_debug.into()),
            ]))
            .context("Specializing basic color shader for sparse failed")?
            .entry_point("main")
            .context("Missing fragment shader (basic color + sparse)")?;
        let vertex_input_state = CubeGeometryVertex::per_vertex().definition(&vs)?;
        let stages_solid = smallvec![
            PipelineShaderStageCreateInfo::new(vs.clone()),
            PipelineShaderStageCreateInfo::new(fs_solid),
        ];
        let stages_sparse = smallvec![
            PipelineShaderStageCreateInfo::new(vs.clone()),
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
            viewport_state: Some(viewport_state),
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
            subpass: Some(PipelineSubpassType::BeginRenderPass(
                Subpass::from(render_passes.color().clone(), 0).context("Missing subpass")?,
            )),
            ..GraphicsPipelineCreateInfo::layout(layout_solid.clone())
        };

        // https://stackoverflow.com/a/59312390: No derivative pipelines
        let sparse_pipeline_info = GraphicsPipelineCreateInfo {
            // This pipeline just needs a shader swap, but can use the same depth test
            stages: stages_sparse,
            layout: layout_sparse,
            ..solid_pipeline_info.clone()
        };
        let will_hybrid_rt = config.raytracing && config.hybrid_rt && ctx.raytracing_supported;

        let fs_unified_nonsparse = if will_hybrid_rt {
            Some(
                self.fs_unified_specular
                    .specialize(HashMap::from_iter([
                        (0, false.into()),
                        (1, config.raytracer_debug.into()),
                    ]))
                    .context("Specializing unified color+specular shader for non-sparse failed")?
                    .entry_point("main")
                    .context("Missing fragment shader (unified color+specular non-sparse)")?,
            )
        } else {
            None
        };

        let translucent_pipeline_info = if will_hybrid_rt {
            let fs_unified_nonsparse = fs_unified_nonsparse
                .as_ref()
                .context("will_hybrid_rt true but no nonsparse unified shader")?;
            let stages = smallvec![
                PipelineShaderStageCreateInfo::new(vs.clone()),
                PipelineShaderStageCreateInfo::new(fs_unified_nonsparse.clone()),
            ];
            let layout = PipelineLayout::new(
                self.device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(self.device.clone())?,
            )?;
            GraphicsPipelineCreateInfo {
                // Read but not write
                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(DepthState {
                        compare_op: CompareOp::Less,
                        write_enable: false,
                    }),
                    depth_bounds: Default::default(),
                    stencil: Default::default(),
                    ..Default::default()
                }),
                color_blend_state: Some(ColorBlendState {
                    attachments: vec![
                        ColorBlendAttachmentState {
                            blend: Some(AttachmentBlend::alpha()),
                            color_write_mask: ColorComponents::all(),
                            color_write_enable: true,
                        },
                        ColorBlendAttachmentState {
                            blend: None,
                            color_write_mask: ColorComponents::all(),
                            color_write_enable: true,
                        },
                        ColorBlendAttachmentState {
                            blend: None,
                            color_write_mask: ColorComponents::all(),
                            color_write_enable: true,
                        },
                    ],
                    ..Default::default()
                }),
                stages,
                layout,
                subpass: Some(PipelineSubpassType::BeginRenderPass(
                    Subpass::from(
                        render_passes
                            .unified()
                            .context("Missing unified renderpass")?
                            .clone(),
                        0,
                    )
                    .context("Missing subpass")?,
                )),
                ..solid_pipeline_info.clone()
            }
        } else {
            GraphicsPipelineCreateInfo {
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
                color_blend_state: Some(ColorBlendState {
                    attachments: vec![ColorBlendAttachmentState {
                        blend: Some(AttachmentBlend::alpha()),
                        color_write_mask: ColorComponents::all(),
                        color_write_enable: true,
                    }],
                    ..Default::default()
                }),
                ..solid_pipeline_info.clone()
            }
        };
        let solid_pipeline =
            GraphicsPipeline::new(self.device.clone(), None, solid_pipeline_info.clone())?;
        let transparent_pipeline =
            GraphicsPipeline::new(self.device.clone(), None, sparse_pipeline_info)?;
        let translucent_pipeline =
            GraphicsPipeline::new(self.device.clone(), None, translucent_pipeline_info.clone())?;

        let transparent_specular_pipeline = match render_passes.specular() {
            None => None,
            Some(specular_renderpass) => {
                let fs_spec_only = self
                    .fs_specular_only
                    .entry_point("main")
                    .context("Missing vertex shader")?;
                let stages = smallvec![
                    PipelineShaderStageCreateInfo::new(vs.clone()),
                    PipelineShaderStageCreateInfo::new(fs_spec_only),
                ];
                let layout = PipelineLayout::new(
                    self.device.clone(),
                    PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                        .into_pipeline_layout_create_info(self.device.clone())?,
                )?;

                let pipeline_info = GraphicsPipelineCreateInfo {
                    stages,
                    layout,
                    subpass: Some(PipelineSubpassType::BeginRenderPass(
                        Subpass::from(specular_renderpass.clone(), 0)
                            .context("specular renderpass missing subpass 0")?
                            .clone(),
                    )),
                    color_blend_state: Some(ColorBlendState {
                        attachments: vec![
                            ColorBlendAttachmentState {
                                blend: None,
                                color_write_mask: ColorComponents::all(),
                                color_write_enable: true,
                            },
                            ColorBlendAttachmentState {
                                blend: None,
                                color_write_mask: ColorComponents::all(),
                                color_write_enable: true,
                            },
                        ],
                        ..Default::default()
                    }),
                    ..solid_pipeline_info.clone()
                };
                Some(GraphicsPipeline::new(
                    self.device.clone(),
                    None,
                    pipeline_info,
                )?)
            }
        };

        let solid_pipeline_heavy = match render_passes.unified() {
            None => None,
            Some(unified_renderpass) => {
                if will_hybrid_rt {
                    let fs_unified_nonsparse = fs_unified_nonsparse
                        .as_ref()
                        .context("will_hybrid_rt true but no nonsparse unified shader")?;
                    let stages = smallvec![
                        PipelineShaderStageCreateInfo::new(vs),
                        PipelineShaderStageCreateInfo::new(fs_unified_nonsparse.clone()),
                    ];
                    let layout = PipelineLayout::new(
                        self.device.clone(),
                        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                            .into_pipeline_layout_create_info(self.device.clone())?,
                    )?;

                    let pipeline_info = GraphicsPipelineCreateInfo {
                        stages,
                        layout,
                        subpass: Some(PipelineSubpassType::BeginRenderPass(
                            Subpass::from(unified_renderpass.clone(), 0)
                                .context("unified renderpass missing subpass 0")?
                                .clone(),
                        )),
                        color_blend_state: Some(ColorBlendState {
                            attachments: vec![
                                // Color doesn't blend for solid heavy
                                ColorBlendAttachmentState {
                                    blend: None,
                                    color_write_mask: ColorComponents::all(),
                                    color_write_enable: true,
                                },
                                // Specular G-buffers don't blend either
                                ColorBlendAttachmentState {
                                    blend: None,
                                    color_write_mask: ColorComponents::all(),
                                    color_write_enable: true,
                                },
                                ColorBlendAttachmentState {
                                    blend: None,
                                    color_write_mask: ColorComponents::all(),
                                    color_write_enable: true,
                                },
                            ],
                            ..Default::default()
                        }),
                        ..solid_pipeline_info.clone()
                    };
                    Some(GraphicsPipeline::new(
                        self.device.clone(),
                        None,
                        pipeline_info,
                    )?)
                } else {
                    None
                }
            }
        };

        let layout = solid_pipeline
            .layout()
            .set_layouts()
            .get(0)
            .with_context(|| "descriptor set 0 missing")?;
        let atlas_descriptor_set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                tex.diffuse.write_descriptor_set(0),
                tex.emissive.write_descriptor_set(1),
            ],
            [],
        )?;
        let specular_atlas_descriptor_set = match transparent_specular_pipeline.as_ref() {
            Some(x) => {
                let layout = x
                    .layout()
                    .set_layouts()
                    .get(0)
                    .with_context(|| "descriptor set 0 missing")?;
                let descriptor_set = DescriptorSet::new(
                    ctx.descriptor_set_allocator.clone(),
                    layout.clone(),
                    [
                        tex.diffuse.write_descriptor_set(0),
                        tex.specular.write_descriptor_set(1),
                    ],
                    [],
                )?;
                Some(descriptor_set)
            }
            None => None,
        };
        let unified_atlas_descriptor_set = if will_hybrid_rt {
            let layout = translucent_pipeline
                .layout()
                .set_layouts()
                .get(0)
                .with_context(|| "descriptor set 0 missing")?;
            let descriptor_set = DescriptorSet::new(
                ctx.descriptor_set_allocator.clone(),
                layout.clone(),
                [
                    tex.diffuse.write_descriptor_set(0),
                    tex.emissive.write_descriptor_set(1),
                    tex.specular.write_descriptor_set(2),
                ],
                [],
            )?;
            descriptor_set
        } else {
            atlas_descriptor_set.clone()
        };

        Ok(CubePipelineWrapper {
            solid_pipeline,
            solid_pipeline_specular: solid_pipeline_heavy,
            transparent_pipeline,
            translucent_pipeline,
            transparent_specular_pipeline,
            atlas_descriptor_set,
            specular_atlas_descriptor_set,
            unified_atlas_descriptor_set,
            start_time: Instant::now(),
            max_draw_indexed_index_value: self
                .device
                .physical_device()
                .properties()
                .max_draw_indexed_index_value,
        })
    }
}
impl CubePipelineProvider {
    pub(crate) fn make_pipeline(
        &self,
        wnd: &VulkanWindow,
        texture_atlas: &TextureAtlas,
        global_config: &LiveRenderConfig,
    ) -> Result<CubePipelineWrapper> {
        let rp = if wnd.raytracing_supported {
            RenderPasses::MainAndSpecular(
                wnd.renderpasses.get_by_framebuffer_id(MAIN_FRAMEBUFFER)?,
                wnd.renderpasses
                    .get_by_framebuffer_id(SPECULAR_FRAMEBUFFER)?,
                wnd.renderpasses
                    .get_by_framebuffer_id(UNIFIED_FRAMEBUFFER)?,
            )
        } else {
            RenderPasses::MainOnly(wnd.renderpasses.get_by_framebuffer_id(MAIN_FRAMEBUFFER)?)
        };
        self.build_pipeline(
            wnd.context(),
            ImageId::MainColor.viewport_state(&wnd.viewport, *global_config),
            rp,
            texture_atlas,
            global_config,
        )
    }
}

pub(crate) enum RenderPasses {
    MainOnly(Arc<RenderPass>),
    MainAndSpecular(Arc<RenderPass>, Arc<RenderPass>, Arc<RenderPass>),
}
impl RenderPasses {
    fn color(&self) -> &Arc<RenderPass> {
        match self {
            RenderPasses::MainOnly(x) => x,
            RenderPasses::MainAndSpecular(x, _, _) => x,
        }
    }
    fn specular(&self) -> Option<&Arc<RenderPass>> {
        match self {
            RenderPasses::MainOnly(_) => None,
            RenderPasses::MainAndSpecular(_, x, _) => Some(x),
        }
    }

    fn unified(&self) -> Option<&Arc<RenderPass>> {
        match self {
            RenderPasses::MainOnly(_) => None,
            RenderPasses::MainAndSpecular(_, _, x) => Some(x),
        }
    }
}
