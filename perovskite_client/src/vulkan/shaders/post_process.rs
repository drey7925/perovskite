use crate::vulkan::shaders::LiveRenderConfig;
use crate::vulkan::{
    CommandBufferBuilder, FramebufferAndLoadOpId, FramebufferHolder, ImageId, LoadOp, VulkanWindow,
};
use anyhow::{Context, Result};
use smallvec::smallvec;
use std::collections::HashMap;
use std::sync::Arc;
use vulkano::command_buffer::{SubpassContents, SubpassEndInfo};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::image::sampler::{
    BorderColor, Filter, Sampler, SamplerAddressMode, SamplerCreateInfo,
};
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, ColorBlendAttachmentState, ColorBlendState, ColorComponents,
};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, FrontFace, RasterizationState};
use vulkano::pipeline::graphics::subpass::PipelineSubpassType;
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::Subpass;
use vulkano::shader::{ShaderModule, SpecializationConstant};

vulkano_shaders::shader! {
    shaders: {
        gen_uv: {
            ty: "vertex",
            src: r"#version 460
                const vec2 vertices[3] = vec2[](
                vec2(-1.0, -1.0),
                vec2(-1.0, 3.0),
                vec2(3.0, -1.0)
                );
                const vec2 uvs[3] = vec2[](
                vec2(0, 0),
                vec2(0, 2),
                vec2(2, 0)
                );

                layout(location = 0) out vec2 uv;
                void main() {
                    gl_Position = vec4(vertices[gl_VertexIndex], 0.5, 1.0);
                    uv = uvs[gl_VertexIndex];
                }"
        },
        extract_overbright: {
        ty: "fragment",
        src: r"
            #version 460
            layout(location = 0) in vec2 uv;
            layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput main_color;
            layout(location = 0) out vec4 f_color;
            layout (constant_id = 0) const float BLOOM_STRENGTH = 1.0;

            void main() {
                f_color = subpassLoad(main_color);
                f_color.rgb = BLOOM_STRENGTH * max((f_color.rgb - 1) / 1000, 0);
            }
            "
        },
        // https://community.arm.com/cfs-file/__key/communityserver-blogs-components-weblogfiles/00-00-00-20-66/siggraph2015_2D00_mmg_2D00_marius_2D00_notes.pdf
        downsample: {
        ty: "fragment",
        src: r"
            #version 460
            // Given w.r.t the input image
            layout(push_constant) uniform PostProcessUniform {
                vec2 ht;
            };
            layout (constant_id = 0) const float CENTER_MUL = 0.5;
            layout (constant_id = 1) const float CORNER_MUL = 0.125;

            layout(location = 0) in vec2 uv;
            layout(location = 0) out vec4 f_color;
            layout(set = 0, binding = 0) uniform sampler2D src;
            void main() {
                vec3 color = texture(src, uv).rgb * CENTER_MUL;
                color += texture(src, uv + vec2(ht.x, ht.y)).rgb * CORNER_MUL;
                color += texture(src, uv + vec2(-ht.x, ht.y)).rgb * CORNER_MUL;
                color += texture(src, uv + vec2(ht.x, -ht.y)).rgb * CORNER_MUL;
                color += texture(src, uv + vec2(-ht.x, -ht.y)).rgb * CORNER_MUL;
                f_color = vec4(color, 1.0);
            }
            "
        },
        upsample: {
        ty: "fragment",
        src: r"
            #version 460
            // Given w.r.t the input image
            layout(push_constant) uniform PostProcessUniform {
                vec2 ht;
            };
            // Adjusted from the original slide deck, center mul optional
            // layout (constant_id = 0) const float CENTER_MUL = 0;
            layout (constant_id = 1) const float CORNER_MUL = 0.16666666667;
            layout (constant_id = 2) const float EDGE_MUL = 0.083333333333;

            layout(location = 0) in vec2 uv;
            layout(location = 0) out vec4 f_color;
            layout(set = 0, binding = 0) uniform sampler2D src;
            void main() {
                vec3 color = vec3(0); //texture(src, uv).rgb * CENTER_MUL;
                color += texture(src, uv + vec2(ht.x, ht.y)).rgb * CORNER_MUL;
                color += texture(src, uv + vec2(-ht.x, ht.y)).rgb * CORNER_MUL;
                color += texture(src, uv + vec2(ht.x, -ht.y)).rgb * CORNER_MUL;
                color += texture(src, uv + vec2(-ht.x, -ht.y)).rgb * CORNER_MUL;
                color += texture(src, uv + vec2(2 * ht.x, 0)).rgb * EDGE_MUL;
                color += texture(src, uv + vec2(-2 * ht.x, 0)).rgb * EDGE_MUL;
                color += texture(src, uv + vec2(0, 2 * ht.y)).rgb * EDGE_MUL;
                color += texture(src, uv + vec2(0, -2 * ht.y)).rgb * EDGE_MUL;
                f_color = vec4(color, 1.0);
            }
            "
        },
        lens_flare: {
        ty: "fragment",
        src: r"
            #version 460

            layout (constant_id = 0) const float FACTOR = 0.01;
            layout(location = 0) in vec2 uv;
            layout(location = 0) out vec4 f_color;
            layout(set = 0, binding = 0) uniform sampler2D src;
            void main() {
                f_color = vec4(0, 0, 0, 1);
                f_color.rgb += texture(src, 1.0 - uv).rgb * FACTOR * vec3(1.0, 0.3, 0.7);
                f_color.rgb += texture(src, 1.62 - 2.24 * uv).rgb * FACTOR * vec3(0.2, 0.6, 0.6);
                f_color.rgb += texture(src, 0.9 - 0.8 * uv).rgb * FACTOR * vec3(0.6, 0.9, 0.2);
                f_color.rgb += texture(src, 0.2 + 0.6 * uv).rgb * FACTOR * vec3(0.6, 0.9, 0.2);
            }
            "
        },
    }
}

pub(crate) struct PostProcessingPipelineWrapper {
    extractor_pipeline: Arc<GraphicsPipeline>,
    lens_flare_pipeline: Arc<GraphicsPipeline>,
    blur_stages: Vec<(PassInfo, Arc<GraphicsPipeline>, PostProcessUniform)>,
    sampler: Arc<Sampler>,
}

impl PostProcessingPipelineWrapper {
    pub(crate) fn bind_and_draw<L>(
        &mut self,
        ctx: &VulkanWindow,
        framebuffer: &FramebufferHolder,
        cmd: &mut CommandBufferBuilder<L>,
    ) -> Result<()> {
        if ctx.renderpasses.config.bloom_strength == 0.0
            && ctx.renderpasses.config.lens_flare_strength == 0.0
        {
            return Ok(());
        }
        let extractor_pfs = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.extractor_pipeline
                .layout()
                .set_layouts()
                .get(0)
                .with_context(|| "Extractor layout missing set 0")?
                .clone(),
            [WriteDescriptorSet::image_view(
                0,
                framebuffer.get_image(ImageId::MainColorResolved)?,
            )],
            [],
        )?;

        framebuffer
            .begin_render_pass(cmd, EXTRACTOR, ctx.renderpasses(), SubpassContents::Inline)
            .context("Begin extractor renderpass")?;
        cmd.bind_pipeline_graphics(self.extractor_pipeline.clone())?;
        cmd.bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            self.extractor_pipeline.layout().clone(),
            0,
            vec![extractor_pfs],
        )?;
        unsafe {
            cmd.draw(3, 1, 0, 0)
                .context("Failed to draw extractor pass")?;
        }
        cmd.end_render_pass(SubpassEndInfo::default())?;

        let lens_flare_pfs = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.lens_flare_pipeline
                .layout()
                .set_layouts()
                .get(0)
                .with_context(|| "Lens flare layout missing set 0")?
                .clone(),
            [WriteDescriptorSet::image_view_sampler(
                0,
                framebuffer.get_image(ImageId::Blur(0))?,
                self.sampler.clone(),
            )],
            [],
        )?;

        if ctx.renderpasses.config.lens_flare_strength > 0.0 {
            framebuffer
                .begin_render_pass(cmd, LENS_FLARE, ctx.renderpasses(), SubpassContents::Inline)
                .context("Begin lens flare renderpass")?;
            cmd.bind_pipeline_graphics(self.lens_flare_pipeline.clone())?;
            cmd.bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.lens_flare_pipeline.layout().clone(),
                0,
                vec![lens_flare_pfs],
            )?;
            unsafe {
                cmd.draw(3, 1, 0, 0)
                    .context("Failed to draw lens flare pass")?;
            }
            cmd.end_render_pass(SubpassEndInfo::default())?;
        }

        for (pass, pipeline, push_constant) in self.blur_stages.iter() {
            let (framebuffer_id, _kernel, source_image, _blend) = pass;

            let pfs = DescriptorSet::new(
                ctx.descriptor_set_allocator.clone(),
                pipeline
                    .layout()
                    .set_layouts()
                    .get(0)
                    .with_context(|| "Pipeline layout missing set 0")?
                    .clone(),
                [WriteDescriptorSet::image_view_sampler(
                    0,
                    framebuffer.get_image(*source_image)?,
                    self.sampler.clone(),
                )],
                [],
            )?;

            framebuffer.begin_render_pass(
                cmd,
                *framebuffer_id,
                ctx.renderpasses(),
                SubpassContents::Inline,
            )?;

            cmd.bind_pipeline_graphics(pipeline.clone())?;
            cmd.bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                vec![pfs],
            )?;
            cmd.push_constants(pipeline.layout().clone(), 0, *push_constant)?;
            unsafe {
                cmd.draw(3, 1, 0, 0).context("Failed to draw blur pass")?;
            }
            cmd.end_render_pass(SubpassEndInfo::default())?;
        }

        Ok(())
    }
}

const EXTRACTOR: FramebufferAndLoadOpId<1, 1> = FramebufferAndLoadOpId {
    color_attachments: [(ImageId::Blur(0), LoadOp::DontCare)],
    depth_stencil_attachment: None,
    input_attachments: [(ImageId::MainColorResolved, LoadOp::Load)],
};

const LENS_FLARE: FramebufferAndLoadOpId<1, 0> = FramebufferAndLoadOpId {
    color_attachments: [(ImageId::MainColorResolved, LoadOp::Load)],
    depth_stencil_attachment: None,
    input_attachments: [],
};

#[derive(Debug, Clone, Copy)]
enum Kernel {
    Down,
    Up,
}

#[derive(Debug, Clone, Copy)]
enum Blend {
    None,
    Additive,
}

type PassInfo = (FramebufferAndLoadOpId<1, 0>, Kernel, ImageId, Blend);

pub(crate) struct PostProcessingPipelineProvider {
    device: Arc<Device>,
    gen_uv: Arc<ShaderModule>,
    extract_overbright: Arc<ShaderModule>,
    downsample: Arc<ShaderModule>,
    upsample: Arc<ShaderModule>,
    lens_flare: Arc<ShaderModule>,
}
impl PostProcessingPipelineProvider {
    pub(crate) fn make_pipeline(
        &self,
        ctx: &VulkanWindow,
        global_config: &LiveRenderConfig,
    ) -> Result<PostProcessingPipelineWrapper> {
        let vs = self
            .gen_uv
            .entry_point("main")
            .context("Missing vertex shader")?;
        let fs_extract = self
            .extract_overbright
            .specialize(HashMap::from_iter([(
                0,
                SpecializationConstant::F32(global_config.bloom_strength),
            )]))?
            .entry_point("main")
            .context("Missing fragment shader: extractor")?;
        let fs_downsample = self
            .downsample
            .entry_point("main")
            .context("Missing fragment shader: downsample")?;
        let fs_upsample = self
            .upsample
            .entry_point("main")
            .context("Missing fragment shader: upsample")?;
        let fs_lens_flare = self
            .lens_flare
            .specialize(HashMap::from_iter([(
                0,
                // compensate for bloom strength (Blur0 is a dual-purpose buffer)
                SpecializationConstant::F32(
                    global_config.lens_flare_strength * 0.03
                        / (global_config.bloom_strength + 0.0000001),
                ),
            )]))?
            .entry_point("main")
            .context("Missing fragment shader: lens flare")?;

        let extractor_stages = smallvec![
            PipelineShaderStageCreateInfo::new(vs.clone()),
            PipelineShaderStageCreateInfo::new(fs_extract)
        ];
        let extractor_layout = PipelineLayout::new(
            self.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&extractor_stages)
                .into_pipeline_layout_create_info(self.device.clone())?,
        )?;

        let vp_size = ImageId::MainColorResolved.dimension(
            ctx.viewport.extent[0] as u32,
            ctx.viewport.extent[1] as u32,
            *global_config,
        );
        let max_steps = u32::min(vp_size.0, vp_size.1).ilog2() - 5;

        let mut passes = vec![];
        let total_steps = global_config.blur_steps.min(max_steps as usize);
        for step in 0..total_steps {
            let input = ImageId::Blur(step as u8);
            let target = FramebufferAndLoadOpId {
                color_attachments: [(ImageId::Blur((step + 1) as u8), LoadOp::DontCare)],
                depth_stencil_attachment: None,
                input_attachments: [],
            };
            passes.push((target, Kernel::Down, input, Blend::None));
        }
        for step in 0..total_steps {
            let input = ImageId::Blur((total_steps - step) as u8);
            let (color, blend, op) = if step == (total_steps - 1) {
                (ImageId::MainColorResolved, Blend::Additive, LoadOp::Load)
            } else {
                (
                    ImageId::Blur((total_steps - step - 1) as u8),
                    Blend::None,
                    LoadOp::DontCare,
                )
            };
            let target = FramebufferAndLoadOpId {
                color_attachments: [(color, op)],
                depth_stencil_attachment: None,
                input_attachments: [],
            };
            passes.push((target, Kernel::Up, input, blend));
        }

        let extractor_pipeline_info = GraphicsPipelineCreateInfo {
            stages: extractor_stages,
            // No bindings or attributes
            vertex_input_state: Some(VertexInputState::new()),
            input_assembly_state: Some(InputAssemblyState::default()),
            rasterization_state: Some(RasterizationState {
                cull_mode: CullMode::Back,
                front_face: FrontFace::CounterClockwise,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            depth_stencil_state: None,
            color_blend_state: Some(ColorBlendState {
                attachments: vec![ColorBlendAttachmentState {
                    blend: None,
                    color_write_mask: ColorComponents::all(),
                    color_write_enable: true,
                }],
                ..Default::default()
            }),
            viewport_state: Some(ImageId::Blur(0).viewport_state(&ctx.viewport, *global_config)),
            subpass: Some(PipelineSubpassType::BeginRenderPass(
                Subpass::from(
                    ctx.renderpasses
                        .get_by_framebuffer_id(EXTRACTOR)
                        .context("Missing extractor renderpass")?,
                    0,
                )
                .context("Missing subpass")?,
            )),
            ..GraphicsPipelineCreateInfo::layout(extractor_layout)
        };
        let extractor_pipeline =
            GraphicsPipeline::new(self.device.clone(), None, extractor_pipeline_info)
                .context("Building overbright extractor pipeline")?;

        let lens_flare_stages = smallvec![
            PipelineShaderStageCreateInfo::new(vs.clone()),
            PipelineShaderStageCreateInfo::new(fs_lens_flare)
        ];
        let lens_flare_layout = PipelineLayout::new(
            self.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&lens_flare_stages)
                .into_pipeline_layout_create_info(self.device.clone())?,
        )?;

        let lens_flare_pipeline_info = GraphicsPipelineCreateInfo {
            stages: lens_flare_stages,
            // No bindings or attributes
            vertex_input_state: Some(VertexInputState::new()),
            input_assembly_state: Some(InputAssemblyState::default()),
            rasterization_state: Some(RasterizationState {
                cull_mode: CullMode::Back,
                front_face: FrontFace::CounterClockwise,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            depth_stencil_state: None,
            color_blend_state: Some(ColorBlendState {
                attachments: vec![ColorBlendAttachmentState {
                    blend: Some(AttachmentBlend::additive()),
                    color_write_mask: ColorComponents::all(),
                    color_write_enable: true,
                }],
                ..Default::default()
            }),
            viewport_state: Some(
                ImageId::MainColorResolved.viewport_state(&ctx.viewport, *global_config),
            ),
            subpass: Some(PipelineSubpassType::BeginRenderPass(
                Subpass::from(
                    ctx.renderpasses
                        .get_by_framebuffer_id(LENS_FLARE)
                        .context("Missing lens flare renderpass")?,
                    0,
                )
                .context("Missing subpass")?,
            )),
            ..GraphicsPipelineCreateInfo::layout(lens_flare_layout)
        };
        let lens_flare_pipeline =
            GraphicsPipeline::new(self.device.clone(), None, lens_flare_pipeline_info)
                .context("Building overbright extractor pipeline")?;

        let mut blur_stages = vec![];
        for pass in passes {
            let (framebuffer, kernel, source_image, blend) = pass;
            let fs = match kernel {
                Kernel::Down => fs_downsample.clone(),
                Kernel::Up => fs_upsample.clone(),
            };
            let blend = match blend {
                Blend::None => None,
                Blend::Additive => Some(AttachmentBlend::additive()),
            };
            let stages = smallvec![
                PipelineShaderStageCreateInfo::new(vs.clone()),
                PipelineShaderStageCreateInfo::new(fs)
            ];
            let layout = PipelineLayout::new(
                self.device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(self.device.clone())?,
            )?;

            let info = GraphicsPipelineCreateInfo {
                stages,
                // No bindings or attributes
                vertex_input_state: Some(VertexInputState::new()),
                input_assembly_state: Some(InputAssemblyState::default()),
                rasterization_state: Some(RasterizationState {
                    cull_mode: CullMode::Back,
                    front_face: FrontFace::CounterClockwise,
                    ..Default::default()
                }),
                multisample_state: Some(MultisampleState::default()),
                depth_stencil_state: None,
                color_blend_state: Some(ColorBlendState {
                    attachments: vec![ColorBlendAttachmentState {
                        blend,
                        color_write_mask: ColorComponents::all(),
                        color_write_enable: true,
                    }],
                    ..Default::default()
                }),
                viewport_state: Some(
                    framebuffer.color_attachments[0]
                        .0
                        .viewport_state(&ctx.viewport, *global_config),
                ),
                subpass: Some(PipelineSubpassType::BeginRenderPass(
                    Subpass::from(
                        ctx.renderpasses
                            .get_by_framebuffer_id(framebuffer)
                            .with_context(|| format!("Missing renderpass {framebuffer}"))?,
                        0,
                    )
                    .context("Missing subpass")?,
                )),
                ..GraphicsPipelineCreateInfo::layout(layout.clone())
            };
            let pipeline = GraphicsPipeline::new(self.device.clone(), None, info)
                .with_context(|| format!("Building pipeline for {framebuffer}/{kernel:?}"))?;

            let source_image_size = framebuffer.color_attachments[0].0.dimension(
                ctx.viewport.extent[0] as u32,
                ctx.viewport.extent[1] as u32,
                *global_config,
            );

            let push_constant = PostProcessUniform {
                ht: [
                    0.5 / (source_image_size.0 as f32),
                    0.5 / (source_image_size.1 as f32),
                ],
            };

            blur_stages.push((pass, pipeline, push_constant));
        }

        Ok(PostProcessingPipelineWrapper {
            extractor_pipeline,
            lens_flare_pipeline,
            blur_stages,
            sampler: Sampler::new(
                ctx.vk_device.clone(),
                SamplerCreateInfo {
                    mag_filter: Filter::Linear,
                    min_filter: Filter::Linear,
                    address_mode: [SamplerAddressMode::ClampToBorder; 3],
                    border_color: BorderColor::FloatTransparentBlack,
                    ..Default::default()
                },
            )?,
        })
    }
    pub(crate) fn new(device: Arc<Device>) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            gen_uv: load_gen_uv(device.clone())?,
            extract_overbright: load_extract_overbright(device.clone())?,
            downsample: load_downsample(device.clone())?,
            upsample: load_upsample(device.clone())?,
            lens_flare: load_lens_flare(device.clone())?,
        })
    }
}
