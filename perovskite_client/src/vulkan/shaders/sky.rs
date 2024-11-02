use crate::game_state::settings::Supersampling;
use crate::vulkan::shaders::{LiveRenderConfig, PipelineProvider, PipelineWrapper, SceneState};
use crate::vulkan::{CommandBufferBuilder, VulkanContext, VulkanWindow};
use anyhow::{Context, Result};
use cgmath::SquareMatrix;
use smallvec::smallvec;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, ColorBlendAttachmentState, ColorBlendState, ColorComponents,
};
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, FrontFace, RasterizationState};
use vulkano::pipeline::graphics::subpass::PipelineSubpassType;
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::graphics::viewport::{Scissor, Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::Subpass;
use vulkano::shader::ShaderModule;
vulkano_shaders::shader! {
    shaders: {
        sky_vtx: {
        ty: "vertex",
        src: r"
            #version 460
                const vec2 vertices[6] = vec2[](
                    vec2(-1.0, -1.0),
                    vec2(-1.0, 1.0),
                    vec2(1.0, 1.0),

                    vec2(-1.0, -1.0),
                    vec2(1.0, 1.0),
                    vec2(1.0, -1.0)
                );

                layout(set = 0, binding = 0) uniform SkyUniformData {
                    // Takes an NDC position and transforms it *back* to world space
                    mat4 inverse_vp_matrix;
                    vec3 sun_direction;
                    // Used for dither
                    float supersampling;
                };

                layout(location = 0) out vec4 global_coords_position;

                void main() {
                    vec4 pos_ndc = vec4(vertices[gl_VertexIndex], 0.5, 1.0);
                    gl_Position = pos_ndc;
                    global_coords_position = inverse_vp_matrix * pos_ndc;
                }
            "
        },
        sky_frag: {
        ty: "fragment",
        src: r"
            #version 460

            layout(location = 0) in vec4 global_coords_position;
            layout(set = 0, binding = 0) uniform SkyUniformData {
                // Takes an NDC position and transforms it *back* to world space
                mat4 inverse_vp_matrix;
                vec3 sun_direction;
                float supersampling;
            };
            layout(location = 0) out vec4 f_color;

            const vec3 base_blue = vec3(0.25, 0.6, 1.0);
            const vec3 base_orange = vec3(1.0, 0.5, 0.1);

            void main() {
                vec3 ngc = normalize(global_coords_position.xyz);
                //f_color.xyz = (ngc / 2.0) + vec3(0.5, 0.5, 0.5);
                vec3 rayleigh = vec3(5.8e-6, 13.5e-6, 33.1e-6);
                float alignment = dot(ngc, sun_direction);

                // TODO This dither has a very weak effect on banding. Implement a real dither.
                vec2 pix = gl_FragCoord.xy / supersampling;
                int ix = int(pix.x) % 2;
                int iy = int(pix.y) % 2;
                float pdither = (float(ix ^ iy) - 0.5) / 3.0;
                alignment += pdither;

                // Extinction effect strongest during sunset and after
                float sun_height = max(-sun_direction.y, 0.0);
                float extinction_strength = 4.0 / (4 * sun_height + 1.0);
                float extinction_unscaled = (alignment * 0.3) + 0.7;
                float extinction = pow(extinction_unscaled, extinction_strength);
                float leakage_correction = clamp(-sun_direction.y * 5.0 + 1.0, 0.0, 1.0);

                vec3 base_color = ((ngc / 2.0) + vec3(0.5, 0.5, 0.5));

                float sunset_strength = 1.25 * max(0.8 - abs(sun_height), 0.0);
                float sunset_blend_factor = max(sunset_strength - max(-ngc.y, 0.0), 0.0);

                base_color = sunset_blend_factor * base_orange + (1.0 - sunset_blend_factor) * base_blue;
                float extra_extinction = max(1.0, 4 * ngc.y + 1.0);

                f_color.xyz = extinction * extra_extinction * leakage_correction * base_color;

                f_color.w = 1.0;
                if (abs(ngc.x - sun_direction.x) < 0.1 &&
                    abs(ngc.y - sun_direction.y) < 0.1 &&
                    abs(ngc.z - sun_direction.z) < 0.1) {
                    f_color.xyz = vec3(1.0, 1.0, 1.0);
                }
            }
        "
        }
    }
}

pub(crate) struct SkyPipelineWrapper {
    pipeline: Arc<GraphicsPipeline>,
    supersampling: Supersampling,
}

impl PipelineWrapper<(), SceneState> for SkyPipelineWrapper {
    type PassIdentifier = ();

    fn draw<L>(
        &mut self,
        builder: &mut CommandBufferBuilder<L>,
        _draw_calls: (),
        _pass: Self::PassIdentifier,
    ) -> anyhow::Result<()> {
        builder.draw(6, 1, 0, 0)?;
        Ok(())
    }

    fn bind<L>(
        &mut self,
        ctx: &VulkanContext,
        per_frame_config: SceneState,
        command_buf_builder: &mut CommandBufferBuilder<L>,
        _pass: Self::PassIdentifier,
    ) -> anyhow::Result<()> {
        command_buf_builder.bind_pipeline_graphics(self.pipeline.clone())?;
        let layout = self.pipeline.layout().clone();
        let per_frame_data = SkyUniformData {
            inverse_vp_matrix: per_frame_config
                .vp_matrix
                .invert()
                .with_context(|| {
                    format!("VP matrix was singular: {:?}", per_frame_config.vp_matrix)
                })?
                .into(),
            sun_direction: per_frame_config.sun_direction.into(),
            supersampling: self.supersampling.to_float(),
        };
        let per_frame_set_layout = layout
            .set_layouts()
            .get(0)
            .with_context(|| "Sky layout missing set 0")?;
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
            per_frame_data,
        )?;
        let per_frame_set = PersistentDescriptorSet::new(
            &ctx.descriptor_set_allocator,
            per_frame_set_layout.clone(),
            [WriteDescriptorSet::buffer(0, uniform_buffer)],
            [],
        )?;
        command_buf_builder.bind_descriptor_sets(
            vulkano::pipeline::PipelineBindPoint::Graphics,
            layout,
            0,
            vec![per_frame_set],
        )?;
        Ok(())
    }
}

pub(crate) struct SkyPipelineProvider {
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
}
impl PipelineProvider for SkyPipelineProvider {
    type DrawCall<'a> = ();
    type PerPipelineConfig<'a> = ();
    type PerFrameConfig = SceneState;
    type PipelineWrapperImpl = SkyPipelineWrapper;

    fn make_pipeline(
        &self,
        ctx: &VulkanWindow,
        _config: Self::PerPipelineConfig<'_>,
        global_config: &LiveRenderConfig,
    ) -> anyhow::Result<Self::PipelineWrapperImpl> {
        let vs = self
            .vs
            .entry_point("main")
            .context("Missing vertex shader")?;
        let fs = self
            .fs
            .entry_point("main")
            .context("Missing fragment shader")?;
        let stages = smallvec![
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];
        let layout = PipelineLayout::new(
            self.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(self.device.clone())?,
        )?;
        let pipeline_info = GraphicsPipelineCreateInfo {
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
            viewport_state: Some(ViewportState {
                viewports: smallvec![Viewport {
                    offset: [0.0, 0.0],
                    depth_range: 0.0..=1.0,
                    extent: [
                        ctx.viewport.extent[0] * global_config.supersampling.to_float(),
                        ctx.viewport.extent[1] * global_config.supersampling.to_float()
                    ],
                }],
                scissors: smallvec![Scissor {
                    offset: [0, 0],
                    extent: [
                        ctx.viewport.extent[0] as u32 * global_config.supersampling.to_int(),
                        ctx.viewport.extent[1] as u32 * global_config.supersampling.to_int()
                    ],
                }],
                ..Default::default()
            }),
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState {
                    // No depth test whatsoever
                    compare_op: CompareOp::Always,
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
            subpass: Some(PipelineSubpassType::BeginRenderPass(
                Subpass::from(ctx.ssaa_render_pass.clone(), 0).context("Missing subpass")?,
            )),
            ..GraphicsPipelineCreateInfo::layout(layout.clone())
        };
        let pipeline = GraphicsPipeline::new(self.device.clone(), None, pipeline_info)?;
        Ok(SkyPipelineWrapper {
            pipeline,
            supersampling: global_config.supersampling,
        })
    }
}

impl SkyPipelineProvider {
    pub(crate) fn new(device: Arc<Device>) -> Result<Self> {
        Ok(SkyPipelineProvider {
            vs: load_sky_vtx(device.clone())?,
            fs: load_sky_frag(device.clone())?,
            device,
        })
    }
}
