use crate::client_state::settings::Supersampling;
use crate::vulkan::block_renderer::BlockRenderer;
use crate::vulkan::shaders::{LiveRenderConfig, SceneState};
use crate::vulkan::{CommandBufferBuilder, DeferredSpecularBuffers, VulkanContext, VulkanWindow};
use anyhow::Context;
use cgmath::{vec3, SquareMatrix, Vector3};
use perovskite_core::coordinates::BlockCoordinate;
use smallvec::smallvec;
use std::collections::HashMap;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, ColorBlendAttachmentState, ColorBlendState, ColorComponents,
};
use vulkano::pipeline::graphics::depth_stencil::{
    CompareOp, DepthStencilState, StencilOp, StencilOpState, StencilOps, StencilState,
};
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
use vulkano::shader::{ShaderModule, SpecializationConstant};

vulkano_shaders::shader! {
    shaders: {
        raytraced_vtx: {
            ty: "vertex",
            path: "src/vulkan/shaders/raytracer_vtx.glsl"
        },
        raytraced_frag: {
            ty: "fragment",
            path: "src/vulkan/shaders/raytracer_frag.glsl"
        },
        trivial_vtx: {
            ty: "vertex",
            src: r"
                #version 460
                const vec2 vertices[3] = vec2[](
                vec2(-1.0, -1.0),
                vec2(-1.0, 3.0),
                vec2(3.0, -1.0)
                );

                layout(location = 0) out vec3 facedir_world;

                void main() {
                    gl_Position = vec4(vertices[gl_VertexIndex], 0.5, 1.0);
                }"
        },
        mark_stencil_frag: {
            ty: "fragment",
            src: r"
                #version 460
                layout (set = 0, binding = 0, rgba8) uniform restrict image2D deferred_specular_color;

                void main() {
                    vec4 spec_color = imageLoad(deferred_specular_color, ivec2(gl_FragCoord.xy));
                    if (spec_color.a < 0.5) {
                        discard;
                    }
                }
            "
        },
        wip_deferred_specular: {
            ty: "fragment",
            src: r"
                #version 460
                layout(location = 0) out vec4 f_color;

                void main() {
                    f_color = vec4(1.0, 0.0, 0.0, 0.5);
                }
            "
        },
    },
    custom_derives: [Debug, Clone, Copy, Default]
}

pub(crate) struct RaytracedPipelineWrapper {
    primary_pipeline: Arc<GraphicsPipeline>,
    mark_stencil_pipeline: Arc<GraphicsPipeline>,
    deferred_specular_pipeline: Arc<GraphicsPipeline>,
    long_term_descriptor_set: Arc<DescriptorSet>,
    supersampling: Supersampling,
    raytrace_control_ssbo_len: u32,
}

impl RaytracedPipelineWrapper {
    pub(crate) fn draw_rt_primary<L>(
        &mut self,
        ctx: &VulkanContext,
        per_frame_config: RaytracingBindings,
        command_buf_builder: &mut CommandBufferBuilder<L>,
    ) -> anyhow::Result<()> {
        let clamped = vec3(
            per_frame_config
                .player_pos
                .x
                .clamp(u32::MIN as f64, u32::MAX as f64),
            per_frame_config
                .player_pos
                .y
                .clamp(u32::MIN as f64, u32::MAX as f64),
            per_frame_config
                .player_pos
                .z
                .clamp(u32::MIN as f64, u32::MAX as f64),
        );

        let player_chunk = BlockCoordinate::try_from(clamped)?.chunk();
        let fine = (1.0 / 16.0)
            * (per_frame_config.player_pos
                - vec3(
                    (player_chunk.x * 16) as f64,
                    (player_chunk.y * 16) as f64,
                    (player_chunk.z * 16) as f64,
                )
                + vec3(0.5, 0.5, 0.5));

        let per_frame_data = RaytracingPerFrameData {
            inverse_vp_matrix: per_frame_config
                .scene_state
                .vp_matrix
                .invert()
                .with_context(|| {
                    format!(
                        "VP matrix was singular: {:?}",
                        per_frame_config.scene_state.vp_matrix
                    )
                })?
                .into(),
            forward_vp_matrix: per_frame_config.scene_state.vp_matrix.into(),
            supersampling: self.supersampling.to_float(),
            coarse_pos: [player_chunk.x, player_chunk.y, player_chunk.z].into(),
            fine_pos: [fine.x as f32, fine.y as f32, fine.z as f32].into(),
            max_cube_info_idx: self.raytrace_control_ssbo_len.into(),
            global_brightness_color: per_frame_config.scene_state.global_light_color.into(),
            sun_direction: [
                per_frame_config.scene_state.sun_direction.x,
                per_frame_config.scene_state.sun_direction.y * -1.0,
                per_frame_config.scene_state.sun_direction.z,
            ]
            .into(),
            render_distance: per_frame_config.render_distance,
            initial_block_id: per_frame_config.scene_state.player_pos_block,
        };

        let primary_layout = self.primary_pipeline.layout().clone();
        let primary_pfs_layout = primary_layout
            .set_layouts()
            .get(1)
            .with_context(|| "Raytraced layout missing set 1")?;
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
        let per_frame_set_primary = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            primary_pfs_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, uniform_buffer.clone()),
                WriteDescriptorSet::buffer(1, per_frame_config.header),
                WriteDescriptorSet::buffer(2, per_frame_config.data),
                WriteDescriptorSet::image_view(3, per_frame_config.depth_view),
                WriteDescriptorSet::image_view(
                    4,
                    per_frame_config.deferred_buffers.specular_strength.clone(),
                ),
                WriteDescriptorSet::image_view(
                    5,
                    per_frame_config.deferred_buffers.specular_ray_dir,
                ),
            ],
            [],
        )?;
        command_buf_builder.bind_pipeline_graphics(self.primary_pipeline.clone())?;

        command_buf_builder.bind_descriptor_sets(
            vulkano::pipeline::PipelineBindPoint::Graphics,
            primary_layout,
            0,
            vec![self.long_term_descriptor_set.clone(), per_frame_set_primary],
        )?;
        unsafe {
            // Safety: TODO
            command_buf_builder.draw(3, 1, 0, 0)?;
        }
        //
        // let mark_layout = self.mark_stencil_pipeline.layout().clone();
        // let mark_pfs_layout = mark_layout
        //     .set_layouts()
        //     .get(0)
        //     .with_context(|| "Raytraced mark-stencil layout missing set 0")?;
        //
        // let per_frame_set_mark = DescriptorSet::new(
        //     ctx.descriptor_set_allocator.clone(),
        //     mark_pfs_layout.clone(),
        //     [
        //         WriteDescriptorSet::image_view(
        //             0,
        //             per_frame_config.deferred_buffers.specular_strength,
        //         ),
        //     ],
        //     [],
        // )?;
        // command_buf_builder.bind_pipeline_graphics(self.mark_stencil_pipeline.clone())?;
        //
        // command_buf_builder.bind_descriptor_sets(
        //     vulkano::pipeline::PipelineBindPoint::Graphics,
        //     mark_layout,
        //     0,
        //     vec![per_frame_set_mark],
        // )?;
        // unsafe {
        //     // Safety: TODO
        //     command_buf_builder.draw(3, 1, 0, 0)?;
        // }
        //
        // command_buf_builder.bind_pipeline_graphics(self.deferred_specular_pipeline.clone())?;
        //
        // unsafe {
        //     // Safety: TODO
        //     command_buf_builder.draw(3, 1, 0, 0)?;
        // }
        Ok(())
    }
}

pub(crate) struct RaytracingBindings {
    pub(crate) scene_state: SceneState,
    pub(crate) data: Subbuffer<[u32]>,
    pub(crate) header: Subbuffer<ChunkMapHeader>,
    pub(crate) depth_view: Arc<ImageView>,
    pub(crate) deferred_buffers: DeferredSpecularBuffers,
    pub(crate) player_pos: Vector3<f64>,
    pub(crate) render_distance: u32,
}

pub(crate) struct RaytracedPipelineProvider {
    device: Arc<Device>,
    vs_rt: Arc<ShaderModule>,
    vs_trivial: Arc<ShaderModule>,
    fs_primary: Arc<ShaderModule>,
    fs_mark: Arc<ShaderModule>,
    fs_deferred: Arc<ShaderModule>,
}
impl RaytracedPipelineProvider {
    pub(crate) fn make_pipeline(
        &self,
        ctx: &VulkanWindow,
        frame_data: &BlockRenderer,
        global_config: &LiveRenderConfig,
    ) -> anyhow::Result<RaytracedPipelineWrapper> {
        let vs = self
            .vs_rt
            .entry_point("main")
            .context("Missing vertex shader")?;
        let vs_trivial = self
            .vs_trivial
            .entry_point("main")
            .context("Missing vertex shader")?;
        let fs_primary = self
            .fs_primary
            .specialize(HashMap::from_iter([
                (
                    0,
                    SpecializationConstant::Bool(global_config.raytracing_reflections),
                ),
                (
                    2,
                    SpecializationConstant::Bool(global_config.raytracer_debug),
                ),
            ]))?
            .entry_point("main")
            .context("Missing fragment shader")?;
        let fs_mark_stencil = self
            .fs_mark
            .entry_point("main")
            .context("Missing fragment shader")?;
        let fs_deferred_specular = self
            .fs_deferred
            .entry_point("main")
            .context("Missing fragment shader")?;
        let stages_primary = smallvec![
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs_primary),
        ];
        let stages_mark_stencil = smallvec![
            PipelineShaderStageCreateInfo::new(vs_trivial.clone()),
            PipelineShaderStageCreateInfo::new(fs_mark_stencil),
        ];
        let stages_deferred_specular = smallvec![
            PipelineShaderStageCreateInfo::new(vs_trivial),
            PipelineShaderStageCreateInfo::new(fs_deferred_specular),
        ];
        let layout = |stages| {
            Ok::<_, anyhow::Error>(PipelineLayout::new(
                self.device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(stages)
                    .into_pipeline_layout_create_info(self.device.clone())?,
            )?)
        };

        let layout_primary = layout(&stages_primary)?;
        let layout_mark = layout(&stages_mark_stencil)?;
        let layout_deferred = layout(&stages_deferred_specular)?;

        let base_pipeline_info = |layout| {
            Ok::<_, anyhow::Error>(GraphicsPipelineCreateInfo {
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
                ..GraphicsPipelineCreateInfo::layout(layout)
            })
        };

        let primary_pipeline = GraphicsPipeline::new(
            self.device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages_primary,
                depth_stencil_state: None,
                color_blend_state: Some(ColorBlendState {
                    attachments: vec![ColorBlendAttachmentState {
                        blend: Some(AttachmentBlend::alpha()),
                        color_write_mask: ColorComponents::all(),
                        color_write_enable: true,
                    }],
                    ..Default::default()
                }),
                subpass: Some(PipelineSubpassType::BeginRenderPass(
                    Subpass::from(ctx.write_color_read_depth_render_pass.clone(), 0)
                        .context("Missing subpass")?,
                )),
                ..base_pipeline_info(layout_primary.clone())?
            },
        )?;

        let mark_stencil_config = StencilOpState {
            ops: StencilOps {
                fail_op: StencilOp::Keep,
                pass_op: StencilOp::Replace,
                depth_fail_op: StencilOp::Keep,
                // Always accept stencil _entering_ the mark stencil stage.
                compare_op: CompareOp::Always,
            },
            compare_mask: u32::MAX,
            write_mask: u32::MAX,
            // and write 1 if we pass
            reference: 1,
        };
        let mark_stencil_pipeline = GraphicsPipeline::new(
            self.device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages_mark_stencil,
                depth_stencil_state: Some(DepthStencilState {
                    depth: None,
                    depth_bounds: None,
                    stencil: Some(StencilState {
                        front: mark_stencil_config,
                        back: mark_stencil_config,
                    }),
                    ..Default::default()
                }),
                color_blend_state: Some(ColorBlendState {
                    attachments: vec![ColorBlendAttachmentState {
                        blend: Some(AttachmentBlend::ignore_source()),
                        color_write_mask: ColorComponents::empty(),
                        // requires an extension that isn't universal, so we disable in other ways
                        color_write_enable: true,
                    }],
                    ..Default::default()
                }),

                subpass: Some(PipelineSubpassType::BeginRenderPass(
                    Subpass::from(ctx.color_depth_render_pass.clone(), 0)
                        .context("Missing subpass")?,
                )),
                ..base_pipeline_info(layout_mark)?
            },
        )?;
        let deferred_specular_stencil = StencilOpState {
            ops: StencilOps {
                fail_op: StencilOp::Keep,
                pass_op: StencilOp::Keep,
                depth_fail_op: StencilOp::Keep,
                // Only accept stencil=1 fragments for deferred specular
                compare_op: CompareOp::Equal,
            },
            compare_mask: u32::MAX,
            write_mask: u32::MAX,
            reference: 1,
        };
        let deferred_specular_pipeline = GraphicsPipeline::new(
            self.device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages_deferred_specular,
                depth_stencil_state: Some(DepthStencilState {
                    depth: None,
                    depth_bounds: Default::default(),
                    stencil: Some(StencilState {
                        front: deferred_specular_stencil,
                        back: deferred_specular_stencil,
                    }),
                    ..Default::default()
                }),
                color_blend_state: Some(ColorBlendState {
                    attachments: vec![ColorBlendAttachmentState {
                        blend: Some(AttachmentBlend::additive()),
                        color_write_mask: ColorComponents::all(),
                        color_write_enable: true,
                    }],
                    ..Default::default()
                }),

                subpass: Some(PipelineSubpassType::BeginRenderPass(
                    Subpass::from(ctx.color_depth_render_pass.clone(), 0)
                        .context("Missing subpass")?,
                )),
                ..base_pipeline_info(layout_deferred)?
            },
        )?;
        let raytrace_control_ssbo = frame_data.raytrace_control_ssbo();
        let raytrace_control_ssbo_len = raytrace_control_ssbo
            .len()
            .try_into()
            .context("raytrace control ssbo len too large")?;
        let descriptor_set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            layout_primary
                .set_layouts()
                .get(0)
                .with_context(|| "descriptor set missing")?
                .clone(),
            [
                frame_data.atlas().write_descriptor_set(0),
                WriteDescriptorSet::buffer(1, raytrace_control_ssbo),
            ],
            [],
        )?;

        Ok(RaytracedPipelineWrapper {
            primary_pipeline,
            mark_stencil_pipeline,
            deferred_specular_pipeline,
            long_term_descriptor_set: descriptor_set,
            supersampling: global_config.supersampling,
            raytrace_control_ssbo_len,
        })
    }
}

impl RaytracedPipelineProvider {
    pub(crate) fn new(device: Arc<Device>) -> anyhow::Result<Self> {
        Ok(RaytracedPipelineProvider {
            vs_rt: load_raytraced_vtx(device.clone())?,
            vs_trivial: load_trivial_vtx(device.clone())?,
            fs_primary: load_raytraced_frag(device.clone())?,
            fs_mark: load_mark_stencil_frag(device.clone())?,
            fs_deferred: load_wip_deferred_specular(device.clone())?,
            device,
        })
    }
}
