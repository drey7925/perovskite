use crate::client_state::settings::Supersampling;
use crate::vulkan::block_renderer::BlockRenderer;
use crate::vulkan::shaders::{LiveRenderConfig, PipelineProvider, PipelineWrapper, SceneState};
use crate::vulkan::{CommandBufferBuilder, VulkanContext, VulkanWindow};
use anyhow::Context;
use cgmath::{vec3, SquareMatrix, Vector3};
use perovskite_core::coordinates::BlockCoordinate;
use smallvec::smallvec;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
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
        raytraced_vtx: {
            ty: "vertex",
            path: "src/vulkan/shaders/raytracer_vtx.glsl"
        },
        raytraced_frag: {
            ty: "fragment",
            path: "src/vulkan/shaders/raytracer_frag.glsl"
        },
    },
    vulkan_version: "1.3",
    spirv_version: "1.3",
}

pub(crate) struct RaytracedPipelineWrapper {
    pipeline: Arc<GraphicsPipeline>,
    descriptor_set: Arc<DescriptorSet>,
    supersampling: Supersampling,
    raytrace_control_ssbo_len: u32,
}

impl PipelineWrapper<(), (SceneState, Subbuffer<[u32]>, Vector3<f64>)>
    for RaytracedPipelineWrapper
{
    type PassIdentifier = ();

    fn draw<L>(
        &mut self,
        builder: &mut CommandBufferBuilder<L>,
        _draw_calls: (),
        _pass: Self::PassIdentifier,
    ) -> anyhow::Result<()> {
        unsafe {
            // Safety: TODO
            builder.draw(6, 1, 0, 0)?;
        }
        Ok(())
    }

    fn bind<L>(
        &mut self,
        ctx: &VulkanContext,
        per_frame_config: (SceneState, Subbuffer<[u32]>, Vector3<f64>),
        command_buf_builder: &mut CommandBufferBuilder<L>,
        _pass: Self::PassIdentifier,
    ) -> anyhow::Result<()> {
        let (per_frame_config, ssbo, player_pos) = per_frame_config;
        command_buf_builder.bind_pipeline_graphics(self.pipeline.clone())?;
        let layout = self.pipeline.layout().clone();

        let player_chunk = BlockCoordinate::try_from(player_pos)?.chunk();
        let fine = player_pos
            - vec3(
                (player_chunk.x * 16) as f64,
                (player_chunk.y * 16) as f64,
                (player_chunk.z * 16) as f64,
            );

        let per_frame_data = RaytracedUniformData {
            inverse_vp_matrix: per_frame_config
                .vp_matrix
                .invert()
                .with_context(|| {
                    format!("VP matrix was singular: {:?}", per_frame_config.vp_matrix)
                })?
                .into(),
            supersampling: self.supersampling.to_float(),
            coarse_pos: [player_chunk.x, player_chunk.y, player_chunk.z].into(),
            fine_pos: [fine.x as f32, fine.y as f32, fine.z as f32].into(),
            max_cube_info_idx: self.raytrace_control_ssbo_len,
        };
        let per_frame_set_layout = layout
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
        let per_frame_set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            per_frame_set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, uniform_buffer),
                WriteDescriptorSet::buffer(1, ssbo),
            ],
            [],
        )?;
        command_buf_builder.bind_descriptor_sets(
            vulkano::pipeline::PipelineBindPoint::Graphics,
            layout,
            0,
            vec![self.descriptor_set.clone(), per_frame_set],
        )?;
        Ok(())
    }
}

pub(crate) struct RaytracedPipelineProvider {
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
}
impl PipelineProvider for RaytracedPipelineProvider {
    type DrawCall<'a> = ();
    type PerPipelineConfig<'a> = &'a BlockRenderer;
    type PerFrameConfig = (SceneState, Subbuffer<[u32]>, Vector3<f64>);
    type PipelineWrapperImpl = RaytracedPipelineWrapper;

    fn make_pipeline(
        &self,
        ctx: &VulkanWindow,
        config: Self::PerPipelineConfig<'_>,
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
        let raytrace_control_ssbo = config.raytrace_control_ssbo();
        let raytrace_control_ssbo_len = raytrace_control_ssbo
            .len()
            .try_into()
            .context("raytrace control ssbo len too large")?;
        let descriptor_set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            dbg!(pipeline.layout().set_layouts().get(0))
                .with_context(|| "descriptor set missing")?
                .clone(),
            [
                config.atlas().write_descriptor_set(0),
                WriteDescriptorSet::buffer(1, raytrace_control_ssbo),
            ],
            [],
        )?;

        Ok(RaytracedPipelineWrapper {
            pipeline,
            descriptor_set,
            supersampling: global_config.supersampling,
            raytrace_control_ssbo_len,
        })
    }
}

impl RaytracedPipelineProvider {
    pub(crate) fn new(device: Arc<Device>) -> anyhow::Result<Self> {
        Ok(RaytracedPipelineProvider {
            vs: load_raytraced_vtx(device.clone())?,
            fs: load_raytraced_frag(device.clone())?,
            device,
        })
    }
}
