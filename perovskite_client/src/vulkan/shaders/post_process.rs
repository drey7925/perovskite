use crate::vulkan::shaders::{LiveRenderConfig, SceneState};
use crate::vulkan::{CommandBufferBuilder, VulkanContext, VulkanWindow};
use anyhow::{Context, Result};
use smallvec::smallvec;
use std::sync::Arc;
use vulkano::device::Device;
use vulkano::pipeline::graphics::color_blend::{
    ColorBlendAttachmentState, ColorBlendState, ColorComponents,
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
        post_process_hdr_frag: {
        ty: "fragment",
        src: r"
            #version 460
                void main() {}
            "
        },
    }
}

pub(crate) struct PostProcessingPipelineWrapper {
    pipeline: Arc<GraphicsPipeline>,
}

impl PostProcessingPipelineWrapper {
    pub(crate) fn bind_and_draw<L>(
        &mut self,
        ctx: &VulkanContext,
        per_frame_config: SceneState,
        command_buf_builder: &mut CommandBufferBuilder<L>,
    ) -> anyhow::Result<()> {
        command_buf_builder.bind_pipeline_graphics(self.pipeline.clone())?;

        Ok(())
    }
}

pub(crate) struct PostProcessingPipelineProvider {
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
}
impl PostProcessingPipelineProvider {
    pub(crate) fn make_pipeline(
        &self,
        ctx: &VulkanWindow,
        global_config: &LiveRenderConfig,
    ) -> anyhow::Result<PostProcessingPipelineWrapper> {
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
                    blend: None,
                    color_write_mask: ColorComponents::all(),
                    color_write_enable: true,
                }],
                ..Default::default()
            }),
            subpass: Some(PipelineSubpassType::BeginRenderPass(
                Subpass::from(ctx.raster_renderpass()?, 0).context("Missing subpass")?,
            )),
            ..GraphicsPipelineCreateInfo::layout(layout.clone())
        };
        let pipeline = GraphicsPipeline::new(self.device.clone(), None, pipeline_info)?;
        Ok(PostProcessingPipelineWrapper { pipeline })
    }
}

impl PostProcessingPipelineWrapper {
    pub(crate) fn new(device: Arc<Device>) -> Result<Self> {
        todo!()
    }
}
