use anyhow::{Context, Result};
use std::sync::Arc;
use vulkano::{
    descriptor_set::{self, PersistentDescriptorSet, WriteDescriptorSet},
    device::Device,
    pipeline::{
        graphics::{
            color_blend::ColorBlendState, depth_stencil::DepthStencilState,
            rasterization::RasterizationState, viewport::ViewportState,
        },
        GraphicsPipeline, Pipeline,
    },
    render_pass::{Framebuffer, Subpass},
    sampler::{Filter, Sampler, SamplerCreateInfo},
    shader::ShaderModule,
};

use super::{frag_blit, vert_2d_blit, PipelineProvider, PipelineWrapper};

pub(crate) struct SupersamplerBlitPipelineWrapper {
    pipeline: Arc<GraphicsPipeline>,
    descriptor_sets: Vec<Arc<PersistentDescriptorSet>>,
}
impl PipelineWrapper<(), usize> for SupersamplerBlitPipelineWrapper {
    type PassIdentifier = ();

    fn draw<L>(
        &mut self,
        builder: &mut crate::vulkan::CommandBufferBuilder<L>,
        _draw_calls: &[()],
        _pass: Self::PassIdentifier,
    ) -> anyhow::Result<()> {
        match builder.draw(6, 1, 0, 0) {
            Ok(x) => (),
            Err(e) => println!("{:?}", e),
        }
        Ok(())
    }

    fn bind<L>(
        &mut self,
        _ctx: &crate::vulkan::VulkanContext,
        per_frame_config: usize,
        command_buf_builder: &mut crate::vulkan::CommandBufferBuilder<L>,
        _pass: Self::PassIdentifier,
    ) -> anyhow::Result<()> {
        let layout = self.pipeline.layout().clone();
        println!("{}", per_frame_config);
        command_buf_builder.bind_pipeline_graphics(self.pipeline.clone());
        command_buf_builder.bind_descriptor_sets(
            vulkano::pipeline::PipelineBindPoint::Graphics,
            layout,
            0,
            vec![self.descriptor_sets[per_frame_config].clone()],
        );
        Ok(())
    }
}

pub(crate) struct SupersamplerBlitPipelineProvider {
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
}
impl SupersamplerBlitPipelineProvider {
    pub(crate) fn new(device: Arc<crate::vulkan::Device>) -> Result<Self> {
        let vs = vert_2d_blit::load_flat_tex(device.clone())?;
        let fs = frag_blit::load(device.clone())?;
        Ok(Self { device, vs, fs })
    }
}
impl PipelineProvider for SupersamplerBlitPipelineProvider {
    type DrawCall = ();
    type PerFrameConfig = usize;
    type PipelineWrapperImpl = SupersamplerBlitPipelineWrapper;

    type PerPipelineConfig<'a> = &'a [Arc<Framebuffer>];

    fn make_pipeline(
        &self,
        ctx: &crate::vulkan::VulkanContext,
        config: Self::PerPipelineConfig<'_>,
    ) -> Result<Self::PipelineWrapperImpl> {
        let pipeline = GraphicsPipeline::start()
            .vertex_shader(self.vs.entry_point("main").unwrap(), ())
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
            .color_blend_state(ColorBlendState::default())
            .render_pass(Subpass::from(ctx.pre_blit_pass.clone(), 1).unwrap())
            .build(self.device.clone())?;
        let descriptor_sets = config
            .iter()
            .map(|framebuffer| {
                let layout = pipeline
                    .layout()
                    .set_layouts()
                    .get(0)
                    .with_context(|| "sampler set missing")
                    .unwrap();
                PersistentDescriptorSet::new(
                    &ctx.descriptor_set_allocator,
                    layout.clone(),
                    [
                        WriteDescriptorSet::image_view(0, framebuffer.attachments()[0].clone()),
                        // WriteDescriptorSet::image_view_sampler(
                        // 0,
                        // framebuffer.attachments()[0].clone(),
                        // Sampler::new(
                        //     ctx.vk_device.clone(),
                        //     SamplerCreateInfo {
                        //         mag_filter: Filter::Nearest,
                        //         min_filter: Filter::Linear,
                        //         ..Default::default()
                        //     },
                        // )
                        // .unwrap(),
                        // )
                    ],
                )
                .unwrap()
            })
            .collect();
        Ok(SupersamplerBlitPipelineWrapper {
            pipeline,
            descriptor_sets,
        })
    }
}
