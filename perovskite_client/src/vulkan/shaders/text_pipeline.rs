use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use cgmath::{vec3, ElementWise};
use vulkano::{
    descriptor_set::DescriptorSet,
    device::Device,
    pipeline::{
        graphics::{
            color_blend::{
                AttachmentBlend, ColorBlendAttachmentState, ColorBlendState, ColorComponents,
            },
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::{CullMode, FrontFace, RasterizationState},
            subpass::PipelineSubpassType,
            vertex_input::{Vertex, VertexDefinition},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::Subpass,
    shader::ShaderModule,
};

use crate::vulkan::{
    shaders::{cube_geometry::MAIN_FRAMEBUFFER, LiveRenderConfig, SceneState},
    text_renderer::{TextOutput, TextVertex},
    CommandBufferBuilder, ImageId, VulkanContext, VulkanWindow,
};

vulkano_shaders::shader! {
    shaders: {
        vert: {
            ty: "vertex",
            path: "src/vulkan/shaders/text.vert",
        },
        frag: {
            ty: "fragment",
            path: "src/vulkan/shaders/text.frag"
        },
    }
}

pub(crate) struct TextPipeline {
    pipeline: Arc<GraphicsPipeline>,
    atlas_descriptor_set: Option<Arc<DescriptorSet>>,
    atlas_descriptor_set_generation: usize,
    max_draw_indexed_count: u32,
}
impl TextPipeline {
    /// Binds the text pipeline and the texture descriptor set, then
    /// draws all glyphs. This is expected to be called during a renderpass
    /// with a single color attachment and a depth attachment, and will not
    /// start or end the renderpass itself.
    pub(crate) fn bind_and_draw<L>(
        &mut self,
        ctx: &VulkanContext,
        per_frame_config: SceneState,
        command_buf_builder: &mut CommandBufferBuilder<L>,
        text_data: &TextOutput,
    ) -> Result<()> {
        if text_data.atlas_major_generation != self.atlas_descriptor_set_generation {
            self.atlas_descriptor_set = Some(
                DescriptorSet::new(
                    ctx.descriptor_set_allocator.clone(),
                    self.pipeline
                        .layout()
                        .set_layouts()
                        .get(0)
                        .context("descriptor set 0 missing")?
                        .clone(),
                    [text_data.atlas.write_descriptor_set(0)],
                    [],
                )
                .context("Recreate atlas descriptor set")?,
            );

            self.atlas_descriptor_set_generation = text_data.atlas_major_generation;
        }

        command_buf_builder.bind_pipeline_graphics(self.pipeline.clone())?;

        let remaining_delta = text_data.origin
            - (per_frame_config
                .player_position
                .mul_element_wise(vec3(1.0, -1.0, 1.0)));

        let push = TextPushConstants {
            vp_matrix: per_frame_config.vp_matrix.into(),
            translation: [
                remaining_delta.x as f32,
                remaining_delta.y as f32,
                remaining_delta.z as f32,
            ],
        };
        command_buf_builder
            .push_constants(self.pipeline.layout().clone(), 0, push)
            .context("text push constants")?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                self.atlas_descriptor_set
                    .clone()
                    .expect("Atlas descriptor set not created"),
            )
            .context("text descriptor sets")?
            .bind_vertex_buffers(0, text_data.vertices.buffer.clone())
            .context("text vertex buffers")?
            .bind_index_buffer(text_data.indices.clone())
            .context("text index buffer")?;

        ensure!(text_data.index_count < self.max_draw_indexed_count);
        unsafe {
            // Safety:
            // Every vertex number that is retrieved from the index buffer must fall within the range of the bound vertex-rate vertex buffers.
            //   - Assured by text renderer, which generates index buffers appropriately. Note that this is a bit against the spirit of safe/unsafe
            //         since a malformed index buffer could be passed in. Alas.
            // Every vertex number that is retrieved from the index buffer, if it is not the special primitive restart value, must be no greater than the max_draw_indexed_index_value device limit.
            //   - Verified above
            // If a descriptor set binding was created with DescriptorBindingFlags::PARTIALLY_BOUND, then if the shader accesses a descriptor in that binding, the descriptor must be initialized and contain a valid resource.
            //   - N/A, we don't set PARTIALLY_BOUND
            // Shader safety: We rely on "Vulkano will validate many of these requirements, but it is only able to do so when the resources involved are statically known."
            //   - TODO: validate these more closely.
            command_buf_builder.draw_indexed(text_data.index_count, 1, 0, 0, 0)?;
        }
        Ok(())
    }
}

pub(crate) struct TextPipelineProvider {
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
}
impl TextPipelineProvider {
    pub(crate) fn new(device: Arc<Device>) -> Result<Self> {
        let vs = load_vert(device.clone())?;
        let fs = load_frag(device.clone())?;
        Ok(Self { device, vs, fs })
    }
    pub(crate) fn make_pipeline(
        &self,
        ctx: &VulkanWindow,
        global_config: &LiveRenderConfig,
    ) -> Result<TextPipeline> {
        let subpass = Subpass::from(ctx.renderpasses.get_by_framebuffer_id(MAIN_FRAMEBUFFER)?, 0)
            .context("Subpass 0 missing")?;
        let vs = self
            .vs
            .entry_point("main")
            .context("Main entry point missing from vertex shader")?;
        let fs = self
            .fs
            .entry_point("main")
            .context("Main entry point missing from fragment shader")?;
        let vertex_input_state = TextVertex::per_vertex().definition(&vs)?;
        let stages = smallvec::smallvec![
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
                    cull_mode: CullMode::None,
                    front_face: FrontFace::CounterClockwise,
                    ..Default::default()
                }),
                multisample_state: Some(MultisampleState::default()),
                viewport_state: Some(
                    ImageId::MainColor.viewport_state(&ctx.viewport, *global_config),
                ),
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
                subpass: Some(PipelineSubpassType::BeginRenderPass(subpass)),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )?;
        Ok(TextPipeline {
            pipeline,
            atlas_descriptor_set: None,
            atlas_descriptor_set_generation: 0,
            max_draw_indexed_count: self
                .device
                .physical_device()
                .properties()
                .max_draw_indexed_index_value,
        })
    }
}
