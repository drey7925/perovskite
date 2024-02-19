use anyhow::{Context, Result};
use cgmath::{vec3, Deg, Matrix4, SquareMatrix};
use image::{DynamicImage, RgbaImage};
use perovskite_core::{
    block_id::BlockId, coordinates::ChunkOffset, protocol::blocks::BlockTypeDef,
};
use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        CopyImageToBufferInfo, PrimaryCommandBufferAbstract, RenderPassBeginInfo, SubpassContents,
    },
    format::Format,
    image::{
        view::{ImageView, ImageViewCreateInfo},
        AttachmentImage, ImageUsage,
    },
    memory::allocator::{AllocationCreateInfo, MemoryUsage},
    pipeline::graphics::viewport::Viewport,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    sync::GpuFuture,
};

use crate::{
    block_renderer::{BlockRenderer, VkChunkPassGpu, VkChunkVertexDataGpu},
    game_state::chunk::{ChunkDataView, ChunkOffsetExt},
    vulkan::shaders::SceneState,
};

use super::{
    make_render_pass,
    shaders::{
        cube_geometry::{
            BlockRenderPass, CubeGeometryDrawCall, CubePipelineProvider, CubePipelineWrapper,
        },
        PipelineWrapper,
    },
    Texture2DHolder, VulkanContext,
};

pub(crate) struct MiniBlockRenderer {
    ctx: Arc<VulkanContext>,
    surface_size: [u32; 2],
    render_pass: Arc<RenderPass>,
    framebuffer: Arc<Framebuffer>,
    target_image: Arc<ImageView<AttachmentImage>>,
    cube_provider: CubePipelineProvider,
    cube_pipeline: CubePipelineWrapper,
    download_buffer: Subbuffer<[u8]>,
    fake_chunk: Box<[BlockId; 18 * 18 * 18]>,
}
impl MiniBlockRenderer {
    pub(crate) fn new(
        ctx: Arc<VulkanContext>,
        surface_size: [u32; 2],
        atlas_texture: &Texture2DHolder,
        air_block: BlockId,
    ) -> Result<Self> {
        // All compliant GPUs should be able to render to R8G8B8A8_SRGB
        let render_pass = make_render_pass(
            ctx.vk_device.clone(),
            Format::R8G8B8A8_SRGB,
            ctx.depth_format,
        )?;
        let target_image = AttachmentImage::with_usage(
            ctx.allocator(),
            surface_size,
            Format::R8G8B8A8_SRGB,
            ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
        )?;
        let create_info = ImageViewCreateInfo {
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
            ..ImageViewCreateInfo::from_image(&target_image)
        };
        let target_image = ImageView::new(target_image, create_info)?;

        let depth_buffer = ImageView::new_default(
            AttachmentImage::transient(ctx.allocator(), surface_size, ctx.depth_format).unwrap(),
        )?;

        let framebuffer_create_info = FramebufferCreateInfo {
            attachments: vec![target_image.clone(), depth_buffer.clone()],
            ..Default::default()
        };
        let framebuffer = Framebuffer::new(render_pass.clone(), framebuffer_create_info)?;
        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [surface_size[0] as f32, surface_size[1] as f32],
            depth_range: 0.0..1.0,
        };

        let cube_provider = CubePipelineProvider::new(ctx.vk_device.clone())?;
        let cube_pipeline =
            cube_provider.build_pipeline(&ctx, viewport, render_pass.clone(), atlas_texture)?;
        let download_buffer = Buffer::new_slice(
            ctx.allocator(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Download,
                ..Default::default()
            },
            target_image.image().mem_size(),
        )?;
        Ok(Self {
            ctx,
            surface_size,
            render_pass,
            target_image,
            framebuffer,
            cube_provider,
            cube_pipeline,
            download_buffer,
            fake_chunk: Box::new([air_block; 18 * 18 * 18]),
        })
    }

    pub(crate) fn render(
        &mut self,
        block_renderer: &BlockRenderer,
        block_id: BlockId,
        block_def: &BlockTypeDef,
    ) -> Result<DynamicImage> {
        let mut vtx = vec![];
        let mut idx = vec![];

        self.fake_chunk[ChunkOffset { x: 0, y: 0, z: 0 }.as_extended_index()] = block_id;
        let fake_chunk_data = FakeChunkDataView {
            block_ids: self.fake_chunk.as_ref(),
        };
        let offset = ChunkOffset { x: 0, y: 0, z: 0 };

        block_renderer.render_single_block(
            block_def,
            block_id,
            offset,
            &fake_chunk_data,
            &mut vtx,
            &mut idx,
            &|_, _| false,
        );

        let mut commands = self.ctx.start_command_buffer()?;
        commands.begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![
                    Some([0.0, 0.0, 0.0, 0.0].into()),
                    Some(self.ctx.depth_clear_value()),
                ],
                ..RenderPassBeginInfo::framebuffer(self.framebuffer.clone())
            },
            SubpassContents::Inline,
        )?;
        let pass = VkChunkPassGpu::from_buffers(vtx, idx, self.ctx.allocator())?;

        if let Some(pass) = pass {
            self.cube_pipeline.bind(
                &self.ctx,
                *SCENE_STATE,
                &mut commands,
                BlockRenderPass::Transparent,
            )?;
            let draw_call = CubeGeometryDrawCall {
                models: VkChunkVertexDataGpu {
                    solid_opaque: None,
                    transparent: Some(pass),
                    translucent: None,
                },
                model_matrix: Matrix4::identity(),
            };
            self.cube_pipeline.draw(
                &mut commands,
                &mut [draw_call],
                BlockRenderPass::Transparent,
            )?;
        }

        commands.next_subpass(SubpassContents::Inline)?;
        commands.end_render_pass()?;
        commands.copy_image_to_buffer(CopyImageToBufferInfo {
            dst_buffer: self.download_buffer.clone(),
            ..CopyImageToBufferInfo::image_buffer(
                self.target_image.image().clone(),
                self.download_buffer.clone(),
            )
        })?;

        let commands = commands.build()?;
        commands.execute(self.ctx.clone_queue())?.flush()?;

        let guard = self.download_buffer.read()?;
        let image = RgbaImage::from_raw(self.surface_size[0], self.surface_size[1], guard.to_vec())
            .context("Failed to create image")?;
        Ok(image.into())
    }
}

static FAKE_LIGHTMAP: [u8; 18 * 18 * 18] = [255; 18 * 18 * 18];

struct FakeChunkDataView<'a> {
    block_ids: &'a [BlockId; 18 * 18 * 18],
}
impl<'a> ChunkDataView for FakeChunkDataView<'a> {
    fn block_ids(&self) -> &[BlockId; 18 * 18 * 18] {
        self.block_ids
    }

    fn lightmap(&self) -> &[u8; 18 * 18 * 18] {
        &FAKE_LIGHTMAP
    }
}

lazy_static::lazy_static! {
    static ref SCENE_STATE: SceneState = {
        let _projection = cgmath::perspective(Deg(45.0), 1.0, 0.01, 1000.);
        let projection = cgmath::ortho(-1., 1., -1., 1., -2., 2.);
        let rotation = Matrix4::from_angle_x(Deg(-30.0)) * Matrix4::from_angle_y(Deg(135.0));
        let coord_system_conversion = Matrix4::from_nonuniform_scale(-1.0, 1.0, 1.0);
        let translation = Matrix4::from_translation(vec3(0.0, 0.0, -1.0));
        let vp_matrix = projection * translation * rotation * coord_system_conversion;
        SceneState {
            vp_matrix,
            clear_color: [0.0, 0.0, 0.0, 0.0],
            global_light_color: [0.0, 0.0, 0.0],
            global_light_direction: vec3(0., 0., 0.)
        }
    };
}
