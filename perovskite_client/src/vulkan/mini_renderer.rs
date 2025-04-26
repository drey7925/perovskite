use anyhow::{Context, Result};
use cgmath::{vec3, Deg, Matrix4, SquareMatrix};
use image::{DynamicImage, RgbaImage};
use perovskite_core::block_id::special_block_defs::AIR_ID;
use perovskite_core::{
    block_id::BlockId, coordinates::ChunkOffset, protocol::blocks::BlockTypeDef,
};
use std::sync::Arc;
use vulkano::command_buffer::{SubpassBeginInfo, SubpassEndInfo};
use vulkano::image::{Image, ImageCreateInfo, ImageLayout, ImageType};
use vulkano::memory::allocator::MemoryTypeFilter;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        CopyImageToBufferInfo, PrimaryCommandBufferAbstract, RenderPassBeginInfo, SubpassContents,
    },
    format::Format,
    image::{
        view::{ImageView, ImageViewCreateInfo},
        ImageUsage,
    },
    memory::allocator::AllocationCreateInfo,
    pipeline::graphics::viewport::Viewport,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    sync::GpuFuture,
    DeviceSize,
};

use super::{
    block_renderer::{BlockRenderer, VkChunkVertexDataGpu},
    make_ssaa_render_pass,
    shaders::{
        cube_geometry::{
            BlockRenderPass, CubeGeometryDrawCall, CubePipelineProvider, CubePipelineWrapper,
        },
        PipelineWrapper,
    },
    Texture2DHolder, VulkanContext,
};
use crate::client_state::settings::Supersampling;
use crate::vulkan::shaders::VkBufferGpu;
use crate::{
    client_state::chunk::{ChunkDataView, ChunkOffsetExt},
    vulkan::shaders::SceneState,
};

pub(crate) struct MiniBlockRenderer {
    ctx: Arc<VulkanContext>,
    surface_size: [u32; 2],
    render_pass: Arc<RenderPass>,
    framebuffer: Arc<Framebuffer>,
    target_image: Arc<ImageView>,
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
    ) -> Result<Self> {
        // All compliant GPUs should be able to render to R8G8B8A8_SRGB
        let render_pass = make_ssaa_render_pass(
            ctx.vk_device.clone(),
            Format::R8G8B8A8_SRGB,
            ctx.depth_format,
        )?;

        let target_image = Image::new(
            ctx.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_SRGB,
                view_formats: vec![Format::R8G8B8A8_SRGB],
                extent: [surface_size[0], surface_size[1], 1],
                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )?;

        let create_info = ImageViewCreateInfo {
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
            ..ImageViewCreateInfo::from_image(&target_image)
        };
        let target_image = ImageView::new(target_image, create_info)?;

        let depth_buffer = ImageView::new_default(Image::new(
            ctx.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: ctx.depth_format,
                view_formats: vec![ctx.depth_format],
                extent: [surface_size[0], surface_size[1], 1],
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                initial_layout: ImageLayout::Undefined,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )?)?;

        let framebuffer_create_info = FramebufferCreateInfo {
            attachments: vec![target_image.clone(), depth_buffer.clone()],
            ..Default::default()
        };
        let framebuffer = Framebuffer::new(render_pass.clone(), framebuffer_create_info)?;
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [surface_size[0] as f32, surface_size[1] as f32],
            depth_range: 0.0..=1.0,
        };

        let cube_provider = CubePipelineProvider::new(ctx.vk_device.clone())?;
        let cube_pipeline = cube_provider.build_pipeline(
            viewport,
            render_pass.clone(),
            atlas_texture,
            Supersampling::None,
        )?;
        let download_buffer = Buffer::new_slice(
            ctx.clone_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            (surface_size[0] as DeviceSize)
                .checked_mul(surface_size[1] as DeviceSize)
                .context("Surface too big")?
                .checked_mul(4)
                .context("Surface too big")?,
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
            fake_chunk: Box::new([AIR_ID; 18 * 18 * 18]),
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
            SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..Default::default()
            },
        )?;
        let pass = VkBufferGpu::from_buffers(&vtx, &idx, self.ctx.clone_allocator())?;

        if let Some(pass) = pass {
            self.cube_pipeline.bind(
                &self.ctx,
                *SCENE_STATE,
                &mut commands,
                BlockRenderPass::Transparent,
            )?;
            let draw_call = CubeGeometryDrawCall {
                models: VkChunkVertexDataGpu {
                    opaque: None,
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
        commands.end_render_pass(SubpassEndInfo {
            ..Default::default()
        })?;
        commands.copy_image_to_buffer(CopyImageToBufferInfo {
            dst_buffer: self.download_buffer.clone(),
            ..CopyImageToBufferInfo::image_buffer(
                self.target_image.image().clone(),
                self.download_buffer.clone(),
            )
        })?;

        let commands = commands.build()?;
        commands.execute(self.ctx.clone_graphics_queue())?.flush()?;

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
    fn is_empty_optimization_hint(&self) -> bool {
        false
    }

    fn block_ids(&self) -> &[BlockId; 18 * 18 * 18] {
        self.block_ids
    }

    fn lightmap(&self) -> &[u8; 18 * 18 * 18] {
        &FAKE_LIGHTMAP
    }
}

lazy_static::lazy_static! {
    static ref SCENE_STATE: SceneState = {
        let projection = cgmath::ortho(-1., 1., -1., 1., -2., 2.);
        let rotation = Matrix4::from_angle_x(Deg(-30.0)) * Matrix4::from_angle_y(Deg(135.0));
        let coord_system_conversion = Matrix4::from_nonuniform_scale(-1.0, 1.0, 1.0);
        let translation = Matrix4::from_translation(vec3(0.0, 0.0, -1.0));
        let vp_matrix = projection * translation * rotation * coord_system_conversion;
        SceneState {
            vp_matrix,
            clear_color: [0.0, 0.0, 0.0, 0.0],
            global_light_color: [0.0, 0.0, 0.0],
            sun_direction: vec3(0., 0., 0.)
        }
    };
}
