use super::{
    block_renderer::{BlockRenderer, VkChunkVertexDataGpu},
    shaders::cube_geometry::{
        CubeDrawStep, CubeGeometryDrawCall, CubePipelineProvider, CubePipelineWrapper,
    },
    LoadOp, RenderPassHolder, RenderPassId, Texture2DHolder, VulkanContext,
};
use crate::client_state::settings::Supersampling;
use crate::vulkan::atlas::TextureAtlas;
use crate::vulkan::block_renderer::VkChunkRaytraceData;
use crate::vulkan::shaders::cube_geometry::RenderPasses;
use crate::vulkan::shaders::{LiveRenderConfig, VkDrawBufferGpu};
use crate::{
    client_state::chunk::{ChunkDataView, ChunkOffsetExt},
    vulkan::shaders::SceneState,
};
use anyhow::{Context, Result};
use cgmath::{vec3, Deg, Matrix4, SquareMatrix};
use enum_map::enum_map;
use image::{DynamicImage, RgbaImage};
use perovskite_core::block_id::special_block_defs::AIR_ID;
use perovskite_core::protocol::map::ClientExtendedData;
use perovskite_core::{
    block_id::BlockId, coordinates::ChunkOffset, protocol::blocks::BlockTypeDef,
};
use smallvec::smallvec;
use std::sync::Arc;
use tinyvec::array_vec;
use vulkano::command_buffer::{SubpassBeginInfo, SubpassEndInfo};
use vulkano::image::{Image, ImageCreateInfo, ImageLayout, ImageType};
use vulkano::memory::allocator::MemoryTypeFilter;
use vulkano::pipeline::graphics::viewport::{Scissor, ViewportState};
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
    ordered_passes_renderpass,
    pipeline::graphics::viewport::Viewport,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    sync::GpuFuture,
    DeviceSize,
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
        atlas_texture: &TextureAtlas,
    ) -> Result<Self> {
        let renderpasses = RenderPassHolder::new(ctx.vk_device.clone(), ctx.non_swapchain_config());
        let render_pass = renderpasses.get(RenderPassId {
            color_attachments: array_vec!({ (Format::R8G8B8A8_SRGB, LoadOp::Clear) }),
            depth_stencil_attachment: Some((ctx.depth_stencil_format, LoadOp::Clear)),
            input_attachments: array_vec!(),
        })?;

        let extent = [surface_size[0], surface_size[1], 1];
        let target_image = Image::new(
            ctx.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_SRGB,
                extent,
                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )?;

        let target_image = ImageView::new_default(target_image)?;

        let depth_stencil_attachment = ImageView::new_default(Image::new(
            ctx.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: ctx.depth_stencil_format,
                extent,
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )?)?;

        let framebuffer_create_info = FramebufferCreateInfo {
            attachments: vec![target_image.clone(), depth_stencil_attachment],
            ..Default::default()
        };
        let framebuffer = Framebuffer::new(render_pass.clone(), framebuffer_create_info)?;
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [surface_size[0] as f32, surface_size[1] as f32],
            depth_range: 0.0..=1.0,
        };

        let viewport_state = ViewportState {
            viewports: smallvec![Viewport {
                offset: [0.0, 0.0],
                depth_range: 0.0..=1.0,
                extent: [viewport.extent[0], viewport.extent[1]],
            }],
            scissors: smallvec![Scissor {
                offset: [0, 0],
                extent: [surface_size[0], surface_size[1]],
            }],
            ..Default::default()
        };

        let cube_provider = CubePipelineProvider::new(ctx.vk_device.clone())?;
        let cube_pipeline = cube_provider.build_pipeline(
            &ctx,
            viewport_state,
            RenderPasses::MainOnly(render_pass.clone()),
            atlas_texture,
            &ctx.non_swapchain_config(),
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
        let pass = VkDrawBufferGpu::from_buffers(&vtx, &idx, self.ctx.clone_allocator())?;

        if let Some(buffer) = pass {
            let mut draw_buffers = enum_map! { _ => None };
            draw_buffers[CubeDrawStep::Opaque] = Some(buffer);
            let draw_call = CubeGeometryDrawCall {
                models: VkChunkVertexDataGpu { draw_buffers },
                model_matrix: Matrix4::identity(),
            };
            self.cube_pipeline.draw_single_step(
                &self.ctx,
                &mut commands,
                *SCENE_STATE,
                &mut [draw_call],
                CubeDrawStep::Transparent,
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

    fn client_ext_data(&self, offset: ChunkOffset) -> Option<&ClientExtendedData> {
        None
    }

    fn raytrace_data(&self) -> Option<&VkChunkRaytraceData> {
        log::warn!("Tried to access raytrace data from FakeChunkDataView");
        None
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
            global_light_color: [0.0, 0.0, 0.0],
            sun_direction: vec3(0., 0., 0.),
            player_pos_block: 0,
        }
    };
}
