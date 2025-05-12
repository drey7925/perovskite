// Copyright 2023 drey7925
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

pub(crate) mod block_renderer;
pub(crate) mod entity_renderer;
pub mod game_renderer;
pub(crate) mod mini_renderer;
pub(crate) mod shaders;
pub(crate) mod util;
// Public for benchmarking
pub mod gpu_chunk_table;

use std::sync::atomic::AtomicBool;
use std::{ops::Deref, sync::Arc};

use anyhow::{bail, Context, Result};
use arc_swap::ArcSwap;
use image::GenericImageView;
use log::warn;
use smallvec::smallvec;

use texture_packer::Rect;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocatorCreateInfo;
use vulkano::command_buffer::{
    BlitImageInfo, BufferImageCopy, CommandBufferUsage, CopyBufferInfo, CopyBufferToImageInfo,
    SubpassBeginInfo,
};
use vulkano::descriptor_set::DescriptorSet;
use vulkano::format::NumericFormat;
use vulkano::image::sampler::{Filter, Sampler, SamplerCreateInfo};
use vulkano::image::{
    Image, ImageAspects, ImageCreateInfo, ImageLayout, ImageSubresourceLayers, ImageType,
};
use vulkano::memory::allocator::{
    AllocationCreateInfo, GenericMemoryAllocatorCreateInfo, MemoryTypeFilter,
};
use vulkano::memory::{MemoryProperties, MemoryPropertyFlags};
use vulkano::swapchain::{ColorSpace, Surface};
use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract, RenderPassBeginInfo,
        SubpassContents,
    },
    descriptor_set::{allocator::StandardDescriptorSetAllocator, WriteDescriptorSet},
    device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures,
        Queue, QueueCreateInfo,
    },
    format::{ClearValue, Format, FormatFeatures},
    image::{view::ImageView, ImageUsage},
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{BuddyAllocator, GenericMemoryAllocator},
    pipeline::{graphics::viewport::Viewport, GraphicsPipeline, Pipeline},
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    swapchain::{Swapchain, SwapchainCreateInfo},
    sync::GpuFuture,
    DeviceSize, Validated, Version,
};
use winit::dpi::Size;
use winit::event_loop::ActiveEventLoop;
use winit::window::WindowAttributes;
use winit::{dpi::PhysicalSize, event_loop::EventLoop, window::Window};

pub(crate) type CommandBufferBuilder<L> = AutoCommandBufferBuilder<L>;

use crate::client_state::settings::{GameSettings, Supersampling};

use self::util::select_physical_device;

pub(crate) type VkAllocator = GenericMemoryAllocator<BuddyAllocator>;

#[derive(Clone)]
pub(crate) struct VulkanContext {
    vk_device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    transfer_queue: Arc<Queue>,
    memory_allocator: Arc<VkAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    /// The format of the image buffer used for rendering to *screen*. Render-to-texture always uses R8G8B8A8_SRGB
    color_format: Format,
    /// The format of the depth buffer used for rendering to *screen*. Probed at startup, and used for both render-to-screen and render-to-texture
    depth_format: Format,

    /// For settings, all GPUs detected on this machine
    all_gpus: Vec<String>,
    max_draw_indexed_index_value: u32,
}

impl VulkanContext {
    pub(crate) fn command_buffer_allocator(&self) -> Arc<StandardCommandBufferAllocator> {
        self.command_buffer_allocator.clone()
    }

    pub(crate) fn clone_allocator(&self) -> Arc<VkAllocator> {
        self.memory_allocator.clone()
    }

    pub(crate) fn allocator(&self) -> &VkAllocator {
        &self.memory_allocator
    }

    pub(crate) fn clone_graphics_queue(&self) -> Arc<Queue> {
        self.graphics_queue.clone()
    }

    pub(crate) fn clone_transfer_queue(&self) -> Arc<Queue> {
        self.transfer_queue.clone()
    }

    pub(crate) fn all_gpus(&self) -> &[String] {
        &self.all_gpus
    }

    fn start_command_buffer(&self) -> Result<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>> {
        let builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.graphics_queue.queue_family_index(),
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )?;

        Ok(builder)
    }

    pub(crate) fn depth_clear_value(&self) -> ClearValue {
        if self.depth_format.aspects().contains(ImageAspects::STENCIL) {
            ClearValue::DepthStencil((1.0, 0))
        } else {
            ClearValue::Depth(1.0)
        }
    }

    pub(crate) fn copy_to_device<T: BufferContents>(&self, data: T) -> Result<Subbuffer<T>> {
        let staging_buffer = Buffer::from_data(
            self.clone_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data,
        )?;
        let target_buffer = Buffer::new_sized(
            self.clone_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )?;

        let transfer_queue = self.clone_transfer_queue();
        let mut command_buffer = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator(),
            transfer_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        command_buffer.copy_buffer(CopyBufferInfo::buffers(
            staging_buffer,
            target_buffer.clone(),
        ))?;
        command_buffer.build()?.execute(transfer_queue)?.flush()?;
        Ok(target_buffer)
    }

    pub(crate) fn iter_to_device<T: BufferContents>(
        &self,
        data: impl ExactSizeIterator<Item = T>,
    ) -> Result<Subbuffer<[T]>> {
        let staging_buffer = Buffer::from_iter(
            self.clone_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data,
        )?;
        let target_buffer = Buffer::new_slice(
            self.clone_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            staging_buffer.len(),
        )?;

        let transfer_queue = self.clone_transfer_queue();
        let mut command_buffer = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator(),
            transfer_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        command_buffer.copy_buffer(CopyBufferInfo::buffers(
            staging_buffer,
            target_buffer.clone(),
        ))?;
        command_buffer.build()?.execute(transfer_queue)?.flush()?;
        Ok(target_buffer)
    }
}

pub(crate) struct VulkanWindow {
    vk_ctx: Arc<VulkanContext>,
    ssaa_render_pass: Arc<RenderPass>,
    post_blit_render_pass: Arc<RenderPass>,
    swapchain: Arc<Swapchain>,
    swapchain_images: Vec<Arc<Image>>,
    framebuffers: Vec<FramebufferHolder>,
    window: Arc<Window>,
    viewport: Viewport,
    want_recreate: AtomicBool,
}

impl Deref for VulkanWindow {
    type Target = VulkanContext;
    fn deref(&self) -> &Self::Target {
        &self.vk_ctx
    }
}

impl VulkanWindow {
    pub(crate) fn request_recreate(&self) {
        self.want_recreate
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn context(&self) -> &VulkanContext {
        &self.vk_ctx
    }

    pub fn clone_context(&self) -> Arc<VulkanContext> {
        self.vk_ctx.clone()
    }

    pub(crate) fn create(
        event_loop: &ActiveEventLoop,
        settings: &Arc<ArcSwap<GameSettings>>,
    ) -> Result<VulkanWindow> {
        let library: Arc<vulkano::VulkanLibrary> =
            vulkano::VulkanLibrary::new().expect("no local Vulkan library/DLL");
        let required_extensions = Surface::required_extensions(event_loop)?;
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: required_extensions,
                application_name: Some("perovskite".to_string()),
                application_version: Version {
                    major: 0,
                    minor: 1,
                    patch: 0,
                },
                ..Default::default()
            },
        )?;

        let attrs = WindowAttributes::new()
            .with_title("Perovskite Game Client")
            .with_min_inner_size(Size::Physical((256, 256).into()));
        let window = Arc::new(event_loop.create_window(attrs)?);
        let surface = Surface::from_window(instance.clone(), window.clone())?;

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window.inner_size().into(),
            depth_range: 0.0..=1.0,
        };

        let device_extensions = DeviceExtensions {
            ext_shader_subgroup_vote: true,
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        let device_features = DeviceFeatures {
            ..DeviceFeatures::empty()
        };

        let (physical_device, graphics_family_index, transfer_family_index) =
            select_physical_device(
                &instance,
                &surface,
                &device_extensions,
                &device_features,
                &settings.load().render.preferred_gpu,
            )?;

        let supersampling = settings.load().render.supersampling;

        let queue_create_infos = if graphics_family_index == transfer_family_index {
            vec![QueueCreateInfo {
                queue_family_index: graphics_family_index,
                ..Default::default()
            }]
        } else {
            vec![
                QueueCreateInfo {
                    queue_family_index: graphics_family_index,
                    ..Default::default()
                },
                QueueCreateInfo {
                    queue_family_index: transfer_family_index,
                    ..Default::default()
                },
            ]
        };

        let (vk_device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos,
                enabled_extensions: device_extensions,
                enabled_features: device_features,
                ..Default::default()
            },
        )?;

        let graphics_queue;
        let transfer_queue;
        if graphics_family_index == transfer_family_index {
            graphics_queue = queues.next().with_context(|| "expected a graphics queue")?;
            transfer_queue = graphics_queue.clone();
        } else {
            graphics_queue = queues.next().with_context(|| "expected a graphics queue")?;
            transfer_queue = queues.next().with_context(|| "expected a transfer queue")?;
        }

        let (swapchain, swapchain_images, color_format) = {
            let caps = physical_device
                .surface_capabilities(&surface, Default::default())
                .expect("failed to get surface capabilities");

            let composite_alpha = caps
                .supported_composite_alpha
                .into_iter()
                .next()
                .context("No supported composite alpha")?;
            let formats = physical_device.surface_formats(&surface, Default::default())?;
            log::info!("Surface available color formats: {formats:?}");
            let (image_format, color_space) = find_best_format(formats)?;
            log::info!("Will render to {image_format:?}, {color_space:?}");
            let mut image_count = caps.min_image_count;
            if let Some(max_image_count) = caps.max_image_count {
                if max_image_count >= 3 {
                    image_count = 3;
                }
            }

            let (swapchain, images) = Swapchain::new(
                vk_device.clone(),
                surface,
                SwapchainCreateInfo {
                    min_image_count: image_count,
                    image_format,
                    image_extent: window.inner_size().into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
                    composite_alpha,
                    image_color_space: color_space,
                    ..Default::default()
                },
            )?;
            (swapchain, images, image_format)
        };

        let depth_format = find_best_depth_format(&physical_device)?;

        let ssaa_render_pass =
            make_ssaa_render_pass(vk_device.clone(), color_format, depth_format)?;
        let post_blit_render_pass = make_post_blit_render_pass(vk_device.clone(), color_format)?;

        // Copied from vulkano source - they implement it only for the freelist allocator, but we
        // need it for the buddy allocator
        let MemoryProperties {
            memory_types,
            memory_heaps,
            ..
        } = vk_device.physical_device().memory_properties();

        let mut block_sizes = vec![0; memory_types.len()];
        let mut memory_type_bits = u32::MAX;

        for (index, memory_type) in memory_types.iter().enumerate() {
            const LARGE_HEAP_THRESHOLD: DeviceSize = 1024 * 1024 * 1024;

            let heap_size = memory_heaps[memory_type.heap_index as usize].size;

            block_sizes[index] = if heap_size >= LARGE_HEAP_THRESHOLD {
                256 * 1024 * 1024
            } else {
                64 * 1024 * 1024
            };

            if memory_type.property_flags.intersects(
                MemoryPropertyFlags::LAZILY_ALLOCATED
                    | MemoryPropertyFlags::PROTECTED
                    | MemoryPropertyFlags::DEVICE_COHERENT
                    | MemoryPropertyFlags::RDMA_CAPABLE,
            ) {
                // VUID-VkMemoryAllocateInfo-memoryTypeIndex-01872
                // VUID-vkAllocateMemory-deviceCoherentMemory-02790
                // Lazily allocated memory would just cause problems for suballocation in general.
                memory_type_bits &= !(1 << index);
            }
        }

        let allocator_params = GenericMemoryAllocatorCreateInfo {
            block_sizes: &block_sizes,
            memory_type_bits,
            ..Default::default()
        };

        let memory_allocator = Arc::new(GenericMemoryAllocator::new(
            vk_device.clone(),
            allocator_params,
        ));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            vk_device.clone(),
            StandardCommandBufferAllocatorCreateInfo {
                primary_buffer_count: 32,
                secondary_buffer_count: 16,
                ..Default::default()
            },
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            vk_device.clone(),
            Default::default(),
        ));
        let framebuffers = get_framebuffers_with_depth(
            &swapchain_images,
            memory_allocator.clone(),
            ssaa_render_pass.clone(),
            post_blit_render_pass.clone(),
            depth_format,
            supersampling,
        )?;
        let all_gpus = instance
            .enumerate_physical_devices()
            .unwrap()
            .map(|x| x.properties().device_name.clone())
            .collect::<Vec<String>>();

        Ok(VulkanWindow {
            vk_ctx: Arc::new(VulkanContext {
                vk_device,
                graphics_queue,
                transfer_queue,
                memory_allocator,
                command_buffer_allocator,
                descriptor_set_allocator,
                color_format,
                depth_format,
                all_gpus,
                max_draw_indexed_index_value: physical_device
                    .properties()
                    .max_draw_indexed_index_value,
            }),
            ssaa_render_pass,
            post_blit_render_pass,
            swapchain,
            swapchain_images,
            framebuffers,
            window,
            viewport,
            want_recreate: AtomicBool::new(false),
        })
    }

    fn recreate_swapchain(
        &mut self,
        size: PhysicalSize<u32>,
        supersampling: Supersampling,
    ) -> Result<()> {
        let size = PhysicalSize::new(size.width.max(1), size.height.max(1));
        let (new_swapchain, new_images) = match self.swapchain.recreate(SwapchainCreateInfo {
            image_extent: size.into(),
            ..self.swapchain.create_info()
        }) {
            Ok(r) => r,
            Err(Validated::Error(e)) => {
                warn!("Ignoring swapchain creation error: {e}");
                return Ok(());
            }
            Err(Validated::ValidationError(e)) => return Err(anyhow::Error::from(e)),
        };
        self.swapchain = new_swapchain;
        self.swapchain_images = new_images.clone();
        self.framebuffers = get_framebuffers_with_depth(
            &new_images,
            self.memory_allocator.clone(),
            self.ssaa_render_pass.clone(),
            self.post_blit_render_pass.clone(),
            self.depth_format,
            supersampling,
        )?;
        Ok(())
    }

    fn start_ssaa_render_pass(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        framebuffer: Arc<Framebuffer>,
        clear_color: [f32; 4],
    ) -> Result<()> {
        builder.begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![
                    // Supersampled color buffer
                    Some(clear_color.into()),
                    // Supersampled depth buffer
                    Some(self.depth_clear_value()),
                ],
                render_pass: self.ssaa_render_pass.clone(),
                ..RenderPassBeginInfo::framebuffer(framebuffer)
            },
            SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..Default::default()
            },
        )?;
        Ok(())
    }

    fn start_post_blit_render_pass(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        framebuffer: Arc<Framebuffer>,
    ) -> Result<()> {
        builder.begin_render_pass(
            RenderPassBeginInfo {
                render_pass: self.post_blit_render_pass.clone(),
                clear_values: vec![None],
                ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
            },
            SubpassBeginInfo {
                contents: SubpassContents::SecondaryCommandBuffers,
                ..Default::default()
            },
        )?;
        Ok(())
    }

    pub(crate) fn window_size(&self) -> (u32, u32) {
        let dims = self.viewport.extent;
        (dims[0] as u32, dims[1] as u32)
    }

    pub(crate) fn swapchain(&self) -> &Swapchain {
        self.swapchain.as_ref()
    }

    pub(crate) fn ssaa_render_pass(&self) -> Arc<RenderPass> {
        self.ssaa_render_pass.clone()
    }
    pub(crate) fn post_blit_render_pass(&self) -> Arc<RenderPass> {
        self.post_blit_render_pass.clone()
    }
}

fn find_best_depth_format(physical_device: &PhysicalDevice) -> Result<Format> {
    const FORMATS_TO_TRY: [Format; 5] = [
        Format::D24_UNORM_S8_UINT,
        Format::X8_D24_UNORM_PACK32,
        Format::D32_SFLOAT,
        Format::D32_SFLOAT_S8_UINT,
        Format::D16_UNORM,
    ];
    for format in FORMATS_TO_TRY {
        if physical_device
            .format_properties(format)?
            .optimal_tiling_features
            .contains(FormatFeatures::DEPTH_STENCIL_ATTACHMENT)
        {
            log::info!("Depth format found: {format:?}");
            if format == Format::D16_UNORM {
                warn!("Depth format D16_UNORM may have low precision and cause visual glitches");
                warn!("According to the vulkan spec, at least one of the other formats should be supported by any compliant GPU.");
                warn!("Please reach out to the devs with details about your GPU.");
            }
            return Ok(format);
        }
    }
    bail!("No depth format found");
}

pub(crate) fn make_ssaa_render_pass(
    vk_device: Arc<Device>,
    output_format: Format,
    depth_format: Format,
) -> Result<Arc<RenderPass>> {
    vulkano::ordered_passes_renderpass!(
        vk_device,
        attachments: {
            color_ssaa: {
                format: output_format,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
            depth: {
                format: depth_format,
                samples: 1,
                load_op: Clear,
                store_op: DontCare,
            },
        },
        passes: [
            {
                color: [color_ssaa],
                depth_stencil: {depth},
                input: [],
            },
        ]
    )
    .context("Renderpass creation failed")
}

fn make_post_blit_render_pass(
    vk_device: Arc<Device>,
    output_format: Format,
) -> Result<Arc<RenderPass>> {
    vulkano::ordered_passes_renderpass!(
        vk_device,
        attachments: {
            color_blitted: {
                format: output_format,
                samples: 1,
                load_op: Load,
                store_op: Store,
            },
        },
        passes: [
            {
                color: [color_blitted],
                depth_stencil: {},
                input: [],
            },
        ]
    )
    .context("Renderpass creation failed")
}

fn find_best_format(
    formats: Vec<(Format, vulkano::swapchain::ColorSpace)>,
) -> Result<(Format, vulkano::swapchain::ColorSpace)> {
    // todo get an HDR format
    // This requires enabling ext_swapchain_colorspace and also getting shaders to do
    // srgb conversions if applicable

    dbg!(formats)
        .iter()
        .find(|(format, space)| {
            *space == ColorSpace::SrgbNonLinear
                && format.numeric_format_color() == Some(NumericFormat::SRGB)
        })
        .cloned()
        .with_context(|| "Could not find an image format")
}

#[derive(Clone)]
pub(crate) struct FramebufferHolder {
    supersampling: Supersampling,
    ssaa_framebuffer: Arc<Framebuffer>,
    // Vector starting with the source and ending with the target. It may be a singleton, but must
    // not be empty.
    blit_path: Vec<Arc<Image>>,
    post_blit_framebuffer: Arc<Framebuffer>,
}

impl FramebufferHolder {
    pub(crate) fn blit_supersampling(
        &self,
        command_buf_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> Result<()> {
        for i in 0..(self.blit_path.len() - 1) {
            command_buf_builder.blit_image(BlitImageInfo {
                filter: Filter::Linear,
                ..BlitImageInfo::images(self.blit_path[i].clone(), self.blit_path[i + 1].clone())
            })?;
        }

        Ok(())
    }
}

// Helper to get framebuffers for swapchain images
// Modified from a sample function from the vulkano examples.
//
// Vulkano examples are under copyright:
// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.
pub(crate) fn get_framebuffers_with_depth(
    images: &[Arc<Image>],
    allocator: Arc<VkAllocator>,
    ssaa_renderpass: Arc<RenderPass>,
    post_blit_renderpass: Arc<RenderPass>,
    depth_format: Format,
    supersampling: Supersampling,
) -> Result<Vec<FramebufferHolder>> {
    images
        .iter()
        .map(|image| -> anyhow::Result<_> {
            let view = ImageView::new_default(image.clone()).unwrap();
            let ssaa_depth_view = ImageView::new_default(Image::new(
                allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: depth_format,
                    view_formats: vec![depth_format],
                    // TODO make adjustable
                    extent: [
                        image.extent()[0] * supersampling.to_int(),
                        image.extent()[1] * supersampling.to_int(),
                        1,
                    ],
                    usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                    initial_layout: ImageLayout::Undefined,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )?)?;
            // Now build the blit path
            let blit_path = if supersampling == Supersampling::None {
                // The blit path is trivial - just the original image
                // In fact, we do no blitting here - see the implementation of FramebufferHolder
                vec![image.clone()]
            } else {
                let mut blit_path = vec![];
                let mut multiplier = supersampling.to_int();

                for i in 0..supersampling.blit_steps() {
                    let usage = if i == 0 {
                        // First image: We render to it, and we blit from it
                        ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC
                    } else {
                        // Intermediate image: We blit into it, and we blit from it
                        ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST
                    };
                    log::debug!(
                        "Creating blit path image {i} with usage {usage:?}, {multiplier}x samples"
                    );

                    let buffer = Image::new(
                        allocator.clone(),
                        ImageCreateInfo {
                            image_type: ImageType::Dim2d,
                            format: image.format(),
                            view_formats: vec![image.format()],
                            extent: [
                                image.extent()[0] * multiplier,
                                image.extent()[1] * multiplier,
                                1,
                            ],
                            usage,
                            initial_layout: ImageLayout::Undefined,
                            ..Default::default()
                        },
                        AllocationCreateInfo {
                            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                            ..Default::default()
                        },
                    )?;
                    blit_path.push(buffer);
                    multiplier /= 2;
                }
                blit_path.push(image.clone());
                blit_path
            };

            // We always render into the first image in the blit path
            let ssaa_color_view = ImageView::new_default(blit_path[0].clone())?;

            let ssaa_framebuffer = Framebuffer::new(
                ssaa_renderpass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![ssaa_color_view, ssaa_depth_view],
                    ..Default::default()
                },
            )?;
            let post_blit_framebuffer = Framebuffer::new(
                post_blit_renderpass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )?;

            Ok(FramebufferHolder {
                supersampling,
                ssaa_framebuffer,
                blit_path,
                post_blit_framebuffer,
            })
        })
        .collect::<Result<Vec<_>>>()
}

pub(crate) struct Texture2DHolder {
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    sampler: Arc<Sampler>,
    image_view: Arc<ImageView>,
    dimensions: [u32; 2],
}
impl Texture2DHolder {
    // Creates a texture, uploads it to the device, and returns a TextureHolder
    pub(crate) fn create(
        ctx: &VulkanContext,
        image: &image::DynamicImage,
    ) -> Result<Texture2DHolder> {
        let img_rgba = image
            .as_rgba8()
            .with_context(|| "rgba8 buffer was empty")?
            .clone()
            .into_vec();

        let dimensions = image.dimensions();
        let image = Image::new(
            ctx.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_SRGB,
                view_formats: vec![Format::R8G8B8A8_SRGB],
                extent: [image.width(), image.height(), 1],
                usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )?;

        let region = BufferImageCopy {
            buffer_offset: 0,
            image_subresource: ImageSubresourceLayers::from_parameters(Format::R8G8B8A8_SRGB, 1),
            image_extent: [dimensions.0, dimensions.1, 1],
            ..Default::default()
        };

        let source = Buffer::from_iter(
            ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            img_rgba.iter().cloned(),
        )?;

        let mut copy_builder = AutoCommandBufferBuilder::primary(
            ctx.command_buffer_allocator.clone(),
            ctx.transfer_queue.queue_family_index(),
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )?;

        copy_builder.copy_buffer_to_image(CopyBufferToImageInfo {
            regions: smallvec![region],
            ..CopyBufferToImageInfo::buffer_image(source, image.clone())
        })?;

        copy_builder
            .build()?
            .execute(ctx.transfer_queue.clone())?
            .flush()?;

        let sampler = Sampler::new(
            ctx.vk_device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Linear,
                ..Default::default()
            },
        )?;
        let image_view = ImageView::new_default(image)?;

        Ok(Texture2DHolder {
            descriptor_set_allocator: ctx.descriptor_set_allocator.clone(),
            sampler,
            image_view,
            dimensions: [dimensions.0, dimensions.1],
        })
    }

    fn descriptor_set(
        &self,
        pipeline: &GraphicsPipeline,
        set: usize,
        binding: u32,
    ) -> Result<Arc<DescriptorSet>> {
        let layout = pipeline
            .layout()
            .set_layouts()
            .get(set)
            .with_context(|| "uniform set missing")?;
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [self.write_descriptor_set(binding)],
            [],
        )?;
        Ok(descriptor_set)
    }

    pub(crate) fn write_descriptor_set(&self, binding: u32) -> WriteDescriptorSet {
        WriteDescriptorSet::image_view_sampler(
            binding,
            self.image_view.clone(),
            self.sampler.clone(),
        )
    }

    pub(crate) fn dimensions(&self) -> (u32, u32) {
        (self.dimensions[0], self.dimensions[1])
    }
    pub(crate) fn clone_image_view(&self) -> Arc<ImageView> {
        self.image_view.clone()
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RectF32 {
    l: f32,
    t: f32,
    w: f32,
    h: f32,
}
impl RectF32 {
    fn new(l: f32, t: f32, w: f32, h: f32) -> Self {
        Self { l, t, w, h }
    }
    fn top(&self) -> f32 {
        self.t
    }
    fn bottom(&self) -> f32 {
        self.t + self.h
    }
    fn left(&self) -> f32 {
        self.l
    }
    fn right(&self) -> f32 {
        self.l + self.w
    }

    fn div(&self, dimensions: (u32, u32)) -> RectF32 {
        RectF32::new(
            self.l / dimensions.0 as f32,
            self.t / dimensions.1 as f32,
            self.w / dimensions.0 as f32,
            self.h / dimensions.1 as f32,
        )
    }
}
impl From<Rect> for RectF32 {
    fn from(rect: Rect) -> Self {
        Self::new(rect.x as f32, rect.y as f32, rect.w as f32, rect.h as f32)
    }
}
impl From<&Rect> for RectF32 {
    fn from(rect: &Rect) -> Self {
        (*rect).into()
    }
}
