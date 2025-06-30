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
mod atlas;
pub mod gpu_chunk_table;
pub(crate) mod raytrace_buffer;

use anyhow::{bail, ensure, Context, Result};
use arc_swap::ArcSwap;
use clap::error::ContextKind::Usage;
use image::GenericImageView;
use log::warn;
use parking_lot::Mutex;
use smallvec::smallvec;
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};
use std::{ops::Deref, sync::Arc};
use texture_packer::Rect;
use tracy_client::span;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocatorCreateInfo;
use vulkano::command_buffer::{
    BlitImageInfo, BufferImageCopy, CommandBufferUsage, CopyBufferInfo, CopyBufferToImageInfo,
    SubpassBeginInfo,
};
use vulkano::descriptor_set::DescriptorSet;
use vulkano::format::NumericFormat;
use vulkano::image::sampler::{Filter, Sampler, SamplerCreateInfo};
use vulkano::image::view::ImageViewCreateInfo;
use vulkano::image::{
    AllocateImageError, Image, ImageAspects, ImageCreateInfo, ImageLayout, ImageSubresourceLayers,
    ImageSubresourceRange, ImageType,
};
use vulkano::memory::allocator::{
    AllocationCreateInfo, GenericMemoryAllocatorCreateInfo, MemoryTypeFilter,
};
use vulkano::memory::{MemoryProperties, MemoryPropertyFlags};
use vulkano::pipeline::graphics::viewport::{Scissor, ViewportState};
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

use self::util::select_physical_device;
use crate::client_state::settings::{GameSettings, Supersampling};
use crate::vulkan::shaders::raytracer::TexRef;
use crate::vulkan::shaders::LiveRenderConfig;

pub(crate) type VkAllocator = GenericMemoryAllocator<BuddyAllocator>;

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
    depth_stencil_format: Format,

    swapchain_len: usize,
    /// For settings, all GPUs detected on this machine
    all_gpus: Vec<String>,
    max_draw_indexed_index_value: u32,

    pub(crate) reclaim_u32: BufferReclaim<u32>,
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
        if self
            .depth_stencil_format
            .aspects()
            .contains(ImageAspects::STENCIL)
        {
            ClearValue::DepthStencil((1.0, 0))
        } else {
            ClearValue::Depth(1.0)
        }
    }

    pub(crate) fn val_to_device_via_staging<T: BufferContents>(
        &self,
        data: T,
        usage: BufferUsage,
    ) -> Result<Subbuffer<T>> {
        let staging_buffer = {
            let _span = span!("build staging buffer");
            Buffer::from_data(
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
            )?
        };
        let target_buffer = {
            let _span = span!("build target buffer");
            Buffer::new_sized(
                self.clone_allocator(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_DST | usage,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )?
        };

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
        let fut = {
            let _span = span!("build target buffer future");
            command_buffer.build()?.execute(transfer_queue)?
        };
        {
            let _span = span!("flush target buffer future");
            fut.flush()?
        }
        Ok(target_buffer)
    }

    pub(crate) fn iter_to_device_via_staging<T: BufferContents>(
        &self,
        data: impl ExactSizeIterator<Item = T>,
        usage: BufferUsage,
    ) -> Result<Subbuffer<[T]>> {
        let staging_buffer = {
            let _span = span!("build staging buffer");
            Buffer::from_iter(
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
            )?
        };
        let target_buffer = {
            let _span = span!("build target buffer");
            Buffer::new_slice(
                self.clone_allocator(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_DST | usage,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                staging_buffer.len(),
            )?
        };

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
        let fut = {
            let _span = span!("build target buffer future");
            command_buffer.build()?.execute(transfer_queue)?
        };
        {
            let _span = span!("flush target buffer future");
            fut.flush()?
        }

        Ok(target_buffer)
    }

    pub(crate) fn iter_to_device_via_staging_with_reclaim<T: BufferContents>(
        &self,
        data: impl ExactSizeIterator<Item = T>,
        reclaim_type: ReclaimType,
        reclaim: &BufferReclaim<T>,
        size_class: DeviceSize,
    ) -> Result<ReclaimableBuffer<T>> {
        ensure!(data.len() >= size_class as usize);
        let staging_buffer = {
            let _span = span!("build staging buffer");
            let buf = reclaim.take_or_create(&self, ReclaimType::CpuTransferSrc, size_class)?;
            let mut guard = buf.buffer.write()?;
            for (src, dst) in data.zip(guard.iter_mut()) {
                *dst = src;
            }
            drop(guard);
            buf
        };

        let target_buffer = {
            let _span = span!("build target buffer");
            reclaim.take_or_create(&self, reclaim_type, size_class)?
        };

        let transfer_queue = self.clone_transfer_queue();
        let mut command_buffer = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator(),
            transfer_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        command_buffer.copy_buffer(CopyBufferInfo::buffers(
            staging_buffer.buffer.clone(),
            target_buffer.buffer.clone(),
        ))?;
        let fut = {
            let _span = span!("build target buffer future");
            command_buffer.build()?.execute(transfer_queue)?
        };
        {
            let _span = span!("flush target buffer future");
            fut.flush()?
        }
        reclaim.give_buffer(staging_buffer, None, Duration::from_secs(1));

        Ok(target_buffer)
    }

    pub(crate) fn copy_to_device<T: ?Sized>(
        &self,
        src: Subbuffer<T>,
        dst: Subbuffer<T>,
    ) -> Result<()> {
        let transfer_queue = self.clone_transfer_queue();
        let mut command_buffer = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator(),
            transfer_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        command_buffer.copy_buffer(CopyBufferInfo::buffers(src, dst.clone()))?;
        let fut = {
            let _span = span!("build target buffer future");
            command_buffer.build()?.execute(transfer_queue)?
        };
        {
            let _span = span!("flush target buffer future");
            fut.flush()?
        }
        Ok(())
    }
}

pub(crate) struct VulkanWindow {
    vk_ctx: Arc<VulkanContext>,
    clear_color_depth_render_pass: Arc<RenderPass>,
    color_depth_render_pass: Arc<RenderPass>,
    write_color_read_depth_render_pass: Arc<RenderPass>,
    color_only_render_pass: Arc<RenderPass>,
    depth_stencil_only_clearing_render_pass: Arc<RenderPass>,
    depth_stencil_only_render_pass: Arc<RenderPass>,
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

        let render_config = settings.load().render.build_global_config();

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
                    image_count = caps.min_image_count.max(3);
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
        let swapchain_len = swapchain_images.len();

        let depth_stencil_format = find_best_depth_format(&physical_device)?;

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

        let clear_color_depth_render_pass = make_clearing_raster_render_pass(
            vk_device.clone(),
            color_format,
            depth_stencil_format,
        )?;
        let color_depth_render_pass =
            make_color_depth_render_pass(vk_device.clone(), color_format, depth_stencil_format)?;
        let write_color_read_depth_render_pass = make_write_color_read_depth_render_pass(
            vk_device.clone(),
            color_format,
            depth_stencil_format,
        )?;
        let color_only_render_pass = make_color_only_renderpass(vk_device.clone(), color_format)?;
        let depth_stencil_only_render_pass =
            make_depth_only_renderpass(vk_device.clone(), depth_stencil_format)?;
        let depth_stencil_only_clearing_render_pass =
            make_depth_only_clearing_renderpass(vk_device.clone(), depth_stencil_format)?;

        let framebuffers = get_framebuffers_with_depth(
            &swapchain_images,
            memory_allocator.clone(),
            clear_color_depth_render_pass.clone(),
            write_color_read_depth_render_pass.clone(),
            color_only_render_pass.clone(),
            depth_stencil_only_clearing_render_pass.clone(),
            color_format,
            depth_stencil_format,
            render_config,
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
                depth_stencil_format,
                all_gpus,
                swapchain_len,
                max_draw_indexed_index_value: physical_device
                    .properties()
                    .max_draw_indexed_index_value,
                reclaim_u32: BufferReclaim::new(),
            }),
            clear_color_depth_render_pass,
            color_depth_render_pass,
            write_color_read_depth_render_pass,
            color_only_render_pass,
            depth_stencil_only_render_pass,
            depth_stencil_only_clearing_render_pass,
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
        config: LiveRenderConfig,
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
            self.clear_color_depth_render_pass.clone(),
            self.write_color_read_depth_render_pass.clone(),
            self.color_only_render_pass.clone(),
            self.depth_stencil_only_clearing_render_pass.clone(),
            self.color_format,
            self.depth_stencil_format,
            config,
        )?;
        Ok(())
    }

    fn start_raster_render_pass_and_clear(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        framebuffer: &FramebufferHolder,
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
                render_pass: self.clear_color_depth_render_pass.clone(),
                ..RenderPassBeginInfo::framebuffer(framebuffer.color_depth_stencil_pre_blit.clone())
            },
            SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..Default::default()
            },
        )?;
        Ok(())
    }

    fn start_color_depth_render_pass(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        framebuffer: &FramebufferHolder,
    ) -> Result<()> {
        builder.begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![None, None],
                render_pass: self.color_depth_render_pass.clone(),
                ..RenderPassBeginInfo::framebuffer(framebuffer.color_depth_stencil_pre_blit.clone())
            },
            SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..Default::default()
            },
        )?;
        Ok(())
    }

    fn start_color_write_depth_read_render_pass(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        framebuffer: &FramebufferHolder,
    ) -> Result<()> {
        builder.begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![None, None],
                render_pass: self.write_color_read_depth_render_pass.clone(),
                ..RenderPassBeginInfo::framebuffer(framebuffer.color_read_depth_pre_blit.clone())
            },
            SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..Default::default()
            },
        )?;
        Ok(())
    }

    fn start_color_only_render_pass(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        is_pre_blit: bool,
        framebuffers: &FramebufferHolder,
        contents: SubpassContents,
    ) -> Result<()> {
        let framebuffer = if is_pre_blit {
            framebuffers.color_only_pre_blit.clone()
        } else {
            framebuffers.color_only_post_blit.clone()
        };

        builder.begin_render_pass(
            RenderPassBeginInfo {
                render_pass: self.color_only_render_pass.clone(),
                clear_values: vec![None],
                ..RenderPassBeginInfo::framebuffer(framebuffer)
            },
            SubpassBeginInfo {
                contents,
                ..Default::default()
            },
        )?;
        Ok(())
    }

    fn start_deferred_specular_depth_only_render_pass(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        framebuffer: &FramebufferHolder,
    ) -> Result<()> {
        builder.begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![None],
                render_pass: self.depth_stencil_only_render_pass.clone(),
                ..RenderPassBeginInfo::framebuffer(
                    framebuffer
                        .deferred_specular_buffers
                        .as_ref()
                        .context("Missing deferred specular buffers")?
                        .specular_framebuffer
                        .clone(),
                )
            },
            SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..Default::default()
            },
        )?;
        Ok(())
    }

    fn start_deferred_specular_depth_only_render_pass_and_clear(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        framebuffer: &FramebufferHolder,
    ) -> Result<()> {
        builder.begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some(self.depth_clear_value())],
                render_pass: self.depth_stencil_only_clearing_render_pass.clone(),
                ..RenderPassBeginInfo::framebuffer(
                    framebuffer
                        .deferred_specular_buffers
                        .as_ref()
                        .context("Missing deferred specular buffers")?
                        .specular_framebuffer
                        .clone(),
                )
            },
            SubpassBeginInfo {
                contents: SubpassContents::Inline,
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

    pub(crate) fn color_only_render_pass(&self) -> Arc<RenderPass> {
        self.color_only_render_pass.clone()
    }
}

fn find_best_depth_format(physical_device: &PhysicalDevice) -> Result<Format> {
    const FORMATS_TO_TRY: [Format; 2] = [Format::D24_UNORM_S8_UINT, Format::D32_SFLOAT_S8_UINT];
    for format in FORMATS_TO_TRY {
        if physical_device
            .format_properties(format)?
            .optimal_tiling_features
            .contains(FormatFeatures::DEPTH_STENCIL_ATTACHMENT)
        {
            log::info!("Depth format found: {format:?}");
            return Ok(format);
        }
    }
    bail!("No depth format found");
}

pub(crate) fn make_clearing_raster_render_pass(
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
            depth_stencil: {
                format: depth_format,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        passes: [
            {
                color: [color_ssaa],
                depth_stencil: {depth_stencil},
                input: [],
            },
        ]
    )
    .context("Renderpass creation failed")
}

pub(crate) fn make_color_depth_render_pass(
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
                load_op: Load,
                store_op: Store,
            },
            depth_stencil: {
                format: depth_format,
                samples: 1,
                load_op: Load,
                store_op: Store,
            },
        },
        passes: [
            {
                color: [color_ssaa],
                depth_stencil: {depth_stencil},
                input: [],
            },
        ]
    )
    .context("Renderpass creation failed")
}

pub(crate) fn make_write_color_read_depth_render_pass(
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
                load_op: Load,
                store_op: Store,
            },
            depth_stencil: {
                format: depth_format,
                samples: 1,
                load_op: Load,
                store_op: Store,
            },
        },
        passes: [
            {
                color: [color_ssaa],
                depth_stencil: {},
                input: [depth_stencil],
            },
        ]
    )
    .context("Renderpass creation failed")
}

fn make_color_only_renderpass(
    vk_device: Arc<Device>,
    output_format: Format,
) -> Result<Arc<RenderPass>> {
    vulkano::ordered_passes_renderpass!(
        vk_device,
        attachments: {
            color: {
                format: output_format,
                samples: 1,
                load_op: Load,
                store_op: Store,
            },
        },
        passes: [
            {
                color: [color],
                depth_stencil: {},
                input: [],
            },
        ]
    )
    .context("Renderpass creation failed")
}

fn make_depth_only_renderpass(
    vk_device: Arc<Device>,
    depth_format: Format,
) -> Result<Arc<RenderPass>> {
    vulkano::ordered_passes_renderpass!(
        vk_device,
        attachments: {
            depth: {
                format: depth_format,
                samples: 1,
                load_op: Load,
                store_op: Store,
            },
        },
        passes: [
            {
                color: [],
                depth_stencil: {depth},
                input: [],
            },
        ]
    )
    .context("Renderpass creation failed")
}

fn make_depth_only_clearing_renderpass(
    vk_device: Arc<Device>,
    depth_format: Format,
) -> Result<Arc<RenderPass>> {
    vulkano::ordered_passes_renderpass!(
        vk_device,
        attachments: {
            depth: {
                format: depth_format,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        passes: [
            {
                color: [],
                depth_stencil: {depth},
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

    formats
        .iter()
        .find(|(format, space)| {
            *space == ColorSpace::SrgbNonLinear
                && format.numeric_format_color() == Some(NumericFormat::SRGB)
        })
        .cloned()
        .with_context(|| "Could not find an image format")
}

#[derive(Clone)]
pub(crate) struct DeferredSpecularBuffers {
    // R8G8B8A8_UNORM. We have three components, RGB isn't a supported type, so we use RGBA.
    // Not clear what alpha can be used for in general; for now alpha = 0.0 will prevent any
    // specular raytracing from occurring (i.e. it's a validity bit, but we can reclaim it at the
    // cost of VRAM BW and use specular_ray_dir.alpha instead). Always uses main target resolution
    // storage image usage
    specular_strength: Arc<ImageView>,
    // The direction of the ray, corresponding to spec_ray_dir
    // R32G32B32A32_UINT - this would naturally be R32G32B32, but that's not well-supported.
    // Note that the ray origin is derived based on the depth buffer.
    // r/g/b are x/y/z reinterpreted as uint32, alpha is the block id
    // Always uses main target resolution
    // storage image usage
    specular_ray_dir: Arc<ImageView>,
    // Direction of the ray, but cleaned up for each downsampled pixel. `mask_specular_stencil_frag`
    // prepares this.
    specular_ray_dir_downsampled: Arc<ImageView>,
    // The actual computed color, in R8G8B8U8_UNORM. Smaller resolution; raytracer will render to
    // this when doing the deferred specular pass
    // Is a storage | transfer_dst
    specular_raw_color: Arc<ImageView>,
    // Native depth type (we really only need the stencil but VK_FORMAT_S8_UINT isn't guaranteed, so
    // we use the standard depth type for now). Smaller resolution
    // depth attachment usage
    specular_stencil: Arc<ImageView>,
    // A depth/stencil only framebuffer
    specular_framebuffer: Arc<Framebuffer>,
}

#[derive(Clone)]
pub(crate) struct FramebufferHolder {
    image_i: usize,
    supersampling: Supersampling,
    color_depth_stencil_pre_blit: Arc<Framebuffer>,
    color_read_depth_pre_blit: Arc<Framebuffer>,
    color_only_pre_blit: Arc<Framebuffer>,
    deferred_specular_buffers: Option<DeferredSpecularBuffers>,
    // Vector starting with the source and ending with the target. It may be a singleton, but must
    // not be empty.
    blit_path: Vec<Arc<Image>>,
    color_only_post_blit: Arc<Framebuffer>,
    depth_only_view: Arc<ImageView>,
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

pub(crate) fn get_framebuffers_with_depth(
    images: &[Arc<Image>],
    allocator: Arc<VkAllocator>,
    raster_render_pass: Arc<RenderPass>,
    write_color_read_depth_render_pass: Arc<RenderPass>,
    color_only_render_pass: Arc<RenderPass>,
    depth_stencil_only_render_pass: Arc<RenderPass>,
    color_format: Format,
    depth_format: Format,
    config: LiveRenderConfig,
) -> Result<Vec<FramebufferHolder>> {
    images
        .iter()
        .enumerate()
        .map(|(image_i, final_image)| -> anyhow::Result<_> {
            let final_image_view = ImageView::new_default(final_image.clone()).unwrap();
            let image_extent = final_image.extent();
            let (depth_stencil_view, depth_only_view) = make_depth_buffer_and_attachments(
                allocator.clone(),
                depth_format,
                config.supersampling,
                image_extent,
            )?;
            // Now build the blit path
            let blit_path = if config.supersampling == Supersampling::None {
                // The blit path is trivial - just the original image
                // In fact, we do no blitting here - see the implementation of FramebufferHolder
                vec![final_image.clone()]
            } else {
                let mut blit_path = vec![];
                let mut multiplier = config.supersampling.to_int();

                for i in 0..config.supersampling.blit_steps() {
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
                            format: final_image.format(),
                            view_formats: vec![final_image.format()],
                            extent: [
                                image_extent[0] * multiplier,
                                image_extent[1] * multiplier,
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
                blit_path.push(final_image.clone());
                blit_path
            };

            // We always render into the first image in the blit path
            let color_view = ImageView::new_default(blit_path[0].clone())?;

            let color_depth_stencil_pre_blit = Framebuffer::new(
                raster_render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![color_view.clone(), depth_stencil_view.clone()],
                    ..Default::default()
                },
            )?;

            let color_read_depth_pre_blit = Framebuffer::new(
                write_color_read_depth_render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![color_view.clone(), depth_only_view.clone()],
                    ..Default::default()
                },
            )?;

            let color_only_pre_blit = Framebuffer::new(
                color_only_render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![color_view],
                    ..Default::default()
                },
            )?;

            let color_only_post_blit = Framebuffer::new(
                color_only_render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![final_image_view],
                    ..Default::default()
                },
            )?;

            let deferred_specular_buffers = if config.raytracing {
                let extent = [
                    image_extent[0] * config.supersampling.to_int(),
                    image_extent[1] * config.supersampling.to_int(),
                    1,
                ];
                let downsampled_extent = [
                    image_extent[0] * config.supersampling.to_int()
                        / config.raytracing_specular_downsampling,
                    image_extent[1] * config.supersampling.to_int()
                        / config.raytracing_specular_downsampling,
                    1,
                ];

                let specular_raw_color = make_image_and_view(
                    allocator.clone(),
                    downsampled_extent,
                    Format::R8G8B8A8_UNORM,
                    ImageUsage::STORAGE | ImageUsage::TRANSFER_DST,
                )?;
                let specular_stencil = make_image_and_view(
                    allocator.clone(),
                    downsampled_extent,
                    depth_format,
                    ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                )?;
                Some(DeferredSpecularBuffers {
                    specular_strength: make_image_and_view(
                        allocator.clone(),
                        extent,
                        Format::R8G8B8A8_UNORM,
                        ImageUsage::STORAGE,
                    )?,
                    specular_ray_dir: make_image_and_view(
                        allocator.clone(),
                        extent,
                        Format::R32G32B32A32_UINT,
                        ImageUsage::STORAGE,
                    )?,
                    specular_ray_dir_downsampled: make_image_and_view(
                        allocator.clone(),
                        downsampled_extent,
                        Format::R32G32B32A32_UINT,
                        ImageUsage::STORAGE,
                    )?,
                    specular_framebuffer: Framebuffer::new(
                        depth_stencil_only_render_pass.clone(),
                        FramebufferCreateInfo {
                            attachments: vec![specular_stencil.clone()],
                            ..Default::default()
                        },
                    )?,
                    specular_raw_color,
                    specular_stencil,
                })
            } else {
                None
            };

            Ok(FramebufferHolder {
                image_i,
                supersampling: config.supersampling,
                color_read_depth_pre_blit,
                color_depth_stencil_pre_blit,
                color_only_pre_blit,
                deferred_specular_buffers,
                blit_path,
                color_only_post_blit,
                depth_only_view,
            })
        })
        .collect::<Result<Vec<_>>>()
}

fn make_image_and_view(
    allocator: Arc<VkAllocator>,
    extent: [u32; 3],
    format: Format,
    usage: ImageUsage,
) -> Result<Arc<ImageView>> {
    let image = Image::new(
        allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format,
            view_formats: vec![format],
            extent,
            usage,
            initial_layout: ImageLayout::Undefined,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )?;
    Ok(ImageView::new_default(image)?)
}

pub(crate) fn make_depth_buffer_and_attachments(
    allocator: Arc<VkAllocator>,
    depth_stencil_format: Format,
    supersampling: Supersampling,
    image_extent: [u32; 3],
) -> Result<(Arc<ImageView>, Arc<ImageView>)> {
    let depth_stencil_buffer = Image::new(
        allocator,
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: depth_stencil_format,
            view_formats: vec![depth_stencil_format],
            extent: [
                image_extent[0] * supersampling.to_int(),
                image_extent[1] * supersampling.to_int(),
                1,
            ],
            usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT
                | ImageUsage::INPUT_ATTACHMENT
                | ImageUsage::TRANSIENT_ATTACHMENT,
            initial_layout: ImageLayout::Undefined,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )?;
    let depth_stencil_view = ImageView::new_default(depth_stencil_buffer.clone())?;
    let depth_only_create_info = ImageViewCreateInfo {
        subresource_range: ImageSubresourceRange {
            aspects: ImageAspects::DEPTH,
            ..depth_stencil_buffer.subresource_range()
        },
        ..ImageViewCreateInfo::from_image(&depth_stencil_buffer)
    };
    let depth_only_view = ImageView::new(depth_stencil_buffer.clone(), depth_only_create_info)?;
    Ok((depth_stencil_view, depth_only_view))
}

pub(crate) struct Texture2DHolder {
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    sampler: Arc<Sampler>,
    image_view: Arc<ImageView>,
    dimensions: [u32; 2],
}
impl Texture2DHolder {
    /// Creates a texture, uploads it to the device, and returns a TextureHolder
    /// The image should be in SRGB.
    pub(crate) fn from_srgb(
        ctx: &VulkanContext,
        image: image::RgbaImage,
    ) -> Result<Texture2DHolder> {
        let dimensions = image.dimensions();
        let img_rgba = image.into_vec();

        let image = Image::new(
            ctx.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_SRGB,
                view_formats: vec![Format::R8G8B8A8_SRGB],
                extent: [dimensions.0, dimensions.1, 1],
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
impl From<RectF32> for TexRef {
    fn from(value: RectF32) -> Self {
        TexRef {
            // reorder here to avoid extra swizzles in the shader
            top_left: [value.l, value.t],
            width_height: [value.w, value.h],
        }
    }
}

#[repr(usize)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum ReclaimType {
    CpuTransferSrc = 0,
    GpuSsboTransferDst = 1,
}

impl ReclaimType {
    fn buffer_create_info(&self) -> BufferCreateInfo {
        match self {
            ReclaimType::CpuTransferSrc => BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            ReclaimType::GpuSsboTransferDst => BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
        }
    }
    fn allocation_create_info(&self) -> AllocationCreateInfo {
        match self {
            ReclaimType::CpuTransferSrc => AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            ReclaimType::GpuSsboTransferDst => AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        }
    }
}

pub(crate) struct ReclaimableBuffer<T> {
    reclaim_type: ReclaimType,
    buffer: Subbuffer<[T]>,
    expiration: Instant,
    sequester: Option<usize>,
}

impl<T> Deref for ReclaimableBuffer<T> {
    type Target = Subbuffer<[T]>;
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

struct ReclaimInner<T> {
    pending: Vec<ReclaimableBuffer<T>>,
    ready: Vec<ReclaimBuffersByType<T>>,
}

pub(crate) struct BufferReclaim<T> {
    inner: Mutex<ReclaimInner<T>>,
}

impl<T: BufferContents> BufferReclaim<T> {
    pub(crate) fn take_or_create(
        &self,
        ctx: &VulkanContext,
        reclaim_type: ReclaimType,
        capacity: DeviceSize,
    ) -> Result<ReclaimableBuffer<T>> {
        if let Some(x) = self.take_buffer(reclaim_type, capacity) {
            Ok(x)
        } else {
            let inner = Buffer::new_slice(
                ctx.memory_allocator.clone(),
                reclaim_type.buffer_create_info(),
                reclaim_type.allocation_create_info(),
                capacity,
            )?;
            Ok(ReclaimableBuffer {
                reclaim_type,
                buffer: inner,
                // doesn't matter, will be updated when received for reclaim
                expiration: Instant::now(),
                sequester: None,
            })
        }
    }
    fn new() -> BufferReclaim<T> {
        BufferReclaim {
            inner: Mutex::new(ReclaimInner {
                pending: vec![],
                ready: vec![ReclaimBuffersByType::new(), ReclaimBuffersByType::new()],
            }),
        }
    }

    fn give_buffer(
        &self,
        mut buffer: ReclaimableBuffer<T>,
        sequester: Option<usize>,
        ttl: Duration,
    ) {
        buffer.expiration = Instant::now() + ttl;
        buffer.sequester = sequester;
        let mut inner = self.inner.lock();
        Self::clean_expired(&mut inner);
        if buffer.sequester.is_some() {
            inner.pending.push(buffer);
        } else {
            inner.ready[buffer.reclaim_type as usize].give_buffer(buffer);
        }
    }

    fn take_buffer(
        &self,
        reclaim_type: ReclaimType,
        size: DeviceSize,
    ) -> Option<ReclaimableBuffer<T>> {
        let mut inner = self.inner.lock();
        Self::clean_expired(&mut inner);
        inner.ready[reclaim_type as usize].take_buffer(size)
    }

    fn clean_expired(inner: &mut ReclaimInner<T>) {
        clean_expired(Instant::now(), &mut inner.pending);
    }
    fn unsequester(&self, frame: usize) {
        let mut inner = self.inner.lock();
        Self::clean_expired(&mut inner);
        let mut i = 0;
        while i < inner.pending.len() {
            let candidate = &inner.pending[i];
            if candidate.sequester == Some(frame) {
                let buffer = inner.pending.swap_remove(i);
                inner.ready[buffer.reclaim_type as usize].give_buffer(buffer);
            } else {
                i += 1;
            }
        }
    }
}

fn clean_expired<T>(exp: Instant, buffers: &mut Vec<ReclaimableBuffer<T>>) {
    let mut i = 0;
    while i < buffers.len() {
        let candidate = &buffers[i];
        if candidate.expiration < exp {
            drop(buffers.swap_remove(i));
        } else {
            i += 1;
        }
    }
}

struct ReclaimBuffersByType<T> {
    by_size_class: BTreeMap<DeviceSize, Vec<ReclaimableBuffer<T>>>,
}

impl<T> ReclaimBuffersByType<T> {
    fn new() -> Self {
        Self {
            by_size_class: Default::default(),
        }
    }
    fn give_buffer(&mut self, mut buffer: ReclaimableBuffer<T>) {
        match self.by_size_class.entry(buffer.len()) {
            Entry::Vacant(x) => {
                x.insert(vec![buffer]);
            }
            Entry::Occupied(x) => {
                x.into_mut().push(buffer);
            }
        }
    }

    fn take_buffer(&mut self, size: DeviceSize) -> Option<ReclaimableBuffer<T>> {
        self.by_size_class.get_mut(&size).and_then(|x| x.pop())
    }

    fn clean_expired(&mut self) {
        let now = Instant::now();
        for class in self.by_size_class.values_mut() {
            clean_expired(now, class);
        }
    }
}
