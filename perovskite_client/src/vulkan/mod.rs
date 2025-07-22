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

use anyhow::{bail, ensure, Context, Error, Result};
use arc_swap::ArcSwap;
use clap::error::ContextKind::Usage;
use enum_map::EnumMap;
use image::GenericImageView;
use log::warn;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use smallvec::smallvec;
use std::collections::hash_map::Entry;
use std::collections::BTreeMap;
use std::fmt::{Display, Formatter, Write};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};
use std::{ops::Deref, sync::Arc};
use texture_packer::Rect;
use tinyvec::array_vec;
use tracy_client::span;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocatorCreateInfo;
use vulkano::command_buffer::{
    BlitImageInfo, BufferImageCopy, CommandBufferUsage, CopyBufferInfo, CopyBufferToImageInfo,
    CopyImageInfo, SubpassBeginInfo,
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
use vulkano::render_pass::{
    AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp,
    RenderPassCreateInfo, SubpassDescription,
};
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
use crate::vulkan::shaders::cube_geometry::CubeGeometryVertex;
use crate::vulkan::shaders::raytracer::{
    TexRef, RAYTRACING_REQUIRED_EXTENSIONS, RAYTRACING_REQUIRED_FEATURES,
};
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
    swapchain_format: Format,
    /// Probed at startup
    depth_stencil_format: Format,

    swapchain_len: usize,
    /// For settings, all GPUs detected on this machine
    all_gpus: Vec<String>,
    max_draw_indexed_index_value: u32,

    u32_reclaimer: Arc<BufferReclaim<u32>>,
    cgv_reclaimer: Arc<BufferReclaim<CubeGeometryVertex>>,

    raytracing_supported: bool,
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

    pub(crate) fn current_gpu_name(&self) -> &str {
        self.vk_device
            .physical_device()
            .properties()
            .device_name
            .as_ref()
    }

    pub(crate) fn all_gpus(&self) -> &[String] {
        &self.all_gpus
    }

    pub(crate) fn raytracing_supported(&self) -> bool {
        self.raytracing_supported
    }

    pub(crate) fn swapchain_format(&self) -> Format {
        self.swapchain_format
    }

    pub(crate) fn depth_stencil_format(&self) -> Format {
        self.depth_stencil_format
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

    pub(crate) fn iter_to_device_via_staging_with_reclaim_and_flush<T: BufferContents>(
        &self,
        data: impl ExactSizeIterator<Item = T>,
        reclaim_type: ReclaimType,
        reclaim: Arc<BufferReclaim<T>>,
        size_class: DeviceSize,
    ) -> Result<ReclaimableBuffer<T>> {
        let mut command_buffer = self.start_transfer_buffer()?;

        let target_buffer = self.iter_to_device_via_staging_with_reclaim(
            data,
            reclaim_type,
            reclaim.clone(),
            size_class,
            &mut command_buffer,
        )?;
        self.finish_transfer_buffer(command_buffer)?;
        Ok(target_buffer)
    }

    pub(crate) fn start_transfer_buffer(&self) -> Result<TransferBuffer> {
        let transfer_queue = self.clone_transfer_queue();

        Ok(TransferBuffer {
            builder: AutoCommandBufferBuilder::primary(
                self.command_buffer_allocator(),
                transfer_queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )?,
            cleanups: vec![],
        })
    }

    pub(crate) fn finish_transfer_buffer(&self, mut buf: TransferBuffer) -> Result<()> {
        let fut = {
            let _span = span!("build target buffer future");
            buf.builder.build()?.execute(self.transfer_queue.clone())?
        };
        {
            let _span = span!("flush target buffer future");
            fut.flush()?
        }
        {
            let _span = span!("transfer buffer cleanups");
            for cleanup in buf.cleanups.drain(..) {
                cleanup();
            }
        }

        Ok(())
    }

    pub(crate) fn iter_to_device_via_staging_with_reclaim<T: BufferContents>(
        &self,
        data: impl ExactSizeIterator<Item = T>,
        reclaim_type: ReclaimType,
        reclaim: Arc<BufferReclaim<T>>,
        size_class: DeviceSize,
        command_buffer: &mut TransferBuffer,
    ) -> Result<ReclaimableBuffer<T>> {
        let data_len = data.len();
        ensure!(data_len <= size_class as usize);
        let mut staging_buffer = {
            let _span = span!("build staging buffer");
            let buf =
                reclaim.take_or_create_slice(&self, ReclaimType::CpuTransferSrc, size_class)?;
            let mut guard = buf.buffer.write()?;
            for (src, dst) in data.zip(guard.iter_mut()) {
                *dst = src;
            }
            drop(guard);
            buf
        };
        staging_buffer.valid_len = data_len as DeviceSize;

        let mut target_buffer = {
            let _span = span!("build target buffer");
            reclaim.take_or_create_slice(&self, reclaim_type, size_class)?
        };
        target_buffer.valid_len = data_len as DeviceSize;

        command_buffer.builder.copy_buffer(CopyBufferInfo::buffers(
            staging_buffer.buffer.clone(),
            target_buffer.buffer.clone(),
        ))?;
        command_buffer.cleanups.push(Box::new(move || {
            reclaim.give_buffer(staging_buffer, None, Duration::from_secs(5));
        }));
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

    /// Constructs a render config suitable for render-to-texture only
    pub(crate) fn non_swapchain_config(&self) -> LiveRenderConfig {
        LiveRenderConfig {
            supersampling: Supersampling::None,
            hdr: false,
            raytracing: false,
            hybrid_rt: false,
            render_distance: 1,
            raytracer_debug: false,
            raytracing_specular_downsampling: 1,
            blur_steps: 0,
            bloom_strength: 0.0,
            lens_flare_strength: 0.0,
            formats: SelectedFormats {
                swapchain: Format::R8G8B8A8_SRGB,
                color: Format::R8G8B8A8_SRGB,
                depth_stencil: self.depth_stencil_format,
            },
        }
    }

    pub fn cgv_reclaimer(&self) -> &Arc<BufferReclaim<CubeGeometryVertex>> {
        &self.cgv_reclaimer
    }

    pub fn u32_reclaimer(&self) -> &Arc<BufferReclaim<u32>> {
        &self.u32_reclaimer
    }
}

pub(crate) struct TransferBuffer {
    builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    cleanups: Vec<Box<dyn FnOnce()>>,
}

pub(crate) struct RenderPassHolder {
    vk_device: Arc<Device>,
    passes: Mutex<FxHashMap<RenderPassId, Arc<RenderPass>>>,
    config: LiveRenderConfig,
}

impl RenderPassHolder {
    pub(crate) fn get_by_framebuffer_id<const M: usize, const N: usize>(
        &self,
        framebuffer_id: FramebufferAndLoadOpId<M, N>,
    ) -> Result<Arc<RenderPass>> {
        let renderpass_id = RenderPassId {
            color_attachments: framebuffer_id
                .color_attachments
                .iter()
                .map(|(image, op)| (image.image_format(&self.config.formats), *op))
                .collect(),
            depth_stencil_attachment: framebuffer_id
                .depth_stencil_attachment
                .iter()
                .map(|(image, op)| (image.image_format(&self.config.formats), *op))
                .next(),
            input_attachments: framebuffer_id
                .input_attachments
                .iter()
                .map(|(image, op)| (image.image_format(&self.config.formats), *op))
                .collect(),
        };
        self.get(renderpass_id)
    }
}

impl RenderPassHolder {
    pub(crate) fn new(vk_device: Arc<Device>, config: LiveRenderConfig) -> Self {
        Self {
            vk_device,
            config,
            passes: Mutex::new(FxHashMap::default()),
        }
    }
    pub(crate) fn get(&self, id: RenderPassId) -> Result<Arc<RenderPass>> {
        match self.passes.lock().entry(id) {
            Entry::Occupied(entry) => Ok(entry.get().clone()),
            Entry::Vacant(entry) => Ok(entry
                .insert(Self::build_render_pass(id, &self.vk_device)?)
                .clone()),
        }
    }

    fn build_render_pass(id: RenderPassId, vk_device: &Arc<Device>) -> Result<Arc<RenderPass>> {
        log::debug!("Building render pass: {}", id);
        let mut attachments = vec![];
        let mut subpass = SubpassDescription {
            input_attachments: vec![],
            color_attachments: vec![],
            depth_stencil_attachment: None,
            ..Default::default()
        };

        for (format, op) in id.color_attachments {
            let idx = attachments.len();
            attachments.push(AttachmentDescription {
                format,
                load_op: op.to_vulkano(),
                store_op: AttachmentStoreOp::Store,
                initial_layout: ImageLayout::ColorAttachmentOptimal,
                final_layout: ImageLayout::ColorAttachmentOptimal,
                ..Default::default()
            });
            subpass.color_attachments.push(Some(AttachmentReference {
                attachment: idx as u32,
                layout: ImageLayout::ColorAttachmentOptimal,
                ..Default::default()
            }));
        }
        if let Some((format, op)) = id.depth_stencil_attachment {
            let idx = attachments.len();
            attachments.push(AttachmentDescription {
                format,
                load_op: op.to_vulkano(),
                store_op: AttachmentStoreOp::Store,
                initial_layout: ImageLayout::DepthStencilAttachmentOptimal,
                final_layout: ImageLayout::DepthStencilAttachmentOptimal,
                ..Default::default()
            });
            subpass.depth_stencil_attachment = Some(AttachmentReference {
                attachment: idx as u32,
                layout: ImageLayout::DepthStencilAttachmentOptimal,
                ..Default::default()
            });
        }

        for (format, op) in id.input_attachments {
            let idx = attachments.len();
            attachments.push(AttachmentDescription {
                format,
                load_op: op.to_vulkano(),
                store_op: AttachmentStoreOp::Store,
                initial_layout: ImageLayout::ShaderReadOnlyOptimal,
                final_layout: ImageLayout::ShaderReadOnlyOptimal,
                ..Default::default()
            });
            subpass.input_attachments.push(Some(AttachmentReference {
                attachment: idx as u32,
                layout: ImageLayout::ShaderReadOnlyOptimal,
                ..Default::default()
            }));
        }

        let info = RenderPassCreateInfo {
            attachments,
            subpasses: vec![subpass],
            ..Default::default()
        };

        RenderPass::new(vk_device.clone(), info)
            .with_context(|| format!("Failed to build renderpass for {id}"))
    }
}

pub(crate) struct VulkanWindow {
    vk_ctx: Arc<VulkanContext>,
    renderpasses: RenderPassHolder,
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

    pub(crate) fn renderpasses(&self) -> &RenderPassHolder {
        &self.renderpasses
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
                    patch: 2,
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

        let mandatory_extensions = DeviceExtensions {
            khr_swapchain: true,
            khr_storage_buffer_storage_class: true,
            ..DeviceExtensions::empty()
        };
        // For now, no features are mandatory
        let mandatory_features = DeviceFeatures::empty();

        let all_gpus = instance
            .enumerate_physical_devices()?
            .map(|x| x.properties().device_name.clone())
            .collect::<Vec<String>>();

        let (physical_device, graphics_family_index, transfer_family_index) =
            select_physical_device(
                &instance,
                &surface,
                &mandatory_extensions,
                &mandatory_features,
                &settings.load().render.preferred_gpu,
            )?;

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

        let mut enabled_extensions = mandatory_extensions;
        let mut enabled_features = mandatory_features;
        let mut raytracing_supported = false;
        if physical_device
            .supported_features()
            .contains(&RAYTRACING_REQUIRED_FEATURES)
            && physical_device
                .supported_extensions()
                .contains(&RAYTRACING_REQUIRED_EXTENSIONS)
        {
            enabled_extensions |= RAYTRACING_REQUIRED_EXTENSIONS;
            enabled_features |= RAYTRACING_REQUIRED_FEATURES;
            raytracing_supported = true;
        }

        let (vk_device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos,
                enabled_extensions,
                enabled_features,
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

        let (swapchain, swapchain_images, swapchain_color_format) = {
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
            let (image_format, color_space) = best_swapchain_format(formats)?;
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
                    image_usage: ImageId::SwapchainColor.usage(),
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

        let vk_ctx = Arc::new(VulkanContext {
            vk_device,
            graphics_queue,
            transfer_queue,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
            swapchain_format: swapchain_color_format,
            depth_stencil_format,
            all_gpus,
            swapchain_len,
            max_draw_indexed_index_value: physical_device.properties().max_draw_indexed_index_value,
            u32_reclaimer: Arc::new(BufferReclaim::new()),
            cgv_reclaimer: Arc::new(BufferReclaim::new()),
            raytracing_supported,
        });
        let render_config = settings.load().render.build_global_config(&vk_ctx);

        let renderpasses = RenderPassHolder::new(vk_ctx.vk_device.clone(), render_config);

        let framebuffers =
            FramebufferHolder::make_framebuffers(&swapchain_images, vk_ctx.clone(), render_config)?;
        Ok(VulkanWindow {
            vk_ctx,
            renderpasses,
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
        self.renderpasses.config = config;
        self.framebuffers =
            FramebufferHolder::make_framebuffers(&new_images, self.vk_ctx.clone(), config)?;
        Ok(())
    }

    pub(crate) fn window_size(&self) -> (u32, u32) {
        let dims = self.viewport.extent;
        (dims[0] as u32, dims[1] as u32)
    }

    pub(crate) fn swapchain(&self) -> &Swapchain {
        self.swapchain.as_ref()
    }

    pub(crate) fn ui_renderpass(&self) -> Result<Arc<RenderPass>> {
        self.renderpasses
            .get_by_framebuffer_id(FramebufferAndLoadOpId {
                color_attachments: [(ImageId::SwapchainColor, LoadOp::Load)],
                depth_stencil_attachment: None,
                input_attachments: [],
            })
    }

    pub(crate) fn raster_renderpass(&self) -> Result<Arc<RenderPass>> {
        self.renderpasses
            .get_by_framebuffer_id(FramebufferAndLoadOpId {
                color_attachments: [(ImageId::MainColor, LoadOp::Load)],
                depth_stencil_attachment: Some((ImageId::MainDepthStencil, LoadOp::Load)),
                input_attachments: [],
            })
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

fn best_swapchain_format(formats: Vec<(Format, ColorSpace)>) -> Result<(Format, ColorSpace)> {
    formats
        .iter()
        .find(|(format, space)| {
            *space == ColorSpace::SrgbNonLinear
                && format.numeric_format_color() == Some(NumericFormat::SRGB)
        })
        .copied()
        .with_context(|| "Could not find an image format")
}

#[derive(Clone)]
pub(crate) struct RaytraceBuffers {
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
    // The actual computed color, in R16G16B16A16_SFLOAT. Smaller resolution; raytracer will render to
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SelectedFormats {
    pub(crate) swapchain: Format,
    pub(crate) color: Format,
    pub(crate) depth_stencil: Format,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, enum_map::Enum)]
pub(crate) enum ImageId {
    MainColor,
    MainDepthStencil,
    MainDepthStencilDepthOnly,
    /// A special depth buffer to allow glass to have separate depth/overdraw control for its
    /// emission pass: Transparent-but-shiny fragments are discarded in color and don't write to the
    /// main depth buffer; on the second pass they do write to this depth buffer while emitting
    /// reflection information into the deferred reflection buffer
    TransparentWithSpecularDepth,
    MainColorResolved,
    SwapchainColor,
    RtSpecRawColor,
    RtSpecStencil,
    RtSpecStrength,
    RtSpecRayDir,
    RtSpecRayDirDownsampled,
    Blur(u8),
}

impl ImageId {
    fn image_format(&self, f: &SelectedFormats) -> Format {
        match self {
            ImageId::MainColor | ImageId::MainColorResolved => f.color,
            ImageId::MainDepthStencil => f.depth_stencil,
            ImageId::MainDepthStencilDepthOnly => f.depth_stencil,
            ImageId::TransparentWithSpecularDepth => f.depth_stencil,
            ImageId::SwapchainColor => f.swapchain,
            ImageId::RtSpecRawColor => Format::R16G16B16A16_SFLOAT,
            ImageId::RtSpecStencil => f.depth_stencil,
            ImageId::RtSpecStrength => Format::R8G8B8A8_UNORM,
            ImageId::RtSpecRayDir => Format::R32G32B32A32_UINT,
            ImageId::RtSpecRayDirDownsampled => Format::R32G32B32A32_UINT,
            ImageId::Blur(_) => f.color,
        }
    }

    fn dimension(&self, base_x: u32, base_y: u32, config: LiveRenderConfig) -> (u32, u32) {
        let supersampling = config.supersampling.to_int();
        let base = (base_x, base_y);
        let upsampled = (base_x * supersampling, base_y * supersampling);
        let rt_deferred = (
            upsampled.0 / config.raytracing_specular_downsampling,
            upsampled.1 / config.raytracing_specular_downsampling,
        );

        match self {
            ImageId::MainColor => upsampled,
            ImageId::MainDepthStencil => upsampled,
            ImageId::MainDepthStencilDepthOnly => upsampled,
            ImageId::TransparentWithSpecularDepth => upsampled,
            ImageId::MainColorResolved => base,
            ImageId::SwapchainColor => base,
            ImageId::RtSpecRawColor => rt_deferred,
            ImageId::RtSpecStencil => rt_deferred,
            ImageId::RtSpecStrength => upsampled,
            ImageId::RtSpecRayDir => upsampled,
            ImageId::RtSpecRayDirDownsampled => rt_deferred,
            ImageId::Blur(n) => (base.0 >> n, base.1 >> n),
        }
    }

    fn abbreviation(&self) -> &'static str {
        match self {
            // I give up on trying to use a letter for each attachment
            // all the raytrace specular buffers get really annoying abbreviations
            // Have some CJK instead; worthwhile terminals should support them anyway
            ImageId::MainColor => "色",
            ImageId::MainDepthStencil => "深",
            ImageId::MainDepthStencilDepthOnly => "半",
            ImageId::TransparentWithSpecularDepth => "玻",
            ImageId::MainColorResolved => "小",
            ImageId::SwapchainColor => "面",
            ImageId::RtSpecRawColor => "光映",
            ImageId::RtSpecStencil => "光切",
            ImageId::RtSpecStrength => "光艶",
            ImageId::RtSpecRayDir => "光方",
            ImageId::RtSpecRayDirDownsampled => "光角",
            ImageId::Blur(0) => "暈原",
            ImageId::Blur(_) => "暈路",
        }
    }

    fn usage(&self) -> ImageUsage {
        match self {
            ImageId::MainColor => {
                ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::TRANSFER_SRC
                    | ImageUsage::INPUT_ATTACHMENT
            }
            ImageId::MainDepthStencil => {
                ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    | ImageUsage::INPUT_ATTACHMENT
                    | ImageUsage::TRANSFER_SRC
            }
            ImageId::MainDepthStencilDepthOnly => {
                ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    | ImageUsage::INPUT_ATTACHMENT
                    | ImageUsage::TRANSFER_SRC
            }
            ImageId::TransparentWithSpecularDepth => {
                ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSFER_DST
            }
            ImageId::MainColorResolved => {
                ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::TRANSFER_SRC
                    | ImageUsage::TRANSFER_DST
                    | ImageUsage::SAMPLED
            }
            ImageId::SwapchainColor => ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
            ImageId::RtSpecRawColor => ImageUsage::COLOR_ATTACHMENT | ImageUsage::STORAGE,
            ImageId::RtSpecStencil => ImageUsage::DEPTH_STENCIL_ATTACHMENT,
            ImageId::RtSpecStrength => ImageUsage::COLOR_ATTACHMENT | ImageUsage::STORAGE,
            ImageId::RtSpecRayDir => ImageUsage::COLOR_ATTACHMENT | ImageUsage::STORAGE,
            ImageId::RtSpecRayDirDownsampled => ImageUsage::COLOR_ATTACHMENT | ImageUsage::STORAGE,
            ImageId::Blur(0) => {
                ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST
            }
            ImageId::Blur(_) => ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
        }
    }

    pub(crate) fn clear_value(&self) -> ClearValue {
        const FLOAT: ClearValue = ClearValue::Float([0.0; 4]);
        const DEPTH_STENCIL: ClearValue = ClearValue::DepthStencil((1.0, 0));
        const DEPTH: ClearValue = ClearValue::Depth(1.0);
        const UINT: ClearValue = ClearValue::Uint([0; 4]);
        match self {
            ImageId::MainColor | ImageId::MainColorResolved => FLOAT,
            ImageId::MainDepthStencil => DEPTH_STENCIL,
            ImageId::MainDepthStencilDepthOnly => DEPTH,
            ImageId::TransparentWithSpecularDepth => DEPTH_STENCIL,
            ImageId::SwapchainColor => FLOAT,
            ImageId::RtSpecRawColor => FLOAT,
            ImageId::RtSpecStencil => DEPTH_STENCIL,
            ImageId::RtSpecStrength => FLOAT,
            ImageId::RtSpecRayDir => UINT,
            ImageId::RtSpecRayDirDownsampled => UINT,
            ImageId::Blur(_) => FLOAT,
        }
    }

    pub(crate) fn viewport_state(
        &self,
        viewport: &Viewport,
        config: LiveRenderConfig,
    ) -> ViewportState {
        let (x, y) = self.dimension(viewport.extent[0] as u32, viewport.extent[1] as u32, config);
        ViewportState {
            viewports: smallvec![Viewport {
                offset: [0.0, 0.0],
                depth_range: 0.0..=1.0,
                extent: [x as f32, y as f32],
            }],
            scissors: smallvec![Scissor {
                offset: [0, 0],
                extent: [x, y],
            }],
            ..Default::default()
        }
    }
}

impl Default for ImageId {
    fn default() -> Self {
        Self::MainColor
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct FramebufferAndLoadOpId<const M: usize, const N: usize> {
    pub(crate) color_attachments: [(ImageId, LoadOp); M],
    pub(crate) depth_stencil_attachment: Option<(ImageId, LoadOp)>,
    pub(crate) input_attachments: [(ImageId, LoadOp); N],
}
impl<const M: usize, const N: usize> FramebufferAndLoadOpId<M, N> {
    fn framebuffer_id(&self) -> FramebufferId {
        FramebufferId {
            color_attachments: self.color_attachments.iter().map(|x| x.0).collect(),
            depth_stencil_attachment: self.depth_stencil_attachment.map(|x| x.0),
            input_attachments: self.input_attachments.iter().map(|x| x.0).collect(),
        }
    }
}
impl<const M: usize, const N: usize> Display for FramebufferAndLoadOpId<M, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("Co:")?;
        for (attachment, op) in self.color_attachments.iter() {
            f.write_str(attachment.abbreviation())?;
            f.write_char(op.op_char())?
        }
        if let Some((attachment, op)) = &self.depth_stencil_attachment {
            f.write_str("/Dp:")?;
            f.write_str(attachment.abbreviation())?;
            f.write_char(op.op_char())?
        }
        f.write_str("/Rd:")?;
        for ((attachment, op)) in self.input_attachments.iter() {
            f.write_str(attachment.abbreviation())?;
            f.write_char(op.op_char())?
        }
        Ok(())
    }
}

impl<const N: usize> ColorAttachmentsWithinLimits for FramebufferAndLoadOpId<0, N> {}
impl<const N: usize> ColorAttachmentsWithinLimits for FramebufferAndLoadOpId<1, N> {}
impl<const N: usize> ColorAttachmentsWithinLimits for FramebufferAndLoadOpId<2, N> {}
impl<const N: usize> ColorAttachmentsWithinLimits for FramebufferAndLoadOpId<3, N> {}
impl<const N: usize> ColorAttachmentsWithinLimits for FramebufferAndLoadOpId<4, N> {}
impl<const M: usize> InputAttachmentsWithinLimits for FramebufferAndLoadOpId<M, 0> {}
impl<const M: usize> InputAttachmentsWithinLimits for FramebufferAndLoadOpId<M, 1> {}
impl<const M: usize> InputAttachmentsWithinLimits for FramebufferAndLoadOpId<M, 2> {}
impl<const M: usize> InputAttachmentsWithinLimits for FramebufferAndLoadOpId<M, 3> {}
impl<const M: usize> InputAttachmentsWithinLimits for FramebufferAndLoadOpId<M, 4> {}

trait ColorAttachmentsWithinLimits {}

trait InputAttachmentsWithinLimits {}

pub(crate) trait SupportedByVulkanCore {}
impl<T: ColorAttachmentsWithinLimits + InputAttachmentsWithinLimits + ?Sized> SupportedByVulkanCore
    for T
{
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct FramebufferId {
    color_attachments: tinyvec::ArrayVec<[ImageId; 8]>,
    depth_stencil_attachment: Option<ImageId>,
    input_attachments: tinyvec::ArrayVec<[ImageId; 8]>,
}
impl Display for FramebufferId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("Co:")?;
        for attachment in self.color_attachments.iter() {
            f.write_str(attachment.abbreviation())?;
        }
        if let Some(attachment) = &self.depth_stencil_attachment {
            f.write_str("/Dp:")?;
            f.write_str(attachment.abbreviation())?;
        }
        f.write_str("/Rd:")?;
        for (attachment) in self.input_attachments.iter() {
            f.write_str(attachment.abbreviation())?;
        }
        Ok(())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum LoadOp {
    Load,
    DontCare,
    Clear,
}
impl Default for LoadOp {
    fn default() -> Self {
        LoadOp::Load
    }
}
impl LoadOp {
    fn to_vulkano(&self) -> AttachmentLoadOp {
        match self {
            LoadOp::Load => AttachmentLoadOp::Load,
            LoadOp::Clear => AttachmentLoadOp::Clear,
            LoadOp::DontCare => AttachmentLoadOp::DontCare,
        }
    }

    fn op_char(self) -> char {
        match self {
            LoadOp::Load => ' ',
            LoadOp::Clear => '#',
            LoadOp::DontCare => '?',
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct RenderPassId {
    color_attachments: tinyvec::ArrayVec<[(Format, LoadOp); 8]>,
    depth_stencil_attachment: Option<(Format, LoadOp)>,
    input_attachments: tinyvec::ArrayVec<[(Format, LoadOp); 4]>,
}
impl Display for RenderPassId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn lookup(f: Format) -> &'static str {
            match f {
                Format::R8G8B8A8_UNORM => "U08",
                Format::A2R10G10B10_UNORM_PACK32 => "U10",
                Format::R16G16B16A16_SFLOAT => "F16",
                Format::R32G32B32A32_UINT => "I32",
                Format::D32_SFLOAT => "D32",
                Format::D32_SFLOAT_S8_UINT => "Z32",
                Format::D24_UNORM_S8_UINT => "D24",
                Format::X8_D24_UNORM_PACK32 => "Z24",
                Format::R8G8B8A8_SRGB => "s08",
                Format::B8G8R8A8_SRGB => "b08",
                _ => "???",
            }
        }

        f.write_str("Co:")?;
        for (format, op) in self.color_attachments.iter() {
            f.write_str(lookup(*format))?;
            f.write_char(op.op_char())?;
        }
        if let Some((format, op)) = &self.depth_stencil_attachment {
            f.write_str("/Dp:")?;
            f.write_str(lookup(*format))?;
            f.write_char(op.op_char())?;
        }
        f.write_str("/Rd:")?;
        for (format, op) in self.input_attachments.iter() {
            f.write_str(lookup(*format))?;
            f.write_char(op.op_char())?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, enum_map::Enum)]
pub(crate) enum SamplerId {
    Linear(ImageId),
    Nearest(ImageId),
}

pub(crate) struct FramebufferHolder {
    ctx: Arc<VulkanContext>,
    image_i: usize,
    base_extent: [u32; 3],
    config: LiveRenderConfig,
    image_views: Mutex<EnumMap<ImageId, Option<Arc<ImageView>>>>,
    samplers: Mutex<EnumMap<SamplerId, Option<Arc<Sampler>>>>,
    // All images to blit through
    blit_path: Vec<Arc<Image>>,
    framebuffers: Mutex<FxHashMap<FramebufferId, Arc<Framebuffer>>>,
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

    pub(crate) fn blit_final(
        &self,
        command_buf_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> Result<()> {
        command_buf_builder.blit_image(BlitImageInfo {
            filter: Filter::Nearest,
            ..BlitImageInfo::images(
                self.blit_path.last().unwrap().clone(),
                self.try_get_image(ImageId::SwapchainColor)?.image().clone(),
            )
        })?;
        Ok(())
    }

    pub(crate) fn make_framebuffers(
        images: &[Arc<Image>],
        ctx: Arc<VulkanContext>,
        config: LiveRenderConfig,
    ) -> Result<Vec<FramebufferHolder>> {
        images
            .iter()
            .enumerate()
            .map(|(image_i, swapchain_image)| -> anyhow::Result<_> {
                Self::new(ctx.clone(), config, image_i, swapchain_image)
            })
            .collect::<Result<Vec<_>>>()
    }

    /// Returns a framebuffer for the given ID, as well as a renderpass with all color and depth
    /// attachments set to the specified load op. For custom combinations of load ops, the caller
    /// must retrieve their own renderpass from the renderpass holder with a manually specified key.
    pub(crate) fn get_framebuffer<const M: usize, const N: usize>(
        &self,
        framebuffer_id: FramebufferAndLoadOpId<M, N>,
        renderpasses: &RenderPassHolder,
    ) -> Result<(Arc<Framebuffer>, Arc<RenderPass>)> {
        let renderpass = renderpasses.get_by_framebuffer_id(framebuffer_id)?;
        let framebuffer = match self
            .framebuffers
            .lock()
            .entry(framebuffer_id.framebuffer_id())
        {
            Entry::Occupied(entry) => Ok::<Arc<Framebuffer>, anyhow::Error>(entry.get().clone()),
            Entry::Vacant(entry) => Ok(entry
                .insert(
                    self.build_framebuffer(framebuffer_id.framebuffer_id(), renderpass.clone())?,
                )
                .clone()),
        }?;
        Ok((framebuffer, renderpass))
    }

    fn build_framebuffer(
        &self,
        framebuffer_id: FramebufferId,
        renderpass: Arc<RenderPass>,
    ) -> Result<Arc<Framebuffer>> {
        log::debug!("Building framebuffer {}", framebuffer_id);
        let mut attachments = vec![];
        for id in framebuffer_id.color_attachments {
            attachments.push(self.get_image(id)?);
        }
        if let Some(id) = framebuffer_id.depth_stencil_attachment {
            attachments.push(self.get_image(id)?);
        }
        for id in framebuffer_id.input_attachments {
            attachments.push(self.get_image(id)?);
        }

        Ok(Framebuffer::new(
            renderpass,
            FramebufferCreateInfo {
                attachments,
                ..Default::default()
            },
        )
        .with_context(|| format!("building framebuffer {}", framebuffer_id))?)
    }

    fn get_image(&self, id: ImageId) -> Result<Arc<ImageView>> {
        let mut guard = self.image_views.lock();
        let entry = &mut guard[id];
        if let Some(entry) = entry {
            Ok(entry.clone())
        } else {
            let image = Self::make_image_and_view(&self.ctx, self.base_extent, id, self.config)?;
            *entry = Some(image.clone());
            Ok(image)
        }
    }

    fn try_get_image(&self, id: ImageId) -> Result<Arc<ImageView>> {
        let mut guard = self.image_views.lock();
        if let Some(x) = guard[id].clone() {
            Ok(x)
        } else {
            bail!("image_view doesn't exist for {id:?}");
        }
    }

    fn new(
        ctx: Arc<VulkanContext>,
        config: LiveRenderConfig,
        image_i: usize,
        swapchain_image: &Arc<Image>,
    ) -> Result<FramebufferHolder> {
        let mut views = EnumMap::default();
        let swapchain_view = ImageView::new_default(swapchain_image.clone())?;
        let base_extent = swapchain_image.extent();
        let depth_stencil_view =
            Self::make_image_and_view(&ctx, base_extent, ImageId::MainDepthStencil, config)?;

        let depth_only_create_info = ImageViewCreateInfo {
            subresource_range: ImageSubresourceRange {
                aspects: ImageAspects::DEPTH,
                ..depth_stencil_view.image().subresource_range()
            },
            ..ImageViewCreateInfo::from_image(depth_stencil_view.image())
        };
        let depth_only_view =
            ImageView::new(depth_stencil_view.image().clone(), depth_only_create_info)?;
        views[ImageId::MainDepthStencil] = Some(depth_stencil_view);
        views[ImageId::MainDepthStencilDepthOnly] = Some(depth_only_view);
        views[ImageId::SwapchainColor] = Some(swapchain_view);
        let main_color = Self::make_image_and_view(&ctx, base_extent, ImageId::MainColor, config)?;
        views[ImageId::MainColor] = Some(main_color.clone());

        let mut blit_path = vec![];
        blit_path.push(main_color.image().clone());

        let mut multiplier = config.supersampling.to_int() / 2;
        for i in 0..config.supersampling.blit_steps() {
            log::debug!("Creating blit path image {i}, {multiplier}x samples");
            let mut usage = ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST;
            if i == config.supersampling.blit_steps() - 1 {
                usage |= ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::INPUT_ATTACHMENT
                    | ImageUsage::SAMPLED;
            }
            let buffer = Image::new(
                ctx.clone_allocator(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: config.formats.color,
                    extent: [base_extent[0] * multiplier, base_extent[1] * multiplier, 1],
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
        views[ImageId::MainColorResolved] = Some(ImageView::new_default(
            blit_path
                .last()
                .context("Blit path was empty; this should never happen")?
                .clone(),
        )?);

        Ok(FramebufferHolder {
            image_i,
            config,
            base_extent,
            ctx,
            image_views: Mutex::new(views),
            framebuffers: Mutex::new(FxHashMap::default()),
            samplers: Mutex::new(EnumMap::default()),
            blit_path,
        })
    }

    fn make_image_and_view(
        ctx: &VulkanContext,
        base_extent: [u32; 3],
        image_type: ImageId,
        config: LiveRenderConfig,
    ) -> Result<Arc<ImageView>> {
        let format = image_type.image_format(&config.formats);
        let usage = image_type.usage();
        let extent = image_type.dimension(base_extent[0], base_extent[1], config);
        let extent = [extent.0, extent.1, 1];
        let image = Image::new(
            ctx.clone_allocator(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format,
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

    pub(crate) fn begin_render_pass<L, const M: usize, const N: usize>(
        &self,
        cmd: &mut AutoCommandBufferBuilder<L>,
        framebuffer_and_load_op_id: FramebufferAndLoadOpId<M, N>,
        renderpasses: &RenderPassHolder,
        contents: SubpassContents,
    ) -> Result<()> {
        let (framebuffer, render_pass) =
            self.get_framebuffer(framebuffer_and_load_op_id, renderpasses)?;
        let mut clear_values = Vec::with_capacity(framebuffer.attachments().len());

        for (image, op) in framebuffer_and_load_op_id
            .color_attachments
            .iter()
            .chain(framebuffer_and_load_op_id.depth_stencil_attachment.iter())
            .chain(framebuffer_and_load_op_id.input_attachments.iter())
        {
            clear_values.push(if *op == LoadOp::Clear {
                Some(image.clear_value())
            } else {
                None
            });
        }

        cmd.begin_render_pass(
            RenderPassBeginInfo {
                render_pass,
                clear_values,
                ..RenderPassBeginInfo::framebuffer(framebuffer)
            },
            SubpassBeginInfo {
                contents,
                ..Default::default()
            },
        )?;
        Ok(())
    }
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

    fn div_texref(&self, dimensions: (u32, u32)) -> TexRef {
        TexRef {
            top_left: [self.l / dimensions.0 as f32, self.t / dimensions.1 as f32],
            width_height: [self.w / dimensions.0 as f32, self.h / dimensions.1 as f32],
        }
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

#[derive(Clone, Copy, Debug, enum_map::Enum)]
pub(crate) enum ReclaimType {
    CpuTransferSrc,
    GpuSsboTransferDst,
    CubeGeometryVtx,
    CubeGeometryIdx,
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
            ReclaimType::CubeGeometryVtx => BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            ReclaimType::CubeGeometryIdx => BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER | BufferUsage::TRANSFER_DST,
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
            _ => AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        }
    }
}

pub(crate) struct ReclaimableBuffer<T> {
    reclaim_type: ReclaimType,
    buffer: Subbuffer<[T]>,
    valid_len: DeviceSize,
    expiration: Instant,
    sequester: Option<usize>,
}
impl<T> ReclaimableBuffer<T> {
    pub(crate) fn valid_len(&self) -> DeviceSize {
        self.valid_len
    }
}

impl<T> Deref for ReclaimableBuffer<T> {
    type Target = Subbuffer<[T]>;
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

struct ReclaimInner<T> {
    pending: Vec<ReclaimableBuffer<T>>,
    ready: enum_map::EnumMap<ReclaimType, ReclaimBuffersByType<T>>,
}

pub(crate) struct BufferReclaim<T> {
    inner: Mutex<ReclaimInner<T>>,
}

impl<T: BufferContents> BufferReclaim<T> {
    pub(crate) fn take_or_create_slice(
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
                valid_len: 0,
            })
        }
    }

    pub(crate) fn size_class(capacity: DeviceSize) -> DeviceSize {
        let size = size_of::<T>() as DeviceSize;
        let bytes = capacity * size;
        bytes.next_power_of_two() / size
    }

    fn new() -> BufferReclaim<T> {
        BufferReclaim {
            inner: Mutex::new(ReclaimInner {
                pending: vec![],
                ready: Default::default(),
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
            inner.ready[buffer.reclaim_type].give_buffer(buffer);
        }
    }

    fn take_buffer(
        &self,
        reclaim_type: ReclaimType,
        size: DeviceSize,
    ) -> Option<ReclaimableBuffer<T>> {
        let mut inner = self.inner.lock();
        Self::clean_expired(&mut inner);
        inner.ready[reclaim_type].take_buffer(size)
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
                inner.ready[buffer.reclaim_type].give_buffer(buffer);
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
        use std::collections::btree_map::Entry;
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

impl<T> Default for ReclaimBuffersByType<T> {
    fn default() -> Self {
        Self::new()
    }
}
