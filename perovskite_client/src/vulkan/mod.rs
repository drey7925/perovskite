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

pub mod game_renderer;
pub(crate) mod mini_renderer;
pub(crate) mod shaders;
pub(crate) mod util;

use std::{ops::Deref, sync::Arc};

use anyhow::{bail, Context, Result};
use arc_swap::ArcSwap;
use log::warn;

use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract, RenderPassBeginInfo,
        SubpassContents,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, DeviceExtensions, Features, Queue,
        QueueCreateInfo,
    },
    format::{ClearValue, Format, FormatFeatures, NumericType},
    image::{
        view::ImageView, AttachmentImage, ImageAccess, ImageUsage, ImmutableImage, SwapchainImage,
    },
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{FreeListAllocator, GenericMemoryAllocator, StandardMemoryAllocator},
    pipeline::{graphics::viewport::Viewport, GraphicsPipeline, Pipeline},
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    sampler::{Filter, Sampler, SamplerCreateInfo},
    swapchain::{Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError},
    sync::GpuFuture,
    Version,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    dpi::PhysicalSize,
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

pub(crate) type CommandBufferBuilder<L> =
    AutoCommandBufferBuilder<L, Arc<StandardCommandBufferAllocator>>;

use crate::game_state::settings::GameSettings;

use self::util::select_physical_device;

#[derive(Clone)]
pub(crate) struct VulkanContext {
    vk_device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<GenericMemoryAllocator<Arc<FreeListAllocator>>>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    /// The format of the image buffer used for rendering to *screen*. Render-to-texture always uses R8G8B8A8_SRGB
    color_format: Format,
    /// The format of the depth buffer used for rendering to *screen*. Probed at startup, and used for both render-to-screen and render-to-texture
    depth_format: Format,
}
impl VulkanContext {
    pub(crate) fn clone_allocator(&self) -> Arc<GenericMemoryAllocator<Arc<FreeListAllocator>>> {
        self.memory_allocator.clone()
    }

    pub(crate) fn allocator(&self) -> &GenericMemoryAllocator<Arc<FreeListAllocator>> {
        &self.memory_allocator
    }

    pub(crate) fn clone_queue(&self) -> Arc<Queue> {
        self.queue.clone()
    }

    fn start_command_buffer(&self) -> Result<CommandBufferBuilder<PrimaryAutoCommandBuffer>> {
        let builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )?;

        Ok(builder)
    }

    pub(crate) fn depth_clear_value(&self) -> ClearValue {
        if self.depth_format.type_stencil().is_some() {
            ClearValue::DepthStencil((1.0, 0))
        } else {
            ClearValue::Depth(1.0)
        }
    }
}

pub(crate) struct VulkanWindow {
    vk_ctx: Arc<VulkanContext>,
    render_pass: Arc<RenderPass>,
    swapchain: ArcSwap<Swapchain>,
    swapchain_images: ArcSwap<Vec<Arc<SwapchainImage>>>,
    framebuffers: ArcSwap<Vec<Arc<Framebuffer>>>,
    window: Arc<Window>,
    viewport: ArcSwap<Viewport>,
}
impl Deref for VulkanWindow {
    type Target = VulkanContext;
    fn deref(&self) -> &Self::Target {
        &self.vk_ctx
    }
}

impl VulkanWindow {
    pub fn context(&self) -> &VulkanContext {
        &self.vk_ctx
    }

    pub fn clone_context(&self) -> Arc<VulkanContext> {
        self.vk_ctx.clone()
    }

    pub(crate) fn create(
        event_loop: &EventLoop<()>,
        settings: &Arc<ArcSwap<GameSettings>>,
    ) -> Result<VulkanWindow> {
        let library: Arc<vulkano::VulkanLibrary> =
            vulkano::VulkanLibrary::new().expect("no local Vulkan library/DLL");
        let required_extensions = vulkano_win::required_extensions(&library);
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

        let surface = WindowBuilder::new()
            .with_title("Perovskite Game Client")
            .build_vk_surface(event_loop, instance.clone())?;

        let window = surface
            .object()
            .with_context(|| "Surface was missing its object")?
            .clone()
            .downcast::<Window>()
            .map_err(|_| anyhow::Error::msg("downcast to Window failed"))?;

        // TODO adjust this
        window.set_min_inner_size(Some(PhysicalSize::new(256, 256)));

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: window.inner_size().into(),
            depth_range: 0.0..1.0,
        };

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        let device_features = Features {
            ..Features::empty()
        };

        let (physical_device, queue_family_index) = select_physical_device(
            &instance,
            &surface,
            &device_extensions,
            &settings.load().render.preferred_gpu,
        )?;

        let (vk_device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                enabled_features: device_features,
                ..Default::default()
            },
        )?;
        let queue = queues.next().with_context(|| "expected a queue")?;

        let (swapchain, swapchain_images, color_format) = {
            let caps = physical_device
                .surface_capabilities(&surface, Default::default())
                .expect("failed to get surface capabilities");

            let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
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
                    image_format: Some(image_format),
                    image_extent: window.inner_size().into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha,
                    image_color_space: color_space,
                    ..Default::default()
                },
            )?;
            (swapchain, images, image_format)
        };

        let depth_format = find_best_depth_format(&physical_device)?;

        let render_pass =
            make_render_pass(vk_device.clone(), swapchain.image_format(), depth_format)?;

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(vk_device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            vk_device.clone(),
            Default::default(),
        ));
        let descriptor_set_allocator =
            Arc::new(StandardDescriptorSetAllocator::new(vk_device.clone()));
        let framebuffers = get_framebuffers_with_depth(
            &swapchain_images,
            &memory_allocator,
            render_pass.clone(),
            depth_format,
        );

        Ok(VulkanWindow {
            vk_ctx: Arc::new(VulkanContext {
                vk_device,
                queue,
                memory_allocator,
                command_buffer_allocator,
                descriptor_set_allocator,
                color_format,
                depth_format,
            }),
            render_pass,
            swapchain: ArcSwap::new(swapchain),
            swapchain_images: ArcSwap::new(Arc::new(swapchain_images)),
            framebuffers: ArcSwap::new(Arc::new(framebuffers)),
            window,
            viewport: ArcSwap::new(Arc::new(viewport)),
        })
    }

    fn recreate_swapchain(&self, size: PhysicalSize<u32>) -> Result<()> {
        let size = PhysicalSize::new(size.width.max(1), size.height.max(1));
        let old_swapchain = self.swapchain.load().clone();
        let (new_swapchain, new_images) = match old_swapchain.recreate(SwapchainCreateInfo {
            image_extent: size.into(),
            ..old_swapchain.create_info()
        }) {
            Ok(r) => r,
            Err(SwapchainCreationError::ImageExtentNotSupported {
                provided,
                min_supported,
                max_supported,
            }) => {
                warn!(
                    "Ignoring ImageExtentNotSupported: provided {:?}, min {:?}, max {:?}",
                    provided, min_supported, max_supported
                );
                return Ok(());
            }
            Err(e) => panic!("failed to recreate swapchain: {e}"),
        };
        self.swapchain.store(new_swapchain);
        self.swapchain_images.store(Arc::new(new_images.clone()));
        self.framebuffers
            .store(Arc::new(get_framebuffers_with_depth(
                &new_images,
                &self.memory_allocator,
                self.render_pass.clone(),
                self.depth_format,
            )));
        Ok(())
    }

    fn start_render_pass(
        &self,
        builder: &mut CommandBufferBuilder<PrimaryAutoCommandBuffer>,
        framebuffer: Arc<Framebuffer>,
        clear_color: [f32; 4],
    ) -> Result<()> {
        builder.begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some(clear_color.into()), Some(self.depth_clear_value())],
                ..RenderPassBeginInfo::framebuffer(framebuffer)
            },
            SubpassContents::Inline,
        )?;
        Ok(())
    }

    pub(crate) fn window_size(&self) -> (u32, u32) {
        let dims = self.viewport.load().dimensions;
        (dims[0] as u32, dims[1] as u32)
    }

    pub(crate) fn swapchain_surface(&self) -> Arc<Surface> {
        self.swapchain.load().surface().clone()
    }
    pub(crate) fn swapchain_format(&self) -> Format {
        self.color_format
    }

    pub(crate) fn clone_render_pass(&self) -> Arc<RenderPass> {
        self.render_pass.clone()
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
                log::warn!(
                    "Depth format D16_UNORM may have low precision and cause visual glitches"
                );
                log::warn!("According to the vulkan spec, at least one of the other formats should be supported by any compliant GPU.");
                log::warn!("Please reach out to the devs with details about your GPU.");
            }
            return Ok(format);
        }
    }
    bail!("No depth format found");
}

pub(crate) fn make_render_pass(
    vk_device: Arc<Device>,
    output_format: Format,
    depth_format: Format,
) -> Result<Arc<RenderPass>> {
    vulkano::ordered_passes_renderpass!(
        vk_device,
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: output_format,
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: depth_format,
                samples: 1,
            },
        },
        passes: [
            {
                color: [color],
                depth_stencil: {depth},
                input: [],
            },
            {
                color: [color],
                depth_stencil: {},
                input: []
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
        .find(|(format, _space)| format.type_color() == Some(NumericType::SRGB))
        .cloned()
        .with_context(|| "Could not find an image format")
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
    images: &[Arc<SwapchainImage>],
    allocator: &StandardMemoryAllocator,
    render_pass: Arc<RenderPass>,
    depth_format: Format,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            let depth_buffer = ImageView::new_default(
                AttachmentImage::transient(
                    allocator,
                    image.dimensions().width_height(),
                    depth_format,
                )
                .unwrap(),
            )
            .unwrap();

            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view, depth_buffer],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

pub(crate) struct Texture2DHolder {
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    sampler: Arc<Sampler>,
    image_view: Arc<ImageView<ImmutableImage>>,
    dimensions: vulkano::image::ImageDimensions,
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
        let mut builder = AutoCommandBufferBuilder::primary(
            &ctx.command_buffer_allocator,
            ctx.queue.queue_family_index(),
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )?;
        let dimensions = vulkano::image::ImageDimensions::Dim2d {
            width: image.width(),
            height: image.height(),
            array_layers: 1,
        };
        let image = ImmutableImage::from_iter(
            &ctx.memory_allocator,
            img_rgba,
            dimensions,
            vulkano::image::MipmapsCount::Log2,
            // Ideally, we would probe for support between RGB and BGR, but we can't swizzle the
            // colors currently, so BGR formats wouldn't render properly.
            // According to https://registry.khronos.org/vulkan/site/spec/latest/chapters/formats.html#features-required-format-support,
            // any compliant GPU should support RGB.
            Format::R8G8B8A8_SRGB,
            &mut builder,
        )?;

        builder.build()?.execute(ctx.queue.clone())?.flush()?;

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
            dimensions,
        })
    }

    fn descriptor_set(
        &self,
        pipeline: &GraphicsPipeline,
        set: usize,
        binding: u32,
    ) -> Result<Arc<PersistentDescriptorSet>> {
        let layout = pipeline
            .layout()
            .set_layouts()
            .get(set)
            .with_context(|| "uniform set missing")?;
        let descriptor_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            layout.clone(),
            [WriteDescriptorSet::image_view_sampler(
                binding,
                self.image_view.clone(),
                self.sampler.clone(),
            )],
        )?;
        Ok(descriptor_set)
    }

    pub(crate) fn dimensions(&self) -> (u32, u32) {
        (self.dimensions.width(), self.dimensions.height())
    }
    pub(crate) fn clone_image_view(&self) -> Arc<ImageView<ImmutableImage>> {
        self.image_view.clone()
    }
}
