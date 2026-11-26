// Copyright 2026 drey7925
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

//! Minimal headless Vulkano renderer for `egui`, requiring no winit window or swapchain surface.
//!
//! Used by the headless renderer entry point to capture menus and loading screens directly into
//! an offscreen Vulkan target image.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use egui::{ClippedPrimitive, ImageData, RawInput, TextureFilter, TextureId, TextureOptions};
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
    PrimaryCommandBufferAbstract, SubpassContents, SubpassEndInfo,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
    ColorComponents,
};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::subpass::PipelineSubpassType;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Scissor, Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::Subpass;
use vulkano::sync::GpuFuture;

use crate::vulkan::shaders::egui_adapter::set_up_fonts;
use crate::vulkan::{FramebufferAndLoadOpId, FramebufferHolder, ImageId, LoadOp, VulkanWindow};

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 460
            layout(location = 0) in vec2 pos;
            layout(location = 1) in vec2 uv;
            layout(location = 2) in vec4 color;

            layout(push_constant) uniform PushConstants {
                vec2 screen_size;
            } pc;

            layout(location = 0) out vec4 v_color;
            layout(location = 1) out vec2 v_uv;

            void main() {
                gl_Position = vec4(
                    2.0 * pos.x / pc.screen_size.x - 1.0,
                    2.0 * pos.y / pc.screen_size.y - 1.0,
                    0.0,
                    1.0
                );
                v_color = color;
                v_uv = uv;
            }
        "
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 460
            layout(location = 0) in vec4 v_color;
            layout(location = 1) in vec2 v_uv;

            layout(set = 0, binding = 0) uniform sampler2D font_texture;

            layout(location = 0) out vec4 f_color;

            vec3 linear_from_srgb(vec3 srgb) {
                bvec3 cutoff = lessThan(srgb, vec3(0.04045));
                vec3 lower = srgb / vec3(12.92);
                vec3 higher = pow((srgb + vec3(0.055)) / vec3(1.055), vec3(2.4));
                return mix(higher, lower, cutoff);
            }

            void main() {
                vec4 tex_srgba = texture(font_texture, v_uv);
                vec4 srgba = v_color * tex_srgba;
                vec3 rgb = srgba.a > 0.0 ? (srgba.rgb / srgba.a) : vec3(0.0);
                vec3 linear_rgb = linear_from_srgb(clamp(rgb, 0.0, 1.0));
                f_color = vec4(linear_rgb * srgba.a, srgba.a);
            }
        "
    }
}

#[derive(BufferContents, Vertex, Copy, Clone, Debug)]
#[repr(C)]
struct EguiVertex {
    #[format(R32G32_SFLOAT)]
    pos: [f32; 2],
    #[format(R32G32_SFLOAT)]
    uv: [f32; 2],
    #[format(R8G8B8A8_UNORM)]
    color: [u8; 4],
}

struct GpuTexture {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
    descriptor_set: Arc<DescriptorSet>,
}

pub(crate) struct HeadlessEguiRenderer {
    pub(crate) egui_ctx: egui::Context,
    pipeline: Arc<GraphicsPipeline>,
    textures: HashMap<TextureId, GpuTexture>,
}

impl HeadlessEguiRenderer {
    pub(crate) fn new(ctx: &VulkanWindow) -> Result<Self> {
        let mut egui_ctx = egui::Context::default();
        set_up_fonts(&mut egui_ctx);
        let mut style = (*egui_ctx.style()).clone();
        style
            .text_styles
            .insert(egui::TextStyle::Body, egui::FontId::proportional(16.0));
        egui_ctx.set_style(style);
        egui_extras::install_image_loaders(&egui_ctx);

        let subpass = Subpass::from(ctx.ui_renderpass()?, 0)
            .context("Could not find subpass 0 for headless egui")?;

        let vs_mod = vs::load(ctx.vk_device.clone())?;
        let fs_mod = fs::load(ctx.vk_device.clone())?;
        let vs_entry = vs_mod.entry_point("main").context("Missing egui vs main")?;
        let fs_entry = fs_mod.entry_point("main").context("Missing egui fs main")?;

        let vertex_input_state = EguiVertex::per_vertex().definition(&vs_entry)?;
        let stages = [
            PipelineShaderStageCreateInfo::new(vs_entry),
            PipelineShaderStageCreateInfo::new(fs_entry),
        ];

        let layout = PipelineLayout::new(
            ctx.vk_device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(ctx.vk_device.clone())?,
        )?;

        let blend = AttachmentBlend {
            src_color_blend_factor: BlendFactor::One,
            dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
            color_blend_op: BlendOp::Add,
            src_alpha_blend_factor: BlendFactor::One,
            dst_alpha_blend_factor: BlendFactor::OneMinusSrcAlpha,
            alpha_blend_op: BlendOp::Add,
        };

        let pipeline = GraphicsPipeline::new(
            ctx.vk_device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState {
                    cull_mode: CullMode::None,
                    ..Default::default()
                }),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState {
                        blend: Some(blend),
                        color_write_mask: ColorComponents::all(),
                        color_write_enable: true,
                    },
                )),
                dynamic_state: [DynamicState::Viewport, DynamicState::Scissor]
                    .into_iter()
                    .collect(),
                subpass: Some(PipelineSubpassType::BeginRenderPass(subpass)),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )?;

        Ok(Self {
            egui_ctx,
            pipeline,
            textures: HashMap::new(),
        })
    }

    pub(crate) fn render_frame(
        &mut self,
        ctx: &VulkanWindow,
        framebuffer: &FramebufferHolder,
        screen_size: [u32; 2],
        pixels_per_point: f32,
        time_secs: f64,
        mut run_ui: impl FnMut(&egui::Context),
    ) -> Result<()> {
        let logical_w = screen_size[0] as f32 / pixels_per_point;
        let logical_h = screen_size[1] as f32 / pixels_per_point;
        let raw_input = RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(logical_w, logical_h),
            )),
            time: Some(time_secs),
            viewports: std::iter::once((
                egui::ViewportId::ROOT,
                egui::ViewportInfo {
                    native_pixels_per_point: Some(pixels_per_point),
                    ..Default::default()
                },
            ))
            .collect(),
            ..Default::default()
        };

        let full_output = self.egui_ctx.run(raw_input, |egui_ctx| {
            run_ui(egui_ctx);
        });

        for (id, delta) in &full_output.textures_delta.set {
            self.update_texture(ctx, *id, delta)?;
        }

        let primitives = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);

        self.draw_primitives(
            ctx,
            framebuffer,
            screen_size,
            full_output.pixels_per_point,
            &primitives,
        )?;

        for id in &full_output.textures_delta.free {
            self.textures.remove(id);
        }

        Ok(())
    }

    fn update_texture(
        &mut self,
        ctx: &VulkanWindow,
        id: TextureId,
        delta: &egui::epaint::ImageDelta,
    ) -> Result<()> {
        let (patch_w, patch_h, patch_bytes) = image_data_to_rgba8(&delta.image);

        let (width, height, pixels) = match delta.pos {
            Some([x, y]) => {
                let existing = self
                    .textures
                    .get_mut(&id)
                    .context("Partial texture update for unknown egui texture")?;
                for row in 0..patch_h {
                    let dst_y = y + row;
                    if dst_y >= existing.height {
                        continue;
                    }
                    let copy_w = patch_w.min(existing.width.saturating_sub(x));
                    let src_offset = row * patch_w * 4;
                    let dst_offset = (dst_y * existing.width + x) * 4;
                    existing.pixels[dst_offset..dst_offset + copy_w * 4]
                        .copy_from_slice(&patch_bytes[src_offset..src_offset + copy_w * 4]);
                }
                (existing.width, existing.height, existing.pixels.clone())
            }
            None => (patch_w, patch_h, patch_bytes),
        };

        let descriptor_set =
            self.upload_texture(ctx, width as u32, height as u32, &pixels, delta.options)?;
        self.textures.insert(
            id,
            GpuTexture {
                width,
                height,
                pixels,
                descriptor_set,
            },
        );
        Ok(())
    }

    fn upload_texture(
        &self,
        ctx: &VulkanWindow,
        width: u32,
        height: u32,
        pixels: &[u8],
        options: TextureOptions,
    ) -> Result<Arc<DescriptorSet>> {
        let staging = Buffer::from_iter(
            ctx.clone_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            pixels.iter().copied(),
        )?;

        let image = Image::new(
            ctx.clone_allocator(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_UNORM,
                extent: [width.max(1), height.max(1), 1],
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )?;

        let mut builder = AutoCommandBufferBuilder::primary(
            ctx.command_buffer_allocator(),
            ctx.graphics_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        builder
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(staging, image.clone()))?;
        builder
            .build()?
            .execute(ctx.graphics_queue.clone())?
            .then_signal_fence_and_flush()?
            .wait(None)?;

        let image_view = ImageView::new_default(image)?;
        let sampler = Sampler::new(
            ctx.vk_device.clone(),
            SamplerCreateInfo {
                mag_filter: to_vk_filter(options.magnification),
                min_filter: to_vk_filter(options.minification),
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..Default::default()
            },
        )?;

        let layout = self
            .pipeline
            .layout()
            .set_layouts()
            .first()
            .context("Missing descriptor set layout 0")?
            .clone();
        let descriptor_set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            layout,
            [WriteDescriptorSet::image_view_sampler(
                0, image_view, sampler,
            )],
            [],
        )?;
        Ok(descriptor_set)
    }

    fn draw_primitives(
        &self,
        ctx: &VulkanWindow,
        framebuffer: &FramebufferHolder,
        screen_size: [u32; 2],
        pixels_per_point: f32,
        primitives: &[ClippedPrimitive],
    ) -> Result<()> {
        let mut builder = ctx.start_command_buffer()?;
        framebuffer.begin_render_pass(
            &mut builder,
            FramebufferAndLoadOpId {
                color_attachments: [(ImageId::SwapchainColor, LoadOp::Clear)],
                depth_stencil_attachment: None,
                input_attachments: [],
            },
            ctx.renderpasses(),
            SubpassContents::Inline,
        )?;

        builder.bind_pipeline_graphics(self.pipeline.clone())?;
        builder.set_viewport(
            0,
            [Viewport {
                offset: [0.0, 0.0],
                extent: [screen_size[0] as f32, screen_size[1] as f32],
                depth_range: 0.0..=1.0,
            }]
            .into_iter()
            .collect(),
        )?;

        let push = vs::PushConstants {
            screen_size: [
                screen_size[0] as f32 / pixels_per_point,
                screen_size[1] as f32 / pixels_per_point,
            ],
        };
        builder.push_constants(self.pipeline.layout().clone(), 0, push)?;

        for clipped in primitives {
            let egui::epaint::Primitive::Mesh(mesh) = &clipped.primitive else {
                continue;
            };
            if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                continue;
            }
            let Some(texture) = self.textures.get(&mesh.texture_id) else {
                continue;
            };

            let clip = clipped.clip_rect;
            let min_x = (clip.min.x * pixels_per_point).round().max(0.0) as u32;
            let min_y = (clip.min.y * pixels_per_point).round().max(0.0) as u32;
            let max_x = (clip.max.x * pixels_per_point)
                .round()
                .clamp(0.0, screen_size[0] as f32) as u32;
            let max_y = (clip.max.y * pixels_per_point)
                .round()
                .clamp(0.0, screen_size[1] as f32) as u32;
            if max_x <= min_x || max_y <= min_y {
                continue;
            }

            let vertices: Vec<EguiVertex> = mesh
                .vertices
                .iter()
                .map(|v| EguiVertex {
                    pos: [v.pos.x, v.pos.y],
                    uv: [v.uv.x, v.uv.y],
                    color: v.color.to_array(),
                })
                .collect();

            let vertex_buffer = Buffer::from_iter(
                ctx.clone_allocator(),
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                vertices,
            )?;

            let index_buffer = Buffer::from_iter(
                ctx.clone_allocator(),
                BufferCreateInfo {
                    usage: BufferUsage::INDEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                mesh.indices.iter().copied(),
            )?;

            builder.set_scissor(
                0,
                [Scissor {
                    offset: [min_x, min_y],
                    extent: [max_x - min_x, max_y - min_y],
                }]
                .into_iter()
                .collect(),
            )?;
            builder.bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                texture.descriptor_set.clone(),
            )?;
            builder.bind_vertex_buffers(0, vertex_buffer)?;
            builder.bind_index_buffer(index_buffer)?;
            unsafe {
                builder.draw_indexed(mesh.indices.len() as u32, 1, 0, 0, 0)?;
            }
        }

        builder.end_render_pass(SubpassEndInfo::default())?;
        builder
            .build()?
            .execute(ctx.graphics_queue.clone())?
            .then_signal_fence_and_flush()?
            .wait(None)?;
        Ok(())
    }
}

fn image_data_to_rgba8(image: &ImageData) -> (usize, usize, Vec<u8>) {
    match image {
        ImageData::Color(color) => {
            let mut bytes = Vec::with_capacity(color.pixels.len() * 4);
            for p in &color.pixels {
                bytes.extend_from_slice(&p.to_array());
            }
            (color.width(), color.height(), bytes)
        }
        ImageData::Font(font) => {
            let mut bytes = Vec::with_capacity(font.pixels.len() * 4);
            for p in font.srgba_pixels(None) {
                bytes.extend_from_slice(&p.to_array());
            }
            (font.width(), font.height(), bytes)
        }
    }
}

fn to_vk_filter(filter: TextureFilter) -> Filter {
    match filter {
        TextureFilter::Nearest => Filter::Nearest,
        TextureFilter::Linear => Filter::Linear,
    }
}
