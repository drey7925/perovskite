use std::sync::Arc;

use egui::TextureId;
use egui_winit_vulkano::{Gui, GuiConfig};
use parking_lot::Mutex;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferInheritanceInfo},
    image::SampleCount,
    render_pass::Subpass,
};
use winit::event::WindowEvent;

use crate::{
    client_state::ClientState,
    game_ui::egui_ui::EguiUi,
    vulkan::{Texture2DHolder, VulkanWindow},
};

use super::{
    flat_texture::{FlatTexPipelineProvider, FlatTexPipelineWrapper},
    LiveRenderConfig,
};
use crate::client_state::settings::Supersampling;
use crate::client_state::tool_controller::ToolState;
use crate::main_menu::InputCapture;
use crate::vulkan::shaders::flat_texture::FlatPipelineConfig;
use anyhow::{Context, Result};
use vulkano::image::sampler::{Filter, SamplerCreateInfo};
use winit::event_loop::ActiveEventLoop;

// Main thread components of egui rendering (e.g. the Gui which contains a non-Send event loop)
pub(crate) struct EguiAdapter {
    gui_adapter: Gui,
    egui_ui: Arc<Mutex<EguiUi>>,
    atlas_texture_id: TextureId,
    atlas: Arc<Texture2DHolder>,
    flat_overlay_pipeline: FlatTexPipelineWrapper,
    flat_overlay_provider: FlatTexPipelineProvider,
}
impl EguiAdapter {
    pub(crate) fn window_event(&mut self, event: &WindowEvent) -> bool {
        if self.egui_ui.lock().wants_user_events() {
            self.gui_adapter.update(event)
        } else {
            // egui isn't drawing; don't try to interact with it
            false
        }
    }

    pub(crate) fn new(
        ctx: &VulkanWindow,
        event_loop: &ActiveEventLoop,
        egui_ui: Arc<Mutex<EguiUi>>,
    ) -> Result<EguiAdapter> {
        let config = GuiConfig {
            allow_srgb_render_target: true,
            is_overlay: true,
            samples: SampleCount::Sample1,
        };
        let mut gui_adapter = Gui::new_with_subpass(
            event_loop,
            ctx.swapchain.surface().clone(),
            ctx.graphics_queue.clone(),
            Subpass::from(ctx.color_only_render_pass.clone(), 0)
                .context("Could not find subpass 0")?,
            ctx.swapchain.image_format(),
            config,
        );
        let atlas = egui_ui.lock().clone_texture_atlas();

        let sampler_create_info = SamplerCreateInfo {
            mag_filter: Filter::Nearest,
            min_filter: Filter::Linear,
            ..Default::default()
        };

        let atlas_texture_id =
            gui_adapter.register_user_image_view(atlas.clone_image_view(), sampler_create_info);

        let flat_overlay_provider = FlatTexPipelineProvider::new(ctx.vk_device.clone())?;
        let flat_overlay_pipeline = flat_overlay_provider.make_pipeline(
            ctx,
            FlatPipelineConfig {
                atlas: atlas.as_ref(),
                subpass: Subpass::from(ctx.color_only_render_pass.clone(), 0)
                    .context("Post-blit subpass 0 missing")?,
                enable_depth_stencil: false,
                enable_supersampling: false,
            },
            &LiveRenderConfig {
                supersampling: Supersampling::None,
                // irrelevant for this pipeline
                raytracing: false,
                raytracing_reflections: false,
                raytracer_debug: false,
                // irrelevant for this pipeline
                render_distance: 0,
            },
        )?;

        set_up_fonts(&mut gui_adapter.egui_ctx);

        Ok(EguiAdapter {
            gui_adapter,
            egui_ui,
            atlas_texture_id,
            atlas,
            flat_overlay_pipeline,
            flat_overlay_provider,
        })
    }

    pub(crate) fn draw<L>(
        &mut self,
        ctx: &VulkanWindow,
        builder: &mut crate::vulkan::CommandBufferBuilder<L>,
        client_state: &ClientState,
        input_capture: &mut InputCapture,
        tool_state: &ToolState,
    ) -> Result<()> {
        let mut egui = self.egui_ui.lock();
        self.gui_adapter.begin_frame();
        egui.draw_all_uis(
            &self.gui_adapter.egui_ctx,
            self.atlas_texture_id,
            client_state,
            input_capture,
            ctx,
            tool_state,
        );
        let cmdbuf = self
            .gui_adapter
            .draw_on_subpass_image([ctx.window_size().0, ctx.window_size().1]);
        builder.execute_commands(cmdbuf)?;

        if let Some(draw_call) = egui.get_carried_itemstack(ctx, client_state)? {
            let mut secondary_builder = AutoCommandBufferBuilder::secondary(
                ctx.command_buffer_allocator.clone(),
                ctx.graphics_queue.queue_family_index(),
                vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
                CommandBufferInheritanceInfo {
                    render_pass: Some(
                        Subpass::from(ctx.color_only_render_pass.clone(), 0)
                            .with_context(|| "Render subpass 0 not found")?
                            .into(),
                    ),
                    ..Default::default()
                },
            )?;
            self.flat_overlay_pipeline
                .bind(ctx, &mut secondary_builder)?;
            self.flat_overlay_pipeline
                .draw(&mut secondary_builder, &[draw_call])?;
            builder.execute_commands(secondary_builder.build()?)?;
        }

        Ok(())
    }

    pub(crate) fn notify_resize(&mut self, ctx: &VulkanWindow) -> Result<()> {
        self.flat_overlay_pipeline = self.flat_overlay_provider.make_pipeline(
            ctx,
            FlatPipelineConfig {
                atlas: self.atlas.as_ref(),
                subpass: Subpass::from(ctx.color_only_render_pass.clone(), 0)
                    .context("Post-blit subpass 0 missing")?,
                enable_depth_stencil: false,
                enable_supersampling: false,
            },
            &LiveRenderConfig {
                supersampling: Supersampling::None,
                raytracing: false,
                raytracing_reflections: false,
                render_distance: 0,
                raytracer_debug: false,
            },
        )?;
        Ok(())
    }
}

pub(crate) fn set_up_fonts(egui_ctx: &mut egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "NotoSans-Light".to_owned(),
        egui::FontData::from_static(include_bytes!("../../fonts/NotoSans-Light.ttf")).into(),
    );
    fonts.font_data.insert(
        "NotoSansJP-Light".to_owned(),
        egui::FontData::from_static(include_bytes!("../../fonts/NotoSansJP-Light.ttf")).into(),
    );
    fonts.font_data.insert(
        "MPlus1Code-Light".to_owned(),
        egui::FontData::from_static(include_bytes!("../../fonts/MPLUS1Code-Light.ttf")).into(),
    );
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "NotoSans-Light".to_owned());
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(1, "NotoSansJP-Light".to_owned());
    fonts
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .insert(0, "MPlus1Code-Light".to_owned());

    let mut mono_fonts = fonts
        .families
        .get(&egui::FontFamily::Monospace)
        .unwrap()
        .clone();
    mono_fonts.push("NotoSansJP-Light".to_owned());
    mono_fonts.push("NotoSans-Light".to_owned());
    // A font family that tries to render with monospace and falls back to noto
    // for wider character selection
    fonts.families.insert(
        egui::FontFamily::Name("MonospaceBestEffort".into()),
        mono_fonts,
    );
    egui_ctx.set_fonts(fonts);
}
