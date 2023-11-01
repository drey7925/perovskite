use std::sync::Arc;

use egui::TextureId;
use egui_winit_vulkano::{Gui, GuiConfig};
use parking_lot::Mutex;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferInheritanceInfo, SubpassContents},
    image::SampleCount,
    render_pass::Subpass, sampler::SamplerCreateInfo,
};
use winit::{event::WindowEvent, event_loop::EventLoopWindowTarget};

use crate::{
    game_state::ClientState,
    game_ui::egui_ui::EguiUi,
    vulkan::{Texture2DHolder, VulkanWindow},
};

use anyhow::{Context, Result};

use super::{
    flat_texture::{FlatTexPipelineProvider, FlatTexPipelineWrapper},
    PipelineProvider, PipelineWrapper,
};

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
        ctx: &crate::vulkan::VulkanWindow,
        event_loop: &EventLoopWindowTarget<()>,
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
            ctx.queue.clone(),
            Subpass::from(ctx.render_pass.clone(), 1).context("Could not find subpass 0")?,
            ctx.swapchain.image_format(),
            config,
        );
        let atlas = egui_ui.lock().clone_texture_atlas();

        let mut sampler_create_info = SamplerCreateInfo::default();
        sampler_create_info.mag_filter = vulkano::sampler::Filter::Nearest;
        sampler_create_info.min_filter = vulkano::sampler::Filter::Linear;

        let atlas_texture_id = gui_adapter.register_user_image_view(atlas.clone_image_view(), sampler_create_info);

        let flat_overlay_provider = FlatTexPipelineProvider::new(ctx.vk_device.clone())?;
        let flat_overlay_pipeline =
            flat_overlay_provider.make_pipeline(ctx, (atlas.as_ref(), 1))?;
        
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
    ) -> Result<()> {
        let mut egui = self.egui_ui.lock();
        self.gui_adapter.begin_frame();
        egui.draw_all_uis(
            &self.gui_adapter.egui_ctx,
            self.atlas_texture_id,
            client_state,
        );
        let cmdbuf = self
            .gui_adapter
            .draw_on_subpass_image([ctx.window_size().0, ctx.window_size().1]);
        builder.next_subpass(SubpassContents::SecondaryCommandBuffers)?;
        builder.execute_commands(cmdbuf)?;

        if let Some(draw_call) = egui.get_carried_itemstack(ctx, client_state)? {
            let mut secondary_builder = AutoCommandBufferBuilder::secondary(
                &ctx.command_buffer_allocator,
                ctx.queue.queue_family_index(),
                vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
                CommandBufferInheritanceInfo {
                    render_pass: Some(
                        Subpass::from(ctx.render_pass.clone(), 1)
                            .with_context(|| "Render subpass 1 not found")?
                            .into(),
                    ),
                    ..Default::default()
                },
            )?;
            self.flat_overlay_pipeline
                .bind(ctx, (), &mut secondary_builder, ())
                .unwrap();
            self.flat_overlay_pipeline
                .draw(&mut secondary_builder, &[draw_call], ())?;
            builder.execute_commands(secondary_builder.build()?)?;
        }

        Ok(())
    }

    pub(crate) fn notify_resize(&mut self, ctx: &VulkanWindow) -> Result<()> {
        self.flat_overlay_pipeline = self
            .flat_overlay_provider
            .make_pipeline(ctx, (self.atlas.as_ref(), 1))?;
        Ok(())
    }
}

fn set_up_fonts(egui_ctx: &mut egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "NotoSans-Light".to_owned(),
        egui::FontData::from_static(include_bytes!(
            "../../fonts/NotoSans-Light.ttf"
        )),
    );
    fonts.font_data.insert(
        "NotoSansJP-Light".to_owned(),
        egui::FontData::from_static(include_bytes!(
            "../../fonts/NotoSansJP-Light.ttf"
        )),
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
    egui_ctx.set_fonts(fonts);
}
