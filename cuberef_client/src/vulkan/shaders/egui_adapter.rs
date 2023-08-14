use std::sync::Arc;

use egui::TextureId;
use egui_winit_vulkano::{Gui, GuiConfig};
use parking_lot::Mutex;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferInheritanceInfo, SubpassContents, SecondaryAutoCommandBuffer},
    image::SampleCount,
    render_pass::Subpass,
};
use winit::{
    event::WindowEvent,
    event_loop::{EventLoopWindowTarget},
};

use crate::{
    game_state::ClientState,
    game_ui::egui_ui::EguiUi,
    vulkan::{Texture2DHolder, VulkanContext},
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
        if self.egui_ui.lock().wants_draw() {
            self.gui_adapter.update(event)
        } else {
            // egui isn't drawing; don't try to interact with it
            false
        }
    }

    pub(crate) fn new(
        ctx: &crate::vulkan::VulkanContext,
        event_loop: &EventLoopWindowTarget<()>,
        egui_ui: Arc<Mutex<EguiUi>>,
    ) -> Result<EguiAdapter> {
        let config = GuiConfig {
            preferred_format: Some(ctx.swapchain.image_format()),
            is_overlay: true,
            samples: SampleCount::Sample1,
        };
        let mut gui_adapter = Gui::new_with_subpass(
            event_loop,
            ctx.swapchain.surface().clone(),
            ctx.queue.clone(),
            Subpass::from(ctx.render_pass.clone(), 0).context("Could not find subpass 0")?,
            config,
        );
        let atlas = egui_ui.lock().clone_texture_atlas();
        let atlas_texture_id = gui_adapter.register_user_image_view(atlas.clone_image_view());

        let flat_overlay_provider = FlatTexPipelineProvider::new(ctx.vk_device.clone())?;
        let flat_overlay_pipeline =
            flat_overlay_provider.make_pipeline(ctx, (atlas.as_ref(), 0))?;
        Ok(EguiAdapter {
            gui_adapter,
            egui_ui,
            atlas_texture_id,
            atlas,
            flat_overlay_pipeline,
            flat_overlay_provider,
        })
    }

    pub(crate) fn draw(
        &mut self,
        ctx: &VulkanContext,
        client_state: &ClientState,
    ) -> Result<Vec<SecondaryAutoCommandBuffer>> {
        let mut egui = self.egui_ui.lock();
        if egui.wants_draw() {
            self.gui_adapter.begin_frame();
            egui.draw_all_uis(
                &self.gui_adapter.egui_ctx,
                self.atlas_texture_id,
                client_state,
            );
            let egui_cmdbuf = self
                .gui_adapter
                .draw_on_subpass_image([ctx.window_size().0, ctx.window_size().1]);
            

            if let Some(draw_call) = egui.get_carried_itemstack(ctx, client_state)? {
                let mut overlay_builder = AutoCommandBufferBuilder::secondary(
                    &ctx.command_buffer_allocator,
                    ctx.queue.queue_family_index(),
                    vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
                    CommandBufferInheritanceInfo {
                        render_pass: Some(
                            Subpass::from(ctx.render_pass.clone(), 0)
                                .with_context(|| "Render subpass 0 not found")?
                                .into(),
                        ),
                        ..Default::default()
                    },
                )?;
                self.flat_overlay_pipeline
                    .bind(ctx, (), &mut overlay_builder, ())
                    .unwrap();
                self.flat_overlay_pipeline
                    .draw(&mut overlay_builder, &[draw_call], ())?;
                Ok(vec![egui_cmdbuf, overlay_builder.build()?])
            } else {
                Ok(vec![egui_cmdbuf])
            }
        } else {
            Ok(vec![])
        }
    }

    pub(crate) fn notify_resize(&mut self, ctx: &VulkanContext) -> Result<()> {
        self.flat_overlay_pipeline = self
            .flat_overlay_provider
            .make_pipeline(ctx, (self.atlas.as_ref(), 0))?;
        Ok(())
    }
}
