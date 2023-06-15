use std::sync::Arc;

use egui::TextureId;
use egui_winit_vulkano::{Gui, GuiConfig};
use parking_lot::Mutex;
use vulkano::{command_buffer::SubpassContents, image::SampleCount, render_pass::Subpass};
use winit::{event::WindowEvent, event_loop::EventLoop};

use crate::{game_state::ClientState, game_ui::egui_ui::EguiUi, vulkan::VulkanContext};

use anyhow::{Context, Result};

// Main thread components of egui rendering (e.g. the Gui which contains a non-Send event loop)
pub(crate) struct EguiAdapter {
    gui_adapter: Gui,
    egui_ui: Arc<Mutex<EguiUi>>,
    atlas_texture_id: TextureId,
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
        event_loop: &EventLoop<()>,
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
            Subpass::from(ctx.render_pass.clone(), 1).context("Could not find subpass 0")?,
            config,
        );
        let atlas_texture_id =
            gui_adapter.register_user_image_view(egui_ui.lock().texture_atlas().clone_image_view());
        Ok(EguiAdapter {
            gui_adapter,
            egui_ui,
            atlas_texture_id,
        })
    }

    pub(crate) fn draw(
        &mut self,
        ctx: &VulkanContext,
        builder: &mut crate::vulkan::CommandBufferBuilder,
        client_state: &ClientState,
    ) -> Result<()> {
        let mut egui = self.egui_ui.lock();
        if egui.wants_draw() {
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
            Ok(())
        } else {
            // We still need to advance to the next subpass, even if we aren't drawing anything
            builder.next_subpass(SubpassContents::Inline)?;
            Ok(())
        }
    }
}
