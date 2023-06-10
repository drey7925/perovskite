use egui_winit_vulkano::{Gui, GuiConfig};
use vulkano::{command_buffer::SubpassContents, image::SampleCount, render_pass::Subpass};
use winit::{event::WindowEvent, event_loop::EventLoop};

use crate::{game_ui::GameUi, vulkan::VulkanContext};

use anyhow::{Context, Result};

pub(crate) struct EguiAdapter {
    gui: Gui,
}
impl EguiAdapter {
    pub(crate) fn window_event(&mut self, event: &WindowEvent) -> bool {
        self.gui.update(event)
    }

    pub(crate) fn new(
        ctx: &crate::vulkan::VulkanContext,
        event_loop: &EventLoop<()>,
    ) -> Result<EguiAdapter> {
        let mut config = GuiConfig::default();
        config.preferred_format = dbg!(Some(ctx.swapchain.image_format()));
        config.is_overlay = true;
        config.samples = SampleCount::Sample4;
        let gui = Gui::new_with_subpass(
            event_loop,
            ctx.swapchain.surface().clone(),
            ctx.queue.clone(),
            Subpass::from(ctx.render_pass.clone(), 1).context("Could not find subpass 0")?,
            config,
        );
        Ok(EguiAdapter {
            gui,
        })
    }

    pub(crate) fn draw(
        &mut self,
        ctx: &VulkanContext,
        builder: &mut crate::vulkan::CommandBufferBuilder,
        game_ui: &mut GameUi,
    ) -> Result<()> {
        if game_ui.egui_active() {
            self.gui.begin_frame();
            game_ui.draw_egui(&self.gui.egui_ctx);
            let cmdbuf = self
                .gui
                .draw_on_subpass_image([ctx.window_size().0, ctx.window_size().1]);
            builder.next_subpass(SubpassContents::SecondaryCommandBuffers)?;
            builder.execute_commands(cmdbuf)?;
            Ok(())
        } else {
            Ok(())
        }
    }
}
