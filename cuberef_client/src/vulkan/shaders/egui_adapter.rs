use egui_winit_vulkano::{Gui, GuiConfig};
use parking_lot::Mutex;
use tracy_client::span;
use vulkano::{render_pass::Subpass, command_buffer::SubpassContents, image::SampleCount};
use winit::{event::WindowEvent, event_loop::EventLoop};

use crate::game_ui::GameUi;

use super::PipelineWrapper;
use anyhow::{Result, Context};

pub(crate) struct EguiAdapter {
    gui: Gui,
    image_dimensions: [u32; 2],
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
            config
        );
        Ok(EguiAdapter {
            gui,
            image_dimensions: ctx.framebuffers[0].extent(),
        })
    }
}
impl PipelineWrapper<&Mutex<GameUi>, ()> for EguiAdapter {
    type PassIdentifier = ();

    fn draw(
        &mut self,
        builder: &mut crate::vulkan::CommandBufferBuilder,
        draw_calls: &[&Mutex<GameUi>],
        _pass: Self::PassIdentifier,
    ) -> Result<()> {
        self.gui.begin_frame();
        for &call in draw_calls {
            let _span = span!("draw egui");
            call.lock().draw_egui(&self.gui.egui_ctx)
        }
        let cmdbuf = self.gui.draw_on_subpass_image(self.image_dimensions);
        builder.execute_commands(cmdbuf)?;
        Ok(())
    }

    fn bind(
        &mut self,
        ctx: &crate::vulkan::VulkanContext,
        _per_frame_config: (),
        command_buf_builder: &mut crate::vulkan::CommandBufferBuilder,
        _pass: Self::PassIdentifier,
    ) -> Result<()> {
        command_buf_builder.next_subpass(SubpassContents::SecondaryCommandBuffers)?;
        self.image_dimensions = [ctx.window_size().0, ctx.window_size().1];
        Ok(())
    }
}
