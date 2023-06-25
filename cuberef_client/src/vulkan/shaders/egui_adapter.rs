use std::sync::Arc;

use egui::TextureId;
use egui_winit_vulkano::{Gui, GuiConfig};
use parking_lot::Mutex;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferInheritanceInfo, SubpassContents},
    image::SampleCount,
    render_pass::Subpass,
};
use winit::{
    event::WindowEvent,
    event_loop::{EventLoop, EventLoopWindowTarget},
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
            Subpass::from(ctx.render_pass.clone(), 1).context("Could not find subpass 0")?,
            config,
        );
        let atlas = egui_ui.lock().clone_texture_atlas();
        let atlas_texture_id = gui_adapter.register_user_image_view(atlas.clone_image_view());

        let flat_overlay_provider = FlatTexPipelineProvider::new(ctx.vk_device.clone())?;
        let flat_overlay_pipeline =
            flat_overlay_provider.make_pipeline(ctx, (atlas.as_ref(), 1))?;
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
        ctx: &VulkanContext,
        builder: &mut crate::vulkan::CommandBufferBuilder<L>,
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
        } else {
            // We still need to advance to the next subpass, even if we aren't drawing anything
            builder.next_subpass(SubpassContents::Inline)?;
            Ok(())
        }
    }

    pub(crate) fn notify_resize(&mut self, ctx: &VulkanContext) -> Result<()> {
        self.flat_overlay_pipeline = self
            .flat_overlay_provider
            .make_pipeline(ctx, (self.atlas.as_ref(), 1))?;
        Ok(())
    }
}
