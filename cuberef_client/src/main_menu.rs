use std::ops::Deref;

use anyhow::Context;
use egui::{Color32, Layout, ProgressBar, TextEdit};
use vulkano::{image::SampleCount, render_pass::Subpass};
use winit::{event::WindowEvent, event_loop::EventLoop};

use crate::vulkan::{
    game_renderer::{ConnectionState, GameState},
    VulkanContext,
};

pub(crate) struct MainMenu {
    egui_gui: egui_winit_vulkano::Gui,
    host_field: String,
    user_field: String,
    pass_field: String,
}
impl MainMenu {
    pub(crate) fn new(ctx: &VulkanContext, event_loop: &EventLoop<()>) -> MainMenu {
        let gui_config = egui_winit_vulkano::GuiConfig {
            preferred_format: Some(ctx.swapchain().image_format()),
            is_overlay: true,
            samples: SampleCount::Sample1,
        };
        let egui_gui = egui_winit_vulkano::Gui::new_with_subpass(
            event_loop,
            ctx.swapchain().surface().clone(),
            ctx.clone_queue(),
            Subpass::from(ctx.clone_render_pass(), 1)
                .context("Could not find subpass 0")
                .unwrap(),
            gui_config,
        );
        MainMenu {
            egui_gui,
            host_field: "".to_string(),
            user_field: "".to_string(),
            pass_field: "".to_string(),
        }
    }

    fn draw_ui<F>(&mut self, game_state: &mut GameState, connect_callback: F)
    where
        F: FnOnce(String, String, String) -> ConnectionState,
    {
        egui::CentralPanel::default().show(&self.egui_gui.egui_ctx, |ui| {
            ui.set_enabled(matches!(game_state, GameState::MainMenu));
            ui.visuals_mut().override_text_color = Some(Color32::WHITE);
            ui.with_layout(Layout::left_to_right(egui::Align::Min), |ui| {
                let label = ui.label("Server address: ");
                let editor = TextEdit::singleline(&mut self.host_field);
                ui.add(editor).labelled_by(label.id);
            });
            ui.with_layout(Layout::left_to_right(egui::Align::Min), |ui| {
                let label = ui.label("Username: ");
                let editor = TextEdit::singleline(&mut self.user_field);
                ui.add(editor).labelled_by(label.id);
            });
            ui.with_layout(Layout::left_to_right(egui::Align::Min), |ui| {
                let label = ui.label("Password (currently unused): ");
                let editor = TextEdit::singleline(&mut self.pass_field).password(true);
                ui.add(editor).labelled_by(label.id);
            });

            let connect_button = egui::Button::new("Connect");

            if ui.add(connect_button).clicked() {
                *game_state = GameState::Connecting(connect_callback(
                    self.host_field.clone(),
                    self.user_field.clone(),
                    self.pass_field.clone(),
                ));
            }
        });
        match game_state {
            GameState::MainMenu => {}
            GameState::Connecting(state) => {
                let progress = state.progress.borrow();
                let (progress_fraction, state) = progress.deref();
                egui::Window::new("Connecting...").collapsible(false).show(
                    &self.egui_gui.egui_ctx,
                    |ui| {
                        ui.visuals_mut().override_text_color = Some(Color32::WHITE);
                        ui.label(state);
                        ui.add(ProgressBar::new(*progress_fraction));
                        // TODO cancel button
                    },
                );
            }
            GameState::ConnectError(e) => {
                let message = e.clone();
                egui::Window::new("Connection error")
                    .collapsible(false)
                    .show(&self.egui_gui.egui_ctx, |ui| {
                        ui.visuals_mut().override_text_color = Some(Color32::WHITE);
                        ui.label(message);
                        if ui.button("OK").clicked() {
                            *game_state = GameState::MainMenu;
                        }
                    });
            }
            GameState::Active(_) => {
                egui::Window::new("Connection error")
                    .collapsible(false)
                    .show(&self.egui_gui.egui_ctx, |ui| {
                        ui.visuals_mut().override_text_color = Some(Color32::WHITE);
                        ui.label(
                            "Game is active. This message should not appear (please file a bug)",
                        );
                    });
            }
        }
    }
    pub(crate) fn draw<L, F>(
        &mut self,
        ctx: &VulkanContext,
        game_state: &mut GameState,
        builder: &mut crate::vulkan::CommandBufferBuilder<L>,
        connect_callback: F,
    ) where
        F: FnOnce(String, String, String) -> ConnectionState,
    {
        self.egui_gui.begin_frame();
        self.draw_ui(game_state, connect_callback);
        builder
            .next_subpass(vulkano::command_buffer::SubpassContents::SecondaryCommandBuffers)
            .unwrap();
        let secondary = self
            .egui_gui
            .draw_on_subpass_image([ctx.window_size().0, ctx.window_size().1]);

        builder.execute_commands(secondary).unwrap();
    }
    pub(crate) fn update(&mut self, event: &WindowEvent) {
        self.egui_gui.update(event);
    }
}
