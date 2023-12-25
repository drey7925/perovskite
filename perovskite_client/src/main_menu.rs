use std::{ops::Deref, sync::Arc};

use anyhow::Context;
use arc_swap::ArcSwap;
use egui::{Color32, Layout, ProgressBar, RichText, TextEdit};
use tokio::sync::{oneshot, watch};
use vulkano::{image::SampleCount, render_pass::Subpass};
use winit::{event::WindowEvent, event_loop::EventLoop};

use crate::{
    game_state::settings::GameSettings,
    vulkan::{
        game_renderer::{ConnectionSettings, ConnectionState, GameState},
        VulkanWindow,
    },
};

pub(crate) struct MainMenu {
    egui_gui: egui_winit_vulkano::Gui,
    host_field: String,
    user_field: String,
    pass_field: String,
    register_host_field: String,
    register_user_field: String,
    register_pass_field: String,
    confirm_pass_field: String,
    show_register_popup: bool,
    previous_servers: Vec<String>,
    settings: Arc<ArcSwap<GameSettings>>,
}
impl MainMenu {
    pub(crate) fn new(
        ctx: &VulkanWindow,
        event_loop: &EventLoop<()>,
        settings: Arc<ArcSwap<GameSettings>>,
    ) -> MainMenu {
        let gui_config = egui_winit_vulkano::GuiConfig {
            allow_srgb_render_target: true,
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
            ctx.swapchain().image_format(),
            gui_config,
        );
        let settings_guard = settings.load();
        MainMenu {
            egui_gui,
            host_field: settings_guard.last_hostname.clone(),
            user_field: settings_guard.last_username.clone(),
            pass_field: "".to_string(),
            register_host_field: "".to_string(),
            register_user_field: "".to_string(),
            register_pass_field: "".to_string(),
            confirm_pass_field: "".to_string(),
            show_register_popup: false,
            previous_servers: settings_guard.previous_servers.clone(),
            settings,
        }
    }

    fn draw_ui(&mut self, game_state: &mut GameState) -> Option<ConnectionSettings> {
        let mut result = None;

        egui::CentralPanel::default().show(&self.egui_gui.egui_ctx, |ui| {
            ui.set_enabled(matches!(game_state, GameState::MainMenu) && !self.show_register_popup);
            ui.visuals_mut().override_text_color = Some(Color32::WHITE);
            if cfg!(debug_assertions) {
                ui.label(
                    RichText::new(
                        "Debug binary; will be unplayably slow.\nRecompile with --release!",
                    )
                    .color(Color32::RED)
                    .size(16.0)
                    .strong(),
                );
            }

            ui.with_layout(Layout::left_to_right(egui::Align::Min), |ui| {
                let label = ui.label("Server address: ");
                let editor = TextEdit::singleline(&mut self.host_field);
                ui.add(editor).labelled_by(label.id);
            });

            ui.with_layout(Layout::left_to_right(egui::Align::Min), |ui| {
                let label = ui.label("Recent servers: ");
                egui::ComboBox::from_id_source(label.id)
                    .selected_text("Select...")
                    .width(256.0)
                    .show_ui(ui, |ui| {
                        let mut fake_selectable = self.host_field.clone();
                        ui.selectable_value(
                            &mut fake_selectable,
                            "select".to_string(),
                            "Select...",
                        );
                        for server in self.previous_servers.iter().rev() {
                            ui.selectable_value(&mut self.host_field, server.clone(), server);
                        }
                    });
            });

            ui.with_layout(Layout::left_to_right(egui::Align::Min), |ui| {
                let label = ui.label("Username: ");
                let editor = TextEdit::singleline(&mut self.user_field);
                ui.add(editor).labelled_by(label.id);
            });
            ui.with_layout(Layout::left_to_right(egui::Align::Min), |ui| {
                let label = ui.label("Password: ");
                let editor = TextEdit::singleline(&mut self.pass_field).password(true);
                ui.add(editor).labelled_by(label.id);
            });

            let connect_button = egui::Button::new("Connect");
            let connect_enabled = matches!(game_state, GameState::MainMenu);
            let button_response = ui.add_enabled(connect_enabled, connect_button);
            if button_response.clicked()
                || (connect_enabled && ui.input(|i| i.key_pressed(egui::Key::Enter)))
            {
                self.settings.rcu(|x| {
                    let old = x.clone();
                    let mut new = GameSettings {
                        last_hostname: self.host_field.trim().to_string(),
                        last_username: self.user_field.trim().to_string(),
                        ..old.deref().clone()
                    };
                    new.push_hostname(self.host_field.trim().to_string());
                    new
                });
                self.previous_servers = self.settings.load().previous_servers.clone();
                if let Err(e) = self.settings.load().save_to_disk() {
                    log::error!("Failure saving settings: {}", e);
                }
                let (state, settings) = make_connection(
                    self.host_field.trim().to_string(),
                    self.user_field.trim().to_string(),
                    self.pass_field.trim().to_string(),
                    false,
                );
                self.pass_field.clear();
                *game_state = GameState::Connecting(state);
                result = Some(settings);
            }
            let register_button = egui::Button::new("Register New Account");

            if ui.add(register_button).clicked() {
                self.register_host_field = self.host_field.clone();
                self.register_user_field = self.user_field.clone();
                self.show_register_popup = true;
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
                        if ui.button("OK").clicked()
                            || ui.input(|i| i.key_pressed(egui::Key::Enter))
                        {
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
        if self.show_register_popup {
            egui::Window::new("Register new account: ").show(&self.egui_gui.egui_ctx, |ui| {
                ui.visuals_mut().override_text_color = Some(Color32::WHITE);
                ui.with_layout(Layout::left_to_right(egui::Align::Min), |ui| {
                    let label = ui.label("Server address: ");
                    let editor = TextEdit::singleline(&mut self.register_host_field);
                    ui.add(editor).labelled_by(label.id);
                });
                ui.with_layout(Layout::left_to_right(egui::Align::Min), |ui| {
                    let label = ui.label("Username: ");
                    let editor = TextEdit::singleline(&mut self.register_user_field);
                    ui.add(editor).labelled_by(label.id);
                });

                ui.with_layout(Layout::left_to_right(egui::Align::Min), |ui| {
                    let label = ui.label("Password: ");
                    let editor = TextEdit::singleline(&mut self.register_pass_field).password(true);
                    ui.add(editor).labelled_by(label.id);
                });
                ui.with_layout(Layout::left_to_right(egui::Align::Min), |ui| {
                    let label = ui.label("Confirm password: ");
                    let editor = TextEdit::singleline(&mut self.confirm_pass_field).password(true);
                    ui.add(editor).labelled_by(label.id);
                });
                if password_has_special_chars(self.register_pass_field.trim()) {
                    ui.colored_label(Color32::LIGHT_RED, textwrap::dedent("
                    Warning: Password contains special characters.
                    This is not recommended because special characters might be encoded inconsistently on different systems.").trim());
                }
                let register_button = egui::Button::new("Register");
                if ui.add(register_button).clicked() {
                    self.show_register_popup = false;
                    if self.register_pass_field != self.confirm_pass_field {
                        *game_state =
                            GameState::ConnectError("Passwords did not match".to_string());
                    } else if self.register_pass_field.trim().is_empty() {
                        *game_state = GameState::ConnectError(
                            "Please specify a non-empty password".to_string(),
                        );
                    } else {
                        self.settings.rcu(|x| {
                            let old = x.clone();
                            let mut new = GameSettings {
                                last_hostname: self.register_host_field.trim().to_string(),
                                last_username: self.register_user_field.trim().to_string(),
                                ..old.deref().clone()
                            };
                            new.push_hostname(self.register_host_field.trim().to_string());
                            new
                        });
                        self.previous_servers = self.settings.load().previous_servers.clone();
                        if let Err(e) = self.settings.load().save_to_disk() {
                            log::error!("Failure saving settings: {}", e);
                        }
                        let (state, settings) = make_connection(
                            self.register_host_field.trim().to_string(),
                            self.register_user_field.trim().to_string(),
                            self.register_pass_field.trim().to_string(),
                            true,
                        );
                        *game_state = GameState::Connecting(state);
                        result = Some(settings);
                    }
                    self.pass_field.clear();
                    self.register_pass_field.clear();
                    self.confirm_pass_field.clear();
                }
                let cancel_button = egui::Button::new("Cancel");
                if ui.add(cancel_button).clicked() || ui.input(|x| x.key_pressed(egui::Key::Escape)) {
                    self.show_register_popup = false;
                    self.register_pass_field.clear();
                    self.confirm_pass_field.clear();
                }
            });
        }

        result
    }

    pub(crate) fn draw<L>(
        &mut self,
        ctx: &VulkanWindow,
        game_state: &mut GameState,
        builder: &mut crate::vulkan::CommandBufferBuilder<L>,
    ) -> Option<ConnectionSettings> {
        self.egui_gui.begin_frame();
        let result = self.draw_ui(game_state);
        builder
            .next_subpass(vulkano::command_buffer::SubpassContents::SecondaryCommandBuffers)
            .unwrap();
        let secondary = self
            .egui_gui
            .draw_on_subpass_image([ctx.window_size().0, ctx.window_size().1]);

        builder.execute_commands(secondary).unwrap();
        result
    }
    pub(crate) fn update(&mut self, event: &WindowEvent) {
        self.egui_gui.update(event);
    }
}

fn password_has_special_chars(password: &str) -> bool {
    password.chars().any(|x| !x.is_ascii_graphic())
}

fn make_connection(
    host: String,
    user: String,
    pass: String,
    register: bool,
) -> (ConnectionState, ConnectionSettings) {
    let progress = watch::channel((0.0, "Starting connection...".to_string()));
    let result = oneshot::channel();

    let state = ConnectionState {
        progress: progress.1,
        result: result.1,
    };
    let settings = ConnectionSettings {
        host,
        user,
        pass,
        register,
        progress: progress.0,
        result: result.0,
    };
    (state, settings)
}
