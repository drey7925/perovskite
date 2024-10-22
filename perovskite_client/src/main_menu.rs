use std::fmt::Display;
use std::ops::ControlFlow;
use std::str::FromStr;
use std::{ops::Deref, sync::Arc};

use anyhow::Context;
use arc_swap::ArcSwap;
use egui::{
    CollapsingHeader, Color32, FontId, InnerResponse, Layout, ProgressBar, RichText, TextEdit, Ui,
};
use tokio::sync::{oneshot, watch};
use vulkano::command_buffer::SubpassContents::SecondaryCommandBuffers;
use vulkano::command_buffer::{SubpassBeginInfo, SubpassEndInfo};
use vulkano::{image::SampleCount, render_pass::Subpass};
use winit::event::ElementState;
use winit::{event::WindowEvent, event_loop::EventLoop};

use crate::game_state::input::{BoundAction, Keybind, KeybindSettings};
use crate::game_state::settings::Supersampling;
use crate::vulkan::shaders::egui_adapter::set_up_fonts;
use crate::vulkan::VulkanContext;
use crate::{
    game_state::settings::GameSettings,
    vulkan::{
        game_renderer::{ConnectionSettings, ConnectionState, GameState},
        VulkanWindow,
    },
};

#[derive(PartialEq, Eq, Debug, Copy, Clone)]
pub(crate) enum InputCapture {
    NotCapturing,
    Capturing(BoundAction),
    Captured(BoundAction, Keybind),
}

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
    show_settings_popup: bool,
    previous_servers: Vec<String>,
    settings: Arc<ArcSwap<GameSettings>>,
    prospective_settings: GameSettings,
    settings_parse_error_acknowledged: bool,
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
        let mut egui_gui = egui_winit_vulkano::Gui::new_with_subpass(
            event_loop,
            ctx.swapchain().surface().clone(),
            ctx.clone_graphics_queue(),
            Subpass::from(ctx.post_blit_render_pass(), 0)
                .context("Could not find subpass 0")
                .unwrap(),
            ctx.swapchain().image_format(),
            gui_config,
        );
        set_up_fonts(&mut egui_gui.egui_ctx);
        let mut style = (*egui_gui.egui_ctx.style()).clone();
        style
            .text_styles
            .insert(egui::TextStyle::Body, FontId::proportional(16.0));
        egui_gui.egui_ctx.set_style(style);
        let settings_guard = settings.load();
        let prospective_settings = (**settings_guard).clone();
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
            show_settings_popup: false,
            previous_servers: settings_guard.previous_servers.clone(),
            settings,
            prospective_settings,
            settings_parse_error_acknowledged: false,
        }
    }

    fn draw_ui(
        &mut self,
        game_state: &mut GameState,
        vk_ctx: &mut VulkanWindow,
        input_capture: &mut InputCapture,
    ) -> Option<ConnectionSettings> {
        let mut result = None;
        let enable_main_controls = self.should_enable_main_controls(&game_state);
        egui::CentralPanel::default().show(&self.egui_gui.egui_ctx, |ui| {
            ui.set_enabled(enable_main_controls);
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
            ui.label(
                RichText::new(
                    "TEST ONLY binary for debugging login issues, DO NOT USE REAL PASSWORDS",
                )
                    .color(Color32::RED)
                    .size(16.0)
                    .strong(),
            );
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

            let settings_button = egui::Button::new("Settings");
            if ui.add(settings_button).clicked() {
                self.show_settings_popup = true;
            }
        });

        if !self.settings_parse_error_acknowledged {
            if let Some(err) = self
                .settings
                .load()
                .internal_parsing_failure_message
                .as_ref()
            {
                egui::Window::new("Settings error").collapsible(false).show(
                    &self.egui_gui.egui_ctx,
                    |ui| {
                        ui.visuals_mut().override_text_color = Some(Color32::WHITE);
                        ui.label(
                            "The settings file could not be parsed. Default settings will be used; recent servers will not be saved.",
                        );
                        ui.label(err);
                        if ui.button("OK").clicked()
                            || ui.input(|i| i.key_pressed(egui::Key::Enter))
                        {
                            self.settings_parse_error_acknowledged = true;
                        }
                    },
                );
            }
        }

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
            egui::Window::new("Register new account").show(&self.egui_gui.egui_ctx, |ui| {
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
        if self.show_settings_popup {
            match draw_settings_menu(
                &mut self.egui_gui.egui_ctx,
                &self.settings,
                &mut self.prospective_settings,
                input_capture,
                vk_ctx,
            ) {
                ControlFlow::Continue(_) => {}
                ControlFlow::Break(_) => {
                    vk_ctx.request_recreate();
                    self.show_settings_popup = false;
                }
            };
        }

        result
    }

    pub(crate) fn draw<L>(
        &mut self,
        ctx: &mut VulkanWindow,
        game_state: &mut GameState,
        builder: &mut crate::vulkan::CommandBufferBuilder<L>,
        input_capture: &mut InputCapture,
    ) -> Option<ConnectionSettings> {
        self.egui_gui.begin_frame();
        let result = self.draw_ui(game_state, ctx, input_capture);
        let secondary = self
            .egui_gui
            .draw_on_subpass_image([ctx.window_size().0, ctx.window_size().1]);

        builder.execute_commands(secondary).unwrap();
        result
    }
    pub(crate) fn update(&mut self, event: &WindowEvent) {
        self.egui_gui.update(event);
    }
    fn should_enable_main_controls(&self, game_state: &GameState) -> bool {
        if let GameState::MainMenu = game_state {
            if self.show_register_popup {
                false
            } else if self
                .settings
                .load()
                .internal_parsing_failure_message
                .is_some()
                && !self.settings_parse_error_acknowledged
            {
                false
            } else if self.show_settings_popup {
                false
            } else {
                true
            }
        } else {
            false
        }
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

pub(crate) fn draw_settings_menu(
    egui_ctx: &egui::Context,
    settings: &ArcSwap<GameSettings>,
    prospective_settings: &mut GameSettings,
    input_capture: &mut InputCapture,
    vk_ctx: &VulkanContext,
) -> ControlFlow<(), ()> {
    let mut result = ControlFlow::Continue(());

    let max_height = (egui_ctx.available_rect().height() - 96.0).max(128.0);
    egui::Window::new("Settings").max_height(max_height).show(egui_ctx, |ui| {
        ui.label("Mouse over the name of a setting to view its description.");
        if prospective_settings.internal_parsing_failure_message.is_some() {
            ui.label("Warning: Settings file is corrupt; saving will overwrite with default settings.");
        }
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.set_min_width(250.0);

            CollapsingHeader::new("Input settings")
                .default_open(true)
                .show(ui, |ui| {
                    draw_input_settings(ui, prospective_settings, input_capture);
                });


            CollapsingHeader::new("Render settings")
                .default_open(true)
                .show(ui, |ui| {
                    draw_render_settings(ui, prospective_settings, vk_ctx);
                });
        });
        ui.with_layout(Layout::left_to_right(egui::Align::Min), |ui| {
            let save_button = egui::Button::new("Save");
            if ui.add(save_button).clicked() {
                settings.store(Arc::new(prospective_settings.clone()));
                if let Err(e) = prospective_settings.save_to_disk() {
                    log::error!("Failure saving settings: {}", e);
                    // TODO show error popup
                }
                result = ControlFlow::Break(());
            }
            let cancel_button = egui::Button::new("Cancel");
            if ui.add(cancel_button).clicked() || ui.input(|x| x.key_pressed(egui::Key::Escape)) {
                *prospective_settings = (**settings.load()).clone();
                result = ControlFlow::Break(());
            }
        });
    });
    result
}

fn draw_input_settings(
    ui: &mut Ui,
    prospective_settings: &mut GameSettings,
    input_capture: &mut InputCapture,
) -> InnerResponse<()> {
    egui::Grid::new("input_grid")
        .num_columns(4)
        .spacing([40.0, 4.0])
        .striped(true)
        .show(ui, |ui| {
            ui.label("Camera sensitivity")
                .on_hover_text("How sensitive the camera is. Higher values are more sensitive.");
            ui.add(
                egui::Slider::new(
                    &mut prospective_settings.input.camera_sensitivity,
                    0.01..=1.0,
                ),
            );
            ui.end_row();
            ui.label("Inverse scroll sensitivity")
                .on_hover_text("How many pixels of scrolling correspond to one slot in the hotbar. Higher values mean lower sensitivity.");
            ui.add(
                egui::Slider::new(
                    &mut prospective_settings.input.scroll_inverse_sensitivity,
                    20..=400,
                )
            );
            ui.end_row();

            if let InputCapture::Captured(action, keybind) = input_capture {
                prospective_settings.input.set(*action, *keybind);
                *input_capture = InputCapture::NotCapturing;
            }
            for action in BoundAction::all_bound_actions() {
                ui.label(action.user_friendly_name()).on_hover_text(action.tooltip());

                let button_text = if *input_capture == InputCapture::Capturing(*action) {
                    "...".to_string()
                } else {
                    format!("{:?}", prospective_settings.input.get(*action))
                };

                if ui.button(button_text).clicked() {
                    *input_capture = InputCapture::Capturing(*action);
                }

                if ui.button("Reset").clicked() {
                    let default = KeybindSettings::default().get(*action);
                    prospective_settings.input.set(*action, default);
                }

                ui.end_row();
            }

        })
}

fn draw_render_settings(
    ui: &mut Ui,
    prospective_settings: &mut GameSettings,
    vk_ctx: &VulkanContext,
) {
    egui::Grid::new("render_grid")
        .num_columns(2)
        .spacing([40.0, 4.0])
        .striped(true)
        .show(ui, |ui| {
            ui.label("Mesh threads")
                .on_hover_text("Mesh threads convert preprocessed chunk data into 3D meshes that can be rendered.");
            ui.add(
                egui::Slider::new(
                    &mut prospective_settings.render.num_mesh_workers,
                    1..=8,
                )
                    .suffix(" threads"),
            );
            ui.end_row();

            ui.label("Pre-mesh threads")
                .on_hover_text("Pre-mesh threads compute lighting and prepare neighbor data as input to mesh threads.");
            ui.add(
                egui::Slider::new(
                    &mut prospective_settings
                        .render
                        .num_neighbor_propagators,
                    1..=8,
                )
                    .suffix(" threads"),
            );
            ui.end_row();

            // TODO: Re-implement the placement guide in a less obtrusive manner
            // before re-enabling.
            //
            // ui.label("Show placement guide");
            // ui.checkbox(
            //     &mut self.prospective_settings.render.show_placement_guide,
            //     "Enable",
            // );
            // ui.end_row();

            let gpu_label = ui.label("Preferred GPU")
                .on_hover_text("The GPU to use, if available.");
            let selected_gpu =
                if prospective_settings.render.preferred_gpu.is_empty() {
                    "Select..."
                } else {
                    &prospective_settings.render.preferred_gpu
                };
            let mut preferred_gpu =
                prospective_settings.render.preferred_gpu.clone();
            egui::ComboBox::from_id_source(gpu_label.id)
                .selected_text(selected_gpu)
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut preferred_gpu,
                        String::new(),
                        "No preference",
                    );
                    for gpu in vk_ctx.all_gpus() {
                        ui.selectable_value(
                            &mut preferred_gpu,
                            gpu.to_string(),
                            gpu.to_string(),
                        );
                    }
                });
            prospective_settings.render.preferred_gpu = preferred_gpu;
            ui.end_row();

            ui.label("Scale inventory at high DPI")
                .on_hover_text("When on a high-density display, whether to make inventory tiles larger based on the display's scale.");
            ui.checkbox(
                &mut prospective_settings
                    .render
                    .scale_inventories_with_high_dpi,
                "Enable",
            );
            ui.end_row();

            ui.label("FOV")
                .on_hover_text("The field of view. Higher values give a wider view.");
            ui.add(
                egui::Slider::new(
                    &mut prospective_settings.render.fov_degrees,
                    30.0..=170.0,
                )
                    .suffix("°"),
            );
            ui.end_row();

            let ssaa_label = ui.label("Supersampling")
                .on_hover_text("Smooths edges of geometry and textures, at the expense of performance.");
            egui::ComboBox::from_id_source(ssaa_label.id)
                .selected_text(
                    match prospective_settings.render.supersampling {
                        Supersampling::None => "Disabled",
                        Supersampling::X2 => "x2",
                        Supersampling::X4 => "x4",
                        Supersampling::X8 => "x8",
                    },
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut prospective_settings.render.supersampling,
                        Supersampling::None,
                        "Disabled",
                    );
                    ui.selectable_value(
                        &mut prospective_settings.render.supersampling,
                        Supersampling::X2,
                        "x2",
                    );
                    ui.selectable_value(
                        &mut prospective_settings.render.supersampling,
                        Supersampling::X4,
                        "x4",
                    );
                    ui.selectable_value(
                        &mut prospective_settings.render.supersampling,
                        Supersampling::X8,
                        "x8",
                    );
                });
            ui.end_row();
        });
}
