use anyhow::Result;
use egui::{
    vec2, Align2, Button, Color32, Context, Id, ScrollArea, Sense, Stroke, TextEdit, TextStyle,
    TextureId, Vec2b,
};
use perovskite_core::chat::ChatMessage;
use perovskite_core::items::ItemStackExt;
use perovskite_core::protocol::items::ItemStack;
use perovskite_core::protocol::ui::{self as proto, PopupResponse};
use perovskite_core::protocol::{items::item_def::QuantityType, ui::PopupDescription};

use super::hud::render_number;
use super::{get_texture, FRAME_UNSELECTED};
use crate::client_state::items::InventoryViewManager;
use crate::client_state::settings::GameSettings;
use crate::client_state::tool_controller::ToolState;
use crate::client_state::{ClientPerformanceMetrics, GameAction, InventoryAction};
use crate::main_menu::{draw_settings_menu, InputCapture};
use crate::vulkan::shaders::flat_texture::{FlatTextureDrawBuilder, FlatTextureDrawCall};
use crate::vulkan::{VulkanContext, VulkanWindow};
use crate::{
    client_state::{items::ClientItemManager, ClientState},
    vulkan::Texture2DHolder,
};
use arc_swap::ArcSwap;
use egui::load::SizedTexture;
use egui_plot::{BarChart, Orientation};
use parking_lot::MutexGuard;
use perovskite_core::protocol::game_rpc::ServerPerformanceMetrics;
use rustc_hash::FxHashMap;
use std::ops::ControlFlow;
use std::time::{Duration, Instant};
use std::{collections::HashMap, sync::Arc, usize};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InvClickType {
    LeftClick,
    RightClick,
}

pub(crate) struct EguiUi {
    texture_atlas: Arc<Texture2DHolder>,
    atlas_coords: HashMap<String, texture_packer::Rect>,
    item_defs: Arc<ClientItemManager>,

    inventory_open: bool,
    pause_menu_open: bool,
    settings_menu_open: bool,
    chat_open: bool,
    chat_force_request_focus: bool,
    chat_force_cursor_to_end: bool,
    chat_force_scroll_to_end: bool,
    pub(crate) inventory_view: Option<PopupDescription>,
    scale: f32,
    scale_override: f32,

    debug_open: bool,
    perf_open: bool,
    interact_menu_last_min: usize,

    visible_popups: Vec<PopupDescription>,

    text_fields: FxHashMap<(u64, String), String>,
    checkboxes: FxHashMap<(u64, String), bool>,

    pub(crate) inventory_manipulation_view_id: Option<u64>,
    last_mouse_position: egui::Pos2,
    stack_carried_by_mouse_offset: (f32, f32),
    allow_inventory_interaction: bool,
    allow_button_interaction: bool,

    chat_message_input: String,
    chat_scroll_counter: usize,

    prospective_settings: GameSettings,
    server_perf_records: ServerPerformanceMetrics,
    client_perf_records: ClientPerformanceMetrics,

    status_bar: Option<(Instant, String)>,
}
impl EguiUi {
    pub(crate) fn new(
        texture_atlas: Arc<Texture2DHolder>,
        atlas_coords: HashMap<String, texture_packer::Rect>,
        item_defs: Arc<ClientItemManager>,
        settings: Arc<ArcSwap<GameSettings>>,
    ) -> EguiUi {
        EguiUi {
            texture_atlas,
            atlas_coords,
            item_defs,
            inventory_open: false,
            pause_menu_open: false,
            settings_menu_open: false,
            chat_open: false,
            chat_force_request_focus: false,
            chat_force_cursor_to_end: false,
            chat_force_scroll_to_end: false,
            debug_open: false,
            perf_open: false,
            interact_menu_last_min: 0,

            inventory_view: None,
            scale: 1.0,
            scale_override: 1.0,
            visible_popups: vec![],
            text_fields: FxHashMap::default(),
            checkboxes: FxHashMap::default(),
            inventory_manipulation_view_id: None,
            last_mouse_position: egui::Pos2 { x: 0., y: 0. },
            stack_carried_by_mouse_offset: (0., 0.),
            allow_inventory_interaction: true,
            allow_button_interaction: true,

            chat_message_input: String::new(),
            chat_scroll_counter: 0,

            prospective_settings: (**settings.load()).clone(),
            server_perf_records: ServerPerformanceMetrics::default(),
            client_perf_records: ClientPerformanceMetrics::default(),
            status_bar: None,
        }
    }
    pub(crate) fn wants_user_events(&self) -> bool {
        self.inventory_open
            || !self.visible_popups.is_empty()
            || self.pause_menu_open
            || self.chat_open
    }
    pub(crate) fn open_inventory(&mut self) {
        self.inventory_open = true;
    }
    pub(crate) fn open_pause_menu(&mut self) {
        self.pause_menu_open = true;
    }
    pub(crate) fn open_chat(&mut self) {
        self.chat_open = true;
        self.chat_force_request_focus = true;
        self.chat_force_scroll_to_end = true;
    }
    pub(crate) fn set_allow_inventory_interaction(&mut self, allow: bool) {
        self.allow_inventory_interaction = allow;
    }
    pub(crate) fn set_allow_button_interaction(&mut self, allow: bool) {
        self.allow_button_interaction = allow;
    }
    pub(crate) fn open_chat_slash(&mut self) {
        self.chat_open = true;
        self.chat_message_input = "/".to_string();
        self.chat_force_request_focus = true;
        self.chat_force_cursor_to_end = true;
        self.chat_force_scroll_to_end = true;
    }

    pub(crate) fn toggle_debug(&mut self) {
        self.debug_open = !self.debug_open;
    }
    pub(crate) fn toggle_perf(&mut self) {
        self.perf_open = !self.perf_open;
    }
    pub(crate) fn draw_all_uis(
        &mut self,
        ctx: &Context,
        atlas_texture_id: TextureId,
        client_state: &ClientState,
        input_capture: &mut InputCapture,
        vk_ctx: &VulkanWindow,
        tool_state: &ToolState,
    ) {
        self.scale = ctx.input(|i| i.pixels_per_point);
        // TODO have more things controlled by the scale. e.g. font sizes?
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::Plus)) {
            self.scale_override += 0.1;
        } else if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::Minus)) {
            self.scale_override -= 0.1;
        } else if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::Num0)) {
            self.scale_override = 1.0;
        }
        self.scale_override = self.scale_override.clamp(0.1, 2.0);

        if !self.visible_popups.is_empty() {
            for popup in self
                .visible_popups
                .clone()
                .iter()
                .take(self.visible_popups.len() - 1)
            {
                // Only consider control flow from the last popup
                let _ = self.draw_popup(popup, ctx, atlas_texture_id, client_state, false);
            }
            let result = self.draw_popup(
                &self.visible_popups.last().unwrap().clone(),
                ctx,
                atlas_texture_id,
                client_state,
                !self.pause_menu_open,
            );
            if result.is_break() {
                self.visible_popups.pop();
            }
        } else if self.inventory_open {
            let result = self.draw_popup(
                &self
                    .inventory_view
                    .clone()
                    .unwrap_or_else(|| PopupDescription {
                        popup_id: u64::MAX,
                        title: "Error".to_string(),
                        element: vec![proto::UiElement {
                            element: Some(proto::ui_element::Element::Label(
                                "The server hasn't sent a definition for the inventory popup"
                                    .to_string(),
                            )),
                        }],
                    }),
                ctx,
                atlas_texture_id,
                client_state,
                !self.pause_menu_open,
            );
            if result.is_break() {
                self.inventory_open = false;
            }
        }

        if let Some(pos) = ctx.input(|i| i.pointer.hover_pos()) {
            self.last_mouse_position = pos;
        }

        if self.pause_menu_open {
            self.draw_pause_menu(ctx, client_state, !self.settings_menu_open);
        }

        if self.settings_menu_open {
            match draw_settings_menu(
                ctx,
                &client_state.settings,
                &mut self.prospective_settings,
                input_capture,
                vk_ctx,
            ) {
                ControlFlow::Continue(()) => {}
                ControlFlow::Break(()) => {
                    vk_ctx.request_recreate();
                    self.settings_menu_open = false;
                }
            }
        }

        // these render a TopBottomPanel, so they need to be last
        if let Some((timeout, message)) = &self.status_bar {
            if *timeout > Instant::now() {
                self.render_status_bar(ctx, &message)
            }
        }
        if self.debug_open {
            self.render_debug(ctx, client_state, tool_state);
        }
        client_state
            .want_server_perf
            .store(self.perf_open, std::sync::atomic::Ordering::Relaxed);
        if self.perf_open {
            self.render_perf(ctx, client_state);
        }
        self.maybe_render_interact_menu(ctx, client_state, tool_state);

        // Chat comes last, since otherwise its panel messes with coordinate calculations
        // for hover text
        self.render_chat_history(ctx, client_state);
    }

    pub(crate) fn push_status_bar(&mut self, duration: Duration, message: String) {
        self.status_bar = Some((Instant::now() + duration, message))
    }

    pub(crate) fn is_perf_panel_open(&self) -> bool {
        self.perf_open
    }

    fn get_text_fields(&self, popup_id: u64) -> HashMap<String, String> {
        self.text_fields
            .iter()
            .filter(|((popup, _), _)| popup == &popup_id)
            .map(|((_, form_key), value)| (form_key.clone(), value.clone()))
            .collect()
    }
    fn get_checkboxes(&self, popup_id: u64) -> HashMap<String, bool> {
        self.checkboxes
            .iter()
            .filter(|((popup, _), _)| popup == &popup_id)
            .map(|((_, form_key), value)| (form_key.clone(), *value))
            .collect()
    }
    fn clear_fields(&mut self, popup: &PopupDescription) {
        self.text_fields
            .retain(|(popup_id, _), _| popup_id != &popup.popup_id);
        self.checkboxes
            .retain(|(popup_id, _), _| popup_id != &popup.popup_id);
    }

    fn render_element(
        &mut self,
        ui: &mut egui::Ui,
        popup: &PopupDescription,
        element: &proto::UiElement,
        id: Id,
        atlas_texture_id: TextureId,
        client_state: &ClientState,
        clicked_button: &mut Option<(String, bool)>,
    ) {
        match &element.element {
            Some(proto::ui_element::Element::Label(label)) => {
                ui.label(label);
            }
            Some(proto::ui_element::Element::TextField(text_field)) => {
                let value = self
                    .text_fields
                    .entry((popup.popup_id, text_field.key.clone()))
                    .or_insert(text_field.initial.clone());
                // todo support multiline, other styling
                if text_field.multiline {
                    ScrollArea::both()
                        .id_source("scroll_".to_string() + text_field.key.as_str())
                        .max_width(320.0)
                        .max_height(240.0)
                        .show(ui, |ui| {
                            let label = ui.label(text_field.label.clone());
                            let resp = ui
                                .add_enabled(text_field.enabled, TextEdit::multiline(value))
                                .labelled_by(label.id);
                            if !text_field.hover_text.is_empty() {
                                resp.on_hover_text(text_field.hover_text.as_str());
                            }
                        });
                } else {
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Min), |ui| {
                        let label = ui.label(text_field.label.clone());
                        let resp = ui
                            .add_enabled(text_field.enabled, TextEdit::singleline(value))
                            .labelled_by(label.id);
                        if !text_field.hover_text.is_empty() {
                            resp.on_hover_text(text_field.hover_text.as_str());
                        }
                    });
                };
            }
            Some(proto::ui_element::Element::Checkbox(checkbox)) => {
                let value = self
                    .checkboxes
                    .entry((popup.popup_id, checkbox.key.clone()))
                    .or_insert(checkbox.initial);
                ui.checkbox(value, checkbox.label.clone());
            }
            Some(proto::ui_element::Element::Button(button_def)) => {
                let button = Button::new(button_def.label.clone());
                if ui
                    .add_enabled(button_def.enabled && self.allow_button_interaction, button)
                    .clicked()
                    && self.allow_button_interaction
                {
                    *clicked_button = Some((button_def.key.clone(), button_def.will_close_popup));
                }
            }
            Some(proto::ui_element::Element::Inventory(inventory)) => {
                let mut inventory_manager = client_state.inventories.lock();
                let label = if inventory.label.is_empty() {
                    "Inventory (click to expand/collapse)"
                } else {
                    inventory.label.as_str()
                };
                egui::CollapsingHeader::new(label)
                    .default_open(true)
                    .id_source(Id::new("collapsing_header_inv").with(inventory.inventory_key))
                    .show(ui, |ui| {
                        self.draw_inventory_view(
                            ui,
                            inventory.inventory_key,
                            &mut inventory_manager,
                            atlas_texture_id,
                            client_state,
                        );
                    });
            }
            Some(proto::ui_element::Element::SideBySide(side_by_side)) => {
                egui::CollapsingHeader::new(&side_by_side.header)
                    .id_source(id)
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.with_layout(egui::Layout::left_to_right(egui::Align::TOP), |ui| {
                            for (index, sub_element) in side_by_side.element.iter().enumerate() {
                                self.render_element(
                                    ui,
                                    popup,
                                    sub_element,
                                    id.with(index),
                                    atlas_texture_id,
                                    client_state,
                                    clicked_button,
                                );
                            }
                        });
                    });
            }
            None => {
                ui.label(
                    "Invalid/missing popup item entry. You may need to update your game client.",
                );
            }
        }
    }

    fn draw_popup(
        &mut self,
        popup: &PopupDescription,
        ctx: &Context,
        atlas_texture_id: TextureId,
        client_state: &ClientState,
        enabled: bool,
    ) -> ControlFlow<(), ()> {
        let center = [ctx.screen_rect().max.x * 0.4, ctx.screen_rect().max.y * 0.4];

        egui::Window::new(popup.title.clone())
            .id(Id::new(("SERVER_POPUP", popup.popup_id)))
            .collapsible(false)
            .resizable(true)
            .default_pos(center)
            .show(ctx, |ui| {
                ui.set_enabled(enabled);
                ui.visuals_mut().override_text_color = Some(Color32::WHITE);

                let mut clicked_button = None;
                for (index, element) in popup.element.iter().enumerate() {
                    self.render_element(
                        ui,
                        popup,
                        element,
                        Id::new("popup_elem").with(popup.popup_id).with(index),
                        atlas_texture_id,
                        client_state,
                        &mut clicked_button,
                    );
                }
                if let Some((button_name, close)) = clicked_button {
                    send_event(
                        client_state,
                        GameAction::PopupResponse(PopupResponse {
                            popup_id: popup.popup_id,
                            closed: false,
                            clicked_button: button_name,
                            text_fields: self.get_text_fields(popup.popup_id),
                            checkboxes: self.get_checkboxes(popup.popup_id),
                        }),
                    );
                    if close {
                        self.clear_fields(popup);
                        return ControlFlow::Break(());
                    }
                }
                if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    send_event(
                        client_state,
                        GameAction::PopupResponse(PopupResponse {
                            popup_id: popup.popup_id,
                            closed: true,
                            clicked_button: "".to_string(),
                            text_fields: self.get_text_fields(popup.popup_id),
                            checkboxes: self.get_checkboxes(popup.popup_id),
                        }),
                    );
                    self.clear_fields(popup);
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            })
            .and_then(|x| x.inner)
            .unwrap_or(ControlFlow::Continue(()))
    }

    pub(crate) fn show_popup(&mut self, desc: &PopupDescription) {
        self.visible_popups.push(desc.clone())
    }

    pub(crate) fn get_carried_itemstack(
        &self,
        vk_ctx: &VulkanContext,
        client_state: &ClientState,
    ) -> Result<Option<FlatTextureDrawCall>> {
        if let Some(stack) = self
            .inventory_manipulation_view_id
            .and_then(|x| {
                client_state
                    .inventories
                    .lock()
                    .inventory_views
                    .get(&x)
                    .and_then(|x| x.contents().get(0).cloned())
            })
            .flatten()
        {
            let scale_for_sizing = if client_state
                .settings
                .load()
                .render
                .scale_inventories_with_high_dpi
            {
                self.scale
            } else {
                1.0
            } / self.scale_override;

            let x = self.stack_carried_by_mouse_offset.0 + self.last_mouse_position.x;
            let y = self.stack_carried_by_mouse_offset.1 + self.last_mouse_position.y;
            let texture_rect = get_texture(&stack, &self.atlas_coords, &self.item_defs);
            // only needed for its size, a bit inefficient but fine for now
            let frame_pixels = *self.atlas_coords.get(FRAME_UNSELECTED).unwrap();
            // todo get rid of this hardcoding :(
            let position_rect = texture_packer::Rect::new(
                // These want the actual physical scale
                (x * self.scale) as u32,
                (y * self.scale) as u32,
                // and these should be adjusted per the inventory scale
                ((frame_pixels.w - 4) as f32 * scale_for_sizing) as u32,
                ((frame_pixels.h - 4) as f32 * scale_for_sizing) as u32,
            );

            let mut builder = FlatTextureDrawBuilder::new();
            builder.rect(position_rect, texture_rect, self.texture_atlas.dimensions());
            let frame_topright = (position_rect.right(), position_rect.top());
            // todo unify this with hud.rs
            if stack.stackable() && stack.quantity != 1 {
                render_number(
                    frame_topright,
                    stack.quantity,
                    &mut builder,
                    &self.atlas_coords,
                    &self.texture_atlas,
                )
            }

            Ok(Some(builder.build(vk_ctx)?))
        } else {
            Ok(None)
        }
    }

    fn draw_inventory_view(
        &mut self,
        ui: &mut egui::Ui,
        view_id: u64,
        inventory_manager: &mut MutexGuard<InventoryViewManager>,
        atlas_texture: TextureId,
        client_state: &ClientState,
    ) {
        let mut clicked_index = None;
        let inventory = match inventory_manager.inventory_views.get(&view_id) {
            Some(x) => x,
            None => {
                ui.label(format!("Inventory {} missing/not loaded", view_id));
                return;
            }
        };

        let dims = inventory.dimensions;
        let contents = inventory.contents();

        if contents.len() != dims.0 as usize * dims.1 as usize {
            ui.label(format!(
                "Error: inventory is {}*{} but server only sent {} stacks",
                dims.0,
                dims.1,
                contents.len()
            ));
            return;
        }

        let frame_pixels = *self.atlas_coords.get(FRAME_UNSELECTED).unwrap();

        let frame_uv = self.pixel_rect_to_uv(frame_pixels);

        let frame_size = if client_state
            .settings
            .load()
            .render
            .scale_inventories_with_high_dpi
        {
            vec2(frame_pixels.w as f32, frame_pixels.h as f32) * self.scale_override
        } else {
            vec2(frame_pixels.w as f32, frame_pixels.h as f32) * self.scale_override / (self.scale)
        };
        let inv_rect = egui::Rect::from_min_size(
            ui.cursor().min,
            vec2(frame_size.x * dims.1 as f32, frame_size.y * dims.0 as f32),
        );
        ui.allocate_rect(inv_rect, Sense::click_and_drag());

        for row in 0..dims.0 {
            for col in 0..dims.1 {
                let index = ((row * dims.1) + col) as usize;
                let frame_image = egui::Image::from_texture(SizedTexture {
                    id: atlas_texture,
                    size: frame_size,
                })
                .uv(frame_uv)
                .sense(Sense::click_and_drag());

                let min_corner =
                    inv_rect.min + vec2(frame_size.x * col as f32, frame_size.y * row as f32);

                let frame_rect = egui::Rect::from_min_size(min_corner, frame_size);
                let drawing_rect = egui::Rect::from_min_size(
                    min_corner + vec2(2.0, 2.0),
                    frame_size - vec2(4.0, 4.0),
                );

                let frame_response = ui.put(frame_rect, frame_image);

                if frame_response.clicked() && self.allow_inventory_interaction {
                    clicked_index = Some((index, InvClickType::LeftClick));
                    self.stack_carried_by_mouse_offset = (-frame_size.x / 2., -frame_size.y / 2.)
                } else if frame_response.clicked_by(egui::PointerButton::Secondary) {
                    clicked_index = Some((index, InvClickType::RightClick));
                }

                if let Some(stack) = &contents[index] {
                    frame_response.on_hover_text(self.item_defs.get(&stack.item_name).map_or_else(
                        || format!("Unknown item {}", stack.item_name),
                        |x| x.display_name.clone(),
                    ));

                    if !stack.item_name.is_empty() {
                        let texture_uv = self.get_texture_uv(stack);
                        let image = egui::Image::from_texture(SizedTexture {
                            id: atlas_texture,
                            size: drawing_rect.size(),
                        })
                        .uv(texture_uv)
                        .sense(Sense::hover());
                        ui.put(drawing_rect, image);
                    }
                    let stacking_type = self
                        .item_defs
                        .get(&stack.item_name)
                        .and_then(|x| x.quantity_type.clone())
                        .unwrap_or(QuantityType::Stack(256));

                    match stacking_type {
                        QuantityType::Stack(_) => {
                            if stack.quantity != 1 {
                                let mut text = stack.quantity.to_string();
                                let label_rect = egui::Rect::from_min_size(
                                    min_corner + frame_size - vec2(36.0, 28.0),
                                    vec2(36.0, 20.0),
                                );
                                ui.with_layout(
                                    egui::Layout::left_to_right(egui::Align::TOP),
                                    |ui| {
                                        ui.style_mut().visuals.extreme_bg_color =
                                            Color32::from_rgba_unmultiplied(0, 0, 0, 128);
                                        ui.style_mut().visuals.window_stroke = Stroke::NONE;
                                        ui.put(
                                            label_rect,
                                            TextEdit::singleline(&mut text)
                                                .font(TextStyle::Heading)
                                                .text_color(Color32::WHITE)
                                                .interactive(false)
                                                .horizontal_align(egui::Align::Max),
                                        );
                                    },
                                );
                            }
                        }
                        QuantityType::Wear(max_wear) => {
                            // pass
                            // todo replace temporary text with a wear bar
                            let wear_bar_corner = drawing_rect.left_bottom() + vec2(0.0, -4.0);
                            let wear_level =
                                ((stack.current_wear as f32) / (max_wear as f32)).clamp(0.0, 1.0);
                            let wear_bar_rectangle = egui::Rect::from_min_size(
                                wear_bar_corner,
                                vec2(wear_level * drawing_rect.width(), 2.0),
                            );
                            let wear_bucket = ((wear_level * 8.0) as u8).clamp(0, 7);
                            let wear_texture = format!("builtin:wear_{}", wear_bucket);
                            let texture_uv = self.pixel_rect_to_uv(
                                *self
                                    .atlas_coords
                                    .get(&wear_texture)
                                    .unwrap_or_else(|| panic!("Missing texture {}", wear_texture)),
                            );

                            let wear_bar_image = egui::Image::from_texture(SizedTexture {
                                id: atlas_texture,
                                size: vec2(wear_bar_rectangle.width(), wear_bar_rectangle.height()),
                            })
                            .uv(texture_uv);
                            ui.put(wear_bar_rectangle, wear_bar_image);
                        }
                    }
                }
            }
        }
        if let Some(manipulation_view) = self.inventory_manipulation_view_id {
            if let Some((index, click_type)) = clicked_index {
                handle_moves(
                    view_id,
                    index,
                    inventory_manager,
                    manipulation_view,
                    click_type,
                    view_id,
                    client_state,
                );
            }
        }
    }

    pub(crate) fn clone_texture_atlas(&self) -> Arc<Texture2DHolder> {
        self.texture_atlas.clone()
    }

    pub(crate) fn pixel_rect_to_uv(&self, pixel_rect: texture_packer::Rect) -> egui::Rect {
        let width = self.texture_atlas.dimensions().0 as f32;
        let height = self.texture_atlas.dimensions().1 as f32;
        let left = pixel_rect.left() as f32 / width;
        let right = (pixel_rect.right() + 1) as f32 / width;

        let top = pixel_rect.top() as f32 / height;
        let bottom = (pixel_rect.bottom() + 1) as f32 / height;

        egui::Rect::from_x_y_ranges(left..=right, top..=bottom)
    }

    pub(crate) fn get_texture_uv(&self, item: &ItemStack) -> egui::Rect {
        let pixel_rect = get_texture(item, &self.atlas_coords, &self.item_defs);
        self.pixel_rect_to_uv(pixel_rect)
    }

    fn draw_pause_menu(&mut self, ctx: &Context, client_state: &ClientState, enabled: bool) {
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.pause_menu_open = false;
        }
        egui::Window::new("Game paused")
            .collapsible(false)
            .resizable(false)
            .enabled(enabled)
            .anchor(egui::Align2::CENTER_CENTER, vec2(0.0, 0.0))
            .show(ctx, |ui| {
                if ui.add_enabled(true, Button::new("Resume")).clicked() {
                    self.pause_menu_open = false;
                }
                if ui.add_enabled(true, Button::new("Settings")).clicked() {
                    self.prospective_settings = (**client_state.settings.load()).clone();
                    match self.prospective_settings.audio.fill_audio_devices() {
                        Ok(_) => {}
                        Err(e) => {
                            log::error!("Failure loading audio devices: {}", e);
                            self.prospective_settings.audio.output_devices.clear();
                            self.prospective_settings
                                .audio
                                .output_devices
                                .push("[No devices]".to_string());
                        }
                    };
                    self.settings_menu_open = true;
                }
                if ui
                    .add_enabled(true, Button::new("Return to Main Menu"))
                    .clicked()
                {
                    client_state.shutdown.cancel();
                }
                if ui.add_enabled(true, Button::new("Quit")).clicked() {
                    client_state.shutdown.cancel();
                    *client_state.wants_exit_from_game.lock() = true;
                }
            });
    }

    fn render_chat_history(&mut self, ctx: &Context, client_state: &ClientState) {
        let chat = client_state.chat.lock();
        if self.chat_open {
            egui::TopBottomPanel::top("chat_panel")
                .max_height(260.0)
                .show(ctx, |ui| {
                    let scroll_area = egui::ScrollArea::vertical()
                        .auto_shrink([false, true])
                        .min_scrolled_height(240.0)
                        .id_source("chat_history")
                        .max_height(240.0);
                    let messages = &chat.message_history;
                    scroll_area.show(ui, |ui| {
                        for message in messages {
                            ui.horizontal_top(|ui| {
                                let formatted_message = format_chat(message);
                                ui.label(formatted_message.0);
                                ui.add(egui::Label::new(formatted_message.1).wrap());
                            });
                        }
                        if self.chat_scroll_counter != messages.len()
                            || self.chat_force_scroll_to_end
                        {
                            ui.scroll_to_cursor(Some(egui::Align::Max));
                            self.chat_scroll_counter = messages.len();
                            self.chat_force_scroll_to_end = false;
                        }
                    });
                    let editor = ui.add(
                        TextEdit::singleline(&mut self.chat_message_input)
                            .hint_text("Type a message; press Enter to send or Escape to close.")
                            .lock_focus(true)
                            .desired_width(f32::INFINITY)
                            .font(egui::FontId::proportional(16.0)),
                    );
                    if self.chat_force_request_focus {
                        self.chat_force_request_focus = false;
                        ui.ctx().memory_mut(|m| m.request_focus(editor.id));
                    }
                    if self.chat_force_cursor_to_end {
                        self.chat_force_cursor_to_end = false;
                        let id = editor.id;
                        if let Some(mut state) = TextEdit::load_state(ui.ctx(), id) {
                            let ccursor =
                                egui::text::CCursor::new(self.chat_message_input.chars().count());
                            state
                                .cursor
                                .set_char_range(Some(egui::text::CCursorRange::one(ccursor)));
                            state.store(ui.ctx(), id);
                            ui.ctx().memory_mut(|m| m.request_focus(id)); // give focus back to the `TextEdit`.
                        }
                    }
                    ui.memory_mut(|m| m.request_focus(editor.id));
                    if ui.input(|i| i.key_pressed(egui::Key::Enter))
                        && !self.chat_message_input.trim().is_empty()
                    {
                        // If this is failing to unwrap, we have pretty serious problems and may as well crash
                        client_state
                            .actions
                            .blocking_send(GameAction::ChatMessage(self.chat_message_input.clone()))
                            .unwrap();

                        self.chat_message_input.clear();
                    } else if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                        self.chat_message_input.clear();
                        self.chat_open = false;
                    }
                });
        } else {
            const SHORT_MESSAGE_HISTORY_LEN: usize = 8;
            const SHORT_MESSAGE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
            let messages = &chat
                .message_history
                .iter()
                .rev()
                .take(SHORT_MESSAGE_HISTORY_LEN)
                .rev()
                .filter(|x| x.timestamp().elapsed() < SHORT_MESSAGE_TIMEOUT)
                .collect::<Vec<_>>();
            if !messages.is_empty() {
                egui::TopBottomPanel::top("chat_panel")
                    .max_height(240.0)
                    .frame(egui::Frame {
                        fill: Color32::from_black_alpha(192),
                        stroke: Stroke {
                            width: 0.0,
                            color: Color32::TRANSPARENT,
                        },
                        ..Default::default()
                    })
                    .show(ctx, |ui| {
                        let scroll_area = egui::ScrollArea::vertical()
                            .auto_shrink([false, true])
                            .min_scrolled_height(240.0)
                            .max_height(240.0);

                        scroll_area.show(ui, |ui| {
                            for message in &messages
                                [messages.len().saturating_sub(SHORT_MESSAGE_HISTORY_LEN)..]
                            {
                                ui.horizontal_top(|ui| {
                                    let formatted_message = format_chat(message);
                                    ui.label(formatted_message.0);
                                    ui.add(egui::Label::new(formatted_message.1).wrap());
                                });
                            }
                            ui.scroll_to_cursor(Some(egui::Align::Max));
                        });
                    });
            }
        }
    }

    fn render_status_bar(&self, ctx: &Context, message: &str) {
        egui::TopBottomPanel::bottom("status_bar")
            .max_height(60.0)
            .frame(egui::Frame {
                fill: Color32::from_black_alpha(192),
                stroke: Stroke {
                    width: 0.0,
                    color: Color32::TRANSPARENT,
                },
                ..Default::default()
            })
            .show(ctx, |ui| {
                ui.label(
                    egui::RichText::new(message)
                        .font(egui::FontId::monospace(16.0))
                        .color(Color32::WHITE),
                );
            });
    }

    fn render_hover_text_only(&mut self, ctx: &Context, state: &ClientState, text: Option<&str>) {
        if let Some(text) = text {
            let contents = |ui: &mut egui::Ui| {
                ui.label(
                    egui::RichText::new(text)
                        .font(egui::FontId::new(
                            16.0,
                            egui::FontFamily::Name("MonospaceBestEffort".into()),
                        ))
                        .color(Color32::WHITE),
                );
            };
            if state.settings.load().display.hover_text_on_bottom_panel {
                egui::TopBottomPanel::bottom("hover_text")
                    .frame(egui::Frame {
                        fill: Color32::from_black_alpha(192),
                        stroke: Stroke {
                            width: 0.0,
                            color: Color32::TRANSPARENT,
                        },
                        ..Default::default()
                    })
                    .show(ctx, contents);
            } else {
                let center = [
                    ctx.screen_rect().center().x + 5.0,
                    ctx.screen_rect().center().y + 5.0,
                ];
                egui::Window::new("Hover text")
                    .title_bar(false)
                    .anchor(Align2::LEFT_TOP, center)
                    .frame(egui::Frame {
                        fill: Color32::from_black_alpha(192),
                        stroke: Stroke {
                            width: 0.0,
                            color: Color32::TRANSPARENT,
                        },
                        ..Default::default()
                    })
                    .show(ctx, contents);
            }
        }
    }

    fn maybe_render_interact_menu(
        &mut self,
        ctx: &Context,
        state: &ClientState,
        tool_state: &ToolState,
    ) {
        let opts = match &tool_state.interact_key_options {
            Some(x) => x,
            None => {
                self.interact_menu_last_min = 0;
                self.render_hover_text_only(ctx, state, tool_state.hover_text.as_deref());
                return;
            }
        };
        if opts.is_empty() {
            self.interact_menu_last_min = 0;
            self.render_hover_text_only(ctx, state, tool_state.hover_text.as_deref());
            return;
        }
        //    selection
        //    v
        // <--*>
        // ^
        // view_range_min
        let mut min = self.interact_menu_last_min;
        let mut max = (min + 4).min(opts.len() - 1);
        if tool_state.selected_interact_option <= min {
            min = tool_state.selected_interact_option.saturating_sub(1);
            max = (min + 4).min(max);
        } else if tool_state.selected_interact_option >= max {
            max = (tool_state.selected_interact_option + 1).min(opts.len() - 1);
            min = max.saturating_sub(4);
        }
        self.interact_menu_last_min = min;

        let min_sigil = if min == 0 { "    " } else { " ↑  " };
        let max_sigil = if max >= opts.len() - 1 {
            "    "
        } else {
            " ↓  "
        };
        let contents = |ui: &mut egui::Ui| {
            if let Some(hover_text) = &tool_state.hover_text {
                ui.label(
                    egui::RichText::new(hover_text)
                        .font(egui::FontId::new(
                            16.0,
                            egui::FontFamily::Name("MonospaceBestEffort".into()),
                        ))
                        .color(Color32::WHITE),
                );
            }
            for idx in min..=max {
                let entry = &opts[idx];
                let sigil = if idx == tool_state.selected_interact_option {
                    "[>] "
                } else if idx == min {
                    min_sigil
                } else if idx == max {
                    max_sigil
                } else {
                    "    "
                };
                ui.label(
                    egui::RichText::new(format!("{sigil}{}", &entry.label))
                        .font(egui::FontId::new(
                            16.0,
                            egui::FontFamily::Name("MonospaceBestEffort".into()),
                        ))
                        .color(if idx == tool_state.selected_interact_option {
                            Color32::WHITE
                        } else {
                            Color32::DARK_GRAY
                        }),
                );
            }
        };
        if state.settings.load().display.hover_text_on_bottom_panel {
            egui::TopBottomPanel::bottom("interact_key_menu")
                .frame(egui::Frame {
                    fill: Color32::from_black_alpha(192),
                    stroke: Stroke {
                        width: 0.0,
                        color: Color32::TRANSPARENT,
                    },
                    ..Default::default()
                })
                .show(ctx, contents);
        } else {
            let center = [
                ctx.screen_rect().center().x + 5.0,
                ctx.screen_rect().center().y + 5.0,
            ];
            egui::Window::new("Interact Menu")
                .title_bar(false)
                .anchor(Align2::LEFT_TOP, center)
                .frame(egui::Frame {
                    fill: Color32::from_black_alpha(192),
                    stroke: Stroke {
                        width: 0.0,
                        color: Color32::TRANSPARENT,
                    },
                    ..Default::default()
                })
                .show(ctx, contents);
        }
    }

    fn render_debug(&mut self, ctx: &Context, state: &ClientState, tool_state: &ToolState) {
        egui::TopBottomPanel::bottom("debug_panel")
            .max_height(240.0)
            .frame(egui::Frame {
                fill: Color32::from_black_alpha(192),
                stroke: Stroke {
                    width: 0.0,
                    color: Color32::TRANSPARENT,
                },
                ..Default::default()
            })
            .show(ctx, |ui| {
                let pos = state.weakly_ordered_last_position();
                ui.label(
                    egui::RichText::new(format!(
                        "Location: ({:.3}, {:.3}, {:.3}), facing ({:.1}, {:.1})",
                        // cast to i32 to avoid spamming a ton of decimal places from the
                        // smoothing algorithm
                        pos.position.x,
                        pos.position.y,
                        pos.position.z,
                        pos.face_direction.0,
                        pos.face_direction.1
                    ))
                    .font(egui::FontId::monospace(16.0))
                    .color(Color32::WHITE),
                );
                let pointee = tool_state.pointee_debug(state);
                let color = if pointee.is_some() {
                    Color32::WHITE
                } else {
                    Color32::DARK_GRAY
                };
                ui.label(
                    egui::RichText::new(pointee.as_deref().unwrap_or("No pointee"))
                        .font(egui::FontId::monospace(16.0))
                        .color(color),
                );
                {
                    let entity_lock = state.entities.lock();
                    if let Some(entity_id) = entity_lock.attached_to_entity {
                        let id = entity_id.entity_id;
                        let tei = entity_id.trailing_entity_index;
                        if let Some(entity) = entity_lock.entities.get(&entity_id.entity_id) {
                            let start_tick = state.timekeeper.now();
                            let debug_speed = entity.debug_speed(start_tick);
                            let estimated_buffer = entity.estimated_buffer(start_tick).max(0.0);
                            let estimated_buffer_count = entity.estimated_buffer_count();
                            let cms = entity.debug_cms();
                            let cme = entity.debug_cme(start_tick);
                            ui.label(
                                egui::RichText::new(format!(
                                    "Attached entity: {id}:{tei}, speed: {debug_speed:.2}, \
                                estimated buffer: {estimated_buffer:.2}, \
                                estimated buffer count: {estimated_buffer_count}, \
                                current move sequence: {cms}, current move elapsed: {cme:.2}"
                                ))
                                .font(egui::FontId::monospace(16.0))
                                .color(Color32::WHITE),
                            );
                            if state.settings.load().debug.extra_entity_debug {
                                let buffer_debug = entity.buffer_debug(start_tick);
                                ui.label(
                                    egui::RichText::new(buffer_debug)
                                        .font(egui::FontId::monospace(16.0))
                                        .color(Color32::WHITE),
                                );
                            }
                        } else {
                            ui.label(
                                egui::RichText::new(format!("Attached entity: {id}:{tei}"))
                                    .font(egui::FontId::monospace(16.0))
                                    .color(Color32::DARK_GRAY),
                            );
                        }
                    }

                    let (av, ai) = state.chunks.average_solid_batch_occupancy();
                    ui.label(
                        egui::RichText::new(format!(
                            "Average solid mesh batch: {av} vertices, {ai} indices"
                        ))
                        .font(egui::FontId::monospace(16.0))
                        .color(Color32::WHITE),
                    );
                }
            });
    }

    fn get_bars_and_update_records(
        current_data: &Vec<u64>,
        records: &mut Vec<u64>,
        color: egui::Color32,
    ) -> Vec<egui_plot::Bar> {
        let color2 = color.gamma_multiply(0.8);

        if records.len() < current_data.len() {
            records.resize(current_data.len(), 0);
        }
        let mut result = vec![];
        for (i, (data, record)) in current_data.iter().zip(records.iter_mut()).enumerate() {
            *record = (*record).max(*data);
            let diff = *record - *data;
            result.push(egui_plot::Bar {
                name: "".to_string(),
                orientation: Orientation::Vertical,
                argument: i as f64,
                value: *data as f64,
                base_offset: None,
                bar_width: 1.0,
                stroke: Stroke::NONE,
                fill: if i % 2 == 0 { color } else { color2 },
            });
            result.push(egui_plot::Bar {
                name: "".to_string(),
                orientation: Orientation::Vertical,
                argument: i as f64,
                value: diff as f64,
                base_offset: Some(*data as f64),
                bar_width: 1.0,
                stroke: Stroke::NONE,
                fill: Color32::DARK_GRAY,
            });
        }
        result
    }

    fn render_perf_chart(
        ui: &mut egui::Ui,
        title: &str,
        data: &Vec<u64>,
        records: &mut Vec<u64>,
        plot_name: &str,
        color: egui::Color32,
    ) {
        let mut bars = vec![];
        bars.append(&mut Self::get_bars_and_update_records(data, records, color));
        ui.label(
            egui::RichText::new(format!(
                "{}\n(chart max={})",
                title,
                records.iter().copied().max().unwrap_or(0)
            ))
            .color(Color32::WHITE),
        );
        egui_plot::Plot::new(plot_name)
            .allow_drag(false)
            .allow_scroll(false)
            .auto_bounds(Vec2b::TRUE)
            .allow_boxed_zoom(false)
            .show_x(false)
            .show_y(false)
            .show_grid(Vec2b::FALSE)
            .width(data.len() as f32 * 8.0)
            .height(48.0)
            .show_axes(Vec2b::FALSE)
            .show(ui, |ui| ui.bar_chart(BarChart::new(bars)));
    }

    fn render_perf(&mut self, ctx: &Context, state: &ClientState) {
        let server_perf = state.server_perf();
        let client_perf = state.client_perf();

        egui::TopBottomPanel::bottom("perf_panel")
            .max_height(240.0)
            .frame(egui::Frame {
                fill: Color32::from_black_alpha(192),
                stroke: Stroke {
                    width: 0.0,
                    color: Color32::TRANSPARENT,
                },
                ..Default::default()
            })
            .show(ctx, |ui| {
                ui.horizontal_top(|ui| {
                    if let Some(perf) = server_perf {
                        ui.vertical(|ui| {
                            Self::render_perf_chart(
                                ui,
                                "Shard writeback queue",
                                &perf.mapshard_writeback_len,
                                &mut self.server_perf_records.mapshard_writeback_len,
                                "writeback_plot",
                                Color32::RED,
                            );
                        });
                        ui.vertical(|ui| {
                            Self::render_perf_chart(
                                ui,
                                "Shard occupancy",
                                &perf.mapshard_loaded_chunks,
                                &mut self.server_perf_records.mapshard_loaded_chunks,
                                "occupancy_plot",
                                Color32::RED,
                            );
                        });
                    } else {
                        ui.label("No server perf data");
                    }

                    if let Some(perf) = client_perf {
                        ui.vertical(|ui| {
                            Self::render_perf_chart(
                                ui,
                                "Lighting/neighbor queue",
                                &perf.nprop_queue_lens,
                                &mut self.client_perf_records.nprop_queue_lens,
                                "writeback_plot",
                                Color32::GREEN,
                            );
                        });
                        ui.vertical(|ui| {
                            Self::render_perf_chart(
                                ui,
                                "Mesh queue",
                                &perf.mesh_queue_lens,
                                &mut self.client_perf_records.mesh_queue_lens,
                                "occupancy_plot",
                                Color32::GREEN,
                            );
                        });
                    } else {
                        ui.label("No server perf data");
                    }
                });
            });
    }
}

fn format_chat(message: &ChatMessage) -> (egui::RichText, egui::RichText) {
    let color = message.origin_color();
    (
        egui::RichText::new(format!(
            "[{} seconds ago] {}",
            message.timestamp().elapsed().as_secs(),
            message.origin()
        ))
        .color(Color32::from_rgb(color.0, color.1, color.2))
        .size(16.0),
        egui::RichText::strong(message.text().into()).size(16.0),
    )
}

// Allow unnecessary_unwrap until https://github.com/rust-lang/rust/issues/53667 is stabilized
#[allow(clippy::unnecessary_unwrap)]
fn handle_moves(
    clicked_inv_id: u64,
    index: usize,
    inventory_manager: &mut MutexGuard<InventoryViewManager>,
    manipulation_view: u64,
    click_type: InvClickType,
    view_key: u64,
    client_state: &ClientState,
) {
    // we have to do a bit of a song-and-dance until get_many_mut is stabilized
    // This code generates the actual request that the server will authoritatively validate.
    // For fluid gameplay, we need to generate requests that we expect the server will fulfill as-is
    // and update our own views consistently (the server will send us an update regardless)
    //
    // For now, it'll be a bit inefficient and awkward
    // TODO refactor it into perovskite_core so it can be properly shared
    let inventory = inventory_manager
        .inventory_views
        .get(&clicked_inv_id)
        .unwrap();
    let clicked_stack = inventory.contents()[index].clone();
    let can_place = inventory.can_place;
    let can_take = inventory.can_take;
    let take_exact = inventory.take_exact;
    let place_without_swap = inventory.put_without_swap;

    let carried_stack = match inventory_manager
        .inventory_views
        .get_mut(&manipulation_view)
        .and_then(|x| x.contents_mut().get_mut(0))
    {
        Some(x) => x,
        None => {
            log::warn!("Manipulation view was empty or not loaded");
            return;
        }
    };

    let action = if clicked_stack.is_some() && carried_stack.is_some() && !place_without_swap {
        // Case 1: both stacks have items
        let clicked_stack = clicked_stack.unwrap();
        let carried_stack_ref = carried_stack.as_ref().unwrap();

        if can_place
            && clicked_stack.item_name == carried_stack_ref.item_name
            && clicked_stack.stackable()
        {
            // You can place into the inventory, and you may or may not be able to take.
            // If the items match, place what you're holding
            let mut quantity = clicked_stack
                .max_stack()
                .saturating_sub(clicked_stack.quantity)
                .min(carried_stack_ref.quantity);
            if click_type == InvClickType::RightClick {
                quantity = quantity.min(1);
            }
            // move out of the carried stack
            if quantity > 0 && (quantity == carried_stack_ref.quantity) {
                *carried_stack = None;
            } else if quantity > 0 {
                carried_stack.as_mut().unwrap().quantity -= quantity;
            } else {
                // early return, don't send anything
                return;
            }
            // and into the clicked stack
            inventory_manager
                .inventory_views
                .get_mut(&view_key)
                .unwrap()
                .contents_mut()[index]
                .as_mut()
                .unwrap()
                .quantity += quantity;
            InventoryAction {
                source_view: manipulation_view,
                source_slot: 0,
                destination_view: view_key,
                destination_slot: index,
                count: quantity,
                swap: false,
            }
        } else if can_take
            && clicked_stack.item_name == carried_stack_ref.item_name
            && carried_stack_ref.stackable()
        {
            // You can take from the inventory, and your items match. Take from it.

            let mut quantity = carried_stack_ref
                .max_stack()
                .saturating_sub(carried_stack_ref.quantity)
                .min(clicked_stack.quantity);
            if !take_exact && click_type == InvClickType::RightClick {
                quantity = quantity.min((clicked_stack.quantity + 1) / 2);
            }
            if take_exact && quantity < clicked_stack.quantity {
                // we can't take enough, but we need to take exactly what's given, give up
                return;
            }

            // move into the carried stack
            carried_stack.as_mut().unwrap().quantity += quantity;
            // and out of the clicked stack
            if quantity > 0 && (quantity == clicked_stack.quantity) {
                inventory_manager
                    .inventory_views
                    .get_mut(&view_key)
                    .unwrap()
                    .contents_mut()[index] = None;
            } else if quantity > 0 {
                inventory_manager
                    .inventory_views
                    .get_mut(&view_key)
                    .unwrap()
                    .contents_mut()[index]
                    .as_mut()
                    .unwrap()
                    .quantity -= quantity;
            } else {
                // early return, don't send anything
                return;
            }

            InventoryAction {
                source_view: view_key,
                source_slot: index,
                destination_view: manipulation_view,
                destination_slot: 0,
                count: quantity,
                swap: false,
            }
        } else if can_place && can_take {
            // You can both give and take, and your items don't match. Swap.

            let carried = carried_stack_ref.clone();
            *carried_stack = Some(clicked_stack);
            inventory_manager
                .inventory_views
                .get_mut(&view_key)
                .unwrap()
                .contents_mut()[index] = Some(carried);

            InventoryAction {
                source_view: manipulation_view,
                source_slot: 0,
                destination_view: view_key,
                destination_slot: index,
                count: 0,
                swap: true,
            }
        } else {
            // Both have items, we couldn't combine stacks in either direction,
            // and we don't have enough permissions to swap. Nothing else to do.
            return;
        }
    }
    // We can place, and the destination is empty, or we aren't swapping.
    else if carried_stack.is_some() && can_place {
        let old_carried_stack = carried_stack.clone();
        let source_quantity = carried_stack.as_ref().unwrap().quantity;
        let mut quantity = source_quantity;
        if click_type == InvClickType::RightClick {
            quantity = quantity.min(1);
        }
        if quantity == source_quantity {
            // move out of the carried stack
            *carried_stack = None;
            // and into the destination
            inventory_manager
                .inventory_views
                .get_mut(&view_key)
                .unwrap()
                .contents_mut()[index] = old_carried_stack;
        } else {
            carried_stack.as_mut().unwrap().quantity -= quantity;
            inventory_manager
                .inventory_views
                .get_mut(&view_key)
                .unwrap()
                .contents_mut()[index] = Some(ItemStack {
                quantity,
                ..old_carried_stack.unwrap()
            });
        }
        InventoryAction {
            source_view: manipulation_view,
            source_slot: 0,
            destination_view: view_key,
            destination_slot: index,
            count: quantity,
            swap: false,
        }
    } else if clicked_stack.is_some() && can_take {
        let source_quantity = clicked_stack.as_ref().unwrap().quantity;
        let mut quantity = source_quantity;
        if !take_exact && click_type == InvClickType::RightClick {
            quantity = quantity.min((quantity + 1) / 2);
        }
        if source_quantity == quantity {
            // move into the carried stack
            *carried_stack = clicked_stack;
            // and out of the clicked stack
            inventory_manager
                .inventory_views
                .get_mut(&view_key)
                .unwrap()
                .contents_mut()[index] = None;
        } else {
            *carried_stack = Some(ItemStack {
                quantity,
                ..clicked_stack.unwrap()
            });
            inventory_manager
                .inventory_views
                .get_mut(&view_key)
                .unwrap()
                .contents_mut()[index]
                .as_mut()
                .unwrap()
                .quantity -= quantity;
        }
        InventoryAction {
            source_view: view_key,
            source_slot: index,
            destination_view: manipulation_view,
            destination_slot: 0,
            count: quantity,
            swap: false,
        }
    } else {
        // nothing to move, return
        return;
    };

    send_event(client_state, GameAction::Inventory(action));
}

fn send_event(client_state: &ClientState, action: GameAction) {
    if client_state.actions.try_send(action).is_err() {
        log::info!("Sending action failed; server disconnected or lagging badly");
    }
}
