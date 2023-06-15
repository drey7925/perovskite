use std::{collections::HashMap, fmt::format, sync::Arc, usize};

use cuberef_core::protocol::ui as proto;
use cuberef_core::protocol::{items::item_def::QuantityType, ui::PopupDescription};
use egui::{vec2, Color32, Sense, Stroke, TextEdit, TextStyle, TextureId};
use log::warn;

use crate::{
    game_state::{
        items::{ClientInventory, ClientItemManager},
        ClientState,
    },
    vulkan::Texture2DHolder,
};

use super::{FRAME_UNSELECTED, UNKNOWN_TEXTURE};

pub(crate) struct EguiUi {
    texture_atlas: Arc<Texture2DHolder>,
    atlas_coords: HashMap<String, texture_packer::Rect>,
    item_defs: Arc<ClientItemManager>,

    inventory_open: bool,
    pub(crate) inventory_view: Option<PopupDescription>,
}
impl EguiUi {
    pub(crate) fn new(
        texture_atlas: Arc<Texture2DHolder>,
        atlas_coords: HashMap<String, texture_packer::Rect>,
        item_defs: Arc<ClientItemManager>,
    ) -> EguiUi {
        EguiUi {
            texture_atlas,
            atlas_coords,
            item_defs,
            inventory_open: false,
            inventory_view: None,
        }
    }
    pub(crate) fn wants_draw(&self) -> bool {
        self.inventory_open
        // todo other popups
    }
    pub(crate) fn open_inventory(&mut self) {
        self.inventory_open = true;
    }
    pub(crate) fn draw_all_uis(
        &mut self,
        ctx: &egui::Context,
        atlas_texture_id: TextureId,
        client_state: &ClientState,
    ) {
        if self.inventory_open {
            self.draw_popup(
                &self.inventory_view.clone().unwrap_or_else(|| {
                    PopupDescription {
                        popup_id: u64::MAX,
                        title: "Error".to_string(),
                        element: vec![proto::UiElement {
                            element: Some(proto::ui_element::Element::Label(
                                "The server hasn't sent a definition for the inventory popup".to_string(),
                            )),
                        }],
                    }
                }),
                ctx,
                atlas_texture_id,
                client_state,
            )
        }
    }

    pub(crate) fn draw_popup(
        &mut self,
        popup: &proto::PopupDescription,
        ctx: &egui::Context,
        atlas_texture_id: TextureId,
        client_state: &ClientState,
    ) {
        // todo send a response
        let mut text_field_contents = HashMap::new();
        egui::Window::new(popup.title.clone()).show(ctx, |ui| {
            if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                // todo close the correct view, send events back
                self.inventory_open = false;
            }
            for element in &popup.element {
                match &element.element {
                    Some(proto::ui_element::Element::Label(label)) => {
                        ui.label(label);
                    }
                    Some(proto::ui_element::Element::TextField(text_field)) => {
                        let mut value = text_field.initial.clone();
                        // todo support multiline, other styling
                        let editor = egui::TextEdit::singleline(&mut value);
                        ui.add_enabled(text_field.enabled, editor);
                        text_field_contents.insert(text_field.key.clone(), value);
                    }
                    Some(proto::ui_element::Element::Button(button_def)) => {
                        let button = egui::Button::new(button_def.label.clone());
                        if ui.add_enabled(button_def.enabled, button).clicked() {
                            log::info!("todo handle button click");
                        }
                    }
                    Some(proto::ui_element::Element::Inventory(inventory)) => {
                        let inventory_manager = client_state.inventories.lock();
                        let view = inventory_manager
                            .inventory_views
                            .get(&inventory.inventory_key);
                        match view {
                            Some(x) => {
                                self.draw_inventory_view(ui, x, atlas_texture_id);
                            }
                            None => {
                                ui.label(format!(
                                    "Error: Inventory view {} not loaded",
                                    inventory.inventory_key
                                ));
                            }
                        }
                    }
                    None => todo!(),
                }
            }
        });
    }

    fn draw_inventory_view(
        &self,
        ui: &mut egui::Ui,
        inventory: &ClientInventory,
        atlas_texture: TextureId,
    ) {
        let dims = inventory.dimensions;
        let contents = inventory.contents();

        if contents.len() != dims.0 as usize * dims.1 as usize {
            ui.label(format!("Error: inventory is {}*{} but server only sent {} stacks", dims.0, dims.1, contents.len()));
            return;
        }

        let frame_pixels = *self.atlas_coords.get(FRAME_UNSELECTED).unwrap();
        let frame_uv = self.pixel_rect_to_uv(frame_pixels);

        let (inv_rect, _) = ui.allocate_exact_size(
            vec2(
                (frame_pixels.w * dims.1) as f32,
                (frame_pixels.h * dims.0) as f32,
            ),
            Sense::click_and_drag(),
        );

        let frame_size = vec2((frame_pixels.w) as f32, (frame_pixels.h) as f32);
        for row in 0..dims.0 {
            for col in 0..dims.1 {
                let index = ((row * dims.1) + col) as usize;
                let frame_image = egui::Image::new(atlas_texture, frame_size).uv(frame_uv);

                let min_corner = inv_rect.min
                    + vec2((frame_pixels.w * col) as f32, (frame_pixels.h * row) as f32);

                let frame_rect = egui::Rect::from_min_size(min_corner, frame_size);
                let drawing_rect = egui::Rect::from_min_size(
                    min_corner + vec2(2.0, 2.0),
                    frame_size - vec2(4.0, 4.0),
                );

                ui.put(frame_rect, frame_image);
                if let Some(stack) = &contents[index] {
                    if !stack.item_name.is_empty() {
                        let texture_uv = self.get_texture_uv(stack);
                        let image = egui::Image::new(
                            atlas_texture,
                            vec2(drawing_rect.width(), drawing_rect.height()),
                        )
                        .uv(texture_uv)
                        .sense(Sense::click_and_drag());
                        let response = ui.put(drawing_rect, image);
                        response.on_hover_text(self.item_defs.get(&stack.item_name).map_or_else(
                            || format!("Unknown item {}", stack.item_name),
                            |x| x.display_name.clone(),
                        ));
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
                                    min_corner + vec2(40.0, 2.0),
                                    frame_size - vec2(38.0, 48.0),
                                );
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::TOP),
                                    |ui| {
                                        ui.style_mut().visuals.extreme_bg_color =
                                            Color32::from_rgba_unmultiplied(0, 0, 0, 128);
                                        ui.style_mut().visuals.window_stroke = Stroke::NONE;
                                        //ui.put(label_rect, Label::new(text));
                                        ui.put(
                                            label_rect,
                                            TextEdit::singleline(&mut text)
                                                .font(TextStyle::Heading)
                                                .text_color(Color32::WHITE)
                                                .interactive(false)
                                                .horizontal_align(egui::Align::Max),
                                        )
                                    },
                                );
                            }
                        }
                        QuantityType::Wear(_) => {
                            warn!("TODO render wear");
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn texture_atlas(&self) -> &Texture2DHolder {
        self.texture_atlas.as_ref()
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

    pub(crate) fn get_texture_uv(
        &self,
        item: &cuberef_core::protocol::items::ItemStack,
    ) -> egui::Rect {
        let pixel_rect: texture_packer::Rect = self
            .item_defs
            .get(&item.item_name)
            .and_then(|x| x.inventory_texture.as_ref())
            .map(|x| x.texture_name.as_ref())
            .and_then(|x: &str| self.atlas_coords.get(x).copied())
            .unwrap_or(*self.atlas_coords.get(UNKNOWN_TEXTURE).unwrap());
        self.pixel_rect_to_uv(pixel_rect)
    }
}
