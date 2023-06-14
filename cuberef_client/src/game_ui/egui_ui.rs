use std::{collections::HashMap, sync::Arc, usize};

use cuberef_core::protocol::items::item_def::QuantityType;
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
        }
    }
    pub(crate) fn wants_draw(&self) -> bool {
        false
    }
    pub(crate) fn draw_ui(
        &self,
        ctx: &egui::Context,
        atlas_texture_id: TextureId,
        client_state: &ClientState,
    ) {
        egui::Window::new("test window").show(ctx, |ui| {
            ui.heading("test app");
            ui.horizontal(|ui| {
                let name_label = ui.label("test label: ");
                ui.text_edit_singleline(&mut "foo")
                    .labelled_by(name_label.id)
            });
            if ui.button("test button").clicked() {};
            if let Some(main_inv) = client_state.hud.lock().hotbar_view_id {
                self.draw_inventory(
                    ui,
                    // testonly
                    client_state
                        .inventories
                        .lock()
                        .inventory_views
                        .get(&main_inv)
                        .unwrap(),
                    atlas_texture_id,
                );
            }
        });
    }

    fn draw_inventory(
        &self,
        ui: &mut egui::Ui,
        inventory: &ClientInventory,
        atlas_texture: TextureId,
    ) {
        let dims = inventory.dimensions;
        let contents = inventory.contents();
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
