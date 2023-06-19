use anyhow::Result;
use cuberef_core::protocol::items::ItemStack;
use cuberef_core::protocol::ui as proto;
use cuberef_core::protocol::{items::item_def::QuantityType, ui::PopupDescription};
use egui::{vec2, Color32, Sense, Stroke, TextEdit, TextStyle, TextureId};
use log::warn;
use parking_lot::MutexGuard;
use std::{collections::HashMap, sync::Arc, usize};

use crate::game_state::items::InventoryViewManager;
use crate::game_state::{GameAction, InventoryAction};
use crate::vulkan::shaders::flat_texture::{FlatTextureDrawBuilder, FlatTextureDrawCall};
use crate::vulkan::VulkanContext;
use crate::{
    game_state::{items::ClientItemManager, ClientState},
    vulkan::Texture2DHolder,
};

use super::hud::render_number;
use super::{get_texture, FRAME_UNSELECTED};

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
    pub(crate) inventory_view: Option<PopupDescription>,
    scale: f32,

    pub(crate) inventory_manipulation_view_id: Option<u64>,
    last_mouse_position: egui::Pos2,
    stack_carried_by_mouse_offset: (f32, f32),
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
            scale: 1.0,
            inventory_manipulation_view_id: None,
            last_mouse_position: egui::Pos2 { x: 0., y: 0. },
            stack_carried_by_mouse_offset: (0., 0.),
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
        self.scale = ctx.input(|i| i.pixels_per_point);
        if self.inventory_open {
            self.draw_popup(
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
            )
        }
        if let Some(pos) = ctx.input(|i| i.pointer.hover_pos()) {
            self.last_mouse_position = pos;
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
                        let mut inventory_manager = client_state.inventories.lock();

                        self.draw_inventory_view(
                            ui,
                            inventory.inventory_key,
                            &mut inventory_manager,
                            atlas_texture_id,
                            client_state,
                        );
                    }
                    None => todo!(),
                }
            }
        });
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
            let x = self.stack_carried_by_mouse_offset.0 + self.last_mouse_position.x;
            let y = self.stack_carried_by_mouse_offset.1 + self.last_mouse_position.y;
            let texture_rect = get_texture(&stack, &self.atlas_coords, &self.item_defs);
            // only needed for its size, a bit inefficient but fine for now
            let frame_pixels = *self.atlas_coords.get(FRAME_UNSELECTED).unwrap();
            // todo get rid of this hardcoding :(
            let position_rect = texture_packer::Rect::new(
                (x * self.scale) as u32,
                (y * self.scale) as u32,
                ((frame_pixels.w - 4) as f32 * self.scale) as u32,
                ((frame_pixels.h - 4) as f32 * self.scale) as u32,
            );

            let mut builder = FlatTextureDrawBuilder::new();
            builder.rect(position_rect, texture_rect, self.texture_atlas.dimensions());
            let frame_topright = (position_rect.right(), position_rect.top());
            // todo unify this with hud.rs
            if stack.max_stack > 1 && stack.quantity != 1 {
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
                let frame_image = egui::Image::new(atlas_texture, frame_size)
                    .uv(frame_uv)
                    .sense(Sense::click_and_drag());

                let min_corner = inv_rect.min
                    + vec2((frame_pixels.w * col) as f32, (frame_pixels.h * row) as f32);

                let frame_rect = egui::Rect::from_min_size(min_corner, frame_size);
                let drawing_rect = egui::Rect::from_min_size(
                    min_corner + vec2(2.0, 2.0),
                    frame_size - vec2(4.0, 4.0),
                );

                let frame_response = ui.put(frame_rect, frame_image);

                if frame_response.clicked() {
                    clicked_index = dbg!(Some((index, InvClickType::LeftClick)));
                    self.stack_carried_by_mouse_offset = (-frame_size.x / 2., -frame_size.y / 2.)
                } else if frame_response.clicked_by(egui::PointerButton::Secondary) {
                    clicked_index = dbg!(Some((index, InvClickType::RightClick)));
                }

                if let Some(stack) = &contents[index] {
                    frame_response.on_hover_text(self.item_defs.get(&stack.item_name).map_or_else(
                        || format!("Unknown item {}", stack.item_name),
                        |x| x.display_name.clone(),
                    ));

                    if !stack.item_name.is_empty() {
                        let texture_uv = self.get_texture_uv(stack);
                        let image = egui::Image::new(
                            atlas_texture,
                            vec2(drawing_rect.width(), drawing_rect.height()),
                        )
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
                            warn!("TODO render wear bar");
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

    pub(crate) fn texture_atlas(&self) -> &Texture2DHolder {
        self.texture_atlas.as_ref()
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

    pub(crate) fn get_texture_uv(
        &self,
        item: &cuberef_core::protocol::items::ItemStack,
    ) -> egui::Rect {
        let pixel_rect = get_texture(item, &self.atlas_coords, &self.item_defs);
        self.pixel_rect_to_uv(pixel_rect)
    }
}

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
    // For fluid gameplay, we need to generate requests that we expect the server will fulfull as-is
    // and update our own views consistently (the server will send us an update regardless)
    //
    // For now, it'll be a bit inefficient and awkward
    // TODO refactor it into cuberef_core so it can be properly shared
    let inventory = inventory_manager
        .inventory_views
        .get(&clicked_inv_id)
        .unwrap();
    let clicked_stack = inventory.contents()[index].clone();
    let can_place = inventory.can_place;
    let can_take = dbg!(inventory.can_take);
    let take_exact = inventory.take_exact;
    drop(inventory);
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

    let action = if clicked_stack.is_some() && carried_stack.is_some() {
        // Case 1: both stacks have items
        let clicked_stack_ref = clicked_stack.unwrap().clone();
        let carried_stack_ref = carried_stack.as_ref().unwrap();

        if can_place
            && clicked_stack_ref.item_name == carried_stack_ref.item_name
            && carried_stack_ref.stackable
        {
            // You can place into the inventory, and you may or may not be able to take.
            // If the items match, place what you're holding
            let mut quantity = clicked_stack_ref
                .max_stack
                .saturating_sub(clicked_stack_ref.quantity)
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
            && clicked_stack_ref.item_name == carried_stack_ref.item_name
            && clicked_stack_ref.stackable
        {
            // You can take from the inventory, and your items match. Take from it.

            let mut quantity = carried_stack_ref
                .max_stack
                .saturating_sub(carried_stack_ref.quantity)
                .min(clicked_stack_ref.quantity);
            if !take_exact && click_type == InvClickType::RightClick {
                quantity = quantity.min((clicked_stack_ref.quantity + 1) / 2);
            }
            if take_exact && quantity < clicked_stack_ref.quantity {
                // we can't take enough but we need to take exactly what's given, give up
                return;
            }
            drop(carried_stack_ref);
            // move into the carried stack
            carried_stack.as_mut().unwrap().quantity += quantity;
            // and out of the clicked stack
            if quantity > 0 && (quantity == clicked_stack_ref.quantity) {
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
            let clicked = clicked_stack_ref.clone();
            *carried_stack = Some(clicked);
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
    // We can place, and the destination is empty
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
            *carried_stack = clicked_stack.clone();
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

    if let Err(_) = client_state.actions.try_send(GameAction::Inventory(action)) {
        log::info!("Sending action failed; server disconnected or lagging badly");
    }
}
