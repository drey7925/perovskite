// Copyright 2023 drey7925
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Arc};

use anyhow::{Context, Result};
use perovskite_core::protocol::items::{item_stack::QuantityType, ItemStack};
use texture_packer::Rect;
use winit::dpi::PhysicalPosition;

use crate::{
    client_state::{
        input::BoundAction, items::ClientItemManager, tool_controller::ToolState, ClientState,
    },
    game_ui::{FRAME_SELECTED_ALT, FRAME_UNSELECTED_ALT},
    vulkan::{
        shaders::flat_texture::{FlatTextureDrawBuilder, FlatTextureDrawCall},
        Texture2DHolder, VulkanWindow,
    },
};

use super::{get_texture, CROSSHAIR, DIGIT_ATLAS, FRAME_SELECTED, FRAME_UNSELECTED};

pub(crate) struct GameHud {
    pub(crate) texture_coords: HashMap<String, Rect>,
    pub(crate) texture_atlas: Arc<Texture2DHolder>,
    pub(crate) item_defs: Arc<ClientItemManager>,
    pub(crate) last_size: (u32, u32),

    pub(crate) hotbar_slot: u32,
    pub(crate) hotbar_view_id: Option<u64>,

    pub(crate) crosshair_draw_call: Option<FlatTextureDrawCall>,
    pub(crate) hotbar_draw_call: Option<FlatTextureDrawCall>,
    pub(crate) full_inv_draw_call: Option<FlatTextureDrawCall>,

    pub(crate) fps_counter: fps_counter::FPSCounter,
}
impl GameHud {
    pub(crate) fn hotbar_slot(&self) -> u32 {
        self.hotbar_slot
    }

    pub(crate) fn update_and_render(
        &mut self,
        ctx: &VulkanWindow,
        client_state: &ClientState,
        tool_state: &ToolState,
    ) -> Result<Vec<FlatTextureDrawCall>> {
        let (slot_delta, slot_selection, full_popup, release_pos) = {
            let mut input_lock = client_state.input.lock();
            (
                input_lock.take_scroll_slots(),
                input_lock.take_hotbar_selection(),
                input_lock.is_pressed(BoundAction::QuickInventory),
                if input_lock.take_just_released(BoundAction::QuickInventory) {
                    Some(input_lock.last_cursor_pos())
                } else {
                    None
                },
            )
        };

        if let Some(total_slots) = self.hotbar_view_id.and_then(|x| {
            client_state
                .inventories
                .lock()
                .inventory_views
                .get(&x)
                .map(|x| x.dimensions.1)
        }) {
            if slot_delta != 0 {
                let new_slot = if self.hotbar_slot >= total_slots {
                    // special case: we'ee in the special slot
                    // Note sign error (here and below), slot increment is negative to make the scroll wheel
                    // intuitive
                    if slot_delta > 0 {
                        total_slots.saturating_sub(slot_delta as u32)
                    } else {
                        ((slot_delta.abs() - 1) as u32).min(total_slots - 1)
                    }
                } else {
                    (self.hotbar_slot as i32 - slot_delta).rem_euclid(total_slots.try_into()?)
                        as u32
                };
                self.set_slot(new_slot, client_state);
            }
            if let Some(x) = slot_selection {
                if x < total_slots {
                    self.set_slot(x, client_state);
                }
            }
        }

        let window_size = ctx.window_size();

        if let Some(pos) = release_pos {
            if let Some(slot) = self.find_full_slot(pos, window_size, client_state) {
                self.set_slot(slot, client_state);
            }
        }

        if self.crosshair_draw_call.is_none() || window_size != self.last_size {
            self.crosshair_draw_call = Some(self.recreate_crosshair(ctx, window_size)?);
        }

        if self.hotbar_draw_call.is_none() || window_size != self.last_size {
            if let Some((hotbar, full_inv)) =
                self.recreate_hotbar(ctx, window_size, client_state)?
            {
                self.hotbar_draw_call = Some(hotbar);
                self.full_inv_draw_call = Some(full_inv);
            }
        }

        self.last_size = window_size;

        let mut outputs = vec![];
        outputs.push(self.crosshair_draw_call.as_ref().unwrap().clone());

        if full_popup && release_pos.is_none() {
            if let Some(x) = self.full_inv_draw_call.as_ref() {
                outputs.push(x.clone());
            }
        } else {
            if let Some(x) = self.hotbar_draw_call.as_ref() {
                outputs.push(x.clone());
            }
        }

        let mut per_frame_builder = FlatTextureDrawBuilder::new();
        let fps = self.fps_counter.tick() as u32;
        render_number(
            (window_size.0, 0),
            fps,
            &mut per_frame_builder,
            &self.texture_coords,
            &self.texture_atlas,
        );

        if let Some(dig_progress) = tool_state.dig_progress {
            render_progress_bar(
                (window_size.0 / 2 - 32, window_size.1 / 2 + 16),
                64,
                dig_progress,
                &mut per_frame_builder,
                &self.texture_coords,
                &self.texture_atlas,
            );
        }

        outputs.push(per_frame_builder.build(ctx)?);

        Ok(outputs)
    }

    pub(crate) fn clone_atlas(&self) -> Arc<Texture2DHolder> {
        self.texture_atlas.clone()
    }

    fn recreate_crosshair(
        &self,
        ctx: &VulkanWindow,
        window_size: (u32, u32),
    ) -> Result<FlatTextureDrawCall> {
        let mut builder = FlatTextureDrawBuilder::new();
        builder.centered_rect(
            (window_size.0 / 2, window_size.1 / 2),
            *self.texture_coords.get(CROSSHAIR).unwrap(),
            self.texture_atlas.dimensions(),
            1,
        );
        builder.build(ctx)
    }

    fn find_full_slot(
        &self,
        pos: PhysicalPosition<f64>,
        window_size: (u32, u32),
        client_state: &ClientState,
    ) -> Option<u32> {
        let unselected_frame = *self.texture_coords.get(FRAME_UNSELECTED).unwrap();

        let w = unselected_frame.w;
        let h = unselected_frame.h;

        let dims = client_state
            .inventories
            .lock()
            .inventory_views
            .get(&self.hotbar_view_id?)?
            .dimensions;

        let left_offset = 0.5 * (dims.1 as f64) * (w as f64);
        let top_offset = 0.5 * (dims.0 as f64) * (h as f64);

        let full_inv_corner = (
            (window_size.0 / 2).saturating_sub(left_offset as u32),
            (window_size.1 / 2).saturating_sub(top_offset as u32),
        );

        for j in 0..dims.0 {
            for i in 0..dims.1 {
                let index = j * dims.1 + i;

                let offset_x = i * w as u32;
                let offset_y = j * h as u32;

                let x_match = pos.x >= (full_inv_corner.0 + offset_x) as f64
                    && pos.x < (full_inv_corner.0 + offset_x + w) as f64;
                let y_match = pos.y >= (full_inv_corner.1 + offset_y) as f64
                    && pos.y < (full_inv_corner.1 + offset_y + h) as f64;

                if x_match && y_match {
                    return Some(index);
                }
            }
        }

        None
    }

    fn recreate_hotbar(
        &self,
        ctx: &VulkanWindow,
        window_size: (u32, u32),
        client_state: &ClientState,
    ) -> Result<Option<(FlatTextureDrawCall, FlatTextureDrawCall)>> {
        let mut main_builder = FlatTextureDrawBuilder::new();
        let mut full_inv_builder = FlatTextureDrawBuilder::new();
        let unselected_frame = *self.texture_coords.get(FRAME_UNSELECTED).unwrap();
        let selected_frame = *self.texture_coords.get(FRAME_SELECTED).unwrap();
        let unselected_frame_alt = *self.texture_coords.get(FRAME_UNSELECTED_ALT).unwrap();

        let w = unselected_frame.w;
        let h = unselected_frame.h;

        let inv_lock = client_state.inventories.lock();
        let view_id = match self.hotbar_view_id {
            Some(x) => x,
            None => return Ok(None),
        };

        let main_inv = inv_lock
            .inventory_views
            .get(&view_id)
            .with_context(|| "Couldn't find main player inventory")?;
        if main_inv.dimensions.0 == 0 {
            return Ok(None);
        }
        if (main_inv.dimensions.0 as usize) > main_inv.contents().len() {
            log::warn!("Hotbar contents vec is too small for the hotbar dimension");
            return Ok(None);
        }
        let hotbar_slots = main_inv.dimensions.1;
        let left_offset = 0.5 * (hotbar_slots as f64) * (w as f64);

        // Top left corner of the frames
        let frame0_corner = (
            (window_size.0 / 2).saturating_sub(left_offset as u32),
            window_size.1.saturating_sub(h),
        );

        let top_offset = 0.5 * (main_inv.dimensions.0 as f64) * (h as f64);

        let full_inv_corner = (
            (window_size.0 / 2).saturating_sub(left_offset as u32),
            (window_size.1 / 2).saturating_sub(top_offset as u32),
        );

        for i in 0..hotbar_slots {
            let frame = if i == self.hotbar_slot {
                selected_frame
            } else {
                unselected_frame
            };

            self.render_inventory_tile(
                &mut main_builder,
                w,
                h,
                main_inv
                    .contents()
                    .get(i as usize)
                    .unwrap_or(&None)
                    .as_ref(),
                frame0_corner,
                i,
                0,
                frame,
            );
        }

        // specially selected item, right next to the hotbar
        if self.hotbar_slot >= hotbar_slots {
            self.render_inventory_tile(
                &mut main_builder,
                w,
                h,
                main_inv
                    .contents()
                    .get(self.hotbar_slot as usize)
                    .unwrap_or(&None)
                    .as_ref(),
                frame0_corner,
                hotbar_slots,
                0,
                *self.texture_coords.get(FRAME_SELECTED_ALT).unwrap(),
            );
        }

        for j in 0..main_inv.dimensions.0 {
            for i in 0..main_inv.dimensions.1 {
                let index = j * main_inv.dimensions.1 + i;
                self.render_inventory_tile(
                    &mut full_inv_builder,
                    w,
                    h,
                    main_inv
                        .contents()
                        .get(index as usize)
                        .unwrap_or(&None)
                        .as_ref(),
                    full_inv_corner,
                    i,
                    j,
                    unselected_frame_alt,
                );
            }
        }

        Ok(Some((
            main_builder.build(ctx)?,
            full_inv_builder.build(ctx)?,
        )))
    }

    fn render_inventory_tile(
        &self,
        builder: &mut FlatTextureDrawBuilder,
        tile_w: u32,
        tile_h: u32,
        stack: Option<&ItemStack>,
        frame0_corner: (u32, u32),
        i: u32,
        j: u32,
        frame: Rect,
    ) {
        let offset_x = i * tile_w;
        let offset_y = j * tile_h;
        let item_rect = Rect::new(
            frame0_corner.0 + 2 + offset_x,
            frame0_corner.1 + 2 + offset_y,
            tile_w - 4,
            tile_h - 4,
        );

        if let Some(stack) = stack {
            let tex_coord = self.get_texture(stack);
            builder.rect(item_rect, tex_coord, self.clone_atlas().dimensions());

            let frame_topright = (
                frame0_corner.0 + offset_x + tile_w - 2,
                frame0_corner.1 + 2 + offset_y,
            );
            let frame_bottomleft = (
                frame0_corner.0 + offset_x + 2,
                frame0_corner.1 + tile_h - 8 + offset_y,
            );
            // todo handle items that have a wear bar
            match stack.quantity_type {
                Some(QuantityType::Stack(_)) => {
                    if stack.quantity != 1 {
                        render_number(
                            frame_topright,
                            stack.quantity,
                            builder,
                            &self.texture_coords,
                            &self.texture_atlas,
                        );
                    }
                }
                Some(QuantityType::Wear(total_wear)) => render_wear_bar(
                    frame_bottomleft,
                    tile_w - 4,
                    stack.current_wear,
                    total_wear,
                    builder,
                    &self.texture_coords,
                    &self.texture_atlas,
                ),
                None => {}
            }
        }
        let frame_rect = Rect::new(
            frame0_corner.0 + offset_x,
            frame0_corner.1 + offset_y,
            tile_w,
            tile_h,
        );
        builder.rect(frame_rect, frame, self.texture_atlas().dimensions());
    }

    pub(crate) fn invalidate_hotbar(&mut self) {
        self.hotbar_draw_call = None;
    }

    pub(crate) fn get_texture(&self, item: &perovskite_core::protocol::items::ItemStack) -> Rect {
        get_texture(item, &self.texture_coords, &self.item_defs)
    }

    fn set_slot(&mut self, slot: u32, client_state: &ClientState) {
        self.hotbar_slot = slot;
        let stack = self.hotbar_view_id.and_then(|x| {
            client_state
                .inventories
                .lock()
                .inventory_views
                .get(&x)
                .and_then(|x| x.contents()[slot as usize].clone())
        });
        let item = stack
            .and_then(|x| client_state.items.get(&x.item_name))
            .cloned();
        client_state
            .tool_controller
            .lock()
            .change_held_item(client_state, slot, item);
        self.hotbar_draw_call = None;
    }

    pub(crate) fn update_held_item(&mut self, client_state: &ClientState) {
        let stack = self.hotbar_view_id.and_then(|x| {
            client_state
                .inventories
                .lock()
                .inventory_views
                .get(&x)
                .and_then(|x| x.contents()[self.hotbar_slot as usize].clone())
        });
        let item = stack
            .and_then(|x| client_state.items.get(&x.item_name))
            .cloned();
        client_state
            .tool_controller
            .lock()
            .change_held_item(client_state, self.hotbar_slot, item);
        self.hotbar_draw_call = None;
    }

    pub(crate) fn texture_atlas(&self) -> &Texture2DHolder {
        self.texture_atlas.as_ref()
    }
}

fn render_wear_bar(
    frame_bottomleft: (u32, u32),
    total_width: u32,
    current_wear: u32,
    max_wear: u32,
    builder: &mut FlatTextureDrawBuilder,
    texture_coords: &HashMap<String, Rect>,
    texture_atlas: &Texture2DHolder,
) {
    let wear_level = ((current_wear as f32) / (max_wear as f32)).clamp(0.0, 1.0);
    let draw_width = (wear_level * total_width as f32) as u32;

    let wear_bucket = ((wear_level * 8.0) as u8).clamp(0, 7);
    let wear_texture = format!("builtin:wear_{}", wear_bucket);
    let wear_uv = texture_coords.get(&wear_texture).copied().unwrap();

    builder.rect(
        Rect::new(frame_bottomleft.0, frame_bottomleft.1, draw_width, 6),
        wear_uv,
        texture_atlas.dimensions(),
    );
}

fn render_progress_bar(
    frame_bottomleft: (u32, u32),
    total_width: u32,
    progress: f64,
    builder: &mut FlatTextureDrawBuilder,
    texture_coords: &HashMap<String, Rect>,
    texture_atlas: &Texture2DHolder,
) {
    let progress_bg = texture_coords["builtin:progress_bg"];
    let progress_fg = texture_coords["builtin:progress_fg"];
    builder.rect(
        Rect::new(frame_bottomleft.0, frame_bottomleft.1, total_width, 6),
        progress_bg,
        texture_atlas.dimensions(),
    );
    let active_width = (progress * total_width as f64) as u32;
    if active_width > 0 {
        builder.rect(
            Rect::new(frame_bottomleft.0, frame_bottomleft.1, active_width, 6),
            progress_fg,
            texture_atlas.dimensions(),
        );
    }
}

// Numbers are right-aligned, with pos being the rightmost point
pub(crate) fn render_number(
    pos: (u32, u32),
    mut number: u32,
    builder: &mut FlatTextureDrawBuilder,
    atlas_coords: &HashMap<String, Rect>,
    atlas: &Texture2DHolder,
) {
    let digits_frame = atlas_coords[DIGIT_ATLAS];
    let mut x = pos.0.saturating_sub(DIGIT_WIDTH);
    loop {
        let digit = number % 10;
        builder.rect(
            Rect::new(x, pos.1, DIGIT_WIDTH, digits_frame.h),
            Rect::new(
                digits_frame.x + digit * DIGIT_WIDTH,
                digits_frame.y,
                DIGIT_WIDTH,
                digits_frame.h,
            ),
            atlas.dimensions(),
        );
        if x < DIGIT_WIDTH {
            break;
        }
        x -= DIGIT_WIDTH - 1;
        number /= 10;
        if number == 0 {
            return;
        }
    }
}

const DIGIT_WIDTH: u32 = 13;
