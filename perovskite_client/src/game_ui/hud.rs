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
use perovskite_core::protocol::items::item_stack::QuantityType;
use texture_packer::Rect;

use crate::{
    game_state::{items::ClientItemManager, ClientState},
    vulkan::{
        shaders::flat_texture::{self, FlatTextureDrawBuilder, FlatTextureDrawCall},
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
    ) -> Result<Vec<FlatTextureDrawCall>> {
        let slot_delta = client_state.input.lock().take_scroll_slots();
        if slot_delta != 0 {
            if let Some(slots) = self.hotbar_view_id.and_then(|x| {
                client_state
                    .inventories
                    .lock()
                    .inventory_views
                    .get(&x)
                    .map(|x| x.dimensions.1)
            }) {
                let new_slot =
                    (self.hotbar_slot as i32 - slot_delta).rem_euclid(slots.try_into().unwrap());
                self.set_slot(new_slot.try_into().unwrap(), client_state);
            }
        }

        let window_size = ctx.window_size();
        if self.crosshair_draw_call.is_none() || window_size != self.last_size {
            self.crosshair_draw_call = Some(self.recreate_crosshair(ctx, window_size)?);
        }

        if self.hotbar_draw_call.is_none() || window_size != self.last_size {
            self.hotbar_draw_call = self.recreate_hotbar(ctx, window_size, client_state)?;
        }

        self.last_size = window_size;

        let mut outputs = vec![];
        outputs.push(self.crosshair_draw_call.as_ref().unwrap().clone());
        if let Some(x) = self.hotbar_draw_call.as_ref() {
            outputs.push(x.clone());
        }

        let mut fps_builder = FlatTextureDrawBuilder::new();
        let fps = self.fps_counter.tick() as u32;
        render_number(
            (window_size.0, 0),
            fps,
            &mut fps_builder,
            &self.texture_coords,
            &self.texture_atlas,
        );
        outputs.push(fps_builder.build(ctx)?);

        let mut net_err_builder = FlatTextureDrawBuilder::new();
        let net_err = client_state.timekeeper.get_offset();
        render_number(
            (window_size.0, 16),
            net_err as u32,
            &mut net_err_builder,
            &self.texture_coords,
            &self.texture_atlas,
        );
        outputs.push(net_err_builder.build(ctx)?);

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
        let mut builder = flat_texture::FlatTextureDrawBuilder::new();
        builder.centered_rect(
            (window_size.0 / 2, window_size.1 / 2),
            *self.texture_coords.get(CROSSHAIR).unwrap(),
            self.texture_atlas.dimensions(),
            1,
        );
        builder.build(ctx)
    }

    fn recreate_hotbar(
        &self,
        ctx: &VulkanWindow,
        window_size: (u32, u32),
        client_state: &ClientState,
    ) -> Result<Option<FlatTextureDrawCall>> {
        let mut builder = flat_texture::FlatTextureDrawBuilder::new();
        let unselected_frame = *self.texture_coords.get(FRAME_UNSELECTED).unwrap();
        let selected_frame = *self.texture_coords.get(FRAME_SELECTED).unwrap();

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

        for i in 0..hotbar_slots {
            let offset = i * w;
            let item_rect = Rect::new(
                frame0_corner.0 + 2 + offset,
                frame0_corner.1 + 2,
                w - 4,
                h - 4,
            );
            let stack = &main_inv.contents()[i as usize];
            if let Some(stack) = stack {
                let tex_coord = self.get_texture(stack);
                builder.rect(item_rect, tex_coord, self.clone_atlas().dimensions());

                let frame_topright = (frame0_corner.0 + offset + w - 2, frame0_corner.1 + 2);
                let frame_bottomleft = (frame0_corner.0 + offset + 2, frame0_corner.1 + h - 8);
                // todo handle items that have a wear bar
                match stack.quantity_type {
                    Some(QuantityType::Stack(_)) => {
                        if stack.quantity != 1 {
                            render_number(
                                frame_topright,
                                stack.quantity,
                                &mut builder,
                                &self.texture_coords,
                                &self.texture_atlas,
                            );
                        }
                    }
                    Some(QuantityType::Wear(total_wear)) => render_wear_bar(
                        frame_bottomleft,
                        w - 4,
                        stack.current_wear,
                        total_wear,
                        &mut builder,
                        &self.texture_coords,
                        &self.texture_atlas,
                    ),
                    None => {}
                }
            }

            let frame_rect = Rect::new(frame0_corner.0 + offset, frame0_corner.1, w, h);
            if i == self.hotbar_slot {
                builder.rect(
                    frame_rect,
                    selected_frame,
                    self.texture_atlas().dimensions(),
                )
            } else {
                builder.rect(
                    frame_rect,
                    unselected_frame,
                    self.texture_atlas().dimensions(),
                )
            }
        }

        Ok(Some(builder.build(ctx)?))
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
            .update_item(client_state, slot, item);
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
