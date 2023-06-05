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

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use anyhow::{Context, Error, Result};

use texture_packer::{importer::ImageImporter, Rect};
use winit::{
    dpi::PhysicalPosition,
    event::{DeviceEvent, ElementState, Event, KeyboardInput},
};

use crate::{
    cube_renderer::AsyncTextureLoader,
    game_state::{items::ClientItemManager, tool_controller::ToolController, ClientState},
    vulkan::{
        shaders::flat_texture::{self, FlatTextureDrawBuilder, FlatTextureDrawCall},
        Texture2DHolder, VulkanContext,
    },
};

pub(crate) struct GameUi {
    texture_coords: HashMap<String, Rect>,
    texture_atlas: Arc<Texture2DHolder>,
    item_defs: Arc<ClientItemManager>,
    last_size: (u32, u32),

    hotbar_slot: u32,

    crosshair_draw_call: Option<FlatTextureDrawCall>,
    hotbar_draw_call: Option<FlatTextureDrawCall>,
    pixel_scroll_pos: i32,

    fps_counter: fps_counter::FPSCounter
}
impl GameUi {
    pub(crate) async fn new<T>(
        item_defs: Arc<ClientItemManager>,
        texture_loader: T,
        ctx: &VulkanContext,
    ) -> Result<Self>
    where
        T: AsyncTextureLoader,
    {
        let (texture_atlas, texture_coords) =
            build_texture_atlas(&item_defs, texture_loader, ctx).await?;

        Ok(GameUi {
            texture_coords,
            texture_atlas,
            item_defs,
            last_size: (0, 0),
            hotbar_slot: 0,
            crosshair_draw_call: None,
            hotbar_draw_call: None,
            pixel_scroll_pos: 0,
            fps_counter: fps_counter::FPSCounter::new()
        })
    }

    pub(crate) fn hotbar_slot(&self) -> u32 {
        self.hotbar_slot
    }

    pub(crate) fn window_event(
        &mut self,
        event: &Event<()>,
        client_state: &ClientState,
        tool_controller: &mut ToolController,
    ) {
        match *event {
            Event::DeviceEvent {
                event:
                    DeviceEvent::Key(KeyboardInput {
                        scancode, state, ..
                    }),
                ..
            } => {
                if state == ElementState::Pressed && (2..=9).contains(&scancode) {
                    let slot = scancode - 2;
                    self.set_slot(slot, client_state, tool_controller);
                }
            }
            Event::DeviceEvent {
                event: DeviceEvent::MouseWheel { delta },
                ..
            } => {
                let slot_delta = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => {
                        self.pixel_scroll_pos = 0;
                        y.round() as i32
                    }
                    winit::event::MouseScrollDelta::PixelDelta(PhysicalPosition { y, .. }) => {
                        self.pixel_scroll_pos += y.round() as i32;
                        // todo make config
                        const SMOOTH_SCROLL_PIXELS_PER_SLOT: i32 = 100;
                        let slot_delta = self
                            .pixel_scroll_pos
                            .div_euclid(SMOOTH_SCROLL_PIXELS_PER_SLOT);
                        self.pixel_scroll_pos = self.pixel_scroll_pos.rem_euclid(SMOOTH_SCROLL_PIXELS_PER_SLOT);
                        slot_delta
                    }
                };
                if slot_delta != 0 {
                    let slots = client_state.inventories.lock().main_inv().dimensions.1 as i32;
                    // Feels more intutive with the sign flipped
                    let new_slot = (self.hotbar_slot as i32 - slot_delta).rem_euclid(slots);
                    self.set_slot(new_slot.try_into().unwrap(), client_state, tool_controller);
                }
            }
            _ => {}
        }
    }

    pub(crate) fn render(
        &mut self,
        ctx: &VulkanContext,
        client_state: &ClientState,
    ) -> Result<Vec<FlatTextureDrawCall>> {
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
        self.render_number((128, 0), fps, &mut fps_builder);
        outputs.push(fps_builder.build(ctx)?);

        Ok(outputs)
    }

    pub(crate) fn texture(&self) -> Arc<Texture2DHolder> {
        self.texture_atlas.clone()
    }

    fn recreate_crosshair(
        &self,
        ctx: &VulkanContext,
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

    // Numbers are right-aligned, with pos being the rightmost point
    fn render_number(
        &self,
        pos: (u32, u32),
        mut number: u32,
        builder: &mut FlatTextureDrawBuilder,
    ) {
        let digits_frame = self.texture_coords[DIGIT_ATLAS];
        let mut x = pos.0 - DIGIT_WIDTH;
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
                self.texture_atlas.dimensions(),
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

    fn recreate_hotbar(
        &self,
        ctx: &VulkanContext,
        window_size: (u32, u32),
        client_state: &ClientState,
    ) -> Result<Option<FlatTextureDrawCall>> {
        let mut builder = flat_texture::FlatTextureDrawBuilder::new();
        let unselected_frame = *self.texture_coords.get(FRAME_UNSELECTED).unwrap();
        let selected_frame = *self.texture_coords.get(FRAME_SELECTED).unwrap();

        let w = unselected_frame.w;
        let h = unselected_frame.h;

        let inv_lock = client_state.inventories.lock();
        let inv_key = &inv_lock.main_inv_key;
        if inv_key.is_empty() {
            return Ok(None);
        }

        let main_inv = inv_lock
            .inventories
            .get(inv_key)
            .with_context(|| "Couldn't find main player inventory")?;
        if main_inv.dimensions.0 == 0 {
            return Ok(None);
        }
        let hotbar_slots = main_inv.dimensions.1;
        let left_offset = 0.5 * (hotbar_slots as f64) * (w as f64);

        // Top left corner of the frames
        let frame0_corner = (
            (window_size.0 / 2).saturating_sub(left_offset as u32),
            window_size.1 - h,
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
                builder.rect(item_rect, tex_coord, self.texture().dimensions());

                let frame_topright = (frame0_corner.0 + offset + w - 2, frame0_corner.1 + 2);
                // todo handle items that have a wear bar
                self.render_number(frame_topright, stack.quantity, &mut builder);
            }

            let frame_rect = Rect::new(frame0_corner.0 + offset, frame0_corner.1, w, h);
            if i == self.hotbar_slot {
                builder.rect(frame_rect, selected_frame, self.texture().dimensions())
            } else {
                builder.rect(frame_rect, unselected_frame, self.texture().dimensions())
            }
        }

        Ok(Some(builder.build(ctx)?))
    }

    pub(crate) fn invalidate_hotbar(&mut self) {
        self.hotbar_draw_call = None;
    }

    fn get_texture(&self, item: &cuberef_core::protocol::items::ItemStack) -> Rect {
        let Some(item) = self.item_defs.get(&item.item_name) else {
            return *self.texture_coords.get(UNKNOWN_TEXTURE).unwrap();
        };
        let Some(tex_ref) = &item.inventory_texture else {
            return *self.texture_coords.get(UNKNOWN_TEXTURE).unwrap();
        };
        self.texture_coords
            .get(&tex_ref.texture_name)
            .copied()
            .unwrap_or(*self.texture_coords.get(UNKNOWN_TEXTURE).unwrap())
    }

    fn set_slot(
        &mut self,
        slot: u32,
        client_state: &ClientState,
        tool_controller: &mut ToolController,
    ) {
        self.hotbar_slot = slot;
        let stack = client_state.inventories.lock().main_inv().contents()[slot as usize].clone();
        let item = stack
            .and_then(|x| client_state.items.get(&x.item_name))
            .cloned();
        tool_controller.update_item(client_state, slot, item);
        self.hotbar_draw_call = None;
    }
}

async fn build_texture_atlas<T>(
    item_defs: &ClientItemManager,
    mut texture_loader: T,
    ctx: &VulkanContext,
) -> Result<(Arc<Texture2DHolder>, HashMap<String, Rect>)>
where
    T: AsyncTextureLoader,
{
    let all_texture_names = item_defs
        .all_item_defs()
        .flat_map(|x| &x.inventory_texture)
        .map(|x| x.texture_name.clone())
        .collect::<HashSet<_>>();

    let mut textures = HashMap::new();
    for name in all_texture_names {
        let texture = texture_loader.load_texture(&name).await?;
        textures.insert(name, texture);
    }

    let config = texture_packer::TexturePackerConfig {
        // todo tweak these or make into a setting
        allow_rotation: false,
        max_width: 1024,
        max_height: 1024,
        border_padding: 2,
        texture_padding: 2,
        texture_extrusion: 2,
        trim: false,
        texture_outlines: false,
    };
    let mut texture_packer = texture_packer::TexturePacker::new_skyline(config);

    texture_packer
        .pack_own(
            String::from(CROSSHAIR),
            ImageImporter::import_from_memory(include_bytes!("crosshair.png")).unwrap(),
        )
        .map_err(|x| Error::msg(format!("Texture pack failed: {:?}", x)))?;
    texture_packer
        .pack_own(
            String::from(UNKNOWN_TEXTURE),
            ImageImporter::import_from_memory(include_bytes!("../block_unknown.png")).unwrap(),
        )
        .map_err(|x| Error::msg(format!("Texture pack failed: {:?}", x)))?;
    texture_packer
        .pack_own(
            String::from(FRAME_SELECTED),
            ImageImporter::import_from_memory(include_bytes!("frame_selected.png")).unwrap(),
        )
        .map_err(|x| Error::msg(format!("Texture pack failed: {:?}", x)))?;
    texture_packer
        .pack_own(
            String::from(FRAME_UNSELECTED),
            ImageImporter::import_from_memory(include_bytes!("frame_unselected.png")).unwrap(),
        )
        .map_err(|x| Error::msg(format!("Texture pack failed: {:?}", x)))?;
    texture_packer
        .pack_own(
            String::from(DIGIT_ATLAS),
            ImageImporter::import_from_memory(include_bytes!("digit_atlas.png")).unwrap(),
        )
        .map_err(|x| Error::msg(format!("Texture pack failed: {:?}", x)))?;
    texture_packer
        .pack_own(
            String::from(TEST_ITEM),
            ImageImporter::import_from_memory(include_bytes!("testonly_pickaxe.png")).unwrap(),
        )
        .map_err(|x| Error::msg(format!("Texture pack failed: {:?}", x)))?;

    for (name, texture) in textures {
        texture_packer
            .pack_own(name, texture)
            .map_err(|x| Error::msg(format!("Texture pack failed: {:?}", x)))?;
    }
    let texture_atlas = texture_packer::exporter::ImageExporter::export(&texture_packer)
        .map_err(|x| Error::msg(format!("Texture atlas export failed: {:?}", x)))?;
    let texture_coords = texture_packer
        .get_frames()
        .iter()
        .map(|(k, v)| (k.clone(), v.frame))
        .collect();

    let texture_atlas = Arc::new(Texture2DHolder::create(ctx, &texture_atlas)?);
    Ok((texture_atlas, texture_coords))
}

const CROSSHAIR: &str = "builtin:crosshair";
const FRAME_SELECTED: &str = "builtin:frame_selected";
const FRAME_UNSELECTED: &str = "builtin:frame_unselected";
const TEST_ITEM: &str = "builtin:test_item";
const DIGIT_ATLAS: &str = "builtin:digit_atlas";
const DIGIT_WIDTH: u32 = 13;
const UNKNOWN_TEXTURE: &str = "builtin:unknown";
