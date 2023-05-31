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


use std::time::{Instant};
use std::{sync::Arc};

use cgmath::{Deg, Zero};
use cuberef_core::coordinates::{BlockCoordinate, ChunkCoordinate, PlayerPositionUpdate};


use log::warn;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use tokio::sync::mpsc;
use winit::event::{DeviceEvent, ElementState, Event, KeyboardInput, MouseButton, WindowEvent};

use crate::cube_renderer::{BlockRenderer, ClientBlockTypeManager};
use crate::game_state::chunk::ClientChunk;
use crate::game_ui::GameUi;

use self::items::{ClientItemManager, InventoryManager};
use self::tool_controller::{ToolController, ToolState};

pub(crate) mod chunk;
pub(crate) mod items;
pub(crate) mod physics;
pub(crate) mod tool_controller;
mod utils;

#[derive(Debug, Clone, Copy)]
pub(crate) struct DigTapAction {
    pub(crate) target: BlockCoordinate,
    pub(crate) prev: Option<BlockCoordinate>,
    pub(crate) item_slot: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PlaceAction {
    pub(crate) target: BlockCoordinate,
    pub(crate) anchor: Option<BlockCoordinate>,
    pub(crate) item_slot: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum GameAction {
    Dig(DigTapAction),
    Tap(DigTapAction),
    Place(PlaceAction),
}

// todo clean up, make private
pub(crate) struct ClientState {
    pub(crate) block_types: Arc<ClientBlockTypeManager>,
    pub(crate) items: Arc<ClientItemManager>,
    pub(crate) last_update: parking_lot::Mutex<Instant>,
    pub(crate) physics_state: parking_lot::Mutex<physics::PhysicsState>,
    pub(crate) chunks: parking_lot::Mutex<FxHashMap<ChunkCoordinate, ClientChunk>>,
    pub(crate) inventories: parking_lot::Mutex<InventoryManager>,
    pub(crate) tool_controller: parking_lot::Mutex<ToolController>,
    pub(crate) shutdown: tokio_util::sync::CancellationToken,
    pub(crate) actions: mpsc::Sender<GameAction>,

    // cube_renderer doesn't currently need a mutex because it's stateless
    // and keeps its cached geometry data in the chunks themselves
    pub(crate) cube_renderer: Arc<BlockRenderer>,
    // GameUi manages its own state.
    pub(crate) game_ui: Arc<Mutex<GameUi>>,
}
impl ClientState {
    pub(crate) fn window_event(&self, event: &Event<()>) {
        match *event {
            Event::DeviceEvent {
                device_id: _,
                event: DeviceEvent::MouseMotion { delta: (x, y) },
            } => {
                self.physics_state.lock().mouse_delta(x, y);
            }
            Event::DeviceEvent {
                device_id: _,
                event: DeviceEvent::Button { button, state },
            } => {
                log::info!("btn: {button}, {state:?}");
            }
            Event::DeviceEvent {
                device_id: _,
                event:
                    DeviceEvent::Key(KeyboardInput {
                        scancode, state, ..
                    }),
            } => {
                self.physics_state.lock().keyboard_input(scancode, state);
            }
            Event::WindowEvent {
                window_id: _,
                event:
                    WindowEvent::MouseInput {
                        device_id: _,
                        state,
                        button,
                        ..
                    },
            } => {
                log::info!("Mouse: {state:?}, {button:?}");
                match (button, state) {
                    (MouseButton::Left, ElementState::Pressed) => {
                        self.tool_controller.lock().mouse_down()
                    }
                    (MouseButton::Left, ElementState::Released) => {
                        self.tool_controller.lock().mouse_up()
                    }
                    (MouseButton::Right, ElementState::Released) => {
                        self.tool_controller.lock().secondary_mouse_down()
                    }
                    _ => {}
                }
            }

            _ => {}
        }
        self.game_ui
            .lock()
            .window_event(event, self, &mut self.tool_controller.lock());
    }
    pub(crate) fn last_position(&self) -> PlayerPositionUpdate {
        let lock = self.physics_state.lock();
        PlayerPositionUpdate {
            // TODO tick, velocity
            tick: 0,
            position: lock.pos(),
            velocity: cgmath::Vector3::zero(),
            face_direction: lock.angle(),
        }
    }
    pub(crate) fn next_frame(&self, aspect_ratio: f64) -> FrameState {
        let delta = {
            let mut lock = self.last_update.lock();
            let now = Instant::now();
            let delta = now - *lock;
            *lock = now;
            delta
        };

        let (player_position, (az, el)) =
            self.physics_state
                .lock()
                .update_and_get(self, aspect_ratio, delta);

        let rotation = cgmath::Matrix4::from_angle_x(Deg(el))
            * cgmath::Matrix4::from_angle_y(Deg(az) + Deg(180.));
        let projection = cgmath::perspective(Deg(45.0), aspect_ratio, 0.01, 1000.);
        let view_proj_matrix = (projection * rotation)
            .cast()
            .unwrap();

        let mut tool_state = self.tool_controller.lock().update(self, delta);
        if let Some(action) = tool_state.action.take() {
            match self.actions.try_send(action) {
                Ok(_) => {}
                Err(e) => warn!("Failure sending action: {e:?}"),
            }
        }

        FrameState {
            view_proj_matrix,
            player_position,
            tool_state,
        }
    }

    pub(crate) fn should_capture(&self) -> bool {
        self.physics_state.lock().should_capture()
    }
}

pub(crate) struct FrameState {
    pub(crate) view_proj_matrix: cgmath::Matrix4<f32>,
    pub(crate) player_position: cgmath::Vector3<f64>,
    pub(crate) tool_state: ToolState,
}
