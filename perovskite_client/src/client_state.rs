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

use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use arc_swap::ArcSwap;
use cgmath::{Deg, Vector3, Zero};
use egui::Color32;
use enum_map::{Enum, EnumMap};
use perovskite_core::constants::block_groups::DEFAULT_SOLID;
use perovskite_core::constants::permissions;
use perovskite_core::coordinates::{BlockCoordinate, PlayerPositionUpdate};

use log::warn;
use parking_lot::Mutex;
use perovskite_core::game_actions::ToolTarget;
use perovskite_core::protocol;
use perovskite_core::protocol::game_rpc::{ServerPerformanceMetrics, SetClientState};
use perovskite_core::protocol::ui::ToolHint;
use perovskite_core::time::TimeState;
use seqlock::SeqLock;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracy_client::span;
use winit::event::{DeviceEvent, WindowEvent};

use crate::client_state::input::InputWrapper;
use crate::game_ui::egui_ui::EguiUi;
use crate::game_ui::hud::GameHud;
use crate::vulkan::block_renderer::BlockRenderer;
use crate::vulkan::entity_renderer::EntityRenderer;
use crate::vulkan::far_geometry::FarGeometryState;
use crate::vulkan::shaders::SceneState;
use crate::vulkan::{text_renderer, VulkanContext};

use self::block_types::ClientBlockTypeManager;
use self::chat::ChatState;
use self::entities::EntityState;
use self::input::BoundAction;
use self::items::{ClientItemManager, InventoryViewManager};
use self::lightcycle::LightCycle;
use self::settings::GameSettings;
use self::timekeeper::Timekeeper;
use self::tool_controller::{ToolController, ToolState};

pub(crate) mod block_types;
pub(crate) mod chat;
pub(crate) mod chunk;
pub(crate) mod entities;
pub(crate) mod input;
pub(crate) mod items;
pub(crate) mod lightcycle;
pub(crate) mod physics;
pub(crate) mod settings;
pub(crate) mod timekeeper;
pub(crate) mod tool_controller;

#[derive(Debug, Clone, Copy)]
pub(crate) struct DigTapAction {
    pub(crate) target: ToolTarget,
    pub(crate) prev: Option<BlockCoordinate>,
    pub(crate) item_slot: u32,
    pub(crate) player_pos: PlayerPositionUpdate,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PlaceAction {
    pub(crate) target: Option<BlockCoordinate>,
    pub(crate) anchor: ToolTarget,
    pub(crate) item_slot: u32,
    pub(crate) player_pos: PlayerPositionUpdate,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct InventoryAction {
    pub(crate) source_view: u64,
    pub(crate) source_slot: usize,
    pub(crate) destination_view: u64,
    pub(crate) destination_slot: usize,
    pub(crate) count: u32,
    pub(crate) swap: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct InteractKeyAction {
    pub(crate) target: ToolTarget,
    pub(crate) item_slot: u32,
    pub(crate) player_pos: PlayerPositionUpdate,
    pub(crate) menu_entry: String,
}

#[derive(Debug, Clone)]
pub(crate) enum GameAction {
    Dig(DigTapAction),
    Tap(DigTapAction),
    Place(PlaceAction),
    Inventory(InventoryAction),
    PopupResponse(protocol::ui::PopupResponse),
    InteractKey(InteractKeyAction),
    ChatMessage(String),
}

// todo clean up, make private
pub(crate) struct ClientState {
    pub(crate) settings: Arc<ArcSwap<GameSettings>>,
    pub(crate) input: InputWrapper,

    pub(crate) block_types: Arc<ClientBlockTypeManager>,
    pub(crate) items: Arc<ClientItemManager>,
    pub(crate) last_update: Mutex<u64>,
    pub(crate) physics_state: Mutex<physics::PhysicsState>,
    pub(crate) chunks: chunk::ChunkManager,
    pub(crate) inventories: Mutex<InventoryViewManager>,
    pub(crate) tool_controller: Mutex<ToolController>,
    pub(crate) shutdown: tokio_util::sync::CancellationToken,
    pub(crate) actions: mpsc::Sender<GameAction>,
    pub(crate) light_cycle: Arc<Mutex<LightCycle>>,
    pub(crate) chat: Arc<Mutex<ChatState>>,
    pub(crate) entities: Mutex<EntityState>,
    pub(crate) far_geometry: Mutex<FarGeometryState>,

    // block_renderer doesn't currently need a mutex because it's stateless
    // and keeps its cached geometry data in the chunks themselves
    pub(crate) block_renderer: Arc<BlockRenderer>,
    pub(crate) entity_renderer: Arc<EntityRenderer>,
    // GameHud manages its own state.
    pub(crate) hud: Arc<Mutex<GameHud>>,
    pub(crate) egui: Arc<Mutex<EguiUi>>,

    pub(crate) pending_error: Mutex<Option<anyhow::Error>>,
    pub(crate) wants_exit_from_game: Mutex<bool>,
    pub(crate) last_position_weak: SeqLock<PlayerPositionUpdate>,

    pub(crate) timekeeper: Arc<Timekeeper>,

    pub(crate) audio: Arc<audio::EngineHandle>,
    pub(crate) world_audio: tokio::sync::Mutex<audio::MapSoundState>,
    pub(crate) text_renderer: Arc<text_renderer::TextEngine>,

    pub(crate) tool_hints: Mutex<HashMap<String, ToolHint>>,

    pub(crate) server_perf: Mutex<Option<ServerPerformanceMetrics>>,
    pub(crate) client_perf: Mutex<Option<ClientPerformanceMetrics>>,
    pub(crate) want_server_perf: AtomicBool,
    pub(crate) want_new_client_perf: tokio::sync::Notify,
    pub(crate) render_distance: AtomicU32,
    pub(crate) far_geometry_enabled: AtomicBool,
    pub(crate) alt_diffuse_enabled: AtomicBool,
    pub(crate) settings_generation: AtomicU64,
}

const PROJ_NEAR: f64 = 0.05;
const PROJ_FAR: f64 = 10000.;

impl ClientState {
    pub(crate) fn new(
        settings: Arc<ArcSwap<GameSettings>>,
        shutdown: CancellationToken,
        block_types: Arc<ClientBlockTypeManager>,
        chunks: chunk::ChunkManager,
        items: Arc<ClientItemManager>,
        action_sender: mpsc::Sender<GameAction>,
        hud: GameHud,
        egui: EguiUi,
        block_renderer: BlockRenderer,
        entity_renderer: EntityRenderer,
        far_geometry: FarGeometryState,
        timekeeper: Arc<Timekeeper>,
        audio: Arc<audio::EngineHandle>,
    ) -> Result<ClientState> {
        let audio_clone = audio.clone();
        Ok(ClientState {
            settings: settings.clone(),
            input: InputWrapper::new(settings.clone()),
            block_types,
            items,
            last_update: Mutex::new(timekeeper.now()),
            physics_state: Mutex::new(physics::PhysicsState::new(settings.clone())),
            chunks,
            inventories: Mutex::new(InventoryViewManager::new()),
            tool_controller: Mutex::new(ToolController::new()),
            shutdown,
            actions: action_sender,
            light_cycle: Arc::new(Mutex::new(LightCycle::new(TimeState::new(
                Duration::from_secs(24 * 60),
                0.0,
            )))),
            chat: Arc::new(Mutex::new(ChatState::new())),
            entities: Mutex::new(EntityState::new()?),

            block_renderer: Arc::new(block_renderer),
            entity_renderer: Arc::new(entity_renderer),
            far_geometry: Mutex::new(far_geometry),
            hud: Arc::new(Mutex::new(hud)),
            egui: Arc::new(Mutex::new(egui)),
            pending_error: Mutex::new(None),
            wants_exit_from_game: Mutex::new(false),
            last_position_weak: SeqLock::new(PlayerPositionUpdate {
                position: Vector3::zero(),
                velocity: Vector3::zero(),
                face_direction: (0.0, 0.0),
            }),
            timekeeper,
            audio,
            world_audio: tokio::sync::Mutex::new(MapSoundState::new(audio_clone)),
            text_renderer: Arc::new(text_renderer::TextEngine::new()),

            tool_hints: Mutex::new(HashMap::new()),
            server_perf: Mutex::new(None),
            client_perf: Mutex::new(None),
            want_server_perf: AtomicBool::new(false),
            want_new_client_perf: tokio::sync::Notify::new(),
            render_distance: AtomicU32::new(settings.load().render.render_distance),
            far_geometry_enabled: AtomicBool::new(settings.load().render.enable_far_geometry),
            alt_diffuse_enabled: AtomicBool::new(false),
            settings_generation: AtomicU64::new(settings.load().generation),
        })
    }

    pub(crate) fn window_event(&self, event: &WindowEvent) {
        let mut input = self.input.lock();
        input.window_event(event);
    }

    pub(crate) fn device_event(&self, event: &DeviceEvent) {
        let mut input = self.input.lock();
        input.handle_device_event(event);
    }
    pub(crate) fn last_position(&self) -> PlayerPositionUpdate {
        let lock = self.physics_state.lock();
        PlayerPositionUpdate {
            // TODO tick, velocity
            position: lock.pos(),
            velocity: Vector3::zero(),
            face_direction: lock.angle(),
        }
    }

    /// Returns the player's last position without requiring the physics lock (which could cause a lock order violation)
    /// This may be a frame behind
    pub(crate) fn weakly_ordered_last_position(&self) -> PlayerPositionUpdate {
        self.last_position_weak.read()
    }

    pub(crate) fn next_frame(&self, aspect_ratio: f64, tick: u64) -> FrameState {
        let _span = span!("client_state next_frame");
        {
            let settings_generation = self.settings_generation.load(Ordering::Acquire);
            let settings = self.settings.load();
            if settings_generation != settings.generation {
                // These are only atomic for atomicity, not for synchronization.
                // We can freely interleave them and don't care about the exact order.
                self.settings_generation
                    .store(settings.generation, Ordering::Release);
                self.render_distance
                    .store(settings.render.render_distance, Ordering::Release);
                self.far_geometry_enabled
                    .store(settings.render.enable_far_geometry, Ordering::Release);
                drop(settings);
                log::info!("New settings generation {}", settings_generation);
            }
        }

        let egui_wants_events;
        {
            let _span = span!("inputs to egui");
            self.timekeeper.update_frame();

            let mut input = self.input.lock();
            egui_wants_events = self.egui.lock().wants_user_events();
            input.set_modal_active(egui_wants_events);
            if input.take_just_pressed(BoundAction::Inventory) {
                self.egui.lock().open_inventory();
            } else if input.take_just_pressed(BoundAction::Menu) {
                self.egui.lock().open_pause_menu();
            } else if input.take_just_pressed(BoundAction::Chat) {
                self.egui.lock().open_chat();
            } else if input.take_just_pressed(BoundAction::ChatSlash) {
                self.egui.lock().open_chat_slash();
            } else if input.take_just_pressed(BoundAction::DebugPanel) {
                self.egui.lock().toggle_debug();
            } else if input.take_just_pressed(BoundAction::PerfPanel) {
                self.egui.lock().toggle_perf();
            } else if input.take_just_pressed(BoundAction::ViewRangeUp) {
                let incr = |x| {
                    if x >= 10 {
                        u32::min(x + 5, 150)
                    } else {
                        x + 1
                    }
                };
                // We're the only thread updating it; other threads will read
                // https://github.com/rust-lang/rust/issues/135894 for easier mechanism
                let old_distance = self
                    .render_distance
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| Some(incr(x)))
                    .unwrap();
                let new_distance = incr(old_distance);
                self.egui.lock().push_status_bar(
                    Duration::from_secs(5),
                    format!("Asking server to send up to {new_distance} chunks view distance"),
                );
            } else if input.take_just_pressed(BoundAction::ViewRangeDown) {
                let decr = |x| {
                    if x > 10 {
                        u32::min(x - 5, 150)
                    } else {
                        u32::max(x - 1, 4)
                    }
                };
                // We're the only thread updating it; other threads will read
                // https://github.com/rust-lang/rust/issues/135894 for easier mechanism
                let old_distance = self
                    .render_distance
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| Some(decr(x)))
                    .unwrap();
                let new_distance = decr(old_distance);
                self.egui.lock().push_status_bar(
                    Duration::from_secs(5),
                    format!("Asking server to send up to {new_distance} chunks view distance"),
                );
            } else if input.take_just_pressed(BoundAction::ToggleFarGeometry) {
                let new_enabled = !self
                    .far_geometry_enabled
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| Some(!x))
                    .unwrap();
                self.egui.lock().push_status_bar(
                    Duration::from_secs(5),
                    format!(
                        "Far geometry {}",
                        if new_enabled { "enabled" } else { "disabled" }
                    ),
                );
            } else if input.take_just_pressed(BoundAction::ToggleAltDiffuse) {
                let new_enabled = !self
                    .alt_diffuse_enabled
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| Some(!x))
                    .unwrap();
                self.egui.lock().push_status_bar(
                    Duration::from_secs(5),
                    format!(
                        "Far geometry {}",
                        if new_enabled { "enabled" } else { "disabled" }
                    ),
                );
            }
        }

        let delta = {
            let mut lock = self.last_update.lock();
            let delta = tick - *lock;
            *lock = tick;
            delta
        };
        let mut entity_attachment = 0;
        let (mut player_position, player_velocity, (az, el), current_block) = self
            .physics_state
            .lock()
            .update_and_get(self, aspect_ratio, Duration::from_nanos(delta), tick);
        {
            let _span = span!("entity attachment");
            let entity_lock = self.entities.lock();
            if let Some(entity_target) = entity_lock.attached_to_entity {
                if let Some(entity) = entity_lock.entities.get(&entity_target.entity_id) {
                    entity_attachment = entity_target.entity_id;
                    // We may not have a position if it's a trailing entity beyond the backbuffer
                    if let Some(position) = entity.attach_position(
                        tick,
                        &self.entity_renderer,
                        entity_target.trailing_entity_index,
                    ) {
                        player_position = position;
                    }
                }
            }
        }
        let ppu = PlayerPositionUpdate {
            position: player_position,
            velocity: player_velocity,
            face_direction: (az, el),
        };
        self.audio.update_position(
            tick,
            player_position,
            player_velocity.cast().unwrap(),
            az * PI / 180.,
            entity_attachment,
        );

        let rotation = cgmath::Matrix4::from_angle_x(Deg(el))
            * cgmath::Matrix4::from_angle_y(Deg(180.) - Deg(az));
        // TODO figure out why this is needed
        let coordinate_correction = cgmath::Matrix4::from_nonuniform_scale(-1., 1., 1.);

        let projection = cgmath::perspective(
            Deg(self.settings.load().render.fov_degrees),
            aspect_ratio,
            PROJ_NEAR,
            PROJ_FAR,
        );
        let view_proj_matrix = (projection * coordinate_correction * rotation)
            .cast()
            .unwrap();

        let mut tool_state = {
            let _span = span!("tool controller update");
            self.tool_controller.lock().update(self, ppu, delta, tick)
        };
        if let Some(action) = tool_state.action.take() {
            match self.actions.try_send(action) {
                Ok(_) => {}
                Err(e) => warn!("Failure sending action: {e:?}"),
            }
        }

        *self.last_position_weak.lock_write() = ppu;

        // TODO: Either do something with sky (like give it to the sky shader), or remove it
        let (_sky, lighting, sun_direction) = self.light_cycle.lock().get_colors();

        self.want_new_client_perf.notify_waiters();
        FrameState {
            scene_state: SceneState {
                vp_matrix: view_proj_matrix,
                global_light_color: lighting.into(),
                sun_direction,
                player_pos_block: current_block.0,
                alt_diffuse_enabled: self.alt_diffuse_enabled.load(Ordering::Relaxed),
                player_position,
            },
            player_position,
            tool_state,
            ime_enabled: egui_wants_events,
        }
    }

    pub(crate) fn cancel_requested(&self) -> bool {
        self.shutdown.is_cancelled()
    }

    pub(crate) fn handle_server_update(&self, state_update: &SetClientState) -> Result<()> {
        let position = state_update
            .position
            .as_ref()
            .and_then(|x| x.position.clone());
        {
            let mut physics_lock = self.physics_state.lock();

            if let Some(pos_vector) = position {
                physics_lock.set_position(pos_vector.try_into()?);
            }
            physics_lock.update_permissions(&state_update.permission, self);
        }
        {
            let mut egui_lock = self.egui.lock();
            egui_lock.inventory_view = state_update.inventory_popup.clone();
            egui_lock.inventory_manipulation_view_id =
                Some(state_update.inventory_manipulation_view);
            egui_lock.set_allow_inventory_interaction(
                state_update
                    .permission
                    .iter()
                    .any(|p| p == permissions::INVENTORY),
            );
            egui_lock.set_allow_button_interaction(
                state_update
                    .permission
                    .iter()
                    .any(|p| p == permissions::TAP_INTERACT),
            );
        }
        {
            let mut tc_lock = self.tool_controller.lock();
            tc_lock.update_permissions(&state_update.permission);
        }
        {
            let mut hud_lock = self.hud.lock();
            hud_lock.hotbar_view_id = Some(state_update.hotbar_inventory_view);
            hud_lock.update_held_item(self);
            hud_lock.invalidate_hotbar();
        }

        {
            let mut time_lock = self.light_cycle.lock();
            time_lock
                .time_state_mut()
                .set_time(state_update.time_of_day)?;
            time_lock
                .time_state_mut()
                .set_day_length(Duration::from_secs_f64(state_update.day_length_sec));
        }

        {
            let mut entities_lock = self.entities.lock();
            entities_lock.attached_to_entity = state_update.attached_to_entity;
        }
        {
            *self.tool_hints.lock() = state_update.tool_hints.clone();
        }
        Ok(())
    }

    pub(crate) fn server_perf(&self) -> Option<ServerPerformanceMetrics> {
        self.server_perf.lock().clone()
    }

    pub(crate) fn client_perf(&self) -> Option<ClientPerformanceMetrics> {
        self.client_perf.lock().clone()
    }

    pub(crate) fn clone_vk_ctx(&self) -> Arc<VulkanContext> {
        self.block_renderer.clone_vk_ctx()
    }

    pub(crate) fn vk_ctx(&self) -> &VulkanContext {
        self.block_renderer.vk_ctx()
    }
}

#[derive(Enum, Debug, PartialEq, Eq)]
pub(crate) enum LampId {
    /// client send queue
    CSendQ,
    /// inbound worker currently busy
    InBusy,
}

#[derive(Clone)]
pub(crate) struct ClientPerformanceMetrics {
    pub(crate) nprop_queue_lens: Vec<u64>,
    pub(crate) mesh_queue_lens: Vec<u64>,
    pub(crate) timekeeper_raw: [i64; ClientPerformanceMetrics::TIMEKEEPER_CHART_LEN],
    pub(crate) timekeeper_smoothed: [i64; ClientPerformanceMetrics::TIMEKEEPER_CHART_LEN],
    /// timekeeper_raw and timekeeper_smooth should write to this index.
    pub(crate) timekeeper_breakpoint: usize,
    pub(crate) lamps: EnumMap<LampId, (Color32, Color32)>,
}
impl Default for ClientPerformanceMetrics {
    fn default() -> Self {
        Self {
            nprop_queue_lens: vec![],
            mesh_queue_lens: vec![],
            timekeeper_raw: [0; Self::TIMEKEEPER_CHART_LEN],
            timekeeper_smoothed: [0; Self::TIMEKEEPER_CHART_LEN],
            timekeeper_breakpoint: 0,
            lamps: EnumMap::from_fn(|_| Self::LAMP_COLOR_OFF),
        }
    }
}
impl ClientPerformanceMetrics {
    pub(crate) const TIMEKEEPER_CHART_LEN: usize = 160;
    pub(crate) const LAMP_COLOR_OFF: (Color32, Color32) = (Color32::WHITE, Color32::from_gray(10));
    pub(crate) const LAMP_COLOR_YELLOW: (Color32, Color32) = (Color32::BLACK, Color32::YELLOW);
    pub(crate) const LAMP_COLOR_RED: (Color32, Color32) = (Color32::WHITE, Color32::RED);
    pub(crate) const LAMP_COLOR_BLUE: (Color32, Color32) = (Color32::BLACK, Color32::LIGHT_BLUE);
    pub(crate) fn update_timekeeper(&mut self, timekeeper: &Timekeeper) {
        self.timekeeper_raw[self.timekeeper_breakpoint] = timekeeper.get_raw_offset();
        self.timekeeper_smoothed[self.timekeeper_breakpoint] = timekeeper.get_smoothed_offset();
        self.timekeeper_breakpoint = (self.timekeeper_breakpoint + 1) % Self::TIMEKEEPER_CHART_LEN;
    }
}

pub(crate) struct FrameState {
    pub(crate) scene_state: SceneState,
    pub(crate) player_position: Vector3<f64>,
    pub(crate) tool_state: ToolState,
    pub(crate) ime_enabled: bool,
}

use crate::audio;
use crate::audio::MapSoundState;
use perovskite_core::protocol::blocks::{self as blocks_proto, CubeVariantEffect};

pub(crate) fn make_fallback_blockdef() -> blocks_proto::BlockTypeDef {
    blocks_proto::BlockTypeDef {
        render_info: Some(blocks_proto::block_type_def::RenderInfo::Cube(
            blocks_proto::CubeRenderInfo {
                tex_left: None,
                tex_right: None,
                tex_top: None,
                tex_bottom: None,
                tex_front: None,
                tex_back: None,
                render_mode: blocks_proto::CubeRenderMode::SolidOpaque.into(),
                variant_effect: CubeVariantEffect::None.into(),
            },
        )),
        groups: vec![DEFAULT_SOLID.to_string()],
        ..Default::default()
    }
}
