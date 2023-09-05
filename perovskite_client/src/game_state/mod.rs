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

use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use arc_swap::ArcSwap;
use cgmath::{Deg, Zero};
use perovskite_core::constants::block_groups::DEFAULT_SOLID;
use perovskite_core::coordinates::{BlockCoordinate, ChunkCoordinate, PlayerPositionUpdate};

use perovskite_core::block_id::BlockId;
use perovskite_core::lighting::{ChunkColumn, Lightfield};
use perovskite_core::protocol;
use perovskite_core::protocol::game_rpc::{MapDeltaUpdate, MapDeltaUpdateBatch};
use log::warn;
use parking_lot::{Mutex, RwLockReadGuard};
use rustc_hash::{FxHashMap, FxHashSet};
use tokio::sync::mpsc;
use tracy_client::span;
use winit::event::Event;

use crate::block_renderer::{fallback_texture, BlockRenderer, ClientBlockTypeManager};
use crate::game_state::chunk::ClientChunk;
use crate::game_ui::egui_ui::EguiUi;
use crate::game_ui::hud::GameHud;

use self::chat::ChatState;
use self::chunk::{ChunkDataView, SnappyDecodeHelper};
use self::input::{BoundAction, InputState};
use self::items::{ClientItemManager, InventoryViewManager};
use self::settings::GameSettings;
use self::tool_controller::{ToolController, ToolState};

pub(crate) mod chunk;
pub(crate) mod input;
pub(crate) mod items;
pub(crate) mod physics;
pub(crate) mod settings;
pub(crate) mod tool_controller;
pub(crate) mod chat;

#[derive(Debug, Clone, Copy)]
pub(crate) struct DigTapAction {
    pub(crate) target: BlockCoordinate,
    pub(crate) prev: Option<BlockCoordinate>,
    pub(crate) item_slot: u32,
    pub(crate) player_pos: PlayerPositionUpdate,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PlaceAction {
    pub(crate) target: BlockCoordinate,
    pub(crate) anchor: Option<BlockCoordinate>,
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

#[derive(Debug, Clone, Copy)]
pub(crate) struct InteractKeyAction {
    pub(crate) target: BlockCoordinate,
    #[allow(unused)]
    pub(crate) item_slot: u32,
    pub(crate) player_pos: PlayerPositionUpdate,
}

#[derive(Debug, Clone)]
pub(crate) enum GameAction {
    Dig(DigTapAction),
    Tap(DigTapAction),
    Place(PlaceAction),
    Inventory(InventoryAction),
    PopupResponse(perovskite_core::protocol::ui::PopupResponse),
    InteractKey(InteractKeyAction),
    ChatMessage(String),
}

pub(crate) type ChunkMap = FxHashMap<ChunkCoordinate, Arc<ClientChunk>>;
pub(crate) type LightColumnMap = FxHashMap<(i32, i32), ChunkColumn>;
pub(crate) struct ChunkManager {
    chunks: parking_lot::RwLock<ChunkMap>,
    renderable_chunks: parking_lot::RwLock<ChunkMap>,
    light_columns: parking_lot::RwLock<LightColumnMap>,
}
impl ChunkManager {
    pub(crate) fn new() -> ChunkManager {
        ChunkManager {
            chunks: parking_lot::RwLock::new(FxHashMap::default()),
            renderable_chunks: parking_lot::RwLock::new(FxHashMap::default()),
            light_columns: parking_lot::RwLock::new(FxHashMap::default()),
        }
    }
    /// Locks the chunk manager and returns a struct that can be used to access chunks in a read/write manner,
    /// but CANNOT be used to insert or remove chunks.
    pub(crate) fn read_lock(&self) -> ChunkManagerView {
        let _span = span!("Acquire chunk read lock");
        let guard = self.chunks.read();
        ChunkManagerView { guard }
    }

    /// Clones the chunk manager and returns a clone with the following properties:
    /// * The clone does not hold any locks on the data in this chunk manager (i.e. insertions and deletions)
    ///   are possible while the cloned view is live.
    /// * The clone does not track any insertions/deletions of this chunk manager.
    ///    * it will not show chunks inserted after cloned_view returned
    ///    * if chunks are deleted after cloned_view returned, they will remain in the cloned view, and their
    ///      memory will not be released until the cloned view is dropped, due to Arcs owned by the cloned view.
    pub(crate) fn renderable_chunks_cloned_view(&self) -> ChunkManagerClonedView {
        ChunkManagerClonedView {
            data: self.renderable_chunks.read().clone(),
        }
    }

    pub(crate) fn remove(&self, coord: &ChunkCoordinate) -> Option<Arc<ClientChunk>> {
        // Lock order: chunks -> renderable_chunks -> light_columns
        let mut chunks_lock = {
            let _span = span!("Acquire global chunk lock");
            self.chunks.write()
        };
        let mut render_chunks_lock = {
            let _span = span!("Acquire global renderable chunk lock");
            self.renderable_chunks.write()
        };
        let mut light_columns_lock = {
            let _span = span!("Acquire global light column lock");
            self.light_columns.write()
        };
        // The column should be present, so we unwrap.
        let column = light_columns_lock.get_mut(&(coord.x, coord.z)).unwrap();
        column.remove(coord.y);
        if column.is_empty() {
            light_columns_lock.remove(&(coord.x, coord.z));
        }
        render_chunks_lock.remove(coord);
        chunks_lock.remove(coord)
    }
    // Returns the number of additional chunks below the given chunk that need lighting updates.
    pub(crate) fn insert_or_update(
        &self,
        coord: ChunkCoordinate,
        proto: protocol::game_rpc::MapChunk,
        snappy_helper: &mut SnappyDecodeHelper,
        block_types: &ClientBlockTypeManager,
    ) -> anyhow::Result<usize> {
        // Lock order: chunks -> [renderable_chunks] -> light_columns
        let mut chunks_lock = {
            let _span = span!("Acquire global chunk lock");
            self.chunks.write()
        };
        let mut light_columns_lock = {
            let _span = span!("Acquire global light column lock");
            self.light_columns.write()
        };
        let light_column = light_columns_lock
            .entry((coord.x, coord.z))
            .or_insert_with(ChunkColumn::empty);
        let occlusion = match chunks_lock.entry(coord) {
            std::collections::hash_map::Entry::Occupied(chunk_entry) => chunk_entry
                .get()
                .update_from(proto, snappy_helper, block_types)?,
            std::collections::hash_map::Entry::Vacant(x) => {
                let (chunk, occlusion) =
                    ClientChunk::from_proto(proto, snappy_helper, block_types)?;
                light_column.insert_empty(coord.y);
                x.insert(Arc::new(chunk));
                occlusion
            }
        };

        let mut light_cursor = light_column.cursor_into(coord.y);
        *light_cursor.current_occlusion_mut() = occlusion;
        light_cursor.mark_valid();
        let extra_chunks = light_cursor.propagate_lighting();
        Ok(extra_chunks)
    }

    pub(crate) fn apply_delta_batch(
        &self,
        batch: &MapDeltaUpdateBatch,
        block_types: &ClientBlockTypeManager,
    ) -> Result<(FxHashSet<ChunkCoordinate>, Vec<BlockCoordinate>, bool)> {
        let mut needs_remesh = FxHashSet::default();
        let mut unknown_coords = vec![];
        let mut missing_coord = false;

        let chunk_lock = self.chunks.read();
        let light_lock = self.light_columns.read();
        for update in batch.updates.iter() {
            let block_coord: BlockCoordinate = match &update.block_coord {
                Some(x) => x.into(),
                None => {
                    log::warn!("Got delta with missing block_coord {:?}", update);
                    missing_coord = true;
                    continue;
                }
            };
            let chunk_coord = block_coord.chunk();
            let extra_chunks = match chunk_lock.get(&chunk_coord) {
                Some(x) => {
                    if !x.apply_delta(update).unwrap() {
                        None
                    } else {
                        let occlusion = x.get_occlusion(block_types);
                        let light_column = light_lock.get(&(chunk_coord.x, chunk_coord.z)).unwrap();
                        let mut light_cursor = light_column.cursor_into(chunk_coord.y);
                        *light_cursor.current_occlusion_mut() = occlusion;
                        light_cursor.mark_valid();
                        let extra_chunks = light_cursor.propagate_lighting();
                        Some(extra_chunks as i32)
                    }
                }
                None => {
                    log::warn!("Got delta for unknown chunk {:?}", block_coord);
                    unknown_coords.push(block_coord);
                    None
                }
            };
            // unwrap because we expect all errors to be internal.
            if let Some(extra_chunks) = extra_chunks {
                let chunk = block_coord.chunk();
                needs_remesh.insert(chunk);
                for i in -1..=1 {
                    for j in (-1 - extra_chunks as i32)..=1 {
                        for k in -1..=1 {
                            if let Some(neighbor) = chunk.try_delta(i, j, k) {
                                needs_remesh.insert(neighbor);
                            }
                        }
                    }
                }
            }
        }
        Ok((needs_remesh, unknown_coords, missing_coord))
    }

    pub(crate) fn cloned_neighbors_fast(&self, chunk: ChunkCoordinate) -> FastChunkNeighbors {
        let chunk_lock = self.chunks.read();
        let lights_lock = self.light_columns.read();
        let mut data =
            std::array::from_fn(|_| std::array::from_fn(|_| std::array::from_fn(|_| None)));
        let mut inbound_lights = [[[Lightfield::zero(); 3]; 3]; 3];
        for i in -1..=1 {
            for k in -1..=1 {
                let light_column = chunk
                    .try_delta(i, 0, k)
                    .and_then(|delta| lights_lock.get(&(delta.x, delta.z)));
                for j in -1..=1 {
                    if let Some(delta) = chunk.try_delta(i, j, k) {
                        data[(k + 1) as usize][(j + 1) as usize][(i + 1) as usize] =
                            chunk_lock.get(&delta).map(|x| Box::new(x.chunk_data().block_ids().clone()));
                        if let Some(light_column) = light_column {
                            inbound_lights[(k + 1) as usize][(j + 1) as usize][(i + 1) as usize] =
                                light_column
                                    .get_incoming_light(delta.y)
                                    .unwrap_or_else(|| Lightfield::zero());
                        }
                    }
                }
            }
        }
        FastChunkNeighbors {
            neighbors: data,
            center: chunk_lock.get(&chunk).cloned(),
            inbound_lights,
        }
    }

    /// If the chunk is present, meshes it. If any geometry was generated, inserts it into
    /// the renderable chunks. If no geometry was generated, removes it from the renderable chunks.
    /// Returns true if the chunk was present in the *main* list of chunks, false otherwise.
    pub(crate) fn maybe_mesh_and_maybe_promote(
        &self,
        coord: ChunkCoordinate,
        renderer: &BlockRenderer,
    ) -> Result<bool> {
        // Lock order notes: self.chunks -> self.renderable_chunks
        // Neighbor propagation and mesh generation only need self.chunks
        // Rendering only needs self.renderable_chunks, and that lock is limited in scope
        // to Self::cloned_view_renderable. The render thread should never end up needing
        // both chunks and renderable_chunks locked at the same time (and the render thread's
        // use of chunks is likewise scoped to the physics code)
        let src = self.chunks.read();
        if let Some(chunk) = src.get(&coord) {
            if chunk.mesh_with(renderer)? {
                // We take the lock after mesh_with to keep the lock scope as short as possible
                // and avoid blocking the render thread
                let mut dst = self.renderable_chunks.write();
                dst.insert(coord, chunk.clone());
            } else {
                // Likewise.
                let mut dst = self.renderable_chunks.write();
                dst.remove(&coord);
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }
}
pub(crate) struct ChunkManagerView<'a> {
    guard: RwLockReadGuard<'a, ChunkMap>,
}
impl<'a> ChunkManagerView<'a> {
    pub(crate) fn get(&'a self, coord: &ChunkCoordinate) -> Option<&'a Arc<ClientChunk>> {
        self.guard.get(coord)
    }
    pub(crate) fn contains_key(&self, coord: &ChunkCoordinate) -> bool {
        self.guard.contains_key(coord)
    }
}

pub(crate) struct FastChunkNeighbors {
    center: Option<Arc<ClientChunk>>,
    neighbors: [[[Option<Box<[BlockId; 18 * 18 * 18]>>; 3]; 3]; 3],
    inbound_lights: [[[Lightfield; 3]; 3]; 3],
}
impl FastChunkNeighbors {
    pub(crate) fn get(&self, coord_xyz: (i32, i32, i32)) -> Option<&[BlockId; 18 * 18 * 18]> {
        assert!((-1..=1).contains(&coord_xyz.0));
        assert!((-1..=1).contains(&coord_xyz.1));
        assert!((-1..=1).contains(&coord_xyz.2));
        self.neighbors[(coord_xyz.2 + 1) as usize][(coord_xyz.1 + 1) as usize]
            [(coord_xyz.0 + 1) as usize]
            .as_deref()
    }
    pub(crate) fn center(&self) -> Option<&ClientChunk> {
        self.center.as_deref()
    }
    pub(crate) fn inbound_light(&self, coord_xyz: (i32, i32, i32)) -> Lightfield {
        assert!((-1..=1).contains(&coord_xyz.0));
        assert!((-1..=1).contains(&coord_xyz.1));
        assert!((-1..=1).contains(&coord_xyz.2));
        self.inbound_lights[(coord_xyz.2 + 1) as usize][(coord_xyz.1 + 1) as usize]
            [(coord_xyz.0 + 1) as usize]
    }
}

pub(crate) struct ChunkManagerClonedView {
    data: ChunkMap,
}
impl Deref for ChunkManagerClonedView {
    type Target = ChunkMap;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

// todo clean up, make private
pub(crate) struct ClientState {
    pub(crate) settings: Arc<ArcSwap<GameSettings>>,
    pub(crate) input: Mutex<InputState>,

    pub(crate) block_types: Arc<ClientBlockTypeManager>,
    pub(crate) items: Arc<ClientItemManager>,
    pub(crate) last_update: parking_lot::Mutex<Instant>,
    pub(crate) physics_state: parking_lot::Mutex<physics::PhysicsState>,
    pub(crate) chunks: ChunkManager,
    pub(crate) inventories: parking_lot::Mutex<InventoryViewManager>,
    pub(crate) tool_controller: parking_lot::Mutex<ToolController>,
    pub(crate) shutdown: tokio_util::sync::CancellationToken,
    pub(crate) actions: mpsc::Sender<GameAction>,

    pub(crate) chat: Arc<Mutex<ChatState>>,

    // block_renderer doesn't currently need a mutex because it's stateless
    // and keeps its cached geometry data in the chunks themselves
    pub(crate) block_renderer: Arc<BlockRenderer>,
    // GameHud manages its own state.
    pub(crate) hud: Arc<Mutex<GameHud>>,
    pub(crate) egui: Arc<Mutex<EguiUi>>,

    pub(crate) pending_error: Mutex<Option<String>>,
    pub(crate) wants_exit_from_game: Mutex<bool>,
    // This is a leaf mutex - consider using some sort of atomic instead
    pub(crate) last_position_weak: Mutex<PlayerPositionUpdate>,
}
impl ClientState {
    pub(crate) fn new(
        settings: Arc<ArcSwap<GameSettings>>,
        block_types: Arc<ClientBlockTypeManager>,
        items: Arc<ClientItemManager>,
        action_sender: mpsc::Sender<GameAction>,
        hud: GameHud,
        egui: EguiUi,
        block_renderer: BlockRenderer,
    ) -> ClientState {
        ClientState {
            settings: settings.clone(),
            input: Mutex::new(InputState::new(settings)),
            block_types,
            items,
            last_update: Mutex::new(Instant::now()),
            physics_state: Mutex::new(physics::PhysicsState::new()),
            chunks: ChunkManager::new(),
            inventories: Mutex::new(InventoryViewManager::new()),
            tool_controller: Mutex::new(ToolController::new()),
            shutdown: tokio_util::sync::CancellationToken::new(),
            actions: action_sender,
            chat: Arc::new(Mutex::new(ChatState::new())),
            block_renderer: Arc::new(block_renderer),
            hud: Arc::new(Mutex::new(hud)),
            egui: Arc::new(Mutex::new(egui)),
            pending_error: Mutex::new(None),
            wants_exit_from_game: Mutex::new(false),
            last_position_weak: Mutex::new(PlayerPositionUpdate {
                position: cgmath::Vector3::zero(),
                velocity: cgmath::Vector3::zero(),
                face_direction: (0.0, 0.0),
            }),
        }
    }

    pub(crate) fn window_event(&self, event: &Event<()>) {
        let mut input = self.input.lock();
        input.event(event);
    }
    pub(crate) fn last_position(&self) -> PlayerPositionUpdate {
        let lock = self.physics_state.lock();
        PlayerPositionUpdate {
            // TODO tick, velocity
            position: lock.pos(),
            velocity: cgmath::Vector3::zero(),
            face_direction: lock.angle(),
        }
    }

    /// Returns the player's last position without requiring the physics lock (which could cause a lock order violation)
    /// This may be a frame behind
    pub(crate) fn weakly_ordered_last_position(&self) -> PlayerPositionUpdate {
        let _lock = self.physics_state.lock();
        *self.last_position_weak.lock()
    }

    pub(crate) fn next_frame(&self, aspect_ratio: f64) -> FrameState {
        {
            let mut input = self.input.lock();
            input.set_modal_active(self.egui.lock().wants_user_events());
            if input.take_just_pressed(BoundAction::Inventory) {
                self.egui.lock().open_inventory();
            } else if input.take_just_pressed(BoundAction::Menu) {
                self.egui.lock().open_pause_menu();
            } else if input.take_just_pressed(BoundAction::Chat) {
                self.egui.lock().open_chat();
            } else if input.take_just_pressed(BoundAction::ChatSlash) {
                self.egui.lock().open_chat_slash();
            }
        }

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
            * cgmath::Matrix4::from_angle_y(Deg(180.) - Deg(az));
        // TODO figure out why this is needed
        let coordinate_correction = cgmath::Matrix4::from_nonuniform_scale(-1., 1., 1.);
        let projection = cgmath::perspective(Deg(45.0), aspect_ratio, 0.01, 1000.);
        let view_proj_matrix = (projection * coordinate_correction * rotation)
            .cast()
            .unwrap();

        let mut tool_state = self.tool_controller.lock().update(self, delta);
        if let Some(action) = tool_state.action.take() {
            match self.actions.try_send(action) {
                Ok(_) => {}
                Err(e) => warn!("Failure sending action: {e:?}"),
            }
        }

        *self.last_position_weak.lock() = PlayerPositionUpdate {
            position: player_position,
            velocity: cgmath::Vector3::zero(),
            face_direction: (az, el),
        };

        FrameState {
            view_proj_matrix,
            player_position,
            tool_state,
        }
    }

    pub(crate) fn cancel_requested(&self) -> bool {
        self.shutdown.is_cancelled()
    }
}

pub(crate) struct FrameState {
    pub(crate) view_proj_matrix: cgmath::Matrix4<f32>,
    pub(crate) player_position: cgmath::Vector3<f64>,
    pub(crate) tool_state: ToolState,
}

use perovskite_core::protocol::blocks::{self as blocks_proto, CubeVariantEffect};

pub(crate) fn make_fallback_blockdef() -> blocks_proto::BlockTypeDef {
    blocks_proto::BlockTypeDef {
        render_info: Some(blocks_proto::block_type_def::RenderInfo::Cube(
            blocks_proto::CubeRenderInfo {
                tex_left: fallback_texture(),
                tex_right: fallback_texture(),
                tex_top: fallback_texture(),
                tex_bottom: fallback_texture(),
                tex_front: fallback_texture(),
                tex_back: fallback_texture(),
                render_mode: blocks_proto::CubeRenderMode::SolidOpaque.into(),
                variant_effect: CubeVariantEffect::None.into(),
            },
        )),
        groups: vec![DEFAULT_SOLID.to_string()],
        ..Default::default()
    }
}
