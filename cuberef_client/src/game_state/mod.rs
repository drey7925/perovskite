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

use arc_swap::ArcSwap;
use cgmath::{Deg, Zero};
use cuberef_core::constants::block_groups::DEFAULT_SOLID;
use cuberef_core::coordinates::{BlockCoordinate, ChunkCoordinate, PlayerPositionUpdate};

use cuberef_core::block_id::BlockId;
use cuberef_core::protocol;
use log::warn;
use parking_lot::{Mutex, RwLockReadGuard};
use rustc_hash::FxHashMap;
use tokio::sync::mpsc;
use tracy_client::span;
use winit::event::Event;

use crate::block_renderer::{fallback_texture, BlockRenderer, ClientBlockTypeManager};
use crate::game_state::chunk::ClientChunk;
use crate::game_ui::egui_ui::EguiUi;
use crate::game_ui::hud::GameHud;

use self::chunk::ChunkDataView;
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
    pub(crate) item_slot: u32,
    pub(crate) player_pos: PlayerPositionUpdate,
}

#[derive(Debug, Clone)]
pub(crate) enum GameAction {
    Dig(DigTapAction),
    Tap(DigTapAction),
    Place(PlaceAction),
    Inventory(InventoryAction),
    PopupResponse(cuberef_core::protocol::ui::PopupResponse),
    InteractKey(InteractKeyAction),
}

pub(crate) type ChunkMap = FxHashMap<ChunkCoordinate, Arc<ClientChunk>>;
pub(crate) struct ChunkManager {
    chunks: parking_lot::RwLock<ChunkMap>,
}
impl ChunkManager {
    pub(crate) fn new() -> ChunkManager {
        ChunkManager {
            chunks: parking_lot::RwLock::new(FxHashMap::default()),
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
    ///      memory will not be released until the cloned view is dropped.
    pub(crate) fn cloned_view(&self) -> ChunkManagerClonedView {
        ChunkManagerClonedView {
            data: self.chunks.read().clone(),
        }
    }
    pub(crate) fn insert(
        &self,
        coord: ChunkCoordinate,
        chunk: ClientChunk,
    ) -> Option<Arc<ClientChunk>> {
        let mut lock = {
            let _span = span!("Acquire global chunk lock");
            self.chunks.write()
        };
        lock.insert(coord, Arc::new(chunk))
    }
    pub(crate) fn remove(&self, coord: &ChunkCoordinate) -> Option<Arc<ClientChunk>> {
        let mut lock = {
            let _span = span!("Acquire global chunk lock");
            self.chunks.write()
        };
        lock.remove(coord)
    }

    pub(crate) fn insert_or_update(
        &self,
        coord: ChunkCoordinate,
        proto: protocol::game_rpc::MapChunk,
    ) -> anyhow::Result<()> {
        let mut lock = {
            let _span = span!("Acquire global chunk lock");
            self.chunks.write()
        };
        match lock.entry(coord) {
            std::collections::hash_map::Entry::Occupied(x) => x.get().update_from(proto),
            std::collections::hash_map::Entry::Vacant(x) => {
                x.insert(Arc::new(ClientChunk::from_proto(proto)?));
                Ok(())
            }
        }
    }

    pub(crate) fn cloned_neighbors(&self, chunk: ChunkCoordinate) -> ChunkManagerClonedView {
        let lock = self.chunks.read();
        let mut data = ChunkMap::default();
        for i in -1..=1 {
            for j in -1..=1 {
                for k in -1..=1 {
                    if let Some(delta) = chunk.try_delta(i, j, k) {
                        if let Some(chunk) = lock.get(&delta) {
                            data.insert(delta, chunk.clone());
                        }
                    }
                }
            }
        }
        ChunkManagerClonedView { data }
    }

    pub(crate) fn cloned_neighbors_fast(&self, chunk: ChunkCoordinate) -> FastChunkNeighbors {
        let lock = self.chunks.read();
        let mut data =
            std::array::from_fn(|_| std::array::from_fn(|_| std::array::from_fn(|_| None)));
        for i in -1..=1 {
            for j in -1..=1 {
                for k in -1..=1 {
                    if let Some(delta) = chunk.try_delta(i, j, k) {
                        data[(k + 1) as usize][(j + 1) as usize][(i + 1) as usize] =
                            lock.get(&delta).cloned();
                    }
                }
            }
        }
        FastChunkNeighbors { data }
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

pub(crate) struct FastNeighborLockCache<'a> {
    backing: PhantomData<&'a FastChunkNeighbors>,
    // does NOT contain (0,0,0)
    neighbors: [[[Option<ChunkDataView<'a>>; 3]; 3]; 3],
}
impl<'a> FastNeighborLockCache<'a> {
    pub(crate) fn new(backing: &FastChunkNeighbors) -> FastNeighborLockCache {
        let mut neighbors =
            std::array::from_fn(|_| std::array::from_fn(|_| std::array::from_fn(|_| None)));
        for i in -1..=1 {
            for j in -1..=1 {
                for k in -1..=1 {
                    if i != 0 || j != 0 || k != 0 {
                        if let Some(chunk) = backing.get((i, j, k)) {
                            neighbors[(k + 1) as usize][(j + 1) as usize][(i + 1) as usize] =
                                Some(chunk.chunk_data());
                        }
                    }
                }
            }
        }
        FastNeighborLockCache {
            backing: PhantomData,
            neighbors,
        }
    }

    pub(crate) fn get(&self, coord_xyz: (i32, i32, i32)) -> Option<&ChunkDataView> {
        self.neighbors[(coord_xyz.2 + 1) as usize][(coord_xyz.1 + 1) as usize]
            [(coord_xyz.0 + 1) as usize]
            .as_ref()
    }
}

pub(crate) struct FastNeighborSliceCache<'a, 'b> {
    backing: PhantomData<&'a FastNeighborLockCache<'b>>,
    // does NOT contain (0,0,0)
    neighbors: [Option<&'a [BlockId; 18 * 18 * 18]>; 27],
}
impl<'a, 'b> FastNeighborSliceCache<'a, 'b> {
    pub(crate) fn new(backing: &'a FastNeighborLockCache<'b>) -> FastNeighborSliceCache<'a, 'b> {
        let mut neighbors = [None; 27];
        for i in -1..=1 {
            for j in -1..=1 {
                for k in -1..=1 {
                    if i != 0 || j != 0 || k != 0 {
                        if let Some(chunk) = backing.get((i, j, k)) {
                            neighbors[((k + 1) * 9) as usize
                                + ((j + 1) * 3) as usize
                                + (i + 1) as usize] = Some(chunk.block_ids());
                        }
                    }
                }
            }
        }
        FastNeighborSliceCache {
            backing: PhantomData,
            neighbors,
        }
    }

    pub(crate) fn get(&self, coord_xyz: (i32, i32, i32)) -> Option<&'a [BlockId; 18 * 18 * 18]> {
        assert!((-1..=1).contains(&coord_xyz.0));
        assert!((-1..=1).contains(&coord_xyz.1));
        assert!((-1..=1).contains(&coord_xyz.2));
        self.neighbors[((coord_xyz.2 + 1) * 9) as usize
            + ((coord_xyz.1 + 1) * 3) as usize
            + (coord_xyz.0 + 1) as usize]
    }
}

pub(crate) struct FastChunkNeighbors {
    data: [[[Option<Arc<ClientChunk>>; 3]; 3]; 3],
}
impl FastChunkNeighbors {
    pub(crate) fn get(&self, coord_xyz: (i32, i32, i32)) -> Option<&Arc<ClientChunk>> {
        assert!((-1..=1).contains(&coord_xyz.0));
        assert!((-1..=1).contains(&coord_xyz.1));
        assert!((-1..=1).contains(&coord_xyz.2));
        self.data[(coord_xyz.2 + 1) as usize][(coord_xyz.1 + 1) as usize]
            [(coord_xyz.0 + 1) as usize]
            .as_ref()
    }
}

pub(crate) struct ChunkManagerClonedView {
    data: ChunkMap,
}
impl ChunkManagerClonedView {
    pub(crate) fn get(&self, coord: &ChunkCoordinate) -> Option<&Arc<ClientChunk>> {
        self.data.get(coord)
    }
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

    /// Returns the player's last position without requiring any locks
    /// This may be a frame behind
    pub(crate) fn weakly_ordered_last_position(&self) -> PlayerPositionUpdate {
        let _lock = self.physics_state.lock();
        *self.last_position_weak.lock()
    }

    pub(crate) fn next_frame(&self, aspect_ratio: f64) -> FrameState {
        {
            let mut input = self.input.lock();
            input.set_modal_active(self.egui.lock().wants_draw());
            if input.take_just_pressed(BoundAction::Inventory) {
                self.egui.lock().open_inventory();
            } else if input.take_just_pressed(BoundAction::Menu) {
                self.egui.lock().open_pause_menu();
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

use cuberef_core::protocol::blocks::{self as blocks_proto, CubeVariantEffect};

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
