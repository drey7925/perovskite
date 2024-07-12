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

use std::ops::Deref;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use arc_swap::ArcSwap;
use cgmath::{vec3, Deg, InnerSpace, Vector3, Zero};
use perovskite_core::constants::block_groups::DEFAULT_SOLID;
use perovskite_core::constants::permissions;
use perovskite_core::coordinates::{BlockCoordinate, ChunkCoordinate, PlayerPositionUpdate};

use log::warn;
use parking_lot::{Mutex, RwLockReadGuard};
use perovskite_core::block_id::BlockId;
use perovskite_core::lighting::{ChunkColumn, Lightfield};
use perovskite_core::protocol;
use perovskite_core::protocol::game_rpc::{MapDeltaUpdateBatch, SetClientState};
use perovskite_core::time::TimeState;
use rustc_hash::{FxHashMap, FxHashSet};
use tokio::sync::mpsc;
use tracy_client::span;
use winit::event::Event;

use crate::game_state::chunk::ClientChunk;
use crate::game_ui::egui_ui::EguiUi;
use crate::game_ui::hud::GameHud;
use crate::vulkan::block_renderer::{fallback_texture, BlockRenderer};
use crate::vulkan::entity_renderer::EntityRenderer;
use crate::vulkan::shaders::cube_geometry::CubeGeometryDrawCall;
use crate::vulkan::shaders::SceneState;
use crate::vulkan::VkAllocator;

use self::block_types::ClientBlockTypeManager;
use self::chat::ChatState;
use self::chunk::{
    ChunkDataView, MeshBatch, MeshBatchBuilder, MeshResult, SnappyDecodeHelper,
    TARGET_BATCH_OCCUPANCY,
};
use self::entities::EntityState;
use self::input::{BoundAction, InputState};
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
    PopupResponse(protocol::ui::PopupResponse),
    InteractKey(InteractKeyAction),
    ChatMessage(String),
}

pub(crate) type ChunkMap = FxHashMap<ChunkCoordinate, Arc<ClientChunk>>;
pub(crate) type LightColumnMap = FxHashMap<(i32, i32), ChunkColumn>;
pub(crate) struct ChunkManager {
    chunks: parking_lot::RwLock<ChunkMap>,
    renderable_chunks: parking_lot::RwLock<ChunkMap>,
    light_columns: parking_lot::RwLock<LightColumnMap>,
    mesh_batches: Mutex<(FxHashMap<u64, MeshBatch>, MeshBatchBuilder)>,
}
impl ChunkManager {
    pub(crate) fn new() -> ChunkManager {
        ChunkManager {
            chunks: parking_lot::RwLock::new(FxHashMap::default()),
            renderable_chunks: parking_lot::RwLock::new(FxHashMap::default()),
            light_columns: parking_lot::RwLock::new(FxHashMap::default()),
            mesh_batches: Mutex::new((FxHashMap::default(), MeshBatchBuilder::new())),
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
        if let Some(chunk) = chunks_lock.get(coord) {
            // todo this is racy
            if let Some(batch) = chunk.get_batch() {
                self.spill(&chunks_lock, batch, "remove");
            }
        }
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
    ) -> Result<usize> {
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
                    warn!("Got delta with missing block_coord {:?}", update);
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
                    warn!("Got delta for unknown chunk {:?}", block_coord);
                    unknown_coords.push(block_coord);
                    None
                }
            };
            // unwrap because we expect all errors to be internal.
            if let Some(extra_chunks) = extra_chunks {
                let chunk = block_coord.chunk();
                needs_remesh.insert(chunk);
                for i in -1..=1 {
                    for j in (-1 - extra_chunks)..=1 {
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

    pub(crate) fn cloned_neighbors_fast(
        &self,
        chunk: ChunkCoordinate,
        result: &mut FastChunkNeighbors,
    ) {
        let _ = span!("cloned_neighbors_fast");
        let chunk_lock = {
            let _span = span!("chunk read lock");
            self.chunks.read()
        };
        let lights_lock = {
            let _span = span!("light column read lock");
            self.light_columns.read()
        };

        for i in -1..=1 {
            for k in -1..=1 {
                let light_column = chunk
                    .try_delta(i, 0, k)
                    .and_then(|delta| lights_lock.get(&(delta.x, delta.z)));
                for j in -1..=1 {
                    if let Some(delta) = chunk.try_delta(i, j, k) {
                        if let Some(chunk) = chunk_lock.get(&delta) {
                            result.neighbors[(k + 1) as usize][(j + 1) as usize]
                                [(i + 1) as usize]
                                .0 = true;
                            result.neighbors[(k + 1) as usize][(j + 1) as usize][(i + 1) as usize]
                                .1
                                .copy_from_slice(chunk.chunk_data().block_ids());
                        } else {
                            result.neighbors[(k + 1) as usize][(j + 1) as usize]
                                [(i + 1) as usize]
                                .0 = false;
                        }

                        if let Some(light_column) = light_column {
                            result.inbound_lights[(k + 1) as usize][(j + 1) as usize]
                                [(i + 1) as usize] = light_column
                                .get_incoming_light(delta.y)
                                .unwrap_or_else(Lightfield::zero);
                        }
                    } else {
                        result.neighbors[(k + 1) as usize][(j + 1) as usize][(i + 1) as usize].0 =
                            false;
                    }
                }
            }
        }
        result.center = chunk_lock.get(&chunk).cloned();
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
            match chunk.mesh_with(renderer)? {
                MeshResult::SameMesh => {
                    // todo we shouldn't need to spill here, but if we do, we get artifacts with
                    // double-rendered water. Why?
                    // Is this a bug here, or does the spill here just mask a bug elsewhere?
                    //self.spill(chunk, &src, "samemesh");
                }
                MeshResult::NewMesh(batch) => {
                    // We take the lock after mesh_with to keep the lock scope as short as possible
                    // and avoid blocking the render thread
                    let mut dst = self.renderable_chunks.write();
                    dst.insert(coord, chunk.clone());
                    if let Some(batch) = batch {
                        self.spill(&src, batch, "newmesh");
                    }
                }
                MeshResult::EmptyMesh(batch) => {
                    // Likewise.
                    let mut dst = self.renderable_chunks.write();
                    dst.remove(&coord);
                    if let Some(batch) = batch {
                        self.spill(&src, batch, "emptymesh");
                    }
                }
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub(crate) fn do_batch_round(
        &self,
        player_pos: Vector3<f64>,
        allocator: &VkAllocator,
    ) -> Result<()> {
        let _span = span!("batch_round");
        let mut chunks = self.chunks.read();
        let mut keys = chunks.keys().cloned().collect::<Vec<_>>();
        // Sort with closest items at the end
        keys.sort_unstable_by_key(|coord| {
            let world_coord = vec3(
                coord.x as f64 * 16.0 + 8.0,
                coord.y as f64 * 16.0 + 8.0,
                coord.z as f64 * 16.0 + 8.0,
            );
            -1 * ((player_pos - world_coord).magnitude2() as i64)
        });
        let mut counter = 0;
        let mut needs_resort = true;

        'outer: while let Some(coord) = keys.pop() {
            counter += 1;
            let world_coord = vec3(
                coord.x as f64 * 16.0 + 8.0,
                coord.y as f64 * 16.0 + 8.0,
                coord.z as f64 * 16.0 + 8.0,
            );
            if (player_pos - world_coord).magnitude() > 30.0 {
                let chunk = match chunks.get(&coord) {
                    Some(chunk) => chunk,
                    None => continue 'outer,
                };
                // Don't touch chunks that were meshed very recently
                // 500 milliseconds is our target for a round of chunk loading, so we'll wait that long as a starting point
                if chunk.last_meshed().elapsed() < Duration::from_millis(500) {
                    continue 'outer;
                }

                if let Some(chunk_data) = chunk.data_for_batch() {
                    // We want to keep some locality so we'll work on the chunks closest to the one we just picked.
                    if needs_resort {
                        keys.sort_unstable_by_key(|next_coord| {
                            let next_coord_f64 = vec3(
                                next_coord.x as f64 * 16.0 + 8.0,
                                next_coord.y as f64 * 16.0 + 8.0,
                                next_coord.z as f64 * 16.0 + 8.0,
                            );
                            let current_coord_f64 = vec3(
                                coord.x as f64 * 16.0 + 8.0,
                                coord.y as f64 * 16.0 + 8.0,
                                coord.z as f64 * 16.0 + 8.0,
                            );
                            -1 * ((next_coord_f64 - current_coord_f64).magnitude2() as i64)
                        });
                        needs_resort = false;
                    }
                    {
                        let _span = span!("batch_one_chunk");
                        let mut batches = {
                            let _span = span!("get batches lock");
                            self.mesh_batches.lock()
                        };
                        batches.1.append(coord, &chunk_data);
                        chunk.set_batch(batches.1.id());

                        if batches.1.occupancy() >= TARGET_BATCH_OCCUPANCY {
                            let new_batch = batches.1.build_and_reset(allocator)?;
                            batches.0.insert(new_batch.id(), new_batch);
                            // We're going to start a new batch. As soon as we put an item into it, we should
                            // try to get some locality around it.
                            needs_resort = true;
                        }
                    }
                    continue 'outer;
                }
            }
            if counter % 100 == 99 {
                RwLockReadGuard::bump(&mut chunks);
            }
        }
        Ok(())
    }

    pub(crate) fn make_batched_draw_calls(
        &self,
        player_position: Vector3<f64>,
    ) -> (Vec<CubeGeometryDrawCall>, FxHashSet<ChunkCoordinate>) {
        let mut calls = vec![];
        let mut handled = FxHashSet::default();
        let batches = self.mesh_batches.lock();
        for (_id, batch) in batches.0.iter() {
            calls.push(batch.make_draw_call(player_position));
            for coord in batch.coords() {
                if !handled.insert(*coord) {
                    //log::warn!("Already handled chunk {:?}", coord);
                }
                // It's tempting to verify that chunk_batch == Some(*id) here, but that's not always true
                // Due to the concurrency of the chunk manager, there is a race condition - we could have removed
                // the batch assignment of this chunk but not yet gotten far enough through spilling. (note that we don't have the chunk lock,
                // only the subordinate render-only chunk lock).
                //
                // This is also why we return a hashset - it's the only authoritative data on what chunks we *actually* rendered from batches. Since
                // batches themselves are immutable, if we encounter a batch here, we know that we at least have a consistent view of it.
                //
                // The following snippet can be used for debugging to see the rate of this race condition.
                //
                // let (chunk_batch, reason) = chunks.get(coord).unwrap().get_batch_debug();
                // if chunk_batch != Some(*id) {
                //     println!(
                //         "Chunk {:?} should be in batch {:?}, was in {:?}, reason {:?}",
                //         coord, chunk_batch, id, reason
                //     );
                // }
            }
        }
        (calls, handled)
    }

    fn spill(
        &self,
        chunks: &FxHashMap<ChunkCoordinate, Arc<ClientChunk>>,
        batch_id: u64,
        _reason: &str,
    ) {
        {
            let _span = span!("batch_spill");
            // Possibly spill a mesh batch if necessary
            let mut batch_lock = self.mesh_batches.lock();
            if batch_lock.1.id() == batch_id {
                // We are spilling the current batch builder itself
                for spilled_coord in batch_lock.1.chunks() {
                    if let Some(spilled_chunk) = chunks.get(&spilled_coord) {
                        spilled_chunk.spill_back_to_solo(batch_id);
                    }
                }
                batch_lock.1.reset();
            } else {
                // We are spilling a finished batch. It's possible we may not find this batch, in case a different thread spilled it already.
                if let Some(spilled_batch) = batch_lock.0.remove(&batch_id) {
                    let _batch_finished = false;
                    for spilled_coord in spilled_batch.coords() {
                        if let Some(spilled_chunk) = chunks.get(spilled_coord) {
                            spilled_chunk.spill_back_to_solo(batch_id);
                        }
                    }
                }
            }
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

type ChunkWithEdgesBuffer = (bool, Box<[BlockId; 18 * 18 * 18]>);

pub(crate) struct FastChunkNeighbors {
    center: Option<Arc<ClientChunk>>,
    neighbors: [[[ChunkWithEdgesBuffer; 3]; 3]; 3],
    inbound_lights: [[[Lightfield; 3]; 3]; 3],
}
impl FastChunkNeighbors {
    pub(crate) fn get(&self, coord_xyz: (i32, i32, i32)) -> Option<&[BlockId; 18 * 18 * 18]> {
        assert!((-1..=1).contains(&coord_xyz.0));
        assert!((-1..=1).contains(&coord_xyz.1));
        assert!((-1..=1).contains(&coord_xyz.2));
        let chunk = &self.neighbors[(coord_xyz.2 + 1) as usize][(coord_xyz.1 + 1) as usize]
            [(coord_xyz.0 + 1) as usize];
        if chunk.0 {
            Some(&chunk.1)
        } else {
            None
        }
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

impl Default for FastChunkNeighbors {
    fn default() -> Self {
        Self {
            center: None,
            neighbors: std::array::from_fn(|_| {
                std::array::from_fn(|_| {
                    std::array::from_fn(|_| (false, Box::new([BlockId(0); 18 * 18 * 18])))
                })
            }),
            inbound_lights: std::array::from_fn(|_| {
                std::array::from_fn(|_| std::array::from_fn(|_| Lightfield::zero()))
            }),
        }
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
    pub(crate) last_update: Mutex<Instant>,
    pub(crate) physics_state: Mutex<physics::PhysicsState>,
    pub(crate) chunks: ChunkManager,
    pub(crate) inventories: Mutex<InventoryViewManager>,
    pub(crate) tool_controller: Mutex<ToolController>,
    pub(crate) shutdown: tokio_util::sync::CancellationToken,
    pub(crate) actions: mpsc::Sender<GameAction>,
    pub(crate) light_cycle: Arc<Mutex<LightCycle>>,
    pub(crate) chat: Arc<Mutex<ChatState>>,
    pub(crate) entities: Mutex<EntityState>,

    // block_renderer doesn't currently need a mutex because it's stateless
    // and keeps its cached geometry data in the chunks themselves
    pub(crate) block_renderer: Arc<BlockRenderer>,
    pub(crate) entity_renderer: Arc<EntityRenderer>,
    // GameHud manages its own state.
    pub(crate) hud: Arc<Mutex<GameHud>>,
    pub(crate) egui: Arc<Mutex<EguiUi>>,

    pub(crate) pending_error: Mutex<Option<String>>,
    pub(crate) wants_exit_from_game: Mutex<bool>,
    // This is a leaf mutex - consider using some sort of atomic instead
    pub(crate) last_position_weak: Mutex<PlayerPositionUpdate>,

    pub(crate) timekeeper: Timekeeper,
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
        entity_renderer: EntityRenderer,
        timekeeper: Timekeeper,
    ) -> Result<ClientState> {
        Ok(ClientState {
            settings: settings.clone(),
            input: Mutex::new(InputState::new(settings.clone())),
            block_types,
            items,
            last_update: Mutex::new(Instant::now()),
            physics_state: Mutex::new(physics::PhysicsState::new(settings)),
            chunks: ChunkManager::new(),
            inventories: Mutex::new(InventoryViewManager::new()),
            tool_controller: Mutex::new(ToolController::new()),
            shutdown: tokio_util::sync::CancellationToken::new(),
            actions: action_sender,
            light_cycle: Arc::new(Mutex::new(LightCycle::new(TimeState::new(
                Duration::from_secs(24 * 60),
                0.0,
            )))),
            chat: Arc::new(Mutex::new(ChatState::new())),
            entities: Mutex::new(EntityState::new(&block_renderer)?),

            block_renderer: Arc::new(block_renderer),
            entity_renderer: Arc::new(entity_renderer),
            hud: Arc::new(Mutex::new(hud)),
            egui: Arc::new(Mutex::new(egui)),
            pending_error: Mutex::new(None),
            wants_exit_from_game: Mutex::new(false),
            last_position_weak: Mutex::new(PlayerPositionUpdate {
                position: Vector3::zero(),
                velocity: Vector3::zero(),
                face_direction: (0.0, 0.0),
            }),
            timekeeper,
        })
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
            velocity: Vector3::zero(),
            face_direction: lock.angle(),
        }
    }

    /// Returns the player's last position without requiring the physics lock (which could cause a lock order violation)
    /// This may be a frame behind
    pub(crate) fn weakly_ordered_last_position(&self) -> PlayerPositionUpdate {
        *self.last_position_weak.lock()
    }

    pub(crate) fn next_frame(&self, aspect_ratio: f64) -> FrameState {
        let mut egui_wants_events = false;
        {
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
            } else if input.take_just_pressed(BoundAction::PhysicsDebug) {
                self.egui.lock().toggle_debug();
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

        let projection = cgmath::perspective(
            Deg(self.settings.load().render.fov_degrees),
            aspect_ratio,
            0.01,
            1000.,
        );
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
            velocity: Vector3::zero(),
            face_direction: (az, el),
        };

        let (sky, lighting, light_direction) = self.light_cycle.lock().get_colors();
        FrameState {
            scene_state: SceneState {
                vp_matrix: view_proj_matrix,
                clear_color: [sky.x, sky.y, sky.z, 1.],
                global_light_color: lighting.into(),
                global_light_direction: light_direction,
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
            )
        }
        {
            let mut tc_lock = self.tool_controller.lock();
            tc_lock.update_permissions(&state_update.permission);
        }
        {
            let mut hud_lock = self.hud.lock();
            hud_lock.hotbar_view_id = Some(state_update.hotbar_inventory_view);
            hud_lock.invalidate_hotbar();
        }

        {
            let mut time_lock = self.light_cycle.lock();
            time_lock
                .time_state_mut()
                .set_time(state_update.time_of_day);
            time_lock
                .time_state_mut()
                .set_day_length(Duration::from_secs_f64(state_update.day_length_sec));
        }

        {
            let mut entities_lock = self.entities.lock();
            entities_lock.attached_to_entity = match state_update.attached_to_entity {
                0 => None,
                id => Some(id),
            }
        }
        Ok(())
    }
}

pub(crate) struct FrameState {
    pub(crate) scene_state: SceneState,
    pub(crate) player_position: Vector3<f64>,
    pub(crate) tool_state: ToolState,
    pub(crate) ime_enabled: bool,
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
