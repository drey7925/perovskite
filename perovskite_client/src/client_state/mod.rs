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

use std::f64::consts::PI;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use arc_swap::ArcSwap;
use cgmath::{vec3, Deg, InnerSpace, Matrix4, Vector3, Zero};
use dashmap::DashMap;
use egui::Color32;
use enum_map::{Enum, EnumMap};
use perovskite_core::constants::block_groups::DEFAULT_SOLID;
use perovskite_core::constants::permissions;
use perovskite_core::coordinates::{
    BlockCoordinate, ChunkCoordinate, ChunkOffset, PlayerPositionUpdate,
};

use log::warn;
use parking_lot::Mutex;
use perovskite_core::block_id::BlockId;
use perovskite_core::game_actions::ToolTarget;
use perovskite_core::lighting::{ChunkColumn, Lightfield};
use perovskite_core::protocol;
use perovskite_core::protocol::game_rpc::{
    MapDeltaUpdateBatch, ServerPerformanceMetrics, SetClientState,
};
use perovskite_core::time::TimeState;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use seqlock::SeqLock;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracy_client::span;
use winit::event::{DeviceEvent, WindowEvent};

use crate::client_state::chunk::ClientChunk;
use crate::game_ui::egui_ui::EguiUi;
use crate::game_ui::hud::GameHud;
use crate::vulkan::block_renderer::BlockRenderer;
use crate::vulkan::entity_renderer::EntityRenderer;
use crate::vulkan::shaders::cube_geometry::CubeGeometryDrawCall;
use crate::vulkan::shaders::SceneState;
use crate::vulkan::VulkanContext;

use self::block_types::ClientBlockTypeManager;
use self::chat::ChatState;
use self::chunk::{ChunkDataView, MeshBatch, MeshBatchBuilder, MeshResult, TARGET_BATCH_OCCUPANCY};
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
pub(crate) type ChunkDashmap = DashMap<ChunkCoordinate, Arc<ClientChunk>, FxBuildHasher>;
/// A non-dashmapped version of the chunk map, used for non-concurrent cases like passing
/// chunks to the raytracer
pub(crate) type ChunkMap = FxHashMap<ChunkCoordinate, Arc<ClientChunk>>;
pub(crate) type LightColumnDashmap =
    DashMap<(i32, i32), ChunkColumn<DefaultSyncBackend>, FxBuildHasher>;
pub(crate) struct ChunkManager {
    chunks: ChunkDashmap,
    renderable_chunks: ChunkDashmap,
    light_columns: LightColumnDashmap,
    mesh_batches: Mutex<(FxHashMap<u64, MeshBatch>, MeshBatchBuilder)>,
    raytrace_buffers: Arc<RaytraceBufferManager>,
}
impl ChunkManager {
    pub(crate) fn new(raytrace_buffers: Arc<RaytraceBufferManager>) -> ChunkManager {
        ChunkManager {
            chunks: DashMap::with_hasher(FxBuildHasher),
            renderable_chunks: DashMap::with_hasher(FxBuildHasher),
            light_columns: DashMap::with_hasher(FxBuildHasher),
            mesh_batches: Mutex::new((FxHashMap::default(), MeshBatchBuilder::new())),
            raytrace_buffers,
        }
    }

    /// Clones the chunk manager and returns a clone with the following properties:
    /// * The clone does not hold any locks on the data in this chunk manager (i.e. insertions and deletions
    ///   are possible while the cloned view is live).
    /// * The clone does not track any insertions/deletions of this chunk manager.
    ///    * It will not show chunks inserted after this function returns.
    ///    * If chunks are deleted after cloned_view returned, they will remain in the cloned view, and their
    ///      memory will not be released until the cloned view is dropped, due to Arcs owned by the cloned view.
    pub(crate) fn renderable_chunks_cloned_view(&self) -> ChunkMap {
        self.renderable_chunks
            .iter()
            .map(|x| (x.key().clone(), x.value().clone()))
            .collect()
    }

    pub(crate) fn remove(&self, coord: &ChunkCoordinate) -> Option<Arc<ClientChunk>> {
        if let Some(chunk) = self.chunks.get(coord) {
            let batch = chunk.get_batch();
            drop(chunk); // avoid reentrant lock on the dashmap
            if let Some(batch) = batch {
                // todo this is racy
                self.spill(batch, "remove");
            }
        }
        // "Lock order": chunks -> renderable_chunks -> light_columns
        // Same note applies - these are dashmap maps, so treat outstanding Refs as equivalent to
        // locks for deadlock analysis

        let chunk = self.chunks.remove(coord);
        // The column should be present, so we unwrap.
        if chunk.is_some() {
            let mut column = self.light_columns.get_mut(&(coord.x, coord.z)).unwrap();
            column.value_mut().remove(coord.y);
            if column.is_empty() {
                drop(column);
                self.light_columns.remove(&(coord.x, coord.z));
            }
        }
        self.renderable_chunks.remove(coord);
        chunk.map(|x| x.1)
    }
    fn handle_mapchunk_audio(
        &self,
        client_state: &ClientState,
        coord: ChunkCoordinate,
        block_ids: &[u32; 4096],
    ) {
        let tick = client_state.timekeeper.now();
        let pos = client_state.weakly_ordered_last_position();
        let block_types = client_state.block_types.deref();

        let mut sounds: smallvec::SmallVec<[_; 16]> = smallvec::SmallVec::new();
        for (i, block) in block_ids.iter().enumerate() {
            if let Some((id, volume)) = block_types.block_sound(BlockId::from(*block)) {
                sounds.push((i, id, volume));
            }
        }
        if !sounds.is_empty() {
            let mut map_sound = client_state.world_audio.lock();
            for (i, id, volume) in sounds.into_iter() {
                map_sound.insert_or_update(
                    tick,
                    pos.position,
                    coord.with_offset(ChunkOffset::from_index(i)),
                    id.get(),
                    volume,
                )
            }
        }
    }

    // Returns the number of additional chunks below the given chunk that need lighting updates.
    pub(crate) fn insert_or_update(
        &self,
        client_state: &ClientState,
        coord: ChunkCoordinate,
        block_ids: &[u32; 4096],
        ced: Vec<ClientExtendedData>,
        block_types: &ClientBlockTypeManager,
    ) -> Result<usize> {
        let mut light_column = self
            .light_columns
            .entry((coord.x, coord.z))
            .or_insert_with(ChunkColumn::empty);
        let occlusion = match self.chunks.entry(coord) {
            dashmap::Entry::Occupied(chunk_entry) => {
                client_state.world_audio.lock().remove_chunk(coord);
                chunk_entry
                    .get()
                    .update_from(coord, block_ids, ced, block_types)?
            }
            dashmap::Entry::Vacant(x) => {
                let (chunk, occlusion) =
                    ClientChunk::from_proto(coord, block_ids, ced, block_types)?;
                light_column.value_mut().insert_empty(coord.y);
                x.insert(Arc::new(chunk));
                occlusion
            }
        };
        self.handle_mapchunk_audio(client_state, coord, block_ids);

        let mut light_cursor = light_column.cursor_into(coord.y);
        *light_cursor.current_occlusion_mut() = occlusion;
        light_cursor.mark_valid();
        let extra_chunks = light_cursor.propagate_lighting();
        Ok(extra_chunks)
    }

    pub(crate) fn apply_delta_batch(
        &self,
        batch: &mut MapDeltaUpdateBatch,
        block_types: &ClientBlockTypeManager,
    ) -> Result<(FxHashSet<ChunkCoordinate>, Vec<BlockCoordinate>, bool)> {
        let mut needs_remesh = FxHashSet::default();
        let mut unknown_coords = vec![];
        let mut missing_coord = false;

        for update in batch.updates.drain(..) {
            let block_coord: BlockCoordinate = match &update.block_coord {
                Some(x) => x.into(),
                None => {
                    warn!("Got delta with missing block_coord {:?}", update);
                    missing_coord = true;
                    continue;
                }
            };
            let chunk_coord = block_coord.chunk();
            let extra_chunks = match self.chunks.get(&chunk_coord) {
                Some(x) => {
                    if !x.apply_delta(update)? {
                        None
                    } else {
                        let occlusion = x.get_occlusion(block_types);
                        let light_column = self
                            .light_columns
                            .get(&(chunk_coord.x, chunk_coord.z))
                            .expect("Missing light column during delta update");
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

        let center_chunk = self.chunks.get(&chunk);
        if center_chunk
            .as_ref()
            .is_none_or(|x| x.chunk_data().is_empty_optimization_hint())
        {
            result.outcome = ChunkNeighborOutcome::DontMesh;
            return;
        }

        result.center = center_chunk.map(|x| x.value().clone());
        result.outcome = ChunkNeighborOutcome::ShouldMesh;

        for i in -1..=1 {
            for k in -1..=1 {
                let light_column = chunk
                    .try_delta(i, 0, k)
                    .and_then(|delta| self.light_columns.get(&(delta.x, delta.z)));
                for j in -1..=1 {
                    if let Some(delta) = chunk.try_delta(i, j, k) {
                        if let Some(chunk) = self.chunks.get(&delta) {
                            if i != 0 || j != 0 || k != 0 {
                                result.neighbors[(k + 1) as usize][(j + 1) as usize]
                                    [(i + 1) as usize]
                                    .0 = true;
                                result.neighbors[(k + 1) as usize][(j + 1) as usize]
                                    [(i + 1) as usize]
                                    .1
                                    .copy_from_slice(chunk.chunk_data().block_ids());
                            }
                        } else {
                            result.neighbors[(k + 1) as usize][(j + 1) as usize]
                                [(i + 1) as usize]
                                .0 = false;
                        }

                        if let Some(light_column) = light_column.as_ref() {
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
    }

    /// If the chunk is present, meshes it. If any geometry was generated, inserts it into
    /// the renderable chunks. If no geometry was generated, removes it from the renderable chunks.
    /// Returns true if the chunk was present in the *main* list of chunks, false otherwise.
    pub(crate) fn maybe_mesh_and_maybe_promote(
        &self,
        coord: ChunkCoordinate,
        renderer: &BlockRenderer,
    ) -> Result<bool> {
        // "Lock order": notes: self.chunks -> self.renderable_chunks
        // Neighbor propagation and mesh generation only need self.chunks
        // Rendering only needs self.renderable_chunks, and that lock is limited in scope
        // to Self::cloned_view_renderable. The render thread should never end up needing
        // both chunks and renderable_chunks locked at the same time (and the render thread's
        // use of chunks is likewise scoped to the physics code).
        //
        // Note that this isn't literal lock order since both are dashmaps. However, a dashmap is
        // deadlock-prone under certain re-entrant use, so treat accesses/outstanding refs to it as
        // effective locks for lock order considerations.

        if let Some(chunk) = self.chunks.get(&coord) {
            let raster_state = chunk.mesh_with(renderer, Some(&self.raytrace_buffers))?;

            match raster_state {
                MeshResult::SameMesh => {
                    // Don't spill or anything
                }
                MeshResult::NewMesh(batch) => {
                    // We take the lock after mesh_with to keep the lock scope as short as possible
                    // and avoid blocking the render thread
                    self.renderable_chunks.insert(coord, chunk.clone());
                    if let Some(batch) = batch {
                        self.spill(batch, "newmesh");
                    }
                }
                MeshResult::EmptyMesh(batch) => {
                    // Likewise.
                    self.renderable_chunks.remove(&coord);
                    if let Some(batch) = batch {
                        self.spill(batch, "emptymesh");
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
        vk_ctx: &VulkanContext,
    ) -> Result<()> {
        let _span = span!("batch_round");
        let mut keys = self
            .chunks
            .iter()
            .map(|x| x.key().clone())
            .collect::<Vec<_>>();
        // Sort with closest items at the end
        keys.sort_unstable_by_key(|coord| {
            let world_coord = vec3(
                coord.x as f64 * 16.0 + 8.0,
                coord.y as f64 * 16.0 + 8.0,
                coord.z as f64 * 16.0 + 8.0,
            );
            -1 * ((player_pos - world_coord).magnitude2() as i64)
        });
        let mut needs_resort = true;

        'outer: while let Some(coord) = keys.pop() {
            let world_coord = vec3(
                coord.x as f64 * 16.0 + 8.0,
                coord.y as f64 * 16.0 + 8.0,
                coord.z as f64 * 16.0 + 8.0,
            );
            if (player_pos - world_coord).magnitude() > 30.0 {
                let chunk = match self.chunks.get(&coord) {
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
                            let new_batch = batches.1.build_and_reset(vk_ctx)?;
                            batches.0.insert(new_batch.id(), new_batch);
                            // We're going to start a new batch. As soon as we put an item into it, we should
                            // try to get some locality around it.
                            needs_resort = true;
                        }
                    }
                    continue 'outer;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn average_solid_batch_occupancy(&self) -> (usize, usize) {
        let mut vertices = 0;
        let mut indices = 0;

        let batches = self.mesh_batches.lock();
        if batches.0.is_empty() {
            return (0, 0);
        }

        for (_id, batch) in batches.0.iter() {
            let (v, i) = batch.solid_occupancy();
            vertices += v;
            indices += i;
        }
        (vertices / batches.0.len(), indices / batches.0.len())
    }

    pub(crate) fn make_batched_draw_calls(
        &self,
        player_position: Vector3<f64>,
        vp_matrix: Matrix4<f32>,
    ) -> (Vec<CubeGeometryDrawCall>, FxHashSet<ChunkCoordinate>) {
        let mut calls = vec![];
        let mut handled = FxHashSet::default();
        let batches = self.mesh_batches.lock();
        for (_id, batch) in batches.0.iter() {
            if let Some(call) = batch.make_draw_call(player_position, vp_matrix) {
                calls.push(call);
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
        }
        (calls, handled)
    }

    fn spill(&self, batch_id: u64, _reason: &str) {
        {
            let _span = span!("batch_spill");
            // Possibly spill a mesh batch if necessary
            let mut batch_lock = self.mesh_batches.lock();
            if batch_lock.1.id() == batch_id {
                // We are spilling the current batch builder itself
                for spilled_coord in batch_lock.1.chunks() {
                    if let Some(spilled_chunk) = self.chunks.get(&spilled_coord) {
                        spilled_chunk.spill_back_to_solo(batch_id);
                    }
                }
                batch_lock.1.reset();
            } else {
                // We are spilling a finished batch. It's possible we may not find this batch, in case a different thread spilled it already.
                if let Some(spilled_batch) = batch_lock.0.remove(&batch_id) {
                    let _batch_finished = false;
                    for spilled_coord in spilled_batch.coords() {
                        if let Some(spilled_chunk) = self.chunks.get(spilled_coord) {
                            spilled_chunk.spill_back_to_solo(batch_id);
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn raytrace_buffers(&self) -> &RaytraceBufferManager {
        &self.raytrace_buffers
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChunkNeighborOutcome {
    ShouldMesh,
    DontMesh,
}

struct ChunkWithEdgesBuffer(bool, Box<[BlockId; 18 * 18 * 18]>);

pub(crate) struct FastChunkNeighbors {
    outcome: ChunkNeighborOutcome,
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
    #[inline]
    pub(crate) fn inbound_light(&self, coord_xyz: (i32, i32, i32)) -> Lightfield {
        assert!((-1..=1).contains(&coord_xyz.0));
        assert!((-1..=1).contains(&coord_xyz.1));
        assert!((-1..=1).contains(&coord_xyz.2));
        self.inbound_lights[(coord_xyz.2 + 1) as usize][(coord_xyz.1 + 1) as usize]
            [(coord_xyz.0 + 1) as usize]
    }
    pub(crate) fn should_mesh(&self) -> bool {
        self.outcome == ChunkNeighborOutcome::ShouldMesh
    }
}

impl Default for FastChunkNeighbors {
    fn default() -> Self {
        Self {
            center: None,
            neighbors: std::array::from_fn(|_| {
                std::array::from_fn(|_| {
                    std::array::from_fn(|_| {
                        ChunkWithEdgesBuffer(false, Box::new([BlockId(0); 18 * 18 * 18]))
                    })
                })
            }),
            inbound_lights: std::array::from_fn(|_| {
                std::array::from_fn(|_| std::array::from_fn(|_| Lightfield::zero()))
            }),
            outcome: ChunkNeighborOutcome::DontMesh,
        }
    }
}

// todo clean up, make private
pub(crate) struct ClientState {
    pub(crate) settings: Arc<ArcSwap<GameSettings>>,
    pub(crate) input: Mutex<InputState>,

    pub(crate) block_types: Arc<ClientBlockTypeManager>,
    pub(crate) items: Arc<ClientItemManager>,
    pub(crate) last_update: Mutex<u64>,
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

    pub(crate) pending_error: Mutex<Option<anyhow::Error>>,
    pub(crate) wants_exit_from_game: Mutex<bool>,
    pub(crate) last_position_weak: SeqLock<PlayerPositionUpdate>,

    pub(crate) timekeeper: Arc<Timekeeper>,

    pub(crate) audio: Arc<audio::EngineHandle>,
    pub(crate) world_audio: Mutex<audio::MapSoundState>,

    pub(crate) server_perf: Mutex<Option<ServerPerformanceMetrics>>,
    pub(crate) client_perf: Mutex<Option<ClientPerformanceMetrics>>,
    pub(crate) want_server_perf: AtomicBool,
    pub(crate) want_new_client_perf: tokio::sync::Notify,
    pub(crate) render_distance: AtomicU32,
}

const PROJ_NEAR: f64 = 0.05;
const PROJ_FAR: f64 = 10000.;

impl ClientState {
    pub(crate) fn new(
        settings: Arc<ArcSwap<GameSettings>>,
        shutdown: CancellationToken,
        block_types: Arc<ClientBlockTypeManager>,
        chunks: ChunkManager,
        items: Arc<ClientItemManager>,
        action_sender: mpsc::Sender<GameAction>,
        hud: GameHud,
        egui: EguiUi,
        block_renderer: BlockRenderer,
        entity_renderer: EntityRenderer,
        timekeeper: Arc<Timekeeper>,
        audio: Arc<audio::EngineHandle>,
    ) -> Result<ClientState> {
        let audio_clone = audio.clone();
        Ok(ClientState {
            settings: settings.clone(),
            input: Mutex::new(InputState::new(settings.clone())),
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
            world_audio: Mutex::new(MapSoundState::new(audio_clone)),
            server_perf: Mutex::new(None),
            client_perf: Mutex::new(None),
            want_server_perf: AtomicBool::new(false),
            want_new_client_perf: tokio::sync::Notify::new(),
            render_distance: AtomicU32::new(settings.load().render.render_distance),
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
        let egui_wants_events;
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

        let mut tool_state = self.tool_controller.lock().update(self, ppu, delta, tick);
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
        Ok(())
    }

    pub(crate) fn server_perf(&self) -> Option<ServerPerformanceMetrics> {
        self.server_perf.lock().clone()
    }

    pub(crate) fn client_perf(&self) -> Option<ClientPerformanceMetrics> {
        self.client_perf.lock().clone()
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
    pub(crate) const TIMEKEEPER_CHART_LEN: usize = 256;
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
use crate::vulkan::raytrace_buffer::RaytraceBufferManager;
use perovskite_core::protocol::blocks::{self as blocks_proto, CubeVariantEffect};
use perovskite_core::protocol::map::ClientExtendedData;
use perovskite_core::sync::DefaultSyncBackend;

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
