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

use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cgmath::{vec3, vec4, ElementWise, InnerSpace, Matrix4, Vector3, Vector4};
use dashmap::DashMap;
use enum_map::{enum_map, EnumMap};
use itertools::Itertools;
use log::warn;
use perovskite_core::constants::{
    CHUNK_SIZE, CHUNK_SIZE_F64, CHUNK_SIZE_I32, CHUNK_SIZE_U8, CHUNK_VOLUME, PADDED_CHUNK_OFFSET,
    PADDED_CHUNK_SIZE, PADDED_CHUNK_VOLUME,
};
use perovskite_core::coordinates::{BlockCoordinate, ChunkOffset, ChunkOffsetForOcclusionExt};
use perovskite_core::protocol::game_rpc::{self as rpc_proto, MapDeltaUpdateBatch};
use perovskite_core::sync::DefaultSyncBackend;
use perovskite_core::vertical_occlusion::{ChunkColumn, OcclusionField, Occlusions};
use perovskite_core::{block_id::BlockId, coordinates::ChunkCoordinate};

use anyhow::{ensure, Context, Result};
use egui::ahash::HashMapExt;
use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use tracy_client::span;
use vulkano::buffer::Subbuffer;

use super::block_types::ClientBlockTypeManager;
use crate::client_state::ClientState;
use crate::vulkan::block_renderer::{
    BlockRenderer, VkChunkRaytraceData, VkChunkVertexDataCpu, VkChunkVertexDataGpu,
};
use crate::vulkan::raytrace_buffer::RaytraceBufferManager;
use crate::vulkan::shaders::cube_geometry::{
    CubeDrawStep, CubeGeometryDrawCall, CubeGeometryVertex,
};
use crate::vulkan::shaders::{VkBufferCpu, VkDrawBufferGpu};
use crate::vulkan::util::check_frustum;
use crate::vulkan::{BufferReclaim, ReclaimType, ReclaimableBuffer, TransferBuffer, VulkanContext};
use perovskite_core::protocol::map::ClientExtendedData;
use perovskite_core::util::AtomicInstant;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet, FxHasher};
use vulkano::DeviceSize;

pub(crate) trait ChunkDataView {
    fn is_empty_optimization_hint(&self) -> bool;
    fn block_ids(&self) -> &[BlockId; PADDED_CHUNK_VOLUME];
    fn lightmap(&self) -> &[u8; PADDED_CHUNK_VOLUME];
    // fn weather(&self) -> &bitvec::BitArr!(for PADDED_CHUNK_VOLUME);
    fn get_block(&self, offset: ChunkOffset) -> BlockId {
        self.block_ids()[offset.as_padded_index()]
    }
    #[allow(dead_code)]
    fn client_ext_data(&self, offset: ChunkOffset) -> Option<&ClientExtendedData>;
    fn raytrace_data(&self) -> Option<&VkChunkRaytraceData>;

    fn effective_rt_data(
        &self,
    ) -> Option<(&[u32; PADDED_CHUNK_VOLUME], &[u8; PADDED_CHUNK_VOLUME])> {
        let rt_blocks = if let Some(rt) = self.raytrace_data() {
            rt.blocks.as_deref()
        } else {
            return None;
        };
        let chunk = rt_blocks.unwrap_or_else(|| bytemuck::must_cast_ref(self.block_ids()));
        let lights = self.lightmap();
        Some((chunk, lights))
    }
}

pub(crate) struct LockedChunkDataView<'a>(RwLockReadGuard<'a, ChunkData>);

impl ChunkDataView for LockedChunkDataView<'_> {
    fn is_empty_optimization_hint(&self) -> bool {
        !(self
            .0
            .block_ids
            .as_ref()
            .is_some_and(|x| x.iter().any(|&v| v != BlockId(0))))
    }
    fn block_ids(&self) -> &[BlockId; PADDED_CHUNK_VOLUME] {
        self.0.block_ids.as_deref().unwrap_or(&ZERO_CHUNK)
    }

    fn lightmap(&self) -> &[u8; PADDED_CHUNK_VOLUME] {
        self.0.lightmap.as_deref().unwrap_or(&ZERO_LIGHTMAP)
    }

    // fn weather(&self) -> &bitvec::BitArr!(for PADDED_CHUNK_VOLUME) {
    //     self.0
    //         .weather
    //         .as_deref()
    //         .unwrap_or(&bitvec::array::BitArray::ZERO)
    // }

    fn client_ext_data(&self, offset: ChunkOffset) -> Option<&ClientExtendedData> {
        self.0.client_ext_data.get(&(offset.as_index() as u16))
    }

    fn raytrace_data(&self) -> Option<&VkChunkRaytraceData> {
        self.0.raytrace_data.as_ref()
    }
}

pub(crate) struct ChunkDataViewMut<'a>(RwLockWriteGuard<'a, ChunkData>);

pub(crate) fn hash_rt(
    data: Option<(&[u32; PADDED_CHUNK_VOLUME], &[u8; PADDED_CHUNK_VOLUME])>,
) -> (u64, u64) {
    let mut blocks_hasher = FxHasher::default();
    let mut lights_hasher = FxHasher::default();
    data.map(|x| x.0).hash(&mut blocks_hasher);
    data.map(|x| x.1).hash(&mut lights_hasher);
    (blocks_hasher.finish(), lights_hasher.finish())
}

impl<'a> ChunkDataViewMut<'a> {
    pub(crate) fn is_empty_optimization_hint(&self) -> bool {
        !self
            .0
            .block_ids
            .as_ref()
            .is_some_and(|x| x.iter().any(|&v| v != BlockId(0)))
    }

    pub(crate) fn block_ids(&self) -> &[BlockId; PADDED_CHUNK_VOLUME] {
        self.0.block_ids.as_deref().unwrap_or(&ZERO_CHUNK)
    }
    pub(crate) fn block_ids_mut(&mut self) -> Option<&mut [BlockId; PADDED_CHUNK_VOLUME]> {
        self.0.block_ids.as_deref_mut()
    }
    pub(crate) fn lightmap_mut(&mut self) -> &mut [u8; PADDED_CHUNK_VOLUME] {
        self.0.lightmap.get_or_insert_with(|| {
            log::warn!("Filling nonexisting lightmap in mutator; likely a bug");
            Box::new([0; PADDED_CHUNK_VOLUME])
        })
    }

    pub(crate) fn lightmap(&self) -> &[u8; PADDED_CHUNK_VOLUME] {
        self.0.lightmap.as_deref().unwrap_or(&ZERO_LIGHTMAP)
    }

    // TODO(#57): use this for raytracer decisions; needs serialization and space in the rt data structure so
    // unlikely in the first pass
    // #[allow(unused)]
    // pub(crate) fn weather(&self) -> &bitvec::BitArr!(for PADDED_CHUNK_VOLUME) {
    //     self.0
    //         .weather
    //         .as_deref()
    //         .unwrap_or(&bitvec::array::BitArray::ZERO)
    // }

    // pub(crate) fn weather_mut(&mut self) -> &mut bitvec::BitArr!(for PADDED_CHUNK_VOLUME) {
    //     self.0.weather.get_or_insert_with(|| {
    //         log::warn!("Filling nonexisting weather in mutator; likely a bug");
    //         Box::new(bitvec::array::BitArray::ZERO)
    //     })
    // }

    pub(crate) fn set_state(&mut self, state: ChunkRenderState) {
        self.0.render_state = state;
    }

    #[allow(dead_code)]
    pub(crate) fn get_block(&self, offset: ChunkOffset) -> BlockId {
        self.0
            .block_ids
            .as_ref()
            .map(|x| x[offset.as_padded_index()])
            .unwrap_or(BlockId(0))
    }
    #[allow(dead_code)]
    pub(crate) fn raytrace_data_mut(&mut self) -> Option<&mut VkChunkRaytraceData> {
        self.0.raytrace_data.as_mut()
    }
    pub(crate) fn downgrade(self) -> LockedChunkDataView<'a> {
        LockedChunkDataView(RwLockWriteGuard::downgrade(self.0))
    }

    fn effective_rt_data(
        &self,
    ) -> Option<(&[u32; PADDED_CHUNK_VOLUME], &[u8; PADDED_CHUNK_VOLUME])> {
        let rt_blocks = if let Some(rt) = &self.0.raytrace_data {
            rt.blocks.as_deref()
        } else {
            return None;
        };
        let chunk = rt_blocks.unwrap_or_else(|| bytemuck::must_cast_ref(self.block_ids()));
        let lights = self.lightmap();
        Some((chunk, lights))
    }
}

const ZERO_CHUNK: [BlockId; PADDED_CHUNK_VOLUME] = [BlockId(0); PADDED_CHUNK_VOLUME];
const ZERO_LIGHTMAP: [u8; PADDED_CHUNK_VOLUME] = [0; PADDED_CHUNK_VOLUME];

pub(crate) enum ChunkRenderState {
    /// We don't have this chunk's neighbors yet. It's not ready to render.
    NeedProcessing,
    /// We processed the chunk's neighbors, but we don't have anything to render.
    NoRender,

    /// We expect the chunk to have something to render, and we've prepared its neighbors.
    ReadyToRender,
    /// We don't expect the chunk to have anything to render, but the renderer should verify this and
    /// use a bright error texture + log an error message if it renders anything.
    #[allow(unused)]
    AuditNoRender,
}

/// Holds a cache of data that's used to actually turn a chunk into a mesh
pub(crate) struct ChunkData {
    /// Block_ids within this chunk *and* its neighbors.
    /// Future memory optimization - None if the chunk is all-air
    pub(crate) block_ids: Option<Box<[BlockId; PADDED_CHUNK_VOLUME]>>,
    /// The calculated lightmap for this chunk + immediate neighbor blocks. The data is encoded as follows:
    /// Bottom four bits: local lighting
    /// Top four bits: global lighting
    /// TODO assess whether to optimize for time or memory
    ///
    /// This is stored separate from the block_ids because it might be used for lighting non-chunk meshes, such as
    /// players, entities, etc.
    ///
    /// This can't be optimized with None since we need to propagate light through the chunk
    pub(crate) lightmap: Option<Box<[u8; PADDED_CHUNK_VOLUME]>>,
    // pub(crate) weather: Option<Box<bitvec::BitArr!(for PADDED_CHUNK_VOLUME)>>,
    render_state: ChunkRenderState,

    raytrace_data: Option<VkChunkRaytraceData>,
    // A hash ***which may collide*** over raytracing data. We're making a reasonable tradeoff that
    // in case of collision, a chunk will be stale until the next rebuild (or until another change
    // is detected via the hash)
    raytrace_hash: (u64, u64),
    client_ext_data: FxHashMap<u16, ClientExtendedData>,
}

pub(crate) const TARGET_BATCH_OCCUPANCY: usize = 128;

struct ChunkMesh {
    solo_cpu: Option<VkChunkVertexDataCpu>,
    solo_gpu: Option<VkChunkVertexDataGpu>,
    batch: Option<u64>,
}
impl Drop for ChunkMesh {
    fn drop(&mut self) {
        if let Some(cpu_data) = self.solo_cpu.take() {
            for (reclaimer, buffer) in RECLAIMERS.values().zip(cpu_data.draw_buffers.into_values())
            {
                if let Some(buffer) = buffer {
                    reclaimer.put(buffer.idx, buffer.vtx);
                }
            }
        }
    }
}

/// State of a chunk on the client
///
/// Data flow and locking are as follows:
/// * Chunk updates acquire a global write lock on the chunk data for all chunks
/// * Neighbor propagation (run after chunk updates) acquires a global read lock, a local write lock on the chunk data for the chunk
///     being processed, and local read locks on neighboring chunks as needed. This is the only layer that needs to take neigbor locks.
///     * Note that if there are multiple neighbor propagation workers (and hence multiple writers), then this data structure needs to be split up
///         further, with locks for input and output data. Otherwise, a deadlock would be possible where two workers have write locks and need
///         read locks on each other's write-locked chunk.
/// * Mesh generation acquires a local read lock on the chunk data. It doesn't need a recursive lock, and hence multiple mesh generator threads
///     shouldn't lead to write starvation.
///
/// To prevent deadlocks, the lock order is as follows:
/// * Chunk updates acquire the global chunk lock first
pub(crate) struct ClientChunk {
    coord: ChunkCoordinate,
    chunk_data: RwLock<ChunkData>,
    chunk_mesh: Mutex<ChunkMesh>,
    last_meshed: AtomicInstant,
    // Speedup hint only
    has_solo_hint: AtomicBool,
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub(crate) enum MeshResult {
    NewMesh(Option<u64>),
    SameMesh,
    EmptyMesh(Option<u64>),
}

// Used to reuse allocations from mesh vectors
pub(crate) struct MeshVectorReclaim {
    sender: crossbeam_channel::Sender<(Vec<u32>, Vec<CubeGeometryVertex>)>,
    receiver: crossbeam_channel::Receiver<(Vec<u32>, Vec<CubeGeometryVertex>)>,
}
impl MeshVectorReclaim {
    fn new(cap: usize) -> MeshVectorReclaim {
        let (sender, receiver) = crossbeam_channel::bounded(cap);
        MeshVectorReclaim { sender, receiver }
    }
    pub(crate) fn take(&self) -> Option<(Vec<u32>, Vec<CubeGeometryVertex>)> {
        match self.receiver.try_recv() {
            Ok((mut idx, mut vtx)) => {
                idx.clear();
                vtx.clear();
                Some((idx, vtx))
            }
            Err(_) => None,
        }
    }

    pub(crate) fn put(&self, idx: Vec<u32>, vtx: Vec<CubeGeometryVertex>) {
        // Drop - if we don't have space, just drop it
        drop(self.sender.try_send((idx, vtx)));
    }
}

// Consider moving these from static into actual owners
// Or maybe not, this is really an extension of the allocator (which is global), so there's an
// argument for making these global as well. In any case they need to be lock-free MPMC anyway for
// concurrency reasons, so shared references are OK anyway
lazy_static::lazy_static! {
    pub(crate) static ref RECLAIMERS: EnumMap<CubeDrawStep, MeshVectorReclaim> = enum_map! {
        _ => MeshVectorReclaim::new(4096),
    };
}

impl ClientChunk {
    pub(crate) fn from_proto(
        coord: ChunkCoordinate,
        block_ids: &[u32; CHUNK_VOLUME],
        ced: Vec<ClientExtendedData>,
        block_types: &ClientBlockTypeManager,
    ) -> Result<(ClientChunk, OcclusionField, OcclusionField)> {
        let (light_occlusion, weather_occlusion) = get_occlusion_for_proto(block_ids, block_types);
        let block_ids = Self::expand_ids(block_ids);
        let lightmap = if block_ids.is_some() {
            Some(bytemuck::zeroed_box())
        } else {
            None
        };
        // let weather = if block_ids.is_some() {
        //     Some(Box::new(bitvec::array::BitArray::ZERO))
        // } else {
        //     None
        // };

        let mut client_ext_data: FxHashMap<u16, ClientExtendedData> =
            FxHashMap::with_capacity(ced.len());
        for val in ced {
            client_ext_data.insert(val.offset_in_chunk as u16, val);
        }
        Ok((
            ClientChunk {
                coord,
                chunk_data: RwLock::new(ChunkData {
                    block_ids,
                    render_state: ChunkRenderState::NeedProcessing,
                    lightmap,
                    // weather,
                    client_ext_data,
                    raytrace_data: None,
                    raytrace_hash: hash_rt(None),
                }),
                chunk_mesh: Mutex::new(ChunkMesh {
                    solo_cpu: None,
                    solo_gpu: None,
                    batch: None,
                }),
                last_meshed: AtomicInstant::new(),
                has_solo_hint: AtomicBool::new(false),
            },
            light_occlusion,
            weather_occlusion,
        ))
    }

    pub(crate) fn last_meshed(&self) -> Instant {
        // Relaxed since this is used just for approximate hints
        self.last_meshed.get_relaxed()
    }

    pub(crate) fn mesh_with(
        &self,
        renderer: &BlockRenderer,
        raytracer: Option<&RaytraceBufferManager>,
    ) -> Result<MeshResult> {
        let _span = span!("mesh_with");
        let mut data = self.chunk_data_mut();
        let new_rt_data = renderer.build_raytrace_data(data.block_ids());
        let old_hash = data.0.raytrace_hash;
        data.0.raytrace_data = new_rt_data;
        data.0.raytrace_hash = hash_rt(data.effective_rt_data());

        let data = data.downgrade();
        let raster_result = self.mesh_raster_with(renderer, &data)?;

        if let Some(rt) = raytracer {
            let rt_data = data.effective_rt_data();
            if let Some(rt_data) = rt_data {
                let blocks = if old_hash.0 != data.0.raytrace_hash.0 {
                    Some(rt_data.0)
                } else {
                    None
                };

                let lights = if old_hash.1 != data.0.raytrace_hash.1 {
                    Some(rt_data.1)
                } else {
                    None
                };
                rt.push_chunk(self.coord, blocks, lights)?
            } else {
                // TODO: push empty chunks
            }
        }
        Ok(raster_result)
    }

    pub(crate) fn invalidate_mesh(&self) {
        let mut mesh = self.chunk_mesh.lock();
        if let Some(cpu_data) = mesh.solo_cpu.take() {
            for (reclaimer, buffer) in RECLAIMERS.values().zip(cpu_data.draw_buffers.into_values())
            {
                if let Some(buffer) = buffer {
                    reclaimer.put(buffer.idx, buffer.vtx);
                }
            }
        }
        mesh.solo_cpu = None;
        mesh.solo_gpu = None;
        mesh.batch = None;
    }

    fn mesh_raster_with(
        &self,
        renderer: &BlockRenderer,
        data: &LockedChunkDataView,
    ) -> Result<MeshResult> {
        let vertex_data = match data.0.render_state {
            ChunkRenderState::NeedProcessing => Some(renderer.mesh_chunk(&data)?),
            ChunkRenderState::NoRender => None,
            ChunkRenderState::ReadyToRender => Some(renderer.mesh_chunk(&data)?),
            ChunkRenderState::AuditNoRender => {
                let result = renderer.mesh_chunk(&data)?;
                if result.draw_buffers.values().any(|x| x.is_some()) {
                    log::warn!("Failed no-render audit for {:?}", self.coord);
                }
                Some(result)
            }
        };
        // The ordering doesn't matter; we're doing this under the lock
        let mut mesh_lock = self.chunk_mesh.lock();
        self.last_meshed.update_now_relaxed();
        let old_batch = mesh_lock.batch;
        if let Some(vertex_data) = vertex_data {
            if Some(&vertex_data) == mesh_lock.solo_cpu.as_ref() {
                Ok(MeshResult::SameMesh)
            } else {
                *mesh_lock = ChunkMesh {
                    solo_gpu: Some(vertex_data.to_gpu(renderer.clone_vk_allocator())?),
                    solo_cpu: Some(vertex_data),
                    batch: None,
                };
                self.has_solo_hint.store(true, Ordering::Relaxed);
                Ok(MeshResult::NewMesh(old_batch))
            }
        } else {
            *mesh_lock = ChunkMesh {
                solo_cpu: None,
                solo_gpu: None,
                batch: None,
            };
            self.has_solo_hint.store(false, Ordering::Relaxed);
            Ok(MeshResult::EmptyMesh(old_batch))
        }
    }

    fn expand_ids(ids: &[u32; CHUNK_VOLUME]) -> Option<Box<[BlockId; PADDED_CHUNK_VOLUME]>> {
        if ids.iter().all(|&x| x == 0) {
            return None;
        }
        let mut result: Box<[BlockId; PADDED_CHUNK_VOLUME]> = bytemuck::zeroed_box();
        for i in 0..CHUNK_SIZE {
            for j in 0..CHUNK_SIZE {
                for k in 0..CHUNK_SIZE {
                    result[(i + 1) * PADDED_CHUNK_SIZE * PADDED_CHUNK_SIZE
                        + (j + 1) * PADDED_CHUNK_SIZE
                        + (k + 1)] =
                        BlockId::from(ids[i * CHUNK_SIZE * CHUNK_SIZE + j * CHUNK_SIZE + k]);
                }
            }
        }
        Some(result)
    }

    pub(crate) fn data_for_batch(&self) -> Option<VkChunkVertexDataCpu> {
        let lock = self.chunk_mesh.lock();
        if lock.batch.is_some() {
            // Already in a batch
            return None;
        }
        lock.solo_cpu.clone()
    }

    pub(crate) fn set_batch(&self, id: u64) {
        self.has_solo_hint.store(false, Ordering::Relaxed);
        self.last_meshed.update_now_relaxed();
        let mut lock = self.chunk_mesh.lock();
        lock.batch = Some(id);
    }

    pub(crate) fn get_batch(&self) -> Option<u64> {
        let _span = span!("chunk_mesh lock get_batch");
        self.chunk_mesh.lock().batch
    }

    pub(crate) fn spill_back_to_solo(&self, expecting: u64) -> Option<u64> {
        let _span = span!("chunk_mesh lock spill_back_to_solo");
        self.has_solo_hint.store(true, Ordering::Relaxed);
        let mut lock = self.chunk_mesh.lock();
        let current_batch = lock.batch.take();
        if current_batch.is_some_and(|x| x != expecting) {
            log::error!(
                "Mismatched batch ID in spill_back_to_solo for {:?}: expected {}, got {:?}",
                self.coord,
                expecting,
                current_batch,
            );
        }
        self.last_meshed.update_now_relaxed();
        lock.batch.take()
    }

    pub(crate) fn update_from(
        &self,
        coord: ChunkCoordinate,
        block_ids: &[u32; CHUNK_VOLUME],
        ced: Vec<ClientExtendedData>,
        block_types: &ClientBlockTypeManager,
    ) -> Result<(OcclusionField, OcclusionField)> {
        ensure!(coord == self.coord);
        let (light_occlusion, weather_occlusion) = get_occlusion_for_proto(block_ids, block_types);
        let ids = Self::expand_ids(block_ids);
        let mut client_ext_data = FxHashMap::with_capacity(ced.len());
        for val in ced {
            client_ext_data.insert(val.offset_in_chunk as u16, val);
        }
        let mut data_guard = self.chunk_data.write();
        data_guard.block_ids = ids;
        data_guard.render_state = ChunkRenderState::NeedProcessing;
        data_guard.client_ext_data = client_ext_data;
        Ok((light_occlusion, weather_occlusion))
    }

    pub(crate) fn apply_delta(&self, proto: rpc_proto::MapDeltaUpdate) -> Result<bool> {
        let block_coord: BlockCoordinate = proto
            .block_coord
            .clone()
            .with_context(|| "block_coord missing in MapDeltaUpdate")?
            .into();
        ensure!(block_coord.chunk() == self.coord);

        let mut chunk_data = self.chunk_data.write();
        if chunk_data.block_ids.is_none() {
            chunk_data.block_ids = Some(bytemuck::zeroed_box());
        }
        let old_id = chunk_data.block_ids.as_mut().unwrap()[block_coord.offset().as_padded_index()];
        let new_id = BlockId::from(proto.new_id);
        // TODO future optimization: check whether a change in variant actually requires
        // a redraw
        if old_id != new_id {
            chunk_data.render_state = ChunkRenderState::NeedProcessing;
            chunk_data.block_ids.as_mut().unwrap()[block_coord.offset().as_padded_index()] = new_id;
        }
        match proto.new_client_ext_data {
            None => {
                chunk_data
                    .client_ext_data
                    .remove(&(block_coord.offset().as_index() as u16));
            }
            Some(x) => {
                chunk_data
                    .client_ext_data
                    .insert(block_coord.offset().as_index() as u16, x);
            }
        }
        Ok(old_id != new_id)
    }

    pub(crate) fn make_draw_call(
        &self,
        coord: ChunkCoordinate,
        player_position: Vector3<f64>,
        view_proj_matrix: Matrix4<f32>,
    ) -> Option<CubeGeometryDrawCall> {
        // We calculate the transform in f64, and *then* narrow to f32
        // Otherwise, we get catastrophic cancellation when far from the origin

        // Relative origin in player-relative coordinates, where the player is at 0,0,0 and
        // the X/Y/Z axes are aligned with the global axes (i.e. NOT rotated)
        //
        // player_position is in game coordinates (+Y up) and the result should be in Vulkan
        // coordinates (+Y down)

        // For some reason, we save 200 usec per frame (on AlderLake-P) when we pass coord as a parameter.
        // It seems like the cache miss to get self.coord loaded is painfully slow
        let relative_origin = (Vector3::new(
            CHUNK_SIZE_F64 * coord.x as f64,
            CHUNK_SIZE_F64 * coord.y as f64,
            CHUNK_SIZE_F64 * coord.z as f64,
        ) - player_position)
            .mul_element_wise(Vector3::new(1., -1., 1.));
        let translation = Matrix4::from_translation(relative_origin.cast().unwrap());
        if check_frustum(view_proj_matrix * translation, CHUNK_CORNERS) {
            let lock = self.chunk_mesh.lock();
            lock.solo_gpu.as_ref().map(|solo_gpu| CubeGeometryDrawCall {
                models: solo_gpu.clone(),
                model_matrix: translation,
            })
        } else {
            None
        }
    }

    pub(crate) fn get_occlusions(
        &self,
        block_types: &ClientBlockTypeManager,
    ) -> (OcclusionField, OcclusionField) {
        let data = self.chunk_data.read();
        let mut light = OcclusionField::zero();
        let mut weather = OcclusionField::zero();
        for x in 0..CHUNK_SIZE_U8 {
            for z in 0..CHUNK_SIZE_U8 {
                'inner: for y in 0..CHUNK_SIZE_U8 {
                    let id = data
                        .block_ids
                        .as_ref()
                        .map(|ids| ids[(ChunkOffset { x, y, z }).as_padded_index()])
                        .unwrap_or(BlockId(0));
                    let blocks_light = block_types.propagates_light(id);
                    let blocks_weather = block_types.propagates_weather(id);
                    if !blocks_light {
                        light.set(x, z, true);
                    }
                    if !blocks_weather {
                        weather.set(x, z, true);
                    }
                    // fast path: we expect many blocks to block both. We can accept some
                    // inefficiency for the rarer blocks that block only one.
                    if blocks_light && blocks_weather {
                        break 'inner;
                    }
                }
            }
        }
        (light, weather)
    }

    pub(crate) fn get_single(&self, offset: ChunkOffset) -> BlockId {
        self.chunk_data
            .read()
            .block_ids
            .as_deref()
            .map(|x| x[offset.as_padded_index()])
            .unwrap_or(BlockId(0))
    }
    pub(crate) fn get_single_with_extended_data(
        &self,
        offset: ChunkOffset,
    ) -> (BlockId, Option<ClientExtendedData>) {
        let id = self
            .chunk_data
            .read()
            .block_ids
            .as_deref()
            .map(|x| x[offset.as_padded_index()])
            .unwrap_or(BlockId(0));
        let ext_data = self
            .chunk_data
            .read()
            .client_ext_data
            .get(&(offset.as_index() as u16))
            .cloned();
        (id, ext_data)
    }

    pub(crate) fn chunk_data(&self) -> LockedChunkDataView<'_> {
        let _span = span!("chunk_data");
        LockedChunkDataView(self.chunk_data.read())
    }

    pub(crate) fn chunk_data_mut(&self) -> ChunkDataViewMut<'_> {
        let _span = span!("chunk_data_mut");
        ChunkDataViewMut(self.chunk_data.write())
    }
}
// todo fix jank that requires this to be 17.0 rather than 16.0
const CHUNK_WITH_JANK: f32 = CHUNK_SIZE_F64 as f32 + 1.0;
const CHUNK_CORNERS: [Vector4<f32>; 8] = [
    vec4(-1.0, 1.0, -1.0, 1.),
    vec4(CHUNK_WITH_JANK, 1.0, -1.0, 1.),
    vec4(-1.0, -CHUNK_WITH_JANK, -1.0, 1.),
    vec4(CHUNK_WITH_JANK, -CHUNK_WITH_JANK, -1.0, 1.),
    vec4(-1.0, 1.0, CHUNK_WITH_JANK, 1.),
    vec4(CHUNK_WITH_JANK, 1.0, CHUNK_WITH_JANK, 1.),
    vec4(-1.0, -CHUNK_WITH_JANK, CHUNK_WITH_JANK, 1.),
    vec4(CHUNK_WITH_JANK, -CHUNK_WITH_JANK, CHUNK_WITH_JANK, 1.),
];

fn get_occlusion_for_proto(
    block_ids: &[u32; CHUNK_VOLUME],
    block_types: &ClientBlockTypeManager,
) -> (OcclusionField, OcclusionField) {
    let mut light_occlusion = OcclusionField::zero();
    let mut weather_occlusion = OcclusionField::zero();
    for x in 0..CHUNK_SIZE_U8 {
        for z in 0..CHUNK_SIZE_U8 {
            'inner: for y in 0..CHUNK_SIZE_U8 {
                let id = block_ids[(ChunkOffset { x, y, z }).as_index()];
                if !block_types.propagates_light(id.into()) {
                    light_occlusion.set(x, z, true);
                    break 'inner;
                }
            }
            'weather_inner: for y in 0..CHUNK_SIZE_U8 {
                let id = block_ids[(ChunkOffset { x, y, z }).as_index()];
                if !block_types.propagates_weather(id.into()) {
                    weather_occlusion.set(x, z, true);
                    break 'weather_inner;
                }
            }
        }
    }
    (light_occlusion, weather_occlusion)
}

static NEXT_MESH_BATCH_ID: AtomicUsize = AtomicUsize::new(0);

pub(crate) struct MeshBatch {
    id: u64,
    vertex_buffers: EnumMap<CubeDrawStep, Option<ReclaimableBuffer<CubeGeometryVertex>>>,
    index_buffers: EnumMap<CubeDrawStep, Option<ReclaimableBuffer<u32>>>,

    chunks: smallvec::SmallVec<[ChunkCoordinate; TARGET_BATCH_OCCUPANCY]>,
    base_position: Vector3<f64>,
}

impl MeshBatch {
    pub(crate) fn solid_occupancy(&self) -> (usize, usize) {
        (
            self.vertex_buffers[CubeDrawStep::OpaqueSimple]
                .as_deref()
                .map(Subbuffer::len)
                .unwrap_or(0) as usize,
            self.index_buffers[CubeDrawStep::OpaqueSimple]
                .as_deref()
                .map(Subbuffer::len)
                .unwrap_or(0) as usize,
        )
    }
}

impl MeshBatch {
    pub(crate) fn coords(&self) -> &[ChunkCoordinate] {
        &self.chunks
    }
    pub(crate) fn make_draw_call(
        &self,
        player_position: Vector3<f64>,
        view_proj_matrix: Matrix4<f32>,
    ) -> Option<CubeGeometryDrawCall> {
        let matrix = Matrix4::from_translation(
            (self.base_position - player_position).mul_element_wise(vec3(1.0, -1.0, 1.0)),
        );

        let mut any_frustum_pass = false;
        for coord in &self.chunks {
            let relative_origin = (Vector3::new(
                CHUNK_SIZE_F64 * coord.x as f64,
                CHUNK_SIZE_F64 * coord.y as f64,
                CHUNK_SIZE_F64 * coord.z as f64,
            ) - player_position)
                .mul_element_wise(Vector3::new(1., -1., 1.));
            let translation = Matrix4::from_translation(relative_origin.cast().unwrap());
            if check_frustum(view_proj_matrix * translation, CHUNK_CORNERS) {
                any_frustum_pass = true;
            }
        }

        if !any_frustum_pass {
            return None;
        }

        Some(CubeGeometryDrawCall {
            models: VkChunkVertexDataGpu {
                draw_buffers: enum_map! {
                    step => {
                        if self.vertex_buffers[step].is_some() && self.index_buffers[step].is_some() {
                            Some(VkDrawBufferGpu {
                                num_indices: self.index_buffers[step].as_ref().unwrap().valid_len() as u32,
                                vtx: self.vertex_buffers[step].as_deref().unwrap().clone(),
                                idx: self.index_buffers[step].as_deref().unwrap().clone(),
                            })
                        } else {
                            None
                        }
                    }
                },
            },
            model_matrix: matrix.cast().unwrap(),
        })
    }
    pub fn id(&self) -> u64 {
        self.id
    }
}

pub(crate) struct MeshBatchBuilder {
    id: u64,
    vertex_buffers: EnumMap<CubeDrawStep, Vec<CubeGeometryVertex>>,
    index_buffers: EnumMap<CubeDrawStep, Vec<u32>>,
    base_position: Option<Vector3<f64>>,
    chunks: smallvec::SmallVec<[ChunkCoordinate; TARGET_BATCH_OCCUPANCY]>,
}
impl MeshBatchBuilder {
    pub(crate) fn new() -> MeshBatchBuilder {
        MeshBatchBuilder {
            id: next_batch_id(),
            vertex_buffers: enum_map! {
                _ => vec![]
            },
            index_buffers: enum_map! {
                _ => vec![]
            },
            base_position: None,
            chunks: smallvec::SmallVec::new(),
        }
    }
    pub(crate) fn occupancy(&self) -> usize {
        self.chunks.len()
    }

    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn chunks(&self) -> &[ChunkCoordinate] {
        &self.chunks
    }

    fn extend_buffer(
        input: &VkBufferCpu<CubeGeometryVertex>,
        vtx: &mut Vec<CubeGeometryVertex>,
        idx: &mut Vec<u32>,
        delta_offset: Vector3<f32>,
    ) {
        let base_index = vtx.len();
        vtx.extend(input.vtx.iter().map(|v| CubeGeometryVertex {
            position: [
                v.position[0] + delta_offset.x,
                v.position[1] - delta_offset.y,
                v.position[2] + delta_offset.z,
            ],
            ..*v
        }));
        idx.extend(input.idx.iter().map(|idx| idx + base_index as u32));
    }

    pub(crate) fn append(&mut self, coord: ChunkCoordinate, cpu: &VkChunkVertexDataCpu) {
        let _span = span!("batch_append");
        let chunk_pos = vec3(
            coord.x as f64 * CHUNK_SIZE_F64,
            coord.y as f64 * CHUNK_SIZE_F64,
            coord.z as f64 * CHUNK_SIZE_F64,
        );
        let base_position = *self.base_position.get_or_insert_with(|| chunk_pos);
        for (step, buffer) in cpu.draw_buffers.iter() {
            if let Some(buffer) = buffer.as_ref() {
                Self::extend_buffer(
                    buffer,
                    &mut self.vertex_buffers[step],
                    &mut self.index_buffers[step],
                    (chunk_pos - base_position).cast().unwrap(),
                );
            }
        }
        self.chunks.push(coord);
    }

    pub(crate) fn build_and_reset(
        &mut self,
        ctx: &VulkanContext,
    ) -> Result<(MeshBatch, TransferBuffer)> {
        let mut any_commands = false;

        let mut cmdbuf = ctx.start_transfer_buffer()?;
        let mut process_vtx = |x: &[CubeGeometryVertex]| -> anyhow::Result<_> {
            if x.is_empty() {
                Ok(None)
            } else {
                let len = x.len();
                let target_buffer = ctx.iter_to_device_via_staging_with_reclaim(
                    x.iter().copied(),
                    ReclaimType::GpuVtxTransferDst,
                    ctx.cgv_reclaimer().clone(),
                    BufferReclaim::<CubeGeometryVertex>::size_class(len as DeviceSize),
                    &mut cmdbuf,
                )?;
                any_commands = true;
                Ok(Some(target_buffer))
            }
        };
        let mut vertex_buffers = enum_map! { _ => None};
        for (step, buf) in self.vertex_buffers.iter() {
            vertex_buffers[step] = process_vtx(buf)?
        }

        let mut process_idx = |x: &[u32]| -> anyhow::Result<_> {
            if x.is_empty() {
                Ok(None)
            } else {
                let len = x.len();
                let target_buffer = ctx.iter_to_device_via_staging_with_reclaim(
                    x.iter().copied(),
                    ReclaimType::GpuIdxTransferDst,
                    ctx.u32_reclaimer().clone(),
                    BufferReclaim::<u32>::size_class(len as DeviceSize),
                    &mut cmdbuf,
                )?;
                any_commands = true;
                Ok(Some(target_buffer))
            }
        };

        let mut index_buffers = enum_map! { _ => None};
        for (step, buf) in self.index_buffers.iter() {
            index_buffers[step] = process_idx(buf)?
        }

        let result = MeshBatch {
            id: self.id,
            vertex_buffers,
            index_buffers,
            base_position: self
                .base_position
                .expect("Expected base position; called build_and_reset on an empty batch"),
            chunks: self.chunks.clone(),
        };

        self.reset();
        Ok((result, cmdbuf))
    }

    pub(crate) fn reset(&mut self) {
        for buf in self.vertex_buffers.values_mut() {
            buf.clear();
        }
        for buf in self.index_buffers.values_mut() {
            buf.clear();
        }

        self.chunks.clear();
        self.base_position = None;
        self.id = next_batch_id();
    }
}

fn next_batch_id() -> u64 {
    NEXT_MESH_BATCH_ID.fetch_add(1, Ordering::Relaxed) as u64
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

    pub(crate) fn get_chunk(
        &self,
        coord: ChunkCoordinate,
    ) -> Option<dashmap::mapref::one::Ref<'_, ChunkCoordinate, Arc<ClientChunk>>> {
        self.chunks.get(&coord)
    }

    /// Clones the chunk manager and returns a clone with the following properties:
    /// * The clone does not hold any locks on the data in this chunk manager (i.e. insertions and deletions
    ///   are possible while the cloned view is live).
    /// * The clone does not track any insertions/deletions of this chunk manager.
    ///    * It will not show chunks inserted after this function returns.
    ///    * If chunks are deleted after cloned_view returned, they will remain in the cloned view, and their
    ///      memory will not be released until the cloned view is dropped, due to Arcs owned by the cloned view.
    pub(crate) fn renderable_chunks_cloned_view<T>(&self) -> T
    where
        T: FromIterator<(ChunkCoordinate, Arc<ClientChunk>)>,
    {
        self.renderable_chunks
            .iter()
            .map(|x| (x.key().clone(), x.value().clone()))
            .collect()
    }

    fn extend_from_renderable_chunks(
        &self,
        dst: &mut impl Extend<(ChunkCoordinate, Arc<ClientChunk>)>,
        key_filter: impl Fn(&ChunkCoordinate) -> bool,
    ) {
        dst.extend(
            self.renderable_chunks
                .iter()
                .filter(|ref_item| key_filter(ref_item.key()))
                .map(|x| (x.key().clone(), x.value().clone())),
        )
    }

    /// Returns (loaded chunks, renderable chunks). Renderable chunks are loaded chunks that have
    /// been meshed and contain something to draw.
    pub(crate) fn chunk_counts(&self) -> (usize, usize) {
        (self.chunks.len(), self.renderable_chunks.len())
    }

    /// Returns (loaded, total) for the chunks within `radius` chunks (in each axis) of `center`.
    pub(crate) fn loaded_within(&self, center: ChunkCoordinate, radius: i32) -> (usize, usize) {
        let mut loaded = 0;
        let mut total = 0;
        for dx in -radius..=radius {
            for dy in -radius..=radius {
                for dz in -radius..=radius {
                    if let Some(coord) = center.try_delta(dx, dy, dz) {
                        total += 1;
                        if self.chunks.contains_key(&coord) {
                            loaded += 1;
                        }
                    }
                }
            }
        }
        (loaded, total)
    }

    /// Removes the chunk from the renderable chunks, but not from the chunks map.
    /// This is used when the chunk is no longer renderable, but is still loaded (e.g. an all-air chunk).
    pub(crate) fn invalidate_mesh(&self, coord: &ChunkCoordinate) {
        let _span = span!("invalidate_mesh");
        let chunk = self.renderable_chunks.remove(coord).map(|x| x.1);
        if let Some(chunk) = chunk {
            let batch = chunk.get_batch();
            if let Some(batch) = batch {
                self.spill(batch, "invalidate_mesh");
            }
            chunk.invalidate_mesh();
        }
    }

    pub(crate) fn remove(&self, coord: &ChunkCoordinate) -> Option<Arc<ClientChunk>> {
        let _span = span!("chunk remove");
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
                // TODO: this carries a race/ordering risk, doesn't it? IIUC we only get lucky because
                // the network routines run on a single async task without issuing inserts/removes in parallel
                drop(column);
                self.light_columns.remove(&(coord.x, coord.z));
            }
        }
        self.renderable_chunks.remove(coord);
        chunk.map(|x| x.1)
    }

    async fn handle_mapchunk_audio(
        &self,
        client_state: &ClientState,
        coord: ChunkCoordinate,
        block_ids: &[u32; CHUNK_VOLUME],
    ) {
        let tick = client_state.timekeeper.now();
        let pos = client_state.weakly_ordered_last_position();

        let mut sounds: smallvec::SmallVec<[_; 16]> = smallvec::SmallVec::new();
        for (i, block) in block_ids.iter().enumerate() {
            if let Some((id, volume)) = client_state.block_types.block_sound(BlockId::from(*block))
            {
                sounds.push((i, id, volume));
            }
        }

        let mut map_sound = client_state.world_audio.lock().await;
        map_sound.remove_chunk(coord);
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

    async fn handle_mapchunk_text(
        &self,
        client_state: &ClientState,
        coord: ChunkCoordinate,
        ced: &[ClientExtendedData],
    ) {
        client_state.text_renderer.remove_chunk(coord).await;
        for ced in ced {
            if ced.rendered_text.is_empty() {
                continue;
            }
            client_state
                .text_renderer
                .insert_or_update(
                    coord.with_offset(ChunkOffset::from_index(ced.offset_in_chunk as usize)),
                    ced.rendered_text.clone(),
                )
                .await;
        }
    }

    // Returns the number of additional chunks below the given chunk that need lighting updates.
    pub(crate) async fn insert_or_update(
        &self,
        client_state: &ClientState,
        coord: ChunkCoordinate,
        block_ids: &[u32; CHUNK_VOLUME],
        ced: Vec<ClientExtendedData>,
        block_types: &ClientBlockTypeManager,
    ) -> Result<usize> {
        self.handle_mapchunk_text(client_state, coord, &ced).await;
        let mut light_column = self
            .light_columns
            .entry((coord.x, coord.z))
            .or_insert_with(ChunkColumn::empty);
        let (light_occlusion, weather_occlusion) = match self.chunks.entry(coord) {
            dashmap::Entry::Occupied(chunk_entry) => {
                client_state.world_audio.lock().await.remove_chunk(coord);
                tokio::task::block_in_place(|| {
                    chunk_entry
                        .get()
                        .update_from(coord, block_ids, ced, block_types)
                })?
            }
            dashmap::Entry::Vacant(x) => {
                let (chunk, light_occlusion, weather_occlusion) =
                    ClientChunk::from_proto(coord, block_ids, ced, block_types)?;
                light_column.value_mut().insert_empty(coord.y);
                x.insert(Arc::new(chunk));
                (light_occlusion, weather_occlusion)
            }
        };
        self.handle_mapchunk_audio(client_state, coord, block_ids)
            .await;

        let mut occ_cursor = light_column.cursor_into(coord.y);
        *occ_cursor.current_light_occlusion_mut() = light_occlusion;
        *occ_cursor.current_weather_occlusion_mut() = weather_occlusion;
        occ_cursor.mark_valid();
        let extra_chunks = occ_cursor.propagate_occlusion();
        Ok(extra_chunks)
    }

    pub(crate) fn apply_delta_batch(
        &self,
        batch: &mut MapDeltaUpdateBatch,
        block_types: &ClientBlockTypeManager,
    ) -> Result<(
        FxHashSet<ChunkCoordinate>,
        FxHashSet<ChunkCoordinate>,
        Vec<BlockCoordinate>,
        bool,
    )> {
        let mut needs_remesh = FxHashSet::default();
        let mut priority_remesh = FxHashSet::default();
        let mut unknown_coords = vec![];
        let mut missing_coord = false;

        for update in batch.updates.drain(..) {
            let Some(block_coord) = update.block_coord else {
                warn!("Got delta with missing block_coord {:?}", update);
                missing_coord = true;
                continue;
            };
            let chunk_coord = block_coord.chunk();
            let extra_chunks = match self.chunks.get(&chunk_coord) {
                Some(x) => {
                    if !x.apply_delta(update)? {
                        None
                    } else {
                        priority_remesh.insert(chunk_coord);
                        let (light_occlusion, weather_occlusion) = x.get_occlusions(block_types);
                        let light_column = self
                            .light_columns
                            .get(&(chunk_coord.x, chunk_coord.z))
                            .expect("Missing light column during delta update");
                        let mut light_cursor = light_column.cursor_into(chunk_coord.y);

                        *light_cursor.current_light_occlusion_mut() = light_occlusion;
                        *light_cursor.current_weather_occlusion_mut() = weather_occlusion;
                        light_cursor.mark_valid();
                        let extra_chunks = light_cursor.propagate_occlusion();
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
        Ok((needs_remesh, priority_remesh, unknown_coords, missing_coord))
    }

    pub(crate) fn cloned_neighbors_fast(
        &self,
        chunk: ChunkCoordinate,
        result: &mut FastChunkNeighbors,
    ) {
        let _span = span!("cloned_neighbors_fast");

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
                            let Occlusions { light, weather, .. } = light_column
                                .get_incoming_light_and_weather(delta.y)
                                .unwrap_or(Occlusions::zero());
                            result.inbound_lights[(k + 1) as usize][(j + 1) as usize]
                                [(i + 1) as usize] = light;
                            result.inbound_weather[(k + 1) as usize][(j + 1) as usize]
                                [(i + 1) as usize] = weather;
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
        const HALF_CHUNK: f64 = CHUNK_SIZE_F64 / 2.0;
        keys.sort_unstable_by_key(|coord| {
            let world_coord = vec3(
                coord.x as f64 * CHUNK_SIZE_F64 + HALF_CHUNK,
                coord.y as f64 * CHUNK_SIZE_F64 + HALF_CHUNK,
                coord.z as f64 * CHUNK_SIZE_F64 + HALF_CHUNK,
            );
            -1 * ((player_pos - world_coord).magnitude2() as i64)
        });
        let mut needs_resort = true;

        'outer: while let Some(coord) = keys.pop() {
            let world_coord = vec3(
                coord.x as f64 * CHUNK_SIZE_F64 + HALF_CHUNK,
                coord.y as f64 * CHUNK_SIZE_F64 + HALF_CHUNK,
                coord.z as f64 * CHUNK_SIZE_F64 + HALF_CHUNK,
            );
            if (player_pos - world_coord).magnitude2() > 900.0 {
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
                                next_coord.x as f64 * CHUNK_SIZE_F64 + HALF_CHUNK,
                                next_coord.y as f64 * CHUNK_SIZE_F64 + HALF_CHUNK,
                                next_coord.z as f64 * CHUNK_SIZE_F64 + HALF_CHUNK,
                            );
                            let current_coord_f64 = vec3(
                                coord.x as f64 * CHUNK_SIZE_F64 + HALF_CHUNK,
                                coord.y as f64 * CHUNK_SIZE_F64 + HALF_CHUNK,
                                coord.z as f64 * CHUNK_SIZE_F64 + HALF_CHUNK,
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
                            let (new_batch, transfer_buffer) = batches.1.build_and_reset(vk_ctx)?;
                            parking_lot::MutexGuard::unlocked(&mut batches, || {
                                vk_ctx.finish_transfer_buffer(transfer_buffer)
                            })?;
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

pub(crate) struct Minimap {
    /// The X/Z coordinate of the bottom-left-most column in this minimap, given
    /// as blocks, not chunks. However, for now it's chunk-aligned, i.e. divisible by
    /// [CHUNK_SIZE]
    pub(crate) min_coord: (i32, i32),
    /// The actual image holding pixel data.
    pub(crate) image: image::RgbaImage,
    /// The scale factor, in pixels per block. Zooming in: greater values. Currently, this
    /// is fixed at 1.0 for the initial revision.
    pub(crate) scale: f32,
}

pub(crate) struct MinimapScratchpad {
    /// Stores block data, resized up on demand and keeps its memory allocation over the lifetime
    /// of the scratchpad
    blocks: Vec<BlockId>,
    chunks: Vec<(ChunkCoordinate, Arc<ClientChunk>)>,
}
impl Default for MinimapScratchpad {
    fn default() -> Self {
        Self {
            blocks: Vec::new(),
            chunks: Vec::new(),
        }
    }
}

impl ChunkManager {
    pub(crate) fn new_minimap(
        &self,
        min_coord: (i32, i32),
        extent: (i32, i32),
        scratchpad: &mut MinimapScratchpad,
        types: &ClientBlockTypeManager,
    ) -> Minimap {
        let min_x_c = min_coord.0.div_euclid(CHUNK_SIZE_I32);
        let min_z_c = min_coord.1.div_euclid(CHUNK_SIZE_I32);
        let max_x_c = (min_coord.0 + extent.0).div_euclid(CHUNK_SIZE_I32);
        let max_z_c = (min_coord.1 + extent.1).div_euclid(CHUNK_SIZE_I32);

        let real_extent_x = (max_x_c - min_x_c + 1) * CHUNK_SIZE_I32;
        let real_extent_z = (max_z_c - min_z_c + 1) * CHUNK_SIZE_I32;

        scratchpad.blocks.clear();
        scratchpad.blocks.resize(
            real_extent_x as usize * real_extent_z as usize,
            BlockId::AIR,
        );

        // don't reserve for chunks because we don't really know how many we really need:
        // some X/Z values may be empty, and the vertical height of each slice is not known
        // _a priori_.
        scratchpad.chunks.clear();

        self.extend_from_renderable_chunks(&mut scratchpad.chunks, |c| {
            (c.x >= min_x_c && c.x <= max_x_c) && (c.z >= min_z_c && c.z <= max_z_c)
        });

        // bitwise not, to sort by y descending without an overflow issue for i32::MIN
        scratchpad.chunks.sort_unstable_by_key(|x| !x.0.y);

        for ((chunk_x, chunk_z), column) in scratchpad
            .chunks
            .iter()
            .chunk_by(|entry| (entry.0.x, entry.0.z))
            .into_iter()
        {
            let rel_x = chunk_x - min_x_c;
            let rel_z = chunk_z - min_z_c;

            let start_idx_2d = (rel_x * real_extent_z) as usize + rel_z as usize;
            let mut local_occupancy = OcclusionField::zero();

            for (_coord, chunk) in column {
                if local_occupancy.all_set() {
                    break;
                }
                let cd = chunk.chunk_data();
                for i in 0..CHUNK_SIZE_I32 {
                    for j in 0..CHUNK_SIZE_I32 {
                        let dst_idx = start_idx_2d + (i * real_extent_z + j) as usize * CHUNK_SIZE;
                        if local_occupancy.get(i as usize, j as usize) {
                            continue;
                        }
                        let src_start = (i, 0, j).as_padded_index();
                        let src_end = src_start + CHUNK_SIZE;

                        for id in cd.block_ids()[src_start..src_end].iter().rev() {
                            if *id != BlockId::AIR {
                                local_occupancy.set(i as usize, j as usize, true);
                                scratchpad.blocks[dst_idx] = *id;
                                break;
                            }
                        }
                    }
                }
            }
        }
        // Avoid holding outstanding Arc<ClientChunk>... Is there a better way to enforce this
        // while still getting to keep the allocations around?
        scratchpad.chunks.clear();

        let mut image = image::RgbaImage::new(real_extent_x as u32, real_extent_z as u32);

        for (&block_id, pixel) in scratchpad.blocks.iter().zip(image.pixels_mut()) {
            let argb = types.lod_color_argb(block_id).top_argb;
            let r = ((argb >> 16) & 0xff) as u8;
            let g = ((argb >> 8) & 0xff) as u8;
            let b = ((argb >> 0) & 0xff) as u8;
            // We don't have a way to actually use the alpha to blend with at the moment;
            // just use 255
            pixel.0 = [r, g, b, 255];
        }

        Minimap {
            min_coord,
            image,
            scale: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChunkNeighborOutcome {
    ShouldMesh,
    DontMesh,
}

struct ChunkWithEdgesBuffer(bool, Box<[BlockId; PADDED_CHUNK_VOLUME]>);

pub(crate) struct FastChunkNeighbors {
    outcome: ChunkNeighborOutcome,
    center: Option<Arc<ClientChunk>>,
    neighbors: [[[ChunkWithEdgesBuffer; 3]; 3]; 3],
    inbound_lights: [[[OcclusionField; 3]; 3]; 3],
    inbound_weather: [[[OcclusionField; 3]; 3]; 3],
}
impl FastChunkNeighbors {
    pub(crate) fn get(
        &self,
        coord_xyz: (i32, i32, i32),
    ) -> Option<&[BlockId; PADDED_CHUNK_VOLUME]> {
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
    pub(crate) fn inbound_light(&self, coord_xyz: (i32, i32, i32)) -> OcclusionField {
        assert!((-1..=1).contains(&coord_xyz.0));
        assert!((-1..=1).contains(&coord_xyz.1));
        assert!((-1..=1).contains(&coord_xyz.2));
        self.inbound_lights[(coord_xyz.2 + 1) as usize][(coord_xyz.1 + 1) as usize]
            [(coord_xyz.0 + 1) as usize]
    }
    #[inline]
    pub(crate) fn inbound_weather(&self, coord_xyz: (i32, i32, i32)) -> OcclusionField {
        assert!((-1..=1).contains(&coord_xyz.0));
        assert!((-1..=1).contains(&coord_xyz.1));
        assert!((-1..=1).contains(&coord_xyz.2));
        self.inbound_weather[(coord_xyz.2 + 1) as usize][(coord_xyz.1 + 1) as usize]
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
                    std::array::from_fn(|_| ChunkWithEdgesBuffer(false, bytemuck::zeroed_box()))
                })
            }),
            inbound_lights: std::array::from_fn(|_| {
                std::array::from_fn(|_| std::array::from_fn(|_| OcclusionField::zero()))
            }),
            inbound_weather: std::array::from_fn(|_| {
                std::array::from_fn(|_| std::array::from_fn(|_| OcclusionField::zero()))
            }),
            outcome: ChunkNeighborOutcome::DontMesh,
        }
    }
}
