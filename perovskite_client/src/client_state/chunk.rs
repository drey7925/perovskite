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
use std::ops::RangeInclusive;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use cgmath::{vec3, vec4, ElementWise, Matrix4, Vector3, Vector4, Zero};
use enum_map::{enum_map, EnumMap};
use perovskite_core::coordinates::{BlockCoordinate, ChunkOffset};
use perovskite_core::lighting::Lightfield;
use perovskite_core::protocol::game_rpc as rpc_proto;
use perovskite_core::{block_id::BlockId, coordinates::ChunkCoordinate};

use anyhow::{ensure, Context, Result};
use bytemuck::{cast_slice, must_cast_slice};
use egui::ahash::HashMapExt;
use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use tracy_client::span;
use vulkano::buffer::Subbuffer;

use super::block_types::ClientBlockTypeManager;
use crate::vulkan::block_renderer::{
    BlockRenderer, VkChunkRaytraceData, VkChunkVertexDataCpu, VkChunkVertexDataGpu,
};
use crate::vulkan::raytrace_buffer::RaytraceBufferManager;
use crate::vulkan::shaders::cube_geometry::{
    CubeDrawStep, CubeGeometryDrawCall, CubeGeometryVertex,
};
use crate::vulkan::shaders::{VkBufferCpu, VkDrawBufferGpu};
use crate::vulkan::util::check_frustum;
use crate::vulkan::{BufferReclaim, ReclaimType, ReclaimableBuffer, VulkanContext};
use perovskite_core::protocol::map::ClientExtendedData;
use perovskite_core::util::AtomicInstant;
use rustc_hash::{FxHashMap, FxHasher};
use vulkano::DeviceSize;

pub(crate) trait ChunkDataView {
    fn is_empty_optimization_hint(&self) -> bool;
    fn block_ids(&self) -> &[BlockId; 18 * 18 * 18];
    fn lightmap(&self) -> &[u8; 18 * 18 * 18];
    fn get_block(&self, offset: ChunkOffset) -> BlockId {
        self.block_ids()[offset.as_extended_index()]
    }
    #[allow(dead_code)]
    fn client_ext_data(&self, offset: ChunkOffset) -> Option<&ClientExtendedData>;
    fn raytrace_data(&self) -> Option<&VkChunkRaytraceData>;

    fn effective_rt_data(&self) -> Option<(&[u32; 5832], &[u8; 5832])> {
        let rt_blocks = if let Some(rt) = self.raytrace_data() {
            rt.blocks.as_deref()
        } else {
            return None;
        };
        let chunk =
            rt_blocks.unwrap_or_else(|| must_cast_slice(self.block_ids()).try_into().unwrap());
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
    fn block_ids(&self) -> &[BlockId; 18 * 18 * 18] {
        self.0.block_ids.as_deref().unwrap_or(&ZERO_CHUNK)
    }

    fn lightmap(&self) -> &[u8; 18 * 18 * 18] {
        self.0.lightmap.as_deref().unwrap_or(&ZERO_LIGHTMAP)
    }

    fn client_ext_data(&self, offset: ChunkOffset) -> Option<&ClientExtendedData> {
        self.0.client_ext_data.get(&(offset.as_index() as u16))
    }

    fn raytrace_data(&self) -> Option<&VkChunkRaytraceData> {
        self.0.raytrace_data.as_ref()
    }
}

pub(crate) struct ChunkDataViewMut<'a>(RwLockWriteGuard<'a, ChunkData>);

pub(crate) fn hash_rt(data: Option<(&[u32], &[u8])>) -> (u64, u64) {
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

    pub(crate) fn block_ids(&self) -> &[BlockId; 18 * 18 * 18] {
        self.0.block_ids.as_deref().unwrap_or(&ZERO_CHUNK)
    }
    pub(crate) fn block_ids_mut(&mut self) -> Option<&mut [BlockId; 18 * 18 * 18]> {
        self.0.block_ids.as_deref_mut()
    }
    pub(crate) fn lightmap_mut(&mut self) -> &mut [u8; 18 * 18 * 18] {
        self.0.lightmap.get_or_insert_with(|| {
            log::warn!("Filling nonexisting lightmap in mutator; likely a bug");
            Box::new([0; 18 * 18 * 18])
        })
    }

    pub(crate) fn lightmap(&self) -> &[u8; 18 * 18 * 18] {
        self.0.lightmap.as_deref().unwrap_or(&ZERO_LIGHTMAP)
    }
    pub(crate) fn set_state(&mut self, state: ChunkRenderState) {
        self.0.render_state = state;
    }

    #[allow(dead_code)]
    pub(crate) fn get_block(&self, offset: ChunkOffset) -> BlockId {
        self.0
            .block_ids
            .as_ref()
            .map(|x| x[offset.as_extended_index()])
            .unwrap_or(BlockId(0))
    }
    #[allow(dead_code)]
    pub(crate) fn raytrace_data_mut(&mut self) -> Option<&mut VkChunkRaytraceData> {
        self.0.raytrace_data.as_mut()
    }
    pub(crate) fn downgrade(self) -> LockedChunkDataView<'a> {
        LockedChunkDataView(RwLockWriteGuard::downgrade(self.0))
    }

    fn effective_rt_data(&self) -> Option<(&[u32], &[u8])> {
        let rt_blocks = if let Some(rt) = &self.0.raytrace_data {
            rt.blocks.as_deref()
        } else {
            return None;
        };
        let chunk = rt_blocks.unwrap_or_else(|| cast_slice(self.block_ids()).try_into().unwrap());
        let lights = self.lightmap();
        Some((chunk, lights))
    }
}

const ZERO_CHUNK: [BlockId; 18 * 18 * 18] = [BlockId(0); 18 * 18 * 18];
const ZERO_LIGHTMAP: [u8; 18 * 18 * 18] = [0; 18 * 18 * 18];

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
    pub(crate) block_ids: Option<Box<[BlockId; 18 * 18 * 18]>>,
    /// The calculated lightmap for this chunk + immediate neighbor blocks. The data is encoded as follows:
    /// Bottom four bits: local lighting
    /// Top four bits: global lighting
    /// TODO assess whether to optimize for time or memory
    ///
    /// This is stored separate from the block_ids because it might be used for lighting non-chunk meshes, such as
    /// players, entities, etc.
    ///
    /// This can't be optimized with None since we need to propagate light through the chunk
    pub(crate) lightmap: Option<Box<[u8; 18 * 18 * 18]>>,

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
        block_ids: &[u32; 4096],
        ced: Vec<ClientExtendedData>,
        block_types: &ClientBlockTypeManager,
    ) -> Result<(ClientChunk, Lightfield)> {
        let occlusion = get_occlusion_for_proto(block_ids, block_types);
        let block_ids = Self::expand_ids(block_ids);
        let lightmap = if block_ids.is_some() {
            Some(Box::new([0; 18 * 18 * 18]))
        } else {
            None
        };

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
            occlusion,
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

    fn expand_ids(ids: &[u32; 16 * 16 * 16]) -> Option<Box<[BlockId; 18 * 18 * 18]>> {
        if ids.iter().all(|&x| x == 0) {
            return None;
        }
        let mut result = Box::new([BlockId(u32::MAX); 18 * 18 * 18]);
        for i in 0..16 {
            for j in 0..16 {
                for k in 0..16 {
                    result[(i + 1) * 18 * 18 + (j + 1) * 18 + (k + 1)] =
                        BlockId::from(ids[i * 16 * 16 + j * 16 + k]);
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
        self.chunk_mesh.lock().batch
    }

    pub(crate) fn spill_back_to_solo(&self, expecting: u64) -> Option<u64> {
        self.has_solo_hint.store(true, Ordering::Relaxed);
        let mut lock = self.chunk_mesh.lock();
        self.last_meshed.update_now_relaxed();
        if lock.batch.is_some_and(|b| b != expecting) {
            panic!("Expected batch {:?}, got {:?}", expecting, lock.batch);
        }
        lock.batch.take()
    }

    pub(crate) fn update_from(
        &self,
        coord: ChunkCoordinate,
        block_ids: &[u32; 4096],
        ced: Vec<ClientExtendedData>,
        block_types: &ClientBlockTypeManager,
    ) -> Result<Lightfield> {
        ensure!(coord == self.coord);
        let occlusion = get_occlusion_for_proto(block_ids, block_types);
        let ids = Self::expand_ids(block_ids);
        let mut client_ext_data = FxHashMap::with_capacity(ced.len());
        for val in ced {
            client_ext_data.insert(val.offset_in_chunk as u16, val);
        }
        let mut data_guard = self.chunk_data.write();
        data_guard.block_ids = ids;
        data_guard.render_state = ChunkRenderState::NeedProcessing;
        data_guard.client_ext_data = client_ext_data;
        Ok(occlusion)
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
            chunk_data.block_ids = Some(Box::new([BlockId(0); 18 * 18 * 18]));
        }
        let old_id =
            chunk_data.block_ids.as_mut().unwrap()[block_coord.offset().as_extended_index()];
        let new_id = BlockId::from(proto.new_id);
        // TODO future optimization: check whether a change in variant actually requires
        // a redraw
        if old_id != new_id {
            chunk_data.render_state = ChunkRenderState::NeedProcessing;
            chunk_data.block_ids.as_mut().unwrap()[block_coord.offset().as_extended_index()] =
                new_id;
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
            16. * coord.x as f64,
            16. * coord.y as f64,
            16. * coord.z as f64,
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

    pub(crate) fn get_occlusion(&self, block_types: &ClientBlockTypeManager) -> Lightfield {
        let data = self.chunk_data.read();
        let mut occlusion = Lightfield::zero();
        for x in 0..16 {
            for z in 0..16 {
                'inner: for y in 0..16 {
                    let id = data
                        .block_ids
                        .as_ref()
                        .map(|ids| ids[(ChunkOffset { x, y, z }).as_extended_index()])
                        .unwrap_or(BlockId(0));
                    if !block_types.propagates_light(id) {
                        occlusion.set(x, z, true);
                        break 'inner;
                    }
                }
            }
        }
        occlusion
    }

    pub(crate) fn get_single(&self, offset: ChunkOffset) -> BlockId {
        self.chunk_data
            .read()
            .block_ids
            .as_deref()
            .map(|x| x[offset.as_extended_index()])
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
            .map(|x| x[offset.as_extended_index()])
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
        let _ = span!("chunk_data");
        LockedChunkDataView(self.chunk_data.read())
    }

    pub(crate) fn chunk_data_mut(&self) -> ChunkDataViewMut<'_> {
        let _ = span!("chunk_data_mut");
        ChunkDataViewMut(self.chunk_data.write())
    }
}
// todo fix jank that requires this to be 17.0 rather than 16.0
const CHUNK_CORNERS: [Vector4<f32>; 8] = [
    vec4(-1.0, 1.0, -1.0, 1.),
    vec4(17.0, 1.0, -1.0, 1.),
    vec4(-1.0, -17.0, -1.0, 1.),
    vec4(17.0, -17.0, -1.0, 1.),
    vec4(-1.0, 1.0, 17.0, 1.),
    vec4(17.0, 1.0, 17.0, 1.),
    vec4(-1.0, -17.0, 17.0, 1.),
    vec4(17.0, -17.0, 17.0, 1.),
];

fn get_occlusion_for_proto(
    block_ids: &[u32; 4096],
    block_types: &ClientBlockTypeManager,
) -> Lightfield {
    let mut occlusion = Lightfield::zero();
    for x in 0..16 {
        for z in 0..16 {
            'inner: for y in 0..16 {
                let id = block_ids[(ChunkOffset { x, y, z }).as_index()];
                if !block_types.propagates_light(id.into()) {
                    occlusion.set(x, z, true);
                    break 'inner;
                }
            }
        }
    }
    occlusion
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
                16. * coord.x as f64,
                16. * coord.y as f64,
                16. * coord.z as f64,
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
    base_position: Vector3<f64>,
    chunks: smallvec::SmallVec<[ChunkCoordinate; TARGET_BATCH_OCCUPANCY]>,
}
impl MeshBatchBuilder {
    pub(crate) fn new() -> MeshBatchBuilder {
        MeshBatchBuilder {
            id: next_id(),
            vertex_buffers: enum_map! {
                _ => vec![]
            },
            index_buffers: enum_map! {
                _ => vec![]
            },
            base_position: Vector3::zero(),
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
            coord.x as f64 * 16.0,
            coord.y as f64 * 16.0,
            coord.z as f64 * 16.0,
        );
        if self.base_position == Vector3::zero() {
            self.base_position = chunk_pos;
        }
        for (step, buffer) in cpu.draw_buffers.iter() {
            if let Some(buffer) = buffer.as_ref() {
                Self::extend_buffer(
                    buffer,
                    &mut self.vertex_buffers[step],
                    &mut self.index_buffers[step],
                    (chunk_pos - self.base_position).cast().unwrap(),
                );
            }
        }
        self.chunks.push(coord);
    }

    pub(crate) fn build_and_reset(&mut self, ctx: &VulkanContext) -> Result<MeshBatch> {
        let mut any_commands = false;

        let mut cmdbuf = ctx.start_transfer_buffer()?;
        let mut process_vtx = |x: &[CubeGeometryVertex]| -> anyhow::Result<_> {
            if x.is_empty() {
                Ok(None)
            } else {
                let len = x.len();
                let target_buffer = ctx.iter_to_device_via_staging_with_reclaim(
                    x.iter().copied(),
                    ReclaimType::CubeGeometryVtx,
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
                    ReclaimType::CubeGeometryIdx,
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
            base_position: self.base_position,
            chunks: self.chunks.clone(),
        };

        if any_commands {
            ctx.finish_transfer_buffer(cmdbuf)?;
        }

        self.reset();
        Ok(result)
    }

    pub(crate) fn reset(&mut self) {
        for buf in self.vertex_buffers.values_mut() {
            buf.clear();
        }
        for buf in self.index_buffers.values_mut() {
            buf.clear();
        }

        self.chunks.clear();
        self.base_position = Vector3::zero();
        self.id = next_id();
    }
}

fn next_id() -> u64 {
    NEXT_MESH_BATCH_ID.fetch_add(1, Ordering::Relaxed) as u64
}

pub(crate) trait ChunkOffsetExt {
    fn as_extended_index(&self) -> usize;
}
impl ChunkOffsetExt for ChunkOffset {
    #[inline]
    fn as_extended_index(&self) -> usize {
        // This unusual order matches that in coordinates.rs and is designed to be cache-friendly for
        // vertical iteration, which are commonly used in global lighting.
        (self.x as usize + 1) * 18 * 18 + (self.z as usize + 1) * 18 + (self.y as usize + 1)
    }
}
impl ChunkOffsetExt for (i32, i32, i32) {
    #[inline]
    fn as_extended_index(&self) -> usize {
        const VALID_RANGE: RangeInclusive<i32> = -1..=16;
        debug_assert!(VALID_RANGE.contains(&self.0));
        debug_assert!(VALID_RANGE.contains(&self.1));
        debug_assert!(VALID_RANGE.contains(&self.2));
        // See comment in ChunkOffsetExt for ChunkOffset
        ((self.0 + 1) as usize) * 18 * 18 + ((self.2 + 1) as usize) * 18 + ((self.1 + 1) as usize)
    }
}
impl ChunkOffsetExt for (i8, i8, i8) {
    #[inline]
    fn as_extended_index(&self) -> usize {
        const VALID_RANGE: RangeInclusive<i8> = -1..=16;
        debug_assert!(VALID_RANGE.contains(&self.0));
        debug_assert!(VALID_RANGE.contains(&self.1));
        debug_assert!(VALID_RANGE.contains(&self.2));
        // See comment in ChunkOffsetExt for ChunkOffset
        ((self.0 + 1) as usize) * 18 * 18 + ((self.2 + 1) as usize) * 18 + ((self.1 + 1) as usize)
    }
}
