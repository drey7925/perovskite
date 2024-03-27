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

use std::ops::{Deref, RangeInclusive};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use cgmath::{vec3, vec4, ElementWise, Matrix4, Vector3, Vector4, Zero};
use perovskite_core::coordinates::{BlockCoordinate, ChunkOffset};
use perovskite_core::lighting::Lightfield;
use perovskite_core::protocol::game_rpc as rpc_proto;
use perovskite_core::protocol::map::StoredChunk;
use perovskite_core::{block_id::BlockId, coordinates::ChunkCoordinate};

use anyhow::{ensure, Context, Result};

use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use tracy_client::span;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{
    AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryUsage,
};

use crate::block_renderer::{
    BlockRenderer, ClientBlockTypeManager, VkChunkPassCpu, VkChunkPassGpu, VkChunkVertexDataCpu,
    VkChunkVertexDataGpu,
};
use crate::vulkan::shaders::cube_geometry::{CubeGeometryDrawCall, CubeGeometryVertex};
use prost::Message;

pub(crate) trait ChunkDataView {
    fn block_ids(&self) -> &[BlockId; 18 * 18 * 18];
    fn lightmap(&self) -> &[u8; 18 * 18 * 18];
    fn get_block(&self, offset: ChunkOffset) -> BlockId {
        self.block_ids()[offset.as_extended_index()]
    }
}

pub(crate) struct LockedChunkDataView<'a>(RwLockReadGuard<'a, ChunkData>);
impl ChunkDataView for LockedChunkDataView<'_> {
    fn block_ids(&self) -> &[BlockId; 18 * 18 * 18] {
        &self.0.block_ids
    }

    fn lightmap(&self) -> &[u8; 18 * 18 * 18] {
        &self.0.lightmap
    }
}

pub(crate) struct ChunkDataViewMut<'a>(RwLockWriteGuard<'a, ChunkData>);
impl ChunkDataViewMut<'_> {
    pub(crate) fn block_ids(&self) -> &[BlockId; 18 * 18 * 18] {
        &self.0.block_ids
    }
    pub(crate) fn block_ids_mut(&mut self) -> &mut [BlockId; 18 * 18 * 18] {
        &mut self.0.block_ids
    }
    pub(crate) fn lightmap_mut(&mut self) -> &mut [u8; 18 * 18 * 18] {
        &mut self.0.lightmap
    }
    pub(crate) fn set_state(&mut self, state: BlockIdState) {
        self.0.data_state = state;
    }

    pub(crate) fn get_block(&self, offset: ChunkOffset) -> BlockId {
        self.0.block_ids[offset.as_extended_index()]
    }
}

pub(crate) enum BlockIdState {
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
    pub(crate) block_ids: Box<[BlockId; 18 * 18 * 18]>,
    /// The calculated lightmap for this chunk + immediate neighbor blocks. The data is encoded as follows:
    /// Bottom four bits: local lighting
    /// Top four bits: global lighting
    /// TODO assess whether to optimize for time or memory
    ///
    /// This is stored separate from the block_ids because it might be used for lighting non-chunk meshes, such as
    /// players, entities, etc. If None, not calculated yet.
    pub(crate) lightmap: Box<[u8; 18 * 18 * 18]>,

    data_state: BlockIdState,
}

pub(crate) struct SnappyDecodeHelper {
    snappy_decoder: snap::raw::Decoder,
    snappy_output_buffer: Vec<u8>,
}
impl SnappyDecodeHelper {
    fn decode<T>(&mut self, data: &[u8]) -> Result<T>
    where
        T: Message + Default,
    {
        let decode_len = snap::raw::decompress_len(data)?;
        if self.snappy_output_buffer.len() < decode_len {
            self.snappy_output_buffer.resize(decode_len, 0);
        }
        let decompressed_len = self
            .snappy_decoder
            .decompress(data, &mut self.snappy_output_buffer)?;
        Ok(T::decode(&self.snappy_output_buffer[0..decompressed_len])?)
    }

    pub(crate) fn new() -> SnappyDecodeHelper {
        SnappyDecodeHelper {
            snappy_decoder: snap::raw::Decoder::new(),
            snappy_output_buffer: Vec::new(),
        }
    }
}
pub(crate) const TARGET_BATCH_OCCUPANCY: usize = 32;

struct ChunkMesh {
    solo_cpu: Option<VkChunkVertexDataCpu>,
    solo_gpu: Option<VkChunkVertexDataGpu>,
    batch: Option<u64>,
}
impl Drop for ChunkMesh {
    fn drop(&mut self) {
        if let Some(cpu_data) = self.solo_cpu.take() {
            if let Some(solid) = cpu_data.solid_opaque {
                SOLID_RECLAIMER.put(solid.idx, solid.vtx);
            }
            if let Some(transparent) = cpu_data.transparent {
                TRANSPARENT_RECLAIMER.put(transparent.idx, transparent.vtx);
            }
            if let Some(translucent) = cpu_data.translucent {
                TRANSLUCENT_RECLAIMER.put(translucent.idx, translucent.vtx);
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
    last_meshed: Mutex<Instant>,
    // Speedup hint only
    has_solo_hint: AtomicBool,
}

#[derive(PartialEq, Eq)]
pub(crate) enum MeshResult {
    NewMesh(Option<u64>),
    SameMesh,
    EmptyMesh(Option<u64>),
}

// TODO move these from static into actual owners
pub(crate) struct MeshVectorReclaim {
    sender: flume::Sender<(Vec<u32>, Vec<CubeGeometryVertex>)>,
    receiver: flume::Receiver<(Vec<u32>, Vec<CubeGeometryVertex>)>,
}
impl MeshVectorReclaim {
    fn new() -> MeshVectorReclaim {
        let (sender, receiver) = flume::bounded(256);
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

lazy_static::lazy_static! {
    pub(crate) static ref SOLID_RECLAIMER: MeshVectorReclaim = MeshVectorReclaim::new();
    pub(crate) static ref TRANSPARENT_RECLAIMER: MeshVectorReclaim = MeshVectorReclaim::new();
    pub(crate) static ref TRANSLUCENT_RECLAIMER: MeshVectorReclaim = MeshVectorReclaim::new();
}

impl ClientChunk {
    pub(crate) fn from_proto(
        proto: rpc_proto::MapChunk,
        snappy_helper: &mut SnappyDecodeHelper,
        block_types: &ClientBlockTypeManager,
    ) -> Result<(ClientChunk, Lightfield)> {
        let coord = proto
            .chunk_coord
            .with_context(|| "Missing chunk_coord")?
            .into();
        let data = snappy_helper
            .decode::<StoredChunk>(&proto.snappy_encoded_bytes)?
            .chunk_data
            .with_context(|| "inner chunk_data missing")?;
        let block_ids: &[u32; 4096] = match &data {
            perovskite_core::protocol::map::stored_chunk::ChunkData::V1(v1_data) => {
                ensure!(v1_data.block_ids.len() == 4096);
                v1_data.block_ids.deref().try_into().unwrap()
            }
        };
        let occlusion = get_occlusion_for_proto(block_ids, block_types);
        Ok((
            ClientChunk {
                coord,
                chunk_data: RwLock::new(ChunkData {
                    block_ids: Self::expand_ids(block_ids),
                    data_state: BlockIdState::NeedProcessing,
                    lightmap: Box::new([0; 18 * 18 * 18]),
                }),
                chunk_mesh: Mutex::new(ChunkMesh {
                    solo_cpu: None,
                    solo_gpu: None,
                    batch: None,
                }),
                last_meshed: Mutex::new(Instant::now()),
                has_solo_hint: AtomicBool::new(false),
            },
            occlusion,
        ))
    }

    pub(crate) fn last_meshed(&self) -> Instant {
        *self.last_meshed.lock()
    }

    pub(crate) fn mesh_with(&self, renderer: &BlockRenderer) -> Result<MeshResult> {
        let data = self.chunk_data();

        let vertex_data = match data.0.data_state {
            BlockIdState::NeedProcessing => Some(renderer.mesh_chunk(&data)?),
            BlockIdState::NoRender => None,
            BlockIdState::ReadyToRender => Some(renderer.mesh_chunk(&data)?),
            BlockIdState::AuditNoRender => {
                let result = renderer.mesh_chunk(&data)?;
                if result.solid_opaque.is_some()
                    || result.transparent.is_some()
                    || result.translucent.is_some()
                {
                    log::warn!("Failed no-render audit for {:?}", self.coord);
                }
                Some(result)
            }
        };
        *self.last_meshed.lock() = Instant::now();
        let mut mesh_lock = self.chunk_mesh.lock();
        let old_batch = mesh_lock.batch;
        if let Some(vertex_data) = vertex_data {
            if Some(&vertex_data) == mesh_lock.solo_cpu.as_ref() {
                Ok(MeshResult::SameMesh)
            } else {
                *mesh_lock = ChunkMesh {
                    solo_gpu: Some(vertex_data.to_gpu(renderer.allocator())?),
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

    fn expand_ids(ids: &[u32; 16 * 16 * 16]) -> Box<[BlockId; 18 * 18 * 18]> {
        let mut result = Box::new([BlockId(u32::MAX); 18 * 18 * 18]);
        for i in 0..16 {
            for j in 0..16 {
                for k in 0..16 {
                    result[(i + 1) * 18 * 18 + (j + 1) * 18 + (k + 1)] =
                        BlockId::from(ids[i * 16 * 16 + j * 16 + k]);
                }
            }
        }
        result
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
        *self.last_meshed.lock() = Instant::now();
        let mut lock = self.chunk_mesh.lock();
        lock.batch = Some(id);
    }

    pub(crate) fn get_batch(&self) -> Option<u64> {
        self.chunk_mesh.lock().batch
    }

    pub(crate) fn spill_back_to_solo(&self, expecting: u64) -> Option<u64> {
        self.has_solo_hint.store(true, Ordering::Relaxed);
        *self.last_meshed.lock() = Instant::now();
        let mut lock = self.chunk_mesh.lock();
        if lock.batch.is_some_and(|b| b != expecting) {
            panic!("Expected batch {:?}, got {:?}", expecting, lock.batch);
        }
        lock.batch.take()
    }

    pub(crate) fn update_from(
        &self,
        proto: rpc_proto::MapChunk,
        snappy_helper: &mut SnappyDecodeHelper,
        block_types: &ClientBlockTypeManager,
    ) -> Result<Lightfield> {
        let coord: ChunkCoordinate = proto
            .chunk_coord
            .with_context(|| "Missing chunk_coord")?
            .into();
        ensure!(coord == self.coord);
        let data = snappy_helper
            .decode::<StoredChunk>(&proto.snappy_encoded_bytes)?
            .chunk_data
            .with_context(|| "inner chunk_data missing")?;
        let block_ids: &[u32; 4096] = match &data {
            perovskite_core::protocol::map::stored_chunk::ChunkData::V1(v1_data) => {
                ensure!(v1_data.block_ids.len() == 4096);
                v1_data.block_ids.deref().try_into().unwrap()
            }
        };
        let occlusion = get_occlusion_for_proto(block_ids, block_types);
        // unwrap is safe: we verified the length
        let mut data_guard = self.chunk_data.write();
        data_guard.block_ids = Self::expand_ids(block_ids);
        data_guard.data_state = BlockIdState::NeedProcessing;
        Ok(occlusion)
    }

    pub(crate) fn apply_delta(&self, proto: &rpc_proto::MapDeltaUpdate) -> Result<bool> {
        let block_coord: BlockCoordinate = proto
            .block_coord
            .clone()
            .with_context(|| "block_coord missing in MapDeltaUpdate")?
            .into();
        ensure!(block_coord.chunk() == self.coord);

        let mut chunk_data = self.chunk_data.write();
        let old_id = chunk_data.block_ids[block_coord.offset().as_extended_index()];
        let new_id = BlockId::from(proto.new_id);
        // TODO future optimization: check whether a change in variant actually requires
        // a redraw
        if old_id != new_id {
            chunk_data.data_state = BlockIdState::NeedProcessing;
            chunk_data.block_ids[block_coord.offset().as_extended_index()] = new_id;
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
        if Self::check_frustum(view_proj_matrix * translation) {
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
                    let id = data.block_ids[(ChunkOffset { x, y, z }).as_extended_index()];
                    if !block_types.propagates_light(id) {
                        occlusion.set(x, z, true);
                        break 'inner;
                    }
                }
            }
        }
        occlusion
    }

    fn check_frustum(transformation: Matrix4<f32>) -> bool {
        #[inline]
        fn mvmul4(matrix: Matrix4<f32>, vector: Vector4<f32>) -> Vector4<f32> {
            // This is the implementation hidden behind the simd feature gate
            matrix[0] * vector[0]
                + matrix[1] * vector[1]
                + matrix[2] * vector[2]
                + matrix[3] * vector[3]
        }

        #[inline]
        fn overlaps(min1: f32, max1: f32, min2: f32, max2: f32) -> bool {
            min1 <= max2 && min2 <= max1
        }
        // todo fix jank that requires this to be 17.0 rather than 16.0
        const CORNERS: [Vector4<f32>; 8] = [
            vec4(-1.0, 1.0, -1.0, 1.),
            vec4(17.0, 1.0, -1.0, 1.),
            vec4(-1.0, -17.0, -1.0, 1.),
            vec4(17.0, -17.0, -1.0, 1.),
            vec4(-1.0, 1.0, 17.0, 1.),
            vec4(17.0, 1.0, 17.0, 1.),
            vec4(-1.0, -17.0, 17.0, 1.),
            vec4(17.0, -17.0, 17.0, 1.),
        ];
        let mut ndc_min = vec4(f32::INFINITY, f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut ndc_max = vec4(
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        );

        for corner in CORNERS {
            let mut ndc = mvmul4(transformation, corner);
            let ndcw = ndc.w;
            // We don't want to flip the x/y/z components when the ndc is negative, since
            // then we'll span the frustum.
            // We also want to avoid an ndc of exactly zero
            ndc /= ndc.w.abs().max(0.000001);
            ndc_min = vec4(
                ndc_min.x.min(ndc.x),
                ndc_min.y.min(ndc.y),
                0.0,
                ndc_min.w.min(ndcw),
            );
            ndc_max = vec4(
                ndc_max.x.max(ndc.x),
                ndc_max.y.max(ndc.y),
                0.0,
                ndc_max.w.max(ndcw),
            );
        }
        // Simply dividing by w as we go isn't enough; we need to also ensure that at least
        // one point is actually in the front clip space: https://stackoverflow.com/a/51798873/1424875
        //
        // This check is a bit conservative; it's possible that one point is in front of the camera, but not within
        // the frustum, while other points cause the overlap check to pass. However, it's good enough for now.
        ndc_max.w > 0.0
            && overlaps(ndc_min.x, ndc_max.x, -1., 1.)
            && overlaps(ndc_min.y, ndc_max.y, -1., 1.)
    }

    pub(crate) fn get_single(&self, offset: ChunkOffset) -> BlockId {
        self.chunk_data.read().block_ids[offset.as_extended_index()]
    }

    pub(crate) fn coord(&self) -> ChunkCoordinate {
        self.coord
    }

    pub(crate) fn chunk_data(&self) -> LockedChunkDataView<'_> {
        LockedChunkDataView(self.chunk_data.read())
    }

    pub(crate) fn chunk_data_mut(&self) -> ChunkDataViewMut<'_> {
        ChunkDataViewMut(self.chunk_data.write())
    }
}

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
    solid_vtx: Option<Subbuffer<[CubeGeometryVertex]>>,
    solid_idx: Option<Subbuffer<[u32]>>,
    transparent_vtx: Option<Subbuffer<[CubeGeometryVertex]>>,
    transparent_idx: Option<Subbuffer<[u32]>>,
    translucent_vtx: Option<Subbuffer<[CubeGeometryVertex]>>,
    translucent_idx: Option<Subbuffer<[u32]>>,

    chunks: smallvec::SmallVec<[ChunkCoordinate; TARGET_BATCH_OCCUPANCY]>,
    base_position: Vector3<f64>,
}
impl MeshBatch {
    pub(crate) fn coords(&self) -> &[ChunkCoordinate] {
        &self.chunks
    }
    pub(crate) fn make_draw_call(&self, player_position: Vector3<f64>) -> CubeGeometryDrawCall {
        let matrix = Matrix4::from_translation(
            (self.base_position - player_position).mul_element_wise(vec3(1.0, -1.0, 1.0)),
        );
        CubeGeometryDrawCall {
            models: VkChunkVertexDataGpu {
                solid_opaque: if self.solid_vtx.is_some() && self.solid_idx.is_some() {
                    Some(VkChunkPassGpu {
                        vtx: self.solid_vtx.as_ref().unwrap().clone(),
                        idx: self.solid_idx.as_ref().unwrap().clone(),
                    })
                } else {
                    None
                },

                transparent: if self.transparent_vtx.is_some() && self.transparent_idx.is_some() {
                    Some(VkChunkPassGpu {
                        vtx: self.transparent_vtx.as_ref().unwrap().clone(),
                        idx: self.transparent_idx.as_ref().unwrap().clone(),
                    })
                } else {
                    None
                },

                translucent: if self.translucent_vtx.is_some() && self.translucent_idx.is_some() {
                    Some(VkChunkPassGpu {
                        vtx: self.translucent_vtx.as_ref().unwrap().clone(),
                        idx: self.translucent_idx.as_ref().unwrap().clone(),
                    })
                } else {
                    None
                },
            },
            model_matrix: matrix.cast().unwrap(),
        }
    }
    pub fn id(&self) -> u64 {
        self.id
    }
}

pub(crate) struct MeshBatchBuilder {
    id: u64,
    solid_vtx: Vec<CubeGeometryVertex>,
    solid_idx: Vec<u32>,
    transparent_vtx: Vec<CubeGeometryVertex>,
    transparent_idx: Vec<u32>,
    translucent_vtx: Vec<CubeGeometryVertex>,
    translucent_idx: Vec<u32>,
    base_position: Vector3<f64>,
    chunks: smallvec::SmallVec<[ChunkCoordinate; TARGET_BATCH_OCCUPANCY]>,
}
impl MeshBatchBuilder {
    pub(crate) fn new() -> MeshBatchBuilder {
        MeshBatchBuilder {
            id: next_id(),
            // rough initial estimate, but should be good enough for now
            solid_vtx: Vec::with_capacity(TARGET_BATCH_OCCUPANCY * 4000),
            solid_idx: Vec::with_capacity(TARGET_BATCH_OCCUPANCY * 6000),
            transparent_vtx: Vec::new(),
            transparent_idx: Vec::new(),
            translucent_vtx: Vec::new(),
            translucent_idx: Vec::new(),
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
        input: &VkChunkPassCpu,
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
        if let Some(opaque) = cpu.solid_opaque.as_ref() {
            Self::extend_buffer(
                opaque,
                &mut self.solid_vtx,
                &mut self.solid_idx,
                (chunk_pos - self.base_position).cast().unwrap(),
            );
        }
        if let Some(transparent) = cpu.transparent.as_ref() {
            Self::extend_buffer(
                transparent,
                &mut self.transparent_vtx,
                &mut self.transparent_idx,
                (chunk_pos - self.base_position).cast().unwrap(),
            );
        }
        if let Some(translucent) = cpu.translucent.as_ref() {
            Self::extend_buffer(
                translucent,
                &mut self.translucent_vtx,
                &mut self.translucent_idx,
                (chunk_pos - self.base_position).cast().unwrap(),
            );
        }
        self.chunks.push(coord);
    }

    pub(crate) fn build_and_reset(
        &mut self,
        allocator: &GenericMemoryAllocator<Arc<FreeListAllocator>>,
    ) -> Result<MeshBatch> {
        fn build_vertex(
            buf: &[CubeGeometryVertex],
            allocator: &GenericMemoryAllocator<Arc<FreeListAllocator>>,
        ) -> Result<Option<Subbuffer<[CubeGeometryVertex]>>> {
            if buf.is_empty() {
                Ok(None)
            } else {
                Ok(Some(Buffer::from_iter(
                    allocator,
                    BufferCreateInfo {
                        usage: BufferUsage::VERTEX_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        usage: MemoryUsage::Upload,
                        ..Default::default()
                    },
                    buf.iter().copied(),
                )?))
            }
        }

        fn build_index(
            buf: &[u32],
            allocator: &GenericMemoryAllocator<Arc<FreeListAllocator>>,
        ) -> Result<Option<Subbuffer<[u32]>>> {
            if buf.is_empty() {
                Ok(None)
            } else {
                Ok(Some(Buffer::from_iter(
                    allocator,
                    BufferCreateInfo {
                        usage: BufferUsage::INDEX_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        usage: MemoryUsage::Upload,
                        ..Default::default()
                    },
                    buf.iter().copied(),
                )?))
            }
        }
        let result = MeshBatch {
            id: self.id,
            solid_vtx: build_vertex(&self.solid_vtx, allocator)?,
            solid_idx: build_index(&self.solid_idx, allocator)?,
            transparent_vtx: build_vertex(&self.transparent_vtx, allocator)?,
            transparent_idx: build_index(&self.transparent_idx, allocator)?,
            translucent_vtx: build_vertex(&self.translucent_vtx, allocator)?,
            translucent_idx: build_index(&self.translucent_idx, allocator)?,
            base_position: self.base_position,
            chunks: self.chunks.clone(),
        };

        self.reset();
        Ok(result)
    }

    pub(crate) fn reset(&mut self) {
        self.solid_vtx.clear();
        self.solid_idx.clear();
        self.transparent_vtx.clear();
        self.transparent_idx.clear();
        self.translucent_vtx.clear();
        self.translucent_idx.clear();
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
    fn as_extended_index(&self) -> usize {
        // This unusual order matches that in coordinates.rs and is designed to be cache-friendly for
        // vertical iteration, which are commonly used in global lighting.
        (self.x as usize + 1) * 18 * 18 + (self.z as usize + 1) * 18 + (self.y as usize + 1)
    }
}
impl ChunkOffsetExt for (i32, i32, i32) {
    fn as_extended_index(&self) -> usize {
        const VALID_RANGE: RangeInclusive<i32> = -1..=16;
        assert!(VALID_RANGE.contains(&self.0));
        assert!(VALID_RANGE.contains(&self.1));
        assert!(VALID_RANGE.contains(&self.2));
        // See comment in ChunkOffsetExt for ChunkOffset
        ((self.0 + 1) as usize) * 18 * 18 + ((self.2 + 1) as usize) * 18 + ((self.1 + 1) as usize)
    }
}
impl ChunkOffsetExt for (i8, i8, i8) {
    fn as_extended_index(&self) -> usize {
        const VALID_RANGE: RangeInclusive<i8> = -1..=16;
        assert!(VALID_RANGE.contains(&self.0));
        assert!(VALID_RANGE.contains(&self.1));
        assert!(VALID_RANGE.contains(&self.2));
        // See comment in ChunkOffsetExt for ChunkOffset
        ((self.0 + 1) as usize) * 18 * 18 + ((self.2 + 1) as usize) * 18 + ((self.1 + 1) as usize)
    }
}
