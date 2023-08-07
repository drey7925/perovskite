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
use std::sync::atomic::AtomicBool;

use cgmath::{ElementWise, Matrix4, Vector3};
use cuberef_core::coordinates::{BlockCoordinate, ChunkOffset};
use cuberef_core::protocol::game_rpc as rpc_proto;
use cuberef_core::{block_id::BlockId, coordinates::ChunkCoordinate};

use anyhow::{ensure, Context, Result};

use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use tracy_client::span;

use crate::block_renderer::{BlockRenderer, VkChunkVertexData};
use crate::vulkan::shaders::cube_geometry::CubeGeometryDrawCall;

use super::{ChunkManagerClonedView, ChunkManagerView};

pub(crate) struct ChunkDataView<'a>(RwLockReadGuard<'a, ChunkData>);
impl ChunkDataView<'_> {
    pub(crate) fn block_ids(&self) -> &[BlockId; 18 * 18 * 18] {
        &self.0.block_ids
    }

    pub(crate) fn lightmap(&self) -> &Box<[u8; 18 * 18 * 18]> {
        &self.0.lightmap
    }

    pub(crate) fn get_block(&self, offset: ChunkOffset) -> BlockId {
        self.0.block_ids[offset.as_extended_index()]
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
    BlockIdsNeedProcessing,
    /// We processed the chunk's neighbors, but we don't have anything to render.
    BlockIdsNoRender,

    /// We expect the chunk to have something to render, and we've prepared its neighbors.
    BlockIdsReadyToRender,
    /// We don't expect the chunk to have anything to render, but the renderer should verify this and
    /// use a bright error texture + log an error message if it renders anything.
    BlockIdsWithNeighborsAuditNoRender,
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
    /// If None, then this chunk doesn't contain any light
    ///
    /// This is stored separate from the block_ids because it might be used for lighting non-chunk meshes, such as
    /// players, entities, etc. If None, not calculated yet.
    pub(crate) lightmap: Box<[u8; 18 * 18 * 18]>,

    data_state: BlockIdState,
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
    cached_vertex_data: Mutex<Option<VkChunkVertexData>>,
}
impl ClientChunk {
    pub(crate) fn from_proto(proto: rpc_proto::MapChunk) -> Result<ClientChunk> {
        let coord = proto
            .chunk_coord
            .with_context(|| "Missing chunk_coord")?
            .into();
        let data = proto
            .chunk_data
            .with_context(|| "chunk_data missing")?
            .chunk_data
            .with_context(|| "inner chunk_data missing")?;
        let block_ids: &[u32; 4096] = match &data {
            cuberef_core::protocol::map::stored_chunk::ChunkData::V1(v1_data) => {
                ensure!(v1_data.block_ids.len() == 4096);
                v1_data.block_ids.deref().try_into().unwrap()
            }
        };
        Ok(ClientChunk {
            coord,
            chunk_data: RwLock::new(ChunkData {
                block_ids: Self::expand_ids(block_ids),
                data_state: BlockIdState::BlockIdsNeedProcessing,
                lightmap: Box::new([0; 18 * 18 * 18]),
            }),
            cached_vertex_data: Mutex::new(None),
        })
    }

    pub(crate) fn mesh_with(&self, renderer: &BlockRenderer) -> Result<()> {
        let data = self.chunk_data();
        let vertex_data = match data.0.data_state {
            BlockIdState::BlockIdsNeedProcessing => {
                log::warn!("BlockIdsNeedProcessing for {:?}", self.coord);
                None
                //Some(renderer.mesh_chunk(&data)?)
            }
            BlockIdState::BlockIdsNoRender => None,
            BlockIdState::BlockIdsReadyToRender => Some(renderer.mesh_chunk(&data)?),
            BlockIdState::BlockIdsWithNeighborsAuditNoRender => {
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
        *self.cached_vertex_data.lock() = vertex_data;
        Ok(())
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

    pub(crate) fn update_from(&self, proto: rpc_proto::MapChunk) -> Result<()> {
        let coord: ChunkCoordinate = proto
            .chunk_coord
            .with_context(|| "Missing chunk_coord")?
            .into();
        ensure!(coord == self.coord);
        let data = proto
            .chunk_data
            .with_context(|| "chunk_data missing")?
            .chunk_data
            .with_context(|| "inner chunk_data missing")?;
        let block_ids: &[u32; 4096] = match &data {
            cuberef_core::protocol::map::stored_chunk::ChunkData::V1(v1_data) => {
                ensure!(v1_data.block_ids.len() == 4096);
                v1_data.block_ids.deref().try_into().unwrap()
            }
        };
        // unwrap is safe: we verified the length
        let mut data_guard = self.chunk_data.write();
        data_guard.block_ids = Self::expand_ids(block_ids);
        data_guard.data_state = BlockIdState::BlockIdsNeedProcessing;
        Ok(())
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
            chunk_data.data_state = BlockIdState::BlockIdsNeedProcessing;
            chunk_data.block_ids[block_coord.offset().as_extended_index()] = new_id;
        }

        Ok(old_id != new_id)
    }
    pub(crate) fn make_draw_call(
        &self,
        player_position: Vector3<f64>,
    ) -> Option<CubeGeometryDrawCall> {
        // We calculate the transform in f64, and *then* narrow to f32
        // Otherwise, we get catastrophic cancellation when far from the origin

        // Relative origin in player-relative coordinates, where the player is at 0,0,0 and
        // the X/Y/Z axes are aligned with the global axes (i.e. NOT rotated)
        //
        // player_position is in game coordinates (+Y up) and the result should be in Vulkan
        // coordinates (+Y down)
        let relative_origin = (Vector3::new(
            16. * self.coord.x as f64,
            16. * self.coord.y as f64,
            16. * self.coord.z as f64,
        ) - player_position)
            .mul_element_wise(Vector3::new(1., -1., 1.));
        self.cached_vertex_data.lock().as_ref().and_then(|x| {
            Some(CubeGeometryDrawCall {
                models: x.clone_if_nonempty()?,
                model_matrix: Matrix4::from_translation(relative_origin.cast().unwrap()),
            })
        })
    }

    pub(crate) fn get_single(&self, offset: ChunkOffset) -> BlockId {
        self.chunk_data.read().block_ids[offset.as_extended_index()]
    }

    pub(crate) fn coord(&self) -> ChunkCoordinate {
        self.coord
    }

    pub(crate) fn chunk_data(&self) -> ChunkDataView<'_> {
        ChunkDataView(self.chunk_data.read())
    }

    pub(crate) fn chunk_data_mut(&self) -> ChunkDataViewMut<'_> {
        ChunkDataViewMut(self.chunk_data.write())
    }
}

pub(crate) trait ChunkOffsetExt {
    fn as_extended_index(&self) -> usize;
}
impl ChunkOffsetExt for ChunkOffset {
    fn as_extended_index(&self) -> usize {
        (self.z as usize + 1) * 18 * 18 + (self.y as usize + 1) * 18 + (self.x as usize + 1)
    }
}
impl ChunkOffsetExt for (i32, i32, i32) {
    fn as_extended_index(&self) -> usize {
        const VALID_RANGE: RangeInclusive<i32> = -1..=16;
        assert!(VALID_RANGE.contains(&self.0));
        assert!(VALID_RANGE.contains(&self.1));
        assert!(VALID_RANGE.contains(&self.2));
        ((self.2 + 1) as usize) * 18 * 18 + ((self.1 + 1) as usize) * 18 + ((self.0 + 1) as usize)
    }
}
impl ChunkOffsetExt for (i8, i8, i8) {
    fn as_extended_index(&self) -> usize {
        const VALID_RANGE: RangeInclusive<i8> = -1..=16;
        assert!(VALID_RANGE.contains(&self.0));
        assert!(VALID_RANGE.contains(&self.1));
        assert!(VALID_RANGE.contains(&self.2));
        ((self.2 + 1) as usize) * 18 * 18 + ((self.1 + 1) as usize) * 18 + ((self.0 + 1) as usize)
    }
}
