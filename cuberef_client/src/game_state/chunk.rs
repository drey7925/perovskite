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


use cgmath::{ElementWise, Matrix4, Vector3};
use cuberef_core::coordinates::{BlockCoordinate, ChunkOffset};
use cuberef_core::protocol::game_rpc as rpc_proto;
use cuberef_core::{block_id::BlockId, coordinates::ChunkCoordinate};

use anyhow::{ensure, Context, Result};

use tracy_client::span;

use crate::cube_renderer::{BlockRenderer, VkChunkVertexData};
use crate::vulkan::shaders::cube_geometry::{CubeGeometryDrawCall};

use super::ChunkManagerView;

pub(crate) struct ClientChunk {
    coord: ChunkCoordinate,
    block_ids: Vec<BlockId>,
    cached_vertex_data: Option<VkChunkVertexData>,
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
        let block_ids = match data {
            cuberef_core::protocol::map::stored_chunk::ChunkData::V1(v1_data) => {
                ensure!(v1_data.block_ids.len() == 4096);
                v1_data
                    .block_ids
                    .iter()
                    .map(|&x| BlockId::from(x))
                    .collect::<Vec<_>>()
            }
        };
        Ok(ClientChunk {
            coord,
            block_ids,
            cached_vertex_data: None,
        })
    }

    pub(crate) fn block_ids(&self) -> &[BlockId; 4096] {
        (self.block_ids.deref()).try_into().unwrap()
    }

    pub(crate) fn apply_delta(&mut self, proto: &rpc_proto::MapDeltaUpdate) -> Result<bool> {
        let block_coord: BlockCoordinate = proto
            .block_coord
            .clone()
            .with_context(|| "block_coord missing in MapDeltaUpdate")?
            .into();
        ensure!(block_coord.chunk() == self.coord);
        let old_id = self.block_ids[block_coord.offset().as_index()];
        let new_id = BlockId::from(proto.new_id);
        // TODO future optimization: check whether a change in variant actually requires
        // a redraw
        if old_id != new_id {
            self.block_ids[block_coord.offset().as_index()] = new_id;
            self.cached_vertex_data = None;
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
        self.cached_vertex_data
            .as_ref()
            .and_then(|x| {
                Some(CubeGeometryDrawCall {
                    models: x.clone_if_nonempty()?,
                    model_matrix: Matrix4::from_translation(relative_origin.cast().unwrap()),
                })
            })
    }

    pub(crate) fn get(&self, offset: ChunkOffset) -> BlockId {
        self.block_ids[offset.as_index()]
    }

    pub(crate) fn coord(&self) -> ChunkCoordinate {
        self.coord
    }
}

pub(crate) fn maybe_mesh_chunk(
    chunk_coord: ChunkCoordinate,
    chunk_data: &ChunkManagerView,
    cube_renderer: &BlockRenderer,
) -> Result<()> {
    if chunk_data.contains_key(&chunk_coord) {
        mesh_chunk(chunk_coord, chunk_data, cube_renderer)
    } else {
        Ok(())
    }
}

pub(crate) fn mesh_chunk(
    chunk_coord: ChunkCoordinate,
    chunk_data: &ChunkManagerView,
    cube_renderer: &BlockRenderer,
) -> Result<()> {
    let _span = span!("meshing");
    let mut current_chunk = chunk_data
        .get_mut(&chunk_coord)
        .with_context(|| "The chunk being meshed is not loaded")?;
    let vertex_data = cube_renderer.mesh_chunk(chunk_data, current_chunk.deref())?;
    current_chunk.cached_vertex_data = Some(vertex_data);

    Ok(())
}
