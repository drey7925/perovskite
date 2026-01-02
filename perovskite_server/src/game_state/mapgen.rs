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

use super::game_map::MapChunk;
use crate::game_state::event::EventInitiator;
use perovskite_core::block_id::special_block_defs::AIR_ID;
use perovskite_core::block_id::BlockId;
use perovskite_core::coordinates::{BlockCoordinate, ChunkCoordinate};
use std::ops::RangeInclusive;

pub trait MapgenInterface: Send + Sync {
    /// Generate a chunk using this map generator.
    ///
    /// Args:
    ///     coord: The coordinate of the chunk to generate.
    ///     chunk: The chunk to fill.
    ///
    /// Implementations should strive to remain consistent, even over
    /// different versions of the same code. The requisite level of consistency
    /// is subjective and depends on scale - e.g. noise that affects major topography (like elevation)
    /// should be consistent to avoid massive jumps in the map, whereas
    /// inconsistent noise that affects minor details (like trees) will have a smaller effect
    /// on the gameplay experience, and inconsistent IID noise that affects single blocks (like
    /// ore generation) will probably not be noticed.
    ///
    /// ^ Caveat: This consistency is mostly important for gameplay experience, and to a small
    /// extent performance/view distance (due to terrain_range_hint). However, sudden jumps and
    /// large inconsistencies are OK if they match your intended gameplay/style, or if you need
    /// to make a major change and accept such inconsistencies as a consequence of that change.
    ///
    /// Users of the map must NOT assume block-for-block consistency between calls to fill_chunk
    /// (noting that it will likely be called only once, barring crashes that cause unsaved
    /// changes to the map).
    fn fill_chunk(&self, coord: ChunkCoordinate, chunk: &mut MapChunk);

    /// Provide an estimate of chunk coordinate Y values (one value -> 16 blocks) where terrain
    /// is most likely to be seen at the given (X, Z) chunk coordinate, or None for no hint
    ///
    /// This is used for speeding up chunk loading. Favor speed over exact precision, but prefer to
    /// err on the side of including a chunk in the range if unsure.
    fn terrain_range_hint(&self, _chunk_x: i32, _chunk_z: i32) -> Option<RangeInclusive<i32>> {
        None
    }

    /// Estimate the height of the terrain at the given (X, Z) coordinate.
    ///
    /// WIP, subject to change! Potentially will get more metadata, more outputs, vectorization, etc.
    fn far_mesh_estimate(&self, x: f64, z: f64) -> FarMeshPoint;

    /// Prints debugging information regarding map generation. The definition of this is up to
    /// the implementor, and can include whatever information is most useful for developing this
    /// specific mapgen.
    ///
    /// By default, does nothing.
    fn dump_debug(&self, _pos: BlockCoordinate, _initiator: &EventInitiator<'_>) {}

    /// When far_mesh_estimate returns a water_height, this is the block type to use for water
    fn water_type(&self, _x: f64, _z: f64) -> BlockId {
        AIR_ID
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FarMeshPoint {
    /// The height of the terrain at this point.
    pub height: f32,
    /// The block type at this point. This is used to estimate the appearance of the terrain.
    pub block_type: BlockId,
    /// The height of the water at this point.
    pub water_height: f32,
}

pub(crate) mod far_mesh;
