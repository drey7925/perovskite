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

use perovskite_core::coordinates::{BlockCoordinate, ChunkCoordinate};
use std::ops::RangeInclusive;

use super::game_map::MapChunk;

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

    /// Prints debugging information regarding map generation. The definition of this is up to
    /// the implementor, and can include whatever information is most useful for developing this
    /// specific mapgen.
    ///
    /// By default, does nothing.
    fn dump_debug(&self, _pos: BlockCoordinate) {}
}
