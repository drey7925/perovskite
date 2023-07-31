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

use std::{fmt::Debug, ops::RangeInclusive, hash::{Hash, Hasher}};

use anyhow::{ensure, Result};
use rustc_hash::FxHasher;

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct BlockCoordinate {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Debug for BlockCoordinate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("({}, {}, {})", self.x, self.y, self.z))
    }
}
impl BlockCoordinate {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
    #[inline]
    pub fn offset(&self) -> ChunkOffset {
        // rem_euclid(16) result should always fit into u8.
        ChunkOffset {
            x: self.x.rem_euclid(16) as u8,
            y: self.y.rem_euclid(16) as u8,
            z: self.z.rem_euclid(16) as u8,
        }
    }
    #[inline]
    pub fn chunk(&self) -> ChunkCoordinate {
        ChunkCoordinate {
            x: self.x.div_euclid(16),
            y: self.y.div_euclid(16),
            z: self.z.div_euclid(16),
        }
    }

    pub fn try_delta(&self, x: i32, y: i32, z: i32) -> Option<BlockCoordinate> {
        let x = self.x.checked_add(x)?;
        let y = self.y.checked_add(y)?;
        let z = self.z.checked_add(z)?;

        Some(BlockCoordinate { x, y, z })
    }
}
impl From<BlockCoordinate> for crate::protocol::coordinates::BlockCoordinate {
    fn from(value: BlockCoordinate) -> Self {
        crate::protocol::coordinates::BlockCoordinate {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}
impl From<&crate::protocol::coordinates::BlockCoordinate> for BlockCoordinate {
    fn from(value: &crate::protocol::coordinates::BlockCoordinate) -> Self {
        BlockCoordinate {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}
impl From<crate::protocol::coordinates::BlockCoordinate> for BlockCoordinate {
    fn from(value: crate::protocol::coordinates::BlockCoordinate) -> Self {
        (&value).into()
    }
}

#[inline]
fn try_convert(value: f64) -> Result<i32> {
    ensure!(value.is_finite(), "val was not finite");
    ensure!(
        value <= (i32::MAX as f64) && value >= (i32::MIN as f64),
        "Value is out of bounds as i32"
    );
    Ok(value.round() as i32)
}

impl TryFrom<cgmath::Vector3<f64>> for BlockCoordinate {
    type Error = anyhow::Error;

    fn try_from(value: cgmath::Vector3<f64>) -> std::result::Result<Self, Self::Error> {
        Ok(BlockCoordinate {
            x: try_convert(value.x)?,
            y: try_convert(value.y)?,
            z: try_convert(value.z)?,
        })
    }
}

/// Represents an offset of a block within a chunk.
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct ChunkOffset {
    pub x: u8,
    pub y: u8,
    pub z: u8,
}
impl ChunkOffset {
    #[inline(always)]
    fn debug_check(&self) {
        debug_assert!(self.x < 16);
        debug_assert!(self.y < 16);
        debug_assert!(self.z < 16);
    }
    #[inline]
    pub fn as_index(&self) -> usize {
        self.debug_check();
        256 * (self.z as usize) + 16 * (self.y as usize) + (self.x as usize)
    }
    #[inline]
    pub fn from_index(index: usize) -> ChunkOffset {
        assert!(index < 4096);
        ChunkOffset {
            x: (index % 16) as u8,
            y: ((index / 16) % 16) as u8,
            z: ((index / 256) % 16) as u8,
        }
    }
    pub fn try_delta(&self, x: i8, y: i8, z: i8) -> Option<ChunkOffset> {
        let x = self.x as i8 + x;
        let y = self.y as i8 + y;
        let z = self.z as i8 + z;
        if !(0..16).contains(&x) || !(0..16).contains(&y) || !(0..16).contains(&z) {
            None
        } else {
            Some(ChunkOffset {
                x: x as u8,
                y: y as u8,
                z: z as u8,
            })
        }
    }
}
impl Debug for ChunkOffset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("({}, {}, {})", self.x, self.y, self.z))
    }
}

/// Represents a location of a map chunk.
/// 
/// Each coordinate spans 16 blocks, covering the range [chunk_coord.x * 16, chunk_coord.x * 16 + 15].
/// e.g. chunk 0,1,2 covers x:[0, 15], y:[16, 31], z:[32, 47]
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub struct ChunkCoordinate {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}
impl ChunkCoordinate {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        let result = Self { x, y, z };
        assert!(result.is_in_bounds());
        result
    }

    /// Returns a new block coordinate with the given offset within this chunk.
    #[inline]
    pub fn with_offset(&self, offset: ChunkOffset) -> BlockCoordinate {
        offset.debug_check();
        BlockCoordinate {
            x: self.x * 16 + (offset.x as i32),
            y: self.y * 16 + (offset.y as i32),
            z: self.z * 16 + (offset.z as i32),
        }
    }
    /// Returns the Manhattan distance between the two coordinates
    pub fn manhattan_distance(&self, other: ChunkCoordinate) -> u32 {
        self.x
            .abs_diff(other.x)
            .saturating_add(self.y.abs_diff(other.y))
            .saturating_add(self.z.abs_diff(other.z))
    }
    /// Returns the L-infinity (max distance along all three dimensions) norm between the two coordinates
    pub fn l_infinity_norm_distance(&self, other: ChunkCoordinate) -> u32 {
        self.x
            .abs_diff(other.x)
            .max(self.y.abs_diff(other.y))
            .max(self.z.abs_diff(other.z))
    }
    /// Returns true if the coordinate is in-bounds. Because *block* coordinates need to
    /// fit into an i32, not every possible chunk coordinate is actually in-bounds.
    pub fn is_in_bounds(&self) -> bool {
        const BOUNDS_RANGE: RangeInclusive<i32> = (i32::MIN / 16)..=(i32::MAX / 16);
        BOUNDS_RANGE.contains(&self.x)
            && BOUNDS_RANGE.contains(&self.y)
            && BOUNDS_RANGE.contains(&self.z)
    }
    /// Adds the given offset to the coordinate, and returns it, if it is in-bounds.
    pub fn try_delta(&self, x: i32, y: i32, z: i32) -> Option<ChunkCoordinate> {
        let x = self.x.checked_add(x)?;
        let y = self.y.checked_add(y)?;
        let z = self.z.checked_add(z)?;
        let candidate = ChunkCoordinate { x, y, z };
        if candidate.is_in_bounds() {
            Some(candidate)
        } else {
            None
        }
    }
    /// Convenience helper to hash a ChunkCoordinate to a u64.
    /// The result is not guaranteed to be the same between versions or runs,
    /// and hence should not be persisted.
    pub fn hash_u64(&self) -> u64 {
        let mut hasher = FxHasher::default();
        self.hash(&mut hasher);
        hasher.finish()
    }
}
impl Debug for ChunkCoordinate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("({}, {}, {})", self.x, self.y, self.z))
    }
}

impl From<ChunkCoordinate> for crate::protocol::coordinates::ChunkCoordinate {
    fn from(value: ChunkCoordinate) -> Self {
        crate::protocol::coordinates::ChunkCoordinate {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}
impl From<&crate::protocol::coordinates::ChunkCoordinate> for ChunkCoordinate {
    fn from(value: &crate::protocol::coordinates::ChunkCoordinate) -> Self {
        ChunkCoordinate {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}
impl From<crate::protocol::coordinates::ChunkCoordinate> for ChunkCoordinate {
    fn from(value: crate::protocol::coordinates::ChunkCoordinate) -> Self {
        (&value).into()
    }
}

impl TryFrom<cgmath::Vector3<f64>> for crate::protocol::coordinates::Vec3D {
    type Error = anyhow::Error;

    fn try_from(value: cgmath::Vector3<f64>) -> std::result::Result<Self, Self::Error> {
        ensure!(
            value.x.is_finite() && value.y.is_finite() && value.z.is_finite(),
            "vec3D contained NaN or inf"
        );
        Ok(crate::protocol::coordinates::Vec3D {
            x: value.x,
            y: value.y,
            z: value.z,
        })
    }
}
impl TryFrom<&crate::protocol::coordinates::Vec3D> for cgmath::Vector3<f64> {
    type Error = anyhow::Error;

    fn try_from(
        value: &crate::protocol::coordinates::Vec3D,
    ) -> std::result::Result<Self, Self::Error> {
        ensure!(
            value.x.is_finite() && value.y.is_finite() && value.z.is_finite(),
            "vec3D contained NaN or inf"
        );
        Ok(cgmath::Vector3 {
            x: value.x,
            y: value.y,
            z: value.z,
        })
    }
}
impl TryFrom<crate::protocol::coordinates::Vec3D> for cgmath::Vector3<f64> {
    type Error = anyhow::Error;

    fn try_from(value: crate::protocol::coordinates::Vec3D) -> Result<Self> {
        (&value).try_into()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PlayerPositionUpdate {
    // The client-side tick for this position
    pub tick: u64,
    // The position, blocks
    pub position: cgmath::Vector3<f64>,
    // The velocity, blocks per second
    pub velocity: cgmath::Vector3<f64>,
    // The facing direction, normalized
    pub face_direction: (f64, f64),
}
