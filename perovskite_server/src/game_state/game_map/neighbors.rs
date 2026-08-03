use std::ops::Range;
use std::sync::atomic::Ordering;
use std::{fmt::Debug, time::Instant};

use anyhow::{Context, Result};
use bytemuck::cast_slice;
use perovskite_core::vertical_occlusion::Occlusions;
use perovskite_core::{
    block_id::BlockId,
    constants::{
        CHUNK_BITS, CHUNK_SIZE, CHUNK_SIZE_I32, EXTENDED_CHUNK_OFFSET, EXTENDED_CHUNK_VOLUME,
        EXTENDED_OVERLAP_RANGES_XZ, EXTENDED_OVERLAP_RANGES_Y,
    },
    coordinates::{BlockCoordinate, ChunkCoordinate, ChunkOffset, ChunkOffsetForOcclusionExt},
    sync::{GenericRwLock, SyncBackend},
    vertical_occlusion::{
        propagate_light_and_occlusion, ChunkBuffer, LightScratchpad, NeighborBuffer, OcclusionField,
    },
};
use smallvec::{smallvec, SmallVec};

use crate::{
    game_state::{blocks::BlockTypeManager, game_map::MapChunkHolder},
    BlockingRegionToken,
};

/// Provides access to a chunk, and a partial grid of its neighbors.
/// The chunk is whatever the chunk size is (see [`perovskite_core::constants::CHUNK_SIZE`]),
/// and [`perovskite_core::constants::EXTENDED_CHUNK_OFFSET`] worth of blocks in each direction
/// are also available - NOT 27 full chunks.
///
/// If you need 27 full chunks of CHUNK_SIZE, please file a feature request. So far, we have not
/// needed this; all natural neighbor interactions we've encountered so far are either adjacent (so
/// only one block of neighbors is needed), or related to light (so 16 blocks are needed).
///
/// In the future, it's possible that some neighbors will (configurably) only provide one block of
/// neighbor data, for those timers that only need that one adjacent neighbor to do their work.
pub struct ChunkNeighbors {
    // Intentionally a block coordinate, not a chunk coordinate, so we can do block-wise math directly
    // without needing a conversion.
    center: BlockCoordinate,
    presence_bitmap: u32,
    blocks: Box<[u32; EXTENDED_CHUNK_VOLUME]>,
    lightfields: Box<[OcclusionField; 3 * 3 * 3]>,
    weatherfields: Box<[OcclusionField; 3 * 3 * 3]>,
}

impl ChunkNeighbors {
    /// Get the neighbors of a chunk.
    /// Note: This is guaranteed to return the neighbors of the chunk in question, assuming that the chunk is loaded.
    ///
    /// Chunks that are *not* the neighbor may or may not be returned arbitrarily (due to optimizations in the timer engine). Do not
    /// rely on their presence. They may also be returned inconsistently (i.e. either before or after the timer callback's effects)
    pub fn get_block(&self, coord: BlockCoordinate) -> Option<BlockId> {
        let block = self.blocks[self.block_index(coord)?].into();
        Some(block)
    }

    fn neighbor_index(cx: i32, cz: i32, cy: i32) -> i32 {
        (cx + 1) * 9 + (cz + 1) * 3 + (cy + 1)
    }

    fn block_index(&self, coord: BlockCoordinate) -> Option<usize> {
        let dx = coord.x - self.center.x;
        let dz = coord.z - self.center.z;
        let dy = coord.y - self.center.y;
        const RANGE: Range<i32> = -EXTENDED_CHUNK_OFFSET..(CHUNK_SIZE_I32 + EXTENDED_CHUNK_OFFSET);
        if !RANGE.contains(&dx) || !RANGE.contains(&dz) || !RANGE.contains(&dy) {
            return None;
        }
        let cx = dx >> CHUNK_BITS;
        let cz = dz >> CHUNK_BITS;
        let cy = dy >> CHUNK_BITS;
        let neighbor_index = Self::neighbor_index(cx, cy, cz);
        if self.presence_bitmap & (1 << neighbor_index) == 0 {
            return None;
        } else {
            Some((dx, dy, dz).as_extended_index())
        }
    }

    pub(crate) fn populate_lighting(
        &mut self,
        block_ids: &BlockTypeManager,
        light: &mut LightScratchpad,
    ) {
        let adapter = ChunkNeighborsAdapter(self);
        propagate_light_and_occlusion(
            adapter,
            light,
            #[inline]
            |id| block_ids.allows_light_propagation(id),
            #[inline]
            |id| block_ids.allows_weather_propagation(id),
            #[inline]
            |id| block_ids.light_emission(id),
        );
    }
}
impl Default for ChunkNeighbors {
    fn default() -> Self {
        Self {
            center: BlockCoordinate::new(0, 0, 0),
            presence_bitmap: 0,
            blocks: bytemuck::zeroed_box(),
            lightfields: Box::new([OcclusionField::zero(); 27]),
            weatherfields: Box::new([OcclusionField::zero(); 27]),
        }
    }
}

impl Debug for ChunkNeighbors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChunkNeighbors")
            .field("center", &self.center)
            .field("presence_bitmap", &self.presence_bitmap)
            .finish()
    }
}

struct NeighborChunkProxy<'a> {
    blocks: &'a ChunkNeighbors,
    base_offset: (i32, i32, i32),
    y_start: i32,
}
impl ChunkBuffer for NeighborChunkProxy<'_> {
    fn get(&self, offset: ChunkOffset) -> BlockId {
        let index = (
            offset.x as i32 + self.base_offset.0,
            offset.y as i32 + self.base_offset.1,
            offset.z as i32 + self.base_offset.2,
        )
            .as_extended_index();
        BlockId(self.blocks.blocks[index])
    }

    fn vertical_slice(&self, x: u8, z: u8) -> &[BlockId] {
        let index = (
            x as i32 + self.base_offset.0,
            self.y_start + self.base_offset.1,
            z as i32 + self.base_offset.2,
        )
            .as_extended_index();

        let len = if self.base_offset.1 == -1 {
            EXTENDED_CHUNK_OFFSET as usize
        } else {
            CHUNK_SIZE
        };
        if index > self.blocks.blocks.len() - len {
            panic!(
                "Index out of bounds: index {}, len {}, size {}, base_offset {:?}, x/ys/z {:?}, center {}, bitmap {:x} -> index_bits {:?}",
                index,
                self.blocks.blocks.len(),
                CHUNK_SIZE,
                self.base_offset,
                (x, self.y_start, z),
                self.blocks.center,
                self.blocks.presence_bitmap,
                (
                    x as i32 + self.base_offset.0,
                    self.y_start + self.base_offset.1,
                    z as i32 + self.base_offset.2,
                )
            );
        }

        let subslice: &[BlockId] = cast_slice(&self.blocks.blocks[index..index + len]);
        subslice
    }
}

// private newtype to work around https://doc.rust-lang.org/error_codes/E0446.html
struct ChunkNeighborsAdapter<'a>(&'a ChunkNeighbors);

impl<'a> NeighborBuffer for ChunkNeighborsAdapter<'a> {
    type Chunk<'b>
        = NeighborChunkProxy<'b>
    where
        Self: 'b;

    fn get(&self, dx: i32, dy: i32, dz: i32) -> Option<Self::Chunk<'_>> {
        let neighbor_index = ChunkNeighbors::neighbor_index(dx, dy, dz);
        if self.0.presence_bitmap & (1 << neighbor_index) == 0 {
            None
        } else {
            Some(NeighborChunkProxy {
                blocks: self.0,
                base_offset: (
                    dx * CHUNK_SIZE_I32,
                    dy * CHUNK_SIZE_I32,
                    dz * CHUNK_SIZE_I32,
                ),
                y_start: EXTENDED_OVERLAP_RANGES_Y[(dy + 1) as usize].1.start,
            })
        }
    }

    fn inbound_light(&self, dx: i32, dy: i32, dz: i32) -> OcclusionField {
        self.0.lightfields[ChunkNeighbors::neighbor_index(dx, dy, dz) as usize]
    }

    fn inbound_weather(&self, dx: i32, dy: i32, dz: i32) -> OcclusionField {
        self.0.weatherfields[ChunkNeighbors::neighbor_index(dx, dy, dz) as usize]
    }
}

pub(super) fn build_neighbors<S: SyncBackend, L: SyncBackend>(
    neighbor_data: &mut ChunkNeighbors,
    center_coord: ChunkCoordinate,
    game_map: &super::ServerGameMap<S, L>,
    copy_data: bool,
    token: &BlockingRegionToken,
    mut interest_check: impl FnMut(&MapChunkHolder<S>) -> bool,
) -> Result<(bool, bool, Option<Instant>)> {
    let buf = &mut neighbor_data.blocks;
    let mut presence_bitmap = 0u32;
    let mut any_interests = false;
    let mut center_interest = false;
    let mut update_times: SmallVec<[_; 27]> = smallvec![];
    for (cx, x_fine_range, x_base) in EXTENDED_OVERLAP_RANGES_XZ {
        for (cz, z_fine_range, z_base) in EXTENDED_OVERLAP_RANGES_XZ {
            for (cy, y_fine_range, y_base) in EXTENDED_OVERLAP_RANGES_Y {
                if let Some(neighbor_coord) = center_coord.try_delta(cx, cy, cz) {
                    let shard = game_map.live_chunks[super::shard_id(neighbor_coord)].lock_read();
                    if let Some(neighbor_holder) = shard.chunks.get(&neighbor_coord) {
                        if interest_check(neighbor_holder) {
                            any_interests = true;
                            if cx == 0 && cy == 0 && cz == 0 {
                                center_interest = true;
                            }
                        }
                        update_times.push(neighbor_holder.last_written.get_acquire());

                        if let Some(contents) = neighbor_holder.try_get_read(token)? {
                            let neighbor_index = ChunkNeighbors::neighbor_index(cx, cy, cz);
                            presence_bitmap |= 1 << neighbor_index;
                            if copy_data {
                                let light_column = shard
                                    .light_columns
                                    .get(&(neighbor_coord.x, neighbor_coord.z))
                                    .with_context(|| {
                                        format!(
                                            "Missing lightmap for present chunk {:?}",
                                            neighbor_coord
                                        )
                                    })?;
                                let Occlusions { light, weather, .. } = light_column
                                    .get_incoming_light_and_weather(neighbor_coord.y)
                                    .unwrap_or(Occlusions::zero());

                                for x_fine in x_fine_range.clone().into_iter() {
                                    for z_fine in z_fine_range.clone().into_iter() {
                                        let src_offset = ChunkOffset::new(
                                            x_fine as u8,
                                            y_fine_range.start as u8,
                                            z_fine as u8,
                                        )
                                        .as_index();
                                        let dst_offset = (
                                            x_fine + x_base,
                                            y_fine_range.start + y_base,
                                            z_fine + z_base,
                                        )
                                            .as_extended_index();
                                        let len = y_fine_range.end - y_fine_range.start;

                                        for offset in 0..len {
                                            buf[dst_offset + offset as usize] = contents.block_ids
                                                [src_offset + offset as usize]
                                                .load(Ordering::Relaxed);
                                        }
                                    }
                                }

                                neighbor_data.lightfields[neighbor_index as usize] = light;
                                neighbor_data.weatherfields[neighbor_index as usize] = weather;
                            }
                        }
                    }
                }
            }
        }
    }
    neighbor_data.center = center_coord.with_offset(ChunkOffset { x: 0, y: 0, z: 0 });
    neighbor_data.presence_bitmap = presence_bitmap;
    Ok((
        any_interests,
        center_interest,
        update_times.into_iter().max(),
    ))
}

#[test]
fn test_build_neighbors() {
    use crate::server::testonly_in_memory;
    let server = testonly_in_memory().unwrap();
    let chunk_offset = -5;
    let offset = chunk_offset * CHUNK_SIZE_I32;
    server.run_task_in_server(|gs| {
        for i in -32..64 {
            gs.game_map()
                .set_block(
                    BlockCoordinate {
                        x: 12,
                        y: 3,
                        z: i + offset,
                    },
                    BlockId((i + 100) as u32),
                    None,
                )
                .unwrap();
        }

        let mut neighbors = ChunkNeighbors::default();
        let center_coord = ChunkCoordinate::new(0, 0, chunk_offset);
        let (_matches, _center_matches, _latest_update) = build_neighbors(
            &mut neighbors,
            center_coord,
            &gs.game_map(),
            true,
            &BlockingRegionToken,
            |_| true,
        )
        .unwrap();
        for i in -16..48 {
            print!(
                "{}={:?} ",
                i,
                neighbors
                    .get_block(BlockCoordinate::new(12, 3, i + offset))
                    .map(|x| x.0)
            );
            assert_eq!(
                neighbors.get_block(BlockCoordinate::new(12, 3, i + offset)),
                Some(BlockId::from((i + 100) as u32))
            );
        }
        println!();
    });
}
