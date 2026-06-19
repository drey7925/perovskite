use super::MapChunk;
use anyhow::Error;
use perovskite_core::block_id::BlockId;
use perovskite_core::constants::{
    CHUNK_BITS, CHUNK_SIZE, CHUNK_SIZE_I32, CHUNK_VOLUME, EXTENDED_CHUNK_OFFSET,
    EXTENDED_CHUNK_VOLUME, EXTENDED_OVERLAP_RANGES,
};
use perovskite_core::coordinates::ChunkOffsetForLightingExt;
use perovskite_core::lighting::{
    propagate_light, ChunkBuffer, LightScratchpad, Lightfield, NeighborBuffer,
};
use rand::distributions::Bernoulli;
use rand::prelude::Distribution;
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use smallvec::{smallvec, SmallVec};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::{
    sync::{Arc, Weak},
    time::{Duration, Instant},
};
use tokio_util::sync::CancellationToken;
use tracy_client::{plot, span};

use crate::game_state::game_map::{
    MapChunkHolder, MapChunkInnerWriteGuard, WritebackReq, NUM_CHUNK_SHARDS,
};
use crate::{run_handler, BlockingRegionToken, NonExhaustive};
use perovskite_core::sync::{DefaultSyncBackend, GenericRwLock, SyncBackend};

use crate::game_state::blocks::{BlockTypeManager, ExtendedDataHolder, InlineContext};
use crate::game_state::event::{EventInitiator, HandlerContext, InitiatorState};
use crate::game_state::GameState;

use anyhow::{ensure, Context, Result};
use bytemuck::cast_slice;
use perovskite_core::coordinates::{BlockCoordinate, ChunkCoordinate, ChunkOffset};
use tokio::{sync::mpsc, task::JoinHandle};

pub struct TimerState {
    pub prev_tick_time: Instant,
    pub current_tick_time: Instant,
}

struct ShardState {
    timer_state: TimerState,
    // pre-allocated here to avoid allocations in the timer path
    neighbor_buffer: Option<ChunkNeighbors>,
    lighting_buffer: Option<LightScratchpad>,
}

pub trait TimerInlineCallback: Send + Sync {
    /// Called once for each block on the map that matched the block type configured for the timer.
    /// This may be invoked concurrently for multiple blocks and multiple chunks.
    ///
    /// Args:
    /// * coordinate: Location of the block this is being called for
    /// * state: The ShardState for this run of the timer.
    /// * block_type: Mutable reference to the block type in the block.
    /// * data: Mutable reference to the extended data in the block.
    /// * ctx: Context for the callback
    fn inline_callback(
        &self,
        coordinate: BlockCoordinate,
        timer_state: &TimerState,
        block_type: &mut BlockId,
        data: &mut ExtendedDataHolder,
        ctx: &InlineContext,
    ) -> Result<()>;
}

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
    center: BlockCoordinate,
    presence_bitmap: u32,
    blocks: Box<[u32; EXTENDED_CHUNK_VOLUME]>,
    lightfields: Box<[Lightfield; 3 * 3 * 3]>,
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
        light: &mut Option<LightScratchpad>,
    ) {
        let light = light.get_or_insert_with(LightScratchpad::default);
        let adapter = ChunkNeighborsAdapter(self);
        propagate_light(
            adapter,
            light,
            #[inline]
            |id| block_ids.allows_light_propagation(id),
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
            lightfields: Box::new([Lightfield::zero(); 27]),
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

    fn vertical_slice(&self, x: u8, z: u8) -> &[BlockId; CHUNK_SIZE] {
        let index = (
            x as i32 + self.base_offset.0,
            self.y_start,
            z as i32 + self.base_offset.2,
        )
            .as_extended_index();
        if index > self.blocks.blocks.len() - CHUNK_SIZE {
            panic!(
                "Index out of bounds: {}, {}, {}, {:?}, {:?}, {}, {} -> {:?}",
                index,
                self.blocks.blocks.len(),
                CHUNK_SIZE,
                self.base_offset,
                (x, z),
                self.blocks.center,
                self.blocks.presence_bitmap,
                (
                    x as i32 + self.base_offset.0,
                    0 + self.base_offset.1,
                    z as i32 + self.base_offset.2,
                )
            );
        }
        let subslice: &[BlockId] = cast_slice(&self.blocks.blocks[index..index + CHUNK_SIZE]);
        subslice.try_into().unwrap()
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
                y_start: EXTENDED_OVERLAP_RANGES[(dy + 1) as usize].1.start,
            })
        }
    }

    fn inbound_light(&self, dx: i32, dy: i32, dz: i32) -> Lightfield {
        self.0.lightfields[ChunkNeighbors::neighbor_index(dx, dy, dz) as usize]
    }
}

pub trait BulkUpdateCallback: Send + Sync {
    /// Called once for each chunk that *might* contain one of the block types configured for this timer.
    /// *This is a probabilistic check, and the callback may be called even if a configured block type is not actually
    ///    present.*
    ///
    /// Performance tip: Iterating in x/z/y (y on the innermost loop) order is the most cache-friendly order possible.
    ///
    /// **Warning**: Trying to access the map via ctx can cause deadlocks.
    ///
    /// Args:
    ///   * ctx: Context. It is deadlock-prone to access inventories and block inventories. This
    ///         param might be removed in an upcoming cleanup.
    ///   * chunk_coordinate: The coord of the chunk in question
    ///   * timer_state: Details about the timer being run
    ///   * chunk: The writeable chunk that can be updated. Note that changes made via chunk are
    ///         not visible in `neighbors`.
    ///   * neighbors: For bulk update with neighbors, this contains neighbor data. This does not
    ///         show changes made via `chunk`.
    ///   * lights: For bulk update with neighbors when light computation is enabled. Does not show
    ///         changes made via `chunk`
    fn bulk_update_callback(
        &self,
        ctx: &HandlerContext<'_>,
        chunk_coordinate: ChunkCoordinate,
        timer_state: &TimerState,
        chunk: &mut MapChunk,
        neighbors: Option<&ChunkNeighbors>,
        lights: Option<&LightScratchpad>,
    ) -> Result<()>;
}

pub trait VerticalNeighborTimerCallback: Send + Sync {
    /// Called once for each chunk that *might* contain one of the block types configured for this timer.
    ///
    /// In particular, this will be called when either upper or lower contains the block type in question.
    ///
    /// *This is a probabilistic check, and the callback may be called even if a configured block type is not
    ///    present.*
    ///
    /// Performance tip: Iterating in x/z/y (y on the innermost loop) order is the most cache-friendly order possible.
    ///
    fn vertical_neighbor_callback(
        &self,
        ctx: &HandlerContext<'_>,
        upper: ChunkCoordinate,
        lower: ChunkCoordinate,
        upper_chunk: &mut MapChunk,
        lower_chunk: &mut MapChunk,
        timer_state: &TimerState,
    ) -> Result<()>;
}

pub enum TimerCallback {
    /// Callback operating on one block at a time. The engine may call it concurrently for multiple
    /// blocks in a chunk, or for multiple chunks. The timing of the callback may be changed between versions,
    /// but the engine will call it once per matching block per timer cycle.
    PerBlockLocked(Box<dyn TimerInlineCallback>),
    /// Callback operating on entire chunks. The engine may call it concurrently for multiple chunks.
    /// This callback will be called once per timer cycle, with the chunk locked for edit.
    ///
    /// No neighbor data is passed.
    BulkUpdate(Box<dyn BulkUpdateCallback>),
    /// Callback operating on entire chunks. The engine may call it concurrently for multiple chunks.
    /// This callback will be called once per timer cycle, with the chunk locked for edit.
    ///
    /// Neighbor data is passed.
    BulkUpdateWithNeighbors(Box<dyn BulkUpdateCallback>),
    /// A fast callback that gives you locked access to *two* vertically contiguous chunks
    /// at a time. The iteration order is top-to-bottom, and only vertically contiguous chunks
    /// are supported. This takes advantage of lighting-related acceleration structures.
    ///
    /// Note that the sharding policy of this timer may be significantly different from other
    /// timer types. In the current implementation, sharding is done based on vertical slices
    /// of the loaded map, rather than on a chunk-by-chunk basis.
    ///
    /// Experimental, subject to change (even more so than everything else in this crate)
    LockedVerticalNeighors(Box<dyn VerticalNeighborTimerCallback>),
}

/// Control for a map timer.
pub struct TimerSettings {
    /// The time between ticks of the timer
    pub interval: Duration,
    /// The number of shards
    pub shards: usize,
    /// How strongly to stagger the shards.
    /// 0.0 - fire all shards at approximately the same time
    /// 1.0 - spread the shards as far apart as possible
    /// Intermediate values - spread the shards out, but in a smaller span of time
    pub spreading: f64,
    /// The set of block types (matches ignoring variant) that this timer should act on
    pub block_types: Vec<BlockId>,
    /// If set, do *not* use block bloom filters to determine whether a block is present.
    /// This is useful for bulk update callbacks that might need to run in all chunks.
    pub ignore_block_type_presence_check: bool,
    /// The probability that the action will be taken for each matching block. Each matching block is
    /// sampled independently, using an unspecified RNG that does not derive from the game seed.
    ///
    /// **Warning:** Ignored for handlers that act on entire chunks (e.g. BulkUpdate or BulkUpdateWithNeighbors)
    pub per_block_probability: f64,
    /// If the bulk handler leaves a chunk unchanged, do not run the bulk handler for that chunk
    /// again until the next time the chunk is modified by external means.
    ///
    /// For bulk handlers with neighbors, the handler will run if the chunk or any neighbors have been modified.
    ///
    /// **Warning:** For per-block actions with a probability, this will idle the timer for a chunk
    /// if eligible blocks are present, but not selected due to the probability.
    pub idle_chunk_after_unchanged: bool,
    /// For bulk update handlers, whether lighting data is populated into the neighbor buffer.
    /// * Only applies for bulk handlers w/ neighbors
    pub populate_lighting: bool,
    /// For a bulk callback with neighbors: If true, relevant blocks in neignbor chunks count
    /// for the block type presence check. Otherwise, only chunks in the current block are considered.
    pub include_neighbors_in_block_presence_check: bool,
    pub _ne: NonExhaustive,
}
impl Default for TimerSettings {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(1),
            shards: 8,
            spreading: 1.0,
            block_types: Default::default(),
            ignore_block_type_presence_check: false,
            per_block_probability: 1.0,
            idle_chunk_after_unchanged: false,
            populate_lighting: false,
            include_neighbors_in_block_presence_check: false,
            _ne: NonExhaustive(()),
        }
    }
}

pub(super) struct GameMapTimer {
    // The name of the timer; in the future this will be used for catching up tasks on unloaded chunks
    name: String,
    // The action to take on each block when the timer fires
    callback: TimerCallback,
    settings: TimerSettings,
    cancellation: CancellationToken,
    // This mutex must be held across an await (for each handle), so we use a tokio async mutex here.
    tasks: tokio::sync::Mutex<Vec<JoinHandle<Result<()>>>>,

    block_types: FxHashSet<u32>,
}
impl GameMapTimer {
    fn spawn_shards(
        self: &Arc<Self>,
        game_state: Arc<GameState>,
        fine_shards_per_coarse: usize,
    ) -> Result<()> {
        ensure!(!self.cancellation.is_cancelled());
        let mut tasks = self.tasks.blocking_lock();
        // Hacky: Delay 0.1 seconds to allow shards to start up
        let first_run = Instant::now() + Duration::from_millis(100);
        for coarse_shard in 0..NUM_CHUNK_SHARDS {
            for fine_shard in 0..fine_shards_per_coarse {
                let start_time = first_run
                    + (self.settings.interval.mul_f64(
                        self.settings.spreading
                            * (fine_shard + fine_shards_per_coarse * coarse_shard) as f64
                            / (fine_shards_per_coarse * NUM_CHUNK_SHARDS) as f64,
                    ));
                {
                    let cloned_self = self.clone();
                    let cloned_game_state = game_state.clone();
                    tasks.push(crate::spawn_async(
                        &format!("timer_{}_shard_{}", self.name, fine_shard),
                        // TODO error-check this
                        // It's brittle on shutdown due to closed channels
                        // We should probably shut down the timers before shutting down the rest of the map
                        async move {
                            cloned_self
                                .run_shard(
                                    start_time,
                                    coarse_shard,
                                    fine_shard,
                                    fine_shards_per_coarse,
                                    cloned_game_state,
                                )
                                .await
                        },
                    )?);
                }
            }
        }

        Ok(())
    }
    #[tracing::instrument(
        name = "timer_shard",
        level = "trace",
        skip(self, start_time, coarse_shard, fine_shard, fine_shards_per_coarse, game_state),
        fields(
            timer_name = %self.name,
        )
    )]
    async fn run_shard(
        &self,
        start_time: Instant,
        coarse_shard: usize,
        fine_shard: usize,
        fine_shards_per_coarse: usize,
        game_state: Arc<GameState>,
    ) -> Result<()> {
        let mut interval = tokio::time::interval_at(start_time.into(), self.settings.interval);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut shard_state = ShardState {
            timer_state: TimerState {
                prev_tick_time: start_time - self.settings.interval,
                current_tick_time: Instant::now(),
            },
            neighbor_buffer: None,
            lighting_buffer: None,
        };

        // Todo detect skipped ticks and adjust accordingly
        while !self.cancellation.is_cancelled() {
            tokio::select! {
                _ = interval.tick() => {
                    self.do_tick_work(coarse_shard, fine_shard, fine_shards_per_coarse, game_state.clone(), &mut shard_state)?;
                },
                _ = self.cancellation.cancelled() => {
                    break;
                },
            }
        }
        Ok(())
    }

    fn do_tick_work(
        &self,
        coarse_shard: usize,
        fine_shard: usize,
        fine_shards_per_coarse: usize,
        game_state: Arc<GameState>,
        shard_state: &mut ShardState,
    ) -> Result<()> {
        let current_tick_start = Instant::now();
        shard_state.timer_state.current_tick_time = current_tick_start;

        crate::block_in_place(|token| {
            self.delegate_locking_path(
                coarse_shard,
                fine_shard,
                fine_shards_per_coarse,
                game_state,
                shard_state,
                token,
            )
        })?;
        shard_state.timer_state.prev_tick_time = current_tick_start;
        Ok(())
    }

    #[cfg(any(test, feature = "test-support", doctest))]
    fn run_inline(&self, game_state: Arc<GameState>) -> Result<()> {
        let mut fake_shard_state = ShardState {
            timer_state: TimerState {
                prev_tick_time: Instant::now() - self.settings.interval,
                current_tick_time: Instant::now(),
            },
            neighbor_buffer: None,
            lighting_buffer: None,
        };

        for coarse_shard in 0..NUM_CHUNK_SHARDS {
            for fine_shard in 0..self.settings.shards {
                self.do_tick_work(
                    coarse_shard,
                    fine_shard,
                    self.settings.shards,
                    game_state.clone(),
                    &mut fake_shard_state,
                )?;
            }
        }

        Ok(())
    }

    fn do_vertical_neighbor_locking(
        &self,
        coarse_shard: usize,
        fine_shard: usize,
        fine_shards_per_coarse: usize,
        game_state: Arc<GameState>,
        state: &mut ShardState,
        token: &BlockingRegionToken,
    ) -> Result<()> {
        const MAX_TIME_UNDER_SHARD_LOCK: Duration = Duration::from_millis(50);
        let _span = span!("timer tick hand-over-hand");
        let mut writeback_permit = Some(game_state.game_map().get_writeback_permit(coarse_shard)?);
        let mut writeback_permit2 = Some(game_state.game_map().get_writeback_permit(coarse_shard)?);

        // Read the 2D slice coords we will work on, and then unlock
        let mut read_lock = {
            let _span = span!("acquire game_map read lock");
            game_state.game_map().live_chunks[coarse_shard].read()
        };
        let mut last_unlock_time = Instant::now();

        let mut coords = {
            let _span = span!("read and filter chunks");
            read_lock
                .light_columns
                .keys()
                .filter(|&coord| {
                    let mut hasher = FxHasher::default();
                    coord.hash(&mut hasher);
                    hasher.finish() % fine_shards_per_coarse as u64 == fine_shard as u64
                })
                .copied()
                .collect::<Vec<_>>()
        };
        // Basic sort to try to increase locality a bit
        coords.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        for (x, z) in coords.into_iter() {
            {
                // we have a light column
                let light_col = match read_lock.light_columns.get(&(x, z)) {
                    Some(light_col) => light_col.copy_keys(),
                    None => {
                        continue;
                    }
                };
                // It's a bit of a pain to get safe deadlock-free borrows to work correctly if we actually use the chunk cursor.
                // To be clear, this is not a limitation of the borrow-checker; it is a limitation of the logic I managed to
                // come up with when I tried the nifty approach here

                // array_windows is unstable, so we have to do this manually
                for i in (0..light_col.len() - 1).rev() {
                    let upper_y = light_col[i + 1];
                    let lower_y = light_col[i];
                    if upper_y == (lower_y + 1) {
                        // We can actually work on a chunk
                        if writeback_permit.is_none() {
                            writeback_permit = Some(reacquire_writeback_permit::<
                                DefaultSyncBackend,
                                DefaultSyncBackend,
                            >(
                                &game_state, coarse_shard, &mut read_lock
                            )?);
                        }
                        if writeback_permit2.is_none() {
                            writeback_permit2 = Some(reacquire_writeback_permit::<
                                DefaultSyncBackend,
                                DefaultSyncBackend,
                            >(
                                &game_state, coarse_shard, &mut read_lock
                            )?);
                        }
                        // Lock ordering: It's important that we lock these upper-to-lower
                        // TODO: consider whether true hand-over-hand improves performance.
                        //    Answer: probably not, and it's potentially deadlock-prone.
                        // Note that https://github.com/Amanieu/parking_lot/issues/505 does not
                        //    apply here - we have a shard lock.
                        let upper_coord = ChunkCoordinate::new(x, upper_y, z);
                        let lower_coord = ChunkCoordinate::new(x, lower_y, z);
                        let upper_chunk = match read_lock.chunks.get(&upper_coord) {
                            Some(upper_chunk) => upper_chunk,
                            None => {
                                continue;
                            }
                        };
                        let lower_chunk = match read_lock.chunks.get(&lower_coord) {
                            Some(lower_chunk) => lower_chunk,
                            None => {
                                continue;
                            }
                        };
                        let passed_block_presence = self.settings.ignore_block_type_presence_check
                            || self.settings.block_types.iter().any(|x| {
                                upper_chunk
                                    .block_bloom_filter
                                    .maybe_contains(x.base_id() as u64)
                            })
                            || self.settings.block_types.iter().any(|x| {
                                lower_chunk
                                    .block_bloom_filter
                                    .maybe_contains(x.base_id() as u64)
                            });
                        if !passed_block_presence {
                            continue;
                        }
                        let last_update = upper_chunk
                            .last_written
                            .get_acquire()
                            .max(lower_chunk.last_written.get_acquire());
                        let should_run = !self.settings.idle_chunk_after_unchanged
                            || last_update >= state.timer_state.prev_tick_time;
                        if should_run {
                            self.handle_chunk_vertical_pairs(
                                upper_coord,
                                lower_coord,
                                upper_chunk,
                                lower_chunk,
                                &game_state,
                                &mut writeback_permit,
                                &mut writeback_permit2,
                                state,
                                token,
                            )?;
                        }

                        if last_unlock_time.elapsed() > MAX_TIME_UNDER_SHARD_LOCK {
                            let _span = span!("timer bumping read lock");
                            parking_lot::RwLock::bump_read(&mut read_lock);
                            last_unlock_time = Instant::now();
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn do_tick_locking_with_neighbors(
        &self,
        coarse_shard: usize,
        fine_shard: usize,
        fine_shards_per_coarse: usize,
        game_state: Arc<GameState>,
        state: &mut ShardState,
        token: &BlockingRegionToken,
    ) -> Result<()> {
        let _span = span!("timer tick with neighbors");
        let mut writeback_permit = Some(game_state.game_map().get_writeback_permit(coarse_shard)?);

        // Read the coords, and then unlock.
        let mut coords = {
            let _span = span!("read and filter chunks");
            game_state.game_map().live_chunks[coarse_shard]
                .read()
                .chunks
                .keys()
                .filter(|&coord| {
                    coord.hash_u64() % fine_shards_per_coarse as u64 == fine_shard as u64
                })
                .copied()
                .collect::<Vec<_>>()
        };
        // Basic sort to try to increase locality a bit
        coords.sort_unstable_by(|a, b| {
            a.x.cmp(&b.x)
                .then_with(|| a.z.cmp(&b.z))
                .then_with(|| a.y.cmp(&b.y))
        });
        plot!("timer tick coords", coords.len() as f64);

        for coord in coords.into_iter() {
            if writeback_permit.is_none() {
                // We don't hold a read lock, so this doesn't risk deadlock
                writeback_permit = Some(game_state.game_map().get_writeback_permit(coarse_shard)?);
            }
            // this does the locking twice, with the benefit that it elides the memory copying
            // associated with build_neighbors if we don't end up actually using that neighbor data
            let (matches, center_matches, latest_update) = self.build_neighbors(
                &mut state.neighbor_buffer,
                coord,
                game_state.game_map(),
                false,
                token,
            )?;

            let actually_matches = if self.settings.include_neighbors_in_block_presence_check {
                matches
            } else {
                center_matches
            };

            let should_run = (!self.settings.idle_chunk_after_unchanged
                || latest_update.is_some_and(|x| x >= state.timer_state.prev_tick_time))
                && (actually_matches || self.settings.ignore_block_type_presence_check);

            if should_run {
                let (_, _, _) = self.build_neighbors(
                    &mut state.neighbor_buffer,
                    coord,
                    game_state.game_map(),
                    true,
                    token,
                )?;
                if self.settings.populate_lighting {
                    state
                        .neighbor_buffer
                        .as_mut()
                        .context("Neighbor buffer was None after building")?
                        .populate_lighting(game_state.block_types(), &mut state.lighting_buffer);
                }
                let shard = game_state.game_map().live_chunks[coarse_shard].read();
                if let Some(holder) = shard.chunks.get(&coord) {
                    if let Some(mut chunk) = holder.try_get_write(token)? {
                        match &self.callback {
                            TimerCallback::BulkUpdateWithNeighbors(_) => {
                                self.run_bulk_handler(
                                    &game_state,
                                    holder,
                                    &mut chunk,
                                    coord,
                                    state.neighbor_buffer.as_ref(),
                                    state.lighting_buffer.as_ref(),
                                    state,
                                    &mut writeback_permit,
                                    token,
                                )?;
                            }
                            _ => unreachable!(),
                        }
                        // if chunk.dirty {
                        //     // This has a small missed optimization - until the chunk is written back, this
                        //     // will keep firing. If further optimizations is needed, track whether
                        //     // *this* bulk updater modified the chunk, and use that for setting
                        //     // the last_written timestamp
                        //     *holder.last_written.lock() = state.timer_state.current_tick_time;
                        //     writeback_permit
                        //         .take()
                        //         .unwrap()
                        //         .send(WritebackReq::Chunk(coord));
                        // }
                    }
                }
            }
        }

        Ok(())
    }

    // Picks the locking path based on the type of callback
    fn delegate_locking_path(
        &self,
        coarse_shard: usize,
        fine_shard: usize,
        fine_shards_per_coarse: usize,
        game_state: Arc<GameState>,
        state: &mut ShardState,
        token: &BlockingRegionToken,
    ) -> Result<()> {
        match &self.callback {
            TimerCallback::PerBlockLocked(_) | TimerCallback::BulkUpdate(_) => self
                .do_tick_fast_lock_path(
                    coarse_shard,
                    fine_shard,
                    fine_shards_per_coarse,
                    game_state,
                    state,
                    token,
                ),
            TimerCallback::BulkUpdateWithNeighbors(_) => self.do_tick_locking_with_neighbors(
                coarse_shard,
                fine_shard,
                fine_shards_per_coarse,
                game_state,
                state,
                token,
            ),
            TimerCallback::LockedVerticalNeighors(_) => self.do_vertical_neighbor_locking(
                coarse_shard,
                fine_shard,
                fine_shards_per_coarse,
                game_state,
                state,
                token,
            ),
        }
    }

    // Fast lock path - no inter-chunk interactions allowed
    fn do_tick_fast_lock_path(
        &self,
        coarse_shard: usize,
        fine_shard: usize,
        fine_shards_per_coarse: usize,
        game_state: Arc<GameState>,
        state: &ShardState,
        token: &BlockingRegionToken,
    ) -> Result<()> {
        let _span = span!("timer tick fast");
        let mut writeback_permit = Some(game_state.game_map().get_writeback_permit(coarse_shard)?);
        let mut read_lock = {
            let _span = span!("acquire game_map read lock");
            game_state.game_map().live_chunks[coarse_shard].read()
        };
        let coords = read_lock
            .chunks
            .keys()
            .filter(|x| x.hash_u64() % fine_shards_per_coarse as u64 == fine_shard as u64)
            .cloned()
            .collect::<Vec<_>>();
        plot!("timer tick coords", coords.len() as f64);
        for (i, coord) in coords.into_iter().enumerate() {
            if writeback_permit.is_none() {
                writeback_permit = Some(reacquire_writeback_permit::<
                    DefaultSyncBackend,
                    DefaultSyncBackend,
                >(
                    &game_state, coarse_shard, &mut read_lock
                )?);
            } else if i % 100 == 0 {
                let _span = span!("timer bumping");
                parking_lot::RwLock::bump_read(&mut read_lock);
            }
            if let Some(chunk) = read_lock.chunks.get(&coord) {
                if self.settings.ignore_block_type_presence_check
                    || self
                        .settings
                        .block_types
                        .iter()
                        .any(|x| chunk.block_bloom_filter.maybe_contains(x.base_id() as u64))
                {
                    let last_update = chunk.last_written.get_acquire();
                    let should_run = !self.settings.idle_chunk_after_unchanged
                        || last_update >= state.timer_state.prev_tick_time;
                    if should_run {
                        self.handle_chunk_no_neighbors(
                            coord,
                            chunk,
                            &game_state,
                            &mut writeback_permit,
                            state,
                            token,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    fn handle_chunk_no_neighbors(
        &self,
        coord: ChunkCoordinate,
        holder: &MapChunkHolder<DefaultSyncBackend>,
        game_state: &Arc<GameState>,
        writeback_permit: &mut Option<mpsc::Permit<'_, WritebackReq>>,
        state: &ShardState,
        token: &BlockingRegionToken,
    ) -> Result<()> {
        assert!(writeback_permit.is_some());

        if let Some(mut chunk) = holder.try_get_write(token)? {
            match &self.callback {
                TimerCallback::PerBlockLocked(_) => {
                    self.run_per_block_handler(game_state, chunk, holder, coord, state, token)?;
                }
                TimerCallback::BulkUpdate(_) => {
                    let chunk_update = holder.last_written.get_acquire();
                    if !self.settings.idle_chunk_after_unchanged
                        || chunk_update >= state.timer_state.prev_tick_time
                    {
                        self.run_bulk_handler(
                            game_state,
                            holder,
                            &mut chunk,
                            coord,
                            None,
                            None,
                            state,
                            writeback_permit,
                            token,
                        )?;
                    }
                }
                TimerCallback::BulkUpdateWithNeighbors(_) => {
                    unreachable!()
                }
                TimerCallback::LockedVerticalNeighors(_) => {
                    unreachable!()
                }
            }
        }

        Ok(())
    }

    fn handle_chunk_vertical_pairs(
        &self,
        upper_coord: ChunkCoordinate,
        lower_coord: ChunkCoordinate,
        upper_holder: &MapChunkHolder<DefaultSyncBackend>,
        lower_holder: &MapChunkHolder<DefaultSyncBackend>,
        game_state: &Arc<GameState>,
        upper_writeback_permit: &mut Option<mpsc::Permit<'_, WritebackReq>>,
        lower_writeback_permit: &mut Option<mpsc::Permit<'_, WritebackReq>>,
        state: &ShardState,
        token: &BlockingRegionToken,
    ) -> Result<()> {
        assert!(upper_writeback_permit.is_some());
        assert!(lower_writeback_permit.is_some());
        let mut upper_chunk = match upper_holder.try_get_write(token)? {
            Some(x) => x,
            None => {
                return Ok(());
            }
        };
        let mut lower_chunk = match lower_holder.try_get_write(token)? {
            Some(x) => x,
            None => {
                return Ok(());
            }
        };
        let upper_old_block_ids = upper_chunk.clone_block_ids();
        let lower_old_block_ids = lower_chunk.clone_block_ids();

        let ctx = HandlerContext {
            tick: 0,
            initiator: EventInitiator::Engine,
            game_state: game_state.clone(),
            initiator_state: InitiatorState::default(),
        };

        match &self.callback {
            TimerCallback::LockedVerticalNeighors(x) => {
                run_handler!(
                    || x.vertical_neighbor_callback(
                        &ctx,
                        upper_coord,
                        lower_coord,
                        &mut upper_chunk,
                        &mut lower_chunk,
                        &state.timer_state,
                    ),
                    "vertical_neighbor_timer",
                    &EventInitiator::Engine
                )?;
            }
            _ => {
                unreachable!()
            }
        }
        self.reconcile_after_bulk_handler(
            &upper_old_block_ids,
            &mut upper_chunk,
            upper_holder,
            game_state,
            upper_coord,
            upper_writeback_permit,
            state.timer_state.current_tick_time,
        )?;
        self.reconcile_after_bulk_handler(
            &lower_old_block_ids,
            &mut lower_chunk,
            lower_holder,
            game_state,
            lower_coord,
            lower_writeback_permit,
            state.timer_state.current_tick_time,
        )?;

        Ok(())
    }

    fn run_per_block_handler(
        &self,
        game_state: &Arc<GameState>,
        mut chunk: MapChunkInnerWriteGuard<'_, DefaultSyncBackend>,
        holder: &MapChunkHolder<DefaultSyncBackend>,
        coord: ChunkCoordinate,
        state: &ShardState,
        token: &BlockingRegionToken,
    ) -> Result<(), Error> {
        let mut rng = rand::thread_rng();
        let sampler = Bernoulli::new(self.settings.per_block_probability)?;
        let map = game_state.game_map();
        for i in 0..CHUNK_VOLUME {
            let block_id = BlockId(chunk.block_ids[i].load(Ordering::Relaxed));
            assert!(holder
                .block_bloom_filter
                .maybe_contains(block_id.base_id() as u64));
            if self.block_types.contains(&block_id.base_id()) && sampler.sample(&mut rng) {
                match self.run_per_block_callback(
                    holder,
                    &mut chunk,
                    ChunkOffset::from_index(i),
                    map,
                    coord,
                    game_state,
                    state,
                    token,
                ) {
                    Ok(()) => {
                        // continue
                    }
                    Err(e) => {
                        log::error!("Timer callback {} failed: {:?}", self.name, e);
                    }
                }
            }
        }
        Ok(())
    }

    fn run_per_block_callback<S: SyncBackend>(
        &self,
        holder: &MapChunkHolder<S>,
        chunk: &mut MapChunkInnerWriteGuard<'_, S>,
        offset: ChunkOffset,
        map: &super::ServerGameMap<S>,
        coord: ChunkCoordinate,
        game_state: &Arc<GameState>,
        state: &ShardState,
        token: &BlockingRegionToken,
    ) -> Result<()> {
        match &self.callback {
            TimerCallback::PerBlockLocked(cb) => {
                map.mutate_block_atomically_locked(
                    holder,
                    chunk,
                    coord.with_offset(offset),
                    |block_id, extended_data| {
                        let ctx = InlineContext {
                            // todo actual ticks
                            tick: 0,
                            initiator: EventInitiator::Engine,
                            location: coord.with_offset(offset),
                            block_types: game_state.game_map().block_type_manager(),
                            items: game_state.item_manager(),
                            deferred_actions: smallvec::SmallVec::new(),
                        };
                        run_handler!(
                            || cb.inline_callback(
                                coord.with_offset(offset),
                                &state.timer_state,
                                block_id,
                                extended_data,
                                &ctx
                            ),
                            "timer_inline_locked",
                            &EventInitiator::Engine
                        )?;
                        static WARNING_LOGGED_ALREADY: AtomicBool = AtomicBool::new(false);
                        if !ctx.deferred_actions.is_empty() {
                            if WARNING_LOGGED_ALREADY.swap(true, Ordering::Relaxed) {
                                log::error!(
                                    "Fixme: timers have inefficient deferred action implementation"
                                );
                            }
                            let gs_clone = game_state.clone();
                            tokio::task::spawn_blocking(move || {
                                let ctx2 = HandlerContext {
                                    tick: 0,
                                    initiator: EventInitiator::Engine,
                                    game_state: gs_clone,
                                    initiator_state: InitiatorState::default(),
                                };
                                for action in ctx.deferred_actions {
                                    let res = run_handler!(
                                        || action(&ctx2),
                                        "timer_deferred",
                                        &EventInitiator::Engine
                                    );
                                    if let Err(e) = res {
                                        log::error!("Timer deferred action failed: {}", e);
                                    }
                                }
                            });
                        }

                        Ok(())
                    },
                    map,
                    super::BroadcastMode::DoBroadcast,
                    token,
                )?;
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    async fn await_shutdown(&self) -> Result<()> {
        for task in self.tasks.lock().await.drain(..) {
            task.await??;
        }
        Ok(())
    }

    fn run_bulk_handler<S: SyncBackend>(
        &self,
        game_state: &Arc<GameState>,
        holder: &MapChunkHolder<S>,
        chunk: &mut MapChunkInnerWriteGuard<'_, S>,
        coord: ChunkCoordinate,
        neighbor_data: Option<&ChunkNeighbors>,
        light_data: Option<&LightScratchpad>,
        state: &ShardState,
        permit: &mut Option<mpsc::Permit<'_, WritebackReq>>,
        _token: &BlockingRegionToken,
    ) -> Result<()> {
        let old_block_ids: Box<[u32; CHUNK_VOLUME]> = chunk.clone_block_ids();
        let ctx = HandlerContext {
            tick: 0,
            initiator: EventInitiator::Engine,
            game_state: game_state.clone(),
            initiator_state: InitiatorState::default(),
        };
        match &self.callback {
            TimerCallback::BulkUpdate(cb) => {
                assert!(neighbor_data.is_none());
                run_handler!(
                    || cb.bulk_update_callback(
                        &ctx,
                        coord,
                        &state.timer_state,
                        chunk,
                        neighbor_data,
                        None,
                    ),
                    "timer_bulk_update",
                    &EventInitiator::Engine
                )?;
            }
            TimerCallback::BulkUpdateWithNeighbors(cb) => {
                assert!(neighbor_data.is_some());
                run_handler!(
                    || cb.bulk_update_callback(
                        &ctx,
                        coord,
                        &state.timer_state,
                        chunk,
                        neighbor_data,
                        light_data,
                    ),
                    "timer_bulk_update_with_neighbors",
                    &EventInitiator::Engine
                )?;
            }
            _ => unreachable!(),
        };
        self.reconcile_after_bulk_handler(
            &old_block_ids,
            chunk,
            holder,
            game_state,
            coord,
            permit,
            state.timer_state.current_tick_time,
        )?;
        Ok(())
    }

    fn build_neighbors(
        &self,
        neighbor_data: &mut Option<ChunkNeighbors>,
        center_coord: ChunkCoordinate,
        game_map: &super::ServerGameMap<impl SyncBackend>,
        copy_data: bool,
        token: &BlockingRegionToken,
    ) -> Result<(bool, bool, Option<Instant>)> {
        let neighbor_data = neighbor_data.get_or_insert_with(Default::default);

        let buf = &mut neighbor_data.blocks;
        let mut presence_bitmap = 0u32;
        let mut any_blooms_match = false;
        let mut center_bloom_matches = false;
        let mut update_times: SmallVec<[_; 27]> = smallvec![];
        for (cx, x_fine_range, x_base) in EXTENDED_OVERLAP_RANGES {
            for (cz, z_fine_range, z_base) in EXTENDED_OVERLAP_RANGES {
                for (cy, y_fine_range, y_base) in EXTENDED_OVERLAP_RANGES {
                    if let Some(neighbor_coord) = center_coord.try_delta(cx, cy, cz) {
                        let shard =
                            game_map.live_chunks[super::shard_id(neighbor_coord)].lock_read();
                        if let Some(neighbor_holder) = shard.chunks.get(&neighbor_coord) {
                            if self.settings.block_types.iter().any(|x| {
                                neighbor_holder
                                    .block_bloom_filter
                                    .maybe_contains(x.base_id() as u64)
                            }) {
                                any_blooms_match = true;
                                if cx == 0 && cy == 0 && cz == 0 {
                                    center_bloom_matches = true;
                                }
                            }
                            update_times.push(neighbor_holder.last_written.get_acquire());

                            if let Some(contents) = neighbor_holder.try_get_read(token)? {
                                let neighbor_index = ChunkNeighbors::neighbor_index(cx, cy, cz);
                                presence_bitmap |= 1 << neighbor_index;
                                if copy_data {
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
                                                buf[dst_offset + offset as usize] = contents
                                                    .block_ids
                                                    [src_offset + offset as usize]
                                                    .load(Ordering::Relaxed);
                                            }
                                        }
                                    }

                                    let light_column = shard
                                        .light_columns
                                        .get(&(neighbor_coord.x, neighbor_coord.z))
                                        .with_context(|| {
                                            format!(
                                                "Missing lightmap for present chunk {:?}",
                                                neighbor_coord
                                            )
                                        })?;
                                    let light = light_column
                                        .get_incoming_light(neighbor_coord.y)
                                        .unwrap_or(Lightfield::zero());
                                    neighbor_data.lightfields[neighbor_index as usize] = light;
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
            any_blooms_match,
            center_bloom_matches,
            update_times.into_iter().max(),
        ))
    }

    fn reconcile_after_bulk_handler<S: SyncBackend>(
        &self,
        old_block_ids: &[u32; CHUNK_VOLUME],
        chunk: &mut MapChunkInnerWriteGuard<'_, S>,
        holder: &MapChunkHolder<S>,
        game_state: &Arc<GameState>,
        coord: ChunkCoordinate,
        permit: &mut Option<mpsc::Permit<'_, WritebackReq>>,
        _update_time: Instant,
    ) -> Result<()> {
        let mut seen_blocks = FxHashSet::default();
        let mut any_updated = false;
        let mut updates = vec![];
        for i in 0..CHUNK_VOLUME {
            let old_block_id = BlockId::from(old_block_ids[i]);
            let new_block_id = BlockId::from(chunk.block_ids[i].load(Ordering::Relaxed));
            let may_have_client_ext = game_state
                .block_types()
                .has_client_side_extended_data(new_block_id);
            if old_block_id != new_block_id || may_have_client_ext {
                if seen_blocks.insert(new_block_id.base_id()) {
                    // this generates expensive `LOCK OR %rax(%r8,%rdx,8) as well as an expensive DIV
                    // on x86_64. Without forking the library, it seems unavoidable
                    holder
                        .block_bloom_filter
                        .insert(new_block_id.base_id() as u64);
                }
                chunk.dirty = true;
                any_updated = true;
                let new_ext = match chunk.extended_data.get(&(i as super::OffsetIndex)) {
                    None => None,
                    Some(x) => {
                        if may_have_client_ext {
                            super::client_serialize_inner(
                                chunk.coord.with_offset(ChunkOffset::from_index(i)),
                                x,
                                game_state.block_types(),
                                new_block_id,
                            )?
                        } else {
                            None
                        }
                    }
                };
                if updates.len() < AGGREGATE_UPDATES_THRESHOLD {
                    let update = super::BlockUpdate {
                        location: coord.with_offset(ChunkOffset::from_index(i)),
                        id: new_block_id,
                        old_id: old_block_id,
                        new_ext_data: new_ext,
                    };
                    updates.push(update);
                }
            }
        }

        if updates.len() < AGGREGATE_UPDATES_THRESHOLD {
            for update in updates {
                game_state.game_map().broadcast_block_id_change(update);
            }
        } else {
            game_state
                .game_map()
                .broadcast_bulk_chunk_update(chunk.coord);
        }

        if any_updated {
            permit.take().unwrap().send(WritebackReq::Chunk(coord));
            holder.last_written.update_now_release();
        }
        Ok(())
    }
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

        struct DummyCallback;
        impl TimerInlineCallback for DummyCallback {
            fn inline_callback(
                &self,
                _coordinate: BlockCoordinate,
                _timer_state: &TimerState,
                _block_type: &mut BlockId,
                _data: &mut ExtendedDataHolder,
                _ctx: &InlineContext,
            ) -> Result<()> {
                Ok(())
            }
        }

        let timer = GameMapTimer {
            name: "Test".to_string(),
            block_types: FxHashSet::from_iter(vec![0]),
            cancellation: CancellationToken::new(),
            callback: TimerCallback::PerBlockLocked(Box::new(DummyCallback)),
            settings: TimerSettings {
                interval: Duration::from_secs(1),
                shards: 1,
                ignore_block_type_presence_check: true,
                ..Default::default()
            },
            tasks: vec![].into(),
        };
        let mut neighbors = Some(ChunkNeighbors::default());
        let center_coord = ChunkCoordinate::new(0, 0, chunk_offset);
        let (_matches, _center_matches, _latest_update) = timer
            .build_neighbors(
                &mut neighbors,
                center_coord,
                &gs.game_map(),
                true,
                &BlockingRegionToken,
            )
            .unwrap();
        let nn = neighbors.as_ref().unwrap();
        for i in -16..48 {
            print!(
                "{}={:?} ",
                i,
                nn.get_block(BlockCoordinate::new(12, 3, i + offset))
                    .map(|x| x.0)
            );
            assert_eq!(
                nn.get_block(BlockCoordinate::new(12, 3, i + offset)),
                Some(BlockId::from((i + 100) as u32))
            );
        }
        println!();
    });
}

const AGGREGATE_UPDATES_THRESHOLD: usize = 128;

fn reacquire_writeback_permit<'a, 'b, S: SyncBackend, L: SyncBackend>(
    game_state: &'a Arc<GameState>,
    coarse_shard: usize,
    read_lock: &mut S::ReadGuard<'_, super::MapShard<S, L>>,
) -> Result<mpsc::Permit<'b, WritebackReq>, Error>
where
    'a: 'b,
{
    if let Some(permit) = game_state
        .game_map()
        .try_get_writeback_permit(coarse_shard)?
    {
        Ok(permit)
    } else {
        // We need to release the read lock to get a permit, as the writeback thread needs to get a write lock
        // to make progress.
        let permit_or = S::RwLock::reader_unlocked(read_lock, || {
            game_state.game_map().get_writeback_permit(coarse_shard)
        });
        Ok(permit_or?)
    }
}

pub(super) struct TimerController<S: SyncBackend, L: SyncBackend> {
    pub(super) map: Arc<super::ServerGameMap<S, L>>,
    pub(super) game_state: Weak<GameState>,
    pub(super) cancellation: CancellationToken,
    pub(super) timers: FxHashMap<String, Arc<GameMapTimer>>,
}
impl<S: SyncBackend, L: SyncBackend> TimerController<S, L> {
    pub(super) async fn shutdown(&self) -> Result<()> {
        self.cancellation.cancel();
        for timer in self.timers.values() {
            timer.await_shutdown().await?;
        }

        Ok(())
    }

    pub(super) fn spawn_timer(
        &mut self,
        gs: Arc<GameState>,
        name: String,
        settings: TimerSettings,
        callback: TimerCallback,
    ) -> Result<()> {
        let shards = (settings.shards - 1) / NUM_CHUNK_SHARDS + 1;
        let timer = Arc::new(GameMapTimer {
            name: name.clone(),
            block_types: FxHashSet::from_iter(settings.block_types.iter().map(|x| x.base_id())),
            callback,
            settings,
            cancellation: self.cancellation.clone(),
            tasks: tokio::sync::Mutex::new(vec![]),
        });
        timer.spawn_shards(gs, shards)?;
        self.timers.insert(name, timer);
        Ok(())
    }

    #[cfg(any(test, feature = "test-support", doctest))]
    pub(super) fn run_timer(&self, name: &str) -> Result<()> {
        let timer = self
            .timers
            .get(name)
            .with_context(|| format!("Timer not found: {}", name))?;
        // Trigger all shards
        timer.run_inline(
            self.game_state
                .upgrade()
                .context("Game state has been dropped")?,
        )?;
        Ok(())
    }

    #[cfg(any(test, feature = "test-support", doctest))]
    pub(super) fn run_all_timers(&self) -> Result<()> {
        let gs = self
            .game_state
            .upgrade()
            .context("Game state has been dropped")?;
        for timer in self.timers.values() {
            timer.run_inline(gs.clone())?;
        }
        Ok(())
    }
}
