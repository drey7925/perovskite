//! Implementation details of light propagation that need to be shared between the client and server
//! The only type interesting to plugins (for now) is [LightScratchpad]

#[cfg(test)]
mod tests;

use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

use bitvec::prelude as bv;

use crate::block_id::BlockId;
use crate::constants::{
    CHUNK_SIZE, CHUNK_SIZE_I32, EXTENDED_CHUNK_OFFSET, EXTENDED_CHUNK_SIZE, EXTENDED_CHUNK_VOLUME,
    EXTENDED_OVERLAP_RANGES, PADDED_CHUNK_VOLUME,
};
use crate::coordinates::{ChunkOffset, ChunkOffsetForOcclusionExt};
use crate::sync::{GenericMutex, SyncBackend};
use std::{collections::BTreeMap, sync::atomic::AtomicUsize};

/// A 256-bit bitfield indicating what XZ positions within a chunk. This requires mutable
/// access to modify, but does not entail atomic or mutex operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OcclusionField {
    // u32 should generally fit into atomics, making this easier to translate to an atomic impl
    // if needed in the future
    data: bitvec::array::BitArray<[u32; (CHUNK_SIZE * CHUNK_SIZE) / 32], bitvec::order::Lsb0>,
}
static_assertions::const_assert_eq!((CHUNK_SIZE * CHUNK_SIZE) % 32, 0);

impl OcclusionField {
    #[inline]
    pub const fn zero() -> Self {
        OcclusionField {
            data: bv::BitArray::ZERO,
        }
    }
    #[inline]
    pub fn all_on() -> Self {
        !OcclusionField {
            data: bv::BitArray::ZERO,
        }
    }
    #[inline]
    pub fn any_set(&self) -> bool {
        self.data.any()
    }

    #[inline(always)]
    pub fn set(&mut self, x: impl Into<usize>, z: impl Into<usize>, arg: bool) {
        self.data.set(x.into() * CHUNK_SIZE + z.into(), arg);
    }
    #[inline(always)]
    pub fn get(&self, x: impl Into<usize>, z: impl Into<usize>) -> bool {
        self.data[x.into() * CHUNK_SIZE + z.into()]
    }
}

macro_rules! delegate_bin_op {
    ($trait:ident, $method:ident) => {
        impl $trait for OcclusionField {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: Self) -> Self::Output {
                OcclusionField {
                    data: self.data.$method(rhs.data),
                }
            }
        }
    };
}
macro_rules! delegate_bin_op_assign {
    ($trait:ident, $method:ident) => {
        impl $trait for OcclusionField {
            #[inline]
            fn $method(&mut self, rhs: Self) {
                self.data.$method(rhs.data);
            }
        }
    };
}
delegate_bin_op!(BitOr, bitor);
delegate_bin_op!(BitAnd, bitand);
delegate_bin_op!(BitXor, bitxor);
delegate_bin_op_assign!(BitOrAssign, bitor_assign);
delegate_bin_op_assign!(BitAndAssign, bitand_assign);
delegate_bin_op_assign!(BitXorAssign, bitxor_assign);

impl Not for OcclusionField {
    type Output = Self;
    #[inline]
    fn not(self) -> Self::Output {
        OcclusionField { data: !self.data }
    }
}

/// Representation of the occlusion for a column of chunks within the map.
pub struct ChunkColumn<S: SyncBackend> {
    // *Chunk* coordinate y-values that are loaded in memory.
    // Locking order: key1 > key2 (key1 is geometrically above key2) => lock-for-key1 is taken first
    present: BTreeMap<i32, S::Mutex<ChunkOcclusionState>>,
    generation: AtomicUsize,
}
impl<S: SyncBackend> ChunkColumn<S> {
    pub fn empty() -> Self {
        Self {
            present: BTreeMap::new(),
            generation: AtomicUsize::new(0),
        }
    }
    /// Inserts an empty chunk occlusion state for the given chunk.
    /// Panics if the chunk is already present.
    pub fn insert_empty(&mut self, chunk_y: i32) {
        assert!(self
            .present
            .insert(chunk_y, S::Mutex::new(ChunkOcclusionState::empty()))
            .is_none());
        let mut cur = self.cursor_into(chunk_y);
        cur.step_occlusion();
    }

    pub fn is_empty(&self) -> bool {
        self.present.is_empty()
    }

    /// Removes an entry, panics if the chunk is not present.
    pub fn remove(&mut self, chunk_y: i32) {
        assert!(self.present.remove(&chunk_y).is_some());
        let predecessor = self.present.range(chunk_y..).next();
        let predecessor_lock = predecessor.map(|x| x.1.lock());
        let successor = self.present.range(..chunk_y).next_back();
        if let Some(successor) = successor {
            let successor_lock = successor.1.lock();
            ChunkColumnCursor::<S> {
                generation: self
                    .generation
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                map: &self.present,
                current_pos: *successor.0,
                prev_pos: predecessor.map(|x| *x.0),
                previous: predecessor_lock,
                current: successor_lock,
            }
            .propagate_occlusion();
        }
    }

    /// Returns a cursor that starts in a state where it can propagate light *into* the given chunk
    /// from the previous chunk above it.
    pub fn cursor_into(&self, chunk_y: i32) -> ChunkColumnCursor<'_, S> {
        // Lock ordering: predecessor read() is called before current write()
        let (prev_pos, predecessor) = self
            .present
            .range((chunk_y + 1)..)
            .next()
            .map(|(pos, guard)| (*pos, guard.lock()))
            .unzip();

        let current = self.present.get(&chunk_y).unwrap().lock();
        ChunkColumnCursor {
            generation: self
                .generation
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            map: &self.present,
            current_pos: chunk_y,
            prev_pos,
            previous: predecessor,
            current,
        }
    }
    /// Returns a cursor that starts in a state where it can propagate light *out* of the given chunk
    /// into the next chunk below it.
    pub fn cursor_out_of(&self, chunk_y: i32) -> Option<ChunkColumnCursor<'_, S>> {
        // Lock ordering: current read() is called before successor write()
        let current = self.present.get(&chunk_y).unwrap().lock();
        let successor = self.present.range(..chunk_y).next_back()?;
        Some(ChunkColumnCursor {
            generation: self
                .generation
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            map: &self.present,
            current_pos: *successor.0,
            prev_pos: Some(chunk_y),
            previous: Some(current),
            current: successor.1.lock(),
        })
    }

    /// Returns a cursor out of the first possible chunk, into the second chunk.
    pub fn cursor_out_of_first(&self) -> Option<ChunkColumnCursor<'_, S>> {
        self.cursor_out_of(*self.present.last_key_value()?.0)
    }

    /// Returns a cursor out of the first possible chunk, into the second chunk.
    pub fn cursor_into_first(&self) -> Option<ChunkColumnCursor<'_, S>> {
        Some(self.cursor_into(*self.present.last_key_value()?.0))
    }

    /// Returns the incoming light and weather for the given chunk.
    /// Takes a short-lived read lock.
    pub fn get_incoming_light_and_weather(
        &self,
        y: i32,
    ) -> Option<(OcclusionField, OcclusionField)> {
        self.present.get(&y).map(|x| {
            let lock = x.lock();
            (lock.incoming_light, lock.incoming_weather)
        })
    }

    pub fn copy_keys(&self) -> Vec<i32> {
        self.present.keys().copied().collect()
    }
}

/// A cursor that can be used to perform light propagation in the column.
pub struct ChunkColumnCursor<'a, S: SyncBackend> {
    generation: usize,
    map: &'a BTreeMap<i32, S::Mutex<ChunkOcclusionState>>,
    current_pos: i32,
    prev_pos: Option<i32>,
    pub(crate) previous: Option<S::Guard<'a, ChunkOcclusionState>>,
    pub(crate) current: S::Guard<'a, ChunkOcclusionState>,
}
impl<'a, S: SyncBackend> ChunkColumnCursor<'a, S> {
    pub fn current_light_occlusion_mut(&mut self) -> &mut OcclusionField {
        &mut self.current.light_occlusion
    }
    pub fn current_weather_occlusion_mut(&mut self) -> &mut OcclusionField {
        &mut self.current.weather_occlusion
    }

    pub fn current_valid(&self) -> bool {
        self.current.valid
    }

    pub fn advance(self) -> Option<Self> {
        // Lock order: drop the previous lock, downgrade the current, and then take the successor
        drop(self.previous);
        let new_previous = self.current;
        let new_current = self.map.range(..self.current_pos).next_back()?;
        Some(ChunkColumnCursor {
            generation: self.generation,
            map: self.map,
            prev_pos: Some(self.current_pos),
            current_pos: *new_current.0,
            previous: Some(new_previous),
            current: new_current.1.lock(),
        })
    }

    /// The current chunk the cursor is at (would propagate light into here)
    pub fn current_pos(&self) -> i32 {
        self.current_pos
    }
    ///
    pub fn previous_pos(&self) -> Option<i32> {
        self.prev_pos
    }

    /// Propagate occlusion without advancing the cursor.
    pub fn step_occlusion(&mut self) {
        // #[cfg(test)] {
        //     println!("step_occlusion {:?} -> {:?}", self.prev_pos, self.current_pos);
        // }
        let prev_outgoing_light = self
            .previous
            .as_ref()
            .map_or(OcclusionField::all_on(), |x| x.outgoing_light());
        let prev_outgoing_weather = self
            .previous
            .as_ref()
            .map_or(OcclusionField::all_on(), |x| x.outgoing_weather());
        self.current.incoming_light = prev_outgoing_light;
        self.current.incoming_weather = prev_outgoing_weather;
    }

    /// Propagate occlusion from this cursor all the way down, consuming the cursor.
    pub fn propagate_occlusion(mut self) -> usize {
        // #[cfg(test)] {
        //     println!(
        //         "ChunkColumnCursor::propagate_occlusion start {:?} -> {}",
        //         self.prev_pos, self.current_pos
        //     );
        // }
        // prev_diff is used only for checking an invariant; we can remove it once we're sure that the
        // algorithm is correct
        // let mut prev_diff = OcclusionField::all_on();

        let mut counter = 0;
        // The light being transferred from one chunk to the next
        let mut prev_outgoing_light = self
            .previous
            .as_ref()
            .map_or(OcclusionField::all_on(), |x| x.outgoing_light());
        let mut prev_outgoing_weather = self
            .previous
            .as_ref()
            .map_or(OcclusionField::all_on(), |x| x.outgoing_weather());
        loop {
            // #[cfg(test)] {
            //     println!(
            //         "ChunkColumnCursor::propagate_occlusion {:?} -> {}. Current valid? {}",
            //         self.prev_pos,
            //         self.current_pos,
            //         self.current_valid()
            //     );
            // }
            // We should have advanced into a chunk with valid occlusion.
            // However, this is not always true. We may enter a chunk undergoing a deferred load,
            // and we do not have valid occlusion for it yet. However, once it loads, it'll fix up light anyway,
            // so this is of no concern.
            //
            // This assertion has not tripped for any other reason yet. To be safe, we'll log it.
            //
            // assert!(self.current.valid);

            let old_outgoing_light = self.current.outgoing_light();
            self.current.incoming_light = prev_outgoing_light;
            let new_outgoing_light = self.current.outgoing_light();

            let old_outgoing_weather = self.current.outgoing_weather();
            self.current.incoming_weather = prev_outgoing_weather;
            let new_outgoing_weather = self.current.outgoing_weather();

            let outgoing_diffs = (new_outgoing_light ^ old_outgoing_light)
                | (new_outgoing_weather ^ old_outgoing_weather);

            // If the outbound light we calculated is the same as the previous outbound light
            // for this chunk, we're done.
            // The first chunk is a special case - its inbound light isn't changing, and we
            // just updated its occlusion.
            if counter > 0 && !outgoing_diffs.any_set() {
                break;
            }
            // This is not actually an invariant. we could hit an out-of-date chunk that never had
            // occlusion propagated after an edit.
            // if (outgoing_diffs & !prev_diff).any_set() {
            //     eprintln!("OcclusionField invariant violated");
            //     eprintln!("Prev diff: {:?}", prev_diff);
            //     eprintln!("mismatch: {:?}", outgoing_diffs & !prev_diff);
            //     panic!();
            // }
            // prev_diff = if counter > 0 {
            //     outgoing_diffs
            // } else {
            //     OcclusionField::all_on()
            // };
            prev_outgoing_light = new_outgoing_light;
            prev_outgoing_weather = new_outgoing_weather;

            self = match self.advance() {
                Some(x) => x,
                None => {
                    break;
                }
            };
            counter += 1;
        }
        counter
    }

    pub fn mark_valid(&mut self) {
        self.current.valid = true;
    }
}

/// The occlusion state of a chunk
pub(crate) struct ChunkOcclusionState {
    pub(crate) valid: bool,
    /// xz coordinates that have light coming from above
    pub(crate) incoming_light: OcclusionField,
    /// xz coordinates where some block in the chunk stops light from passing
    pub(crate) light_occlusion: OcclusionField,

    pub(crate) incoming_weather: OcclusionField,
    pub(crate) weather_occlusion: OcclusionField,
}
impl ChunkOcclusionState {
    pub(crate) fn empty() -> Self {
        Self {
            valid: false,
            incoming_light: OcclusionField::zero(),
            light_occlusion: OcclusionField::zero(),
            incoming_weather: OcclusionField::zero(),
            weather_occlusion: OcclusionField::zero(),
        }
    }
    pub(crate) fn outgoing_light(&self) -> OcclusionField {
        self.incoming_light & !self.light_occlusion
    }
    pub(crate) fn outgoing_weather(&self) -> OcclusionField {
        self.incoming_weather & !self.weather_occlusion
    }
}

/// Holds state of a light propagation calculation. Exposed to callers so they can keep a
/// scratchpad around instead of constantly making new allocations.
pub struct LightScratchpad {
    light_buffer: Box<[u8; EXTENDED_CHUNK_VOLUME]>,
    // weather doesn't need fall-off, so use padded rather than extended
    weather_buffer: Box<bitvec::BitArr!(for PADDED_CHUNK_VOLUME)>,
    visit_queue: Vec<(i32, i32, i32, u8)>,
    light_propagation_cache: Box<bitvec::BitArr!(for EXTENDED_CHUNK_VOLUME)>,
}
impl LightScratchpad {
    pub fn clear(&mut self) {
        self.light_buffer.fill(0);
        self.weather_buffer.fill(false);
        self.visit_queue.clear();
        self.light_propagation_cache.fill(false);
    }
    /// Returns the light at the given coordinate, packed with global light in the upper 4 bits and
    /// local light in the lower 4 bits
    #[inline(always)]
    pub fn get_packed_u4_u4(&self, x: i32, y: i32, z: i32) -> u8 {
        self.light_buffer[(x, y, z).as_extended_index()]
    }

    #[inline(always)]
    pub fn get_global_light(&self, x: i32, y: i32, z: i32) -> u8 {
        self.get_packed_u4_u4(x, y, z) >> 4
    }

    #[inline(always)]
    pub fn get_local_light(&self, x: i32, y: i32, z: i32) -> u8 {
        self.get_packed_u4_u4(x, y, z) & 0xf
    }

    #[inline(always)]
    pub fn get_weather(&self, x: i32, y: i32, z: i32) -> bool {
        self.weather_buffer[(x, y, z).as_extended_index()]
    }

    #[inline(always)]
    pub fn weather(&self) -> &bitvec::BitArr!(for PADDED_CHUNK_VOLUME) {
        &self.weather_buffer
    }
}
impl Default for LightScratchpad {
    fn default() -> Self {
        Self {
            light_buffer: bytemuck::zeroed_box(),
            weather_buffer: Box::new(bitvec::array::BitArray::ZERO),
            visit_queue: Vec::new(),
            light_propagation_cache: Box::new(bitvec::array::BitArray::ZERO),
        }
    }
}

#[inline]
fn check_propagation_and_push<F>(
    queue: &mut Vec<(i32, i32, i32, u8)>,
    light_buffer: &mut [u8; EXTENDED_CHUNK_VOLUME],
    i: i32,
    j: i32,
    k: i32,
    light_level: u8,
    light_propagation: F,
) where
    F: Fn(i32, i32, i32) -> bool,
{
    const MIN_RANGE: i32 = -EXTENDED_CHUNK_OFFSET;
    const MAX_RANGE: i32 = CHUNK_SIZE_I32 + EXTENDED_CHUNK_OFFSET;
    if i < MIN_RANGE
        || j < MIN_RANGE
        || k < MIN_RANGE
        || i >= MAX_RANGE
        || j >= MAX_RANGE
        || k >= MAX_RANGE
    {
        return;
    }
    if !light_propagation(i, j, k) {
        return;
    }
    let old_level = light_buffer[(i + EXTENDED_CHUNK_OFFSET) as usize
        * EXTENDED_CHUNK_SIZE
        * EXTENDED_CHUNK_SIZE
        + (k + EXTENDED_CHUNK_OFFSET) as usize * EXTENDED_CHUNK_SIZE
        + (j + EXTENDED_CHUNK_OFFSET) as usize];
    // Take the maximum value of the upper and lower nibbles independently
    let max_level =
        ((old_level & 0xf).max(light_level & 0xf)) | (old_level & 0xf0).max(light_level & 0xf0);
    if max_level == old_level {
        return;
    }

    light_buffer[(i + EXTENDED_CHUNK_OFFSET) as usize
        * EXTENDED_CHUNK_SIZE
        * EXTENDED_CHUNK_SIZE
        + (k + EXTENDED_CHUNK_OFFSET) as usize * EXTENDED_CHUNK_SIZE
        + (j + EXTENDED_CHUNK_OFFSET) as usize] = max_level;
    let i_dist = (-1 - i).max(i - CHUNK_SIZE_I32);
    let j_dist = (-1 - j).max(j - CHUNK_SIZE_I32);
    let k_dist = (-1 - k).max(k - CHUNK_SIZE_I32);
    let dist = i_dist + j_dist + k_dist;
    let max_level = (light_level >> 4).max(light_level & 0xf);
    if dist < (max_level as i32) {
        queue.push((i, j, k, light_level));
    }
}

pub trait ChunkBuffer {
    /// Returns a single block at the given coordinate
    fn get(&self, offset: ChunkOffset) -> BlockId;
    /// Returns a slice of (x, 0, z), (x, 1, z), ..., (x, 15, z)
    fn vertical_slice(&self, x: u8, z: u8) -> &[BlockId; CHUNK_SIZE];
}

/// A type that holds a chunk and the immediately adjacent chunks
pub trait NeighborBuffer {
    /// The underlying chunk that this buffer holds.
    type Chunk<'a>: ChunkBuffer
    where
        Self: 'a;
    /// Returns this chunk if available. If not, None is returned.
    fn get(&self, dx: i32, dy: i32, dz: i32) -> Option<Self::Chunk<'_>>;
    /// Returns the lightmap at the top of this chunk.
    fn inbound_light(&self, dx: i32, dy: i32, dz: i32) -> OcclusionField;
    fn inbound_weather(&self, dx: i32, dy: i32, dz: i32) -> OcclusionField;
}

/// Fills in scratchpad with light in the center chunk of the neighbor buffer.
// Critical function, inline it into every caller even at the cost of build time
// In particular, we want to make sure that the compiler can see through the light data lookup
// functions and inline them
#[inline]
pub fn propagate_light_and_occlusion(
    neighbors: impl NeighborBuffer,
    scratchpad: &mut LightScratchpad,
    propagates_light: impl Fn(BlockId) -> bool,
    propagates_weather: impl Fn(BlockId) -> bool,
    light_emission: impl Fn(BlockId) -> u8,
) {
    scratchpad.clear();

    // First, scan through the neighborhood looking for light sources.
    // Indices are reordered to achieve better cache locality.
    // x is the major index, z is intermediate, and y is the minor index

    for (x_coarse, x_fine_range, x_base) in EXTENDED_OVERLAP_RANGES {
        for (z_coarse, z_fine_range, z_base) in EXTENDED_OVERLAP_RANGES {
            for (y_coarse, y_fine_range, y_base) in EXTENDED_OVERLAP_RANGES {
                // println!(
                //     "x_coarse: {}, x_fine_range: {:?}, x_base: {}",
                //     x_coarse, x_fine_range, x_base
                // );
                // println!(
                //     "z_coarse: {}, z_fine_range: {:?}, z_base: {}",
                //     z_coarse, z_fine_range, z_base
                // );
                // println!(
                //     "y_coarse: {}, y_fine_range: {:?}, y_base: {}",
                //     y_coarse, y_fine_range, y_base
                // );
                let sub_chunk = neighbors.get(x_coarse, y_coarse, z_coarse);

                let global_inbound_lights = neighbors.inbound_light(x_coarse, y_coarse, z_coarse);
                if let Some(chunk) = sub_chunk {
                    for x_fine in x_fine_range.clone().into_iter() {
                        for z_fine in z_fine_range.clone().into_iter() {
                            let x = x_base + x_fine;
                            let z = z_base + z_fine;

                            let subslice = &chunk.vertical_slice(x_fine as u8, z_fine as u8)
                                [y_fine_range.start as usize..y_fine_range.end as usize];
                            // consider unrolling this loop
                            let mut global_light =
                                global_inbound_lights.get(x_fine as u8, z_fine as u8);
                            let mut global_weather = neighbors
                                .inbound_weather(x_coarse, y_coarse, z_coarse)
                                .get(x_fine as u8, z_fine as u8);
                            for (&block_id, y_fine) in subslice
                                .iter()
                                .zip(y_fine_range.clone())
                                .rev()
                                .take(CHUNK_SIZE)
                            {
                                let y = y_base + y_fine as i32;
                                let propagates_light = propagates_light(block_id);
                                let light_emission = light_emission(block_id);

                                if light_emission > 0 {
                                    // println!(
                                    //     "Light emission at ({}, {}, {}) = {}",
                                    //     x, y, z, light_emission
                                    // );
                                    // println!(
                                    //     "coarse: ({}, {}, {}), fine: ({}, {}, {}) -> effective: ({}, {}, {})
                                    //     ",
                                    //     x_coarse, y_coarse, z_coarse, x_fine, y_fine, z_fine, x,
                                    //     y, z
                                    // );
                                }
                                scratchpad
                                    .light_propagation_cache
                                    .set((x, y, z).as_extended_index(), propagates_light);

                                if !propagates_light {
                                    global_light = false;
                                }

                                if !propagates_weather(block_id) {
                                    global_weather = false;
                                }
                                if let Some(idx) = (x, y, z).try_as_padded_index() {
                                    scratchpad.weather_buffer.set(idx, global_weather);
                                }

                                let global_bits = if global_light { 15 << 4 } else { 0 };
                                let effective_emission = light_emission | global_bits;
                                if effective_emission > 0 {
                                    check_propagation_and_push(
                                        &mut scratchpad.visit_queue,
                                        &mut scratchpad.light_buffer,
                                        x,
                                        y,
                                        z,
                                        effective_emission,
                                        |_, _, _| true,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let propagates_light_check =
        |x: i32, y: i32, z: i32| scratchpad.light_propagation_cache[(x, y, z).as_extended_index()];

    // Then, while the scratchpad.visit_queue is non-empty, attempt to propagate light
    while let Some((x, y, z, light_level)) = scratchpad.visit_queue.pop() {
        // println!(
        //     "Propagating light from ({}, {}, {}) with level {}",
        //     x, y, z, light_level
        // );
        let decremented =
            ((light_level & 0xf).saturating_sub(0x1)) | ((light_level & 0xf0).saturating_sub(0x10));
        check_propagation_and_push(
            &mut scratchpad.visit_queue,
            &mut scratchpad.light_buffer,
            x - 1,
            y,
            z,
            decremented,
            propagates_light_check,
        );
        check_propagation_and_push(
            &mut scratchpad.visit_queue,
            &mut scratchpad.light_buffer,
            x + 1,
            y,
            z,
            decremented,
            propagates_light_check,
        );
        check_propagation_and_push(
            &mut scratchpad.visit_queue,
            &mut scratchpad.light_buffer,
            x,
            y - 1,
            z,
            decremented,
            propagates_light_check,
        );
        check_propagation_and_push(
            &mut scratchpad.visit_queue,
            &mut scratchpad.light_buffer,
            x,
            y + 1,
            z,
            decremented,
            propagates_light_check,
        );
        check_propagation_and_push(
            &mut scratchpad.visit_queue,
            &mut scratchpad.light_buffer,
            x,
            y,
            z - 1,
            decremented,
            propagates_light_check,
        );
        check_propagation_and_push(
            &mut scratchpad.visit_queue,
            &mut scratchpad.light_buffer,
            x,
            y,
            z + 1,
            decremented,
            propagates_light_check,
        );
    }
}
