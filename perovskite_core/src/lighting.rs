//! Implementation details of light propagation that need to be shared between the client and server
//! The only type interesting to plugins (for now) is [LightScratchpad]

mod tests;

use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

use bitvec::field::BitField;
use bitvec::prelude as bv;

use crate::block_id::BlockId;
use crate::coordinates::ChunkOffset;
use crate::sync::{GenericMutex, SyncBackend};
use std::{collections::BTreeMap, sync::atomic::AtomicUsize};

/// A 256-bit bitfield indicating what XZ positions within a chunk. This requires mutable
/// access to modify, but does not entail atomic or mutex operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lightfield {
    // usize should match the native register size which will make it easier to
    // translate this to atomics in the future, if necessary.
    data: bitvec::array::BitArray<[u32; 8], bitvec::order::Lsb0>,
}

impl Lightfield {
    #[inline]
    pub const fn zero() -> Self {
        Lightfield {
            data: bv::BitArray::ZERO,
        }
    }
    #[inline]
    pub fn all_on() -> Self {
        !Lightfield {
            data: bv::BitArray::ZERO,
        }
    }
    #[inline]
    pub fn any_set(&self) -> bool {
        self.data.any()
    }
    #[inline]
    pub fn serialize(&self) -> [u32; 8] {
        [
            self.data[0..31].load_le(),
            self.data[32..63].load_le(),
            self.data[64..95].load_le(),
            self.data[96..127].load_le(),
            self.data[128..159].load_le(),
            self.data[160..191].load_le(),
            self.data[192..223].load_le(),
            self.data[224..255].load_le(),
        ]
    }
    pub fn deserialize(arr: [u32; 8]) -> Self {
        let mut data = bv::BitArray::ZERO;
        data[0..31].store_le(arr[0]);
        data[32..63].store_le(arr[1]);
        data[64..95].store_le(arr[2]);
        data[96..127].store_le(arr[3]);
        data[128..159].store_le(arr[4]);
        data[160..191].store_le(arr[5]);
        data[192..223].store_le(arr[6]);
        data[224..255].store_le(arr[7]);

        Lightfield { data }
    }

    #[inline(always)]
    pub fn set(&mut self, x: u8, z: u8, arg: bool) {
        self.data.set((x as usize) * 16 + (z as usize), arg);
    }
    #[inline(always)]
    pub fn get(&self, x: u8, z: u8) -> bool {
        self.data[(x as usize) * 16 + (z as usize)]
    }
}

macro_rules! delegate_bin_op {
    ($trait:ident, $method:ident) => {
        impl $trait for Lightfield {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: Self) -> Self::Output {
                Lightfield {
                    data: self.data.$method(rhs.data),
                }
            }
        }
    };
}
macro_rules! delegate_bin_op_assign {
    ($trait:ident, $method:ident) => {
        impl $trait for Lightfield {
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

impl Not for Lightfield {
    type Output = Self;
    #[inline]
    fn not(self) -> Self::Output {
        Lightfield { data: !self.data }
    }
}

/// Representation of the lighting for a column of chunks within the map.
pub struct ChunkColumn<S: SyncBackend> {
    // *Chunk* coordinate y-values that are loaded in memory.
    // Locking order: key1 > key2 (key1 is geometrically above key2) => lock-for-key1 is taken first
    present: BTreeMap<i32, S::Mutex<ChunkLightingState>>,
    generation: AtomicUsize,
}
impl<S: SyncBackend> ChunkColumn<S> {
    pub fn empty() -> Self {
        Self {
            present: BTreeMap::new(),
            generation: AtomicUsize::new(0),
        }
    }
    /// Inserts an empty chunk lighting state for the given chunk.
    /// Panics if the chunk is already present.
    pub fn insert_empty(&mut self, chunk_y: i32) {
        assert!(self
            .present
            .insert(chunk_y, S::Mutex::new(ChunkLightingState::empty()))
            .is_none());
        let mut cur = self.cursor_into(chunk_y);
        cur.step_lighting();
    }

    pub fn is_empty(&self) -> bool {
        self.present.is_empty()
    }

    /// Removes an entry, panics if the chunk is not present.
    pub fn remove(&mut self, chunk_y: i32) {
        #[cfg(test)]
        {
            // println!("remove {chunk_y}");
        }
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
            .propagate_lighting();
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

    /// Returns the incoming light for the given chunk.
    /// Takes a short-lived read lock.
    pub fn get_incoming_light(&self, y: i32) -> Option<Lightfield> {
        self.present.get(&y).map(|x| x.lock().incoming)
    }

    pub fn copy_keys(&self) -> Vec<i32> {
        self.present.keys().copied().collect()
    }
}

/// A cursor that can be used to perform light propagation in the column.
pub struct ChunkColumnCursor<'a, S: SyncBackend> {
    generation: usize,
    map: &'a BTreeMap<i32, S::Mutex<ChunkLightingState>>,
    current_pos: i32,
    prev_pos: Option<i32>,
    pub(crate) previous: Option<S::Guard<'a, ChunkLightingState>>,
    pub(crate) current: S::Guard<'a, ChunkLightingState>,
}
impl<'a, S: SyncBackend> ChunkColumnCursor<'a, S> {
    pub fn current_occlusion_mut(&mut self) -> &mut Lightfield {
        &mut self.current.occlusion
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

    pub fn current_pos(&self) -> i32 {
        self.current_pos
    }
    pub fn previous_pos(&self) -> Option<i32> {
        self.prev_pos
    }

    pub fn step_lighting(&mut self) {
        // #[cfg(test)] {
        //     println!("step_lighting {:?} -> {:?}", self.prev_pos, self.current_pos);
        // }
        let prev_outgoing = self
            .previous
            .as_ref()
            .map_or(Lightfield::all_on(), |x| x.outgoing());
        self.current.incoming = prev_outgoing;
    }

    pub fn propagate_lighting(mut self) -> usize {
        // #[cfg(test)] {
        //     println!(
        //         "ChunkColumnCursor::propagate_lighting start {:?} -> {}",
        //         self.prev_pos, self.current_pos
        //     );
        // }
        // prev_diff is used only for checking an invariant; we can remove it once we're sure that the
        // algorithm is correct
        // let mut prev_diff = Lightfield::all_on();

        let mut counter = 0;
        // The light being transferred from one chunk to the next
        let mut prev_outgoing = self
            .previous
            .as_ref()
            .map_or(Lightfield::all_on(), |x| x.outgoing());
        loop {
            // #[cfg(test)] {
            //     println!(
            //         "ChunkColumnCursor::propagate_lighting {:?} -> {}. Current valid? {}",
            //         self.prev_pos,
            //         self.current_pos,
            //         self.current_valid()
            //     );
            // }
            // We should have advanced into a chunk with valid lighting.
            // However, this is not always true. We may enter a chunk undergoing a deferred load,
            // and we do not have valid lighting for it yet. However, once it loads, it'll fix up light anyway,
            // so this is of no concern.
            //
            // This assertion has not tripped for any other reason yet. To be safe, we'll log it.
            //
            // assert!(self.current.valid);

            let old_outgoing = self.current.outgoing();
            self.current.incoming = prev_outgoing;
            let new_outgoing = self.current.outgoing();

            let outgoing_diffs = new_outgoing ^ old_outgoing;

            // If the outbound light we calculated is the same as the previous outbound light
            // for this chunk, we're done.
            // The first chunk is a special case - its inbound light isn't changing, and we
            // just updated its occlusion.
            if counter > 0 && !outgoing_diffs.any_set() {
                break;
            }
            // This is not actually an invariant. we could hit an out-of-date chunk that never had
            // lighting propagated after an edit.
            // if (outgoing_diffs & !prev_diff).any_set() {
            //     eprintln!("Lightfield invariant violated");
            //     eprintln!("Prev diff: {:?}", prev_diff);
            //     eprintln!("mismatch: {:?}", outgoing_diffs & !prev_diff);
            //     panic!();
            // }
            // prev_diff = if counter > 0 {
            //     outgoing_diffs
            // } else {
            //     Lightfield::all_on()
            // };
            prev_outgoing = new_outgoing;

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

/// The lighting state of a chunk
pub struct ChunkLightingState {
    pub(crate) valid: bool,
    /// xz coordinates that have light coming from above
    pub(crate) incoming: Lightfield,
    /// xz coordinates where some block in the chunk stops light from passing
    pub(crate) occlusion: Lightfield,
}
impl ChunkLightingState {
    pub(crate) fn empty() -> Self {
        Self {
            valid: false,
            incoming: Lightfield::zero(),
            occlusion: Lightfield::zero(),
        }
    }
    pub(crate) fn outgoing(&self) -> Lightfield {
        self.incoming & !self.occlusion
    }
}

/// Holds state of a light propagation calculation. Exposed to callers so they can keep a
/// scratchpad around instead of constantly making new allocations.
pub struct LightScratchpad {
    light_buffer: Box<[u8; 48 * 48 * 48]>,
    visit_queue: Vec<(i32, i32, i32, u8)>,
    propagation_cache: Box<bitvec::BitArr!(for 48*48*48)>,
}
impl LightScratchpad {
    pub fn clear(&mut self) {
        self.light_buffer.fill(0);
        self.visit_queue.clear();
        self.propagation_cache.fill(false);
    }
    /// Returns the light at the given coordinate, packed with global light in the upper 4 bits and
    /// local light in the lower 4 bits
    #[inline(always)]
    pub fn get_packed_u4_u4(&self, x: i32, y: i32, z: i32) -> u8 {
        self.light_buffer[(x + 16) as usize * 48 * 48 + (z + 16) as usize * 48 + (y + 16) as usize]
    }

    #[inline(always)]
    pub fn get_global_light(&self, x: i32, y: i32, z: i32) -> u8 {
        self.get_packed_u4_u4(x, y, z) >> 4
    }

    #[inline(always)]
    pub fn get_local_light(&self, x: i32, y: i32, z: i32) -> u8 {
        self.get_packed_u4_u4(x, y, z) & 0xf
    }
}
impl Default for LightScratchpad {
    fn default() -> Self {
        Self {
            light_buffer: Box::new([0; 48 * 48 * 48]),
            visit_queue: Vec::new(),
            propagation_cache: Box::new(bitvec::array::BitArray::ZERO),
        }
    }
}

#[inline]
fn check_propagation_and_push<F>(
    queue: &mut Vec<(i32, i32, i32, u8)>,
    light_buffer: &mut [u8; 48 * 48 * 48],
    i: i32,
    j: i32,
    k: i32,
    light_level: u8,
    light_propagation: F,
) where
    F: Fn(i32, i32, i32) -> bool,
{
    if i < -16 || j < -16 || k < -16 || i >= 32 || j >= 32 || k >= 32 {
        return;
    }
    if !light_propagation(i, j, k) {
        return;
    }
    let old_level =
        light_buffer[(i + 16) as usize * 48 * 48 + (k + 16) as usize * 48 + (j + 16) as usize];
    // Take the maximum value of the upper and lower nibbles independently
    let max_level =
        ((old_level & 0xf).max(light_level & 0xf)) | (old_level & 0xf0).max(light_level & 0xf0);
    if max_level == old_level {
        return;
    }

    light_buffer[(i + 16) as usize * 48 * 48 + (k + 16) as usize * 48 + (j + 16) as usize] =
        max_level;
    let i_dist = (-1 - i).max(i - 16);
    let j_dist = (-1 - j).max(j - 16);
    let k_dist = (-1 - k).max(k - 16);
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
    fn vertical_slice(&self, x: u8, z: u8) -> &[BlockId; 16];
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
    fn inbound_light(&self, dx: i32, dy: i32, dz: i32) -> Lightfield;
}

/// Fills in scratchpad with light in the center chunk of the neighbor buffer.
// Critical function, inline it into every caller even at the cost of build time
// In particular, we want to make sure that the compiler can see through the light data lookup
// functions and inline them
pub fn propagate_light(
    neighbors: impl NeighborBuffer,
    scratchpad: &mut LightScratchpad,
    propagates_light: impl Fn(BlockId) -> bool,
    light_emission: impl Fn(BlockId) -> u8,
) {
    scratchpad.clear();

    // First, scan through the neighborhood looking for light sources.
    // Indices are reordered to achieve better cache locality.
    // x is the major index, z is intermediate, and y is the minor index
    for x_coarse in -1i32..=1 {
        for z_coarse in -1i32..=1 {
            for y_coarse in -1i32..=1 {
                let slice = neighbors.get(x_coarse, y_coarse, z_coarse);

                let global_inbound_lights = neighbors.inbound_light(x_coarse, y_coarse, z_coarse);
                if let Some(chunk) = slice {
                    for x_fine in 0i32..16 {
                        for z_fine in 0i32..16 {
                            let x = x_coarse * 16 + x_fine;
                            let z = z_coarse * 16 + z_fine;

                            let subslice = chunk.vertical_slice(x_fine as u8, z_fine as u8);
                            // consider unrolling this loop
                            let mut global_light =
                                global_inbound_lights.get(x_fine as u8, z_fine as u8);
                            for (y_fine, &block_id) in subslice.iter().enumerate().rev().take(16) {
                                let y = y_coarse * 16 + y_fine as i32;
                                let propagates_light = propagates_light(block_id);
                                scratchpad.propagation_cache.set(
                                    ((x + 16) * 48 * 48 + (z + 16) * 48 + (y + 16)) as usize,
                                    propagates_light,
                                );
                                let light_emission = light_emission(block_id);
                                if !propagates_light {
                                    global_light = false;
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

    let propagates_light_check = |x: i32, y: i32, z: i32| {
        scratchpad.propagation_cache
            [(x + 16) as usize * 48 * 48 + (z + 16) as usize * 48 + (y + 16) as usize]
    };

    // Then, while the scratchpad.visit_queue is non-empty, attempt to propagate light
    while let Some((x, y, z, light_level)) = scratchpad.visit_queue.pop() {
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
