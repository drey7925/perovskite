//! Implementation details of light propagation that need to be shared between the client and server
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

use bitvec::field::BitField;
use bitvec::prelude as bv;

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use std::{collections::BTreeMap, sync::atomic::AtomicUsize};

/// A 256-bit bitfield indicating what XZ positions within a chunk. This requires mutable
/// access to modify, but does not entail atomic or mutex operations.
#[derive(Clone, Copy, Debug)]
pub struct Lightfield {
    // usize should match the native register size which will make it easier to
    // translate this to atomics in the future, if necessary.
    data: bitvec::array::BitArray<[u32; 8], bitvec::order::Lsb0>,
}

impl Lightfield {
    pub const fn zero() -> Self {
        Lightfield {
            data: bv::BitArray::ZERO,
        }
    }
    pub fn all_on() -> Self {
        !Lightfield {
            data: bv::BitArray::ZERO,
        }
    }
    pub fn any_set(&self) -> bool {
        self.data.any()
    }
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

    pub fn set(&mut self, x: u8, z: u8, arg: bool) {
        self.data.set((x as usize) * 16 + (z as usize), arg);
    }
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
pub struct ChunkColumn {
    // *Chunk* coordinate y-values that are loaded in memory.
    // Locking order: key1 > key2 (key1 is geometrically above key2) => lock-for-key1 is taken first
    present: BTreeMap<i32, RwLock<ChunkLightingState>>,
    generation: AtomicUsize,
}
impl ChunkColumn {
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
            .insert(chunk_y, RwLock::new(ChunkLightingState::empty()))
            .is_none());
    }

    pub fn is_empty(&self) -> bool {
        self.present.is_empty()
    }

    /// Removes an entry, panics if the chunk is not present.
    pub fn remove(&mut self, chunk_y: i32) {
        assert!(self.present.remove(&chunk_y).is_some());
    }

    /// Returns a cursor that starts in a state where it can propagate light *into* the given chunk
    /// from the previous chunk above it.
    pub fn cursor_into(&self, chunk_y: i32) -> ChunkColumnCursor<'_> {
        // Lock ordering: predecessor read() is called before current write()
        let (prev_pos, predecessor) = self
            .present
            .range((chunk_y + 1)..)
            .next()
            .map(|(pos, guard)| (*pos, guard.read()))
            .unzip();

        let current = self.present.get(&chunk_y).unwrap().write();
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
    pub fn cursor_out_of(&self, chunk_y: i32) -> Option<ChunkColumnCursor<'_>> {
        // Lock ordering: current read() is called before successor write()
        let current = self.present.get(&chunk_y).unwrap().read();
        let successor = self.present.range(..chunk_y).next_back()?;
        Some(ChunkColumnCursor {
            generation: self
                .generation
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            map: &self.present,
            current_pos: *successor.0,
            prev_pos: Some(chunk_y),
            previous: Some(current),
            current: successor.1.write(),
        })
    }

    /// Returns a cursor out of the first possible chunk, into the second chunk.
    pub fn cursor_out_of_first(&self) -> Option<ChunkColumnCursor<'_>> {
        self.cursor_out_of(*self.present.last_key_value()?.0)
    }

    /// Returns the incoming light for the given chunk.
    /// Takes a short-lived read lock.
    pub fn get_incoming_light(&self, y: i32) -> Option<Lightfield> {
        self.present.get(&y).map(|x| x.read().incoming)
    }

    pub fn copy_keys(&self) -> Vec<i32> {
        self.present.keys().copied().collect()
    }
}

/// A cursor that can be used to perform light propagation in the column.
pub struct ChunkColumnCursor<'a> {
    generation: usize,
    map: &'a BTreeMap<i32, RwLock<ChunkLightingState>>,
    current_pos: i32,
    prev_pos: Option<i32>,
    pub(crate) previous: Option<RwLockReadGuard<'a, ChunkLightingState>>,
    pub(crate) current: RwLockWriteGuard<'a, ChunkLightingState>,
}
impl<'a> ChunkColumnCursor<'a> {
    pub fn current_occlusion_mut(&mut self) -> &mut Lightfield {
        &mut self.current.occlusion
    }

    pub fn current_valid(&self) -> bool {
        self.current.valid
    }

    pub fn advance(self) -> Option<Self> {
        // Lock order: drop the previous lock, downgrade the current, and then take the successor
        drop(self.previous);
        let new_previous = RwLockWriteGuard::downgrade(self.current);
        let new_current = self.map.range(..self.current_pos).next_back()?;
        Some(ChunkColumnCursor {
            generation: self.generation,
            map: self.map,
            prev_pos: Some(self.current_pos),
            current_pos: *new_current.0,
            previous: Some(new_previous),
            current: new_current.1.write(),
        })
    }

    pub fn current_pos(&self) -> i32 {
        self.current_pos
    }
    pub fn previous_pos(&self) -> Option<i32> {
        self.prev_pos
    }

    pub fn propagate_lighting(mut self) -> usize {
        // prev_diff is used only for checking an invariant; we can remove it once we're sure that the
        // algorithm is correct
        let mut prev_diff = Lightfield::all_on();

        let mut counter = 0;
        // The light being transferred from one chunk to the next
        let mut prev_outgoing = self
            .previous
            .as_ref()
            .map_or(Lightfield::all_on(), |x| x.outgoing());
        loop {
            // We should have advanced into a chunk with valid lighting.
            assert!(self.current.valid);

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
            if (outgoing_diffs & !prev_diff).any_set() {
                eprintln!("Lightfield invariant violated");
                eprintln!("Prev diff: {:?}", prev_diff);
                eprintln!("mismatch: {:?}", outgoing_diffs & !prev_diff);
                panic!();
            }
            prev_diff = if counter > 0 {
                outgoing_diffs
            } else {
                Lightfield::all_on()
            };
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
