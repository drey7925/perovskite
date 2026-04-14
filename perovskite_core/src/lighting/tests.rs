use crate::lighting::{ChunkColumn, Lightfield};
use crate::sync::GenericRwLock;
use crate::sync::{DefaultSyncBackend, SyncBackend, TestonlyLoomBackend};
use std::sync::Arc;
#[test]
fn light_column_basic_test() {
    let mut col = ChunkColumn::<DefaultSyncBackend>::empty();
    col.insert_empty(3);
    col.insert_empty(5);
    col.insert_empty(7);
    let mut cursor = col.cursor_into(7);
    assert_eq!(cursor.current_pos, 7);
    assert_eq!(cursor.prev_pos, None);
    *cursor.current_occlusion_mut() = lf(1);
    cursor.mark_valid();
    cursor = cursor.advance().unwrap();

    assert_eq!(cursor.current_pos, 5);
    assert_eq!(cursor.prev_pos, Some(7));
    *cursor.current_occlusion_mut() = lf(2);
    cursor.mark_valid();
    cursor = cursor.advance().unwrap();

    assert_eq!(cursor.current_pos, 3);
    assert_eq!(cursor.prev_pos, Some(5));
    *cursor.current_occlusion_mut() = lf(3);
    cursor.mark_valid();
    assert!(cursor.advance().is_none());

    let cursor = col.cursor_into_first().unwrap();
    assert_eq!(cursor.current_pos, 7);
    assert_eq!(cursor.prev_pos, None);
    // 2 additional chunks
    assert_eq!(cursor.propagate_lighting(), 2);

    assert_eq!(col.get_incoming_light(8), None);
    assert_eq!(col.get_incoming_light(7), Some(Lightfield::all_on()));
    assert_eq!(
        col.get_incoming_light(5),
        Some(Lightfield::all_on() & !lf(1))
    );
    assert_eq!(
        col.get_incoming_light(3),
        Some(Lightfield::all_on() & !(lf(1) | lf(2)))
    );

    assert_eq!(col.get_incoming_light(1), None);
    col.insert_empty(1);
    let mut cursor = col.cursor_into(1);
    *cursor.current_occlusion_mut() = lf(7);
    cursor.mark_valid();
    cursor.propagate_lighting();

    assert_eq!(
        col.get_incoming_light(1),
        Some(Lightfield::all_on() & !(lf(1) | lf(2) | lf(3)))
    );
}
fn lf(i: u8) -> Lightfield {
    let mut lf = Lightfield::zero();
    lf.set(0usize, i, true);
    lf
}

#[test]
fn loom_test_concurrently_insert_remove() {
    if cfg!(debug_assertions) {
        panic!("loom tests are too slow to be run in debug mode; either ignore this test, or use --release");
    }
    loom::model(move || {
        let mut col = ChunkColumn::<TestonlyLoomBackend>::empty();

        assert_eq!(col.get_incoming_light(1), None);
        col.insert_empty(1);
        col.cursor_into(1).mark_valid();

        let col = Arc::new(<TestonlyLoomBackend as SyncBackend>::RwLock::new(col));
        let mut threads = vec![];
        // let cc = col.clone();
        // threads.push(loom::thread::spawn(move || {
        //     let mut cc = cc.lock_write();
        //     cc.insert_empty(3);
        //     let cc = <TestonlyLoomBackend as SyncBackend>::RwLock::downgrade_writer(cc);
        //     let mut cursor = cc.cursor_into(3);
        //     *cursor.current_occlusion_mut() = lf(4) | lf(5) | lf(6) | lf(7);
        //     cursor.mark_valid();
        //     cursor.propagate_lighting();
        // }));
        let cc = col.clone();
        threads.push(loom::thread::spawn(move || {
            let mut ccl = cc.lock_write();
            ccl.insert_empty(5);
            let ccl = <TestonlyLoomBackend as SyncBackend>::RwLock::downgrade_writer(ccl);
            let mut cursor = ccl.cursor_into(5);
            *cursor.current_occlusion_mut() = lf(2) | lf(3) | lf(6) | lf(7);
            cursor.mark_valid();
            cursor.propagate_lighting();
            // cursor dropped here
            drop(ccl);

            let mut ccl = cc.lock_write();
            ccl.remove(5);
        }));
        let cc = col.clone();
        threads.push(loom::thread::spawn(move || {
            // println!("4 add-rm-add thread");
            let mut ccl = cc.lock_write();
            ccl.insert_empty(4);
            let ccl = <TestonlyLoomBackend as SyncBackend>::RwLock::downgrade_writer(ccl);
            let mut cursor = ccl.cursor_into(4);
            *cursor.current_occlusion_mut() = lf(2) | lf(3) | lf(7) | lf(8) | lf(9);
            cursor.mark_valid();
            cursor.propagate_lighting();

            // println!("4 add-rm-add added1");
            drop(ccl);

            let mut ccl = cc.lock_write();
            ccl.remove(4);
            // println!("4 add-rm-add removed");
            drop(ccl);

            let mut ccl = cc.lock_write();

            ccl.insert_empty(4);
            let ccl = <TestonlyLoomBackend as SyncBackend>::RwLock::downgrade_writer(ccl);
            let mut cursor = ccl.cursor_into(4);
            *cursor.current_occlusion_mut() = lf(7) | lf(8);
            cursor.mark_valid();
            cursor.propagate_lighting();
            // cursor dropped here
            // println!("4 add-rm-add added2");

            drop(ccl);

            // println!("4 add-rm-add thread done");
        }));
        let cc = col.clone();
        threads.push(loom::thread::spawn(move || {
            let mut cc = cc.lock_write();
            cc.insert_empty(7);
            let cc = <TestonlyLoomBackend as SyncBackend>::RwLock::downgrade_writer(cc);
            let mut cursor = cc.cursor_into(7);
            *cursor.current_occlusion_mut() = lf(1) | lf(3) | lf(5) | lf(7);
            cursor.mark_valid();
            cursor.propagate_lighting();
        }));

        for thread in threads {
            thread.join().unwrap();
        }

        let col = col.lock_read();
        // let mut cursor = col.cursor_into(7);
        // cursor.propagate_lighting();

        assert_eq!(
            col.get_incoming_light(1),
            Some(!(lf(1) | lf(3) | lf(5) | lf(7) | lf(8)))
        );
    });
}

#[test]
fn non_loom_corruption_test() {
    let mut ccl = ChunkColumn::<DefaultSyncBackend>::empty();
    ccl.insert_empty(1);
    ccl.cursor_into(1).mark_valid();

    ccl.insert_empty(5);
    let mut cursor = ccl.cursor_into(5);
    *cursor.current_occlusion_mut() = lf(2) | lf(3) | lf(6) | lf(7);
    cursor.mark_valid();
    cursor.propagate_lighting();
    // // cursor dropped here
    // ccl.remove(5);

    ccl.insert_empty(4);

    let mut cursor = ccl.cursor_into(4);
    *cursor.current_occlusion_mut() = lf(2) | lf(3) | lf(7) | lf(8);
    cursor.mark_valid();
    cursor.propagate_lighting();

    ccl.remove(4);

    // ccl.insert_empty(4);
    // let mut cursor = ccl.cursor_into(4);
    // *cursor.current_occlusion_mut() = lf(7) | lf(8);
    // cursor.mark_valid();
    // cursor.propagate_lighting();
    // cursor dropped here
    ccl.insert_empty(7);
    let mut cursor = ccl.cursor_into(7);
    *cursor.current_occlusion_mut() = lf(1) | lf(3) | lf(5) | lf(7);
    cursor.mark_valid();
    cursor.propagate_lighting();
}

/// Tests for light propagation in the 3D scratchpad.
///
/// These tests focus on the coordinate logic and cross-chunk-boundary propagation
/// rather than the full physics of lighting. They treat the current implementation
/// as ground truth in preparation for a refactor.
///
/// Conventions used throughout:
///   - `propagates_light` is unconditionally `true` (block type is irrelevant).
///   - `light_emission` is `|b: BlockId| b.0 as u8` – the raw 32-bit block ID
///     truncated to u8 is the local emission level.
///   - No inbound sky-light (all `inbound_light` fields are `Lightfield::zero()`).
///   - `CHUNK_SIZE` is used everywhere instead of the literal 16.
mod propagation_tests {
    use crate::block_id::BlockId;
    use crate::constants::{CHUNK_SIZE, CHUNK_SIZE_I32, CHUNK_VOLUME};
    use crate::coordinates::ChunkOffset;
    use crate::lighting::{
        propagate_light, ChunkBuffer, LightScratchpad, Lightfield, NeighborBuffer,
    };

    // ── Test infrastructure ──────────────────────────────────────────────────

    /// A chunk stored in the same [x][z][y] order as `ChunkOffset::as_index`,
    /// backed by a heap-allocated Vec so it can be moved cheaply.
    struct SimpleChunk(Vec<BlockId>);

    impl SimpleChunk {
        fn empty() -> Self {
            Self(vec![BlockId(0); CHUNK_VOLUME])
        }

        /// Set a single block. The layout `[x][z][y]` matches `ChunkOffset::as_index`.
        fn set(&mut self, x: usize, y: usize, z: usize, id: BlockId) {
            debug_assert!(x < CHUNK_SIZE && y < CHUNK_SIZE && z < CHUNK_SIZE);
            self.0[CHUNK_SIZE * CHUNK_SIZE * x + CHUNK_SIZE * z + y] = id;
        }

        fn set_all(&mut self, id: BlockId) {
            self.0.fill(id);
        }
    }

    /// Thin reference wrapper so the associated lifetime can be expressed.
    struct SimpleChunkRef<'a>(&'a SimpleChunk);

    impl ChunkBuffer for SimpleChunkRef<'_> {
        fn get(&self, offset: ChunkOffset) -> BlockId {
            self.0 .0[offset.as_index()]
        }

        fn vertical_slice(&self, x: u8, z: u8) -> &[BlockId; CHUNK_SIZE] {
            let base = ChunkOffset::new(x, 0, z).as_index();
            (&self.0 .0[base..base + CHUNK_SIZE]).try_into().unwrap()
        }
    }

    /// A 3×3×3 neighborhood of optional test chunks indexed by
    /// `(dx+1, dy+1, dz+1)` where each component is in `{-1, 0, 1}`.
    struct SimpleNeighborBuffer {
        chunks: [[[Option<SimpleChunk>; 3]; 3]; 3],
        inbound_lights: [[[Lightfield; 3]; 3]; 3],
    }

    impl SimpleNeighborBuffer {
        fn new() -> Self {
            Self {
                // Option<T>: Default is always None, regardless of whether T: Default.
                chunks: Default::default(),
                inbound_lights: [[[Lightfield::zero(); 3]; 3]; 3],
            }
        }

        fn set_chunk(&mut self, dx: i32, dy: i32, dz: i32, chunk: SimpleChunk) {
            self.chunks[(dx + 1) as usize][(dy + 1) as usize][(dz + 1) as usize] = Some(chunk);
        }

        /// Convenience: mutable access to the center chunk.
        /// Panics if the center chunk has not been set yet.
        fn center_mut(&mut self) -> &mut SimpleChunk {
            self.chunks[1][1][1].as_mut().expect("center chunk not set")
        }
    }

    impl NeighborBuffer for SimpleNeighborBuffer {
        type Chunk<'a>
            = SimpleChunkRef<'a>
        where
            Self: 'a;

        fn get(&self, dx: i32, dy: i32, dz: i32) -> Option<Self::Chunk<'_>> {
            self.chunks[(dx + 1) as usize][(dy + 1) as usize][(dz + 1) as usize]
                .as_ref()
                .map(SimpleChunkRef)
        }

        fn inbound_light(&self, dx: i32, dy: i32, dz: i32) -> Lightfield {
            self.inbound_lights[(dx + 1) as usize][(dy + 1) as usize][(dz + 1) as usize]
        }
    }

    /// Run `propagate_light` with the test-standard closures and return the
    /// populated scratchpad.
    fn run(nb: SimpleNeighborBuffer) -> LightScratchpad {
        let mut pad = LightScratchpad::default();
        propagate_light(
            nb,
            &mut pad,
            |_: BlockId| true,      // propagates_light: always true
            |b: BlockId| b.0 as u8, // light_emission: block-id value → level
        );
        pad
    }

    // ── Tests ────────────────────────────────────────────────────────────────

    /// Single emitting block (level 15) placed at the centre of the chunk.
    ///
    /// Checks:
    /// - The source itself is at level 15.
    /// - Every face-adjacent block is at level 14.
    /// - Light decreases by exactly 1 per step of Manhattan distance along
    ///   each axis.
    #[test]
    fn test_single_source_in_center() {
        let mut nb = SimpleNeighborBuffer::new();
        nb.set_chunk(0, 0, 0, SimpleChunk::empty());
        let mid = CHUNK_SIZE / 2; // = 8 for the default CHUNK_SIZE of 16
        nb.center_mut().set(mid, mid, mid, BlockId(15));

        let pad = run(nb);
        let c = mid as i32;

        // Source
        assert_eq!(pad.get_local_light(c, c, c), 15, "source");

        // Six face-adjacent blocks
        assert_eq!(pad.get_local_light(c + 1, c, c), 14, "+x");
        assert_eq!(pad.get_local_light(c - 1, c, c), 14, "-x");
        assert_eq!(pad.get_local_light(c, c + 1, c), 14, "+y");
        assert_eq!(pad.get_local_light(c, c - 1, c), 14, "-y");
        assert_eq!(pad.get_local_light(c, c, c + 1), 14, "+z");
        assert_eq!(pad.get_local_light(c, c, c - 1), 14, "-z");

        // Gradient along +x: level = 15 - d, only within the centre chunk.
        // Source is at x=mid; the chunk ends at x=CHUNK_SIZE-1, so we can go at most
        // CHUNK_SIZE-1-mid steps before leaving the chunk (= mid-1 steps for a centred source).
        // Blocks beyond the chunk edge are absent so no light reaches them.
        for d in 0..(CHUNK_SIZE_I32 - c) {
            assert_eq!(
                pad.get_local_light(c + d, c, c),
                15u8.saturating_sub(d as u8),
                "along +x at d={d}"
            );
        }
        // Gradient along -z: source at z=mid, going to z=0 (still within the chunk).
        for d in 0..=c {
            assert_eq!(
                pad.get_local_light(c, c, c - d),
                15u8.saturating_sub(d as u8),
                "along -z at d={d}"
            );
        }
    }

    /// Light source at `x_fine = 0` of the +x neighbour chunk.
    ///
    /// In extended coordinates the source sits at `x = CHUNK_SIZE`, directly
    /// adjacent to the centre chunk's face at `x = CHUNK_SIZE - 1`.  Light
    /// must cross the x-chunk boundary, arriving at level 14 and decrementing
    /// by 1 for each additional block traveled into the centre chunk.
    #[test]
    fn test_positive_x_neighbor_crosses_edge() {
        let mid = CHUNK_SIZE / 2;
        let mut nb = SimpleNeighborBuffer::new();
        nb.set_chunk(0, 0, 0, SimpleChunk::empty());

        let mut neighbor = SimpleChunk::empty();
        // x_fine = 0  →  extended x = CHUNK_SIZE  (one step beyond centre's +x face)
        neighbor.set(0, mid, mid, BlockId(15));
        nb.set_chunk(1, 0, 0, neighbor);

        let pad = run(nb);
        let cy = mid as i32;
        let cz = mid as i32;

        // CHUNK_SIZE-1 is the first block inside the centre from the +x face: 1 step → 14
        assert_eq!(
            pad.get_local_light(CHUNK_SIZE_I32 - 1, cy, cz),
            14,
            "1 block in"
        );
        // 2 blocks in → 13
        assert_eq!(
            pad.get_local_light(CHUNK_SIZE_I32 - 2, cy, cz),
            13,
            "2 blocks in"
        );

        // Full gradient inward from the +x boundary
        for d in 1..=(CHUNK_SIZE / 2) as i32 {
            assert_eq!(
                pad.get_local_light(CHUNK_SIZE_I32 - d, cy, cz),
                15u8.saturating_sub(d as u8),
                "d={d} blocks in from +x boundary"
            );
        }
    }

    /// Light source at `z_fine = CHUNK_SIZE - 1` of the -z neighbour chunk.
    ///
    /// In extended coordinates the source sits at `z = -1`, directly adjacent
    /// to `z = 0` of the centre chunk.  Light crosses the z-chunk boundary,
    /// arriving at level 14 and falling by 1 for each additional block.
    #[test]
    fn test_negative_z_neighbor_crosses_edge() {
        let mid = CHUNK_SIZE / 2;
        let mut nb = SimpleNeighborBuffer::new();
        nb.set_chunk(0, 0, 0, SimpleChunk::empty());

        let mut neighbor = SimpleChunk::empty();
        // z_fine = CHUNK_SIZE-1  →  extended z = -1  (adjacent to z=0 in centre)
        neighbor.set(mid, mid, CHUNK_SIZE - 1, BlockId(15));
        nb.set_chunk(0, 0, -1, neighbor);

        let pad = run(nb);
        let cx = mid as i32;
        let cy = mid as i32;

        // z = 0 in centre: 1 step from source → 14
        assert_eq!(pad.get_local_light(cx, cy, 0), 14, "z=0");
        // z = 1: 2 steps → 13
        assert_eq!(pad.get_local_light(cx, cy, 1), 13, "z=1");

        // Gradient moving deeper along +z into the centre chunk
        for d in 0..=(CHUNK_SIZE / 2) as i32 {
            assert_eq!(
                pad.get_local_light(cx, cy, d),
                14u8.saturating_sub(d as u8),
                "z={d}"
            );
        }
    }

    /// Light source in the diagonal corner neighbour `(dx=-1, dy=0, dz=+1)`
    /// at the corner of that chunk nearest to the centre chunk.
    ///
    /// In extended coordinates the source is at `(x=-1, z=CHUNK_SIZE)`.
    /// To enter the centre chunk it must travel through two intermediate
    /// edges, each hosted in a flanking edge-chunk that we supply as empty
    /// so that the propagation cache marks those positions as passable.
    ///
    /// Both possible 2-step routes agree on level 13 at the centre corner:
    ///
    /// ```text
    /// Route A (through (0,0,+1) chunk):
    ///   (-1,cy,CS) → (0,cy,CS) → (0,cy,CS-1)   [levels 15,14,13]
    ///
    /// Route B (through (-1,0,0) chunk):
    ///   (-1,cy,CS) → (-1,cy,CS-1) → (0,cy,CS-1) [levels 15,14,13]
    /// ```
    #[test]
    fn test_corner_neighbor_crosses_two_chunk_edges() {
        let mid = CHUNK_SIZE / 2;
        let mut nb = SimpleNeighborBuffer::new();
        nb.set_chunk(0, 0, 0, SimpleChunk::empty());
        // The two edge-chunks that the light flows through on its way to the centre.
        // Without them the propagation cache would mark those positions as opaque.
        nb.set_chunk(-1, 0, 0, SimpleChunk::empty());
        nb.set_chunk(0, 0, 1, SimpleChunk::empty());

        // Corner chunk (-1, 0, +1): source at (x_fine=CHUNK_SIZE-1, y=mid, z_fine=0)
        // Extended coords: x = -1*CHUNK_SIZE + (CHUNK_SIZE-1) = -1
        //                  z =  1*CHUNK_SIZE + 0               = CHUNK_SIZE
        let mut corner = SimpleChunk::empty();
        corner.set(CHUNK_SIZE - 1, mid, 0, BlockId(15));
        nb.set_chunk(-1, 0, 1, corner);

        let pad = run(nb);
        let cy = mid as i32;

        // Corner of centre chunk (x=0, z=CHUNK_SIZE-1): 2 steps → 13
        assert_eq!(
            pad.get_local_light(0, cy, CHUNK_SIZE_I32 - 1),
            13,
            "corner of centre"
        );

        // One step deeper in z → 12
        assert_eq!(
            pad.get_local_light(0, cy, CHUNK_SIZE_I32 - 2),
            12,
            "one deeper in z"
        );
        // One step deeper in x → 12
        assert_eq!(
            pad.get_local_light(1, cy, CHUNK_SIZE_I32 - 1),
            12,
            "one deeper in x"
        );
    }

    /// No light sources anywhere and no inbound sky-light: every position in
    /// the centre chunk must have both local and global light equal to zero.
    #[test]
    fn test_no_sources_all_dark() {
        let mut nb = SimpleNeighborBuffer::new();
        nb.set_chunk(0, 0, 0, SimpleChunk::empty());

        let pad = run(nb);

        for x in 0..CHUNK_SIZE_I32 {
            for y in 0..CHUNK_SIZE_I32 {
                for z in 0..CHUNK_SIZE_I32 {
                    assert_eq!(pad.get_local_light(x, y, z), 0, "local ({x},{y},{z})");
                    assert_eq!(pad.get_global_light(x, y, z), 0, "global ({x},{y},{z})");
                }
            }
        }
    }

    #[test]
    fn test_all_sources_doesnt_crash_or_panic() {
        let mut nb = SimpleNeighborBuffer::new();
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    let mut chunk = SimpleChunk::empty();
                    chunk.set_all(BlockId(15));
                    nb.set_chunk(x, y, z, chunk);
                }
            }
        }
        let pad = run(nb);
        for x in 0..CHUNK_SIZE_I32 {
            for y in 0..CHUNK_SIZE_I32 {
                for z in 0..CHUNK_SIZE_I32 {
                    assert_eq!(pad.get_local_light(x, y, z), 15, "local ({x},{y},{z})");
                }
            }
        }
    }
}

#[test]
fn loom_hand_over_hand_deadlock_check() {
    if cfg!(debug_assertions) {
        panic!("loom tests are too slow to be run in debug mode; either ignore this test, or use --release");
    }
    loom::model(move || {
        let mut col = ChunkColumn::<TestonlyLoomBackend>::empty();

        col.insert_empty(3);
        col.insert_empty(2);
        col.insert_empty(1);

        let col = Arc::new(col);
        let mut threads = vec![];
        let cc = col.clone();
        threads.push(loom::thread::spawn(move || {
            cc.cursor_out_of(3);
        }));
        let cc = col.clone();
        threads.push(loom::thread::spawn(move || {
            cc.cursor_into(2);
        }));
        let cc = col.clone();
        threads.push(loom::thread::spawn(move || {
            cc.cursor_out_of(2);
        }));
        let cc = col.clone();
        threads.push(loom::thread::spawn(move || {
            cc.cursor_into(1);
        }));
        for thread in threads {
            thread.join().unwrap();
        }
    })
}
