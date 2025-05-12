//! GPU-friendly data structures, mostly used for the new raytracer.

use crate::vulkan::gpu_chunk_table::ht_consts::{
    FLAG_HASHTABLE_PRESENT, OFFSET_K1, OFFSET_K2, OFFSET_K3, OFFSET_MXC, OFFSET_N,
};
use anyhow::Context;
use perovskite_core::coordinates::ChunkCoordinate;
use rand::distributions::uniform::SampleRange;
use rustc_hash::{FxHashMap, FxHashSet};
use std::ops::Deref;

/// Computes a parametrized hash over the coordinate.
///
/// Params:
///     coord: The coordinate to hash
///     k1, k2, k3: Discovered hash parameters
///
pub fn phash(coord: ChunkCoordinate, k1: u32, k2: u32, k3: u32, p: u32, n: u32) -> u32 {
    let ChunkCoordinate { x, y, z } = coord;
    assert!(n.is_power_of_two());
    let x = x as u32;
    let y = y as u32;
    let z = z as u32;
    let xp = x.wrapping_mul(k1);
    let yp = y.wrapping_mul(k2);
    let zp = z.wrapping_mul(k3);
    ((xp.wrapping_add(yp).wrapping_add(zp)) % p) & (n - 1)
}

/// The actual data payload of each chunk
pub const CHUNK_LEN: usize = 18 * 18 * 18;
// 18 * 18 * 18, overaligned to 32 bytes
pub const CHUNK_LIGHTS_OFFSET: usize = 5856;
pub const CHUNK_LIGHTS_LEN: usize = 18 * 18 * 18 / 4; // 1458
/// Data payload + alignment
pub const CHUNK_STRIDE: usize = 5856 + 1472; // 7328

pub mod ht_consts {
    /// If set in flags, this entry has a valid chunk. If unset, x/y/z are nonsensical, with
    /// arbitrary (likely 0 values), and the same holds for the corresponding chunk data
    ///
    /// Note that if this is absent, the WHOLE flags must be 0
    pub const FLAG_HASHTABLE_PRESENT: u32 = 1;

    pub const OFFSET_N: usize = 0;
    pub const OFFSET_MXC: usize = 1;
    pub const OFFSET_K1: usize = 2;
    pub const OFFSET_K2: usize = 3;
    pub const OFFSET_K3: usize = 4;
}

const PRIME: u32 = 1610612741;

/// Builds a hashtable that is efficiently accessible on a GPU.
///
/// Layout (all values are u32, offsets are given w.r.t a `[u32]`, i.e. offset 1 is 4 bytes):
///
/// * N is a power of two, and must be at least 8 (since we need 4N + 32 to be 32-int, i.e. 128-byte
///   aligned). Why 128 bytes rather than the 64-byte alignment typical on CPU? GPU cacheline can be
///   that large.
///
/// Each row is 4 ints, i.e. 16 bytes
///
/// ```ignore                   0         1        2         3
/// 0..3                     [   N   ][  MXC   ][   k1   ][  k2    ]
/// 4..7                     [   k3  ][     reserved/padding       ]
///    ...                   [ reserved/padding                    ]
/// 28..31                   [ reserved/padding                    ]
/// 32..35                   [ flags0 ][   x0  ][   y0   ][   z0   ]
/// 36..39                   [ flags1 ][   x1  ][   y1   ][   z1   ]
/// 4i+32 .. 4i + 35         [ f_i    ][   x_i ][   y_i  ][   z_i  ]
/// 4(N-1)+32 .. 4(N-1) + 35 [ x_n-1 ][ y_n-1  ][ z_n-1  ][ f_n-1  ]
/// 4N+32 .. 5863            [ Chunk 0 data, as CHUNK_STRIDE ints  ]
/// ...                      [ More chunks                ...      ]
/// ```
///
/// Each chunk internally is 5832 ints (blocks), 1458 ints (lights, u8 packed into u32 using machine
/// endianness, e.g. on x86 `[1u8, 0u8, 0u8, 0u8]` -> `1u32`.)
///
/// Chunk x/y/z are transmuted i32 -> u32 when stored.
///
/// Each chunk can be found starting at int offset 4N + 32 + CHUNK_LEN*i.
///
/// Data buffers are the blocks in a chunk, ordered X/Z/Y. [perovskite_core::coordinates::ChunkOffset::as_index]
/// may be used to index into this.
pub struct ChunkHashtableBuilder<T, U>
where
    T: Deref<Target = [u32]>,
    U: Deref<Target = [u8]>,
{
    // Not a FxHashMap: paranoid that we don't have a hard guarantee of iteration order being same
    // on each iteration of an unchanged map (although it's a reasonable assumption to make)
    chunks: FxHashMap<ChunkCoordinate, (T, U)>,
}
impl<T, U> ChunkHashtableBuilder<T, U>
where
    T: Deref<Target = [u32]>,
    U: Deref<Target = [u8]>,
{
    pub fn new() -> ChunkHashtableBuilder<T, U> {
        ChunkHashtableBuilder {
            chunks: FxHashMap::default(),
        }
    }

    /// Adds a chunk to this builder.
    pub fn add_chunk(&mut self, coord: ChunkCoordinate, data: T, lights: U) {
        self.chunks.insert(coord, (data, lights));
    }

    /// Builds a chunk hashtable.
    ///
    /// Args:
    ///     max_tries: Maximum number of attempts to make before giving up and taking the best
    ///         mapping possible.
    ///     max_probe_len: If the longest collision probe takes <= max_probe_len values, stop and
    ///         accept the attempt. This does not include the first one (i.e., first try w/o
    ///         collision is counted as 0)
    pub fn build(&self, max_tries: usize, max_probe_len: usize) -> anyhow::Result<Vec<u32>> {
        //const SENTINEL_FILLER: u32 = 0xcdcdcdcd;
        // oversize to 1.25x
        let expanded = self.chunks.len() + (self.chunks.len() >> 2) + (self.chunks.len() >> 3);
        let n = expanded.max(8).next_power_of_two();
        let mut data = Vec::new();
        data.resize((4 + CHUNK_STRIDE) * n + 32, 0);
        let n32 = n.try_into().context("n overflowed u32")?;
        data[0] = n32;

        let mut rng = rand::thread_rng();

        // K values, max num probes, table
        let mut best_probes = usize::MAX;
        let mut best_k: Option<(u32, u32, u32, Vec<Option<ChunkCoordinate>>)> = None;
        'tries: for _ in 0..max_tries {
            let k1 = (100000000..u32::MAX).sample_single(&mut rng);
            let k2 = (100000000..u32::MAX).sample_single(&mut rng);
            let k3 = (100000000..u32::MAX).sample_single(&mut rng);
            let mut mapping: Vec<Option<ChunkCoordinate>> = Vec::new();
            mapping.resize(n, None);
            let mut max_probes = 0;
            for (coord, _) in &self.chunks {
                let mut slot = phash(*coord, k1, k2, k3, PRIME, n32) as usize;
                let mut probes = 0;
                while mapping[slot].is_some() {
                    probes += 1;
                    slot = (slot + 1) & (n - 1);
                }
                if probes >= best_probes {
                    continue 'tries;
                }
                mapping[slot] = Some(*coord);
                max_probes = max_probes.max(probes);
            }
            if max_probes < best_probes {
                best_k = Some((k1, k2, k3, mapping));
                best_probes = max_probes;
            }
            if max_probes <= max_probe_len {
                break;
            }
        }
        let (k1, k2, k3, table) = best_k.unwrap();
        data[OFFSET_MXC] = best_probes
            .try_into()
            .context("max_probes overflowed u32")?;
        data[OFFSET_K1] = k1;
        data[OFFSET_K2] = k2;
        data[OFFSET_K3] = k3;
        dbg!(&data[0..5]);

        for (i, entry) in table.iter().enumerate() {
            if let Some(coord) = entry {
                let control_base = (4 * i) + 32;
                data[control_base] = FLAG_HASHTABLE_PRESENT;
                data[control_base + 1] = coord.x as u32;
                data[control_base + 2] = coord.y as u32;
                data[control_base + 3] = coord.z as u32;
                let data_base = 4 * n + 32 + CHUNK_STRIDE * i;
                let (blocks, lights) = self.chunks.get(coord).unwrap();
                data[data_base..data_base + CHUNK_LEN].copy_from_slice(blocks.deref());
            }
        }

        Ok(data)
    }
}

pub fn gpu_table_lookup(table: &[u32], key: ChunkCoordinate) -> u32 {
    let n = table[OFFSET_N];
    let mxc = table[OFFSET_MXC];
    let k1 = table[OFFSET_K1];
    let k2 = table[OFFSET_K2];
    let k3 = table[OFFSET_K3];

    let x = key.x as u32;
    let y = key.y as u32;
    let z = key.z as u32;

    let mut slot = phash(key, k1, k2, k3, PRIME, n);

    for _ in 0..=mxc {
        let base = (slot as usize) * 4 + 32;
        if table[base + 3] & FLAG_HASHTABLE_PRESENT == 0 {
            return u32::MAX;
        }
        if table[base] == x && table[base + 1] == y && table[base + 2] == z {
            return slot;
        }
        slot = (slot + 1) & (n - 1);
    }
    u32::MAX
}

#[test]
fn test_build_and_probe() {
    let mut builder = ChunkHashtableBuilder::new();
    let data = vec![0u32; 18 * 18 * 18];
    let lights = vec![0u8; 18 * 18 * 18];
    for x in -15..15 {
        for y in -15..15 {
            for z in -15..15 {
                builder.add_chunk(
                    ChunkCoordinate::new(x, y, z),
                    data.as_slice(),
                    lights.as_slice(),
                );
            }
        }
    }
    let table = builder.build(10, 3).unwrap();
    println!(
        "Table n: {} mxc {} k1 {} k2 {} k3 {}",
        table[0], table[1], table[2], table[3], table[4]
    );
    for x in -20..20 {
        for y in -20..20 {
            for z in -20..20 {
                if gpu_table_lookup(&table, ChunkCoordinate::new(x, y, z)) == u32::MAX {
                    println!("!!! {x} {y} {z}")
                }
            }
        }
    }
    assert_eq!(
        u32::MAX,
        gpu_table_lookup(&table, ChunkCoordinate::new(1, 100, 2))
    );
}
