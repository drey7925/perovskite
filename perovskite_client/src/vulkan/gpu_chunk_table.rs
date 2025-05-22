//! GPU-friendly data structures, mostly used for the new raytracer.

use crate::vulkan::gpu_chunk_table::ht_consts::{
    FLAG_HASHTABLE_PRESENT, OFFSET_K1, OFFSET_K2, OFFSET_K3, OFFSET_MXC, OFFSET_N,
};
use crate::vulkan::shaders::raytracer::ChunkMapHeader;
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
pub fn phash(coord: ChunkCoordinate, k1: u32, k2: u32, k3: u32, n_minus_one: u32) -> u32 {
    let ChunkCoordinate { x, y, z } = coord;
    assert!((n_minus_one + 1).is_power_of_two());
    let x = x as u32;
    let y = y as u32;
    let z = z as u32;
    let xp = x.wrapping_mul(k1);
    let yp = y.wrapping_mul(k2);
    let zp = z.wrapping_mul(k3);
    ((xp.wrapping_add(yp).wrapping_add(zp)) % PRIME) & n_minus_one
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
    /// If set, this chunk includes more complex and expensive-to-render geometry.
    pub const FLAG_HASHTABLE_HEAVY: u32 = 2;

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
/// * N is a power of two, and must be at least 8 (since we need 4N to be 32-int, i.e. 128-byte
///   aligned). Why 128 bytes rather than the 64-byte alignment typical on CPU? GPU cacheline can be
///   that large.
///
/// Each row is 4 ints, i.e. 16 bytes
///
/// ```ignore                   0         1        2         3
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
/// Each chunk can be found starting at int offset 4N + CHUNK_LEN*i.
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
    pub fn build(
        &self,
        max_tries: usize,
        max_probe_len: usize,
    ) -> anyhow::Result<(Vec<u32>, ChunkMapHeader)> {
        // oversize to 1.75x
        let expanded = self.chunks.len() + (self.chunks.len() >> 2) + (self.chunks.len() >> 3);
        let n = expanded.max(8).next_power_of_two();
        let mut data = Vec::new();
        data.resize((4 + CHUNK_STRIDE) * n + 32, 0);
        let n_minus_one = (n - 1).try_into().context("n overflowed u32")?;

        let mut rng = rand::thread_rng();

        // K values, max num probes, table
        let mut best_probes = usize::MAX;
        let mut best_k: Option<(u32, u32, u32, Vec<_>)> = None;

        // Calculate the min and max coordinates of all chunks
        let mut min_x = i32::MAX;
        let mut min_y = i32::MAX;
        let mut min_z = i32::MAX;
        let mut max_x = i32::MIN;
        let mut max_y = i32::MIN;
        let mut max_z = i32::MIN;

        for (coord, _) in &self.chunks {
            min_x = min_x.min(coord.x);
            min_y = min_y.min(coord.y);
            min_z = min_z.min(coord.z);
            max_x = max_x.max(coord.x);
            max_y = max_y.max(coord.y);
            max_z = max_z.max(coord.z);
        }

        'tries: for _ in 0..max_tries {
            let k1 = (100000000..u32::MAX).sample_single(&mut rng);
            let k2 = (100000000..u32::MAX).sample_single(&mut rng);
            let k3 = (100000000..u32::MAX).sample_single(&mut rng);
            let mut mapping: Vec<Option<(ChunkCoordinate, usize)>> = Vec::new();
            mapping.resize(n, None);
            let mut max_probes = 0;
            for (coord, _) in &self.chunks {
                let mut new_coord = *coord;
                let mut slot = phash(new_coord, k1, k2, k3, n_minus_one) as usize;
                let mut new_probes = 0;
                while let Some((present_coord, present_probes)) = &mut mapping[slot].as_mut() {
                    if new_probes > *present_probes {
                        new_probes += 1;
                        max_probes = max_probes.max(new_probes);
                        std::mem::swap(&mut new_coord, present_coord);
                        std::mem::swap(&mut new_probes, present_probes);
                    }
                    new_probes += 1;

                    slot = (slot + 1) & (n - 1);
                }
                if new_probes >= best_probes {
                    continue 'tries;
                }
                mapping[slot] = Some((new_coord, new_probes));
                max_probes = max_probes.max(new_probes);
            }
            assert_eq!(
                max_probes,
                mapping
                    .iter()
                    .map(|x| x.as_ref().map(|x| x.1).unwrap_or(0))
                    .max()
                    .unwrap()
            );
            if max_probes < best_probes {
                best_k = Some((k1, k2, k3, mapping));
                best_probes = max_probes;
            }
            if max_probes <= max_probe_len {
                break;
            }
        }
        let (k1, k2, k3, table) = best_k.unwrap();

        for (i, entry) in table.iter().enumerate() {
            if let Some((coord, _)) = entry {
                let control_base = 4 * i;
                data[control_base] = FLAG_HASHTABLE_PRESENT;
                data[control_base + 1] = coord.x as u32;
                data[control_base + 2] = coord.y as u32;
                data[control_base + 3] = coord.z as u32;
                let data_base = 4 * n + CHUNK_STRIDE * i;
                let light_base = 4 * n + CHUNK_STRIDE * i + CHUNK_LIGHTS_OFFSET;
                let (blocks, lights) = self.chunks.get(coord).unwrap();
                data[data_base..data_base + CHUNK_LEN].copy_from_slice(blocks.deref());
                data[light_base..light_base + CHUNK_LIGHTS_LEN]
                    .copy_from_slice(bytemuck::cast_slice(lights));
            }
        }
        let mxc: u32 = best_probes
            .try_into()
            .context("max_probes overflowed u32")?;
        let header = ChunkMapHeader {
            n_minus_one,
            mxc: mxc.into(),
            k: [k1, k2, k3].into(),
            min_chunk: [min_x, min_y, min_z].into(),
            max_chunk: [max_x, max_y, max_z].into(),
        };

        Ok((data, header))
    }
}

pub fn build_empty() -> anyhow::Result<(Vec<u32>, ChunkMapHeader)> {
    ChunkHashtableBuilder::<Vec<u32>, Vec<u8>>::new().build(1, 1)
}

pub fn gpu_table_lookup(table: &[u32], header: &ChunkMapHeader, key: ChunkCoordinate) -> u32 {
    let x = key.x as u32;
    let y = key.y as u32;
    let z = key.z as u32;

    let mut slot = phash(
        key,
        header.k[0],
        header.k[1],
        header.k[2],
        header.n_minus_one,
    );
    let mxc = *header.mxc;
    for _ in 0..=mxc {
        let base = (slot as usize) * 4;
        if table[base] & FLAG_HASHTABLE_PRESENT == 0 {
            return u32::MAX;
        }
        if table[base + 1] == x && table[base + 2] == y && table[base + 3] == z {
            return slot;
        }
        slot = (slot + 1) & (header.n_minus_one);
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
    let (table, header) = builder.build(10, 3).unwrap();

    for x in -15..15 {
        for y in -15..15 {
            for z in -15..15 {
                if gpu_table_lookup(&table, &header, ChunkCoordinate::new(x, y, z)) == u32::MAX {
                    panic!("Lookup failed for {x} {y} {z}")
                }
            }
        }
    }
    assert_eq!(
        u32::MAX,
        gpu_table_lookup(&table, &header, ChunkCoordinate::new(1, 100, 2))
    );
}
