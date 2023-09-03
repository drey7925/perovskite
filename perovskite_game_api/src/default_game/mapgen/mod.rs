use std::{
    ops::BitXor,
    sync::Arc,
};

use perovskite_core::{
    constants::blocks::AIR,
    coordinates::{BlockCoordinate, ChunkCoordinate, ChunkOffset},
};
use perovskite_server::game_state::{
    blocks::{BlockTypeHandle, BlockTypeManager},
    mapgen::MapgenInterface, game_map::MapChunkStronglyConsistentData,
};
use noise::{MultiFractal, NoiseFn};


use crate::blocks::BlockTypeHandleWrapper;

use super::{
    basic_blocks::{DIRT, DIRT_WITH_GRASS, STONE, WATER},
    foliage::{MAPLE_LEAVES, MAPLE_TREE},
};

const ELEVATION_FINE_INPUT_SCALE: f64 = 1.0 / 60.0;
const ELEVATION_FINE_OUTPUT_SCALE: f64 = 10.0;
const ELEVATION_COARSE_INPUT_SCALE: f64 = 1.0 / 800.0;
const ELEVATION_COARSE_OUTPUT_SCALE: f64 = 60.0;
const ELEVATION_OFFSET: f64 = 20.0;
const TREE_DENSITY_INPUT_SCALE: f64 = 1.0 / 240.0;
const TREE_DENSITY_OUTPUT_SCALE: f64 = 0.0625;
const TREE_DENSITY_OUTPUT_OFFSET: f64 = 0.03125;
struct ElevationNoise {
    coarse: noise::RidgedMulti<noise::SuperSimplex>,
    fine: noise::SuperSimplex,
}
impl ElevationNoise {
    fn new(seed: u32) -> Box<ElevationNoise> {
        Box::new(ElevationNoise {
            coarse: noise::RidgedMulti::new(seed).set_persistence(0.8),
            fine: noise::SuperSimplex::new(seed.wrapping_add(1)),
        })
    }
    fn get(&self, x: i32, z: i32) -> i32 {
        let coarse_pos = [
            x as f64 * ELEVATION_COARSE_INPUT_SCALE,
            z as f64 * ELEVATION_COARSE_INPUT_SCALE,
        ];
        let coarse_height = self.coarse.get(coarse_pos) * ELEVATION_COARSE_OUTPUT_SCALE;

        let fine_pos = [
            x as f64 * ELEVATION_FINE_INPUT_SCALE,
            z as f64 * ELEVATION_FINE_INPUT_SCALE,
        ];
        let fine_height = self.fine.get(fine_pos) * ELEVATION_FINE_OUTPUT_SCALE;
        let raw_height = coarse_height + fine_height + ELEVATION_OFFSET;
        // Create more flat ground near water level
        let mut adjusted_height = raw_height; // (raw_height - 1.) - 1. * ((raw_height - 1.) / 1.1).tanh();
        if adjusted_height < 2. {
            adjusted_height = adjusted_height / 2. + 1.;
        }
        adjusted_height as i32
    }
}

/// An ore that the mapgen should generate. Note: This struct is subject to being extended with new fields
/// in the future
pub struct OreDefinition {
    pub block: BlockTypeHandleWrapper,
    /// When the generated noise is smaller than this value, the ore is generated.
    /// TODO figure out and document the range of the noise
    /// This is expressed as a spline with the input being the depth at which we are generating ore.
    pub noise_cutoff: splines::spline::Spline<f64, f64>,
    /// The scale of the noise that generates this ore. The higher the scale, the larger and more spread-apart the clumps are.
    pub noise_scale: (f64, f64, f64),
}

struct DefaultMapgen {
    air: BlockTypeHandle,
    dirt: BlockTypeHandle,
    dirt_grass: BlockTypeHandle,
    stone: BlockTypeHandle,
    water: BlockTypeHandle,
    // todo organize the foliage (and more of the blocks in general) in a better way
    maple_tree: BlockTypeHandle,
    maple_leaves: BlockTypeHandle,

    elevation_noise: Box<ElevationNoise>,
    tree_density_noise: noise::Billow<noise::SuperSimplex>,
    ores: Vec<(OreDefinition, noise::SuperSimplex)>,
    seed: u32,
}
impl MapgenInterface for DefaultMapgen {
    fn fill_chunk(&self, chunk_coord: ChunkCoordinate, chunk: &mut (Box<[u32; 4096]>, MapChunkStronglyConsistentData)) {
        // todo subdivide by surface vs underground, etc. This is a very minimal MVP
        // let mut height_map = Box::new([[0i32; 16]; 16]);
        // for x in 0..16 {
        //     for z in 0..16 {
        //         let xg = 16 * chunk_coord.x + (x as i32);
        //         let zg = 16 * chunk_coord.z + (z as i32);

        //         let elevation = self.elevation_noise.get(xg, zg);
        //         height_map[x as usize][z as usize] = elevation;
        //         for y in 0..16 {
        //             let offset = ChunkOffset { x, y, z };
        //             let block_coord = chunk_coord.with_offset(offset);

        //             let vert_offset = block_coord.y - elevation;
        //             let block = if vert_offset > 0 {
        //                 if block_coord.y > 0 {
        //                     self.air
        //                 } else {
        //                     self.water
        //                 }
        //             } else if vert_offset == 0 && block_coord.y >= 0 {
        //                 self.dirt_grass
        //             } else if vert_offset > -3 {
        //                 // todo variable depth of dirt
        //                 self.dirt
        //             } else {
        //                 self.generate_ore(block_coord)
        //             };
        //             //chunk.0[offset.as_index()] = (offset, block, None);
        //         }
        //     }
        // }
        // self.generate_vegetation(chunk_coord, chunk, &height_map);
    }
}

// impl DefaultMapgen {
//     #[inline]
//     fn generate_ore(&self, coord: BlockCoordinate) -> BlockTypeHandle {
//         for (ore, noise) in &self.ores {
//             let cutoff = ore
//                 .noise_cutoff
//                 .clamped_sample(-(coord.y as f64))
//                 .unwrap_or(0.);
//             let noise_coord = [
//                 coord.x as f64 / ore.noise_scale.0,
//                 coord.y as f64 / ore.noise_scale.1,
//                 coord.z as f64 / ore.noise_scale.2,
//             ];
//             let sample = noise.get(noise_coord);
//             if sample < cutoff {
//                 return ore.block.0;
//             }
//         }

//         self.stone
//     }

//     #[inline]
//     fn fast_uniform_2d(&self, x: i32, z: i32, seed: u32) -> f64 {
//         // Ugly, awful RNG. However, it's stable and passes enough simple correlation tests
//         // to be useful for putting trees into a voxel game.
//         const K: u64 = 0x517cc1b727220a95;
//         const M: u64 = 0x72c07af023017001;

//         let mut hash = seed as u64;
//         hash = hash
//             .rotate_left(5)
//             .bitxor((x as u64).wrapping_mul(M))
//             .swap_bytes()
//             .wrapping_mul(K);
//         hash = hash
//             .rotate_left(5)
//             .bitxor((z as u64).wrapping_mul(M))
//             .swap_bytes()
//             .wrapping_mul(K);
//         hash = hash
//             .rotate_left(5)
//             .bitxor((seed as u64).wrapping_mul(M))
//             .swap_bytes()
//             .wrapping_mul(K);
//         (hash as f64) / (u64::MAX as f64)
//     }

//     fn generate_vegetation(
//         &self,
//         chunk_coord: ChunkCoordinate,
//         chunk: &mut MapChunk,
//         heightmap: &[[i32; 16]; 16],
//     ) {

//         for i in -3..18 {
//             for j in -3..18 {
//                 let block_xz = (chunk_coord.x * 16)
//                     .checked_add(i)
//                     .zip((chunk_coord.z * 16).checked_add(j));
//                 if let Some((x, z)) = block_xz {
//                     let y = if x.div_euclid(16) == chunk_coord.x && z.div_euclid(16) == chunk_coord.z {
//                         heightmap[x.rem_euclid(16) as usize][z.rem_euclid(16) as usize]
//                     } else {
//                         self.elevation_noise.get(x, z)
//                     };
//                     if y <= 0 {
//                         continue;
//                     }
//                     let tree_value = self.fast_uniform_2d(x, z, self.seed);
//                     let tree_cutoff = self.tree_density_noise.get([
//                         (x as f64) * TREE_DENSITY_INPUT_SCALE,
//                         (z as f64) * TREE_DENSITY_INPUT_SCALE,
//                     ]) * TREE_DENSITY_OUTPUT_SCALE
//                         + TREE_DENSITY_OUTPUT_OFFSET;
//                     if tree_value < tree_cutoff {
//                         self.make_tree(chunk_coord, chunk, x, y, z);
//                     }
//                 }
//             }
//         }
//     }


//     // Generate a tree at the given location. The y coordinate represents the dirt block just below the tree.
//     fn make_tree(
//         &self,
//         chunk_coord: ChunkCoordinate,
//         chunk: &mut MapChunk,
//         x: i32,
//         y: i32,
//         z: i32,
//     ) {

//         for h in 1..=4 {
//             let coord = BlockCoordinate::new(x, y + h, z);
//             if coord.chunk() == chunk_coord {
//                 chunk.set_block(coord.offset(), self.maple_tree, None);
//             }
//         }
//         for h in 3..=5 {
//             for i in -2..=2 {
//                 for j in -2..=2 {
//                     if i == 0 && j == 0 && h <= 4 {
//                         continue;
//                     }
//                     let coord = BlockCoordinate::new(x + i, y + h, z + j);
//                     if coord.chunk() == chunk_coord {
//                         chunk.set_block(coord.offset(), self.maple_leaves, None);
//                     }
//                 }
//             }
//         }
//     }
// }

pub(crate) fn build_mapgen(
    blocks: Arc<BlockTypeManager>,
    seed: u32,
    ores: Vec<OreDefinition>,
) -> Arc<dyn MapgenInterface> {
    Arc::new(DefaultMapgen {
        air: blocks.get_by_name(AIR).expect("air"),
        dirt: blocks.get_by_name(DIRT.0).expect("dirt"),
        dirt_grass: blocks.get_by_name(DIRT_WITH_GRASS.0).expect("dirt_grass"),
        stone: blocks.get_by_name(STONE.0).expect("stone"),
        // TODO 0xfff is a magic number, give it a real constant definition
        water: blocks.get_by_name(WATER.0).expect("water").with_variant(0xfff).unwrap(),
        maple_tree: blocks.get_by_name(MAPLE_TREE.0).expect("maple_tree"),
        maple_leaves: blocks.get_by_name(MAPLE_LEAVES.0).expect("maple_leaves"),

        elevation_noise: ElevationNoise::new(seed),
        tree_density_noise: noise::Billow::new(seed + 1),
        ores: ores
            .into_iter()
            .enumerate()
            .map(|(i, ore)| {
                (
                    ore,
                    noise::SuperSimplex::new(seed.wrapping_add(5000).wrapping_add(i as u32)),
                )
            })
            .collect(),
        seed,
    })
}
