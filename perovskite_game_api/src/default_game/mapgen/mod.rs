use std::{
    ops::{BitXor, Range},
    sync::Arc,
};

use noise::{MultiFractal, NoiseFn};
use perovskite_core::{
    constants::blocks::AIR,
    coordinates::{BlockCoordinate, ChunkCoordinate, ChunkOffset},
};
use perovskite_server::game_state::{
    blocks::{BlockTypeHandle, BlockTypeManager},
    game_map::MapChunk,
    mapgen::MapgenInterface,
};

use crate::blocks::BlockTypeHandleWrapper;

use super::{
    basic_blocks::{DESERT_SAND, DESERT_STONE, DIRT, DIRT_WITH_GRASS, SAND, STONE, WATER},
    foliage::{MAPLE_LEAVES, MAPLE_TREE, CACTUS},
};

const ELEVATION_FINE_INPUT_SCALE: f64 = 1.0 / 60.0;
const ELEVATION_FINE_OUTPUT_SCALE: f64 = 10.0;
const ELEVATION_COARSE_INPUT_SCALE: f64 = 1.0 / 800.0;
const ELEVATION_COARSE_OUTPUT_SCALE: f64 = 60.0;
const ELEVATION_OFFSET: f64 = 20.0;
const TREE_DENSITY_INPUT_SCALE: f64 = 1.0 / 240.0;
const TREE_DENSITY_OUTPUT_SCALE: f64 = 0.0625;
const TREE_DENSITY_OUTPUT_OFFSET: f64 = 0.03125;
const CACTUS_DENSITY_INPUT_SCALE: f64 = 1.0 / 120.0;
const CACTUS_DENSITY_OUTPUT_SCALE: f64 = 0.03125;
const CACTUS_DENSITY_OUTPUT_OFFSET: f64 = 0.03125;

const BEACH_TENDENCY_INPUT_SCALE: f64 = 1.0 / 240.0;
const DESERT_TENDENCY_INPUT_SCALE: f64 = 1.0 / 480.0;

// Next seed offset: 6

#[derive(Clone, Copy, Debug)]
enum Biome {
    DefaultGrassy,
    SandyBeach,
    Desert,
}

struct BiomeNoise {
    sand_tendency: noise::SuperSimplex,
    desert_tendency: noise::SuperSimplex,
}
impl BiomeNoise {
    fn new(seed: u32) -> BiomeNoise {
        BiomeNoise {
            sand_tendency: noise::SuperSimplex::new(seed.wrapping_add(3)),
            desert_tendency: noise::SuperSimplex::new(seed.wrapping_add(4)),
        }
    }
    fn get(&self, x: i32, z: i32, elevation: f64) -> Biome {
        let sand_value = self.sand_tendency.get([
            x as f64 * BEACH_TENDENCY_INPUT_SCALE,
            z as f64 * BEACH_TENDENCY_INPUT_SCALE,
        ]);
        if sand_value > 0.3 {
            let excess = sand_value - 0.3;
            // sandy beach
            let range = (excess * -10.0)..=(excess * 3.0 + 1.0);
            if range.contains(&elevation) {
                return Biome::SandyBeach;
            }
        };
        let desert_value = self.desert_tendency.get([
            x as f64 * DESERT_TENDENCY_INPUT_SCALE,
            z as f64 * DESERT_TENDENCY_INPUT_SCALE,
        ]);
        if desert_value > 0.4 {
            // The more extreme the desert value, the closer it can approach sea level
            // 0.4 deserts start at approx 6
            // 0.7 deserts start at sea level
            let cutoff = (20.0 * (0.8 - desert_value)).max(10.0);
            if elevation > cutoff {
                return Biome::Desert;
            }
        }
        return Biome::DefaultGrassy;
    }
}
struct ElevationNoise {
    coarse: noise::RidgedMulti<noise::SuperSimplex>,
    fine: noise::SuperSimplex,
}
impl ElevationNoise {
    fn new(seed: u32) -> ElevationNoise {
        ElevationNoise {
            coarse: noise::RidgedMulti::new(seed).set_persistence(0.8),
            fine: noise::SuperSimplex::new(seed.wrapping_add(1)),
        }
    }
    fn get(&self, x: i32, z: i32) -> f64 {
        // todo consider using the biome to adjust heights
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
        adjusted_height
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
    sand: BlockTypeHandle,
    stone: BlockTypeHandle,

    desert_stone: BlockTypeHandle,
    desert_sand: BlockTypeHandle,

    water: BlockTypeHandle,
    // todo organize the foliage (and more of the blocks in general) in a better way
    maple_tree: BlockTypeHandle,
    maple_leaves: BlockTypeHandle,
    cactus: BlockTypeHandle,

    elevation_noise: ElevationNoise,
    tree_density_noise: noise::Billow<noise::SuperSimplex>,
    cactus_density_noise: noise::Billow<noise::SuperSimplex>,
    biome_noise: BiomeNoise,
    ores: Vec<(OreDefinition, noise::SuperSimplex)>,
    seed: u32,
}
impl MapgenInterface for DefaultMapgen {
    fn fill_chunk(&self, chunk_coord: ChunkCoordinate, chunk: &mut MapChunk) {
        // todo subdivide by surface vs underground, etc. This is a very minimal MVP
        let mut height_map = Box::new([[0f64; 16]; 16]);
        let mut biome_map = Box::new([[Biome::DefaultGrassy; 16]; 16]);
        for x in 0..16 {
            for z in 0..16 {
                let xg = 16 * chunk_coord.x + (x as i32);
                let zg = 16 * chunk_coord.z + (z as i32);

                let elevation = self.elevation_noise.get(xg, zg);
                let biome = self.biome_noise.get(xg, zg, elevation);
                biome_map[x as usize][z as usize] = biome;
                height_map[x as usize][z as usize] = elevation;
                for y in 0..16 {
                    let offset = ChunkOffset { x, y, z };
                    let block_coord = chunk_coord.with_offset(offset);

                    let vert_offset = block_coord.y - (elevation as i32);
                    let block = match biome {
                        Biome::DefaultGrassy => {
                            self.generate_default_biome(vert_offset, block_coord)
                        }
                        Biome::SandyBeach => self.generate_sandy_beach(vert_offset, block_coord),
                        Biome::Desert => self.generate_desert(vert_offset, block_coord),
                    };
                    chunk.set_block(offset, block, None);
                }
            }
        }
        self.generate_vegetation(chunk_coord, chunk, &height_map, &biome_map);
    }
}

impl DefaultMapgen {
    #[inline]
    fn generate_ore(&self, coord: BlockCoordinate) -> BlockTypeHandle {
        for (ore, noise) in &self.ores {
            let cutoff = ore
                .noise_cutoff
                .clamped_sample(-(coord.y as f64))
                .unwrap_or(0.);
            let noise_coord = [
                coord.x as f64 / ore.noise_scale.0,
                coord.y as f64 / ore.noise_scale.1,
                coord.z as f64 / ore.noise_scale.2,
            ];
            let sample = noise.get(noise_coord);
            if sample < cutoff {
                return ore.block.0;
            }
        }

        self.stone
    }

    #[inline]
    fn fast_uniform_2d(&self, x: i32, z: i32, seed: u32) -> f64 {
        // Ugly, awful RNG. However, it's stable and passes enough simple correlation tests
        // to be useful for putting trees into a voxel game.
        const K: u64 = 0x517cc1b727220a95;
        const M: u64 = 0x72c07af023017001;

        let mut hash = seed as u64;
        hash = hash
            .rotate_left(5)
            .bitxor((x as u64).wrapping_mul(M))
            .swap_bytes()
            .wrapping_mul(K);
        hash = hash
            .rotate_left(5)
            .bitxor((z as u64).wrapping_mul(M))
            .swap_bytes()
            .wrapping_mul(K);
        hash = hash
            .rotate_left(5)
            .bitxor((seed as u64).wrapping_mul(M))
            .swap_bytes()
            .wrapping_mul(K);
        (hash as f64) / (u64::MAX as f64)
    }

    fn generate_vegetation(
        &self,
        chunk_coord: ChunkCoordinate,
        chunk: &mut MapChunk,
        heightmap: &[[f64; 16]; 16],
        biome_map: &[[Biome; 16]; 16],
    ) {
        for i in -3..18 {
            for j in -3..18 {
                let block_xz = (chunk_coord.x * 16)
                    .checked_add(i)
                    .zip((chunk_coord.z * 16).checked_add(j));
                if let Some((x, z)) = block_xz {
                    let (y, biome) =
                        if x.div_euclid(16) == chunk_coord.x && z.div_euclid(16) == chunk_coord.z {
                            (
                                heightmap[x.rem_euclid(16) as usize][z.rem_euclid(16) as usize] as i32,
                                biome_map[x.rem_euclid(16) as usize][z.rem_euclid(16) as usize],
                            )
                        } else {
                            let elevation = self.elevation_noise.get(x, z);
                            let biome = self.biome_noise.get(x, z, elevation);
                            (elevation as i32, biome)
                        };
                    if y <= 0 {
                        continue;
                    }
                    match biome {
                        Biome::DefaultGrassy => {
                            let tree_value = self.fast_uniform_2d(x, z, self.seed);
                            let tree_cutoff = self.tree_density_noise.get([
                                (x as f64) * TREE_DENSITY_INPUT_SCALE,
                                (z as f64) * TREE_DENSITY_INPUT_SCALE,
                            ]) * TREE_DENSITY_OUTPUT_SCALE
                                + TREE_DENSITY_OUTPUT_OFFSET;
                            if tree_value < tree_cutoff {
                                self.make_tree(chunk_coord, chunk, x, y, z);
                            }
                        },
                        Biome::Desert => {
                            let cactus_value = self.fast_uniform_2d(x, z, self.seed.wrapping_add(1));
                            let cactus_cutoff = self.cactus_density_noise.get([
                                (x as f64) * CACTUS_DENSITY_INPUT_SCALE,
                                (z as f64) * CACTUS_DENSITY_INPUT_SCALE,
                            ]) * CACTUS_DENSITY_OUTPUT_SCALE
                                + CACTUS_DENSITY_OUTPUT_OFFSET;
                            if cactus_value < cactus_cutoff {
                                let cactus_height =  (2.0 * self.fast_uniform_2d(x, z, self.seed.wrapping_add(2)) + 2.5) as i32;
                                self.make_cactus(chunk_coord, chunk, x, y, z, cactus_height);
                            }
                        },
                        Biome::SandyBeach => {
                            // TODO beach plants?
                        }
                    }
                }
            }
        }
    }

    // Generate a tree at the given location. The y coordinate represents the dirt block just below the tree.
    fn make_tree(
        &self,
        chunk_coord: ChunkCoordinate,
        chunk: &mut MapChunk,
        x: i32,
        y: i32,
        z: i32,
    ) {
        for h in 1..=4 {
            let coord = BlockCoordinate::new(x, y + h, z);
            if coord.chunk() == chunk_coord {
                chunk.set_block(coord.offset(), self.maple_tree, None);
            }
        }
        for h in 3..=5 {
            for i in -2..=2 {
                for j in -2..=2 {
                    if i == 0 && j == 0 && h <= 4 {
                        continue;
                    }
                    let coord = BlockCoordinate::new(x + i, y + h, z + j);
                    if coord.chunk() == chunk_coord {
                        chunk.set_block(coord.offset(), self.maple_leaves, None);
                    }
                }
            }
        }
    }

    // Generate a cactus at the given location. The y coordinate represents the dirt block just below the cactus.
    fn make_cactus(
            &self,
            chunk_coord: ChunkCoordinate,
            chunk: &mut MapChunk,
            x: i32,
            y: i32,
            z: i32,
            height: i32
        ) {
            for h in 1..=height {
                let coord = BlockCoordinate::new(x, y + h, z);
                if coord.chunk() == chunk_coord {
                    chunk.set_block(coord.offset(), self.cactus, None);
                }
            }
        }

    fn generate_default_biome(
        &self,
        vert_offset: i32,
        block_coord: BlockCoordinate,
    ) -> perovskite_core::block_id::BlockId {
        if vert_offset > 0 {
            if block_coord.y > 0 {
                self.air
            } else {
                self.water
            }
        } else if vert_offset == 0 && block_coord.y >= 0 {
            self.dirt_grass
        } else if vert_offset > -3 {
            // todo variable depth of dirt
            self.dirt
        } else {
            self.generate_ore(block_coord)
        }
    }

    fn generate_sandy_beach(
        &self,
        vert_offset: i32,
        block_coord: BlockCoordinate,
    ) -> perovskite_core::block_id::BlockId {
        if vert_offset > 0 {
            if block_coord.y > 0 {
                self.air
            } else {
                self.water
            }
        } else if vert_offset > -3 {
            // todo variable depth
            self.sand
        } else {
            self.generate_ore(block_coord)
        }
    }

    fn generate_desert(
        &self,
        vert_offset: i32,
        block_coord: BlockCoordinate,
    ) -> perovskite_core::block_id::BlockId {
        if vert_offset > 0 {
            if block_coord.y > 0 {
                self.air
            } else {
                self.water
            }
        } else if vert_offset > -3 {
            // todo variable depth
            self.desert_sand
        } else if vert_offset > -10 {
            // todo variable depth
            self.desert_stone
        } else {
            self.generate_ore(block_coord)
        }
    }
}

pub(crate) fn build_mapgen(
    blocks: Arc<BlockTypeManager>,
    seed: u32,
    ores: Vec<OreDefinition>,
) -> Arc<dyn MapgenInterface> {
    Arc::new(DefaultMapgen {
        air: blocks.get_by_name(AIR).expect("air"),
        dirt: blocks.get_by_name(DIRT.0).expect("dirt"),
        dirt_grass: blocks.get_by_name(DIRT_WITH_GRASS.0).expect("dirt_grass"),
        sand: blocks.get_by_name(SAND.0).expect("sand"),
        stone: blocks.get_by_name(STONE.0).expect("stone"),
        desert_sand: blocks.get_by_name(DESERT_SAND.0).expect("desert_sand"),
        desert_stone: blocks.get_by_name(DESERT_STONE.0).expect("desert_stone"),
        // TODO 0xfff is a magic number, give it a real constant definition
        water: blocks
            .get_by_name(WATER.0)
            .expect("water")
            .with_variant(0xfff)
            .unwrap(),
        maple_tree: blocks.get_by_name(MAPLE_TREE.0).expect("maple_tree"),
        maple_leaves: blocks.get_by_name(MAPLE_LEAVES.0).expect("maple_leaves"),
        cactus: blocks.get_by_name(CACTUS.0).expect("cactus"),

        elevation_noise: ElevationNoise::new(seed),
        biome_noise: BiomeNoise::new(seed),
        tree_density_noise: noise::Billow::new(seed.wrapping_add(2)),
        cactus_density_noise: noise::Billow::new(seed.wrapping_add(5)),
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
