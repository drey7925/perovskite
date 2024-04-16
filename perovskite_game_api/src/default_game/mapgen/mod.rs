use std::{ops::BitXor, sync::Arc};

use noise::{MultiFractal, NoiseFn};
use perovskite_core::{
    block_id::BlockId,
    constants::blocks::AIR,
    coordinates::{BlockCoordinate, ChunkCoordinate, ChunkOffset},
};
use perovskite_server::game_state::{
    blocks::{BlockTypeHandle, BlockTypeManager},
    game_map::MapChunk,
    mapgen::MapgenInterface,
};
use rand::seq::SliceRandom;

use super::{
    basic_blocks::{
        DESERT_SAND, DESERT_STONE, DIRT, DIRT_WITH_GRASS, LIMESTONE_DARK, LIMESTONE_LIGHT, SAND,
        STONE, WATER,
    },
    foliage::{CACTUS, MAPLE_LEAVES, MAPLE_TREE, TALL_GRASS, TERRESTRIAL_FLOWERS},
};

mod karst;

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

const TALL_GRASS_DENSITY_INPUT_SCALE: f64 = 1.0 / 120.0;
const TALL_GRASS_DENSITY_OUTPUT_SCALE: f64 = 0.03125;
const TALL_GRASS_DENSITY_OUTPUT_OFFSET: f64 = 0.03125;

const FLOWER_DENSITY_INPUT_SCALE: f64 = 1.0 / 240.0;
const FLOWER_DENSITY_OUTPUT_SCALE: f64 = 0.03125;
const FLOWER_DENSITY_OUTPUT_OFFSET: f64 = 0.03125;

const BEACH_TENDENCY_INPUT_SCALE: f64 = 1.0 / 240.0;
const DESERT_TENDENCY_INPUT_SCALE: f64 = 1.0 / 480.0;
const KARST_TENDENCY_INPUT_SCALE: f64 = 1.0 / 7200.0;

// Next seed offset: 15
// Offsets 10/11/12/13/14 are used for karst

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Biome {
    DefaultGrassy,
    SandyBeach,
    Desert,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Macrobiome {
    RollingHills,
    Karst,
}

struct MacrobiomeNoise {
    karst_tendency: noise::SuperSimplex,
}
impl MacrobiomeNoise {
    fn new(seed: u32) -> MacrobiomeNoise {
        MacrobiomeNoise {
            karst_tendency: noise::SuperSimplex::new(seed.wrapping_add(9)),
        }
    }
    fn get(&self, x: i32, z: i32) -> (Macrobiome, f64) {
        let karst_value = self.karst_tendency.get([
            x as f64 * KARST_TENDENCY_INPUT_SCALE,
            z as f64 * KARST_TENDENCY_INPUT_SCALE,
        ]);
        if karst_value > 0.3 {
            (
                Macrobiome::Karst,
                ((0.5 - karst_value) * 5.0).clamp(0.0, 1.0),
            )
        } else {
            (Macrobiome::RollingHills, 0.0)
        }
    }
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
        Biome::DefaultGrassy
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

struct CaveNoise {
    cave: noise::Fbm<noise::SuperSimplex>,
}
impl CaveNoise {
    fn new(seed: u32) -> CaveNoise {
        CaveNoise {
            cave: noise::Fbm::new(seed.wrapping_add(6)),
        }
    }

    // Returns (is_cave, ore_bias)
    fn get(&self, block_coord: BlockCoordinate) -> (bool, f64) {
        const CAVE_NOISE_INPUT_SCALE: f64 = 1.0 / 240.0;
        const CAVE_SQUASH_FACTOR: f64 = 4.0;
        // If we're at -10 or above, bias against caves. counter_bias ranges from 0.0 to 0.20
        let counter_bias = (block_coord.y as f64 + 20.0).clamp(0.0, 20.0) / 50.0;
        let noise = self.cave.get([
            block_coord.x as f64 * CAVE_NOISE_INPUT_SCALE,
            block_coord.y as f64 * CAVE_NOISE_INPUT_SCALE * CAVE_SQUASH_FACTOR,
            block_coord.z as f64 * CAVE_NOISE_INPUT_SCALE,
        ]);
        let threshold = 0.6 + counter_bias;
        let is_cave = noise > threshold;
        // At the edge, bias by up to 1.0. If the noise is well below the threshold, clamp to 0.
        let ore_bias = (noise - threshold + 0.1).clamp(0.0, 0.1) * 1.0;
        (is_cave, ore_bias)
    }
}

/// An ore that the mapgen should generate. Note: This struct is subject to being extended with new fields
/// in the future
pub struct OreDefinition {
    pub block: BlockId,
    /// When the generated noise is larger than this value, the ore is generated.
    /// TODO figure out and document the range of the noise
    /// This is expressed as a spline with the input being the depth at which we are generating ore.
    pub noise_cutoff: splines::spline::Spline<f64, f64>,
    /// The scale of the noise that generates this ore. The higher the scale, the larger and more spread-apart the clumps are.
    pub noise_scale: (f64, f64, f64),
    /// How much to multiply the ore bias from caves. This adjusts noise_scale. 0.0 -> no adjustment near caves.
    /// Positive values -> More likely to be near cave
    /// Negative values -> Less likely to be near cave
    pub cave_bias_effect: f64,
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
    tall_grass: BlockTypeHandle,
    flowers: Vec<BlockTypeHandle>,

    elevation_noise: ElevationNoise,
    tree_density_noise: noise::Billow<noise::SuperSimplex>,

    flower_density_noise: noise::Billow<noise::SuperSimplex>,
    cactus_density_noise: noise::Billow<noise::SuperSimplex>,
    tall_grass_density_noise: noise::Billow<noise::SuperSimplex>,

    karst_noise: karst::KarstGenerator,

    biome_noise: BiomeNoise,
    macrobiome_noise: MacrobiomeNoise,
    cave_noise: CaveNoise,
    ores: Vec<(OreDefinition, noise::SuperSimplex)>,
    seed: u32,
}

impl DefaultMapgen {
    #[inline]
    fn prefill_single(&self, xg: i32, zg: i32) -> (Macrobiome, f64) {
        let (macrobiome, rolling_hills_blend, karst_blend) = self.macrobiome_single(xg, zg);
        let mut elevation = 0.0;
        if rolling_hills_blend > 0.0 {
            elevation += self.elevation_noise.get(xg, zg) * rolling_hills_blend;
        }
        if karst_blend > 0.0 {
            elevation += self.karst_noise.height(xg, zg) * karst_blend;
        }
        (macrobiome, elevation)
    }

    fn macrobiome_single(&self, xg: i32, zg: i32) -> (Macrobiome, f64, f64) {
        let (macrobiome, blend_factor) = self.macrobiome_noise.get(xg, zg);
        match macrobiome {
            Macrobiome::RollingHills => (Macrobiome::RollingHills, 1.0, 0.0),
            Macrobiome::Karst => (Macrobiome::Karst, blend_factor, 1.0 - blend_factor),
        }
    }

    fn prefill(
        &self,
        chunk_coord: ChunkCoordinate,
        height_map: &mut [[f64; 16]; 16],
        macrobiome_map: &mut [[Macrobiome; 16]; 16],
    ) {
        let mut rolling_hills_blend = [[0.0; 16]; 16];
        let mut karst_blend = [[0.0; 16]; 16];

        for x in 0..16 {
            for z in 0..16 {
                let xg = (chunk_coord.x * 16) + x as i32;
                let zg = (chunk_coord.z * 16) + z as i32;
                (
                    macrobiome_map[x][z],
                    rolling_hills_blend[x][z],
                    karst_blend[x][z],
                ) = self.macrobiome_single(xg, zg);
            }
        }
        if rolling_hills_blend.iter().flatten().any(|&x| x > 0.0) {
            for x in 0..16 {
                for z in 0..16 {
                    height_map[x][z] += rolling_hills_blend[x][z]
                        * self.elevation_noise.get(
                            (chunk_coord.x * 16) + x as i32,
                            (chunk_coord.z * 16) + z as i32,
                        );
                }
            }
        }
        if karst_blend.iter().flatten().any(|&x| x > 0.0) {
            for x in 0..16 {
                for z in 0..16 {
                    height_map[x][z] += karst_blend[x][z]
                        * self.karst_noise.height(
                            (chunk_coord.x * 16) + x as i32,
                            (chunk_coord.z * 16) + z as i32,
                        );
                }
            }
        }
    }
}

impl MapgenInterface for DefaultMapgen {
    fn fill_chunk(&self, chunk_coord: ChunkCoordinate, chunk: &mut MapChunk) {
        // todo subdivide by surface vs underground, etc. This is a very minimal MVP
        let mut height_map = Box::new([[0.0; 16]; 16]);
        let mut macrobiome_map = Box::new([[Macrobiome::RollingHills; 16]; 16]);
        let mut biome_map = Box::new([[Biome::DefaultGrassy; 16]; 16]);

        self.prefill(chunk_coord, &mut height_map, &mut macrobiome_map);

        for x in 0..16 {
            for z in 0..16 {
                let macrobiome = macrobiome_map[x as usize][z as usize];
                match macrobiome {
                    Macrobiome::RollingHills => {
                        self.generate_rolling_hills(
                            chunk_coord,
                            x,
                            z,
                            &mut biome_map,
                            &height_map,
                            chunk,
                        );
                    }
                    Macrobiome::Karst => {
                        self.karst_noise
                            .generate(chunk_coord, x, z, &height_map, chunk, |coord| {
                                self.generate_ore(coord)
                            });
                    }
                }
            }
        }
        self.generate_vegetation(chunk_coord, chunk, &height_map, &biome_map);
    }
}

impl DefaultMapgen {
    #[inline]
    fn generate_ore(&self, coord: BlockCoordinate) -> BlockTypeHandle {
        let (is_cave, cave_bias) = self.cave_noise.get(coord);
        if is_cave {
            // TODO - lava, etc?
            return self.air;
        }
        for (ore, noise) in &self.ores {
            let mut cutoff = ore
                .noise_cutoff
                .clamped_sample(-(coord.y as f64))
                .unwrap_or(0.);
            cutoff -= ore.cave_bias_effect * cave_bias;
            let noise_coord = [
                coord.x as f64 / ore.noise_scale.0,
                coord.y as f64 / ore.noise_scale.1,
                coord.z as f64 / ore.noise_scale.2,
            ];
            let sample = noise.get(noise_coord);
            if sample > cutoff {
                return ore.block;
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
                    let (y, biome) = if x.div_euclid(16) == chunk_coord.x
                        && z.div_euclid(16) == chunk_coord.z
                    {
                        (
                            heightmap[x.rem_euclid(16) as usize][z.rem_euclid(16) as usize] as i32,
                            biome_map[x.rem_euclid(16) as usize][z.rem_euclid(16) as usize],
                        )
                    } else {
                        let (macrobiome, elevation) = self.prefill_single(x, z);

                        let biome = match macrobiome {
                            Macrobiome::RollingHills => self.biome_noise.get(x, z, elevation),
                            // TODO karst biomes
                            Macrobiome::Karst => Biome::DefaultGrassy,
                        };
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

                            let flower_value =
                                self.fast_uniform_2d(x, z, self.seed.wrapping_add(3));
                            let flower_cutoff = self.flower_density_noise.get([
                                (x as f64) * FLOWER_DENSITY_INPUT_SCALE,
                                (z as f64) * FLOWER_DENSITY_INPUT_SCALE,
                            ]) * FLOWER_DENSITY_OUTPUT_SCALE
                                + FLOWER_DENSITY_OUTPUT_OFFSET;
                            if flower_value < flower_cutoff {
                                self.make_simple_foliage(
                                    chunk_coord,
                                    chunk,
                                    x,
                                    y,
                                    z,
                                    *self.flowers.choose(&mut rand::thread_rng()).unwrap(),
                                );
                            }

                            let tall_grass_value =
                                self.fast_uniform_2d(x, z, self.seed.wrapping_add(4));
                            let tall_grass_cutoff = self.flower_density_noise.get([
                                (x as f64) * FLOWER_DENSITY_INPUT_SCALE,
                                (z as f64) * FLOWER_DENSITY_INPUT_SCALE,
                            ]) * FLOWER_DENSITY_OUTPUT_SCALE
                                + FLOWER_DENSITY_OUTPUT_OFFSET;
                            if tall_grass_value < tall_grass_cutoff {
                                self.make_simple_foliage(
                                    chunk_coord,
                                    chunk,
                                    x,
                                    y,
                                    z,
                                    self.tall_grass,
                                );
                            }
                        }
                        Biome::Desert => {
                            let cactus_value =
                                self.fast_uniform_2d(x, z, self.seed.wrapping_add(1));
                            let cactus_cutoff = self.cactus_density_noise.get([
                                (x as f64) * CACTUS_DENSITY_INPUT_SCALE,
                                (z as f64) * CACTUS_DENSITY_INPUT_SCALE,
                            ]) * CACTUS_DENSITY_OUTPUT_SCALE
                                + CACTUS_DENSITY_OUTPUT_OFFSET;
                            if cactus_value < cactus_cutoff {
                                let cactus_height =
                                    (2.0 * self.fast_uniform_2d(x, z, self.seed.wrapping_add(2))
                                        + 2.5) as i32;
                                self.make_cactus(chunk_coord, chunk, x, y, z, cactus_height);
                            }
                        }
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
        height: i32,
    ) {
        for h in 1..=height {
            let coord = BlockCoordinate::new(x, y + h, z);
            if coord.chunk() == chunk_coord {
                chunk.set_block(coord.offset(), self.cactus, None);
            }
        }
    }

    fn generate_default_biome<F>(
        &self,
        vert_offset: i32,
        block_coord: BlockCoordinate,
        gen_ore: F,
    ) -> perovskite_core::block_id::BlockId
    where
        F: Fn() -> perovskite_core::block_id::BlockId,
    {
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
            gen_ore()
        }
    }

    fn generate_sandy_beach<F>(
        &self,
        vert_offset: i32,
        block_coord: BlockCoordinate,
        gen_ore: F,
    ) -> perovskite_core::block_id::BlockId
    where
        F: Fn() -> perovskite_core::block_id::BlockId,
    {
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
            gen_ore()
        }
    }

    fn generate_desert<F>(
        &self,
        vert_offset: i32,
        block_coord: BlockCoordinate,
        gen_ore: F,
    ) -> perovskite_core::block_id::BlockId
    where
        F: Fn() -> perovskite_core::block_id::BlockId,
    {
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
            gen_ore()
        }
    }

    fn make_simple_foliage(
        &self,
        chunk_coord: ChunkCoordinate,
        chunk: &mut MapChunk,
        x: i32,
        y: i32,
        z: i32,
        block: BlockId,
    ) {
        let coord = BlockCoordinate::new(x, y + 1, z);
        if coord.chunk() == chunk_coord {
            chunk.set_block(coord.offset(), block, None);
        }
    }

    #[inline(always)]
    fn generate_rolling_hills(
        &self,
        chunk_coord: ChunkCoordinate,
        x: u8,
        z: u8,
        biome_map: &mut [[Biome; 16]; 16],
        height_map: &[[f64; 16]; 16],
        chunk: &mut MapChunk,
    ) {
        let xg = 16 * chunk_coord.x + (x as i32);
        let zg = 16 * chunk_coord.z + (z as i32);

        let elevation = height_map[x as usize][z as usize];
        let biome = self.biome_noise.get(xg, zg, elevation);

        biome_map[x as usize][z as usize] = biome;

        for y in 0..16 {
            let offset = ChunkOffset { x, y, z };
            let block_coord = chunk_coord.with_offset(offset);

            let vert_offset = block_coord.y - (elevation as i32);

            let gen_ore = || self.generate_ore(block_coord);

            let block = match biome {
                Biome::DefaultGrassy => {
                    self.generate_default_biome(vert_offset, block_coord, gen_ore)
                }
                Biome::SandyBeach => self.generate_sandy_beach(vert_offset, block_coord, gen_ore),
                Biome::Desert => self.generate_desert(vert_offset, block_coord, gen_ore),
            };
            chunk.set_block(offset, block, None);
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
        tall_grass: blocks.get_by_name(TALL_GRASS.0).expect("tall_grass"),

        flowers: TERRESTRIAL_FLOWERS
            .iter()
            .map(|f| blocks.get_by_name(f.0 .0).expect("flower missing"))
            .collect(),

        cactus: blocks.get_by_name(CACTUS.0).expect("cactus"),

        elevation_noise: ElevationNoise::new(seed),
        biome_noise: BiomeNoise::new(seed),
        macrobiome_noise: MacrobiomeNoise::new(seed),
        karst_noise: karst::KarstGenerator::new(seed, &blocks),
        cave_noise: CaveNoise::new(seed),
        tree_density_noise: noise::Billow::new(seed.wrapping_add(2)),
        flower_density_noise: noise::Billow::new(seed.wrapping_add(7)),
        tall_grass_density_noise: noise::Billow::new(seed.wrapping_add(8)),
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
