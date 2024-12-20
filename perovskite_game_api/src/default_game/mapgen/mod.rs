use std::ops::RangeInclusive;
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
        CLAY, DESERT_SAND, DESERT_STONE, DIRT, DIRT_WITH_GRASS, SAND, SILT_DAMP, SILT_DRY, STONE,
        WATER,
    },
    foliage::{
        CACTUS, MAPLE_LEAVES, MAPLE_TREE, MARSH_GRASS, TALL_GRASS, TALL_REED, TERRESTRIAL_FLOWERS,
    },
};

mod karst;

const TREE_DENSITY_INPUT_SCALE: f64 = 1.0 / 240.0;
const TREE_DENSITY_OUTPUT_SCALE: f64 = 0.0625;
const TREE_DENSITY_OUTPUT_OFFSET: f64 = 0.03125;
const CACTUS_DENSITY_INPUT_SCALE: f64 = 1.0 / 120.0;
const CACTUS_DENSITY_OUTPUT_SCALE: f64 = 0.03125;
const CACTUS_DENSITY_OUTPUT_OFFSET: f64 = 0.03125;

const TALL_GRASS_DENSITY_INPUT_SCALE: f64 = 1.0 / 120.0;
const TALL_GRASS_DENSITY_OUTPUT_SCALE: f64 = 0.03125;
const TALL_GRASS_DENSITY_OUTPUT_OFFSET: f64 = 0.03125;

const VINES_DENSITY_INPUT_SCALE: f64 = 1.0 / 120.0;
const VINES_DENSITY_OUTPUT_SCALE: f64 = 0.1;
const VINES_DENSITY_OUTPUT_OFFSET: f64 = 0.1;

const FLOWER_DENSITY_INPUT_SCALE: f64 = 1.0 / 240.0;
const FLOWER_DENSITY_OUTPUT_SCALE: f64 = 0.03125;
const FLOWER_DENSITY_OUTPUT_OFFSET: f64 = 0.03125;

const BEACH_TENDENCY_INPUT_SCALE: f64 = 1.0 / 240.0;
const DESERT_TENDENCY_INPUT_SCALE: f64 = 1.0 / 480.0;
const KARST_TENDENCY_INPUT_SCALE: f64 = 1.0 / 7200.0;

// Next seed offset: 23
// Offsets 10-18 are used for karst

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Biome {
    DefaultGrassy,
    SandyBeach,
    Desert,
    Saltmarsh,
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
    saltmarsh_tendency: noise::SuperSimplex,
}
impl BiomeNoise {
    fn new(seed: u32) -> BiomeNoise {
        BiomeNoise {
            sand_tendency: noise::SuperSimplex::new(seed.wrapping_add(3)),
            desert_tendency: noise::SuperSimplex::new(seed.wrapping_add(4)),
            saltmarsh_tendency: noise::SuperSimplex::new(seed.wrapping_add(22)),
        }
    }
    fn get(&self, x: i32, z: i32, elevation: f64, coastal_flatness: f64) -> Biome {
        let sand_value = self.sand_tendency.get([
            x as f64 * BEACH_TENDENCY_INPUT_SCALE,
            z as f64 * BEACH_TENDENCY_INPUT_SCALE,
        ]);
        if sand_value > 0.2 {
            let excess = sand_value - 0.2;
            // sandy beach
            let range = (excess * -10.0)..=(excess * 3.0 + 0.5);
            if range.contains(&elevation) {
                return Biome::SandyBeach;
            }
        };
        let marsh_value = self.saltmarsh_tendency.get([
            x as f64 * BEACH_TENDENCY_INPUT_SCALE,
            z as f64 * BEACH_TENDENCY_INPUT_SCALE,
        ]) + (coastal_flatness * 3.0);
        if marsh_value > 0.5 {
            let excess = marsh_value - 0.4;
            // sandy beach
            let range = (excess * -5.0)..=(excess * 1.0 + 0.5);
            if range.contains(&elevation) {
                return Biome::Saltmarsh;
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
    coarse_nr: noise::RidgedMulti<noise::SuperSimplex>,
    fine: noise::SuperSimplex,
    extra_coarse: noise::SuperSimplex,
    ridge_strength: noise::SuperSimplex,
    sea_level_flatness: noise::SuperSimplex,
}
impl ElevationNoise {
    fn new(seed: u32) -> ElevationNoise {
        ElevationNoise {
            coarse: noise::RidgedMulti::new(seed).set_persistence(0.8),
            coarse_nr: noise::RidgedMulti::new(seed).set_persistence(0.8),
            fine: noise::SuperSimplex::new(seed.wrapping_add(1)),
            extra_coarse: noise::SuperSimplex::new(seed.wrapping_add(19)),
            ridge_strength: noise::SuperSimplex::new(seed.wrapping_add(20)),
            sea_level_flatness: noise::SuperSimplex::new(seed.wrapping_add(21)),
        }
    }

    fn get(&self, x: i32, z: i32) -> (f64, f64) {
        const ELEVATION_FINE_INPUT_SCALE: f64 = 1.0 / 80.0;
        const COASTAL_FLATNESS_SCALE: f64 = 1.0 / 160.0;
        const ELEVATION_FINE_OUTPUT_SCALE: f64 = 10.0;
        const ELEVATION_COARSE_INPUT_SCALE: f64 = 1.0 / 1200.0;
        const ELEVATION_EXTRA_COARSE_INPUT_SCALE: f64 = 1.0 / 4800.0;
        const ELEVATION_COARSE_OUTPUT_SCALE: f64 = 60.0;
        const ELEVATION_EXTRA_COARSE_OUTPUT_SCALE: f64 = 40.0;
        const ELEVATION_OFFSET: f64 = 20.0;

        let coarse_pos = [
            x as f64 * ELEVATION_COARSE_INPUT_SCALE,
            z as f64 * ELEVATION_COARSE_INPUT_SCALE,
        ];
        let coarse_ridge_height = self.coarse.get(coarse_pos) * ELEVATION_COARSE_OUTPUT_SCALE;
        let coarse_nr_height = self.coarse_nr.get(coarse_pos) * ELEVATION_COARSE_OUTPUT_SCALE;

        let extra_coarse_pos = [
            x as f64 * ELEVATION_COARSE_INPUT_SCALE,
            z as f64 * ELEVATION_COARSE_INPUT_SCALE,
        ];
        let coarse_blend = (self.ridge_strength.get(extra_coarse_pos) + 0.5).clamp(0.0, 1.0);
        let coarse_height =
            (coarse_ridge_height * coarse_blend) + (coarse_nr_height * (1.0 - coarse_blend));

        let extra_coarse_height =
            self.extra_coarse.get(extra_coarse_pos) * ELEVATION_EXTRA_COARSE_OUTPUT_SCALE;

        let fine_pos = [
            x as f64 * ELEVATION_FINE_INPUT_SCALE,
            z as f64 * ELEVATION_FINE_INPUT_SCALE,
        ];

        let flatness_pos = [
            x as f64 * COASTAL_FLATNESS_SCALE,
            z as f64 * COASTAL_FLATNESS_SCALE,
        ];
        let fine_height = self.fine.get(fine_pos) * ELEVATION_FINE_OUTPUT_SCALE;
        let raw_height = coarse_height + fine_height + ELEVATION_OFFSET + extra_coarse_height;
        // Create more flat ground near water level
        let mut adjusted_height = raw_height;
        let flatness = self.sea_level_flatness.get(flatness_pos) * 1.5;
        let flatness_control = (flatness * 3.5 + 0.45).clamp(0.0, 4.0);
        if flatness_control > 0.1 {
            adjusted_height -= flatness_control * (adjusted_height / flatness_control).tanh()
        }
        if adjusted_height < -1.0 {
            adjusted_height /= 2.0;
        }
        (adjusted_height, flatness)
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
    stone: BlockTypeHandle,
    sand: BlockTypeHandle,
    silt_dry: BlockTypeHandle,
    silt_damp: BlockTypeHandle,
    clay: BlockTypeHandle,

    desert_stone: BlockTypeHandle,
    desert_sand: BlockTypeHandle,

    water: BlockTypeHandle,
    // todo organize the foliage (and more of the blocks in general) in a better way
    maple_tree: BlockTypeHandle,
    maple_leaves: BlockTypeHandle,
    cactus: BlockTypeHandle,
    tall_grass: BlockTypeHandle,
    flowers: Vec<BlockTypeHandle>,

    marsh_grass: BlockTypeHandle,
    tall_reed: BlockTypeHandle,

    elevation_noise: ElevationNoise,
    tree_density_noise: noise::Billow<noise::SuperSimplex>,

    flower_density_noise: noise::Billow<noise::SuperSimplex>,
    cactus_density_noise: noise::Billow<noise::SuperSimplex>,
    tall_grass_density_noise: noise::Billow<noise::SuperSimplex>,

    karst_noise: karst::KarstGenerator,

    // These tiles are used for the temporary prespawned railways until I have a chance to implement
    // a suitable gameplay experience for creating long tracks
    rail_testonly: BlockTypeHandle,
    signal_testonly: BlockTypeHandle,
    glass_testonly: BlockTypeHandle,

    biome_noise: BiomeNoise,
    macrobiome_noise: MacrobiomeNoise,
    cave_noise: CaveNoise,
    ores: Vec<(OreDefinition, noise::SuperSimplex)>,
    seed: u32,
}

impl DefaultMapgen {
    #[inline]
    fn prefill_single(&self, xg: i32, zg: i32) -> (Macrobiome, f64, f64) {
        let (macrobiome, rolling_hills_blend, karst_blend) = self.macrobiome_single(xg, zg);
        let mut elevation = 0.0;
        let mut flatness = 0.0;
        if rolling_hills_blend > 0.0 {
            let (rh_elevation, coastal_flatness) = self.elevation_noise.get(xg, zg);
            flatness = coastal_flatness;
            elevation += rh_elevation * rolling_hills_blend;
        }
        if karst_blend > 0.0 {
            let (raw_elev, _floor, _ceil) = self.karst_noise.height(xg, zg);
            elevation += raw_elev * karst_blend;
        }
        (macrobiome, elevation, flatness)
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
        cave_floor_map: &mut [[f64; 16]; 16],
        cave_ceil_map: &mut [[f64; 16]; 16],
        macrobiome_map: &mut [[Macrobiome; 16]; 16],
        biome_map: &mut [[Biome; 16]; 16],
    ) {
        let mut rolling_hills_blend = [[0.0; 16]; 16];
        let mut karst_blend = [[0.0; 16]; 16];
        let mut coastal_flatness_map = [[0.0; 16]; 16];

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
        let any_rolling_hills = rolling_hills_blend.iter().flatten().any(|&x| x > 0.0);
        if any_rolling_hills {
            for x in 0..16 {
                for z in 0..16 {
                    let (elevation, flatness) = self.elevation_noise.get(
                        (chunk_coord.x * 16) + x as i32,
                        (chunk_coord.z * 16) + z as i32,
                    );
                    coastal_flatness_map[x][z] = flatness;
                    height_map[x][z] += rolling_hills_blend[x][z] * elevation;
                }
            }
        }
        if karst_blend.iter().flatten().any(|&x| x > 0.0) {
            for x in 0..16 {
                for z in 0..16 {
                    let (raw_height, raw_cave_floor, raw_cave_ceil) = self.karst_noise.height(
                        (chunk_coord.x * 16) + x as i32,
                        (chunk_coord.z * 16) + z as i32,
                    );
                    height_map[x][z] += karst_blend[x][z] * raw_height;
                    cave_floor_map[x][z] += karst_blend[x][z] * raw_cave_floor;
                    cave_ceil_map[x][z] += karst_blend[x][z] * raw_cave_ceil;
                }
            }
        }
        if any_rolling_hills {
            for x in 0..16 {
                for z in 0..16 {
                    biome_map[x][z] = self.biome_noise.get(
                        (chunk_coord.x * 16) + x as i32,
                        (chunk_coord.z * 16) + z as i32,
                        height_map[x][z],
                        coastal_flatness_map[x][z],
                    );
                }
            }
        }
    }
}

impl MapgenInterface for DefaultMapgen {
    fn fill_chunk(&self, chunk_coord: ChunkCoordinate, chunk: &mut MapChunk) {
        // todo subdivide by surface vs underground, etc. This is a very minimal MVP
        let mut height_map = [[0.0; 16]; 16];
        let mut cave_floor_map: [[f64; 16]; 16] = [[0.0; 16]; 16];
        let mut cave_ceil_map: [[f64; 16]; 16] = [[0.0; 16]; 16];
        let mut macrobiome_map = [[Macrobiome::RollingHills; 16]; 16];
        let mut biome_map = [[Biome::DefaultGrassy; 16]; 16];

        self.prefill(
            chunk_coord,
            &mut height_map,
            &mut cave_floor_map,
            &mut cave_ceil_map,
            &mut macrobiome_map,
            &mut biome_map,
        );

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
                        self.karst_noise.generate(
                            chunk_coord,
                            x,
                            z,
                            &height_map,
                            &cave_floor_map,
                            &cave_ceil_map,
                            chunk,
                            |coord| self.generate_ore(coord),
                        );
                    }
                }
            }
        }
        self.generate_vegetation(chunk_coord, chunk, &height_map, &macrobiome_map, &biome_map);

        if chunk_coord.y == 0 {
            // Rails running along Z axis, every half kilometer
            if chunk_coord.x % 32 == 0 {
                for x in 6..10 {
                    for z in 0..16 {
                        chunk.set_block(ChunkOffset { x, y: 3, z }, self.stone, None);
                        if x == 7 || x == 8 {
                            chunk.set_block(
                                ChunkOffset { x, y: 4, z },
                                self.rail_testonly.with_variant(0).unwrap(),
                                None,
                            );
                        }
                        chunk.set_block(ChunkOffset { x, y: 5, z }, self.air, None);
                        chunk.set_block(
                            ChunkOffset { x, y: 6, z },
                            if z == 5 && chunk_coord.z % 4 == 0 {
                                self.glass_testonly
                            } else if z == 4 && chunk_coord.z % 4 == 0 && x == 7 {
                                self.signal_testonly.with_variant(16).unwrap()
                            } else if z == 6 && chunk_coord.z % 4 == 0 && x == 8 {
                                self.signal_testonly.with_variant(18).unwrap()
                            } else {
                                self.air
                            },
                            None,
                        );
                    }
                }
                if chunk_coord.z % 4 == 0 {
                    for x in [6, 9] {
                        // Gantries
                        chunk.set_block(ChunkOffset { x, y: 4, z: 5 }, self.glass_testonly, None);
                        chunk.set_block(ChunkOffset { x, y: 5, z: 5 }, self.glass_testonly, None);
                        chunk.set_block(ChunkOffset { x, y: 6, z: 5 }, self.glass_testonly, None);
                    }
                }
            }
        }
    }

    fn terrain_range_hint(&self, chunk_x: i32, chunk_z: i32) -> Option<RangeInclusive<i32>> {
        let xg0 = chunk_x * 16;
        let zg0 = chunk_z * 16;
        let mut max_h = f64::MIN;
        let mut min_h = f64::MAX;
        for (dx, dz) in [(0, 0), (0, 15), (15, 0), (15, 15)] {
            let xg = xg0 + dx;
            let zg = zg0 + dz;
            let (_, rh_blend, karst_blend) = self.macrobiome_single(xg, zg);
            let mut height = 0.0;
            if rh_blend > 0.0 {
                height += rh_blend * self.elevation_noise.get(xg, zg).0;
            }
            if karst_blend > 0.0 {
                height += karst_blend * self.karst_noise.profile(xg, zg);
                min_h = min_h.min(self.karst_noise.floor(xg, zg));
            }
            max_h = max_h.max(height);
            min_h = min_h.min(height);
        }
        let min_c = ((min_h - 16.0) as i32).max(-64).div_euclid(16);
        let max_c = ((max_h + 16.0) as i32).clamp(15, 4096).div_euclid(16);
        Some(min_c..=max_c)
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
        macrobiome_map: &[[Macrobiome; 16]; 16],
        biome_map: &[[Biome; 16]; 16],
    ) {
        for i in -3..18 {
            for j in -3..18 {
                let block_xz = (chunk_coord.x * 16)
                    .checked_add(i)
                    .zip((chunk_coord.z * 16).checked_add(j));
                if let Some((x, z)) = block_xz {
                    let (elevation, macrobiome, biome) = if x.div_euclid(16) == chunk_coord.x
                        && z.div_euclid(16) == chunk_coord.z
                    {
                        (
                            heightmap[x.rem_euclid(16) as usize][z.rem_euclid(16) as usize],
                            macrobiome_map[x.rem_euclid(16) as usize][z.rem_euclid(16) as usize],
                            biome_map[x.rem_euclid(16) as usize][z.rem_euclid(16) as usize],
                        )
                    } else {
                        let (macrobiome, elevation, flatness) = self.prefill_single(x, z);

                        let biome = match macrobiome {
                            Macrobiome::RollingHills => {
                                self.biome_noise.get(x, z, elevation, flatness)
                            }
                            // TODO karst biomes
                            Macrobiome::Karst => Biome::DefaultGrassy,
                        };
                        (elevation, macrobiome, biome)
                    };
                    let ground_y = elevation.floor() as i32;
                    if ground_y < 0 {
                        continue;
                    }
                    if ground_y < (chunk_coord.y * 16).saturating_sub(24) {
                        continue;
                    }
                    if ground_y > (chunk_coord.y * 16).saturating_add(24) {
                        continue;
                    }
                    match (macrobiome, biome) {
                        (Macrobiome::RollingHills, Biome::DefaultGrassy) => {
                            let tree_value = self.fast_uniform_2d(x, z, self.seed);
                            let tree_cutoff = self.tree_density_noise.get([
                                (x as f64) * TREE_DENSITY_INPUT_SCALE,
                                (z as f64) * TREE_DENSITY_INPUT_SCALE,
                            ]) * TREE_DENSITY_OUTPUT_SCALE
                                + TREE_DENSITY_OUTPUT_OFFSET;
                            if tree_value < tree_cutoff {
                                self.make_tree(chunk_coord, chunk, x, ground_y, z);
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
                                    ground_y,
                                    z,
                                    *self.flowers.choose(&mut rand::thread_rng()).unwrap(),
                                );
                            }

                            let tall_grass_value =
                                self.fast_uniform_2d(x, z, self.seed.wrapping_add(4));
                            let tall_grass_cutoff = self.flower_density_noise.get([
                                (x as f64) * TALL_GRASS_DENSITY_INPUT_SCALE,
                                (z as f64) * TALL_GRASS_DENSITY_INPUT_SCALE,
                            ]) * TALL_GRASS_DENSITY_OUTPUT_SCALE
                                + TALL_GRASS_DENSITY_OUTPUT_OFFSET;
                            if tall_grass_value < tall_grass_cutoff {
                                self.make_simple_foliage(
                                    chunk_coord,
                                    chunk,
                                    x,
                                    ground_y,
                                    z,
                                    self.tall_grass,
                                );
                            }
                        }
                        (Macrobiome::RollingHills, Biome::Desert) => {
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
                                self.make_cactus_like(
                                    chunk_coord,
                                    chunk,
                                    x,
                                    ground_y,
                                    z,
                                    cactus_height,
                                    self.cactus,
                                );
                            }
                        }
                        (Macrobiome::RollingHills, Biome::SandyBeach | Biome::Saltmarsh) => {
                            let marsh_grass_value =
                                self.fast_uniform_2d(x, z, self.seed.wrapping_add(3));
                            let marsh_grass_cutoff = self.flower_density_noise.get([
                                (x as f64) * FLOWER_DENSITY_INPUT_SCALE,
                                (z as f64) * FLOWER_DENSITY_INPUT_SCALE,
                            ]) * FLOWER_DENSITY_OUTPUT_SCALE
                                + (4.0 * FLOWER_DENSITY_OUTPUT_OFFSET);
                            if marsh_grass_value < marsh_grass_cutoff {
                                if biome == Biome::Saltmarsh
                                    && elevation < 0.5
                                    && marsh_grass_value < (0.25 * marsh_grass_cutoff)
                                {
                                    // tall reeds should only grow in marshes
                                    self.make_cactus_like(
                                        chunk_coord,
                                        chunk,
                                        x,
                                        ground_y,
                                        z,
                                        3,
                                        self.tall_reed,
                                    );
                                } else {
                                    self.make_simple_foliage(
                                        chunk_coord,
                                        chunk,
                                        x,
                                        ground_y,
                                        z,
                                        self.marsh_grass,
                                    );
                                }
                            }
                        }
                        (Macrobiome::Karst, _) => {
                            if ground_y <= 20 {
                                let tree_value = self.fast_uniform_2d(x, z, self.seed);
                                let tree_cutoff = self.tree_density_noise.get([
                                    (x as f64) * TREE_DENSITY_INPUT_SCALE,
                                    (z as f64) * TREE_DENSITY_INPUT_SCALE,
                                ]) * TREE_DENSITY_OUTPUT_SCALE
                                    + TREE_DENSITY_OUTPUT_OFFSET;
                                if tree_value < tree_cutoff {
                                    self.make_tree(chunk_coord, chunk, x, ground_y, z);
                                }
                            } else {
                                // vines
                                // TODO: Apply vines to the whole surface of a karst formation,
                                //       not just the top
                                let tall_grass_value =
                                    self.fast_uniform_2d(x, z, self.seed.wrapping_add(4)) - 0.2;
                                let tall_grass_cutoff = self.flower_density_noise.get([
                                    (x as f64) * VINES_DENSITY_INPUT_SCALE,
                                    (z as f64) * VINES_DENSITY_INPUT_SCALE,
                                ]) * VINES_DENSITY_OUTPUT_OFFSET
                                    + VINES_DENSITY_OUTPUT_SCALE;

                                if tall_grass_value < tall_grass_cutoff {
                                    self.make_simple_foliage(
                                        chunk_coord,
                                        chunk,
                                        x,
                                        ground_y,
                                        z,
                                        self.tall_grass,
                                    );
                                }
                            }
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
        // saturating_add used here since overflow is fine, the chunk checks will all fail from
        // wraparound
        for h in 1..=4 {
            let coord = BlockCoordinate::new(x, y.wrapping_add(h), z);
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
                    let coord = BlockCoordinate::new(
                        x.wrapping_add(i),
                        y.wrapping_add(h),
                        z.wrapping_add(j),
                    );
                    if coord.chunk() == chunk_coord {
                        chunk.set_block(coord.offset(), self.maple_leaves, None);
                    }
                }
            }
        }
    }

    // Generate a cactus at the given location. The y coordinate represents the dirt block just below the cactus.
    fn make_cactus_like(
        &self,
        chunk_coord: ChunkCoordinate,
        chunk: &mut MapChunk,
        x: i32,
        y: i32,
        z: i32,
        height: i32,
        block_id: BlockId,
    ) {
        for h in 1..=height {
            let coord = BlockCoordinate::new(x, y.wrapping_add(h), z);
            if coord.chunk() == chunk_coord {
                chunk.set_block(coord.offset(), block_id, None);
            }
        }
    }

    fn generate_default_biome<F>(
        &self,
        vert_offset: i32,
        block_coord: BlockCoordinate,
        gen_ore: F,
    ) -> BlockId
    where
        F: Fn() -> BlockId,
    {
        if vert_offset > 0 {
            if block_coord.y > 0 {
                self.air
            } else {
                self.water
            }
        } else if vert_offset == 0 && block_coord.y >= 0 {
            self.dirt_grass
            // TODO: at high elevations and wide spaces, we need more variety
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
    ) -> BlockId
    where
        F: Fn() -> BlockId,
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

    fn generate_saltmarsh<F>(
        &self,
        vert_offset: i32,
        block_coord: BlockCoordinate,
        gen_ore: F,
        true_elevation: f64,
    ) -> BlockId
    where
        F: Fn() -> BlockId,
    {
        if vert_offset > 0 {
            if block_coord.y > 0 {
                self.air
            } else {
                self.water
            }
        } else if vert_offset > -1 {
            // todo actual silt block, todo soft silt?
            let dither =
                self.fast_uniform_2d(block_coord.x, block_coord.z, self.seed.wrapping_add(5)) * 0.1;
            if true_elevation + dither > 1.3 {
                self.sand
            } else if true_elevation + dither > 0.3 {
                self.silt_dry
            } else if true_elevation + dither > -1.0 {
                self.silt_damp
            } else {
                self.clay
            }
        } else if vert_offset > -3 {
            self.silt_dry
        } else {
            gen_ore()
        }
    }

    fn generate_desert<F>(
        &self,
        vert_offset: i32,
        block_coord: BlockCoordinate,
        gen_ore: F,
    ) -> BlockId
    where
        F: Fn() -> BlockId,
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
            if chunk.get_block(coord.offset()) == self.air {
                chunk.set_block(coord.offset(), block, None);
            }
        }
    }

    #[inline(always)]
    fn generate_rolling_hills(
        &self,
        chunk_coord: ChunkCoordinate,
        x: u8,
        z: u8,
        biome_map: &[[Biome; 16]; 16],
        height_map: &[[f64; 16]; 16],
        chunk: &mut MapChunk,
    ) {
        let xg = 16 * chunk_coord.x + (x as i32);
        let zg = 16 * chunk_coord.z + (z as i32);

        let elevation = height_map[x as usize][z as usize];
        let biome = biome_map[x as usize][z as usize];

        for y in 0..16 {
            let offset = ChunkOffset { x, y, z };
            let block_coord = chunk_coord.with_offset(offset);

            let vert_offset = block_coord.y - (elevation.floor() as i32);

            let gen_ore = || self.generate_ore(block_coord);

            let block = match biome {
                Biome::DefaultGrassy => {
                    self.generate_default_biome(vert_offset, block_coord, gen_ore)
                }
                Biome::SandyBeach => self.generate_sandy_beach(vert_offset, block_coord, gen_ore),
                Biome::Saltmarsh => {
                    self.generate_saltmarsh(vert_offset, block_coord, gen_ore, elevation)
                }
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
        stone: blocks.get_by_name(STONE.0).expect("stone"),

        sand: blocks.get_by_name(SAND.0).expect("sand"),
        silt_dry: blocks.get_by_name(SILT_DRY.0).expect("silt_dry"),
        silt_damp: blocks.get_by_name(SILT_DAMP.0).expect("silt_dry"),
        clay: blocks.get_by_name(CLAY.0).expect("silt_dry"),

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
        marsh_grass: blocks.get_by_name(MARSH_GRASS.0).expect("marsh_grass"),
        tall_reed: blocks.get_by_name(TALL_REED.0).expect("tall_reed"),

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

        rail_testonly: blocks.get_by_name("carts:rail_tile").expect("rail"),
        signal_testonly: blocks.get_by_name("carts:signal").expect("signal"),
        glass_testonly: blocks.get_by_name("default:glass").expect("glass"),
    })
}
