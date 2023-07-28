use std::sync::Arc;

use cuberef_core::{
    constants::blocks::AIR,
    coordinates::{BlockCoordinate, ChunkCoordinate, ChunkOffset},
};
use cuberef_server::game_state::{
    blocks::{BlockTypeHandle, BlockTypeManager, BlockTypeName},
    game_map::MapChunk,
    mapgen::MapgenInterface,
};
use noise::{MultiFractal, NoiseFn};

use crate::{NonExhaustive, blocks::BlockTypeHandleWrapper};

use super::basic_blocks::{DIRT, DIRT_WITH_GRASS, STONE, WATER};

const ELEVATION_FINE_INPUT_SCALE: f64 = 1.0 / 60.0;
const ELEVATION_FINE_OUTPUT_SCALE: f64 = 10.0;
const ELEVATION_COARSE_INPUT_SCALE: f64 = 1.0 / 800.0;
const ELEVATION_COARSE_OUTPUT_SCALE: f64 = 60.0;
const ELEVATION_OFFSET: f64 = 20.0;
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
    fn get(&self, x: i32, y: i32) -> i32 {
        let coarse_pos = [
            x as f64 * ELEVATION_COARSE_INPUT_SCALE,
            y as f64 * ELEVATION_COARSE_INPUT_SCALE,
        ];
        let coarse_height = self.coarse.get(coarse_pos) * ELEVATION_COARSE_OUTPUT_SCALE;

        let fine_pos = [
            x as f64 * ELEVATION_FINE_INPUT_SCALE,
            y as f64 * ELEVATION_FINE_INPUT_SCALE,
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

    elevation_noise: Box<ElevationNoise>,
    ores: Vec<(OreDefinition, noise::SuperSimplex)>,
}
impl MapgenInterface for DefaultMapgen {
    fn fill_chunk(&self, chunk_coord: ChunkCoordinate, chunk: &mut MapChunk) {
        // todo subdivide by surface vs underground, etc. This is a very minimal MVP
        for x in 0..16 {
            for z in 0..16 {
                let xg = 16 * chunk_coord.x + (x as i32);
                let zg = 16 * chunk_coord.z + (z as i32);

                let elevation = self.elevation_noise.get(xg, zg);
                for y in 0..16 {
                    let offset = ChunkOffset { x, y, z };
                    let block_coord = chunk_coord.with_offset(offset);

                    let vert_offset = block_coord.y - elevation;
                    let block = if vert_offset > 0 {
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
                    };
                    chunk.set_block(offset, block, None);
                }
            }
        }
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
        water: blocks.get_by_name(WATER.0).expect("water"),
        elevation_noise: ElevationNoise::new(seed),
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
    })
}
