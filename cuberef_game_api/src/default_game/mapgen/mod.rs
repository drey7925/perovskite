use std::{sync::Arc};

use cuberef_core::{
    constants::blocks::AIR,
    coordinates::{ChunkCoordinate, ChunkOffset},
};
use cuberef_server::game_state::{
    blocks::{BlockTypeHandle, BlockTypeManager},
    game_map::MapChunk,
    mapgen::MapgenInterface,
};
use noise::{NoiseFn};

use super::basic_blocks::{DIRT, DIRT_WITH_GRASS, STONE, WATER};

const ELEVATION_FINE_INPUT_SCALE: f64 = 1.0 / 100.0;
const ELEVATION_FINE_OUTPUT_SCALE: f64 = 20.0;
const ELEVATION_COARSE_INPUT_SCALE: f64 = 1.0 / 800.0;
const ELEVATION_COARSE_OUTPUT_SCALE: f64 = 20.0;
struct ElevationNoise {
    coarse: noise::HybridMulti<noise::SuperSimplex>,
    fine: noise::SuperSimplex,
}
impl ElevationNoise {
    fn new(seed: u32) -> Box<ElevationNoise> {
        Box::new(ElevationNoise {
            coarse: noise::HybridMulti::new(seed),
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
        (coarse_height + fine_height) as i32
    }
}

struct DefaultMapgen {
    air: BlockTypeHandle,
    dirt: BlockTypeHandle,
    dirt_grass: BlockTypeHandle,
    stone: BlockTypeHandle,
    water: BlockTypeHandle,

    elevation_noise: Box<ElevationNoise>,
}
impl MapgenInterface for DefaultMapgen {
    fn fill_chunk(&self, coord: ChunkCoordinate, chunk: &mut MapChunk) {
        // todo subdivide by surface vs underground, etc. This is a very minimal MVP
        for x in 0..16 {
            for z in 0..16 {
                let xg = 16 * coord.x + (x as i32);
                let zg = 16 * coord.z + (z as i32);

                let elevation = self.elevation_noise.get(xg, zg);
                for y in 0..16 {
                    let offset = ChunkOffset { x, y, z };
                    let coord2 = coord.with_offset(offset);
                    debug_assert!(coord2.chunk() == coord);

                    let vert_offset = coord2.y - elevation;
                    let block = if vert_offset > 0 {
                        if coord2.y > 0 {
                            self.air
                        } else {
                            self.water
                        }
                    } else if vert_offset == 0 {
                        self.dirt_grass
                    } else if vert_offset > -3 {
                        // todo variable depth of dirt
                        self.dirt
                    } else {
                        self.stone
                    };
                    chunk.set_block(offset, block, None);
                }
            }
        }
    }
}

pub(crate) fn build_mapgen(blocks: Arc<BlockTypeManager>, seed: u32) -> Arc<dyn MapgenInterface> {
    Arc::new(DefaultMapgen {
        air: blocks.get_by_name(AIR).expect("air"),
        dirt: blocks.get_by_name(DIRT.0).expect("dirt"),
        dirt_grass: blocks.get_by_name(DIRT_WITH_GRASS.0).expect("dirt_grass"),
        stone: blocks.get_by_name(STONE.0).expect("stone"),
        water: blocks.get_by_name(WATER.0).expect("water"),
        elevation_noise: ElevationNoise::new(seed),
    })
}
