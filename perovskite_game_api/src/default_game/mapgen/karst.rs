//! Generates karst-like terrain, similar to places like Ha Long Bay and Ninh Binh

use noise::NoiseFn;
use perovskite_core::{
    block_id::BlockId,
    constants::blocks::AIR,
    coordinates::{BlockCoordinate, ChunkCoordinate, ChunkOffset},
};
use perovskite_server::game_state::{blocks::BlockTypeManager, game_map::MapChunk};

use crate::default_game::basic_blocks::{DIRT_WITH_GRASS, LIMESTONE_DARK, LIMESTONE_LIGHT, WATER};

const PROFILE_INPUT_SCALE: f64 = 1.0 / 160.0;
const VALLEY_INPUT_SCALE: f64 = 1.0 / 800.0;
const MODULATING_INPUT_SCALE: f64 = 1.0 / 80.0;

pub(crate) struct KarstGenerator {
    // The profile that the tops of mountains follow
    // Offset 10
    karst_profile_noise: noise::SuperSimplex,
    // The profile that the valleys follow
    // Offset 11
    karst_valley_noise: noise::SuperSimplex,
    // The noise that switches between the mountain and valley
    // Offset 12
    modulating_noise: noise::SuperSimplex,
    // Limestone type noise
    // Offset 13
    limestone_noise: noise::SuperSimplex,

    light_limestone: BlockId,
    dark_limestone: BlockId,
    dirt_grass: BlockId,
    air: BlockId,
    water: BlockId,
}
impl KarstGenerator {
    pub(crate) fn new(seed: u32, blocks: &BlockTypeManager) -> Self {
        Self {
            karst_profile_noise: noise::SuperSimplex::new(seed.wrapping_add(10)),
            karst_valley_noise: noise::SuperSimplex::new(seed.wrapping_add(11)),
            modulating_noise: noise::SuperSimplex::new(seed.wrapping_add(12)),
            limestone_noise: noise::SuperSimplex::new(seed.wrapping_add(13)),
            light_limestone: blocks
                .get_by_name(LIMESTONE_LIGHT.0)
                .expect("limestone_light"),
            dark_limestone: blocks
                .get_by_name(LIMESTONE_DARK.0)
                .expect("limestone_dark"),
            dirt_grass: blocks.get_by_name(DIRT_WITH_GRASS.0).expect("dirt_grass"),
            air: blocks.get_by_name(AIR).expect("air"),
            water: blocks
                .get_by_name(WATER.0)
                .expect("water")
                .with_variant(0xfff)
                .unwrap(),
        }
    }
    #[inline(always)]
    pub(crate) fn generate(
        &self,
        chunk_coord: ChunkCoordinate,
        x: u8,
        z: u8,
        height_map: &mut [[f64; 16]; 16],
        chunk: &mut MapChunk,
        blend_factor: f64,
        blend_elevation: f64,
        gen_ore: impl Fn(BlockCoordinate) -> BlockId,
    ) {
        let xg = 16 * chunk_coord.x + (x as i32);
        let zg = 16 * chunk_coord.z + (z as i32);

        let profile = self.karst_profile_noise.get([
            xg as f64 * PROFILE_INPUT_SCALE,
            zg as f64 * PROFILE_INPUT_SCALE,
        ]) * 30.0
            + 60.0;
        let valley = self.karst_valley_noise.get([
            xg as f64 * VALLEY_INPUT_SCALE,
            zg as f64 * VALLEY_INPUT_SCALE,
        ]) * 20.0;
        let raw_modulation = self.modulating_noise.get([
            xg as f64 * MODULATING_INPUT_SCALE,
            zg as f64 * MODULATING_INPUT_SCALE,
        ]);

        let modulation = (raw_modulation * 12.0 - 4.0).tanh() * 0.5 + 0.5;
        let elevation = (profile * modulation) + (valley * (1.0 - modulation));
        let blended_elevation =
            (blend_elevation * blend_factor) + (elevation * (1.0 - blend_factor));
        height_map[x as usize][z as usize] = blended_elevation;
        for y in 0..16 {
            let offset = ChunkOffset { x, y, z };
            let block_coord = chunk_coord.with_offset(offset);

            let vert_offset = block_coord.y - (blended_elevation as i32);

            let block = if vert_offset > 0 {
                if block_coord.y > 0 {
                    self.air
                } else {
                    self.water
                }
            } else if vert_offset == 0 && block_coord.y >= 0 {
                self.dirt_grass
            } else if block_coord.y > -32 {
                self.light_limestone
            } else {
                gen_ore(block_coord)
            };

            chunk.set_block(offset, block, None);
        }
    }
}
