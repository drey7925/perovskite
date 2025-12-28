//! Generates karst-like terrain, similar to places like Ha Long Bay and Ninh Binh

use cgmath::Vector3;
use noise::NoiseFn;
use perovskite_core::{
    block_id::BlockId,
    constants::blocks::AIR,
    coordinates::{BlockCoordinate, ChunkCoordinate, ChunkOffset},
};
use perovskite_server::game_state::{blocks::BlockTypeManager, game_map::MapChunk};

use crate::default_game::basic_blocks::{
    DIRT, DIRT_WITH_GRASS, LIMESTONE, LIMESTONE_DARK, LIMESTONE_LIGHT, WATER,
};

const PROFILE_INPUT_SCALE: f64 = 1.0 / 160.0;
const VALLEY_INPUT_SCALE: f64 = 1.0 / 800.0;
const SLOW_MODULATOR_INPUT_SCALE: f64 = 1.0 / 640.0;
const MODULATING_INPUT_SCALE: f64 = 1.0 / 160.0;
const MODULATION_SLOPE_INPUT_SCALE: f64 = 1.0 / 80.0;
const LIMESTONE_NOISE_SCALE: Vector3<f64> = Vector3::new(0.5, 0.03, 0.5);
const LIMESTONE_SLOW_NOISE_SCALE: f64 = 1.0 / 40.0;
const CAVES_INPUT_SCALE: f64 = 1.0 / 16.0;

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
    // Contributes to limestone noise
    // Offset 14
    limestone_slow_noise: noise::SuperSimplex,
    // Biases the modulator, changing slowly
    // Offset 15
    slow_modulator: noise::SuperSimplex,
    // These two contribute to the direction bias for caves
    // Offset 16
    cave_control: noise::SuperSimplex,
    // Adjusts the modulation slope feeding into the tanh
    // Offset 17
    modulation_slope: noise::SuperSimplex,

    limestone: BlockId,
    light_limestone: BlockId,
    dark_limestone: BlockId,
    dirt_grass: BlockId,
    dirt: BlockId,
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
            limestone_slow_noise: noise::SuperSimplex::new(seed.wrapping_add(14)),
            slow_modulator: noise::SuperSimplex::new(seed.wrapping_add(15)),
            cave_control: noise::SuperSimplex::new(seed.wrapping_add(16)),
            modulation_slope: noise::SuperSimplex::new(seed.wrapping_add(17)),
            limestone: blocks.get_by_name(LIMESTONE.0).expect("limestone"),
            light_limestone: blocks
                .get_by_name(LIMESTONE_LIGHT.0)
                .expect("limestone_light"),
            dark_limestone: blocks
                .get_by_name(LIMESTONE_DARK.0)
                .expect("limestone_dark"),

            dirt_grass: blocks.get_by_name(DIRT_WITH_GRASS.0).expect("dirt_grass"),
            dirt: blocks.get_by_name(DIRT.0).expect("dirt"),
            air: blocks.get_by_name(AIR).expect("air"),
            water: blocks
                .get_by_name(WATER.0)
                .expect("water")
                .with_variant(0xfff)
                .unwrap(),
        }
    }

    #[inline]
    pub(crate) fn height(&self, xg: i32, zg: i32) -> (f32, f32, f32) {
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
        ]) + self.slow_modulator.get([
            xg as f64 * SLOW_MODULATOR_INPUT_SCALE,
            zg as f64 * SLOW_MODULATOR_INPUT_SCALE,
        ]) * 0.25;

        let cave_control = self
            .cave_control
            .get([xg as f64 * CAVES_INPUT_SCALE, zg as f64 * CAVES_INPUT_SCALE]);

        // The tanh lands between -1 and 1
        // Multiplying by 0.6 gives us -0.6 to 0.6
        // We subsequently add 0.4 and clamp to [0, 1] to get a sharp cutoff on the
        // bottom
        let recentered = raw_modulation * 12.0 - 4.0;
        let scaled = recentered
            * (0.7
                + 0.4
                    * self.modulation_slope.get([
                        xg as f64 * MODULATION_SLOPE_INPUT_SCALE,
                        zg as f64 * MODULATION_SLOPE_INPUT_SCALE,
                    ]));
        let modulation = (scaled.tanh() * 0.6 + 0.4).clamp(0.0, 1.0);
        // The cave excess is higher when modulation is higher, and is clamped to 0.
        let cave_excess = (modulation * 0.8 - cave_control - 0.3).clamp(0.0, 0.5);

        let height = (profile * modulation) + (valley * (1.0 - modulation));
        let valley_factor = if height < 0.0 { 1.25 } else { 0.5 };
        let mut cave_ceiling = (valley * valley_factor) + (16.0 * cave_excess);
        if cave_ceiling < -0.0 || raw_modulation < 0.3 {
            // avoid weird floating mountains that sit upon a cushion of water
            // also avoid weird patches of bare limestone whenever we spawn a cave not in a valley.
            cave_ceiling = valley - 100.0;
        }

        (height as f32, valley as f32, cave_ceiling as f32)
    }

    pub(crate) fn profile(&self, xg: i32, zg: i32) -> f32 {
        self.karst_profile_noise.get([
            xg as f64 * PROFILE_INPUT_SCALE,
            zg as f64 * PROFILE_INPUT_SCALE,
        ]) as f32
            * 30.0
            + 60.0
    }

    pub(crate) fn floor(&self, xg: i32, zg: i32) -> f32 {
        self.karst_valley_noise.get([
            xg as f64 * VALLEY_INPUT_SCALE,
            zg as f64 * VALLEY_INPUT_SCALE,
        ]) as f32
            * 20.0
    }

    pub(crate) fn generate(
        &self,
        chunk_coord: ChunkCoordinate,
        x: u8,
        z: u8,
        height_map: &[[f32; 16]; 16],
        cave_floors: &[[f32; 16]; 16],
        cave_ceiling: &[[f32; 16]; 16],
        water_heights: &[[f32; 16]; 16],
        chunk: &mut MapChunk,
        gen_ore: impl Fn(BlockCoordinate) -> BlockId,
    ) {
        let xg = 16 * chunk_coord.x + (x as i32);
        let zg = 16 * chunk_coord.z + (z as i32);

        let blended_elevation = height_map[x as usize][z as usize];
        let cave_floor = cave_floors[x as usize][z as usize];
        let cave_ceiling = cave_ceiling[x as usize][z as usize];
        let water_height = water_heights[x as usize][z as usize];

        for y in 0..16 {
            let offset = ChunkOffset { x, y, z };
            let block_coord = chunk_coord.with_offset(offset);

            let vert_offset = block_coord.y - (blended_elevation as i32);

            let block = if vert_offset > 0
                || ((block_coord.y as f32) > cave_floor && (block_coord.y as f32) < cave_ceiling)
            {
                if block_coord.y as f32 > water_height {
                    self.air
                } else {
                    self.water
                }
            } else if vert_offset == 0 {
                if block_coord.y >= 0 {
                    self.dirt_grass
                } else {
                    self.dirt
                }
            } else if block_coord.y > -32 {
                self.get_limestone(xg, block_coord.y, zg)
            } else {
                gen_ore(block_coord)
            };

            chunk.set_block(offset, block, None);
        }
    }

    pub(super) fn get_limestone(&self, xg: i32, y: i32, zg: i32) -> BlockId {
        let limestone_noise = self.limestone_noise.get([
            xg as f64 * LIMESTONE_NOISE_SCALE.x,
            y as f64 * LIMESTONE_NOISE_SCALE.y,
            zg as f64 * LIMESTONE_NOISE_SCALE.z,
        ]);
        let slow_limestone_noise = 0.25
            * self.limestone_slow_noise.get([
                xg as f64 * LIMESTONE_SLOW_NOISE_SCALE,
                zg as f64 * LIMESTONE_SLOW_NOISE_SCALE,
            ]);
        if limestone_noise + slow_limestone_noise > 0.25 {
            self.dark_limestone
        } else if limestone_noise + slow_limestone_noise < -0.25 {
            self.light_limestone
        } else {
            self.limestone
        }
    }
}
