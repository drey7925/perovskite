//! Station-manager test scenario and shared world-setup helpers.
//!
//! This module hosts the world layout used both by interactive `dev_server`
//! sessions (via `--scenario stations_fork`) and by unit tests. Keeping the
//! mapgen + helper in one place means that templates authored interactively
//! against the dev server can be replayed exactly inside the test fixture.
//!
//! # World layout (`stations_fork` scenario)
//!
//! All tracks sit at `y = 0`. Y is up. The mapgen places straight track tiles
//! (`carts:rail_tile`, variant 0) along ZPlus/ZMinus on the following columns:
//!
//! * **Four-track main line** for `z < -64`:
//!     - `x = -2, -1, 0, 1`
//! * **Two two-track branch lines** for `z >= 64`:
//!     - West pair: `x = -32, -31`
//!     - East pair: `x =  31,  32`
//!
//! The slab `-64 <= z < 64` is intentionally left blank by the mapgen; it is
//! the region where hand-crafted station templates are placed in later test
//! steps.
//!
//! All other `y < 0` blocks are filled with `default:dirt`; `y >= 0` is air
//! except for the rail tiles listed above.
#![cfg(any(test, feature = "test-support"))]

use std::sync::Arc;

use anyhow::Context;
use perovskite_core::{
    block_id::{special_block_defs::AIR_ID, BlockId},
    constants::CHUNK_SIZE_I32,
    coordinates::{BlockCoordinate, ChunkCoordinate, ChunkOffset},
};
use perovskite_server::{
    game_state::{
        game_map::MapChunk,
        mapgen::{FarMeshPoint, MapgenInterface},
    },
    server::Server,
};
use prost::Message;

use crate::{
    carts::game_state::game_map::templates::SerializedTemplate,
    carts::signals::DEFAULT_SIGNAL_VARIANT,
    default_game::basic_blocks::DIRT,
    game_builder::{GameBuilder, StaticBlockName},
};
use perovskite_server::game_state::event::EventInitiator;

/// X columns that carry tracks for the four-track main line (`z < -64`).
pub const FOUR_TRACK_COLUMNS: [i32; 4] = [-2, -1, 0, 1];
/// X columns that carry tracks for the two two-track branch lines (`z >= 64`).
pub const TWO_TRACK_COLUMNS: [i32; 4] = [-16, -15, 15, 16];
/// World Y of all tracks placed by this mapgen.
pub const TRACK_Y: i32 = 0;
/// Inclusive lower bound of the hand-crafted station region (template area).
pub const STATION_REGION_Z_MIN: i32 = -64;
/// Exclusive upper bound of the hand-crafted station region (template area).
pub const STATION_REGION_Z_MAX: i32 = 64;

/// Flatland-derived mapgen for the `stations_fork` scenario.
///
/// * `y <  0`              → `ground_block`
/// * `y == 0`, columns in [`FOUR_TRACK_COLUMNS`], `z <  STATION_REGION_Z_MIN`  → `rail_block`
/// * `y == 0`, columns in [`TWO_TRACK_COLUMNS`],  `z >= STATION_REGION_Z_MAX`  → `rail_block`
/// * everything else       → air
///
/// TODO: automatic signals on gantries
pub struct StationsForkMapgen {
    pub ground_block: BlockId,
    pub rail_block: BlockId,
    pub signal_block: BlockId,
}

impl MapgenInterface for StationsForkMapgen {
    fn fill_chunk(&self, coord: ChunkCoordinate, chunk: &mut MapChunk) {
        if coord.y < 0 {
            chunk.fill(self.ground_block);
        } else {
            chunk.fill(AIR_ID);
        }
        // Tracks live exclusively at world y = 0, which is offset_y = 0 in chunk_y = 0.
        // signals live at y = 2, which is also in chunk y = 0
        if coord.y != 0 {
            return;
        }
        for ox in 0u8..CHUNK_SIZE_I32 as u8 {
            let world_x = coord.x * CHUNK_SIZE_I32 + ox as i32;
            let is_four_track = FOUR_TRACK_COLUMNS.contains(&world_x);
            let is_two_track = TWO_TRACK_COLUMNS.contains(&world_x);
            if !is_four_track && !is_two_track {
                continue;
            }
            for oz in 0u8..CHUNK_SIZE_I32 as u8 {
                let world_z = coord.z * CHUNK_SIZE_I32 + oz as i32;
                let place = (is_four_track && world_z < STATION_REGION_Z_MIN)
                    || (is_two_track && world_z >= STATION_REGION_Z_MAX);
                if place {
                    chunk.set_block(ChunkOffset::new(ox, 0, oz), self.rail_block, None);
                    // simple signal placeement, no pretty ganties but functional
                    if world_z.rem_euclid(256) == 128 {
                        let signal_variant = match world_x {
                            -2 | -1 | -16 | 15 => DEFAULT_SIGNAL_VARIANT,
                            _ => DEFAULT_SIGNAL_VARIANT | 2,
                        };
                        chunk.set_block(
                            ChunkOffset::new(ox, 2, oz),
                            self.signal_block.with_variant_unchecked(signal_variant),
                            None,
                        );
                    }
                }
            }
        }
    }

    fn far_mesh_estimate(&self, _x: f64, _z: f64) -> FarMeshPoint {
        FarMeshPoint {
            height: 0.0,
            block_type: self.ground_block,
            water_height: 0.0,
        }
    }
}

const FORK_STATION_BYTES: &[u8] = include_bytes!("testdata/station_fork.test_geometry");
pub(crate) const FORK_STATION_ORIGIN: BlockCoordinate = BlockCoordinate::new(-16, -1, -64);

pub fn load_fork_station(server: &Server) -> anyhow::Result<()> {
    server.run_task_in_server(|server| {
        let serialized =
            SerializedTemplate::decode(FORK_STATION_BYTES).context("Failed to decode template")?;
        let in_mem = serialized
            .to_in_mem(server.block_types())
            .context("Failed to convert to in_mem")?;
        server
            .game_map()
            .apply_template(&in_mem, FORK_STATION_ORIGIN, 0, &EventInitiator::Engine)
            .context("Failed to apply template")?;
        Ok(())
    })
}

/// Sets up a `GameBuilder` for the `stations_fork` scenario:
///
/// 1. Calls [`crate::configure_default_game`] (so `carts:rail_tile` and the
///    default-game blocks are registered).
/// 2. Installs [`StationsForkMapgen`] as the world's map generator.
///
/// Equivalent helpers are wired into both the dev server (`--scenario
/// stations_fork`) and the station test fixtures so the two paths share an
/// identical world layout.
pub fn configure_stations_fork(game: &mut GameBuilder) -> anyhow::Result<()> {
    crate::configure_default_game(game)?;
    let rail_block = game
        .get_block(StaticBlockName("carts:rail_tile"))
        .context("carts:rail_tile not registered after configure_default_game")?;
    let signal_block = game
        .get_block(StaticBlockName("carts:signal"))
        .context("carts:signal not registered")?;
    let ground_block = game
        .get_block(DIRT)
        .context("default:dirt not registered after configure_default_game")?;
    game.unstable_server_builder_mut()
        .set_mapgen(move |_, _, _| {
            Ok(Arc::new(StationsForkMapgen {
                ground_block,
                rail_block,
                signal_block,
            }))
        });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{IsBlock, TestFixture};
    use googletest::prelude::*;
    use perovskite_core::coordinates::BlockCoordinate;

    /// Brings the stations_fork world up and verifies the mapgen placed a rail tile at
    /// (-2, 0, -65) — i.e. on the four-track main line, just past the station region.
    #[gtest]
    fn stations_fork_world_loads(fixture: &TestFixture) -> googletest::Result<()> {
        fixture.start_server(|builder| configure_stations_fork(builder))?;
        fixture.run_assertions_in_server(|gs| {
            let rail_block = gs
                .block_types()
                .get_by_name("carts:rail_tile")
                .expect("carts:rail_tile");
            expect_that!(
                gs.game_map()
                    .get_block(BlockCoordinate::new(-2, 0, -65))
                    .or_fail()?,
                IsBlock(rail_block)
            );
            Ok(())
        })?;
        Ok(())
    }
}
