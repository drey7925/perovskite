// For now, hide unused warnings since they're distracting
#![allow(dead_code)]

//! Higher-level routing network primitives.
//!
//! The track scanner (`tracks.rs`) handles individual tile traversal. This module
//! builds the graph layer on top of it: waypoints and interlocking signals are
//! graph nodes; the track segments between them are edges. `find_adjacency` is
//! the fundamental edge-finding primitive; routing and planning algorithms will
//! be layered over it.

use anyhow::{Context, Result};
use perovskite_core::{block_id::BlockId, coordinates::BlockCoordinate};
use perovskite_server::game_state::{
    blocks::{CompassDirection, ProtoCompassDirection},
    game_map::ServerGameMap,
};

use super::{
    signals::get_infra_block_name,
    tracks::{ScanOutcome, ScanState},
    CartsGameBuilderExtension,
};

/// The kind of endpoint found during an adjacency scan.
///
/// Field-less so that this type can be serialized as a protobuf enum in a later
/// step when we add adjacency caching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, prost::Enumeration, Hash)]
#[repr(i32)]
pub(crate) enum AdjacencyHitKind {
    /// The step limit was exhausted without finding any waypoint or terminator.
    StepLimitExhausted = 0,
    /// The track physically ended: disconnected track, out-of-bounds coordinate,
    /// or unavailable chunk.
    EndOfTrack = 1,
    /// A correctly-facing interlocking signal was found. This is the entrance to
    /// an interlocking and forms a graph node.
    InterlockingSignal = 2,
    /// A correctly-facing waypoint block (`carts:waypoint`) was found.
    WaypointBlock = 3,
    /// A signal (automatic or interlocking) was found facing against the scan
    /// direction. This is an error condition: either the track is one-way in the
    /// other direction, or signals are misconfigured.
    BackwardsSignal = 4,
    /// A correctly-facing automatic signal was found at the exit of an interlocking.
    /// Used as a terminal in `scan_interlocking_routes` results.
    EndOfInterlockingSignal = 5,
    /// A correctly-facing starting signal was found within an interlocking.
    /// Used as an intermediate waypoint in `scan_interlocking_routes` results.
    StartingSignal = 6,
}

/// The result of a single adjacency scan.
///
/// `track_coord` is always the rail-tile coordinate (Y position of the track
/// block itself, **not** Y+2 of any signal/waypoint above it).
///
/// `travel_direction` is the direction the scanner was traveling when it
/// encountered the endpoint. It is derived from the signal or waypoint variant
/// bits (easy) rather than from `ScanState` (hard for diagonal tiles).
/// `Some` for `InterlockingSignal`, `WaypointBlock`, and `BackwardsSignal`;
/// `None` for `StepLimitExhausted` and `EndOfTrack`.
///
/// The terminating `ScanState` is returned as the second element of the tuple
/// from `find_adjacency` and is kept separate to avoid coupling it to
/// serialization or caching of `AdjacencyHit`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct AdjacencyHit {
    pub(crate) kind: AdjacencyHitKind,
    /// Rail-tile coordinate under the hit signal/waypoint, or the last scanned
    /// tile for `StepLimitExhausted`/`EndOfTrack`.
    pub(crate) track_coord: BlockCoordinate,
    /// Direction the scanner was traveling when the endpoint was reached.
    /// `Some` for signal/waypoint hits; `None` for `StepLimitExhausted` and
    /// `EndOfTrack`.
    pub(crate) travel_direction: Option<CompassDirection>,
    /// Number of tile advances taken before the scan terminated.
    pub(crate) step_count: usize,
    /// If named, the name of the waypoint/signal/etc.
    pub(crate) name: Option<String>,
}

#[derive(Clone, prost::Message)]
pub(crate) struct CachedHit {
    #[prost(enumeration = "AdjacencyHitKind", tag = "1")]
    pub(crate) kind: i32,
    #[prost(message, tag = "2")]
    pub(crate) track_coord: Option<BlockCoordinate>,
    #[prost(enumeration = "ProtoCompassDirection", tag = "3")]
    pub(crate) travel_direction: i32,
    #[prost(uint64, tag = "4")]
    pub(crate) step_count: u64,
    #[prost(string, tag = "5")]
    pub(crate) name: String,
}

impl From<AdjacencyHit> for CachedHit {
    fn from(hit: AdjacencyHit) -> CachedHit {
        CachedHit {
            kind: hit.kind as i32,
            track_coord: Some(hit.track_coord),
            travel_direction: ProtoCompassDirection::from(hit.travel_direction) as i32,
            step_count: hit.step_count as u64,
            name: hit.name.unwrap_or_default(),
        }
    }
}

impl TryFrom<CachedHit> for AdjacencyHit {
    type Error = anyhow::Error;
    fn try_from(value: CachedHit) -> Result<Self, Self::Error> {
        Ok(AdjacencyHit {
            kind: value.kind(),
            track_coord: value.track_coord.context("Missing track_coord")?,
            travel_direction: value.travel_direction().into(),
            step_count: value.step_count as usize,
            name: if value.name.is_empty() {
                None
            } else {
                Some(value.name)
            },
        })
    }
}

/// Scan forward from `initial_state`, looking for the next routing graph node.
///
/// Advances one tile at a time, starting from the tile **after**
/// `initial_state.block_coord` (i.e., the starting tile is not itself checked
/// for signals/waypoints). At each new tile the block at Y+2 is inspected:
///
/// | Block type | Facing correctly? | Action |
/// |---|---|---|
/// | Waypoint | Yes | Stop — `WaypointBlock` (success) |
/// | Waypoint | No | Ignore, keep scanning |
/// | Automatic signal | Yes | Ignore, keep scanning |
/// | Automatic signal | No | Stop — `BackwardsSignal` (failure) |
/// | Interlocking signal | Yes | Stop — `InterlockingSignal` (success) |
/// | Interlocking signal | No | Stop — `BackwardsSignal` (failure) |
/// | Starting signal | Either | Ignore (only found inside interlockings) |
/// | Speedpost / other | — | Ignore |
///
/// After `step_limit` advances without hitting a terminator the scan returns
/// `StepLimitExhausted`.
///
/// Map read errors are propagated as `Err`; callers must not treat them as
/// end-of-track.
///
/// # Diverging routes
/// The scanner never sets `ScanState::is_diverging`. Switches outside
/// interlockings are decorative; only the interlocking pathfinder (`interlocking.rs`)
/// sets the diverging bit.
pub(crate) fn find_adjacency(
    initial_state: ScanState,
    step_limit: usize,
    game_map: &ServerGameMap,
    cart_config: &CartsGameBuilderExtension,
) -> Result<(AdjacencyHit, ScanState)> {
    let mut state = initial_state;

    let mut cursor = game_map.new_cursor();

    for step in 0..step_limit {
        let advance_result: Result<ScanOutcome> = state.advance::<false>(&mut cursor, cart_config);

        let next_state = match advance_result {
            Ok(ScanOutcome::Success(s)) => s,
            Ok(_) => {
                return Ok((
                    AdjacencyHit {
                        kind: AdjacencyHitKind::EndOfTrack,
                        track_coord: state.block_coord,
                        travel_direction: None,
                        step_count: step,
                        name: None,
                    },
                    state,
                ));
            }
            Err(e) => return Err(e),
        };
        state = next_state;

        // Check Y+2 for signals and waypoints.
        let Some(signal_coord) = state.block_coord.try_delta(0, 2, 0) else {
            continue;
        };
        let block = cursor.get_block(signal_coord)?;
        if block == BlockId::AIR {
            continue;
        }

        let variant = block.variant();
        // The signal/waypoint variant encodes the direction its front face points
        // (the direction a correctly-oriented cart is traveling when it passes it).
        let facing_dir = CompassDirection::from_rotation_variant(variant);

        if (block.equals_ignore_variant(cart_config.automatic_signal)
            || block.equals_ignore_variant(cart_config.interlocking_signal))
            && !state.signal_rotation_ok(variant)
        {
            // Backwards signal: track is one-way in the other direction.
            return Ok((
                AdjacencyHit {
                    kind: AdjacencyHitKind::BackwardsSignal,
                    track_coord: state.block_coord,
                    // We were traveling the opposite of the signal's facing.
                    travel_direction: Some(facing_dir.opposite()),
                    step_count: step + 1,
                    name: None,
                },
                state,
            ));
        }

        // We want a name if we have a right-side variant or any interlocking signal
        if (block.equals_ignore_variant(cart_config.waypoint) && state.signal_rotation_ok(variant))
            || block.equals_ignore_variant(cart_config.interlocking_signal)
        {
            let (block, name) = get_infra_block_name(signal_coord, game_map, cart_config)?;
            if block.equals_ignore_variant(cart_config.waypoint) {
                if state.signal_rotation_ok(variant) {
                    // Correctly-facing waypoint: stop successfully.
                    return Ok((
                        AdjacencyHit {
                            kind: AdjacencyHitKind::WaypointBlock,
                            track_coord: state.block_coord,
                            travel_direction: Some(facing_dir),
                            step_count: step + 1,
                            name,
                        },
                        state,
                    ));
                }
                // Wrong-way waypoint: ignore and keep scanning.
            } else if block.equals_ignore_variant(cart_config.interlocking_signal) {
                if state.signal_rotation_ok(variant) {
                    // Correctly-facing interlocking signal: interlocking entrance, graph node.
                    return Ok((
                        AdjacencyHit {
                            kind: AdjacencyHitKind::InterlockingSignal,
                            track_coord: state.block_coord,
                            travel_direction: Some(facing_dir),
                            step_count: step + 1,
                            name,
                        },
                        state,
                    ));
                } else {
                    // Backwards interlocking signal: stop, failure.
                    return Ok((
                        AdjacencyHit {
                            kind: AdjacencyHitKind::BackwardsSignal,
                            track_coord: state.block_coord,
                            travel_direction: Some(facing_dir.opposite()),
                            step_count: step + 1,
                            name: None,
                        },
                        state,
                    ));
                }
            }
        }
        // Starting signals, speedposts, and all other infrastructure: ignore.
    }

    Ok((
        AdjacencyHit {
            kind: AdjacencyHitKind::StepLimitExhausted,
            track_coord: state.block_coord,
            travel_direction: None,
            step_count: step_limit,
            name: None,
        },
        state,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::carts::tracks::TileId;
    use crate::test_support::TestFixture;
    use googletest::prelude::*;
    use perovskite_core::coordinates::BlockCoordinate;
    use perovskite_server::game_state::GameState;

    /// Initial scan state at track tile (0, 64, z), scanning ZPlus on a
    /// straight variant=0 rail.
    fn initial_state(z: i32) -> ScanState {
        ScanState {
            block_coord: BlockCoordinate::new(0, 64, z),
            is_reversed: false,
            is_diverging: false,
            allowable_speed: 90.0,
            odometer: 0.0,
            // variant=0 straight track: atlas (0,0), rotation=0, no flip
            current_tile_id: TileId::from_variant(0, false, false),
        }
    }

    /// Place a straight ZPlus track from z=`z_start` to z=`z_end` (inclusive) at Y=64.
    /// `infra` specifies (z, block) pairs placed at Y=66 (the signal/waypoint slot).
    fn setup_track(
        gs: &GameState,
        config: &CartsGameBuilderExtension,
        z_start: i32,
        z_end: i32,
        infra: &[(i32, BlockId)],
    ) -> googletest::Result<()> {
        // variant=0 straight track pointing ZPlus (atlas tile (0,0), rotation=0)
        let rail = config.rail_block.with_variant_unchecked(0);
        for z in z_start..=z_end {
            gs.game_map()
                .set_block(BlockCoordinate::new(0, 64, z), rail, None)
                .or_fail()?;
        }
        for &(z, block) in infra {
            gs.game_map()
                .set_block(BlockCoordinate::new(0, 66, z), block, None)
                .or_fail()?;
        }
        Ok(())
    }

    fn start(fixture: &TestFixture) -> googletest::Result<()> {
        fixture.start_server(|builder| crate::configure_default_game(builder))
    }

    #[gtest]
    fn step_limit_exhausted(fixture: &TestFixture) -> googletest::Result<()> {
        start(fixture)?;
        fixture.run_assertions_in_server(|gs: &GameState| {
            let config = gs.extension::<CartsGameBuilderExtension>().unwrap();
            setup_track(gs, config, 0, 10, &[])?;
            let (hit, final_state) =
                find_adjacency(initial_state(0), 5, gs.game_map(), config).or_fail()?;
            expect_that!(hit.kind, eq(AdjacencyHitKind::StepLimitExhausted));
            // 5 advances from z=0 land on z=5.
            expect_that!(hit.track_coord, eq(BlockCoordinate::new(0, 64, 5)));
            expect_that!(hit.step_count, eq(5));
            expect_that!(final_state.block_coord, eq(BlockCoordinate::new(0, 64, 5)));
            Ok(())
        })
    }

    #[gtest]
    fn end_of_track(fixture: &TestFixture) -> googletest::Result<()> {
        start(fixture)?;
        fixture.run_assertions_in_server(|gs: &GameState| {
            let config = gs.extension::<CartsGameBuilderExtension>().unwrap();
            // Track only from z=0 to z=3; z=4 is absent.
            setup_track(gs, config, 0, 3, &[])?;
            let (hit, final_state) =
                find_adjacency(initial_state(0), 20, gs.game_map(), config).or_fail()?;
            expect_that!(hit.kind, eq(AdjacencyHitKind::EndOfTrack));
            // Last valid tile is z=3; we fail trying to advance to z=4.
            expect_that!(hit.track_coord, eq(BlockCoordinate::new(0, 64, 3)));
            // 3 advances: z=0→1, 1→2, 2→3; fail on the 4th attempt (step index 3).
            expect_that!(hit.step_count, eq(3));
            expect_that!(final_state.block_coord, eq(BlockCoordinate::new(0, 64, 3)));
            Ok(())
        })
    }

    #[gtest]
    fn right_way_waypoint(fixture: &TestFixture) -> googletest::Result<()> {
        start(fixture)?;
        fixture.run_assertions_in_server(|gs: &GameState| {
            let config = gs.extension::<CartsGameBuilderExtension>().unwrap();
            // variant=0 waypoint faces ZPlus — correct for a ZPlus scan.
            let waypoint = config.waypoint.with_variant_unchecked(0);
            setup_track(gs, config, 0, 10, &[(3, waypoint)])?;
            let (hit, final_state) =
                find_adjacency(initial_state(0), 20, gs.game_map(), config).or_fail()?;
            expect_that!(hit.kind, eq(AdjacencyHitKind::WaypointBlock));
            expect_that!(hit.track_coord, eq(BlockCoordinate::new(0, 64, 3)));
            expect_that!(hit.travel_direction, eq(Some(CompassDirection::ZPlus)));
            expect_that!(hit.step_count, eq(3));
            expect_that!(final_state.block_coord, eq(BlockCoordinate::new(0, 64, 3)));
            Ok(())
        })
    }

    #[gtest]
    fn wrong_way_waypoint_is_skipped(fixture: &TestFixture) -> googletest::Result<()> {
        start(fixture)?;
        fixture.run_assertions_in_server(|gs: &GameState| {
            let config = gs.extension::<CartsGameBuilderExtension>().unwrap();
            // variant=2 waypoint faces ZMinus — backwards for a ZPlus scan, so ignored.
            let waypoint_backwards = config.waypoint.with_variant_unchecked(2);
            // Place an interlocking signal further along so the scan has somewhere to stop.
            let interlocking = config.interlocking_signal.with_variant_unchecked(0);
            setup_track(
                gs,
                config,
                0,
                10,
                &[(3, waypoint_backwards), (7, interlocking)],
            )?;
            let (hit, final_state) =
                find_adjacency(initial_state(0), 20, gs.game_map(), config).or_fail()?;
            // The backwards waypoint at z=3 must be silently skipped.
            expect_that!(hit.kind, eq(AdjacencyHitKind::InterlockingSignal));
            expect_that!(hit.track_coord, eq(BlockCoordinate::new(0, 64, 7)));
            expect_that!(hit.travel_direction, eq(Some(CompassDirection::ZPlus)));
            expect_that!(hit.step_count, eq(7));
            expect_that!(final_state.block_coord, eq(BlockCoordinate::new(0, 64, 7)));
            Ok(())
        })
    }

    #[gtest]
    fn right_way_auto_signal_is_skipped(fixture: &TestFixture) -> googletest::Result<()> {
        start(fixture)?;
        fixture.run_assertions_in_server(|gs: &GameState| {
            let config = gs.extension::<CartsGameBuilderExtension>().unwrap();
            // variant=0 automatic signal faces ZPlus — correct, so skipped (not a graph node).
            let auto_signal = config.automatic_signal.with_variant_unchecked(0);
            let interlocking = config.interlocking_signal.with_variant_unchecked(0);
            setup_track(gs, config, 0, 10, &[(3, auto_signal), (7, interlocking)])?;
            let (hit, final_state) =
                find_adjacency(initial_state(0), 20, gs.game_map(), config).or_fail()?;
            expect_that!(hit.kind, eq(AdjacencyHitKind::InterlockingSignal));
            expect_that!(hit.track_coord, eq(BlockCoordinate::new(0, 64, 7)));
            expect_that!(hit.step_count, eq(7));
            expect_that!(final_state.block_coord, eq(BlockCoordinate::new(0, 64, 7)));
            Ok(())
        })
    }

    #[gtest]
    fn backwards_auto_signal(fixture: &TestFixture) -> googletest::Result<()> {
        start(fixture)?;
        fixture.run_assertions_in_server(|gs: &GameState| {
            let config = gs.extension::<CartsGameBuilderExtension>().unwrap();
            // variant=2 automatic signal faces ZMinus — backwards for a ZPlus scan.
            let auto_backwards = config.automatic_signal.with_variant_unchecked(2);
            setup_track(gs, config, 0, 10, &[(4, auto_backwards)])?;
            let (hit, final_state) =
                find_adjacency(initial_state(0), 20, gs.game_map(), config).or_fail()?;
            expect_that!(hit.kind, eq(AdjacencyHitKind::BackwardsSignal));
            expect_that!(hit.track_coord, eq(BlockCoordinate::new(0, 64, 4)));
            // We were traveling ZPlus; signal faces ZMinus.
            expect_that!(hit.travel_direction, eq(Some(CompassDirection::ZPlus)));
            expect_that!(hit.step_count, eq(4));
            expect_that!(final_state.block_coord, eq(BlockCoordinate::new(0, 64, 4)));
            Ok(())
        })
    }

    #[gtest]
    fn right_way_interlocking_signal(fixture: &TestFixture) -> googletest::Result<()> {
        start(fixture)?;
        fixture.run_assertions_in_server(|gs: &GameState| {
            let config = gs.extension::<CartsGameBuilderExtension>().unwrap();
            let interlocking = config.interlocking_signal.with_variant_unchecked(0);
            setup_track(gs, config, 0, 10, &[(5, interlocking)])?;
            let (hit, final_state) =
                find_adjacency(initial_state(0), 20, gs.game_map(), config).or_fail()?;
            expect_that!(hit.kind, eq(AdjacencyHitKind::InterlockingSignal));
            expect_that!(hit.track_coord, eq(BlockCoordinate::new(0, 64, 5)));
            expect_that!(hit.travel_direction, eq(Some(CompassDirection::ZPlus)));
            expect_that!(hit.step_count, eq(5));
            expect_that!(final_state.block_coord, eq(BlockCoordinate::new(0, 64, 5)));
            Ok(())
        })
    }

    #[gtest]
    fn backwards_interlocking_signal(fixture: &TestFixture) -> googletest::Result<()> {
        start(fixture)?;
        fixture.run_assertions_in_server(|gs: &GameState| {
            let config = gs.extension::<CartsGameBuilderExtension>().unwrap();
            // variant=2 interlocking signal faces ZMinus — backwards.
            let interlocking_backwards = config.interlocking_signal.with_variant_unchecked(2);
            setup_track(gs, config, 0, 10, &[(4, interlocking_backwards)])?;
            let (hit, final_state) =
                find_adjacency(initial_state(0), 20, gs.game_map(), config).or_fail()?;
            expect_that!(hit.kind, eq(AdjacencyHitKind::BackwardsSignal));
            expect_that!(hit.track_coord, eq(BlockCoordinate::new(0, 64, 4)));
            expect_that!(hit.travel_direction, eq(Some(CompassDirection::ZPlus)));
            expect_that!(hit.step_count, eq(4));
            expect_that!(final_state.block_coord, eq(BlockCoordinate::new(0, 64, 4)));
            Ok(())
        })
    }
}
