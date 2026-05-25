use googletest::prelude::*;
use perovskite_core::coordinates::BlockCoordinate;
use perovskite_server::game_state::{
    blocks::CompassDirection, event::EventInitiator, game_map::templates::SerializedTemplate,
};
use prost::Message;

use super::interlocking::{
    apply_interlocking_routes_to_signals, scan_interlocking_routes, RoutingPath,
};
use super::network::AdjacencyHitKind;
use crate::test_support::TestFixture;
use rustc_hash::FxHashMap;

// on dev_server:
//  /place_template perovskite_game_api/src/carts/testdata/simple_interlocking.test_geometry 0 -1 0
//  /save_template perovskite_game_api/src/carts/testdata/simple_interlocking.test_geometry 0 -1 0 [fill in size]
const SIMPLE_INTERLOCKING_BYTES: &[u8] =
    include_bytes!("testdata/simple_interlocking.test_geometry");

/// World-space origin where the simple_interlocking template is placed.
/// Template local (0,0,0) maps to this coordinate; tracks sit at world y=0 (template y=1).
pub(crate) const SIMPLE_INTERLOCKING_ORIGIN: BlockCoordinate = BlockCoordinate::new(0, -1, 0);

/// Loads the simple_interlocking template into the world at [`SIMPLE_INTERLOCKING_ORIGIN`].
/// Call this after `fixture.start_server(...)`.
pub(crate) fn load_simple_interlocking(fixture: &TestFixture) -> googletest::Result<()> {
    fixture.run_with_context(|ctx| {
        let serialized = SerializedTemplate::decode(SIMPLE_INTERLOCKING_BYTES)
            .map_err(|e| anyhow::anyhow!("Failed to decode template: {}", e))
            .or_fail()?;
        let in_mem = serialized.to_in_mem(ctx.block_types()).or_fail()?;
        ctx.game_map()
            .apply_template(
                &in_mem,
                SIMPLE_INTERLOCKING_ORIGIN,
                0,
                &EventInitiator::Engine,
            )
            .or_fail()?;
        Ok(())
    })
}

/// Verifies that single_pathfind_attempt can acquire each of the four approach tracks
/// through the simple interlocking concurrently, each returning a non-empty route with
/// pending signal changes, and that dropping all routes rolls the transactions back.
#[gtest]
fn test_simple_interlocking_acquires(fixture: &TestFixture) -> googletest::Result<()> {
    fixture.start_server(|builder| crate::configure_default_game(builder))?;
    load_simple_interlocking(fixture)?;

    fixture.run_with_context(|ctx| {
        let config = ctx
            .extension::<super::CartsGameBuilderExtension>()
            .expect("CartsGameBuilderExtension")
            .clone();

        let starts = [
            (BlockCoordinate::new(2, 0, 4), CompassDirection::ZPlus),
            (BlockCoordinate::new(3, 0, 4), CompassDirection::ZPlus),
            (BlockCoordinate::new(4, 0, 60), CompassDirection::ZMinus),
            (BlockCoordinate::new(5, 0, 60), CompassDirection::ZMinus),
        ];

        let mut routes = vec![];
        for (coord, dir) in starts {
            let scan_state =
                super::tracks::ScanState::create_at(coord, dir, ctx.game_map(), &config)
                    .or_fail()?
                    .expect("ScanState::create_at should succeed at a rail tile");

            let route = super::interlocking::single_pathfind_attempt(
                &ctx,
                "",
                (ctx.startup_counter(), 0),
                scan_state,
                1024,
                &config,
                None,
                0.0,
            )
            .or_fail()?
            .expect("interlocking should be acquirable on each approach track");

            expect_that!(route.inner().steps.len(), gt(0));
            expect_that!(route.pending_change_count(), gt(0));
            routes.push(route);
        }

        // Explicitly dropping all routes rolls back all signal transactions.
        drop(routes);

        Ok(())
    })
}

/// Verifies that the template loads and contains all expected marker blocks within
/// the scan range x=[0,9], y=0, z=[0,66] in world coordinates.
#[gtest]
fn test_simple_interlocking_loads(fixture: &TestFixture) -> googletest::Result<()> {
    fixture.start_server(|builder| crate::configure_default_game(builder))?;
    load_simple_interlocking(fixture)?;

    fixture.run_with_context(|ctx| {
        let rail = ctx
            .block_types()
            .get_by_name("carts:rail_tile")
            .expect("carts:rail_tile");
        let obsidian = ctx
            .block_types()
            .get_by_name("default:obsidian")
            .expect("default:obsidian");
        let stone = ctx
            .block_types()
            .get_by_name("default:stone")
            .expect("default:stone");
        let cactus = ctx
            .block_types()
            .get_by_name("default:cactus")
            .expect("default:cactus");
        let maple = ctx
            .block_types()
            .get_by_name("default:maple_tree")
            .expect("default:maple_tree");

        let mut found_rail = false;
        let mut found_obsidian = false;
        let mut found_stone = false;
        let mut found_cactus = false;
        let mut found_maple = false;

        for x in 0i32..=9 {
            for z in 0i32..=66 {
                let coord = BlockCoordinate::new(x, 0, z);
                let block = ctx.game_map().get_block(coord).or_fail()?;
                if block.equals_ignore_variant(rail) {
                    found_rail = true;
                }
                if block.equals_ignore_variant(obsidian) {
                    found_obsidian = true;
                }
                if block.equals_ignore_variant(stone) {
                    found_stone = true;
                }
                if block.equals_ignore_variant(cactus) {
                    found_cactus = true;
                }
                if block.equals_ignore_variant(maple) {
                    found_maple = true;
                }
            }
        }

        expect_that!(found_rail, eq(true));
        expect_that!(found_obsidian, eq(true));
        expect_that!(found_stone, eq(true));
        expect_that!(found_cactus, eq(true));
        expect_that!(found_maple, eq(true));

        Ok(())
    })
}

/// Expected path descriptor used to find and assert on an `InterlockingPathResult`.
struct ExpectedPath {
    kind: AdjacencyHitKind,
    track_coord: BlockCoordinate,
    travel_direction: Option<CompassDirection>,
    /// Ordered list of waypoint names along this path (empty until waypoints are added to the template).
    waypoint_names: Vec<Option<&'static str>>,
}

/// Finds the path in `paths` matching `expected` and asserts its waypoints.
fn assert_path_exists(
    paths: &FxHashMap<super::interlocking::InterlockingPathResult, Vec<RoutingPath>>,
    expected: &ExpectedPath,
) -> googletest::Result<()> {
    let path = paths
        .keys()
        .find(|p| {
            p.endpoint.kind == expected.kind
                && p.endpoint.track_coord == expected.track_coord
                && p.endpoint.travel_direction == expected.travel_direction
                && p.via
                    .iter()
                    .map(|x| x.name.as_ref().map(String::as_str))
                    .eq(expected.waypoint_names.iter().copied())
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "expected path not found: {:?} at {:?} dir {:?}",
                expected.kind,
                expected.track_coord,
                expected.travel_direction,
            )
        })
        .or_fail()?;

    Ok(())
}

/// Verifies that `scan_interlocking_routes` returns exactly 3 distinct paths from each
/// Z+ approach track (x=2 and x=3), with hardcoded endpoint coordinates and directions.
///
/// Expected paths (empty waypoint lists will be updated when waypoints are added to the template):
///   - EndOfInterlockingSignal at (2, 0, 56) traveling ZPlus
///   - EndOfInterlockingSignal at (3, 0, 56) traveling ZPlus
///   - EndOfTrack at (6, 0, 48) with no direction
#[gtest]
fn test_scan_interlocking_routes_zplus(fixture: &TestFixture) -> googletest::Result<()> {
    fixture.start_server(|builder| crate::configure_default_game(builder))?;
    load_simple_interlocking(fixture)?;

    fixture.run_with_context(|ctx| {
        let config = ctx
            .extension::<super::CartsGameBuilderExtension>()
            .expect("CartsGameBuilderExtension")
            .clone();

        for (start_coord, dir, expected_paths) in [
            (
                BlockCoordinate::new(2, 0, 4),
                CompassDirection::ZPlus,
                vec![
                    // can stay on track 2
                    ExpectedPath {
                        kind: AdjacencyHitKind::EndOfInterlockingSignal,
                        track_coord: BlockCoordinate::new(2, 0, 56),
                        travel_direction: Some(CompassDirection::ZPlus),
                        waypoint_names: vec![Some("t2_zp".into())],
                    },
                    // can switch onto track 3 and back (2-3-2)
                    ExpectedPath {
                        kind: AdjacencyHitKind::EndOfInterlockingSignal,
                        track_coord: BlockCoordinate::new(2, 0, 56),
                        travel_direction: Some(CompassDirection::ZPlus),
                        waypoint_names: vec![Some("t3_zp".into())],
                    },
                    // can stay on track 3 - bur no way to do 3-2-3
                    ExpectedPath {
                        kind: AdjacencyHitKind::EndOfInterlockingSignal,
                        track_coord: BlockCoordinate::new(3, 0, 56),
                        travel_direction: Some(CompassDirection::ZPlus),
                        waypoint_names: vec![Some("t3_zp".into())],
                    },
                    // can run on track 4 (no waypoints on it)
                    ExpectedPath {
                        kind: AdjacencyHitKind::EndOfInterlockingSignal,
                        track_coord: BlockCoordinate::new(2, 0, 56),
                        travel_direction: Some(CompassDirection::ZPlus),
                        waypoint_names: vec![],
                    },
                    // can run on track 4 (no waypoints on it)
                    ExpectedPath {
                        kind: AdjacencyHitKind::EndOfInterlockingSignal,
                        track_coord: BlockCoordinate::new(3, 0, 56),
                        travel_direction: Some(CompassDirection::ZPlus),
                        waypoint_names: vec![],
                    },
                    // only path to end of track is via track 2->3->4->5->6 missing all waypoints
                    ExpectedPath {
                        kind: AdjacencyHitKind::EndOfTrack,
                        track_coord: BlockCoordinate::new(6, 0, 48),
                        travel_direction: None,
                        waypoint_names: vec![],
                    },
                ],
            ),
            (
                BlockCoordinate::new(3, 0, 4),
                CompassDirection::ZPlus,
                vec![
                    ExpectedPath {
                        kind: AdjacencyHitKind::EndOfInterlockingSignal,
                        track_coord: BlockCoordinate::new(2, 0, 56),
                        travel_direction: Some(CompassDirection::ZPlus),
                        waypoint_names: vec![Some("t3_zp".into())], // update when waypoints are added to the template
                    },
                    ExpectedPath {
                        kind: AdjacencyHitKind::EndOfInterlockingSignal,
                        track_coord: BlockCoordinate::new(3, 0, 56),
                        travel_direction: Some(CompassDirection::ZPlus),
                        waypoint_names: vec![Some("t3_zp").into()], // update when waypoints are added to the template
                    },
                    // can run on track 4 (no waypoints on it)
                    ExpectedPath {
                        kind: AdjacencyHitKind::EndOfInterlockingSignal,
                        track_coord: BlockCoordinate::new(2, 0, 56),
                        travel_direction: Some(CompassDirection::ZPlus),
                        waypoint_names: vec![],
                    },
                    // can run on track 4 (no waypoints on it)
                    ExpectedPath {
                        kind: AdjacencyHitKind::EndOfInterlockingSignal,
                        track_coord: BlockCoordinate::new(3, 0, 56),
                        travel_direction: Some(CompassDirection::ZPlus),
                        waypoint_names: vec![],
                    },
                    ExpectedPath {
                        kind: AdjacencyHitKind::EndOfTrack,
                        track_coord: BlockCoordinate::new(6, 0, 48),
                        travel_direction: None,
                        waypoint_names: vec![], // update when waypoints are added to the template
                    },
                ],
            ),
        ] {
            let scan_state =
                super::tracks::ScanState::create_at(start_coord, dir, ctx.game_map(), &config)
                    .or_fail()?
                    .expect("ScanState::create_at should succeed at a rail tile");

            let paths =
                scan_interlocking_routes(scan_state, 1024, ctx.game_map(), &config).or_fail()?;

            println!(
                "paths from {:?}: {:?}",
                start_coord,
                paths.keys().collect::<Vec<_>>()
            );

            expect_that!(paths.len(), eq(expected_paths.len()));
            for expected in &expected_paths {
                assert_path_exists(&paths, expected)?;
            }
        }

        Ok(())
    })
}

/// Verifies that the Z- approach at x=5 (which has a siding) returns 3 distinct paths:
///   - EndOfInterlockingSignal at (5, 0, 6) traveling ZMinus
///   - EndOfInterlockingSignal at (4, 0, 6) traveling ZMinus
///   - EndOfTrack at (6, 0, 16) with no direction (the siding dead end)
#[gtest]
fn test_scan_interlocking_routes_zminus_x5_has_siding(
    fixture: &TestFixture,
) -> googletest::Result<()> {
    fixture.start_server(|builder| crate::configure_default_game(builder))?;
    load_simple_interlocking(fixture)?;

    fixture.run_with_context(|ctx| {
        let config = ctx
            .extension::<super::CartsGameBuilderExtension>()
            .expect("CartsGameBuilderExtension")
            .clone();

        let start_coord = BlockCoordinate::new(5, 0, 60);

        let scan_state = super::tracks::ScanState::create_at(
            start_coord,
            CompassDirection::ZMinus,
            ctx.game_map(),
            &config,
        )
        .or_fail()?
        .expect("ScanState::create_at should succeed at a rail tile");

        let paths =
            scan_interlocking_routes(scan_state, 1024, ctx.game_map(), &config).or_fail()?;

        println!(
            "paths from {:?}: {:?}",
            start_coord,
            paths.keys().collect::<Vec<_>>()
        );

        let expected_paths = [
            ExpectedPath {
                kind: AdjacencyHitKind::EndOfInterlockingSignal,
                track_coord: BlockCoordinate::new(5, 0, 6),
                travel_direction: Some(CompassDirection::ZMinus),
                waypoint_names: vec![], // update when waypoints are added to the template
            },
            ExpectedPath {
                kind: AdjacencyHitKind::EndOfInterlockingSignal,
                track_coord: BlockCoordinate::new(4, 0, 6),
                travel_direction: Some(CompassDirection::ZMinus),
                waypoint_names: vec![], // update when waypoints are added to the template
            },
            ExpectedPath {
                kind: AdjacencyHitKind::EndOfTrack,
                track_coord: BlockCoordinate::new(6, 0, 16),
                travel_direction: None,
                waypoint_names: vec![], // update when waypoints are added to the template
            },
        ];

        expect_that!(paths.len(), eq(expected_paths.len()));
        for expected in &expected_paths {
            assert_path_exists(&paths, expected)?;
        }

        Ok(())
    })
}

/// Verifies that the Z- approach at x=4 (no siding) returns exactly 2 paths:
///   - EndOfInterlockingSignal at (4, 0, 6) traveling ZMinus
///   - EndOfInterlockingSignal at (5, 0, 6) traveling ZMinus
/// No dead end is reachable from x=4 because x=4 has no siding branch.
#[gtest]
fn test_scan_interlocking_routes_zminus_x4_no_siding(
    fixture: &TestFixture,
) -> googletest::Result<()> {
    fixture.start_server(|builder| crate::configure_default_game(builder))?;
    load_simple_interlocking(fixture)?;

    fixture.run_with_context(|ctx| {
        let config = ctx
            .extension::<super::CartsGameBuilderExtension>()
            .expect("CartsGameBuilderExtension")
            .clone();

        let start_coord = BlockCoordinate::new(4, 0, 60);

        let scan_state = super::tracks::ScanState::create_at(
            start_coord,
            CompassDirection::ZMinus,
            ctx.game_map(),
            &config,
        )
        .or_fail()?
        .expect("ScanState::create_at should succeed at a rail tile");

        let paths =
            scan_interlocking_routes(scan_state, 1024, ctx.game_map(), &config).or_fail()?;

        println!(
            "paths from {:?}: {:?}",
            start_coord,
            paths.keys().collect::<Vec<_>>()
        );

        let expected_paths = [
            ExpectedPath {
                kind: AdjacencyHitKind::EndOfInterlockingSignal,
                track_coord: BlockCoordinate::new(4, 0, 6),
                travel_direction: Some(CompassDirection::ZMinus),
                waypoint_names: vec![], // update when waypoints are added to the template
            },
            ExpectedPath {
                kind: AdjacencyHitKind::EndOfInterlockingSignal,
                track_coord: BlockCoordinate::new(5, 0, 6),
                travel_direction: Some(CompassDirection::ZMinus),
                waypoint_names: vec![], // update when waypoints are added to the template
            },
        ];

        expect_that!(paths.len(), eq(expected_paths.len()));
        for expected in &expected_paths {
            assert_path_exists(&paths, expected)?;
        }

        Ok(())
    })
}

/// Verifies that `apply_interlocking_routes_to_signals` writes correct routing tables
/// into each interlocking signal block: the set of terminal destinations recorded in
/// the routing table must match the set of terminals returned by a fresh
/// `scan_interlocking_routes` call from the same signal.
#[gtest]
fn test_routing_tables(fixture: &TestFixture) -> googletest::Result<()> {
    fixture.start_server(|builder| crate::configure_default_game(builder))?;
    load_simple_interlocking(fixture)?;

    fixture.run_with_context(|ctx| {
        let config = ctx
            .extension::<super::CartsGameBuilderExtension>()
            .expect("CartsGameBuilderExtension")
            .clone();

        // Run scan+apply from each approach track
        let starting_points = [
            (BlockCoordinate::new(2, 0, 4), CompassDirection::ZPlus),
            (BlockCoordinate::new(3, 0, 4), CompassDirection::ZPlus),
            (BlockCoordinate::new(4, 0, 60), CompassDirection::ZMinus),
            (BlockCoordinate::new(5, 0, 60), CompassDirection::ZMinus),
        ];
        for (coord, dir) in starting_points {
            let scan_state =
                super::tracks::ScanState::create_at(coord, dir, ctx.game_map(), &config)
                    .or_fail()?
                    .expect("valid track");
            let routes =
                scan_interlocking_routes(scan_state, 1024, ctx.game_map(), &config).or_fail()?;
            apply_interlocking_routes_to_signals(&routes, ctx.game_map()).or_fail()?;
        }

        // Find all interlocking signals at Y=2
        let mut signal_coords: Vec<(BlockCoordinate, u16)> = vec![];
        for x in 0i32..=9 {
            for z in 0i32..=66 {
                let coord = BlockCoordinate::new(x, 2, z);
                let block = ctx.game_map().get_block(coord).or_fail()?;
                if block.equals_ignore_variant(config.interlocking_signal) {
                    signal_coords.push((coord, block.variant() & 3));
                }
            }
        }
        expect_that!(signal_coords.is_empty(), eq(false));

        let mut checked = 0usize;
        for (signal_coord, rotation) in signal_coords {
            let facing = CompassDirection::from_rotation_variant(rotation);
            let Some(track_coord) = signal_coord.try_delta(0, -2, 0) else {
                continue;
            };

            // Read signal's routing tables
            let tables = ctx
                .game_map()
                .get_block_with_extended_data(signal_coord, |_, ext| {
                    Ok(ext.custom_data.as_ref().and_then(|cd| {
                        cd.downcast_ref::<super::signals::SignalConfig>().map(|sc| {
                            (
                                sc.left_paths.clone(),
                                sc.right_paths.clone(),
                                sc.forward_paths.clone(),
                            )
                        })
                    }))
                })
                .or_fail()?;

            let Some((left_paths, right_paths, forward_paths)) = tables.1 else {
                continue;
            };
            if left_paths.is_empty() && right_paths.is_empty() && forward_paths.is_empty() {
                continue; // Not reached by any starting scan; skip
            }

            // Fresh scan for ground truth
            let Some(scan_state) =
                super::tracks::ScanState::create_at(track_coord, facing, ctx.game_map(), &config)
                    .or_fail()?
            else {
                continue;
            };
            let fresh_routes =
                scan_interlocking_routes(scan_state, 1024, ctx.game_map(), &config).or_fail()?;

            fn terminal_key(
                track_coord: Option<perovskite_core::coordinates::BlockCoordinate>,
            ) -> (i32, i32, i32) {
                track_coord.map(|c| (c.x, c.y, c.z)).unwrap_or((-1, -1, -1))
            }

            use rustc_hash::FxHashSet;
            // Collect terminal (kind, x, y, z) from routing tables
            let routing_terminals: FxHashSet<(i32, i32, i32, i32)> = left_paths
                .iter()
                .chain(right_paths.iter())
                .chain(forward_paths.iter())
                .filter_map(|p| p.items.last())
                .map(|item| {
                    let (x, y, z) = terminal_key(item.track_coord.clone());
                    (item.kind as i32, x, y, z)
                })
                .collect();

            let scan_terminals: FxHashSet<(i32, i32, i32, i32)> = fresh_routes
                .into_keys()
                .map(|r| {
                    let c = r.endpoint.track_coord;
                    (r.endpoint.kind as i32, c.x, c.y, c.z)
                })
                .collect();

            expect_that!(
                &routing_terminals,
                eq(&scan_terminals),
                "routing table mismatch for signal at {:?} facing {:?}",
                signal_coord,
                facing
            );
            checked += 1;
        }
        expect_that!(checked, gt(0));
        Ok(())
    })
}
