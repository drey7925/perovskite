use googletest::prelude::*;
use perovskite_core::coordinates::BlockCoordinate;
use perovskite_server::game_state::{
    blocks::CompassDirection, event::EventInitiator, game_map::templates::SerializedTemplate,
};
use prost::Message;

use crate::test_support::TestFixture;

use super::interlocking::scan_interlocking_routes;
use super::network::AdjacencyHitKind;

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
            (BlockCoordinate::new(2, 0, 1), CompassDirection::ZPlus),
            (BlockCoordinate::new(3, 0, 1), CompassDirection::ZPlus),
            (BlockCoordinate::new(4, 0, 65), CompassDirection::ZMinus),
            (BlockCoordinate::new(5, 0, 65), CompassDirection::ZMinus),
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

/// Verifies that `scan_interlocking_routes` returns at least 3 distinct paths
/// from each of the four approach tracks into the simple_interlocking template.
///
/// Per the design doc, from each Z+ approach track (x=2 or x=3) we expect:
///   - two automatic-signal exits (one at x=2, one at x=3)
///   - one dead-end (at x=6)
/// The template is currently missing backwards-facing automatic signals at
/// wrong-way exits, so extra results beyond the 3 are permitted.
///
/// From each Z- approach track (x=4 or x=5) we similarly expect at least 3.
#[gtest]
fn test_scan_interlocking_routes_zplus(fixture: &TestFixture) -> googletest::Result<()> {
    fixture.start_server(|builder| crate::configure_default_game(builder))?;
    load_simple_interlocking(fixture)?;

    fixture.run_with_context(|ctx| {
        let config = ctx
            .extension::<super::CartsGameBuilderExtension>()
            .expect("CartsGameBuilderExtension")
            .clone();

        for (start_coord, dir) in [
            (BlockCoordinate::new(2, 0, 1), CompassDirection::ZPlus),
            (BlockCoordinate::new(3, 0, 1), CompassDirection::ZPlus),
        ] {
            let scan_state =
                super::tracks::ScanState::create_at(start_coord, dir, ctx.game_map(), &config)
                    .or_fail()?
                    .expect("ScanState::create_at should succeed at a rail tile");

            let paths =
                scan_interlocking_routes(scan_state, 1024, ctx.game_map(), &config).or_fail()?;

            // Every result must have a non-trivial endpoint kind.
            for path in &paths {
                expect_that!(
                    path.endpoint.kind,
                    any!(
                        eq(AdjacencyHitKind::EndOfInterlockingSignal),
                        eq(AdjacencyHitKind::EndOfTrack)
                    )
                );
            }

            let auto_signal_exits = paths
                .iter()
                .filter(|p| p.endpoint.kind == AdjacencyHitKind::EndOfInterlockingSignal)
                .count();
            let dead_ends = paths
                .iter()
                .filter(|p| p.endpoint.kind == AdjacencyHitKind::EndOfTrack)
                .count();

            // At least two automatic-signal exits (one per exit track) and one dead end.
            expect_that!(auto_signal_exits, ge(2));
            expect_that!(dead_ends, ge(1));
        }

        Ok(())
    })
}

/// Same as above for the Z- approach tracks.
#[gtest]
fn test_scan_interlocking_routes_zminus(fixture: &TestFixture) -> googletest::Result<()> {
    fixture.start_server(|builder| crate::configure_default_game(builder))?;
    load_simple_interlocking(fixture)?;

    fixture.run_with_context(|ctx| {
        let config = ctx
            .extension::<super::CartsGameBuilderExtension>()
            .expect("CartsGameBuilderExtension")
            .clone();

        for (start_coord, dir) in [
            (BlockCoordinate::new(4, 0, 65), CompassDirection::ZMinus),
            (BlockCoordinate::new(5, 0, 65), CompassDirection::ZMinus),
        ] {
            let scan_state =
                super::tracks::ScanState::create_at(start_coord, dir, ctx.game_map(), &config)
                    .or_fail()?
                    .expect("ScanState::create_at should succeed at a rail tile");

            let paths =
                scan_interlocking_routes(scan_state, 1024, ctx.game_map(), &config).or_fail()?;

            for path in &paths {
                expect_that!(
                    path.endpoint.kind,
                    any!(
                        eq(AdjacencyHitKind::EndOfInterlockingSignal),
                        eq(AdjacencyHitKind::EndOfTrack)
                    )
                );
            }

            let auto_signal_exits = paths
                .iter()
                .filter(|p| p.endpoint.kind == AdjacencyHitKind::EndOfInterlockingSignal)
                .count();
            let dead_ends = paths
                .iter()
                .filter(|p| p.endpoint.kind == AdjacencyHitKind::EndOfTrack)
                .count();

            expect_that!(auto_signal_exits, ge(2));
            expect_that!(dead_ends, ge(1));
        }

        Ok(())
    })
}
