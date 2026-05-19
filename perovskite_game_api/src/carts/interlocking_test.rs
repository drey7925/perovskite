use googletest::prelude::*;
use perovskite_core::coordinates::BlockCoordinate;
use perovskite_server::game_state::{
    event::EventInitiator, game_map::templates::SerializedTemplate,
};
use prost::Message;

use crate::test_support::TestFixture;

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
