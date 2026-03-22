---
name: block_testing
description: Testing strategies and infrastructure for game block behavior in perovskite_game_api. Use when asked to write tests for block placement, dig handlers, timers, inventories, or other game mechanics.
---

The test infrastructure lives in `perovskite_game_api/src/test_support.rs` and is gated behind the `test-support` feature flag (which also enables `perovskite_server/test-support` and `googletest`).

All testing uses the **googletest** crate (v0.14.2). Most tests use the `#[gtest]` macro; simple unit tests that don't need a live server use the standard `#[test]` macro.

## Setup: Enabling Tests

Tests that use `TestFixture` must be in a crate or binary that depends on `perovskite_game_api` with `features = ["test-support"]`. Import everything with:

```rust
use perovskite_game_api::test_support::*;
use googletest::prelude::*;
```

## Core: `TestFixture`

`TestFixture` is a googletest fixture — it implements `googletest::fixtures::Fixture` and is set up/torn down automatically per test. It provides a thread-local in-memory world backed by `InMemGameDatabase`.

### Lifecycle

```rust
#[gtest]
fn my_test(fixture: &TestFixture) -> googletest::Result<()> {
    // 1. Start the server with the content under test
    fixture.start_server(|builder| crate::configure_default_game(builder))?;

    // 2. Run assertions / manipulations
    fixture.run_assertions_in_server(|gs| { ... })?;

    // 3. Server is stopped automatically on tear_down
    Ok(())
}
```

`start_server` creates a flatland world filled with air by default. Pass a `GameBuilder` callback to register blocks, items, timers, etc. Call `configure_default_game` to load the full default game content.

### Running Assertions

**`run_assertions_in_server`** — access `GameState` directly:

```rust
fixture.run_assertions_in_server(|gs: &GameState| {
    gs.game_map().set_block(ZERO_COORD, MY_BLOCK, None).or_fail()?;
    expect_that!(gs.game_map().get_block(ZERO_COORD).or_fail()?, IsBlock(MY_BLOCK));
    Ok(())
})?;
```

**`run_with_context`** — access a full `HandlerContext` (needed for popups, inventory views, item manager, etc.). The initiator is `EventInitiator::Plugin("TestSupport_TestFixture")`:

```rust
fixture.run_with_context(|ctx| {
    ctx.block_types().get_by_name(MY_BLOCK.0).expect("not found");
    ctx.item_manager().get_item(MY_ITEM.0).expect("not found");
    ctx.game_map().set_block(ZERO_COORD, block_id, None).or_fail()?;
    Ok(())
})?;
```

`HandlerContext` derefs to `GameState`, so `ctx.game_map()`, `ctx.block_types()`, and `ctx.item_manager()` all work.

### Running Timers

Timers registered via the block/game API don't fire automatically in tests. Drive them manually:

```rust
fixture.run_timer_inline("default:furnace_timer")?;   // run one named timer
fixture.run_all_timers_inline()?;                     // run every registered timer once
```

These are the primary way to test time-dependent game logic (liquid flow, furnace smelting, crop growth, etc.).

## Convenience Constant

```rust
pub const ZERO_COORD: BlockCoordinate = BlockCoordinate::new(0, 0, 0);
```

Use `ZERO_COORD` as the default coordinate for single-block tests. For multi-block tests, construct coordinates directly: `BlockCoordinate::new(x, y, z)`. Positive Y is up.

## Flatland Mapgen

By default, `start_server` sets up an all-air world. To test blocks that need a solid ground, use `GameBuilderTestExt::set_flatland_mapgen`:

```rust
fixture.start_server(|builder| {
    builder.set_flatland_mapgen(DIRT);  // negative Y = DIRT, non-negative Y = air
    // ... register other content ...
    Ok(())
})?;
```

`FlatlandMapgen` is also public and implements `MapgenInterface` if you need it directly.

## Custom Matchers

### `IsBlock<T>` — match block type, ignoring variant

```rust
expect_that!(gs.game_map().get_block(coord).or_fail()?, IsBlock(MY_BLOCK));
// Also works with AIR_ID:
expect_that!(gs.game_map().get_block(coord).or_fail()?, IsBlock(AIR_ID));
```

`IsBlock` accepts anything that implements `TryToBlockId` — typically a `BuiltBlock`, a `BlockId`, or a `StaticBlockName`.

### `IsBlockWithVariant<T>` — match block type AND variant

```rust
expect_that!(gs.game_map().get_block(coord).or_fail()?, IsBlockWithVariant(specific_block_id));
```

The error message shows human-readable block names and the variant hex value for both expected and actual.

### `IsItemStack<T, U>` — match item name and quantity/wear

```rust
expect_that!(item_stack, IsItemStack(MY_ITEM.0, eq(5)));
expect_that!(item_stack, IsItemStack(MY_ITEM.0, lt(10u32)));

// In combination with elements_are! for Vec<ItemStack>:
expect_that!(
    dig_result.item_stacks,
    elements_are![IsItemStack(DIRT.0, eq(1))]
);
```

The second argument is any `Matcher<u32>` from the googletest prelude: `eq`, `lt`, `gt`, `ge`, `le`, etc.

## Key `GameState` / Map Operations

These are available inside both `run_assertions_in_server` and `run_with_context`:

```rust
// Place a block (None = no extended data)
gs.game_map().set_block(coord, block_id, None).or_fail()?;

// Read a block
let id: BlockId = gs.game_map().get_block(coord).or_fail()?;

// Dig a block (fires dig_handler_inline + dig_handler_full, returns drops)
let result = gs.game_map().dig_block(coord, &EventInitiator::Engine, None).or_fail()?;
// result.item_stacks: Vec<ItemStack>

// Read block + extended data atomically
let (block_id, value) = ctx.game_map().get_block_with_extended_data(coord, |data| {
    Ok(data.simple_data.get("key").cloned())
})?;

// Mutate block + extended data atomically
ctx.game_map().mutate_block_atomically(coord, |block_id, ext| {
    let data = ext.get_or_insert_with(Default::default);
    data.simple_data.insert("key".into(), "val".into());
    Ok(())
})?;

// Run all registered timers
gs.game_map().run_all_timers_inline().or_fail()?;
```

## Testing Inventory Behavior

Inventories are accessed through **popups** created in a `run_with_context` call. This pattern mimics what a player sees when they open an interactive block:

```rust
fixture.run_with_context(|ctx| {
    // Set block first
    ctx.game_map().set_block(ZERO_COORD, furnace_id, None).or_fail()?;

    // Open the popup (this also initializes the block's inventories)
    let popup = make_furnace_popup(&ctx, ZERO_COORD).or_fail()?;

    // Get a specific inventory view by name (names are defined by the block)
    let fuel_view = popup.inventory_views().get(FURNACE_FUEL).unwrap();

    // Put an item in slot 0
    let leftover = fuel_view.put(&popup, 0, item_stack.into()).or_fail()?;
    expect_that!(leftover, none());  // none() = nothing was rejected

    Ok(())
})?;

// Advance time
fixture.run_timer_inline("default:furnace_timer")?;

// Check the result
fixture.run_with_context(|ctx| {
    let popup = make_furnace_popup(&ctx, ZERO_COORD).or_fail()?;
    let output_view = popup.inventory_views().get(FURNACE_OUTPUT).unwrap();
    let contents = output_view.peek(&popup).or_fail()?;
    expect_that!(contents[0], some(IsItemStack(IRON_INGOT.0, ge(1u32))));
    Ok(())
})?;
```

## Testing Dig Drops

```rust
fixture.run_assertions_in_server(|gs| {
    gs.game_map().set_block(ZERO_COORD, DIRT_WITH_GRASS, None).or_fail()?;

    let result = gs.game_map()
        .dig_block(ZERO_COORD, &EventInitiator::Engine, None)
        .or_fail()?;

    expect_that!(result.item_stacks, elements_are![IsItemStack(DIRT.0, eq(1))]);
    expect_that!(gs.game_map().get_block(ZERO_COORD).or_fail()?, IsBlock(AIR_ID));
    Ok(())
})?;
```

## Testing Liquid Flow

Liquid flow is timer-driven. Set the liquid block somewhere with a free space below, then run timers:

```rust
fixture.run_assertions_in_server(|gs| {
    let water_id = gs.block_types()
        .get_by_name(WATER.0)
        .expect("WATER not found")
        .with_variant_unchecked(0xfff);  // variant encodes fill level

    let above = BlockCoordinate::new(0, 2, 0);
    let below = BlockCoordinate::new(0, 1, 0);

    gs.game_map().set_block(above, water_id, None).or_fail()?;
    gs.game_map().set_block(below, AIR_ID, None).or_fail()?;

    gs.game_map().run_all_timers_inline().or_fail()?;

    expect_that!(gs.game_map().get_block(below).or_fail()?, IsBlock(WATER));
    Ok(())
})?;
```

Note: chunk boundaries (e.g., Y=15 and Y=16 are in adjacent chunks) are worth testing separately as edge cases.

## Testing Variants

```rust
fixture.run_assertions_in_server(|gs| {
    gs.game_map().set_block(ZERO_COORD, my_block.with_variant_unchecked(3), None).or_fail()?;
    let id = gs.game_map().get_block(ZERO_COORD).or_fail()?;
    // IsBlock ignores variant:
    expect_that!(id, IsBlock(my_block));
    // IsBlockWithVariant checks exact match:
    expect_that!(id, IsBlockWithVariant(my_block.with_variant_unchecked(3)));
    // Read raw variant:
    assert_eq!(id.variant(), 3);
    Ok(())
})?;
```

## Pure Unit Tests (No Fixture)

For logic that doesn't need a live server (e.g., recipe matching, pure algorithms):

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_pure_logic() {
        // Direct struct instantiation and assertion — no TestFixture needed
        let recipe = MyRecipe { ... };
        assert!(recipe.matches(&stacks));
    }
}
```

## Common Imports

```rust
#[cfg(test)]
mod tests {
    use crate::test_support::{TestFixture, ZERO_COORD, IsBlock, IsBlockWithVariant, IsItemStack};
    use googletest::prelude::*;
    use perovskite_core::block_id::special_block_defs::AIR_ID;
    use perovskite_core::coordinates::BlockCoordinate;
    use perovskite_server::game_state::{GameState, event::EventInitiator};
}
```