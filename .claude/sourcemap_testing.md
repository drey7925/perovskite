# Test-Support Sourcemap

Everything conditionally compiled under `test-support` (or `#[cfg(any(test, feature = "test-support", doctest))]`).

---

## Feature Flag Dependency Chain

```
perovskite_game_api/Cargo.toml:129
  test-support = ["perovskite_server/test-support", "googletest"]

perovskite_server/Cargo.toml:96
  test-support = ["googletest"]

perovskite_game_api/Cargo.toml:158  (dev_server binary)
  required-features = ["default_game", "server", "test-support"]
```

Dev dependencies auto-enable `test-support`:
- `perovskite_game_api/Cargo.toml` dev-dep on `perovskite_server` with `features = ["test-support"]`

---

## Core Test Infrastructure — perovskite_game_api

**Module gate:** `perovskite_game_api/src/lib.rs:150`
```rust
#[cfg(any(test, feature = "test-support", doctest))]
pub mod test_support;
```

### `perovskite_game_api/src/test_support.rs`

| Item | Kind | Notes |
|------|------|-------|
| `TestFixture` | struct | googletest `Fixture`; thread-local in-memory world |
| `TestFixture::start_server` | method | registers game content, creates flatland world |
| `TestFixture::stop_server` | method | called automatically on tear_down |
| `TestFixture::run_assertions_in_server` | method | closure receives `&GameState` |
| `TestFixture::run_with_context` | method | closure receives `HandlerContext` (deref→GameState) |
| `TestFixture::run_timer_inline` | method | drives a single named timer synchronously |
| `TestFixture::run_all_timers_inline` | method | drives every registered timer once |
| `FlatlandMapgen` | struct | implements `MapgenInterface`; neg-Y=block, rest=air |
| `GameBuilderTestExt` | trait | extension on `GameBuilder` |
| `GameBuilderTestExt::set_flatland_mapgen` | method | configures `FlatlandMapgen` on the builder |
| `IsBlock<T>` | matcher | googletest: matches block type, ignores variant |
| `IsBlockWithVariant<T>` | matcher | googletest: matches block type + exact variant |
| `IsItemStack<T,U>` | matcher | googletest: matches item name + qty/wear |
| `ZERO_COORD` | const | `BlockCoordinate::new(0,0,0)` |

---

## Timer Helpers — perovskite_server

**`perovskite_server/src/game_state/game_map.rs`**

| Item | Line | Gate |
|------|------|------|
| `ServerGameMap::run_timer_inline(name)` | ~1863 | `#[cfg(any(test, feature="test-support", doctest))]` |
| `ServerGameMap::run_all_timers_inline()` | ~1878 | same |

**`perovskite_server/src/game_state/game_map/timers.rs`**

| Item | Line | Notes |
|------|------|-------|
| `Timer::run_inline()` | ~494 | executes timer with fake shard state |
| `TimerController::run_timer(name)` | ~1485 | runs single named timer |
| `TimerController::run_all_timers()` | ~1500 | runs all timers |

---

## Entity Coroutine Testing — perovskite_server

**`perovskite_server/src/game_state/entities.rs`** (module gate, line ~36):
```rust
// Currently unconditionally exported (cfg guard commented out):
pub mod entity_test_helpers;
```

**`perovskite_server/src/game_state/entities/entity_test_helpers.rs`**

(highly WIP, API subject to substantial change, verify against that file's current state)

| Item | Kind | Notes |
|------|------|-------|
| `CoroutineTester` | struct | drives an `EntityCoroutine` synchronously |
| `CoroutineTester::new(coro, queue_type, start_pos, gs)` | constructor | calls `advance` once to initialize |
| `CoroutineTester::advance(gs)` | method | steps the coroutine one planned move; must be called inside an async runtime (e.g. inside `run_assertions_in_server`) |
| `CoroutineTester::current_position()` | method | position before current move completes |
| `CoroutineTester::post_queue_position()` | method | position after all queued moves complete |
| `CoroutineTester::move_buffer()` | method | total seconds of moves currently queued |
| `CoroutineTester::is_engaged()` | method | false after `StopCoroutineControl` or `ImmediateDespawn` |

Import path: `perovskite_server::game_state::entities::entity_test_helpers::CoroutineTester`

---

## Server-Level Test Ext — perovskite_server

**`perovskite_server/src/server.rs:233`** — gate: `#[cfg(feature = "test-support")]`

```rust
pub mod test_support {
    pub trait EventTestExt {
        fn create_context<'a>(&self, initiator: EventInitiator<'a>) -> HandlerContext<'a>;
    }
    impl EventTestExt for Server { ... }
}
```

Used for directly constructing a `HandlerContext` at the `Server` level (lower-level than `TestFixture::run_with_context`).

---

## Carts Test Modules

**`perovskite_game_api/src/carts/mod.rs:63`**
```rust
#[cfg(any(test, feature = "test-support"))]
pub mod station_tests;
```

**`perovskite_game_api/src/carts/station_tests.rs`** — gate: `#![cfg(any(test, feature = "test-support"))]`
- Station template loading and validation tests
- Imports: `IsBlock`, `TestFixture`

**`perovskite_game_api/src/carts/interlocking_test.rs`**
- Interlocking route logic tests
- Imports: `TestFixture`

**`perovskite_game_api/src/carts/network.rs:284`**
- Network adjacency calculation tests
- Imports: `TestFixture`

---

## Files With Tests Using test-support

| File | Test subject | Key imports |
|------|-------------|-------------|
| `perovskite_game_api/src/animals/mod.rs:229` | duck coroutine smoke | `TestFixture`, `GameBuilderTestExt`, `CoroutineTester` |
| `perovskite_game_api/src/default_game/basic_blocks.rs:1311` | water flow | `TestFixture`, `IsBlock` |
| `perovskite_game_api/src/default_game/furnace.rs:430` | furnace smelting | `TestFixture`, `IsItemStack`, `ZERO_COORD` |
| `perovskite_game_api/src/carts/interlocking_test.rs:12` | interlocking routes | `TestFixture` |
| `perovskite_game_api/src/carts/network.rs:284` | cart network adjacency | `TestFixture` |
| `perovskite_game_api/src/carts/station_tests.rs:247` | station templates | `TestFixture`, `IsBlock` |
| `perovskite_game_api/src/autobuild/machines.rs:1295` | autobuilder machines | `TestFixture`, `GameBuilderTestExt`, `IsBlock` |

---

## Development Server

**`perovskite_game_api/src/bin/dev_server.rs:29`**
```rust
use perovskite_game_api::test_support::GameBuilderTestExt;
```
Uses `set_flatland_mapgen` to create a flat world for manual testing.


