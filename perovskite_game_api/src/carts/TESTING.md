# Cart Interlocking Test Infrastructure

## Test geometry files

### `testdata/simple_interlocking.test_geometry`

A serialized `SerializedTemplate` (protobuf via `prost::Message`) capturing a simple interlocking
layout used as the primary test fixture for interlocking logic.

This template includes four tracks - two in each direction, as well as a few dead ends on a fifth column:

* x = 2: runs in +z direction, from z = 1 to z = 65
* x = 3; runs in +z direction, from z = 1 to z = 65,
* x = 4; runs in -z direction, from z = 65 to z = 1.
* x = 5; runs in -z direction, from z = 65 to z = 1.

Note that these Z coordinates are for the physical track; see interlocking_test.rs for start points where it's
reasonable to search for an interlocking.

**Placement in tests**

| Parameter | Value |
|---|---|
| World-space origin | `(0, -1, 0)` (`SIMPLE_INTERLOCKING_ORIGIN`) |
| Template local `(0,0,0)` → world | `(0, -1, 0)` |
| Tracks sit at world Y | `0` (template Y = 1 relative to origin) |
| Scan range for sanity check | x ∈ [0, 9], y = 0, z ∈ [0, 66] |

**Marker blocks** (unique block types used as spatial markers in the layout)

| Block name | Role |
|---|---|
| `carts:rail_tile` | Track tiles (expected in large numbers) |
| `default:obsidian` | Marker |
| `default:stone` | Marker |
| `default:cactus` | Marker |
| `default:maple_tree` | Marker |

Other block types present: `carts:switch_unset`, `carts:enhanced_signal`, `carts:signal`,
`carts:gantry`, `default:dirt`, `builtin:air`.

## Test helper API

Defined in `perovskite_game_api/src/carts/interlocking_test.rs` (compiled only under `cfg(test)`).
Declared as `#[cfg(test)] mod interlocking_test` in `carts/mod.rs`.

```rust
// Constant: world-space origin for the simple_interlocking template
pub(crate) const SIMPLE_INTERLOCKING_ORIGIN: BlockCoordinate;

// Load the simple_interlocking template into the world. Call after fixture.start_server(...).
pub(crate) fn load_simple_interlocking(fixture: &TestFixture) -> googletest::Result<()>;
```

## Running the tests

```sh
cargo test --package perovskite_game_api --features test-support --lib -- carts::interlocking_test
```

## Current tests

| Test | File | Description |
|---|---|---|
| `test_simple_interlocking_loads` | `interlocking_test.rs` | Loads the template at `SIMPLE_INTERLOCKING_ORIGIN` and verifies all five expected block types are present within the scan range. |

## Known issues / notes

