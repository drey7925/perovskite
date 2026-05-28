# Cart System Design

This document describes the design, conventions, and behavior of the cart subsystem in
`perovskite_game_api`. It is intended as a living reference for future feature work and
contributors. The authoritative source of truth is always the source code; when in doubt,
read the code.

---

## Table of Contents

1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [Block Types and Block Layout Conventions](#block-types-and-block-layout-conventions)
4. [Coordinate System and Rotation Conventions](#coordinate-system-and-rotation-conventions)
5. [Track Tiles](#track-tiles)
6. [Cart Entity and Coroutine](#cart-entity-and-coroutine)
7. [Movement Planning](#movement-planning)
8. [Signalling Philosophy](#signalling-philosophy)
9. [Automatic Signals](#automatic-signals)
10. [Interlocking Signals](#interlocking-signals)
11. [Starting Signals](#starting-signals)
12. [Signal Variant Bits](#signal-variant-bits)
13. [Switches](#switches)
14. [Interlocking Pathfinding](#interlocking-pathfinding)
15. [Speed Control and Speedposts](#speed-control-and-speedposts)
16. [Slope Blocks](#slope-blocks)
17. [Pending Actions and Odometer](#pending-actions-and-odometer)
18. [Circuit Integration](#circuit-integration)
19. [Spawning a Cart](#spawning-a-cart)
20. [Track Builder Tools](#track-builder-tools)
21. [Extension Pattern](#extension-pattern)
22. [Known Limitations and Future Work](#known-limitations-and-future-work)

---

## Overview

The carts system provides high-speed automated cart transport. Carts follow predefined
tracks, respect a signal system that prevents collisions, and can navigate complex
junctions (interlockings) that include switches and multiple signal types.

Key design principles:

- **No collision detection.** Signals act as mutexes. A cart cannot enter a track section
  unless it has successfully acquired the protecting signal. This is efficient: no scanning
  of other cart positions is needed.
- **Ahead-of-time motion planning.** Carts scan up to 2048 blocks of track ahead and
  compute a braking curve (acceleration, coast, deceleration) before they need to execute
  it. The motion plan is a queue of discrete movement commands submitted to the entity
  system.
- **Odometer-based scheduling.** Signal acquisitions, signal releases, and switch releases
  are all triggered by a cumulative distance counter (the odometer), not by wall-clock
  time. This decouples track scanning from movement execution.
- **Atomic interlocking transactions.** When a cart needs to navigate a junction, it
  attempts to atomically acquire all signals and set all switches along a complete path
  before it moves. If this fails (because another cart holds a conflicting resource), the
  whole transaction is rolled back and retried after a random backoff.
- **State stored in block variants.** Signal and switch states are encoded in the variant
  bits of block IDs stored in the game map. No separate data structures are needed for
  per-signal occupancy.

---

## File Structure

| File | Responsibility |
|---|---|
| `mod.rs` | Cart entity registration, coroutine, movement planning, pending-action scheduler, speedposts, gantry, switch blocks |
| `tracks.rs` | Track tile atlas, `TileId` encoding, `TrackTile` geometry, `ScanState` and the `advance()` loop, slope blocks, rail block registration |
| `signals.rs` | Signal block registration, variant-bit constants, automatic/interlocking/starting signal acquire/release logic, signal popup UI, circuit callbacks |
| `interlocking.rs` | Interlocking pathfinder (`single_pathfind_attempt`), transaction (`SignalTransaction`), switch handling, async `interlock_cart` task |
| `track_tool.rs` | Player-facing track-building item (manual placement helper and autorouter) |

---

## Block Types and Block Layout Conventions

### Rail Blocks

| Block name | Description |
|---|---|
| `carts:rail_tile` | Standard flat rail (single block, texture atlas driven) |
| `carts:rail_slope_1_1` | Steep slope, 1:1 grade (45°) |
| `carts:rail_slope_1_8` … `carts:rail_slope_8_8` | Gradual slopes, 1/8 through 8/8 grades |

All rail blocks occupy one block in the world. A rail tile's shape and connection
direction are fully encoded in the block's variant bits (see below).

### Infrastructure Blocks

| Block name | Description | Position relative to track |
|---|---|---|
| `carts:signal` | Automatic signal | **Y+2** above the rail tile it protects |
| `carts:enhanced_signal` | Interlocking signal | **Y+2** above the rail tile it protects |
| `carts:starting_signal` | Starting (platform) signal | **Y+2** above the rail tile it protects |
| `carts:switch_unset` | Switch, idle state | **Y−1** below the rail tile at the branch point |
| `carts:switch_straight` | Switch, set for straight route | **Y−1** below the rail tile at the branch point |
| `carts:switch_diverging` | Switch, set for diverging route | **Y−1** below the rail tile at the branch point |
| `carts:speedpost1` | Speed limit 3 m/s | **Y+2** above the rail tile (same slot as signals) |
| `carts:speedpost2` | Speed limit 30 m/s | **Y+2** above the rail tile (same slot as signals) |
| `carts:speedpost3` | Speed limit 90 m/s | **Y+2** above the rail tile (same slot as signals) |
| `carts:waypoint` | Named routing waypoint | **Y+2** above the rail tile (same slot as signals) |
| `carts:gantry` | Decorative overhead frame | Above track |

**Signals, speedposts, and waypoints are all looked up by computing `track_coord.try_delta(0, 2, 0)`**
(mod.rs:1250, interlocking.rs:280). The block 2 above a rail tile is checked on every scan
step; `parse_signal()` first checks whether it is a speedpost, then whether it is a signal
block. A given position can therefore be either a signal or a speedpost, but not both.
Waypoints occupy the same Y+2 slot and are mutually exclusive with signals and speedposts at
a given track coordinate.

**Switches are looked up by computing `track_coord.try_delta(0, -1, 0)`** (interlocking.rs:499).
A switch must be placed exactly 1 block below the track tile at the branch point.

### Block Groups

All cart infrastructure blocks are registered in the `RAIL_INFRA_GROUP` fast block group
(mod.rs). This allows quick bulk checks (e.g., "is this block rail infrastructure?")
without comparing against every individual block ID.

---

## Coordinate System and Rotation Conventions

The game uses a right-handed coordinate system where **Y is up**. The horizontal plane is
X–Z.

In tile atlas space, **variant = 0 is defined with +Z as the "forward" direction** (i.e.,
the direction a cart travels on a straight tile at rotation 0). The tile atlas comment in
`tracks.rs` shows:

```
Game map coordinates for variant = 0
Top view
X <--+
     |
     V
     Z
```

So in the default orientation: X increases to the left, Z increases downward (in the
map view). In world terms, **rotation 0 → +Z direction, rotation 1 → +X, rotation 2 →
−Z, rotation 3 → −X** (clockwise when viewed from above).

`eval_rotation(x, z, flip_x, rotation)` (tracks.rs:324) applies a flip then a rotation:

| Rotation | Output (x, z) |
|---|---|
| 0 | (x, z) |
| 1 | (z, −x) |
| 2 | (−x, −z) |
| 3 | (−z, x) |

If `flip_x` is set, the x component is negated before the rotation is applied.

---

## Track Tiles

### TileId

`TileId` is a 16-bit value packed as follows (tracks.rs, `mod c`):

| Bits | Mask | Field |
|---|---|---|
| 0–1 | `ROTATION_BITS` | Rotation 0–3 (clockwise) |
| 2–5 | `X_SELECTOR` | X coordinate in tile atlas (0–15) |
| 6–9 | `Y_SELECTOR` | Y coordinate in tile atlas (0–10) |
| 10 | `FLIP_X_BIT` | Flip X flag |
| 12 | `DIVERGING_ROUTE` | This is a diverging-route tile (internal use) |
| 13 | `REVERSE_SCAN` | Cart is scanning in reverse (internal use) |
| 14 | `SLOPE_ENCODING` | Special encoding for slopes |
| 15 | `ENTRY_PRESENT` | Sentinel: TileId is valid (not empty) |

Only bits 0–10 are stored in the block variant on the game map. Bits 12–15 are used
internally during scanning and are never written to the map.

### TrackTile

Each `TileId` (atlas position + rotation + flip) maps to a `TrackTile` struct
(tracks.rs:381). Key fields:

- **`next_delta` / `prev_delta`**: How to step to the next/previous block coordinate when
  scanning forward or backward. These are 8-bit encoded (X, Y, Z) deltas, defined for
  variant=0 and automatically rotated during scanning.
- **`next_diverging_delta` / `prev_diverging_delta`**: Deltas for the diverging branch.
- **`secondary_coord` / `secondary_tile`**: Some tiles require a secondary helper block at
  an adjacent coordinate (e.g., the other half of a wide switch).
- **`tertiary_coord` / `tertiary_tile`**: Optional third helper block.
- **`allowed_next_tiles` / `allowed_prev_tiles`**: Which tile IDs are accepted at the
  next/previous block. A tile ID of `(0, 0)` is special: it matches any
  "straight-track-eligible" connection, honoring approach direction.
- **`physical_x_offset` / `physical_z_offset`**: The physical position of the cart center
  within the block, expressed in 1/128ths of a block. Used to smoothly position carts on
  curved or offset tiles.
- **`diverging_physical_x_offset` / `diverging_physical_z_offset`**: Same for the
  diverging route.
- **`max_speed` / `diverging_max_speed`**: Maximum speed (m/s) through this tile.
- **`switch_length`**: If nonzero, the tile is a switch tile. The value is a countdown
  that determines how long the switch must remain locked after a cart traverses it, to
  prevent sideswiping.

### Tile atlas: (0, 0) — Straight track

Atlas position (0, 0) is the straight track tile and is by far the most important tile to
understand. At **rotation = 0** (variant = 0 in a block with no flip), the tile runs along
the **+Z axis**: `next_delta = (0, 0, +1)` and `prev_delta = (0, 0, −1)`.

`straight_through_spawn_dirs = 0b0100_0001`:
- Bit 0 (value 1): rotation 0 → **forward** (non-reversed) ZPlus traversal is valid.
- Bit 6 (value 64): rotation 2 → **reverse** ZMinus traversal is valid.

Rotating via the standard 90°-clockwise encoding:

| Block variant low-2 bits | CompassDirection | Meaning |
|---|---|---|
| 0 | ZPlus | Forward running in +Z direction |
| 1 | XPlus | Forward running in +X direction |
| 2 | ZMinus | Forward running in −Z direction |
| 3 | XMinus | Forward running in −X direction |

A **signal or waypoint** at Y+2 uses the same low-2 variant bits to encode which direction
a correctly-oriented cart is traveling when it passes (the "front" face points away from the
approaching cart). Consequently:
- `signal_rotation_ok(variant & 3)` returns **true** if the signal's facing direction
  matches the current scan direction (i.e., the cart is approaching the front of the signal).
- For a forward ZPlus scan on a variant=0 straight tile: `signal_rotation_ok(0)` = true,
  `signal_rotation_ok(2)` = false.

**Testing shorthand:** All signal/waypoint tests that scan in ZPlus should place
variant=0 blocks for "facing correctly" and variant=2 blocks for "backwards."

### The (0, 0) tile as a wildcard sentinel

When `TileId(x=0, y=0)` appears in a tile's `allowed_next_tiles` or `allowed_prev_tiles`
array, it is **not** a reference to the straight track tile itself. It is a wildcard meaning
"any tile whose `straight_track_eligible_connections` bitfield accepts approach from this
direction." Almost every tile type (including switches and diagonals at their entry
positions) declares itself straight-track-eligible for at least one approach direction.
This lets the scanner avoid enumerating every compatible tile explicitly; a single (0,0)
entry covers all of them.

The straight track tile at (0,0) itself also uses this sentinel in its own
`allowed_next_tiles` so that it connects to any compatible tile in sequence.

### Slope Tiles

Slopes are registered as separate block types (`rail_slope_1_1`, `rail_slope_1_8` through
`rail_slope_8_8`). Their `TileId` uses `SLOPE_ENCODING` (bit 14) and encodes the
numerator and denominator of the grade. Slopes are scanned by checking the block at Y−1
(the block below the slope block).

Maximum speeds: 6 m/s for 1:1 slopes, 60 m/s for 1/8–8/8 gradual slopes.

---

## Cart Entity and Coroutine

### Entity Registration

The cart entity class is registered as `"carts:high_speed_minecart"` with:
- Move queue type: `Buffer64` (64-entry move buffer).
- Attachment offset: 1 block above the cart center (so players ride on top).
- Trailing entities: one per block of cart length beyond the first (supports multi-block
  carts; currently only length 1 is used in practice).

### CartCoroutine

`CartCoroutine` (mod.rs:860) is the core per-cart state machine. It implements
`EntityCoroutine` and is polled by the entity system via `plan_move()`. Key fields:

| Field | Purpose |
|---|---|
| `config` | Clone of `CartsGameBuilderExtension` (all registered block IDs) |
| `scan_state` | Current position in the track scan (`ScanState`) |
| `scheduled_segments` | Deque of `ScheduledSegment`s: moves with computed braking curves |
| `unplanned_segments` | Deque of raw `TrackSegment`s awaiting braking curve computation |
| `last_speed_post_indication` | Speed limit from the most recently seen speedpost |
| `last_submitted_move_exit_speed` | Exit speed of the last movement submitted to the entity system |
| `cleared_signals` | Cache: signals already checked and confirmed permissive this scan cycle |
| `held_signal` | The signal currently "held" (acquired) by this cart; released when the next signal is acquired |
| `precomputed_steps` | Interlocking route returned by the async pathfinder |
| `cart_name` | Name used for route selection at interlocking signals |
| `pending_actions` | Binary heap of `PendingActionEntry` (odometer-triggered actions) |
| `cart_length` | Number of blocks long the cart is (currently always 1) |
| `interlocking_resume_state` | If the cart is stopped at a starting signal, the state needed to resume |
| `cancellation` | `CancellationToken` for cancelling in-flight async tasks on despawn |
| `spawned_task_count` | `AsyncRefcount` tracking in-flight async tasks |

### ScanState

`ScanState` (tracks.rs:1516) records the current scan position:

| Field | Meaning |
|---|---|
| `block_coord` | World coordinate of the current track block |
| `is_reversed` | True if the cart is scanning in the reverse direction for this tile |
| `is_diverging` | True if the cart is on the diverging route through this tile |
| `allowable_speed` | Maximum speed from track geometry at this tile |
| `odometer` | Cumulative distance (meters) travelled from the cart's spawn position |
| `current_tile_id` | The `TileId` of the block at `block_coord` |

---

## Movement Planning

### Overview

Each time the entity system calls `plan_move()` (which translates to `plan_move_impl()` in
mod.rs), the cart coroutine:

1. **Scans ahead** via `scan_tracks()` until it has scanned up to 2048 meters of track, or
   until it hits a deferral (a block not yet loaded from disk). New raw `TrackSegment`s are
   appended to `unplanned_segments`.

2. **Promotes schedulable segments** via `promote_schedulable()`: filters the unplanned
   queue and moves segments that are safe to schedule (i.e., any braking can be performed
   in time) into `scheduled_segments`, computing the braking curve for each.

3. **Submits moves** to the entity system from `scheduled_segments` until the move buffer
   is full.

### Scanning: `scan_tracks()`

The scan loop repeatedly calls `ScanState::advance()` to step from one track block to the
next. Each step:

- Reads the block at the current `block_coord`.
- Validates the tile against the expected list of allowed next tiles.
- Checks for a signal at `block_coord + (0, 2, 0)`.
- Checks for a speedpost at `block_coord`.
- Appends a new `TrackSegment` with the tile's `max_speed` and current odometer position.

If a signal is encountered, the cart attempts to acquire it. Depending on the signal type
and its state, the scan may pause and defer (returning a `Deferral` to the entity system,
which will call `continuation()` once the deferred result is available).

### Braking Curve: `schedule_single_segment()`

For each `TrackSegment`, `schedule_single_segment()` computes up to three sub-moves:

1. An acceleration phase (if the cart is below the segment's speed limit).
2. A coast phase at constant speed (if needed).
3. A deceleration phase (if the next segment requires a lower speed).

Acceleration is capped at `MAX_ACCEL = 8.0 m/s²`. Emergency braking uses the same value.
If the current plan cannot honor a required stopping distance (e.g., a signal was acquired
later than expected), the cart schedules emergency braking.

### `promote_schedulable()`

A segment is unconditionally schedulable if the segment immediately before it was speed-
limited by track geometry (not by a signal). This means the cart knows it can decelerate
in time. Otherwise, the cart only schedules if:

- Fewer than 2 moves are buffered, or
- The cart is at a standstill, or
- The remaining scheduled time is less than 2 seconds.

This conservative approach prevents over-committing the braking curve before signal states
are known.

---

## Signalling Philosophy

Signals in this system act as **mutexes**, not as track circuits. Each cart is responsible
for:

1. Acquiring a signal before it is allowed to enter the protected section.
2. Transitioning the signal to a "traffic present" state as it passes.
3. Releasing the signal once it has cleared the section (triggered by acquiring the *next*
   signal, or by despawning).

This avoids the need to scan all cart positions anywhere in the network. The game map
itself stores signal state in variant bits, so block map reads/writes serve as the
synchronization mechanism.

---

## Automatic Signals

Automatic signals (`carts:signal`) provide simple headway separation on unidirectional
track segments. They are the simplest signal type and are used between interlockings.

### State Machine

| Variant bits set | Meaning |
|---|---|
| `VARIANT_RESTRICTIVE` only | Idle: signal can be acquired |
| `VARIANT_PERMISSIVE` | Acquired: a cart has been granted permission to pass |
| `VARIANT_RESTRICTIVE` \| `VARIANT_RESTRICTIVE_TRAFFIC` | Cart has entered the block; no new cart may acquire |

### Acquisition Sequence

1. Cart scan encounters the signal at `track_coord + (0, 2, 0)`.
2. Cart calls `automatic_signal_acquire()` (signals.rs). This performs a
   compare-and-swap on the block: if the block is in `VARIANT_RESTRICTIVE` (idle), it
   transitions to `VARIANT_PERMISSIVE`.
3. If the CAS succeeds, the cart records this as the `held_signal` and schedules a
   `SignalEnterBlock` action at the odometer position of the signal tile.
4. When the odometer reaches the signal tile (`SignalEnterBlock` fires), the signal is
   transitioned to `VARIANT_RESTRICTIVE | VARIANT_RESTRICTIVE_TRAFFIC`.
5. When the cart acquires the *next* signal (of any type), a `SignalRelease` action fires
   for the previously held signal, resetting it to `VARIANT_RESTRICTIVE`.

### Failure Modes

- If the signal is `VARIANT_PERMISSIVE`: unexpected state (another cart was spawned into a
  live signal section). The cart waits.
- If the signal is `VARIANT_RESTRICTIVE_TRAFFIC`: blocked by preceding cart. The cart
  waits.

---

## Interlocking Signals

Interlocking signals (`carts:enhanced_signal`) are used at junctions where multiple
conflicting routes converge or diverge. They support bidirectional track and left/right
route selection.

The principle: before a cart enters a junction, it must find a **complete path** through
the junction to an exit (an automatic signal, a starting signal set to stop, or a physical
end of track). Only once that complete path is found and all its signals and switches
atomically acquired does the cart enter.

See [Interlocking Pathfinding](#interlocking-pathfinding) for the detailed algorithm.

After pathfinding succeeds, the cart receives a `Vec<InterlockingStep>` (stored in
`precomputed_steps`). Each step records which signal or switch to acquire at which odometer
position, allowing the main coroutine to execute the plan incrementally.

---

## Starting signals

Starting signals (`carts:starting_signal`) are a special interlocking signal that permits
a cart to stop within a junction—for example, at a station platform or loading hopper.

The user (or circuit automation) controls whether the signal is in **proceed** or **stop**
mode.

### Stop mode

- A cart approaching from the **front** treats the signal as end-of-track: it decelerates
  and stops just before the signal.
- The cart sets `VARIANT_STARTING_HELD` as it approaches, to prevent a cart behind it from
  also trying to enter the stopped section.
- The cart then enters `InterlockingResumeState` and waits asynchronously until the signal
  is cleared by the user or circuit.

### Proceed mode

- A cart approaching from the **front** treats the signal like a normal interlocking
  signal: it acquires the signal, passes through, and continues to the next signal.
- A cart approaching from the **back** may pass through only if `VARIANT_STARTING_HELD`
  is not set (meaning no cart is in the process of stopping in front).

---

## Signal Variant Bits

All three signal types (automatic, interlocking, starting) share the same variant
bit layout (signals.rs), and all three also register `SignalConfig` as their
extended-data type and reuse `spawn_signal_popup` for their UI. That means
`SignalConfig::signal_nickname`, `cached_paths`, and the route-pattern fields are
readable/writable identically on all three — only the pathfinder and interlocking
semantics differ.

| Constant | Value | Meaning |
|---|---|---|
| Direction (bits 0–1) | `0b00`–`0b11` | Which direction the signal faces (rotation, clockwise) |
| `VARIANT_PERMISSIVE` | 4 | Signal is clear; a cart has acquired it |
| `VARIANT_RESTRICTIVE_TRAFFIC` | 8 | A cart is in the protected block |
| `VARIANT_RESTRICTIVE` | 16 | Signal is not clear (idle or blocked) |
| `VARIANT_RESTRICTIVE_EXTERNAL` | 32 | An external circuit is holding the signal red |
| `VARIANT_RIGHT` | 64 | Interlocking: route leads to the right |
| `VARIANT_LEFT` | 128 | Interlocking: route leads to the left |
| `VARIANT_PRELOCKED` | 256 | Interlocking pathfinder has tentatively reserved this signal |
| `VARIANT_STARTING_HELD` | 512 | A cart is approaching/stopped at this starting signal's front |

The visual appearance of the signal is driven entirely by which of these bits are set
(using per-variant-bit AABB show/hide in the client).

---

## Switches

Switch blocks are placed at **Y−1** below the branch-point rail tile. Three block states exist:

| Block name | Meaning |
|---|---|
| `carts:switch_unset` | Idle; not currently allocated to any route |
| `carts:switch_straight` | Locked for straight-through movement |
| `carts:switch_diverging` | Locked for diverging movement |

On cold map load, all switches are reset to `switch_unset` by a postprocessor (mod.rs,
the cold-load handler). This ensures no stale route locks survive a server restart.

### Switch Lock Duration

The `switch_length` field on a `TrackTile` indicates how many track tiles the cart must
travel past the switch before the switch can be released. This prevents a cart's trailing
body from sideswiping a switch that was released too early. A value of 0 means the switch
can be released immediately.

---

## Interlocking Pathfinding

The pathfinder (`interlocking.rs`, `single_pathfind_attempt()`) is called asynchronously
via `interlock_cart()` and runs under the entity system's deferred-result mechanism.

### Algorithm

1. Start at the first interlocking signal tile.
2. Scan forward tile-by-tile (up to 2048 tiles).
3. At each signal, call `query_interlocking_signal()` to determine whether to go straight,
   diverge left, or diverge right, based on the cart's name pattern matching against the
   signal's configured routes.
4. At each branch point, check the switch at Y−1. Decide whether to set it straight or
   diverging based on the route decision from the preceding signal.
5. For each signal and switch encountered, attempt to atomically set the `VARIANT_PRELOCKED`
   / preliminary state using `SignalTransaction`.
6. If the scan reaches an automatic signal, a starting signal set to stop, or a physical
   track end, the transaction is **committed**: all prelocked signals are set permissive,
   all switches are set to their route state.
7. If a conflict is found (a signal or switch already held by another cart), the
   transaction is **rolled back** (RAII drop of `SignalTransaction`), and the pathfinder
   returns `None`. The cart retries after a random backoff of 500–1000 ms.

### Route Selection

`query_interlocking_signal()` (signals.rs) inspects the signal's `SignalConfig` extended
data, which stores `left_routes` and `right_routes` as lists of name patterns. The cart
name is matched against these patterns using simple wildcard matching (`*` and `?`). If no
pattern matches, the route defaults to **straight**.

For manual dispatch, the signal can be in `ManuallySignalledNoDecision` mode, causing the
cart to wait until an external circuit sends a bus message with the cart ID and the desired
route.

---

## Speed Control and Speedposts

Speed limits come from three sources, evaluated as a minimum:

1. **Track geometry**: `TrackTile::max_speed` (or `diverging_max_speed`). The inherent
   maximum is 90 m/s (≈320 km/h). Curves, switches, and slopes impose lower limits.
2. **Speedposts**: blocks placed at Y+2 above a rail tile (the same position as signals),
   scanned at each tile during `scan_tracks()`. `parse_signal()` checks for a speedpost
   first, before checking for signal types. Three levels:
   - `carts:speedpost1`: 3 m/s
   - `carts:speedpost2`: 30 m/s
   - `carts:speedpost3`: 90 m/s
3. **Signal-imposed stops**: when a signal must be approached at zero speed (e.g., a
   starting signal set to stop), the braking curve targets zero at that point.

The `last_speed_post_indication` field in `CartCoroutine` carries the most recent
speedpost value forward. It persists until overridden by a later speedpost.

Acceleration: constant at `MAX_ACCEL = 8.0 m/s²` for both acceleration and braking.
Emergency braking also uses this value.

---

## Slope Blocks

Slopes are implemented as separate block types rather than as variant configurations of
the flat rail block:

| Block | Grade | Max speed |
|---|---|---|
| `carts:rail_slope_1_1` | 1:1 (45°) | 6 m/s |
| `carts:rail_slope_1_8` | 1/8 | 60 m/s |
| … | … | … |
| `carts:rail_slope_8_8` | 8/8 (same as 1:1 but different registration) | 60 m/s |

Slope tiles are scanned differently: the block at Y−1 (below the slope block) is checked
rather than the current Y. The tile's physical Y offset is non-zero, displacing the cart
vertically as it traverses the slope.

---

## Pending Actions and Odometer

Actions that must occur at specific track positions are queued in `pending_actions`
(a `BinaryHeap<PendingActionEntry>`), sorted by odometer value. The action types are:

| Action | When it fires | Effect |
|---|---|---|
| `SignalEnterBlock` | When the cart reaches the signal tile | Transition signal to `RESTRICTIVE \| RESTRICTIVE_TRAFFIC` |
| `SignalRelease` | When the cart acquires the next signal (i.e., previous block cleared) | Reset signal to `RESTRICTIVE` |
| `StartingSignalEnterBlockReverse` | Reverse approach to a starting signal | Special-case transition |
| `SwitchRelease` | When the cart has travelled `switch_length` tiles past the switch | Reset switch to `switch_unset` |

Each action is spawned as a short-lived async task via `spawn_delayed()` with a calculated
wall-clock duration derived from the odometer distance and scheduled speed. A
`CancellationToken` allows all outstanding tasks to be cancelled on cart despawn.

---

## Circuit Integration

Signals integrate with the game's circuit system (`crate::circuits`):

- **`VARIANT_RESTRICTIVE_EXTERNAL`**: When an external circuit input is active on a signal,
  this bit is set, causing the signal to remain red regardless of cart state.
- **Bus messages**: After a successful (or failed) interlocking acquisition, the
  interlocking signal emits a bus message. The message includes the signal coordinate, cart
  name, cart ID, and outcome. External circuits can use this to trigger further automation.
- **Manual route dispatch**: An interlocking signal can be configured for manual dispatch.
  In this mode, it emits a message asking for a route decision and waits for an external
  bus message containing the cart ID and the chosen route.

---

## Spawning a Cart

Carts are spawned by right-clicking a rail block with the `carts:minecart` item
(`place_cart()` in mod.rs). The placement logic:

1. Validates the target block is a rail block.
2. Calls `ScanState::spawn_at()` to determine the cart's initial tile, direction, and
   diverging state, based on the player's facing direction (`az_direction`).
3. Initializes a `CartCoroutine` and registers a new entity at the block coordinate (Y+1).

The spawning direction is constrained by the `straight_through_spawn_dirs` and
`diverging_dirs_spawn_dirs` bitfields on the tile, which specify which approach directions
are valid for spawning.

---

## Track Builder Tools

`track_tool.rs` provides two player-facing tools registered as items:

- **Manual builder**: The player places track by pointing at existing track and clicking.
  The tool infers the correct tile variant based on the direction of the existing endpoint.
- **Autorouter**: Given a start and end point, attempts to automatically lay a track path
  between them, choosing tiles and variants appropriately.

These are builder conveniences and do not affect runtime cart behavior.

---

## Extension Pattern

`CartsGameBuilderExtension` (mod.rs:63) holds all registered block IDs and the cart entity
class ID. It implements both `GameBuilderExtension` (registered during game construction)
and `GameStateExtension` (available server-wide at runtime via the extension registry).

It is cloned into each `CartCoroutine` so that async coroutine code can look up block IDs
without accessing the global game builder.

Helper methods:
- `is_any_rail_block(block)`: returns true if the block is any of the rail types.
- `parse_speedpost(block)`: returns the speed limit if the block is a speedpost.
- `parse_slope(block)`: returns `(numerator, denominator, rotation)` for slope blocks.
- `slope_tile(block)`: returns the `TileId` for a slope block.

---

## Known Limitations and Future Work

The following are known gaps and areas identified for future development:

- **Dispatch / scheduling**: There is currently no automated dispatch system. Carts start
  moving when placed and follow route patterns based on cart name. Future work should add
  timetable-based or demand-based dispatch.
- **Stopping at stations without starting signals**: Stopping mid-route without an
  interlocking is not supported. Starting signals are required for any planned stop.
- **Cart length**: Only carts of length 1 are used in practice. Multi-block carts are
  partially implemented (trailing entities) but not fully exercised.
- **Cargo / inventory**: No mechanism exists yet for carts to carry or transfer items.
  The starting signal is designed to support loading/unloading pauses when this is added.
- **Unloaded chunks**: If a cart approaches an unloaded chunk, the scan defers. The cart
  will wait but does not trigger chunk loading. Behavior in this scenario is ad hoc.
- **No slip switches**: `switch_length`'s upper bit is reserved for future slip-switch
  support.
- **Route patterns are name-based**: Route selection at interlocking signals uses cart name
  wildcard matching. A more structured routing table (e.g., destination codes) may be
  desirable.
- **Manual dispatch bus protocol**: The bus message protocol for manual dispatch is
  functional but not formally documented or stabilized.
- **Speedpost appearance**: Speedpost blocks are currently simple cubes with no directional
  orientation. A TODO comment in `mod.rs` notes they should be updated.
