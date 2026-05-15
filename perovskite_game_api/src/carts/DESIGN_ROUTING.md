# Cart routing system design

This is the high-level design of cart routing. It is intended to complement DESIGN_MOVEMENT.md - that doc focuses
on lower-level cart movement and operations, while the document focuses on a higher-level routing system.

This is a greenfield design - this doc will be filled in as development proceeds.


## Important factors to keep in mind

* Carts move at up to 90 m/s. This is fast in world-coordinate terms, but slow in terms of computational time. A
  well-written pathfinding operation can easily run faster than a cart, all while warming the cache for that cart.
* Individual block reads (including with extended data) are cheap.
* Immediate consistency is neither achievable at scale nor desired (some degree of eventual consistency and discovery delay is acceptable and expected as pat of gameplay)
* `force_disable_track_placer` is an unfortunate name for a non-carts-specific library functionality. This disables track*ing* the event initiator that placed a block, and doesn't have much to do with placing tracks in particular.

## Design rules

### Diverging routes are interlocking-only

**Rule:** Diverging routes (setting `ScanState::is_diverging = true`) only apply inside interlockings. Switches
placed outside an interlocking are decorative/visual and carry no routing meaning. The network-layer scanner
(`network.rs`) must never set the diverging bit of a `ScanState`; only the interlocking pathfinder
(`interlocking.rs`) may do so.

Practically: when the network scanner advances past a switch tile that is outside an interlocking, it always
follows the straight-through path regardless of the physical switch state. The `ScanState::is_diverging` field
should remain `false` throughout a network adjacency scan.

## Testing conventions for network.rs

The following conventions make unit tests for the higher-level routing layer simple and consistent:

* **Z+ is the canonical test direction.** Scanning in the ZPlus direction means we encounter variant 0 of all
  signal and waypoint blocks as the "facing correctly" case. The one exception is when we need the back face of a
  starting signal, which requires variant 2.

* **Separate concerns from the track scanner.** Network-layer tests assume the track scanner (`ScanState::advance`)
  is correct and do not test it.

* **Real server via `TestFixture`.** Tests spin up a full game server using `TestFixture` and
  `configure_default_game` (which already includes `register_carts` — do **not** call `register_carts` again or
  you will get "Resource already exists" errors). Retrieve the real `CartsGameBuilderExtension` via
  `gs.extension::<CartsGameBuilderExtension>().unwrap()` and place blocks with `gs.game_map().set_block(...)`.

* **`find_adjacency` returns `Result<AdjacencyHit>`.** Map errors are propagated as `Err` rather than silently
  treated as end-of-track or air. Callers must `.or_fail()?` (in tests) or `?`-propagate (in production code).

* **Straight track only.** For nearly all tests, use variant=0 tracks (straight ZPlus) at Y=64, and scan in the
  ZPlus direction. Place variant=0 signals/waypoints for "facing correctly" and variant=2 for "backwards" (facing
  ZMinus). Signals/waypoints go at Y=66 (Y+2 above the track tile).

## Waypoints

Waypoints are the fundamental building block of the cart routing system.

A waypoint is a location that's relevant for cart routing/dispatch in some way. It has a location (block coordinate)
and a direction (CompassDirection) - bidirectional tracks can have two waypoints near each other or potentially at the same location
(note that right now, there's no way to put two waypoints at the same location, since all of the existing waypoint-bearing blocks are unidirectional, but that may change!). The coordiante and direction together uniquely identify a waypoint.

In graph-based pathfinding, waypoints are the nodes of the graph, and the edges are the segments of track between
waypoints. Waypoint-bearing blocks are placed at the Y+2 position above the track tile they annotate (the same slot
used by signals and speedposts). Despite the marker being at Y+2, **the waypoint's coordinate is the track tile
itself** (i.e. the block at Y-2 relative to the waypoint marker). This matches the existing signal convention
documented in DESIGN_MOVEMENT.md.

The `carts:waypoint` block (`signals.rs: register_waypoint_block`) is the first concrete waypoint block. It carries
a `WaypointConfig` (a single `name: String` field) in its extended data. Direction is encoded in the low 2 bits of
the block variant, identical to how signals encode their facing.

Examples of waypoints include:
* Interlocking entrances (first interlocking signal encountered when approaching an interlocking)
* Interlocking exits (first non-interlocking signal, or dead end, encountered when leaving an interlocking)
* Starting signals for interlocked platform stops
* `carts:waypoint` blocks placed by players as named routing nodes
* Other signals that may arise as part of the routing design.

Note that automatic signals (which are extremely frequent) are explicitly and intentionally NOT waypoints.

### Waypoint data

**Naming:** Waypoints have a human-readable `name` string (e.g. "Spawnville Station"). The naming format is flat for
now; hierarchical naming (e.g. "Spawnville/Platform 1/northbound") will be layered on later once the routing graph
design stabilises.

**Waypoint properties:** For routing functionality, waypoints will need to have distinct types; see the list of example types above.

We have not yet committed to whether waypoint data is cached with a reference to a waypoint, or loaded on the fly
from the actual waypoint block each time. Presumably, it's just as easy to mess up a link between waypoints (by disturbing track) as it is
to change the waypoint itself, so it's probably fine to cache waypoint data along with coordinates when caching adjacencies.