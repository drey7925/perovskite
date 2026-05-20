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

* **`find_adjacency` returns `Result<(AdjacencyHit, ScanState)>`.** The `ScanState` at termination is the second
  tuple element and is kept separate from `AdjacencyHit` so the hit can be serialized/cached without coupling it
  to scanner internals. Map errors are propagated as `Err` rather than silently treated as end-of-track or air.
  Callers must `.or_fail()?` (in tests) or `?`-propagate (in production code).

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

### Scanning an interlocking

This is an extension of finding an adjacency that also must consider different routes possible in an interlocking.

A scan logically takes a starting scan state, at an incoming interlocking signal, and produces a collection of structs, each containing:

* An AdjacencyHit for each possible end of scan (dead ends, automatic signals)
* A smallvec of all waypoints and starting signals (encountered in forward order with passing signal_rotation_ok) as AdjacencyHit that
  were encountered along the path

Then deduped based on both the adjacency hit and the list of waypoints encountered.

New adjacency hit kinds:
* EndOfInterlockingSignal
* StartingSignal

We can do this by BFS/DFS with a few considerations:

* Entirely ignore resume logic when copying from the existing pathfinder. It is not even an input here.
* We're just passively scanning, do not acquire signals or switches or mutate any blocks yet.
* We only consider interlocking signals where signal_rotation_ok check passes
* Disregard signals being acquired, contended, etc.
* We have to track whether we can actually diverge (i.e. when we hit an interlocking signal of any kind, set left_pending, right_pending
  as true, and take the first left branch and one right branch we see; the first signal implicit in the start of the scan can also route left/right, so left_pending and right_pending should have initial value of true).
      * If we do diverge, left_pending and right_pending go to false until we hit another signal.
* We have to verify that there is actually a switch block of any kind (including a set one) when we see a Some value for get_switch_length()
* Like in the existing code, encountering an interlocking signal backwards means an invalid path - not even end of track - it doesn't contribute
  a possible end state, and shouldn't go into the collection.
* Loop prevention is important (detecting if a signal has been seen before). A naive FxHashSet of encountered coordinate + direction is good enough as long as it's properly cloned when branching. We do not get this for free - the existing function did get it for free because it mutated the map, and we are not doing that here.

Essentially, a rework/copy of single_pathfind_attempt to not mutate or acquire, track what left/right moves are possible (rather than were actually called for), and collect all outputs into a collection rather than just acquiring one path.

Test notes:
* On our existing test interlocking, starting from either of the Z+ tracks (x = 2 or 3) should get us three outcomes (automatic signals at x=2, x=3, and a dead end on x=6). Starting from the Z- tracks (z = 4 ot 5) should also get us 3 outcomes (automatic signals at x=4, x=5, and a dead end on x=6).
* Currently, the template is missing backwards-facing automatic signals at wrong-way exits, so it's OK if we get extra results in addition to the above.
* A better template with loops and other oddities will be added as part of later work.