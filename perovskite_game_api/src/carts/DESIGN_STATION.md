# Station Manager — Primitives Inventory & Data-Flow Sketch

This is a **reference document**, not a design. The goal is to enumerate the
engine/game-API primitives that a "station manager" block could compose, and to
sketch the topology of state and messages between carts, signals, waypoints,
the manager block, and the player. Concrete gameplay, scheduling algorithms,
and routing semantics are intentionally deferred to a later pass.

Companion docs: `DESIGN_MOVEMENT.md` (cart/signal mechanics), `DESIGN_ROUTING.md`
(waypoints, network adjacency, routing tables).

---

## 1. Concept recap (so the reader of this doc has a frame)

A *station* is a coherent cluster of cart infrastructure that, from a player's
point of view, behaves as one entity: it holds, dispatches, and announces carts;
shows arrivals/departures; lets the player request transport; sequences loading
hoppers, etc. We want one (or at most: one central + one per platform) **station
manager block** with a friendly in-world facade ("mascot") that internally:

- knows which signals/waypoints/displays belong to it,
- talks to approaching/stopped carts to negotiate dispatch,
- exposes a UI for both server-ops configuration and casual player interaction.

---

## 2. Primitives available today

### 2.1 Block-level primitives (engine / game_api)

| Primitive | Where | What it gives a station block |
|---|---|---|
| `BlockBuilder` + appearance | `blocks.rs` | Visual: mascot can be a custom mesh, AABB stack, or rotating cube. Variants drive bit-toggled geometry for "states" (e.g. "open", "alarm"). |
| `interact_key_handler` + popups | `advanced_block_features` skill | Player UI: configure routes, name station, kick a cart, schedule a manual departure. |
| `simple_data: HashMap<String,String>` | `ExtendedData` | Cheap, schema-less config: nickname, owner, flags. |
| `custom_data` + prost message | `register_proto_serialization_handlers::<T>()` | Rich, versioned state: route tables, timetables, pending dispatch queue, per-platform metadata, learned topology cache. *Cap is practical, not enforced — the chunk is rewritten on every dirty.* |
| `inventories` map | `ExtendedData::inventories` | If a station handles cargo (ticket dispenser, lost-and-found, fuel/tokens), no extra infra needed. |
| `mutate_block_atomically` / `get_block_with_extended_data` | `game_map.rs` | Atomic CAS on (block id + extended data) — the canonical synchronization point. Holds a chunk write lock; keep closures short. |
| `try_mutate_block_atomically` (non-blocking) | same | Useful for the cart coroutine which must not block on chunks. |
| `client_extended_data` + `BlockHoverText` | `make_client_extended_data` | Server-rendered text on the block — directly usable for arrival/departure boards or mascot speech bubbles. **No new client work needed.** |
| Variant bits (12 bits, 4096 states) | `BlockId::variant` | Could expose at-a-glance state ("idle", "boarding", "alert") without an ExtendedData round-trip. |
| Block timers | `perovskite_server/src/game_state/game_map/timers.rs` | Periodic per-block ticks — natural place to run "scan signals", "expire dispatch slot", "garbage-collect stale cart records". |
| `ctx.run_deferred` | `HandlerContext` | Run work *outside* the chunk lock from inside a handler — needed any time the manager wants to touch other blocks (its signals) without deadlocking. |
| Chat messages to players | `Player::send_chat_message` | "Your cart is arriving on platform 2 in 30 seconds." Already used by carts. |

### 2.2 Cart / signal primitives we can either reuse or extend

| Primitive | File:line | Relevance to station |
|---|---|---|
| `SignalConfig` (prost) with `cached_paths`, `forward_paths`/`left_paths`/`right_paths`, `signal_nickname`, route patterns | `signals.rs`, `interlocking.rs` | Existing per-signal config — station can read these to *understand its own topology* without re-deriving it, or write to them to programmatically alter route dispatch. **All three signal block types (automatic, interlocking, starting) register `SignalConfig` handlers and share the same popup**, so `signal_nickname` and friends are readable identically on all of them — only the pathfinder/interlocking semantics differ. |
| `WaypointConfig { name, cached_path }` | `signals.rs:782` | Named graph nodes. The station can use waypoint names as platform/track IDs without inventing a new naming scheme. |
| `RoutingPath` / `RoutingTablePath` / `PathDecision::{Forward,Left,Right}` | `interlocking.rs` | Pre-computed adjacency table mapping every signal decision to a downstream sequence of (waypoints + endpoint). The station can consume these to know "if I tell signal X to go Left, the cart will reach waypoints A, B, then dead-end at C." |
| `scan_interlocking_routes` / `apply_interlocking_routes_to_signals` | `interlocking.rs` | Already implements the multi-route topology walk. A station could re-run this on demand or react to its output. |
| `InterlockingResumeState` + starting-signal "held" behaviour | `interlocking.rs:710`, `signals.rs` | The existing way a cart parks itself at a platform and waits for permission. Station manager flipping a starting signal between proceed/stop *is* the dispatch primitive. |
| `VARIANT_STARTING_HELD` + `VARIANT_RESTRICTIVE_EXTERNAL` | `signals.rs:185-200` | External holds and "cart is here" indicator — readable as a poll signal by the manager. |
| `ScheduledSegment` / `pending_actions` / odometer | `mod.rs` | Cart's planning horizon (~2048 m). The cart already commits decisions ahead of time, so signals it acquires *now* can be turned into estimated arrival times. |
| `parse_signal()` route to `query_interlocking_signal` w/ `ManuallySignalledNoDecision` | `signals.rs`/`interlocking.rs` | Existing manual-dispatch hook: signal asks for a decision; station can be the thing that answers via bus message. |
| `cart_name` (cart's identity used for route pattern matching) | `mod.rs:886`, `query_interlocking_signal` | Already the "addressing scheme" between carts and signals. Stations can extend it (prefixes / hierarchical names) without engine changes. |

### 2.3 Circuits as the message bus

Confirmed via subagent survey of `circuits/`:

- `BusMessage { sender: BlockCoordinate, data: HashMap<String,String> }` — opaque
  string K/V envelope; no size/type enforcement at the API level. Already used
  by interlocking signals to report success/failure.
- `CircuitBlockCallbacks::on_bus_message` — the receive callback.
- `CircuitBlockCallbacks::update_connectivity` + `get_live_connectivities` —
  topology discovery: which "wired" blocks does this block see? Bidirectional.
- `recalculate_wire` (`circuits/wire.rs`) — BFS along wire blocks discovering
  every endpoint connected to a wire; this is the mechanism the station would
  use to enumerate its associated signals.
- `transmit_bus_message`, `transmit_edge` — outbound, called from inside any
  circuit callback. They are *pull-based*: you must be inside a callback to
  emit; there is no "fire from a tokio task" path yet (see §6 wish-list).
- TTL = 256 hops per propagation, fanout cap = 256 neighbors.

This means: **a station block already has, via circuits, a usable mechanism to
discover what signals/waypoints are wired to it and exchange string-keyed
messages with them.** The signal→circuit message that `send_signal_bus_message`
already emits (signal coord, nickname, cart_id, cart_name, outcome, at_signal_now)
is most of what a station needs to react to cart arrivals.

### 2.4 Cart-side hooks (existing & viable)

- The cart coroutine is allowed to `defer_async` to run arbitrary async work
  during `plan_move` (this is how `interlock_cart` runs). Anything that needs
  to consult or notify a station block from inside the cart's planning step
  can be done the same way.
- The cart already maintains `cleared_signals` (a per-scan cache) and
  `interlocking_resume_state`. A station-aware extension could piggyback on
  these without redesign — e.g., `precomputed_steps` already records signals
  the cart will pass and at which odometer offsets, which is exactly the data
  needed for ETA computation.

### 2.5 Test / dev primitives

- `TestFixture` + `configure_default_game` (per `DESIGN_ROUTING.md` testing
  conventions) — straight ZPlus tracks at Y=64, variant=0/2 for facing.
- `block_testing` skill — covers timer-driven tests and inventory tests.
- Real game map; no mocking, so the manager can be exercised end-to-end with
  actual signals/carts at unit-test cost.

---

## 3. Where state can live (storage matrix)

| Where | Persistence | Cost to read/write | Good for | Bad for |
|---|---|---|---|---|
| Station block `custom_data` (prost) | Persistent w/ chunk | Cheap on hit, requires the chunk to be loaded | Per-station authoritative state: routes, timetables, learned topology cache, dispatch queue | Data needed when station chunk is unloaded |
| Station block `simple_data` | Persistent | Cheap | Nickname, ownership, simple flags | Anything structured |
| Station block variant bits | Persistent, inline | Free | Display state ("idle/boarding/alert"), 4096 enum values | Anything with structure |
| Signal/waypoint extended data (`SignalConfig`, `WaypointConfig`) | Persistent | Cheap | Per-signal hints the station has *assigned* (e.g. "this starting signal belongs to platform 2"); nickname; cached paths | Cross-signal coordination |
| Cart in-memory `CartCoroutine` fields | Per-cart, lost on despawn | Free | Cart's local view of "where am I going" | Anything multi-cart |
| Cart name (string) | Per-cart, sticky | Free | Routing intent, station addressing | Anything wide |
| Global `GameStateExtension` (`CartsGameBuilderExtension` pattern) | Process lifetime | Cheap | Block ID handles, registry, per-server caches like "global station index" | Anything that should survive restart and not be re-derived |
| Database (`KeySpace::Plugin`) | Persistent | Modest | A future station/timetable registry not tied to a specific block | Hot-path reads |

A reasonable default: **authoritative station config in the manager block's
custom_data; ephemeral cart↔station handoff via circuit bus messages; learned
topology cached in custom_data with manual or scan-triggered refresh** (mirrors
how `cached_paths` is already managed on signals).

---

## 4. Communication channels & their properties

```
                   ┌─────────────────────┐
                   │  Player (UI/chat)   │
                   └──────────┬──────────┘
                              │ popups, chat, interact key
                              ▼
                   ┌─────────────────────┐
       circuit     │  Station Manager    │ block_timer ticks
       bus msgs    │  Block(s)           │ scheduled async tasks
   ◄──────────────►│  - custom_data      │
                   │  - variant (UI)     │
                   │  - inventory (opt)  │
                   └──────┬───┬──────────┘
                          │   │
              circuit bus │   │ map writes (mutate_block_atomically)
              + map polls │   │ (SignalConfig flags, VARIANT_STARTING_HELD,
                          │   │  VARIANT_RESTRICTIVE_EXTERNAL)
                          ▼   ▼
                   ┌──────────────────────┐
                   │  Signals / waypoints │
                   │  (existing)          │
                   └──────┬───────────────┘
                          │ existing acquisition protocol +
                          │ send_signal_bus_message on success/fail
                          ▼
                   ┌──────────────────────┐
                   │  Cart coroutine      │
                   │  (deferred async ok) │
                   └──────────────────────┘
```

### 4.1 Manager ↔ signals/waypoints

- **Outbound**: direct map writes via `mutate_block_atomically`
  (e.g., flip `VARIANT_RESTRICTIVE_EXTERNAL`, set starting signal proceed/stop,
  rewrite `SignalConfig::*_paths` or `signal_nickname`). Discovery either via
  wired circuit topology, or via cached "known children" list in the manager's
  custom_data.
- **Inbound** (today): the interlocking system *already* emits a bus message on
  every interlocking acquisition attempt; if the manager is wired into that
  circuit, it gets cart arrivals "for free".

### 4.2 Manager ↔ carts

Three usable patterns; pick per use-case:

1. **Manager pulls from cart's footprint in signals.** When a cart acquires the
   starting signal, the existing bus message tells the manager "cart X has
   stopped at platform 2". No cart-side changes needed.
2. **Cart polls manager on each wakeup.** The cart coroutine already has
   ample power to defer async work and read arbitrary blocks. On every
   `plan_move`, or when it next encounters a "near-station" interlocking signal,
   it can read the upcoming station block and react to whatever instruction is
   queued for `cart_name == self.cart_name`. Acceptable freshness: tens of ms.
3. **Manager sets state on signals; cart sees it during normal scan.** The
   existing route-pattern / `ManuallySignalledNoDecision` mechanism is exactly
   this: manager writes intent to a signal; cart reads it during scan; no new
   plumbing required.

Pattern (3) is the cheapest, (1) gives the manager visibility, (2) gives the
richest cart-side reactivity. None require new engine primitives — but a true
message-delivery-to-cart channel is in the "wish-list" below.

### 4.3 Manager ↔ player

- Interact key popup for ops/admin (route tables, naming, manual override).
- `make_client_extended_data` + `BlockHoverText` for at-a-glance info on the block
  face (next arrival, current dispatch).
- Chat messages on cart boarding/alighting; can be triggered from either the
  manager's bus-message handler or the cart's existing player attachment hooks.
- Variant-bit visual states for an animated mascot.

### 4.4 Manager ↔ manager (multi-block stations)

If a station is split across "central manager + one per platform":

- Discovery via the circuit wire scan (same as signals).
- Cross-block consistency via small messages, with one block treated as the
  authoritative owner; followers store a back-pointer in their `custom_data`.
- Cold-load ordering caveat: chunks can load independently, so a "is my parent
  still alive" check on first tick / on each timer is needed (mirrors how the
  switch postprocessor resets stale state at cold load).

---

## 5. Data-flow sketches (intentionally high-level)

### 5.1 Cart arrives at a station platform

1. Cart's track scan reaches the platform's interlocking signal.
2. `interlock_cart` runs the existing pathfinder; if the platform's starting
   signal is in `stop` mode, the cart's planned path terminates *at* the
   starting signal (existing behaviour).
3. `send_signal_bus_message` fires from the interlocking signal — the **station
   manager, wired into that circuit, receives the message** and learns
   `cart_id`, `cart_name`, signal coord, nickname, "at_signal_now" flag.
4. Manager updates its `custom_data` (or `simple_data`) to record arrival.
5. Manager optionally pushes a status to its display via
   `make_client_extended_data` (triggered by the next chunk read, or proactively
   via a block update).
6. Player presses interact, gets boarding popup. Or the manager dispatches
   automatically per its timetable.

### 5.2 Manager dispatches a held cart

1. Manager decides "platform 2 leaves now" (timer tick, player action, or
   inbound message from a central manager).
2. Manager flips the corresponding starting signal's mode from `stop` to
   `proceed` via `mutate_block_atomically` on the signal block — clearing
   `VARIANT_RESTRICTIVE_EXTERNAL` (or whichever bit we wire up). *No cart-side
   change needed; the existing `interlocking_resume_state` polling already
   wakes the cart.*
3. Cart's existing resume code re-runs `interlock_cart`, the pathfinder
   commits, cart leaves.
4. Bus message on success — manager records "platform 2 cleared".

### 5.3 Manager learns its own topology

1. On initial setup, player wires station block into the local signal/waypoint
   circuit network.
2. Manager runs its own `update_connectivity` (circuit callback) → calls
   `get_live_connectivities`/wire scan → enumerates wired signal & waypoint
   block coordinates.
3. For each discovered signal/waypoint, manager reads `SignalConfig` /
   `WaypointConfig` (incl. `cached_paths` / `cached_path`) to derive the local
   routing graph.
4. Manager stores the derived "platform map" in its own `custom_data` so it
   doesn't have to rescan every tick. Manual "rescan" button in the popup,
   exactly like the existing "Scan Interlocking Routes" pattern.

### 5.4 Player asks for transport ("take me to Spawnville")

1. Player interacts with manager → popup with a list of *reachable destination
   waypoint names* (derived from cached topology + routing tables on
   downstream interlocking signals).
2. Manager picks (or reserves) a cart, sets its `cart_name` to a routing
   pattern matching the destination, flips the relevant starting signal to
   proceed, attaches player to the cart entity.

---

## 6. Gaps / engine primitives worth adding

These are **not blockers** — gameplay can be designed around them — but flagging
for prioritization:

1. **Message → cart delivery primitive.** Today carts only learn about external
   state by re-reading blocks on the next scan/wakeup. A typed mailbox on the
   `CartCoroutine` (drained at the top of `plan_move`) would let a manager
   push an event ("change destination, you've been re-routed") without
   requiring the cart to poll. Workaround: store the message in the next
   station block; have the cart read it on each scan — but this couples the
   message to a physical location and a specific moment.

2. **Stable cart identity beyond `cart_name`.** `cart_name` doubles as both
   routing intent and identity. A separate `cart_handle` (entity ID is fine,
   but accessible to circuits / handlers without the entity-system gymnastics
   already needed today) would let the manager track a specific cart through
   re-naming.

3. **Bus-message emission from outside a circuit callback.** Currently only
   `on_bus_message`/`on_incoming_edge`/`update_connectivity` can call
   `transmit_bus_message`. A block_timer or async task wanting to push an
   update has to first synthesize a fake edge to itself, or schedule a
   `run_deferred` that goes through a circuit handler. A direct
   "emit from anywhere with a HandlerContext" path would simplify station
   timers, scheduled announcements, etc.

4. **Larger / typed BusMessage payload.** `HashMap<String, String>` is fine for
   ad-hoc fields, but anything structured (a list of upcoming arrivals, a
   timetable) ends up reinventing a serialization layer. A `Vec<u8>` payload
   slot (or attached prost message) would help.

5. **Topology-change hook on track/signal placement.** Today caches must be
   manually refreshed (see `DESIGN_ROUTING.md` "Caches are intentionally not
   automatically invalidated"). A station that auto-resyncs would need either
   a hook, or periodic timer-driven re-scans. The latter is sufficient for
   a first pass.

6. **Block→block reference type that survives renames.** When the manager
   stores "my platform 2 starting signal is at (x,y,z)", it has no way to
   know if that block was dug and replaced with something else. A weak
   reference (coord + expected block id, with mismatch reporting) would
   reduce defensive coding.

7. **Cart ETA query.** The cart knows its `scheduled_segments` and
   `precomputed_steps`. Surfacing "ETA to coord C" via an `EntityManager`
   query would let displays show arrival countdowns without each manager
   reimplementing the math.

---

## 7. Open questions deferred to next iteration

Listed here so the next-context run knows what we *didn't* pin down:

- **Granularity**: one block per station vs. one central + one per platform.
  Primitives support both; gameplay choice.
- **Naming scheme**: flat (`"Spawnville/P2"`) vs. structured prost ID. The
  existing `cart_name` wildcard matcher is flat; staying flat avoids
  inventing a parallel scheme prematurely.
- **Dispatch policy**: timetable / on-demand / hybrid — independent of every
  primitive above.
- **Multi-station coordination** (network-wide schedules): would need either
  a global registry (DB-backed plugin keyspace) or chained circuit links.
- **Cargo / passenger model**: blocked on cart inventory primitive
  (`DESIGN_MOVEMENT.md` known limitations).
- **Trust / permissions**: who can flip a station's dispatch? Today
  placer-tracking + `simple_data` "owner" mirrors the locked-chest pattern.

---

## 8. Baseline station test plan (pre-station-block)

A "station" today, with no manager block implemented, is just **tracks + interlocking
signals + (optionally) a starting signal**. Cart behaviour at such a layout is fully
determined by primitives that already exist (`signals.rs`, `interlocking.rs`, the cart
coroutine in `mod.rs`). This section enumerates the *station-level* behaviours we want
to lock down with tests **before** we begin adding any station-block logic, so that
later work has a known-good baseline to regress against.

### 8.1 Out of scope (covered elsewhere)

These already have coverage in `interlocking_test.rs` and should **not** be duplicated:

- `single_pathfind_attempt` acquiring/rolling back routes through an interlocking.
- `scan_interlocking_routes` enumerating distinct paths through a junction.
- `apply_interlocking_routes_to_signals` writing routing tables that match a fresh scan.
- The mapgen + template-load smoke test for the `stations_fork` world
  (`station_tests.rs::stations_fork_world_loads`).

The baseline tests should treat the pathfinder and routing-table machinery as a black
box and assert only on cart-observable / station-observable outcomes.

### 8.2 Fixture

All tests in this section reuse the existing `stations_fork` scenario
(`station_tests.rs`):

- Mapgen produces a four-track main line at `z < -64` (`x ∈ {-2,-1,0,1}`) and two
  two-track branches at `z ≥ 64` (`x ∈ {-16,-15, 15,16}`).
- The hand-crafted template `testdata/station_fork.test_geometry` is loaded at
  `FORK_STATION_ORIGIN` (`-16, -1, -64`), filling the `-64 ≤ z < 64` slab with the
  station throat: switches, interlocking signals, at least one starting signal per
  platform, and waypoints labelling each branch.
- Tracks at `y = 0`; signals at `y = 2`.

We add at most one additional hand-edited template variant if the existing fork doesn't
already include the platform-side starting signals and the per-platform route patterns;
the dev-server `stations_fork` scenario lets us author it interactively and re-export.

### 8.3 Test inventory

Each test is intentionally *narrow* — it asserts one station-level behaviour, using the
shared fixture. Numbering is for cross-referencing in commits/PRs only.

#### station_fork test description

Everything takes place with tracks on Y=0 and signals/waypoints at Y=2.

`station_fork`'s scenario begins with a single junction station, with a four-track line
in the -Z direction, and two two-track branches in the +Z direction.

The four-track line has two interlocking signals that correspond to entering the
station in the Z+ direction, nicknamed mainline_outer_enter and mainline_inner_enter.
There are also automatic signals marking the exit of the station, nicknamed mainline_outer_exit
and mainline_inner_exit, with travel in the Z- direction.

The two two-track branches have signals marked east_enter, east_exit, west_enter, west_exit;
these tracks have corresponding platforms with starting signals, called stop_east_zplus, stop_west_zplus,
stop_east_zminus, stop_west_zminus. Effectively, the platforms are assigned based on the
Z+ side (split east vs west) line taken by the cart; both zminus platforms can reach either
of the two zminus tracks on the four-track line, and either of the two zplus tracks on the
four-track line can reach either of the zplus platforms.

#### Future testing work on station_fork

station_fork has infinitely extending tracks, so it will be possible to add extra stations along
the lines running from the central junction station; on the four-track line it is also possible to
set up either local-only or all-track stations in the future. This will be done after we identify
specific test scenarios that require multiple stations.

#### T1. Fork template world is sane for station tests

**Purpose:** belt-and-suspenders that the fixture exposes the inputs the rest of the
tests assume.

**Setup:** `configure_stations_fork` + `load_fork_station`.

**Assertions:**
- For each expected platform (one platform per branch x-column, identified by waypoint
  name `"west_outer" / "west_inner" / "east_inner" / "east_outer"` — names TBD when
  the template is finalized), there is a starting signal block at the documented
  `(x, 2, z)` and an interlocking signal upstream of it on the approach side.
- The four mainline approach tracks (`x ∈ {-2,-1,0,1}` at `z = -65`) are rail tiles.
- `scan_interlocking_routes` starting from each mainline approach reaches **at least
  one** path terminating at each branch's starting signal. (Counts, not contents:
  contents are interlocking-graph territory.)

This test is what catches "we edited the template and broke the fixture" before any
subsequent test reports a confusing failure.

#### T2. Cart parks at a starting signal set to STOP

**Purpose:** the most fundamental station primitive — a held cart actually holds.

**Setup:** fork fixture, all platform starting signals in stop mode (the default after
mapgen, assuming we don't pre-clear them). Spawn one cart on a mainline approach with a
`cart_name` that routes it to a specific platform.

**Assertions:**
- After advancing simulated time by N ticks (enough for the cart to traverse the
  approach + throat at full speed), the cart's `vec_coord()` is within one tile of the
  target starting signal and its velocity is `~0`.
- `VARIANT_STARTING_HELD` is set on the target starting signal (this is the bit the
  station manager will eventually poll).
- The cart's `interlocking_resume_state` (read via whichever test hook we add to
  inspect cart coroutine state — see §8.5) is `Some(_)`.
- No `VARIANT_PERMISSIVE` bits remain set on the other platform starting signals
  (negative check: this cart did not contaminate sibling platforms).

#### T3. Cart resumes after starting signal cleared

**Purpose:** the dispatch primitive — flipping a starting signal releases a held cart.

**Setup:** as T2, run until the cart is held at the platform.

**Action:** atomically clear `VARIANT_STARTING_HELD` *and* set the starting signal to
proceed mode via `mutate_block_atomically` (one CAS — the same operation a future
station manager would perform). Advance time.

**Assertions:**
- Cart's `vec_coord()` advances past the starting signal within N more ticks.
- Eventually cart reaches end-of-track (or the next holding point on the branch) and
  remains stationary there with no errors logged.
- Starting signal's variant returns to a non-held idle state (exact bit pattern TBD
  by what `signals.rs` actually leaves behind once the cart has cleared).

#### T4. External hold (`VARIANT_RESTRICTIVE_EXTERNAL`) blocks dispatch

**Purpose:** the bit that a station manager will eventually use to gate proceed-mode
starting signals must actually gate them.

**Setup:** fork fixture, set the target platform's starting signal to **proceed** but
also set `VARIANT_RESTRICTIVE_EXTERNAL`. Spawn cart.

**Assertions:**
- Cart parks at the starting signal exactly as in T2.
- Clearing `VARIANT_RESTRICTIVE_EXTERNAL` (without touching the proceed/stop bit)
  releases the cart, mirroring T3.

This is the most important test for the future station manager: it freezes the contract
that flipping this one bit is sufficient to control dispatch, without the manager having
to touch the rest of the variant.

#### T5. Two carts contend for the same platform

**Purpose:** the interlocking already serialises access to a single platform; verify
this from the station's perspective.

**Setup:** fork fixture, target platform held in stop. Spawn cart A on one mainline
approach, cart B on another, **both with `cart_name` patterns selecting the same
platform**, with B starting later or further upstream so A is guaranteed to reach the
throat first.

**Assertions:**
- A reaches the starting signal and is held (as in T2).
- B is held *upstream* of the throat — either at the last interlocking signal before
  the conflicting route, or stopped earlier — but **not** colocated with A.
- Releasing A (clearing the starting signal) eventually leads to B acquiring the
  platform and parking at the same starting signal.
- At no point are both carts inside the throat simultaneously (poll positions across
  several ticks; assert distance ≥ some safe threshold, or that they share no track
  tile).

#### T6. Convergent carts route to different platforms

**Purpose:** the routing-pattern mechanism (cart_name → interlocking decision) routes
each cart to its intended platform when both are simultaneously in flight.

**Setup:** fork fixture, all platform starting signals held in stop. Spawn cart A
named for the west-outer platform and cart B named for the east-outer platform on
distinct mainline approaches, simultaneously.

**Assertions:**
- After enough ticks for both to traverse the throat: A parks at west-outer's starting
  signal, B parks at east-outer's. Neither cart ends up on the wrong branch.
- Both `VARIANT_STARTING_HELD` bits are set on their respective platforms.
- The throat's interlocking signals have settled back to idle (no stuck `PRELOCKED`).

#### T7. Cart name not matched by any route

**Purpose:** a cart whose `cart_name` matches no platform route — i.e., a player tries
to "go to a place that doesn't exist" — does not silently take a wrong branch and does
not deadlock the interlocking.

**Setup:** fork fixture, spawn one cart with `cart_name = "nonexistent_destination"`.

**Assertions:**
- Cart either (a) stops cleanly at the first interlocking signal that has no matching
  route (today's `ManuallySignalledNoDecision` path), or (b) takes the documented
  default branch — whichever the *current* code does. The test pins the behaviour
  rather than prescribing it; if we later change it, this test changes with it.
- No other cart spawned afterwards on a *valid* route is blocked by the stranded cart
  (assuming the stranded cart is on a branch the other doesn't need; this also
  documents the "stranded cart blocks the throat" failure mode if that's what we see).

#### T8. Cold-load preserves starting-signal hold state

**Purpose:** the starting-signal variant is persistent; switches are not (existing
postprocessor resets them). Make sure the station-relevant bits survive a fixture
restart so a future station manager can rely on it.

**Setup:** fork fixture; cart spawned, held at platform, world is then shut down and
restarted using the existing fixture restart mechanism (if available — otherwise
flagged as a follow-up).

**Assertions:**
- After restart, the starting signal still has `VARIANT_STARTING_HELD` (or whatever
  bit pattern we declare to be the persistent "held" indicator).
- All switches in the throat have been reset to `switch_unset` by the existing
  postprocessor (negative check on residue from the locked transaction).

If fixture restart isn't supported in `TestFixture` today, this test is deferred and
noted as a wish-list item — the cold-load reset logic is already covered indirectly by
the existing switch postprocessor unit tests.

#### T9. Multiple sequential dispatches from one platform

**Purpose:** the platform isn't "stuck" after one dispatch — repeated hold/release
cycles work.

**Setup:** fork fixture, single platform, single mainline approach. Loop 3+ times:
spawn cart → wait for park → release → wait for clear → assert signal reverted to a
re-armable state.

**Assertions:**
- Each cart parks and is released without manual intervention beyond flipping the
  documented bit(s).
- No bit residue (`PRELOCKED`, `STARTING_HELD`, etc.) on the starting signal between
  iterations.

### 8.4 Out-of-scope for the baseline pass

Explicit non-goals, so we don't gold-plate:

- Anything that requires a station *block* (UI, custom_data routes, displays, chat
  announcements, circuit bus reception).
- Timetable / schedule semantics.
- ETA computation.
- Multi-station coordination.
- Player boarding/alighting (separate concern, depends on the entity-attach paths).
- Any test that exists only to exercise the interlocking graph algorithm.

### 8.5 Likely small helpers we'll need to add (test-support only)

These do **not** belong in production code paths and should live in `#[cfg(any(test,
feature = "test-support"))]` modules:

1. **`wait_until_cart_parked(fixture, cart_id, near: BlockCoordinate, max_ticks)`** —
   advances ticks (or sleeps + polls, matching whatever the fixture uses today) until
   the cart's reported position is within a small radius of `near` and velocity is
   approximately zero, or fails the test after `max_ticks`.
2. **`set_starting_signal(fixture, coord, mode: enum {Stop, ProceedHeld, Proceed})`**
   — wraps `mutate_block_atomically` with the right variant-bit recipe so tests don't
   each open-code the bit math. Mirrors the operation a future station manager will
   perform.
3. **`read_signal_variant(fixture, coord) -> u16`** — convenience for asserting on
   variant bits without bespoke `get_block_with_extended_data` closures.
4. **`platform_coords(fixture) -> &'static [(name, starting_signal_coord, target_track_coord)]`**
   — a single source of truth for "where the platforms are" in the fork template, so
   tests reference platforms by symbolic name instead of hard-coded `(x, y, z)`.

Helpers 1 and 4 are the highest-leverage; 2 and 3 fall out trivially.

### 8.6 Suggested ordering

A reasonable sequence for actually implementing the suite (one PR per group is fine):

1. T1 (fixture sanity) + helpers 3, 4.
2. T2 + helper 1.
3. T3 + helper 2.
4. T4 (locks the contract for the future station manager).
5. T5–T7 (multi-cart and routing semantics).
6. T8 (cold-load; defer if fixture restart isn't ready).
7. T9 (repeated-cycle smoke).

Once this suite is green and stable, we have a regression net: subsequent station-block
work can be evaluated by "does it break any of T1–T9, and does it require any new
station-level tests beyond them?"

---

## 9. Suggested next investigation passes (for future runs)

1. Pick a *minimum viable* station behaviour (e.g., "single platform that
   holds, lets a player board, and dispatches on interact") and walk through
   exactly which existing primitives suffice. Identify the first real gap.
2. Prototype the circuit-wire topology scan from a stub station block; verify
   that it can enumerate the signals and waypoints of `interlocking_test`'s
   fixture template without engine changes.
3. Decide whether the cart needs a real inbound-message channel, or whether
   "manager writes intent into the next signal/waypoint the cart will read"
   suffices for all near-term use cases.
4. Iterate on multi-block (central + platform) consistency story with a small
   working prototype, before committing to a config schema.
