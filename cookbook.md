# Perovskite Modding Cookbook

This document catalogs the emergent techniques available in `perovskite_game_api` and the underlying `perovskite_server` primitives. It focuses on how these tools *compose* into complex, compelling gameplay systems — tech trees, electric grids, logistics networks, automated machines, and more.

The source map files in `.claude/` describe *what* each module contains. This cookbook describes *how to combine* those modules into things that don't exist yet.

---

## Table of Contents

1. [Block State Storage](#1-block-state-storage)
2. [Variant Bits: Compact State Encoding](#2-variant-bits-compact-state-encoding)
3. [Block Groups: O(1) Type Checking](#3-block-groups-o1-type-checking)
4. [Event Handlers: Reacting to Player Actions](#4-event-handlers-reacting-to-player-actions)
5. [Timers: Recurring Block Updates](#5-timers-recurring-block-updates)
6. [Block Swapping: State Machines Without Data](#6-block-swapping-state-machines-without-data)
7. [Neighbor Inspection: Reading the Environment](#7-neighbor-inspection-reading-the-environment)
8. [BFS Connectivity: Finding Connected Components](#8-bfs-connectivity-finding-connected-components)
9. [Signal Propagation: Nets and Busses](#9-signal-propagation-nets-and-busses)
10. [Axis-Aligned Boxes: Directional Visuals](#10-axis-aligned-boxes-directional-visuals)
11. [Multi-Block Machines: Sense/Act Coordination](#11-multi-block-machines-senseact-coordination)
12. [Atomic Group Movement](#12-atomic-group-movement)
13. [Bulk Writes and Undo](#13-bulk-writes-and-undo)
14. [Two-Click Tools: The Autobuilder Trait](#14-two-click-tools-the-autobuilder-trait)
15. [GameState Extensions: Plugin-Wide Shared State](#15-gamestate-extensions-plugin-wide-shared-state)
16. [Entity Coroutines: Autonomous Behavior](#16-entity-coroutines-autonomous-behavior)
17. [Physics-Based Entity Movement](#17-physics-based-entity-movement)
18. [Inventory System: Containers and Crafting](#18-inventory-system-containers-and-crafting)
19. [Popup UI: Server-Driven Dialogs](#19-popup-ui-server-driven-dialogs)
20. [Terrain Following and Path Planning](#20-terrain-following-and-path-planning)
21. [Composing a Tech Mod: Putting It Together](#21-composing-a-tech-mod-putting-it-together)

---

## 1. Block State Storage

Every block can carry state beyond its type identity. There are three layers, all living in `ExtendedData` (`perovskite_server/src/game_state/blocks.rs:104–151`):

### Layer 1: Protobuf Custom Data

For structured state that needs schema evolution and binary compactness.

```rust
// 1. Define a protobuf message in your proto file
message FurnaceState {
    uint32 fuel_remaining = 1;
    uint32 smelt_progress = 2;
    bool is_lit = 3;
}

// 2. Register serialization handlers during block registration
block.register_proto_serialization_handlers::<FurnaceState>();

// 3. In any handler, read and modify
ctx.game_map().mutate_block_atomically(coord, |_block, ext| {
    let state = ext.get_or_insert_default()
        .custom_data
        .downcast_mut::<FurnaceState>()
        .expect("furnace state");
    state.fuel_remaining -= 1;
    Ok(ControlFlow::Continue(()))
})?;
```

**Example in codebase:** Microcontroller stores script AST, port registers, memory arrays, and a circular bus message buffer — all in one `MicrocontrollerExtendedData` struct. See `perovskite_game_api/src/circuits/gates/microcontroller.rs:755–858`.

### Layer 2: Key-Value Simple Data

For lightweight, human-readable metadata that doesn't need a schema.

```rust
// Write
ext.simple_data.insert("owner".to_string(), player_name.to_string());
ext.simple_data.insert("temp".to_string(), "800".to_string());

// Read
let owner = ext.simple_data.get("owner").map(String::as_str);
let temp: u32 = ext.simple_data.get("temp")
    .and_then(|s| s.parse().ok())
    .unwrap_or(0);
```

Use for: furnace temperature, lock owner, last-touched timestamp, configuration flags.

### Layer 3: Nested Inventories

Blocks can host named `Inventory` objects directly inside their `ExtendedData`. These are accessed by name and created on demand.

```rust
// Get-or-create a 9×3 inventory named "input"
let inv = ext.get_or_insert_default()
    .inventory_mut("input".to_string(), (9, 3));
let leftover = inv.try_insert(item_stack);
```

Multiple inventories per block are fine: a furnace might have `"input"`, `"fuel"`, and `"output"`. See machine inventory wiring in `perovskite_game_api/src/autobuild/machines.rs:849–897`.

### The Dirty Bit

**Critical:** `ExtendedData` only persists if marked dirty. Mutation via `DerefMut` sets this automatically, but if you mutate through an `Arc<Mutex<...>>` stored inside `custom_data`, you must call `ext.set_dirty()` manually or your changes will be lost on chunk unload.

---

## 2. Variant Bits: Compact State Encoding

Every `BlockId` carries a 12-bit variant field. These bits are free for any block to use, and they're cheap — no heap allocation, no serialization roundtrip. The key is that different bit ranges can serve different purposes simultaneously.

### Rotation (Bits 0–1)

`CubeAppearanceBuilder::set_rotate_laterally()` uses bits 0–1 for 90° NESW rotation. The engine auto-rotates the mesh. Use `rotate_nesw_azimuth_to_variant(player_azimuth)` (`perovskite_game_api/src/blocks.rs:1305–1315`) to set on placement.

### Connectivity Masks (Bits 0–7, circuits pattern)

Wire uses 8 bits as a bitmask of which directions the wire connects:

```rust
const VARIANT_XPLUS:       u16 = 1;   // connects right
const VARIANT_XMINUS:      u16 = 2;   // connects left
const VARIANT_ZPLUS:       u16 = 4;   // connects forward
const VARIANT_ZMINUS:      u16 = 8;   // connects backward
const VARIANT_XPLUS_ABOVE: u16 = 16;  // connects right + up
// ... etc
```

On neighbor update, OR together all connection bits and call `block_id.with_variant_unchecked(combined_mask)`. The visual appearance then conditionally renders arms based on which bits are set. See `perovskite_game_api/src/circuits/wire.rs:28–56`.

**Reuse:** Pipe networks, power conduits, data cables, fluid channels — any system where a block must visually and logically know which of its 6 faces connect to a neighbor of the same type.

### Accumulation Counter (Bits 2–11)

The farming module uses upper variant bits as an accumulator: each timer tick adds a fixed amount, and when the counter trips a threshold, a state transition fires. This implements probabilistic thresholds deterministically.

```rust
pub struct InteractionAccumulator {
    pub add: u8,   // increment per event
    pub trip: u8,  // threshold to trigger
}

impl InteractionAccumulator {
    pub fn apply(&self, id: BlockId) -> (BlockId, bool) {
        let old_acc = id.variant() >> 2;
        let new_variant = id.variant() + (4 * self.add as u16);
        let new_id = id.with_variant_unchecked(
            new_variant.min(BLOCK_VARIANT_MASK as u16)
        );
        (new_id, old_acc.saturating_add(self.add as u16) >= self.trip as u16)
    }
}
```

**Reuse:** Fatigue/durability for blocks that crack under repeated hits, heat buildup in machines, charge accumulation.

### Placer Tracking (Bits 10–11)

Call `.set_track_placer()` on a `BlockBuilder` to have the place handler automatically encode whether a player or autobuild placed the block. Autobuild tools read these bits to avoid overwriting player work.

```rust
const VARIANT_PLACER_PLAYER:   u16 = 0x800;
const VARIANT_PLACED_AUTOBUILD: u16 = 0x400;
```

### Rail/Slope Encoding (tracks.rs)

Carts compress track geometry into 12 bits: atlas X/Y (tile appearance), rotation (0–3), flip_x, reverse-scan direction, diverging-route flag, slope encoding. See `perovskite_game_api/src/carts/tracks.rs:44–226`. This demonstrates that 12 bits is ample for most block state — rotation + 2–3 mode bits + an accumulator all fit.

### Visual features

The client renderer interprets variant bits in several ways depending on the block's render info:

**`CubeVariantEffect` (set on `CubeRenderInfo`):**
- `RotateNesw` — bits 0–1 rotate the whole block mesh in 90° steps: 0 = front toward Z−, 1 = X−, 2 = Z+, 3 = X+. Top/bottom textures rotate to match. (`block_renderer.rs:923–927`)
- `Liquid` — variant 0–7 encodes flow height; variant 0xfff is a source block (rendered same as 7). The renderer samples the same variant field from all 4 horizontal neighbors and computes a per-corner height by taking the max of the four surrounding variants, producing smooth inter-block height blending. A gradient normal is derived from the height differences and encoded on the top face. (`block_renderer.rs:1415–1488`)
- `CubeVariantHeight` — same height encoding (variant 0–7 → height) but without neighbor sampling, so each block's top surface is flat and unconnected to neighbors. The top face is forced-visible when height < full. (`block_renderer.rs:1389–1408`)

**`AxisAlignedBox` per-box controls:**
- `variant_mask` — each sub-box has an independent mask matched against the block's *render selector* (`perovskite_core::render_selector`): bits [0,12) are the server-sent variant, bits [12,18) are neighbor-presence bits computed client-side (X+, X−, Y+, Y−, Z+, Z−; a neighbor counts if it is solid opaque or the same base block). The box is skipped entirely if `selector & mask == 0` (a zero mask means always render). Variant bits drive wire arms and other server-driven multi-part blocks; neighbor bits drive fences and similar auto-connecting geometry with no server involvement. (`blocks.proto:119`, `block_renderer.rs`, `render_selector.rs`)
- `rotation` — `NESW` applies `variant % 4` to rotate that individual box around Y, independently of other boxes in the same block. (`block_renderer.rs:1054–1057`)
- `top_slope_x / top_slope_z` / `bottom_slope_x / bottom_slope_z` — static (not variant-driven) linear warps applied to the top or bottom face vertices: `y_adjusted = y + slope_x * x + slope_z * z`. Used by sloped rail tiles. (`blocks.proto:115–124`, `block_renderer.rs:1272–1273`)

**`DynamicCrop` on `TextureReference`:**
- Selects a cell from a sprite-sheet atlas based on variant bits. `x_selector_bits` masks the variant to get a column index (0 to `x_cells-1`); `y_selector_bits` similarly for row. Bit ranges must be contiguous. (`render.proto:104–123`, `block_renderer.rs:457–485`)
- `flip_x_bit` / `flip_y_bit` mirror the selected cell if the corresponding variant bit is set; `extra_flip_x/y` inverts the polarity.
- Example use: a block with 4 horizontal variants and 2 vertical variants uses `x_selector_bits = 0b0011, x_cells = 4, y_selector_bits = 0b0100, y_cells = 2` to pick one of 8 tiles from a 4×2 sprite sheet without defining separate block types.

**`TextureTransform` (static, not variant-driven):**
- A per-`TextureReference` flip or rotate applied before any crop: `FlipHorizontal`, `FlipVertical`, `RotateClockwise`, `Rotate180`, `RotateCounterClockwise`. Useful for reusing a single texture asset in multiple orientations. Note: if a normal map is present, the normal vectors are **not** automatically adjusted to match. (`render.proto:27–40`)

---

## 3. Block Groups: O(1) Type Checking

Block groups are the primary way to ask "is this block a member of category X?" without hard-coding block IDs everywhere.

### Registering a Group

```rust
// During server setup (GameBuilder):
game.inner.register_fast_block_group("pipes:conductive");

// During block registration (BlockBuilder):
block.add_block_group("pipes:conductive");
```

### Checking at Runtime

```rust
// In any handler (after server init):
if let Some(group) = ctx.block_types().fast_block_group("pipes:conductive") {
    if group.contains(neighbor_block_id) {
        // neighbor is a pipe conductor
    }
}
```

Internally this is a `bitvec` indexed by block base index — constant time regardless of how many blocks are in the group.

Consider calling `fast_block_group` once and then using it for many lookups, in situations where you expect to do many lookups
in a row. `fast_block_group` (which borrows from block_types) is a single hashmap lookup.

### Built-in Groups to Know

| Group | Meaning |
|-------|---------|
| `trivially_replaceable` | Air-like; machines/tools can overwrite without asking |
| `default:solid` | Has collision |
| `default:liquid` | Fluid physics |
| `variant_encodes_placer` | Upper bits track who placed this block |
| `natural_ground` | Terrain; autobuild may overwrite |

### Pattern: Peer Detection

Circuit wires use groups to quickly skip non-circuit neighbors during BFS, avoiding an expensive callback-map lookup on every adjacent block. Your pipe system should do the same: register `"mypipes:pipe"`, check it before checking pipe-specific state.

---

## 4. Event Handlers: Reacting to Player Actions

Blocks can attach handlers for five interaction events. Each has an *inline* (atomic, runs under chunk lock) and *full* (racy, has full `GameState` access) variant.

### Handler Types

```rust
// Dig: player removes block. MUST set block to AIR_ID in inline handler.
pub type InlineInteractionHandler =
    dyn Fn(InlineContext, &mut BlockId, &mut ExtendedDataHolder, Option<&ItemStack>)
        -> Result<BlockInteractionResult>;

// Tap: player right-clicks without placing. Block stays.
// Same signature as dig inline.

// Step-on: player walks onto block. Position unsynchronized; may be lost.
pub step_on_handler_inline: Option<Box<InlineInteractionHandler>>,
pub step_on_handler_full:   Option<Box<FullInteractionHandler>>,

// Interact key: player presses 'E' (or named key). Returns optional popup.
pub type InteractKeyHandler =
    dyn Fn(HandlerContext, BlockCoordinate, &str) -> Result<Option<Popup>>;

// Fixup: called when block moves (piston) or is imported (template).
pub fixup_handler_inline: Option<Box<InlineGenericHandler<FixupReason, ()>>>,
```

See `perovskite_server/src/game_state/blocks.rs:215–410`.

### Inline vs. Full Handler: When to Use Each

| Use inline when... | Use full when... |
|--------------------|-----------------|
| You need atomic guarantee (block can't change mid-handler) | You need to read/write other blocks or inventories |
| Updating the block itself (set new variant, clear it) | Checking neighbor state for chain reactions |
| Simple state transitions | Spawning entities, sending chat, opening popups |

**Never** access `game_map` from inside an inline handler — you'll deadlock.

### Step-On Trick: Pressure Plates

The `step_on_handler` fires when any player enters a block. Combine with a deferred call to trigger a circuit edge or machine cycle:

```rust
.set_step_on_handler_full(Box::new(|ctx, coord, _item| {
    ctx.run_deferred(move |ctx| {
        dispatch_signal(&ctx, coord, true)
    });
    Ok(BlockInteractionResult::default())
}))
```

### Multi-Function Interact Keys

The `&str` argument to `InteractKeyHandler` is the key name from `interact_key_option` in the block proto. An empty string is the default. Registering multiple named keys lets one block offer a menu of actions (different crafting modes, open vs. close, configure vs. operate).

---

## 5. Timers: Recurring Block Updates

The timer system (`perovskite_server/src/game_state/game_map/timers.rs`) drives all time-based block behavior: crop growth, oscillators, liquid spread, machine cycles.

### Registration

```rust
game_builder.inner.add_timer(
    "myplugin:my_timer",
    TimerSettings {
        interval: Duration::from_secs(2),
        shards: 16,          // parallel execution buckets
        spreading: 1.0,      // spread firings evenly across interval
        block_types: vec![my_block.id, my_block_lit.id],
        per_block_probability: 0.3,   // 30% chance per matching block per tick
        populate_lighting: true,       // include light data in callback
        idle_chunk_after_unchanged: false,
        ..Default::default()
    },
    TimerCallback::BulkUpdateWithNeighbors(Box::new(MyTimerImpl)),
);
```

See `perovskite_game_api/src/farming/crops.rs:446–462` for the farming crop growth timer as a full working example.

### The Four Callback Types

| Type | Neighbors? | Use for |
|------|-----------|---------|
| `PerBlockLocked` | No | Simple per-block transitions |
| `BulkUpdate` | No | Chunk-wide homogeneous updates |
| `BulkUpdateWithNeighbors` | Yes (27-chunk grid) | Growth checks, liquid spread, signal propagation |
| `LockedVerticalNeighbors` | Vertical pairs | Falling blocks, vertical light propagation |

### BulkUpdateWithNeighbors Callback Signature

```rust
impl BulkUpdateCallback for MyTimer {
    fn bulk_update_callback(
        &self,
        ctx: &HandlerContext<'_>,
        chunk_coordinate: ChunkCoordinate,
        _timer_state: &TimerState,
        chunk: &mut MapChunk,         // write here
        neighbors: Option<&ChunkNeighbors>,  // read neighbors here
        lights: Option<&LightScratchpad>,
    ) -> Result<()> {
        for x in 0..16 { for z in 0..16 { for y in 0..16 {
            let offset = ChunkOffset::new(x, y, z);
            let block = chunk.get_block(offset);
            if block.equals_ignore_variant(MY_BLOCK) {
                let coord = chunk_coordinate.with_offset(offset);
                // check neighbors
                if let Some(below) = coord.try_delta(0, -1, 0)
                    .and_then(|c| neighbors.as_ref()?.get_block(c))
                {
                    if below.equals_ignore_variant(SOIL) {
                        chunk.set_block(offset, NEXT_STAGE, None);
                    }
                }
            }
        }}}
        Ok(())
    }
}
```

**Important:** Never call `ctx.game_map()` from inside a bulk callback. The chunk is already locked; a second lock attempt will deadlock.

### Deferred Edge Transmission from Timers

For circuit-style timers (oscillators, delay lines), fire the edge *after* releasing the chunk lock using `ctx.run_deferred()`:

```rust
// Collect pending signal changes into a Vec during the callback
// Then after the callback returns:
ctx.run_deferred(move |ctx| {
    for (coord, high) in pending_signals {
        dispatch_signal(&ctx, coord, high);
    }
    Ok(())
});
```

See `perovskite_game_api/src/circuits/simple_blocks.rs:418–471` for the oscillator block.

### Performance: `idle_chunk_after_unchanged`

Setting `idle_chunk_after_unchanged: true` skips chunks that haven't changed since the last tick. Use this for stable systems (liquids that have reached equilibrium, soil moisture). Don't use it for growth systems where every block must be RNG-evaluated even if nothing changed last tick.

---

## 6. Block Swapping: State Machines Without Data

The simplest form of state machine: each state is a different block type, and transitioning means replacing one block ID with another. No `ExtendedData` needed for the state itself.

### In Timer Callbacks

```rust
// crops.rs pattern: each growth stage is a distinct block
chunk.set_block(offset, next_stage_block_id, None);
```

`MapChunk::set_block` returns the old `(BlockId, Option<ExtendedData>)`, letting you preserve extended data across the transition if needed.

### `swap_blocks` for Conveyors

```rust
// Swap two blocks within the same chunk (e.g., item on conveyor advances one step)
chunk.swap_blocks(offset_a, offset_b);

// Swap across chunk boundaries (e.g., item moves from one chunk to another)
MapChunk::swap_blocks_across_chunks(&mut chunk_a, &mut chunk_b, offset_a, offset_b);
```

Both preserve `ExtendedData`. See `perovskite_server/src/game_state/game_map.rs:392–447`.

### Transition Target Enum Pattern

The farming module uses an `InteractionTransitionTarget` enum with variants like `NextStage`, `JumpToRandomStage(Vec<usize>)`, `ChangeBlockType(BlockId)`, `Remove` (→ air). This is worth copying for any multi-stage system where you need configurability:

```rust
enum MyTransition {
    Next,
    JumpTo(BlockId),
    Remove,
    DoNothing,
}
```

---

## 7. Neighbor Inspection: Reading the Environment

### From Inside a Timer (BulkUpdateWithNeighbors)

```rust
// ChunkNeighbors covers ±1 chunk in all 6 directions (27 total)
// Access any coordinate, even outside the current chunk:
let neighbor_block = neighbors?.get_block(any_block_coord)?;
```

This is the primary way to read neighbor state without acquiring locks: the system pre-loads all 27 chunks before calling your callback. Read-only during the callback.

### From a Full Handler (In-World Queries)

```rust
// Single block read
let block = ctx.game_map().try_get_block(coord)?;

// Check a block group
if ctx.block_types().fast_block_group("ore")
    .map(|g| g.contains(block))
    .unwrap_or(false) { ... }
```

### Checking the 6 Faces

```rust
for (dx, dy, dz) in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)] {
    if let Some(neighbor_coord) = coord.try_delta(dx, dy, dz) {
        if let Ok(neighbor) = ctx.game_map().try_get_block(neighbor_coord) {
            // process neighbor
        }
    }
}
```

`try_delta` returns `None` if the coordinate would overflow the world bounds.

### Soil Moisture Pattern: Gradient Diffusion

The farming soil module encodes *distance from water* in variant bits, creating a smooth moisture gradient without a separate data structure. On each timer tick it scans a 3×3×3 neighborhood, takes the max moisture level of any neighbor (decremented by 1), and stores it as the block's own variant. This implements diffusion/propagation purely through block swaps + variant updates. See `perovskite_game_api/src/farming/soil.rs:41–85`.

**Reuse:** Heat diffusion from furnaces, gas concentration gradients, signal strength attenuation. This doesn't need to be done for light; light propagation
is provided by the game engine.

---

## 8. BFS Connectivity: Finding Connected Components

BFS is the backbone of circuits (wire nets), machine discovery, and any system where "connected" blocks must act together.

### The Circuit Wire Pattern

`recalculate_wire()` in `perovskite_game_api/src/circuits/wire.rs:230–342` implements the definitive BFS propagation loop:

```rust
fn recalculate_wire(ctx, first_wire, who_signalled) {
    let mut queue = VecDeque::from([first_wire]);
    let mut visited_wires = FxHashSet::default();
    let mut non_wire_neighbors = FxHashMap::default(); // coord → incoming pins

    while let Some(coord) = queue.pop_front() {
        if !visited_wires.insert(coord) { continue; }
        if visited_wires.len() > MAX_WIRE_FANOUT { break; } // hard cap

        for live in get_live_connectivities(ctx, coord) {
            let neighbor = live.neighbor_coord;
            if neighbor_is_wire(ctx, neighbor) {
                queue.push_back(neighbor);
            } else {
                // Collect non-wire endpoints for callbacks
                non_wire_neighbors.entry(neighbor).or_default().push(live);
            }
        }
    }

    // Sample all sources, determine net state
    let net_high = non_wire_neighbors.iter()
        .any(|(coord, pins)| sample_any_pin_high(ctx, coord, pins));

    // Atomically update all wires in one pass
    for wire_coord in &visited_wires {
        mutate_block_atomically(*wire_coord, |block| {
            *block = if net_high { wire_on } else { wire_off };
        });
    }

    // Notify all non-wire endpoints
    for (coord, pins) in non_wire_neighbors {
        on_incoming_edge(ctx, coord, pins, net_high);
    }
}
```

### The Machine Discovery Pattern

`trigger_machine_cycle()` in `perovskite_game_api/src/autobuild/machines.rs:610–731` uses BFS for a different goal: discovering which blocks belong to a multi-block machine.

```rust
let mut visit_queue = vec![start_coord];
let mut visited = FxHashSet::default();
let mut machines = FxHashMap::default(); // coord → (MachineDef, BlockId)

while let Some(coord) = visit_queue.pop() {
    if !visited.insert(coord) { continue; }
    if visited.len() > MAX_MACHINE_BLOCKS { return Err("too large"); }

    let block = ctx.game_map().try_get_block(coord)?;
    if let Some(def) = machines_ext.block_types.get(&block.base_id()) {
        machines.insert(coord, (def, block));
        for (dx, dy, dz) in SIX_FACES {
            if let Some(neighbor) = coord.try_delta(dx, dy, dz) {
                visit_queue.push(neighbor);
            }
        }
    }
}
```

### Connectivity Rules: Bidirectional Validation

`get_live_connectivities()` (`perovskite_game_api/src/circuits/mod.rs:273–323`) only considers a neighbor "connected" if the neighbor also has a connectivity rule pointing back. This prevents one-sided connections and is essential for any port-based system (gates connect to wires, not the reverse):

```rust
// For each connectivity rule this block has:
for rule in my_rules {
    let neighbor_coord = rule.eval(coord, my_variant);
    let neighbor_block = get_block(neighbor_coord);
    let neighbor_rules = get_connectivity_rules(neighbor_block);
    // Only "live" if neighbor has a reciprocal rule pointing back to us
    if neighbor_rules.iter().any(|r| r.eval(neighbor_coord, neighbor_variant) == coord) {
        yield LiveConnectivity { neighbor_coord, ... };
    }
}
```

---

## 9. Signal Propagation: Nets and Busses

The circuits module demonstrates how to build a full multi-block signal network on top of block state + event handlers. The key patterns generalize to any network protocol.

### Connectivity Rules with Rotation

Define connectivity as a table of (dx, dy, dz, variant_bit, rotation_mode) tuples:

```rust
// Wire: connects in all 4 horizontal directions, unrotated
BlockConnectivity::unrotated(1,  0, 0, VARIANT_XPLUS),
BlockConnectivity::unrotated(-1, 0, 0, VARIANT_XMINUS),

// Gate output: connects "forward" from the gate's facing direction
BlockConnectivity::rotated_nesw_with_variant(0, 0, -1, 0),
```

`rotated_nesw_with_variant` applies the block's rotation variant to the offset, so a gate facing east connects eastward regardless of how you placed it.

### TTL / Loop Prevention

Any propagation algorithm needs a hard fanout cap. Circuit wires cap at `MAX_WIRE_FANOUT = 256` blocks. Machine BFS caps at `MAX_MACHINE_BLOCKS = 256`. For your own networks, pick a similar limit and enforce it early in the BFS loop:

```rust
if visited.len() > MAX_NET_SIZE {
    log::warn!("network at {:?} exceeds size limit", start_coord);
    break;
}
```

### Deferred Callbacks Prevent Recursion

When a signal propagation would trigger another signal (e.g., a gate output changes, which feeds into another gate), use `ctx.run_deferred()` to break the call chain. Otherwise you risk stack overflow on large networks. The circuits system uses `CircuitHandlerContext` with an explicit TTL counter for the same reason.

### Multi-Bit Busses

The microcontroller block treats the wire network as a multi-bit bus: each pin can carry a value, not just a single high/low bit. The `sample_pin()` callback returns a `u64`, and the bus master aggregates pin values from all connected drivers. To implement a data bus: store the bus value in extended data, propagate via BFS like a wire net, deliver the full value to all receivers.

---

## 10. Axis-Aligned Boxes: Directional Visuals

For any block that should look different depending on which neighbors it connects to, use `add_box_with_variant_mask()` to conditionally show sub-boxes. There are two mechanisms, which can be combined:

- **Server-driven (variant bits, [0,12) of the mask):** the server updates the block's variant when neighbors change (e.g. circuit wires, which need the connectivity for game logic anyway).
- **Client-driven (neighbor-presence bits, [12,18) of the mask):** the client shows the box automatically when a connecting neighbor (solid opaque, or same base block) is present in that direction — no server round trip, no variant churn, and no variant bits consumed. Use `render_selector::NEIGHBOR_XPLUS` etc. See `make_fence` in `perovskite_game_api/src/default_game/shaped_blocks.rs` for a complete example (fence arms that also collide and are tool-targetable only when connected).

### Wire Appearance Example

```rust
fn build_wire_aabox(builder: &mut CubeAppearanceBuilder) {
    // Always-present central nub
    builder.add_box((-0.1, 0.1), (-0.5, -0.4), (-0.1, 0.1), &texture);

    // Conditional arms — only shown when the matching variant bit is set
    builder.add_box_with_variant_mask(
        (0.1, 0.5), (-0.5, -0.4), (-0.1, 0.1),  // extends in +X
        &texture,
        VARIANT_XPLUS,   // shown iff bit 0 is set in block variant
    );
    builder.add_box_with_variant_mask(
        (-0.5, -0.1), (-0.5, -0.4), (-0.1, 0.1), // extends in -X
        &texture,
        VARIANT_XMINUS,  // shown iff bit 1 is set
    );
    // ... etc for Z+ Z- and the "above" variants
}
```

See `perovskite_game_api/src/circuits/wire.rs:59–145` for the full implementation.

**The update loop:** When a neighbor changes, call `update_connectivity()` which re-queries all live connections, ORs together their variant bits, and calls `block_id.with_variant_unchecked(new_bits)`. The client updates the visual automatically on the next chunk mesh rebuild.

**Reuse:** Pipes that show open ends toward connected neighbors, power cables, fences, walls — anything with directional connection geometry.

---

## 11. Multi-Block Machines: Sense/Act Coordination

The autobuild machines system (`perovskite_game_api/src/autobuild/machines.rs`) shows how to build a group of blocks that collectively perform work each "cycle."

### The MachineAction Trait

```rust
pub trait MachineAction: Send + Sync {
    // Phase 1: read environment, return intent (e.g., "I want to move north")
    fn sense(&self, ctx: &HandlerContext, state: &ActionState) -> Result<SenseInput> {
        Ok(Default::default())
    }
    // Phase 2: after all sense() calls aggregated, perform work
    fn act(&self, ctx: &HandlerContext, state: &ActionState) -> Result<ActionOutcome>;
}
```

`ActionState` gives each block its own coordinate, block ID (with variant/orientation), and the aggregated `SenseOutput` from all blocks in the machine.

### The Cycle

`trigger_machine_cycle(ctx, start_coord)`:

1. **BFS discovery** — finds all connected machine blocks (max 256)
2. **Sense phase** — calls `sense()` on every block; aggregates movement requests (conflict if blocks disagree)
3. **Act phase** — calls `act()` on every block with the aggregated sense data
4. **Movement phase** — if sense phase produced a consensus `Movement::Some(delta)`, sorts blocks along the movement axis and moves them front-to-back

### Block Roles in a Machine

| Role | Built-in block | Action |
|------|---------------|--------|
| Dig (fixed dir) | `machine_dig_up/down` | `DigFixedDeltaAction(0, 1, 0)` |
| Dig (facing dir) | `machine_dig_facing` | `DigFacingDirectionAction` |
| Place (fixed dir) | `machine_place_up/down` | `PlaceFixedDeltaAction(0, -1, 0)` |
| Place (facing dir) | `machine_place_facing` | `PlaceFacingDirectionAction` |
| Move | `machine_move_one` | `MoveOneAction` |
| Structural | `machine_manual_trigger` | `DoNothingAction` |

A player builds a machine by placing any combination of these blocks in a connected cluster. A trigger block fires the cycle.

### Composing Custom Roles

```rust
// Build a "drill head" that digs facing AND proposes forward movement
struct DrillHeadAction;
impl MachineAction for DrillHeadAction {
    fn sense(&self, _ctx, state) -> Result<SenseInput> {
        let dir = CompassDirection::from_rotation_variant(state.machine_block_id.variant());
        let (dx, dz) = dir.to_delta_xz();
        Ok(SenseInput {
            requested_movement: Some((dx, 0, dz)),
            ..Default::default()
        })
    }
    fn act(&self, ctx, state) -> Result<ActionOutcome> {
        // Dig the block in the facing direction
        let target = state.machine_coord.try_delta(dx, 0, dz)?;
        ctx.game_map().dig_block(ctx, target, None)?;
        Ok(Default::default())
    }
}
```

Combine actions with `CombinedAction<T, U>` (`machines.rs:556–576`) to put multiple behaviors in one block.

---

## 12. Atomic Group Movement

`MoveOneAction` demonstrates how to move an entire multi-block structure atomically, preventing mid-move collisions.

### Algorithm

1. **Sort blocks along movement vector** (reverse order — "front" blocks move first):
   ```rust
   blocks_to_move.sort_by_key(|(_, (mx, my, mz))| 
       -(dx * mx + dy * my + dz * mz));
   ```
2. **Test and move each block atomically** using `mutate_block_atomically` + `ControlFlow`:
   ```rust
   ctx.game_map().mutate_block_atomically(dst_coord, |block, ext| {
       if !ctx.block_types().is_trivially_replaceable(*block) {
           Ok(ControlFlow::Break(())) // abort — something is in the way
       } else {
           *block = src_block;
           *ext = src_ext;
           Ok(ControlFlow::Continue(()))
       }
   })?;
   ```
3. **Clear vacated source blocks** with `AIR_ID`.

This pattern generalizes to any "vehicle" structure — a submarine, a flying ship, a piston head — where you need to verify all destinations are clear before committing any move.

Note that at the moment, this is not truly atomic, but this pattern is likely to be the easiest to refit to a transactional API when it becomes available.

---

## 13. Bulk Writes and Undo

`BatchedWrite` (`perovskite_game_api/src/autobuild/mod.rs:165–395`) and `BatchedUndo` (lines 397–420) solve the "place/remove many blocks efficiently, with the option to undo" problem.

### BatchedWrite

```rust
let mut write = BatchedWrite::new();
for coord in my_coords {
    write.push(coord, block_id, OverwriteBehavior::DetectConflicts);
}
let undo = write.commit(ctx, &write_params)?;
// Returns a BatchedUndo containing pre-write state
```

Internally: sorts by chunk → reads pre-write state (for undo + conflict detection) → writes in chunk-order for cache efficiency.

`OverwriteBehavior` variants:
- `DetectConflicts` — honor `WriteParameters` (respect player-placed flags)
- `ForceOverwrite` — ignore all conflict detection
- `SilentlySkip` — skip occupied blocks without error

### BatchedUndo

```rust
// Store in player transient data
player.with_transient_data::<BatchedUndo, _>(|x| *x = undo);

// Later, on /undo:
let undo = player.with_transient_data::<BatchedUndo, _>(|x| std::mem::take(x));
undo.undo(ctx)?;  // restores all blocks + their ExtendedData
```

One undo level per player. Both `BlockId` and `ExtendedData` (inventories, custom data) are fully restored.

### Pattern: Safe World Editing

Any tool that bulk-modifies the world should use `BatchedWrite` + `BatchedUndo` + register a `/undo` command. This gives players confidence to use automated tools without fear of irreversible mistakes.

---

## 14. Two-Click Tools: The Autobuilder Trait

For player-facing tools that require a "select start, then execute" UX, implement `Autobuilder` (`perovskite_game_api/src/autobuild/mod.rs:427–502`).

```rust
pub trait Autobuilder {
    // Serialized settings (protobuf) — survive server restart
    type Settings: PersistentData + prost::Message + Default + Clone;
    // In-memory selection state — cleared on logout
    type SelectionState: Any + Send + Sync + Default + Clone + 'static;

    // First click (tap): update SelectionState
    fn tap(&self, ctx, coord, player, settings, state) -> Result<SelectionState>;

    // Second click (place): do the work, return BatchedUndo
    fn build(&self, ctx, coord, player, settings, state) 
        -> Result<Option<BatchedUndo>>;

    fn make_settings_popup(&self, ...) -> Popup; // dig-click shows this
    fn current_hint(&self, state) -> Option<ToolHint>; // real-time HUD

    const TOOL_ID: &'static str;
}

// Wire it up:
configure_item::<MyTool>(game, item_builder);
```

**State layers:**
- `Settings` (protobuf-backed, per-player): width, material, mode — anything the player configures via the settings popup.
- `SelectionState` (transient): the first-click coordinate, inferred direction, any mid-operation state.

---

## 15. GameState Extensions: Plugin-Wide Shared State

Any data that needs to be shared across multiple blocks, items, or timers within your plugin — or across plugins — lives in the `GameState` extension map.

```rust
// 1. Define your extension
#[derive(Clone)]
pub struct MyPluginState {
    pub copper_ore_block: FastBlockName,
    pub smelter_group: String,
    pub global_power_level: Arc<AtomicU32>,
}
impl GameStateExtension for MyPluginState {}

impl GameBuilderExtension for MyPluginState {
    fn pre_run(&mut self, server_builder: &mut ServerBuilder) {
        server_builder.add_extension(self.clone());
    }
}

// 2. Register during init
let ext = builder.builder_extension_mut::<MyPluginState>();
ext.copper_ore_block = copper_ore.name.clone();

// 3. Retrieve in any handler
let state = ctx.game_state.extension::<MyPluginState>()
    .expect("MyPlugin not initialized");
let power = state.global_power_level.load(Ordering::Relaxed);
```

See `perovskite_game_api/src/farming/mod.rs:31–42` for the farming plugin's use of this pattern to share block name handles across the soil, crop, and hoe subsystems.

**Use for:** Ore registries, global resource pools, cross-plugin event buses, rate-limit managers, per-world config.

---

## 16. Entity Coroutines: Autonomous Behavior

Entities (animals, carts, NPCs) express autonomous behavior through the `EntityCoroutine` trait rather than traditional game-loop callbacks.

### The Two-Method Pattern

```rust
pub trait EntityCoroutine {
    // Called when the entity needs its next move queued
    fn plan_move(self: Pin<&mut Self>, services: EntityCoroutineServices<'_>)
        -> CoroutineResult;

    // Called when a deferred async operation completes
    fn continuation(self: Pin<&mut Self>, services, result: ContinuationResult)
        -> CoroutineResult;
}
```

`plan_move` can return:
- `QueueSingleMovement(Movement)` — queue one move and wait
- `QueueUpMultiple(Vec<Movement>)` — queue several moves in advance
- `Deferral(deferral)` — spawn an async task; `continuation` will be called with the result

### The Deferral Pattern for Async Work

```rust
fn plan_move(self: Pin<&mut Self>, services) -> CoroutineResult {
    // Kick off an expensive async operation (pathfinding, DB query, etc.)
    let deferral = Deferral::defer_and_reinvoke(
        "pathfind",
        async move {
            let path = compute_path(start, goal).await;
            (path, ())
        }
    );
    CoroutineResult::Deferred(deferral)
}

fn continuation(self: Pin<&mut Self>, services, result: ContinuationResult) 
    -> CoroutineResult 
{
    let path = result.downcast::<Vec<BlockCoordinate>>();
    // Convert path steps to Movement queue
    let moves = path_to_movements(path);
    CoroutineResult::Successful(EntityMoveDecision::QueueUpMultiple(moves))
}
```

This keeps the main entity scheduler non-blocking: expensive work runs on tokio while the entity just "waits" at its current position.

### Duck AI Pattern: Probabilistic Local Pathfinding

The duck coroutine (`perovskite_game_api/src/animals/mod.rs:28–121`) shows the minimal viable animal AI:

1. 50% chance to pause (queue a zero-velocity movement for 1–2 seconds)
2. Scan 8 neighbors for preferred terrain (water)
3. Weight candidates by alignment with current heading (dot product → inertia)
4. Sample weighted random direction
5. Queue a 1-second movement in that direction

**Reuse template:** Replace the water check with patrol waypoints (guard), distance-to-player (predator), or item density (forager). The inertia weighting prevents erratic zigzag motion in all cases.

### Entity Lifecycle

- **Create:** `entity_manager.insert(entity_type, position, coroutine)`
- **Update:** Scheduler calls `plan_move` when movement buffer runs low
- **Remove:** `entity_manager.remove(entity_id)` → calls `pre_delete()` on coroutine for cleanup (release signals, cancel pending tasks)

Entities are stored in struct-of-arrays layout for cache efficiency. Position updates use quadratic kinematics: `pos(t) = s + v·t + 0.5·a·t²`.

---

## 17. Physics-Based Entity Movement

Carts demonstrate how to represent physically plausible movement as a sequence of pre-computed kinematic segments rather than frame-by-frame integration.

### TrackSegment + Kinematic Scheduling

```rust
struct TrackSegment {
    from: Vector3<f64>,
    to: Vector3<f64>,
    max_speed: f64,       // speed limit on this segment
    starting_odometer: f64,
}
```

For each segment, `schedule_single_segment()` computes:
- **Acceleration phase:** time and distance to reach `max_speed` from entry speed
- **Cruise phase:** remaining distance at constant speed
- **Deceleration phase:** brake to next segment's entry speed requirement

The result is a `Movement { velocity, acceleration, move_time }` that the entity system can interpolate client-side without further computation.

**Brake curve calculation:** To find the maximum safe entry speed for a segment given the required exit speed, solve backwards: `entry² = exit² + 2·a·distance`. Propagate this backwards from the end of a planned route to the start to ensure the entity can always stop in time. See `perovskite_game_api/src/carts/track_tool.rs:1449–1597`.

### Rail Following: Track Scan State

The `ScanState::advance()` algorithm (`perovskite_game_api/src/carts/tracks.rs:1587–1934`) defines how a cart moves from one track tile to the next:

1. Parse the current tile's `TileId` from variant bits
2. Look up the tile's `next_delta` (an encoded coordinate offset, pre-rotated for this tile's orientation)
3. Compute the neighbor coordinate
4. Validate the neighbor's `TileId` against an allowlist of valid successors
5. Check secondary/tertiary "helper" tiles (for switch validation)
6. Return the new `ScanState`

This pattern generalizes: any block-to-block traversal (pipes, conduits, roads) can be expressed as a table of (tile_type → valid successors, delta to next) rules. The key is pre-computing rotated versions of each rule so the traversal code is orientation-independent.

### Inferring Connectivity from Existing Structure

`determine_track_exit()` (`perovskite_game_api/src/carts/track_tool.rs:575–646`) determines which end of an existing track segment to extend from:

1. Scan forward from the endpoint
2. Scan backward from the endpoint
3. If one direction is already connected and the other is disconnected, the disconnected end is where to extend
4. `DisconnectedTrack(coord, straight_valid)` tells you the track "wants" to connect at `coord` but can't yet

**Reuse:** Any tool that extends an existing network from its open end uses this two-direction scan + disambiguation pattern.

---

## 18. Inventory System: Containers and Crafting

### InventoryKey

Persistent inventories (chests, player hotbars, machine buffers) are identified by `InventoryKey` — a 128-bit UUID. The inventory lives in the database under `KeySpace::Inventory` and is loaded on demand.

Block-local inventories (small buffers, transient crafting grids) live in `ExtendedData::inventories` instead, keyed by a string name.

### Atomic Multi-Inventory Operations

For any operation that moves items between two InventoryKey-based inventories (e.g., auto-crafting extracts from input, inserts to output), use:

```rust
game_state.inventory_manager().mutate_inventories_atomically(
    &[input_key, output_key],  // sorted internally to prevent deadlock
    |invs| {
        let input = &mut invs[0];
        let output = &mut invs[1];
        // Move items...
        Ok(())
    }
)?;
```

**Never** call `mutate_inventory_atomically` from inside an item handler (it will panic). Use `ctx.run_deferred()` to escape the handler context first.

### In-block inventories

Inventories stored inside a block's ExtendedData are "plain old data" that uses the same lifecycle, locking, etc as all other extended data in the map.
Use `mutate_block_atomically` to access it for edit, then insert/take/move/peek at whatever you need (if only peeking, consider `get_block_with_extended_data` and pass a callback to inspect what you need)

### Virtual Inventory Views for Crafting

The popup/UI system supports `VirtualOutput` and `VirtualInput` inventory views that compute their contents dynamically:

- `VirtualInput`: callbacks fire when a player places an item (validate recipe ingredients)
- `VirtualOutput`: contents computed on demand (the crafting result for current inputs)
- Taking from `VirtualOutput` triggers consumption of `VirtualInput` contents

This is the standard pattern for crafting tables, auto-assemblers, creative menus, and any "transform inputs → output" UI.

---

## 19. Popup UI: Server-Driven Dialogs

The server can send arbitrary UI dialogs to players, built from a set of primitive widgets.

```rust
// From inside an interact_key_handler:
fn my_handler(ctx: HandlerContext, coord: BlockCoordinate, _key: &str) 
    -> Result<Option<Popup>> 
{
    let mut popup = ctx.new_popup().title("Smelter");

    popup.label("Fuel");
    popup.inventory_view_block(coord, "fuel", "Fuel Slot", (1, 1))?;

    popup.label("Input");
    popup.inventory_view_block(coord, "input", "Input", (9, 1))?;

    popup.label("Output");
    popup.inventory_view_block(coord, "output", "Output", (9, 1))?;

    popup.button("Start Smelting", "start");
    popup.on_button(Box::new(|resp| {
        if let PopupAction::ButtonClicked(key) = &resp.user_action {
            if key == "start" {
                begin_smelt_cycle(&resp.ctx, coord)?;
            }
        }
        Ok(())
    }))?;

    Ok(Some(popup))
}
```

### Inventory View Backing Types

| Type | Backed by | Use |
|------|-----------|-----|
| `inventory_view_stored(key)` | Database `InventoryKey` | Chest contents, player main inventory |
| `inventory_view_block(coord, name)` | `ExtendedData::inventories` | Machine I/O slots |
| `VirtualOutput` | Closure | Crafting result preview |
| `VirtualInput` | Closure | Recipe input validation |
| `Transient` | Popup lifetime | Temporary drag-and-drop space |

---

## 20. Terrain Following and Path Planning

The road tool (`perovskite_game_api/src/autobuild/mod.rs:855–1161`) demonstrates a complete pipeline for routing a path across terrain.

### Axis Normalization Trick

Normalize a 2D path to always be "X-major" by conditionally transposing coordinates:

```rust
let transposer: fn(BlockCoordinate) -> BlockCoordinate = 
    if abs_dx > abs_dz { |c| c }              // X-dominant: identity
    else               { |c| BlockCoordinate::new(c.z, c.y, c.x) }; // Z-dominant: swap X/Z
// All path logic uses X as primary axis
// Transpose back when writing blocks: transposer(computed_coord)
```

This halves the code for 2D path algorithms.

### Height Map Generation + Smoothing

1. Sample terrain height at each step with `probe_ground_level(y_min, y_max)` — scans downward for the first natural block
2. Store in an `ndarray::Array2<Option<i32>>`
3. Apply a 3×3 median filter over the interior to smooth spikes:
   ```rust
   array.windows([3, 3]).into_iter().for_each(|window| {
       let mut vals: SmallVec<[i32; 9]> = window.iter().flatten().copied().collect();
       vals.sort();
       *center = vals[vals.len() / 2];
   });
   ```

### Slope Validation and Bridge Mode

After generating height values, validate the slope constraint (e.g., max 1:3 rise-per-run):

```rust
for i in 1..heights.len() {
    heights[i] = heights[i].max(heights[i-1] - 1).min(heights[i-1] + 1);
}
if (heights[i] - heights[i-1]).abs() > MAX_SLOPE_PER_STEP {
    // Request bridge mode: straight line from start height to end height
}
```

### Step Type Detection for Curves

When routing a rail/road path, compare consecutive step directions. A direction change between steps is a `SharpTurn`; consecutive same-direction steps are `Straight`. Distribute slopes only across straight steps.

---

## 21. Composing a Tech Mod: Putting It Together

Here's how these techniques layer into some example tech systems.

### Electric Grid

**Goal:** Blocks emit/consume power; wires carry it; a generator block produces it; a deficit causes brownout.

1. **Wire blocks:** Variant bits = connectivity mask (bits 0–7). `update_connectivity()` callback re-checks neighbors and updates variant. Axis-aligned box appearance shows connected arms.
2. **BFS net discovery:** On any change to a wire or device, BFS from the changed block to find the connected component.
3. **Power balance:** After BFS, query each device's `sense()` — generators report positive watts, consumers report negative. Sum = net balance. Store result in a `GameState` extension (`Arc<AtomicI32>`). This can feed into any sort of algorithm of choice - from a simple match-supply-and-demand to a full-blown distributed power-market simulation with marginal pricing and price curves.

### Furnace / Smelter

1. **Block state:** Protobuf `FurnaceState { fuel_ticks, smelt_ticks, is_lit }` in custom data. Three `ExtendedData` inventories: `"input"`, `"fuel"`, `"output"`.
2. **Interact key → popup:** Shows all three inventory views + a status label.
3. **Timer:** `BulkUpdateWithNeighbors` timer ticks all furnace blocks. Each tick: decrement `fuel_ticks`; if zero, consume a fuel item and reset. Decrement `smelt_ticks`; if zero, consume input item, produce output item, reset.
4. **Block swap:** `is_lit` flag causes the block to swap to a lit variant (different texture) on state change.
5. **Circuit input:** A `tap_handler` connected to a circuit wire can enable/disable the furnace remotely.

### Turtle / Mining Robot

1. **Machine blocks:** A "turtle head" (digs facing direction), a "brain" (stores program), a trigger (receives circuit pulse). All connected via BFS machine discovery.
2. **MachineAction:** Brain `sense()` reads its program from `ExtendedData` and returns the next movement delta. Head `sense()` also requests the same delta (consensus required for `Movement::Some`). Head `act()` digs the block at `current_coord + delta`.
3. **Program storage:** A Rhai script (like the microcontroller) or a simple integer sequence stored in protobuf.
4. **Circuit trigger:** Wire a redstone pulse to the machine_manual_trigger to run one cycle per pulse. Or use a timer for autonomous operation.

### Pipe / Fluid Network

1. **Pipe blocks:** Variant bits = connectivity mask. `update_connectivity()` on place/dig.
2. **Fluid state:** Variant bits 8–11 = fluid type index (0 = empty, 1–15 = fluid types). Or use protobuf for more fluid types.
3. **Flow simulation:** `BulkUpdateWithNeighbors` timer. Each tick: if a pipe has fluid and a connected neighbor is empty, move fluid toward lower elevation (or toward lowest-pressure neighbor).
4. **Source/sink:** A pump block's `MachineAction::act()` pushes fluid from its inventory into adjacent pipe inventory. A tank block stores large quantities in `ExtendedData`.
5. **Visual:** Conditional AABoxes for the pipe connections; color variant for fluid color (dye the pipe texture with the fluid's color).

### Logistics Belt

1. **Belt direction:** `RotationMode::RotateNesw` for facing direction (bits 0–1). Variant bits 2–3 = speed tier.
2. **Item state:** Each belt block stores `Option<ItemStack>` in `ExtendedData::simple_data` (or custom protobuf for multiple slots).
3. **Tick:** Timer advances items one belt block in the belt's direction. Use `chunk.swap_blocks_across_chunks()` if the next belt is in a different chunk, or write to `ExtendedData` directly if same chunk.
4. **Junction:** When a belt's successor has two incoming belts, alternate which one feeds (round-robin stored in variant bits).
5. **Filter:** A sorter block reads item type from the incoming item's `ItemStack::item_id`, checks against a config stored in `ExtendedData::simple_data`, and routes to the appropriate output direction.

---

*This cookbook was generated from analysis of `perovskite_game_api` and `perovskite_server`. Cross-reference `.claude/sourcemap_game_api.md` and `.claude/sourcemap_server.md` for API overview, and `perovskite_game_api/src/circuits/` for the most comprehensive working example of almost all these patterns in one place. Rustdoc/source code has the most authoritative API documentation.*
