
This skill covers adding circuit-aware blocks to the game — blocks that send, receive, or respond
to digital signals and bus messages. For basic BlockBuilder usage (appearance, drops, block groups,
physics) see `define_new_block`. For advanced handler slots (dig, interact, extended data, timers)
see `advanced_block_features`.

Key source files:
- `perovskite_game_api/src/circuits/mod.rs` — core types, trait, helper fns, registration
- `perovskite_game_api/src/circuits/simple_blocks.rs` — source, lamp, oscillator
- `perovskite_game_api/src/circuits/switches.rs` — switch, button
- `perovskite_game_api/src/circuits/gates.rs` — AND/NOT/XOR/delay/DFF gates
- `perovskite_game_api/src/carts/signals.rs` — bus message receive example (interlocking signal)
- `perovskite_game_api/src/carts/interlocking.rs` — bus message send example

---

## Principles

### The two registration steps

Every circuit block requires **two** registration calls. Missing either one silently breaks the
block:

1. `.register_circuit_callbacks()` on the `BlockBuilder` — wires the dig/place hooks that notify
   the circuit engine when the block is added or removed. It also adds the block to the
   `CIRCUITS_GROUP` block group, and force-disables track-placer (see below).

2. `builder.define_circuit_callbacks(block_id, Box::new(MyCallbacks), CircuitBlockProperties { … })`
   — registers the block's connectivity and its `CircuitBlockCallbacks` implementation.

Both must be called for **every** `BlockId** that participates in circuits, including the "on"
variant of a dual-block pair.

### Dual-block pattern

Circuit blocks with a visible on/off state are modeled as **two separate blocks**: one for "off"
and one for "on". The "on" block:
- Has `HIDDEN_FROM_CREATIVE` so it doesn't appear in the creative inventory
- Has `set_simple_dropped_item(OFF_BLOCK_NAME, 1)` so digging it drops the off block
- Gets its own `define_circuit_callbacks` registration with the same `CircuitBlockProperties`

The block type stored on the map changes between the two when the signal state changes.

### Variants and rotation

`register_circuit_callbacks` automatically calls `force_disable_track_placer()`, because many circuits need free use of the variant bits for state. They are yours to use. It's also a fair bet
that circuit blocks weren't placed by the map generator, and are
otherwise valuable structures, so placer tracking isn't critical anyway.

For blocks that face a direction (switches, gates, signals), use
`BlockConnectivity::rotated_nesw_with_variant` so that connectivity rotates with the block.
The lowest two variant bits encode the facing direction when `RotationMode::RotateHorizontally`
is used.

### Spurious on_incoming_edge calls

The engine may deliver `on_incoming_edge` even when no real state change has occurred. Always
re-read the actual incoming pin states inside the handler (via `get_incoming_pin_states`) rather
than trusting the `state` argument alone. The `state` argument is still useful as a fast-path
hint, but is not guaranteed to be the only transition you receive.

### Overheat safety

`on_overheat` is called by a plain `&HandlerContext`, not a `CircuitHandlerContext`. This is
intentional — constructing a new `CircuitHandlerContext` inside `on_overheat` risks infinite
recursion or deadlock. Replace the block with a non-circuit dead block and do **not** call
`transmit_edge` or `transmit_bus_message`.

### Atomicity and race safety

In `on_incoming_edge`, always check that the block is still what you expect inside
`mutate_block_atomically` before changing it:

```rust
ctx.game_map().mutate_block_atomically(coord, |block, _| {
    if block.equals_ignore_variant(self.off) || block.equals_ignore_variant(self.on) {
        // safe to mutate
    }
    Ok(/* … */)
})?;
```

`try_get_block` (used in `sample_pin`) may return `None` if the chunk has been unloaded between
when the event was queued and when it runs — always guard with `match` / `?`.

---

## Mini API Reference

### Types

```rust
// perovskite_game_api::circuits
pub enum PinState { Low, High }
// PinState supports: ! (not), | (or), & (and), ^ (xor), From<bool>

pub struct BlockConnectivity {
    pub dx: i8, pub dy: i8, pub dz: i8,
    pub rotation_mode: ConnectivityRotation,  // NoRotation | RotateNeswWithVariant
    pub id: u32,   // arbitrary tag your callbacks can use to identify the pin role
}
impl BlockConnectivity {
    pub const fn unrotated(dx, dy, dz, id) -> Self;
    pub const fn rotated_nesw_with_variant(dx, dy, dz, id) -> Self;
    /// Returns the target BlockCoordinate, accounting for variant-based rotation.
    /// Returns None at map edge.
    pub fn eval(&self, coord: BlockCoordinate, variant: u16) -> Option<BlockCoordinate>;
}

pub struct CircuitBlockProperties {
    pub connectivity: Vec<BlockConnectivity>,
}

pub struct BusMessage {
    pub sender: BlockCoordinate,
    pub data: HashMap<String, String>,  // arbitrary string key-value payload
}

pub enum PinReading {
    Valid(PinState),
    CantGetBlock,       // chunk unloaded or map edge
    NotCircuitBlock,    // block isn't registered with circuits
    TtlExceeded,        // too deep a call chain; overheat will be triggered
}
```

### CircuitBlockCallbacks trait

```rust
pub trait CircuitBlockCallbacks: Send + Sync + 'static {
    // Called when a neighboring block is placed/removed, possibly changing live connections.
    // Default impl does nothing. Useful for wires that update their visual variant.
    fn update_connectivity(&self, ctx: &CircuitHandlerContext<'_>, coord: BlockCoordinate)
        -> Result<()> { Ok(()) }

    // Called when a connected input may be transitioning.
    // May be called spuriously. Re-read actual state via get_incoming_pin_states.
    // Propagate outputs by calling transmit_edge for each connected output.
    fn on_incoming_edge(&self, ctx: &CircuitHandlerContext<'_>,
        coordinate: BlockCoordinate, from: BlockCoordinate, state: PinState) -> Result<()> { Ok(()) }

    // Called when a bus message arrives from a connected block.
    // `from` is the immediate neighbor that sent it (may differ from message.sender if routed).
    fn on_bus_message(&self, ctx: &CircuitHandlerContext<'_>,
        coordinate: BlockCoordinate, from: BlockCoordinate, message: &BusMessage) -> Result<()> { Ok(()) }

    // Called by the engine to sample this block's current output toward `destination`.
    // Must be fast — no locks or heavy computation. Use cached variant state where possible.
    // Check that `destination` matches the expected output pin before returning High.
    fn sample_pin(&self, ctx: &CircuitHandlerContext<'_>,
        coord: BlockCoordinate, destination: BlockCoordinate) -> PinState { PinState::Low }

    // Called when the block must self-destruct due to combinational loop / overload.
    // Use a plain HandlerContext — DO NOT create a CircuitHandlerContext or call transmit_edge.
    fn on_overheat(&self, ctx: &HandlerContext, coord: BlockCoordinate) {}
}
```

### Helper functions

```rust
// perovskite_game_api::circuits

/// Returns all actually-connected neighbors and their current output states.
pub fn get_incoming_pin_states(ctx: &CircuitHandlerContext<'_>, coord: BlockCoordinate)
    -> SmallVec<[(BlockConnectivity, BlockCoordinate, PinState); 8]>;

/// Returns all currently-live connections (no pin state sampling).
pub fn get_live_connectivities(ctx: &CircuitHandlerContext<'_>, coord: BlockCoordinate)
    -> SmallVec<[(BlockConnectivity, BlockCoordinate); 8]>;

/// Directly samples one neighbor's output pin toward `into`.
pub fn get_pin_state(ctx: &CircuitHandlerContext<'_>, coord: BlockCoordinate, into: BlockCoordinate)
    -> PinReading;
```

### Event dispatch

```rust
// perovskite_game_api::circuits::events

/// Convert a HandlerContext to a CircuitHandlerContext (used in full handlers and timers).
pub fn make_root_context<'a>(ctx: &'a HandlerContext) -> CircuitHandlerContext<'a>;

/// Notify dest_coord that from_coord's output may be changing to new_state.
/// If dest_coord is a wire, the full net is re-evaluated. Otherwise on_incoming_edge is called directly.
pub fn transmit_edge(ctx: &CircuitHandlerContext<'_>,
    dest_coord: BlockCoordinate, from_coord: BlockCoordinate, new_state: PinState) -> Result<()>;

/// Like transmit_edge but also delivers a BusMessage to the destination's on_bus_message.
/// pin_state can be PinState::Low if you only need the message channel.
pub fn transmit_bus_message(ctx: &CircuitHandlerContext<'_>,
    dest_coord: BlockCoordinate, from_coord: BlockCoordinate,
    pin_state: PinState, message: BusMessage) -> Result<()>;
```

### Registration

```rust
// perovskite_game_api::circuits

/// Must be called once before registering any circuit blocks.
pub fn register_circuits(builder: &mut GameBuilder) -> Result<()>;

// Trait implemented by GameBuilder
pub trait CircuitGameBuilder {
    fn define_circuit_callbacks(&mut self, block_id: BlockId,
        callbacks: Box<dyn CircuitBlockCallbacks>,
        properties: CircuitBlockProperties) -> Result<()>;
}

// Trait implemented by BlockBuilder
pub trait CircuitBlockBuilder {
    /// Injects dig/place hooks and CIRCUITS_GROUP membership.
    /// Also calls force_disable_track_placer().
    fn register_circuit_callbacks(self) -> BlockBuilder;
}
```

### Imports

```rust
use perovskite_game_api::circuits::{
    BlockConnectivity, CircuitBlockBuilder, CircuitBlockCallbacks, CircuitBlockProperties,
    CircuitGameBuilder, PinState, PinReading, BusMessage,
    get_incoming_pin_states, get_live_connectivities, get_pin_state,
    events::{make_root_context, transmit_edge, transmit_bus_message},
    register_circuits,
};
use perovskite_core::constants::item_groups::HIDDEN_FROM_CREATIVE;
use perovskite_server::game_state::event::HandlerContext;
```

---

## Worked Examples

### 1. Simple source — always outputs High

This is the simplest possible block: constant output, no input handling.

```rust
// mod.rs or similar
const MY_SOURCE: StaticBlockName = StaticBlockName("myplugin:source");
const SOURCE_CONNECTIVITIES: [BlockConnectivity; 4] = [
    BlockConnectivity::unrotated( 1, 0,  0, 0),
    BlockConnectivity::unrotated( 0, 0,  1, 0),
    BlockConnectivity::unrotated(-1, 0,  0, 0),
    BlockConnectivity::unrotated( 0, 0, -1, 0),
];

struct MySourceCallbacks;
impl CircuitBlockCallbacks for MySourceCallbacks {
    fn sample_pin(&self, _ctx: &CircuitHandlerContext<'_>,
        _coord: BlockCoordinate, _destination: BlockCoordinate) -> PinState {
        PinState::High   // always driven high
    }
    // on_overheat: default (do nothing) is fine for a constant source
}

pub fn register(builder: &mut GameBuilder) -> Result<()> {
    register_circuits(builder)?;  // idempotent; safe to call multiple times

    let block = builder.add_block(
        BlockBuilder::new(MY_SOURCE)
            .set_display_name("My Source")
            .set_cube_single_texture(MY_TEX)
            .set_light_emission(4)
            .register_circuit_callbacks(),   // step 1
    )?;
    builder.define_circuit_callbacks(        // step 2
        block.id,
        Box::new(MySourceCallbacks),
        CircuitBlockProperties { connectivity: SOURCE_CONNECTIVITIES.to_vec() },
    )?;
    Ok(())
}
```

Reference: `simple_blocks.rs` — `SourceBlockCallbacks`.

---

### 2. Lamp — sink that visually reflects input state

Two blocks (off/on), OR-reduces all inputs, no output.

```rust
const LAMP_OFF: StaticBlockName = StaticBlockName("myplugin:lamp_off");
const LAMP_ON:  StaticBlockName = StaticBlockName("myplugin:lamp_on");
const LAMP_CONNECTIVITIES: [BlockConnectivity; 4] = [
    BlockConnectivity::unrotated( 1, 0,  0, 0),
    BlockConnectivity::unrotated( 0, 0,  1, 0),
    BlockConnectivity::unrotated(-1, 0,  0, 0),
    BlockConnectivity::unrotated( 0, 0, -1, 0),
];

struct LampCallbacks { lamp_off: BlockId, lamp_on: BlockId }
impl CircuitBlockCallbacks for LampCallbacks {
    fn on_incoming_edge(&self, ctx: &CircuitHandlerContext<'_>,
        coord: BlockCoordinate, _from: BlockCoordinate, _state: PinState) -> Result<()> {
        // OR-reduce all inputs (don't trust the 'state' hint alone)
        let any_high = get_incoming_pin_states(ctx, coord)
            .iter()
            .any(|(_, _, s)| s == &PinState::High);
        let desired = if any_high { self.lamp_on } else { self.lamp_off };
        ctx.game_map().mutate_block_atomically(coord, |block, _| {
            if block.base_id() == self.lamp_off.base_id()
                || block.base_id() == self.lamp_on.base_id()
            {
                *block = desired;
            }
            Ok(())
        })?;
        Ok(())
    }

    fn sample_pin(&self, _ctx: &CircuitHandlerContext<'_>, _coord: BlockCoordinate,
        _destination: BlockCoordinate) -> PinState {
        PinState::Low   // lamp is a pure sink
    }

    fn on_overheat(&self, ctx: &HandlerContext, coord: BlockCoordinate) {
        // Replace with air or a dead block; do NOT call transmit_edge
        let _ = ctx.game_map().set_block(coord, AIR_ID, None);
    }
}

pub fn register(builder: &mut GameBuilder) -> Result<()> {
    register_circuits(builder)?;

    let lamp_off = builder.add_block(
        BlockBuilder::new(LAMP_OFF)
            .set_display_name("Lamp")
            .set_cube_single_texture(LAMP_OFF_TEX)
            .set_light_emission(0)
            .register_circuit_callbacks(),
    )?;
    let lamp_on = builder.add_block(
        BlockBuilder::new(LAMP_ON)
            .set_display_name("Lamp (on)")
            .set_cube_single_texture(LAMP_ON_TEX)
            .set_light_emission(15)
            .add_item_group(HIDDEN_FROM_CREATIVE)
            .set_simple_dropped_item(LAMP_OFF.0, 1)
            .register_circuit_callbacks(),
    )?;
    // Both block IDs must be registered
    for id in [lamp_off.id, lamp_on.id] {
        builder.define_circuit_callbacks(
            id,
            Box::new(LampCallbacks { lamp_off: lamp_off.id, lamp_on: lamp_on.id }),
            CircuitBlockProperties { connectivity: LAMP_CONNECTIVITIES.to_vec() },
        )?;
    }
    Ok(())
}
```

Reference: `simple_blocks.rs` — `SimpleLampCallbacks` and `register_simple_blocks`.

---

### 3. Directional switch — player-toggled source with rotation

Uses rotation variant, interact handler, and `transmit_edge` from a full handler.

```rust
const SWITCH_OFF: StaticBlockName = StaticBlockName("myplugin:switch_off");
const SWITCH_ON:  StaticBlockName = StaticBlockName("myplugin:switch_on");

// Output goes forward (z+1 in local facing), also connects one block down
const SWITCH_CONNECTIVITIES: [BlockConnectivity; 2] = [
    BlockConnectivity::rotated_nesw_with_variant(0, 0, 1, 0),
    BlockConnectivity::rotated_nesw_with_variant(0, -1, 1, 0),
];

pub fn register(builder: &mut GameBuilder) -> Result<()> {
    register_circuits(builder)?;

    let switch_off_name = builder.inner.blocks().make_block_name(SWITCH_OFF.0);
    let switch_on_name  = builder.inner.blocks().make_block_name(SWITCH_ON.0);

    let switch_off = builder.add_block(
        BlockBuilder::new(SWITCH_OFF)
            .set_display_name("Switch")
            // ... appearance ...
            .set_allow_light_propagation(true)
            .add_interact_key_menu_entry("", "Turn on")
            .force_disable_track_placer()      // already done by register_circuit_callbacks,
                                               // but explicit here because we use variant bits
            .add_modifier(|block| {
                block.interact_key_handler = Some(Box::new(move |ctx, coord, _| {
                    let off = ctx.block_types().resolve_name(&switch_off_name).unwrap();
                    let on  = ctx.block_types().resolve_name(&switch_on_name).unwrap();
                    let mut targets: smallvec::SmallVec<[_; 2]> = smallvec::SmallVec::new();
                    ctx.game_map().mutate_block_atomically(coord, |id, _| {
                        if id.equals_ignore_variant(off) {
                            *id = on.with_variant_of(*id);
                            for conn in SWITCH_CONNECTIVITIES {
                                if let Some(dest) = conn.eval(coord, id.variant()) {
                                    targets.push((dest, coord));
                                }
                            }
                        }
                        Ok(())
                    })?;
                    let cctx = make_root_context(&ctx);
                    for (dest, src) in targets {
                        transmit_edge(&cctx, dest, src, PinState::High)?;
                    }
                    Ok(None)
                }));
            })
            .register_circuit_callbacks(),
    )?;

    let switch_on = builder.add_block(
        BlockBuilder::new(SWITCH_ON)
            .set_display_name("Switch (on)")
            // ... appearance ...
            .set_allow_light_propagation(true)
            .add_item_group(HIDDEN_FROM_CREATIVE)
            .set_simple_dropped_item(SWITCH_OFF.0, 1)
            .add_interact_key_menu_entry("", "Turn off")
            .add_modifier(|block| {
                // mirror of above, sends PinState::Low
                block.interact_key_handler = Some(Box::new(move |ctx, coord, _| {
                    // ... (see switches.rs for full implementation)
                    Ok(None)
                }));
            })
            .register_circuit_callbacks(),
    )?;

    builder.define_circuit_callbacks(switch_off.id,
        Box::new(SourceBlockCallbacks(PinState::Low)),
        CircuitBlockProperties { connectivity: SWITCH_CONNECTIVITIES.to_vec() })?;
    builder.define_circuit_callbacks(switch_on.id,
        Box::new(SourceBlockCallbacks(PinState::High)),
        CircuitBlockProperties { connectivity: SWITCH_CONNECTIVITIES.to_vec() })?;
    Ok(())
}
```

The key pattern when toggling a switch:
1. Inside `mutate_block_atomically`, atomically swap the block and collect destination coordinates.
2. Outside the lock, call `transmit_edge` for each destination.

Reference: `switches.rs` — `register_switches`.

---

### 4. Momentary button — auto-resets after 250 ms

Same as switch, but after turning on it schedules a deferred turn-off:

```rust
// Inside the button_off interact handler, after transmitting the High edge:
ctx.run_deferred_delayed(Duration::from_millis(250), move |ctx| {
    let mut targets: smallvec::SmallVec<[_; 2]> = smallvec::SmallVec::new();
    ctx.game_map().mutate_block_atomically(coord, |id, _| {
        let variant = id.variant();
        if id.equals_ignore_variant(button_on) {
            *id = button_off.with_variant_unchecked(variant);
            for conn in SWITCH_CONNECTIVITIES {
                if let Some(dest) = conn.eval(coord, variant) {
                    targets.push((dest, coord));
                }
            }
        }
        Ok(())
    })?;
    let cctx = make_root_context(&ctx);
    for (dest, src) in targets {
        transmit_edge(&cctx, dest, src, PinState::Low)?;
    }
    Ok(())
});
```

Reference: `switches.rs` — button interact handler.

---

### 5. Combinational gate — truth-table based, directional output

The gate has a fixed output direction (back, id=0) and up to three input directions (left/front/right).
The connectivity `id` field acts as a bitmask bit: left=0b100, front=0b010, right=0b001.

```rust
// Output pin: back (rotated)
const OUTPUT_CONN: BlockConnectivity = BlockConnectivity::rotated_nesw_with_variant(0, 0, -1, 0);
// Input pins with id bits for truth table lookup
const LEFT_CONN:  BlockConnectivity = BlockConnectivity::rotated_nesw_with_variant( 1, 0, 0, 0b100);
const FRONT_CONN: BlockConnectivity = BlockConnectivity::rotated_nesw_with_variant( 0, 0, 1, 0b010);
const RIGHT_CONN: BlockConnectivity = BlockConnectivity::rotated_nesw_with_variant(-1, 0, 0, 0b001);

struct AndGateImpl { on: BlockId, off: BlockId, broken: BlockId }
impl CircuitBlockCallbacks for AndGateImpl {
    fn on_incoming_edge(&self, ctx: &CircuitHandlerContext<'_>,
        coord: BlockCoordinate, _from: BlockCoordinate, _state: PinState) -> Result<()> {
        // Build an index from connectivity IDs and compute truth table
        let inbound = get_incoming_pin_states(ctx, coord);
        let mut input_bits = 0u8;
        for (conn, _, state) in &inbound {
            if *state == PinState::High {
                input_bits |= conn.id as u8;
            }
        }
        // AND truth table: high only when both left(0b100) and right(0b001) are high
        let truth_table: u8 = (1 << 0b101) | (1 << 0b111);
        let result_high = truth_table & (1 << input_bits) != 0;
        let new_state = PinState::from(result_high);

        let (changed, variant) = ctx.game_map().mutate_block_atomically(coord, |block, _| {
            let v = block.variant();
            if result_high && block.equals_ignore_variant(self.off) {
                *block = self.on.with_variant(v)?;
                Ok((true, v))
            } else if !result_high && block.equals_ignore_variant(self.on) {
                *block = self.off.with_variant(v)?;
                Ok((true, v))
            } else {
                Ok((false, 0))
            }
        })?;

        if changed {
            if let Some(dest) = OUTPUT_CONN.eval(coord, variant) {
                transmit_edge(ctx, dest, coord, new_state)?;
            }
        }
        Ok(())
    }

    fn sample_pin(&self, ctx: &CircuitHandlerContext<'_>,
        coord: BlockCoordinate, destination: BlockCoordinate) -> PinState {
        let block = match ctx.game_map().try_get_block(coord) {
            Some(b) => b, None => return PinState::Low,
        };
        // Only drive the output pin; inputs always read Low
        if OUTPUT_CONN.eval(coord, block.variant()) != Some(destination) {
            return PinState::Low;
        }
        if block.equals_ignore_variant(self.on) { PinState::High } else { PinState::Low }
    }

    fn on_overheat(&self, ctx: &HandlerContext, coord: BlockCoordinate) {
        let _ = ctx.game_map().set_block(coord, self.broken, None);
    }
}
```

Reference: `gates.rs` — `CombinationalGateImpl`, `AND_GATE_CONFIG`.

---

### 6. Oscillator — timer-driven periodic signal

A 1 Hz oscillator uses `BulkUpdateCallback` so it can efficiently update all oscillator blocks in
a chunk without per-block timer overhead:

```rust
use perovskite_server::game_state::game_map::{BulkUpdateCallback, TimerCallback, TimerSettings};
use perovskite_core::coordinates::ChunkOffset;

struct OscillatorTimer { on: BlockId, off: BlockId }
impl BulkUpdateCallback for OscillatorTimer {
    fn bulk_update_callback(&self, ctx: &HandlerContext<'_>,
        chunk_coordinate: ChunkCoordinate, _timer_state: &TimerState,
        chunk: &mut MapChunk, _neighbors: Option<&ChunkNeighbors>,
        _lights: Option<&LightScratchpad>) -> Result<()> {
        let cctx = make_root_context(ctx);
        let mut transitions = vec![];

        for dx in 0..16 { for dz in 0..16 { for dy in 0..16 {
            let offset = ChunkOffset::new(dx, dy, dz);
            let block = chunk.get_block(offset);
            let coord = chunk_coordinate.with_offset(offset);
            if block.equals_ignore_variant(self.off) {
                chunk.set_block(offset, self.on, None);
                for conn in SOURCE_CONNECTIVITIES {
                    if let Some(dest) = conn.eval(coord, 0) {
                        transitions.push((dest, coord, PinState::High));
                    }
                }
            } else if block.equals_ignore_variant(self.on) {
                chunk.set_block(offset, self.off, None);
                for conn in SOURCE_CONNECTIVITIES {
                    if let Some(dest) = conn.eval(coord, 0) {
                        transitions.push((dest, coord, PinState::Low));
                    }
                }
            }
        }}}

        // Transmit edges outside the chunk lock via run_deferred
        cctx.run_deferred(|ctx| {
            let cctx = make_root_context(ctx);
            for (dest, src, state) in transitions {
                transmit_edge(&cctx, dest, src, state)?;
            }
            Ok(())
        });
        Ok(())
    }
}

// Registration
builder.inner.add_timer(
    "myplugin_osc_1Hz",
    TimerSettings {
        interval: Duration::from_secs(1),
        shards: 16,
        spreading: 0.1,
        block_types: vec![oscillator_off.id, oscillator_on.id],
        ..Default::default()
    },
    TimerCallback::BulkUpdate(Box::new(OscillatorTimer {
        on: oscillator_on.id, off: oscillator_off.id,
    })),
);
```

Key points:
- Mutate the chunk **directly** inside `bulk_update_callback` (`chunk.set_block`).
- Collect transitions, then `run_deferred` to call `transmit_edge` after the lock is released.

Reference: `simple_blocks.rs` — `OscillatorTimerHandler`.

---

### 7. Bus message send and receive

Bus messages carry a `HashMap<String, String>` payload. The sender iterates its connectivity list
and calls `transmit_bus_message` for each neighbor, or just on the neighbors that are relevant (if
it should only be sent on some ports). The receiver implements `on_bus_message` and reads the keys
it cares about.

**Sending** (from a domain-logic handler, e.g. when a cart acquires a signal):

```rust
use perovskite_game_api::circuits::{BusMessage, events::{make_root_context, transmit_bus_message}};

fn emit_bus_event(ctx: &HandlerContext, source_coord: BlockCoordinate,
    connectivities: &[BlockConnectivity], variant: u16,
    payload: HashMap<String, String>) -> Result<()> {
    let message = BusMessage { sender: source_coord, data: payload };
    let cctx = make_root_context(ctx);
    for conn in connectivities {
        if let Some(dest) = conn.eval(source_coord, variant) {
            transmit_bus_message(&cctx, dest, source_coord, PinState::Low, message.clone())?;
        }
    }
    Ok(())
}

// Caller:
let mut data = HashMap::new();
data.insert("event_type".to_string(), "cart_passed".to_string());
data.insert("cart_id".to_string(), cart_id.to_string());
emit_bus_event(ctx, signal_coord, &SIGNAL_CONNECTIVITIES, block.variant(), data)?;
```

**Receiving** (in `CircuitBlockCallbacks::on_bus_message`):

```rust
fn on_bus_message(&self, ctx: &CircuitHandlerContext<'_>,
    coordinate: BlockCoordinate, from: BlockCoordinate, message: &BusMessage) -> Result<()> {
    // Guard against loopback (block receiving its own message)
    if from == coordinate { return Ok(()); }

    let cart_id = match message.data.get("cart_id").and_then(|s| s.parse::<u32>().ok()) {
        Some(id) => id,
        None => return Ok(()),
    };
    let decision = match message.data.get("decision").map(|s| s.as_str()) {
        Some("left")  => Route::Left,
        Some("right") => Route::Right,
        _             => return Ok(()),
    };

    ctx.game_map().mutate_block_atomically(coordinate, |_block, ext| {
        let data = ext.get_or_insert_with(Default::default);
        // store the decision in extended data for later use
        data.simple_data.insert("pending_decision".to_string(), format!("{:?}", decision));
        Ok(())
    })?;
    Ok(())
}
```

`PinState::Low` is the conventional pin state when the message is the only payload (not a digital
signal). The engine still routes the message through wires using the standard circuit topology.

Reference: `carts/interlocking.rs` — `send_signal_bus_message`; `carts/signals.rs` — `InterlockingSignalCircuitCallbacks::on_bus_message`.

---

### 8. Integrating circuit behavior with non-circuit domain logic (e.g. carts/signals)

A block does **not** need to look like a lamp or gate. It can have full interact popups, extended
data, and domain-specific logic while also participating in circuits. The circuit part is an
orthogonal layer:

```rust
// A block that is primarily controlled by carts but also listens on the circuit bus
builder.add_block(
    BlockBuilder::new(MY_SIGNAL_BLOCK)
        .set_axis_aligned_boxes_appearance(/* ... */)
        // Domain-specific extended data
        .add_modifier(|bt| {
            bt.interact_key_handler = Some(Box::new(|ctx, coord, _| { /* popup */ Ok(None) }));
            bt.deserialize_extended_data_handler = Some(Box::new(my_deserialize));
            bt.serialize_extended_data_handler   = Some(Box::new(my_serialize));
        })
        .add_interact_key_menu_entry("", "Properties")
        // Variant bits used for signal state flags, so disable placer tracking
        .set_extra_variant_func(Box::new(|_ctx, _coord, _stack, old_variant| {
            Ok(old_variant | VARIANT_RESTRICTIVE)  // set initial state on placement
        }))
        .force_disable_track_placer()
        // Circuit layer — injects dig/place hooks and CIRCUITS_GROUP
        .register_circuit_callbacks(),
)?;

// Circuit callbacks can be minimal if bus messages are the real mechanism
struct MySignalCallbacks;
impl CircuitBlockCallbacks for MySignalCallbacks {
    fn on_bus_message(&self, ctx: &CircuitHandlerContext<'_>,
        coordinate: BlockCoordinate, from: BlockCoordinate, message: &BusMessage) -> Result<()> {
        // Interpret message, update extended data
        Ok(())
    }
    // sample_pin, on_incoming_edge: default impls (Low / no-op)
}
```

Reference: `carts/signals.rs` — `register_single_signal`, `InterlockingSignalCircuitCallbacks`.