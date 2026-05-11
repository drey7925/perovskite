---
name: mod_structure
description: How to structure a new mod or subsystem in perovskite_game_api — registration pattern, GameBuilder/GameState extensions, BlockBuilder extension traits, and cross-mod ecosystem hooks.
---

This guide covers the idiomatic patterns for structuring a new mod (feature module) in `perovskite_game_api`. Use it when adding a new self-contained subsystem or when wiring one mod into another's ecosystem.

## The Primary Registration Function

Every mod exposes a single top-level entry point, called from `configure_default_game()` in `lib.rs`:

```rust
// perovskite_game_api/src/mycrate/mod.rs
pub fn register_mycrate(game: &mut GameBuilder) -> Result<()> {
    // 1. Register textures
    include_texture_bytes!(game, MY_TEX, "textures/my_tex.png")?;

    // 2. Register blocks, items, entities
    let my_block = game.add_block(BlockBuilder::new(MY_BLOCK_NAME)...)?;

    // 3. Populate builder extension with IDs needed at runtime
    let ext = game.builder_extension_mut::<MycrateBuilderExt>();
    ext.my_block = my_block.id;

    Ok(())
}
```

Feature-gate the call in `lib.rs`:

```rust
#[cfg(feature = "mycrate")]
{
    mycrate::register_mycrate(game)?;
}
```

## GameBuilder Extensions and GameState Extensions

This is the pattern for storing block/entity IDs and other config at build time and making them available at runtime in handlers.

### Define the extension struct

One struct can serve as both the builder-time accumulator and the runtime state if it's `Clone` (see carts); or use two separate types if the runtime state differs significantly (see circuits, which uses `CircuitGameBuilderExt` + `CircuitGameStateExtension`).

```rust
// Builder-time + runtime state (simple case, like farming and carts):
#[derive(Clone, Default)]
struct MycrateGameBuilderExt {
    my_block: BlockId,
    my_entity: EntityClassId,
    // etc.
}

impl GameBuilderExtension for MycrateGameBuilderExt {
    fn pre_run(&mut self, server_builder: &mut ServerBuilder) {
        // Transfer to runtime — clone self if it IS the runtime state.
        server_builder.add_extension(self.clone());
    }
}

// Mark it accessible via ctx.extension::<MycrateGameBuilderExt>()
impl GameStateExtension for MycrateGameBuilderExt {}
```

For more complex cases where the runtime type differs (see `circuits`), use an `Option<RuntimeState>` inside the builder extension:

```rust
#[derive(Default)]
struct MycrateBuilderExt {
    state: Option<MycrateRuntimeState>,
}

impl GameBuilderExtension for MycrateBuilderExt {
    fn pre_run(&mut self, server_builder: &mut ServerBuilder) {
        let state = self.state.take().expect("pre_run called twice");
        server_builder.add_extension(state);
    }
}

impl Default for MycrateBuilderExt {
    fn default() -> Self {
        Self { state: Some(MycrateRuntimeState::default()) }
    }
}
```

### Access during registration

```rust
let ext = game.builder_extension_mut::<MycrateBuilderExt>();
ext.my_block = registered_block_id;
```

`builder_extension_mut` auto-creates via `Default` on first call — no manual init needed.

### Access at runtime (inside handlers)

```rust
let ext = ctx.extension::<MycrateGameBuilderExt>()?;
let my_block_id = ext.my_block;
```

## BlockBuilder Extension Traits

When your mod needs to imbue blocks with behavior from its subsystem, define an extension trait on `BlockBuilder`. This lets other mods opt their blocks into your ecosystem with a single method call.

Pattern from `circuits/mod.rs`:

```rust
// In your mod's pub API:
pub trait MycrateBlockBuilder {
    fn register_mycrate_behavior(self) -> BlockBuilder;
}

impl MycrateBlockBuilder for BlockBuilder {
    fn register_mycrate_behavior(self) -> BlockBuilder {
        self
            .add_block_group(MYCRATE_BLOCK_GROUP)
            .add_modifier(|bt| {
                // Wrap or set handlers; preserve old handlers:
                let old_dig = bt.dig_handler_full.take();
                bt.dig_handler_full = Some(Box::new(move |ctx, coord, tool| {
                    let result = old_dig.as_deref()
                        .map(|h| h(ctx, coord, tool))
                        .transpose()?
                        .unwrap_or_default();
                    // ... add your subsystem logic here
                    Ok(result)
                }));
            })
    }
}
```

This is consumed as:

```rust
game.add_block(
    BlockBuilder::new(SOME_BLOCK)
        .set_cube_single_texture(TEX)
        .register_mycrate_behavior()  // opt this block into your subsystem
)?;
```

**Note**: `add_modifier` closures run just before block registration.

## Ecosystem Extension Points (Cross-Mod Integration)

A mod can expose helper functions that let other mods register their blocks/items/entities into the first mod's ecosystem. This avoids hard dependencies in either direction.

### Pattern 1: define_* functions (circuits model)

The circuits module exposes:

```rust
pub trait CircuitGameBuilder {
    fn define_circuit_callbacks(
        &mut self,
        block_id: BlockId,
        callbacks: Box<dyn CircuitBlockCallbacks>,
        properties: CircuitBlockProperties,
    ) -> Result<()>;
}

impl CircuitGameBuilder for GameBuilder {
    fn define_circuit_callbacks(...) -> Result<()> {
        register_circuits(self)?;  // Lazy/idempotent init
        let ext = self.builder_extension_mut::<CircuitGameBuilderExt>();
        let state = ext.resulting_state.as_mut().expect("pre_run already called");
        state.callbacks.insert(block_id.base_id(), callbacks);
        state.basic_properties.insert(block_id.base_id(), properties);
        Ok(())
    }
}
```

The lazy `register_circuits(self)?` at the top makes calling order irrelevant — the dependent mod doesn't need to know whether circuits was initialized before it.

### Pattern 2: register_* helper functions (machines model)

```rust
// In autobuild/machines.rs:
pub fn register_machine_type(
    game_builder: &mut GameBuilder,
    block_builder: BlockBuilder,
    def: impl Into<MachineDef>,
) -> Result<()> {
    let ext = game_builder.builder_extension_mut::<MachinesBuilderExt>();
    let block_types = ext.block_types.as_mut().context("pre_run already called")?;
    // Register and store
    let block = game_builder.add_block(block_builder.add_block_group(AUTOBUILD_MACHINES_GROUP))?;
    block_types.insert(block.id.base_id(), def.into());
    Ok(())
}
```

External mods call this to plug in their machine type without touching internals.

### Cross-mod usage example (carts/signals.rs using circuits)

`carts` depends on `circuits` by calling its public API during registration:

```rust
// In carts/signals.rs register_signal_blocks():
circuits::register_circuits(game_builder)?;   // idempotent init

let signal_block = game_builder.add_block(
    BlockBuilder::new(SIGNAL_BLOCK)
        // ... appearance ...
        .register_circuit_callbacks()           // BlockBuilder extension trait
)?;

game_builder.define_circuit_callbacks(         // ecosystem extension point
    signal_block.id,
    Box::new(SignalCircuitCallbacks { ... }),
    CircuitBlockProperties { connectivity: vec![...] },
)?;
```

## The Autobuild Sharp Edge

`autobuild` uses a `BatchedWrite` type that is **consumed by value** when committed:

```rust
pub fn commit(mut self, ctx: &HandlerContext, params: WriteParameters) -> Result<BatchedUndo>
```

This is intentional — once you call `commit()`, the batch is gone. The typical usage:

```rust
let mut batch = BatchedWrite::new();
batch.add_block(coord, block_id);
// ... more adds ...
let undo = batch.commit(ctx, write_params)?;
// batch is now moved/dropped; cannot be reused
```

If you need to add blocks conditionally and commit later, either collect into a `Vec<(BlockCoordinate, BlockId)>` first, or commit multiple separate batches.

The `Autobuilder` trait (for tool types) uses generic associated types for persistent and transient state, wired up through `configure_item::<T>(item)`.

## Idempotency Pattern

Registration functions called as ecosystem dependencies should be idempotent. The circuits module does this with a flag:

```rust
pub fn register_circuits(builder: &mut GameBuilder) -> Result<()> {
    let ext = builder.builder_extension_mut::<CircuitGameBuilderExt>();
    if ext.initialized {
        return Ok(());
    }
    ext.initialized = true;
    // ... register blocks, groups, etc.
}
```

## Summary of Key Files

| Pattern | Primary Example File |
|---|---|
| Registration function | `circuits/mod.rs` `register_circuits()`, `carts/mod.rs` `register_carts()` |
| GameBuilderExtension | `farming/mod.rs` `FarmingGameStateExtension`, `carts/mod.rs` `CartsGameBuilderExtension` |
| Separate builder/runtime types | `circuits/mod.rs` `CircuitGameBuilderExt` / `CircuitGameStateExtension` |
| BlockBuilder extension trait | `circuits/mod.rs` `CircuitBlockBuilder` trait + `impl CircuitBlockBuilder for BlockBuilder` |
| Ecosystem extension function | `circuits/mod.rs` `CircuitGameBuilder::define_circuit_callbacks()` |
| Cross-mod use of ecosystem | `carts/signals.rs` calling `circuits::` APIs |
| register_* helper | `autobuild/machines.rs` `register_machine_type()` |
| Value-consuming API sharp edge | `autobuild/mod.rs` `BatchedWrite::commit(self, ...)` |