# Perovskite Game API Crate Source Map

High-level API for game content: block/item definition, optional features (circuits, farming, animals, autobuild).

## `game_builder` module
**File**: `/c/cuberef/perovskite_game_api/src/game_builder.rs`

Game and server initialization:
- `GameBuilder` — fluent builder for configuring and starting a game server
  - `testonly_in_memory()` — in-memory game for tests
  - `initialize_default_game()` — registers default game content
  - Block/item registration methods
  - Command registration
- Type aliases: `BlockName`, `ItemName`, `StaticItemName`, `FastBlockName`
- Server startup: spawning gRPC listener, block synchronization

**Key abstractions**: Builder pattern for game setup; separates configuration from execution.

## `blocks` module
**File**: `/c/cuberef/perovskite_game_api/src/blocks.rs`

High-level block definition API:
- `DroppedItem` enum — what a block yields when dug: None, Fixed, Dynamic (closure), NotDiggable
- `RotationMode` — how a block rotates based on player facing: by azimuth, by axis, etc.
- Block appearance builders:
  - `CubeAppearanceBuilder` — solid cube with texture + rotation support
  - `PlantLikeRenderInfo` — flora-style rendering
  - `CustomMeshRenderInfo` — custom geometry
- Interaction handlers: `InlineInteractionHandler` for block-specific dig/tap logic
- `BlockType` — internal definition (server-side API marked unstable)

Future work will add more config for blocks, from a game-behavior-centric point of view, rather than a low-level block-centric or callback-centric one.

**Key abstractions**: Variant encoding (rotation stored in variant bits); deferred builder pattern for appearance; closure-based item drops.

**Alternatives**:
* Use `add_modifier` as an escape hatch to get a perovskite_server::game_state::blocks level view onto a block and its handlers.
* Define blocks directly using perovskite_server::game_state::blocks.

## `items` module
**File**: `/c/cuberef/perovskite_game_api/src/items.rs`

High-level item definition API:
- `ItemActionTarget` enum — what blocks an item can target: Block, BlockGroup, Any
- `ItemAction` enum — what an item does: PlaceBlock, DigBlock, DoNothing
- `StackDecrement` enum — how stack decrements: Fixed, WearMultiplied
- `ItemHandler` — action definition per item state
- Item appearance builders: `ItemAppearanceBuilder`
- Quantity types: Stack (stackable) vs. Wear (tool durability)

Future work will add more config for blocks, from a game-behavior-centric point of view, rather than a low-level block-centric one.

**Key abstractions**: Item handlers are data-driven configurations, not closures; separate logic from appearance.

**Alternatives**: Define items directly using perovskite_server::game_state::items.


## `blocks::variants` submodule
Variant encoding utilities:
- `rotate_nesw_azimuth_to_variant()` — convert player facing to block rotation variant

## `default_game` module (feature-gated)
**File**: `/c/cuberef/perovskite_game_api/src/default_game/`

Default game content:
- Block types: dirt, stone, wood, ores, decorative blocks
- Items: tools (pickaxe, axe, shovel), resources (ingots, gems)
- Block groups: VARIANT_ENCODES_PLACER and other classification
- Map generator integration
- Tool wear and mining difficulty

**Key abstractions**: Hierarchical material progression; crafting prerequisites encoded as block/item availability.

## `circuits` module (feature-gated: `circuits`)
**File**: `/c/cuberef/perovskite_game_api/src/circuits/`

Digital logic circuits system:
- `register_circuits(game: &mut GameBuilder)` — initialization entry point
- Circuit block types: wires, logic gates (AND, OR, NOT, XOR), latches, microcontroller units
- Signal propagation: high/low on multi-bit busses
- Extended data: circuit state persisted in block extended data (protobuf-serialized)
- Interaction handlers: configuration UI for gates, microcontroller programming

**Key abstractions**: Circuit state as block extended data; event-driven update on signal change.

## `farming` module (feature-gated: `farming`)
**File**: `/c/cuberef/perovskite_game_api/src/farming/`

Crop growth system:
- `initialize_farming(game: &mut GameBuilder)` — registration entry point
- Crop types: wheat, vegetables
- Growth stages: seedling -> mature -> harvestable
- Extended data: growth state and timer
- Block timers: crop growth driven by per-block timers in game_map

**Key abstractions**: Crop state as block extended data; timer-based progression.

## `autobuild` module (feature-gated: `autobuild`)
**File**: `/c/cuberef/perovskite_game_api/src/autobuild/`

Automated building tools (WIP):
- `initialize_autobuild(game: &mut GameBuilder)` — registration entry point
- Autobuild items: tools for filling/clearing volumes, pattern brushes, undo
- Template system: store and replay building actions
- Undo mechanism: recent history of block changes

**Key abstractions**: Action history; bulk block operations; pattern templates.

## `animals` module (feature-gated: `animals`)
**File**: `/c/cuberef/perovskite_game_api/src/animals/`

Creature/mob system:
- `register_duck(game: &mut GameBuilder)` — example animal registration
- Entity types: duck, other mobs (future)
- Behavior: movement, AI, interactions
- Appearance: entity mesh/texture from assets

**Key abstractions**: Entity-based mobs; coroutine-driven AI (via `EntityManager`).

## `carts` module (feature-gated: `carts`)
**File**: `/c/cuberef/perovskite_game_api/src/carts/`

Minecart/rail system:
- `register_carts(game: &mut GameBuilder)` — registration entry point
- Rail block types: straight, curve, slope
- Cart entities: track following, passenger/cargo capacity
- Physics: gravity, friction, propulsion

**Key abstractions**: Entity-based movement constrained to rail geometry.

**Future work**: Automatic cart pathfinding, interoperation with circuits, automated railway stations

## `discord` module (feature-gated: `discord`)
**File**: `/c/cuberef/perovskite_game_api/src/discord/`

Discord integration:
- `connect(game: &mut GameBuilder)` — setup entry point
- Chat bridge: in-game messages <-> Discord webhook/bot
- Event webhook: player join/leave, milestones

**Key abstractions**: Async HTTP requests to Discord API.

## `colors` module
**File**: `/c/cuberef/perovskite_game_api/src/colors.rs`

Unified color palette and dye system:
- `register_dyes(game: &mut GameBuilder)` — dye item registration
- Dye colors: primary palette + variants
- Texture colorization: apply color overlay to base texture (single texture -> many color variants)

**Key abstractions**: Color palette abstraction; texture transformation for efficiency.

## `test_support` module (feature-gated: `test-support`)
Testing utilities for game content development.

## Top-level `configure_default_game()`
**File**: `/c/cuberef/perovskite_game_api/src/lib.rs`

Orchestrator function that conditionally registers all enabled features:
- Calls `game.initialize_default_game()`
- Conditionally calls `circuits::register_circuits()`, `farming::initialize_farming()`, etc.

**Key abstractions**: Feature coordination; single entry point for full game setup.

## Re-exports
- `BlockCoordinate`, `ChunkCoordinate`, `ChunkOffset` — from perovskite_core
- `constants` — from perovskite_core (for default block groups, textures)
- Server unstable API (gated by `unstable_api` feature) — access to `BlockTypeManager`, `ItemManager`, raw handlers

---

### Game Content Development Workflow

1. **Define block type**: Use `GameBuilder::register_block()` with `CubeAppearanceBuilder`
2. **Define item type**: Use `GameBuilder::register_item()` with appearance and handlers
3. **Add interaction**: Attach `ItemHandler` with action + target filter
4. **Extend behavior**: Provide custom `InlineInteractionHandler` closure for complex logic
5. **Persist state**: Store custom data as protobuf in block's `ExtendedData`
6. **Time-based events**: Use `game_map.schedule_timer()` for crop growth, decay, etc.

### Module Navigation Guidance

**For block definitions**: Start at `game_builder::GameBuilder::register_block()`, then `blocks::CubeAppearanceBuilder`.

**For item interactions**: See `items::ItemActionTarget` and `items::ItemHandler` for declarative approach.

**For circuits/farming/etc**: Find `register_*()` function and trace block/item registrations.

**For autobuild patterns**: Examine action history and template storage in autobuild module.

**For animals/mobs**: Check entity coroutine registration in `animals::register_duck()`.

**For extended data**: Look at circuit/farming modules for protobuf serialization examples.

**For colors**: Use `colors::register_dyes()` and texture colorization primitives.

**For feature coordination**: The `configure_default_game()` function shows how to selectively enable modules.
