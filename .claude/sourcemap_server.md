# Perovskite Server Crate Source Map

Core server implementation: game state management, database, networking, and game loop.

## `database` module
**File**: `/c/cuberef/perovskite_server/src/database/`

Abstraction layer over persistent storage with optional failure injection:
- `GameDatabase` trait — key-value interface: `get()`, `put()`, `delete()`, `flush()`, `read_prefix()`
- `KeySpace` enum — logical namespaces: Metadata, MapchunkData, Plugin, Inventory, Player, UserMeta, Entity, DisasterRecovery
- `rocksdb` submodule — RocksDB implementation
- `failure_injection` submodule (feature-gated) — chaos testing wrapper
- `InMemGameDatabase` — in-memory implementation for testing

**Key abstractions**: Typed key namespacing; optional chaos monkey for resilience testing.

## `game_state` module
**File**: `/c/cuberef/perovskite_server/src/game_state/` (directory with submodules)

Central game state orchestration and world simulation.

### `game_state::GameState` (main struct)
**File**: `/c/cuberef/perovskite_server/src/game_state/mod.rs`

Main server state container holding references to all subsystems:
- `map: Arc<ServerGameMap>` — world data and chunk storage
- `mapgen: Arc<dyn MapgenInterface>` — terrain generation
- `database: Arc<dyn GameDatabase>` — persistent storage
- `inventory_manager: InventoryManager` — non-embedded inventory storage
- `item_manager: ItemManager` — item type registry
- `player_manager: Arc<PlayerManager>` — connected player state
- `media_resources: MediaManager` — assets sent to clients
- `chat: Arc<ChatState>` — chat command dispatch
- `entities: Arc<EntityManager>` — entity lifecycle and behavior
- `time_state: Mutex<TimeState>` — in-game time (day/night cycle)
- `game_behaviors: GameBehaviors` — configurable game rules

**Key abstractions**: Arc-based central registry; shutdown coordination via `await_start_shutdown()`; per-crate extension storage via `TypeMap`.

### `game_state::blocks`
**File**: `/c/cuberef/perovskite_server/src/game_state/blocks.rs`

Block type registry and interaction system:
- `BlockTypeManager` — registry of all block definitions; lookup by `BlockId` or name
- `BlockType` — metadata for a block: appearance, physics, groups, interaction handlers
- `ExtendedData` — per-block-instance state: custom data (protobuf-serialized), key-value store, nested inventories
- `BlockInteractionResult` — output of dig/tap handler: items dropped + tool wear
- `InlineInteractionHandler` — closure type for custom block interactions
- Custom data system: `CustomData` trait with `Clone + Send + Sync + 'static`; protobuf serialization helpers

**Key abstractions**: Type-erased custom data (downcasting); handler-based interaction model; inventory embedding.

### `game_state::items`
**File**: `/c/cuberef/perovskite_server/src/game_state/items.rs`

Item type registry and usage:
- `ItemManager` — registry of all items; lookup by name
- `Item` — item definition: prototype + optional handlers for dig, tap, place_on_block, place_on_entity
- `ItemStack` — runtime instance: item name, quantity or wear, embedded metadata
- `ItemInteractionResult` — handler output: updated stack + obtained items
- Handler types: `BlockInteractionHandler`, `EntityInteractionHandler`, `PlaceHandler`
- `PointeeBlockCoords` — raycast hit info: selected block and optional preceding block

**Key abstractions**: Pluggable interaction handlers per item; tool wear vs. stack quantity distinction.

### `game_state::game_map`
**File**: `/c/cuberef/perovskite_server/src/game_state/game_map.rs`

Map chunk storage, update coalescing, and lighting:
- `ServerGameMap<S: SyncBackend>` — chunk cache and database interface
- `MapChunk` — 16×16×16 block data + lighting + extended data
- `CasOutcome::Match` / `Mismatch` — compare-and-swap result for atomic updates
- Light propagation: `Lightfield`, `propagate_light()`
- Update coalescing: bulk update callbacks, timer-based handlers
- Templates: in-memory chunk patterns for paste/undo operations

**Key abstractions**: Generic over `SyncBackend` (thread-safe or single-threaded); lazy chunk loading from DB; light update coalescing for performance.

### `game_state::game_behaviors`
Game configuration (day length, difficulty, default behaviors).

### `game_state::event`
**File**: `/c/cuberef/perovskite_server/src/game_state/event.rs`

Event context and handler infrastructure:
- `EventInitiator<'a>` enum: `Engine`, `Player(PlayerInitiator)`, `WeakPlayerRef`, `Plugin(String)`
- `HandlerContext` — provides access to `GameState`, initiator, and location for handler closures
- Handler macros: `run_handler!()`, `run_async_handler!()` with panic catching and tracing

**Key abstractions**: Distinguishing player-driven vs. engine-driven vs. plugin-driven events; weak refs to avoid shutdown deadlocks.

### `game_state::player`
**File**: `/c/cuberef/perovskite_server/src/game_state/player.rs`

Player state and management:
- `Player` — user session: name, position, inventory key, permissions, entity ID
- `PlayerManager` — registry of connected players; database persistence
- `PlayerState` (internal) — mutable state: position, inventory, permissions, extended data
- Position tracking: fast seqlock-based read, slower mutex-protected write
- Inventory views: `InventoryView`, `InventoryViewWithContext`
- Popups: `Popup` for client dialogs

**Key abstractions**: Weak reference to `GameState` prevents shutdown deadlock; seqlock for lock-free position reads.

### `game_state::entities`
**File**: `/c/cuberef/perovskite_server/src/game_state/entities.rs`

Entity lifecycle and coroutine management:
- `EntityManager` — entity registry and coroutine scheduler
- Entity types: `EntityTypeManager`, `EntityTypeAssignment` (wire proto)
- Control flags: CONTROL_PRESENT, CONTROL_AUTONOMOUS, CONTROL_SIMPLE_PHYSICS, etc.
- Coroutines: `DeferredPrivate<T>`, `Completion<T>` for async work
- Appearance: `EntityAppearance` (protobuf)

**Key abstractions**: Sharded entity storage; coroutine-driven behavior; move queue for locomotion.

### `game_state::inventory`
Inventory storage and view management (referenced in player and extended data).

### `game_state::client_ui`
Client UI primitives: `Popup` dialogs.

### `game_state::chat`
Chat and command dispatch infrastructure.

### `game_state::handlers`
**File**: `/c/cuberef/perovskite_server/src/game_state/handlers.rs`

Handler execution utilities:
- `run_handler_impl()` — wraps closures with panic catching
- `CoalesceResult` trait — combining results from multiple handlers
- Macros: `run_handler!()`, `run_async_handler!()`

## `network_server` module
**File**: `/c/cuberef/perovskite_server/src/network_server/` (directory)

gRPC service implementation:
- `grpc_service` — `PerovskiteGameServerImpl` implementing the tonic-generated service
- `auth` — authentication (OPAQUE) integration for login
- `client_context` — per-client state tracking

**Key abstractions**: gRPC-over-HTTP/2 using tonic; one handler per connected client.

## `server` module
**File**: `/c/cuberef/perovskite_server/src/server.rs`

Server initialization and configuration:
- `ServerArgs` — CLI arguments: data_dir, bind_addr, port, database tuning, profiling
- Server builders: `with_tempdir()` for testing, `in_memory()` for transient servers
- Main entry point orchestration: database init, block/item registration, game setup

**Key abstractions**: Statically-linked binary; single monolith combining engine + game content + assets.

## `media` module
**File**: `/c/cuberef/perovskite_server/src/media.rs`

Asset/resource management:
- `MediaManager` — registry of textures, sounds, meshes sent to clients
- `SoundKey(u32)` — opaque identifier for registered audio samples
- `Resource` — holds data (owned bytes or file path) + SHA256 hash
- `SampledAudioDetails` — metadata for audio resources

**Key abstractions**: Lazy loading; hash-based deduplication; sampled audio vs. other resources.

## `formats` module
**File**: `/c/cuberef/perovskite_server/src/formats.rs`

Format conversions:
- `load_obj_mesh()` — OBJ to `CustomMesh` (Vulkan coordinate system conversion)

**Key abstractions**: Vertex format transformation; UV and normal coordinate flipping for engine conventions.

## `mapgen` module
**File**: `/c/cuberef/perovskite_server/src/game_state/mapgen.rs`

Map generation interface (plugin point):
- `MapgenInterface` trait — procedural world generation
- `FarMeshPoint` — distant terrain outline points

**Key abstractions**: Pluggable terrain generator; supports WIP features.

---

### Cross-Module Data Flow

1. **Game loop**: `GameState::run()` -> block updates -> light propagation -> entity coroutines -> network updates
    * This is a misnomer: rather than a loop, this is primarily event-driven, with timers driven independently of a precise tick system.
2. **Player interaction**: gRPC request -> `HandlerContext` -> item/block handler -> map update -> notification to all players
3. **Chunk loading**: Client requests chunk -> DB lookup -> light calculation -> protobuf encoding -> network send
4. **Inventory**: `InventoryKey` -> `InventoryManager` -> `Inventory` structure with metadata

### Module Navigation Guidance

**To understand player actions**: Start at `game_state::event::HandlerContext`, then `game_state::items::Item` for handler signatures.

**To track block updates**: `game_state::game_map::ServerGameMap::set_block()` or `dig_block()`, then light propagation.

**To add custom block data**: Define protobuf message, register with `register_proto_serialization_handlers()` in `BlockType`.

**To hook entity behavior**: Examine `EntityManager` coroutine system and `EntityTypeManager` for handler registration.

**To access player state**: Use `Player::last_position()`, `main_inventory()`, or acquire `PlayerState` lock for mutations.

**For database operations**: All persistent data flows through `GameDatabase` trait via `KeySpace` prefixes.
