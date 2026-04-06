# Perovskite Source Map — Workspace Overview

Complete AI-navigational map of the Perovskite voxel game engine workspace. A statically-linked monolithic server (engine + game content) with a thin Vulkan client.

## Workspace Structure

```
perovskite/
├── perovskite_core/        Shared types & protocols (gRPC)
├── perovskite_server/      Game server (game state, networking, database)
├── perovskite_client/      Vulkan client (rendering, UI, audio)
└── perovskite_game_api/    Content APIs (blocks, items, circuits, farming, autobuild, etc.)
```

## High-Level Architecture

### Server (Monolithic)
- **Engine core** (`perovskite_server`)
  - `GameState` — central coordinator holding map, players, entities, inventory, chat
  - `ServerGameMap` — chunk storage with lazy loading and light propagation
  - `BlockTypeManager`, `ItemManager` — type registries
  - `PlayerManager` — connected player tracking
  - `EntityManager` — entity lifecycle and coroutine-driven behavior
  - `ChatState` — command dispatch
  - `InventoryManager` — persistent inventory storage (outside of chunks)
  - `MediaManager` — asset (texture/sound) registry

- **Networking** (`network_server`)
  - tonic gRPC service: `PerovskiteGameServerImpl`
  - HTTP/2 bidirectional streams for game events
  - Per-client `ClientContext` state tracking

- **Database** (`database`)
  - Abstraction trait `GameDatabase` with KeySpace namespacing
  - RocksDB implementation (real-server feature)
  - In-memory implementation (testing)
  - Optional failure injection for chaos testing

- **Game Content** (`perovskite_game_api`)
  - Block/item registration APIs
  - Default game: procedural terrain, mining, resources, crafting
  - Optional modules: circuits, farming, animals, carts, autobuild, Discord
  - Dye/color system with texture colorization

### Client (Thin)
- **Rendering** (`vulkan`)
  - Vulkan + winit
  - GPU-resident chunk meshes
  - Terrain far-field LOD
  - Entity rendering
  - HUD/UI overlay

- **Networking** (`net_client`)
  - gRPC client streaming to server
  - Request/response coalescing
  - Chunk/entity update reception

- **Client State** (`client_state`)
  - Local cache: chunks, entities, inventory
  - Optimistic updates for latency hiding

- **UI** (`game_ui`, `main_menu`)
  - Inventory, hotbar, chat, settings
  - Server connection dialog
  - Popup dialogs (received from server)

- **Audio** (`audio`, `media`)
  - Sound playback
  - Asset loading (textures, samples, meshes)

### Shared (`perovskite_core`)
- **Coordinate systems**: `BlockCoordinate`, `ChunkCoordinate`, `ChunkOffset`
- **Block IDs**: `BlockId` with variant encoding (12-bit variants, 20-bit base index)
- **Lighting**: `Lightfield` (bitfield), light propagation algorithm
- **Protocols**: tonic-generated gRPC modules (`perovskite::protocol::*`)
- **Chat**: `ChatMessage` with origin and color
- **Auth**: OPAQUE cipher suite (Ristretto255 + Argon2)
- **Constants**: block groups, textures, item rules, permissions
- **Sync**: thread-safe or single-threaded backend abstraction

## Key Abstractions & Patterns

### 1. Block Extended Data
Blocks can store custom state beyond variant bits:
- **ExtendedData** struct: custom data (protobuf-serialized, type-erased), key-value store, nested inventories
- Used by: circuits (logic state), farming (growth stage), custom mechanisms
- Persistence: serialized to database, sent to client on chunk load

### 2. Event Handling System
Interactions are driven by handlers attached to blocks/items:
- **EventInitiator** enum: `Engine`, `Player(PlayerInitiator)`, `WeakPlayerRef`, `Plugin`
- **HandlerContext**: provides access to `GameState`, map, position, initiator
- **InlineInteractionHandler**: closure-based block interaction (`(&HandlerContext, &mut BlockId, &mut ExtendedData, &mut ItemStack) -> Result<BlockInteractionResult>`)
- **ItemHandler**: data-driven action + target + stack decrement rule

### 3. Chunk Storage & Lighting
- **MapChunk**: 16×16×16 blocks with extended data per block + lighting
- **Lightfield**: compact bitfield (256 bits = 16×16 XZ positions per Y level)
- **Light propagation**: coalesced updates; changes to one block trigger neighbor propagation
- **Lazy loading**: chunks loaded from database on first access, cached in memory

### 4. Entity Coroutines
- **EntityManager**: sharded entity storage with coroutine scheduler
- **DeferredCall**: async work deferred to tokio executor
- **Completion**: result delivered back to entity coroutine
- Control flags: PRESENT, AUTONOMOUS, SIMPLE_PHYSICS, SUSPENDED, DEQUEUE_WHEN_READY
- Used by: animals, carts, future custom NPCs

### 5. Inventory System
- **Inventory**: collection of stacks with persistent key
- **InventoryKey**: identifier for persistent inventory (player main inv, chest contents, etc.)
- **InventoryView**: UI representation of inventory for client
- **ItemStack**: item + quantity (or wear for tools) + metadata
- Persistence: stored separately in database under `KeySpace::Inventory`

### 6. Game Content Registration
- **GameBuilder**: fluent API for defining blocks, items, entities
- Block appearance builders: `CubeAppearanceBuilder`, `PlantLikeRenderInfo`, custom mesh
- Item appearance + handlers: declarative action target + effect + stack change
- Feature-gated modules: circuits, farming, animals, autobuild coordinated via `configure_default_game()`

### 7. Client–Server Synchronization
- **Optimistic updates**: client modifies local state immediately, server confirms
- **Chunk streaming**: server sends chunks on-demand as player explores
- **Entity interpolation**: client interpolates between position updates from server
- **Popups**: server-driven UI (inventory, crafting, dialogs)

### 8. Type Erasure & Extensions
- **GameState::extensions**: `TypeMap` for per-crate extensibility
- **CustomData**: type-erased block state (downcasting at runtime)
- **Entity::Any**: extensible entity behavior storage

## Core Data Structures

| Structure | Purpose | Location |
|-----------|---------|----------|
| `BlockId(u32)` | Opaque block identifier (variant in lower 12 bits) | perovskite_core::block_id |
| `BlockCoordinate` | 3D integer position (can convert to/from chunk + offset) | perovskite_core::coordinates |
| `ChunkCoordinate` | 16³ chunk position | perovskite_core::coordinates |
| `MapChunk` | 16³ block data + extended data + lighting | perovskite_server::game_state::game_map |
| `ExtendedData` | Custom state for a block: protobuf + key-value + inventories | perovskite_server::game_state::blocks |
| `ItemStack` | Item + quantity/wear + metadata | perovskite_server::game_state::items |
| `Inventory` | Collection of ItemStacks with persistence key | perovskite_server::game_state::inventory |
| `Player` | Name, position, inventory, permissions, entity ID | perovskite_server::game_state::player |
| `Entity` | Type, position, appearance, coroutine state | perovskite_server::game_state::entities |
| `GameState` | Central coordinator (map, players, entities, database, etc.) | perovskite_server::game_state |

## Network Protocol

**Transport**: gRPC over HTTP/2 (tonic)
**Language**: Protobuf 3

### Service: `PerovskiteGameServer`
Methods (bidirectional streaming):
- `connect_to_game(client_id) -> stream GameStreamMessage` — main game loop stream
  - Client sends: player actions (dig, place, move), chat, etc.
  - Server sends: chunk updates, entity state, chat, popups

### Message Types
- **BlockTypeDef** — block appearance, physics, groups
- **MapChunk** — chunk data (block IDs, extended data serialized, lighting)
- **EntityTypeAssignment** — entity type metadata
- **ItemDef** — item appearance, interaction rules
- **ChatMessage** — origin, text, color
- **Popup** — UI dialog (inventory view, crafting grid, etc.)
- **InventoryAction** — player inventory interaction

### Coordinate Transmission
- `WireBlockCoordinate(x, y, z)` — individual block position
- `WireChunkCoordinate(x, y, z)` — chunk position
- Converted to/from Rust types at boundary

## Important Patterns to Know

### 1. Shutdown Coordination
`GameState` held in `Arc`; reference count used to detect shutdown. Holders must:
- Await `GameState::await_start_shutdown()` in tokio select
- Drop Arc on shutdown to unblock server cleanup
- Database flushed when last Arc is dropped

### 2. Handler Execution
Handlers wrapped in `run_handler!()` or `run_async_handler!()` macros:
- Panic catching (closure failures don't crash server)
- Tracy tracing for profiling
- Error logging and propagation

### 3. Permission Checks
`EventInitiator::check_permission_if_player(permission: &str) -> bool`
- Players: checked against their permission set
- Engine/plugins: always allowed
- Used in chat commands, actions

### 4. Block Update Coalescing
Multiple block updates batched and propagated together:
- Lighting updates coalesced per chunk
- Callbacks fired at end of update cycle
- Reduces redundant calculations

### 5. Optional Feature Coordination
Game content modules enabled via Cargo features:
- `circuits`, `farming`, `animals`, `carts`, `autobuild`, `discord`
- Each registers blocks/items via `initialize_*()` / `register_*(game: &mut GameBuilder)`
- Coordinated in `configure_default_game()` top-level function

## Debugging & Profiling

- **Tracy integration**: `tracy_client` crate; GUI profiler for CPU/GPU
- **Tracing**: structured logging with `tracing` crate
- **Failure injection**: optional `db_failure_injection` feature for chaos testing
- **Trace buffers**: circular event logs per handler for forensics

## Next Steps for Navigation

1. **To add a new block type**: See `perovskite_game_api::game_builder::GameBuilder::register_block()` and example in default_game
2. **To add a new item**: See `perovskite_game_api::items::ItemHandler` and `GameBuilder::register_item()`
3. **To hook block interactions**: Provide `InlineInteractionHandler` closure to block definition
4. **To add custom entity behavior**: See `EntityManager` coroutine system and `register_entity_type()`
5. **To persist block data**: Define protobuf message in perovskite_server, use `register_proto_serialization_handlers()`
6. **To add chat command**: See `ChatState::register_command()` in `game_state::chat`
7. **To render custom geometry**: Use `CustomMesh` in block appearance or entity mesh
8. **To add audio**: Register with `MediaManager` and trigger via handler
9. **For performance**: Profile with Tracy; check light propagation coalescing and chunk mesh generation
10. **For client rendering**: Examine `vulkan::game_renderer` pipeline and shader code

## Crate Dependencies Summary

- **perovskite_core**: (minimal, foundation)
  - tokio, tonic, prost (gRPC)
  - cgmath (vector math)
  - parking_lot (synchronization)
  
- **perovskite_server**: (core engine)
  - perovskite_core
  - rocksdb (real-server feature), in-memory DB (testing)
  - tokio, tonic (async runtime + gRPC)
  - various game-dev deps (rand, bitvec, etc.)

- **perovskite_game_api**: (content)
  - perovskite_core, perovskite_server (unstable_api feature)
  
- **perovskite_client**: (rendering)
  - perovskite_core
  - vulkano (Vulkan bindings)
  - winit (windowing + input)
  - tonic (gRPC client)
  - cpal (audio)
