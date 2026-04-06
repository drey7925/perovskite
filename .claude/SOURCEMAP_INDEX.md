# Perovskite Sourcemap Index

Complete navigation guide for the Perovskite voxel game engine codebase. Four crates + detailed cross-crate reference.

## Documents

### 1. [`sourcemap_overview.md`](./sourcemap_overview.md) — START HERE
**High-level workspace architecture & key abstractions**

- Workspace structure and crate organization
- Server, client, and shared subsystem overview
- 8 core abstraction patterns (extended data, events, chunks, entities, inventory, content registration, sync, type erasure)
- Network protocol summary
- Important patterns (shutdown, handlers, permissions, coalescing, features)
- Quick navigation links

**Use this to**: Understand the big picture, find which crate handles a feature, learn key data structures.

### 2. [`sourcemap_core.md`](./sourcemap_core.md)
**perovskite_core: Shared types and protocols**

- `coordinates` — `BlockCoordinate`, `ChunkCoordinate`, `ChunkOffset`
- `block_id` — `BlockId` with variant encoding, block registry
- `lighting` — `Lightfield` bitfield, light propagation
- `auth` — OPAQUE cryptography (Ristretto255 + Argon2)
- `chat` — `ChatMessage` with color metadata
- `protocol` — tonic-generated gRPC modules (blocks, map, game_rpc, items, entities, ui, audio, etc.)
- `constants` — block groups, textures, item rules
- Utility modules: sync, util, far_sheet, time, game_actions, items

**Use this to**: Understand core types, find coordinate conversion helpers, see protocol message types.

### 3. [`sourcemap_server.md`](./sourcemap_server.md)
**perovskite_server: Game engine and state management**

- **database** — `GameDatabase` trait, `KeySpace` namespacing, RocksDB + in-memory backends
- **game_state** — Central `GameState` orchestrator + 9 subsystems:
  - `blocks` — `BlockTypeManager`, `BlockType`, `ExtendedData` (custom block state), interaction handlers
  - `items` — `ItemManager`, `Item`, `ItemStack`, item handlers (dig, tap, place)
  - `game_map` — `ServerGameMap`, `MapChunk`, lighting, update coalescing, templates
  - `player` — `Player`, `PlayerManager`, position tracking, permissions
  - `entities` — `EntityManager`, coroutine-driven behavior, appearance
  - `event` — `EventInitiator` enum, `HandlerContext`, handler macros
  - `chat` — Command dispatch
  - `inventory` — Inventory storage and views
  - `client_ui` — Popups
- **network_server** — gRPC service, auth, per-client context
- **server** — `ServerArgs`, initialization, CLI
- **media** — Asset registry (textures, sounds, meshes)
- **formats** — Format conversions (OBJ to CustomMesh)
- **mapgen** — Map generation interface (plugin point)

**Use this to**: Find where game state lives, understand block/item/player/entity mechanics, hook into game events.

### 4. [`sourcemap_client.md`](./sourcemap_client.md)
**perovskite_client: Rendering, UI, and networking**

- **client_main** — winit event loop, Tracy profiler integration
- **vulkan** — Vulkan renderer, `GameApplication`, frame loop, chunk mesh generation, entity rendering, UI overlay
- **client_state** — Local cache: chunks, entities, inventory (eventual consistency with server)
- **net_client** — gRPC client, bidirectional streaming, request/response coalescing
- **game_ui** — HUD, inventory, chat, popups
- **main_menu** — Server connection, settings, character selection
- **media** — Asset loading (textures, sounds, meshes), cache, hash validation
- **audio** — Sound playback and mixing

**Use this to**: Understand rendering pipeline, find UI systems, see how client syncs with server.

## Quick Navigation by Task

### Adding a new block type
1. Start: [`sourcemap_game_api.md`](./sourcemap_game_api.md) → `game_builder::GameBuilder::register_block()`
2. See: `blocks::CubeAppearanceBuilder` for appearance configuration
3. Reference: `default_game` module for examples
4. Persist data: use `ExtendedData` + protobuf (see `sourcemap_server.md` → `game_state::blocks`)

### Adding a new item
1. Start: [`sourcemap_game_api.md`](./sourcemap_game_api.md) → `items::ItemHandler`
2. Define: `ItemActionTarget` (what block type), `ItemAction` (place/dig), `StackDecrement` rule
3. Register: `GameBuilder::register_item()`
4. Reference: `perovskite_game_api::default_game` for tool examples

### Understanding block interactions
1. Overview: [`sourcemap_overview.md`](./sourcemap_overview.md) → "Event Handling System" pattern
2. Handler types: [`sourcemap_server.md`](./sourcemap_server.md) → `game_state::event` and `game_state::items`
3. Custom block logic: `InlineInteractionHandler` closure in `game_state::blocks`
4. Permission checks: `EventInitiator::check_permission_if_player()`

### Implementing custom block state (circuits, farming, etc.)
1. Overview: [`sourcemap_overview.md`](./sourcemap_overview.md) → "Block Extended Data" pattern
2. Define: protobuf message for state
3. Register: `register_proto_serialization_handlers()` in block definition
4. Access: `ExtendedData::custom_data`, downcasting to concrete type
5. Examples: [`sourcemap_game_api.md`](./sourcemap_game_api.md) → `circuits` and `farming` modules

### Adding entity behavior (animals, NPCs, etc.)
1. Overview: [`sourcemap_overview.md`](./sourcemap_overview.md) → "Entity Coroutines" pattern
2. Types: [`sourcemap_server.md`](./sourcemap_server.md) → `game_state::entities::EntityManager`
3. Register: `EntityTypeManager`, define appearance and behavior
4. Coroutines: `DeferredCall` for async work, `Completion` for results
5. Example: [`sourcemap_game_api.md`](./sourcemap_game_api.md) → `animals::register_duck()`

### Rendering custom geometry
1. Block custom mesh: [`sourcemap_game_api.md`](./sourcemap_game_api.md) → `blocks::CustomMeshRenderInfo`
2. Entity custom mesh: See entity appearance in [`sourcemap_server.md`](./sourcemap_server.md) → `game_state::entities`
3. Client rendering: [`sourcemap_client.md`](./sourcemap_client.md) → `vulkan::game_renderer`
4. Asset format: [`sourcemap_server.md`](./sourcemap_server.md) → `formats::load_obj_mesh()`

### Hooking network messages
1. Protocol: [`sourcemap_core.md`](./sourcemap_core.md) → `protocol` module (tonic-generated)
2. Server handler: [`sourcemap_server.md`](./sourcemap_server.md) → `network_server::grpc_service`
3. Client receiver: [`sourcemap_client.md`](./sourcemap_client.md) → `net_client`
4. State update: [`sourcemap_client.md`](./sourcemap_client.md) → `client_state`

### Understanding persistence
1. Database layer: [`sourcemap_server.md`](./sourcemap_server.md) → `database` module
2. KeySpace namespaces: Metadata, MapchunkData, Plugin, Inventory, Player, UserMeta, Entity, DisasterRecovery
3. Block data: stored in MapChunk + extended data
4. Player data: inventory, permissions, extended data in Player KeySpace
5. Custom data: protobuf serialization in `game_state::blocks`

### Debugging a performance issue
1. Profiler: Use Tracy (integrated in both server and client)
2. Light propagation: [`sourcemap_server.md`](./sourcemap_server.md) → `game_map::light propagation coalescing`
3. Chunk mesh generation: [`sourcemap_client.md`](./sourcemap_client.md) → `vulkan::game_renderer`
4. Network sync: [`sourcemap_client.md`](./sourcemap_client.md) → `net_client` message batching
5. Handler execution: [`sourcemap_server.md`](./sourcemap_server.md) → `game_state::handlers::run_handler_impl()`

### Adding a chat command
1. Command registration: [`sourcemap_server.md`](./sourcemap_server.md) → `game_state::chat::ChatState`
2. Handler: `HandlerContext` and `EventInitiator` for permission checking
3. Response: Send message via `EventInitiator::send_chat_message()`
4. Example: Default game chat commands in perovskite_game_api

### Understanding audio
1. Server-side: [`sourcemap_server.md`](./sourcemap_server.md) → `media::MediaManager`, register sound
2. Trigger: Item/block handler calls game event to queue audio
3. Client-side: [`sourcemap_client.md`](./sourcemap_client.md) → `media` (asset loading), `audio` (playback)
4. Sample format: PCM, provided via server resource with SHA256 hash

### Modifying a feature-gated module
1. Feature list: `circuits`, `farming`, `animals`, `carts`, `autobuild`, `discord`
2. Module location: [`sourcemap_game_api.md`](./sourcemap_game_api.md) → each feature section
3. Registration: Module's `register_*()` or `initialize_*()` function
4. Coordination: `configure_default_game()` calls all registered features
5. Cargo.toml: Enable features to include in build

## Key Data Flows

### Player Interaction (dig/place/tap)
```
Client gRPC request
  → network_server::grpc_service
    → GameState event handler
      → Item dig_handler (if present)
        → Block interaction handler
          → game_map.set_block() / dig_block()
            → Light propagation
              → Database write (on chunk flush)
      → Inventory update
  → Network response: updated chunk + item stack
    → client_state cache update
      → vulkan re-mesh affected chunks
```

### Chunk Loading
```
[Client OutboundContext — client_workers.rs]
  Timer fires (~every frame) or game action triggered
  → send_position_update(last_position)
    → Attach ClientPacing { pending_chunks, distance_limit } to the message
        pending_chunks = sum of all MeshWorker queue lengths (backpressure signal)
        distance_limit = current render distance setting
    → Send StreamToServer::PositionUpdate via bidirectional gRPC stream
    Note: if previous position update has not been acked yet, the update is
    withheld (avoids flooding the server with unacknowledged positions)

[Server InboundWorker — client_context.rs]
  Receives StreamToServer::PositionUpdate
  → Reads pacing.pending_chunks and pacing.distance_limit from the message
  → Adjusts chunk_aimd (AIMD controller, shared with MapChunkSenders):
      - network backlogged (RTT > 250ms):  aimd.decrease() twice
      - pending_chunks > 1024:             aimd.decrease()
      - pending_chunks < 256:              aimd.increase()
      - block event lagged:                aimd.decrease_to_floor()
    AIMD parameters: additive_increase=64, multiplicative_decrease=0.5,
    initial=128, max=4096 chunks per position update
  → Publishes PositionAndPacing { kinematics, chunks_to_send=aimd.get(),
      client_requested_render_distance } on a watch channel

[Server MapChunkSender — client_context.rs]  (two instances, sharded by parity)
  Watches the PositionAndPacing watch channel for changes
  → send_chunks_for_position_update(update)
    → Phase 1: Unsubscribe out-of-range chunks
        → Sends StreamToClient::MapChunkUnsubscribe to client
    → Phase 2: Load and send in-range chunks not yet subscribed
        → Iterates candidate coords (zigzag order, up to chunks_to_send limit)
        → game_map().serialize_for_client(coord, load_if_missing=true, ...)
            → ServerGameMap::get_chunk() — loads from DB or runs mapgen
              → Deserialize blocks + extended data + lighting
            → Protobuf encode as StoredChunk
        → Snappy-compress encoded bytes
        → Send StreamToClient { server_message: MapChunk { snappy_encoded_bytes } }
          via outbound mpsc → NetPrioritizer → gRPC stream
          (server-pushed; there is no client-initiated "request chunk" RPC)
        → If chunks_to_send budget exhausted before all chunks sent: aimd.increase()
          (signals server can send more next update)

[Client InboundContext — client_workers.rs]
  Receives StreamToClient::MapChunk
  → handle_mapchunk()
    → Snappy-decode → StoredChunk protobuf
    → client_state.chunks.insert_or_update() — stores block IDs + client extended data
    → Enqueue coord (+ neighbors) for neighbor propagation
      → NeighborPropagator computes lighting neighbor data
        → MeshWorker generates chunk mesh
          → MeshBatcher batches draw calls
            → Vulkan renders next frame
```

### Entity Coroutine
```
EntityManager::tick()
  → Get entity, check control flags
    → If AUTONOMOUS + NEEDS_CALC:
      → Run coroutine closure
        → May call DeferredCall for async work
        → Tokio task executes, sends Completion back
    → Queue move command
  → Next frame: entity moves, clients notified
```

### Chat Message
```
Client: /command args
  → gRPC request: send_chat_message()
    → network_server handler
      → ChatState.handle_command()
        → Handler closure with EventInitiator context
          → Permission check
          → Execute action
      → Send response message
  → Broadcast to all players
    → Each client receives, display in HUD
```

## File Structure Quick Ref

```
perovskite_core/
├── src/
│   ├── lib.rs (module declarations + protocol re-exports)
│   ├── coordinates.rs (BlockCoordinate, ChunkCoordinate, ChunkOffset)
│   ├── block_id.rs (BlockId, variant handling)
│   ├── lighting.rs (Lightfield, propagation)
│   ├── auth.rs (OPAQUE cipher suite)
│   ├── chat.rs (ChatMessage)
│   └── ... (constants, sync, util, etc.)

perovskite_server/
├── src/
│   ├── lib.rs (module declarations)
│   ├── database/ (GameDatabase trait + implementations)
│   ├── game_state/ (9 submodules: blocks, items, game_map, player, entities, event, etc.)
│   ├── network_server/ (gRPC service, auth, client context)
│   ├── server.rs (ServerArgs, initialization)
│   ├── media.rs (MediaManager)
│   ├── formats.rs (OBJ loading)
│   └── ... (other utilities)

perovskite_client/
├── src/
│   ├── lib.rs (module declarations)
│   ├── client_main.rs (winit event loop)
│   ├── vulkan/ (rendering, game_renderer)
│   ├── client_state.rs (local cache)
│   ├── net_client.rs (gRPC client)
│   ├── game_ui.rs (HUD, inventory, chat)
│   ├── main_menu.rs (connection dialog)
│   ├── media.rs (asset loading)
│   ├── audio.rs (sound playback)
│   └── ... (other utilities)

perovskite_game_api/
├── src/
│   ├── lib.rs (feature-gated module declarations + configure_default_game())
│   ├── game_builder.rs (GameBuilder)
│   ├── blocks.rs (block appearance, rotation, interaction)
│   ├── items.rs (item actions, targets, handlers)
│   ├── default_game/ (blocks, items, map generator)
│   ├── circuits/ (logic gates, microcontroller, etc.)
│   ├── farming/ (crop growth)
│   ├── autobuild/ (building tools)
│   ├── animals/ (mobs)
│   ├── carts/ (minecarts on rails)
│   ├── discord/ (Discord integration)
│   ├── colors.rs (dye system, texture colorization)
│   └── ... (test_support, etc.)
```

## Typical Starting Points by Role

**Game Content Developer**:
1. Read: [`sourcemap_game_api.md`](./sourcemap_game_api.md) and [`sourcemap_overview.md`](./sourcemap_overview.md) → "Event Handling System"
2. Start with: `GameBuilder` for block/item registration
3. Extend: Custom interaction handlers and extended data for complex logic

**Engine Developer**:
1. Read: [`sourcemap_server.md`](./sourcemap_server.md) (focus on game_state subsystems)
2. Understand: Event dispatch, handler execution, synchronization patterns
3. Explore: Performance optimizations in light propagation, chunk loading, entity management

**Client Developer**:
1. Read: [`sourcemap_client.md`](./sourcemap_client.md)
2. Understand: Vulkan rendering pipeline, client-server sync, UI rendering
3. Extend: Custom shaders, optimization for different hardware

**Systems/Infra**:
1. Read: [`sourcemap_overview.md`](./sourcemap_overview.md) → "Network Protocol" and "Database"
2. Focus: [`sourcemap_server.md`](./sourcemap_server.md) → database, network_server
3. Extend: Persistence layer, scaling strategies, failure recovery
