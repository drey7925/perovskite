# Perovskite Client Crate Source Map

Thin Vulkan-based client for rendering, input, and networking with server.

## `client_main` module
**File**: `/c/cuberef/perovskite_client/src/client_main.rs`

Client initialization and event loop:
- `ClientArgs` — CLI arguments: server host:port
- `run_client()` — main entry point
  - Initializes Tracy profiler
  - Creates winit `EventLoop`
  - Instantiates `GameApplication` and runs event loop

**Key abstractions**: Thin wrapper around winit; single-threaded event loop.

## `vulkan` module
**File**: `/c/cuberef/perovskite_client/src/vulkan/`

Rendering engine (Vulkan):
- `GameApplication` — winit `ApplicationHandler`; manages render state and frame loop
- `game_renderer` — frame rendering: vertex/fragment shaders, mesh submission, swapchain management
- Rendering pipeline:
  - Chunk mesh generation (on demand from server data)
  - Terrain far-field rendering (distant outlines)
  - Entity rendering (players, animals, etc.)
  - UI overlay (HUD, chat, inventory)
- Memory management: GPU buffer pools, texture atlasing

**Key abstractions**: GPU-resident chunk meshes; deferred terrain LOD; frame-synchronized rendering.

## `client_state` module
**File**: `/c/cuberef/perovskite_client/src/client_state.rs`

Client-side game state tracking:
- `ClientState` — local cache of visible world: loaded chunks, entities, player position
- Chunk status: unloaded, loading, loaded, mesh ready
- Entity cache: received entity updates, interpolated positions
- Inventory local copy: for UI and interaction preview
- Chat buffer: recent messages for display

**Key abstractions**: Eventual consistency with server; optimistic updates for latency hiding.

## `net_client` module (private)
**File**: `/c/cuberef/perovskite_client/src/net_client.rs`

gRPC client for server communication:
- Network stream: bidirectional gRPC over HTTP/2 (tonic)
- Message types: game RPC requests (block updates, player action), responses (world state, entity updates)
- Queue: outgoing requests coalesced before send
- Retries/backoff: connection loss handling

**Key abstractions**: Async gRPC streams; message batching for efficiency.

## `game_ui` module (private)
**File**: `/c/cuberef/perovskite_client/src/game_ui.rs`

In-game UI rendering and interaction:
- HUD elements: hotbar, selected item, health/hunger
- Inventory screen: grid layout, item tooltips, drag-and-drop preview
- Chat UI: message display, input box, command completion
- Popup dialogs: received from server for inventory/crafting

**Key abstractions**: Overlay rendered on top of 3D scene; input routing between game and UI.

## `main_menu` module (private)
**File**: `/c/cuberef/perovskite_client/src/main_menu.rs`

Server connection UI:
- Server selection/entry: host input, recent servers
- Character creation/selection (future)
- Settings panel: graphics, audio, controls
- Main menu state machine

**Key abstractions**: Pre-game state; server list persistence.

## `media` module (private)
**File**: `/c/cuberef/perovskite_client/src/media.rs`

Asset loading and management:
- Texture loading: server-provided textures -> GPU memory
- Audio loading: sampled sounds -> audio system
- Model loading: OBJ meshes -> GPU buffers
- Asset cache: hash-based deduplication (server provides SHA256)

**Key abstractions**: Lazy loading; hash validation; fallback textures on missing assets.

## `audio` module (public for early testing)
**File**: `/c/cuberef/perovskite_client/src/audio.rs`

Audio playback:
- Audio device management (likely using cpal or similar)
- Sound mixing: multiple simultaneous effects
- Spatial audio (future): 3D positioning, attenuation
- BGM: background music playback

**Key abstractions**: Abstracted audio backend; spatial sound calculations.

---

### Client Architecture Overview

```
winit EventLoop
  ├─ Input handling (keyboard, mouse, gamepad)
  ├─ Window events (resize, focus, close)
  └─ Frame loop (game_renderer::GameApplication)
       ├─ net_client: Receive server updates
       ├─ client_state: Update local cache
       ├─ game_ui: Render UI overlays
       ├─ vulkan: Render 3D world
       └─ audio: Play queued sounds
```

### Data Flow

1. **Server connection**: `main_menu` -> `net_client` establishes gRPC stream to server
2. **World updates**: Server sends chunk data + entity updates -> `client_state` caches -> `vulkan` generates meshes
3. **Player input**: `game_ui` / `main_menu` -> player action -> `net_client` enqueues RPC request
4. **Rendering**: `vulkan::game_renderer` pulls chunks from `client_state`, generates meshes, renders frame
5. **Audio**: Server sends audio events -> `media` loads sample -> `audio` plays

### Module Navigation Guidance

**For rendering**: Start at `vulkan::game_renderer::GameApplication` for frame loop and pipeline structure.

**For chunk mesh generation**: Examine `vulkan` module's mesh builder; converts server block data to vertex buffers.

**For networking**: Check `net_client` for message format and stream handling; uses tonic's `perovskite::protocol::game_rpc`.

**For UI**: Look at `game_ui` for layout and input handling; popups received from server contain inventory/crafting data.

**For audio**: Examine `audio` module for playback primitives and `media` for asset loading.

**For asset loading**: See `media` module for texture atlas, hash validation, and fallback handling.

**For client state**: Inspect `client_state` for chunk cache structure and entity interpolation.

**For connection setup**: Start at `main_menu` and `net_client` for gRPC endpoint configuration and authentication.
