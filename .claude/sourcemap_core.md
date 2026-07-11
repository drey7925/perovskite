# Perovskite Core Crate Source Map

Core shared types, protocols, and foundational abstractions used across server and client.

## `coordinates` module
**File**: `perovskite_core/src/coordinates.rs`

Defines spatial coordinate systems for the voxel world:
- `BlockCoordinate` — 3D integer coordinate for individual blocks; provides `offset()` (within-chunk position) and `chunk()` (containing chunk) conversions
- `ChunkCoordinate` — 3D coordinate for 16×16×16 chunks
- `ChunkOffset` — 8-bit offset of a block within its chunk (0–15 per axis)

**Key abstractions**: Coordinate algebra (checked/unchecked delta), tiebreaking orderings for sorted structures.

## `block_id` module
**File**: `perovskite_core/src/block_id.rs`

Block type registry and ID encoding:
- `BlockId(u32)` — opaque block identifier; lower 12 bits encode variant (rotation, state), upper 20 bits are base block index
- `BlockError` — registration errors (duplicate name, invalid variant, ID exhaustion)
- `special_block_defs::AIR_ID`, `UNLOADED_CHUNK_BLOCK_ID`, `INVALID_BLOCK_ID`
- Constants: `MAX_BLOCK_DEFS` (~1M), `BLOCK_VARIANT_MASK` (0xfff)

**Key abstractions**: Variant-aware block encoding; helper methods for base ID and variant extraction.

## `render_selector` module
**File**: `perovskite_core/src/render_selector.rs`

Bit layout of the *render selector*, the u32 word that conditional AABB geometry masks
(`AxisAlignedBox::variant_mask`) are matched against on the client:
- bits [0,12) — server-sent block variant; bits [12,18) — client-computed neighbor-presence bits (`NEIGHBOR_XPLUS` … `NEIGHBOR_ZMINUS`); bit 31 — `ALWAYS` (set in every selector, so an all-ones mask means "always active")
- `from_variant(u16) -> u32`, `NEIGHBOR_OFFSETS` (world-space offsets in bit order), `VALID_MASK_BITS`
- Used by the client mesher, physics, and tool raycast; enables fence-style auto-connecting geometry without server round trips (see `make_fence` in game_api `shaped_blocks.rs`)

## `lighting` module
**File**: `perovskite_core/src/lighting.rs`

Light propagation shared between client and server:
- `Lightfield` — 256-bit bitfield (16×16 XZ positions in a chunk); serializes/deserializes as `[u32; 8]`
- `LightScratchpad` (public re-export) — used during light propagation calculations
- `propagate_light()` — core light algorithm

**Key abstractions**: Compact bitfield-based representation of lit positions within a chunk column.

## `auth` module
**File**: `perovskite_core/src/auth.rs`

Authentication cryptography:
- `PerovskiteOpaqueAuth` — OPAQUE cipher suite using Ristretto255 + Argon2 KSF
- `Argon2_4096_3_1` — configured key derivation

**Key abstractions**: Opaque-ke integration for password-authenticated key exchange.

**Warnings**: Authentication parameters affect existing auth information in databases. Do not change
without also preparing a migration path for users, or as part of an otherwise-breaking change.

## `chat` module
**File**: `perovskite_core/src/chat.rs`

Chat messaging types and colors:
- `ChatMessage` — timestamped origin + text with color; builder methods `with_color()`, `with_origin()`
- Color constants: `SERVER_MESSAGE_COLOR`, `SERVER_WARNING_COLOR`, `SERVER_ERROR_COLOR`
- Serialization: `color_to_fixed32()`, `color_from_fixed32()`

**Key abstractions**: Immutable chat representation; origin color as metadata.

## `protocol` module (gRPC/Protobuf)
**File**: `perovskite_core/src/lib.rs` (mod definition)

Re-exports tonic-generated protobuf modules:
- `perovskite::protocol::blocks` — `BlockTypeDef`, `BlockTypeDefVariant`, etc.
- `perovskite::protocol::map` — map chunk wire format
- `perovskite::protocol::game_rpc` — main game server RPC interface
- `perovskite::protocol::coordinates` — wire coordinate formats
- `perovskite::protocol::render` — `TextureReference`, `CustomMesh`
- `perovskite::protocol::items` — `ItemDef`, `ItemStack` (wire formats)
- `perovskite::protocol::players` — player state wire format
- `perovskite::protocol::ui` — popup/dialog messages
- `perovskite::protocol::entities` — entity appearance and types
- `perovskite::protocol::audio` — `SampledSound` and audio resources

Also provides version headers and descriptor set for gRPC reflection.

## `constants` module
Game constants: block groups (AIR, DEFAULT_SOLID, DEFAULT_LIQUID, DEFAULT_GAS, TRIVIALLY_REPLACEABLE), textures (FALLBACK_UNKNOWN_TEXTURE), item interaction rules, permissions.

## `game_actions` module
Game action definitions (shared action types for client–server interaction).

## `items` module
Item-related types and utilities.

## `sync` module
Synchronization primitives (`GenericMutex`, `SyncBackend`, `DefaultSyncBackend`) for optional thread-safe vs. single-threaded backends.

## `util` module
Utility functions and tracing infrastructure.

## `far_sheet` module
Far mesh / terrain outline storage for distant geometry.

## `time` module
Game time state tracking.

---

### Module Navigation Guidance

**For coordinate conversions**: Start at `BlockCoordinate::chunk()`, `BlockCoordinate::offset()`.

**For block type lookups**: Use `BlockId` methods: `base_id()`, `variant()`, `with_variant()`.

**For light calculations**: Inspect `Lightfield` serialization and `propagate_light()`.

**For auth**: Refer to `PerovskiteOpaqueAuth` cipher suite configuration in server's `AuthService`.

**For protocol messages**: All wire types are in `perovskite::protocol::*` (tonic-generated).
