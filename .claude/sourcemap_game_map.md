# game_map Module Source Map

Deep-dive navigation for `perovskite_server::game_state::game_map` — the world chunk storage, lighting, atomic block mutation, timer system, and background I/O tasks.

**Files covered:**
- `/c/cuberef/perovskite_server/src/game_state/game_map.rs` (~4,551 lines)
- `/c/cuberef/perovskite_server/src/game_state/game_map/templates.rs` (99 lines)
- `/c/cuberef/perovskite_server/src/game_state/game_map/loom_tests.rs` (146 lines)

---

## Top-Level Types

### `CasOutcome`
Result of compare-and-set operations:
- `Match { old_block, old_ext_data }` — predicate passed, block was changed
- `Mismatch { actual_block }` — predicate failed, no mutation occurred

### `MapChunk`
A 16×16×16 block region held in memory:
- `block_ids: [CachelineAligned<AtomicU32>; 4096]` — block data, lock-free readable
- `extended_data: FxHashMap<u16, ExtendedDataHolder>` — per-block custom state (sparse)
- `dirty: bool` — needs writeback to database
- `light_valid: bool` — lighting is current
- `bloom: BlockBloomFilter` — fast presence check for blocks of a type
- `client_extended_data_cache: Option<Vec<Option<...>>>` — cached serialized ext data for network

Key methods:
- `block_ids_as_u32()` — raw array for fast client serialization
- `extended_data_for_mut(offset)` — borrow-or-create extended data slot
- `set_block_id(offset, id)` — write to atomic slot
- `fill_lighting_for_load()` — initialize lighting from block occlusion on chunk load

### `ServerGameMap<S, L>`
Main world map. Generic over two sync backends:
- `S: SyncBackend` — real (thread-safe) or loom-testable
- `L: SyncBackend` — separate backend for light columns (prevents lock-order issues)

Fields:
- `shards: [MapShard<S, L>; NUM_CHUNK_SHARDS]` — 16 shards for distributed locking
- `db: Arc<dyn GameDatabase>` — persistent storage
- `mapgen: Arc<dyn MapgenInterface>` — terrain generation
- `block_type_manager: Arc<BlockTypeManager>` — block definitions
- `timers: Mutex<Vec<TimerHandle>>` — registered map timers
- `writeback_permits: Semaphore` — backpressure for writeback queue depth
- `prefetch_tx: Sender<ChunkCoordinate>` — channel to prefetch background task

---

## Internal Structure

### `MapShard<S, L>`
One of 16 partitions of the chunk map:
- `chunks: RwLock<FxHashMap<ChunkCoordinate, Arc<MapChunkHolder<S>>>>` — the chunk cache
- `light_columns: RwLock<FxHashMap<(i32, i32), LightColumn<L>>>` — vertical light state

Sharding key: `coord.hash() % NUM_CHUNK_SHARDS`

### `MapChunkHolder<S>`
Wraps a `MapChunk` with lifecycle state and async notification:
- `state: RwCondvar<HolderState<S>>` — current load state + waiter notifications
- `last_access: AtomicU64` — timestamp for LRU eviction
- `coord: ChunkCoordinate` — self-referential for logging

### `HolderState`
Three-state chunk load lifecycle:
- `Empty` — placeholder inserted while loading (others wait on condvar)
- `Err(anyhow::Error)` — load/generate failed
- `Ok(S::RwLock<MapChunk>)` — loaded, ready for access

### `MapChunkOuterGuard<'a, S, L>`
RAII guard returned by `get_chunk()`:
- Holds `Arc<MapChunkHolder>` + read lock on `HolderState`
- Exposes `chunk()` → `&MapChunk` and `chunk_mut()` → `&mut MapChunk`
- Dropping releases the shard read lock

---

## Chunk Loading Flow

**`get_chunk(coord, load_if_missing)`**

```
1. Acquire shard read lock
2. If chunk exists in cache → return MapChunkOuterGuard (fast path)
3. If load_if_missing=false → return None
4. Acquire writeback permit (backpressure — limits concurrent loads)
5. Upgrade to shard write lock
6. Insert Empty MapChunkHolder (others waiting will block on condvar)
7. Downgrade to read lock (atomic, no gap)
8. Spawn load task:
   a. Try DB: database.get(KeySpace::MapchunkData, coord_key)
      → deserialize StoredChunk proto → fill MapChunk
   b. On miss: mapgen.generate_chunk(coord) → fill MapChunk
   c. fill_lighting_for_load() — scan Y columns for first opaque block
9. call holder.fill(Ok(chunk)) — notifies all waiters via condvar
10. Return MapChunkOuterGuard
```

**Prefetch** (background task): `prefetch_tx.send(coord)` queues a coord; background task calls `get_chunk(coord, true)` without blocking callers.

---

## Block Access Methods

All block access goes through `ServerGameMap`; chunk is loaded on demand.

| Method | Lock taken | Notes |
|--------|-----------|-------|
| `get_block(coord)` | Chunk read | Returns `BlockId`, no ext data |
| `try_get_block(coord)` | Chunk read | Returns `None` if chunk not cached |
| `get_block_with_extended_data(coord, cb)` | Chunk read | Callback receives `(BlockId, Option<&ExtendedData>)` |
| `bulk_read_chunk(coord, cb)` | Chunk read | Callback receives `&MapChunk` |
| `set_block(coord, block, ext_data)` | Chunk write | Marks dirty, enqueues writeback, broadcasts |
| `bulk_write_chunk(coord, cb)` | Chunk write | Callback mutates `&mut MapChunk` |
| `compare_and_set_block(coord, expected, new)` | Chunk write | Simple equality CAS |
| `compare_and_set_block_predicate(coord, pred, new)` | Chunk write | Closure-based predicate CAS |
| `mutate_block_atomically(coord, mutator)` | Chunk write | Mutator sees `(&mut BlockId, &mut ExtendedData)` |
| `try_mutate_block_atomically(coord, mutator)` | Chunk write (try) | Non-blocking; returns `None` if lock busy |

### `mutate_block_atomically_locked` — the core mutation primitive
Called by all write methods after acquiring the chunk write lock:
1. Extract extended data → wrap in `ExtendedDataHolder`
2. Run mutator closure
3. If dirty: update bloom filter, mark chunk dirty, enqueue writeback
4. If lighting changed: propagate via `update_lighting()`
5. If block type changed: regenerate client ext data cache if needed
6. Broadcast `UpdateBroadcast::Block` to all connected clients

---

## Block Interaction Methods

Higher-level methods that invoke block/item handler closures:

- `dig_block(coord, tool, initiator)` — calls `run_block_interaction` with Dig context
- `tap_block(coord, tool, initiator)` — calls `run_block_interaction` with Tap context
- `fixup_block(coord, initiator)` — re-runs place handler for world repair

### `run_block_interaction(coord, ...)`
Two-phase handler execution to avoid holding chunk lock during full handler:

**Phase 1** (inside `mutate_block_atomically`, chunk shard lock held):
- `block.dig_handler_inline` / `tap_handler_inline`
- `InlineContext`: may mutate `BlockId` and `ExtendedData` in place
- Lighting updated after inline mutation

**Phase 2** (chunk lock released, avoids deadlock):
- `block.dig_handler_full` / `tap_handler_full`
- `HandlerContext`: full `GameState` access; can call `set_block`, spawn timers, send chat
- Results from both phases coalesced via `CoalesceResult`

---

## Lighting System

### `LightColumn` (in `game_state::game_map`, light columns live in shards)
Tracks occlusion state per (x, z) column across all Y chunks:
- `occlusion: FxHashMap<ChunkCoordinate, Lightfield>` — per-chunk occlusion bitfield
- `top_opaque_y: FxHashMap<(i32,i32), i32>` — highest opaque block per XZ position

### `fill_lighting_for_load`
On chunk load, scans the 16×16 XZ grid from top to find first opaque block per column. Initializes `Lightfield` and updates light column.

### `update_lighting` (called from `mutate_block_atomically_locked`)
When a block changes opacity:
1. Look up light column for the chunk
2. Recompute occlusion for affected XZ column
3. Propagate changes up/down to neighboring chunks' `Lightfield`
4. Mark affected chunks dirty for client resend

**Coalescing**: Multiple block updates in one tick batch lighting recalculations; only the final lighting state is propagated.

---

## Timer System

Timers drive periodic per-block or per-chunk logic (e.g., crop growth, circuit updates).

### `TimerSettings`
Configuration for a registered timer:
- `interval: Duration` — how often to fire
- `shards: u32` — number of sub-shards for staggered execution
- `spreading: f64` — randomize interval ± this fraction
- `block_types: Option<Vec<BlockId>>` — if set, only run on chunks containing these blocks (bloom filter)
- `callback: TimerCallback` — the callback variant to use
- `_non_exhaustive: NonExhaustive` — forward-compatibility marker

### `TimerCallback`
Four callback modes for different access patterns:

| Variant | Lock held | Use case |
|---------|----------|---------|
| `PerBlockLocked(Arc<dyn TimerInlineCallback>)` | Chunk write | Per-block mutation, uses bloom filter |
| `BulkUpdate(Arc<dyn BulkUpdateCallback>)` | Chunk write | Whole-chunk batch update |
| `BulkUpdateWithNeighbors(Arc<dyn BulkUpdateCallback>)` | Chunk write + 26 neighbor reads | Needs neighbor block data |
| `LockedVerticalNeighbors(Arc<dyn VerticalNeighborTimerCallback>)` | Two vertically adjacent chunk writes | Lighting across chunk boundaries |

### `TimerInlineCallback` trait
```rust
fn timer_callback(
    &self,
    ctx: &HandlerContext,
    coord: BlockCoordinate,
    block_id: &mut BlockId,
    ext_data: &mut ExtendedData,
    state: &TimerState,
) -> Result<()>;
```

### `BulkUpdateCallback` trait
```rust
fn bulk_update_callback(
    &self,
    ctx: &HandlerContext,
    coord: ChunkCoordinate,
    chunk: &mut MapChunk,
    neighbors: Option<&ChunkNeighbors>,
    state: &TimerState,
) -> Result<()>;
```

### `ChunkNeighbors`
48×48×48 buffer (3×3×3 chunk neighborhood) for `BulkUpdateWithNeighbors`:
- `block_at(coord: BlockCoordinate) -> BlockId` — read-only neighbor access
- Built by copying block IDs from the 26 surrounding chunks before invoking callback

### `TimerState`
```rust
pub struct TimerState {
    pub prev_tick: Instant,
    pub current_tick: Instant,
}
```

### `GameMapTimer`
Internal timer implementation:
- Tracks shard state (`ShardState`): which chunks have been processed this tick
- Divides work across `shards` sub-shards, spread over `interval`
- Uses bloom filter to skip chunks with no relevant blocks (fast path)
- Tracks `ShardState { started_at, shards_remaining }` per tick

### `TimerController<S, L>`
Runs all registered timers:
- Spawns one tokio task per timer
- Each task sleeps until next shard boundary, runs shard work, loops
- `register_timer(settings, callback) -> TimerHandle`

### Timer execution flow (`do_tick_work`, line 3459)
```
For each chunk in shard:
  1. Check bloom filter against timer's block_types → skip if no match
  2. Acquire chunk write lock
  3. Dispatch to callback:
     PerBlockLocked: iterate all blocks in chunk, call timer_callback per block
     BulkUpdate: call bulk_update_callback(chunk, None)
     BulkUpdateWithNeighbors:
       → Read-lock 26 neighbors
       → Build ChunkNeighbors buffer
       → call bulk_update_callback(chunk, Some(&neighbors))
     LockedVerticalNeighbors:
       → Acquire write lock on Y+1 chunk
       → Call vertical_neighbor_callback(lower_chunk, upper_chunk)
  4. Mark chunk dirty if callback modified blocks
  5. Enqueue writeback
```

---

## Background Tasks

### `GameMapWriteback<S, L>`
One background task per shard; coalesces and persists dirty chunks:

**`gather()`** — coalescing logic:
1. Wait for first writeback request from mpsc channel
2. Poll channel for up to `WRITEBACK_COALESCE_TIME` (few ms), collecting more requests
3. Dedup by coordinate (latest write wins)
4. Return sorted batch (up to `WRITEBACK_COALESCE_MAX_SIZE`)

**`run()` loop:**
1. `gather()` → deduplicated list of coords
2. For each coord: lock chunk, serialize to `StoredChunk` proto, call `db.put()`
3. Release permit (unblocks `get_chunk()` callers waiting on backpressure)

### `MapCacheCleanup<S, L>`
Background task per shard that evicts cold chunks from memory:
1. Periodically scan shard's chunk map
2. Sort by `last_access` timestamp (LRU order)
3. Evict chunks older than threshold if within memory budget
4. Only evicts chunks where `dirty=false` (writeback task ensures this)

### Prefetch Task
Receives `ChunkCoordinate` from `prefetch_tx` channel; calls `get_chunk(coord, true)` to warm cache before a player arrives.

### Backpressure
`writeback_permits: Semaphore` limits concurrent in-flight chunk loads/writebacks. `get_chunk()` acquires a permit before loading; `GameMapWriteback` releases it after persisting. `in_pushback()` reports when semaphore is near capacity.

---

## Client Serialization

### `serialize_for_client(coord, load_if_missing, block_type_mgr)`
Produces `StoredChunk` proto for network transmission:
1. Load chunk (or return `None` if not cached and `load_if_missing=false`)
2. Copy 4096 block IDs as raw u32 array
3. Collect extended data: for each block with ext data, serialize via registered proto handler → `MapChunkExtData` entries
4. Include lighting `Lightfield` bytes
5. Return `StoredChunk { block_ids, ext_data, lighting, ... }`

Caller (network_server) Snappy-compresses the encoded bytes before sending.

---

## Shutdown & Persistence

### `do_shutdown()`
Graceful shutdown sequence:
1. Signal all timer tasks to stop
2. Drain writeback queues (wait for all dirty chunks to persist)
3. Flush database (ensure fsync)
4. Drop all chunk data

### `purge_and_flush()`
Synchronous flush for testing / disaster recovery:
1. Collect all dirty chunks across all shards
2. Serialize and write each to database
3. Call `db.flush()`
4. Clear in-memory chunk cache

---

## Templates (`game_map/templates.rs`)

### `MASK: BlockId`
Special sentinel `BlockId(1)` meaning "leave existing block unchanged" when applying a template.

### `InMemTemplate`
In-memory 3D array of blocks for paste/undo/structure operations:
- Storage layout: X outermost, Z middle, Y innermost (matches GameMap chunk layout)
- `new_empty(size: (i32, i32, i32))` — fills with MASK blocks
- `block_at(x, y, z) -> BlockId`
- `ext_data_at(x, y, z) -> Option<&ExtendedData>`
- `set_block_at(x, y, z, block_id, extended_data: Option<ExtendedData>)`

Extended data stored in `BTreeMap<usize, ExtendedData>` keyed by linearized index.

### `apply_template(template, origin, rotation)` on `ServerGameMap`
Applies an `InMemTemplate` to the world:
1. Iterate template coordinates
2. Skip positions where `template.block_at(x,y,z) == MASK`
3. For each real block: `set_block(world_coord, block_id, ext_data)`
4. Rotation applied as 0°/90°/180°/270° around Y axis

---

## Concurrency Model Summary

```
ServerGameMap
 ├─ 16 MapShards, each with:
 │   ├─ RwLock<HashMap<ChunkCoord, Arc<MapChunkHolder>>>
 │   └─ RwLock<HashMap<(x,z), LightColumn>>
 │
 │  (shard determined by coord hash; avoids cross-shard locking)
 │
 ├─ MapChunkHolder
 │   └─ RwCondvar<HolderState>
 │       ├─ Empty  → readers wait on condvar
 │       ├─ Err    → load failed
 │       └─ Ok(RwLock<MapChunk>)
 │           └─ block_ids: [AtomicU32]  ← lock-free reads
 │
 ├─ Background tasks (per shard):
 │   ├─ GameMapWriteback  → coalesce & persist dirty chunks
 │   └─ MapCacheCleanup   → LRU eviction of cold chunks
 │
 ├─ Timer tasks (per TimerSettings):
 │   └─ GameMapTimer → staggered per-shard tick work
 │
 └─ Semaphore: writeback_permits → backpressure on concurrent loads
```

**Lock ordering** (never acquire in reverse order to avoid deadlock):
1. Shard `chunks` RwLock (outer)
2. `MapChunkHolder` inner RwLock (inner)
3. `LightColumn` RwLock (separate shard-level lock, never held with chunk write lock)

Timer `BulkUpdateWithNeighbors` acquires one write lock + 26 read locks; always takes write lock first, then read locks in deterministic coordinate order.

---

## Concurrency Testing (`game_map/loom_tests.rs`)

Uses the `loom` crate for model-checked concurrency:
- `test_load_store_purge()` — 3 threads doing concurrent mutations + `purge_and_flush()`; verifies total write count
- `test_lighting()` — 2 threads doing concurrent block updates; verifies lighting consistency
- `make_loom_map()` — creates `ServerGameMap` using `TestonlyLoomBackend` instead of `DefaultSyncBackend`

These tests exercise the specific lock patterns that would be vulnerable to data races or deadlocks under thread interleaving.

---

## Quick Navigation

| Task | Location |
|------|---------|
| Read a block | `ServerGameMap::get_block()` line 1338 |
| Write a block | `ServerGameMap::set_block()` line 1441 |
| Atomic read-modify-write | `ServerGameMap::mutate_block_atomically()` line 1642 |
| CAS with predicate | `ServerGameMap::compare_and_set_block_predicate()` line 1544 |
| Run dig/tap handlers | `ServerGameMap::run_block_interaction()` line 1893 |
| How a chunk loads | `ServerGameMap::get_chunk()` line 1970 |
| How lighting propagates | `fill_lighting_for_load()` line 878, `update_lighting()` |
| Register a timer | `ServerGameMap::register_timer()` line 4364 |
| Timer callback interfaces | `TimerInlineCallback` line 3096, `BulkUpdateCallback` line 3230 |
| How timers execute | `GameMapTimer::do_tick_work()` line 3459 |
| Chunk writeback | `GameMapWriteback::gather()` line 2793 |
| Paste a structure | `ServerGameMap::apply_template()` line 2525, `InMemTemplate` |
| Client chunk serialization | `ServerGameMap::serialize_for_client()` line 2260 |
| Shutdown sequence | `ServerGameMap::do_shutdown()` line 2309 |
| Backpressure check | `ServerGameMap::in_pushback()` line 2440 |
