---
name: debug_rpc_tool
description: Using and extending the server's debug RPC interface (perovskite_core/proto/debug.proto, perovskite_server/src/network_server/debug_rpc.rs) and its CLI, perovskite_server/src/bin/debug_client.rs. Use this when asked to inspect live server state (block/item defs, recent player events, loaded chunks) or to add a new debug RPC.
---

The debug RPC interface is a small gRPC service, separate from the main game protocol, meant for interactive
inspection of a running server by humans and LLM/agent tooling. It is driven by a fake player
(`internal:debug_client`) that the server keeps connected so that RPCs can piggyback on real game-state APIs
(e.g. chunk loading around a "working coordinate").

## Using the tool

1. Start (or use an already-running) server with the debug interface enabled:
   ```
   cargo run -p perovskite_game_api --release --features=server,discord -- \
       --data-dir=/tmp/foo --enable_debug_interface --port=28273
   ```
2. **Build the CLI once, then invoke the executable directly** (defaults to `grpc://localhost:28273`):
   ```
   cargo build -p perovskite_server --bin debug_client --release
   cp target/release/debug_client.exe /path/to/stable/dir/debug_client.exe

   /path/to/stable/dir/debug_client.exe find-block-defs '^default:dirt'
   /path/to/stable/dir/debug_client.exe find-item-defs '.*'
   /path/to/stable/dir/debug_client.exe last-events
   /path/to/stable/dir/debug_client.exe set-working-coord -- -400 0 128
   ```
   The `--` before the coordinates in `set-working-coord` is required whenever any of them is negative —
   without it, clap parses e.g. `-400` as an unrecognized flag rather than a positional argument.

   `--help` on the binary or any subcommand lists arguments; output is always Rust's pretty-printed
   `{:#?}` Debug format, so new RPCs don't need custom formatting code.

   Available flags:

   * **--endpoint** - endpoint of the form `grpc://host:port`. If unspecified, defaults to localhost:28273 without TLS
       (matching the default for a typical running server).
   * **--verbose** - use to get full protos, remove to get quick skim-friendly summaries.

**Build it once and reuse the binary.** Copy it to a stable location outside the `target/` tree as shown
above, and keep invoking that copy — don't rebuild before every call. Only rebuild (and re-copy) it when
`debug_client.rs` or something it depends on (the `debug` proto, `perovskite_core`) changes.

**`cargo run` is for iterating on `debug_client.rs` itself.** While actively editing the CLI, it's fine to
use `cargo run -p perovskite_server --bin debug_client -- <args>` so each change is picked up automatically;
switch back to the built executable once you're done changing it.

## Extending the tool

Adding a new debug RPC touches three files, in this order:

1. **`perovskite_core/proto/debug.proto`** — add `FooReq`/`FooResp` messages and an `rpc Foo(FooReq) returns
   (FooResp);` line in the `PerovskiteDebug` service. Keep request/response messages simple and
   self-describing; prefer `optional` fields with sensible defaults over required fields so the client CLI
   can expose them as optional flags.
2. **`perovskite_server/src/network_server/debug_rpc.rs`** — implement the new method on
   `impl PerovskiteDebug for DebugServer`. `DebugServer` holds an `Arc<GameState>`, a `PlayerContext` for the
   fake debug player, a rolling `last_events` buffer, and a `working_coord` shared with the background chunk
   -loading task — reuse these rather than adding new fields unless the RPC needs genuinely new state.
3. **`perovskite_server/src/bin/debug_client.rs`** — add a variant to the `Command` enum (with doc comments,
   which `clap` turns into `--help` text), a matching arm in `main`'s `match cli.command`, and a `run_*`
   async function alongside `run_find_block_defs`/`run_last_events`/etc. that calls the client method and
   prints results via `{:#?}` on success or `print_rpc_error` on failure. Follow the existing functions as a
   template — no new formatting/error-handling patterns are needed.

After changing the `.proto` file, a normal `cargo build` regenerates the Rust bindings (via the crate's
build script/prost-tonic codegen) — no separate protoc step is required.

Do not add debug subcommands whose only purpose is to intentionally trigger a server error or panic (e.g. to
test client-side error handling) — those are validated ad hoc during development, not kept as permanent CLI
surface.