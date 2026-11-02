---
name: dev_server
description: Starting and customizing the dev server (perovskite_game_api/src/bin/dev_server.rs) - an in-memory, throwaway server with the default game, selectable world scenarios (flat dirt, real default mapgen, stations fork), and a player-action recorder. Referenced by headless_render, game_input_demo, and debug_rpc_tool.
---

`perovskite_game_api/src/bin/dev_server.rs` brings up the default game on `127.0.0.1:28275` with an
in-memory database. The world is thrown away on exit, so accounts must be registered on every run
(any username/password works). All players get all permissions. Player dig/tap/place/interact
actions are logged to a recording file (see `game_input_demo`).

## Running

From the workspace root:

```
cargo run -p perovskite_game_api --bin dev_server --features default_game,server,test-support -- [--scenario <name>] [--output <file>]
```

The command **blocks** until the server shuts down (type `/stop` in game chat, or kill the process).
Wait for `Serving on 127.0.0.1:28275` in the log before connecting. When driving it from an agent,
run it in the background.

## Scenarios (`--scenario`)

| Scenario | World |
|----------|-------|
| *(none)* | Flat dirt world (`set_flatland_mapgen(DIRT)`), with a stone block at (0,-1,0) marking the origin. Fast to load; best for block/UI/behavior work. |
| `default_mapgen` | The real, pre-configured terrain mapgen of `default_game`: `configure_default_game` with no mapgen override. Use it to check terrain, trees/foliage, ores, lighting, and anything that depends on natural world generation. Chunk generation is slower than flat dirt, especially in a debug build. No origin marker is placed. |
| `stations_fork` | The cart fork-station test scene (`carts::station_tests`). |

Adding a scenario: add a `Scenario` variant, a `parse_scenario` arm, a match arm in the world setup
in `main`, and (if it needs a running server) a match arm after `into_server()`.

## Customizing initial state

`dev_server.rs` has two marked blocks:

- `CONTENT SETUP` (before `into_server`): spawn location, items given on join, extra registrations.
- `SCENE SETUP` (via `run_task_in_server`): world edits that need a running server.

See `game_input_demo` for worked examples (spawn override, `GiveItemsOnJoin`). Revert these edits
when done unless they're reusable.

## Other tools that use the dev server

- `headless_render`: renders a PNG of the world from a camera pose. Use `--scenario default_mapgen`
  to look at real terrain.
- `game_input_demo`: ask the user to play on the dev server and read back the action recording.
- `debug_rpc_tool`: the dev server already has the debug RPC interface enabled (via `tempdir_args`),
  served on the same port as the game. Point the CLI at it with
  `debug_client --endpoint grpc://127.0.0.1:28275 <subcommand>` (the CLI's default is 28273).
