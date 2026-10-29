---
name: headless_render
description: Render a single frame of the game client to a PNG without a window or GPU (lavapipe), from a given camera position and angle. Use this to visually check client rendering changes or server-side world/block changes in an agent loop.
---

`perovskite_client/src/bin/headless_render.rs` logs into a server, pins the camera (no physics),
waits for nearby chunks to be meshed, renders one frame into an offscreen image, and writes a PNG.
It runs the real client workers and the real `ActiveGame` render path, minus egui and audio.

## IMPORTANT: always use a release build of the client

**Run `headless_render` with `cargo run --release` (or `cargo build --release`), even when the
server is a debug build you're iterating on.** A debug client against a debug server does not work
reliably:

- Login takes about 30s, because the client-side password hashing (Argon2) is very slow
  unoptimized. In release it takes about 3s.
- A debug client is slow to process incoming chunks. Position-update acks queue up behind chunk
  data, and the client stops sending while it waits for them. The server then times out the
  connection after 10s of silence ("Client inbound context ... timed out" in the server log,
  "outbound loop crashed: channel closed" in the client log). This is a known protocol gap,
  not something to work around in the skill.

A debug server is fine: it only makes chunk loading slower.

## Prerequisites

- Mesa's software Vulkan driver: `apt-get install -y mesa-vulkan-drivers` (check with
  `vulkaninfo --summary`, from `vulkan-tools`; expect `llvmpipe`).
- If the machine also has a real GPU, force lavapipe with
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`.

## Steps

1. Start the dev server (see the `game_input_demo` skill for scene setup); it listens on
   `127.0.0.1:28275` with an in-memory world, so the account must be registered on each run:

   ```bash
   cargo run -p perovskite_game_api --bin dev_server --features default_game,server,test-support
   ```

2. Wait for `Serving on 127.0.0.1:28275` in the server log, then render:

   ```bash
   cargo run --release -p perovskite_client --bin headless_render -- \
     --server grpc://127.0.0.1:28275 --user agent --pass agentpass --register \
     --pos 0 14 -30 --az 200 --el -35 --width 800 --height 450 --out frame.png
   ```

   Use `--register` only on the first login to a given server instance. Keep `--release`
   (see above).

3. Read `frame.png` to check the result.

## Camera conventions

- `--pos X Y Z` is the eye position in world coordinates (blocks).
- `--az` is the azimuth in degrees: 0 looks toward +Z, 90 toward +X, 180 toward -Z.
- `--el` is the elevation in degrees: negative looks down.

## Useful flags

- `--no-hud` hides the crosshair and hotbar.
- `--render-distance N` (default 8 chunks) trades load time for view distance.
- `--near-radius N` (default 2, keep it at most 3) is how many chunks around the camera must
  arrive before the settle timer starts.
- `--settle-secs` / `--timeout-secs` control how long the drawable-chunk count must stay
  unchanged, and the overall cap on waiting.
- `--settings-from-disk` uses the interactive client's settings (e.g. for supersampling or HDR).

## Timing

With a release client against a debug dev server, expect about 5s to connect and about 10s
for the chunks near the camera. Waiting for the drawable-chunk count to settle can take another
30-40s, because a debug server generates and sends chunks slowly. Lower `--render-distance` or
`--settle-secs` if you only care about nearby blocks.
