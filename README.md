# &🧊 Perovskite

Perovskite is a 3D voxel game engine inspired by [Minetest](https://minetest.net).

It is written in Rust, with two goals:

* Give me an opportunity to learn Rust
* Create a decently playable game

***At this time, it is not a playable game.***

As this is a learning project, I may spend time on silly things that may not be useful to players or improved gameplay,
if I think I can learn something from doing so. Likewise, I may use design patterns inconsistently (or make other
inconsistencies) to increase the breadth of my learning. If there is enough interest or need, I can refactor and clean
up code later.

## Why the name?

Naming Rust projects after oxide minerals is fun. Unfortunately, all the cool oxide minerals are already used for really
cool rust projects, or could otherwise cause confusion.

[Perovskite](https://en.wikipedia.org/wiki/Perovskite) is a cool mineral that inspired artificial structures with a
bunch of
cool [aspirational applications](https://en.wikipedia.org/wiki/Perovskite_(structure)#Aspirational_applications).

## What's the current state?

It is possible to define blocks and dig them. Inventory support is mostly present. There's a furnace and a crafting
mechanism. Game content (i.e. all the blocks you'd expect) is not yet present.

Entities are under active development, but are not yet stable. The entity system is currently focused on
autonomous movement, and low-latency player control will be added in the future.

I intend to wrap the engine-specific behaviors in a lightweight API that makes it easier to define game logic. That API
is still not stable, but I intend to make that API more stable than the low-level engine API.

At the moment, the network API is not yet stable - the client and server must be built from the same code.
The intent is to stabilize the API to a reasonable extent later on, but having an unstable API allows for faster feature
iteration.

## How is it different from similar voxel games?

The limitations:

* The implementation is a bit more incoherent since it's a learning project
* The game lacks a lot of content
* A lot of basic features, like a player model, are still missing
* No scripting language yet - all game logic is written in Rust
    * And no dynamic content loading yet. It might be possible once crABI stabilizes.
* TLS support is a WIP. At this time, you need to bring your own certificate.
    * Let's Encrypt (certbot) works for this. One day, I'd like to implement automatic certificate provisioning using
      ACME + ALPN challenges. This ran into some problems on my first attempt.
    * TLS-secured servers should use address format `https://domain:port`
    * Unsecured servers should use the existing format `grpc://domain:port`

The benefits:

* The engine is optimized for scalability and performance where possible
    * Note that not all of the components have been optimized yet
* The minecarts can move at an unusually high speed, comparable to real-world HSR
* All game logic is written in Rust
* Authentication uses a secure remote password protocol that avoids sending the password over the network
* Communication is secured with TLS

## How do I play it?

### Prerequisites

A Rust toolchain is required; on Windows I use the MSVC toolchain for testing and development.

The minimum Rust version at this time is 1.87. This is subject to increase at any time (but the latest stable should
always be supported for the default build; i.e., no nightly features are required by default). Some experimental
features may require a nightly compiler.

On Linux, you'll need a handful of packages; Ubuntu/Debian users should
`apt install protobuf-compiler libasound-dev libdbus-1-dev pkg-config`

On Windows, you'll also need `protoc` and `ninja` (e.g. use these exact package names with chocolatey `choco install`)

### Server

First, build and run the server:

```
$ cargo build  --features= --bin perovskite_game_api --release
$ target/release/perovskite_game_api --data-dir /path/to/data-directory
```

The default server port is 28273.

The following `--features` can be used to do performance debugging of the server:

* `deadlock_detection` - Prints stacktraces when certain deadlocks occur
* `tracy` - Exports trace spans and metrics to [tracy](https://github.com/wolfpld/tracy).
    * Depending on your system configuration, thread stacks and thread scheduling events may be visible.
      e.g. on Windows, this requires running the game server as an admin.
* `dhat-heap` - Tracks heap allocations with dhat. Slow, and suspected to cause a rare deadlock.
* `tokio-console` - Exports tokio stats to [Tokio console](https://github.com/tokio-rs/console). Requires `$RUSTFLAGS`
  to contain `--cfg tokio_unstable`.

`--release` is recommended.

At the moment, the render distance is hardcoded at the server, using constants in
`src/network_server/client_context.rs`.

Use Ctrl+C to exit, and Ctrl+C a second time to force quit. At the moment, there are some deadlocks that cause
graceful shutdown to hang.

### Client

Then, build and run the client:

```
$ cargo build --features= --bin perovskite_client --release
$ target/release/perovskite_client
```

The only supported feature is `--features=tracy`, with similar behavior to the same feature on the server.

To reach a client on the local machine, use `grpc://localhost:28273` as the server address.

### Default keybinds

* move with WASD
* space to go up
* left-click to dig
* right-click to place
* F to interact with a block (e.g. furnace/chest).
* I for inventory
* P to toggle between normal physics, flying, and noclip. This is useful if you spawn underground.
    * left-control to descend
* Left-shift to sprint (if you have server permission to do so)
* Escape for menu

These can be adjusted in the settings file, whose filename is printed in the log when the client starts up.

## What are the requirements?

It runs on Windows and Linux. Mac is an open question due to issues involving Vulkan libraries.

Aspirationally, I'd like the game to be somewhat playable in single-player with only one or two CPU cores
and a very basic GPU.

I'm not quite there yet, and it's possible that I'll drift further away from this goal as I add features.
I tend to test with a fairly powerful laptop and a high-end GPU. I aim for the following performance goals:

* 165 FPS (matching my monitor refresh rate) and no noticeable interaction latency when using my gaming machine - to
  ensure there are not bottlenecks at the high end of scaling at lower render distance
* 60-90 FPS on max render distance
* 30-50 FPS on Intel integrated graphics when the client and server are limited to a few E-cores.
    * This will require both optimizations and better control over load (e.g. settings, dynamic render distance, etc)

## What's on the near-term roadmap?

Tons of stuff, in no particular order, and not yet prioritized. Everything in this list is a maybe, depending
on my available time and mood.

* Rendering and display:
    * TBD
    * Maybe a raytraced system?
    * Fog for distant chunks
* Game map
    * Further optimized APIs
        * Block visitors? (e.g. `for_each_connected` and similar taking closures and running them efficiently)
    * Out-of-line chunk loading (e.g. offloaded onto a dedicated executor)
    * Map bypass (i.e. stream chunks directly to the client for display)
* Block handling
    * Further refinements of signs/text
    * Implement place_on handler
    * Implement step_on handler and wire up to client context
    * Further client-side extended data
* Entities
    * Optimized (shader-assisted?) entity renderer
    * Simpler (tokio-driven) entity API (i.e. use a tokio task rather than the custom coroutine)
* Net code
    * Testing and optimization for slow WANs, avoiding head-of-line blocking
    * Player action validation
    * Adaptive and adjustable chunk load distance
* Content
    * Torch (doable w/ custom geometry + custom handlers)
    * Mapgen
        * Trees - keep refining
        * Sand and other surface material variety
        * Mapgen improvement
    * More simple tools (textures needed)
    * Cobblestone, bricks, etc
    * Minecarts
        * Integration with circuits for station automation
        * Freight minecarts (requires some work for persistence to avoid losing minecart contents during restart)
    * Pneumatic tubes
        * Tubes to send items between chests and furnaces, as a basis for some kind of machines later on
    * Circuit
        * In-inventory microcontroller (e.g. for handheld devices and smartcards)
           * Depends on extended data for inventory items 
* Further audio development
* Bugfixes for known issues
    * Trees intersect each other
    * Only binds to IPv6 on Windows

## What are the stability guarantees?

None at the moment. I test with either the latest or almost-latest stable Rust version, on Windows x64.

perovskite_server's API can change in breaking ways. perovskite_game_api (as well as anything it re-exports by default)
should be reasonably stable once it's written. I intend to re-export some unstable APIs behind a feature flag.

## Code style

* Formatted following `cargo fmt`
* No unsafe
* No nightly-only features; should build on stable Rust.

## Who is behind this?

I work on low-level infrastructure at a major hyperscaler and produce this game in my free time. You can find me on
discord as `drey7925`.

Note that this project is not endorsed, sponsored, or supported by my employer or any affiliates.

## AI policy

Some code generation and chore tasks in this project are automated with LLM technology. Please see AI_AGENTS.md for examples
of tasks that I've been automating.

Most of the engine is handwritten; I haven't been able to get LLMs to generate reliable enough code for those components
yet, and I strongly prefer to have direct design and implementation control anyway.

Aspirationally, after making some API improvements, context instruction files, etc, I hope to make code generation for
game _content_ viable, for a few reasons:

* I don't enjoy writing game content as much as engine code anyway, but feel that game content is necessary to showcase
  the engine
* Being able to automatically convert from pseudocode, Lua, etc to Rust would make modding and content creation accessible
  to a wider range of technical backgrounds

No AI is used for image generation, textures, "AI art" generation, etc.

### Pull requests and AI

Please indicate in any pull requests if code was primarily generated with an AI tool/agent.

Please do not contribute media (textures, audio, etc.) that you did not either photograph, record, or draw yourself.
Non-AI tools like image/audio editors, denoising, etc are fine.
