# &🧊 Cuberef

Cuberef is a 3D voxel game engine inspired by [Minetest](https://minetest.net).

It is written in Rust, with two goals:

* Give me an opportunity to learn Rust
* Create a decently playable game

***At this time, it is not a playable game.***

As this is a learning project, I may spend time on silly things that may not be useful to players or improved gameplay, if I think I can learn something from doing so. Likewise, I may use design patterns inconsistently (or make other inconsistencies) to increase the breadth of my learning. If there is enough interest or need, I can refactor and clean up code later.

## Why the name?

It's a silly pun related to Rust. Rust leans heavily on references, or refs for short, of various types (immutable refs, mutable refs, static refs, etc), and they are a fundamental aspect of the language.

**Note:** The name is likely to change soon.

## What's the current state?

It is possible to define blocks and dig them. Inventory support is mostly present. There's a furnace and a crafting mechanism. Game content (i.e. all the blocks you'd expect) is not yet present.

I intend to wrap the engine-specific behaviors in a lightweight API that makes it easier to define game logic. That API is still not stable, but I intend to make that API more stable than the low-level engine API.

At the moment, the network API is not yet stable - the client and server must be built from the same code.
The intent is to stabilize the API to a reasonable extent later on, but having an unstable API allows for faster feature iteration.

## How do I play it?

### Server

First, build and run the server:

```
$ cargo build  --features= --bin cuberef_game_api --release
$ target/release/cuberef_game_api --data-dir /path/to/data-directory
```

The default server port is 28273.

The following `--features` can be used to do performance debugging of the server:

* `deadlock_detection` - Prints stacktraces when certain deadlocks occur
* `tracy` - Exports trace spans and metrics to [tracy](https://github.com/wolfpld/tracy).
    * Depending on your system configuration, thread stacks and thread scheduling events may be visible.
      e.g. on Windows, this requires running the game server as an admin.
* `dhat-heap` - Tracks heap allocations with dhat. Slow, and suspected to cause a rare deadlock.
* `tokio-console` - Exports tokio stats to [Tokio console](https://github.com/tokio-rs/console). Requires `$RUSTFLAGS` to contain `--cfg tokio_unstable`.

`--release` is recommended.

At the moment, the render distance is hardcoded at the server, using constants in `src/network_server/client_context.rs`.

Use Ctrl+C to exit, and Ctrl+C a second time to force quit. At the moment, there are some deadlocks that cause
graceful shutdown to hang.

### Client

Then, build and run the client:

```
$ cargo build --features= --bin cuberef_client --release
$ target/release/cuberef_client
```

The only supported feature is `--features=tracy`, with similar behavior to the same feature on the server.

To reach a client on the local machine, use `grpc://localhost:28273` as the server address.

## What's on the near-term roadmap?

Tons of stuff, in no particular order, and not yet prioritized:

* Rendering and display:
    * Rotating blocks based on item variants
    * Non-cube blocks (e.g. torches, plants). Eventually custom geometry, but not there yet.
    * Lighting
* Block types
    * Liquids
* Game state and interactions:
    * Chat
    * Commands
* Game map
    * Support for falling blocks (e.g. sand)
    * Further optimized APIs
        * Block visitors? (e.g. `for_each_connected` and similar taking closures and running them efficiently)
        * Fast neighbor checks (e.g. for timers) that avoid re-locking and extra ops
    * Multithreaded mapgen?
* Entities
    * Initial design of non-block entities (e.g. other player characters)
    * TBD based on the initial design
* Net code
    * Caching images/resources
    * Testing and optimization for slow WANs
    * Player action validation
    * Adaptive and adjustable chunk load distance
* Content
    * Mapgen
        * Trees - keep refining
        * Sand and other surface material variety
        * Ores
        * Simple caves
        * Slightly more interesting elevation profile
    * Simple tools
    * Locked chests
    * Cobblestone, bricks, etc
* Bugfixes for known issues
    * Inconsistency with block ID assignment when unknown blocks are present
    * Trees intersect each other
    * Only binds to IPv6 on Windows

## What are the stability guarantees?

None at the moment. I test with either the latest or almost-latest stable Rust version, on Windows x64.

cuberef_server's API can change in breaking ways. cuberef_game_api (as well as anything it re-exports by default) should be reasonably stable once it's written. I intend to re-export some unstable APIs behind a feature flag.

## Who is behind this?

A pseudoanonymous software engineer (drey7925). This project is not endorsed, sponsored, or supported by my employer or any affiliates.
