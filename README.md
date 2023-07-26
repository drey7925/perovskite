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

It is possible to define blocks and dig them. Inventory support is partially present.

I intend to wrap the engine-specific behaviors in a lightweight API that makes it easier to define game logic. That API has not yet been designed or written.

## What's on the near-term roadmap?

Tons of stuff, in no particular order, and not yet prioritized:

* Rendering and display:
    * Rotating blocks based on item variants
    * Non-cube blocks (e.g. torches, plants). Eventually custom geometry, but not there yet.
    * Lighting
* Game state and interactions:
    * Chat
    * Commands
* Game map
    * Support for falling blocks (e.g. sand)
    * Further optimized APIs
        * Block visitors? (e.g. `for_each_connected` and similar taking closures and running them efficiently)
* Entities
    * Initial design of non-block entities (e.g. other player characters)
    * TBD based on the initial design
* Net code
    * Caching images/resources
    * Testing and optimization for slow WANs
    * Player action validation
* Content
    * Mapgen
        * Trees
        * Sand and other surface material variety
        * Ores
        * Simple caves
        * Slightly more interesting elevation profile
    * Simple tools
    * Locked chests
    * Cobblestone, bricks, etc
* Bugfixes for known issues
    * Inconsistency with block ID assignment when unknown blocks are present

## What are the stability guarantees?

None at the moment. I test with either the latest or almost-latest stable Rust version, on Windows x64.

cuberef_server's API can change in breaking ways. cuberef_game_api (as well as anything it re-exports by default) should be reasonably stable once it's written. I intend to re-export some unstable APIs behind a feature flag.

## Who is behind this?

A pseudoanonymous software engineer (drey7925). This project is not endorsed, sponsored, or supported by my employer or any affiliates.
