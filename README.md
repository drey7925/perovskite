# &🧊 Cuberef

Cuberef is a 3D voxel game engine inspired by [Minetest](https://minetest.net).

It is written in Rust, with two goals:

* Give me an opportunity to learn Rust
* Create a decently playable game

***At this time, it is not a playable game.***

As this is a learning project, I may spend time on silly things that may not be useful to players or improved gameplay, if I think I can learn something from doing so. Likewise, I may use design patterns inconsistently (or make other inconsistencies) to increase the breadth of my learning. If there is enough interest or need, I can refactor and clean up code later.

## Why the name?

It's a silly pun related to Rust. Rust leans heavily on references, or refs for short, of various types (immutable refs, mutable refs, static refs, etc), and they are a fundamental aspect of the language.

## What's the current state?

It is possible to define blocks and dig them. Inventory support is underway.

I intend to wrap the engine-specific behaviors in a lightweight API that makes it easier to define game logic. That API has not yet been designed or written.

## What are the stability guarantees?

None at the moment. The MSRV is the latest stable version. I test the client on Windows x86_64, and the server on Windows x86_64 and Linux x86_64.

cuberef_server's API can change in breaking ways. cuberef_game_api (as well as anything it re-exports by default) should be reasonably stable once it's written. I intend to re-export some unstable APIs behind a feature flag.

## Who is behind this?

A pseudoanonymous software engineer (drey7925). This project is not endorsed, sponsored, or supported by my employer or any affiliates.
