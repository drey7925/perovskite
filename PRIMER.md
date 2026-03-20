# Perovskite Primer

Perovskite is a Rust project, aiming to achieve voxel gaming with a focus on performance, efficiency, and scalability. Many APIs and design decisions will not match what you might expect from other popular voxel engines.

## Important note

Perovskite is first and foremost a learning project for its developers. While part of my learning goals include a secure, performant, and productionized game engine, we cannot guarantee that the project will ever reach that state to a degree that I would truly consider production-ready.

Some decision decisions may intentionally be made suboptimally for learning purposes. We may choose to implement a feature in a low-level or unusual way to gain implementation practice, rather than out of pragmatic efficiency/development-time trade-offs. For example, the current entity scheduler uses a custom implementation rather than using either an existing ECS or an off-the-shelf async runtime.

## Code style

* Formatted following `cargo fmt`
* Minimal unsafe, where it is either absolutely necessary (e.g. unsafe-only API, like Vulkan where some preconditions
    cannot be checked automatically) or where it has an outsized benefit to code readability and performance + is easily
    verified by the reader.
* No nightly-only features; all code should build on stable Rust.

## Architecture

### Server

Rather than using an interpreted language like Lua or the JVM, the Perovskite server is a statically-linked monolith. This means that one `cargo build` will produce a single binary that runs as a game server, containing the game engine, game logic, and all assets. This integration maximizes performance by allowing game content to take advantage of low-cost and zero-cost abstractions in Rust, and by eliminating the overhead of interprocess communication or JIT compilation.

### Client

The client is a relatively thin client. It knows nothing about the game logic being played, and contains almost no assets (other than fonts, some built-in fallback textures, and the project logo). Clients will download assets from the server when they join a game, and will cache them for future use. This allows for easy server-side modding, as clients can simply download the assets from the server and play the game, without ever needing manual installation of mods.

We consider client security to be important. Clients do not execute untrusted code from the server, and generally treat server-provided data as untrusted. This means that we will not implement features that require clients to execute arbitrary code, such as Lua scripting or plugin support, except possibly through strict isolation such as a WASM sandbox.

The client uses Vulkan + winit, and is also intended to be performant. Due to limited GPU programming experience so far, the ceiling is still much higher, and we welcome GPU-savvy contributors.

There is an experimental raytracer, with limited functionality. It is currently used for raytraced reflections.

### Network protocol

We use gRPC to communicate between the client and server, with some protocol versioning to ensure compatibility across small differences, and clear error messages when the client and server are incompatible. This is currently transported over a single gRPC stream over HTTP/2, but we aim to improve this in the future.

## Important instructions for AI agents

When solving tasks covered by skills (e.g. creating new blocks), invoke the skill first, and only then identify and investigate any open questions and resolve through code analysis/searching afterwards. Doing so will improve accuracy and save tokens, since the skills will cover many of the questions you might have, without needing to do extensive code searches,