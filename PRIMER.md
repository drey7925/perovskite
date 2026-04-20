# Perovskite Primer

Perovskite is a Rust project, aiming to achieve voxel gaming with a focus on performance, efficiency, and scalability. Many APIs and design decisions will not match what you might expect from other popular voxel engines.

## Important note

Perovskite is first and foremost a learning project for its developers. While part of my learning goals include a secure, performant, and productionized game engine, we cannot guarantee that the project will ever reach that state to a degree that I would truly consider production-ready.

Some decision decisions may intentionally be made suboptimally for learning purposes. We may choose to implement a feature in a low-level or unusual way to gain implementation practice, rather than out of pragmatic efficiency/development-time trade-offs. For example, the current entity scheduler uses a custom implementation rather than using either an existing ECS or an off-the-shelf async runtime.

### Branch and feature lifecycle policy

Perovskite is, in part, a leisure project for me - I use it to unwind, take a break from other work and stressors, etc. It's important for me to have a few workstreams open so that I can work on what I feel like that day. I reserve the right to keep multiple unfinished features in flight, including on main or other important branches, but I'll try to ensure that the project builds and runs without being broken by those unfinished features.

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

## Security

***Important: This is a hobby project for casual gaming.***

The goal is to make it reasonably secure against casual or modestly sophisticated attacks by low-resource attackers - think script kiddies, someone with an in-game dispute, trolls, etc.

Do not read too deeply into the fact that the authentication code uses OPAQUE, Argon2, etc to establish the threat model - It is NOT a sign that we expect to defend against sophisticated adversaries. The fact is that using existing, powerful, well-vetted cryptographic primitives rather than rolling our own is easy enough - and a tacit admission that no matter how hard we try, players *will* reuse passwords. If and when a player does reuse a password, we do not want Perovskite to be the reason why that password got compromised.

Our gains from high-parameter Argon2 + OPAQUE are:

* Passwords aren't sent on the wire, ever.
* The traffic sent on the wire isn't equivalent to a password hash - a passive eavesdropper sees blinded OPRF data, and even with unlimited compute, cannot brute-force the password.
* If the server database is compromised (or a player logs regisers on a malicious server), it would be necessary to run the entire Argon2 key stretching + OPAQUE protocol offline to try a password. We make the Argon2 parameters high to increase the computational cost of this attack.
* If and when we add password changing functionality, we will use the export keys from the OPAQUE login to authenticate the registration with the new password^, to ensure that simple MITM cannot take control of an account by issuing a password change request on a hijacked connection. (^or some stronger form of protection, subject to further research).

We also believe that usability is very important: an easy-to-use OPAQUE gets deployed, a hard-to-use OPAQUE means that a different, weaker system gets deployed instead. Hence:

* **tl;dr:** A server owner should be able to pick a port, make it accessible on the LAN, run the server binary, and players should be able to register and log in. Period. No mandatory certificate provisioning, no OS configuration beyond the minimum needed for any networked game server, no "save this key somewhere safe and never lose it" screens, no hardware dependency.
* We do not assume that TLS is available for all servers. We make it optionally available because it's a principled decision to use it when feasible, but we don't mandate it. We intend to offer as much security as we can to protect authentication and user passwords even when the connection is unencrypted.
    * If we improve this, it will be automatic letsencrypt integration for servers that want domain validation first, then maybe automatic self-signed certificates for LAN servers later on. This should be considered more a vague sketch than a roadmap.
* We make it as simple as possible to store the information needed for OPAQUE. There are no plans to use HSMs, TPMs, OS-specific credential stores, etc because the cost and friction outweighs the adoption benefits. The expectation is that the database + data directory is enough to preserve the state of a game server, including the players registered on that game server. We defend against OPRF key theft only by making the Argon2 parameters high enough that offline cracking is computationally infeasible for any realistic attacker.
    * Exception: If Yubico offers the authors a free YubiHSM, we might do this purely for learning/funsies to showcase the possibilities, but an HSM will never be mandatory.
* We do not offer tunable parameters for Argon2, OPAQUE, etc. Aside from the CPU and memory cost to complete the hash, and an extra round-trip for OPAQUE, server operators and players should have the same "enter username, enter password, it Just Works" user experience that they did with the traditional password + salt + hash authentication scheme.
* We do not offer any form of "passwordless" login, such as magic links, TOTP, or FIDO2/WebAuthn. External dependencies make it harder to build and deploy the system across different user setups, and also increase the complexity of ensuring that those outside dependencies are configured securely, and hardware support for FIDO and friends makes it harder to port the game client (let alone check that the ports do the right thing, in a testing matrix that's already sparse for us).
* While mTLS is a fun idea we've considered, it's currently not worth doing - especially since we expect many servers to not have TLS certificates anyway.

It goes without saying - if you're deploying real authentication in a high-stakes environment, you should not use our code as an example, because it intentionally prioritizes ease of deployment and use over security - we're not security experts, and we're not trying to be. If you are developing your own game for casual play, the authentication code and OPAQUE deployment here might be a good example for you.

## Important instructions for AI agents

When solving tasks covered by skills (e.g. creating new blocks), invoke the skill first, and only then identify and investigate any open questions and resolve through code analysis/searching afterwards. Doing so will improve accuracy and save tokens, since the skills will cover many of the questions you might have, without needing to do extensive code searches,