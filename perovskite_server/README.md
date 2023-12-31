This crate contains the game server component of Perovskite.

The API of this crate is unstable and subject to change at any time, including in backwards-incompatible ways.

Most plugins should link against `perovskite_game_api` instead, which includes base content. However, it is possible
to directly depend on this crate for lower level game logic.

If you're looking to *play* Perovskite, you can build `perovskite_game_api` with default features instead for a game server,
and `perovskite_client` for the client.