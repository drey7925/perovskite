This crate contains core definitions shared between Perovskite clients and servers. It doesn't make
sense to use on its own, unless you're writing some kind of middleware, loadbalancer, custom client,
protocol translator, etc.

The constants and definitions will generally offer a somewhat stable API; the network protocol may evolve quickly, with
protocol version mismatches mediated by logic in the client and server crates.

However, its definitions are useful in conjunction with the APIs in either `perovskite_game_api` (if you want to write game content) or `perovskite_server` (if you need lower level engine access).

Please see the `perovskite_game_api` and `perovskite_client` crates to run a game server and join a game, respectively.
