---
name: entity_testing
description: Testing strategies and infrastructure for game entity behavior in perovskite_game_api. Use when asked to write tests for entities, including animals, minecarts, etc.
---

## Testing Entity Coroutines

Entity movement logic (`EntityCoroutine` implementations) is tested with `CoroutineTester` from
`perovskite_server::game_state::entities::entity_test_helpers`. It drives the coroutine
synchronously step-by-step inside `run_assertions_in_server` (which provides the required async
runtime).

```rust
use perovskite_server::game_state::entities::entity_test_helpers::CoroutineTester;
use perovskite_server::game_state::entities::MoveQueueType;

fixture.run_assertions_in_server(|gs| {
    let start_pos = Vector3::new(0.0, 0.5, 0.0);
    let coro = MyCoroutine { ... };

    // Constructor calls advance() once; start_pos must be in a loaded chunk
    let mut tester = CoroutineTester::new(
        Box::pin(coro),
        MoveQueueType::SingleMove,  // or Buffer8, Buffer64
        start_pos,
        gs,
    ).or_fail()?;

    // Step one planned move
    tester.advance(gs).or_fail()?;

    // Inspect state
    println!("pos={:?} buffer={}s engaged={}",
        tester.current_position(),   // position before current move completes
        tester.move_buffer(),        // total seconds currently queued
        tester.is_engaged(),         // false after StopCoroutineControl/ImmediateDespawn
    );

    // Position after all queued moves drain
    let final_pos = tester.post_queue_position();

    Ok(())
})?;
```

`CoroutineTester::advance` must run inside an async runtime because deferred moves call
`tokio::runtime::Handle::current().block_on(...)`. The `run_assertions_in_server` closure already
satisfies this. The chunks around `start_pos` are warmed up automatically by each `advance` call.

See `perovskite_game_api/src/animals/mod.rs` for a full example (`duck_coroutine_smoke`).