use criterion::async_executor::AsyncExecutor;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use perovskite_core::block_id::BlockId;
use perovskite_game_api::{
    carts, default_game::DefaultGameBuilder, game_builder::GameBuilder, BlockCoordinate,
};
use std::future::Future;
use tokio::runtime::Handle;

struct AsyncExecutorAdapter {
    handle: Handle,
}
impl AsyncExecutor for AsyncExecutorAdapter {
    fn block_on<T>(&self, future: impl Future<Output = T>) -> T {
        self.handle.block_on(future)
    }
}

fn map_benchmarks(c: &mut Criterion) {
    let (mut game, work_dir) = GameBuilder::using_tempdir().unwrap();
    game.initialize_default_game().unwrap();
    carts::register_carts(&mut game).unwrap();
    game.run_task_in_server(|gs| {
        c.bench_function("null_block_in_place", |b| {
            b.to_async(AsyncExecutorAdapter {
                handle: Handle::current(),
            })
            .iter(|| async move { tokio::task::block_in_place(move || {}) })
        });
        c.bench_function("null_spawn_blocking_await", |b| {
            b.to_async(AsyncExecutorAdapter {
                handle: Handle::current(),
            })
            .iter(|| async move { tokio::task::spawn_blocking(move || {}).await })
        });

        gs.game_map()
            .get_block(BlockCoordinate::new(0, 0, 0))
            .unwrap();

        c.bench_function("get_block", |b| {
            b.iter(|| {
                gs.game_map()
                    .get_block(black_box(BlockCoordinate::new(0, 0, 0)))
                    .unwrap();
            })
        });

        c.bench_function("set_block", |b| {
            b.iter(|| {
                gs.game_map()
                    .set_block(
                        black_box(BlockCoordinate::new(0, 0, 0)),
                        black_box(BlockId::from(0)),
                        black_box(None),
                    )
                    .unwrap();
            })
        });

        c.bench_function("try_get_block", |b| {
            b.iter(|| {
                gs.game_map()
                    .try_get_block(black_box(BlockCoordinate::new(0, 0, 0)))
                    .unwrap();
            })
        });

        Ok(())
    })
    .unwrap();

    std::fs::remove_dir_all(work_dir).unwrap();
}

criterion_group!(benches, map_benchmarks);
criterion_main!(benches);
