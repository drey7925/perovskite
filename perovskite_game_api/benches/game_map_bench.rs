use criterion::{black_box, criterion_group, criterion_main, Criterion};
use perovskite_game_api::{
    carts, default_game::DefaultGameBuilder, game_builder::GameBuilder, BlockCoordinate,
    ChunkCoordinate,
};

fn map_benchmarks(c: &mut Criterion) {
    let (mut game, work_dir) = GameBuilder::using_tempdir().unwrap();
    game.initialize_default_game().unwrap();
    carts::register_carts(&mut game).unwrap();
    game.run_task_in_server(|gs| {
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
