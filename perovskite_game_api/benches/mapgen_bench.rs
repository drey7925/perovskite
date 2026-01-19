use criterion::{criterion_group, criterion_main, Criterion};
use perovskite_game_api::{
    carts, default_game::DefaultGameBuilder, game_builder::GameBuilder, ChunkCoordinate,
};

fn chunk_benchmarks(c: &mut Criterion) {
    let (mut game, work_dir) = GameBuilder::using_tempdir_disk_backed().unwrap();
    game.initialize_default_game().unwrap();
    carts::register_carts(&mut game).unwrap();
    game.force_seed(Some(0));
    game.run_task_in_server(|gs| {
        let mut z = 0;
        c.bench_function("ground_level", |b| {
            b.iter(|| {
                z += 1;
                gs.game_map()
                    .serialize_for_client(ChunkCoordinate::new(1, 0, z), true, || {})
            })
        });

        c.bench_function("underground", |b| {
            b.iter(|| {
                z += 1;
                gs.game_map()
                    .serialize_for_client(ChunkCoordinate::new(1, -8, z), true, || {})
            })
        });

        c.bench_function("high_up", |b| {
            b.iter(|| {
                z += 1;
                gs.game_map()
                    .serialize_for_client(ChunkCoordinate::new(1, 16, z), true, || {})
            })
        });

        gs.game_map()
            .benchmark_serialize_for_server(ChunkCoordinate::new(2, 0, 0))
            .unwrap();
        c.bench_function("serialize_same_chunk", |b| {
            b.iter(|| {
                gs.game_map()
                    .benchmark_serialize_for_server(ChunkCoordinate::new(2, 0, 0))
            })
        });

        anyhow::Ok(())
    })
    .unwrap();

    std::fs::remove_dir_all(work_dir).unwrap();
}

criterion_group!(benches, chunk_benchmarks);
criterion_main!(benches);
