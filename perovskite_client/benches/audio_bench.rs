use criterion::{criterion_group, criterion_main, Criterion};
use perovskite_client::audio::generated_eqns::travel_time_newton_raphson;
use std::hint::black_box;

fn travel_time_newton_raphson_benchmark(c: &mut Criterion) {
    c.bench_function("nr_4iters", |b| {
        b.iter(|| {
            let mut dt = 0.0;
            let rt = black_box(0.0);
            let px = black_box(-20.0);
            let py = black_box(1.0);
            let pz = black_box(0.0);
            let vx = black_box(90.0);
            let vy = black_box(0.0);
            let vz = black_box(0.0);
            let ax = black_box(0.0);
            let ay = black_box(0.0);
            let az = black_box(9.0);
            for _ in 0..4 {
                dt = travel_time_newton_raphson(dt, rt, px, py, pz, vx, vy, vz, ax, ay, az);
            }
            dt
        })
    });
    c.bench_function("nr_8iters", |b| {
        b.iter(|| {
            let mut dt = 0.0;
            let rt = black_box(0.0);
            let px = black_box(-20.0);
            let py = black_box(1.0);
            let pz = black_box(0.0);
            let vx = black_box(90.0);
            let vy = black_box(0.0);
            let vz = black_box(0.0);
            let ax = black_box(0.0);
            let ay = black_box(0.0);
            let az = black_box(9.0);
            for _ in 0..8 {
                dt = travel_time_newton_raphson(dt, rt, px, py, pz, vx, vy, vz, ax, ay, az);
            }
            dt
        })
    });
}

criterion_group!(benches, travel_time_newton_raphson_benchmark);
criterion_main!(benches);
