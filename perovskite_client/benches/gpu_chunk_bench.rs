use criterion::{black_box, criterion_group, criterion_main, Criterion};
use perovskite_client::vulkan::gpu_chunk_table::{gpu_table_lookup, ChunkHashtableBuilder};
use perovskite_core::coordinates::ChunkCoordinate;

fn build_hashtable(c: &mut Criterion) {
    let data = vec![0; 4096];
    let lights = vec![0; 4096];
    let mut builder = ChunkHashtableBuilder::new();
    for x in -15..15 {
        for y in -15..15 {
            for z in -15..15 {
                builder.add_chunk(
                    ChunkCoordinate::new(x, y, z),
                    data.as_slice(),
                    lights.as_slice(),
                );
            }
        }
    }
    c.bench_function("build_hashtable_10tries_27kchunk", |b| {
        b.iter(|| black_box(&builder).build(10, 3).unwrap())
    });

    let (table, header) = builder.build(10, 3).unwrap();

    c.bench_function("lookup_27kchunk_hit", |b| {
        let header_bb = black_box(&header);
        b.iter(|| {
            gpu_table_lookup(
                black_box(&table),
                header_bb,
                black_box(ChunkCoordinate::new(10, 10, 10)),
            )
        })
    });

    c.bench_function("lookup_27kchunk_miss", |b| {
        let header_bb = black_box(&header);
        b.iter(|| {
            gpu_table_lookup(
                black_box(&table),
                header_bb,
                black_box(ChunkCoordinate::new(30, 10, 10)),
            )
        })
    });
}

criterion_group!(benches, build_hashtable);
criterion_main!(benches);
