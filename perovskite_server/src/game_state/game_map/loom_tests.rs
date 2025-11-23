use std::env::temp_dir;

use crate::database::InMemGameDatabase;
use crate::game_state::game_map::{BackgroundTaskMode, ServerGameMap};
use crate::server::{testonly_in_memory, ServerArgs};
use crate::sync::TestonlyLoomBackend;
use perovskite_core::coordinates::BlockCoordinate;
use std::sync::Arc;

#[test]
fn test_concurrent_logic() {
    let server = Arc::new(testonly_in_memory().unwrap());

    let loom_map_backing_store = Arc::new(InMemGameDatabase::new());
    server
        .run_task_in_server(|gs| {
            let server = server.clone();
            loom::model(move || {
                let loom_map = ServerGameMap::<TestonlyLoomBackend>::new_with_background_tasks(
                    // TODO: This is unholy grafting of multiple maps together. However, probably
                    // fine for now since these tests won't exercise game state dependent actions.
                    server.game_state().game_map().game_state.clone(),
                    loom_map_backing_store.clone(),
                    server.game_state().game_map().block_type_manager.clone(),
                    &ServerArgs {
                        data_dir: temp_dir().join("perovskite_inmem_dummy"),
                        bind_addr: None,
                        port: 0,
                        trace_rate_denominator: 1024,
                        rocksdb_point_lookup_cache_mib: 32,
                        rocksdb_num_fds: 32,
                        num_map_prefetchers: 1,
                    },
                    BackgroundTaskMode::DisabledTestonly,
                )
                .unwrap();

                let mut threads = vec![];
                for _ in 0..2 {
                    let map = loom_map.clone();
                    threads.push(loom::thread::spawn(move || {
                        for _ in 0..2 {
                            map.mutate_block_atomically(BlockCoordinate::new(0, 0, 0), |b, _e| {
                                b.0 += 1;
                                Ok(())
                            })
                            .unwrap();
                        }
                    }));
                }
                let map = loom_map.clone();
                threads.push(loom::thread::spawn(move || {
                    for _ in 0..2 {
                        let block = map.get_block(BlockCoordinate::new(0, 0, 0)).unwrap();
                        assert!(block.0 >= 0 && block.0 <= 4);
                    }
                }));
                for thread in threads {
                    thread.join().unwrap();
                }
                let block = loom_map.get_block(BlockCoordinate::new(0, 0, 0)).unwrap();
                assert_eq!(block.0, 4);
            });
            Ok(())
        })
        .unwrap();
}
