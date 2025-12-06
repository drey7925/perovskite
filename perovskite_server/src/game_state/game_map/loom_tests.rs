use std::env::temp_dir;

use crate::database::InMemGameDatabase;
use crate::game_state::game_map::{BackgroundTaskMode, ServerGameMap};
use crate::server::{testonly_in_memory, Server, ServerArgs};
use perovskite_core::coordinates::BlockCoordinate;
use perovskite_core::sync::{DefaultSyncBackend, SyncBackend, TestonlyLoomBackend};
use std::sync::Arc;

#[test]
fn test_load_store_purge() {
    let server = Arc::new(testonly_in_memory().unwrap());
    server
        .run_task_in_server(|_gs| {
            let server = server.clone();
            let mut loom = loom::model::Builder::default();
            loom.preemption_bound = Some(3);
            loom.check(move || {
                let loom_map = make_loom_map::<DefaultSyncBackend>(&server);

                let block = loom_map.get_block(BlockCoordinate::new(0, 0, 0)).unwrap();
                assert_eq!(block.0, 0);

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
                        if block.0 > 4 {
                            panic!("block was {}", block.0);
                        }
                    }
                }));

                let map = loom_map.clone();
                threads.push(loom::thread::spawn(move || {
                    map.purge_and_flush();
                }));

                for thread in threads {
                    thread.join().unwrap();
                }
                let block = loom_map.get_block(BlockCoordinate::new(0, 0, 0)).unwrap();
                assert_eq!(block.0, 4);
                tokio::runtime::Handle::current()
                    .block_on(loom_map.do_shutdown())
                    .unwrap();
            });
            Ok(())
        })
        .unwrap();
}

#[test]
fn test_lighting() {
    let server = Arc::new(testonly_in_memory().unwrap());
    server
        .run_task_in_server(|_gs| {
            let server = server.clone();
            let mut loom = loom::model::Builder::default();
            loom.preemption_bound = Some(2);
            loom.check(move || {
                let loom_map = make_loom_map::<TestonlyLoomBackend>(&server);
                let coords = [
                    BlockCoordinate::new(0, 0, 0),
                    BlockCoordinate::new(0, 64, 0),
                    BlockCoordinate::new(0, 256, 0),
                ];
                for coord in coords {
                    assert_eq!(loom_map.get_block(coord).unwrap().0, 0);
                }

                let mut threads = vec![];

                for coord in coords {
                    let map = loom_map.clone();
                    threads.push(loom::thread::spawn(move || {
                        for val in [0, 65536, 0, 65536] {
                            map.mutate_block_atomically(coord, |b, _e| {
                                b.0 = val;
                                Ok(())
                            })
                            .unwrap();
                        }
                    }));
                }

                let map = loom_map.clone();
                threads.push(loom::thread::spawn(move || {
                    map.purge_and_flush();
                }));
                for thread in threads {
                    thread.join().unwrap();
                }
                for coord in coords {
                    assert_eq!(loom_map.get_block(coord).unwrap().0, 65536);
                }
                tokio::runtime::Handle::current()
                    .block_on(loom_map.do_shutdown())
                    .unwrap();
            });
            Ok(())
        })
        .unwrap();
}

/// Makes a new empty Loom map with the specified lighting lock backend <L>. If L is
/// TestonlyLoomBackend, Loom interleaves light operations. If L is a non-Loom backend,
/// only the main game map concurrency is tested.
fn make_loom_map<L: SyncBackend>(
    server: &Arc<Server>,
) -> Arc<ServerGameMap<TestonlyLoomBackend, L>> {
    let loom_map_backing_store = Arc::new(InMemGameDatabase::new());
    let loom_map = ServerGameMap::<TestonlyLoomBackend, L>::new_with_background_tasks(
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
    loom_map
}
