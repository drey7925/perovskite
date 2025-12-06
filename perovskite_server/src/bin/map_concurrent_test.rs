use cgmath::num_traits::WrappingAdd;
use parking_lot::Mutex;
use perovskite_core::coordinates::{BlockCoordinate, ChunkCoordinate, ChunkOffset};
use perovskite_core::util::TraceBuffer;
use perovskite_server::game_state::event::run_traced_sync;
use perovskite_server::game_state::GameState;
use perovskite_server::server::{testonly_in_memory, GameDatabase, Server};
use rand::rngs::ThreadRng;
use rand::{Rng, RngCore};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier, OnceLock};
use std::thread::sleep;
use std::time::Duration;
use tokio::time::Instant;

struct SlowLoadSaveDb {
    base: Arc<dyn perovskite_server::database::GameDatabase>,
}
#[allow(dead_code)]
impl SlowLoadSaveDb {
    thread_local! {
        static GET_DELAY: AtomicU64 = AtomicU64::new(0);
        static PUT_DELAY: AtomicU64 = AtomicU64::new(0);
    }
    fn get_delay() -> u64 {
        Self::GET_DELAY.with(|v| v.load(Ordering::Relaxed))
    }
    fn put_delay() -> u64 {
        Self::PUT_DELAY.with(|v| v.load(Ordering::Relaxed))
    }
    fn do_get_delay() {
        let delay = Self::get_delay();
        if delay > 0 {
            std::thread::sleep(std::time::Duration::from_micros(delay));
        }
    }
    fn do_put_delay() {
        let delay = Self::put_delay();
        if delay > 0 {
            std::thread::sleep(std::time::Duration::from_micros(delay));
        }
    }
    fn set_get_delay(delay: u64) {
        Self::GET_DELAY.with(|v| v.store(delay, Ordering::Relaxed));
    }
    fn set_put_delay(delay: u64) {
        Self::PUT_DELAY.with(|v| v.store(delay, Ordering::Relaxed));
    }
}
impl GameDatabase for SlowLoadSaveDb {
    fn get(&self, key: &[u8]) -> anyhow::Result<Option<Vec<u8>>> {
        Self::do_get_delay();
        self.base.get(key)
    }

    fn put(&self, key: &[u8], value: &[u8]) -> anyhow::Result<()> {
        Self::do_put_delay();
        self.base.put(key, value)
    }

    fn delete(&self, key: &[u8]) -> anyhow::Result<()> {
        Self::do_put_delay();
        self.base.delete(key)
    }

    fn flush(&self) -> anyhow::Result<()> {
        Self::do_put_delay();
        self.base.flush()
    }

    fn read_prefix(
        &self,
        prefix: &[u8],
        callback: &mut dyn FnMut(&[u8], &[u8]) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        Self::do_get_delay();
        self.base.read_prefix(prefix, callback)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct WorkerStats {
    ops: usize,
    write_checksum: u32,
    duration: Duration,
}
trait Action: Send + Sync + 'static {
    fn act(&mut self, gs: &GameState, rng: &mut ThreadRng, stats: &mut WorkerStats);
    fn describe(&self) -> String {
        format!("{:?}", std::any::type_name::<Self>())
    }
}

struct RandRead(Range<i32>);
impl Action for RandRead {
    fn act(&mut self, gs: &GameState, rng: &mut ThreadRng, stats: &mut WorkerStats) {
        let val = rng.gen_range(self.0.clone());
        let coord = ChunkCoordinate::new(val >> 4 & 0xf, val >> 8, val & 0xf);
        gs.game_map()
            .get_block(coord.with_offset(ChunkOffset::new(0, 0, 0)))
            .unwrap();
        stats.ops += 1;
    }
    fn describe(&self) -> String {
        "randread".into()
    }
}

struct FlushAll {
    freq: f32,
    next_awaken: Option<Instant>,
}
impl Action for FlushAll {
    fn act(&mut self, gs: &GameState, _rng: &mut ThreadRng, stats: &mut WorkerStats) {
        let now = Instant::now();
        let next_awaken = self.next_awaken.get_or_insert_with(Instant::now);
        sleep(*next_awaken - now);
        *next_awaken += Duration::from_secs_f32(1.0 / self.freq);
        gs.game_map().purge_and_flush();
        stats.ops += 1;
    }
    fn describe(&self) -> String {
        "flush_all".into()
    }
}

struct RandWrite(Range<i32>);
impl Action for RandWrite {
    fn act(&mut self, gs: &GameState, rng: &mut ThreadRng, stats: &mut WorkerStats) {
        let val = rng.gen_range(self.0.clone());
        let inc = rng.next_u32();
        let coord = ChunkCoordinate::new(val >> 4 & 0xf, val >> 8, val & 0xf);
        gs.game_map()
            .mutate_block_atomically(coord.with_offset(ChunkOffset::new(0, 0, 0)), |b, _e| {
                b.0 = b.0.wrapping_add(inc);
                Ok(())
            })
            .unwrap();
        stats.ops += 1;
        stats.write_checksum = stats.write_checksum.wrapping_add(inc);
    }
    fn describe(&self) -> String {
        "randwrite".into()
    }
}

struct Worker<A: Action> {
    server: Arc<Server>,
    barrier: Arc<(std::sync::Barrier, AtomicBool)>,
    result: Arc<OnceLock<WorkerStats>>,
    action: Mutex<A>,
    setup: Box<dyn Fn() + Send + Sync + 'static>,
}

impl<A: Action> Worker<A> {
    fn new(
        server: Arc<Server>,
        barrier: Arc<(Barrier, AtomicBool)>,
        action: A,
        setup: impl Fn() + Send + Sync + 'static,
    ) -> Self {
        Self {
            server,
            barrier,
            result: Arc::new(OnceLock::new()),
            action: Mutex::new(action),
            setup: Box::new(setup),
        }
    }
    fn start(self: Arc<Self>) {
        let thread_name = self.action.lock().describe();
        std::thread::Builder::new()
            .name(thread_name)
            .spawn(move || {
                let mut stats = WorkerStats::default();
                self.server
                    .run_task_in_server(|gs| {
                        (self.setup)();

                        let mut rng = rand::thread_rng();
                        let mut action = self.action.lock();
                        self.barrier.0.wait();
                        let start = std::time::Instant::now();
                        while self.barrier.1.load(Ordering::Relaxed) {
                            for _ in 0..16 {
                                action.act(gs, &mut rng, &mut stats);
                            }
                        }
                        stats.duration = start.elapsed();
                        Ok(())
                    })
                    .unwrap();

                self.result.set(stats).unwrap();
            })
            .unwrap();
    }
    fn join(&self) -> WorkerStats {
        self.result.wait().clone()
    }
}

fn main() {
    const FLUSH_HZ: f32 = 10.0;
    const RUN_PROBE: bool = true;
    const N_WRITERS: usize = 2;
    const LOAD_SLEEP_TIME_MICROS: u64 = 1000;
    const WORKING_SET_SIZE: i32 = 256;
    for n_read_thread in [0, 1, 2, 4, 8, 12, 16] {
        let server = Arc::new(
            // testonly_in_memory_with_db(Arc::new(SlowLoadSaveDb {
            //     base: Arc::new(InMemGameDatabase::new()),
            // }))
            testonly_in_memory().unwrap(),
        );
        println!("\n{N_WRITERS}x write + {n_read_thread}x randread, random chunk from working set of {WORKING_SET_SIZE}, forced flush at {FLUSH_HZ} Hz");

        let barrier = Arc::new((
            Barrier::new(n_read_thread + N_WRITERS + 1),
            AtomicBool::new(true),
        ));

        let mut read_workers = vec![];
        let mut write_workers = vec![];
        for _ in 0..n_read_thread {
            read_workers.push(Arc::new(Worker::new(
                server.clone(),
                barrier.clone(),
                RandRead(0..WORKING_SET_SIZE),
                || {
                    SlowLoadSaveDb::set_get_delay(LOAD_SLEEP_TIME_MICROS);
                },
            )));
        }
        for _ in 0..N_WRITERS {
            write_workers.push(Arc::new(Worker::new(
                server.clone(),
                barrier.clone(),
                RandWrite(0..WORKING_SET_SIZE),
                || {
                    SlowLoadSaveDb::set_get_delay(LOAD_SLEEP_TIME_MICROS);
                },
            )));
        }
        let flusher = Arc::new(Worker::new(
            server.clone(),
            barrier.clone(),
            FlushAll {
                freq: FLUSH_HZ,
                next_awaken: None,
            },
            || {
                SlowLoadSaveDb::set_get_delay(LOAD_SLEEP_TIME_MICROS);
            },
        ));

        for worker in read_workers.iter() {
            worker.clone().start();
        }
        for worker in write_workers.iter() {
            worker.clone().start();
        }
        flusher.clone().start();
        sleep(Duration::from_secs(5));

        if RUN_PROBE {
            server
                .run_task_in_server(|gs| {
                    let tracer = TraceBuffer::new(true);
                    run_traced_sync(tracer, || {
                        gs.game_map().get_block(BlockCoordinate::new(0, 0, 0))
                    })
                })
                .unwrap();
        }

        sleep(Duration::from_secs(5));
        barrier.1.store(false, Ordering::Relaxed);
        // for worker in workers.iter() {
        //     println!("{:?}", worker.join())
        // }
        let mut read_rate = 0.0;
        for worker in read_workers.iter() {
            let stats = worker.join();
            read_rate += stats.ops as f64 / stats.duration.as_secs_f64();
        }
        println!("flush: {:?}", flusher.join());

        let mut write_rate = 0.0;
        let mut expected_write_cksum: u32 = 0;

        for worker in write_workers.iter() {
            let stats = worker.join();
            write_rate += stats.ops as f64 / stats.duration.as_secs_f64();
            expected_write_cksum = expected_write_cksum.wrapping_add(stats.write_checksum);
        }

        println!("Reads/sec: {read_rate}");
        println!("Writes/sec: {write_rate}");

        let mut checksum = 0;
        server
            .run_task_in_server(|gs| {
                for i in 0..WORKING_SET_SIZE {
                    let coord = ChunkCoordinate::new(i >> 4 & 0xf, i >> 8, i & 0xf);
                    checksum = checksum.wrapping_add(
                        &gs.game_map()
                            .get_block(coord.with_offset(ChunkOffset::new(0, 0, 0)))
                            .unwrap()
                            .0,
                    );
                }
                Ok(())
            })
            .unwrap();
        println!(
            "Checksum actual {:x}, expected {:x}",
            checksum, expected_write_cksum
        );
    }
}
