use std::{thread, time::Duration};

use rand::Rng;

fn main() {
    let _tracy_client = tracy_client::Client::start();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let mut lt = perovskite_client::loadtester::Loadtester::new();
    let rand = rand::thread_rng().gen_range(0..65536);
    for i in 0..100 {
        lt.start_worker(
            "grpc://localhost:28273".to_string(),
            format!("test_{:x}_{}", rand, i),
        );
        thread::sleep(Duration::from_millis(500));
    }
    thread::sleep(Duration::MAX);
}
