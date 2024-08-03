use cgmath::{Vector3, Zero};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let engine = perovskite_client::audio::start_engine_for_testing().unwrap();
    std::thread::sleep(std::time::Duration::from_secs(60));
}
