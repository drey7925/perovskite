fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let engine = rt
        .block_on(async { perovskite_client::audio::start_engine_for_standalone_test().await })
        .unwrap();
    std::thread::sleep(std::time::Duration::from_secs(60));
}
