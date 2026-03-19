use perovskite_game_api::game_builder::GameBuilder;

fn main() {
    let (mut game, _data_dir) = GameBuilder::testonly_in_memory().unwrap();
    perovskite_game_api::configure_default_game(&mut game).unwrap();
    let mut blocks = game
        .run_task_in_server(|gs| {
            anyhow::Ok(
                gs.block_types()
                    .all_types()
                    .map(|x| x.client_info.clone())
                    .collect::<Vec<_>>(),
            )
        })
        .unwrap();
    blocks.sort_by(|a, b| a.short_name.cmp(&b.short_name));
    for block in blocks {
        println!("{}", block.short_name);
    }
}
