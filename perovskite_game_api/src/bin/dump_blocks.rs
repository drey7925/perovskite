use perovskite_game_api::game_builder::GameBuilder;

fn main() {
    let (mut game, _data_dir) = GameBuilder::testonly_in_memory().unwrap();
    perovskite_game_api::configure_default_game(&mut game).unwrap();
    let mut blocks = game
        .run_task_in_server(|gs| {
            anyhow::Ok(
                gs.block_types()
                    .all_types()
                    .map(|bt| {
                        let mut client_info = bt.client_info.clone();
                        // Exclude block ID — it's an artifact of registration order
                        // Within a game, block IDs are kept consistent through persistent
                        // mapping tables in the database, even as registration order changes
                        // or new blocks are added.
                        //
                        // This tool runs against a fresh database, so there are no
                        // persistent mappings yet. Whatever mapping does arise is not informative,
                        // and just makes the diff noisy.
                        client_info.id = 0;

                        let handlers = handler_summary(bt);
                        (client_info.short_name.clone(), client_info, handlers)
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .unwrap();
    blocks.sort_by(|a, b| a.0.cmp(&b.0));
    for (short_name, client_info, handlers) in blocks {
        println!("=== {} ===", short_name);
        println!("client_info: {:?}", client_info);
        println!("handlers: {}", handlers);
        println!();
    }
}

fn handler_summary(bt: &perovskite_server::game_state::blocks::BlockType) -> String {
    let mut parts = Vec::new();
    if bt.dig_handler_inline.is_some() {
        parts.push("dig_inline");
    }
    if bt.dig_handler_full.is_some() {
        parts.push("dig_full");
    }
    if bt.tap_handler_inline.is_some() {
        parts.push("tap_inline");
    }
    if bt.tap_handler_full.is_some() {
        parts.push("tap_full");
    }
    if bt.step_on_handler_inline.is_some() {
        parts.push("step_on_inline");
    }
    if bt.step_on_handler_full.is_some() {
        parts.push("step_on_full");
    }
    if bt.interact_key_handler.is_some() {
        parts.push("interact_key");
    }
    if bt.make_client_extended_data.is_some() {
        parts.push("client_ext_data");
    }
    if bt.deserialize_extended_data_handler.is_some() {
        parts.push("deser_ext_data");
    }
    if bt.serialize_extended_data_handler.is_some() {
        parts.push("ser_ext_data");
    }
    if bt.fixup_handler_inline.is_some() {
        parts.push("fixup_inline");
    }
    if bt.fixup_handler_full.is_some() {
        parts.push("fixup_full");
    }
    if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join(" ")
    }
}
