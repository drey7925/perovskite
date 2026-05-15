// Dev server entry point for rapid iteration and record/replay based testing.
//
// Brings up the default game with a flat dirt map and logs player actions to a
// recording file. Pass --output <filename> to set the recording path; defaults
// to a timestamped filename in the current working directory.

use std::{
    io::{BufWriter, Write},
    sync::Arc,
};

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::Local;
use parking_lot::Mutex;
use perovskite_core::protocol::game_rpc::{
    self as proto, dig_tap_action::ActionTarget as DigTapTarget,
    interact_key_action::InteractionTarget, place_action::PlaceAnchor,
    stream_to_server::ClientMessage,
};
use perovskite_game_api::{
    default_game::basic_blocks::DIRT, game_builder::GameBuilder, test_support::GameBuilderTestExt,
};
use perovskite_server::game_state::{
    chat::commands::ChatCommandHandler, event::HandlerContext, player::Player, GameState,
    GameStreamInterceptors,
};
use tracing::metadata::LevelFilter;
use tracing_subscriber::prelude::*;

/// Rounded step for spatial position logging (in blocks).
const POSITION_ROUND_UNIT: f64 = 0.25;
/// Rounded step for angular logging (in degrees).
const ANGLE_ROUND_DEGREES: f64 = 15.0;

fn round_to(value: f64, unit: f64) -> f64 {
    (value / unit).round() * unit
}

fn format_player_pos(pos: &Option<proto::PlayerPosition>) -> String {
    match pos {
        None => "none".to_string(),
        Some(p) => {
            let xyz = p.position.as_ref();
            let dir = p.face_direction.as_ref();
            let x = xyz
                .map(|v| round_to(v.x, POSITION_ROUND_UNIT))
                .unwrap_or(0.0);
            let y = xyz
                .map(|v| round_to(v.y, POSITION_ROUND_UNIT))
                .unwrap_or(0.0);
            let z = xyz
                .map(|v| round_to(v.z, POSITION_ROUND_UNIT))
                .unwrap_or(0.0);
            let az = dir
                .map(|a| round_to(a.deg_azimuth, ANGLE_ROUND_DEGREES))
                .unwrap_or(0.0);
            let el = dir
                .map(|a| round_to(a.deg_elevation, ANGLE_ROUND_DEGREES))
                .unwrap_or(0.0);
            format!("({x:.2},{y:.2},{z:.2}) az={az:.0} el={el:.0}")
        }
    }
}

fn format_dig_target(target: &Option<DigTapTarget>) -> String {
    match target {
        None => "none".to_string(),
        Some(DigTapTarget::BlockCoord(c)) => format!("block({},{},{})", c.x, c.y, c.z),
        Some(DigTapTarget::EntityTarget(e)) => format!("entity({})", e.entity_id),
    }
}

fn format_place_anchor(anchor: &Option<PlaceAnchor>) -> String {
    match anchor {
        None => "none".to_string(),
        Some(PlaceAnchor::AnchorBlock(c)) => format!("block({},{},{})", c.x, c.y, c.z),
        Some(PlaceAnchor::AnchorEntity(e)) => format!("entity({})", e.entity_id),
    }
}

fn format_interact_target(target: &Option<InteractionTarget>) -> String {
    match target {
        None => "none".to_string(),
        Some(InteractionTarget::BlockCoord(c)) => format!("block({},{},{})", c.x, c.y, c.z),
        Some(InteractionTarget::EntityTarget(e)) => format!("entity({})", e.entity_id),
    }
}

struct PlayerActionLogger {
    writer: Mutex<BufWriter<std::fs::File>>,
}

impl PlayerActionLogger {
    fn new(filename: &str) -> Result<Self> {
        let file = std::fs::File::create(filename).with_context(|| format!("create {filename}"))?;
        let abs_path =
            std::fs::canonicalize(filename).unwrap_or_else(|_| std::path::PathBuf::from(filename));
        tracing::info!("Recording player actions to {}", abs_path.display());
        let mut writer = BufWriter::new(file);
        let ts = Local::now().format("%Y-%m-%dT%H:%M:%S");
        writeln!(writer, "SESSION_START timestamp={ts}")?;
        writer.flush()?;
        Ok(Self {
            writer: Mutex::new(writer),
        })
    }
}

impl GameStreamInterceptors for PlayerActionLogger {
    fn handle_message(
        &self,
        message: &proto::StreamToServer,
        game_state: &GameState,
        player: &Player,
    ) -> Result<()> {
        let (action, item_slot, target, pos) = match &message.client_message {
            Some(ClientMessage::Dig(m)) => (
                "dig",
                m.item_slot,
                format_dig_target(&m.action_target),
                format_player_pos(&m.position),
            ),
            Some(ClientMessage::Tap(m)) => (
                "tap",
                m.item_slot,
                format_dig_target(&m.action_target),
                format_player_pos(&m.position),
            ),
            Some(ClientMessage::Place(m)) => {
                let coord = m.block_coord.as_ref();
                let block_coord = coord
                    .map(|c| format!("block({},{},{})", c.x, c.y, c.z))
                    .unwrap_or_else(|| "none".to_string());
                let anchor = format_place_anchor(&m.place_anchor);
                let target = format!("{} anchor={}", block_coord, anchor);
                ("place", m.item_slot, target, format_player_pos(&m.position))
            }
            Some(ClientMessage::InteractKey(m)) => (
                "interact",
                m.item_slot,
                format_interact_target(&m.interaction_target),
                format_player_pos(&m.position),
            ),
            _ => return Ok(()),
        };

        let inv = game_state
            .inventory_manager()
            .get(&player.main_inventory())?;
        let item_desc = match inv {
            None => "no_inventory".to_string(),
            Some(inv) => match inv.contents().get(item_slot as usize) {
                Some(Some(stack)) => {
                    format!("{}:{}", stack.item_name(), stack.quantity_or_wear())
                }
                Some(None) => "empty".to_string(),
                None => format!("slot_{item_slot}_oob"),
            },
        };

        let ts = Local::now().format("%H:%M:%S");
        let line = format!(
            "ts={ts} action={action} player={name} item_slot={item_slot} item={item_desc} target={target} player_pos={pos}\n",
            name = player.name(),
        );

        let mut w = self.writer.lock();
        w.write_all(line.as_bytes())?;
        w.flush()?;
        Ok(())
    }
}

struct ShutdownCommand;

#[async_trait]
impl ChatCommandHandler for ShutdownCommand {
    async fn handle(&self, _message: &str, context: &HandlerContext<'_>) -> Result<()> {
        context.start_shutdown();
        Ok(())
    }
}

fn parse_output_filename() -> String {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--output" && i + 1 < args.len() {
            return args[i + 1].clone();
        }
        i += 1;
    }
    format!("recording_{}.txt", Local::now().format("%Y%m%d_%H%M%S"))
}

fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer().with_filter(
                tracing_subscriber::EnvFilter::builder()
                    .with_default_directive(LevelFilter::INFO.into())
                    .from_env_lossy(),
            ),
        )
        .init();

    let output_filename = parse_output_filename();

    let (mut game, temp_dir) = GameBuilder::testonly_in_memory(Some(28275))?;
    perovskite_game_api::configure_default_game(&mut game)?;

    // Override with a simple flat dirt world instead of the complex default mapgen.
    let dirt = game
        .get_block(DIRT)
        .expect("dirt block not registered after configure_default_game");
    game.set_flatland_mapgen(dirt);

    // ==================== SCENE SETUP ====================
    // Modify this section to configure initial world state for a specific demo session.
    // Changes here are ephemeral: the world uses an in-memory database and the temp
    // directory is cleaned up when the server exits.

    // Sample: override the spawn location (add `use cgmath::vec3;` at the top)
    // game.server_builder_mut().game_behaviors_mut().spawn_location =
    //     Box::new(|_player_name| vec3(0.0, 5.0, 0.0));

    // Sample: give items to a player on first connect.
    // See the game_input_demo skill for the full GiveItemsOnJoin pattern.

    // ==================== END SCENE SETUP ====================

    game.add_command(
        "stop",
        Box::new(ShutdownCommand),
        "Gracefully shut down the dev server.",
    )?;

    let logger = Arc::new(PlayerActionLogger::new(&output_filename)?);
    let logger_for_end = logger.clone();
    game.set_stream_interceptors(logger);

    game.run_game_server()?;

    {
        let ts = Local::now().format("%Y-%m-%dT%H:%M:%S");
        let mut w = logger_for_end.writer.lock();
        writeln!(w, "SESSION_END timestamp={ts}")?;
        w.flush()?;
    }

    tracing::info!("Dev server has shut down; cleaning up temp dir");
    if let Err(e) = std::fs::remove_dir_all(&temp_dir) {
        tracing::warn!("Failed to remove temp dir {:?}: {e}", temp_dir);
    }

    Ok(())
}
