// Dev server entry point for rapid iteration and record/replay based testing.
//
// Brings up the default game with a flat dirt map and logs player actions to a
// timestamped recording file in the current working directory.

use std::{
    io::{BufWriter, Write},
    sync::Arc,
};

use anyhow::{Context, Result};
use chrono::Local;
use parking_lot::Mutex;
use perovskite_core::protocol::game_rpc::{
    self as proto,
    dig_tap_action::ActionTarget as DigTapTarget,
    interact_key_action::InteractionTarget,
    stream_to_server::ClientMessage,
};
use perovskite_game_api::{
    default_game::basic_blocks::DIRT,
    game_builder::GameBuilder,
    test_support::GameBuilderTestExt,
};
use perovskite_server::game_state::{player::Player, GameState, GameStreamInterceptors};
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
            let x = xyz.map(|v| round_to(v.x, POSITION_ROUND_UNIT)).unwrap_or(0.0);
            let y = xyz.map(|v| round_to(v.y, POSITION_ROUND_UNIT)).unwrap_or(0.0);
            let z = xyz.map(|v| round_to(v.z, POSITION_ROUND_UNIT)).unwrap_or(0.0);
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
    fn new() -> Result<Self> {
        let filename = format!(
            "recording_{}.txt",
            Local::now().format("%Y%m%d_%H%M%S")
        );
        let file =
            std::fs::File::create(&filename).with_context(|| format!("create {filename}"))?;
        tracing::info!("Recording player actions to {filename}");
        Ok(Self {
            writer: Mutex::new(BufWriter::new(file)),
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
                let target = coord
                    .map(|c| format!("block({},{},{})", c.x, c.y, c.z))
                    .unwrap_or_else(|| "none".to_string());
                (
                    "place",
                    m.item_slot,
                    target,
                    format_player_pos(&m.position),
                )
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

        let line = format!(
            "action={action} player={name} item_slot={item_slot} item={item_desc} target={target} player_pos={pos}\n",
            name = player.name(),
        );

        self.writer.lock().write_all(line.as_bytes())?;
        Ok(())
    }
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

    let (mut game, temp_dir) = GameBuilder::testonly_in_memory()?;
    perovskite_game_api::configure_default_game(&mut game)?;

    // Override with a simple flat dirt world instead of the complex default mapgen.
    let dirt = game
        .get_block(DIRT)
        .expect("dirt block not registered after configure_default_game");
    game.set_flatland_mapgen(dirt);

    let logger = Arc::new(PlayerActionLogger::new()?);
    game.set_stream_interceptors(logger);

    game.run_game_server()?;

    tracing::info!("Dev server has shut down; cleaning up temp dir");
    if let Err(e) = std::fs::remove_dir_all(&temp_dir) {
        tracing::warn!("Failed to remove temp dir {:?}: {e}", temp_dir);
    }

    Ok(())
}
