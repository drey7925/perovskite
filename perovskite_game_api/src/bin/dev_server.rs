// Dev server entry point for rapid iteration and record/replay based testing.
//
// Brings up the default game with a flat dirt map and logs player actions to a
// recording file. Pass --output <filename> to set the recording path; defaults
// to a timestamped filename in the current working directory.

use std::{
    io::{BufWriter, Write},
    sync::Arc,
};

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use chrono::Local;
use parking_lot::Mutex;
use perovskite_core::{
    chat::ChatMessage,
    protocol::game_rpc::{
        self as proto, dig_tap_action::ActionTarget as DigTapTarget,
        interact_key_action::InteractionTarget, place_action::PlaceAnchor,
        stream_to_server::ClientMessage,
    },
};
use perovskite_game_api::{
    autobuild::{self, Autobuilder},
    default_game::basic_blocks::{DIRT, STONE},
    game_builder::{GameBuilder, OwnedTextureName},
    test_support::GameBuilderTestExt,
    BlockCoordinate,
};
use perovskite_server::game_state::{
    chat::commands::ChatCommandHandler,
    client_ui::UiElementContainer,
    event::{EventInitiator, HandlerContext},
    game_map::templates::{self, SerializedTemplate},
    items::Item,
    player::Player,
    GameState, GameStreamInterceptors,
};
use prost::Message;
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

struct PlaceTemplateCommand;

#[async_trait]
impl ChatCommandHandler for PlaceTemplateCommand {
    async fn handle(&self, message: &str, context: &HandlerContext<'_>) -> Result<()> {
        let params = message.split_whitespace().collect::<Vec<_>>();
        let (path, origin) = match params.len() {
            2 => (params[1], BlockCoordinate::new(0, 0, 0)),
            5 => {
                let x: i32 = params[2].parse().context("Invalid x coordinate")?;
                let y: i32 = params[3].parse().context("Invalid y coordinate")?;
                let z: i32 = params[4].parse().context("Invalid z coordinate")?;
                (params[1], BlockCoordinate::new(x, y, z))
            }
            _ => bail!("Usage: /place_template <path> [x y z]"),
        };

        let bytes =
            std::fs::read(path).with_context(|| format!("Failed to read template file {path}"))?;
        let serialized = SerializedTemplate::decode(bytes.as_slice())
            .with_context(|| format!("Failed to decode template from {path}"))?;
        let in_mem = serialized.to_in_mem(context.block_types())?;
        context
            .game_map()
            .apply_template(&in_mem, origin, 0, &EventInitiator::Engine)?;

        let (sx, sy, sz) = in_mem.size();
        context
            .initiator()
            .send_chat_message_async(ChatMessage::new_server_message(format!(
                "Placed template {} at ({}, {}, {}); size: {}x{}x{} (dx dy dz)",
                path, origin.x, origin.y, origin.z, sx, sy, sz
            )))
            .await?;
        Ok(())
    }
}

struct SaveTemplateCommand;

#[async_trait]
impl ChatCommandHandler for SaveTemplateCommand {
    async fn handle(&self, message: &str, context: &HandlerContext<'_>) -> Result<()> {
        let params = message.split_whitespace().collect::<Vec<_>>();
        if params.len() != 8 {
            bail!("Usage: /save_template <filename> <x0> <y0> <z0> <dx> <dy> <dz>");
        }
        let path = params[1];
        let x0: i32 = params[2].parse().context("Invalid x0")?;
        let y0: i32 = params[3].parse().context("Invalid y0")?;
        let z0: i32 = params[4].parse().context("Invalid z0")?;
        let dx: i32 = params[5].parse().context("Invalid dx")?;
        let dy: i32 = params[6].parse().context("Invalid dy")?;
        let dz: i32 = params[7].parse().context("Invalid dz")?;
        if dx <= 0 || dy <= 0 || dz <= 0 {
            bail!("dx, dy, dz must be positive");
        }

        let mut template = templates::InMemTemplate::new_empty(dx, dy, dz);
        for x in x0..x0 + dx {
            for y in y0..y0 + dy {
                for z in z0..z0 + dz {
                    let (block, ext) = context
                        .game_map()
                        .get_block_with_extended_data(BlockCoordinate::new(x, y, z), |_, e| {
                            Ok(Some(e.clone()))
                        })?;
                    template.set_block_at(x - x0, y - y0, z - z0, block, ext);
                }
            }
        }

        let serialized = SerializedTemplate::from_in_mem(&template, context.block_types())?;
        let bytes = serialized.encode_to_vec();
        let len = bytes.len();
        std::fs::write(path, bytes).with_context(|| format!("Failed to write {path}"))?;

        context
            .initiator()
            .send_chat_message_async(ChatMessage::new_server_message(format!(
                "Saved template to {path} ({dx}x{dy}x{dz}, {len} bytes)"
            )))
            .await?;
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

    let texture = OwnedTextureName::from_css_color("#00ffff");
    let mut save_template_tool = Item {
        ..Item::default_with_proto(perovskite_core::protocol::items::ItemDef {
            short_name: "devserver:save_template".to_string(),
            display_name: "Save template tool".to_string(),
            appearance: texture.into(),
            groups: vec![],
            interaction_rules: vec![],
            quantity_type: None,
            sort_key: "devserver:save_template".to_string(),
            tool_range: 50.0,
        })
    };
    autobuild::configure_item::<RegionSaver>(&mut save_template_tool);
    game.unstable_server_builder_mut()
        .items_mut()
        .register_item(save_template_tool)?;

    // ==================== CONTENT SETUP ====================
    // Modify this section to configure initial world state for a specific demo session.
    // Changes here are ephemeral: the world uses an in-memory database and the temp
    // directory is cleaned up when the server exits.

    // Sample: override the spawn location (add `use cgmath::vec3;` at the top)
    // game.server_builder_mut().game_behaviors_mut().spawn_location =
    //     Box::new(|_player_name| vec3(0.0, 5.0, 0.0));

    // Sample: give items to a player on first connect.
    // See the game_input_demo skill for the full GiveItemsOnJoin pattern.

    // ==================== END CONTENT SETUP ====================

    game.add_command(
        "stop",
        Box::new(ShutdownCommand),
        "Gracefully shut down the dev server.",
    )?;
    game.add_command(
        "place_template",
        Box::new(PlaceTemplateCommand),
        "<path> [x y z]: Loads a saved template file and places it at the given coordinates (default 0,0,0).",
    )?;
    game.add_command(
        "save_template",
        Box::new(SaveTemplateCommand),
        "<filename> <x0> <y0> <z0> <dx> <dy> <dz>: Saves a region of the world to a template file.",
    )?;

    let logger = Arc::new(PlayerActionLogger::new(&output_filename)?);
    let logger_for_end = logger.clone();
    game.set_stream_interceptors(logger);

    let server = game.into_server()?;

    server.run_task_in_server(|gs| -> Result<()> {
        // ==================== SCENE SETUP ====================
        // Function calls that require a running server can be done here.
        // For example, let's mark (0,0,0) as stone so the player can find the origin easily.
        gs.game_map()
            .set_block(BlockCoordinate::new(0, -1, 0), STONE, None)?;

        Ok(())
    })?;
    // ==================== END SCENE SETUP ====================

    server.serve()?;

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

#[derive(Default, Clone, Copy, Debug)]
struct RegionSaverState {
    first_corner: Option<BlockCoordinate>,
}

struct RegionSaver;
impl Autobuilder for RegionSaver {
    type Settings = ();

    type SelectionState = RegionSaverState;

    fn make_settings_popup(
        _ctx: &HandlerContext,
        _coord: perovskite_server::game_state::items::PointeeBlockCoords,
        _settings: &Self::Settings,
    ) -> Option<perovskite_server::game_state::client_ui::Popup> {
        None
    }

    fn make_advice_popup(
        ctx: &HandlerContext,
        _state: &Self::SelectionState,
    ) -> perovskite_server::game_state::client_ui::Popup {
        ctx.new_popup().label("Use left-click to select the first corner of the region, right-click to select the second corner. Saves to the current working directory of the dev server.")
    }

    fn should_show_advice(state: &Self::SelectionState) -> bool {
        state.first_corner.is_none()
    }

    fn tap(
        _ctx: &HandlerContext,
        coord: perovskite_server::game_state::items::PointeeBlockCoords,
        _settings: &mut Self::Settings,
        state: &mut Self::SelectionState,
    ) -> Result<()> {
        state.first_corner = Some(coord.selected);
        Ok(())
    }

    fn build(
        ctx: &HandlerContext,
        pointee_coord: perovskite_server::game_state::items::PointeeBlockCoords,
        _settings: &mut Self::Settings,
        state: &mut Self::SelectionState,
    ) -> Result<Option<perovskite_game_api::autobuild::BatchedUndo>> {
        let first = state.first_corner.unwrap();
        let second = pointee_coord.selected;
        let x_min = first.x.min(second.x);
        let x_max = first.x.max(second.x);
        let y_min = first.y.min(second.y);
        let y_max = first.y.max(second.y);
        let z_min = first.z.min(second.z);
        let z_max = first.z.max(second.z);

        let sx = (second.x - first.x).abs() + 1;
        let sy = (second.y - first.y).abs() + 1;
        let sz = (second.z - first.z).abs() + 1;

        let mut template = templates::InMemTemplate::new_empty(sx, sy, sz);

        for x in x_min..=x_max {
            for z in z_min..=z_max {
                for y in y_min..=y_max {
                    let (block, ext) = ctx
                        .game_map()
                        .get_block_with_extended_data(BlockCoordinate::new(x, y, z), |_, e| {
                            Ok(Some(e.clone()))
                        })?;
                    template.set_block_at(x - x_min, y - y_min, z - z_min, block, ext);
                }
            }
        }

        let serialized_template = SerializedTemplate::from_in_mem(&template, ctx.block_types())?;
        let bytes = serialized_template.encode_to_vec();
        let filename = format!(
            "template_{}_mincorner_x{}_{}_y{}_{}_z{}_{}.pvtpl",
            Local::now().format("%Y%m%d_%H%M%S"),
            x_min,
            x_max,
            y_min,
            y_max,
            z_min,
            z_max
        );
        let len = bytes.len();

        let path = std::env::current_dir()?.join(filename);
        std::fs::write(&path, bytes)?;
        let message = format!("Saved template to {}, {} bytes", path.display(), len);
        ctx.initiator()
            .send_chat_message(ChatMessage::new_server_message(message))?;

        Ok(None)
    }

    const TOOL_ID: &'static str = "devserver:save_template";
}
