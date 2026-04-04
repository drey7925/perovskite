//! Tools for players to quickly build structures. Initial focus is on roads, to prove
//! out the UX and get some initial experimentation.

use std::{any::Any, iter};

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use itertools::Itertools;
use ndarray::s;
use perovskite_core::{
    block_id::{special_block_defs::AIR_ID, BlockId},
    chat::{ChatMessage, SERVER_ERROR_COLOR},
    constants::{block_groups::DEFAULT_SOLID, permissions::WORLD_STATE},
    coordinates::{BlockCoordinate, ChunkCoordinate, ChunkOffset},
    protocol::items::{self as items_proto, interaction_rule::DigBehavior, InteractionRule},
};
use perovskite_server::game_state::{
    blocks::{ExtendedData, FastBlockGroup},
    chat::commands::ChatCommandHandler,
    client_ui::{Popup, TextFieldBuilder, UiElementContainer},
    event::{EventInitiator, HandlerContext},
    items::{Item, ItemInteractionResult, ItemStack, PointeeBlockCoords},
    player::{self, Player},
};
use smallvec::SmallVec;

use crate::{
    blocks,
    default_game::block_groups::{NATURAL_AND_STRUCTURAL, NATURAL_GROUND, VARIANT_ENCODES_PLACER},
    game_builder::{GameBuilder, OwnedTextureName},
};

pub fn initialize_autobuild(game: &mut GameBuilder) -> anyhow::Result<()> {
    game.inner.register_command(
        "undo",
        Box::new(UndoAutobuildCommand),
        "Undo the last autobuild operation",
    )?;

    game.inner
        .blocks_mut()
        .register_fast_block_group(NATURAL_GROUND);
    game.inner
        .blocks_mut()
        .register_fast_block_group(NATURAL_AND_STRUCTURAL);
    game.inner
        .blocks_mut()
        .register_fast_block_group(VARIANT_ENCODES_PLACER);

    let texture = OwnedTextureName::from_css_color("#808080");
    let mut road_tool = Item {
        ..Item::default_with_proto(items_proto::ItemDef {
            short_name: "autobuild:road_tool".to_string(),
            display_name: "Road tool".to_string(),
            appearance: texture.into(),
            groups: vec![],
            interaction_rules: vec![],
            quantity_type: None,
            sort_key: "autobuild:road_tool".to_string(),
        })
    };
    configure_item::<RoadTool>(&mut road_tool);
    game.inner.items_mut().register_item(road_tool)?;

    Ok(())
}

fn configure_item<T: Autobuilder>(item: &mut Item) {
    item.place_on_block_handler = Some(Box::new(|ctx, coord, stack| {
        place_on_block_interaction::<T>(ctx, coord, stack)
    }));
    item.dig_handler = Some(Box::new(|ctx, coord, stack| {
        dig_interaction::<T>(ctx, coord, stack)
    }));
    item.tap_handler = Some(Box::new(|ctx, coord, stack| {
        tap_interaction::<T>(ctx, coord, stack)
    }));
    item.proto.interaction_rules = vec![InteractionRule {
        block_group: vec![DEFAULT_SOLID.to_string()],
        tool_wear: 1,
        dig_behavior: Some(DigBehavior::ConstantTime(0.3)),
    }];
}

struct UndoAutobuildCommand;
#[async_trait]
impl ChatCommandHandler for UndoAutobuildCommand {
    async fn handle(&self, _message: &str, context: &HandlerContext<'_>) -> Result<()> {
        if !context.initiator().check_permission_if_player(WORLD_STATE) {
            context.initiator().send_chat_message(
                ChatMessage::new_server_message("You don't have permission to use this command")
                    .with_color(SERVER_ERROR_COLOR),
            )?;
            return Ok(());
        }
        match context.initiator() {
            EventInitiator::Player(p) => {
                let undo = p.player.with_transient_data::<BatchedUndo, _>(|x| std::mem::take(x));
                undo.undo(context)?;
                Ok(())
            }
            EventInitiator::WeakPlayerRef(weak) => {
                weak.try_to_run(|p| {
                    let undo = p.with_transient_data::<BatchedUndo, _>(|x| std::mem::take(x));
                    undo.undo(context)?;
                    Ok(())
                })
                .ok_or_else(|| anyhow::anyhow!("Player not found"))?
            }
            _ => bail!("Autobuild undo command can only be used interactively by players; call methods on the undo object directly to undo (or better yet, file a feature request for proper database transactions; undo is meant as a player aid"),
        }
    }
}

#[derive(Default)]
struct BatchedWrite {
    blocks: Vec<(BlockCoordinate, BlockId)>,
}

#[derive(Default)]
struct BatchedUndo {
    blocks: Vec<(
        ChunkCoordinate,
        Vec<(ChunkOffset, BlockId, Option<ExtendedData>)>,
    )>,
}

impl BatchedWrite {
    /// Writes the blocks to the game map, and returns an undo object.
    ///
    /// This is more efficient than writing blocks one by one, because it allows
    /// the game map to batch the writes.
    fn write(mut self, ctx: &HandlerContext) -> Result<BatchedUndo> {
        let mut undo = BatchedUndo::default();
        let start = std::time::Instant::now();
        let block_count = self.blocks.len();
        self.blocks
            .sort_unstable_by_key(|(c, _)| (c.chunk().x, c.chunk().z, c.chunk().y));
        let sort_time = start.elapsed();
        let write_start = std::time::Instant::now();
        for (key, group) in self
            .blocks
            .into_iter()
            .chunk_by(|(c, _)| c.chunk())
            .into_iter()
        {
            let undos = ctx.game_map().bulk_write_chunk(key, |chunk| {
                let mut undos = Vec::new();
                for (coord, block_id) in group {
                    let (old_block, old_ext_data) = chunk.set_block(coord.offset(), block_id, None);
                    undos.push((coord.offset(), old_block, old_ext_data));
                }
                Ok(undos)
            })?;
            undo.blocks.push((key, undos));
        }
        let write_time = write_start.elapsed();
        let total_time = start.elapsed();
        tracing::info!(
            "BatchedWrite::write: sort_time: {:?} micros, write_time: {:?} micros, total_time: {:?} micros ({:.2} nanoseconds per block)",
            sort_time.as_micros(),
            write_time.as_micros(),
            total_time.as_micros(),
            total_time.as_nanos() as f64 / block_count as f64
        );

        Ok(undo)
    }
}

impl BatchedUndo {
    fn undo(self, ctx: &HandlerContext) -> Result<()> {
        for (key, undos) in self.blocks {
            ctx.game_map().bulk_write_chunk(key, |chunk| {
                for (coord, block_id, ext_data) in undos {
                    chunk.set_block(coord, block_id, ext_data);
                }
                Ok(())
            })?;
        }
        Ok(())
    }
}

trait Autobuilder {
    type Settings: player::PersistentData + prost::Message + Default + Clone;
    type SelectionState: Any + Send + Sync + Default + Clone + 'static;
    /// Creates a popup for configuring the tool.
    fn make_settings_popup(ctx: &HandlerContext, settings: &Self::Settings) -> Popup;
    /// Saves the settings to the player's persistent data. This is provided as a convenience,
    /// to be used in the callback of the settings popup.
    fn save_settings(ctx: &HandlerContext, settings: Self::Settings) -> Result<()> {
        match ctx.initiator() {
            EventInitiator::Player(p) => p.player.with_persistent_data(Self::TOOL_ID, |x| {
                *x = settings.clone();
                Ok(())
            }),
            EventInitiator::WeakPlayerRef(weak) => weak
                .try_to_run(|p| {
                    p.with_persistent_data(Self::TOOL_ID, |x| {
                        *x = settings.clone();
                        Ok(())
                    })
                })
                .ok_or_else(|| anyhow::anyhow!("Player not found"))?,
            _ => bail!("Autobuild can only be used interactively by players"),
        }
    }
    /// Creates a popup for showing advice to the user, e.v. when the state is incomplete or
    /// the user's usage must be corrected.
    fn make_advice_popup(ctx: &HandlerContext, state: &Self::SelectionState) -> Popup;
    /// Returns true if the advice popup should be shown.
    fn should_show_advice(state: &Self::SelectionState) -> bool;

    /// Called when the player taps the tool on the ground.
    fn tap(
        ctx: &HandlerContext,
        coord: PointeeBlockCoords,
        state: &mut Self::SelectionState,
    ) -> Result<()>;

    /// Actually builds. Invoked when the player right-clicks. Returns an undo object that
    /// can be used to undo the build.
    fn build(
        ctx: &HandlerContext,
        right_click_coord: BlockCoordinate,
        settings: Self::Settings,
        state: &mut Self::SelectionState,
    ) -> Result<BatchedUndo>;
    /// A unique identifier for this tool, used to store settings in persistent data.
    /// This should be unique across all autobuilders, and also include `autobuild:` to
    /// avoid collisions with other plugins.
    const TOOL_ID: &'static str;
}

fn place_on_block_interaction<T: Autobuilder>(
    ctx: &HandlerContext,
    coord: PointeeBlockCoords,
    stack: &ItemStack,
) -> Result<ItemInteractionResult> {
    let work = |p: &Player| -> Result<()> {
        let mut data = p.with_transient_data::<T::SelectionState, _>(|x| x.clone());
        let settings = p.with_persistent_data::<T::Settings, _>(T::TOOL_ID, |x| Ok(x.clone()))?;
        if T::should_show_advice(&data) {
            p.show_popup_blocking(T::make_advice_popup(ctx, &data))
        } else {
            match T::build(ctx, coord.selected, settings, &mut data) {
                Ok(undo) => {
                    p.send_chat_message(ChatMessage::new_server_message(
                        "Build successful! Use /undo to undo (only one undo at a time)",
                    ))?;
                    p.with_transient_data::<BatchedUndo, _>(|x| *x = undo);
                }
                Err(e) => {
                    p.send_chat_message(
                        ChatMessage::new_server_message(e.to_string())
                            .with_color(SERVER_ERROR_COLOR),
                    )?;
                }
            }
            p.with_transient_data::<T::SelectionState, _>(|x| *x = data);
            Ok(())
        }
    };

    match ctx.initiator() {
        perovskite_server::game_state::event::EventInitiator::Player(p) => {
            work(p.player)?;
        }
        perovskite_server::game_state::event::EventInitiator::WeakPlayerRef(weak) => {
            match weak.try_to_run(|p| work(p)) {
                None => {}
                Some(Ok(_)) => {}
                Some(Err(e)) => return Err(e),
            }
        }
        _ => {
            bail!("Autobuild can only be used interactively by players; consider calling build functions directly");
        }
    };
    // give the user back their tool
    Ok(Some(stack.clone()).into())
}

fn dig_interaction<T: Autobuilder>(
    ctx: &HandlerContext,
    _coord: PointeeBlockCoords,
    stack: &ItemStack,
) -> Result<ItemInteractionResult> {
    let work = |p: &Player| -> Result<()> {
        let settings = p.with_persistent_data::<T::Settings, _>(T::TOOL_ID, |x| Ok(x.clone()))?;
        p.show_popup_blocking(T::make_settings_popup(ctx, &settings))
    };

    match ctx.initiator() {
        perovskite_server::game_state::event::EventInitiator::Player(p) => {
            work(p.player)?;
        }
        perovskite_server::game_state::event::EventInitiator::WeakPlayerRef(weak) => {
            match weak.try_to_run(|p| work(p)) {
                None => {}
                Some(Ok(_)) => {}
                Some(Err(e)) => return Err(e),
            }
        }
        _ => {
            bail!("Autobuild can only be used interactively by players; consider calling build functions directly");
        }
    };
    // give the user back their tool
    Ok(Some(stack.clone()).into())
}

fn tap_interaction<T: Autobuilder>(
    ctx: &HandlerContext,
    coord: PointeeBlockCoords,
    stack: &ItemStack,
) -> Result<ItemInteractionResult> {
    let work = |p: &Player| -> Result<()> {
        let mut state = p.with_transient_data::<T::SelectionState, _>(|x| x.clone());
        T::tap(ctx, coord, &mut state)?;
        p.with_transient_data::<T::SelectionState, _>(|x| *x = state);
        Ok(())
    };

    match ctx.initiator() {
        perovskite_server::game_state::event::EventInitiator::Player(p) => {
            work(p.player)?;
        }
        perovskite_server::game_state::event::EventInitiator::WeakPlayerRef(weak) => {
            match weak.try_to_run(|p| work(p)) {
                None => {}
                Some(Ok(_)) => {}
                Some(Err(e)) => return Err(e),
            }
        }
        _ => {
            bail!("Autobuild can only be used interactively by players; consider calling build functions directly");
        }
    };
    // give the user back their tool
    Ok(Some(stack.clone()).into())
}

#[derive(Clone, Debug, Default)]
struct RoadToolSelectionState {
    start: Option<BlockCoordinate>,
    pending_end: Option<BlockCoordinate>,
}

#[derive(Clone, prost::Message)]
struct RoadToolSettings {
    #[prost(uint32, tag = "1")]
    block: u32,
    #[prost(uint32, tag = "2", default = "4")]
    width: u32,
    #[prost(uint32, tag = "3", default = "4")]
    tunnel_height: u32,
    #[prost(uint32, tag = "4")]
    slab_block: u32,
    #[prost(uint32, tag = "5", default = "0")]
    edge_block: u32,
    #[prost(uint32, tag = "6", default = "0")]
    edge_slab_block: u32,
    #[prost(bool, tag = "7", default = "false")]
    raised_edges: bool,
}

struct RoadTool;
impl Autobuilder for RoadTool {
    type Settings = RoadToolSettings;
    type SelectionState = RoadToolSelectionState;
    fn make_settings_popup(ctx: &HandlerContext, settings: &Self::Settings) -> Popup {
        ctx.new_popup()
            .title("Road settings")
            .text_field(
                TextFieldBuilder::new("block_id").label("Block").initial(
                    ctx.block_types()
                        .get_block(BlockId(settings.block))
                        .map_or("???", |(b, _)| b.short_name()),
                ),
            )
            .text_field(
                TextFieldBuilder::new("slab_block")
                    .label("Slab Block")
                    .initial(
                        ctx.block_types()
                            .get_block(BlockId(settings.slab_block))
                            .map_or("???", |(b, _)| b.short_name()),
                    ),
            )
            .text_field(
                TextFieldBuilder::new("edge_block")
                    .label("Edge Block")
                    .initial(
                        ctx.block_types()
                            .get_block(BlockId(settings.edge_block))
                            .map_or("???", |(b, _)| b.short_name()),
                    ),
            )
            .text_field(
                TextFieldBuilder::new("edge_slab_block")
                    .label("Edge Slab Block")
                    .initial(
                        ctx.block_types()
                            .get_block(BlockId(settings.edge_slab_block))
                            .map_or("???", |(b, _)| b.short_name()),
                    ),
            )
            .text_field(
                TextFieldBuilder::new("width")
                    .label("Width")
                    .initial(settings.width.to_string()),
            )
            .text_field(
                TextFieldBuilder::new("tunnel_height")
                    .label("Tunnel Height")
                    .initial(settings.tunnel_height.to_string()),
            )
            .checkbox("raised_edges", "Raised Edges", settings.raised_edges, true)
            .button("save", "Save", true, true)
            .set_button_callback(|resp| {
                let block = resp
                    .ctx
                    .block_types()
                    .get_by_name(
                        &resp
                            .textfield_values
                            .get("block_id")
                            .context("Missing block_id")?,
                    )
                    .context("Block not found")?;
                let width = resp
                    .textfield_values
                    .get("width")
                    .context("Missing width")?
                    .parse::<u32>()?;
                let slab_block = resp
                    .ctx
                    .block_types()
                    .get_by_name(
                        &resp
                            .textfield_values
                            .get("slab_block")
                            .context("Missing slab_block")?,
                    )
                    .context("Slab block not found")?;
                let edge_block = resp
                    .ctx
                    .block_types()
                    .get_by_name(
                        &resp
                            .textfield_values
                            .get("edge_block")
                            .context("Missing edge_block")?,
                    )
                    .context("Edge block not found")?;
                let edge_slab_block = resp
                    .ctx
                    .block_types()
                    .get_by_name(
                        &resp
                            .textfield_values
                            .get("edge_slab_block")
                            .context("Missing edge_slab_block")?,
                    )
                    .context("Edge slab block not found")?;
                let tunnel_height = resp
                    .textfield_values
                    .get("tunnel_height")
                    .context("Missing tunnel_height")?
                    .parse::<u32>()?;
                let raised_edges = *resp
                    .checkbox_values
                    .get("raised_edges")
                    .context("Missing raised_edges")?;
                let settings = RoadToolSettings {
                    block: block.0,
                    width,
                    slab_block: slab_block.0,
                    tunnel_height,
                    edge_block: edge_block.0,
                    edge_slab_block: edge_slab_block.0,
                    raised_edges,
                };
                Self::save_settings(&resp.ctx, settings)
            })
    }
    fn make_advice_popup(ctx: &HandlerContext, _state: &Self::SelectionState) -> Popup {
        ctx.new_popup()
        .title("Road tool")
        .label("Tap the tool on the ground to set the starting point. Then, right-click to finish building.")
        .label("Hold the dig/tap button to change road settings.")
        .label("If the width is even, click slightly to the left of your desired centerline, in the direction of the road.")
        .button("acknowledge", "Got it!", true, true)
    }
    fn should_show_advice(state: &Self::SelectionState) -> bool {
        state.start.is_none()
    }

    fn tap(
        ctx: &HandlerContext,
        coord: PointeeBlockCoords,
        state: &mut Self::SelectionState,
    ) -> Result<()> {
        ctx.initiator()
            .send_chat_message(ChatMessage::new_server_message("Start set"))?;
        state.start = Some(coord.selected);
        Ok(())
    }

    fn build(
        ctx: &HandlerContext,
        right_click_coord: BlockCoordinate,
        settings: Self::Settings,
        state: &mut Self::SelectionState,
    ) -> Result<BatchedUndo> {
        let start = state.start.context("No start point")?;
        let end = right_click_coord;
        let initiator = ctx.initiator();
        let mut block = BlockId(settings.block);
        let mut slab_block = BlockId(settings.slab_block);
        let mut edge_block = BlockId(settings.edge_block);
        let mut edge_slab_block = BlockId(settings.edge_slab_block);

        let natural_block_group = ctx
            .block_types()
            .fast_block_group(NATURAL_GROUND)
            .context("No natural ground block group found; internal bug")?;
        let natural_structural_group =
            ctx.block_types()
                .fast_block_group(NATURAL_AND_STRUCTURAL)
                .context("No natural structural block group found; internal bug")?;
        let encodes_placer_group = ctx
            .block_types()
            .fast_block_group(VARIANT_ENCODES_PLACER)
            .context("No encodes placer block group found; internal bug")?;
        if encodes_placer_group.contains(block) {
            block = block.with_variant_unchecked(blocks::variants::VARIANT_PLACED_AUTOBUILD as u16);
        }
        if encodes_placer_group.contains(slab_block) {
            slab_block = slab_block
                .with_variant_unchecked(blocks::variants::VARIANT_PLACED_AUTOBUILD as u16);
        }
        if edge_block == AIR_ID {
            edge_block = block;
        } else if encodes_placer_group.contains(edge_block) {
            edge_block = edge_block
                .with_variant_unchecked(blocks::variants::VARIANT_PLACED_AUTOBUILD as u16);
        }
        if edge_slab_block == AIR_ID {
            edge_slab_block = slab_block;
        } else if encodes_placer_group.contains(edge_slab_block) {
            edge_slab_block = edge_slab_block
                .with_variant_unchecked(blocks::variants::VARIANT_PLACED_AUTOBUILD as u16);
        }

        let abs_dx = (start.x - end.x).abs();
        let abs_dz = (start.z - end.z).abs();

        // Do everything assuming that X > Z, then fix up coordinates during map reads and writes
        let transposer = if abs_dx > abs_dz {
            |coord: BlockCoordinate| coord
        } else {
            |coord: BlockCoordinate| BlockCoordinate::new(coord.z, coord.y, coord.x)
        };
        let tuple_transposer = if abs_dx > abs_dz {
            |coord: (i32, i32)| coord
        } else {
            |coord: (i32, i32)| (coord.1, coord.0)
        };
        let bias_direction = if abs_dx > abs_dz { 1.0 } else { -1.0 };
        let start = transposer(start);
        let end = transposer(end);

        // A heuristic on how long this road is, used to determine how far up/down
        // we'll look for the ground contour
        let heuristic_len = (abs_dx + abs_dz) as i32;

        if heuristic_len > 1000 {
            bail!("Road too long");
        }
        if heuristic_len < 2 {
            bail!("Road too short");
        }
        if settings.width < 1 {
            bail!("Road width must be at least 1");
        }
        if settings.width > 7 {
            bail!("Road width must be at most 7");
        }

        let (start, end, z_bias) = if start.x > end.x {
            (end, start, 0.25 * bias_direction)
        } else {
            (start, end, -0.25 * bias_direction)
        };

        let bridge_mode = state.pending_end.take() == Some(end);
        if bridge_mode {
            initiator
                .send_chat_message(ChatMessage::new_server_message("Building bridge/tunnel"))?;
        } else {
            initiator.send_chat_message(ChatMessage::new_server_message(
                "Building road along ground",
            ))?;
        }

        let global_z_min = end.z.min(start.z) - (1 + settings.width as i32) / 2;
        let global_z_max = end.z.max(start.z) + (1 + settings.width as i32) / 2;

        let mut y_levels = ndarray::Array2::<Option<i32>>::from_elem(
            (
                (end.x - start.x + 1) as usize,
                (global_z_max - global_z_min + 1) as usize,
            ),
            None,
        );

        let y_min = start.y.min(end.y) - 10 - (heuristic_len / 4);
        let y_max = start.y.max(end.y) + 10 + (heuristic_len / 4);

        // Mostly moving along x axis
        for x in start.x..=end.x {
            for z in global_z_min..=global_z_max {
                // The physical coordinate to look at
                let (x_prime, z_prime) = tuple_transposer((x, z));
                let y_level = probe_ground_level(
                    ctx,
                    x_prime,
                    z_prime,
                    (y_min, y_max),
                    &natural_block_group,
                )?;
                y_levels[((x - start.x) as usize, (z - global_z_min) as usize)] = y_level;
            }
        }

        let mut y_levels_smoothed = y_levels.clone();
        let windows = y_levels.windows((3, 3));

        let targets = y_levels_smoothed.slice_mut(s![
            1..(end.x - start.x) as usize,
            1..global_z_max as usize - global_z_min as usize
        ]);

        ndarray::Zip::from(windows)
            .and(targets)
            .for_each(|window, target| {
                let mut valid_y_levels: SmallVec<[i32; 9]> =
                    window.iter().filter_map(|y| *y).collect();
                valid_y_levels.sort_unstable();
                let median = if valid_y_levels.is_empty() {
                    None
                } else {
                    // slight downward bias, but probably more aesthetic since we'll put slabs on top
                    Some(valid_y_levels[valid_y_levels.len() / 2])
                };
                *target = median;
            });

        // Compute a smoothed height
        let mut raw_path_heights: Vec<Option<i32>> =
            Vec::with_capacity((end.x - start.x + 1) as usize);
        for x in start.x..=end.x {
            let t = (x - start.x) as f64 / (end.x - start.x) as f64;
            let z_center = i2f_lerp(start.z, end.z, t);

            let z_min = (z_center - settings.width as f64 / 2.0 + z_bias).ceil() as i32;
            let z_max = (z_center + settings.width as f64 / 2.0 + z_bias).floor() as i32;

            let mut num = 0;
            let mut den = 0;
            for z in z_min..=z_max {
                let target_level =
                    y_levels_smoothed[((x - start.x) as usize, (z - global_z_min) as usize)];
                if let Some(y_tgt) = target_level {
                    num += y_tgt;
                    den += 1;
                }
            }
            if den == 0 {
                raw_path_heights.push(None);
            } else {
                raw_path_heights.push(Some(num / den));
            }
        }

        if raw_path_heights.is_empty() {
            bail!("Segment too short; nothing to do")
        }

        let start_height = raw_path_heights[0];
        let end_height = raw_path_heights[raw_path_heights.len() - 1];

        let start_height = match start_height {
            Some(h) => h,
            None => {
                bail!(
                    "Cannot estimate path height at the start; please try a different start position.",
                );
            }
        };
        let end_height = match end_height {
            Some(h) => h,
            None => {
                bail!(
                    "Cannot estimate path height at the end; please try a different end position.",
                );
            }
        };

        let path_heights = if bridge_mode {
            let mut path_heights = Vec::with_capacity(raw_path_heights.len());
            for i in 0..raw_path_heights.len() {
                path_heights.push(i2f_lerp(
                    start_height,
                    end_height,
                    i as f64 / raw_path_heights.len() as f64,
                ));
            }
            path_heights
        } else {
            if raw_path_heights.iter().any(|h| h.is_none()) {
                state.pending_end = Some(end);
                bail!(
                    "Cannot estimate path height at some point along the path, but we can build a bridge/tunnel instead. Tap again at the same end spot to confirm.",
                );
            }
            let raw_path_heights: Vec<i32> =
                raw_path_heights.into_iter().map(|h| h.unwrap()).collect();
            let mut path_heights = Vec::from_iter(iter::repeat_n(0.0, raw_path_heights.len()));
            let start_height = start_height as f64;
            let end_height = end_height as f64;
            path_heights[0] = start_height;
            path_heights[raw_path_heights.len() - 1] = end_height;

            const MAX_SLOPE: f64 = 1.0 / 3.0;
            if (raw_path_heights[0] - raw_path_heights[raw_path_heights.len() - 1]).abs() as f64
                > MAX_SLOPE * (raw_path_heights.len() as f64)
            {
                bail!("Road is too steep");
            }

            for i in 1..raw_path_heights.len() - 1 {
                let prev = path_heights[i - 1];
                let target_ground = raw_path_heights[i] as f64;

                let dist_to_right = (raw_path_heights.len() - i) as f64;

                let upper_bound =
                    (prev + 1.0 * MAX_SLOPE).min(dist_to_right * MAX_SLOPE + end_height);
                let lower_bound = (prev - 1.0 * MAX_SLOPE)
                    .max(dist_to_right * -MAX_SLOPE + end_height)
                    .min(upper_bound);

                if target_ground > (upper_bound + 3.0) || target_ground < (lower_bound - 3.0) {
                    state.pending_end = Some(end);
                    bail!("Path can't follow the terrain, but we can build a bridge/tunnel. Tap again at the same spot to confirm.")
                }

                let result = target_ground.clamp(lower_bound, upper_bound);

                path_heights[i] = result;
                // TODO: some kind of dynamic programming or optimization to make the path align with more
                // of the contour, instead of this asymmetric simplification that just follows bounds greedily
                // in one direction.
            }

            path_heights
        };

        // todo: detect player-placed blocks and don't overwrite them
        let mut writes = BatchedWrite::default();
        for x in start.x..=end.x {
            let t = (x - start.x) as f64 / (end.x - start.x) as f64;
            let z_center = i2f_lerp(start.z, end.z, t);

            let z_min = (z_center - settings.width as f64 / 2.0 + z_bias).ceil() as i32;
            let z_max = (z_center + settings.width as f64 / 2.0 + z_bias).floor() as i32;

            for z in z_min..=z_max {
                let (main, slab, edge_bias) = if (z_max - z_min >= 3) && (z == z_min || z == z_max)
                {
                    (edge_block, edge_slab_block, 0.5)
                } else {
                    (block, slab_block, -0.125)
                };

                let target_level = path_heights[(x - start.x) as usize] + edge_bias;

                let y_tgt = target_level.floor() as i32;

                let target_coord = transposer(BlockCoordinate::new(x, y_tgt, z));
                writes.blocks.push((target_coord, main));

                let cave_start = if target_level - (y_tgt as f64) > 0.5 {
                    writes
                        .blocks
                        .push((transposer(BlockCoordinate::new(x, y_tgt + 1, z)), slab));
                    y_tgt + 2
                } else {
                    y_tgt + 1
                };

                const TUNNEL_HEIGHT: i32 = 4;

                for y in cave_start..=(cave_start + TUNNEL_HEIGHT) {
                    let target_coord = transposer(BlockCoordinate::new(x, y, z));
                    writes.blocks.push((target_coord, AIR_ID));
                }
            }
        }
        writes.write(ctx)
    }
    const TOOL_ID: &'static str = "autobuild:road_tool";
}

fn i2f_lerp(start: i32, end: i32, t: f64) -> f64 {
    start as f64 + (end - start) as f64 * t
}

fn probe_ground_level(
    ctx: &HandlerContext,
    x: i32,
    z: i32,
    search_range: (i32, i32),
    natural_block_group: &FastBlockGroup,
) -> Result<Option<i32>> {
    for y in (search_range.0..=search_range.1).rev() {
        let block = ctx.game_map().get_block(BlockCoordinate::new(x, y, z))?;
        if natural_block_group.contains(block) {
            return Ok(Some(y));
        }
    }
    Ok(None)
}
