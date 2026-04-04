//! Tools for players to quickly build structures. Initial focus is on roads, to prove
//! out the UX and get some initial experimentation.

use std::{any::Any, iter};

use anyhow::{bail, Context, Result};
use ndarray::s;
use perovskite_core::{
    block_id::{special_block_defs::AIR_ID, BlockId},
    chat::{ChatMessage, SERVER_ERROR_COLOR},
    constants::items::default_item_interaction_rules,
    coordinates::BlockCoordinate,
    protocol::items as items_proto,
};
use perovskite_server::game_state::{
    blocks::FastBlockGroup,
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
            interaction_rules: default_item_interaction_rules(),
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

    /// Actually builds. Invoked when the player right-clicks.
    fn build(
        ctx: &HandlerContext,
        right_click_coord: BlockCoordinate,
        settings: Self::Settings,
        state: &mut Self::SelectionState,
    ) -> Result<()>;
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
            if let Err(e) = T::build(ctx, coord.selected, settings, &mut data) {
                p.send_chat_message(
                    ChatMessage::new_server_message(e.to_string()).with_color(SERVER_ERROR_COLOR),
                )?;
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
                TextFieldBuilder::new("width")
                    .label("Width")
                    .initial(settings.width.to_string()),
            )
            .text_field(
                TextFieldBuilder::new("tunnel_height")
                    .label("Tunnel Height")
                    .initial(settings.tunnel_height.to_string()),
            )
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
                let tunnel_height = resp
                    .textfield_values
                    .get("tunnel_height")
                    .context("Missing tunnel_height")?
                    .parse::<u32>()?;
                let settings = RoadToolSettings {
                    block: block.0,
                    width,
                    slab_block: slab_block.0,
                    tunnel_height,
                };
                Self::save_settings(&resp.ctx, settings)
            })
    }
    fn make_advice_popup(ctx: &HandlerContext, _state: &Self::SelectionState) -> Popup {
        ctx.new_popup()
        .title("Road tool")
        .label("Tap the tool on the ground to set the starting point. Then, right-click to finish building.")
        .label("Hold the dig/tap button to change road settings.")
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
    ) -> Result<()> {
        ctx.initiator()
            .send_chat_message(ChatMessage::new_server_message("Building road"))?;

        let start = state.start.context("No start point")?;
        let end = right_click_coord;
        let initiator = ctx.initiator();
        let mut block = BlockId(settings.block);
        let mut slab_block = BlockId(settings.slab_block);

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
        let start = transposer(start);
        let end = transposer(end);

        // A heuristic on how long this road is, used to determine how far up/down
        // we'll look for the ground contour
        let heuristic_len = (abs_dx + abs_dz) as i32;

        if heuristic_len > 1000 {
            bail!("Road too long");
        }

        let (start, end) = if start.x > end.x {
            (end, start)
        } else {
            (start, end)
        };

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
        let mut raw_path_heights: Vec<i32> = Vec::with_capacity((end.x - start.x + 1) as usize);
        for x in start.x..=end.x {
            let t = (x - start.x) as f64 / (end.x - start.x) as f64;
            let z_center = i2f_lerp(start.z, end.z, t);

            let z_min = (z_center - settings.width as f64 / 2.0).round() as i32;
            let z_max = (z_center + settings.width as f64 / 2.0).round() as i32;

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
                bail!(
                    "Cannot estimate path height near {:?}",
                    tuple_transposer((x, z_center.round() as i32))
                );
            }
            raw_path_heights.push(num / den);
        }

        if raw_path_heights.is_empty() {
            bail!("Segment too short; nothing to do")
        }

        let bridge_mode = state.pending_end.take() == Some(end);

        let start_height = raw_path_heights[0];
        let end_height = raw_path_heights[raw_path_heights.len() - 1];
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
        for x in start.x..=end.x {
            let t = (x - start.x) as f64 / (end.x - start.x) as f64;
            let z_center = i2f_lerp(start.z, end.z, t);

            let z_min = (z_center - settings.width as f64 / 2.0).round() as i32;
            let z_max = (z_center + settings.width as f64 / 2.0).round() as i32;

            for z in z_min..=z_max {
                let target_level = path_heights[(x - start.x) as usize];
                let y_tgt = target_level.round() as i32;
                let target_coord = transposer(BlockCoordinate::new(x, y_tgt, z));
                ctx.game_map().set_block(target_coord, block, None)?;

                const TUNNEL_HEIGHT: i32 = 4;

                let current_block = if target_level.fract() < 0.5 {
                    block
                } else {
                    // todo: base block under slab block
                    slab_block
                };
                let target_coord = transposer(BlockCoordinate::new(x, y_tgt, z));
                ctx.game_map()
                    .set_block(target_coord, current_block, None)?;
                // todo: detect player-placed blocks and don't overwrite them
                for y in (y_tgt + 1)..=(y_tgt + TUNNEL_HEIGHT) {
                    let target_coord = transposer(BlockCoordinate::new(x, y, z));
                    ctx.game_map().set_block(target_coord, AIR_ID, None)?;
                }
            }
        }
        Ok(())
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
