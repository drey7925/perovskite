use std::fmt::Display;

use super::{
    tracks::{eval_rotation, TRACK_TEMPLATES},
    CartsGameBuilderExtension,
};
use crate::{
    autobuild::{
        configure_item, AutobuildExt, Autobuilder, BatchedUndo, BatchedWrite, WriteParameters,
    },
    carts::{
        tracks::{build_block, place_track_interactively, ScanOutcome, ScanState, TileId},
        RAIL_INFRA_GROUP,
    },
    game_builder::OwnedTextureName,
};
use crate::{
    blocks::variants::rotate_nesw_azimuth_to_variant,
    game_builder::{GameBuilder, StaticTextureName},
    include_texture_bytes,
};
use anyhow::{bail, Context, Result};
use itertools::Itertools;
use perovskite_core::{
    chat::{ChatMessage, SERVER_ERROR_COLOR, SERVER_WARNING_COLOR},
    constants::{block_groups::TRIVIALLY_REPLACEABLE, items::default_item_interaction_rules},
    coordinates::BlockCoordinate,
    protocol::{game_rpc::place_action, items as items_proto, ui::ToolHint},
};
use perovskite_server::game_state::items::{ItemInteractionResult, PointeeBlockCoords};
use perovskite_server::game_state::{
    client_ui::{Popup, UiElementContainer},
    event::HandlerContext,
    items::{Item, ItemStack},
    player::Player,
};

pub(crate) fn register_track_tool(
    game_builder: &mut GameBuilder,
    config: &CartsGameBuilderExtension,
) -> Result<()> {
    const TRACK_TOOL_TEXTURE: StaticTextureName = StaticTextureName("carts:track_tool");
    include_texture_bytes!(game_builder, TRACK_TOOL_TEXTURE, "textures/track_tool.png")?;

    game_builder
        .inner
        .blocks_mut()
        .register_fast_block_group(TRIVIALLY_REPLACEABLE);

    let config_clone = config.clone();
    let item = Item {
        place_on_block_handler: Some(Box::new(move |ctx, coord, stack| {
            track_tool_interaction(ctx, coord, stack, &config_clone)
        })),
        ..Item::default_with_proto(items_proto::ItemDef {
            short_name: "carts:track_tool".to_string(),
            display_name: "Track placement tool".to_string(),
            appearance: TRACK_TOOL_TEXTURE.into(),
            groups: vec![],
            interaction_rules: default_item_interaction_rules(),
            quantity_type: None,
            sort_key: "carts:track_tool".to_string(),
            // This tool is used for citybuilding, so let it reach far.
            tool_range: 50.0,
        })
    };
    game_builder.inner.items_mut().register_item(item)?;

    let mut track_autorouter = Item {
        ..Item::default_with_proto(items_proto::ItemDef {
            short_name: "carts:track_autorouter".to_string(),
            display_name: "Track autorouter".to_string(),
            appearance: OwnedTextureName::from_css_color("#ffff00").into(),
            groups: vec![],
            interaction_rules: vec![],
            quantity_type: None,
            sort_key: "carts:track_autorouter".to_string(),
            // Autobuild tools are meant for citybuilding, so we give them a large tool range
            tool_range: 50.0,
        })
    };
    configure_item::<TrackAutorouter>(&mut track_autorouter);
    game_builder
        .inner
        .items_mut()
        .register_item(track_autorouter)?;

    Ok(())
}

fn track_tool_interaction(
    ctx: &HandlerContext,
    coord: PointeeBlockCoords,
    stack: &ItemStack,
    config: &CartsGameBuilderExtension,
) -> Result<ItemInteractionResult> {
    let work = |p: &Player| -> Result<()> {
        let face_dir = rotate_nesw_azimuth_to_variant(p.last_position().face_direction.0);
        let working_block = match find_working_block(ctx, coord, config) {
            Ok(b) => b,
            Err(e) => {
                p.send_chat_message(
                    ChatMessage::new_server_message(e.to_string()).with_color(SERVER_ERROR_COLOR),
                )?;
                return Ok(());
            }
        };
        p.show_popup_blocking(track_build_popup(ctx, config, working_block.0, face_dir)?)?;

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
        _ => {}
    };
    Ok(Some(stack.clone()).into())
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WorkingBlockType {
    NewRail,
    ExistingRail,
}

fn find_working_block(
    ctx: &HandlerContext<'_>,
    coord: PointeeBlockCoords,
    config: &CartsGameBuilderExtension,
) -> Result<(BlockCoordinate, WorkingBlockType)> {
    let anchor_block = ctx.game_map().get_block(coord.selected)?;
    let (working_block, block_type) = if config.is_any_rail_block(anchor_block) {
        (coord.selected, WorkingBlockType::ExistingRail)
    } else if ctx.block_types().is_trivially_replaceable(anchor_block) {
        (coord.selected, WorkingBlockType::NewRail)
    } else {
        if coord.preceding.is_none() || coord.preceding != coord.selected.try_delta(0, 1, 0) {
            bail!("Track tool only works on rails or on top of other blocks");
        }
        (coord.preceding.unwrap(), WorkingBlockType::NewRail)
    };
    Ok((working_block, block_type))
}

fn track_build_popup(
    ctx: &HandlerContext,
    config: &CartsGameBuilderExtension,
    initial_coord: BlockCoordinate,
    face_dir_as_variant: u16,
) -> Result<Popup> {
    let mut popup = ctx.new_popup().title("Track tool");
    for (key, chunk) in &TRACK_TEMPLATES.iter().chunk_by(|x| &x.category) {
        popup = popup.side_by_side_layout(key, |mut group| {
            // todo: disable buttons based on inventory, and also disable buttons for switches if there is no switch
            // actuator underneath
            for template in chunk {
                if template.bifurcate {
                    group = group.button(
                        &(template.id.to_string() + "_l"),
                        &("<-".to_string() + template.name.as_str()),
                        true,
                        false,
                    );
                    group = group.button(
                        &(template.id.to_string() + "_r"),
                        &(template.name.to_string() + "->"),
                        true,
                        false,
                    );
                } else {
                    group = group.button(
                        // this is a hack: right-handed parses as unflipped, so we just
                        // encode the right-handed template ID here.
                        &(template.id.to_string() + "_r"),
                        &template.name,
                        true,
                        false,
                    );
                }
            }
            Ok(group)
        })?;
    }
    let cloned_config = config.clone();
    popup = popup.set_button_callback(move |resp| {
        match resp.user_action {
            perovskite_server::game_state::client_ui::PopupAction::PopupClosed => {}
            perovskite_server::game_state::client_ui::PopupAction::ButtonClicked(btn) => {
                if let Err(e) = build_track(
                    &resp.ctx,
                    &cloned_config,
                    initial_coord,
                    face_dir_as_variant,
                    &btn,
                ) {
                    resp.ctx.initiator().send_chat_message(
                        ChatMessage::new_server_message(e.to_string())
                            .with_color(SERVER_ERROR_COLOR),
                    )?;
                }
            }
        }
        Ok(())
    });
    Ok(popup)
}

fn build_track(
    ctx: &HandlerContext,
    config: &CartsGameBuilderExtension,
    initial_coord: BlockCoordinate,
    face_dir_as_variant: u16,
    template_id: &str,
) -> Result<()> {
    let (template, flip) = if let Some(prefix) = template_id.strip_suffix("_l") {
        (
            TRACK_TEMPLATES
                .iter()
                .find(|x| x.id == prefix)
                .with_context(|| format!("No track template with ID {}", prefix))?,
            true,
        )
    } else if let Some(prefix) = template_id.strip_suffix("_r") {
        (
            TRACK_TEMPLATES
                .iter()
                .find(|x| x.id == prefix)
                .with_context(|| format!("No track template with ID {}", prefix))?,
            false,
        )
    } else {
        bail!("Can't parse template ID {}", template_id);
    };

    // TODO: Check inventory
    let mut bulk_edit = crate::autobuild::BatchedWrite::default();

    for tile in template.entries.iter() {
        let (cx, cz) = eval_rotation(tile.offset_x, tile.offset_z, flip, face_dir_as_variant);
        let coord = initial_coord
            .try_delta(cx, tile.offset_y, cz)
            .with_context(|| {
                format!(
                    "Out of bounds for {:?} + ({:?}, {:?})",
                    initial_coord, cx, cz
                )
            })?;
        let block = build_block(config, tile.tile_id, face_dir_as_variant, flip)
            .with_context(|| format!("Invalid tile ID: {:?}", tile.tile_id))
            .unwrap();
        bulk_edit.blocks.push((coord, block));
    }

    match bulk_edit.write(
        ctx,
        WriteParameters {
            detect_player_placed: true,
            allow_overwrite_autobuilds: false,
            overwrite_failure_action: crate::autobuild::OverwriteFailureAction::Stop,
            additional_replaceable_groups: vec![ctx
                .block_types()
                .fast_block_group(RAIL_INFRA_GROUP)
                .context("Missing RAIL_INFRA_GROUP FastBlockGroup")?],
            ..Default::default()
        },
    ) {
        Ok(undo) => {
            ctx.set_autobuild_undo(undo)?;
        }
        Err(e) => return Err(e),
    }
    Ok(())
}

#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    Debug,
    prost::Enumeration,
    strum_macros::IntoStaticStr,
    strum_macros::Display,
)]
#[repr(i32)]
enum AutorouteSlope {
    Gradual = 0,
    Steep = 1,
}

#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    Debug,
    prost::Enumeration,
    strum_macros::EnumString,
    strum_macros::IntoStaticStr,
)]
#[repr(i32)]
enum AutorouteShift {
    Auto = 0,
    Abrupt = 1,
    Diagonal8 = 2,
    Diagonal16 = 3,
}
impl AutorouteShift {
    fn display_text(&self) -> &'static str {
        match self {
            AutorouteShift::Auto => "Auto",
            AutorouteShift::Abrupt => "Abrupt (right angles)",
            AutorouteShift::Diagonal8 => "Diagonal (1/8)",
            AutorouteShift::Diagonal16 => "Diagonal (1/16)",
        }
    }
}

#[derive(Clone, PartialEq, Eq, prost::Message)]
struct AutorouterSettings {
    #[prost(enumeration = "AutorouteSlope", tag = "1")]
    slope: i32,
    #[prost(enumeration = "AutorouteShift", tag = "2")]
    shift: i32,
    #[prost(int32, tag = "3")]
    block_length: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direction {
    XPlus,
    ZPlus,
    XMinus,
    ZMinus,
}
impl Direction {
    fn from_rotation_variant(variant: u16) -> Self {
        match variant & 3 {
            0 => Self::ZPlus,
            1 => Self::XPlus,
            2 => Self::ZMinus,
            3 => Self::XMinus,
            _ => unreachable!(),
        }
    }
    fn to_variant(&self) -> u16 {
        match self {
            Self::ZPlus => 0,
            Self::XPlus => 1,
            Self::ZMinus => 2,
            Self::XMinus => 3,
        }
    }
    fn to_delta(&self) -> (i32, i32) {
        match self {
            Self::ZPlus => (0, 1),
            Self::XPlus => (1, 0),
            Self::ZMinus => (0, -1),
            Self::XMinus => (-1, 0),
        }
    }
    fn try_from_delta(delta: (i32, i32)) -> Option<Self> {
        match delta {
            (1, 0) => Some(Self::XPlus),
            (0, 1) => Some(Self::ZPlus),
            (-1, 0) => Some(Self::XMinus),
            (0, -1) => Some(Self::ZMinus),
            _ => None,
        }
    }
    fn opposite(&self) -> Self {
        match self {
            Self::XPlus => Self::XMinus,
            Self::ZPlus => Self::ZMinus,
            Self::XMinus => Self::XPlus,
            Self::ZMinus => Self::ZPlus,
        }
    }
}

#[derive(Default, Clone, Debug)]
struct AutorouterState {
    last_placement: Option<(BlockCoordinate, Option<Direction>)>, // None direction means uncommitted
    text_hint: String,
}
fn dot(u: (i32, i32), v: (i32, i32)) -> i32 {
    u.0 * v.0 + u.1 * v.1
}
/// Takes a vector, a unit vector onto, and returns (projected, residual) such that
/// v = projected + residual, and projected is parallel to onto, and residual is orthogonal to onto.
///
/// Undefined result if "onto" is not a unit vector.
fn project(v: (i32, i32), onto: (i32, i32)) -> ((i32, i32), (i32, i32)) {
    let dotted = dot(onto, v);
    let projected = (dotted * onto.0, dotted * onto.1);
    let residual = (v.0 - projected.0, v.1 - projected.1);
    (projected, residual)
}

// still TODO:
// * Slopes
// * Option to avoid building through other tracks
// * Air over track
// * Signals and gantries
// * multitrack?

struct TrackAutorouter;
impl Autobuilder for TrackAutorouter {
    type Settings = AutorouterSettings;
    type SelectionState = AutorouterState;
    const TOOL_ID: &'static str = "carts:track_autorouter";

    fn make_settings_popup(
        ctx: &HandlerContext,
        coord: PointeeBlockCoords,
        settings: &Self::Settings,
    ) -> Option<Popup> {
        // TODO
        None
    }

    fn make_advice_popup(ctx: &HandlerContext, _state: &Self::SelectionState) -> Popup {
        // TODO
        ctx.new_popup()
            .title("Track autorouter")
            .label("TODO provide some instructions here")
    }

    fn should_show_advice(state: &Self::SelectionState) -> bool {
        false
    }

    fn tap(
        ctx: &HandlerContext,
        coord: PointeeBlockCoords,
        _settings: &mut Self::Settings,
        state: &mut Self::SelectionState,
    ) -> Result<()> {
        let (block, direction, selection_type) = compute_working_block_for_tool(ctx, coord)?;
        if selection_type == WorkingBlockType::NewRail {
            state.last_placement = Some((block, None));
            state.text_hint =
                "Initial tile selected. Right-click to build, left-click to reset.".to_string();
        } else {
            state.last_placement = Some((block, Some(direction)));
            state.text_hint =
                "Continuing from selected track. Right-click to extend, left-click to reset."
                    .to_string();
        }
        Ok(())
    }

    fn build(
        ctx: &HandlerContext,
        pointee: PointeeBlockCoords,
        settings: &mut Self::Settings,
        state: &mut Self::SelectionState,
    ) -> Result<Option<crate::autobuild::BatchedUndo>> {
        let (end, end_direction, end_type) = compute_working_block_for_tool(ctx, pointee)?;
        let end_direction = match end_type {
            WorkingBlockType::NewRail => end_direction,
            // If it's an existing rail, we're given the direction looking off its
            // cut end, while we really want to approach that end from our path.
            WorkingBlockType::ExistingRail => end_direction.opposite(),
        };
        let end = (end, end_direction);
        let start = state.last_placement.clone();

        let (undo, end_dir) = if let Some(start) = start {
            build_impl(ctx, settings, start, end)?
        } else {
            // build_single_track(ctx, settings, end)?
            bail!("todo build_single_track");
        };
        if end_type == WorkingBlockType::NewRail {
            let new_end_coord = BlockCoordinate::new(
                end.0.x + end_dir.to_delta().0,
                end.0.y,
                end.0.z + end_dir.to_delta().1,
            );
            state.last_placement = Some((new_end_coord, Some(end_dir)));
            state.text_hint =
                "Continuing segment. Left-click to reset and start a new segment, right-click to extend."
                    .to_string();
        } else {
            state.last_placement = None;
            state.text_hint =
                "Track complete. Click on an existing track to start a new segment.".to_string();
        }
        Ok(Some(undo))
    }

    fn current_hint(
        _settings: &Self::Settings,
        state: &Self::SelectionState,
        _ctx: &HandlerContext,
    ) -> Option<ToolHint> {
        let static_string = if state.text_hint.is_empty() {
            None
        } else {
            Some(state.text_hint.clone())
        };

        let coord = state.last_placement.map(|(c, _)| c);
        if coord.is_some() || static_string.is_some() {
            return Some(ToolHint {
                static_string,
                edit_delta_from: coord.map(|x| x.into()),
                ..Default::default()
            });
        }
        None
    }
}

fn determine_track_exit(
    ctx: &HandlerContext<'_>,
    coord: BlockCoordinate,
    config: &CartsGameBuilderExtension,
) -> Result<BlockCoordinate> {
    let block = ctx.game_map().get_block(coord)?;
    let tile_id = if block.equals_ignore_variant(config.rail_block) {
        TileId::from_variant(block.variant(), false, false)
    } else {
        anyhow::bail!("Not on a track");
    };

    let start_state = ScanState {
        block_coord: coord,
        is_reversed: false,
        is_diverging: false,
        allowable_speed: 90.0,
        current_tile_id: tile_id,
        odometer: 0.0,
    };
    let start_reversed_state = ScanState {
        is_reversed: true,
        ..start_state
    };
    let forward_outcome =
        start_state.advance::<false>(|c| ctx.game_map().get_block(c).into(), config)?;
    let backward_outcome =
        start_reversed_state.advance::<false>(|c| ctx.game_map().get_block(c).into(), config)?;
    match (forward_outcome, backward_outcome) {
        (ScanOutcome::DisconnectedTrack(next_coord, straight_valid), ScanOutcome::Success(_)) => {
            if !straight_valid {
                anyhow::bail!("Existing track cannot connect to a straight track");
            }
            Ok(next_coord)
        }
        (ScanOutcome::Success(_), ScanOutcome::DisconnectedTrack(next_coord, straight_valid)) => {
            if !straight_valid {
                anyhow::bail!("Existing track cannot connect to a straight track");
            }
            Ok(next_coord)
        }
        (ScanOutcome::Success(_), ScanOutcome::Success(_)) => Err(anyhow::anyhow!(
            "Track is already connected on both ends. Cannot connect to it."
        )),
        (ScanOutcome::DisconnectedTrack(_, _), ScanOutcome::DisconnectedTrack(_, _)) => Err(
            anyhow::anyhow!("Track is not connected on exactly one side"),
        ),
        (forward, backward) => Err(anyhow::anyhow!(
            "Track is not connected on exactly one side. Forward: {:?}, Backward: {:?}",
            forward,
            backward
        )),
    }
}

/// computes the working block for a generic track building tool.
/// Args:
/// 	ctx - the handler context
/// 	coord - the pointee block coordinates
fn compute_working_block_for_tool(
    ctx: &HandlerContext<'_>,
    coord: PointeeBlockCoords,
) -> Result<(BlockCoordinate, Direction, WorkingBlockType)> {
    let azimuth = ctx
        .initiator()
        .position()
        .context("Autorouter used without a player position; programmatic usages should call track builder functions directly")?
        .face_direction.0;
    let face_dir = rotate_nesw_azimuth_to_variant(azimuth);
    let config = ctx
        .extension::<CartsGameBuilderExtension>()
        .context("Missing extension")?;
    let (selected_block, block_type) = find_working_block(ctx, coord, config)?;
    let player_direction = Direction::from_rotation_variant(face_dir);
    match block_type {
        WorkingBlockType::NewRail => Ok((selected_block, player_direction, block_type)),
        WorkingBlockType::ExistingRail => {
            let next = determine_track_exit(ctx, selected_block, config)?;
            let next_direction = match (next.x - selected_block.x, next.z - selected_block.z) {
                (1, 0) => Direction::XPlus,
                (-1, 0) => Direction::XMinus,
                (0, 1) => Direction::ZPlus,
                (0, -1) => Direction::ZMinus,
                _ => bail!("Nonadjacent tile found"),
            };
            Ok((next, next_direction, block_type))
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StepType {
    Straight,
    SharpTurn,
}
impl StepType {
    fn allows_slope(&self) -> bool {
        match self {
            StepType::Straight => true,
            StepType::SharpTurn => false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Step {
    x: i32,
    z: i32,
    step_type: StepType,
    rotation: i8,
}

/// Actually builds a track
///
/// Args:
/// 	ctx - the handler context
/// 	settings - the autorouter settings
/// 	start - the start of the track (coordinate + direction)
/// 	end - the end of the track (coordinate + direction)
///
/// Returns:
/// 	- A BatchedUndo to undo the changes
/// 	- The direction of the end of the track (the inferred one if None was passed in)
fn build_impl(
    ctx: &HandlerContext,
    settings: &AutorouterSettings,
    start: (BlockCoordinate, Option<Direction>),
    end: (BlockCoordinate, Direction),
) -> Result<(crate::autobuild::BatchedUndo, Direction)> {
    let ext = ctx
        .extension::<CartsGameBuilderExtension>()
        .context("Missing carts extension")?;
    let slopes = match settings.slope() {
        AutorouteSlope::Gradual => ext.rail_slopes_8.to_vec(),
        AutorouteSlope::Steep => vec![ext.rail_slope_1],
    };

    let mut builder = BatchedWrite::default();

    let delta = (end.0.x - start.0.x, end.0.z - start.0.z);
    if delta.0 == 0 && delta.1 == 0 {
        bail!("Start and end are the same point");
    }
    if delta.0.abs() + delta.1.abs() > 1024 {
        bail!("End is too far from start; max 1024 blocks per step");
    }
    let mut steps = vec![];

    let _delta_y = end.0.y - start.0.y;

    // Step 1: go along the initial direction until we are in line with the end point
    // notation: ds is our differential element, as a vec2
    //           delta is our total delta to our goal point.
    let start_dir = start.1.unwrap_or_else(|| {
        if delta.0.abs() > delta.1.abs() {
            if delta.0 > 0 {
                Direction::XPlus
            } else {
                Direction::XMinus
            }
        } else {
            if delta.1 > 0 {
                Direction::ZPlus
            } else {
                Direction::ZMinus
            }
        }
    });
    let end_dir = end.1;

    let ds = start_dir.to_delta();
    if dot(ds, delta) < 0 {
        bail!("Path ends up behind the start");
    }

    if dot(end_dir.to_delta(), delta) < 0 {
        bail!("Path ends up behind the end");
    }

    let (first_seg, second_seg) = project(delta, ds);

    let initial_seg_end = (start.0.x + first_seg.0, start.0.z + first_seg.1);
    let mut cursor = (start.0.x, start.0.z);

    // Insert synthetic blocks for the start (and later, at the end)
    // These remove edge cases from the curve detection code, hence simplifying it
    steps.push(Step {
        // Back off one step - very possibly onto existing rail but it doesn't matter; these
        // are sentinels
        x: start.0.x - start_dir.to_delta().0,
        z: start.0.z - start_dir.to_delta().1,
        step_type: StepType::Straight,
        rotation: start_dir.to_variant() as i8,
    });

    loop {
        if steps.len() > 1026 {
            bail!("Too many steps (internal logic bug, please report)");
        }

        steps.push(Step {
            x: cursor.0,
            z: cursor.1,
            step_type: StepType::Straight,
            rotation: start_dir.to_variant() as i8,
        });

        if (cursor.0, cursor.1) == initial_seg_end {
            break;
        }
        cursor.0 += ds.0;
        cursor.1 += ds.1;
    }

    if second_seg != (0, 0) {
        let ds = (second_seg.0.signum(), second_seg.1.signum());
        // advance the cursor; this point does double duty in both the first and second segment,
        // and we don't want to double-add it.
        cursor.0 += ds.0;
        cursor.1 += ds.1;
        let second_variant = Direction::try_from_delta(ds)
            .with_context(|| format!("Invalid second segment {:?}", second_seg))?;
        loop {
            if steps.len() > 1026 {
                bail!("Too many steps (internal logic bug, please report)");
            }

            steps.push(Step {
                x: cursor.0,
                z: cursor.1,
                step_type: StepType::Straight,
                rotation: second_variant.to_variant() as i8,
            });

            if (cursor.0, cursor.1) == (end.0.x, end.0.z) {
                break;
            }
            cursor.0 += ds.0;
            cursor.1 += ds.1;
        }
    }

    // Insert synthetic block for the end (back off one step in the final direction)
    steps.push(Step {
        x: end.0.x + end_dir.to_delta().0,
        z: end.0.z + end_dir.to_delta().1,
        step_type: StepType::Straight,
        rotation: end_dir.to_variant() as i8,
    });

    // Up until now, the step_type is not yet known, so it's fixed at "straight"
    // Update it in-place, and also reconcile curve types.
    for i in 1..steps.len() - 1 {
        let prev = steps[i - 1];
        let next = steps[i + 1];

        let cur = &mut steps[i];

        let diff_prev = (prev.x - cur.x, prev.z - cur.z);
        let diff_next = (next.x - cur.x, next.z - cur.z);

        const XP: (i32, i32) = (1, 0);
        const XN: (i32, i32) = (-1, 0);
        const ZP: (i32, i32) = (0, 1);
        const ZN: (i32, i32) = (0, -1);

        let (form, variant) = match (diff_prev, diff_next) {
            (ZN, ZP) => (StepType::Straight, 0),
            (XN, XP) => (StepType::Straight, 1),
            (ZP, ZN) => (StepType::Straight, 2),
            (XP, XN) => (StepType::Straight, 3),
            (XP, ZN) | (ZN, XP) => (StepType::SharpTurn, 0),
            (XN, ZN) | (ZN, XN) => (StepType::SharpTurn, 1),
            (XN, ZP) | (ZP, XN) => (StepType::SharpTurn, 2),
            (XP, ZP) | (ZP, XP) => (StepType::SharpTurn, 3),
            _ => bail!("Invalid step transition {:?} {:?}", diff_prev, diff_next),
        };
        cur.step_type = form;
        cur.rotation = variant;
    }

    // Skip the first and last steps - they're just sentinels
    for s in &steps[1..steps.len() - 1] {
        let block = match s.step_type {
            StepType::Straight => ext.rail_block.with_variant_unchecked(s.rotation as u16),
            StepType::SharpTurn => ext.rail_block.with_variant_unchecked(
                TileId::new(8, 8, s.rotation as u16, false, false, false).block_variant(),
            ),
        };
        builder
            .blocks
            .push((BlockCoordinate::new(s.x, start.0.y, s.z), block))
    }
    let undo = builder.write(
        ctx,
        WriteParameters {
            detect_player_placed: true,
            allow_overwrite_autobuilds: false,
            overwrite_failure_action: crate::autobuild::OverwriteFailureAction::Stop,
            additional_replaceable_groups: vec![ctx
                .block_types()
                .fast_block_group(RAIL_INFRA_GROUP)
                .context("missing rail infra group")?],
        },
    )?;
    Ok((undo, end_dir))
}
