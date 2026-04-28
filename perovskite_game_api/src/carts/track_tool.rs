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
        let anchor_block = ctx.game_map().get_block(coord.selected)?;
        let working_block = if config.is_any_rail_block(anchor_block)
            || ctx.block_types().is_trivially_replaceable(anchor_block)
        {
            coord.selected
        } else {
            if coord.preceding.is_none() || coord.preceding != coord.selected.try_delta(0, 1, 0) {
                p.send_chat_message(
                    ChatMessage::new_server_message(
                        "Track tool only works on rails or on top of other blocks",
                    )
                    .with_color(SERVER_ERROR_COLOR),
                )?;
                return Ok(());
            }
            coord.preceding.unwrap()
        };

        let face_dir = rotate_nesw_azimuth_to_variant(p.last_position().face_direction.0);

        p.show_popup_blocking(track_build_popup(ctx, config, working_block, face_dir)?)?;

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

#[derive(Default, Clone, Debug)]
enum AutorouterState {
    // All preliminary, and will be revised as the autorouter becomes more powerful.
    /// The player has picked a starting point for the track. Their next click will
    /// determine the end of the track.
    #[default]
    Empty,
    /// The player has picked a starting point for the track. Their next click will
    /// determine an ending point for the track, and may or may not require another
    /// click to determine the shift point, depending on the geometry.
    ///
    /// Includes both the coordinate of the existing track to attach to, and the exit
    /// direction.
    StartPicked(CoordinateAndDelta),
    /// The player has picked both the start and end points for the track.
    NeedsShiftPoint(CoordinateAndDelta, CoordinateAndDelta, i32),
    /// Ready to build
    Ready(TrackPlan),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TrackPlanType {
    StraightLine,
    // gives the shift length when returning a pre-plan, or the actual coordinate where
    // the shift should happen when executing a plan.
    StraightWithShift(i32, Option<BlockCoordinate>),
    RightAngleTurn,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct TrackPlan {
    plan_type: TrackPlanType,
    start: CoordinateAndDelta,
    end: CoordinateAndDelta,
    abs_vertical_delta: i32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Orientation {
    Z,
    X,
}
impl Orientation {
    fn variant(&self) -> u16 {
        match self {
            Orientation::Z => 0,
            Orientation::X => 1,
        }
    }
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

fn classify_plan(start: CoordinateAndDelta, end: CoordinateAndDelta) -> Result<TrackPlan> {
    let dx = end.coord.x - start.coord.x;
    let dz = end.coord.z - start.coord.z;
    let dy = end.coord.y - start.coord.y;
    if dx == 0 && dz == 0 {
        anyhow::bail!("Start and end points are the same");
    }

    let displacement = (dx, dz);

    if dot(displacement, start.delta) == 0 || dot(displacement, end.delta) == 0 {
        bail!("The start and end points are not facing each other.")
    }

    if dot(start.delta, end.delta) == 0 {
        // start and end paths are orthogonal to each other
        return Ok(TrackPlan {
            plan_type: TrackPlanType::RightAngleTurn,
            abs_vertical_delta: dy.abs(),
            start,
            end,
        });
    }

    let (main, residual) = project(displacement, start.delta);
    if residual != (0, 0) {
        // The residual should be axis-aligned
        assert!(residual.0 == 0 || residual.1 == 0);
        Ok(TrackPlan {
            plan_type: TrackPlanType::StraightWithShift(residual.0.abs() + residual.1.abs(), None),
            abs_vertical_delta: dy.abs(),
            start,
            end,
        })
    } else {
        // No residual, so the track is a straight line
        Ok(TrackPlan {
            plan_type: TrackPlanType::StraightLine,
            abs_vertical_delta: dy.abs(),
            start,
            end,
        })
    }
}

/// Returns (dx, dz) describing the one direction that the track can connect to, or an error
/// (with a user-facing message) if there is either no connection or a connection on both sides.
fn determine_track_exit(
    ctx: &HandlerContext<'_>,
    coord: BlockCoordinate,
    config: &CartsGameBuilderExtension,
) -> Result<(i32, i32)> {
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
            Ok((next_coord.x - coord.x, next_coord.z - coord.z))
        }
        (ScanOutcome::Success(_), ScanOutcome::DisconnectedTrack(next_coord, straight_valid)) => {
            if !straight_valid {
                anyhow::bail!("Existing track cannot connect to a straight track");
            }
            Ok((next_coord.x - coord.x, next_coord.z - coord.z))
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

#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
struct CoordinateAndDelta {
    coord: BlockCoordinate,
    delta: (i32, i32),
}
impl CoordinateAndDelta {
    fn real_coord(self) -> BlockCoordinate {
        BlockCoordinate {
            x: self.coord.x + self.delta.0,
            y: self.coord.y,
            z: self.coord.z + self.delta.1,
        }
    }
}

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
        settings: &mut Self::Settings,
        state: &mut Self::SelectionState,
    ) -> Result<()> {
        if let AutorouterState::NeedsShiftPoint(start, end, shift_length) = *state {
            *state = AutorouterState::Ready(TrackPlan {
                plan_type: TrackPlanType::StraightWithShift(shift_length, Some(coord.selected)),
                abs_vertical_delta: end.coord.y - start.coord.y,
                start,
                end,
            });
            return Ok(());
        }

        let exit_dir = determine_track_exit(
            ctx,
            coord.selected,
            ctx.extension().context("Missing carts extension")?,
        )?;

        let next_state = match state {
            AutorouterState::Empty => AutorouterState::StartPicked(CoordinateAndDelta {
                coord: coord.selected,
                delta: exit_dir,
            }),
            AutorouterState::StartPicked(start) => {
                let end = CoordinateAndDelta {
                    coord: coord.selected,
                    delta: exit_dir,
                };
                let plan = classify_plan(*start, end)?;
                match plan.plan_type {
                    TrackPlanType::StraightWithShift(shift_length, _) => {
                        AutorouterState::NeedsShiftPoint(*start, end, shift_length)
                    }
                    TrackPlanType::StraightLine => AutorouterState::Ready(plan),
                    TrackPlanType::RightAngleTurn => AutorouterState::Ready(plan),
                }
            }
            AutorouterState::NeedsShiftPoint(start, end, shift_length) => {
                unreachable!();
            }
            AutorouterState::Ready(plan) => AutorouterState::Ready(*plan),
        };
        *state = next_state;
        Ok(())
    }

    fn build(
        ctx: &HandlerContext,
        pointee: PointeeBlockCoords,
        settings: &mut Self::Settings,
        state: &mut Self::SelectionState,
    ) -> Result<crate::autobuild::BatchedUndo> {
        if let AutorouterState::Ready(plan) = *state {
            let res = build_impl(ctx, settings, plan);
            *state = AutorouterState::Empty;
            res
        } else {
            *state = AutorouterState::Empty;
            place_track_interactively(
                ctx,
                pointee,
                ctx.extension::<CartsGameBuilderExtension>()
                    .context("Missing extension")?
                    .rail_block,
                0,
                0,
            )?;
            Ok(BatchedUndo::default())
        }
    }

    fn current_hint(
        _settings: &Self::Settings,
        state: &Self::SelectionState,
        _ctx: &HandlerContext,
    ) -> Option<ToolHint> {
        match state {
            AutorouterState::Empty => Some(ToolHint {
                static_string: Some("Click on a track to begin".to_string()),
                edit_delta_from: None,
            }),
            AutorouterState::StartPicked(start) => Some(ToolHint {
                static_string: Some("Click on the second track to finish selection".to_string()),
                edit_delta_from: Some(start.coord.into()),
            }),
            AutorouterState::NeedsShiftPoint(start, _end, shift_length) => Some(ToolHint {
                static_string: Some(
                    format!(
                        "Click the point where the tracks should shift sideway by {shift_length} blocks to line up",
                    )
                ),
                edit_delta_from: Some(start.coord.into()),
            }),
            AutorouterState::Ready(plan) => Some(ToolHint {
                static_string: Some("Ready to build! Right-click anywhere to build.".to_string()),
                edit_delta_from: Some(plan.start.coord.into()),
            }),
        }
    }
}

enum StepType {
    Straight,
    SharpTurn,
    CrossStraight,
}
impl StepType {
    fn allows_slope(&self) -> bool {
        match self {
            StepType::Straight => true,
            StepType::SharpTurn => false,
            StepType::CrossStraight => false,
        }
    }
}

struct Step {
    x: i32,
    z: i32,
    step_type: StepType,
    rotation: i8,
}

fn range_noninclusive(start: i32, end: i32) -> num::iter::RangeStep<i32> {
    if start < end {
        num::range_step(start, end, 1)
    } else {
        num::range_step(start, end, -1)
    }
}

fn range_inclusive(start: i32, end: i32) -> num::iter::RangeStepInclusive<i32> {
    if start < end {
        num::range_step_inclusive(start, end, 1)
    } else {
        num::range_step_inclusive(start, end, -1)
    }
}

fn build_impl(
    ctx: &HandlerContext,
    settings: &AutorouterSettings,
    plan: TrackPlan,
) -> Result<crate::autobuild::BatchedUndo> {
    let ext = ctx
        .extension::<CartsGameBuilderExtension>()
        .context("Missing carts extension")?;
    let slopes = match settings.slope() {
        AutorouteSlope::Gradual => ext.rail_slopes_8.to_vec(),
        AutorouteSlope::Steep => vec![ext.rail_slope_1],
    };

    let mut builder = BatchedWrite::default();

    let start = plan.start.real_coord();
    let end = plan.end.real_coord();

    let steps: Vec<Step> = match plan.plan_type {
        TrackPlanType::StraightLine => {
            if start.x == end.x {
                let z_range = range_inclusive(start.z, end.z);
                z_range
                    .map(|z| Step {
                        x: start.x,
                        z,
                        step_type: StepType::Straight,
                        rotation: 0,
                    })
                    .collect()
            } else {
                let x_range = range_inclusive(start.x, end.x);
                x_range
                    .map(|x| Step {
                        x,
                        z: start.z,
                        step_type: StepType::Straight,
                        rotation: 1,
                    })
                    .collect()
            }
        }
        TrackPlanType::RightAngleTurn => {
            if plan.start.delta.0 == 0 {
                // along Z then along X
                let z_range = range_noninclusive(start.z, end.z);
                let x_range = range_noninclusive(end.x, start.x);
                let mut steps = vec![];
                steps.extend(z_range.map(|z| Step {
                    x: start.x,
                    z,
                    step_type: StepType::Straight,
                    rotation: 0,
                }));
                steps.push(Step {
                    x: start.x,
                    z: end.z,
                    step_type: StepType::SharpTurn,
                    rotation: 0,
                });
                steps.extend(x_range.map(|x| Step {
                    x,
                    z: end.z,
                    step_type: StepType::Straight,
                    rotation: 1,
                }));
                steps
            } else {
                // along X then along Z
                let x_range = range_noninclusive(start.x, end.x);
                let z_range = range_noninclusive(end.z, start.z);
                let mut steps = vec![];
                steps.extend(x_range.map(|x| Step {
                    x,
                    z: start.z,
                    step_type: StepType::Straight,
                    rotation: 0,
                }));
                steps.push(Step {
                    x: end.x,
                    z: start.z,
                    step_type: StepType::SharpTurn,
                    rotation: 0,
                });
                steps.extend(z_range.map(|z| Step {
                    x: end.x,
                    z,
                    step_type: StepType::Straight,
                    rotation: 1,
                }));
                steps
            }
        }
        TrackPlanType::StraightWithShift(_, _) => bail!("todo"),
    };

    for s in steps {
        let block = match s.step_type {
            StepType::Straight => ext.rail_block.with_variant_unchecked(s.rotation as u16),
            StepType::CrossStraight => ext.rail_block.with_variant_unchecked(s.rotation as u16 ^ 1),
            StepType::SharpTurn => ext.rail_block.with_variant_unchecked(
                TileId::new(8, 8, s.rotation as u16, false, false, false).block_variant(),
            ),
        };
        builder
            .blocks
            .push((BlockCoordinate::new(s.x, plan.start.coord.y, s.z), block))
    }
    builder.write(
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
    )
}
