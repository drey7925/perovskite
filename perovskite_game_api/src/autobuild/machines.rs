//! Autobuild machines: autobuild, but made of blocks with tasks.
//!
//! A player can assemble a handful of blocks, each with a task
//! (e.g. "place carts:rail_tile", "if non-air replace with tunnel_wall",
//! "if air, replace with bridge_deck", "every 32, place track gantry"),
//! almost jacquard loom style config, that you can set up and leave running.
use std::{
    collections::{BTreeMap, HashSet},
    iter,
    ops::{ControlFlow, DerefMut},
};

use anyhow::{bail, Context, Result};
use itertools::Itertools;
use num::Zero;
use perovskite_core::{
    block_id::{special_block_defs::AIR_ID, BlockId},
    chat::ChatMessage,
    coordinates::{BlockCoordinate, PlayerPositionUpdate},
    protocol::render::TextureTransform,
};
use perovskite_server::game_state::{
    blocks::CompassDirection,
    client_ui::UiElementContainer,
    event::HandlerContext,
    items::{PointeeBlockCoords, DIG_ANY_SOLID_STACK},
    GameStateExtension,
};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    blocks::{BlockAppearanceBuilder, BlockBuilder, CubeAppearanceBuilder},
    colors::Color,
    default_game::block_groups::BRITTLE,
    game_builder::{GameBuilder, GameBuilderExtension, StaticBlockName, TextureRefExt},
    include_texture_bytes, NonExhaustive,
};

pub const AUTOBUILD_MACHINES_GROUP: &str = "autobuild:machines";

#[derive(Clone, Debug)]
pub struct ActionState<'a> {
    /// The coordinate of the machine block itself.
    pub machine_coord: BlockCoordinate,
    /// The movement delta for the machine group after this action is completed.
    pub movement_delta: (i32, i32, i32),
    /// Data sensed from the sense step, or Default if either sensing not yet done this cycle,
    /// or nothing was sensed. During the sense stage, this will contain sense data from previous
    /// sensed blocks in the current machine cycle, but the order in which blocks are visited will
    /// be unpredictable;
    pub sense_data: &'a SenseOutput,
    /// The expected BlockId of the machine acting here. Used to detect race conditions, averting
    /// a circular dependency where we need a MachineAction to build a machine block definition, but
    /// would otherwise need the ID from the machine block definition to construct the MachineAction.
    ///
    /// This also includes the block variant, so we know (for free) the orientation of the machine
    /// as it was facing when the framework called a machine function.
    pub machine_block_id: BlockId,
    pub _ne: NonExhaustive,
}

/// Data that machine blocks produce during sensing, and that the framework will aggregate
#[derive(Clone, Debug)]
pub struct SenseInput {
    /// How far the machine should move. At most one block can report this, or the system will refuse to move.
    requested_movement: Option<(i32, i32, i32)>,
    pub _ne: NonExhaustive,
}

#[derive(Clone, Copy, Debug)]
pub enum Movement {
    /// No block requested a movement
    None,
    /// One or more blocks requested the same movement
    Some((i32, i32, i32)),
    /// Two or more blocks requested inconsistent movements.
    Conflict,
}

/// Aggregated data across all of the [SenseInput]s returned by all of the blocks.
#[derive(Clone, Debug)]
pub struct SenseOutput {
    pub movement: Movement,
    pub errors: HashSet<String>,
    pub _ne: NonExhaustive,
}

impl SenseOutput {
    fn merge_in(&mut self, other: SenseInput) {
        if let Some(delta) = other.requested_movement {
            let new_movement = match self.movement {
                Movement::None => Movement::Some(delta),
                Movement::Some(current_delta) if current_delta == delta => Movement::Some(delta),
                _ => {
                    println!("{:?} {:?}", delta, self.movement);
                    self.errors.insert(
                        "Multiple different movements requested by different blocks".to_string(),
                    );
                    Movement::Conflict
                }
            };
            self.movement = new_movement;
        }
    }
}

impl Default for SenseInput {
    fn default() -> Self {
        Self {
            requested_movement: None,
            _ne: NonExhaustive(()),
        }
    }
}

impl Default for SenseOutput {
    fn default() -> Self {
        Self {
            movement: Movement::None,
            errors: HashSet::new(),
            _ne: NonExhaustive(()),
        }
    }
}

/// Currently, just *gameplay-level* errors (distinguished from anyhow::Result which
/// represent *system* errors). More may be added in the future
#[must_use]
pub struct ActionOutcome {
    errors: HashSet<String>,
    pub _ne: NonExhaustive,
}
impl Default for ActionOutcome {
    fn default() -> Self {
        ActionOutcome {
            errors: HashSet::new(),
            _ne: NonExhaustive(()),
        }
    }
}
impl ActionOutcome {
    pub fn extend(&mut self, other: Self) {
        self.errors.extend(other.errors.into_iter());
    }

    pub fn from_error(error: impl Into<String>) -> Self {
        Self {
            errors: HashSet::from_iter(iter::once(error.into())),
            _ne: NonExhaustive(()),
        }
    }
}

#[non_exhaustive]
pub enum MachineCycle {
    /// First step of the cycle: sensing machines act on their ambient properties (and make inventory requests to machines that carry items)
    Sense,
    /// Second step of the cycle: machines that dig remove a block (and possibly store it). Machines that carry items fulfill (or don't fulfill)
    /// the requested items. (note, inventory features are planned for later and not yet implemented; this enum description is forward-looking)
    DigAndStage,
    /// Third step of the cycle: machines that place will place the necessary block (possibly taking it from a block that was dug)
    Place,
}

pub trait MachineAction: Send + Sync {
    fn sense(&self, _ctx: &HandlerContext, _state: &ActionState) -> Result<SenseInput> {
        // Default impl: don't sense anything.
        Ok(Default::default())
    }

    /// The action to perform when the machine is activated. This should just do the intended action
    /// of the machine, but not move itself (despite what movement_delta says). The machine system will
    /// handle movement on its own.
    fn act(&self, ctx: &HandlerContext, state: &ActionState) -> Result<ActionOutcome>;
}

/// Definition for the behavior of a machine.
///
/// Construct with [Into::into] (in the future, there may be parameters in the
/// MachineDef that can be customized)
pub struct MachineDef {
    pub action: Box<dyn MachineAction + Send + Sync + 'static>,
    pub _ne: NonExhaustive,
}
impl<T: MachineAction + Send + Sync + 'static> From<T> for MachineDef {
    fn from(value: T) -> Self {
        MachineDef {
            action: Box::new(value),
            _ne: NonExhaustive(()),
        }
    }
}

struct MachinesBuilderExt {
    block_types: Option<BTreeMap<u32, MachineDef>>,
}
impl Default for MachinesBuilderExt {
    fn default() -> Self {
        Self {
            block_types: Some(BTreeMap::new()),
        }
    }
}

struct MachinesExt {
    block_types: BTreeMap<u32, MachineDef>,
}

impl GameBuilderExtension for MachinesBuilderExt {
    fn pre_run(&mut self, server_builder: &mut perovskite_server::server::ServerBuilder) {
        server_builder
            .blocks_mut()
            .register_fast_block_group(AUTOBUILD_MACHINES_GROUP);
        server_builder.add_extension(MachinesExt {
            block_types: self.block_types.take().expect("pre_run already called"),
        });
    }
}

impl GameStateExtension for MachinesExt {}

/// Registers a new machine type. Note tht the API is in flux.
///
/// Args:
///   game_builder: The game to build this nachine type into
///   block_builder: A block builder with the desired name, appearance, etc of the machine.
///     This will be modified to mix in additional machine properties. This does have a limitation
///     that the same block cannot be used with another framework that takes a BlockBuilder by value;
///     if this is an impediment please open an issue.
///   machine_def: Either a MachineDef or an Action
pub fn register_machine_type(
    game_builder: &mut GameBuilder,
    block_builder: BlockBuilder,
    def: impl Into<MachineDef>,
) -> Result<()> {
    let built = block_builder
        .add_block_group(AUTOBUILD_MACHINES_GROUP)
        .force_disable_track_placer()
        .build_and_deploy_into(game_builder)?;

    let base_id = built.id.base_id();
    let ext = game_builder.builder_extension_mut::<MachinesBuilderExt>();
    let block_types = ext.block_types.as_mut().context("pre_run already called")?;
    if block_types.contains_key(&base_id) {
        bail!("Machine already registered for base ID {:?}", base_id)
    }
    block_types.insert(base_id, def.into());
    Ok(())
}

const DIG_UP: StaticBlockName = StaticBlockName("autobuild:machine_dig_up");
const DIG_DOWN: StaticBlockName = StaticBlockName("autobuild:machine_dig_down");
const DIG_FACING: StaticBlockName = StaticBlockName("autobuild:machine_dig_facing");

const PLACE_UP: StaticBlockName = StaticBlockName("autobuild:machine_place_up");
const PLACE_DOWN: StaticBlockName = StaticBlockName("autobuild:machine_place_down");
const PLACE_FACING: StaticBlockName = StaticBlockName("autobuild:machine_place_facing");

const MOVE_ONE: StaticBlockName = StaticBlockName("autobuild:machine_move_one");
const MANUAL_TRIGGER: StaticBlockName = StaticBlockName("autobuild:machine_manual_trigger");

pub mod base_textures {
    use crate::game_builder::StaticTextureName;

    pub const ARROW_POINTING_UP: StaticTextureName =
        StaticTextureName("autobuild:machine_arrow_pointing_up");
    pub const FRONT: StaticTextureName = StaticTextureName("autobuild:machine_front");
    pub const FRONT_ORIENTED: StaticTextureName =
        StaticTextureName("autobuild:machine_front_oriented");
    pub const BACK: StaticTextureName = StaticTextureName("autobuild:machine_back");
    pub const BACK_ORIENTED: StaticTextureName =
        StaticTextureName("autobuild:machine_back_oriented");
}

/// Enables the machines functionality, and defines a few pre-made machines.
pub fn register_machines(game_builder: &mut GameBuilder) -> Result<()> {
    include_texture_bytes!(
        game_builder,
        base_textures::ARROW_POINTING_UP,
        "textures/machine_arrow_point_up.png"
    )?;
    include_texture_bytes!(
        game_builder,
        base_textures::FRONT,
        "textures/machine_front.png"
    )?;

    include_texture_bytes!(
        game_builder,
        base_textures::FRONT_ORIENTED,
        "textures/machine_front_oriented.png"
    )?;

    include_texture_bytes!(
        game_builder,
        base_textures::BACK,
        "textures/machine_back.png"
    )?;

    include_texture_bytes!(
        game_builder,
        base_textures::BACK_ORIENTED,
        "textures/machine_back_oriented.png"
    )?;

    let dig_up = BlockBuilder::new(DIG_UP)
        .set_display_name("Machine bit: dig up")
        .set_static_hover_text("Machine bit: dig up")
        .add_block_group(BRITTLE)
        .set_appearance(point_up_appearance(game_builder, Color::Red, false)?);
    register_machine_type(
        game_builder,
        add_interact_inventory(dig_up, "Machine bit: dig up".to_string(), false),
        DigFixedDeltaAction(0, 1, 0),
    )?;
    let dig_down = BlockBuilder::new(DIG_DOWN)
        .set_display_name("Machine bit: dig down")
        .set_static_hover_text("Machine bit: dig down")
        .add_block_group(BRITTLE)
        .set_appearance(point_down_appearance(game_builder, Color::Red, false)?);
    register_machine_type(
        game_builder,
        add_interact_inventory(dig_down, "Machine bit: dig down".to_string(), false),
        DigFixedDeltaAction(0, -1, 0),
    )?;
    let dig_facing = BlockBuilder::new(DIG_FACING)
        .set_display_name("Machine bit: dig horizontal")
        .set_static_hover_text("Machine bit: dig horizontal")
        .add_block_group(BRITTLE)
        .set_appearance(point_facing_appearance(game_builder, Color::Red)?);
    register_machine_type(
        game_builder,
        add_interact_inventory(dig_facing, "Machine bit: dig horizontal".to_string(), false),
        DigFacingDirectionAction,
    )?;

    let machine_back_lime =
        Color::colorize_to_texture(&Color::Lime, game_builder, base_textures::BACK)?;

    let manual_trigger = BlockBuilder::new(MANUAL_TRIGGER)
        .set_display_name("Machine bit: cycle start")
        .set_static_hover_text("Machine bit: cycle start")
        .add_block_group(BRITTLE)
        .set_appearance(
            CubeAppearanceBuilder::new()
                // trigger is not orientable.
                .set_single_texture(machine_back_lime)
                .into(),
        )
        .add_interact_key_menu_entry("start", "Start machine cycle")
        .add_modifier(|block| {
            block.interact_key_handler = Some(Box::new(|ctx, coord, _action| {
                let sense_out = trigger_machine_cycle(&ctx, coord)?;
                for error in sense_out.errors {
                    ctx.initiator()
                        .send_chat_message(ChatMessage::new_server_error(format!(
                            "Machine cycle error: {error}"
                        )))?;
                }
                Ok(None)
            }))
        });
    register_machine_type(game_builder, manual_trigger, DoNothingAction)?;

    let move_one = BlockBuilder::new(MOVE_ONE)
        .set_display_name("Machine bit: move one block")
        .set_static_hover_text("Machine bit: move one block")
        .add_block_group(BRITTLE)
        .set_appearance(point_facing_appearance(game_builder, Color::Teal)?);
    register_machine_type(game_builder, move_one, MoveOneAction)?;

    let place_up = BlockBuilder::new(PLACE_UP)
        .set_display_name("Machine bit: place up")
        .set_static_hover_text("Machine bit: place up")
        .add_block_group(BRITTLE)
        .set_appearance(point_up_appearance(game_builder, Color::Green, true)?);
    register_machine_type(
        game_builder,
        add_interact_inventory(place_up, "Machine bit: place up".to_string(), true),
        PlaceFixedDeltaAction(0, 1, 0),
    )?;
    let place_down = BlockBuilder::new(PLACE_DOWN)
        .set_display_name("Machine bit: place down")
        .set_static_hover_text("Machine bit: place down")
        .add_block_group(BRITTLE)
        .set_appearance(point_down_appearance(game_builder, Color::Green, true)?);
    register_machine_type(
        game_builder,
        add_interact_inventory(place_down, "Machine bit: place down".to_string(), true),
        PlaceFixedDeltaAction(0, -1, 0),
    )?;

    let place_facing = BlockBuilder::new(PLACE_FACING)
        .set_display_name("Machine bit: place horizontal")
        .set_static_hover_text("Machine bit: place horizontal")
        .add_block_group(BRITTLE)
        .set_appearance(point_facing_appearance(game_builder, Color::Green)?);
    register_machine_type(
        game_builder,
        add_interact_inventory(
            place_facing,
            "Machine bit: place horizontal".to_string(),
            true,
        ),
        PlaceFacingDirectionAction,
    )?;

    Ok(())
}

trait Apply: Sized {
    fn apply_if(self, cond: bool, f: impl FnOnce(Self) -> Self) -> Self {
        if cond {
            f(self)
        } else {
            self
        }
    }
}
impl<T: Sized> Apply for T {}

fn point_up_appearance(
    game_builder: &mut GameBuilder,
    color: Color,
    orientable: bool,
) -> Result<BlockAppearanceBuilder> {
    // These calls are memoized and won't double-register the texture even if point_up_appearance is called
    // multiple times
    let machine_side =
        Color::colorize_to_texture(&color, game_builder, base_textures::ARROW_POINTING_UP)?;
    let machine_front = Color::colorize_to_texture(
        &color,
        game_builder,
        if orientable {
            base_textures::FRONT_ORIENTED
        } else {
            base_textures::FRONT
        },
    )?;
    let machine_back = Color::colorize_to_texture(
        &color,
        game_builder,
        if orientable {
            base_textures::BACK_ORIENTED
        } else {
            base_textures::BACK
        },
    )?;

    Ok(CubeAppearanceBuilder::new()
        .set_individual_textures(
            &machine_side,
            &machine_side,
            &machine_front,
            &machine_back,
            &machine_side,
            &machine_side,
        )
        .apply_if(orientable, CubeAppearanceBuilder::set_rotate_laterally)
        .into())
}

fn point_down_appearance(
    game_builder: &mut GameBuilder,
    color: Color,
    orientable: bool,
) -> Result<BlockAppearanceBuilder> {
    let machine_side =
        Color::colorize_to_texture(&color, game_builder, base_textures::ARROW_POINTING_UP)?;
    let machine_front = Color::colorize_to_texture(
        &color,
        game_builder,
        if orientable {
            base_textures::FRONT_ORIENTED
        } else {
            base_textures::FRONT
        },
    )?;
    let machine_back = Color::colorize_to_texture(
        &color,
        game_builder,
        if orientable {
            base_textures::BACK_ORIENTED
        } else {
            base_textures::BACK
        },
    )?;
    Ok(CubeAppearanceBuilder::new()
        .set_individual_textures(
            machine_side.with_transform(TextureTransform::Rotate180),
            machine_side.with_transform(TextureTransform::Rotate180),
            &machine_back,
            &machine_front,
            machine_side.with_transform(TextureTransform::Rotate180),
            machine_side.with_transform(TextureTransform::Rotate180),
        )
        .apply_if(orientable, CubeAppearanceBuilder::set_rotate_laterally)
        .into())
}

fn point_facing_appearance(
    game_builder: &mut GameBuilder,
    color: Color,
) -> Result<BlockAppearanceBuilder> {
    let machine_side =
        Color::colorize_to_texture(&color, game_builder, base_textures::ARROW_POINTING_UP)?;
    let machine_front = Color::colorize_to_texture(&color, game_builder, base_textures::FRONT)?;
    let machine_back = Color::colorize_to_texture(&color, game_builder, base_textures::BACK)?;
    Ok(CubeAppearanceBuilder::new()
        .set_individual_textures(
            machine_side.with_transform(TextureTransform::RotateCounterClockwise),
            machine_side.with_transform(TextureTransform::RotateClockwise),
            machine_side.with_transform(TextureTransform::Rotate180),
            machine_side.with_transform(TextureTransform::Rotate180),
            // visual back and front are flipped - we want the machine to face the direction
            // the player faces - the sign flip in CompassDirection is also correct as a result
            &machine_back,
            &machine_front,
        )
        .set_rotate_laterally()
        .into())
}

pub struct DigFixedDeltaAction(i32, i32, i32);
impl MachineAction for DigFixedDeltaAction {
    fn act(&self, ctx: &HandlerContext, state: &ActionState) -> Result<ActionOutcome> {
        dig_at_delta(ctx, state, self.0, self.1, self.2)
    }
}

pub struct DigFacingDirectionAction;
impl MachineAction for DigFacingDirectionAction {
    fn act(&self, ctx: &HandlerContext, state: &ActionState) -> Result<ActionOutcome> {
        let (dx, dz) =
            CompassDirection::from_rotation_variant(state.machine_block_id.variant()).to_delta_xz();
        dig_at_delta(ctx, state, dx, 0, dz)
    }
}

pub struct PlaceFixedDeltaAction(i32, i32, i32);
impl MachineAction for PlaceFixedDeltaAction {
    fn act(&self, ctx: &HandlerContext, state: &ActionState) -> Result<ActionOutcome> {
        place_at_delta(ctx, state, self.0, self.1, self.2)
    }
}

pub struct PlaceFacingDirectionAction;
impl MachineAction for PlaceFacingDirectionAction {
    fn act(&self, ctx: &HandlerContext, state: &ActionState) -> Result<ActionOutcome> {
        let (dx, dz) =
            CompassDirection::from_rotation_variant(state.machine_block_id.variant()).to_delta_xz();
        place_at_delta(ctx, state, dx, 0, dz)
    }
}

pub struct CombinedAction<T: MachineAction, U: MachineAction>(T, U);
impl<T: MachineAction, U: MachineAction> MachineAction for CombinedAction<T, U> {
    fn sense(&self, ctx: &HandlerContext, state: &ActionState) -> Result<SenseInput> {
        let sense0 = self.0.sense(ctx, state)?;
        let sense1 = self.1.sense(ctx, state)?;
        if sense0.requested_movement.is_some() && sense1.requested_movement.is_some() {
            tracing::warn!("CombinedAction<{}, {}>: Both senses returned a requested movement, keeping the first.", std::any::type_name::<T>(), std::any::type_name::<U>());
        }
        Ok(SenseInput {
            requested_movement: sense0.requested_movement.or(sense1.requested_movement),
            _ne: NonExhaustive(()),
        })
    }

    fn act(&self, ctx: &HandlerContext, state: &ActionState) -> Result<ActionOutcome> {
        let mut outcome = ActionOutcome::default();
        outcome.extend(self.0.act(ctx, state)?);
        outcome.extend(self.1.act(ctx, state)?);
        Ok(outcome)
    }
}

/// A machine action that does nothing; useful for machine blocks that are only
/// used for triggering, or only form the superstructure of the machine to connect blocks
/// together
pub struct DoNothingAction;
impl MachineAction for DoNothingAction {
    fn act(&self, _ctx: &HandlerContext, _state: &ActionState) -> Result<ActionOutcome> {
        Ok(Default::default())
    }
}

/// A machine action that proposes moving one block in the facing direction
pub struct MoveOneAction;
impl MachineAction for MoveOneAction {
    fn sense(&self, _ctx: &HandlerContext, state: &ActionState) -> Result<SenseInput> {
        // See notes in the textures for the move_one block for notes about sign errors.
        // In short, the "front" face, facing the player, is actually the opposite of the
        // direction the player is facing.
        let compass_direction =
            CompassDirection::from_rotation_variant(state.machine_block_id.variant());
        let (dx, dz) = compass_direction.to_delta_xz();
        Ok(SenseInput {
            requested_movement: Some((dx, 0, dz)),
            ..Default::default()
        })
    }
    fn act(&self, _ctx: &HandlerContext, _state: &ActionState) -> Result<ActionOutcome> {
        Ok(Default::default())
    }
}

pub const MAX_MACHINE_BLOCKS: usize = 256;

pub fn trigger_machine_cycle(
    ctx: &HandlerContext,
    start_coord: BlockCoordinate,
) -> Result<SenseOutput> {
    let ctx = &ctx.with_plugin_initator("autobuild_machines");
    let extension = ctx
        .extension::<MachinesExt>()
        .context("Missing machines extension")?;
    // ye olde BFS
    let mut visit_queue = vec![start_coord];
    let mut visited = FxHashSet::default();
    let mut machines: FxHashMap<BlockCoordinate, (&MachineDef, BlockId)> = FxHashMap::default();
    while let Some(coord) = visit_queue.pop() {
        if visited.contains(&coord) {
            continue;
        }
        visited.insert(coord);
        let block = ctx.game_map().get_block(coord)?;
        if let Some(config) = extension.block_types.get(&block.base_id()) {
            if machines.len() >= MAX_MACHINE_BLOCKS {
                bail!("Machine is too large")
            }
            machines.insert(coord, (config, block));
            for (dx, dy, dz) in [
                (0, 0, 1),
                (0, 0, -1),
                (0, -1, 0),
                (0, 1, 0),
                (1, 0, 0),
                (-1, 0, 0),
            ] {
                if let Some(new_coord) = coord.try_delta(dx, dy, dz) {
                    visit_queue.push(new_coord);
                }
            }
        }
    }
    let mut sense = SenseOutput::default();
    for (&coord, (config, block)) in machines.iter() {
        let state = ActionState {
            machine_coord: coord,
            machine_block_id: *block,
            // todo
            movement_delta: (0, 0, 0),
            sense_data: &sense,
            _ne: NonExhaustive(()),
        };
        let this_sense = config.action.sense(ctx, &state)?;
        sense.merge_in(this_sense);
    }

    let mut action_outcome = ActionOutcome::default();
    for (&coord, (config, block)) in machines.iter() {
        let state = ActionState {
            machine_coord: coord,
            machine_block_id: *block,
            // todo
            movement_delta: (0, 0, 0),
            sense_data: &sense,
            _ne: NonExhaustive(()),
        };
        action_outcome.extend(config.action.act(ctx, &state)?);
    }
    sense.errors.extend(action_outcome.errors.into_iter());

    if let Movement::Some((dx, dy, dz)) = sense.movement {
        let origin = machines.keys().next().expect(
            "At least one machine matched, so machines must be non-empty, but it was empty (?!)",
        );
        let mut blocks_to_move = machines.keys().map(|c| (*c, *c - *origin)).collect_vec();
        blocks_to_move.sort_by_key(|(_, (mx, my, mz))| -(dx * mx + dy * my + dz * mz));

        let mut move_error: Option<String> = None;
        for (coord, _) in &blocks_to_move {
            if let Some(dst_coord) = coord.try_delta(dx, dy, dz) {
                if machines.contains_key(&dst_coord) {
                    continue;
                }
                let dst_block = ctx.game_map().get_block(dst_coord)?;
                if !ctx.block_types().is_trivially_replaceable(dst_block) {
                    move_error = Some(format!(
                        "A block is in the way of movement at {:?}",
                        dst_coord
                    ));
                    break;
                }
            }
        }
        if move_error.is_none() {
            for (src_coord, _) in &blocks_to_move {
                if let Some(dst_coord) = src_coord.try_delta(dx, dy, dz) {
                    // TODO: transactionality?
                    let (src_block, src_ext) = ctx
                        .game_map()
                        .get_block_with_extended_data(*src_coord, |e| Ok(Some(e.clone())))?;
                    if ctx
                        .game_map()
                        .mutate_block_atomically(dst_coord, move |block, ext| {
                            if !ctx.block_types().is_trivially_replaceable(*block) {
                                Ok(ControlFlow::Break(()))
                            } else {
                                *block = src_block;
                                *ext.deref_mut() = src_ext;
                                Ok(ControlFlow::Continue(()))
                            }
                        })?
                        .is_break()
                    {
                        move_error = Some(format!("mid-movement conflict at {:?}", dst_coord));
                        break;
                    };
                    ctx.game_map().set_block(*src_coord, AIR_ID, None)?;
                }
            }
        }
        if let Some(move_error) = move_error {
            sense.errors.insert(move_error);
        }
    }

    Ok(sense)
}

fn dig_at_delta(
    ctx: &HandlerContext<'_>,
    state: &ActionState,
    dx: i32,
    dy: i32,
    dz: i32,
) -> Result<ActionOutcome> {
    let Some(target) = state.machine_coord.try_delta(dx, dy, dz) else {
        // Out of bounds, not doing anything
        return Ok(ActionOutcome::from_error("Would dig out of bounds"));
    };
    let result = ctx.game_map().dig_block(
        target,
        ctx.initiator(),
        ctx.initiator_state(),
        Some(&DIG_ANY_SOLID_STACK),
    )?;
    let expected_id = state.machine_block_id;
    ctx.game_map()
        .mutate_block_atomically(state.machine_coord, move |id, ext| {
            if *id != expected_id {
                return Ok(ActionOutcome::from_error("machine disappared"));
            }
            let inv = ext
                .get_or_insert_default()
                .inventory_mut(MACHINE_INVENTORY_NAME.to_string(), MACHINE_INV_DEFAULT_SIZE);
            for stack in result.item_stacks {
                // drop the leftover stack
                let _ = inv.try_insert(stack);
            }
            Ok(ActionOutcome::default())
        })
}

fn place_at_delta(
    ctx: &HandlerContext<'_>,
    action_state: &ActionState,
    dx: i32,
    dy: i32,
    dz: i32,
) -> Result<ActionOutcome> {
    let azimuth = CompassDirection::from_rotation_variant(action_state.machine_block_id.variant())
        .to_azimuth();
    let ctx = &ctx.with_initiator_state_update(|state| {
        state.position = Some(PlayerPositionUpdate {
            position: action_state.machine_coord.into(),
            velocity: cgmath::Vector3::zero(),
            face_direction: (azimuth, 0.0),
        })
    });

    let Some(target) = action_state.machine_coord.try_delta(dx, dy, dz) else {
        // Out of bounds, not doing anything
        return Ok(ActionOutcome::from_error("Would place out of bounds"));
    };

    let first_stack =
        ctx.game_map()
            .mutate_block_atomically(action_state.machine_coord, |_block, ext| {
                let Some(inv) = ext
                    .get_or_insert_default()
                    .inventories
                    .get_mut(MACHINE_INVENTORY_NAME)
                else {
                    return Ok(None);
                };
                let Some(first_stack_idx) = inv.contents_mut().iter_mut().position(|x| x.is_some())
                else {
                    return Ok(None);
                };
                Ok(inv.contents_mut()[first_stack_idx].take())
            })?;
    let Some(first_stack) = first_stack else {
        return Ok(ActionOutcome::from_error("placer ran out of material"));
    };

    // TODO: override the facing direction
    let (reinsert, leftover, handler_result) = match ctx.item_manager().run_place_handler(
        first_stack.clone(),
        ctx,
        PointeeBlockCoords {
            selected: target,
            preceding: None,
        },
    ) {
        Ok(x) => (x.updated_stack, x.obtained_items, Result::Ok(())),
        Err(e) => (Some(first_stack), vec![], Result::Err(e)),
    };
    let expected_id = action_state.machine_block_id;
    let outcome =
        ctx.game_map()
            .mutate_block_atomically(action_state.machine_coord, move |id, ext| {
                if *id != expected_id {
                    return Ok(ActionOutcome::from_error("machine disappared"));
                }
                let inv = ext
                    .get_or_insert_default()
                    .inventory_mut(MACHINE_INVENTORY_NAME.to_string(), MACHINE_INV_DEFAULT_SIZE);
                if let Some(stack) = reinsert {
                    // drop the leftover stack
                    let _ = inv.try_insert(stack);
                }

                let leftover_inv = ext.get_or_insert_default().inventory_mut(
                    MACHINE_LEFTOVER_INVENTORY_NAME.to_string(),
                    MACHINE_INV_DEFAULT_SIZE,
                );
                for stack in leftover {
                    let _ = leftover_inv.try_insert(stack);
                }
                Ok(ActionOutcome::default())
            });
    handler_result?;
    outcome
}

pub const MACHINE_INVENTORY_NAME: &str = "machine_inv";
pub const MACHINE_LEFTOVER_INVENTORY_NAME: &str = "machine_leftover";
pub const MACHINE_INV_DEFAULT_SIZE: (u32, u32) = (2, 12);

fn add_interact_inventory(
    builder: BlockBuilder,
    title: String,
    includes_leftover: bool,
) -> BlockBuilder {
    builder.add_modifier(move |block| {
        block.interact_key_handler = Some(Box::new(move |ctx, coord, _action| {
            let popup = ctx.new_popup().title(title.clone());
            ctx.initiator().try_with_player(move |p| {
                let popup = popup.inventory_view_block(
                    MACHINE_INVENTORY_NAME.to_string(),
                    "Contents:",
                    MACHINE_INV_DEFAULT_SIZE,
                    coord,
                    MACHINE_INVENTORY_NAME.to_string(),
                    true,
                    true,
                    false,
                )?;
                let popup = if includes_leftover {
                    popup.inventory_view_block(
                        MACHINE_LEFTOVER_INVENTORY_NAME.to_string(),
                        "Leftover:",
                        MACHINE_INV_DEFAULT_SIZE,
                        coord,
                        MACHINE_LEFTOVER_INVENTORY_NAME.to_string(),
                        true,
                        true,
                        false,
                    )?
                } else {
                    popup
                };
                let popup = popup.inventory_view_stored(
                    "player_inv",
                    "Player inventory:",
                    p.main_inventory(),
                    true,
                    true,
                )?;
                Ok(popup)
            })
        }))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        default_game::basic_blocks::DIRT,
        test_support::{GameBuilderTestExt, IsBlock, TestFixture},
    };
    use googletest::prelude::*;
    use perovskite_core::{block_id::special_block_defs::AIR_ID, coordinates::BlockCoordinate};

    // All machines are placed at positive Y to avoid mapgen interference.
    const MACHINE_COORD: BlockCoordinate = BlockCoordinate::new(0, 16, 0);

    fn start(fixture: &TestFixture) -> googletest::Result<()> {
        fixture.start_server(|builder| {
            crate::configure_default_game(builder)?;
            // Override the default mapgen with all-air so machine tests don't
            // collide with naturally-generated terrain.
            builder.set_flatland_mapgen(perovskite_core::block_id::BlockId::AIR);
            Ok(())
        })
    }

    /// Pre-load a stack of items into a machine's input inventory.
    fn load_machine_inventory(
        fixture: &TestFixture,
        machine: BlockCoordinate,
        item_name: &'static str,
        quantity: u32,
    ) -> googletest::Result<()> {
        fixture.run_with_context(|ctx| {
            let stack = ctx
                .item_manager()
                .get_item(item_name)
                .expect("item not registered")
                .make_stack(quantity);
            ctx.game_map()
                .mutate_block_atomically(machine, |_block, ext| {
                    let inv = ext
                        .get_or_insert_default()
                        .inventory_mut(MACHINE_INVENTORY_NAME.to_string(), MACHINE_INV_DEFAULT_SIZE);
                    let _ = inv.try_insert(stack.clone());
                    Ok(())
                })
                .or_fail()?;
            Ok(())
        })
    }

    /// Count items by name in a machine's input inventory.
    fn count_in_machine_inventory(
        fixture: &TestFixture,
        machine: BlockCoordinate,
        item_name: &str,
    ) -> googletest::Result<u32> {
        let mut total = 0u32;
        fixture.run_with_context(|ctx| {
            let (_, count) = ctx
                .game_map()
                .get_block_with_extended_data(machine, |ext| {
                    let n = ext
                        .inventories
                        .get(MACHINE_INVENTORY_NAME)
                        .map(|inv| {
                            inv.contents()
                                .iter()
                                .flatten()
                                .filter(|s| s.proto.item_name == item_name)
                                .map(|s| s.proto.quantity)
                                .sum::<u32>()
                        })
                        .unwrap_or(0);
                    Ok(Some(n))
                })
                .or_fail()?;
            total = count.unwrap_or(0);
            Ok(())
        })?;
        Ok(total)
    }

    // ────────────────────────────────────────────────────────────────────────
    // Dig machines
    // ────────────────────────────────────────────────────────────────────────

    #[gtest]
    fn test_dig_up_removes_block_above(fixture: &TestFixture) -> googletest::Result<()> {
        start(fixture)?;

        let target = MACHINE_COORD.try_delta(0, 1, 0).unwrap();

        fixture.run_with_context(|ctx| {
            let dig_up_id = ctx.block_types().get_by_name(DIG_UP.0).expect("DIG_UP not registered");
            let dirt_id = ctx.block_types().get_by_name(DIRT.0).expect("DIRT not registered");
            ctx.game_map().set_block(MACHINE_COORD, dig_up_id, None).or_fail()?;
            ctx.game_map().set_block(target, dirt_id, None).or_fail()?;
            Ok(())
        })?;

        fixture.run_with_context(|ctx| {
            trigger_machine_cycle(&ctx, MACHINE_COORD).or_fail()?;
            Ok(())
        })?;

        fixture.run_with_context(|ctx| {
            expect_that!(ctx.game_map().get_block(target).or_fail()?, IsBlock(AIR_ID));
            Ok(())
        })
    }

    #[gtest]
    fn test_dig_down_removes_block_below(fixture: &TestFixture) -> googletest::Result<()> {
        start(fixture)?;

        let target = MACHINE_COORD.try_delta(0, -1, 0).unwrap();

        fixture.run_with_context(|ctx| {
            let dig_down_id =
                ctx.block_types().get_by_name(DIG_DOWN.0).expect("DIG_DOWN not registered");
            let dirt_id = ctx.block_types().get_by_name(DIRT.0).expect("DIRT not registered");
            ctx.game_map().set_block(MACHINE_COORD, dig_down_id, None).or_fail()?;
            ctx.game_map().set_block(target, dirt_id, None).or_fail()?;
            Ok(())
        })?;

        fixture.run_with_context(|ctx| {
            trigger_machine_cycle(&ctx, MACHINE_COORD).or_fail()?;
            Ok(())
        })?;

        fixture.run_with_context(|ctx| {
            expect_that!(ctx.game_map().get_block(target).or_fail()?, IsBlock(AIR_ID));
            Ok(())
        })
    }

    #[gtest]
    fn test_dig_facing_removes_block_ahead(fixture: &TestFixture) -> googletest::Result<()> {
        start(fixture)?;

        // variant 0 = ZPlus: machine digs one block in the +Z direction
        let variant = perovskite_server::game_state::blocks::CompassDirection::ZPlus.to_variant();
        let target = MACHINE_COORD.try_delta(0, 0, 1).unwrap();

        fixture.run_with_context(|ctx| {
            let base_id =
                ctx.block_types().get_by_name(DIG_FACING.0).expect("DIG_FACING not registered");
            let dig_facing_id = base_id.with_variant_unchecked(variant);
            let dirt_id = ctx.block_types().get_by_name(DIRT.0).expect("DIRT not registered");
            ctx.game_map().set_block(MACHINE_COORD, dig_facing_id, None).or_fail()?;
            ctx.game_map().set_block(target, dirt_id, None).or_fail()?;
            Ok(())
        })?;

        fixture.run_with_context(|ctx| {
            trigger_machine_cycle(&ctx, MACHINE_COORD).or_fail()?;
            Ok(())
        })?;

        fixture.run_with_context(|ctx| {
            expect_that!(ctx.game_map().get_block(target).or_fail()?, IsBlock(AIR_ID));
            Ok(())
        })
    }

    #[gtest]
    fn test_dig_stores_drops_in_machine_inventory(fixture: &TestFixture) -> googletest::Result<()> {
        start(fixture)?;

        let target = MACHINE_COORD.try_delta(0, 1, 0).unwrap();

        fixture.run_with_context(|ctx| {
            let dig_up_id = ctx.block_types().get_by_name(DIG_UP.0).expect("DIG_UP not registered");
            let dirt_id = ctx.block_types().get_by_name(DIRT.0).expect("DIRT not registered");
            ctx.game_map().set_block(MACHINE_COORD, dig_up_id, None).or_fail()?;
            ctx.game_map().set_block(target, dirt_id, None).or_fail()?;
            Ok(())
        })?;

        fixture.run_with_context(|ctx| {
            trigger_machine_cycle(&ctx, MACHINE_COORD).or_fail()?;
            Ok(())
        })?;

        let dirt_in_inv = count_in_machine_inventory(fixture, MACHINE_COORD, DIRT.0)?;
        expect_that!(dirt_in_inv, gt(0u32));

        Ok(())
    }

    // ────────────────────────────────────────────────────────────────────────
    // Place machines
    // ────────────────────────────────────────────────────────────────────────

    #[gtest]
    fn test_place_up_places_block_above(fixture: &TestFixture) -> googletest::Result<()> {
        start(fixture)?;

        let target = MACHINE_COORD.try_delta(0, 1, 0).unwrap();

        fixture.run_with_context(|ctx| {
            let place_up_id =
                ctx.block_types().get_by_name(PLACE_UP.0).expect("PLACE_UP not registered");
            ctx.game_map().set_block(MACHINE_COORD, place_up_id, None).or_fail()?;
            ctx.game_map().set_block(target, AIR_ID, None).or_fail()?;
            Ok(())
        })?;
        load_machine_inventory(fixture, MACHINE_COORD, DIRT.0, 5)?;

        fixture.run_with_context(|ctx| {
            let result = trigger_machine_cycle(&ctx, MACHINE_COORD).or_fail()?;
            expect_that!(result.errors, is_empty());
            Ok(())
        })?;

        fixture.run_with_context(|ctx| {
            expect_that!(ctx.game_map().get_block(target).or_fail()?, IsBlock(DIRT));
            Ok(())
        })
    }

    #[gtest]
    fn test_place_down_places_block_below(fixture: &TestFixture) -> googletest::Result<()> {
        start(fixture)?;

        let target = MACHINE_COORD.try_delta(0, -1, 0).unwrap();

        fixture.run_with_context(|ctx| {
            let place_down_id =
                ctx.block_types().get_by_name(PLACE_DOWN.0).expect("PLACE_DOWN not registered");
            ctx.game_map().set_block(MACHINE_COORD, place_down_id, None).or_fail()?;
            ctx.game_map().set_block(target, AIR_ID, None).or_fail()?;
            Ok(())
        })?;
        load_machine_inventory(fixture, MACHINE_COORD, DIRT.0, 5)?;

        fixture.run_with_context(|ctx| {
            let result = trigger_machine_cycle(&ctx, MACHINE_COORD).or_fail()?;
            expect_that!(result.errors, is_empty());
            Ok(())
        })?;

        fixture.run_with_context(|ctx| {
            expect_that!(ctx.game_map().get_block(target).or_fail()?, IsBlock(DIRT));
            Ok(())
        })
    }

    #[gtest]
    fn test_place_facing_places_block_ahead(fixture: &TestFixture) -> googletest::Result<()> {
        start(fixture)?;

        // variant for ZPlus: machine places one block in the +Z direction
        let variant = perovskite_server::game_state::blocks::CompassDirection::ZPlus.to_variant();
        let target = MACHINE_COORD.try_delta(0, 0, 1).unwrap();

        fixture.run_with_context(|ctx| {
            let base_id =
                ctx.block_types().get_by_name(PLACE_FACING.0).expect("PLACE_FACING not registered");
            let place_facing_id = base_id.with_variant_unchecked(variant);
            ctx.game_map().set_block(MACHINE_COORD, place_facing_id, None).or_fail()?;
            ctx.game_map().set_block(target, AIR_ID, None).or_fail()?;
            Ok(())
        })?;
        load_machine_inventory(fixture, MACHINE_COORD, DIRT.0, 5)?;

        fixture.run_with_context(|ctx| {
            let result = trigger_machine_cycle(&ctx, MACHINE_COORD).or_fail()?;
            expect_that!(result.errors, is_empty());
            Ok(())
        })?;

        fixture.run_with_context(|ctx| {
            expect_that!(ctx.game_map().get_block(target).or_fail()?, IsBlock(DIRT));
            Ok(())
        })
    }

    #[gtest]
    fn test_place_out_of_material_reports_error(fixture: &TestFixture) -> googletest::Result<()> {
        start(fixture)?;

        fixture.run_with_context(|ctx| {
            let place_up_id =
                ctx.block_types().get_by_name(PLACE_UP.0).expect("PLACE_UP not registered");
            ctx.game_map().set_block(MACHINE_COORD, place_up_id, None).or_fail()?;
            Ok(())
        })?;
        // inventory intentionally left empty

        fixture.run_with_context(|ctx| {
            let result = trigger_machine_cycle(&ctx, MACHINE_COORD).or_fail()?;
            expect_that!(result.errors, not(is_empty()));
            Ok(())
        })
    }

    #[gtest]
    fn test_place_depletes_machine_inventory(fixture: &TestFixture) -> googletest::Result<()> {
        start(fixture)?;

        let target = MACHINE_COORD.try_delta(0, 1, 0).unwrap();

        fixture.run_with_context(|ctx| {
            let place_up_id =
                ctx.block_types().get_by_name(PLACE_UP.0).expect("PLACE_UP not registered");
            ctx.game_map().set_block(MACHINE_COORD, place_up_id, None).or_fail()?;
            ctx.game_map().set_block(target, AIR_ID, None).or_fail()?;
            Ok(())
        })?;
        load_machine_inventory(fixture, MACHINE_COORD, DIRT.0, 3)?;

        let before = count_in_machine_inventory(fixture, MACHINE_COORD, DIRT.0)?;

        fixture.run_with_context(|ctx| {
            trigger_machine_cycle(&ctx, MACHINE_COORD).or_fail()?;
            Ok(())
        })?;

        let after = count_in_machine_inventory(fixture, MACHINE_COORD, DIRT.0)?;
        expect_that!(after, lt(before));

        Ok(())
    }

    // ────────────────────────────────────────────────────────────────────────
    // Move machine
    // ────────────────────────────────────────────────────────────────────────

    #[gtest]
    fn test_move_one_moves_machine(fixture: &TestFixture) -> googletest::Result<()> {
        start(fixture)?;

        // Layout: MANUAL_TRIGGER at (0,16,0), MOVE_ONE at (0,16,1) facing +Z.
        // After one cycle the whole machine should shift +1 in Z.
        let trigger_coord = MACHINE_COORD;
        let mover_coord = MACHINE_COORD.try_delta(0, 0, 1).unwrap();
        let z_plus = perovskite_server::game_state::blocks::CompassDirection::ZPlus.to_variant();

        fixture.run_with_context(|ctx| {
            let trigger_id = ctx
                .block_types()
                .get_by_name(MANUAL_TRIGGER.0)
                .expect("MANUAL_TRIGGER not registered");
            let mover_base =
                ctx.block_types().get_by_name(MOVE_ONE.0).expect("MOVE_ONE not registered");
            let mover_id = mover_base.with_variant_unchecked(z_plus);
            ctx.game_map().set_block(trigger_coord, trigger_id, None).or_fail()?;
            ctx.game_map().set_block(mover_coord, mover_id, None).or_fail()?;
            // Only the leading edge — the cell the mover vacates last — needs to be free.
            // The trigger's destination (mover_coord) is occupied by the mover itself, so the
            // movement code skips that check automatically.
            let leading_edge = mover_coord.try_delta(0, 0, 1).unwrap();
            ctx.game_map().set_block(leading_edge, AIR_ID, None).or_fail()?;
            Ok(())
        })?;

        fixture.run_with_context(|ctx| {
            let result = trigger_machine_cycle(&ctx, trigger_coord).or_fail()?;
            expect_that!(result.errors, is_empty());
            Ok(())
        })?;

        fixture.run_with_context(|ctx| {
            // Original positions are now air.
            expect_that!(ctx.game_map().get_block(trigger_coord).or_fail()?, IsBlock(AIR_ID));
            expect_that!(ctx.game_map().get_block(mover_coord).or_fail()?, IsBlock(MANUAL_TRIGGER));
            let new_mover = mover_coord.try_delta(0, 0, 1).unwrap();
            expect_that!(ctx.game_map().get_block(new_mover).or_fail()?, IsBlock(MOVE_ONE));
            Ok(())
        })
    }

    // ────────────────────────────────────────────────────────────────────────
    // Manual trigger block (interact key handler)
    // ────────────────────────────────────────────────────────────────────────

    #[gtest]
    fn test_manual_trigger_interact_key_activates_cycle(
        fixture: &TestFixture,
    ) -> googletest::Result<()> {
        start(fixture)?;

        // A DIG_UP adjacent to the trigger so we can verify the cycle ran.
        let trigger_coord = MACHINE_COORD;
        let digger_coord = MACHINE_COORD.try_delta(1, 0, 0).unwrap();
        let target_coord = digger_coord.try_delta(0, 1, 0).unwrap();

        fixture.run_with_context(|ctx| {
            let trigger_id = ctx
                .block_types()
                .get_by_name(MANUAL_TRIGGER.0)
                .expect("MANUAL_TRIGGER not registered");
            let dig_up_id = ctx.block_types().get_by_name(DIG_UP.0).expect("DIG_UP not registered");
            let dirt_id = ctx.block_types().get_by_name(DIRT.0).expect("DIRT not registered");
            ctx.game_map().set_block(trigger_coord, trigger_id, None).or_fail()?;
            ctx.game_map().set_block(digger_coord, dig_up_id, None).or_fail()?;
            ctx.game_map().set_block(target_coord, dirt_id, None).or_fail()?;
            Ok(())
        })?;

        // Call the interact key handler directly instead of trigger_machine_cycle.
        fixture.run_with_context(|ctx| {
            let trigger_id = ctx
                .block_types()
                .get_by_name(MANUAL_TRIGGER.0)
                .expect("MANUAL_TRIGGER not registered");
            let (block_type, _) = ctx.block_types().get_block(trigger_id).or_fail()?;
            let handler = block_type
                .interact_key_handler
                .as_ref()
                .expect("MANUAL_TRIGGER must have an interact key handler");
            handler(ctx.clone(), trigger_coord, "start").or_fail()?;
            Ok(())
        })?;

        fixture.run_with_context(|ctx| {
            expect_that!(ctx.game_map().get_block(target_coord).or_fail()?, IsBlock(AIR_ID));
            Ok(())
        })
    }
}
