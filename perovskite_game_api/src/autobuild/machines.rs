//! Autobuild machines: autobuild, but made of blocks with tasks.
//!
//! A player can assemble a handful of blocks, each with a task
//! (e.g. "place carts:rail_tile", "if non-air replace with tunnel_wall",
//! "if air, replace with bridge_deck", "every 32, place track gantry"),
//! almost jacquard loom style config, that you can set up and leave running.
use std::{
    collections::{BTreeMap, HashSet},
    ops::{ControlFlow, DerefMut},
};

use anyhow::{bail, Context, Result};
use itertools::Itertools;
use perovskite_core::{
    block_id::{special_block_defs::AIR_ID, BlockId},
    chat::ChatMessage,
    coordinates::BlockCoordinate,
};
use perovskite_server::game_state::{
    blocks::CompassDirection, client_ui::UiElementContainer, event::HandlerContext,
    items::DIG_ANY_SOLID_STACK, GameStateExtension,
};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    blocks::{BlockBuilder, CubeAppearanceBuilder},
    colors::Color,
    default_game::block_groups::BRITTLE,
    game_builder::{GameBuilder, GameBuilderExtension, StaticBlockName},
    include_texture_bytes,
};

pub const AUTOBUILD_MACHINES_GROUP: &str = "autobuild:machines";

#[derive(Clone, Debug)]
#[non_exhaustive]
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
}

/// Data that machine blocks produce during sensing, and that the framework will aggregate
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SenseInput {
    /// How far the machine should move. At most one block can report this, or the system will refuse to move.
    requested_movement: Option<(i32, i32, i32)>,
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
#[non_exhaustive]
pub struct SenseOutput {
    pub movement: Movement,
    pub errors: HashSet<String>,
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
        }
    }
}

impl Default for SenseOutput {
    fn default() -> Self {
        Self {
            movement: Movement::None,
            errors: HashSet::new(),
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
    fn act(&self, ctx: &HandlerContext, state: &ActionState) -> Result<()>;
}

/// Definition for the behavior of a machine.
///
/// Construct with [Into::into] (in the future, there may be parameters in the
/// MachineDef that can be customized)
#[non_exhaustive]
pub struct MachineDef {
    pub action: Box<dyn MachineAction + Send + Sync + 'static>,
}
impl<T: MachineAction + Send + Sync + 'static> From<T> for MachineDef {
    fn from(value: T) -> Self {
        MachineDef {
            action: Box::new(value),
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

const DIG_UP: StaticBlockName = StaticBlockName("autobuild:machine_dig_above");
const MOVE_ONE: StaticBlockName = StaticBlockName("autobuild:machine_move_one");
const MANUAL_TRIGGER: StaticBlockName = StaticBlockName("autobuild:machine:manual_trigger");

pub mod base_textures {
    use crate::game_builder::StaticTextureName;

    pub const SIDE_POINTING_UP: StaticTextureName =
        StaticTextureName("autobuild:machine_side_pointing_up");
    pub const SIDE: StaticTextureName = StaticTextureName("autobuild:machine_side");
    pub const FRONT: StaticTextureName = StaticTextureName("autobuild:machine_front");
    pub const BACK: StaticTextureName = StaticTextureName("autobuild:machine_back");
}

/// Enables the machines functionality, and defines a few pre-made machines.
pub fn register_machines(game_builder: &mut GameBuilder) -> Result<()> {
    include_texture_bytes!(
        game_builder,
        base_textures::SIDE_POINTING_UP,
        "textures/machine_side_point_up.png"
    )?;
    include_texture_bytes!(
        game_builder,
        base_textures::SIDE,
        "textures/machine_side.png"
    )?;
    include_texture_bytes!(
        game_builder,
        base_textures::FRONT,
        "textures/machine_front.png"
    )?;
    include_texture_bytes!(
        game_builder,
        base_textures::BACK,
        "textures/machine_back.png"
    )?;

    let machine_side_up_red =
        Color::colorize_to_texture(&Color::Red, game_builder, base_textures::SIDE_POINTING_UP)?;

    let machine_front_red =
        Color::colorize_to_texture(&Color::Red, game_builder, base_textures::FRONT)?;

    let machine_back_red =
        Color::colorize_to_texture(&Color::Red, game_builder, base_textures::BACK)?;
    let dig_up = BlockBuilder::new(DIG_UP)
        .set_display_name("Machine bit: dig up")
        .set_static_hover_text("Machine bit: dig up")
        .add_block_group(BRITTLE)
        .set_appearance(
            CubeAppearanceBuilder::new()
                // TODO apply all needed textures
                .set_individual_textures(
                    &machine_side_up_red,
                    &machine_side_up_red,
                    &machine_front_red,
                    &machine_back_red,
                    &machine_side_up_red,
                    &machine_side_up_red,
                )
                .into(),
        );
    register_machine_type(
        game_builder,
        add_interact_inventory(
            dig_up,
            "Machine bit: dig up".to_string(),
            MACHINE_INV_DEFAULT_SIZE,
        ),
        DigUpAction,
    )?;

    let machine_back_green =
        Color::colorize_to_texture(&Color::Green, game_builder, base_textures::BACK)?;

    let manual_trigger = BlockBuilder::new(MANUAL_TRIGGER)
        .set_display_name("Machine bit: cycle start")
        .set_static_hover_text("Machine bit: cycle start")
        .add_block_group(BRITTLE)
        .set_appearance(
            CubeAppearanceBuilder::new()
                // trigger is not orientable
                .set_single_texture(machine_back_green)
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

    let machine_side_teal =
        Color::colorize_to_texture(&Color::Teal, game_builder, base_textures::SIDE)?;

    let machine_front_teal =
        Color::colorize_to_texture(&Color::Teal, game_builder, base_textures::FRONT)?;

    let machine_back_teal =
        Color::colorize_to_texture(&Color::Teal, game_builder, base_textures::BACK)?;
    let move_one = BlockBuilder::new(MOVE_ONE)
        .set_display_name("Machine bit: move one")
        .set_static_hover_text("Machine bit: move one")
        .add_block_group(BRITTLE)
        .set_appearance(
            CubeAppearanceBuilder::new()
                // TODO apply all needed textures
                // TODO need texture flips...
                .set_individual_textures(
                    &machine_side_teal,
                    &machine_side_teal,
                    &machine_side_teal,
                    &machine_side_teal,
                    // visual back and front are flipped - we want the machine to face the direction
                    // the player faces - the sign flip in CompassDirection is also correct as a result
                    &machine_back_teal,
                    &machine_front_teal,
                )
                .set_rotate_laterally()
                .into(),
        );
    register_machine_type(game_builder, move_one, MoveOneAction)?;

    Ok(())
}

struct DigUpAction;
impl MachineAction for DigUpAction {
    fn act(&self, ctx: &HandlerContext, state: &ActionState) -> Result<()> {
        dig_at_delta(ctx, state, 0, 1, 0)
    }
}

/// A machine action that does nothing; useful for machine blocks that are only
/// used for triggering, or only form the superstructure of the machine to connect blocks
/// together
pub struct DoNothingAction;
impl MachineAction for DoNothingAction {
    fn act(&self, _ctx: &HandlerContext, _state: &ActionState) -> Result<()> {
        Ok(())
    }
}

/// A machine action that proposes moving one block in the facing direction
pub struct MoveOneAction;
impl MachineAction for MoveOneAction {
    fn sense(&self, _ctx: &HandlerContext, state: &ActionState) -> Result<SenseInput> {
        let compass_direction =
            CompassDirection::from_rotation_variant(state.machine_block_id.variant());
        let (dx, dz) = compass_direction.to_delta_xz();
        Ok(SenseInput {
            requested_movement: Some((dx, 0, dz)),
            ..Default::default()
        })
    }
    fn act(&self, _ctx: &HandlerContext, _state: &ActionState) -> Result<()> {
        Ok(())
    }
}

pub const MAX_MACHINE_BLOCKS: usize = 256;

pub fn trigger_machine_cycle(
    ctx: &HandlerContext,
    start_coord: BlockCoordinate,
) -> Result<SenseOutput> {
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
        };
        let this_sense = config.action.sense(ctx, &state)?;
        sense.merge_in(this_sense);
    }

    for (&coord, (config, block)) in machines.iter() {
        let state = ActionState {
            machine_coord: coord,
            machine_block_id: *block,
            // todo
            movement_delta: (0, 0, 0),
            sense_data: &sense,
        };
        config.action.act(ctx, &state)?;
    }

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
) -> Result<()> {
    let Some(target) = state.machine_coord.try_delta(dx, dy, dz) else {
        // Out of bounds, not doing anything
        return Ok(());
    };
    let result = ctx
        .game_map()
        .dig_block(target, ctx.initiator(), Some(&DIG_ANY_SOLID_STACK))?;
    let expected_id = state.machine_block_id;
    ctx.game_map()
        .mutate_block_atomically(state.machine_coord, move |id, ext| {
            if *id != expected_id {
                return Ok(());
            }
            let inv = ext
                .get_or_insert_default()
                .inventory_mut(MACHINE_INVENTORY_NAME.to_string(), MACHINE_INV_DEFAULT_SIZE);
            for stack in result.item_stacks {
                // drop the leftover stack
                let _ = inv.try_insert(stack);
            }
            Ok(())
        })
}

pub const MACHINE_INVENTORY_NAME: &str = "machine_inv";
pub const MACHINE_INV_DEFAULT_SIZE: (u32, u32) = (2, 12);

fn add_interact_inventory(
    builder: BlockBuilder,
    title: String,
    dimension: (u32, u32),
) -> BlockBuilder {
    builder.add_modifier(move |block| {
        block.interact_key_handler = Some(Box::new(move |ctx, coord, _action| {
            let popup = ctx.new_popup().title(title.clone());
            ctx.initiator().try_with_player(move |p| {
                Ok(popup
                    .inventory_view_block(
                        MACHINE_INVENTORY_NAME.to_string(),
                        "Contents:",
                        dimension,
                        coord,
                        MACHINE_INVENTORY_NAME.to_string(),
                        true,
                        true,
                        false,
                    )?
                    .inventory_view_stored(
                        "player_inv",
                        "Player inventory:",
                        p.main_inventory(),
                        true,
                        true,
                    )?)
            })
        }))
    })
}
