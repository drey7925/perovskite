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
    client_ui::{ButtonCallbackExt, RefinementType, TextFieldBuilder, UiElementContainer},
    event::HandlerContext,
    items::{ItemStack, PointeeBlockCoords, DIG_ANY_SOLID_STACK},
    GameStateExtension,
};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    autobuild::machines::item_priorities::DIGGER_PRIORITY,
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

#[derive(Clone, Debug)]
pub struct ExpectedConsumption {
    /// The item we want. Must be set, default value is empty and nonsensical
    pub item_name: String,
    /// The inventory to put it into, if available. Must be set; default is empty and nonsensical.
    pub inventory_name: String,
    /// quantity requested; may fill partially. Default value is 1.
    pub requested_quantity: u32,
    pub _ne: NonExhaustive,
}
impl Default for ExpectedConsumption {
    fn default() -> Self {
        Self {
            item_name: String::new(),
            inventory_name: String::new(),
            requested_quantity: 1,
            _ne: NonExhaustive(()),
        }
    }
}

/// Data that machine blocks produce during sensing, and that the framework will aggregate
#[derive(Clone, Debug)]
pub struct SenseInput {
    /// How far the machine should move. At most one block can report this, or the system will refuse to move.
    pub requested_movement: Option<(i32, i32, i32)>,
    /// The machine expects to consume this material from its inventory (approximate, because it doesn't know
    /// whether placing will succeed or whether the place handler actually consumes the item)
    pub expects_to_consume: Vec<ExpectedConsumption>,
    /// All inventories that include waste/leftover items from this machine. If we have a redistributor, it'll
    /// try to deliver these to any expects_to_consume requests for items. Number is a priority, highest values
    /// fulfill consumption requests earlier.
    pub available_for_take: Vec<(String, i32)>,
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
    /// How far the machine is supposed to move.
    pub movement: Movement,

    /// All inventories that the machine expects to consume items from.
    pub expects_to_consume: Vec<(BlockCoordinate, ExpectedConsumption)>,

    /// All inventories that the machine has available for taking items from. This is guaranteed to be
    /// sorted by priority from highest to lowest.
    pub available_for_take: Vec<(BlockCoordinate, String, i32)>,
    pub errors: HashSet<String>,
    pub _ne: NonExhaustive,
}

impl SenseOutput {
    fn merge_in(&mut self, other: SenseInput, coord: BlockCoordinate) {
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
        for consumption in other.expects_to_consume {
            self.expects_to_consume.push((coord, consumption));
        }
        for (inv, prio) in other.available_for_take {
            self.available_for_take.push((coord, inv, prio));
        }
    }
}

impl Default for SenseInput {
    fn default() -> Self {
        Self {
            requested_movement: None,
            expects_to_consume: Vec::new(),
            available_for_take: Vec::new(),
            _ne: NonExhaustive(()),
        }
    }
}

impl Default for SenseOutput {
    fn default() -> Self {
        Self {
            movement: Movement::None,
            expects_to_consume: Vec::new(),
            available_for_take: Vec::new(),
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
const MOVE_UP: StaticBlockName = StaticBlockName("autobuild:machine_move_up");
const MOVE_DOWN: StaticBlockName = StaticBlockName("autobuild:machine_move_down");
const MANUAL_TRIGGER: StaticBlockName = StaticBlockName("autobuild:machine_manual_trigger");
const DISTRIBUTOR: StaticBlockName = StaticBlockName("autobuild:machine_distributor");

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
        add_interact_inventory(
            dig_up,
            "Machine bit: dig up".to_string(),
            InteractInventoryConfig::default(),
        ),
        DigFixedDeltaAction(0, 1, 0),
    )?;
    let dig_down = BlockBuilder::new(DIG_DOWN)
        .set_display_name("Machine bit: dig down")
        .set_static_hover_text("Machine bit: dig down")
        .add_block_group(BRITTLE)
        .set_appearance(point_down_appearance(game_builder, Color::Red, false)?);
    register_machine_type(
        game_builder,
        add_interact_inventory(
            dig_down,
            "Machine bit: dig down".to_string(),
            InteractInventoryConfig::default(),
        ),
        DigFixedDeltaAction(0, -1, 0),
    )?;
    let dig_facing = BlockBuilder::new(DIG_FACING)
        .set_display_name("Machine bit: dig horizontal")
        .set_static_hover_text("Machine bit: dig horizontal")
        .add_block_group(BRITTLE)
        .set_appearance(point_facing_appearance(game_builder, Color::Red)?);
    register_machine_type(
        game_builder,
        add_interact_inventory(
            dig_facing,
            "Machine bit: dig horizontal".to_string(),
            InteractInventoryConfig::default(),
        ),
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

    let machine_back_purple =
        Color::colorize_to_texture(&Color::Purple, game_builder, base_textures::BACK)?;
    let distributor = BlockBuilder::new(DISTRIBUTOR)
        .set_display_name("Machine bit: distributor")
        .set_static_hover_text("Machine bit: distributor")
        .add_block_group(BRITTLE)
        .set_appearance(
            CubeAppearanceBuilder::new()
                // distributor is not orientable.
                .set_single_texture(machine_back_purple)
                .into(),
        );
    register_machine_type(
        game_builder,
        add_interact_inventory(
            distributor,
            "Distributor".to_string(),
            InteractInventoryConfig::default(),
        ),
        DistributorAction,
    )?;

    let move_one = BlockBuilder::new(MOVE_ONE)
        .set_display_name("Machine bit: move one block")
        .set_static_hover_text("Machine bit: move one block")
        .add_block_group(BRITTLE)
        .set_appearance(point_facing_appearance(game_builder, Color::Teal)?);
    register_machine_type(game_builder, move_one, MoveOneAction)?;

    let move_up = BlockBuilder::new(MOVE_UP)
        .set_display_name("Machine bit: move up")
        .set_static_hover_text("Machine bit: move up")
        .add_block_group(BRITTLE)
        .set_appearance(point_up_appearance(game_builder, Color::Teal, false)?);
    register_machine_type(game_builder, move_up, MoveFixedDeltaAction(0, 1, 0))?;

    let move_down = BlockBuilder::new(MOVE_DOWN)
        .set_display_name("Machine bit: move down")
        .set_static_hover_text("Machine bit: move down")
        .add_block_group(BRITTLE)
        .set_appearance(point_down_appearance(game_builder, Color::Teal, false)?);
    register_machine_type(game_builder, move_down, MoveFixedDeltaAction(0, -1, 0))?;

    let place_up = BlockBuilder::new(PLACE_UP)
        .set_display_name("Machine bit: place up")
        .set_static_hover_text("Machine bit: place up")
        .add_block_group(BRITTLE)
        .set_appearance(point_up_appearance(game_builder, Color::Green, true)?);
    register_machine_type(
        game_builder,
        add_interact_inventory(
            place_up,
            "Machine bit: place up".to_string(),
            InteractInventoryConfig {
                includes_leftover: true,
                includes_refill_with: true,
                _ne: NonExhaustive(()),
            },
        ),
        PlaceFixedDeltaAction(0, 1, 0),
    )?;
    let place_down = BlockBuilder::new(PLACE_DOWN)
        .set_display_name("Machine bit: place down")
        .set_static_hover_text("Machine bit: place down")
        .add_block_group(BRITTLE)
        .set_appearance(point_down_appearance(game_builder, Color::Green, true)?);
    register_machine_type(
        game_builder,
        add_interact_inventory(
            place_down,
            "Machine bit: place down".to_string(),
            InteractInventoryConfig {
                includes_leftover: true,
                includes_refill_with: true,
                _ne: NonExhaustive(()),
            },
        ),
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
            InteractInventoryConfig {
                includes_leftover: true,
                includes_refill_with: true,
                _ne: NonExhaustive(()),
            },
        ),
        PlaceFacingDirectionAction,
    )?;

    Ok(())
}

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
    fn sense(&self, _ctx: &HandlerContext, _state: &ActionState) -> Result<SenseInput> {
        Ok(SenseInput {
            available_for_take: vec![(MACHINE_INVENTORY_NAME.to_string(), DIGGER_PRIORITY)],
            ..Default::default()
        })
    }
    fn act(&self, ctx: &HandlerContext, state: &ActionState) -> Result<ActionOutcome> {
        dig_at_delta(ctx, state, self.0, self.1, self.2)
    }
}

pub struct DigFacingDirectionAction;
impl MachineAction for DigFacingDirectionAction {
    fn sense(&self, _ctx: &HandlerContext, _state: &ActionState) -> Result<SenseInput> {
        Ok(SenseInput {
            available_for_take: vec![(MACHINE_INVENTORY_NAME.to_string(), DIGGER_PRIORITY)],
            ..Default::default()
        })
    }
    fn act(&self, ctx: &HandlerContext, state: &ActionState) -> Result<ActionOutcome> {
        let (dx, dz) =
            CompassDirection::from_rotation_variant(state.machine_block_id.variant()).to_delta_xz();
        dig_at_delta(ctx, state, dx, 0, dz)
    }
}

pub struct PlaceFixedDeltaAction(i32, i32, i32);
impl MachineAction for PlaceFixedDeltaAction {
    fn sense(&self, ctx: &HandlerContext, state: &ActionState) -> Result<SenseInput> {
        sense_place(ctx, state)
    }
    fn act(&self, ctx: &HandlerContext, state: &ActionState) -> Result<ActionOutcome> {
        place_at_delta(ctx, state, self.0, self.1, self.2)
    }
}

pub struct PlaceFacingDirectionAction;
impl MachineAction for PlaceFacingDirectionAction {
    fn sense(&self, ctx: &HandlerContext, state: &ActionState) -> Result<SenseInput> {
        sense_place(ctx, state)
    }
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
            expects_to_consume: sense0
                .expects_to_consume
                .into_iter()
                .chain(sense1.expects_to_consume.into_iter())
                .collect(),
            available_for_take: sense0
                .available_for_take
                .into_iter()
                .chain(sense1.available_for_take.into_iter())
                .unique()
                .collect(),
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

/// A machine action that proposes moving a fixed delta each cycle
pub struct MoveFixedDeltaAction(i32, i32, i32);
impl MachineAction for MoveFixedDeltaAction {
    fn sense(&self, _ctx: &HandlerContext, _state: &ActionState) -> Result<SenseInput> {
        Ok(SenseInput {
            requested_movement: Some((self.0, self.1, self.2)),
            ..Default::default()
        })
    }
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

/// Provides constants for relative priorities of items. Used as a reference to allow custom priority
/// bands to properly interoperate with the built-in ones.
pub mod item_priorities {
    /// The default priority for items that are given.
    pub const DEFAULT_PRIORITY: i32 = 0;
    /// The priority used for items that distributors use when giving away items.
    pub const DISTRIBUTOR_PRIORITY: i32 = 100;
    /// The priority used for items that diggers give away.
    pub const DIGGER_PRIORITY: i32 = 50;
}

/// A machine action that redistributes items to machines that want them, by offering its own inventory and taking
pub struct DistributorAction;
impl MachineAction for DistributorAction {
    // Offers its own inventory, which has leftover from failed inserts, and player-added materials
    fn sense(&self, _ctx: &HandlerContext, _state: &ActionState) -> Result<SenseInput> {
        Ok(SenseInput {
            available_for_take: vec![(MACHINE_INVENTORY_NAME.to_string(), 100)],
            ..Default::default()
        })
    }

    fn act(&self, ctx: &HandlerContext, state: &ActionState) -> Result<ActionOutcome> {
        use perovskite_server::game_state::items::MaybeStack;

        let mut errors = HashSet::new();

        let mut unfulfilled_requests = state.sense_data.expects_to_consume.clone();
        let mut fulfilled_requests: Vec<(BlockCoordinate, String, ItemStack)> = vec![];
        for (src_coord, inv, _prio) in &state.sense_data.available_for_take {
            if unfulfilled_requests.is_empty() {
                break;
            }
            ctx.game_map()
                .mutate_block_atomically(*src_coord, |_block, ext| {
                    let Some(inv) = ext.as_mut().and_then(|x| x.inventories.get_mut(inv)) else {
                        return Ok(());
                    };

                    for entry in inv.contents_mut() {
                        unfulfilled_requests = unfulfilled_requests
                            .drain(..)
                            .flat_map(|(dst_coord, request)| {
                                // implicit stack-is-some check here
                                if entry.try_item_name() == Some(request.item_name.as_str()) {
                                    let taken = entry.take_items(Some(request.requested_quantity));
                                    if let Some(taken) = taken {
                                        let taken_count = taken.quantity();
                                        fulfilled_requests.push((
                                            dst_coord,
                                            request.inventory_name.clone(),
                                            taken,
                                        ));
                                        if taken_count >= request.requested_quantity {
                                            None
                                        } else {
                                            Some((
                                                dst_coord,
                                                ExpectedConsumption {
                                                    requested_quantity: request.requested_quantity
                                                        - taken_count,
                                                    ..request
                                                },
                                            ))
                                        }
                                    } else {
                                        Some((dst_coord, request))
                                    }
                                } else {
                                    Some((dst_coord, request))
                                }
                            })
                            .collect();
                    }
                    Ok(())
                })?;
        }

        let mut leftovers = vec![];
        for (dst_coord, dst_inv, stack) in fulfilled_requests {
            let leftover = ctx
                .game_map()
                .mutate_block_atomically(dst_coord, |_block, ext| {
                    let Some(inv) = ext.as_mut().and_then(|x| x.inventories.get_mut(&dst_inv))
                    else {
                        return Ok(Some(stack));
                    };
                    Ok(inv.try_insert(stack))
                })?;
            if let Some(leftover) = leftover {
                leftovers.push(leftover);
            }
        }

        // insert leftovers into our own inventory
        ctx.game_map()
            .mutate_block_atomically(state.machine_coord, |_block, ext| {
                let inv = ext
                    .get_or_insert_default()
                    .inventory_mut(MACHINE_INVENTORY_NAME.to_string(), MACHINE_INV_DEFAULT_SIZE);
                for leftover in leftovers {
                    if inv.try_insert(leftover).is_some() {
                        // this should never happen if the distributor is working correctly
                        errors.insert("Distributor inventory full".to_string());
                    }
                }
                Ok(())
            })?;

        Ok(ActionOutcome {
            errors,
            ..Default::default()
        })
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
        sense.merge_in(this_sense, coord);
    }

    sense
        .available_for_take
        .sort_unstable_by_key(|(_, _, p)| std::cmp::Reverse(*p));

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

fn sense_place(ctx: &HandlerContext<'_>, state: &ActionState) -> Result<SenseInput> {
    // Report expected consumption of the refill_with item into MACHINE_INVENTORY_NAME
    // when the machine inventory is empty or absent. A future machine type will act on
    // this to automatically restock the machine from an adjacent supply.

    let (_, item_name) =
        ctx.game_map()
            .get_block_with_extended_data(state.machine_coord, |ext| {
                let Some(refill_with) = ext.simple_data.get(REFILL_WITH_KEY).cloned() else {
                    return Ok(None);
                };
                let Some(inventory) = ext.inventories.get(MACHINE_INVENTORY_NAME) else {
                    // inventory is missing, so it's empty and will be lazily created
                    return Ok(Some(refill_with));
                };
                let inv_needs_fill = inventory.contents().iter().any(|s| {
                    s.is_none()
                        || s.as_ref().is_some_and(|stack| {
                            stack.item_name() == refill_with.as_str() && stack.can_accept_more()
                        })
                });
                if inv_needs_fill {
                    Ok(Some(refill_with))
                } else {
                    Ok(None)
                }
            })?;
    let Some(item_name) = item_name else {
        return Ok(SenseInput::default());
    };
    Ok(SenseInput {
        expects_to_consume: vec![ExpectedConsumption {
            item_name,
            inventory_name: MACHINE_INVENTORY_NAME.to_string(),
            ..Default::default()
        }],
        ..Default::default()
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

/// Key in block simple_data for the item name used to refill the machine inventory.
/// A future machine type will act on this to automatically restock the machine from an
/// external source when the machine inventory is empty.
pub const REFILL_WITH_KEY: &str = "refill_with";

#[derive(Clone, Debug)]
pub struct InteractInventoryConfig {
    pub includes_leftover: bool,
    /// Whether to show a "refill with" item picker field in the interact form.
    /// When enabled, the chosen item name is persisted to simple_data under [REFILL_WITH_KEY].
    pub includes_refill_with: bool,
    pub _ne: NonExhaustive,
}
impl Default for InteractInventoryConfig {
    fn default() -> Self {
        Self {
            includes_leftover: false,
            includes_refill_with: false,
            _ne: NonExhaustive(()),
        }
    }
}

trait ApplyIf: Sized {
    fn apply_if(self, cond: bool, f: impl FnOnce(Self) -> Self) -> Self {
        if cond {
            f(self)
        } else {
            self
        }
    }

    fn apply_if_result(self, cond: bool, f: impl FnOnce(Self) -> Result<Self>) -> Result<Self> {
        if cond {
            f(self)
        } else {
            Ok(self)
        }
    }
}
impl<T: Sized> ApplyIf for T {}

fn add_interact_inventory(
    builder: BlockBuilder,
    title: String,
    config: InteractInventoryConfig,
) -> BlockBuilder {
    builder.add_modifier(move |block| {
        block.interact_key_handler = Some(Box::new(move |ctx, coord, _action| {
            let initial_refill_with = if config.includes_refill_with {
                ctx.game_map()
                    .get_block_with_extended_data(coord, |ext| {
                        Ok(ext.simple_data.get(REFILL_WITH_KEY).cloned())
                    })?
                    .1
            } else {
                None
            };
            let popup = ctx.new_popup().title(title.clone());
            ctx.initiator().try_with_player(move |p| {
                let popup = popup
                    .inventory_view_block(
                        MACHINE_INVENTORY_NAME.to_string(),
                        "Contents:",
                        MACHINE_INV_DEFAULT_SIZE,
                        coord,
                        MACHINE_INVENTORY_NAME.to_string(),
                        true,
                        true,
                        false,
                    )?
                    .apply_if_result(config.includes_leftover, |p| {
                        p.inventory_view_block(
                            MACHINE_LEFTOVER_INVENTORY_NAME.to_string(),
                            "Leftover:",
                            MACHINE_INV_DEFAULT_SIZE,
                            coord,
                            MACHINE_LEFTOVER_INVENTORY_NAME.to_string(),
                            true,
                            true,
                            false,
                        )
                    })?
                    .apply_if(config.includes_refill_with, |p| {
                        p.text_field(
                            TextFieldBuilder::new(REFILL_WITH_KEY)
                                .label("Refill with:")
                                .initial(initial_refill_with.clone().unwrap_or_else(String::new))
                                .refinement(RefinementType::ItemType(Default::default())),
                        )
                    })
                    .inventory_view_stored(
                        "player_inv",
                        "Player inventory:",
                        p.main_inventory(),
                        true,
                        true,
                    )?
                    .apply_if(config.includes_refill_with, |p| {
                        p.set_button_callback(
                            (move |resp: perovskite_server::game_state::client_ui::PopupResponse| {
                                let value = resp
                                    .textfield_values
                                    .get(REFILL_WITH_KEY)
                                    .cloned()
                                    .unwrap_or_else(String::new);
                                resp.ctx.game_map().mutate_block_atomically(
                                    coord,
                                    |_block, ext| {
                                        ext.get_or_insert_default()
                                            .simple_data
                                            .insert(REFILL_WITH_KEY.to_string(), value);
                                        Ok(())
                                    },
                                )
                            })
                            .send_errors_to_chat(),
                        )
                    });
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
                    let inv = ext.get_or_insert_default().inventory_mut(
                        MACHINE_INVENTORY_NAME.to_string(),
                        MACHINE_INV_DEFAULT_SIZE,
                    );
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
            let dig_up_id = ctx
                .block_types()
                .get_by_name(DIG_UP.0)
                .expect("DIG_UP not registered");
            let dirt_id = ctx
                .block_types()
                .get_by_name(DIRT.0)
                .expect("DIRT not registered");
            ctx.game_map()
                .set_block(MACHINE_COORD, dig_up_id, None)
                .or_fail()?;
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
            let dig_down_id = ctx
                .block_types()
                .get_by_name(DIG_DOWN.0)
                .expect("DIG_DOWN not registered");
            let dirt_id = ctx
                .block_types()
                .get_by_name(DIRT.0)
                .expect("DIRT not registered");
            ctx.game_map()
                .set_block(MACHINE_COORD, dig_down_id, None)
                .or_fail()?;
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
            let base_id = ctx
                .block_types()
                .get_by_name(DIG_FACING.0)
                .expect("DIG_FACING not registered");
            let dig_facing_id = base_id.with_variant_unchecked(variant);
            let dirt_id = ctx
                .block_types()
                .get_by_name(DIRT.0)
                .expect("DIRT not registered");
            ctx.game_map()
                .set_block(MACHINE_COORD, dig_facing_id, None)
                .or_fail()?;
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
            let dig_up_id = ctx
                .block_types()
                .get_by_name(DIG_UP.0)
                .expect("DIG_UP not registered");
            let dirt_id = ctx
                .block_types()
                .get_by_name(DIRT.0)
                .expect("DIRT not registered");
            ctx.game_map()
                .set_block(MACHINE_COORD, dig_up_id, None)
                .or_fail()?;
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
            let place_up_id = ctx
                .block_types()
                .get_by_name(PLACE_UP.0)
                .expect("PLACE_UP not registered");
            ctx.game_map()
                .set_block(MACHINE_COORD, place_up_id, None)
                .or_fail()?;
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
            let place_down_id = ctx
                .block_types()
                .get_by_name(PLACE_DOWN.0)
                .expect("PLACE_DOWN not registered");
            ctx.game_map()
                .set_block(MACHINE_COORD, place_down_id, None)
                .or_fail()?;
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
            let base_id = ctx
                .block_types()
                .get_by_name(PLACE_FACING.0)
                .expect("PLACE_FACING not registered");
            let place_facing_id = base_id.with_variant_unchecked(variant);
            ctx.game_map()
                .set_block(MACHINE_COORD, place_facing_id, None)
                .or_fail()?;
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
            let place_up_id = ctx
                .block_types()
                .get_by_name(PLACE_UP.0)
                .expect("PLACE_UP not registered");
            ctx.game_map()
                .set_block(MACHINE_COORD, place_up_id, None)
                .or_fail()?;
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
            let place_up_id = ctx
                .block_types()
                .get_by_name(PLACE_UP.0)
                .expect("PLACE_UP not registered");
            ctx.game_map()
                .set_block(MACHINE_COORD, place_up_id, None)
                .or_fail()?;
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
            let mover_base = ctx
                .block_types()
                .get_by_name(MOVE_ONE.0)
                .expect("MOVE_ONE not registered");
            let mover_id = mover_base.with_variant_unchecked(z_plus);
            ctx.game_map()
                .set_block(trigger_coord, trigger_id, None)
                .or_fail()?;
            ctx.game_map()
                .set_block(mover_coord, mover_id, None)
                .or_fail()?;
            // Only the leading edge — the cell the mover vacates last — needs to be free.
            // The trigger's destination (mover_coord) is occupied by the mover itself, so the
            // movement code skips that check automatically.
            let leading_edge = mover_coord.try_delta(0, 0, 1).unwrap();
            ctx.game_map()
                .set_block(leading_edge, AIR_ID, None)
                .or_fail()?;
            Ok(())
        })?;

        fixture.run_with_context(|ctx| {
            let result = trigger_machine_cycle(&ctx, trigger_coord).or_fail()?;
            expect_that!(result.errors, is_empty());
            Ok(())
        })?;

        fixture.run_with_context(|ctx| {
            // Original positions are now air.
            expect_that!(
                ctx.game_map().get_block(trigger_coord).or_fail()?,
                IsBlock(AIR_ID)
            );
            expect_that!(
                ctx.game_map().get_block(mover_coord).or_fail()?,
                IsBlock(MANUAL_TRIGGER)
            );
            let new_mover = mover_coord.try_delta(0, 0, 1).unwrap();
            expect_that!(
                ctx.game_map().get_block(new_mover).or_fail()?,
                IsBlock(MOVE_ONE)
            );
            Ok(())
        })
    }

    // ────────────────────────────────────────────────────────────────────────
    // Manual trigger block (interact key handler)
    // ────────────────────────────────────────────────────────────────────────

    #[gtest]
    fn test_move_up_moves_machine_upward(fixture: &TestFixture) -> googletest::Result<()> {
        start(fixture)?;

        // Layout: MANUAL_TRIGGER at (0,16,0), MOVE_UP at (0,17,0).
        // After one cycle the whole machine should shift +1 in Y.
        let trigger_coord = MACHINE_COORD;
        let mover_coord = MACHINE_COORD.try_delta(0, 1, 0).unwrap();

        fixture.run_with_context(|ctx| {
            let trigger_id = ctx
                .block_types()
                .get_by_name(MANUAL_TRIGGER.0)
                .expect("MANUAL_TRIGGER not registered");
            let mover_id = ctx
                .block_types()
                .get_by_name(MOVE_UP.0)
                .expect("MOVE_UP not registered");
            ctx.game_map()
                .set_block(trigger_coord, trigger_id, None)
                .or_fail()?;
            ctx.game_map()
                .set_block(mover_coord, mover_id, None)
                .or_fail()?;
            let leading_edge = mover_coord.try_delta(0, 1, 0).unwrap();
            ctx.game_map()
                .set_block(leading_edge, AIR_ID, None)
                .or_fail()?;
            Ok(())
        })?;

        fixture.run_with_context(|ctx| {
            let result = trigger_machine_cycle(&ctx, trigger_coord).or_fail()?;
            expect_that!(result.errors, is_empty());
            Ok(())
        })?;

        fixture.run_with_context(|ctx| {
            expect_that!(
                ctx.game_map().get_block(trigger_coord).or_fail()?,
                IsBlock(AIR_ID)
            );
            expect_that!(
                ctx.game_map().get_block(mover_coord).or_fail()?,
                IsBlock(MANUAL_TRIGGER)
            );
            let new_mover = mover_coord.try_delta(0, 1, 0).unwrap();
            expect_that!(
                ctx.game_map().get_block(new_mover).or_fail()?,
                IsBlock(MOVE_UP)
            );
            Ok(())
        })
    }

    #[gtest]
    fn test_move_down_moves_machine_downward(fixture: &TestFixture) -> googletest::Result<()> {
        start(fixture)?;

        // Layout: MANUAL_TRIGGER at (0,16,0), MOVE_DOWN at (0,15,0).
        // After one cycle the whole machine should shift -1 in Y.
        let trigger_coord = MACHINE_COORD;
        let mover_coord = MACHINE_COORD.try_delta(0, -1, 0).unwrap();

        fixture.run_with_context(|ctx| {
            let trigger_id = ctx
                .block_types()
                .get_by_name(MANUAL_TRIGGER.0)
                .expect("MANUAL_TRIGGER not registered");
            let mover_id = ctx
                .block_types()
                .get_by_name(MOVE_DOWN.0)
                .expect("MOVE_DOWN not registered");
            ctx.game_map()
                .set_block(trigger_coord, trigger_id, None)
                .or_fail()?;
            ctx.game_map()
                .set_block(mover_coord, mover_id, None)
                .or_fail()?;
            let leading_edge = mover_coord.try_delta(0, -1, 0).unwrap();
            ctx.game_map()
                .set_block(leading_edge, AIR_ID, None)
                .or_fail()?;
            Ok(())
        })?;

        fixture.run_with_context(|ctx| {
            let result = trigger_machine_cycle(&ctx, trigger_coord).or_fail()?;
            expect_that!(result.errors, is_empty());
            Ok(())
        })?;

        fixture.run_with_context(|ctx| {
            expect_that!(
                ctx.game_map().get_block(trigger_coord).or_fail()?,
                IsBlock(AIR_ID)
            );
            expect_that!(
                ctx.game_map().get_block(mover_coord).or_fail()?,
                IsBlock(MANUAL_TRIGGER)
            );
            let new_mover = mover_coord.try_delta(0, -1, 0).unwrap();
            expect_that!(
                ctx.game_map().get_block(new_mover).or_fail()?,
                IsBlock(MOVE_DOWN)
            );
            Ok(())
        })
    }

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
            let dig_up_id = ctx
                .block_types()
                .get_by_name(DIG_UP.0)
                .expect("DIG_UP not registered");
            let dirt_id = ctx
                .block_types()
                .get_by_name(DIRT.0)
                .expect("DIRT not registered");
            ctx.game_map()
                .set_block(trigger_coord, trigger_id, None)
                .or_fail()?;
            ctx.game_map()
                .set_block(digger_coord, dig_up_id, None)
                .or_fail()?;
            ctx.game_map()
                .set_block(target_coord, dirt_id, None)
                .or_fail()?;
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
            expect_that!(
                ctx.game_map().get_block(target_coord).or_fail()?,
                IsBlock(AIR_ID)
            );
            Ok(())
        })
    }
}
