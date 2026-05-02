//! Autobuild machines: autobuild, but made of blocks with tasks.
//!
//! A player can assemble a handful of blocks, each with a task
//! (e.g. "place carts:rail_tile", "if non-air replace with tunnel_wall",
//! "if air, replace with bridge_deck", "every 32, place track gantry"),
//! almost jacquard loom style config, that you can set up and leave running.
use std::collections::HashMap;

use anyhow::{bail, Result};
use perovskite_core::coordinates::BlockCoordinate;
use perovskite_server::game_state::event::HandlerContext;

use crate::{blocks::BlockAppearanceBuilder, game_builder::GameBuilder};

pub const AUTOBUILD_MACHINES_GROUP: &str = "autobuild:machines";

#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ActionState {
    /// The coordinate of the machine block itself.
    pub machine_coord: BlockCoordinate,
    /// The coordinate that the machine is acting _on_
    pub target_coord: BlockCoordinate,
    /// The movement delta for the machine group after this action is completed.
    pub movement_delta: (i32, i32, i32),
}

#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SenseData {
    should_stop: bool,
    extended: HashMap<String, String>,
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

pub trait MachineAction {
    fn sense(&self, ctx: &HandlerContext, state: &ActionState) -> Result<HashMap<String, String>> {
        // Default impl: don't sense anything.
        Ok(HashMap::new())
    }

    /// The action to perform when the machine is activated. This should just do the intended action
    /// of the machine, but not move itself (despite what movement_delta says). The machine system will
    /// handle movement on its own.
    fn act(
        &self,
        ctx: &HandlerContext,
        state: &ActionState,
        merged_state: &HashMap<String, String>,
    ) -> Result<()>;
}

#[non_exhaustive]
pub struct MachineDef {
    pub appearance: BlockAppearanceBuilder,
    pub action: Box<dyn MachineAction>,
}

pub fn register_machine(game_builder: &mut GameBuilder, def: MachineDef) -> Result<()> {
    bail!("todo")
}
