// placeholders for circuits

use std::ops::Deref;

use crate::{
    blocks::{BlockBuilder, BuiltBlock},
    default_game::DefaultGameBuilder,
    game_builder::{GameBuilder, GameBuilderExtension},
};
use anyhow::Result;
use perovskite_core::{block_id::BlockId, coordinates::BlockCoordinate};
use perovskite_server::game_state::{
    blocks::{BlockInteractionResult, BlockTypeName},
    event::HandlerContext,
    GameStateExtension,
};
use rustc_hash::FxHashMap;

mod wire;

/// Constants for blocks that are part of the circuits mechanism
///
/// These should be used to *detect* circuits; in order to make a block have
/// circuit behavior, the block must also be registered with the circuits plugin directly;
/// simply setting these groups is not enough.
pub mod constants {
    /// All circuits fall into this group
    pub const CIRCUITS_GROUP: &str = "circuits:all_circuits";
    /// A callback should fire whenever a compatible circuit block is placed or removed
    /// in the vicinity, even if is not currently
    pub const CIRCUITS_DETECTS_CONNECTIVITY: &str = "circuits:detects_connectivity";
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ConnectivityRotation {
    /// Coordinates are not rotated, and simply follow global axes
    NoRotation,
    /// Coordinates are rotated based on the variant, same as [crate::blocks::CubeAppearanceBuilder::set_rotate_laterally]
    RotateNeswWithVariant,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BlockConnectivity {
    pub dx: i8,
    pub dy: i8,
    pub dz: i8,
    pub rotation_mode: ConnectivityRotation,
}
impl BlockConnectivity {
    pub const fn unrotated(dx: i8, dy: i8, dz: i8) -> Self {
        Self {
            dx,
            dy,
            dz,
            rotation_mode: ConnectivityRotation::NoRotation,
        }
    }
    pub const fn rotated_nesw_with_variant(dx: i8, dy: i8, dz: i8) -> Self {
        Self {
            dx,
            dy,
            dz,
            rotation_mode: ConnectivityRotation::RotateNeswWithVariant,
        }
    }
    pub fn eval(&self, coord: BlockCoordinate, variant: u16) -> Option<BlockCoordinate> {
        let (dx, dz) = match (self.rotation_mode, variant % 4) {
            (ConnectivityRotation::NoRotation, _) => (self.dx, self.dz),
            (ConnectivityRotation::RotateNeswWithVariant, 0) => (self.dx, self.dz),
            (ConnectivityRotation::RotateNeswWithVariant, 1) => (-self.dz, self.dx),
            (ConnectivityRotation::RotateNeswWithVariant, 2) => (-self.dx, -self.dz),
            (ConnectivityRotation::RotateNeswWithVariant, 3) => (self.dz, -self.dx),
            _ => return None,
        };
        coord.try_delta(dx as i32, self.dy as i32, dz as i32)
    }
}

#[derive(Clone, Debug)]
struct CircuitBlockProperties {
    /// Neighboring coordinates to which this block can connect
    /// A connection is only made if the block at that coordinate itself has
    /// the current block's location in *its* connectivity list
    connectivity: Vec<BlockConnectivity>,
}

pub trait CircuitBlockCallbacks: Send + Sync + 'static {
    fn update_connectivity(&self, ctx: &CircuitHandlerContext<'_>, coord: BlockCoordinate) -> Result<()>;
}

/// Private state for the circuits plugin
struct CircuitGameStateExtension {
    basic_properties: FxHashMap<u32, CircuitBlockProperties>,
    callbacks: FxHashMap<u32, Box<dyn CircuitBlockCallbacks>>,
}
impl GameStateExtension for CircuitGameStateExtension {}

/// A context to be used for circuits related callbacks.
pub struct CircuitHandlerContext<'a> {
    pub inner: &'a HandlerContext<'a>,
}
impl<'a> Deref for CircuitHandlerContext<'a> {
    type Target = HandlerContext<'a>;
    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

pub trait CircuitGameBuilder {
    /// Initialize the circuits content of the plugin:
    /// - makes basic circuits available to players
    /// - makes other functions in this trait available to plugins that need them
    fn initialize_circuits(&mut self) -> Result<()>;
}

trait CircuitGameBuilerPrivate {
    fn register_wire_private(
        &mut self,
        on: BuiltBlock,
        off: BuiltBlock,
        properties: CircuitBlockProperties,
    ) -> Result<()>;
}

impl CircuitGameBuilder for GameBuilder {
    fn initialize_circuits(&mut self) -> Result<()> {
        if self
            .builder_extension::<CircuitGameBuilderExt>()
            .initialized
        {
            return Ok(());
        }
        self.builder_extension::<CircuitGameBuilderExt>()
            .initialized = true;

        self.inner
            .blocks_mut()
            .register_block_group(constants::CIRCUITS_GROUP);

        wire::register_wire(self)?;
        Ok(())
    }
}

impl CircuitGameBuilerPrivate for GameBuilder {
    fn register_wire_private(
        &mut self,
        on: BuiltBlock,
        off: BuiltBlock,
        properties: CircuitBlockProperties,
    ) -> Result<()> {
        fn make_wire_callbacks(block_id: BlockId) -> Box<dyn CircuitBlockCallbacks> {
            Box::new(wire::WireCallbacksImpl { base_id: block_id })
        }

        let inner = self
            .builder_extension::<CircuitGameBuilderExt>()
            .resulting_state
            .as_mut()
            .expect("pre_run already called");
        inner
            .basic_properties
            .insert(on.id.base_id(), properties.clone());
        inner.basic_properties.insert(off.id.base_id(), properties);

        inner.callbacks.insert(on.id.base_id(), make_wire_callbacks(on.id));
        inner.callbacks.insert(off.id.base_id(), make_wire_callbacks(off.id));

        Ok(())
    }
}

struct CircuitGameBuilderExt {
    initialized: bool,
    resulting_state: Option<CircuitGameStateExtension>,
}
impl GameBuilderExtension for CircuitGameBuilderExt {
    fn pre_run(&mut self, server_builder: &mut perovskite_server::server::ServerBuilder) {
        tracing::info!("circuits pre_run");
        server_builder.add_extension(self.resulting_state.take().expect("pre_run already called"));
    }
}
impl Default for CircuitGameBuilderExt {
    fn default() -> Self {
        Self {
            initialized: false,
            resulting_state: Some(CircuitGameStateExtension {
                basic_properties: FxHashMap::default(),
                callbacks: FxHashMap::default(),
            }),
        }
    }
}

pub trait CircuitBlockBuilder {
    /// Adds circuits callbacks to the block being built.
    ///
    /// These callbacks will perform circuits-related tasks and then delegate
    /// to the existing callbacks in the builder.
    fn register_circuit_callbacks(self) -> BlockBuilder;
}

impl CircuitBlockBuilder for BlockBuilder {
    fn register_circuit_callbacks(self) -> BlockBuilder {
        self.add_modifier(Box::new(move |bt| {
            let old_handler = bt.dig_handler_full.take();
            bt.dig_handler_full = Some(Box::new(move |ctx, coord, tool_stack| {
                // Run the initial handler first, so that the map is updated before the circuit
                // callbacks are invoked
                // However, we need the old ID to figure out which nearby blocks need updates
                let old_id = ctx.game_map().try_get_block(coord).unwrap();
                let result = match &old_handler {
                    Some(old_handler) => old_handler(ctx, coord, tool_stack),
                    None => Ok(BlockInteractionResult::default()),
                };
                if let Err(e) = dispatch::dig(old_id, coord, &dispatch::make_circuit_context(&ctx)) {
                    tracing::error!("Error in dig handler: {}", e);
                }
                result
            }));

            bt.client_info
                .groups
                .push(constants::CIRCUITS_GROUP.to_string());
        })).add_item_modifier(
            Box::new(|item| {
                let old_handler = item.place_handler.take();
                item.place_handler = Some(Box::new(move |ctx, coord, anchor, tool_stack| {
                    let result = match &old_handler {
                        Some(old_handler) => old_handler(ctx, coord, anchor, tool_stack),
                        None => Ok(None),
                    };
                    if let Err(e) = dispatch::place(coord, &dispatch::make_circuit_context(&ctx)) {
                        tracing::error!("Error in place handler: {}", e);
                    }
                    result
                }))
            })
        )
    }
}
mod dispatch {
    use anyhow::Result;
    use perovskite_core::{block_id::BlockId, coordinates::BlockCoordinate};
    use perovskite_server::game_state::event::HandlerContext;

    use super::{constants::CIRCUITS_GROUP, CircuitGameStateExtension, CircuitHandlerContext};

    pub(crate) fn make_circuit_context<'a>(
        ctx: &'a HandlerContext<'a>,
    ) -> CircuitHandlerContext<'a> {
        CircuitHandlerContext { inner: ctx }
    }

    pub(crate) fn dig(old_block_id: BlockId, coord: BlockCoordinate, ctx: &CircuitHandlerContext<'_>) -> Result<()> {
        let block_id = match ctx.game_map().try_get_block(coord) {
            Some(x) => x,
            None => {
                tracing::warn!("We're in a block's event handler, but its chunk is unloaded");
                return Ok(());
            }
        };
        let ext = ctx
            .extension::<CircuitGameStateExtension>()
            .expect("circuits gamestate extension not found");
        let circuits_group = ctx
            .block_types()
            .block_group(CIRCUITS_GROUP)
            .expect("circuits group not found");
        let connectivity = match ext.basic_properties.get(&block_id.base_id()) {
            Some(properties) => &properties.connectivity,
            None => {
                tracing::warn!("block {:?} has no circuit properties but was registered with the circuits plugin", block_id);
                return Ok(());
            }
        };

        for connection in connectivity {
            let neighbor = match connection.eval(coord, block_id.variant()) {
                Some(neighbor) => neighbor,
                None => continue,
            };
            let neighbor_block = match ctx.game_map().try_get_block(neighbor) {
                Some(x) => x,
                None => continue,
            };
            if !circuits_group.contains(neighbor_block) {
                continue;
            }
            let neighbor_callbacks = match ext.callbacks.get(&neighbor_block.base_id()) {
                Some(x) => x,
                None => {
                    tracing::warn!(
                        "block {:?} has no circuit callbacks but had the circuits group",
                        neighbor_block
                    );
                    continue;
                }
            };
            neighbor_callbacks.update_connectivity(ctx, neighbor)?;
        }

        Ok(())
    }

    pub(crate) fn place(coord: BlockCoordinate, ctx: &CircuitHandlerContext<'_>) -> Result<()> {
        // todo debug why try_get_block doesn't work here
        let block_id = match ctx.game_map().try_get_block(coord) {
            Some(x) => x,
            None => {
                tracing::warn!("We're in a block's event handler, but its chunk is unloaded");
                return Ok(());
            }
        };
        let ext = ctx
            .extension::<CircuitGameStateExtension>()
            .expect("circuits gamestate extension not found");
        let circuits_group = ctx
            .block_types()
            .block_group(CIRCUITS_GROUP)
            .expect("circuits group not found");
        let connectivity = match ext.basic_properties.get(&block_id.base_id()) {
            Some(properties) => &properties.connectivity,
            None => {
                tracing::warn!("block {:?} has no circuit properties but was registered with the circuits plugin", block_id);
                return Ok(());
            }
        };

        for connection in dbg!(connectivity) {
            let neighbor = match dbg!(connection.eval(coord, block_id.variant())) {
                Some(neighbor) => neighbor,
                None => continue,
            };
            let neighbor_block = match ctx.game_map().try_get_block(neighbor) {
                Some(x) => x,
                None => continue,
            };
            if dbg!(!circuits_group.contains(neighbor_block)) {
                continue;
            }
            let neighbor_callbacks = match ext.callbacks.get(&neighbor_block.base_id()) {
                Some(x) => x,
                None => {
                    tracing::warn!(
                        "block {:?} has no circuit callbacks but had the circuits group",
                        neighbor_block
                    );
                    continue;
                }
            };
            neighbor_callbacks.update_connectivity(ctx, neighbor)?;
        }
        let self_callbacks = ext.callbacks.get(&block_id.base_id()).unwrap();
        self_callbacks.update_connectivity(ctx, coord)?;

        Ok(())
    }
}
