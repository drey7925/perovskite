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
    blocks::{BlockInteractionResult, FastBlockName, FullHandler, InlineContext, InlineHandler},
    event::HandlerContext,
    GameStateExtension,
};
use rustc_hash::FxHashMap;

use self::events::CircuitHandlerContext;

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

/// The connectivity of a block - a relative coordinate and rotation mode
/// indicating whether that relative coordinate gets rotated with the variant.
/// 
/// Connections are made when blocks reciprocally connect to each other. For example,
/// a block with connectivity [BlockConnectivity::unrotated(0, 1, 0)] will connect to
/// a block with connectivity [BlockConnectivity::unrotated(0, -1, 0)] when the two blocks
/// are adjacent.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BlockConnectivity {
    pub dx: i8,
    pub dy: i8,
    pub dz: i8,
    pub rotation_mode: ConnectivityRotation,
}
impl BlockConnectivity {
    /// Constructs a new connectivity. Coordinates are not rotated
    pub const fn unrotated(dx: i8, dy: i8, dz: i8) -> Self {
        Self {
            dx,
            dy,
            dz,
            rotation_mode: ConnectivityRotation::NoRotation,
        }
    }
    /// Constructs a new connectivity. Coordinates are rotated based on the variant, same as [crate::blocks::CubeAppearanceBuilder::set_rotate_laterally]
    pub const fn rotated_nesw_with_variant(dx: i8, dy: i8, dz: i8) -> Self {
        Self {
            dx,
            dy,
            dz,
            rotation_mode: ConnectivityRotation::RotateNeswWithVariant,
        }
    }
    /// Computes the coordinates that this connectivity connects to.
    ///
    /// variant is the source block's variant
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
    /// Called when another block is placed or removed in the vicinity, and may affect
    /// the connections made by *this* block.
    ///
    /// If this block needs to react in some way (e.g. changing its appearance), it should
    /// do so in this function. To do so, it can use [`get_live_connectivities`] and react accordingly
    /// (e.g. by updating its variant). Note that updating a variant from this callback is inherently racy
    /// because another thread could have updated or dug the block - it is wise
    /// to use compare_and_set_block_predicate. See examples in this crate's source code, e.g. impls in in wire.rs
    /// (which is a private API, but is worth looking at as an example).
    fn update_connectivity(
        &self,
        ctx: &CircuitHandlerContext<'_>,
        coord: BlockCoordinate,
    ) -> Result<()> {
        Ok(())
    }

    /// TODO add parameters and document
    fn on_incoming_edge(&self) -> Result<()> {
        Ok(())
    }
}

/// Get the connections being made to the given block at the given coordinate.
pub fn get_live_connectivities(
    ctx: &CircuitHandlerContext<'_>,
    coord: BlockCoordinate,
) -> smallvec::SmallVec<[BlockConnectivity; 8]> {
    let circuit_extension = ctx.inner.extension::<CircuitGameStateExtension>().unwrap();

    let block_id = match ctx.game_map().try_get_block(coord) {
        Some(b) => b,
        None => {
            tracing::warn!("We're in a block's event handler, but its chunk is unloaded");
            return smallvec::smallvec![];
        }
    };
    let connectivity_rules = match circuit_extension.basic_properties.get(&block_id.base_id()) {
        Some(p) => &p.connectivity,
        None => {
            tracing::warn!("block {:?} has no circuit properties but was registered with the circuits plugin", block_id);
            return smallvec::smallvec![];
        }
    };
    let mut result = smallvec::smallvec![];
    for connectivity in connectivity_rules {
        // The variant is irrelevant for wires
        let neighbor_coord = match connectivity.eval(coord, block_id.variant()) {
            Some(c) => c,
            None => continue,
        };
        let neighbor_block = match ctx.inner.game_map().try_get_block(neighbor_coord) {
            Some(b) => b,
            None => continue,
        };
        let neighbor_properties = match circuit_extension
            .basic_properties
            .get(&neighbor_block.base_id())
        {
            Some(p) => p,
            None => continue,
        };
        if neighbor_properties
            .connectivity
            .iter()
            .any(|x| x.eval(neighbor_coord, neighbor_block.variant()) == Some(coord))
        {
            result.push(*connectivity);
        }
    }
    result
}

/// Private state for the circuits plugin
struct CircuitGameStateExtension {
    basic_properties: FxHashMap<u32, CircuitBlockProperties>,
    callbacks: FxHashMap<u32, Box<dyn CircuitBlockCallbacks>>,
}
impl GameStateExtension for CircuitGameStateExtension {}

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

        inner
            .callbacks
            .insert(on.id.base_id(), make_wire_callbacks(on.id));
        inner
            .callbacks
            .insert(off.id.base_id(), make_wire_callbacks(off.id));

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
            let old_full_handler: Option<&'static FullHandler> =
                match bt.dig_handler_full.take().map(Box::leak) {
                    Some(x) => Some(x),
                    None => None,
                };
            let block_name = FastBlockName::new(bt.client_info.short_name.clone());
            bt.dig_handler_full = Some(Box::new(move |ctx, coord, tool_stack| {
                // Run the initial handler first, so that the map is updated before the circuit
                // callbacks are invoked
                // However, we need the old ID to figure out which nearby blocks need updates
                let old_result = match &old_full_handler {
                    Some(old_handler) => old_handler(ctx, coord, tool_stack),
                    None => Ok(BlockInteractionResult::default()),
                };
                if let Err(e) = dispatch::dig(
                    ctx.block_types()
                        .resolve_name(&block_name)
                        .expect("block name not found, yet we're in its handler"),
                    coord,
                    &events::make_root_context(&ctx),
                ) {
                    tracing::error!("Error in circuits dig handler: {}", e);
                }
                old_result
            }));

            bt.client_info
                .groups
                .push(constants::CIRCUITS_GROUP.to_string());
        }))
        .add_item_modifier(Box::new(|item| {
            let old_handler = item.place_handler.take();
            item.place_handler = Some(Box::new(move |ctx, coord, anchor, tool_stack| {
                let result = match &old_handler {
                    Some(old_handler) => old_handler(ctx, coord, anchor, tool_stack),
                    None => Ok(None),
                };
                if let Err(e) = dispatch::place(coord, &events::make_root_context(&ctx)) {
                    tracing::error!("Error in place handler: {}", e);
                }
                result
            }))
        }))
    }
}
mod dispatch {
    use anyhow::Result;
    use perovskite_core::{block_id::BlockId, coordinates::BlockCoordinate};
    use perovskite_server::game_state::event::HandlerContext;

    use super::{
        constants::CIRCUITS_GROUP, events::CircuitHandlerContext, CircuitGameStateExtension,
    };

    pub(crate) fn dig(
        old_block_id: BlockId,
        coord: BlockCoordinate,
        ctx: &CircuitHandlerContext<'_>,
    ) -> Result<()> {
        let ext = ctx
            .extension::<CircuitGameStateExtension>()
            .expect("circuits gamestate extension not found");
        let circuits_group = ctx
            .block_types()
            .block_group(CIRCUITS_GROUP)
            .expect("circuits group not found");
        let connectivity = match ext.basic_properties.get(&old_block_id.base_id()) {
            Some(properties) => &properties.connectivity,
            None => {
                tracing::warn!("block {:?} has no circuit properties but was registered with the circuits plugin", old_block_id);
                return Ok(());
            }
        };

        for connection in connectivity {
            let neighbor = match connection.eval(coord, old_block_id.variant()) {
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
        let self_callbacks = ext.callbacks.get(&block_id.base_id()).unwrap();
        self_callbacks.update_connectivity(ctx, coord)?;

        Ok(())
    }
}

/// The circuits plugin requires events to be associated with a context, which tracks
/// who initiated the event as well as (yet to be defined) ratelimiting/anti-DoS/recursion limiting
/// measures.
pub mod events {
    use std::ops::Deref;

    use perovskite_core::coordinates::BlockCoordinate;
    use perovskite_server::game_state::event::HandlerContext;

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

    /// Creates a new root context for the circuits plugin
    pub fn make_root_context<'a>(ctx: &'a HandlerContext) -> CircuitHandlerContext<'a> {
        CircuitHandlerContext { inner: ctx }
    }

    pub enum EdgeType {
        /// A signal is being driven from low to high
        Rising,
        /// A signal is being driven from high to low
        Falling,
        /// The caller is unsure of what kind of edge is being triggered,
        /// but wants any connected nets to be recalculated.
        Unknown,
    }

    /// Signals that a block's digital output is (potentially) changing.
    /// This can be for any reason, such as a button being pressed, a sensor's
    /// state changing, or a block being placed with an initial high output state.
    ///
    /// If the connection is made directly to another block, an event will be delivered
    /// directly to it. If the connection is made to a wire, the wire will be recalculated
    /// and any connected inputs will be signalled if the net's state transitioned.
    ///
    /// If edge is set to Rising or Falling, then net recalculations may be skipped as an
    /// optimization. It's always safe to set `[EdgeType::Unknown]`.
    ///
    /// Args:
    ///     coord: The coordinate of the block whose output is changing
    ///     connection: The coordinate of the block that the output is connected to
    ///     edge: The type of edge that triggered the signal.
    pub fn transmit_edge(
        ctx: &CircuitHandlerContext<'_>,
        coord: BlockCoordinate,
        connection: BlockCoordinate,
        _edge: EdgeType,
    ) {
        // TODO: use the edge type as an optimization hint
    }
}
