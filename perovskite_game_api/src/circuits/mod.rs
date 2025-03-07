use crate::default_game::DefaultGameBuilder;
use crate::{
    blocks::{BlockBuilder, BuiltBlock},
    game_builder::{GameBuilder, GameBuilderExtension},
};
use anyhow::{Context, Result};
use perovskite_core::{block_id::BlockId, coordinates::BlockCoordinate};
use perovskite_server::game_state::blocks::{InlineContext, InlineHandler};
use perovskite_server::game_state::{
    blocks::{BlockInteractionResult, FastBlockName, FullHandler},
    event::HandlerContext,
    GameStateExtension,
};
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::ops::Deref;

use self::events::CircuitHandlerContext;

mod simple_blocks;
mod wire;
// Todo: consider making this pub if it has useful abstractions for custom gates
mod gates;

mod switches;

/// Constants for blocks that are part of the circuits mechanism
///
/// These should be used to *detect* circuits; in order to make a block have
/// circuit behavior, the block must also be registered with the circuits plugin directly;
/// simply setting these groups is not enough.
pub mod constants {
    /// All circuits fall into this group
    pub const CIRCUITS_GROUP: &str = "circuits:all_circuits";
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
    /// An ID that your block's callbacks can use for any arbitrary purpose. IDs do not need to be
    /// unique, consecutive, etc.
    pub id: u32,
}
impl BlockConnectivity {
    /// Constructs a new connectivity. Coordinates are not rotated
    pub const fn unrotated(dx: i8, dy: i8, dz: i8, id: u32) -> Self {
        Self {
            dx,
            dy,
            dz,
            rotation_mode: ConnectivityRotation::NoRotation,
            id,
        }
    }
    /// Constructs a new connectivity. Coordinates are rotated based on the variant, same as [crate::blocks::CubeAppearanceBuilder::set_rotate_laterally]
    pub const fn rotated_nesw_with_variant(dx: i8, dy: i8, dz: i8, id: u32) -> Self {
        Self {
            dx,
            dy,
            dz,
            rotation_mode: ConnectivityRotation::RotateNeswWithVariant,
            id,
        }
    }
    /// Computes the coordinates that this connectivity connects to.
    ///
    /// variant is the source block's variant.
    ///
    /// Returns None in case of coordinate overflow (i.e. edge of map)
    pub fn eval(&self, coord: BlockCoordinate, variant: u16) -> Option<BlockCoordinate> {
        let (dx, dz) = match (self.rotation_mode, variant % 4) {
            (ConnectivityRotation::NoRotation, _) => (self.dx, self.dz),
            (ConnectivityRotation::RotateNeswWithVariant, 0) => (self.dx, self.dz),
            (ConnectivityRotation::RotateNeswWithVariant, 1) => (self.dz, -self.dx),
            (ConnectivityRotation::RotateNeswWithVariant, 2) => (-self.dx, -self.dz),
            (ConnectivityRotation::RotateNeswWithVariant, 3) => (-self.dz, self.dx),
            _ => unreachable!("shouldn't happen, variant % 4 should always be in [0, 3]"),
        };
        coord.try_delta(dx as i32, self.dy as i32, dz as i32)
    }
}

/// The properties of a block participating in a circuit
#[derive(Clone, Debug)]
pub struct CircuitBlockProperties {
    /// Neighboring coordinates to which this block can connect
    /// A connection is only made if the block at that coordinate itself has
    /// the current block's location in *its* connectivity list
    pub connectivity: Vec<BlockConnectivity>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
/// The state of a pin. For now, this only supports simple digital high/low
/// states.
///
/// However, future implementations may support other types of states, for example
/// something like SERDES, pullups/pulldowns, or possibly simple analog (although not a full
/// physically realizable analog).
///
/// Note the following rules, which do not necessarily match physics:
/// - If a net is driven by at least one High signal, it is High.
/// - If a net is driven by only HighZ or Low signals, it is Low.
#[non_exhaustive]
pub enum PinState {
    /// The signal is being driven low
    Low,
    /// The signal is being driven high
    High,
}
macro_rules! delegate_bin_op {
    ($trait:path, $method:ident) => {
        impl $trait for PinState {
            type Output = PinState;
            #[inline]
            fn $method(self, rhs: Self) -> Self::Output {
                if (self == PinState::High).$method(rhs == PinState::High) {
                    PinState::High
                } else {
                    PinState::Low
                }
            }
        }
    };
}
delegate_bin_op!(std::ops::BitOr, bitor);
delegate_bin_op!(std::ops::BitAnd, bitand);
delegate_bin_op!(std::ops::BitXor, bitxor);
impl std::ops::Not for PinState {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            PinState::Low => PinState::High,
            PinState::High => PinState::Low,
        }
    }
}
impl From<bool> for PinState {
    fn from(value: bool) -> Self {
        if value {
            PinState::High
        } else {
            PinState::Low
        }
    }
}

#[derive(Clone, Debug)]
pub struct BusMessage {
    pub sender: BlockCoordinate,
    pub data: HashMap<String, String>,
}

pub trait CircuitBlockCallbacks: Send + Sync + 'static {
    /// Called when another block is placed or removed in the vicinity, and may affect
    /// the connections made by *this* block.
    ///
    /// If this block needs to react in some way (e.g. changing its appearance), it should
    /// do so in this function. To do so, it can use [`get_live_connectivities`] and react accordingly
    /// (e.g. by updating its variant). Note that updating a variant from this callback is inherently racy
    /// because another thread could have updated or dug the block - it is wise
    /// to use mutate_block_atomically. See examples in this crate's source code, e.g. impls in in wire.rs
    /// (which is a private API, but is worth looking at as an example).
    fn update_connectivity(
        &self,
        _ctx: &CircuitHandlerContext<'_>,
        _coord: BlockCoordinate,
    ) -> Result<()> {
        Ok(())
    }

    /// Called when an input into this block is possibly transitioning.
    /// The implementation should react (e.g. by updating its appearance/variant), and should
    /// also call `[events::transmit_edge]` if any of its ouputs are possibly changing.
    ///
    /// Note that this will receive spurious calls, in cases where a transition did not actually occur, but
    /// the engine was not able to suppress the call. (the cases where we suppress the signal will vary as
    /// further optimizations are added).
    fn on_incoming_edge(
        &self,
        _ctx: &CircuitHandlerContext<'_>,
        _coordinate: BlockCoordinate,
        _from: BlockCoordinate,
        _state: PinState,
    ) -> Result<()> {
        Ok(())
    }

    fn on_bus_message(
        &self,
        _ctx: &CircuitHandlerContext<'_>,
        _coordinate: BlockCoordinate,
        _from: BlockCoordinate,
        _message: &BusMessage,
    ) -> Result<()> {
        Ok(())
    }

    /// Called when the circuits engine needs to determine the state of a pin at the given coordinate.
    ///
    /// This should be as fast as possible.
    ///
    /// It is discouraged, but not forbidden, to call back into the circuits plugin to determine the status of inputs.
    /// Ideally, the state would be cached in the variant, and updated in response to [`on_incoming_edge`].
    ///
    /// Note that this is a bit racy - if this block was dug up by another thread, it may not be present by
    /// the time this function is called. A simple implementation would call `try_get_block` and verify that
    /// the base ID matches before using the variant to make a decision
    ///
    /// More expensive blocks (e.g. a whole microcontroller) may need to use extended data or perform computation.
    ///
    /// Args:
    ///     coord: The coordinate of the block whose output is being sampled
    ///     destination: The coordinate of the block that the output in question is connected to
    fn sample_pin(
        &self,
        _ctx: &CircuitHandlerContext<'_>,
        _coord: BlockCoordinate,
        _destination: BlockCoordinate,
    ) -> PinState {
        PinState::Low
    }

    /// Called when this block should break down.
    ///
    /// This happens to an arbitrary block in a combinational loop if it keeps looping enough times to
    /// exceed the TTL of its context.
    ///
    /// It also happens if a block tries to drive too many standard wires.
    ///
    /// In-game, this should act as if the block were overloaded, overheating, etc.
    /// This should do something that breaks a combinational loop or similar, e.g. by causing
    /// the block to enter a broken, useless state.
    ///
    /// **Warning:** If this leads to the block being replaced, care should be taken that this doesn't lead to
    /// new circuit callbacks (e.g. edges) being triggered, as that can cause infinite recursion of on_overheat
    /// events.
    ///
    /// **Warning:** The context passed here is intentionally a base perovskite HandlerContext, not a circuit handler context.
    /// Making a new circuit handler context is a good way to cause infinite recursion or a deadlock.
    fn on_overheat(&self, _ctx: &HandlerContext, _coord: BlockCoordinate) {
        // By default, do nothing
    }
}

/// Get the connections being made to the given block at the given coordinate.
///
/// This includes only those connections that actually connect to a block which is able to connect back.
#[inline]
pub fn get_live_connectivities(
    ctx: &CircuitHandlerContext<'_>,
    coord: BlockCoordinate,
) -> smallvec::SmallVec<[(BlockConnectivity, BlockCoordinate); 8]> {
    let circuit_extension = ctx.circuits_ext();

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
            tracing::warn!(
                "block {:?} has no circuit properties but was registered with the circuits plugin",
                block_id
            );
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
            result.push((*connectivity, neighbor_coord));
        }
    }
    result
}

/// Get the connections being made to the given block at the given coordinate.
///
/// This includes only those connections that actually connect to a block which is able to connect back.
#[inline]
pub fn get_incoming_pin_states(
    ctx: &CircuitHandlerContext<'_>,
    coord: BlockCoordinate,
) -> smallvec::SmallVec<[(BlockConnectivity, BlockCoordinate, PinState); 8]> {
    let circuit_extension = ctx.circuits_ext();

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
            tracing::warn!(
                "block {:?} has no circuit properties but was registered with the circuits plugin",
                block_id
            );
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
            let state = match get_pin_state(ctx, neighbor_coord, coord) {
                PinReading::Valid(x) => x,
                _ => continue,
            };
            result.push((*connectivity, neighbor_coord, state));
        }
    }
    result
}

/// Private state for the circuits plugin
struct CircuitGameStateExtension {
    basic_properties: FxHashMap<u32, CircuitBlockProperties>,
    callbacks: FxHashMap<u32, Box<dyn CircuitBlockCallbacks>>,
    // Used for fast graph traversal for net/wire updates
    // Sentinel: if basic_wire_off == basic_wire_on, then the wires were
    // not registered properly
    basic_wire_off: BlockId,
    basic_wire_on: BlockId,
}
impl GameStateExtension for CircuitGameStateExtension {}

pub trait CircuitGameBuilder {
    /// Register a block with the circuits plugin. This must be done in addition to actually
    /// injecting circuits related callbacks into the block itself.
    fn define_circuit_callbacks(
        &mut self,
        block_id: BlockId,
        callbacks: Box<dyn CircuitBlockCallbacks>,
        properties: CircuitBlockProperties,
    ) -> Result<()>;
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
    fn define_circuit_callbacks(
        &mut self,
        block_id: BlockId,
        callbacks: Box<dyn CircuitBlockCallbacks>,
        properties: CircuitBlockProperties,
    ) -> Result<()> {
        let ext = self.builder_extension::<CircuitGameBuilderExt>();
        let state = ext
            .resulting_state
            .as_mut()
            .expect("pre_run already called");
        if state.basic_properties.contains_key(&block_id.base_id()) {
            return Err(anyhow::anyhow!(
                "block {:?} has already been registered",
                block_id
            ));
        }
        if state.callbacks.contains_key(&block_id.base_id()) {
            return Err(anyhow::anyhow!(
                "block {:?} has already been registered",
                block_id
            ));
        }
        state
            .basic_properties
            .insert(block_id.base_id(), properties);
        state.callbacks.insert(block_id.base_id(), callbacks);
        Ok(())
    }
}

/// Initialize the circuits plugin's content:
/// - makes basic circuits available to players
/// - makes other functions in this trait available to plugins that need them
pub fn register_circuits(builder: &mut GameBuilder) -> Result<()> {
    if builder
        .builder_extension::<CircuitGameBuilderExt>()
        .initialized
    {
        return Ok(());
    }
    builder
        .builder_extension::<CircuitGameBuilderExt>()
        .initialized = true;

    builder
        .inner
        .blocks_mut()
        .register_fast_block_group(constants::CIRCUITS_GROUP);

    wire::register_wire(builder)?;
    simple_blocks::register_simple_blocks(builder)?;
    gates::register_base_gates(builder)?;
    switches::register_switches(builder)?;
    Ok(())
}

impl CircuitGameBuilerPrivate for GameBuilder {
    fn register_wire_private(
        &mut self,
        on: BuiltBlock,
        off: BuiltBlock,
        properties: CircuitBlockProperties,
    ) -> Result<()> {
        fn make_wire_callbacks(
            block_id: BlockId,
            state: PinState,
        ) -> Box<dyn CircuitBlockCallbacks> {
            Box::new(wire::WireCallbacksImpl {
                base_id: block_id,
                state,
            })
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
            .insert(on.id.base_id(), make_wire_callbacks(on.id, PinState::High));
        inner
            .callbacks
            .insert(off.id.base_id(), make_wire_callbacks(off.id, PinState::Low));
        inner.basic_wire_off = off.id;
        inner.basic_wire_on = on.id;

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
        let state = self.resulting_state.take().expect("pre_run already called");
        if state.basic_wire_off == state.basic_wire_on {
            panic!("Circuit plugin startup incomplete: basic_wire_off == basic_wire_on");
        }
        server_builder.add_extension(state);
    }
}
impl Default for CircuitGameBuilderExt {
    fn default() -> Self {
        Self {
            initialized: false,
            resulting_state: Some(CircuitGameStateExtension {
                basic_properties: FxHashMap::default(),
                callbacks: FxHashMap::default(),
                basic_wire_off: BlockId(0),
                basic_wire_on: BlockId(0),
            }),
        }
    }
}

pub trait CircuitBlockBuilder {
    /// Adds circuits callbacks to the block being built.
    ///
    /// These callbacks will call into the circuits plugin and also delegate
    /// to the existing callbacks already registered on this builder.
    ///
    /// [`define_circuit_callbacks`] should be called on the game builder to also
    /// define the block's circuit-speciifc behavior.
    fn register_circuit_callbacks(self) -> BlockBuilder;
}

impl CircuitBlockBuilder for BlockBuilder {
    fn register_circuit_callbacks(self) -> BlockBuilder {
        self.add_modifier(Box::new(move |bt| {
            let old_inline_handler: Option<Box<InlineHandler>> = bt.dig_handler_inline.take();
            let old_full_handler: Option<Box<FullHandler>> = bt.dig_handler_full.take();
            let block_name = FastBlockName::new(bt.client_info.short_name.clone());

            let checked_inline_handler: Box<InlineHandler> = match old_inline_handler {
                None => Box::new(move |_, _, _, _| Ok(BlockInteractionResult::default())),
                Some(x) => Box::new(move |ctx, block_type: &mut BlockId, ext_data, stack| {
                    let expected_id = ctx
                        .block_types()
                        .resolve_name(&block_name)
                        .context("We're in a handler, but the block type isn't registered?")?;
                    if block_type.equals_ignore_variant(expected_id) {
                        x(ctx, block_type, ext_data, stack)
                    } else {
                        Ok(BlockInteractionResult::default())
                    }
                }),
            };
            bt.dig_handler_full = Some(Box::new(move |ctx, coord, tool_stack| {
                // Run the initial handler first, so that the map is updated before the circuit
                // callbacks are invoked
                // However, we need the old ID to figure out which nearby blocks need updates
                let block_id = ctx.game_map().get_block(coord)?;

                let mut result = ctx.game_map().mutate_block_atomically(coord, |id, ext| {
                    let ictx = ctx.game_map().make_inline_context(
                        coord,
                        ctx.initiator(),
                        ctx.deref(),
                        ctx.tick(),
                    );
                    checked_inline_handler(ictx, id, ext, tool_stack)
                })?;

                result += match old_full_handler.as_deref() {
                    Some(old_handler) => old_handler(ctx, coord, tool_stack)?,
                    None => BlockInteractionResult::default(),
                };

                if let Err(e) = dispatch::dig(block_id, coord, &events::make_root_context(ctx)) {
                    tracing::error!("Error in circuits dig handler: {}", e);
                }
                Ok(result)
            }));

            bt.client_info
                .groups
                .push(constants::CIRCUITS_GROUP.to_string());
        }))
        .add_item_modifier(Box::new(|item| {
            let old_handler = item.place_on_block_handler.take();
            item.place_on_block_handler = Some(Box::new(move |ctx, coord, anchor, tool_stack| {
                let result = match old_handler.as_deref() {
                    Some(old_handler) => old_handler(ctx, coord, anchor, tool_stack),
                    None => Ok(None),
                };
                if let Err(e) = dispatch::place(coord, &events::make_root_context(ctx)) {
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

    use crate::circuits::events::transmit_edge;

    use super::{
        constants::CIRCUITS_GROUP, events::CircuitHandlerContext, CircuitGameStateExtension,
    };

    pub(crate) fn dig(
        old_block_id: BlockId,
        coord: BlockCoordinate,
        ctx: &CircuitHandlerContext<'_>,
    ) -> Result<()> {
        let ext = ctx.circuits_ext();
        let circuits_group = ctx
            .block_types()
            .fast_block_group(CIRCUITS_GROUP)
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
            transmit_edge(ctx, neighbor, coord, super::PinState::Low)?;
        }

        Ok(())
    }

    pub(crate) fn place(coord: BlockCoordinate, ctx: &CircuitHandlerContext<'_>) -> Result<()> {
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
            .fast_block_group(CIRCUITS_GROUP)
            .expect("circuits group not found");
        let connectivity = match ext.basic_properties.get(&block_id.base_id()) {
            Some(properties) => &properties.connectivity,
            None => {
                tracing::warn!("block {:?} has no circuit properties but was registered with the circuits plugin", block_id);
                return Ok(());
            }
        };
        let callbacks = match ext.callbacks.get(&block_id.base_id()) {
            Some(x) => x,
            None => {
                tracing::warn!(
                    "block {:?} has no circuit callbacks but had the circuits group",
                    block_id
                );
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
            let outbound_state = callbacks.sample_pin(ctx, coord, neighbor);
            transmit_edge(ctx, neighbor, coord, outbound_state)?;
            let inbound_state = neighbor_callbacks.sample_pin(ctx, neighbor, coord);
            transmit_edge(ctx, coord, neighbor, inbound_state)?;
        }
        callbacks.update_connectivity(ctx, coord)?;

        Ok(())
    }
}

/// The circuits plugin requires events to be associated with a context, which tracks
/// who initiated the event as well as (yet to be defined) ratelimiting/anti-DoS/recursion limiting
/// measures.
pub mod events {
    use std::ops::Deref;

    use anyhow::{Error, Result};
    use env_logger::builder;

    use perovskite_core::coordinates::BlockCoordinate;
    use perovskite_server::game_state::event::HandlerContext;

    use super::{wire, BusMessage, CircuitGameStateExtension, PinState};

    /// A context to be used for circuits related callbacks.
    pub struct CircuitHandlerContext<'a> {
        pub inner: &'a HandlerContext<'a>,
        pub ttl: u32,
    }
    impl<'a> Deref for CircuitHandlerContext<'a> {
        type Target = HandlerContext<'a>;
        fn deref(&self) -> &Self::Target {
            self.inner
        }
    }
    impl CircuitHandlerContext<'_> {
        pub fn consume_ttl(&self) -> Option<Self> {
            if self.ttl == 0 {
                None
            } else {
                Some(CircuitHandlerContext {
                    inner: self.inner,
                    ttl: self.ttl - 1,
                })
            }
        }
        pub fn consume_multiple_ttl(&self, count: u32) -> Option<Self> {
            if self.ttl < count {
                None
            } else {
                Some(CircuitHandlerContext {
                    inner: self.inner,
                    ttl: self.ttl.checked_sub(count).unwrap(),
                })
            }
        }

        pub(super) fn circuits_ext(&self) -> &CircuitGameStateExtension {
            self.inner
                .extension::<CircuitGameStateExtension>()
                .expect("circuits extension not found")
        }
    }

    const DEFAULT_TTL: u32 = 256;

    /// Creates a new root context for the circuits plugin
    pub fn make_root_context<'a>(ctx: &'a HandlerContext) -> CircuitHandlerContext<'a> {
        CircuitHandlerContext {
            inner: ctx,
            ttl: DEFAULT_TTL,
        }
    }

    /// Signals that a block's digital output is (potentially) changing.
    /// This can be for any reason, such as a button being pressed, a sensor's
    /// state changing, a timer running, etc.
    ///
    /// **Important:** If a block has had register_circuit_callbacks called on it, then the dig handler
    /// will already take care of this. Likewise for the place handler.
    ///
    /// If the connection is made directly to another block, an event will be delivered
    /// directly to it. If the connection is made to a wire, the wire will be recalculated
    /// and any connected inputs will be signalled if the net's state transitioned.
    ///
    /// If edge is set to Rising or Falling, then net recalculations may be skipped as an
    /// optimization. It's always safe to set `[EdgeType::Unknown]`.
    ///
    /// Args:
    ///     dest_coord: The coordinate of the block into which the changing output is connected
    ///     connection: The block that signalled the edge
    ///     edge: The state of the pin that triggered this notification.
    pub fn transmit_edge(
        ctx: &CircuitHandlerContext<'_>,
        dest_coord: BlockCoordinate,
        from_coord: BlockCoordinate,
        new_state: PinState,
    ) -> Result<()> {
        transmit_edge_inner(ctx, dest_coord, from_coord, new_state, None)
    }

    pub fn transmit_bus_message(
        ctx: &CircuitHandlerContext<'_>,
        dest_coord: BlockCoordinate,
        from_coord: BlockCoordinate,
        pin_state: PinState,
        message: BusMessage,
    ) -> Result<()> {
        transmit_edge_inner(ctx, dest_coord, from_coord, pin_state, Some(message))
    }

    fn transmit_edge_inner(
        ctx: &CircuitHandlerContext,
        dest_coord: BlockCoordinate,
        from_coord: BlockCoordinate,
        new_state: PinState,
        bus_message: Option<BusMessage>,
    ) -> Result<(), Error> {
        // TODO: Detect when new_state doesn't match what we get from sampling
        // this would cause a mismatch depending on whether there is an intervening wire or not
        let next_ctx = match ctx.consume_ttl() {
            Some(x) => x,
            None => {
                send_device_overheat(ctx, from_coord);
                return Ok(());
            }
        };

        let dest = match ctx.game_map().try_get_block(dest_coord) {
            Some(x) => x,
            None => return Ok(()),
        };
        let circuits_ext = ctx.circuits_ext();
        // Specialized fast path when we're driving a wire
        if dest.base_id() == circuits_ext.basic_wire_off.0
            || dest.base_id() == circuits_ext.basic_wire_on.0
        {
            if let Err(e) = wire::recalculate_wire(
                &next_ctx,
                dest_coord,
                from_coord,
                new_state,
                bus_message.as_ref(),
            ) {
                tracing::error!("failed to recalculate wire: {}", e);
            }
            return Ok(());
        }
        let dest_callbacks = match circuits_ext.callbacks.get(&dest.base_id()) {
            Some(x) => x,
            None => {
                // Delivering an edge to a non-circuits block
                // We could check if the block is in the circuits group here, but that's
                // probably not worth it
                return Ok(());
            }
        };
        dest_callbacks.on_incoming_edge(&next_ctx, dest_coord, from_coord, new_state)?;
        if let Some(message) = bus_message.as_ref() {
            dest_callbacks.on_bus_message(&next_ctx, dest_coord, from_coord, message)?;
        }

        Ok(())
    }

    pub(super) fn send_device_overheat(ctx: &CircuitHandlerContext<'_>, coord: BlockCoordinate) {
        let block = match ctx.game_map().try_get_block(coord) {
            Some(x) => x,
            None => {
                tracing::warn!(
                    "We're in a block's overheat handler, but its chunk (coord {:?}) is unloaded",
                    coord
                );
                return;
            }
        };
        let circuits_ext = ctx.circuits_ext();
        let callbacks = match circuits_ext.callbacks.get(&block.base_id()) {
            Some(x) => x,
            None => {
                tracing::warn!(
                    "We're in a block's overheat handler, but the block ({:?}) has no circuit callbacks", block
                );
                return;
            }
        };
        callbacks.on_overheat(ctx.inner, coord);
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
/// Result of [`get_pin_state`]
pub enum PinReading {
    /// Pin was successfully read, and has this state
    Valid(PinState),
    /// The block could not be read (e.g. unloaded chunk, edge of map, etc)
    CantGetBlock,
    /// The block at that location is not a circuit block. Note that this is currently not returned
    /// when that block is a circuit block but lacks the necessary connectivity in its connectivity table.
    NotCircuitBlock,
    /// The ttl of the context ran out
    TtlExceeded,
}

/// Returns the state of the pin at the given coordinate.
///
/// Args:
///     ctx: The circuit context for this operation
///     coord: The coordinate of the block whose output is being sampled
///     into: The coordinate of the block that the output in question is connected to
/// Returns:
///     The state of the pin
pub fn get_pin_state(
    ctx: &CircuitHandlerContext<'_>,
    coord: BlockCoordinate,
    into: BlockCoordinate,
) -> PinReading {
    let block = match ctx.game_map().try_get_block(coord) {
        Some(block) => block,
        None => {
            return PinReading::CantGetBlock;
        }
    };
    let callbacks = match ctx.circuits_ext().callbacks.get(&block.base_id()) {
        Some(callbacks) => callbacks,
        None => {
            return PinReading::NotCircuitBlock;
        }
    };
    let next_ctx = match ctx.consume_ttl() {
        Some(x) => x,
        None => {
            return PinReading::TtlExceeded;
        }
    };
    PinReading::Valid(callbacks.sample_pin(&next_ctx, coord, into))
}
