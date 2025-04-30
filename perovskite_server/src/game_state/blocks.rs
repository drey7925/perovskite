// Copyright 2023 drey7925
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

use anyhow::{bail, ensure, Context, Result};
use log::{info, warn};
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;
use std::{
    any::Any,
    borrow::Borrow,
    collections::{hash_map::Entry, HashMap, HashSet},
    ops::{AddAssign, Deref, DerefMut},
    sync::atomic::{AtomicU32, Ordering},
};

use crate::database::database_engine::{GameDatabase, KeySpace};

use super::{
    client_ui::Popup,
    event::{EventInitiator, HandlerContext},
    inventory::Inventory,
    items::{Item, ItemManager, ItemStack},
};
use perovskite_core::{
    block_id::{special_block_defs::AIR_ID, BlockError, BlockId},
    constants::{
        block_groups::{self, DEFAULT_GAS, TRIVIALLY_REPLACEABLE},
        blocks::AIR,
    },
    coordinates::BlockCoordinate,
    protocol::blocks as blocks_proto,
};
use prost::Message;

pub type CustomData = Box<dyn Any + Send + Sync + 'static>;
pub struct ExtendedData {
    /// In-memory extended data that may be associated with a block at a
    /// particular location. Use `downcast_ref` to try to get the inner data
    /// of the Any as a concrete type.
    pub custom_data: Option<CustomData>,

    /// Simple key-value storage associated with a block.
    pub simple_data: HashMap<String, String>,

    /// Inventories that can be shown in inventory views
    /// Note that nothing here prevents you from messing with inventories that are still
    /// visible in popups; they will be stale until client refreshes the popup view
    /// TODO: switch back to std hashmap/btreemap once get_many_mut is stabilized
    pub inventories: hashbrown::HashMap<String, Inventory>,
}
impl Default for ExtendedData {
    fn default() -> Self {
        Self {
            custom_data: None,
            simple_data: HashMap::new(),
            inventories: hashbrown::HashMap::new(),
        }
    }
}

/// The result of interacting with (e.g. digging/tapping) a block.
#[derive(Clone, Debug)]
pub struct BlockInteractionResult {
    /// The item stacks obtained by the player
    pub item_stacks: Vec<ItemStack>,
    /// The wear of the tool that the player used
    pub tool_wear: u32,
}
// We want to make the default value explicit, and it may not always be derivable
#[allow(clippy::derivable_impls)]
impl Default for BlockInteractionResult {
    fn default() -> Self {
        Self {
            item_stacks: vec![],
            tool_wear: 0,
        }
    }
}
impl AddAssign for BlockInteractionResult {
    fn add_assign(&mut self, other: Self) {
        self.item_stacks.extend(other.item_stacks);
        self.tool_wear += other.tool_wear;
    }
}

pub use perovskite_core::block_id::MAX_BLOCK_DEFS;
use perovskite_core::protocol::blocks::SolidPhysicsInfo;
use perovskite_core::protocol::map::ClientExtendedData;

/// Takes (handler context, coordinate being dug, item stack used to dig), returns dropped item stacks.
pub type FullHandler = dyn Fn(&HandlerContext, BlockCoordinate, Option<&ItemStack>) -> Result<BlockInteractionResult>
    + Send
    + Sync;
/// Takes (handler context, mutable reference to the block type in the map,
/// mutable reference to the extended data holder, item stack used to dig), returns dropped item stacks.
pub type InlineHandler = dyn Fn(
        InlineContext,
        &mut BlockId,
        &mut ExtendedDataHolder,
        Option<&ItemStack>,
    ) -> Result<BlockInteractionResult>
    + Send
    + Sync;

pub type ColdLoadPostprocessor = dyn Fn(&mut [BlockId; 4096]) + Send + Sync + 'static;

/// In-memory representation of the different types of blocks that can exist in the world.
///
/// # Handlers
/// Some handlers exist in two forms: foo_handler_full and foo_handler_inline.
///
/// * The full handler is provided access to the entire game map and game state, but as a result,
/// it is inherently racy in order to avoid deadlocks - your handler runs without holding a lock
/// on the chunk or block it was called on, so by the time you call game map functions, *including
/// functions on the block for which the handler was invoked*, it might have been changed.
/// * The inline handler is provided access to only the block to which it pertains, but it is atomic
/// on the server - the block will not change between when the blocktype/handler are looked up, and when
/// it is actually run.
///
/// Note that both handlers are run if present.
///
/// Handlers should prefer to return non-Ok results rather than panicking where possible. The
/// exact behavior for a panic in a handler is still TBD and may include data loss (i.e. recent actions
/// may be buffered in memory but never be written to the database on a panic).
#[allow(clippy::type_complexity)] // Hard to factor types while trait aliases are unstable
pub struct BlockType {
    /// All block details that are common between the client and the server
    pub client_info: blocks_proto::BlockTypeDef,

    /// Called when the block is being loaded into memory. Should be a pure function
    /// and as fast as possible. Typically, this should simply be a wrapper around a protobuf
    /// deserialization step or similar. This callback does not need to handle any block inventories;
    /// they are handled automatically by the engine.
    ///
    /// Only called if extended_data_handling is not None and there's serialized extended data
    /// for this block.
    ///
    /// If this returns an Err, the game will crash since this represents data loss due to corrupt data
    /// in a chunk.
    pub deserialize_extended_data_handler:
        Option<Box<dyn Fn(InlineContext, &[u8]) -> Result<Option<CustomData>> + Send + Sync>>,
    /// Called when the block is being written back to disk. Should be a pure function
    /// and as fast as possible. It should typically be a wrapper around a protobuf serialization
    /// step or similar. This callback does not need to handle any block inventories; they are
    /// handled by the engine.
    ///
    /// Only called if the extended data itself is Some. If this returns None, no extended data will
    /// be serialized to storage - an example use for this would be to avoid the storage overhead if
    /// the extended data is *logically* empty.
    ///
    /// If this returns an Err, the game will crash since this represents data loss. Note that this
    /// can cause other changes to a map chunk to be lost.
    ///
    /// Note that this is likely to be slower on the client as well, especially if it impacts block
    /// rendering/appearance (TBD if this is the case)
    pub serialize_extended_data_handler:
        Option<Box<dyn Fn(InlineContext, &CustomData) -> Result<Option<Vec<u8>>> + Send + Sync>>,
    // NOT YET IMPLEMENTED
    // /// If extended_data_handling is one of the client-side options, called whenever the block
    // /// needs to be sent to a client (subject to the caching policy, see doc for `ExtDataHandling`)
    // ///
    // /// The function is called with the extended data obtained from deserialize_extended_data_handler
    // /// as well as any inventories.
    // ///
    // /// If the response is Some(...), all fields other than id and short_name override the defaults
    // /// for this block.
    // ///
    // /// Should be as fast as possible.
    // ///
    // /// If this returns an Err, the game will crash since this represents data loss due to corrupt data
    // /// in a chunk.
    //  pub extended_data_to_client_side: Option<
    //     Box<
    //         dyn Fn(InlineContext, &ExtendedData) -> Option<blocks_proto::BlockTypeDef>
    //             + Send
    //             + Sync,
    //     >,
    // >,
    /// Called when the client needs this block, and client_info.has_client_extended_data is true.
    ///
    /// This should be fast. If handler is None or return value is Ok(None), no client-side extended
    /// data is sent. The return value does *not* need offset_in_chunk specified; the engine will
    /// overwrite it
    ///
    /// If this returns an Err, the game will crash (the exact semantics from recovering from the
    /// error are not yet clear)
    pub make_client_extended_data: Option<
        Box<
            dyn Fn(InlineContext, &ExtendedData) -> Result<Option<ClientExtendedData>>
                + Send
                + Sync,
        >,
    >,
    /// Called when the block is dug. This function (or dig_handler_inline) should explicitly remove the block (i.e. replace it with air)
    /// if it should be removed from the map when dug.
    ///
    /// See notes in the header for [`BlockType`] for details about this vs [`BlockType::dig_handler_inline`]. Note that if both
    /// _inline and _full return item stacks, they will be merged.
    ///
    /// If this returns Err, a message will be logged and the user that dug the block will get an error message in some
    /// TBD way (for now, just a log message on the client, but eventually a popup or similar)
    ///
    /// Note that if the handler performs changes, they will not be rolled back if the handler subsequently returns Err.
    ///
    /// Also note that if the item has its own dig handler, that item's dig handler is responsible for calling the
    /// dig_block on the map. The item does not need to dig the same block that was originally dug, e.g. it may dig no block,
    /// a different block than the originally dug one, or multiple blocks.
    ///
    /// **Important:** This is racy, and by the time the handler is invoked, the block may have changed.
    /// Furthermore, if the inline handler is also set and does change the block, that inline handler's effect
    /// will also be visible.
    /// However, if this handler is called, it is guaranteed that at some recent point, this block *was* at the given
    /// coordinates (even if changed by a race or by its own inline handler), and someone tried to dig it.
    pub dig_handler_full: Option<Box<FullHandler>>,
    /// Called when the block is dug. To update the block on the map, assign a new value to the BlockTypeRef.
    /// To update extended data, use the DerefMut trait on ExtendedDataHolder
    ///
    /// If this returns Err, a message will be logged, and the user that dug the block will get an error message in some
    /// TBD way. dig_handler_full will not be called in that case
    ///
    /// Note that if the handler performs changes, they will not be rolled back if the handler subsequently returns Err.
    pub dig_handler_inline: Option<Box<InlineHandler>>,
    /// Same as dig_handler_full but for the case when the block is hit with a tool but not dug fully
    pub tap_handler_full: Option<Box<FullHandler>>,
    /// Same as dig_handler_inline but for the case when the block is hit with a tool but not dug fully
    pub tap_handler_inline: Option<Box<InlineHandler>>,

    /// Called when the given item is placed onto this block.
    ///
    /// The second parameter indicates the coordinate
    ///
    /// The third parameter indicates the item being placed.
    /// Note that this is invoked when the place key/button is placed while pointing to this block; it is not
    /// necessarily true that the placement is vertically above this block.
    ///
    /// If this returns Err, a message will be logged and the user that dug the block will get an error message in some
    /// TBD way (for now, just a log message on the client, but eventually a popup or similar)
    ///
    /// NOT YET IMPLEMENTED, Signature subject to change
    pub place_upon_handler: Option<
        Box<dyn Fn(HandlerContext, BlockCoordinate, Option<Item>) -> Result<()> + Send + Sync>,
    >,
    /// Called when the interact key is pressed and an interaction is selected.
    ///
    /// This can mutate the given block (using handlercontext) if desired, and return a popup (if desired)
    ///
    /// The signature of this callback is subject to change.
    ///
    /// Args:
    ///     * HandlerContext: Handler context giving access to the player/initator, game state, etc
    ///     * BlockCoordinate: Coordinate that was interacted with
    ///     * &str: Indicates the name of the named interact key option. Empty for the unnamed default
    ///         handler used when no named handlers are present. The string is passed from the client, and is
    ///         not validated or sanitized.
    ///         Assuming a proper client, this will be one of the entries in interact_key_option in the block proto
    pub interact_key_handler: Option<
        Box<dyn Fn(HandlerContext, BlockCoordinate, &str) -> Result<Option<Popup>> + Send + Sync>,
    >,

    // Internal impl details
    pub(crate) is_unknown_block: bool,
}

impl BlockType {
    pub(crate) fn id(&self, variant: u16) -> Result<BlockId> {
        BlockId::new(self.client_info.id, variant)
    }

    pub fn short_name(&self) -> &str {
        &self.client_info.short_name
    }
}

impl Default for BlockType {
    fn default() -> Self {
        Self {
            client_info: Default::default(),
            deserialize_extended_data_handler: None,
            serialize_extended_data_handler: None,
            make_client_extended_data: None,
            dig_handler_full: None,
            dig_handler_inline: None,
            tap_handler_full: None,
            tap_handler_inline: None,
            place_upon_handler: None,
            interact_key_handler: None,
            is_unknown_block: false,
        }
    }
}

/// Represents extended data for a block on the map.
///
/// # Data loss warning:
/// The dirty bit *must* be set after making meaningful changes to the extended data to avoid
/// data loss (by the game map not writing anything back)
///
/// This is done automatically when accessing the given extended data using a mutable reference,
/// since [`std::ops::DerefMut`] will mark the dirty bit - this includes constructions like:
/// ```rust
/// # use perovskite_core::coordinates::BlockCoordinate;
/// let server = perovskite_server::server::testonly_in_memory().unwrap();
/// # server.run_task_in_server(|gs| {
/// # let coord = BlockCoordinate::new(0, 0, 0);
/// gs.game_map().mutate_block_atomically(coord, |block, ext| {
///     // `ExtendedDataHolder` will DerefMut to `Option<ExtendedData>`, and DerefMut sets the
///     // dirty bit.
///     let ext_inner = ext.get_or_insert_with(Default::default);
///     ext_inner.simple_data.insert("foo".to_string(), "bar".to_string());
///     Ok(())
/// });
/// #    Ok(())
/// # });
/// ```
///
/// However, there is a risk of data loss if:
/// 1. The extended data contains some sort of interior mutability (e.g. Mutex, RwLock, atomics,
///    lock-free data structures, etc), whether behind an Arc or otherwise
/// 2. That data (behind that interior mutability) should be written back to the game database (i.e.
///    the block's extended data serializer reads it, and it's not just some in-memory cache)
/// 3. The only way the extended data is accessed is via an immutable deref (e.g. `ext.map()`,
///    `ext.and_then()`, `ext.deref()`)
/// 4. And `ExtendedDataHolder::set_dirty` is not called manually.
///
/// In this case, if the chunk is unloaded without any other changes, the changes will be lost since
/// the chunk won't be written back to disk by the game map.
pub struct ExtendedDataHolder<'a> {
    extended_data: &'a mut Option<ExtendedData>,
    dirty: bool,
    // todo cache client-side data
}
impl<'a> ExtendedDataHolder<'a> {
    pub fn new(extended_data: &'a mut Option<ExtendedData>) -> Self {
        Self {
            extended_data,
            dirty: false,
        }
    }
    pub(crate) fn dirty(&self) -> bool {
        self.dirty
    }

    pub fn clear(&mut self) {
        if self.extended_data.is_some() {
            self.dirty = true;
        }
        *self.extended_data = None;
    }
    /// Marks the extended data as needing to be written back to disk.
    ///
    /// Note that in most cases, this is not needed. See the docstring for [ExtendedDataHolder] for
    /// more details.
    pub fn set_dirty(&mut self) {
        self.dirty = true;
    }
}
impl<'a> Deref for ExtendedDataHolder<'a> {
    type Target = Option<ExtendedData>;

    fn deref(&self) -> &Self::Target {
        self.extended_data
    }
}
impl<'a> DerefMut for ExtendedDataHolder<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.dirty = true;
        self.extended_data
    }
}

/// A limited context for serialization and deserialization handlers.
/// These handlers run in critical chunk management paths, under critical locks.
/// As a result, they have very restricted access to context and state.
pub struct InlineContext<'a> {
    pub(crate) tick: u64,
    pub(crate) initiator: EventInitiator<'a>,
    pub(crate) location: BlockCoordinate,
    pub(crate) block_types: &'a BlockTypeManager,
    pub(crate) items: &'a ItemManager,
}

impl<'a> InlineContext<'a> {
    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn location(&self) -> BlockCoordinate {
        self.location
    }

    pub fn initiator(&self) -> &EventInitiator {
        &self.initiator
    }

    pub fn block_types(&self) -> &BlockTypeManager {
        self.block_types
    }

    pub fn items(&self) -> &ItemManager {
        self.items
    }
}

/// Manages all of the different block types defined in this game world.
///
/// This struct owns all of the [`BlockType`]s that are registered with it;
/// they can be accessed using either a [`BlockTypeHandle`], or a [`BlockTypeName`]:
/// * A handle refers to a block that is known to be already registered
/// * A name refers to a block that may or may not be registered yet, using its unique short name.
///
/// For example, block A may replace itself with block B in its dig handler, while block B replaces
/// itself with block A when dug. If block A is registered first, it will need to
/// use a `BlockTypeName` to reference block B (by its short name), since block B has
/// not yet been registered yet, and it will be impossible to make a handle to it.
///
/// Handles and names should be considered effectively immutable (but may internally use mutexes/atomics
/// for lookup caching); they can be freely cloned around as needed.
pub struct BlockTypeManager {
    block_types: Vec<BlockType>,
    name_to_base_id_map: FxHashMap<String, u32>,
    init_complete: bool,

    // Separate copy of BlockType.client_info.allow_light_propagation, packed densely
    // to be more cache-friendly
    light_propagation: bitvec::vec::BitVec,
    trivially_replaceable_block_group: bitvec::vec::BitVec,
    has_client_side_extended_data: bitvec::vec::BitVec,
    fast_block_groups: FxHashMap<String, bitvec::vec::BitVec>,
    cold_load_postprocessors: Vec<Box<ColdLoadPostprocessor>>,
}
impl BlockTypeManager {
    pub(crate) fn new() -> BlockTypeManager {
        BlockTypeManager {
            block_types: vec![make_air_block()],
            name_to_base_id_map: FxHashMap::from_iter([(AIR.to_string(), AIR_ID.0)]),
            init_complete: false,
            light_propagation: bitvec::vec::BitVec::new(),
            has_client_side_extended_data: bitvec::vec::BitVec::new(),
            trivially_replaceable_block_group: bitvec::vec::BitVec::new(),
            fast_block_groups: FxHashMap::default(),
            cold_load_postprocessors: Vec::new(),
        }
    }
    /// Given a handle, return the block.
    pub fn get_block(&self, handle: &BlockTypeHandle) -> Result<(&BlockType, u16)> {
        self.get_block_by_id(*handle)
    }

    pub fn is_trivially_replaceable(&self, id: BlockId) -> bool {
        *self
            .trivially_replaceable_block_group
            .get(id.index())
            .as_deref()
            .unwrap_or(&false)
    }

    pub(crate) fn get_block_by_id(&self, id: BlockId) -> Result<(&BlockType, u16)> {
        let block_type = self
            .block_types
            .get(id.index())
            .with_context(|| BlockError::IdNotFound(id.into()))?;
        Ok((block_type, id.variant()))
    }

    #[inline]
    pub(crate) fn allows_light_propagation(&self, id: BlockId) -> bool {
        if id.index() < self.light_propagation.len() {
            self.light_propagation[id.index()]
        } else {
            // unknown blocks don't propagate light
            false
        }
    }

    #[inline]
    pub(crate) fn has_client_side_extended_data(&self, id: BlockId) -> bool {
        if id.index() < self.has_client_side_extended_data.len() {
            self.has_client_side_extended_data[id.index()]
        } else {
            // Unknown blocks don't have client-side extended data
            false
        }
    }

    /// Registers a new block in this block type manager, and returns a handle to it.
    ///
    /// Returns an error if another block is already registered with the same short name, or
    /// if there are too many blocktypes (up to roughly 1 million BlockTypes can be registered)
    pub fn register_block(&mut self, mut block: BlockType) -> Result<BlockTypeHandle> {
        let id = match self
            .name_to_base_id_map
            .entry(block.short_name().to_string())
        {
            Entry::Occupied(x) => {
                let id = BlockId(*x.get());
                let existing = &mut self.block_types[id.index()];
                ensure!(
                    existing.is_unknown_block,
                    "Block with this name already registered: {}",
                    &block.short_name()
                );

                block.client_info.id = id.base_id();
                info!("Registering block {} as {:?}", block.short_name(), id);
                *existing = block;
                id
            }
            Entry::Vacant(x) => {
                let new_id = BlockId(
                    (self.block_types.len() << 12)
                        .try_into()
                        .with_context(|| BlockError::TooManyBlocks)?,
                );
                if new_id.index() >= MAX_BLOCK_DEFS {
                    bail!(BlockError::TooManyBlocks);
                }
                info!(
                    "Registering new block {} as {:?}",
                    block.short_name(),
                    new_id
                );
                block.client_info.id = new_id.base_id();
                self.block_types.push(block);
                x.insert(new_id.base_id());
                new_id
            }
        };

        Ok(id)
    }

    pub(crate) fn from_proto(
        block_proto: blocks_proto::ServerBlockTypeAssignments,
    ) -> Result<BlockTypeManager> {
        let mut manager = BlockTypeManager {
            block_types: Vec::new(),
            name_to_base_id_map: FxHashMap::default(),
            init_complete: false,
            light_propagation: bitvec::vec::BitVec::new(),
            trivially_replaceable_block_group: bitvec::vec::BitVec::new(),
            has_client_side_extended_data: bitvec::vec::BitVec::new(),
            fast_block_groups: FxHashMap::default(),
            cold_load_postprocessors: Vec::new(),
        };
        let max_index = block_proto
            .block_type
            .iter()
            .map(|x| BlockId(x.id).index())
            .max()
            .unwrap_or(0);

        if max_index >= MAX_BLOCK_DEFS {
            bail!(BlockError::TooManyBlocks);
        }

        let present_indices: HashSet<usize> =
            HashSet::from_iter(block_proto.block_type.iter().map(|x| BlockId(x.id).index()));
        for i in 0..=max_index {
            let unknown_block_name = format!("by_id:0x{:x}", i << 12);
            manager.block_types.push(make_unknown_block_serverside(
                &manager,
                BlockId((i as u32) << 12),
                unknown_block_name.clone(),
            ));
            if !present_indices.contains(&i) {
                warn!(
                    "Creating unknown block {} with id 0x{:x}",
                    unknown_block_name,
                    i << 12
                );
                manager
                    .name_to_base_id_map
                    .insert(unknown_block_name, (i << 12) as u32);
            }
        }
        for assignment in block_proto.block_type.iter() {
            let block_id = BlockId(assignment.id);
            ensure!(
                block_id.variant() == 0,
                BlockError::VariantBitsNonzero(assignment.id)
            );
            manager
                .name_to_base_id_map
                .insert(assignment.short_name.clone(), block_id.base_id());
        }
        manager.block_types[AIR_ID.index()] = make_air_block();
        ensure!(manager.name_to_base_id_map.get(AIR).map(|&x| BlockId(x)) == Some(AIR_ID));
        Ok(manager)
    }

    // TODO - this is a bottleneck for the number of blocks that can be defined due to
    // database and proto size constraints
    pub(crate) fn to_proto(&self) -> blocks_proto::ServerBlockTypeAssignments {
        let mut result = Vec::new();
        for (name, &id) in self.name_to_base_id_map.iter() {
            result.push(blocks_proto::BlockTypeAssignment {
                short_name: name.clone(),
                id,
            })
        }
        blocks_proto::ServerBlockTypeAssignments { block_type: result }
    }

    pub(crate) fn to_client_protos(&self) -> Vec<blocks_proto::BlockTypeDef> {
        self.block_types
            .iter()
            .map(|block| block.client_info.clone())
            .collect()
    }

    pub(crate) fn create_or_load(db: &dyn GameDatabase) -> Result<BlockTypeManager> {
        match db.get(&KeySpace::Metadata.make_key(BLOCK_MANAGER_META_KEY_LEGACY))? {
            Some(x) => {
                let result = BlockTypeManager::from_proto(
                    blocks_proto::ServerBlockTypeAssignments::decode(x.borrow())?,
                )?;
                info!(
                    "Loaded block type manager from database with {} definitions",
                    result.block_types.len()
                );
                Ok(result)
            }
            None => {
                info!("Creating new block type manager");
                Ok(BlockTypeManager::new())
            }
        }
    }
    pub(crate) fn save_to(&self, db: &dyn GameDatabase) -> Result<()> {
        db.put(
            &KeySpace::Metadata.make_key(BLOCK_MANAGER_META_KEY_LEGACY),
            &self.to_proto().encode_to_vec(),
        )?;
        db.flush()
    }

    /// Creates a BlockTypeName that refers to the indicated block. See the doc for
    /// [`BlockTypeManager`] and [`FastBlockName`] for an example of where this should be used.
    pub fn make_block_name(&self, name: impl Into<String>) -> FastBlockName {
        FastBlockName {
            name: name.into(),
            base_id: AtomicU32::new(u32::MAX),
        }
    }

    /// Tries to resolve a BlockTypeName from `make_block_name()`. If a block with a matching
    /// name was registered, returns a handle to it. Otherwise, returns None.
    pub fn resolve_name(&self, block_name: &FastBlockName) -> Option<BlockId> {
        let cached = block_name.base_id.load(Ordering::Relaxed);
        if cached == u32::MAX {
            // Need to fill the cache.
            // Multiple threads might race, but they should set the same value anyway
            if let Some(&id) = self.name_to_base_id_map.get(&block_name.name) {
                block_name.base_id.store(id, Ordering::Relaxed);
                Some(id.into())
            } else {
                None
            }
        } else {
            Some(cached.into())
        }
    }

    /// Tries to get a block by name.
    ///
    /// Equivalent to calling make_block_name and then resolve_name back to back, but without the
    /// caching benefits of FastBlockName
    pub fn get_by_name(&self, block_name: &str) -> Option<BlockTypeHandle> {
        if let Some(&id) = self.name_to_base_id_map.get(block_name) {
            Some(id.into())
        } else {
            None
        }
    }

    /// Registers the name of a block group to be available using [`block_group`].
    ///
    /// Note that not all block groups need to be registered this way. However, if
    /// a plugin expects to check whether a block_id is in a group in a performance-critical
    /// situation (e.g. tight loop), using [`has_block_group`] will avoid multiple hashtable
    /// lookups and vector scans in the normal block ID -> block def -> group list process.
    pub fn register_fast_block_group(&mut self, block_group: &str) {
        // Pre-populate an empty bitvec, which we'll fill during startup
        self.fast_block_groups
            .insert(block_group.to_string(), bitvec::vec::BitVec::new());
    }

    /// Returns a block group by name, or None if the block group is not registered.
    ///
    /// The block group must have been registered using [`register_block_group`]. Simply
    /// being defined in a block type is not enough.
    ///
    /// This function will return an empty, nonsensical object if called before the game starts up.
    pub fn fast_block_group(&self, block_group: &str) -> Option<FastBlockGroup> {
        self.fast_block_groups
            .get(block_group)
            .map(|x| FastBlockGroup { blocks: x })
    }

    /// Registers a function to run on block data whenever a chunk is loaded after it was last
    /// modified in a *past* startup of the server (i.e. chunks that were evicted from the
    /// cache during this run, and are now being reloaded do NOT trigger this function).
    ///
    /// This function must ABSOLUTELY be fast - it will be called in the hot path of the load, and
    /// is in the critical path of the thread that triggered the chunk load AND any threads that are
    /// blocked on the condition variable for the same chunk.
    ///
    /// It is currently **undefined** whether this occurs before or after extended data parsing
    /// (this makes a distinction since the block ID is used to select the parsing function)
    pub fn register_cold_load_postprocessor(&mut self, postprocessor: Box<ColdLoadPostprocessor>) {
        self.cold_load_postprocessors.push(postprocessor);
    }

    pub(crate) fn cold_load_postprocessors(&self) -> &[Box<ColdLoadPostprocessor>] {
        &self.cold_load_postprocessors
    }

    // Performs some last setup before the block manager becomes immutable.
    pub(crate) fn pre_build(&mut self) -> Result<()> {
        for (name, group) in self.fast_block_groups.iter_mut() {
            Self::pre_build_block_group(&self.block_types, name, group);
        }
        Self::pre_build_block_group(
            &self.block_types,
            TRIVIALLY_REPLACEABLE,
            &mut self.trivially_replaceable_block_group,
        );
        self.light_propagation.resize(self.block_types.len(), false);
        self.has_client_side_extended_data
            .resize(self.block_types.len(), false);
        for (index, block) in self.block_types.iter().enumerate() {
            if block.client_info.allow_light_propagation {
                self.light_propagation.set(index, true);
            }
            if block.client_info.has_client_extended_data {
                self.has_client_side_extended_data.set(index, true);
            }
        }
        Ok(())
    }

    fn pre_build_block_group(
        block_types: &[BlockType],
        group_name: &str,
        group: &mut bitvec::vec::BitVec,
    ) {
        group.resize(block_types.len(), false);
        for (index, block) in block_types.iter().enumerate() {
            if block.client_info.groups.iter().any(|x| x == group_name) {
                group.set(index, true);
            }
        }
    }
}

pub struct FastBlockGroup<'a> {
    blocks: &'a bitvec::vec::BitVec,
}
impl FastBlockGroup<'_> {
    /// Returns true if the block is in the group, or false if the block is not in the group
    /// and/or the block ID is out of range.
    pub fn contains(&self, block_id: BlockId) -> bool {
        *self
            .blocks
            .get(block_id.index())
            .as_deref()
            .unwrap_or(&false)
    }
    /// Clone the block group into an owned version
    fn to_owned(&self) -> OwnedFastBlockGroup {
        OwnedFastBlockGroup {
            blocks: self.blocks.clone(),
        }
    }
}

pub struct OwnedFastBlockGroup {
    blocks: bitvec::vec::BitVec,
}
impl OwnedFastBlockGroup {
    /// Construct a placeholder, for ease of constructing game state extensions
    /// during the pre-build phase. Contains no blocks.
    pub fn new_empty() -> Self {
        Self {
            blocks: bitvec::vec::BitVec::new(),
        }
    }

    /// Returns true if the block is in the group, or false if the block is not in the group
    /// and/or the block ID is out of range.
    pub fn contains(&self, block_id: BlockId) -> bool {
        *self
            .blocks
            .get(block_id.index())
            .as_deref()
            .unwrap_or(&false)
    }
}

const BLOCK_MANAGER_META_KEY_LEGACY: &[u8] = b"block_types";

const E: blocks_proto::Empty = blocks_proto::Empty {};

struct UnknownBlockExtDataPassthrough {
    data: Vec<u8>,
}

fn unknown_block_serialize_data_passthrough(
    _: InlineContext,
    data: &CustomData,
) -> Result<Option<Vec<u8>>> {
    Ok(Some(
        data.downcast_ref::<UnknownBlockExtDataPassthrough>()
            .context("Unknown block UnknownBlockExtDataPassthrough downcast failed")?
            .data
            .clone(),
    ))
}

fn unknown_block_deserialize_data_passthrough(
    _: InlineContext,
    data: &[u8],
) -> Result<Option<CustomData>> {
    Ok(Some(Box::new(UnknownBlockExtDataPassthrough {
        data: data.to_vec(),
    })))
}

fn make_unknown_block_serverside(
    manager: &BlockTypeManager,
    id: BlockId,
    short_name: String,
) -> BlockType {
    assert_eq!(id.variant(), 0);

    let air = manager.make_block_name(AIR.to_string());

    BlockType {
        client_info: blocks_proto::BlockTypeDef {
            id: id.base_id(),
            short_name,
            render_info: Some(blocks_proto::block_type_def::RenderInfo::Cube(
                blocks_proto::CubeRenderInfo {
                    render_mode: blocks_proto::CubeRenderMode::SolidOpaque.into(),
                    ..Default::default()
                },
            )),
            physics_info: Some(blocks_proto::block_type_def::PhysicsInfo::Solid(
                SolidPhysicsInfo {
                    ..Default::default()
                },
            )),
            groups: vec![block_groups::DEFAULT_SOLID.to_string()],
            base_dig_time: 1.0,
            wear_multiplier: 0.0,
            light_emission: 0,
            allow_light_propagation: false,
            footstep_sound: 0,
            tool_custom_hitbox: None,
            sound_id: 0,
            sound_volume: 0.0,
            interact_key_option: vec![],
            has_client_extended_data: false,
        },
        deserialize_extended_data_handler: Some(Box::new(
            unknown_block_deserialize_data_passthrough,
        )),
        serialize_extended_data_handler: Some(Box::new(unknown_block_serialize_data_passthrough)),
        make_client_extended_data: None,
        dig_handler_full: None,
        dig_handler_inline: Some(Box::new(move |ctx, block, _, _| {
            *block = ctx
                .block_types()
                .resolve_name(&air)
                .context("Couldn't find air block")?;
            Ok(BlockInteractionResult::default())
        })),
        tap_handler_full: None,
        tap_handler_inline: None,
        place_upon_handler: None,
        interact_key_handler: None,
        is_unknown_block: true,
    }
}

fn make_air_block() -> BlockType {
    use perovskite_core::protocol::blocks::block_type_def::{PhysicsInfo, RenderInfo};
    use perovskite_core::protocol::blocks::Empty;
    BlockType {
        client_info: perovskite_core::protocol::blocks::BlockTypeDef {
            id: 0,
            short_name: AIR.to_string(),
            render_info: Some(RenderInfo::Empty(Empty {})),
            physics_info: Some(PhysicsInfo::Air(Empty {})),
            base_dig_time: 1.0,
            groups: vec![DEFAULT_GAS.to_string(), TRIVIALLY_REPLACEABLE.to_string()],
            wear_multiplier: 1.0,
            light_emission: 0,
            allow_light_propagation: true,
            footstep_sound: 0,
            tool_custom_hitbox: None,
            sound_id: 0,
            sound_volume: 0.0,
            interact_key_option: vec![],
            has_client_extended_data: false,
        },
        ..Default::default()
    }
}

/// Traits for types that can be interpreted as a block type.
pub trait TryAsHandle {
    /// Use the given manager to transform this into a handle.
    fn as_handle(&self, manager: &BlockTypeManager) -> Option<BlockTypeHandle>;
}

/// A handle for a blocktype. This can be passed to the block manager it came from to get back the
/// full block definition.
///
/// The game map, and other APIs, use this as a value type to indicate the type of a block (e.g. at a certain location
/// in the map). This struct implements Copy and has no lifetime requirements to make it easier to use in
/// these contexts.
pub type BlockTypeHandle = BlockId;

impl TryAsHandle for BlockTypeHandle {
    #[inline]
    fn as_handle(&self, _manager: &BlockTypeManager) -> Option<BlockTypeHandle> {
        Some(*self)
    }
}
/// A representation of the type of a block that may not have been registered yet. This can be passed to
/// the block manager it came from to try to get back the full block definition.
///
/// This can be used to allow blocks to have circular dependencies on each other through their
/// respective handlers. See the doc for [`BlockTypeManager`] for details and a motivating example.
///
/// For performance, this struct caches the result of the lookup. The caching logic uses atomics, so
/// only a non-mutable & reference is needed to use this struct.
///
/// There is no protection against accidentally using a FastBlockName with two different block type
/// managers, if there are somehow two perovskite worlds running in the same process and sharing
/// objects. However, that should be a rare case.
pub struct FastBlockName {
    name: String,
    base_id: AtomicU32,
}
impl Clone for FastBlockName {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            base_id: AtomicU32::new(self.base_id.load(Ordering::Relaxed)),
        }
    }
}
impl TryAsHandle for FastBlockName {
    fn as_handle(&self, manager: &BlockTypeManager) -> Option<BlockTypeHandle> {
        manager.resolve_name(self)
    }
}
impl TryAsHandle for &FastBlockName {
    fn as_handle(&self, manager: &BlockTypeManager) -> Option<BlockTypeHandle> {
        manager.resolve_name(self)
    }
}
impl TryAsHandle for &str {
    fn as_handle(&self, manager: &BlockTypeManager) -> Option<BlockTypeHandle> {
        manager.get_by_name(self)
    }
}
impl FastBlockName {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            base_id: AtomicU32::new(u32::MAX),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extended_data_autosets_dirty() {
        let mut ext = None;
        let mut edh = ExtendedDataHolder::new(&mut ext);
        assert!(!edh.dirty);
        let _ = edh.get_or_insert_with(Default::default);
        assert!(edh.dirty);
    }
}
