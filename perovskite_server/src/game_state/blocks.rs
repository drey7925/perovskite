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

use std::{
    any::Any,
    borrow::Borrow,
    collections::{hash_map::Entry, HashMap, HashSet},
    ops::{AddAssign, Deref, DerefMut},
    sync::atomic::{AtomicU32, AtomicUsize, Ordering},
};

use anyhow::{bail, ensure, Context, Result};
use log::{info, warn};
use rustc_hash::FxHashMap;

use crate::database::database_engine::{GameDatabase, KeySpace};

use super::{
    client_ui::Popup,
    event::{EventInitiator, HandlerContext},
    inventory::Inventory,
    items::{Item, ItemManager, ItemStack},
};
use perovskite_core::{
    block_id::{BlockError, BlockId},
    constants::blocks::AIR,
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
    /// visible in popups.
    /// TODO: switch back to std hashmap once get_many_mut is stabilized
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

/// Takes (handler context, coordinate being dug, item stack used to dig), returns dropped item stacks.
pub type FullHandler = dyn Fn(&HandlerContext, BlockCoordinate, Option<&ItemStack>) -> Result<BlockInteractionResult>
    + Send
    + Sync;
/// Takes (handler context, mutable reference to the block type in the map,
/// mutable reference to the extended data holder, item stack used to dig), returns dropped item stacks.
pub type InlineHandler = dyn Fn(
        InlineContext,
        &mut BlockTypeHandle,
        &mut ExtendedDataHolder,
        Option<&ItemStack>,
    ) -> Result<BlockInteractionResult>
    + Send
    + Sync;

// How extended data for this block type ought to be handled.
#[derive(PartialEq, Eq)]
pub enum ExtDataHandling {
    /// The block will never have extended data.
    NoExtData,
    /// When storing/loading this block type, it has extended data that is
    /// needed by the block type's plugins/handlers code on the server.
    /// Note that this loading may still be done lazily - they may only be loaded when
    /// an event handler actually tries to read/write them.
    ServerSide,
    /// This block type has extended data that affects the client's behavior.
    /// The xattrs will be loaded and parsed every time the block is sent to a client,
    /// but the result may be cached for use with multiple clients/users.
    /// This may be expensive - a block-specific handler will be called every time
    /// the block is loaded and needs to be sent to a client.
    ///
    /// NOT YET IMPLEMENTED
    ClientSideShared,
    /// Same as ClientSideShared, except the result must not be cached across
    /// clients/users. That is, if the same block is being sent to ten players
    /// in a space, the event handler will need to be run ten times, making this even
    /// more expensive than ClientSideShared.
    ///
    /// Clients may still cache the resulting extended data until it is invalidated
    /// and re-sent, or that part of the map is otherwise dropped from the cache.
    ///
    /// NOT YET IMPLEMENTED
    ClientSideUnshared,
}

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
    /// How extended data ought to be handled for this block.
    pub extended_data_handling: ExtDataHandling,

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
    /// and as fast as possible. Should typically be a wrapper around a protobuf serialization
    /// step or similar. This callback does not need to handle any block inventories; they are automatically
    /// handled by the engine.
    ///
    /// Only called if `extended_data_handling` is not equal to `NoExtData` and the extended data itself is
    /// Some. If this returns None, no extended data will be serialized to storage - an example
    /// use for this would be to avoid the storage overhead if the extended data is *logically*
    /// empty.
    ///
    /// If this returns an Err, the game will crash since this represents data loss. Note that this can cause
    /// other changes to a map chunk to be lost.
    pub serialize_extended_data_handler:
        Option<Box<dyn Fn(InlineContext, &CustomData) -> Result<Option<Vec<u8>>> + Send + Sync>>,
    /// If extended_data_handling is one of the client-side options, called whenever the block
    /// needs to be sent to a client (subject to the caching policy, see doc for `ExtDataHandling`)
    ///
    /// The function is called with the extended data obtained from deserialize_extended_data_handler
    /// as well as any inventories.
    ///
    /// If the response is Some(...), all fields other than id and short_name override the defaults
    /// for this block.
    ///
    /// Should be as fast as possible.
    ///
    /// If this returns an Err, the game will crash since this represents data loss due to corrupt data
    /// in a chunk.
    pub extended_data_to_client_side: Option<
        Box<
            dyn Fn(InlineContext, &ExtendedData) -> Option<blocks_proto::BlockTypeDef>
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
    /// If this returns Err, a message will be logged and the user that dug the block will get an error message in some
    /// TBD way (for now, just a log message on the client). dig_handler_full will not be called in that case
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
    pub place_upon_handler: Option<
        Box<dyn Fn(HandlerContext, BlockCoordinate, Option<Item>) -> Result<()> + Send + Sync>,
    >,
    /// Called when the interact key is pressed and an interaction is selected.
    ///
    /// This can mutate the given block (using handlercontext) if desired, and return a popup (if desired)
    ///
    /// The signature of this callback is subject to change.
    pub interact_key_handler:
        Option<Box<dyn Fn(HandlerContext, BlockCoordinate) -> Result<Option<Popup>> + Send + Sync>>,
    // Internal impl details
    pub(crate) is_unknown_block: bool,
    pub(crate) block_type_manager_id: Option<usize>,
}

impl BlockType {
    pub(crate) fn id(&self, variant: u16) -> Result<BlockId> {
        BlockId::new(self.client_info.id, variant)
    }

    pub fn short_name(&self) -> &str {
        &self.client_info.short_name
    }

    pub fn block_type_ref(&self, variant: u16) -> Result<BlockTypeHandle> {
        if let Some(_unique_id) = self.block_type_manager_id {
            Ok(self.id(variant)?)
        } else {
            bail!(BlockError::BlockNotRegistered(
                self.client_info.short_name.clone()
            ))
        }
    }
}

impl Default for BlockType {
    fn default() -> Self {
        Self {
            client_info: Default::default(),
            extended_data_handling: ExtDataHandling::NoExtData,
            deserialize_extended_data_handler: None,
            serialize_extended_data_handler: None,
            extended_data_to_client_side: None,
            dig_handler_full: None,
            dig_handler_inline: None,
            tap_handler_full: None,
            tap_handler_inline: None,
            place_upon_handler: None,
            interact_key_handler: None,
            is_unknown_block: false,
            block_type_manager_id: None,
        }
    }
}

/// Represents extended data for a block on the map.
///
/// **Data loss warning:** [ExtendedDataHolder::set_dirty()] *must* be called after making meaningful changes to the extended data
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
    /// **Data loss warning:** If this is not set, changes to the extended data may be lost.
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

// Each BlockTypeManager has a unique ID, which
static BLOCK_TYPE_MANAGER_ID: AtomicUsize = AtomicUsize::new(1);

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
    name_to_base_id_map: HashMap<String, u32>,
    unique_id: usize,

    // Separate copy of BlockType.client_info.allow_light_propagation, packed densely in order
    // to be more cache friendly
    init_complete: bool,
    light_propagation: bitvec::vec::BitVec,
    fast_block_groups: FxHashMap<String, bitvec::vec::BitVec>,
}
impl BlockTypeManager {
    pub(crate) fn new() -> BlockTypeManager {
        BlockTypeManager {
            block_types: Vec::new(),
            name_to_base_id_map: HashMap::new(),
            unique_id: BLOCK_TYPE_MANAGER_ID.fetch_add(1, Ordering::Relaxed),
            init_complete: false,
            light_propagation: bitvec::vec::BitVec::new(),
            fast_block_groups: FxHashMap::default(),
        }
    }
    /// Given a handle, return the block.
    pub fn get_block(&self, handle: &BlockTypeHandle) -> Result<(&BlockType, u16)> {
        self.get_block_by_id(*handle)
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

    /// Registers a new block in this block type manager, and returns a handle to it.
    ///
    /// Returns an error if another block is already registered with the same short name, or
    /// if there are too many blocktypes (up to roughly 1 million BlockTypes can be registered)
    pub fn register_block(&mut self, mut block: BlockType) -> Result<BlockTypeHandle> {
        block.block_type_manager_id = Some(self.unique_id);

        let id = match self
            .name_to_base_id_map
            .entry(block.short_name().to_string())
        {
            Entry::Occupied(x) => {
                let id = BlockId(*x.get());
                let existing = &mut self.block_types[id.index()];
                ensure!(existing.is_unknown_block);

                block.client_info.id = id.base_id();
                info!("Registering block {} as {:?}", block.short_name(), id);
                self.light_propagation
                    .set(id.index(), block.client_info.allow_light_propagation);
                *existing = block;
                id
            }
            Entry::Vacant(x) => {
                let new_id = BlockId(
                    (self.block_types.len() << 12)
                        .try_into()
                        .with_context(|| BlockError::TooManyBlocks)?,
                );
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
            name_to_base_id_map: HashMap::new(),
            unique_id: BLOCK_TYPE_MANAGER_ID.fetch_add(1, Ordering::Relaxed),
            init_complete: false,
            light_propagation: bitvec::vec::BitVec::new(),
            fast_block_groups: FxHashMap::default(),
        };
        let max_index = block_proto
            .block_type
            .iter()
            .map(|x| BlockId(x.id).index())
            .max()
            .unwrap_or(0);

        let present_indices: HashSet<usize> =
            HashSet::from_iter(block_proto.block_type.iter().map(|x| BlockId(x.id).index()));
        manager.light_propagation.resize(max_index + 1, false);
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

    pub(crate) fn create_or_load(
        db: &dyn GameDatabase,
        allow_create: bool,
    ) -> Result<BlockTypeManager> {
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
                if !allow_create {
                    bail!("Block type data is missing from the database");
                }
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
    /// [`BlockTypeManager`] and [`BlockTypeName`] for an example of where this should be used.
    pub fn make_block_name(&self, name: String) -> FastBlockName {
        FastBlockName {
            name,
            base_id: AtomicU32::new(u32::MAX),
        }
    }

    /// Tries to resolve a BlockTypeName from `make_block_name()`. If a block with a matching
    /// name was registered, returns a handle to it. Otherwise, returns None.
    pub fn resolve_name(&self, block_name: &FastBlockName) -> Option<BlockTypeHandle> {
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

    /// Tries to get a block by name. Equivalent to calling make_block_name and then resolve_name back to back.
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
        self.fast_block_groups
            .insert(block_group.to_string(), bitvec::vec::BitVec::new());
    }

    /// Returns a block group by name, or None if the block group is not registered.
    ///
    /// The block group must have been registered using [`register_block_group`]. Simply
    /// being defined in a block type is not enough.
    ///
    /// Panics:
    /// This function will panic if called before the game starts up.
    pub fn fast_block_group<'a>(&'a self, block_group: &str) -> Option<FastBlockGroup<'a>> {
        self.fast_block_groups
            .get(block_group)
            .map(|x| FastBlockGroup { blocks: x })
    }

    // Performs some last setup before the block manager becomes immutable.
    pub(crate) fn pre_build(&mut self) -> Result<()> {
        for (name, group) in self.fast_block_groups.iter_mut() {
            group.resize(self.block_types.len(), false);
            for (index, block) in self.block_types.iter().enumerate() {
                if block.client_info.groups.contains(name) {
                    group.set(index, true);
                }
            }
        }
        Ok(())
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
    assert!(id.variant() == 0);

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
            physics_info: Some(blocks_proto::block_type_def::PhysicsInfo::Solid(E)),
            groups: vec![],
            base_dig_time: 1.0,
            wear_multiplier: 0.0,
            light_emission: 0,
            allow_light_propagation: false,
        },
        extended_data_handling: ExtDataHandling::ServerSide,
        deserialize_extended_data_handler: Some(Box::new(
            unknown_block_deserialize_data_passthrough,
        )),
        serialize_extended_data_handler: Some(Box::new(unknown_block_serialize_data_passthrough)),
        extended_data_to_client_side: None,
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
        block_type_manager_id: None,
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
/// managers, if there are somehow two perovskite worlds runninng in the same process and sharing
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
    pub fn new(name: String) -> Self {
        Self {
            name,
            base_id: AtomicU32::new(u32::MAX),
        }
    }
}
