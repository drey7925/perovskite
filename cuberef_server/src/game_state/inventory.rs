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
    borrow::Borrow,
    collections::HashMap,
    fmt::format,
    ops::DerefMut,
    sync::{atomic::AtomicU64, Arc},
};

use crate::{
    database::database_engine::{GameDatabase, KeySpace},
    game_state::items::{ItemStack, MaybeStack},
};
use anyhow::{bail, ensure, Context, Result};
use cuberef_core::{coordinates::BlockCoordinate, protocol::items as items_proto};
use log::warn;
use parking_lot::{Mutex, RwLock};
use prost::Message;
use tokio::sync::broadcast;

use super::{blocks::ExtendedData, GameState};

/// Opaque unique identifier for an inventory.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct InventoryKey {
    id: uuid::Uuid,
}
impl InventoryKey {
    /// Converts an InventoryKey to an opaque byte string
    pub fn as_bytes(&self) -> &[u8] {
        self.id.as_bytes()
    }
    /// Converts an opaque byte string to an InventoryKey.
    pub fn parse_bytes(bytes: &[u8]) -> Result<InventoryKey> {
        Ok(InventoryKey {
            id: *uuid::Uuid::from_bytes_ref(bytes.try_into()?),
        })
    }

    fn to_db_key(self) -> Vec<u8> {
        KeySpace::Inventory.make_key(self.id.as_bytes())
    }
}

/// Server-side representation of an inventory.
/// Note that Inventory structs are mutable and don't imply any access controls
/// themselves, but clients also cannot modify them directly.
///
/// inventories may be presented via a UI; this is an additional layer (TODO write it)
/// That UI will have rules on access control that the server will enforce
#[derive(Debug)]
pub struct Inventory {
    key: Option<InventoryKey>,
    pub dimensions: (u32, u32),
    contents: Vec<Option<ItemStack>>,
}
impl Inventory {
    fn new(dimensions: (u32, u32)) -> Result<Inventory> {
        ensure!(dimensions.0 > 0);
        ensure!(dimensions.1 > 0);
        // To be fully correct, this should promote to usize first, and also deal with struct sizes.
        // However, nobody should be creating a 4-billion-stack inventory.
        let len = dimensions
            .0
            .checked_mul(dimensions.1)
            .with_context(|| "Inventory size overflowed")?;
        let mut contents = Vec::new();
        contents.resize_with(len.try_into().unwrap(), || None);
        Ok(Inventory {
            key: Some(InventoryKey {
                id: uuid::Uuid::new_v4(),
            }),
            dimensions,
            contents,
        })
    }

    pub(crate) fn from_proto(
        proto: items_proto::Inventory,
        key: Option<InventoryKey>,
    ) -> Result<Inventory> {
        let size = proto
            .width
            .checked_mul(proto.height)
            .with_context(|| "Inventory size overflowed")?;
        ensure!(size as usize == proto.contents.len());
        ensure!(size > 0);
        let contents = proto.contents.iter().map(ItemStack::from_proto).collect();
        Ok(Inventory {
            key,
            dimensions: (proto.height, proto.width),
            contents,
        })
    }

    pub(crate) fn to_proto(&self) -> items_proto::Inventory {
        let contents = self
            .contents
            .iter()
            .map(|x| match x {
                Some(x) => x.proto().clone(),
                None => items_proto::ItemStack {
                    item_name: String::from(""),
                    quantity: 0,
                    max_stack: 0,
                    stackable: false,
                },
            })
            .collect();
        items_proto::Inventory {
            height: self.dimensions.0,
            width: self.dimensions.1,
            contents,
        }
    }
    /// Get a mutable ref to the itemstacks in this inventory. This can be used to
    /// modify them, but cannot change the size or structure of the inventory itself
    pub fn contents_mut(&mut self) -> &mut [Option<ItemStack>] {
        &mut self.contents
    }
    /// Get a reference that can be used to read the item stacks
    /// in the inventory
    pub fn contents(&self) -> &[Option<ItemStack>] {
        &self.contents
    }

    /// Try to insert the given item stack into the given inventory.
    /// This will try to merge the given item stack with existing item stacks
    /// that still have space, if merging is possible
    pub fn try_insert(&mut self, mut stack: ItemStack) -> Option<ItemStack> {
        // Try to merge with existing items
        for slot in self.contents_mut().iter_mut().flatten() {
            match slot.try_merge(stack) {
                // If we have remaining stuff, try to keep merging it
                Some(remaining) => stack = remaining,
                // Otherwise, we're done
                None => return None,
            }
        }
        // Otherwise, try to create a new stack
        for slot in self.contents_mut() {
            if slot.is_none() {
                *slot = Some(stack);
                return None;
            }
        }
        Some(stack)
    }
}

/// Game component that manages access to inventories of items
pub struct InventoryManager {
    // For now, just throw a mutex around an (already thread-safe) DB for basic
    // atomicity/mutual exclusion
    //
    // TODO add caching in the future, if needed for performance reasons
    //
    // TODO FIXME there is a massive deadlock in mutate_inventor{y|ies}_atomically here since
    //  the db lock is held during the callback. This should be restructured to avoid this (i.e.
    // switch to a mutex-protected set of keys being accessed + condvar to release threads that block
    // waiting for an inventory mutation to finish)
    db: Mutex<Arc<dyn GameDatabase>>,
    update_sender: broadcast::Sender<InventoryKey>,
}
impl InventoryManager {
    /// Create a new, empty inventory
    pub fn make_inventory(&self, height: u32, width: u32) -> Result<InventoryKey> {
        let inventory = Inventory::new((height, width))?;
        let db = self.db.lock();
        db.put(
            // unwrap OK because we just generated an inventory with a key
            &inventory.key.unwrap().to_db_key(),
            &inventory.to_proto().encode_to_vec(),
        )?;
        Ok(inventory.key.unwrap())
    }
    /// Get a readonly copy of an inventory.
    pub fn get(&self, key: &InventoryKey) -> Result<Option<Inventory>> {
        let db = self.db.lock();
        let bytes = db.get(&key.to_db_key())?;
        match bytes {
            Some(x) => {
                let inv_proto = items_proto::Inventory::decode(x.borrow())?;
                Ok(Some(Inventory::from_proto(inv_proto, Some(*key))?))
            }
            None => Ok(None),
        }
    }

    /// Run the given mutator on the indicated inventory.
    /// The function may mutate the data, or leave it as-is, and it may return a value
    /// to the caller through its own return value.
    pub fn mutate_inventory_atomically<F, T>(&self, key: &InventoryKey, mutator: F) -> Result<T>
    where
        F: FnOnce(&mut Inventory) -> Result<T>,
    {
        let db = self.db.lock();
        let db_key = key.to_db_key();
        let bytes = db.get(&db_key)?;
        let mut inv = match bytes {
            Some(x) => {
                let inv_proto = items_proto::Inventory::decode(x.borrow())?;
                Inventory::from_proto(inv_proto, Some(*key))?
            }
            None => {
                bail!("Inventory ID not found")
            }
        };
        let result = mutator(&mut inv);
        let new_bytes = inv.to_proto().encode_to_vec();
        db.put(&db_key, &new_bytes)?;
        self.broadcast_update(*key);
        result
    }

    /// Run the given mutator on the indicated vec of inventories.
    /// The function may mutate the data, or leave it as-is, and it may return a value to the caller
    /// through its own return value.
    ///
    /// See mutate_inventory_atomically for warnings regarding deadlock safety
    pub fn mutate_inventories_atomically<F, T>(
        &self,
        keys: &[InventoryKey],
        mutator: F,
    ) -> Result<T>
    where
        F: FnOnce(&mut [Inventory]) -> Result<T>,
    {
        let db = self.db.lock();
        let mut inventories = Vec::with_capacity(keys.len());
        for key in keys {
            let bytes = db.get(&key.to_db_key())?;
            let inv = match bytes {
                Some(x) => {
                    let inv_proto = items_proto::Inventory::decode(x.borrow())?;
                    Inventory::from_proto(inv_proto, Some(*key))?
                }
                None => {
                    bail!("Inventory ID not found")
                }
            };
            inventories.push(inv);
        }

        let result = mutator(&mut inventories);
        for inv in inventories {
            let new_bytes = inv.to_proto().encode_to_vec();
            db.put(&inv.key.unwrap().to_db_key(), &new_bytes)?;
            self.broadcast_update(inv.key.unwrap())
        }
        result
    }

    pub(crate) fn new(db: Arc<dyn GameDatabase>) -> InventoryManager {
        let (sender, _) = broadcast::channel(BROADCAST_CHANNEL_SIZE);
        InventoryManager {
            db: db.into(),
            update_sender: sender,
        }
    }

    pub(crate) fn subscribe(&self) -> broadcast::Receiver<InventoryKey> {
        self.update_sender.subscribe()
    }

    fn broadcast_update(&self, key: InventoryKey) {
        match self.update_sender.send(key) {
            Ok(_) => {}
            Err(_) => warn!("No receivers for inventory key update {:?}", key),
        }
    }
}

/// ID for an inventory view.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct InventoryViewId(pub(crate) u64);

#[derive(Clone, Debug)]
pub enum BorrowLocation {
    /// Stored in the inventory manager (inventory key, slot)
    Global(InventoryKey, usize),
    /// Stored in a block (coordinate, key, slot)
    Block(BlockCoordinate, String, usize),
    /// Not borrowing from somewhere
    NotBorrowed,
}
impl BorrowLocation {
    fn or(self, alternative: BorrowLocation) -> BorrowLocation {
        if let Self::NotBorrowed = self {
            return alternative;
        }
        return self;
    }
}

/// A stack of items that came from some real inventory, and is now in a transient inventory
/// (or being held by the user's cursor).
///
/// This is not the same as a "borrow" in the sense of Rust references and lifetimes.
#[derive(Clone, Debug)]
pub struct BorrowedStack {
    /// The location (inventory key, offset) that this stack is borrowing from
    pub borrows_from: BorrowLocation,
    /// The stack that is borrowed from the location indicated.
    pub borrowed_stack: ItemStack,
}

pub struct VirtualOutputCallbacks<T> {
    /// Callback that shows what items are visible in the view. This should be idempotent
    pub peek: Box<dyn Fn(&T) -> Vec<Option<ItemStack>> + Sync + Send>,
    /// Callback of (context, slot#, count). If count is None, user is taking everything visible in peek.
    /// If this view is take_exact, then the impl is free to disregard the count parameter
    pub take: Box<dyn FnMut(&T, usize, Option<u32>) -> Option<ItemStack> + Sync + Send>,
}

/// Where the items in the inventory view actually come from.
///
/// The type parameter T represents the type passed to the callbacks as context.
pub enum ViewBacking<T> {
    /// The inventory view represents an inventory that isn't stored anywhere, but
    /// represents real items that can be moved around.
    /// (e.g. a grid into which recipe ingredients can be placed, a trading interface, etc)
    ///
    /// If this view is deleted, the items within it are returned to the place they were borrowed
    /// from, or alternatively into a stack with space or open slot in the inventory they came from.
    /// Any items with an empty borrows_from don't have a place to return them (they may have come from
    /// a callback of a virtual view)
    ///
    /// TODO figure out what happens if the returned items can't fit or they have an empty borrows_from
    Transient(parking_lot::RwLock<Vec<Option<BorrowedStack>>>),
    /// This inventory view is generated on-the-fly and does not represent real items
    /// e.g. the output of a recipe grid.
    ///
    /// Once an item is taken from the view, it becomes real, and must be placed somewhere.
    ///
    /// This includes a callback that can e.g. consume input items from other views when
    /// an itemstack is taken from the view.
    ///
    /// Nothing interesting happens when this view is deleted, because any inventory changes
    /// through this view have already been performed on other views through the callbacks by then.
    /// The item "appearing" in this view isn't real until it's taken out (at which the callback is invoked)
    ///
    /// Note that VirtualInput and VirtualOutput may be refactored into the same enum variant with the callbacks combined;
    /// however the edge cases involving both input and output are not yet resolved in the current MVP.
    VirtualOutput(parking_lot::RwLock<VirtualOutputCallbacks<T>>),
    /// This inventory view doesn't hold anything. When a stack is placed into it, it is consumed from
    /// its source and a callback will be invoked.
    ///
    /// Nothing interesting happens when this view is deleted, because this view doesn't "hold" any items.
    ///
    /// Note that VirtualInput and VirtualOutput may be refactored into the same enum variant with the callbacks combined;
    /// however the edge cases involving both input and output are not yet resolved in the current MVP.
    VirtualInput(Box<dyn Fn(&T) + Send + Sync>),

    /// This inventory view is stored in the database. Nothing interesting happens when the view
    /// is deleted, because any actions on the view have been written back to the database by then.
    Stored(InventoryKey),

    // This inventory view is stored in the extended data of a block.
    StoredInBlock(BlockCoordinate, String),
}

/// A view into an inventory, meant for user interaction.
///
/// An inventory view has the following operations: Picking up a stack (or subset) from a slot,
/// placing a stack into a slot, rendering the inventory view, and deleting the view.
///
/// An inventory view can be transient, virtual, or backed by a real inventory from the database.
/// See [ViewBacking] for more details.
///
/// Note that an InventoryView is, on its own, inert. It needs to be added to a
/// [PopupBuilder][super::popups::PopupBuilder] to do anything
pub struct InventoryView<T> {
    /// (rows, cols)
    pub(crate) dimensions: (u32, u32),
    /// Whether the user can put things into this view
    pub(crate) can_place: bool,
    /// Whether the user can take things out of this view
    pub(crate) can_take: bool,
    take_exact: bool,
    /// The kind of inventory this view is showing    
    pub(crate) backing: ViewBacking<T>,
    pub(crate) id: InventoryViewId,
    game_state: Arc<GameState>,
}
impl<T> InventoryView<T> {
    pub(crate) fn new_stored(
        inventory_key: InventoryKey,
        game_state: Arc<GameState>,
        can_place: bool,
        can_take: bool,
    ) -> Result<InventoryView<T>> {
        let inventory = game_state
            .inventory_manager()
            .get(&inventory_key)?
            .with_context(|| format!("Inventory {inventory_key:?} not found"))?;
        Ok(InventoryView {
            dimensions: inventory.dimensions,
            can_place,
            can_take,
            take_exact: false,
            backing: ViewBacking::Stored(inventory_key),
            game_state,
            id: next_id(),
        })
    }
    /// Creates a new transient view. This view is inert until added to a UI.
    ///
    /// initial_contents must either have length equal to dimensions.0 * dimensions.1, or be zero length
    pub(crate) fn new_transient(
        game_state: Arc<GameState>,
        dimensions: (u32, u32),
        mut initial_contents: Vec<Option<BorrowedStack>>,
        can_place: bool,
        can_take: bool,
        take_exact: bool,
    ) -> Result<InventoryView<T>> {
        if initial_contents.is_empty() {
            initial_contents.resize_with(dimensions.0 as usize * dimensions.1 as usize, || None);
        }
        ensure!(dimensions.0 as usize * dimensions.1 as usize == initial_contents.len());
        Ok(InventoryView {
            dimensions,
            can_place,
            can_take,
            take_exact,
            backing: ViewBacking::Transient(initial_contents.into()),
            id: next_id(),
            game_state,
        })
    }

    pub(crate) fn new_virtual_output(
        game_state: Arc<GameState>,
        dimensions: (u32, u32),
        callbacks: VirtualOutputCallbacks<T>,
        take_exact: bool,
    ) -> Result<InventoryView<T>> {
        Ok(InventoryView {
            dimensions,
            can_place: false,
            can_take: true,
            take_exact,
            backing: ViewBacking::VirtualOutput(RwLock::new(callbacks)),
            id: next_id(),
            game_state,
        })
    }

    pub(crate) fn new_block(
        game_state: Arc<GameState>,
        dimensions: (u32, u32),
        coord: BlockCoordinate,
        key: String,
        can_place: bool,
        can_take: bool,
        take_exact: bool
    ) -> Result<InventoryView<T>> {
        let actual_dimensions = game_state
            .map()
            .mutate_block_atomically(coord, |_, ext_data| {
                if ext_data.is_none() {
                    *(ext_data.deref_mut()) = Some(ExtendedData {
                        custom_data: None,
                        inventories: HashMap::new(),
                    });
                }
                let inv = ext_data.as_mut().unwrap().inventories.entry(key.clone()).or_insert_with(|| {
                    let mut contents = vec![];
                    contents.resize_with(dimensions.0 as usize * dimensions.1 as usize, || None);
                    Inventory {
                        key: None,
                        dimensions,
                        contents,
                    }
                });
                Ok(inv.dimensions)
            })?;
        Ok(InventoryView {
            dimensions: actual_dimensions,
            can_place,
            can_take,
            take_exact,
            backing: ViewBacking::StoredInBlock(coord, key),
            id: next_id(),
            game_state: game_state,
        })
    }

    /// Clears all the items in the view, returning them to their
    /// respective origin locations
    pub(crate) fn clear_if_transient(&mut self, owner_inv_key: Option<InventoryKey>) -> Result<()> {
        if let ViewBacking::Transient(transient_data) = &mut self.backing {
            for stack in transient_data.write().iter_mut() {
                if let Some(stack) = stack.take() {
                    let leftover = match stack.borrows_from {
                        BorrowLocation::Global(key, slot) => self
                            .game_state
                            .inventory_manager()
                            .mutate_inventory_atomically(&key, |inv| {
                                let mut leftover = inv
                                    .contents_mut()
                                    .get_mut(slot)
                                    .with_context(|| "Out of bounds slot #")?
                                    .try_merge(Some(stack.borrowed_stack));
                                if leftover.is_some() {
                                    leftover = inv.try_insert(leftover.unwrap());
                                }
                                Ok(leftover)
                            })?,
                        BorrowLocation::Block(coord, key, slot) => self
                            .game_state
                            .map()
                            .mutate_block_atomically(coord, |_, ext_data| {
                                match ext_data.as_mut().and_then(|x| x.inventories.get_mut(&key)) {
                                    Some(inv) => {
                                        let mut leftover = inv
                                            .contents_mut()
                                            .get_mut(slot)
                                            .with_context(|| "Out of bounds slot #")?
                                            .try_merge(Some(stack.borrowed_stack));
                                        if leftover.is_some() {
                                            leftover = inv.try_insert(leftover.unwrap());
                                        }
                                        Ok(leftover)
                                    }
                                    None => Ok(Some(stack.borrowed_stack)),
                                }
                            })?,
                        BorrowLocation::NotBorrowed => Some(stack.borrowed_stack),
                    };
                    if leftover.is_some() {
                        if let Some(key) = owner_inv_key {
                            self.game_state
                                .inventory_manager()
                                .mutate_inventory_atomically(&key, |inv| {
                                    let leftover = inv.try_insert(leftover.unwrap());
                                    if leftover.is_some() {
                                        log::warn!("Could not return {:?} to a home; its home inventory was full", leftover.unwrap());
                                        // TODO handle this
                                    }
                                    Ok(())
                                })?;
                        };
                    }
                }
            }
        }
        Ok(())
    }
    /// Returns true if this view is backed by the given inventory key
    pub(crate) fn wants_update_for(&self, key: &InventoryKey) -> bool {
        if let ViewBacking::Stored(our_key) = &self.backing {
            our_key == key
        } else {
            false
        }
    }
}

/// A representation of an inventory view that's independent of the actual callback type involved.
pub struct InventoryViewWithContext<'a, T> {
    pub(crate) view: &'a InventoryView<T>,
    pub(crate) context: &'a T,
}
pub trait TypeErasedInventoryView {
    /// Get the ID of this view
    fn id(&self) -> InventoryViewId;

    fn dimensions(&self) -> (u32, u32);

    /// See the items in this view (e.g. to display the inventory).
    fn peek(&self) -> Result<Vec<Option<ItemStack>>>;
    /// Takes a stack from one of the slots in this view (e.g. when clicked with a cursor).
    /// This will either modify the items in this view (if transient/stored) or possibly modify other
    /// views (if virtual).
    ///
    /// peek should be called immediately after to see what the view should display.
    fn take(&self, slot: usize, count: Option<u32>) -> Result<Option<BorrowedStack>>;
    /// Attempts to place a stack into the given slot in the view.
    /// Returns the leftover stack (possibly the entire provided stack if it cannot be accepted here)
    ///
    /// peek should be called immediate after to see what the view should display.
    fn put(&self, slot: usize, stack: BorrowedStack) -> Result<Option<BorrowedStack>>;

    fn can_place(&self) -> bool;
    fn can_take(&self) -> bool;
    // if true, user must take exactly the amount shown in the stack they're taking. Probably only useful for VirtualOutput
    // stacks (or maybe transient stacks) used as crafting output or similar.
    fn take_exact(&self) -> bool;

    fn to_client_proto(&self) -> Result<cuberef_core::protocol::game_rpc::InventoryUpdate>;
}

impl<'a, T> TypeErasedInventoryView for InventoryViewWithContext<'a, T> {
    fn id(&self) -> InventoryViewId {
        self.view.id
    }

    fn dimensions(&self) -> (u32, u32) {
        self.view.dimensions
    }

    fn peek(&self) -> Result<Vec<Option<ItemStack>>> {
        self.view.peek(self.context)
    }

    fn take(&self, slot: usize, count: Option<u32>) -> Result<Option<BorrowedStack>> {
        self.view.take(self.context, slot, count)
    }

    fn put(&self, slot: usize, stack: BorrowedStack) -> Result<Option<BorrowedStack>> {
        self.view.put(self.context, slot, stack)
    }

    fn can_place(&self) -> bool {
        self.view.can_place
    }

    fn can_take(&self) -> bool {
        self.view.can_take
    }

    fn take_exact(&self) -> bool {
        self.view.take_exact
    }

    fn to_client_proto(&self) -> Result<cuberef_core::protocol::game_rpc::InventoryUpdate> {
        self.view.to_client_proto(self.context)
    }
}

impl TypeErasedInventoryView for &InventoryView<()> {
    fn id(&self) -> InventoryViewId {
        self.id
    }

    fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    fn peek(&self) -> Result<Vec<Option<ItemStack>>> {
        (*self).peek(&())
    }

    fn take(&self, slot: usize, count: Option<u32>) -> Result<Option<BorrowedStack>> {
        (*self).take(&(), slot, count)
    }

    fn put(&self, slot: usize, stack: BorrowedStack) -> Result<Option<BorrowedStack>> {
        (*self).put(&(), slot, stack)
    }

    fn can_place(&self) -> bool {
        self.can_place
    }

    fn can_take(&self) -> bool {
        self.can_take
    }

    fn take_exact(&self) -> bool {
        self.take_exact
    }

    fn to_client_proto(&self) -> Result<cuberef_core::protocol::game_rpc::InventoryUpdate> {
        (*self).to_client_proto(&())
    }
}

impl<T> InventoryView<T> {
    /// See the items in this view (e.g. to display the inventory).
    pub fn peek(&self, context: &T) -> Result<Vec<Option<ItemStack>>> {
        match &self.backing {
            ViewBacking::Transient(contents) => Ok(contents
                .borrow()
                .try_read()
                .context("already borrowed")?
                .iter()
                .map(|x| x.as_ref().map(|x| x.borrowed_stack.clone()))
                .collect()),
            ViewBacking::VirtualOutput(virt_out) => {
                let peeked = (virt_out.read().peek)(context);
                ensure!(peeked.len() == self.dimensions.0 as usize * self.dimensions.1 as usize);
                Ok(peeked)
            }
            ViewBacking::VirtualInput(_) => todo!(),
            ViewBacking::Stored(key) => Ok(self
                .game_state
                .inventory_manager()
                .get(key)?
                .with_context(|| format!("Inventory {key:?} in view {:?} not found", self.id))?
                .contents()
                .to_vec()),
            ViewBacking::StoredInBlock(coord, key) => Ok(self
                .game_state
                .map()
                .get_block_with_extended_data(*coord, |x| {
                    Some(
                        x.inventories
                            .get(key)
                            .map(|x| x.contents.to_vec())
                            .unwrap_or(vec![]),
                    )
                })?
                .1
                .unwrap_or(vec![])),
        }
    }
    /// Takes a stack from one of the slots in this view (e.g. when clicked with a cursor).
    /// This will either modify the items in this view (if transient/stored) or possibly modify other
    /// views (if virtual).
    ///
    /// peek should be called immediately after to see what the view should display.
    pub fn take(
        &self,
        context: &T,
        slot: usize,
        count: Option<u32>,
    ) -> Result<Option<BorrowedStack>> {
        ensure!(slot < (self.dimensions.0 as usize * self.dimensions.1 as usize));
        match &self.backing {
            ViewBacking::Transient(contents) => {
                let mut guard = contents.try_write().context("already borrowed")?;
                // unwrap is ok - we checked the length
                let slot_contents = guard.get_mut(slot).unwrap();
                if slot_contents.is_some() {
                    let borrows_from = slot_contents.as_ref().unwrap().borrows_from.clone();
                    let mut inner = Some(slot_contents.as_ref().unwrap().borrowed_stack.clone());
                    let obtained = inner.take_items(count);

                    if let Some(inner) = inner {
                        slot_contents.as_mut().unwrap().borrowed_stack = inner;
                    } else {
                        *slot_contents = None;
                    }
                    Ok(obtained.map(|obtained| BorrowedStack {
                        borrows_from,
                        borrowed_stack: obtained,
                    }))
                } else {
                    Ok(None)
                }
            }
            ViewBacking::VirtualOutput(virt_out) => Ok((virt_out.write().take)(
                context, slot, count,
            )
            .map(|x| BorrowedStack {
                borrows_from: BorrowLocation::NotBorrowed,
                borrowed_stack: x,
            })),
            ViewBacking::VirtualInput(_) => todo!(),
            ViewBacking::Stored(key) => Ok(self
                .game_state
                .inventory_manager()
                .mutate_inventory_atomically(key, |inv| {
                    Ok(inv.contents_mut().get_mut(slot).unwrap().take_items(count))
                })?
                .map(|obtained| BorrowedStack {
                    borrows_from: BorrowLocation::Global(*key, slot),
                    borrowed_stack: obtained,
                })),
            ViewBacking::StoredInBlock(coord, key) => self
                .game_state
                .map()
                .mutate_block_atomically(*coord, |_, ext_data| {
                    Ok(ext_data
                        .as_mut()
                        .and_then(|x| x.inventories.get_mut(key))
                        .and_then(|x| {
                            x.contents_mut()
                                .get_mut(slot)
                                .unwrap()
                                .take_items(count)
                                .map(|x| BorrowedStack {
                                    borrows_from: BorrowLocation::Block(*coord, key.clone(), slot),
                                    borrowed_stack: x,
                                })
                        }))
                }),
        }
    }

    /// Attempts to place a stack into the given slot in the view.
    /// Returns the leftover stack (possibly the entire provided stack if it cannot be accepted here)
    ///
    /// peek should be called immediate after to see what the view should display.
    pub fn put(
        &self,
        _context: &T,
        slot: usize,
        stack: BorrowedStack,
    ) -> Result<Option<BorrowedStack>> {
        ensure!(slot < (self.dimensions.0 as usize * self.dimensions.1 as usize));
        match &self.backing {
            ViewBacking::Transient(contents) => {
                let mut guard = contents.try_write().context("already borrowed")?;
                let slot_contents = guard.get_mut(slot).unwrap();
                if slot_contents.is_none() {
                    *slot_contents = Some(stack);
                    Ok(None)
                } else {
                    let mut inner = slot_contents.as_mut().unwrap().borrowed_stack.clone();
                    let leftover = inner.try_merge(stack.borrowed_stack);
                    *slot_contents = Some(BorrowedStack {
                        borrows_from: slot_contents
                            .as_ref()
                            .unwrap()
                            .borrows_from
                            .clone()
                            .or(stack.borrows_from.clone()),
                        borrowed_stack: inner,
                    });
                    Ok(leftover.map(|leftover| BorrowedStack {
                        borrows_from: stack.borrows_from,
                        borrowed_stack: leftover,
                    }))
                }
            }
            ViewBacking::VirtualOutput(_) => {
                // can't put into a virtualoutput
                Ok(Some(stack))
            }
            ViewBacking::VirtualInput(_) => todo!(),
            ViewBacking::Stored(key) => Ok(self
                .game_state
                .inventory_manager()
                .mutate_inventory_atomically(key, |inv| {
                    Ok(inv
                        .contents_mut()
                        .get_mut(slot)
                        .with_context(|| "Out of bounds slot #")?
                        .try_merge(Some(stack.borrowed_stack)))
                })?
                .map(|leftover| BorrowedStack {
                    borrows_from: stack.borrows_from,
                    borrowed_stack: leftover,
                })),
            ViewBacking::StoredInBlock(coord, key) => Ok(self
                .game_state
                .map()
                .mutate_block_atomically(*coord, |_, ext_data| {
                    match ext_data.as_mut().and_then(|x| x.inventories.get_mut(key)) {
                        Some(x) => {
                            let leftover = x
                                .contents_mut()
                                .get_mut(slot)
                                .with_context(|| "Out of bounds slot #")?
                                .try_merge(Some(stack.borrowed_stack));
                            Ok(leftover.map(|x| BorrowedStack {
                                borrows_from: stack.borrows_from,
                                borrowed_stack: x,
                            }))
                        }
                        None => Ok(Some(stack)),
                    }
                })?),
        }
    }

    /// Get the ID of this view
    fn id(&self) -> InventoryViewId {
        self.id
    }
    /// Get the dimensions of this view
    fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    fn can_place(&self) -> bool {
        self.can_place
    }

    fn can_take(&self) -> bool {
        self.can_take
    }

    // if true, user must take exactly the amount shown in the stack they're taking. Probably only useful for VirtualOutput
    // stacks (or maybe transient stacks) used as crafting output or similar.
    fn take_exact(&self) -> bool {
        self.take_exact
    }

    fn to_client_proto(
        &self,
        context: &T,
    ) -> Result<cuberef_core::protocol::game_rpc::InventoryUpdate> {
        Ok(cuberef_core::protocol::game_rpc::InventoryUpdate {
            inventory: Some(cuberef_core::protocol::items::Inventory {
                height: self.dimensions().0,
                width: self.dimensions().1,
                contents: self
                    .peek(context)?
                    .into_iter()
                    .map(|x| {
                        x.map_or_else(
                            || cuberef_core::protocol::items::ItemStack {
                                item_name: "".to_string(),
                                quantity: 0,
                                max_stack: 0,
                                stackable: false,
                            },
                            |x| x.proto,
                        )
                    })
                    .collect(),
            }),
            view_id: self.id().0,
            can_place: self.can_place(),
            can_take: self.can_take(),
            take_exact: self.take_exact(),
        })
    }
}

impl<T> Drop for InventoryView<T> {
    fn drop(&mut self) {
        if let Err(e) = self.clear_if_transient(None) {
            log::error!("Failed to drop inventory view: {e:?}");
        }
    }
}

static INVENTORY_VIEW_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_id() -> InventoryViewId {
    InventoryViewId(INVENTORY_VIEW_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
}

const BROADCAST_CHANNEL_SIZE: usize = 32;
