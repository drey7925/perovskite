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

use std::{borrow::Borrow, sync::Arc};

use crate::{
    database::database_engine::{GameDatabase, KeySpace},
    game_state::items::ItemStack,
};
use anyhow::{bail, ensure, Context, Result};
use cuberef_core::protocol::items as items_proto;
use log::warn;
use parking_lot::Mutex;
use prost::Message;
use tokio::sync::broadcast;

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
    key: InventoryKey,
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
            key: InventoryKey {
                id: uuid::Uuid::new_v4(),
            },
            dimensions,
            contents,
        })
    }

    fn from_proto(proto: items_proto::Inventory) -> Result<Inventory> {
        let size = proto
            .width
            .checked_mul(proto.height)
            .with_context(|| "Inventory size overflowed")?;
        ensure!(size as usize == proto.contents.len());
        ensure!(size > 0);
        let contents = proto
            .contents
            .iter()
            .map(ItemStack::from_proto)
            .collect();
        Ok(Inventory {
            key: InventoryKey::parse_bytes(&proto.inventory_key)?,
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
                },
            })
            .collect();
        items_proto::Inventory {
            inventory_key: self.key.as_bytes().to_vec(),
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
    db: Mutex<Arc<dyn GameDatabase>>,
    update_sender: broadcast::Sender<InventoryKey>,
}
impl InventoryManager {
    /// Create a new, empty inventory
    pub fn make_inventory(&self, height: u32, width: u32) -> Result<InventoryKey> {
        let inventory = Inventory::new((height, width))?;
        let db = self.db.lock();
        db.put(
            &inventory.key.to_db_key(),
            &inventory.to_proto().encode_to_vec(),
        )?;
        Ok(inventory.key)
    }
    /// Get a readonly copy of an inventory.
    pub fn get(&self, id: &InventoryKey) -> Result<Option<Inventory>> {
        let db = self.db.lock();
        let bytes = db.get(&id.to_db_key())?;
        match bytes {
            Some(x) => {
                let inv_proto = items_proto::Inventory::decode(x.borrow())?;
                Ok(Some(Inventory::from_proto(inv_proto)?))
            }
            None => Ok(None),
        }
    }

    /// Run the given mutator on the indicated inventory.
    /// The function may mutate the data, or leave it as-is, and it may return a value
    /// to the caller through its own return value.
    ///
    /// TODO identify and document deadlock risks when calling other game state changing
    /// functions from the mutator
    pub fn mutate_inventory_atomically<F, T>(&self, id: &InventoryKey, mutator: F) -> Result<T>
    where
        F: FnOnce(&mut Inventory) -> Result<T>,
    {
        let db = self.db.lock();
        let key = id.to_db_key();
        let bytes = db.get(&key)?;
        let mut inv = match bytes {
            Some(x) => {
                let inv_proto = items_proto::Inventory::decode(x.borrow())?;
                Inventory::from_proto(inv_proto)?
            }
            None => {
                bail!("Inventory ID not found")
            }
        };
        let result = mutator(&mut inv);
        let new_bytes = inv.to_proto().encode_to_vec();
        db.put(&key, &new_bytes)?;
        self.broadcast_update(*id);
        result
    }

    /// Run the given mutator on the indicated vec of inventories.
    /// The function may mutate the data, or leave it as-is, and it may return a value to the caller
    /// through its own return value.
    ///
    /// See mutate_inventory_atomically for warnings regarding deadlock safety
    pub fn mutate_inventories_atomically<F, T>(&self, ids: &[InventoryKey], mutator: F) -> Result<T>
    where
        F: FnOnce(&mut Vec<Inventory>) -> Result<T>,
    {
        let db = self.db.lock();
        let mut inventories = Vec::with_capacity(ids.len());
        for id in ids {
            let bytes = db.get(&id.to_db_key())?;
            let inv = match bytes {
                Some(x) => {
                    let inv_proto = items_proto::Inventory::decode(x.borrow())?;
                    Inventory::from_proto(inv_proto)?
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
            db.put(&inv.key.to_db_key(), &new_bytes)?;
            self.broadcast_update(inv.key)
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

const BROADCAST_CHANNEL_SIZE: usize = 32;
