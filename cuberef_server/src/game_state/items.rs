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

use anyhow::{anyhow, Result};
use cuberef_core::protocol::items::item_def::QuantityType;

use std::collections::HashMap;

use super::blocks::BlockTypeHandle;
use super::event::HandlerContext;

use cuberef_core::coordinates::BlockCoordinate;
use cuberef_core::protocol::items as proto;

/// Result of the dig_handler of an Item.
pub struct DigResult {
    /// An updated version of the item (stack) that was used to dig.
    /// If None, the item disappears (e.g. a pickaxe breaks when its durability runs out)
    /// Does not need to be the same as the original item.
    pub updated_tool: Option<ItemStack>,
    /// Items that were obtained from digging and ought to be added to the player's inventory
    pub obtained_items: Vec<ItemStack>,
}
/// (handler context, coordinate of the block, the block seen on the map then, the item stack in use)
pub type BlockInteractionHandler = dyn Fn(HandlerContext, BlockCoordinate, BlockTypeHandle, &ItemStack) -> Result<DigResult>
    + Send
    + Sync;
/// The parameters are handler context, location where the new block is being placed, anchor block, and the item stack in use.
/// The anchor block is the existing block that the player was pointing to when they clicked the place button.
pub type PlaceHandler = dyn Fn(HandlerContext, BlockCoordinate, BlockCoordinate, &ItemStack) -> Result<Option<ItemStack>>
    + Send
    + Sync;

pub struct Item {
    pub proto: cuberef_core::protocol::items::ItemDef,

    /// Called when the item is used to dig a block.
    /// If this handler is Some, it will override the default behavior to call the dig handler
    /// for the block in question.
    ///
    /// If the block should still be dug in the normal way, this handler is responsible for
    /// calling game_map().dig_block(...). It may call that function multiple times, e.g.
    /// if a tool digs multiple coordinates in one activation.
    ///
    /// If None, the current item stack will not be updated, and the block's dig handler will be run.
    pub dig_handler: Option<Box<BlockInteractionHandler>>,
    /// Same as dig_handler, but called when the block is briefly tapped with the left mouse button without
    /// digging it fully.
    pub tap_handler: Option<Box<BlockInteractionHandler>>,
    /// Called when the itemstack is placed (typicall with rightclick).
    /// If this handler is None, nothing happens.
    /// If this handler is Some, it should call a suitable function of ctx.game_map() if it
    /// wishes to place a block, and then return an updated ItemStack (or None to delete the itemstack)
    ///
    /// The parameters are handler context, location where the new block is being placed, anchor block, and the item stack in use.
    /// The anchor block is the existing block that the player was pointing to when they clicked the place button.
    pub place_handler: Option<Box<PlaceHandler>>,
}

#[derive(Debug, Clone)]
pub struct ItemStack {
    pub proto: proto::ItemStack,
}
impl ItemStack {
    /// Creates an ItemStack of the given item
    pub fn new(item: &Item, quantity_or_wear: u32) -> ItemStack {
        match item.proto.quantity_type {
            Some(QuantityType::Stack(x)) => ItemStack {
                proto: proto::ItemStack {
                    item_name: item.proto.short_name.clone(),
                    quantity: quantity_or_wear,
                    current_wear: 1,
                    quantity_type: Some(proto::item_stack::QuantityType::Stack(x)),
                },
            },
            Some(QuantityType::Wear(x)) => ItemStack {
                proto: proto::ItemStack {
                    item_name: item.proto.short_name.clone(),
                    quantity: 1,
                    current_wear: quantity_or_wear,
                    quantity_type: Some(proto::item_stack::QuantityType::Wear(x)),
                },
            },
            None => ItemStack {
                proto: proto::ItemStack {
                    item_name: item.proto.short_name.clone(),
                    quantity: 1,
                    current_wear: 1,
                    quantity_type: None,
                },
            },
        }
    }
    pub(crate) fn from_proto(proto: &proto::ItemStack) -> Option<ItemStack> {
        if proto.item_name.is_empty() {
            None
        } else {
            Some(ItemStack {
                proto: proto.clone(),
            })
        }
    }

    pub fn proto(&self) -> &proto::ItemStack {
        &self.proto
    }

    /// Tries to merge the provided stack into this one, returning any leftover.
    pub fn try_merge(&mut self, other: ItemStack) -> Option<ItemStack> {
        if self.proto.item_name != other.proto.item_name {
            return Some(other);
        }
        // other isn't stackable
        if !matches!(
            other.proto.quantity_type,
            Some(proto::item_stack::QuantityType::Stack(_))
        ) {
            return Some(other);
        }
        if let Some(proto::item_stack::QuantityType::Stack(max_stack)) = self.proto.quantity_type {
            let move_size = other
                .proto
                .quantity
                .min(max_stack.saturating_sub(self.proto.quantity));
            self.proto.quantity += move_size;
            let remaining = other.proto.quantity - move_size;
            if remaining == 0 {
                None
            } else {
                Some(ItemStack {
                    proto: proto::ItemStack {
                        quantity: remaining,
                        ..other.proto
                    },
                })
            }
        } else {
            // we aren't stackable
            return Some(other);
        }
    }

    /// Tries to merge the provided stack into this one, without allowing leftovers. Returns true on success, false (and self is unmodified) on failure
    pub fn try_merge_all(&mut self, other: ItemStack) -> bool {
        if self.proto.item_name != other.proto.item_name {
            return false;
        }
        // other isn't stackable
        if !matches!(
            other.proto.quantity_type,
            Some(proto::item_stack::QuantityType::Stack(_))
        ) {
            return false;
        }
        if let Some(proto::item_stack::QuantityType::Stack(max_stack)) = self.proto.quantity_type {
            let available_space = max_stack.saturating_sub(self.proto.quantity);
            if available_space < other.proto.quantity {
                return false;
            }
            self.proto.quantity += other.proto.quantity;
            true
        } else {
            false
        }
    }

    pub fn decrement(&self) -> Option<ItemStack> {
        match self.proto.quantity {
            0 | 1 => None,
            x => Some(ItemStack {
                proto: proto::ItemStack {
                    quantity: x - 1,
                    ..self.proto.clone()
                },
            }),
        }
    }
}

pub trait MaybeStack {
    /// Try to merge the provided stack into this one. Returns leftovers.
    fn try_merge(&mut self, other: Self) -> Self;
    /// Try to merge the provided stack into this one. Do not allow leftovers. True if merged successfully.
    fn try_merge_all(&mut self, other: Self) -> bool;
    /// Try to take some subset of items (or the entire stack if count is None).
    /// Returns what could be taken.
    fn take_items(&mut self, count: Option<u32>) -> Self;
    /// Try to take the requested number of items (or the entire stack if count is None).
    /// If the stack doesn't contain enough items, None is returned.
    fn try_take_all(&mut self, count: Option<u32>) -> Self;
}
impl MaybeStack for Option<ItemStack> {
    fn try_merge(&mut self, other: Option<ItemStack>) -> Option<ItemStack> {
        match other {
            Some(other) => match self {
                Some(self_inner) => ItemStack::try_merge(self_inner, other),
                None => {
                    *self = Some(other);
                    None
                }
            },
            None => None,
        }
    }

    fn try_merge_all(&mut self, other: Self) -> bool {
        match other {
            Some(other) => {
                match self {
                    Some(self_inner) => ItemStack::try_merge_all(self_inner, other),
                    None => {
                        *self = Some(other);
                        // We can always insert into an empty stack.
                        true
                    }
                }
            }
            // If the other stack is empty, we have nothing to insert, which we can always do successfully.
            None => true,
        }
    }

    fn take_items(&mut self, count: Option<u32>) -> Option<ItemStack> {
        match count {
            Some(count) => {
                if let Some(self_stack) = self.as_mut() {
                    if let Some(proto::item_stack::QuantityType::Stack(_)) =
                        self_stack.proto.quantity_type
                    {
                        let self_count = self_stack.proto.quantity;
                        let taken = self_count.min(count);
                        let remaining = self_count.saturating_sub(count);
                        if remaining == 0 {
                            self.take()
                        } else {
                            self_stack.proto.quantity = remaining;
                            Some(ItemStack {
                                proto: proto::ItemStack {
                                    quantity: taken,
                                    ..self_stack.proto.clone()
                                },
                            })
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            None => self.take(),
        }
    }

    fn try_take_all(&mut self, count: Option<u32>) -> Self {
        match count {
            Some(count) => {
                if let Some(self_stack) = self.as_mut() {
                    if let Some(proto::item_stack::QuantityType::Stack(_)) =
                        self_stack.proto.quantity_type
                    {
                        let available = self.as_mut().unwrap().proto.quantity;
                        match available.cmp(&count) {
                            std::cmp::Ordering::Less => None,
                            std::cmp::Ordering::Equal => self.take(),
                            std::cmp::Ordering::Greater => {
                                self.as_mut().unwrap().proto.quantity -= count;
                                Some(ItemStack {
                                    proto: proto::ItemStack {
                                        quantity: count,
                                        ..self.as_ref().unwrap().proto.clone()
                                    },
                                })
                            }
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            None => self.take(),
        }
    }
}

pub struct ItemManager {
    items: HashMap<String, Item>,
}
impl ItemManager {
    pub fn from_stack(&self, stack: &ItemStack) -> Option<&Item> {
        self.get_item(&stack.proto.item_name)
    }
    pub fn get_item(&self, name: &str) -> Option<&Item> {
        self.items.get(name)
    }
    pub fn register_item(&mut self, item: Item) -> Result<()> {
        match self.items.entry(item.proto.short_name.clone()) {
            std::collections::hash_map::Entry::Occupied(_) => {
                Err(anyhow!("Item {} already registered", item.proto.short_name))
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                log::info!("Registering item {}", item.proto.short_name);
                entry.insert(item);
                Ok(())
            }
        }
    }

    pub(crate) fn new() -> ItemManager {
        ItemManager {
            items: HashMap::new(),
        }
    }

    pub fn registered_items(&self) -> impl Iterator<Item = &Item> {
        self.items.values()
    }
}
