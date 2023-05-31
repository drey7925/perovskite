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
use cuberef_core::protocol::items as proto;
use cuberef_core::coordinates::BlockCoordinate;

/// Result of the dig_handler of an Item.
pub struct DigResult {
    /// An updated version of the item (stack) that was used to dig.
    /// If None, the item disappears (e.g. a pickaxe breaks when its durability runs out)
    /// Does not need to be the same as the original item.
    pub updated_tool: Option<ItemStack>,
    /// Items that were obtained from digging and ought to be added to the player's inventory
    pub obtained_items: Vec<ItemStack>,
}

pub type BlockInteractionHandler =
    dyn Fn(HandlerContext, BlockCoordinate, BlockTypeHandle, &ItemStack) -> Result<DigResult> + Send + Sync;
pub type PlaceHandler =
    dyn Fn(HandlerContext, BlockCoordinate, BlockCoordinate, &ItemStack) -> Result<Option<ItemStack>> + Send + Sync;

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
    proto: proto::ItemStack,
}
impl ItemStack {
    /// Creates an ItemStack of the given item
    pub(crate) fn new(item: &Item, quantity: u32) -> ItemStack {
        ItemStack {
            proto: proto::ItemStack {
                item_name: item.proto.short_name.clone(),
                quantity,
                max_stack: match item.proto.quantity_type {
                    Some(QuantityType::Stack(x)) => x,
                    Some(QuantityType::Wear(_)) => 0,
                    None => 0,
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

    pub(crate) fn proto(&self) -> &proto::ItemStack {
        &self.proto
    }

    /// Tries to merge the provided stack into this one, returning any leftover.
    pub fn try_merge(&mut self, stack: ItemStack) -> Option<ItemStack> {
        // We aren't stackable.
        if self.proto.max_stack == 0 {
            return Some(stack);
        }
        // The other stack is either non-stackable or wear-based. Don't try to stack,
        // even if we have the same item name and think it's stackable.
        if stack.proto.max_stack == 0 {
            return Some(stack);
        }
        if self.proto.item_name != stack.proto.item_name {
            return Some(stack);
        }

        let move_size = stack
            .proto
            .quantity
            .min(self.proto.max_stack.saturating_sub(self.proto.quantity));

        self.proto.quantity += move_size;
        let remaining = stack.proto.quantity - move_size;
        if remaining == 0 {
            None
        } else {
            Some(ItemStack {
                proto: proto::ItemStack {
                    quantity: remaining,
                    ..stack.proto
                },
                ..stack
            })
        }
    }

    pub(crate) fn decrement(&self) -> Option<ItemStack> {
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

    pub(crate) fn item_types(&self) -> impl Iterator<Item = &Item> {
        self.items.values()
    }
}
