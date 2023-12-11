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

use anyhow::{anyhow, ensure, Result};
use lazy_static::lazy_static;
use perovskite_core::constants::items::default_item_interaction_rules;
use perovskite_core::protocol::items::item_def::QuantityType;
use rustc_hash::FxHashSet;

use std::collections::HashMap;
use std::time::Duration;

use super::blocks::BlockType;
use super::event::HandlerContext;

use perovskite_core::coordinates::BlockCoordinate;
use perovskite_core::protocol::items as proto;

/// Result of the dig_handler of an Item.
pub struct ItemInteractionResult {
    /// An updated version of the item (stack) that was used to dig.
    /// If None, the item disappears (e.g. a pickaxe breaks when its durability runs out)
    /// Does not need to be the same as the original item.
    pub updated_tool: Option<ItemStack>,
    /// Items that were obtained from digging and ought to be added to the player's inventory
    pub obtained_items: Vec<ItemStack>,
}
/// (handler context, coordinate of the block, the item stack in use)
pub type BlockInteractionHandler = dyn Fn(&HandlerContext, BlockCoordinate, &ItemStack) -> Result<ItemInteractionResult>
    + Send
    + Sync;
/// The parameters are handler context, location where the new block is being placed, anchor block, and the item stack in use.
/// The anchor block is the existing block that the player was pointing to when they clicked the place button.
pub type PlaceHandler = dyn Fn(&HandlerContext, BlockCoordinate, BlockCoordinate, &ItemStack) -> Result<Option<ItemStack>>
    + Send
    + Sync;

pub struct Item {
    pub proto: perovskite_core::protocol::items::ItemDef,

    /// Called when the item is used to dig a block.
    /// If this handler is Some, it will override the default behavior to call the dig handler
    /// for the block in question.
    ///
    /// If the block should still be dug in the normal way, this handler is responsible for
    /// calling game_map().dig_block(...). It may call that function multiple times, e.g.
    /// if a tool digs multiple coordinates in one activation.
    /// 
    /// Note - if a plugin calls game_map().dig_block directly, then this handler will not be called.
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

impl Item {
    /// Creates an ItemStack of the given item
    pub fn make_stack(&self, quantity_or_wear: u32) -> ItemStack {
        match self.proto.quantity_type {
            Some(QuantityType::Stack(x)) => ItemStack {
                proto: proto::ItemStack {
                    item_name: self.proto.short_name.clone(),
                    quantity: quantity_or_wear,
                    current_wear: 1,
                    quantity_type: Some(proto::item_stack::QuantityType::Stack(x)),
                },
            },
            Some(QuantityType::Wear(x)) => ItemStack {
                proto: proto::ItemStack {
                    item_name: self.proto.short_name.clone(),
                    quantity: 1,
                    current_wear: quantity_or_wear,
                    quantity_type: Some(proto::item_stack::QuantityType::Wear(x)),
                },
            },
            None => ItemStack {
                proto: proto::ItemStack {
                    item_name: self.proto.short_name.clone(),
                    quantity: 1,
                    current_wear: 1,
                    quantity_type: None,
                },
            },
        }
    }

    /// Creates a singleton itemstack of this item, i.e. one copy, at full wear if applicable
    pub fn singleton_stack(&self) -> ItemStack {
        match self.proto.quantity_type {
            Some(QuantityType::Wear(x)) => ItemStack {
                proto: proto::ItemStack {
                    item_name: self.proto.short_name.clone(),
                    quantity: 1,
                    current_wear: x,
                    quantity_type: Some(proto::item_stack::QuantityType::Wear(x)),
                },
            },
            _ => self.make_stack(1),
        }
    }

    /// Creates a maximal stack of this item, i.e. as many copies as possible if it's stackable
    pub fn make_max_stack(&self) -> ItemStack {
        match self.proto.quantity_type {
            Some(QuantityType::Stack(x)) => ItemStack {
                proto: proto::ItemStack {
                    item_name: self.proto.short_name.clone(),
                    quantity: x,
                    current_wear: 1,
                    quantity_type: Some(proto::item_stack::QuantityType::Stack(x)),
                },
            },
            _ => self.singleton_stack(),
        }
    }

    pub fn stackable(&self) -> bool {
        matches!(
            self.proto.quantity_type,
            Some(proto::item_def::QuantityType::Stack(_))
        )
    }
}

fn eval_interaction_rules<'a>(rules: &'a [proto::InteractionRule], block_type: &BlockType) -> Option<&'a proto::InteractionRule> {
    let block_groups = block_type
        .client_info
        .groups
        .iter()
        .collect::<FxHashSet<_>>();
    rules.iter().find(|rule| rule.block_group.iter().all(|x| block_groups.contains(x)))
}

impl HasInteractionRules for &Item {
    fn get_interaction_rule(&self, block_type: &BlockType) -> Option<&proto::InteractionRule> {
        eval_interaction_rules(&self.proto.interaction_rules, block_type)
    }
}

#[derive(Debug, Clone)]
pub struct ItemStack {
    pub proto: proto::ItemStack,
}
impl ItemStack {
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
            Some(other)
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

    pub fn stackable(&self) -> bool {
        matches!(
            self.proto.quantity_type,
            Some(proto::item_stack::QuantityType::Stack(_))
        )
    }
    pub fn has_wear(&self) -> bool {
        matches!(
            self.proto.quantity_type,
            Some(proto::item_stack::QuantityType::Wear(_))
        )
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
                    } else if self_stack.proto.quantity == count {
                        self.take()
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
                    } else if self_stack.proto.quantity == count {
                        self.take()
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
    /// Gets the item corresponding to the given stack, if it is defined. Returns None if the stack was itself None,
    /// or if the item was not defined (e.g. if the stack was created and then the server restarted with the item removed,
    /// or if the stack contains a totally invalid item name)
    pub fn from_stack(&self, stack: Option<&ItemStack>) -> Option<&Item> {
        stack.and_then(|stack| self.get_item(&stack.proto.item_name))
    }
    pub fn get_item(&self, name: &str) -> Option<&Item> {
        self.items.get(name)
    }
    pub fn register_item(&mut self, item: Item) -> Result<()> {
        if item.proto.short_name.is_empty() {
            return Err(anyhow!("Item must have a non-empty short name"));
        }
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

    pub(crate) fn new() -> Result<ItemManager> {
        let mut manager = ItemManager {
            items: HashMap::new(),
        };
        manager.register_defaults()?;
        Ok(manager)
    }

    pub fn registered_items(&self) -> impl Iterator<Item = &Item> {
        self.items.values()
    }

    fn register_defaults(&mut self) -> Result<()> {
        self.register_item(Item {
            proto: proto::ItemDef {
                short_name: NO_TOOL.to_string(),
                display_name: "No Tool (you should never see this)".to_string(),
                inventory_texture: None,
                groups: vec![],
                interaction_rules: default_item_interaction_rules(),
                quantity_type: None,
                ..Default::default()
            },
            dig_handler: None,
            tap_handler: None,
            place_handler: None,
        })?;
        Ok(())
    }
}

pub trait InteractionRuleExt {
    /// Returns true if the given rule matches the given block
    fn matches(&self, block_type: &BlockType) -> bool;

    /// Returns the time expected to dig this block, or None if it's not possible
    fn dig_time(&self, block: &BlockType) -> Option<Duration>;

    /// Returns the tool wear obtained from digging with this rule
    fn tool_wear(&self, block: &BlockType) -> Result<u32>;
}
impl InteractionRuleExt for proto::InteractionRule {
    fn matches(&self, block_type: &BlockType) -> bool {
        self.block_group
            .iter()
            .all(|x| block_type.client_info.groups.contains(x))
    }

    fn dig_time(&self, block: &BlockType) -> Option<Duration> {
        use proto::interaction_rule::DigBehavior::*;
        match self.dig_behavior {
            Some(ConstantTime(t)) => Some(Duration::from_secs_f64(t)),
            Some(ScaledTime(t)) => {
                Some(Duration::from_secs_f64(t * block.client_info.base_dig_time))
            }
            Some(InstantDig(_) | InstantDigOneshot(_)) => Some(Duration::ZERO),
            Some(Undiggable(_)) | None => None,
        }
    }

    fn tool_wear(&self, block: &BlockType) -> Result<u32> {
        let product = self.tool_wear as f64 * block.client_info.wear_multiplier;
        ensure!(
            product.is_finite(),
            "Tool wear must be finite (not NaN or inf)"
        );
        ensure!(product >= 0.0, "Tool wear cannot be negative");
        ensure!(
            product < u32::MAX as f64,
            "Tool wear overflowed u32::MAX, was {}",
            product
        );
        Ok(product as u32)
    }
}

pub(crate) fn default_dig_handler(
    ctx: &HandlerContext,
    coord: BlockCoordinate,
    stack: &ItemStack,
) -> Result<ItemInteractionResult> {
    let dig_result = ctx
        .game_map()
        .dig_block(coord, ctx.initiator(), Some(stack))?;
    let mut stack = stack.clone();
    if stack.has_wear() {
        stack.proto.current_wear = stack
            .proto
            .current_wear
            .saturating_sub(dig_result.tool_wear)
    }
    Ok(ItemInteractionResult {
        updated_tool: if stack.proto.current_wear > 0 {
            Some(stack.clone())
        } else {
            None
        },
        obtained_items: dig_result.item_stacks,
    })
}

pub trait HasInteractionRules {
    fn get_interaction_rule(&self, block_type: &BlockType) -> Option<&proto::InteractionRule>;
}

impl HasInteractionRules for Option<&Item> {
    fn get_interaction_rule(&self, block_type: &BlockType) -> Option<&proto::InteractionRule> {
        if let Some(item) = self {
            item.get_interaction_rule(block_type)
        } else {
            eval_interaction_rules(&DEFAULT_INTERACTION_RULES, block_type)
        }
    }
}

lazy_static!{
    static ref DEFAULT_INTERACTION_RULES: Vec<proto::InteractionRule> = default_item_interaction_rules();
}

pub(crate) const NO_TOOL: &str = "internal:no_tool";

// This is a small hack, allowing item handlers to take an ItemStack rather than an Option<ItemStack>
pub(crate) fn make_fake_item_for_no_tool() -> ItemStack {
    ItemStack {
        proto: proto::ItemStack {
            item_name: NO_TOOL.to_string(),
            quantity: 0,
            current_wear: 0,
            quantity_type: None,
        },
    }
}

lazy_static! {
    pub(crate) static ref NO_TOOL_STACK: ItemStack = make_fake_item_for_no_tool();
}

pub(crate) fn default_tap_handler(
    ctx: &HandlerContext,
    coord: BlockCoordinate,
    stack: &ItemStack,
) -> Result<ItemInteractionResult> {
    let dig_result = ctx
        .game_map()
        .tap_block(coord, ctx.initiator(), Some(stack))?;
    let mut stack = stack.clone();
    if stack.has_wear() {
        stack.proto.quantity = stack.proto.quantity.saturating_sub(dig_result.tool_wear)
    }
    Ok(ItemInteractionResult {
        updated_tool: if stack.proto.quantity > 0 {
            Some(stack.clone())
        } else {
            None
        },
        obtained_items: dig_result.item_stacks,
    })
}
