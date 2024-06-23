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

use anyhow::anyhow;
use anyhow::Result;

use perovskite_core::protocol::items as items_proto;
use rustc_hash::FxHashMap;
use std::collections::HashMap;

use crate::vulkan::VulkanContext;

pub(crate) struct ClientItemManager {
    item_defs: HashMap<String, items_proto::ItemDef>,
}
impl ClientItemManager {
    pub(crate) fn new(
        items: Vec<items_proto::ItemDef>,
        _vk_ctx: &VulkanContext,
    ) -> Result<ClientItemManager> {
        let mut item_defs = HashMap::new();
        for item in items {
            match item_defs.entry(item.short_name.clone()) {
                std::collections::hash_map::Entry::Occupied(_) => {
                    return Err(anyhow!("Item {} already registered", item.short_name))
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(item);
                }
            }
        }
        Ok(ClientItemManager { item_defs })
    }
    pub(crate) fn get(&self, name: &str) -> Option<&items_proto::ItemDef> {
        self.item_defs.get(name)
    }
    pub(crate) fn all_item_defs(&self) -> impl Iterator<Item = &items_proto::ItemDef> {
        self.item_defs.values()
    }
}

#[derive(Debug)]
pub(crate) struct ClientInventory {
    id: u64,
    pub(crate) dimensions: (u32, u32),
    stacks: Vec<Option<items_proto::ItemStack>>,
    pub(crate) can_place: bool,
    pub(crate) can_take: bool,
    pub(crate) take_exact: bool,
    pub(crate) put_without_swap: bool,
}
impl ClientInventory {
    pub(crate) fn from_proto(
        proto: &perovskite_core::protocol::game_rpc::InventoryUpdate,
    ) -> ClientInventory {
        let inventory = proto.inventory.clone().unwrap();
        ClientInventory {
            id: proto.view_id,
            dimensions: (inventory.height, inventory.width),
            stacks: inventory
                .contents
                .into_iter()
                .map(|x| {
                    if x.item_name.is_empty() {
                        None
                    } else {
                        Some(x)
                    }
                })
                .collect(),
            can_place: proto.can_place,
            can_take: proto.can_take,
            take_exact: proto.take_exact,
            put_without_swap: proto.put_without_swap,
        }
    }
    pub(crate) fn contents(&self) -> &[Option<items_proto::ItemStack>] {
        &self.stacks
    }
    pub(crate) fn contents_mut(&mut self) -> &mut [Option<items_proto::ItemStack>] {
        &mut self.stacks
    }
}

pub(crate) struct InventoryViewManager {
    // TODO encapsulate these better
    // TODO stop leaking them
    pub(crate) inventory_views: FxHashMap<u64, ClientInventory>,
}
impl InventoryViewManager {
    pub(crate) fn new() -> InventoryViewManager {
        InventoryViewManager {
            inventory_views: FxHashMap::default(),
        }
    }
}
