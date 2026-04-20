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
use std::collections::{HashMap, HashSet};

pub(crate) struct ClientItemManager {
    item_defs: HashMap<String, items_proto::ItemDef>,
    sorted_item_names: Vec<String>,
    groups: Vec<String>,
    plugin_prefixes: Vec<String>,
}
impl ClientItemManager {
    pub(crate) fn new(items: Vec<items_proto::ItemDef>) -> Result<ClientItemManager> {
        let mut item_defs = HashMap::new();
        let mut groups = HashSet::new();
        let mut plugin_prefixes = HashSet::new();
        for item in items {
            groups.extend(item.groups.iter().cloned());

            if item.short_name.contains(':') {
                if let Some(prefix) = item.short_name.split(':').next() {
                    plugin_prefixes.insert(prefix.to_string());
                }
            }

            match item_defs.entry(item.short_name.clone()) {
                std::collections::hash_map::Entry::Occupied(_) => {
                    return Err(anyhow!("Item {} already registered", item.short_name))
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(item);
                }
            }
        }
        let mut groups: Vec<String> = groups.into_iter().collect();
        groups.sort_unstable();
        let mut plugin_prefixes: Vec<String> = plugin_prefixes.into_iter().collect();
        plugin_prefixes.sort_unstable();
        let mut sorted_item_names: Vec<String> = item_defs.keys().cloned().collect();
        sorted_item_names.sort_unstable();
        Ok(ClientItemManager {
            item_defs,
            sorted_item_names,
            groups,
            plugin_prefixes,
        })
    }
    pub(crate) fn get(&self, name: &str) -> Option<&items_proto::ItemDef> {
        self.item_defs.get(name)
    }
    pub(crate) fn all_item_defs(&self) -> impl Iterator<Item = &items_proto::ItemDef> {
        self.item_defs.values()
    }
    pub(crate) fn groups(&self) -> &[String] {
        &self.groups
    }
    pub(crate) fn plugin_prefixes(&self) -> &[String] {
        &self.plugin_prefixes
    }
    pub(crate) fn sorted_items(&self) -> impl Iterator<Item = &items_proto::ItemDef> {
        self.sorted_item_names
            .iter()
            .map(|name| self.get(name).unwrap())
    }
}

#[derive(Debug)]
pub(crate) struct ClientInventory {
    _id: u64,
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
            _id: proto.view_id,
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
