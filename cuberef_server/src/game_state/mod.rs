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

pub mod blocks;
pub mod event;
pub mod game_map;
pub mod handlers;
pub mod inventory;
pub mod items;
pub mod mapgen;
pub mod player;

#[cfg(test)]
pub mod tests;

pub mod testutils;

use anyhow::Result;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use crate::database::database_engine::GameDatabase;

use crate::game_state::{game_map::ServerGameMap, mapgen::MapgenInterface};
use crate::resources::MediaManager;

use self::blocks::BlockTypeManager;
use self::inventory::InventoryManager;
use self::items::ItemManager;
use self::player::PlayerManager;

pub struct GameState {
    map: Arc<ServerGameMap>,
    mapgen: Arc<dyn MapgenInterface>,
    database: Arc<dyn GameDatabase>,
    inventory_manager: Arc<InventoryManager>,
    item_manager: Arc<ItemManager>,
    player_manager: Arc<PlayerManager>,
    media_resources: Arc<MediaManager>,
    early_shutdown: CancellationToken,
}

impl GameState {
    pub(crate) fn new(
        db: Arc<dyn GameDatabase>,
        blocks: Arc<BlockTypeManager>,
        items: ItemManager,
        media: MediaManager,
        mapgen: Arc<dyn MapgenInterface>,
    ) -> Result<Arc<Self>> {
        // TODO figure out a way to replace unwrap with error propagation
        Ok(Arc::new_cyclic(|weak| Self {
            map: ServerGameMap::new(weak.clone(), db.clone(), blocks).unwrap(),
            mapgen,
            database: db.clone(),
            inventory_manager: Arc::new(InventoryManager::new(db.clone())),
            item_manager: Arc::new(items),
            player_manager: PlayerManager::new(weak.clone(), db),
            media_resources: Arc::new(media),
            early_shutdown: CancellationToken::new(),
        }))
    }

    pub(crate) fn new_testonly(
        map: Arc<ServerGameMap>,
        mapgen: Arc<dyn MapgenInterface>,
        database: Arc<dyn GameDatabase>,
        inventory_manager: Arc<InventoryManager>,
        item_manager: Arc<ItemManager>,
        player_manager: Arc<PlayerManager>,
        media_resources: Arc<MediaManager>,
        shutdown: CancellationToken,
    ) -> Self {
        Self {
            map,
            mapgen,
            database,
            inventory_manager,
            item_manager,
            player_manager,
            media_resources,
            early_shutdown: shutdown,
        }
    }

    /// Gets the map for this game.
    pub fn map(&self) -> &ServerGameMap {
        self.map.as_ref()
    }

    /// Not implemented yet, and its semantics are still TBD
    pub fn tick(&self) -> u64 {
        0 // TODO
    }

    pub(crate) fn mapgen(&self) -> &dyn MapgenInterface {
        self.mapgen.as_ref()
    }

    pub(crate) fn db(&self) -> &dyn GameDatabase {
        self.database.as_ref()
    }

    pub(crate) fn media_resources(&self) -> &MediaManager {
        &self.media_resources
    }

    /// Start shutting down the game.
    /// This will kick all players off.
    pub fn start_shutdown(&self) {
        self.early_shutdown.cancel();
    }

    // Shut down things that handle events (e.g. map, database)
    // and wait for them to safely flush data.
    pub(crate) async fn finish_shutdown(&self) {
        self.map.request_shutdown();
        self.player_manager.request_shutdown();
        self.map.await_shutdown().await.unwrap();
        self.player_manager.await_shutdown().await.unwrap();
    }

    // Await a call to self.start_shutdown
    pub async fn await_start_shutdown(&self) {
        self.early_shutdown.cancelled().await
    }

    pub fn inventory_manager(&self) -> &InventoryManager {
        self.inventory_manager.as_ref()
    }

    pub fn player_manager(&self) -> &Arc<PlayerManager> {
        &self.player_manager
    }

    pub fn item_manager(&self) -> &ItemManager {
        self.item_manager.as_ref()
    }
}
