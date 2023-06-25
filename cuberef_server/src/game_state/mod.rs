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
pub mod client_ui;
pub mod event;
pub mod game_behaviors;
pub mod game_map;
pub mod handlers;
pub mod inventory;
pub mod items;
pub mod mapgen;
pub mod player;

#[cfg(test)]
pub mod tests;

pub mod testutils;

use anyhow::{bail, Result};
use integer_encoding::VarInt;
use log::{info, warn};
use rand::distributions::Standard;
use rand::prelude::Distribution;
use std::fmt::Debug;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use crate::database::database_engine::{GameDatabase, KeySpace};

use crate::game_state::{game_map::ServerGameMap, mapgen::MapgenInterface};
use crate::media::MediaManager;
use crate::network_server::auth::AuthService;

use self::blocks::BlockTypeManager;
use self::game_behaviors::GameBehaviors;
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
    mapgen_seed: u32,
    game_behaviors: GameBehaviors,
    auth: AuthService
}

impl GameState {
    pub(crate) fn new(
        db: Arc<dyn GameDatabase>,
        blocks: Arc<BlockTypeManager>,
        items: ItemManager,
        media: MediaManager,
        mapgen_provider: Box<dyn FnOnce(Arc<BlockTypeManager>, u32) -> Arc<dyn MapgenInterface>>,
        game_behaviors: GameBehaviors,
    ) -> Result<Arc<Self>> {
        // TODO figure out a way to replace unwrap with error propagation
        let mapgen_seed = get_or_create_seed(db.as_ref(), b"mapgen_seed")?;
        let mapgen = mapgen_provider(blocks.clone(), mapgen_seed);
        Ok(Arc::new_cyclic(|weak| Self {
            map: ServerGameMap::new(weak.clone(), db.clone(), blocks).unwrap(),
            mapgen,
            database: db.clone(),
            inventory_manager: Arc::new(InventoryManager::new(db.clone())),
            item_manager: Arc::new(items),
            player_manager: PlayerManager::new(weak.clone(), db.clone()),
            media_resources: Arc::new(media),
            early_shutdown: CancellationToken::new(),
            mapgen_seed,
            game_behaviors,
            auth: AuthService::create(db).unwrap()
        }))
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

    pub fn game_behaviors(&self) -> &GameBehaviors {
        &self.game_behaviors
    }

    pub(crate) fn auth(&self) -> &AuthService {
        &self.auth
    }
}

fn get_or_create_seed<T: VarInt + Debug + Copy>(db: &dyn GameDatabase, name: &[u8]) -> Result<T>
where
    Standard: Distribution<T>,
{
    let mut key = b"seed_".to_vec();
    key.append(&mut name.to_vec());
    let key = KeySpace::Metadata.make_key(&key);
    match db.get(&key)? {
        Some(x) => match T::decode_var(&x) {
            Some((val, read)) => {
                if read != x.len() {
                    warn!(
                        "Saved seed for {} was {} bytes but only {} bytes were decoded (to {:?})",
                        String::from_utf8_lossy(name),
                        x.len(),
                        read,
                        val
                    )
                }
                info!(
                    "Loaded seed {:?} for {}",
                    val,
                    String::from_utf8_lossy(name)
                );
                Ok(val)
            }
            None => {
                bail!(
                    "Decoding varint for {} seed failed",
                    String::from_utf8_lossy(name)
                );
            }
        },
        None => {
            let seed: T = rand::random();
            db.put(&key, &seed.encode_var_vec())?;
            info!(
                "Generated seed {:?} for {}",
                seed,
                String::from_utf8_lossy(name)
            );
            Ok(seed)
        }
    }
}
