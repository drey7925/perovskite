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
pub mod chat;
pub mod client_ui;
pub mod event;
pub mod game_behaviors;
pub mod game_map;
pub mod handlers;
pub mod inventory;
pub mod items;
pub mod mapgen;
pub mod player;
pub mod entities;

#[cfg(test)]
pub mod tests;

use anyhow::{bail, Result};
use integer_encoding::VarInt;
use log::{info, warn};
use parking_lot::Mutex;
use perovskite_core::time::TimeState;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::database::database_engine::{GameDatabase, KeySpace};

use crate::game_state::{game_map::ServerGameMap, mapgen::MapgenInterface};
use crate::media::MediaManager;
use crate::network_server::auth::AuthService;

use self::blocks::BlockTypeManager;
use self::chat::ChatState;
use self::chat::commands::CommandManager;
use self::entities::EntityManager;
use self::game_behaviors::GameBehaviors;
use self::inventory::InventoryManager;
use self::items::ItemManager;
use self::player::PlayerManager;

pub struct GameState {
    data_dir: PathBuf,
    map: Arc<ServerGameMap>,
    mapgen: Arc<dyn MapgenInterface>,
    database: Arc<dyn GameDatabase>,
    inventory_manager: Arc<InventoryManager>,
    item_manager: Arc<ItemManager>,
    player_manager: Arc<PlayerManager>,
    media_resources: Arc<MediaManager>,
    chat: Arc<ChatState>,
    early_shutdown: CancellationToken,
    mapgen_seed: u32,
    game_behaviors: GameBehaviors,
    auth: AuthService,
    time_state: Mutex<TimeState>,
    player_state_resync: tokio::sync::watch::Sender<()>,
    entities: EntityManager,
    server_start_time: Instant,
}

impl GameState {
    pub(crate) fn new(
        data_dir: PathBuf,
        db: Arc<dyn GameDatabase>,
        blocks: Arc<BlockTypeManager>,
        items: ItemManager,
        media: MediaManager,
        mapgen_provider: Box<dyn FnOnce(Arc<BlockTypeManager>, u32) -> Arc<dyn MapgenInterface>>,
        game_behaviors: GameBehaviors,
        commands: CommandManager
    ) -> Result<Arc<Self>> {
        // TODO figure out a way to replace unwrap with error propagation
        let mapgen_seed = get_or_create_seed(db.as_ref(), b"mapgen_seed")?;
        let mapgen = mapgen_provider(blocks.clone(), mapgen_seed);
        // If we don't have a time of day yet, start in the morning.
        let time_of_day = get_double_meta_value(db.as_ref(), b"time_of_day")?.unwrap_or(0.25);
        let day_length = game_behaviors.day_length;
        Ok(Arc::new_cyclic(|weak| Self {
            data_dir,
            map: ServerGameMap::new(weak.clone(), db.clone(), blocks).unwrap(),
            mapgen,
            database: db.clone(),
            inventory_manager: Arc::new(InventoryManager::new(db.clone())),
            item_manager: Arc::new(items),
            player_manager: PlayerManager::new(weak.clone(), db.clone()).unwrap(),
            media_resources: Arc::new(media),
            chat: Arc::new(ChatState::new(commands)),
            early_shutdown: CancellationToken::new(),
            mapgen_seed,
            game_behaviors,
            auth: AuthService::create(db).unwrap(),
            time_state: Mutex::new(TimeState::new(day_length, time_of_day)),
            player_state_resync: tokio::sync::watch::channel(()).0,
            entities: EntityManager::new(weak.clone()),
            server_start_time: Instant::now(),
        }))
    }

    /// Gets the map for this game.
    pub fn game_map(&self) -> &ServerGameMap {
        self.map.as_ref()
    }

    pub fn block_types(&self) -> &BlockTypeManager {
        self.map.block_type_manager()
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
        put_double_meta_value(
            self.database.as_ref(),
            b"time_of_day",
            self.time_state().lock().time_of_day(),
        )
        .unwrap();
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

    pub fn chat(&self) -> &ChatState {
        &self.chat
    }

    pub(crate) fn auth(&self) -> &AuthService {
        &self.auth
    }

    pub(crate) fn time_state(&self) -> &Mutex<TimeState> {
        &self.time_state
    }

    pub(crate) fn entities(&self) -> &EntityManager {
        &self.entities
    }

    /// Sets the time of day. 0 is midnight, 1 is the next midnight.
    pub fn set_time_of_day(&self, time_of_day: f64) {
        self.time_state.lock().set_time(time_of_day);
        if let Err(_) = self.player_state_resync.send(()) {
            tracing::info!("Player resync had no closures waiting for it.");
            // pass, nobody was listening.
        }
    }

    pub(crate) fn subscribe_player_state_resyncs(&self) -> tokio::sync::watch::Receiver<()> {
        self.player_state_resync.subscribe()
    }

    pub fn data_dir(&self) -> &PathBuf {
        &self.data_dir
    }

    /// Returns the time when the server started.
    pub fn server_start_time(&self) -> Instant {
        self.server_start_time
    }
}

fn get_double_meta_value(db: &dyn GameDatabase, name: &[u8]) -> Result<Option<f64>>
where
    Standard: Distribution<f64>,
{
    let mut key = b"double_".to_vec();
    key.append(&mut name.to_vec());
    let key = KeySpace::Metadata.make_key(&key);
    match db.get(&key)? {
        Some(x) => {
            if x.len() != 8 {
                bail!(
                    "Decoding double for {} metadata failed",
                    String::from_utf8_lossy(name)
                )
            }
            let value = f64::from_le_bytes(x.try_into().unwrap());
            if value.is_nan() {
                bail!(
                    "Decoding double for {} metadata failed - got NaN",
                    String::from_utf8_lossy(name)
                )
            }
            return Ok(Some(value));
        }
        None => {
            return Ok(None);
        }
    }
}

fn put_double_meta_value(db: &dyn GameDatabase, name: &[u8], value: f64) -> Result<()> {
    let mut key = b"double_".to_vec();
    key.append(&mut name.to_vec());
    let key = KeySpace::Metadata.make_key(&key);
    db.put(&key, &value.to_le_bytes())?;
    Ok(())
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
