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
pub mod entities;
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

use anyhow::{bail, Result};
use futures::Future;
use integer_encoding::VarInt;
use log::{info, warn};
use parking_lot::Mutex;
use perovskite_core::time::TimeState;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

use crate::database::database_engine::{GameDatabase, KeySpace};

use crate::game_state::{game_map::ServerGameMap, mapgen::MapgenInterface};
use crate::media::MediaManager;
use crate::network_server::auth::AuthService;

use self::blocks::BlockTypeManager;
use self::chat::commands::CommandManager;
use self::chat::ChatState;
use self::game_behaviors::GameBehaviors;
use self::inventory::InventoryManager;
use self::items::ItemManager;
use self::player::PlayerManager;

/// The main struct representing a Perovskite server.
///
/// It contains the state of the server, including the map, the players, the inventory, etc.
///
/// Note that the game state is usually held in an Arc, and the reference count is used for
/// coordinating server shutdown. This means that if you hold a reference to the game state
/// and do not track shutdown, *the server will hang on shutdown and fail to flush data*.
///
/// **If you hold an Arc, you must await `await_start_shutdown` (possibly in a `tokio::select!` in your
/// main loop) or otherwise detect a shutdown, at which point you must do any cleanup actions and drop your Arc.**
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
    entities: Arc<entities::EntityManager>,
    server_start_time: Instant,
    extensions: type_map::concurrent::TypeMap,
    startup_time: Instant,
}

impl GameState {
    pub(crate) fn new(
        data_dir: PathBuf,
        db: Arc<dyn GameDatabase>,
        blocks: Arc<BlockTypeManager>,
        entity_types: Arc<entities::EntityTypeManager>,
        items: ItemManager,
        media: MediaManager,
        mapgen_provider: Box<dyn FnOnce(Arc<BlockTypeManager>, u32) -> Arc<dyn MapgenInterface>>,
        game_behaviors: GameBehaviors,
        commands: CommandManager,
        extensions: type_map::concurrent::TypeMap,
    ) -> Result<Arc<Self>> {
        let server_start_time = Instant::now();
        // TODO figure out a way to replace unwrap with error propagation
        let mapgen_seed = get_or_create_seed(db.as_ref(), b"mapgen_seed")?;
        let mapgen = mapgen_provider(blocks.clone(), mapgen_seed);
        // If we don't have a time of day yet, start in the morning.
        let time_of_day = get_double_meta_value(db.as_ref(), b"time_of_day")?.unwrap_or(0.25);
        let day_length = game_behaviors.day_length;
        let result = Arc::new_cyclic(|weak| Self {
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
            auth: AuthService::create(db.clone()).unwrap(),
            time_state: Mutex::new(TimeState::new(day_length, time_of_day)),
            player_state_resync: tokio::sync::watch::channel(()).0,
            entities: Arc::new(entities::EntityManager::new(
                db.clone(),
                entity_types,
                server_start_time,
            )),
            server_start_time,
            extensions,
            startup_time: Instant::now(),
        });
        result.entities.clone().start_workers(result.clone());
        Ok(result)
    }

    /// Gets the map for this game.
    pub fn game_map(&self) -> &ServerGameMap {
        self.map.as_ref()
    }

    pub(crate) fn game_map_clone(&self) -> Arc<ServerGameMap> {
        self.map.clone()
    }

    pub fn block_types(&self) -> &BlockTypeManager {
        self.map.block_type_manager()
    }

    /// The time since the server started, in nanoseconds.
    pub fn tick(&self) -> u64 {
        // This limits the server to a runtime of approximately 500 years.
        self.startup_time.elapsed().as_nanos().try_into().unwrap()
    }

    /// Sleep until the given tick
    pub async fn sleep_until_tick(&self, tick: u64) {
        let now = self.tick();
        if now < tick {
            tokio::time::sleep(Duration::from_nanos(tick - now)).await;
        }
    }

    pub fn tick_as_duration(&self) -> Duration {
        self.startup_time.elapsed()
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

    /// Returns true if the server is in the process of shutting down.
    pub fn is_shutting_down(&self) -> bool {
        self.early_shutdown.is_cancelled()
    }
    /// Returns a future that will resolve when the server is in the process of shutting down.
    pub async fn await_shutdown(&self) {
        self.early_shutdown.cancelled().await
    }

    /// Returns an extension that a plugin registered with [crate::server::ServerBuilder::add_extension],
    /// typically at some point during its initialization.
    ///
    /// This is meant for use by a plugin with its own extension (to access its own data).
    ///
    /// A more ergonomic API for *consumers* of a plugin would be to use a mixin trait:
    /// * Create a new public trait and implement it for GameState (coherence rules allow you to implement
    ///   your own trait)
    /// * In the implementation of the trait, call this method to get access to the desired plugin data
    pub fn extension<T: GameStateExtension>(&self) -> Option<&T> {
        self.extensions.get::<T>()
    }

    // Shut down things that handle events (e.g. map, database)
    // and wait for them to safely flush data.
    pub(crate) async fn shut_down(&self) {
        self.player_manager.request_shutdown();
        self.player_manager.await_shutdown().await.unwrap();

        self.map.do_shutdown().await.unwrap();
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

    pub fn entities(&self) -> &entities::EntityManager {
        &self.entities
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

    /// Sets the time of day. 0 is midnight, 1 is the next midnight.
    pub fn set_time_of_day(&self, time_of_day: f64) {
        self.time_state.lock().set_time(time_of_day);
        if self.player_state_resync.send(()).is_err() {
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
            Ok(Some(value))
        }
        None => Ok(None),
    }
}

pub trait GameStateExtension: Send + Sync + 'static {}

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
