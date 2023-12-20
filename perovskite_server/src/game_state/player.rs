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

use std::{
    collections::{hash_map::Entry, HashMap, HashSet},
    ops::Deref,
    pin::Pin,
    sync::{Arc, Weak},
    time::{Duration, Instant},
};

use anyhow::{bail, ensure, Context, Result};
use cgmath::{vec3, Vector3, Zero};
use futures::Future;
use perovskite_core::{
    chat::ChatMessage,
    coordinates::PlayerPositionUpdate,
    protocol::{game_rpc::InventoryAction, players::StoredPlayer},
};

use log::warn;
use parking_lot::{Mutex, MutexGuard, RwLock};
use prost::Message;
use tokio::{select, task::JoinHandle};
use tokio_util::sync::CancellationToken;

use crate::{
    database::database_engine::{GameDatabase, KeySpace},
    game_state::inventory::InventoryViewWithContext,
    network_server::auth::AuthOutcome,
};

use super::{
    client_ui::Popup,
    entities::EntityId,
    event::{EventInitiator, PlayerInitiator},
    inventory::{InventoryKey, InventoryView, InventoryViewId, TypeErasedInventoryView},
    GameState,
};

pub struct Player {
    // Player's in-game name
    pub(crate) name: String,
    // Player's inventory (immutable key that can be used to access mutable inventory)
    // The main inventory is a bit special - it's drawn in the HUD and used for interactions
    pub(crate) main_inventory_key: InventoryKey,
    // Mutable state of the player
    // Impl note: If mutating access is provided to callers, then we have no way of assuring that
    // client(s) are notified of the change.
    //
    // We should ensure that the only way to get mutating access is through something like a PlayerMutView that
    // Derefs to a player and has its own mutating functions that appropriate track whether any updates need
    // to be sent to clients, or only through this struct (which can maintain the necessary invariant)
    pub(crate) state: Mutex<PlayerState>,
    // Sends events for that player; the corresponding receiver will send them over the network to the client.
    sender: PlayerEventSender,
    game_state: Arc<GameState>,
    // TODO - refactor according to the final design of entities
    pub(crate) entity_id: EntityId,
}

impl Player {
    pub fn name(&self) -> &str {
        self.name.as_ref()
    }
    /// Returns the inventory key for this player's main inventory.
    /// This can be used with InventoryManager to view or modify the
    /// player's inventory
    pub fn main_inventory(&self) -> InventoryKey {
        self.main_inventory_key
    }
    pub fn last_position(&self) -> PlayerPositionUpdate {
        self.state.lock().last_position
    }
    pub fn selected_hotbar_slot(&self) -> Option<u32> {
        self.state.lock().hotbar_slot
    }
    fn to_server_proto(&self) -> StoredPlayer {
        let lock = self.state.lock();
        StoredPlayer {
            name: self.name.clone(),
            last_position: Some(lock.last_position.position.try_into().unwrap()),
            main_inventory: self.main_inventory_key.as_bytes().to_vec(),
            permission: lock.granted_permissions.iter().cloned().collect(),
        }
    }
    fn from_server_proto(
        game_state: Arc<GameState>,
        proto: &StoredPlayer,
        sender: PlayerEventSender,
    ) -> Result<Player> {
        let main_inventory_key = InventoryKey::parse_bytes(&proto.main_inventory)?;
        let mut effective_permissions = HashSet::new();
        for permission in &proto.permission {
            if game_state.game_behaviors.has_defined_permission(permission) {
                effective_permissions.insert(permission.clone());
            } else {
                tracing::warn!(
                    "Player {} has unknown permission: {}",
                    proto.name,
                    permission
                );
            }
        }

        effective_permissions.extend(
            game_state
                .game_behaviors
                .ambient_permissions
                .iter()
                .cloned(),
        );

        let position = proto
            .last_position
            .as_ref()
            .with_context(|| "Missing last_position in StoredPlayer")?
            .try_into()?;

        let entity_id = game_state
            .entities
            .insert_entity(position, Vector3::zero())?;

        Ok(Player {
            name: proto.name.clone(),
            main_inventory_key,
            state: Mutex::new(PlayerState {
                last_position: PlayerPositionUpdate {
                    position,
                    velocity: vec3(0., 0., 0.),
                    face_direction: (0., 0.),
                },
                active_popups: vec![],
                inventory_popup: game_state
                    .game_behaviors()
                    .make_inventory_popup
                    .make_inventory_popup(
                        game_state.clone(),
                        proto.name.clone(),
                        effective_permissions.into_iter().collect(),
                        main_inventory_key,
                    )?,
                hotbar_inventory_view: InventoryView::new_stored(
                    main_inventory_key,
                    game_state.clone(),
                    false,
                    false,
                )?,
                inventory_manipulation_view: InventoryView::new_transient(
                    game_state.clone(),
                    (1, 1),
                    vec![None],
                    true,
                    true,
                    false,
                )?,
                hotbar_slot: None,
                granted_permissions: proto.permission.iter().cloned().collect(),
                temporary_permissions: HashSet::new(),
            }),
            sender,
            game_state,
            entity_id,
        })
    }

    fn new_player(
        name: &str,
        game_state: Arc<GameState>,
        sender: PlayerEventSender,
    ) -> Result<Player> {
        let main_inventory_key = game_state.inventory_manager().make_inventory(4, 8)?;
        // TODO provide hooks here
        // TODO custom spawn location

        let effective_permissions = game_state
            .game_behaviors()
            .default_permissions
            .clone()
            .union(&game_state.game_behaviors().ambient_permissions(name))
            .cloned()
            .collect();

        let position = (game_state.game_behaviors().spawn_location)(name);
        let entity_id = game_state
            .entities
            .insert_entity(position, Vector3::zero())?;

        let player = Player {
            name: name.to_string(),
            main_inventory_key,
            state: PlayerState {
                last_position: PlayerPositionUpdate {
                    position,
                    velocity: Vector3::zero(),
                    face_direction: (0., 0.),
                },
                active_popups: vec![],
                inventory_popup: game_state
                    .game_behaviors()
                    .make_inventory_popup
                    .make_inventory_popup(
                        game_state.clone(),
                        name.to_string(),
                        effective_permissions,
                        main_inventory_key,
                    )?,
                hotbar_inventory_view: InventoryView::new_stored(
                    main_inventory_key,
                    game_state.clone(),
                    false,
                    false,
                )?,
                inventory_manipulation_view: InventoryView::new_transient(
                    game_state.clone(),
                    (1, 1),
                    vec![None],
                    true,
                    true,
                    false,
                )?,
                hotbar_slot: None,
                granted_permissions: game_state.game_behaviors().default_permissions.clone(),
                temporary_permissions: HashSet::new(),
            }
            .into(),
            sender,
            game_state,
            entity_id,
        };

        Ok(player)
    }

    pub async fn send_chat_message(&self, message: ChatMessage) -> Result<()> {
        self.sender.chat_messages.send(message).await?;
        Ok(())
    }

    pub async fn kick_player(&self, reason: &str) -> Result<()> {
        self.sender
            .disconnection_message
            .send(format!("Kicked: {reason}"))
            .await?;
        Ok(())
    }

    pub fn grant_permission(&self, permission: &str) -> Result<()> {
        if !self
            .game_state
            .game_behaviors
            .has_defined_permission(permission)
        {
            bail!("Permission {} not defined", permission);
        }

        let mut lock = self.state.lock();
        lock.granted_permissions.insert(permission.to_string());
        self.post_permission_update(lock)
    }

    fn post_permission_update(&self, mut lock: MutexGuard<'_, PlayerState>) -> Result<()> {
        let effective_permissions = lock
            .effective_permissions(&self.game_state, &self.name)
            .into_iter()
            .collect();
        lock.inventory_popup = self
            .game_state
            .game_behaviors()
            .make_inventory_popup
            .make_inventory_popup(
                self.game_state.clone(),
                self.name.clone(),
                effective_permissions,
                self.main_inventory_key,
            )?;
        tokio::task::block_in_place(|| {
            MutexGuard::unlocked(&mut lock, || {
                self.sender.reinit_player_state.blocking_send(false)
            })
        })?;
        Ok(())
    }

    pub fn grant_temporary_permission(&self, permission: &str) -> Result<()> {
        if !self
            .game_state
            .game_behaviors
            .has_defined_permission(permission)
        {
            bail!("Permission {} not defined", permission);
        }
        let mut lock = self.state.lock();
        lock.temporary_permissions.insert(permission.to_string());
        self.post_permission_update(lock)
    }

    pub fn revoke_permission(&self, permission: &str) -> Result<bool> {
        let mut lock = self.state.lock();
        let removed = lock.granted_permissions.remove(permission);
        self.post_permission_update(lock)?;
        Ok(removed)
    }

    pub fn revoke_temporary_permission(&self, permission: &str) -> Result<bool> {
        let mut lock = self.state.lock();
        let removed = lock.temporary_permissions.remove(permission);
        self.post_permission_update(lock)?;
        Ok(removed)
    }

    pub fn clear_temporary_permissions(&self) -> Result<()> {
        let mut lock = self.state.lock();
        lock.temporary_permissions.clear();
        self.post_permission_update(lock)
    }

    pub fn has_permission(&self, permission: &str) -> bool {
        let lock = self.state.lock();
        lock.granted_permissions.contains(permission)
            || lock.temporary_permissions.contains(permission)
            || self
                .game_state
                .game_behaviors()
                .ambient_permissions(&self.name)
                .contains(permission)
    }

    /// Lists the player's effective permissions (ambient, granted, and temporary)
    pub fn effective_permissions(&self) -> HashSet<String> {
        self.state
            .lock()
            .effective_permissions(&self.game_state, &self.name)
    }
    /// Lists the player's granted permissions (i.e. those stored in the database)
    pub fn granted_permissions(&self) -> HashSet<String> {
        self.state.lock().granted_permissions.clone()
    }
    /// Lists the player's temporary permissions (i.e. obtained from /elevate or similar)
    pub fn temporary_permissions(&self) -> HashSet<String> {
        self.state.lock().temporary_permissions.clone()
    }

    pub async fn set_position(&self, position: Vector3<f64>) -> Result<()> {
        self.state.lock().last_position.position = position;
        self.sender.reinit_player_state.send(true).await?;
        Ok(())
    }

    pub fn set_position_blocking(&self, position: Vector3<f64>) -> Result<()> {
        tokio::runtime::Handle::current().block_on(self.set_position(position))
    }
}

pub(crate) struct PlayerState {
    /// The player's last client-reported position
    pub(crate) last_position: PlayerPositionUpdate,
    /// The inventory popup is always loaded, even when it's not shown
    pub(crate) inventory_popup: Popup,
    /// Other active popups for the player. These get deleted when closed.
    pub(crate) active_popups: Vec<Popup>,
    /// The player's main inventory, which is shown in the user hotbar
    pub(crate) hotbar_inventory_view: InventoryView<()>,
    /// A 1x1 transient view used to carry items with the mouse
    pub(crate) inventory_manipulation_view: InventoryView<()>,
    /// The player's currently selected hotbar slot, if known.
    pub(crate) hotbar_slot: Option<u32>,
    /// The player's persistently granted permissions
    /// (i.e. ones stored in the database, not including any that are ambient)
    pub(crate) granted_permissions: HashSet<String>,
    /// Permissions that this player has obtained using /elevate or similar
    /// They are sent to the client, but they are not stored in the database.
    pub(crate) temporary_permissions: HashSet<String>,
}
impl PlayerState {
    pub(crate) fn handle_inventory_action(&mut self, action: &InventoryAction) -> Result<()> {
        let source_view = self.find_inv_view(InventoryViewId(action.source_view))?;
        let destination_view = self.find_inv_view(InventoryViewId(action.destination_view))?;

        if source_view.can_take() && destination_view.can_place() {
            if action.swap {
                let taken_stack = source_view.take(action.source_slot as usize, None)?;
                let other_taken_stack =
                    destination_view.take(action.destination_slot as usize, None)?;

                if let Some(taken_stack) = taken_stack {
                    // verify no leftover
                    ensure!(destination_view
                        .put(action.destination_slot as usize, taken_stack)?
                        .is_none());
                }

                if let Some(other_taken_stack) = other_taken_stack {
                    // verify no leftover
                    ensure!(source_view
                        .put(action.source_slot as usize, other_taken_stack)?
                        .is_none());
                }
            } else {
                let taken_stack =
                    source_view.take(action.source_slot as usize, Some(action.count))?;
                if let Some(taken_stack) = taken_stack {
                    let leftover =
                        destination_view.put(action.destination_slot as usize, taken_stack)?;
                    if let Some(leftover) = leftover {
                        let still_leftover =
                            source_view.put(action.source_slot as usize, leftover)?;
                        if let Some(still_leftover) = still_leftover {
                            log::warn!("Still-leftover items were destroyed: {:?}", still_leftover);
                        }
                    }
                }
            }
        } else {
            warn!("Inventory view(s) not found for {action:?}");
        }
        for popup in self
            .active_popups
            .iter()
            .chain(std::iter::once(&self.inventory_popup))
        {
            if popup
                .inventory_views()
                .values()
                .any(|x| x.id.0 == action.source_view || x.id.0 == action.destination_view)
            {
                popup.invoke_inventory_action_callback();
            }
        }
        Ok(())
    }

    pub(crate) fn find_inv_view<'a>(
        &'a self,
        id: InventoryViewId,
    ) -> Result<Box<dyn TypeErasedInventoryView + 'a>> {
        if self.inventory_manipulation_view.id == id {
            return Ok(Box::new(&self.inventory_manipulation_view));
        }
        if self.hotbar_inventory_view.id == id {
            return Ok(Box::new(&self.hotbar_inventory_view));
        }

        for popup in self
            .active_popups
            .iter()
            .chain(std::iter::once(&self.inventory_popup))
        {
            if let Some(result) = popup.inventory_views().values().find(|x| x.id == id) {
                return Ok(Box::new(InventoryViewWithContext {
                    view: result,
                    context: popup,
                }));
            }
        }

        bail!("View not found");
    }

    pub(crate) fn effective_permissions(
        &self,
        game_state: &GameState,
        name: &str,
    ) -> HashSet<String> {
        let mut effective_permissions = HashSet::new();
        for permission in &self.granted_permissions {
            effective_permissions.insert(permission.clone());
        }
        for permission in &self.temporary_permissions {
            effective_permissions.insert(permission.clone());
        }
        for permission in game_state.game_behaviors().ambient_permissions(name).iter() {
            effective_permissions.insert(permission.clone());
        }
        effective_permissions
    }
}

// Struct held by the client contexts for this player. When dropped, the
// PlayerManager is notified of this via the Drop impl
// TODO make this cleaner and more sensible, this is fine for an initial impl
pub(crate) struct PlayerContext {
    pub(crate) player: Arc<Player>,
    manager: Arc<PlayerManager>,
}
impl PlayerContext {
    pub(crate) fn update_position(&self, pos: PlayerPositionUpdate) {
        self.player.state.lock().last_position = pos;
    }
    /// Internal method that will get more parameters as we add more animation/interaction state in player
    /// position update messages
    pub(crate) fn update_client_position_state(&self, pos: PlayerPositionUpdate, hotbar_slot: u32) {
        let mut lock = self.player.state.lock();
        lock.last_position = pos;
        lock.hotbar_slot = Some(hotbar_slot);
    }
    pub(crate) fn last_position(&self) -> PlayerPositionUpdate {
        self.player.state.lock().last_position
    }
    pub fn name(&self) -> &str {
        &self.player.name
    }
    pub fn make_initiator(&self) -> EventInitiator<'_> {
        EventInitiator::Player(PlayerInitiator {
            player: &self.player,
            weak: Arc::downgrade(&self.player),
            position: self.last_position(),
        })
    }
}
impl Deref for PlayerContext {
    type Target = Player;

    fn deref(&self) -> &Self::Target {
        &self.player
    }
}
impl Drop for PlayerContext {
    fn drop(&mut self) {
        self.manager.drop_disconnect(&self.player.name)
    }
}

struct PlayerEventSender {
    chat_messages: tokio::sync::mpsc::Sender<ChatMessage>,
    disconnection_message: tokio::sync::mpsc::Sender<String>,
    reinit_player_state: tokio::sync::mpsc::Sender<bool>,
}
pub(crate) struct PlayerEventReceiver {
    pub(crate) chat_messages: tokio::sync::mpsc::Receiver<ChatMessage>,
    pub(crate) disconnection_message: tokio::sync::mpsc::Receiver<String>,
    pub(crate) reinit_player_state: tokio::sync::mpsc::Receiver<bool>,
}

fn make_event_channels() -> (PlayerEventSender, PlayerEventReceiver) {
    const CHAT_BUFFER_SIZE: usize = 128;
    let (chat_sender, chat_receiver) = tokio::sync::mpsc::channel(CHAT_BUFFER_SIZE);
    let (disconnection_sender, disconnection_receiver) = tokio::sync::mpsc::channel(2);
    let (reinit_sender, reinit_receiver) = tokio::sync::mpsc::channel(4);
    (
        PlayerEventSender {
            chat_messages: chat_sender,
            disconnection_message: disconnection_sender,
            reinit_player_state: reinit_sender,
        },
        PlayerEventReceiver {
            chat_messages: chat_receiver,
            disconnection_message: disconnection_receiver,
            reinit_player_state: reinit_receiver,
        },
    )
}

/// Manages players and provides access to them and their data.
pub struct PlayerManager {
    db: Arc<dyn GameDatabase>,
    game_state: Weak<GameState>,
    // To avoid leaks, we should avoid situations where arbitrary code can
    // clone Player instances. To do this, we keep Arc<Player> in private fields
    // and only provide &Player borrows
    //
    // This should probably be cleaned up somehow in the future; current approach seems ugly
    active_players: RwLock<HashMap<String, Arc<Player>>>,

    shutdown: CancellationToken,
    writeback: Mutex<Option<JoinHandle<Result<()>>>>,
    writeback_now_sender: tokio::sync::mpsc::Sender<String>,
}
impl PlayerManager {
    fn db_key(player: &str) -> Vec<u8> {
        KeySpace::Player.make_key(player.as_bytes())
    }
    fn game_state(&self) -> Arc<GameState> {
        self.game_state.upgrade().unwrap()
    }
    pub(crate) fn connect(
        self: Arc<Self>,
        name: &str,
    ) -> Result<(PlayerContext, PlayerEventReceiver)> {
        let mut lock = self.active_players.write();
        if lock.contains_key(name) {
            bail!("Player {name} already connected");
        }

        let (sender, receiver) = make_event_channels();
        // TODO - optimization: This is IO done under the global player lock
        // Maybe something like the game map double locks?
        let player = match self.db.get(&Self::db_key(name))? {
            Some(player_proto) => Player::from_server_proto(
                self.game_state(),
                &StoredPlayer::decode(player_proto.as_slice())?,
                sender,
            )?,
            None => {
                log::info!("New player {name} joining");
                let player = Player::new_player(name, self.game_state(), sender)?;
                self.write_back(&player)?;
                player
            }
        };
        let player = Arc::new(player);
        lock.insert(name.to_string(), player.clone());

        Ok((
            PlayerContext {
                player,
                manager: self.clone(),
            },
            receiver,
        ))
    }

    /// Runs the given closure on the provided player.
    pub fn with_connected_player<F, T>(&self, name: &str, closure: F) -> Result<T>
    where
        F: FnOnce(&Player) -> Result<T>,
    {
        tokio::task::block_in_place(|| {
            let lock = self.active_players.read();
            if !lock.contains_key(name) {
                bail!("Player {name} not connected");
            }
            let player = lock.get(name).unwrap();
            closure(&player)
        })
    }

    pub fn for_all_connected_players<F>(&self, mut closure: F) -> Result<()>
    where
        F: FnMut(&Player) -> Result<()>,
    {
        tokio::task::block_in_place(|| {
            let lock = self.active_players.read();
            for (_, player) in lock.iter() {
                closure(&player)?;
            }
            Ok(())
        })
    }

    pub async fn for_all_connected_players_async<F, G>(&self, mut closure: F) -> Result<()>
    where
        F: FnMut(&Player) -> G,
        G: Future<Output = anyhow::Result<()>>,
    {
        let lock = tokio::task::block_in_place(|| self.active_players.read());
        for (_, player) in lock.iter() {
            closure(&player).await?;
        }
        Ok(())
    }

    fn drop_disconnect(&self, name: &str) {
        match self.active_players.write().entry(name.to_string()) {
            Entry::Occupied(entry) => {
                let count = Arc::strong_count(entry.get());
                // We expect 2 - one in the map, and one that we're holding while calling drop_disconnect
                if count != 2 {
                    log::error!("Arc<Player> seems to be leaking; {count} remaining references");
                }
                match self.write_back(entry.get()) {
                    Ok(_) => {
                        log::info!("{name} disconnected; data written back")
                    }
                    Err(e) => {
                        log::error!("Writeback for {name} failed: {:?}", e);
                    }
                }
                entry.remove();
            }
            Entry::Vacant(_) => {
                log::error!("Trying to disconnect player {name} but they're not present. This is a sign of a serious inconsistency.");
                if cfg!(debug_assertions) {
                    panic!("Trying to disconnect player {name} but they're not present. This is a sign of a serious inconsistency.")
                }
            }
        }
    }
    fn write_back(&self, player: &Player) -> Result<()> {
        self.db.put(
            &Self::db_key(&player.name),
            &player.to_server_proto().encode_to_vec(),
        )
    }

    pub(crate) fn new(
        game_state: Weak<GameState>,
        db: Arc<dyn GameDatabase>,
    ) -> Result<Arc<PlayerManager>> {
        let (sender, receiver) = tokio::sync::mpsc::channel(32);
        let result = Arc::new(PlayerManager {
            game_state,
            db,
            active_players: RwLock::new(HashMap::new()),
            shutdown: CancellationToken::new(),
            writeback: Mutex::new(None),
            writeback_now_sender: sender,
        });
        result.start_writeback(receiver)?;
        Ok(result)
    }

    pub(crate) async fn writeback_loop(
        self: Arc<Self>,
        mut receiver: tokio::sync::mpsc::Receiver<String>,
    ) -> Result<()> {
        while !self.shutdown.is_cancelled() {
            let deadline = Instant::now() + PLAYER_WRITEBACK_INTERVAL;
            select! {
                _ = tokio::time::sleep_until(deadline.into()) => {},
                name = receiver.recv() => {
                    if let Some(name) = name {
                        let lock = self.active_players.read();
                        let player = lock.get(&name);
                        if let Some(player) = player {
                            self.write_back(player)?;
                        }
                    }
                }
                _ = self.shutdown.cancelled() => {
                    log::info!("Player writeback detected cancellation");
                }
            }
            tokio::task::block_in_place(|| self.flush())?;
        }
        log::info!("Player writeback exiting");
        Ok(())
    }

    fn start_writeback(
        self: &Arc<Self>,
        receiver: tokio::sync::mpsc::Receiver<String>,
    ) -> Result<()> {
        let clone = self.clone();
        *(self.writeback.lock()) = Some(crate::spawn_async("player_writeback", async {
            clone.writeback_loop(receiver).await
        })?);
        Ok(())
    }

    pub(crate) fn request_shutdown(&self) {
        self.shutdown.cancel();
    }

    pub(crate) async fn await_shutdown(&self) -> Result<()> {
        let writeback_handle = self.writeback.lock().take();
        writeback_handle.unwrap().await??;
        tokio::task::block_in_place(|| self.flush())?;
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        for player in self.active_players.read().values() {
            self.write_back(player)?;
        }
        Ok(())
    }
}

const PLAYER_WRITEBACK_INTERVAL: Duration = Duration::from_secs(10);
