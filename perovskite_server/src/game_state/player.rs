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

use super::{
    client_ui::Popup,
    entities::EntityTypeId,
    event::{EventInitiator, PlayerInitiator},
    inventory::{InventoryKey, InventoryView, InventoryViewId, TypeErasedInventoryView},
    GameState,
};
use crate::{
    database::{GameDatabase, KeySpace},
    game_state::inventory::InventoryViewWithContext,
};
use log::warn;
use parking_lot::{Mutex, MutexGuard, RwLock};
use perovskite_core::protocol::game_rpc::EntityTarget;
use prost::Message;
use seqlock::SeqLock;
use tokio::{select, task::JoinHandle};
use tokio_util::sync::CancellationToken;

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
    // Derefs to a player and has its own mutating functions that appropriately track whether any updates need
    // to be sent to clients, or only through this struct (which can maintain the necessary invariant)
    pub(crate) state: Mutex<PlayerState>,
    fast_pos: SeqLock<PlayerPositionUpdate>,
    // Sends events for that player; the corresponding receiver will send them over the network to the client.
    sender: PlayerEventSender,
    game_state: Arc<GameState>,
    // TODO - refactor according to the final design of entities
    pub(crate) entity_id: u64,
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
        self.fast_pos.read()
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

        let entity_id = game_state.entities().new_entity_blocking(
            position,
            None,
            EntityTypeId {
                class: (&game_state.game_behaviors.player_entity_class)(&proto.name),
                data: Some(proto.name.as_bytes().into()),
            },
            None,
        );

        let ppu = PlayerPositionUpdate {
            position,
            velocity: vec3(0., 0., 0.),
            face_direction: (0., 0.),
        };

        Ok(Player {
            name: proto.name.clone(),
            main_inventory_key,
            state: Mutex::new(PlayerState {
                last_position: ppu,
                active_popups: vec![],
                inventory_popup: game_state
                    .game_behaviors()
                    .make_inventory_popup
                    .make_inventory_popup(
                        game_state.clone(),
                        proto.name.clone(),
                        effective_permissions.into_iter().collect(),
                        main_inventory_key,
                    )?
                    .into(),
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
                attached_to_entity: None,
            }),
            fast_pos: SeqLock::new(ppu),
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

        let effective_permissions = game_state
            .game_behaviors()
            .default_permissions
            .clone()
            .union(&game_state.game_behaviors().ambient_permissions(name))
            .cloned()
            .collect();

        let position = (game_state.game_behaviors().spawn_location)(name);
        let entity_id = game_state.entities().new_entity_blocking(
            position,
            None,
            EntityTypeId {
                class: (game_state.game_behaviors().player_entity_class)(name),
                data: Some(name.as_bytes().into()),
            },
            None,
        );
        let ppu = PlayerPositionUpdate {
            position,
            velocity: Vector3::zero(),
            face_direction: (0., 0.),
        };

        let player = Player {
            name: name.to_string(),
            main_inventory_key,
            state: PlayerState {
                last_position: ppu,
                active_popups: vec![],
                inventory_popup: game_state
                    .game_behaviors()
                    .make_inventory_popup
                    .make_inventory_popup(
                        game_state.clone(),
                        name.to_string(),
                        effective_permissions,
                        main_inventory_key,
                    )?
                    .into(),
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
                attached_to_entity: None,
            }
            .into(),
            fast_pos: SeqLock::new(ppu),
            sender,
            game_state,
            entity_id,
        };

        Ok(player)
    }

    pub async fn send_chat_message_async(&self, message: ChatMessage) -> Result<()> {
        self.sender
            .tx
            .send(PlayerEvent::ChatMessage(message))
            .await?;
        Ok(())
    }
    pub fn send_chat_message(&self, message: ChatMessage) -> Result<()> {
        tokio::runtime::Handle::current().block_on(self.send_chat_message_async(message))
    }

    pub async fn kick_player(&self, reason: &str) -> Result<()> {
        self.sender
            .tx
            .send(PlayerEvent::DisconnectionMessage(format!(
                "Kicked: {reason}"
            )))
            .await?;
        Ok(())
    }

    pub fn kick_player_blocking(&self, reason: &str) -> Result<()> {
        tokio::runtime::Handle::current().block_on(self.kick_player(reason))
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
            )?
            .into();
        // Avoid a deadlock: the thread that consumes from the channel may need the lock in order
        // to make progress.
        drop(lock);
        tokio::task::block_in_place(|| {
            self.sender
                .tx
                .blocking_send(PlayerEvent::ReinitPlayerState(false))
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
        self.fast_pos.lock_write().position = position;
        self.sender
            .tx
            .send(PlayerEvent::ReinitPlayerState(true))
            .await?;
        Ok(())
    }

    pub fn set_position_blocking(&self, position: Vector3<f64>) -> Result<()> {
        tokio::runtime::Handle::current().block_on(self.set_position(position))
    }

    pub async fn attach_to_entity(&self, entity_target: EntityTarget) -> Result<()> {
        self.state.lock().attached_to_entity = Some(entity_target);
        self.sender
            .tx
            .send(PlayerEvent::ReinitPlayerState(true))
            .await?;
        Ok(())
    }

    pub fn attach_to_entity_blocking(&self, entity_target: EntityTarget) -> Result<()> {
        tokio::runtime::Handle::current().block_on(self.attach_to_entity(entity_target))
    }
    pub async fn detach_from_entity(&self) -> Result<Option<EntityTarget>> {
        let old_attachment = std::mem::replace(&mut self.state.lock().attached_to_entity, None);
        self.sender
            .tx
            .send(PlayerEvent::ReinitPlayerState(true))
            .await?;
        Ok(old_attachment)
    }
    pub fn detach_from_entity_blocking(&self) -> Result<Option<EntityTarget>> {
        tokio::runtime::Handle::current().block_on(self.detach_from_entity())
    }

    pub async fn show_popup(&self, popup: Popup) -> Result<()> {
        let mut lock = self.state.lock();
        self.sender
            .tx
            .send(PlayerEvent::UpdatedPopup(popup.id()))
            .await?;
        lock.active_popups.push(popup);
        Ok(())
    }

    pub fn show_popup_blocking(&self, popup: Popup) -> Result<()> {
        tokio::runtime::Handle::current().block_on(self.show_popup(popup))
    }
}

pub(crate) struct PlayerState {
    /// The player's last client-reported position
    pub(crate) last_position: PlayerPositionUpdate,
    /// The inventory popup is always loaded, even when it's not shown
    pub(crate) inventory_popup: Option<Popup>,
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
    /// The player's attached entity, if any
    pub(crate) attached_to_entity: Option<EntityTarget>,
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
                            warn!("Still-leftover items were destroyed: {:?}", still_leftover);
                        }
                    }
                }
            }
        } else {
            warn!("Inventory view(s) not found for {action:?}");
        }
        for popup in self.active_popups.iter().chain(self.inventory_popup.iter()) {
            if popup
                .inventory_views()
                .values()
                .any(|x| x.id.0 == action.source_view || x.id.0 == action.destination_view)
            {
                popup.invoke_inventory_action_callback()?
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

        for popup in self.active_popups.iter().chain(self.inventory_popup.iter()) {
            if let Some(result) = popup.inventory_views().values().find(|x| x.id == id) {
                return Ok(Box::new(InventoryViewWithContext {
                    view: result,
                    context: popup,
                }));
            }
        }

        bail!("View not found");
    }

    pub(crate) fn repair_inventory_popup(
        &mut self,
        name: &str,
        main_inventory_key: InventoryKey,
        game_state: &Arc<GameState>,
    ) -> Result<()> {
        #[cold]
        fn actually_repair(
            state: &mut PlayerState,
            name: &str,
            main_inventory_key: InventoryKey,
            game_state: &Arc<GameState>,
        ) -> Result<()> {
            tracing::error!(
                "Inventory popup wasn't returned for player {}; repairing",
                name
            );
            state.inventory_popup = game_state
                .game_behaviors()
                .make_inventory_popup
                .make_inventory_popup(
                    game_state.clone(),
                    name.to_string(),
                    state
                        .effective_permissions(game_state, name)
                        .iter()
                        .cloned()
                        .collect(),
                    main_inventory_key,
                )?
                .into();
            Ok(())
        }

        if self.inventory_popup.is_none() {
            actually_repair(self, name, main_inventory_key, game_state)?;
        }
        Ok(())
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
    /// Updates the player's position, does not force a resync.
    /// This is used to handle the inbound position updates set by the
    /// player's interactive movement
    pub(crate) fn update_position(&self, pos: PlayerPositionUpdate) {
        self.player.state.lock().last_position = pos;
        *self.fast_pos.lock_write() = pos;
    }
    /// Internal method that will get more parameters as we add more animation/interaction state in player
    /// position update messages
    pub(crate) fn update_client_position_state(&self, pos: PlayerPositionUpdate, hotbar_slot: u32) {
        let mut lock = self.player.state.lock();
        lock.last_position = pos;
        *self.fast_pos.lock_write() = pos;
        lock.hotbar_slot = Some(hotbar_slot);
    }
    pub(crate) fn last_position(&self) -> PlayerPositionUpdate {
        self.player.fast_pos.read()
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
    tx: tokio::sync::mpsc::Sender<PlayerEvent>,
}
pub(crate) struct PlayerEventReceiver {
    pub(crate) rx: tokio::sync::mpsc::Receiver<PlayerEvent>,
}

#[derive(Debug, Clone)]
pub(crate) enum PlayerEvent {
    ChatMessage(ChatMessage),
    DisconnectionMessage(String),
    /// Updates the player's general state of the world (i.e. [perovskite_core::protocol::game_rpc::SetClientState])
    /// bool indicates whether the position should be updated (true if we're teleporting the player, false otherwise)
    ReinitPlayerState(bool),
    UpdatedPopup(u64),
}

fn make_event_channels() -> (PlayerEventSender, PlayerEventReceiver) {
    const CHAT_BUFFER_SIZE: usize = 128;
    let (tx, rx) = tokio::sync::mpsc::channel(CHAT_BUFFER_SIZE);
    (PlayerEventSender { tx }, PlayerEventReceiver { rx })
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
            Some(player_proto) => tokio::task::block_in_place(|| {
                Player::from_server_proto(
                    self.game_state(),
                    &StoredPlayer::decode(player_proto.as_slice())?,
                    sender,
                )
            })?,
            None => {
                log::info!("New player {name} joining");
                let player = tokio::task::block_in_place(|| {
                    Player::new_player(name, self.game_state(), sender)
                })?;
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
            closure(player)
        })
    }

    pub fn for_all_connected_players<F>(&self, mut closure: F) -> Result<()>
    where
        F: FnMut(&Player) -> Result<()>,
    {
        tokio::task::block_in_place(|| {
            let lock = self.active_players.read();
            for (_, player) in lock.iter() {
                closure(player)?;
            }
            Ok(())
        })
    }

    pub async fn for_all_connected_players_async<F, G>(&self, mut closure: F) -> Result<()>
    where
        F: FnMut(&Player) -> G,
        G: Future<Output = Result<()>>,
    {
        let lock = tokio::task::block_in_place(|| self.active_players.read());
        for (_, player) in lock.iter() {
            closure(player).await?;
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
                match self.game_state.upgrade() {
                    None => {
                        log::warn!("Game state gone, not trying to remove player entity");
                    }
                    Some(gs) => {
                        let entity_id = entry.get().entity_id;
                        tokio::task::spawn(async move {
                            gs.entities().remove(entity_id).await;
                            log::info!("Player entity with id {entity_id} removed")
                        });
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

impl Drop for PlayerManager {
    fn drop(&mut self) {
        self.flush().unwrap();
        log::info!("Player manager shutdown complete");
    }
}

const PLAYER_WRITEBACK_INTERVAL: Duration = Duration::from_secs(10);
