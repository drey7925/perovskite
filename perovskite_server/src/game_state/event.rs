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
use std::fmt::Debug;
use std::{num::NonZeroU64, sync::Arc};

use perovskite_core::coordinates::PlayerPositionUpdate;

use super::blocks::BlockTypeManager;
use super::{game_map::ServerGameMap, items::ItemManager, player::Player, GameState, client_ui::Popup};

// Private, lightweight representation of who initiated an event.
// This is used to reconcile responses in the game stream to the requests
// that were sent by the client.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub(crate) struct ClientEventContext {
    client_id: NonZeroU64,
    seq: u64,
}

/// Who initiated an event
#[derive(Clone, Debug)]
pub enum EventInitiator<'a> {
    /// Event was not initiated by a player
    Engine,
    /// Event was initiated by a player, and a reference to that player is provided
    Player(PlayerInitiator<'a>),
    /// Event was initiated by a plugin, and the plugin wants to indicate that it
    /// was the originator. The exact semantics of this variant are still TBD, and
    /// a plugin can still set Engine without being incorrect
    /// 
    /// So far, this is mostly used for error messages and indicating what plugin
    /// sent a chat message to a user.
    Plugin(String)
}
impl EventInitiator<'_> {
    pub fn position(&self) -> Option<PlayerPositionUpdate> {
        match self {
            EventInitiator::Engine => None,
            EventInitiator::Player(p) => Some(p.position),
            EventInitiator::Plugin(_) => None
        }
    }
}

/// Details about a player that initiated an event.
#[derive(Clone)]
pub struct PlayerInitiator<'a> {
    /// The player that initiated the event
    pub player: &'a Player,
    /// The player's reported position and face direction *at the time that they initiated the event*
    /// Note that this is not necessarily the same as the player's current position, or as any position reported
    /// by the player's periodic position updates.
    pub position: PlayerPositionUpdate,
}
impl Debug for PlayerInitiator<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlayerInitiator")
            .field("player", &self.player.name())
            .field("position", &self.position)
            .finish()
    }
}

/// Common details for all events that are passed to event handlers.
#[derive(Clone)]
pub struct HandlerContext<'a> {
    /// sequence number for this event.
    pub(crate) tick: u64,
    /// The character
    pub(crate) initiator: EventInitiator<'a>,
    /// Access to the map
    pub(crate) game_state: Arc<GameState>,
}

impl<'a> HandlerContext<'a> {
    pub fn initiator(&self) -> &EventInitiator {
        &self.initiator
    }
    pub fn tick(&self) -> u64 {
        self.tick
    }
    pub fn game_map(&self) -> &ServerGameMap {
        self.game_state.map()
    }
    pub fn blocks(&self) -> &BlockTypeManager {
        self.game_state.map.block_type_manager()
    }
    pub fn items(&self) -> &ItemManager {
        self.game_state.item_manager()
    }
    pub fn new_popup(&self) -> Popup {
        Popup::new(self.game_state.clone())
    }
}
