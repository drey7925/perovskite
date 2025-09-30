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
use std::ops::Deref;
use std::sync::Weak;
use std::{num::NonZeroU64, sync::Arc};

use anyhow::Result;

use futures::Future;
use perovskite_core::chat::ChatMessage;
use perovskite_core::coordinates::PlayerPositionUpdate;
use perovskite_core::util::TraceBuffer;
use tokio::task::futures::TaskLocalFuture;
use tracing::warn;

use super::{client_ui::Popup, player::Player, GameState};

// Private, lightweight representation of who initiated an event.
// This is used to reconcile responses in the game stream to the requests
// that were sent by the client.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub(crate) struct ClientEventContext {
    client_id: NonZeroU64,
    seq: u64,
}

#[derive(Clone)]
pub struct WeakPlayerRef {
    pub(crate) player: Weak<Player>,
    pub(crate) name: String,
    pub position: PlayerPositionUpdate,
}
impl WeakPlayerRef {
    pub fn try_to_run<T>(&self, f: impl FnOnce(&Player) -> T) -> Option<T> {
        self.player.upgrade().map(|player| f(&player))
    }

    pub fn name(&self) -> &str {
        self.name.as_ref()
    }
}
impl Debug for WeakPlayerRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeakPlayerRef")
            .field("player", &self.name)
            .field("position", &self.position)
            .field("strong_count", &Weak::strong_count(&self.player))
            .finish()
    }
}

/// Who initiated an event
#[derive(Clone, Debug)]
pub enum EventInitiator<'a> {
    /// Event was not initiated by a player
    Engine,
    /// Event was initiated by a player, and a reference to that player is provided
    Player(PlayerInitiator<'a>),
    /// Event was initiated by a player, but we are running in a deferred context.
    /// To avoid deadlock/shutdown issues, the player object may not be available.
    WeakPlayerRef(WeakPlayerRef),
    /// Event was initiated by a plugin, and the plugin wants to indicate that it
    /// was the originator. The exact semantics of this variant are still TBD, and
    /// a plugin can still set Engine without being incorrect
    ///
    /// So far, this is mostly used for error messages and indicating what plugin
    /// sent a chat message to a user.
    Plugin(String),
}
impl EventInitiator<'_> {
    pub fn position(&self) -> Option<PlayerPositionUpdate> {
        match self {
            EventInitiator::Engine => None,
            EventInitiator::Player(p) => Some(p.position),
            EventInitiator::WeakPlayerRef(p) => Some(p.position),
            EventInitiator::Plugin(_) => None,
        }
    }
    pub async fn send_chat_message_async(&self, message: ChatMessage) -> Result<()> {
        match self {
            EventInitiator::Engine => warn!("Attempted to send chat message to engine"),
            EventInitiator::Player(p) => p.player.send_chat_message_async(message).await?,
            EventInitiator::WeakPlayerRef(p) => {
                if let Some(player) = p.player.upgrade() {
                    player.send_chat_message_async(message).await?
                }
            }
            EventInitiator::Plugin(_) => warn!("Attempted to send chat message to plugin"),
        }
        Ok(())
    }
    pub fn send_chat_message(&self, message: ChatMessage) -> Result<()> {
        match self {
            EventInitiator::Engine => warn!("Attempted to send chat message to engine"),
            EventInitiator::Player(p) => p.player.send_chat_message(message)?,
            EventInitiator::WeakPlayerRef(p) => {
                if let Some(player) = p.player.upgrade() {
                    player.send_chat_message(message)?;
                }
            }
            EventInitiator::Plugin(_) => warn!("Attempted to send chat message to plugin"),
        }
        Ok(())
    }
    pub(crate) fn as_debug_str(&self) -> &str {
        match self {
            EventInitiator::Engine => "engine",
            EventInitiator::Player(p) => p.player.name(),
            EventInitiator::WeakPlayerRef(p) => &p.name,
            EventInitiator::Plugin(_) => "plugin",
        }
    }
    /// Checks if the player has the given permission. If the initiator is not a player, then
    /// this always returns true (plugins and engine are assumed to always have permission when acting as themselves)
    pub fn check_permission_if_player(&self, permission: &str) -> bool {
        match self {
            EventInitiator::Engine => true,
            EventInitiator::Player(p) => p.player.has_permission(permission),
            EventInitiator::WeakPlayerRef(p) => p
                .try_to_run(|p| p.has_permission(permission))
                .unwrap_or(false),
            EventInitiator::Plugin(_) => true,
        }
    }

    fn clone_to_static(&self) -> EventInitiator<'static> {
        match self {
            EventInitiator::Engine => EventInitiator::Engine,
            EventInitiator::Player(p) => EventInitiator::WeakPlayerRef(WeakPlayerRef {
                player: p.weak.clone(),
                name: p.player.name.clone(),
                position: p.position,
            }),
            EventInitiator::WeakPlayerRef(p) => EventInitiator::WeakPlayerRef(p.clone()),
            EventInitiator::Plugin(s) => EventInitiator::Plugin(s.clone()),
        }
    }

    pub fn player_name(&self) -> Option<&str> {
        match self {
            EventInitiator::Engine => None,
            EventInitiator::Player(p) => Some(p.player.name()),
            EventInitiator::WeakPlayerRef(p) => Some(&p.name),
            EventInitiator::Plugin(_) => None,
        }
    }
}

/// Details about a player that initiated an event.
#[derive(Clone)]
pub struct PlayerInitiator<'a> {
    /// The player that initiated the event
    pub player: &'a Player,
    pub(crate) weak: Weak<Player>,
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
    pub fn initiator(&self) -> &EventInitiator<'_> {
        &self.initiator
    }
    pub fn tick(&self) -> u64 {
        self.tick
    }
    /// Creates a new popup
    pub fn new_popup(&self) -> Popup {
        Popup::new(self.game_state.clone())
    }
    /// Runs the given function in a new blocking task in the background.
    /// run_deferred returns immediately. This can be used e.g. in a timer while locks are held;
    /// if the deferred task needs the same locks, it'll wait in the background for them to be released.
    ///
    /// Warning: If the function hangs indefinitely, the game cannot exit.
    pub fn run_deferred<F>(&self, f: F)
    where
        F: FnOnce(&HandlerContext) -> Result<()> + 'static + Send,
    {
        let our_clone = HandlerContext {
            tick: self.tick,
            initiator: self.initiator.clone_to_static(),
            game_state: self.game_state.clone(),
        };
        tokio::task::spawn_blocking(move || {
            if let Err(e) = f(&our_clone) {
                tracing::error!("Error in deferred function: {}", e);
            }
        });
    }

    pub fn run_deferred_async<U: Future<Output = Result<()>> + Send + Sync + 'static>(
        &self,
        f: impl FnOnce(HandlerContext<'static>) -> U,
    ) {
        let our_clone = HandlerContext {
            tick: self.tick,
            initiator: self.initiator.clone_to_static(),
            game_state: self.game_state.clone(),
        };
        let fut = f(our_clone);
        tokio::task::spawn(async move {
            if let Err(e) = fut.await {
                tracing::error!("Error in deferred function: {}", e);
            }
        });
    }

    /// Same as run_deferred, but runs the function after the given delay
    ///
    /// Warning: If the function hangs indefinitely, the game cannot exit.
    ///
    /// Consider checking `game_state.is_shutting_down()` in any loops or other unbounded delays,
    /// or consider using a tokio::select to exit early if `game_state.await_shutdown()` returns.
    pub fn run_deferred_delayed(
        &self,
        delay: std::time::Duration,
        f: impl FnOnce(&HandlerContext) -> Result<()> + 'static + Send,
    ) {
        let our_clone = HandlerContext {
            tick: self.tick,
            initiator: self.initiator.clone_to_static(),
            game_state: self.game_state.clone(),
        };
        tokio::task::spawn(async move {
            tokio::time::sleep(delay).await;
            match tokio::task::spawn_blocking(move || f(&our_clone)).await {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => tracing::error!("Error in deferred function: {}", e),
                Err(e) => tracing::error!("Panic in deferred function: {}", e),
            }
        });
    }
}
impl Deref for HandlerContext<'_> {
    type Target = GameState;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.game_state
    }
}

tokio::task_local! {
    static TRACE_BUFFER: TraceBuffer;
}

pub fn log_trace(msg: &'static str) {
    let _ = TRACE_BUFFER.try_with(|t| t.log(msg));
}

pub fn clone_trace_buffer() -> TraceBuffer {
    TRACE_BUFFER.try_with(|t| t.clone()).unwrap_or_else(|_| {
        let buf = TraceBuffer::new(false);
        buf.log("Failed to clone trace buffer, creating a new one");
        buf
    })
}

pub(crate) fn run_traced<F: Future>(buf: TraceBuffer, f: F) -> TaskLocalFuture<TraceBuffer, F> {
    TRACE_BUFFER.scope(buf, f)
}

pub(crate) fn run_traced_sync<T, F: FnOnce() -> T>(buf: TraceBuffer, f: F) -> T {
    TRACE_BUFFER.sync_scope(buf, f)
}
