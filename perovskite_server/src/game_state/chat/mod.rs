use std::sync::Arc;

use perovskite_core::{chat::ChatMessage, constants::permissions};
use tokio::{sync::broadcast, task::block_in_place};

use self::commands::CommandManager;

use super::{
    event::{EventInitiator, HandlerContext},
    GameState,
};
use anyhow::{Result, bail};
pub struct ChatState {
    pub(crate) broadcast_messages: broadcast::Sender<ChatMessage>,
    pub(crate) command_manager: CommandManager,
}
pub mod commands;

const BROADCAST_CAPACITY: usize = 128;
impl ChatState {
    pub(crate) fn new(commands: CommandManager) -> ChatState {
        ChatState {
            broadcast_messages: broadcast::channel(BROADCAST_CAPACITY).0,
            command_manager: commands,
        }
    }
    pub(crate) async fn handle_inbound_chat_message(
        &self,
        initiator: EventInitiator<'_>,
        game_state: Arc<GameState>,
        message: &str,
    ) -> Result<()> {
        if message.starts_with('/') {
            self.handle_slash_command(initiator, game_state, message)
                .await
        } else {
            if !initiator.check_permission_if_player(permissions::CHAT) {
                bail!("You do not have permission to chat");
            }
            let message = match initiator {
                EventInitiator::Player(p) => {
                    ChatMessage::new(format!("<{}>", p.player.name()), message)
                }
                EventInitiator::Engine => ChatMessage::new_server_message(message),
                EventInitiator::Plugin(p) => {
                    ChatMessage::new_server_message(message).with_origin(format!("[{}]", p))
                }
            };
            self.broadcast_messages.send(message)?;
            Ok(())
        }
    }
    /// Sends a chat message to all connected players
    pub fn broadcast_chat_message(&self, message: ChatMessage) -> Result<()> {
        self.broadcast_messages.send(message)?;
        Ok(())
    }

    async fn handle_slash_command(
        &self,
        initiator: EventInitiator<'_>,
        game_state: Arc<GameState>,
        message: &str,
    ) -> Result<()> {
        self.command_manager.handle_command(
            message,
            HandlerContext {
                tick: game_state.tick(),
                initiator: initiator,
                game_state,
            },
        ).await
    }

    /// Returns a broadcast receiver for chat messages
    pub fn subscribe(&self) -> broadcast::Receiver<ChatMessage> {
        self.broadcast_messages.subscribe()
    }
}
