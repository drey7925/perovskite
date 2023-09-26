use std::sync::Arc;

use perovskite_core::chat::ChatMessage;
use tokio::{sync::broadcast, task::block_in_place};

use super::{event::EventInitiator, GameState};
use anyhow::Result;
pub struct ChatState {
    pub(crate) broadcast_messages: broadcast::Sender<ChatMessage>,
}
const BROADCAST_CAPACITY: usize = 128;
impl ChatState {
    pub(crate) fn new() -> ChatState {
        ChatState {
            broadcast_messages: broadcast::channel(BROADCAST_CAPACITY).0,
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
            let origin = match initiator {
                EventInitiator::Player(p) => format!("<{}>", p.player.name()),
                EventInitiator::Engine => "[server]".to_string(),
                EventInitiator::Plugin(p) => format!("[{}]", p),
            };
            let text = message.to_string();

            self.broadcast_messages
                .send(ChatMessage::new(origin, text))?;
            Ok(())
        }
    }

    async fn handle_slash_command(
        &self,
        initiator: EventInitiator<'_>,
        game_state: Arc<GameState>,
        message: &str,
    ) -> Result<()> {
        match initiator {
            EventInitiator::Player(p) => {
                if message.starts_with("/mapdebug") {
                    p.player
                        .send_chat_message(ChatMessage::new("[server]", "Map debug: "))
                        .await?;
                    let shard_sizes = block_in_place(|| game_state.map().debug_shard_sizes());
                    for (shard, size) in shard_sizes.iter().enumerate() {
                        p.player
                            .send_chat_message(ChatMessage::new(
                                "[server]",
                                format!("shard {} has {} chunks", shard, size),
                            ))
                            .await?;
                    }
                    Ok(())
                } else if message.starts_with("/kickme") {
                    p.player.kick_player("Test kick reason").await?;
                    Ok(())
                } else {
                    p.player
                        .send_chat_message(
                            ChatMessage::new(
                                "[server]",
                                "This is where we'd handle a slash command in the future.",
                            )
                            .with_color((0, 255, 255)),
                        )
                        .await?;
                    Ok(())
                }
            }
            initiator => {
                tracing::warn!("Unhandled slash command from {:?}: {}", initiator, message);
                Ok(())
            }
        }
    }

    /// Returns a broadcast receiver for chat messages
    pub fn subscribe(&self) -> broadcast::Receiver<ChatMessage> {
        self.broadcast_messages.subscribe()
    }
}
