use cuberef_core::chat::ChatMessage;
use tokio::sync::broadcast;

use super::event::EventInitiator;
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
        message: &str,
    ) -> Result<()> {
        if message.starts_with('/') {
            self.handle_slash_command(initiator, message).await
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
        message: &str,
    ) -> Result<()> {
        match initiator {
            EventInitiator::Player(p) => {
                p.player
                    .send_chat_message(
                        ChatMessage::new(
                            "[server]",
                            "This is where we'd handle a slash command in the future.",
                        )
                        .with_color((0, 255, 255)),
                    )
                    .await
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
