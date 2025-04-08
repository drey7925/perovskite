use perovskite_core::chat::{ChatMessage, CLIENT_INTERNAL_MESSAGE_COLOR};

pub(crate) struct ChatState {
    pub(crate) message_history: Vec<ChatMessage>,
}
impl ChatState {
    pub(crate) fn new() -> ChatState {
        ChatState {
            message_history: Vec::new(),
        }
    }
    pub(crate) fn show_client_message(&mut self, message: String) {
        self.message_history
            .push(ChatMessage::new("[client]", message).with_color(CLIENT_INTERNAL_MESSAGE_COLOR))
    }
}
