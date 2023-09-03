use std::collections::VecDeque;

use perovskite_core::chat::ChatMessage;

pub(crate) struct ChatState {
    pub(crate) message_history: Vec<ChatMessage>,
}
impl ChatState {
    pub(crate) fn new() -> ChatState {
        ChatState {
            message_history: Vec::new(),
        }
    }
}