use std::{collections::HashMap, panic::AssertUnwindSafe};

use anyhow::{bail, Result};
use futures::FutureExt;
use itertools::Itertools;
use perovskite_core::chat::ChatMessage;
use tonic::async_trait;

use crate::{game_state::event::HandlerContext, run_async_handler};

#[async_trait]
pub trait CommandImplementation: Send + Sync {
    async fn handle(&self, message: &str, context: &HandlerContext<'_>) -> Result<()>;
}

pub struct ChatCommand {
    action: Box<dyn CommandImplementation>,
    /// Help text. E.g. for a command `/give [player] [item] [amount]`, the help text
    /// should be `"<player> <item> [amount] - Gives <item> to <player>"`
    help_text: String,
    // TODO - parameter structure, tab completion?
}
impl ChatCommand {
    pub fn new(action: Box<dyn CommandImplementation>, help_text: String) -> Self {
        Self { action, help_text }
    }
}

pub struct CommandManager {
    commands: HashMap<String, ChatCommand>,
}
impl CommandManager {
    pub fn new() -> Self {
        let mut commands = HashMap::new();
        commands.insert(
            "help".to_string(),
            ChatCommand {
                action: Box::new(HelpCommandImpl),
                help_text: "Lists all available commands.".to_string(),
            },
        );
        Self { commands }
    }
    pub fn add_command(&mut self, name: String, command: ChatCommand) -> Result<()> {
        if self.commands.contains_key(&name) {
            bail!("Command already exists");
        } else {
            self.commands.insert(name, command);
            Ok(())
        }
    }
    /// Handles a command, sending errors back to the user
    ///
    pub async fn handle_command(&self, message: &str, context: HandlerContext<'_>) -> Result<()> {
        if let Err(e) = self.try_handle_command(message, &context).await {
            context
                .initiator
                .send_chat_message(
                    ChatMessage::new_server_message(format!("Command failed: {:?}", e))
                        .with_color((255, 0, 0)),
                )
                .await?;
        }
        Ok(())
    }
    /// Tries to handle a command, returning an error if it fails
    pub async fn try_handle_command(
        &self,
        message: &str,
        context: &HandlerContext<'_>,
    ) -> Result<()> {
        if !message.starts_with('/') {
            bail!("Not a command - expected a /");
        }
        let command_name = message.trim_start_matches('/').split_whitespace().next();
        match command_name {
            Some(name) => {
                if let Some(command) = self.commands.get(name) {
                    let initiator = context.initiator.clone();
                    run_async_handler!(
                        async { command.action.handle(message, context).await },
                        "command",
                        &initiator,
                        command = command_name,
                        initiator = initiator.as_debug_str(),
                    ).await?;

                    Ok(())
                } else {
                    bail!("Command not found");
                }
            }
            None => {
                bail!("Command name was empty");
            }
        }
    }
}

struct HelpCommandImpl;
#[async_trait]
impl CommandImplementation for HelpCommandImpl {
    async fn handle(&self, _message: &str, context: &HandlerContext<'_>) -> Result<()> {
        let content = context
            .game_state
            .chat
            .command_manager
            .commands
            .iter()
            .sorted_by(|(k, _v), (k2, _v2)| k.cmp(k2))
            .map(|(name, cmd)| format!("/{} {}", name, cmd.help_text))
            .join("\n");
        context
            .initiator
            .send_chat_message(ChatMessage::new_server_message(content))
            .await?;
        Ok(())
    }
}
