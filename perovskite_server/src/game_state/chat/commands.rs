use std::{collections::HashMap, panic::AssertUnwindSafe};

use anyhow::{bail, Result};
use itertools::Itertools;
use perovskite_core::{
    chat::ChatMessage,
    constants::permissions::{self, ELIGIBLE_PREFIX},
};
use tonic::async_trait;

use crate::{
    game_state::event::{EventInitiator, HandlerContext, PlayerInitiator},
    run_async_handler,
};

#[async_trait]
pub trait ChatCommandHandler: Send + Sync {
    async fn handle(&self, message: &str, context: &HandlerContext<'_>) -> Result<()>;

    fn should_show_in_help_menu(&self, _context: &HandlerContext<'_>) -> bool {
        true
    }
}

pub struct ChatCommand {
    action: Box<dyn ChatCommandHandler>,
    /// Help text. E.g. for a command `/give [player] [item] [amount]`, the help text
    /// should be `"<player> <item> [amount] - Gives <item> to <player>"`
    help_text: String,
    // TODO - parameter structure, tab completion?
}
impl ChatCommand {
    pub fn new(action: Box<dyn ChatCommandHandler>, help_text: String) -> Self {
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
        commands.insert(
            "grant".to_string(),
            ChatCommand {
                action: Box::new(GrantCommandImpl),
                help_text: "<player> <permission>: Grants a permission to a player.".to_string(),
            },
        );
        commands.insert(
            "revoke".to_string(),
            ChatCommand {
                action: Box::new(RevokeCommandImpl),
                help_text: "<player> <permission>: Revokes a permission from a player.".to_string(),
            },
        );
        commands.insert(
            "elevate".to_string(),
            ChatCommand {
                action: Box::new(ElevateCommandImpl),
                help_text:
                    "[permission]: Temporarily grant yourself the given permission, if eligible. If no permission specified, list eligible permissions."
                        .to_string(),
            },
        );
        commands.insert(
            "permissions".to_string(),
            ChatCommand {
                action: Box::new(ListPermissionsImpl),
                help_text:
                    "[player]: List player's permissions, or if no player is specified, list all defined permissions."
                        .to_string(),
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
                    )
                    .await?;

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
impl ChatCommandHandler for HelpCommandImpl {
    async fn handle(&self, _message: &str, context: &HandlerContext<'_>) -> Result<()> {
        let content = context
            .game_state
            .chat
            .command_manager
            .commands
            .iter()
            .filter(|(_, cmd)| cmd.action.should_show_in_help_menu(context))
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

struct GrantCommandImpl;
#[async_trait]
impl ChatCommandHandler for GrantCommandImpl {
    async fn handle(&self, message: &str, context: &HandlerContext<'_>) -> Result<()> {
        if !context
            .initiator
            .check_permission_if_player(permissions::GRANT)
        {
            bail!("Insufficient permissions");
        }
        let params = message.split_whitespace().collect::<Vec<_>>();

        if params.len() != 3 {
            bail!("Incorrect usage: should be /grant <user> <permission>");
        }
        let (user, permission) = (params[1], params[2]);
        let permissions = if permission == "*" {
            context
                .game_behaviors()
                .defined_permissions
                .iter()
                .map(|x| x.to_string())
                .collect()
        } else {
            vec![permission.to_string()]
        };
        // todo support offline players
        context.player_manager().with_connected_player(user, |p| {
            for permission in permissions {
                p.grant_permission(&permission)?;
            }
            Ok(())
        })?;
        context
            .initiator
            .send_chat_message(ChatMessage::new_server_message(
                "Permission granted".to_string(),
            ))
            .await?;
        Ok(())
    }

    fn should_show_in_help_menu(&self, context: &HandlerContext<'_>) -> bool {
        context
            .initiator()
            .check_permission_if_player(permissions::GRANT)
    }
}

struct RevokeCommandImpl;
#[async_trait]
impl ChatCommandHandler for RevokeCommandImpl {
    async fn handle(&self, message: &str, context: &HandlerContext<'_>) -> Result<()> {
        if !context
            .initiator
            .check_permission_if_player(permissions::GRANT)
        {
            bail!("Insufficient permissions");
        }
        let params = message.split_whitespace().collect::<Vec<_>>();
        if params.len() != 3 {
            bail!("Incorrect usage: should be /grant <user> <permission>");
        }
        let (user, permission) = (params[1], params[2]);

        if let EventInitiator::Player(p) = context.initiator() {
            if p.player.name == user && permission == permissions::GRANT {
                bail!("Cannot revoke your own grant permission");
            }
        }

        let permissions = if permission == "*" {
            context
                .game_behaviors()
                .defined_permissions
                .iter()
                .map(|x| x.to_string())
                .collect()
        } else {
            vec![permission.to_string()]
        };
        // todo support offline players
        let any_failed = context.player_manager().with_connected_player(user, |p| {
            let mut any_failed = false;
            for permission in permissions {
                if !p.revoke_permission(&permission)? {
                    any_failed = true;
                }
            }
            Ok(any_failed)
        })?;
        // We only check if * wasn't used, since we expect some permissions to fail to be revoked
        // when the wildcard is used.
        if permission != "*" && any_failed {
            bail!("Player didn't have the specified permission");
        }
        context
            .initiator()
            .send_chat_message(ChatMessage::new_server_message(
                "Permission revoked".to_string(),
            ))
            .await?;
        Ok(())
    }

    fn should_show_in_help_menu(&self, context: &HandlerContext<'_>) -> bool {
        context
            .initiator()
            .check_permission_if_player(permissions::GRANT)
    }
}

struct ElevateCommandImpl;
#[async_trait]
impl ChatCommandHandler for ElevateCommandImpl {
    async fn handle(&self, message: &str, context: &HandlerContext<'_>) -> Result<()> {
        let params = message.split_whitespace().collect::<Vec<_>>();

        let player_initiator = match context.initiator() {
            EventInitiator::Player(p) => p,
            _ => bail!("This command can only be used by a player"),
        };

        // todo allow a timeout to be set
        match params.len() {
            1 => {
                let eligible = player_initiator
                    .player
                    .effective_permissions()
                    .iter()
                    .filter_map(|x| x.strip_prefix(ELIGIBLE_PREFIX))
                    .join(", ");
                if eligible.is_empty() {
                    context
                        .initiator()
                        .send_chat_message(ChatMessage::new_server_message(
                            "You are not eligible for any permissions",
                        ))
                        .await?;
                } else {
                    context
                        .initiator()
                        .send_chat_message(ChatMessage::new_server_message(format!(
                            "You are eligible for the following permissions: {}",
                            eligible
                        )))
                        .await?;
                }
            }
            2 => {
                if params[1] == "clear" {
                    player_initiator.player.clear_temporary_permissions()?;
                    context
                        .initiator
                        .send_chat_message(ChatMessage::new_server_message(
                            "Temporary permissions cleared",
                        ))
                        .await?;
                } else {
                    let permission_to_check = ELIGIBLE_PREFIX.to_string() + params[1];
                    if !player_initiator.player.has_permission(&permission_to_check) {
                        bail!("Not eligible to sudo this permission");
                    }
                    player_initiator
                        .player
                        .grant_temporary_permission(&params[1])?;
                    context
                        .initiator
                        .send_chat_message(ChatMessage::new_server_message(
                            "Temporary permission granted",
                        ))
                        .await?;
                }
            }
            _ => {
                bail!(
                    "Incorrect usage: should be /elevate, /elevate clear, or /elevate <permission>"
                );
            }
        }

        Ok(())
    }
    fn should_show_in_help_menu(&self, context: &HandlerContext<'_>) -> bool {
        match context.initiator() {
            EventInitiator::Player(p) => p
                .player
                .effective_permissions()
                .iter()
                .any(|x| x.starts_with(ELIGIBLE_PREFIX)),
            _ => true,
        }
    }
}

struct ListPermissionsImpl;
#[async_trait]
impl ChatCommandHandler for ListPermissionsImpl {
    async fn handle(&self, message: &str, context: &HandlerContext<'_>) -> Result<()> {
        if !context
            .initiator
            .check_permission_if_player(permissions::GRANT)
        {
            bail!("Insufficient permissions");
        }
        let params = message.split_whitespace().collect::<Vec<_>>();
        match params.len() {
            1 => {
                // List all defined permissions
                context
                    .initiator
                    .send_chat_message(ChatMessage::new_server_message(format!(
                        "All defined permissions: {}",
                        context
                            .game_behaviors()
                            .defined_permissions
                            .iter()
                            .sorted()
                            .join(", ")
                    )))
                    .await?;
            }
            2 => {
                let username = params[1];
                let message =
                    context
                        .game_state
                        .player_manager()
                        .with_connected_player(username, |p| {
                            let message = format!(
                                "{}'s permissions:\n  granted: {}\n  temporary: {}\n  ambient: {}",
                                username,
                                p.granted_permissions().iter().sorted().join(", "),
                                p.temporary_permissions().iter().sorted().join(", "),
                                context
                                    .game_state
                                    .game_behaviors()
                                    .ambient_permissions(username)
                                    .iter()
                                    .sorted()
                                    .join(", ")
                            );
                            Ok(ChatMessage::new_server_message(message))
                        })?;
                context.initiator.send_chat_message(message).await?;
            }
            _ => {
                bail!("Incorrect usage: should be /permissions <playername> or /permissions");
            }
        }

        Ok(())
    }
}
