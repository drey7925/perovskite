use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use perovskite_core::{
    chat::{ChatMessage, SERVER_WARNING_COLOR},
    constants::permissions,
    protocol::items::item_def::QuantityType,
};
use perovskite_server::game_state::{
    chat::commands::ChatCommandHandler,
    event::{EventInitiator, HandlerContext},
    items::ItemStack,
};

use super::DefaultGameBuilder;

pub(crate) fn register_default_commands(game_builder: &mut DefaultGameBuilder) -> Result<()> {
    game_builder.add_command(
        "whereami",
        Box::new(WhereAmICommand),
        ": Tells you your current coordinates.",
    )?;
    game_builder.add_command(
        "give",
        Box::new(GiveCommand),
        "<recipient> <item> [count]: Gives someone an item.",
    )?;
    game_builder.add_command(
        "giveme",
        Box::new(GiveMeCommand),
        "<item> [count]: Gives you an item.",
    )?;
    game_builder.add_command(
        "settime",
        Box::new(SetTimeCommand),
        "<days>: Sets the game time to the given number of days. 0.0 is midnight, 0.5 is noon, and 1.0 is midnight the next day."
    )?;
    game_builder.add_command(
        "discard",
        Box::new(DiscardCommand),
        ": Discards the currently selected item.",
    )?;
    game_builder.add_command(
        "discardall",
        Box::new(DiscardAllCommand),
        ": Discards ALL items in your inventory.",
    )?;
    Ok(())
}

struct WhereAmICommand;
#[async_trait]
impl ChatCommandHandler for WhereAmICommand {
    async fn handle(&self, _message: &str, context: &HandlerContext<'_>) -> Result<()> {
        if let EventInitiator::Player(p) = context.initiator() {
            p.player
                .send_chat_message(ChatMessage::new_server_message(format!(
                    "You are at {:?}",
                    p.position.position
                )))
                .await?;
        }
        Ok(())
    }
}

struct GiveMeCommand;
#[async_trait]
impl ChatCommandHandler for GiveMeCommand {
    async fn handle(&self, message: &str, context: &HandlerContext<'_>) -> Result<()> {
        if !context
            .initiator()
            .check_permission_if_player(permissions::GIVE)
        {
            bail!("Insufficient permissions");
        }
        // message is either /giveme item or /give item count
        let params = message.split_whitespace().collect::<Vec<_>>();
        let (item, count) = match params.len() {
            2 => (params[1], 1),
            3 => (params[1], params[2].parse().context("Invalid count")?),
            _ => bail!("Incorrect usage: should be /giveme <item> [count]"),
        };

        if let EventInitiator::Player(p) = context.initiator() {
            let player_inventory = p.player.main_inventory();
            let stack = make_stack(item, count, context).await?;

            let leftover = context
                .inventory_manager()
                .mutate_inventory_atomically(&player_inventory, |inv| Ok(inv.try_insert(stack)))?;
            if leftover.is_some() {
                p.player
                    .send_chat_message(
                        ChatMessage::new_server_message("Not enough space in inventory")
                            .with_color(SERVER_WARNING_COLOR),
                    )
                    .await?;
            }
        }
        Ok(())
    }
    fn should_show_in_help_menu(&self, context: &HandlerContext<'_>) -> bool {
        context.initiator().check_permission_if_player(permissions::GIVE)
    }
}

async fn make_stack(item: &str, mut count: u32, context: &HandlerContext<'_>) -> Result<ItemStack> {
    let item = match context.item_manager().get_item(item) {
        Some(item) => item,
        None => bail!("Item not found"),
    };
    if !item.stackable() && count != 1 {
        context
            .initiator()
            .send_chat_message(
                ChatMessage::new_server_message("Item is not stackable, setting count to 1")
                    .with_color(SERVER_WARNING_COLOR),
            )
            .await?;
        count = 1
    }
    if let Some(QuantityType::Wear(x)) = item.proto.quantity_type {
        count = x;
    }
    Ok(item.make_stack(count))
}

struct GiveCommand;
#[async_trait]
impl ChatCommandHandler for GiveCommand {
    async fn handle(&self, message: &str, context: &HandlerContext<'_>) -> Result<()> {
        if !context
            .initiator()
            .check_permission_if_player(permissions::GIVE)
        {
            bail!("Insufficient permissions");
        }
        let params = message.split_whitespace().collect::<Vec<_>>();
        let (recipient, item, count) = match params.len() {
            3 => (params[1], params[2], 1),
            4 => (
                params[1],
                params[2],
                params[3].parse().context("Invalid count")?,
            ),
            _ => bail!("Incorrect usage: should be /giveme <item> [count]"),
        };
        let player_inventory = context
            .player_manager()
            .with_connected_player(recipient, |p| Ok(p.main_inventory()))?;
        let stack = make_stack(item, count, context).await?;
        let leftover = context
            .inventory_manager()
            .mutate_inventory_atomically(&player_inventory, |inv| Ok(inv.try_insert(stack)))?;
        if leftover.is_some() {
            context
                .initiator()
                .send_chat_message(
                    ChatMessage::new_server_message("Not enough space in inventory")
                        .with_color(SERVER_WARNING_COLOR),
                )
                .await?;
        }
        Ok(())
    }
    fn should_show_in_help_menu(&self, context: &HandlerContext<'_>) -> bool {
        context.initiator().check_permission_if_player(permissions::GIVE)
    }
}

struct SetTimeCommand;
#[async_trait]
impl ChatCommandHandler for SetTimeCommand {
    async fn handle(&self, message: &str, context: &HandlerContext<'_>) -> Result<()> {
        if !context
            .initiator()
            .check_permission_if_player(permissions::WORLD_STATE)
        {
            bail!("Insufficient permissions");
        }
        let params = message.split_whitespace().collect::<Vec<_>>();
        if params.len() != 2 {
            bail!("Incorrect usage: should be /settime <time>");
        }
        let time: f64 = params[1].parse()?;
        context.set_time_of_day(time);
        Ok(())
    }
    fn should_show_in_help_menu(&self, context: &HandlerContext<'_>) -> bool {
        context.initiator().check_permission_if_player(permissions::WORLD_STATE)
    }
}

struct DiscardCommand;
#[async_trait]
impl ChatCommandHandler for DiscardCommand {
    async fn handle(&self, _message: &str, context: &HandlerContext<'_>) -> Result<()> {
        if !context
            .initiator()
            .check_permission_if_player(permissions::INVENTORY)
        {
            bail!("Insufficient permissions");
        }
        if let EventInitiator::Player(p) = context.initiator() {
            let inv = p.player.main_inventory();
            let slot = p
                .player
                .selected_hotbar_slot()
                .context("Unknown selected_inv_slot, this is a bug")?;
            context
                .inventory_manager()
                .mutate_inventory_atomically(&inv, |inv| {
                    *inv.contents_mut()
                        .get_mut(slot as usize)
                        .context("Item slot was out of bounds; this is a bug")? = None;
                    Ok(())
                })?;
        }
        Ok(())
    }
    fn should_show_in_help_menu(&self, context: &HandlerContext<'_>) -> bool {
        context.initiator().check_permission_if_player(permissions::INVENTORY)
    }
}

struct DiscardAllCommand;
#[async_trait]
impl ChatCommandHandler for DiscardAllCommand {
    async fn handle(&self, _message: &str, context: &HandlerContext<'_>) -> Result<()> {
        if !context
            .initiator()
            .check_permission_if_player(permissions::INVENTORY)
        {
            bail!("Insufficient permissions");
        }
        if let EventInitiator::Player(p) = context.initiator() {
            let inv = p.player.main_inventory();
            context
                .inventory_manager()
                .mutate_inventory_atomically(&inv, |inv| {
                    for slot in inv.contents_mut().iter_mut() {
                        *slot = None;
                    }
                    Ok(())
                })?;
        }
        Ok(())
    }
    fn should_show_in_help_menu(&self, context: &HandlerContext<'_>) -> bool {
        context.initiator().check_permission_if_player(permissions::INVENTORY)
    }
}