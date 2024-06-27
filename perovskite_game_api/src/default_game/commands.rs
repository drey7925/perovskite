use std::time::Instant;

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use cgmath::Vector3;
use perovskite_core::{
    chat::{ChatMessage, SERVER_WARNING_COLOR},
    constants::permissions,
    coordinates::BlockCoordinate,
    protocol::items::item_def::QuantityType,
};
use perovskite_server::game_state::{
    chat::commands::ChatCommandHandler,
    event::{EventInitiator, HandlerContext},
    items::ItemStack,
};

use crate::game_builder::GameBuilder;

pub(crate) fn register_default_commands(game_builder: &mut GameBuilder) -> Result<()> {
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
    game_builder.add_command(
        "teleport",
        Box::new(TeleportCommand),
        "[player] <x> <y> <z>: Teleports a player to the given coordinates. If no player specified, teleports you.",
    )?;
    game_builder.add_command(
        "benchmark",
        Box::new(TestonlyBenchmarkCommand),
        ": Runs a benchmark to measure the performance of the game.",
    )?;
    game_builder.add_command(
        "panic",
        Box::new(TestonlyPanicCommand),
        ": Panics the player's coroutine.",
    )?;
    game_builder.add_command(
        "attach_to_entity",
        Box::new(AttachEntityCommand),
        "<entity_id>: Attaches yourself to the given entity.",
    )?;
    game_builder.add_command(
        "detach_from_entity",
        Box::new(DetachEntityCommand),
        ": Detaches yourself from the entity you're attached to.",
    )?;
    Ok(())
}

struct WhereAmICommand;
#[async_trait]
impl ChatCommandHandler for WhereAmICommand {
    async fn handle(&self, _message: &str, context: &HandlerContext<'_>) -> Result<()> {
        if let EventInitiator::Player(p) = context.initiator() {
            p.player
                .send_chat_message_async(ChatMessage::new_server_message(format!(
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
                    .send_chat_message_async(
                        ChatMessage::new_server_message("Not enough space in inventory")
                            .with_color(SERVER_WARNING_COLOR),
                    )
                    .await?;
            }
        }
        Ok(())
    }
    fn should_show_in_help_menu(&self, context: &HandlerContext<'_>) -> bool {
        context
            .initiator()
            .check_permission_if_player(permissions::GIVE)
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
            .send_chat_message_async(
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
                .send_chat_message_async(
                    ChatMessage::new_server_message("Not enough space in inventory")
                        .with_color(SERVER_WARNING_COLOR),
                )
                .await?;
        }
        Ok(())
    }
    fn should_show_in_help_menu(&self, context: &HandlerContext<'_>) -> bool {
        context
            .initiator()
            .check_permission_if_player(permissions::GIVE)
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
        context
            .initiator()
            .check_permission_if_player(permissions::WORLD_STATE)
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
        context
            .initiator()
            .check_permission_if_player(permissions::INVENTORY)
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
        context
            .initiator()
            .check_permission_if_player(permissions::INVENTORY)
    }
}

struct TeleportCommand;
#[async_trait]
impl ChatCommandHandler for TeleportCommand {
    async fn handle(&self, message: &str, context: &HandlerContext<'_>) -> Result<()> {
        if !context
            .initiator()
            .check_permission_if_player(permissions::WORLD_STATE)
        {
            bail!("Insufficient permissions");
        }
        let params = message.split_whitespace().collect::<Vec<_>>();
        let (name, coords) = match params.len() {
            4 => {
                let name = match context.initiator() {
                    EventInitiator::Player(p) => p.player.name(),
                    _ => bail!("Incorrect usage: no player specified and caller was not a player"),
                };
                (
                    name,
                    (params[1].parse()?, params[2].parse()?, params[3].parse()?),
                )
            }
            5 => (
                params[1],
                (params[2].parse()?, params[3].parse()?, params[4].parse()?),
            ),
            _ => bail!("Incorrect usage: should be /teleport [player] <x> <y> <z>"),
        };
        let coords = Vector3::<f64>::new(coords.0, coords.1, coords.2);

        if !coords.x.is_finite() || !coords.y.is_finite() || !coords.z.is_finite() {
            bail!("Incorrect usage: a coordinate was infinite/NaN");
        }

        context.player_manager().with_connected_player(name, |p| {
            p.set_position_blocking(coords)?;
            Ok(())
        })?;

        Ok(())
    }
}

struct TestonlyBenchmarkCommand;
#[async_trait]
impl ChatCommandHandler for TestonlyBenchmarkCommand {
    async fn handle(&self, _message: &str, context: &HandlerContext<'_>) -> Result<()> {
        {
            let start = Instant::now();
            let sgm = context.game_map();
            let mut oks = 0;
            tokio::task::block_in_place(|| {
                for _ in 0..1000000 {
                    if std::hint::black_box(sgm.try_get_block(BlockCoordinate::new(0, 0, 0)))
                        .is_some()
                    {
                        oks += 1;
                    };
                }
            });

            let end = Instant::now();
            context
                .initiator()
                .send_chat_message_async(ChatMessage::new_server_message(format!(
                    "Fastpath took {} ms ({:?} per iter), {} ok",
                    (end - start).as_millis(),
                    (end - start) / 1000000,
                    oks
                )))
                .await?;
        }
        {
            let start = Instant::now();
            let sgm = context.game_map();
            let mut oks = 0;
            tokio::task::block_in_place(|| {
                for i in 0..100 {
                    for j in 0..100 {
                        for k in 0..100 {
                            if std::hint::black_box(
                                sgm.try_get_block(BlockCoordinate::new(i, j, k)),
                            )
                            .is_some()
                            {
                                oks += 1;
                            };
                        }
                    }
                }
            });

            let end = Instant::now();
            context
                .initiator()
                .send_chat_message_async(ChatMessage::new_server_message(format!(
                    "Fastpath moving took {} ms ({:?} per iter), {} ok",
                    (end - start).as_millis(),
                    (end - start) / 1000000,
                    oks
                )))
                .await?;
        }
        {
            let start = Instant::now();
            let sgm = context.game_map();
            let mut oks = 0;
            tokio::task::block_in_place(|| {
                for _ in 0..1000000 {
                    if std::hint::black_box(sgm.get_block(BlockCoordinate::new(0, 0, 0))).is_ok() {
                        oks += 1;
                    };
                }
            });

            let end = Instant::now();
            context
                .initiator()
                .send_chat_message_async(ChatMessage::new_server_message(format!(
                    "Slowpath took {} ms ({:?} per iter), {} ok",
                    (end - start).as_millis(),
                    (end - start) / 1000000,
                    oks
                )))
                .await?;
        }
        {
            let start = Instant::now();
            let sgm = context.game_map();
            let mut oks = 0;
            tokio::task::block_in_place(|| {
                for i in 0..100 {
                    for j in 0..100 {
                        for k in 0..100 {
                            if std::hint::black_box(sgm.get_block(BlockCoordinate::new(i, j, k)))
                                .is_ok()
                            {
                                oks += 1;
                            };
                        }
                    }
                }
            });

            let end = Instant::now();
            context
                .initiator()
                .send_chat_message_async(ChatMessage::new_server_message(format!(
                    "Slowpath moving took {} ms ({:?} per iter), {} ok",
                    (end - start).as_millis(),
                    (end - start) / 1000000,
                    oks
                )))
                .await?;
        }
        Ok(())
    }
}

struct TestonlyPanicCommand;
#[async_trait]
impl ChatCommandHandler for TestonlyPanicCommand {
    async fn handle(&self, _message: &str, _context: &HandlerContext<'_>) -> Result<()> {
        panic!("Test panic");
    }
}

struct AttachEntityCommand;
#[async_trait]
impl ChatCommandHandler for AttachEntityCommand {
    async fn handle(&self, message: &str, context: &HandlerContext<'_>) -> Result<()> {
        if let EventInitiator::Player(p) = context.initiator() {
            let parts = message.split(' ').collect::<Vec<_>>();
            if parts.len() != 2 {
                bail!("Incorrect usage: should be /attach entity_id");
            }
            let entity_id = parts[1].parse::<u64>()?;
            p.player.attach_to_entity(entity_id).await?;
            Ok(())
        } else {
            bail!("Only players can attach entities");
        }
    }

    fn should_show_in_help_menu(&self, context: &HandlerContext<'_>) -> bool {
        context
            .initiator()
            .check_permission_if_player(permissions::WORLD_STATE)
    }
}

struct DetachEntityCommand;
#[async_trait]
impl ChatCommandHandler for DetachEntityCommand {
    async fn handle(&self, _message: &str, context: &HandlerContext<'_>) -> Result<()> {
        if let EventInitiator::Player(p) = context.initiator() {
            p.player.detach_from_entity().await?;
            Ok(())
        } else {
            bail!("Only players can attach entities");
        }
    }

    fn should_show_in_help_menu(&self, context: &HandlerContext<'_>) -> bool {
        context
            .initiator()
            .check_permission_if_player(permissions::WORLD_STATE)
    }
}
