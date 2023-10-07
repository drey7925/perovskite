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

use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

use crate::game_builder::GameBuilder;

use anyhow::{bail, Context, Result};

use async_trait::async_trait;
use perovskite_core::{
    chat::{ChatMessage, SERVER_WARNING_COLOR},
    protocol::items::{self as items_proto, item_def::QuantityType},
};
use perovskite_server::game_state::{
    chat::commands::CommandImplementation,
    event::{EventInitiator, HandlerContext, PlayerInitiator},
    items::ItemStack,
};

use self::{
    mapgen::OreDefinition,
    recipes::{RecipeBook, RecipeImpl, RecipeSlot},
};

/// Blocks defined in the default game.
pub mod basic_blocks;
/// Trees, plants, etc
pub mod foliage;
/// Basic server behaviors not covered in other modules
pub mod game_behaviors;
/// Recipes for crafting, smelting, etc
pub mod recipes;
/// Helpers for stairs, slabs, and other blocks derived from a base block
pub mod shaped_blocks;
/// Standard tools - pickaxes, shovels, axes
pub mod tools;

#[cfg(feature = "unstable_api")]
/// Furnace implementation,
pub mod furnace;
#[cfg(not(feature = "unstable_api"))]
mod furnace;

/// Common block groups that are defined in the default game and integrate with its tools
/// See also [perovskite_core::constants::block_groups]
pub mod block_groups {
    /// Brittle, stone-like blocks that are best removed with a pickaxe.
    /// e.g. stone, cobble, bricks
    pub const BRITTLE: &str = "default:brittle";
    /// Granular blocks that are best removed with a shovel, e.g. dirt
    pub const GRANULAR: &str = "default:granular";
    /// Woody blocks that are best removed with an axe, e.g. wood
    pub const FIBROUS: &str = "default:fibrous";
}

/// Control of the map generator.
pub mod mapgen;

/// Builder for a game based on the default game.
/// Eventually, hooks will be added so other game content can integrate closely
/// with default game content (e.g. in the mapgen). For now, this simply wraps
/// a [GameBuilder].
pub struct DefaultGameBuilder {
    inner: GameBuilder,

    crafting_recipes: Arc<RecipeBook<9, ()>>,
    /// Metadata is number of furnace timer ticks (period given by [`furnace::FURNACE_TICK_DURATION`]) that it takes to smelt this recipe
    smelting_recipes: Arc<RecipeBook<1, u32>>,
    // Metadata is number of furnace timer ticks (period tbd) that the fuel lasts for
    // Output item is ignored
    smelting_fuels: Arc<RecipeBook<1, u32>>,

    ores: Vec<OreDefinition>,
}
impl Deref for DefaultGameBuilder {
    type Target = GameBuilder;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl DerefMut for DefaultGameBuilder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
impl DefaultGameBuilder {
    /// Provides access to the [GameBuilder] that this DefaultGameBuilder is wrapping,
    /// e.g. to register blocks and items.
    pub fn game_builder(&mut self) -> &mut GameBuilder {
        &mut self.inner
    }
    /// Creates a new default-game builder using server configuration from the
    /// command line. If argument parsing fails, usage info is printed to
    /// the terminal and the process exits.
    pub fn new_from_commandline() -> Result<DefaultGameBuilder> {
        Self::new_with_builtins(GameBuilder::from_cmdline()?)
    }
    /// Creates a new default-game builder with custom server configuration.
    #[cfg(feature = "unstable_api")]
    pub fn new_from_args(
        args: &perovskite_server::server::ServerArgs,
    ) -> Result<DefaultGameBuilder> {
        Self::new_with_builtins(GameBuilder::from_args(args)?)
    }

    fn new_with_builtins(
        builder: GameBuilder,
    ) -> std::result::Result<DefaultGameBuilder, anyhow::Error> {
        let mut builder = DefaultGameBuilder {
            inner: builder,
            crafting_recipes: Arc::new(RecipeBook::new()),
            smelting_recipes: Arc::new(RecipeBook::new()),
            smelting_fuels: Arc::new(RecipeBook::new()),
            ores: Vec::new(),
        };
        register_defaults(&mut builder)?;
        Ok(builder)
    }

    /// Adds an ore to the mapgen.
    ///
    /// **API skeleton, unimplemented, parameters TBD**
    /// Will be made pub when finalized and implemented
    fn register_ore(&mut self, ore_definition: OreDefinition) {
        self.ores.push(ore_definition);
    }

    /// Returns an Arc for the crafting recipes in this game.
    pub fn crafting_recipes(&mut self) -> Arc<RecipeBook<9, ()>> {
        self.crafting_recipes.clone()
    }
    /// Registers a new crafting recipe.
    ///
    /// If stackable is false, quantity represents item wear. Behavior is undefined if stackable is false,
    /// the item isn't subject to tool wear, and quantity != 1.
    ///
    /// **This API is subject to change.**
    pub fn register_crafting_recipe(
        &mut self,
        slots: [RecipeSlot; 9],
        result: String,
        quantity: u32,
        quantity_type: Option<items_proto::item_stack::QuantityType>,
    ) {
        self.crafting_recipes.register_recipe(RecipeImpl {
            slots,
            result: ItemStack {
                proto: perovskite_core::protocol::items::ItemStack {
                    item_name: result,
                    quantity: if matches!(
                        quantity_type,
                        Some(items_proto::item_stack::QuantityType::Stack(_))
                    ) {
                        quantity
                    } else {
                        1
                    },
                    current_wear: if matches!(
                        quantity_type,
                        Some(items_proto::item_stack::QuantityType::Wear(_))
                    ) {
                        quantity
                    } else {
                        1
                    },
                    quantity_type,
                },
            },
            shapeless: false,
            metadata: (),
        })
    }

    /// Starts a game based on this builder.
    pub fn build_and_run(mut self) -> Result<()> {
        self.crafting_recipes.sort();
        self.smelting_recipes.sort();

        let ores = self.ores.drain(..).collect();
        self.game_builder()
            .inner
            .set_mapgen(move |blocks, seed| mapgen::build_mapgen(blocks, seed, ores));
        self.inner.run_game_server()
    }
}

fn register_defaults(game_builder: &mut DefaultGameBuilder) -> Result<()> {
    basic_blocks::register_basic_blocks(game_builder)?;
    game_behaviors::register_game_behaviors(game_builder)?;
    recipes::register_test_recipes(game_builder);
    tools::register_default_tools(game_builder)?;
    furnace::register_furnace(game_builder)?;
    foliage::register_foliage(game_builder)?;
    register_default_commands(game_builder)?;
    Ok(())
}

fn register_default_commands(game_builder: &mut DefaultGameBuilder) -> Result<()> {
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
impl CommandImplementation for WhereAmICommand {
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
impl CommandImplementation for GiveMeCommand {
    async fn handle(&self, message: &str, context: &HandlerContext<'_>) -> Result<()> {
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
impl CommandImplementation for GiveCommand {
    async fn handle(&self, message: &str, context: &HandlerContext<'_>) -> Result<()> {
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
            .for_connected_player(recipient, |p| Ok(p.main_inventory()))?;
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
}

struct SetTimeCommand;
#[async_trait]
impl CommandImplementation for SetTimeCommand {
    async fn handle(&self, message: &str, context: &HandlerContext<'_>) -> Result<()> {
        let params = message.split_whitespace().collect::<Vec<_>>();
        if params.len() != 2 {
            bail!("Incorrect usage: should be /settime <time>");
        }
        let time: f64 = params[1].parse()?;
        context.set_time_of_day(time);
        Ok(())
    }
}

struct DiscardCommand;
#[async_trait]
impl CommandImplementation for DiscardCommand {
    async fn handle(&self, _message: &str, context: &HandlerContext<'_>) -> Result<()> {
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
}

struct DiscardAllCommand;
#[async_trait]
impl CommandImplementation for DiscardAllCommand {
    async fn handle(&self, _message: &str, context: &HandlerContext<'_>) -> Result<()> {
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
}
