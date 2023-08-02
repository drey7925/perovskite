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

use std::sync::Arc;

use crate::game_builder::{GameBuilder, TextureName};

use anyhow::Result;

use cuberef_core::protocol::items::{self as items_proto, ItemDef};
use cuberef_server::game_state::items::{ItemStack, Item};

use self::{recipes::{RecipeBook, RecipeImpl, RecipeSlot}, mapgen::OreDefinition};

/// Blocks defined in the default game.
pub mod basic_blocks;
/// Basic server behaviors not covered in other modules
pub mod game_behaviors;
/// Recipes for crafting, smelting, etc
pub mod recipes;
/// Standard tools - pickaxes, shovels, axes
pub mod tools;
/// Trees, plants, etc
pub mod foliage;

#[cfg(feature = "unstable_api")]
/// Furnace implementation,
pub mod furnace;
#[cfg(not(feature = "unstable_api"))]
mod furnace;

/// Common block groups that are defined in the default game and integrate with its tools
/// See also [cuberef_core::constants::block_groups]
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

    ores: Vec<OreDefinition>
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
    pub fn new_from_args(args: &cuberef_server::server::ServerArgs) -> Result<DefaultGameBuilder> {
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
                proto: cuberef_core::protocol::items::ItemStack {
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
    Ok(())
}
