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

use crate::game_builder::{GameBuilder, GameBuilderExtension};

use anyhow::{ensure, Result};

use perovskite_core::protocol::items as items_proto;
use perovskite_server::game_state::items::ItemStack;

use self::{
    game_settings::DefaultGameSettings,
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

pub mod item_groups {
    /// Ores that were extracted from stone and have yet to be processed
    pub const RAW_ORES: &str = "default:raw_ores";
    /// Ingots from smelting ore
    pub const METAL_INGOTS: &str = "default:metal_ingots";
    /// Gems and crystals directly usable after extracting
    pub const GEMS: &str = "default:gems";
}

/// Control of the map generator.
pub mod mapgen;

/// Builder for a game based on the default game.
/// Eventually, hooks will be added so other game content can integrate closely
/// with default game content (e.g. in the mapgen). For now, this simply wraps
/// a [GameBuilder].
pub trait DefaultGameBuilder {
    /// Initializes the default game in the current game builder.
    /// This will set the mapgen, game behaviors, inventory menu, register blocks/items, etc.
    /// 
    /// If called multiple times, only the first call will have any effect. Subsequent calls
    /// will be no-ops.
    fn initialize_default_game(&mut self) -> Result<()>;

    /// Returns an Arc for the crafting recipes in this game.
    fn crafting_recipes(&mut self) -> Arc<RecipeBook<9, ()>>;
    /// Registers a new crafting recipe.
    ///
    /// If stackable is false, quantity represents item wear. Behavior is undefined if stackable is false,
    /// the item isn't subject to tool wear, and quantity != 1.
    ///
    /// **This API is subject to change.**
    fn register_crafting_recipe(
        &mut self,
        slots: [RecipeSlot; 9],
        result: String,
        quantity: u32,
        quantity_type: Option<items_proto::item_stack::QuantityType>,
    );

    /// Registers a new smelting fuel
    /// Args:
    ///   - fuel_name: Name of the fuel
    ///   - ticks: Metadata is number of furnace timer ticks (period tbd) that the fuel lasts for
    fn register_smelting_fuel(&mut self, fuel_name: impl Into<String>, ticks: u32);
}

// This is a private type; other plugins cannot name it so they cannot access
// the builder_extension directly.
//
// This ensures that the builder_extension is only made available to plugins via the
// `DefaultGameBuilder` API.
struct DefaultGameBuilderExtension {
    settings: DefaultGameSettings,

    crafting_recipes: Arc<RecipeBook<9, ()>>,
    /// Metadata is number of furnace timer ticks (period given by [`furnace::FURNACE_TICK_DURATION`]) that it takes to smelt this recipe
    smelting_recipes: Arc<RecipeBook<1, u32>>,
    // Metadata is number of furnace timer ticks (period tbd) that the fuel lasts for
    // Output item is ignored
    smelting_fuels: Arc<RecipeBook<1, u32>>,

    ores: Vec<OreDefinition>,
    initialized: bool,
}
impl Default for DefaultGameBuilderExtension {
    fn default() -> Self {
        Self {
            settings: DefaultGameSettings::default(),
            crafting_recipes: Arc::new(RecipeBook::new()),
            smelting_recipes: Arc::new(RecipeBook::new()),
            smelting_fuels: Arc::new(RecipeBook::new()),
            ores: vec![],
            initialized: false,
        }
    }
}

impl GameBuilderExtension for DefaultGameBuilderExtension {
    fn pre_run(&mut self, server_builder: &mut perovskite_server::server::ServerBuilder) {
        tracing::info!("DefaultGame doing pre-run initialization");
        self.crafting_recipes.sort();
        self.smelting_recipes.sort();

        let ores = self.ores.drain(..).collect();
        server_builder.set_mapgen(move |blocks, seed| mapgen::build_mapgen(blocks, seed, ores));
    }
}

impl DefaultGameBuilderExtension {
    /// Adds an ore to the mapgen.
    ///
    /// **API skeleton, parameters TBD**
    /// Will be made pub when finalized and implemented
    fn register_ore(&mut self, ore_definition: OreDefinition) {
        self.ores.push(ore_definition);
    }
}

impl DefaultGameBuilder for GameBuilder {

    fn initialize_default_game(&mut self) -> Result<()> {
        let data_dir = self.data_dir().clone();
        let ext = self.builder_extension::<DefaultGameBuilderExtension>();
        if ext.initialized {
            return Ok(());
        }

        tracing::info!("DefaultGame doing main initialization");
        ext.settings = game_settings::load(&data_dir)?;
        ext.initialized = true;
        basic_blocks::register_basic_blocks(self)?;
        game_behaviors::register_game_behaviors(self)?;
        tools::register_default_tools(self)?;
        furnace::register_furnace(self)?;
        foliage::register_foliage(self)?;
        commands::register_default_commands(self)?;
        Ok(())
    }
    

    /// Returns an Arc for the crafting recipes in this game.
    fn crafting_recipes(&mut self) -> Arc<RecipeBook<9, ()>> {
        self.builder_extension::<DefaultGameBuilderExtension>()
            .crafting_recipes
            .clone()
    }
    /// Registers a new crafting recipe.
    ///
    /// If stackable is false, quantity represents item wear. Behavior is undefined if stackable is false,
    /// the item isn't subject to tool wear, and quantity != 1.
    ///
    /// **This API is subject to change.**
    fn register_crafting_recipe(
        &mut self,
        slots: [RecipeSlot; 9],
        result: String,
        quantity: u32,
        quantity_type: Option<items_proto::item_stack::QuantityType>,
    )  {
        assert!(
            self.builder_extension::<DefaultGameBuilderExtension>().initialized,
            "DefaultGame builder_extension not initialized"
        );
        self.builder_extension::<DefaultGameBuilderExtension>()
            .crafting_recipes
            .register_recipe(RecipeImpl {
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

    fn register_smelting_fuel(&mut self, fuel_name: impl Into<String>, ticks: u32)  {
        assert!(
            self.builder_extension::<DefaultGameBuilderExtension>().initialized,
            "DefaultGame builder_extension not initialized"
        );
        self.builder_extension::<DefaultGameBuilderExtension>()
            .smelting_fuels
            .register_recipe(RecipeImpl {
                slots: [RecipeSlot::Exact(fuel_name.into())],
                result: ItemStack {
                    proto: Default::default(),
                },
                shapeless: false,
                metadata: ticks,
            });
    }
}

mod commands;

pub mod game_settings;
