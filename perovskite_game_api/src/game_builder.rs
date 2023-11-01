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

use std::{arch::x86_64, collections::HashMap, path::Path, time::Duration};

use perovskite_core::{
    constants::{
        block_groups::DEFAULT_GAS, blocks::AIR, items::default_item_interaction_rules,
        textures::FALLBACK_UNKNOWN_TEXTURE,
    },
    protocol::{
        blocks::{
            block_type_def::{PhysicsInfo, RenderInfo},
            BlockTypeDef, Empty,
        },
        items::ItemDef,
        render::TextureReference,
    },
};
use perovskite_server::{
    game_state::{
        blocks::{BlockType, BlockTypeHandle},
        chat::commands::ChatCommandHandler,
        game_map::{TimerCallback, TimerSettings},
        items::Item,
    },
    server::ServerBuilder,
};

use anyhow::Result;

/// Type-safe newtype wrapper for a texture name
#[derive(Clone, Copy)]
pub struct TextureName(pub &'static str);

impl From<TextureName> for TextureReference {
    fn from(value: TextureName) -> Self {
        TextureReference {
            texture_name: value.0.to_string(),
            crop: None,
        }
    }
}

pub(crate) const FALLBACK_UNKNOWN_TEXTURE_NAME: TextureName = TextureName(FALLBACK_UNKNOWN_TEXTURE);

/// Type-safe newtype wrapper for a const/static block name
pub struct StaticBlockName(pub &'static str);
/// Type-safe wrapper for a block name
pub struct BlockName(pub String);
impl From<StaticBlockName> for BlockName {
    fn from(value: StaticBlockName) -> Self {
        BlockName(value.0.to_string())
    }
}

/// Type-safe newtype wrapper for a const/static item name
pub struct StaticItemName(pub &'static str);
/// Type-safe wrapper for an item name
pub struct ItemName(pub String);
impl From<StaticItemName> for ItemName {
    fn from(value: StaticItemName) -> Self {
        ItemName(value.0.to_string())
    }
}

#[cfg(feature = "unstable_api")]
/// Unstable re-export of the raw gameserver API. This API is subject to
/// breaking changes that do not follow semver, before 1.0
use perovskite_server::server as server_api;

use crate::{
    blocks::{BlockBuilder, BuiltBlock, FallingBlocksPropagator, LiquidPropagator, FallingBlocksChunkEdgePropagator},
    maybe_export,
};

/// Stable API for building and configuring a game.
///
/// API stability note: *Before 1.0.0*, it is possible that functions returning
/// `()` or `Result<()>` may be changed to return something other than the empty
/// unit tuple.
pub struct GameBuilder {
    pub(crate) inner: ServerBuilder,
    pub(crate) air_block: BlockTypeHandle,
    pub(crate) liquids_by_flow_time: HashMap<Duration, Vec<BlockTypeHandle>>,
    pub(crate) falling_blocks: Vec<BlockTypeHandle>,
}
impl GameBuilder {
    /// Creates a new game builder using server configuration from the
    /// command line. If argument parsing fails, usage info is printed to
    /// the terminal and the process exits.
    pub fn from_cmdline() -> Result<GameBuilder> {
        Self::new_with_builtins(ServerBuilder::from_cmdline()?)
    }

    /// Creates a new game builder with custom server configuration
    #[cfg(feature = "unstable_api")]
    pub fn from_args(args: &server_api::ServerArgs) -> Result<GameBuilder> {
        Self::new_with_builtins(ServerBuilder::from_args(args)?)
    }

    /// Borrows the ServerBuilder that can be used to directly register
    /// items, blocks, etc using the low-level unstable API.
    #[cfg(feature = "unstable_api")]
    pub fn server_builder_mut(&mut self) -> &mut server_api::ServerBuilder {
        &mut self.inner
    }

    /// Returns the ServerBuilder with everything built so far.
    #[cfg(feature = "unstable_api")]
    pub fn into_server_builder(mut self) -> Result<server_api::ServerBuilder> {
        self.pre_build()?;
        Ok(self.inner)
    }

    fn pre_build(&mut self) -> Result<()> {
        for (&period, liquid_group) in self.liquids_by_flow_time.iter() {
            self.inner.add_timer(
                format!("liquid_flow_{}", period.as_micros()),
                TimerSettings {
                    interval: period,
                    shards: 16,
                    spreading: 1.0,
                    block_types: liquid_group.clone(),
                    per_block_probability: 1.0,
                    ignore_block_type_presence_check: false,
                    idle_chunk_after_unchanged: true,
                    ..Default::default()
                },
                TimerCallback::BulkUpdateWithNeighbors(Box::new(LiquidPropagator {
                    liquids: liquid_group.clone(),
                    air: self.air_block,
                })),
            )
        }

        if !self.falling_blocks.is_empty() {
            self.inner.add_timer(
                "falling_blocks",
                TimerSettings {
                    interval: Duration::from_secs(1),
                    shards: 16,
                    spreading: 1.0,
                    block_types: self.falling_blocks.clone(),
                    per_block_probability: 1.0,
                    ignore_block_type_presence_check: true,
                    idle_chunk_after_unchanged: false,
                    ..Default::default()
                },
                // TimerCallback::BulkUpdate(Box::new(FallingBlocksPropagator {
                //     blocks: self.falling_blocks.clone(),
                //     air: self.air_block,
                // })),
                TimerCallback::LockedVerticalNeighors(Box::new(FallingBlocksChunkEdgePropagator {
                    blocks: self.falling_blocks.clone(),
                    air: self.air_block, 
                }))
            );
        }

        Ok(())
    }

    /// Run the game server
    pub fn run_game_server(mut self) -> Result<()> {
        self.pre_build()?;
        self.inner.build()?.serve()
    }

    // Instantiate some builtin content
    fn new_with_builtins(mut inner: ServerBuilder) -> Result<GameBuilder> {
        inner.media_mut().register_from_memory(
            FALLBACK_UNKNOWN_TEXTURE,
            include_bytes!("media/block_unknown.png"),
        )?;
        const EMPTY: Empty = Empty {};
        let mut air_block = BlockType::default();
        air_block.client_info = BlockTypeDef {
            id: 0,
            short_name: AIR.to_string(),
            render_info: Some(RenderInfo::Empty(EMPTY)),
            physics_info: Some(PhysicsInfo::Air(EMPTY)),
            base_dig_time: 1.0,
            groups: vec![DEFAULT_GAS.to_string()],
            wear_multiplier: 1.0,
            light_emission: 0,
            allow_light_propagation: true,
        };
        let air_block = inner.blocks_mut().register_block(air_block)?;
        Ok(GameBuilder {
            inner,
            air_block,
            liquids_by_flow_time: HashMap::new(),
            falling_blocks: vec![],
        })
    }
    /// Registers a block and its corresponding item in the game.
    pub fn add_block(&mut self, block_builder: BlockBuilder) -> Result<BuiltBlock> {
        block_builder.build_and_deploy_into(self)
    }

    pub fn get_block(&self, block_name: StaticBlockName) -> Option<BlockTypeHandle> {
        self.inner.blocks().get_by_name(block_name.0)
    }

    /// Registers a simple item that cannot be placed, doesn't have a block automatically generated for it, and is not a tool
    /// The item can be stacked in the inventory, but has no other behaviors. If used as a tool, it will behave the same as if
    /// nothing were held in the hand.
    pub fn register_basic_item(
        &mut self,
        short_name: StaticItemName,
        display_name: impl Into<String>,
        texture: TextureName,
        groups: Vec<String>,
    ) -> Result<()> {
        self.inner.items_mut().register_item(Item {
            proto: ItemDef {
                short_name: short_name.0.to_string(),
                display_name: display_name.into(),
                inventory_texture: Some(texture.into()),
                groups,
                interaction_rules: default_item_interaction_rules(),
                quantity_type: Some(
                    perovskite_core::protocol::items::item_def::QuantityType::Stack(256),
                ),
                block_apperance: "".to_string(),
            },
            dig_handler: None,
            tap_handler: None,
            place_handler: None,
        })
    }

    maybe_export!(
        /// Registers a chat command. name should not contain the leading slash
        // TODO: convert this into a builder once we have more features in commands
        fn add_command(
            &mut self,
            name: &str,
            command: Box<dyn ChatCommandHandler>,
            help: &str,
        ) -> Result<()> {
            self.inner.register_command(name, command, help)?;
            Ok(())
        }
    );

    /// Adds a texture to the game by reading from a file.
    ///
    /// tex_name must be unique across all textures; an error will be returned
    /// if it is a duplicate
    pub fn register_texture_file(
        &mut self,
        tex_name: TextureName,
        file_path: impl AsRef<Path>,
    ) -> Result<()> {
        self.inner
            .media_mut()
            .register_from_file(tex_name.0, file_path)
    }

    /// Adds a texture to the game with data passed as bytes.
    /// tex_name must be unique across all textures; an error will be returned
    /// if it is a duplicate
    pub fn register_texture_bytes(&mut self, tex_name: TextureName, data: &[u8]) -> Result<()> {
        self.inner
            .media_mut()
            .register_from_memory(tex_name.0, data)
    }

    pub fn data_dir(&self) -> &std::path::PathBuf {
        self.inner.data_dir()
    }
}

/// Convenience helper for including a texture in the source tree into the game.
///
/// file_name is looked up relative to the current file (see [std::include_bytes]).
///
/// This macro takes the following parameters:
/// * Mutable reference to [GameBuilder]
/// * Texture name ([Tex] object)
/// * File name (string literal)
#[macro_export]
macro_rules! include_texture_bytes {
    ($game_builder:expr, $tex_name:expr, $file_name:literal) => {
        $game_builder.register_texture_bytes($tex_name, include_bytes!($file_name))
    };
}
pub use include_texture_bytes;
