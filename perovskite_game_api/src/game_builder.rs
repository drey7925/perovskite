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
    any::{Any, TypeId},
    collections::HashMap,
    path::Path,
    time::Duration,
};

use perovskite_core::{
    constants::{
        block_groups::TRIVIALLY_REPLACEABLE, items::default_item_interaction_rules,
        textures::FALLBACK_UNKNOWN_TEXTURE,
    },
    protocol::{blocks::Empty, items::ItemDef, render::TextureReference},
};
use perovskite_server::{
    game_state::{
        blocks::BlockTypeHandle,
        chat::commands::ChatCommandHandler,
        game_map::{TimerCallback, TimerSettings},
        items::Item,
    },
    server::ServerBuilder,
};

use anyhow::Result;

/// Type-safe newtype wrapper for a texture name
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TextureName(pub String);

/// Type-safe newtype wrapper for a texture name
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StaticTextureName(pub &'static str);

impl From<StaticTextureName> for TextureReference {
    fn from(value: StaticTextureName) -> Self {
        TextureReference {
            texture_name: value.0.to_string(),
            crop: None,
        }
    }
}
impl From<TextureName> for TextureReference {
    fn from(value: TextureName) -> Self {
        TextureReference {
            texture_name: value.0,
            crop: None,
        }
    }
}
impl From<&TextureName> for TextureReference {
    fn from(value: &TextureName) -> Self {
        TextureReference {
            texture_name: value.0.to_string(),
            crop: None,
        }
    }
}
impl From<StaticTextureName> for TextureName {
    fn from(value: StaticTextureName) -> Self {
        TextureName(value.0.to_string())
    }
}
impl From<&TextureName> for TextureName {
    fn from(value: &TextureName) -> Self {
        value.clone()
    }
}

pub(crate) const FALLBACK_UNKNOWN_TEXTURE_NAME: StaticTextureName =
    StaticTextureName(FALLBACK_UNKNOWN_TEXTURE);

/// Type-safe newtype wrapper for a const/static block name
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StaticBlockName(pub &'static str);

/// Type-safe wrapper for a block name
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
    blocks::{BlockBuilder, BuiltBlock, FallingBlocksChunkEdgePropagator, LiquidPropagator},
    maybe_export,
};

mod private {
    use std::any::Any;

    // This trait is needed to provide both downcasting to a concrete extension type,
    // and to also allow dynamic calls to common methods (e.g. pre_run).
    //
    // To do this, we need to get two different vtables for the object:
    // * One for the Any trait, so we can downcast it
    // * One for the GameBuilderExtension trait, so we can call its common methods
    // If we did this via GameBuilderExtension: Any, we would need an upcast which is not currently stabilized
    // (https://github.com/rust-lang/rust/issues/65991) or some other shenanigans that were hard to reason about,
    // got messy, and didn't work fully
    //
    // Instead, we create a new trait that gets a generic impl; as long as we have a &dyn GameBuilderExtension,
    // we can call (via dynamic dispatch) a concrete as_any which gets us a fat pointer with the appropriate vtable
    // for the concrete impl of Any for the actual concrete extension type.
    //
    // Note that this trait is private, so it can't be used outside of this module - we don't want unusual impls or
    // outside usages.
    //
    // Also note that this is not in a performance-critical path, so I don't mind the dynamic dispatch
    pub trait AsAny: Any + Send + Sync + 'static {
        fn as_any(&mut self) -> &mut dyn Any;
    }
    impl<T: Any + Send + Sync + 'static> AsAny for T {
        fn as_any(&mut self) -> &mut dyn Any {
            self
        }
    }
}

pub trait GameBuilderExtension: private::AsAny {
    /// Called before the server starts running
    /// At this point, there is no longer an opportunity to interact with other
    /// plugins' extensions (their pre_run may already have been called)
    ///
    /// This is the last opportunity for a plugin to modify the parts of the game-state
    /// that are immutable after init (e.g. defined blocks, defined items, etc)
    ///
    /// If there are any extensions that should be made available through the GameState,
    /// then they should be added to the ServerBuilder via [ServerBuilder::add_extension]
    fn pre_run(&mut self, server_builder: &mut ServerBuilder);
}

/// Stable API for building and configuring a game.
///
/// API stability note: *Before 1.0.0*, it is possible that functions returning
/// `()` or `Result<()>` may be changed to return something other than the empty
/// unit tuple.
pub struct GameBuilder {
    pub(crate) inner: ServerBuilder,
    pub(crate) liquids_by_flow_time: HashMap<Duration, Vec<BlockTypeHandle>>,
    pub(crate) falling_blocks: Vec<BlockTypeHandle>,
    // We cannot use a typemap here because we want to be able to iterate
    // over all the extensions for various things like pre_run
    pub(crate) builder_extensions: HashMap<TypeId, Box<dyn GameBuilderExtension>>,
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
        self.inner
            .blocks_mut()
            .register_fast_block_group(TRIVIALLY_REPLACEABLE);
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
                    idle_chunk_after_unchanged: true,
                    ..Default::default()
                },
                TimerCallback::LockedVerticalNeighors(Box::new(FallingBlocksChunkEdgePropagator {
                    blocks: self.falling_blocks.clone(),
                })),
            );
        }
        for extension in self.builder_extensions.values_mut() {
            extension.pre_run(&mut self.inner);
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
        Ok(GameBuilder {
            inner,
            liquids_by_flow_time: HashMap::new(),
            falling_blocks: vec![],
            builder_extensions: HashMap::new(),
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
        short_name: impl Into<ItemName>,
        display_name: impl Into<String>,
        texture: impl Into<TextureReference>,
        groups: Vec<String>,
    ) -> Result<()> {
        self.inner.items_mut().register_item(Item {
            proto: ItemDef {
                short_name: short_name.into().0.to_string(),
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
        tex_name: impl Into<TextureName>,
        file_path: impl AsRef<Path>,
    ) -> Result<()> {
        self.inner
            .media_mut()
            // TODO - this is wretched and should be refactored
            .register_from_file(&tex_name.into().0, file_path)
    }

    /// Adds a texture to the game with data passed as bytes.
    /// tex_name must be unique across all textures; an error will be returned
    /// if it is a duplicate
    pub fn register_texture_bytes(
        &mut self,
        tex_name: impl Into<TextureName>,
        data: &[u8],
    ) -> Result<()> {
        self.inner
            .media_mut()
            .register_from_memory(&tex_name.into().0, data)
    }

    pub fn data_dir(&self) -> &std::path::PathBuf {
        self.inner.data_dir()
    }

    /// Returns an extension that holds state for additional functionality or APIs
    /// for a specific plugin (e.g. default_game providing ore generation to other plugins that want
    /// to generate ores) without the core GameBuilder needing to be aware of it *a priori*.
    pub fn builder_extension<T: GameBuilderExtension + Any + Default + 'static>(
        &mut self,
    ) -> &mut T {
        self.builder_extensions
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::<T>::default())
            .as_any()
            .downcast_mut::<T>()
            .unwrap()
    }
}

/// Convenience helper for including a texture in the source tree into the game.
///
/// file_name is looked up relative to the current file (see [std::include_bytes]).
///
/// This macro takes the following parameters:
/// * Mutable reference to [GameBuilder]
/// * Texture name ([TextureName] object)
/// * File name (string literal)
#[macro_export]
macro_rules! include_texture_bytes {
    ($game_builder:expr, $tex_name:expr, $file_name:literal) => {
        $game_builder.register_texture_bytes($tex_name, include_bytes!($file_name))
    };
}
pub use include_texture_bytes;
