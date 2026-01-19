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
    path::{Path, PathBuf},
    time::Duration,
};

use perovskite_core::{
    constants::{
        block_groups::TRIVIALLY_REPLACEABLE, items::default_item_interaction_rules,
        textures::FALLBACK_UNKNOWN_TEXTURE,
    },
    protocol::{items::ItemDef, render::TextureReference},
};
use perovskite_server::{
    game_state::{
        blocks::BlockTypeHandle,
        chat::commands::ChatCommandHandler,
        game_map::{TimerCallback, TimerSettings},
        items::Item,
        GameState,
    },
    server::{GameDatabase, Server, ServerArgs, ServerBuilder},
};

use anyhow::{Context, Result};

/// Type-safe newtype wrapper for a texture name
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OwnedTextureName(pub String);

/// Type-safe newtype wrapper for a texture name
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StaticTextureName(pub &'static str);

pub trait TextureRefExt {
    /// Makes this texture shiny/reflective. Note that the *diffuse* texture's alpha controls some
    /// shininess behavior: alpha = 0 applies a Fresnel effect where reflections are stronger at
    /// shallow/grazing angles. Alpha = 1 applies constant reflectivity regardless of incident
    /// angle; intermediate values blend between them.
    ///
    /// Alpha channel of *this* texture is still TBD
    fn with_specular(self, tex: impl TextureName) -> TextureReference;

    /// Makes this texture emit strong light. Note that this is only used for screen-space effects
    /// and direct shiny hits when the user has raytracing enabled. This is tone-mapped in some
    /// TBD manner; setting R, G, or B to the max value may be rather bright.
    ///
    /// The meaning of the alpha channel in *this* texture is still TBD
    fn with_emissive(self, tex: impl TextureName) -> TextureReference;

    /// Adjusts normals for reflections. Note that this does not necessarily match other
    /// applications' normal formats.
    /// R = tangent, G = bitangent, 0.5 is neutral, 0 and 1 point left/right and up/down in texture
    /// mapped as vector_component = 2 * (texture_component) - 1
    /// space. Normal component is imputed from the tangent/bitangent components
    fn with_normal_map(self, tex: impl TextureName) -> TextureReference;
}
impl TextureRefExt for TextureReference {
    fn with_specular(self, tex: impl TextureName) -> TextureReference {
        TextureReference {
            rt_specular: tex.name().to_string(),
            ..self
        }
    }

    fn with_emissive(self, tex: impl TextureName) -> TextureReference {
        TextureReference {
            emissive: tex.name().to_string(),
            ..self
        }
    }

    fn with_normal_map(self, tex: impl TextureName) -> TextureReference {
        TextureReference {
            normal_map: tex.name().to_string(),
            ..self
        }
    }
}

impl OwnedTextureName {
    /// Creates a texture that's a solid color; CSS colors of the form `rgb(0 255 0)` or `orange`
    /// are accepted.
    pub fn from_css_color(color: &str) -> OwnedTextureName {
        OwnedTextureName(GENERATED_TEXTURE_CATEGORY_SOLID_FROM_CSS.to_string() + color)
    }
}

pub trait TextureName {
    fn name(&self) -> &str;
}
impl TextureName for OwnedTextureName {
    fn name(&self) -> &str {
        &self.0
    }
}
impl TextureName for StaticTextureName {
    fn name(&self) -> &str {
        self.0
    }
}
impl From<OwnedTextureName> for Option<Appearance> {
    fn from(value: OwnedTextureName) -> Self {
        Some(Appearance::InventoryTexture(value.name().to_string()))
    }
}
impl From<StaticTextureName> for Option<Appearance> {
    fn from(value: StaticTextureName) -> Self {
        Some(Appearance::InventoryTexture(value.name().to_string()))
    }
}

impl From<StaticTextureName> for TextureReference {
    fn from(value: StaticTextureName) -> Self {
        TextureReference {
            diffuse: value.0.to_string(),
            rt_specular: String::new(),
            crop: None,
            emissive: String::new(),
            normal_map: String::new(),
        }
    }
}
impl From<OwnedTextureName> for TextureReference {
    fn from(value: OwnedTextureName) -> Self {
        TextureReference {
            diffuse: value.0,
            rt_specular: String::new(),
            crop: None,
            emissive: String::new(),
            normal_map: String::new(),
        }
    }
}
impl From<&OwnedTextureName> for TextureReference {
    fn from(value: &OwnedTextureName) -> Self {
        TextureReference {
            diffuse: value.0.to_string(),
            rt_specular: String::new(),
            crop: None,
            emissive: String::new(),
            normal_map: String::new(),
        }
    }
}
impl From<StaticTextureName> for OwnedTextureName {
    fn from(value: StaticTextureName) -> Self {
        OwnedTextureName(value.0.to_string())
    }
}
impl From<&OwnedTextureName> for OwnedTextureName {
    fn from(value: &OwnedTextureName) -> Self {
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
impl From<&'static str> for BlockName {
    fn from(value: &'static str) -> Self {
        BlockName(value.to_string())
    }
}
impl From<&FastBlockName> for BlockName {
    fn from(value: &FastBlockName) -> Self {
        BlockName(value.name().to_string())
    }
}
impl From<StaticBlockName> for FastBlockName {
    fn from(value: StaticBlockName) -> Self {
        FastBlockName::new(value.0)
    }
}
impl From<&BlockName> for FastBlockName {
    fn from(value: &BlockName) -> Self {
        FastBlockName::new(&value.0)
    }
}

/// Type-safe newtype wrapper for a const/static item name

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StaticItemName(pub &'static str);

impl From<StaticItemName> for RecipeSlot {
    fn from(value: StaticItemName) -> Self {
        RecipeSlot::Exact(value.0.to_string())
    }
}

/// Type-safe wrapper for an item name
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ItemName(pub String);
impl From<StaticItemName> for ItemName {
    fn from(value: StaticItemName) -> Self {
        ItemName(value.0.to_string())
    }
}
impl From<&str> for ItemName {
    fn from(value: &str) -> Self {
        ItemName(value.to_string())
    }
}
impl From<String> for ItemName {
    fn from(value: String) -> Self {
        ItemName(value)
    }
}

#[cfg(feature = "unstable_api")]
/// Unstable re-export of the raw gameserver API. This API is subject to
/// breaking changes that do not follow semver, before 1.0
use perovskite_server::server as server_api;
use rand::RngCore;

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

    pub(crate) default_solid_footstep_sound: SoundKey,
}
impl GameBuilder {
    /// Creates a new game builder using server configuration from the
    /// command line. If argument parsing fails, usage info is printed to
    /// the terminal and the process exits.
    pub fn from_cmdline() -> Result<GameBuilder> {
        Self::from_serverbuilder(ServerBuilder::from_cmdline()?)
    }

    pub fn using_tempdir_disk_backed() -> Result<(GameBuilder, PathBuf)> {
        let data_dir =
            std::env::temp_dir().join(format!("perovskite-{}", rand::thread_rng().next_u64()));
        let builder = ServerBuilder::from_args(&ServerArgs {
            data_dir: data_dir.clone(),
            bind_addr: None,
            port: 0,
            trace_rate_denominator: usize::MAX,
            rocksdb_num_fds: 512,
            rocksdb_point_lookup_cache_mib: 128,
            num_map_prefetchers: 8,
        })?;

        Ok((Self::from_serverbuilder(builder)?, data_dir))
    }

    /// Creates a new game builder with custom server configuration
    #[cfg(feature = "unstable_api")]
    pub fn from_args(args: &ServerArgs) -> Result<GameBuilder> {
        Self::from_serverbuilder(ServerBuilder::from_args(args)?)
    }

    /// Borrows the ServerBuilder that can be used to directly register
    /// items, blocks, etc using the low-level unstable API.
    #[cfg(feature = "unstable_api")]
    pub fn server_builder_mut(&mut self) -> &mut ServerBuilder {
        &mut self.inner
    }

    /// Returns the ServerBuilder with everything built so far.
    #[cfg(feature = "unstable_api")]
    pub fn into_server_builder(mut self) -> Result<ServerBuilder> {
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

    pub fn force_seed(&mut self, seed: Option<u32>) {
        self.inner.force_seed(seed);
    }

    /// Run the game server. This function will block until the server is stopped.
    pub fn run_game_server(mut self) -> Result<()> {
        self.pre_build()?;
        self.inner.build()?.serve()
    }

    /// Instantiates a server from this game builder, but does not block on running it.
    pub(crate) fn into_server(mut self) -> Result<Server> {
        self.pre_build()?;
        self.inner.build()
    }

    /// Starts a server, runs a task in it, and returns. This is meant for unit-testing
    pub fn run_task_in_server<T>(
        mut self,
        task: impl FnOnce(&GameState) -> Result<T>,
    ) -> Result<T> {
        self.pre_build()?;
        let server = self.inner.build()?;
        server.run_task_in_server(task)
    }

    /// Instantiates a game builder from a server builder
    pub(crate) fn from_serverbuilder(mut inner: ServerBuilder) -> Result<GameBuilder> {
        inner.media_mut().register_from_memory(
            FALLBACK_UNKNOWN_TEXTURE,
            include_bytes!("media/block_unknown.png"),
        )?;

        inner.media_mut().register_from_memory(
            DEFAULT_FOOTSTEP_SOUND_NAME,
            include_bytes!("media/simple_footstep.wav"),
        )?;
        inner.media_mut().register_from_memory(
            SNOW_FOOTSTEP_SOUND_NAME,
            include_bytes!("media/footstep_snow.wav"),
        )?;
        inner.media_mut().register_from_memory(
            GRASS_FOOTSTEP_SOUND_NAME,
            include_bytes!("media/footstep_grass.wav"),
        )?;

        let footstep_key = inner
            .media_mut()
            .register_file_for_sampled_audio(DEFAULT_FOOTSTEP_SOUND_NAME)?;
        inner
            .media_mut()
            .register_file_for_sampled_audio(SNOW_FOOTSTEP_SOUND_NAME)?;
        inner
            .media_mut()
            .register_file_for_sampled_audio(GRASS_FOOTSTEP_SOUND_NAME)?;

        Ok(GameBuilder {
            inner,
            liquids_by_flow_time: HashMap::new(),
            falling_blocks: vec![],
            builder_extensions: HashMap::new(),
            default_solid_footstep_sound: footstep_key,
        })
    }
    /// Registers a block and its corresponding item in the game.
    pub fn add_block(&mut self, block_builder: BlockBuilder) -> Result<BuiltBlock> {
        block_builder.build_and_deploy_into(self)
    }

    pub fn get_block(&self, block_name: StaticBlockName) -> Option<BlockTypeHandle> {
        self.inner.blocks().get_by_name(block_name.0)
    }

    pub fn get_sound_id(&self, sound_name: &str) -> Option<SoundKey> {
        self.inner.media().get_sound_by_name(sound_name)
    }
    pub fn register_sound_from_memory(
        &mut self,
        sound_name: &str,
        data: &[u8],
    ) -> Result<SoundKey> {
        self.inner
            .media_mut()
            .register_from_memory(sound_name, data)?;
        self.inner
            .media_mut()
            .register_file_for_sampled_audio(sound_name)
    }

    /// Registers a simple item that cannot be placed, doesn't have a block automatically generated for it, and is not a tool
    /// The item can be stacked in the inventory up to 256, but has no other behaviors.
    ///
    /// If used as a tool, it will behave the same as if nothing were held in the hand.
    ///
    /// TODO: Create a proper builder abstraction similar to BlockBuilder
    pub fn register_basic_item(
        &mut self,
        short_name: impl Into<ItemName>,
        display_name: impl Into<String>,
        texture: impl TextureName,
        groups: Vec<String>,
        sort_key: impl Into<String>,
    ) -> Result<&Item> {
        let short_name = short_name.into().0.to_string();
        self.inner
            .items_mut()
            .register_item(Item::default_with_proto(ItemDef {
                short_name: short_name.clone(),
                display_name: display_name.into(),
                appearance: Some(Appearance::InventoryTexture(texture.name().to_string())),
                groups,
                interaction_rules: default_item_interaction_rules(),
                quantity_type: Some(
                    perovskite_core::protocol::items::item_def::QuantityType::Stack(256),
                ),
                sort_key: sort_key.into(),
            }))?;
        self.inner
            .items()
            .get_item(&short_name)
            .context("Item just registered, but missing. This shouldn't happen.")
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
        tex_name: impl Into<OwnedTextureName>,
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
        tex_name: impl Into<OwnedTextureName>,
        data: &[u8],
    ) -> Result<()> {
        self.inner
            .media_mut()
            .register_from_memory(&tex_name.into().0, data)
    }

    pub fn data_dir(&self) -> &PathBuf {
        self.inner.data_dir()
    }

    /// Returns an extension that holds state for additional functionality or APIs
    /// for a specific plugin (e.g. default_game providing ore generation to other plugins that want
    /// to generate ores) without the core GameBuilder needing to be aware of it *a priori*.
    pub fn builder_extension_mut<T: GameBuilderExtension + Any + Default + 'static>(
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
/// file_name is looked up relative to the current file (see [include_bytes]).
///
/// This macro takes the following parameters:
/// * Mutable reference to [GameBuilder]
/// * Texture name ([OwnedTextureName] object)
/// * File name (string literal)
#[macro_export]
macro_rules! include_texture_bytes {
    ($game_builder:expr, $tex_name:expr, $file_name:literal) => {
        $game_builder.register_texture_bytes($tex_name, include_bytes!($file_name))
    };
}
use crate::default_game::recipes::RecipeSlot;
pub use include_texture_bytes;
use perovskite_core::constants::GENERATED_TEXTURE_CATEGORY_SOLID_FROM_CSS;
use perovskite_core::protocol::items::item_def::Appearance;
use perovskite_server::game_state::blocks::FastBlockName;
use perovskite_server::media::SoundKey;

pub const DEFAULT_FOOTSTEP_SOUND_NAME: &str = "default:footstep.wav";
pub const SNOW_FOOTSTEP_SOUND_NAME: &str = "default:footstep_snow.wav";
pub const GRASS_FOOTSTEP_SOUND_NAME: &str = "default:footstep_grass.wav";
