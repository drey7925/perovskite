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

use std::path::Path;

use cuberef_core::{
    constants::{blocks::AIR, textures::FALLBACK_UNKNOWN_TEXTURE},
    protocol::{
        blocks::{
            block_type_def::{PhysicsInfo, RenderInfo},
            BlockTypeDef, Empty,
        },
        render::TextureReference,
    },
};
use cuberef_server::{
    game_state::blocks::{BlockType, BlockTypeHandle},
    server::ServerBuilder,
};

use anyhow::Result;

/// Type-safe newtype wrapper for a texture name
pub struct Tex(pub &'static str);

impl From<Tex> for TextureReference {
    fn from(value: Tex) -> Self {
        TextureReference {
            texture_name: value.0.to_string(),
        }
    }
}

/// Type-safe newtype wrapper for a block name
pub struct Block(pub &'static str);

/// Unstable re-export of the raw gameserver API. This API is subject to
/// breaking changes that do not follow semver, before 1.0
#[cfg(feature = "unstable_api")]
pub use cuberef_server::server as server_api;

use crate::blocks::BlockBuilder;

/// Stable API for building and configuring a game.
///
/// API stability note: *Before 1.0.0*, it is possible that functions returning
/// `()` or `Result<()>` may be changed to return something other than the empty
/// unit tuple.
pub struct GameBuilder {
    pub(crate) inner: ServerBuilder,
    pub(crate) air_block: BlockTypeHandle,
}
impl GameBuilder {
    /// Creates a new game builder using server configuration from the
    /// command line. If argument parsing fails, usage info is printed to
    /// the terminal and the process exits.
    pub fn from_cmdline() -> Result<GameBuilder> {
        Self::new_with_builtins(ServerBuilder::from_cmdline()?)
    }
    /// Creates a new game builder with custom server configuration.
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
    pub fn into_server_builder(self) -> server_api::ServerBuilder {
        self.inner
    }
    /// Run the game server
    pub fn run_game_server(self) -> Result<()> {
        self.inner.build()?.serve()
    }

    // Instantiate some builtin content
    fn new_with_builtins(mut inner: ServerBuilder) -> Result<GameBuilder> {
        inner.media().register_from_memory(
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
            groups: vec![],
        };
        let air_block = inner.blocks().register_block(air_block)?;
        Ok(GameBuilder { inner, air_block })
    }
    /// Registers a block and its corresponding item in the game.
    pub fn add_block(&mut self, block_builder: BlockBuilder) -> Result<()> {
        block_builder.build_and_deploy_into(self)
    }

    /// Adds a texture to the game by reading from a file.
    ///
    /// tex_name must be unique across all textures; an error will be returned
    /// if it is a duplicate
    pub fn register_texture_file(
        &mut self,
        tex_name: Tex,
        file_path: impl AsRef<Path>,
    ) -> Result<()> {
        self.inner.media().register_from_file(tex_name.0, file_path)
    }

    /// Adds a texture to the game with data passed as bytes.
    /// tex_name must be unique across all textures; an error will be returned
    /// if it is a duplicate
    pub fn register_texture_bytes(&mut self, tex_name: Tex, data: &[u8]) -> Result<()> {
        self.inner.media().register_from_memory(tex_name.0, data)
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
