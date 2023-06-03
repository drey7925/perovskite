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

use crate::game_builder::GameBuilder;

use anyhow::Result;

/// Blocks defined in the default game.
pub mod basic_blocks;

/// Common block groups that are defined in the default game and integrate with its tools
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
        args: &crate::game_builder::server_api::ServerArgs,
    ) -> Result<DefaultGameBuilder> {
        Self::new_with_builtins(GameBuilder::from_args(args)?)
    }

    fn new_with_builtins(
        mut builder: GameBuilder,
    ) -> std::result::Result<DefaultGameBuilder, anyhow::Error> {
        register_defaults(&mut builder)?;
        Ok(DefaultGameBuilder { inner: builder })
    }

    /// Adds an ore to the mapgen.
    ///
    /// **API skeleton, unimplemented, parameters TBD**
    /// Will be made pub when finalized and implemented
    fn register_ore(_ore_block: &str) {
        unimplemented!()
    }

    pub fn build_and_run(mut self) -> Result<()> {
        self.game_builder()
            .inner
            .set_mapgen(|blocks, seed| mapgen::build_mapgen(blocks, seed));
        self.inner.run_game_server()
    }
}

fn register_defaults(game_builder: &mut GameBuilder) -> Result<()> {
    basic_blocks::register_basic_blocks(game_builder)?;
    Ok(())
}
