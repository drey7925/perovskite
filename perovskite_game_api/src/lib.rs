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
//
// SPDX-License-Identifier: Apache-2.0

//! This crate contains APIs for defining game behaviors/content in Perovskite
//!
//! Some APIs are re-exported directly from perovskite_server and are subject to change;
//! they are available with the `unstable_api` crate feature.

/// Contains utilities for defining types of blocks in the world, as well as
/// items that simply correspond to a block stored in inventory.
pub mod blocks;
/// Common constant values useful to game content.
pub use perovskite_core::constants;

/// Provides functionality to build and start a game and server.
pub mod game_builder;
/// Contains utilities for defining items.
pub mod items;

pub use perovskite_core::coordinates::{BlockCoordinate, ChunkCoordinate, ChunkOffset};

use crate::default_game::DefaultGameBuilder;
use crate::game_builder::GameBuilder;

/// Provides a default set of game content centered around exploring a natural
/// procedurally-generated world, collecting resources through mining, converting
/// resources to useful materials, building structures out of them, etc.
///
/// The set of blocks and game features in the default game will grow as perovskite is
/// developed.
///
/// The default game will provide a map generator (WIP).
#[cfg(feature = "default_game")]
pub mod default_game;

/// Provides digital circuits that can be used in the default game.
#[cfg(feature = "circuits")]
pub mod circuits;

#[cfg(feature = "farming")]
pub mod farming;

#[cfg(feature = "discord")]
pub mod discord;

/// Carts on rails
#[cfg(feature = "carts")]
pub mod carts;

/// Animals
#[cfg(feature = "animals")]
pub mod animals;

/// Provides colors that can be used in the default game, as well as
/// a unified set of items representing dyes in different colors.
/// Also provides functionality to colorize textures automatically,
/// allowing one base texture to be used for multiple blocks in different colors.
pub mod colors;

#[macro_export]
#[cfg(doc)]
macro_rules! maybe_export {
    {
        $(
        $(#[$outer:meta])*
        fn $name:ident $params: tt
            $(-> $rtype:ty)? $body:block
        )+
    } => {
        #[cfg(feature="unstable_api")]
        $($(#[$outer])+)*
        $(pub(crate) fn $name $params $(-> $rtype)? $body)+
    };
    (
        $(#[$outer:meta])* use $p:path as $i:ident) => {
        $(#[$outer])*
        #[cfg(feature="unstable_api")] pub use $p as $i;
    }
}

#[macro_export]
#[cfg(all(not(feature = "unstable_api"), not(doc)))]
macro_rules! maybe_export {
    {
        $(
        $(#[$outer:meta])*
        fn $name:ident $params: tt
            $(-> $rtype:ty)? $body:block
        )+
    } => {
        $($(#[$outer])+
        pub fn $name $params $(-> $rtype)? $body)+
    };
    (
        $(#[$outer:meta])* use $p:path as $i:ident) => {
        $(#[$outer])*
        pub use $p as $i;
    }
}

#[macro_export]
#[cfg(all(feature = "unstable_api", not(doc)))]
macro_rules! maybe_export {
    {
        $(
        $(#[$outer:meta])*
        fn $name:ident $params: tt
            $(-> $rtype:ty)? $body:block
        )+
    } => {
        $($(#[$outer])+
        pub fn $name $params $(-> $rtype)? $body)+
    };
    (
        $(#[$outer:meta])* use $p:path as $i:ident) => {
        $(#[$outer])*
        pub use $p as $i;
    }
}

/// Marker that a struct may be extended in the future
///
/// This cannot be constructed except with Default::default
pub struct NonExhaustive(pub(crate) ());

#[cfg(any(test, feature = "test-support", doctest))]
pub mod test_support;

pub fn configure_default_game(game: &mut GameBuilder) -> anyhow::Result<()> {
    game.initialize_default_game()?;
    colors::register_dyes(game)?;

    #[cfg(feature = "circuits")]
    {
        circuits::register_circuits(game)?;
    }
    #[cfg(feature = "discord")]
    {
        discord::connect(game)?;
    }
    #[cfg(feature = "carts")]
    {
        carts::register_carts(game)?;
    }
    #[cfg(feature = "animals")]
    {
        animals::register_duck(game)?;
    }
    #[cfg(feature = "farming")]
    {
        farming::initialize_farming(game)?;
    }

    Ok(())
}
