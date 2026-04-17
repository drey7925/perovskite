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

/// The size of a chunk in each dimension. This is a fundamental constant for the codebase, and cannot be changed
/// without recompiling both clients and servers, as well as migrating all existing world data (or discarding it).
///
/// Furthermore, much performance tuning has happened with this value set to 16; changing it will likely shift bottlenecks
/// around.
///
/// Test any changes thoroughly!
///
/// TODO: The raytracer has these hardcoded, and they'll need to be moved into specialization constants.
pub const CHUNK_SIZE: usize = 32;
// https://github.com/rust-lang/rfcs/pull/1062 :(
pub const CHUNK_SIZE_U8: u8 = CHUNK_SIZE as u8;
pub const CHUNK_SIZE_I8: i8 = CHUNK_SIZE as i8;
pub const CHUNK_SIZE_I32: i32 = CHUNK_SIZE as i32;
pub const CHUNK_SIZE_F64: f64 = CHUNK_SIZE as f64;
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
/// The number of bits to shift by to get the chunk coordinate from a block coordinate.
pub const CHUNK_BITS: i32 = CHUNK_SIZE.ilog2() as i32;
pub const CHUNK_MASK: i32 = CHUNK_SIZE as i32 - 1;
static_assertions::const_assert_eq!(CHUNK_SIZE, 1 << CHUNK_BITS);
static_assertions::const_assert_eq!(CHUNK_MASK as usize & CHUNK_SIZE, 0);

/// The size of the extended block array for a chunk, including the 1-block border around the chunk.
/// This is currently used only within the client's implementation details, but it will become a part of server-
/// and API-level abstractions in the future.
pub const PADDED_CHUNK_SIZE: usize = CHUNK_SIZE + 2;
pub const PADDED_CHUNK_OFFSET: i32 = 1;
pub const PADDED_CHUNK_VOLUME: usize = PADDED_CHUNK_SIZE * PADDED_CHUNK_SIZE * PADDED_CHUNK_SIZE;

/// The size of the extended block array for a chunk, including the 16-block border around the chunk.
///
/// An extended chunk is used for light propagation calculations, and is large enough to hold
/// the light values for all blocks in the chunk, as well as the light values for the
/// 16-block border around the chunk; the 16-block border is necessary because light can travel up to 16 blocks in a straight line.
///
/// If the chunk size is increased in the future, it is not clear whether this will be CHUNK_SIZE * 3,
/// or CHUNK_SIZE + 32, but EXTENDED_CHUNK_OFFSET will be updated accordingly to reflect that decision.
pub const EXTENDED_CHUNK_SIZE: usize = CHUNK_SIZE + 2 * (EXTENDED_CHUNK_OFFSET as usize);
pub const EXTENDED_CHUNK_SIZE_I32: i32 = EXTENDED_CHUNK_SIZE as i32;
/// An extended chunk size provides 16 extra blocks on all sides of the chunk, so the center chunk begins
/// at an offset of 16 blocks from the start of the array.
pub const EXTENDED_CHUNK_OFFSET: i32 = 16;
pub const EXTENDED_CHUNK_VOLUME: usize =
    EXTENDED_CHUNK_SIZE * EXTENDED_CHUNK_SIZE * EXTENDED_CHUNK_SIZE;

/// When building an extended chunk, these ranges describe the three chunks that make up the extended chunk.
///
/// Each tuple consists of:
///   * The offset in chunk space (-1, 0, 1)
///   * The range of chunk coordinates that should be read from the chunk
///   * The offset in the extended chunk to write the data to
pub const EXTENDED_OVERLAP_RANGES: [(i32, std::ops::Range<i32>, i32); 3] = [
    (
        -1i32,
        (CHUNK_SIZE_I32 - EXTENDED_CHUNK_OFFSET)..CHUNK_SIZE_I32,
        -CHUNK_SIZE_I32,
    ),
    (0, 0..CHUNK_SIZE_I32, 0),
    (1, 0..EXTENDED_CHUNK_OFFSET, CHUNK_SIZE_I32),
];
// Light travels for up to 16 blocks, so the extended chunk must provide at least 16 blocks worth
// of data outside the core chunk.
static_assertions::const_assert!(EXTENDED_CHUNK_OFFSET >= 16);

/// Names for well-known block groups. By using these, different plugins
/// can interoperate effectively.
pub mod block_groups {
    /// Block group for all solid blocks, e.g. dirt, glass, sand, furniture, furnaces
    /// Most tools won't select/point at a block unless it has this group.
    pub const DEFAULT_SOLID: &str = "default:solid";
    /// Block group for all liquid/fluid blocks, e.g. water, lava
    pub const DEFAULT_LIQUID: &str = "default:liquid";
    /// Block group for all gas-like blocks (e.g. air, clouds)
    pub const DEFAULT_GAS: &str = "default:gas";

    /// Blocks that can be replaced by a conflicting block placement (e.g. water, air, very light
    /// plants).
    ///
    /// Note that this is a *gameplay* property,
    /// not a technical one. A trivially-replaceable block is, subjectively, one that is easily replaced
    /// when another block is placed. For example, air and water are trivially replaceable, because placing
    /// a block into them destroys them without giving the player an item. Light grasses and over very light foliage
    /// may also fall into this, depending on gameplay design decisions.
    ///
    /// However, there *is* a technical ramification - if a block is trivially replaceable, then there are more ways in
    /// which it can be destroyed without running a dig handler. If a block has gameplay effects that must happen on
    /// destruction then it probably shouldn't fall into this category.
    ///
    /// Plugins encountering a trivially replaceable block may either treat it as air and replace it without recovering
    /// dropped items, or may run the dig handler and respect its actions, etc.
    pub const TRIVIALLY_REPLACEABLE: &str = "default:trivially_replaceable";

    /// Blocks that cannot be dug by hand or using a generic non-tool item
    pub const TOOL_REQUIRED: &str = "default:tool_required";
    /// Blocks that cannot be dug under any circumstances (other than admon intervention)
    /// Tools available to normal users should specify a dig_behavior of None for this block.
    pub const NOT_DIGGABLE: &str = "default:not_diggable";
    /// Can be instantly dug.
    pub const INSTANT_DIG: &str = "default:instant_dig";
}

pub mod blocks {
    pub const AIR: &str = "builtin:air";
}

pub mod item_groups {
    /// Item group for all items that have a wear bar corresponding to physical wear (e.g. pickaxes)
    /// as opposed to e.g. electrical charge, which could be used in some game plugin.
    ///
    /// The default game doesn't do anything with this group. This is only used as a signal for
    /// other plugins to make this distiction.
    pub const TOOL_WEAR: &str = "default:tool_wear";

    /// Items that should not be shown in the creative inventory
    /// The default game's creative inventory respects this group, and other plugins should
    /// do the same, unless they intentionally want to expose internal items for development/testing/curiosity
    pub const HIDDEN_FROM_CREATIVE: &str = "default:hidden_from_creative";
}

pub mod items {
    use crate::protocol::items::{interaction_rule::DigBehavior, Empty, InteractionRule};

    use super::block_groups::*;
    /// Get the default interaction rules for generic items that aren't some kind of special tool
    /// e.g. stacks of random items/blocks, no item held in the hand, etc
    pub fn default_item_interaction_rules() -> Vec<InteractionRule> {
        vec![
            InteractionRule {
                block_group: vec![DEFAULT_SOLID.to_string(), NOT_DIGGABLE.to_string()],
                dig_behavior: Some(DigBehavior::Undiggable(Empty {})),
                tool_wear: 0,
            },
            InteractionRule {
                block_group: vec![DEFAULT_SOLID.to_string(), TOOL_REQUIRED.to_string()],
                dig_behavior: Some(DigBehavior::Undiggable(Empty {})),
                tool_wear: 0,
            },
            InteractionRule {
                block_group: vec![DEFAULT_SOLID.to_string(), INSTANT_DIG.to_string()],
                dig_behavior: Some(DigBehavior::InstantDigOneshot(Empty {})),
                tool_wear: 0,
            },
            InteractionRule {
                block_group: vec![DEFAULT_SOLID.to_string()],
                dig_behavior: Some(DigBehavior::ConstantTime(1.0)),
                tool_wear: 0,
            },
        ]
    }
}

pub mod textures {
    /// A simple fallback texture.
    pub const FALLBACK_UNKNOWN_TEXTURE: &str = "builtin:unknown";
}

/// Built-in permissions that can be granted to a player.
///
/// This is not an exhaustive list of permissions, and plugins may define
/// new permissions of their own. However, plugins may want to use these
/// for consistency where possible.
pub mod permissions {
    /// The player can fly
    pub const FLY: &str = "default:fly";
    /// The player can sprint
    pub const FAST_MOVE: &str = "default:fast_move";
    /// The player can pass through solid obstacles
    pub const NOCLIP: &str = "default:noclip";
    /// The player can use /give and /giveme commands
    pub const GIVE: &str = "default:give";
    /// The player can grant permissions to themselves/other players
    /// Note that to avoid lock-out, a player cannot revoke this permission
    /// from themselves
    pub const GRANT: &str = "default:grant";
    /// The player can bypass permission checks on inventories (e.g. locked chests)
    pub const BYPASS_INVENTORY_CHECKS: &str = "default:bypass_inventory_checks";
    /// The player has access to a creative inventory
    pub const CREATIVE: &str = "default:creative";
    /// The player can dig and place blocks
    pub const DIG_PLACE: &str = "default:dig_place";
    /// The player can tap blocks and interact (the F key by default). While this can modify the world, it does so in
    /// more limited ways (e.g. tapping a block might cause some in-world automation designed by other players
    /// to run, but generally won't modify the world substantially in the way mining and placing would)
    ///
    /// This controls whether the user can click buttons in server-defined popups as well.
    pub const TAP_INTERACT: &str = "default:tap_interact";
    /// The player may actually log in.
    /// If this is not granted, the player will be immediately disconnected after authentication.
    pub const LOG_IN: &str = "default:log_in";
    /// Allows the user to mess with the world state (e.g. set time, weather, player locations, etc)
    pub const WORLD_STATE: &str = "default:set_world_state";
    /// Allows the player to interact with inventories, popups, etc
    pub const INVENTORY: &str = "default:inventory";
    /// Allows the player to send chat messages to other players. This does NOT affect slash commands
    /// which should check permissions on their own. (e.g. some slash commands may still be useful for players
    /// that are not permitted to chat yet)
    pub const CHAT: &str = "default:chat";
    /// Allows the user to receive metrics regarding server performance and other kinds of debugging
    pub const PERFORMANCE_METRICS: &str = "default:performance_metrics";

    /// The set of permissions that affect client behavior. Only these are sent to clients
    pub const CLIENT_RELEVANT_PERMISSIONS: &[&str] =
        &[FLY, FAST_MOVE, NOCLIP, DIG_PLACE, TAP_INTERACT, INVENTORY];

    pub const ALL_PERMISSIONS: [&str; 14] = [
        FLY,
        FAST_MOVE,
        NOCLIP,
        GIVE,
        GRANT,
        BYPASS_INVENTORY_CHECKS,
        CREATIVE,
        DIG_PLACE,
        TAP_INTERACT,
        LOG_IN,
        WORLD_STATE,
        INVENTORY,
        CHAT,
        PERFORMANCE_METRICS,
    ];

    /// Prefix for permissions that can be self-granted on request, used for
    /// /elevate. Intended to allow an admin to have permissions without them having
    /// an effect at all times.
    ///
    /// e.g. a player having `eligible:default:bypass_inventory_checks` would by default
    /// be subject to inventory checks, but could use /elevate default:bypass_inventory_checks
    /// (command TBD and to be implemented) to *temporarily* enable that permission
    ///
    /// This is implemented in perovskite_server and doesn't require any special handling
    /// in plugins when doing permission checks for actions. Once an eligible permission
    /// is activated with /elevate, that permission will show up when calling permission check functions
    /// on the player without the eligible: prefix.
    ///
    /// However, plugins may check eligible permissions for their own purposes if they wish to.
    pub const ELIGIBLE_PREFIX: &str = "eligible:";
}

/// Prefix for media that are generated on the fly by the client
///
/// Currently supported: `generated:solid_css:[A CSS COLOR]`, e.g.
///   * `generated:solid_css:rgb(0 255 0)` (with embedded spaces)
///   * `generated:solid_css:lime`
pub const GENERATED_TEXTURE_PREFIX: &str = "generated:";
pub const GENERATED_TEXTURE_CATEGORY_SOLID_FROM_CSS: &str = "generated:solid_css:";
