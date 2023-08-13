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

/// Names for well-known block groups. By using these, different plugins
/// can interoperate effectively.
pub mod block_groups {
    /// Block group for all solid blocks, e.g. dirt, glass, sand
    pub const DEFAULT_SOLID: &str = "default:solid";
    /// Block group for all liquid/fluid blocks, e.g. water, lava
    pub const DEFAULT_LIQUID: &str = "default:liquid";

    /// Blocks that cannot be dug by hand or using a generic non-tool item
    pub const TOOL_REQUIRED: &str = "default:tool_required";
    /// Blocks that cannot be dug under any circumstances (other than admon intervention)
    /// Tools available to normal users should specify a dig_behavior of None for this block.
    pub const NOT_DIGGABLE: &str = "default:not_diggable";
    /// Can be instantly dug
    pub const INSTANT_DIG: &str = "default:instant_dig";
}

pub mod blocks {
    pub const AIR: &str = "builtin:air";
}

pub mod item_groups {
    /// Item group for all items that have a wear bar corresponding to physical wear (e.g. pickaxes)
    /// as opposed to e.g. electrical charge, which could be used in some game plugin
    ///
    /// The default game doesn't do anything with this group.
    pub const TOOL_WEAR: &str = "default:tool_wear";

    /// Items that should not be shown in the creative inventory
    /// The default game's creative inventory respects this group, and other plugins should
    /// do the same, unless they intentionally want to expose internal items for development/testing/curiosity
    pub const HIDDEN_FROM_CREATIVE: &str = "default:hidden_from_creative";
}

pub mod items {
    use crate::protocol::items::{interaction_rule::DigBehavior, InteractionRule};

    use super::block_groups::*;
    /// Get the default interaction rules for generic items that aren't some kind of special tool
    /// e.g. stacks of random items/blocks, no item held in the hand, etc
    pub fn default_item_interaction_rules() -> Vec<InteractionRule> {
        vec![
            InteractionRule {
                block_group: vec![NOT_DIGGABLE.to_string()],
                dig_behavior: None,
                tool_wear: 0,
            },
            InteractionRule {
                block_group: vec![TOOL_REQUIRED.to_string()],
                dig_behavior: None,
                tool_wear: 0,
            },
            InteractionRule {
                block_group: vec![INSTANT_DIG.to_string()],
                dig_behavior: Some(DigBehavior::InstantDigOneshot(
                    crate::protocol::items::Empty {},
                )),
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
