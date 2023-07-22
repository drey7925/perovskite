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

use std::collections::HashSet;

use anyhow::Result;
use cuberef_core::{
    constants::{
        block_groups::DEFAULT_SOLID, items::default_item_interaction_rules,
        textures::FALLBACK_UNKNOWN_TEXTURE,
    },
    coordinates::BlockCoordinate,
    protocol::{
        self,
        blocks::{
            block_type_def::{PhysicsInfo, RenderInfo},
            BlockTypeDef, CubeRenderInfo, CubeRenderMode, Empty,
        },
        items as items_proto,
        items::{item_def::QuantityType, ItemDef},
        render::TextureReference,
    },
};
/// Unstable re-export of the raw blocks API. This API is subject to
/// breaking changes that do not follow semver, before 1.0
#[cfg(feature = "unstable_api")]
pub use cuberef_server::game_state::blocks as server_api;

use cuberef_server::game_state::{
    blocks::{BlockType, ExtendedData, InlineHandler, BlockTypeHandle},
    event::HandlerContext,
    game_map::CasOutcome,
    items::{Item, ItemStack},
};

use crate::{
    game_builder::{Block, GameBuilder},
    maybe_export,
};

/// The item obtained when the block is dug.
enum DroppedItem {
    None,
    Fixed(String, u32),
    Dynamic(Box<dyn Fn() -> (String, u32) + Sync + Send>),
}
impl DroppedItem {
    fn build_dig_handler(self, game_builder: &GameBuilder) -> Box<InlineHandler> {
        let air_block = game_builder.air_block;
        match self {
            DroppedItem::None => Box::new(move |_, target_block, _, _| {
                *target_block = air_block;
                Ok(vec![])
            }),
            DroppedItem::Fixed(item, count) => Box::new(move |_, target_block, _, _| {
                *target_block = air_block;
                Ok(vec![ItemStack {
                    proto: protocol::items::ItemStack {
                        item_name: item.clone(),
                        quantity: count,
                        current_wear: 1,
                        quantity_type: Some(items_proto::item_stack::QuantityType::Stack(256))
                    },
                }])
            }),
            DroppedItem::Dynamic(closure) => Box::new(move |_, target_block, _, _| {
                let (item, count) = closure();
                *target_block = air_block;
                Ok(vec![ItemStack {
                    proto: protocol::items::ItemStack {
                        item_name: item,
                        quantity: count,
                        current_wear: 1,
                        quantity_type: Some(items_proto::item_stack::QuantityType::Stack(256))
                    },
                }])
            }),
        }
    }
}
#[cfg(feature = "unstable_api")]
pub type ExtendedDataInitializer = Box<
    dyn Fn(
            HandlerContext,
            BlockCoordinate,
            BlockCoordinate,
            &ItemStack,
        ) -> Result<Option<ExtendedData>>
        + Send
        + Sync,
>;
#[cfg(not(feature = "unstable_api"))]
pub(crate) type ExtendedDataInitializer = Box<
    dyn Fn(
            HandlerContext,
            BlockCoordinate,
            BlockCoordinate,
            &ItemStack,
        ) -> Result<Option<ExtendedData>>
        + Send
        + Sync,
>;

#[cfg(feature = "unstable_api")]
/// Opaque wrapper around BlockTypeHandle. Enabling the `unstable_api` feature will expose the underlying block type handle.
pub struct BlockTypeHandleWrapper(pub BlockTypeHandle);
#[cfg(not(feature = "unstable_api"))]
/// Opaque wrapper around BlockTypeHandle. Enabling the `unstable_api` feature will expose the underlying block type handle.
pub struct BlockTypeHandleWrapper(pub(crate) BlockTypeHandle);


/// Builder for simple blocks.
/// Note that there are behaviors that this builder cannot express, but
/// [server_api::BlockType] (when used directly) can.
pub struct BlockBuilder {
    block_name: String,
    block_groups: HashSet<String>,
    item: Item,
    block_render_info: CubeRenderInfo,
    dropped_item: DroppedItem,
    // Temporarily exposed for water until the API is stabilized.
    pub(crate) physics_info: PhysicsInfo,
    modifier: Option<Box<dyn FnOnce(&mut BlockType)>>,
    /// Same parameters as [cuberef_server::game_state::items::PlaceHandler]
    extended_data_initializer: Option<ExtendedDataInitializer>,
}
impl BlockBuilder {
    /// Create a new block builder that will build a block and a corresponding inventory
    /// item for it.
    pub fn new(name: Block) -> BlockBuilder {
        let name = name.0;
        let item = Item {
            proto: ItemDef {
                short_name: name.into(),
                display_name: name.into(),
                inventory_texture: make_texture_ref(FALLBACK_UNKNOWN_TEXTURE.to_string()),
                groups: vec![],
                interaction_rules: default_item_interaction_rules(),
                quantity_type: Some(QuantityType::Stack(256)),
            },
            dig_handler: None,
            tap_handler: None,
            place_handler: Some(Box::new(|_, _, _, _| {
                panic!("Incomplete place_handler");
            })),
        };

        BlockBuilder {
            block_name: name.into(),
            block_groups: HashSet::from_iter([DEFAULT_SOLID.to_string()].into_iter()),
            item,
            block_render_info: CubeRenderInfo {
                tex_left: make_texture_ref(FALLBACK_UNKNOWN_TEXTURE.to_string()),
                tex_right: make_texture_ref(FALLBACK_UNKNOWN_TEXTURE.to_string()),
                tex_top: make_texture_ref(FALLBACK_UNKNOWN_TEXTURE.to_string()),
                tex_bottom: make_texture_ref(FALLBACK_UNKNOWN_TEXTURE.to_string()),
                tex_front: make_texture_ref(FALLBACK_UNKNOWN_TEXTURE.to_string()),
                tex_back: make_texture_ref(FALLBACK_UNKNOWN_TEXTURE.to_string()),
                // todo autodetect this
                render_mode: CubeRenderMode::SolidOpaque.into(),
            },
            dropped_item: DroppedItem::Fixed(name.into(), 1),
            physics_info: PhysicsInfo::Solid(Empty {}),
            modifier: None,
            extended_data_initializer: None,
        }
    }
    /// Sets the item which will be given to a player that digs this block.
    ///
    /// By default, the block will drop the corresponding item with the same name
    /// (e.g. block `foo:dirt` will drop item `foo:dirt`).
    ///
    /// Note that even if a block (e.g. `foo:iron_ore`) drops a different material
    /// (`foo:iron_ore_chunk`), an item `foo:iron_ore` will be registered. That item
    /// may be unobtainable without special game admin abilities, but if obtained
    /// it will still place the indicated block.
    pub fn set_dropped_item(mut self, item_name: &str, count: u32) -> Self {
        self.dropped_item = DroppedItem::Fixed(item_name.into(), count);
        self
    }
    /// Sets a closure that indicates what item a player will get when digging this
    /// block. Examples may include randomized drop rates.
    ///
    /// This closure takes no arguments. In the future, additional setters may be
    /// added that take closures with parameters to provide more details about the
    /// block being dug; this will be done in a non-API-breaking way.
    pub fn set_dropped_item_closure(
        mut self,
        closure: impl Fn() -> (String, u32) + Send + Sync + 'static,
    ) -> Self {
        self.dropped_item = DroppedItem::Dynamic(Box::new(closure));
        self
    }
    /// Sets that this block should drop nothing when dug
    pub fn set_no_drops(mut self) -> Self {
        self.dropped_item = DroppedItem::None;
        self
    }
    /// Sets the texture for all six faces of this block as well as the item, all to the same value
    pub fn set_texture_all<T>(mut self, texture: T) -> Self
    where
        T: Into<TextureReference>,
    {
        let tex = Some(texture.into());
        self.block_render_info.tex_left = tex.clone();
        self.block_render_info.tex_right = tex.clone();
        self.block_render_info.tex_top = tex.clone();
        self.block_render_info.tex_bottom = tex.clone();
        self.block_render_info.tex_front = tex.clone();
        self.block_render_info.tex_back = tex.clone();
        self.item.proto.inventory_texture = tex;
        self
    }
    /// Sets the texture for all six faces of this block as well as the inventory item, one by one
    pub fn set_individual_textures<T>(
        mut self,
        left: T,
        right: T,
        top: T,
        bottom: T,
        front: T,
        back: T,
        inventory: T,
    ) -> Self
    where
        T: Into<TextureReference>,
    {
        self.block_render_info.tex_left = Some(left.into());
        self.block_render_info.tex_right = Some(right.into());
        self.block_render_info.tex_top = Some(top.into());
        self.block_render_info.tex_bottom = Some(bottom.into());
        self.block_render_info.tex_front = Some(front.into());
        self.block_render_info.tex_back = Some(back.into());
        self.item.proto.inventory_texture = Some(inventory.into());
        self
    }
    /// Indicates that the textures may have transparent pixels. This does not support translucency.
    ///
    /// Stability note: It's possible that we may start autodetecting transparent pixels in texture files.
    /// If that happens, this method will become a deprecated no-op.
    pub fn set_needs_transparency(mut self) -> Self {
        self.block_render_info
            .set_render_mode(CubeRenderMode::Transparent);
        self
    }

    /// Indicates that the textures may have translucent pixels. The behavior of this is still TBD.
    ///
    /// Stability note: The signature of this method will likely remain the same, but the render behavior may change.
    /// Additional controls might be added in the future (as separate methods)
    pub fn set_needs_translucency(mut self) -> Self {
        self.block_render_info
            .set_render_mode(CubeRenderMode::Translucent);
        self
    }

    /// Adds a group to the list of groups for this block.
    /// These can affect diggability, dig speed, and other behavior in
    /// the map.
    ///
    /// See [crate::constants::block_groups] for useful values.
    pub fn add_block_group(mut self, group: &str) -> Self {
        self.block_groups.insert(group.into());
        self
    }

    /// Adds a group to the list of groups for the item corresponding to this block.
    /// These can affect crafting with this block (crafting API is TBD)
    ///
    /// See [crate::constants::block_groups] for useful values.
    pub fn add_item_group(mut self, group: &str) -> Self {
        self.item.proto.groups.push(group.into());
        self
    }
    /// Set the display name visible when hovering in the inventory.
    pub fn set_inventory_display_name(mut self, display_name: &str) -> Self {
        self.item.proto.display_name = display_name.into();
        self
    }

    maybe_export!(
        /// Run arbitrary changes on the block definition just before it's registered
        fn set_modifier(mut self, modifier: Box<dyn FnOnce(&mut BlockType)>) -> Self {
            self.modifier = Some(modifier);
            self
        }
    );
    maybe_export!(
        /// Sets a function to generate extended data just before the block is placed
        /// This will be run even if the block is not placed due to a conflicting material
        fn set_extended_data_initializer(
            mut self,
            extended_data_initializer: ExtendedDataInitializer,
        ) -> Self {
            self.extended_data_initializer = Some(extended_data_initializer);
            self
        }
    );

    pub(crate) fn build_and_deploy_into(mut self, game_builder: &mut GameBuilder) -> Result<BlockTypeHandleWrapper> {
        let mut block = BlockType::default();
        block.client_info = BlockTypeDef {
            id: 0,
            short_name: self.block_name.clone(),
            base_dig_time: 1.0,
            groups: self.block_groups.into_iter().collect(),
            render_info: Some(RenderInfo::Cube(self.block_render_info)),
            physics_info: Some(self.physics_info),
        };
        block.dig_handler_inline = Some(self.dropped_item.build_dig_handler(game_builder));
        if let Some(modifier) = self.modifier {
            (modifier)(&mut block);
        }
        let block_handle = game_builder.inner.blocks_mut().register_block(block)?;

        let mut item = self.item;
        let air_block = game_builder.air_block;
        let extended_data_initializer = self.extended_data_initializer.take();
        item.place_handler = Some(Box::new(move |ctx, coord, anchor, stack| {
            if stack.proto().quantity == 0 {
                return Ok(None);
            }
            let extended_data = match &extended_data_initializer {
                Some(x) => (x)(ctx.clone(), coord, anchor, stack)?,
                None => None,
            };

            match ctx
                .game_map()
                // TODO be more flexible with placement (e.g. water)
                .compare_and_set_block(coord, air_block, block_handle, extended_data, false)?
                .0
            {
                CasOutcome::Match => Ok(stack.decrement()),
                CasOutcome::Mismatch => Ok(Some(stack.clone())),
            }
        }));
        game_builder.inner.items_mut().register_item(item)?;
        Ok(BlockTypeHandleWrapper(block_handle))
    }
}

fn make_texture_ref(tex_name: String) -> Option<TextureReference> {
    Some(TextureReference {
        texture_name: tex_name,
    })
}
