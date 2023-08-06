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

use anyhow::{Context, Result};
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
            BlockTypeDef, CubeRenderInfo, CubeRenderMode, CubeVariantEffect, Empty,
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
    blocks::{BlockInteractionResult, BlockType, BlockTypeHandle, ExtendedData, InlineHandler},
    event::HandlerContext,
    game_map::CasOutcome,
    items::{InteractionRuleExt, Item, ItemStack},
};

use crate::{
    game_builder::{BlockName, GameBuilder, ItemName},
    maybe_export,
};

/// The item obtained when the block is dug.
/// TODO stabilize item stack and expose a vec<itemstack> option
enum DroppedItem {
    None,
    Fixed(String, u32),
    Dynamic(Box<dyn Fn() -> (ItemName, u32) + Sync + Send + 'static>),
}
impl DroppedItem {
    fn build_dig_handler_inner<F>(closure: F, game_builder: &GameBuilder) -> Box<InlineHandler>
    where
        F: Fn() -> Vec<ItemStack> + Sync + Send + 'static,
    {
        let air_block = game_builder.air_block;
        Box::new(move |ctx, target_block, ext_data, stack| {
            let block_type = ctx.block_types().get_block(&target_block)?.0;
            let rule = ctx
                .items()
                .from_stack(stack)
                .and_then(|item| item.get_interaction_rule(block_type));
            if rule.as_ref().and_then(|x| x.dig_time(block_type)).is_some() {
                *target_block = air_block;
                ext_data.clear();

                Ok(BlockInteractionResult {
                    item_stacks: closure(),
                    tool_wear: match rule {
                        Some(rule) => rule.tool_wear(block_type)?,
                        None => 0,
                    },
                })
            } else {
                Ok(Default::default())
            }
        })
    }

    fn build_dig_handler(self, game_builder: &GameBuilder) -> Box<InlineHandler> {
        match self {
            DroppedItem::None => Self::build_dig_handler_inner(|| vec![], game_builder),
            DroppedItem::Fixed(item, count) => Self::build_dig_handler_inner(
                move || {
                    vec![ItemStack {
                        proto: protocol::items::ItemStack {
                            item_name: item.clone(),
                            quantity: count,
                            current_wear: 1,
                            quantity_type: Some(items_proto::item_stack::QuantityType::Stack(256)),
                        },
                    }]
                },
                game_builder,
            ),
            DroppedItem::Dynamic(closure) => Self::build_dig_handler_inner(
                move || {
                    let (item, count) = closure();
                    vec![ItemStack {
                        proto: protocol::items::ItemStack {
                            item_name: item.0.to_string(),
                            quantity: count,
                            current_wear: 1,
                            quantity_type: Some(items_proto::item_stack::QuantityType::Stack(256)),
                        },
                    }]
                },
                game_builder,
            ),
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
impl From<BlockTypeHandle> for BlockTypeHandleWrapper {
    fn from(handle: BlockTypeHandle) -> Self {
        Self(handle)
    }
}

enum VariantEffect {
    None,
    RotateNesw,
}

/// Builder for simple blocks.
/// Note that there are behaviors that this builder cannot express, but
/// [server_api::BlockType] (when used directly) can.
#[must_use = "Builders do nothing unless used; set_foo will return a new builder."]
pub struct BlockBuilder {
    block_name: String,
    item: Item,
    dropped_item: DroppedItem,
    modifier: Option<Box<dyn FnOnce(&mut BlockType)>>,
    /// Same parameters as [cuberef_server::game_state::items::PlaceHandler]
    extended_data_initializer: Option<ExtendedDataInitializer>,
    // Exposed within the crate while not all APIs are complete
    pub(crate) client_info: BlockTypeDef,
    variant_effect: VariantEffect,
}
impl BlockBuilder {
    /// Create a new block builder that will build a block and a corresponding inventory
    /// item for it.
    pub fn new(name: BlockName) -> BlockBuilder {
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
                panic!("Incomplete place_handler; item registration was not completed properly");
            })),
        };

        BlockBuilder {
            block_name: name.into(),
            item,
            dropped_item: DroppedItem::Fixed(name.into(), 1),
            modifier: None,
            extended_data_initializer: None,
            client_info: BlockTypeDef {
                id: 0,
                short_name: name.into(),
                base_dig_time: 1.0,
                groups: vec![DEFAULT_SOLID.to_string()],
                wear_multiplier: 1.0,
                light_emission: 0,
                allow_light_propagation: false,
                render_info: Some(RenderInfo::Cube(CubeRenderInfo {
                    tex_left: make_texture_ref(FALLBACK_UNKNOWN_TEXTURE.to_string()),
                    tex_right: make_texture_ref(FALLBACK_UNKNOWN_TEXTURE.to_string()),
                    tex_top: make_texture_ref(FALLBACK_UNKNOWN_TEXTURE.to_string()),
                    tex_bottom: make_texture_ref(FALLBACK_UNKNOWN_TEXTURE.to_string()),
                    tex_front: make_texture_ref(FALLBACK_UNKNOWN_TEXTURE.to_string()),
                    tex_back: make_texture_ref(FALLBACK_UNKNOWN_TEXTURE.to_string()),
                    // todo autodetect this
                    render_mode: CubeRenderMode::SolidOpaque.into(),
                    variant_effect: CubeVariantEffect::None.into(),
                })),
                physics_info: Some(PhysicsInfo::Solid(Empty {})),
            },
            variant_effect: VariantEffect::None,
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
        closure: impl Fn() -> (ItemName, u32) + Send + Sync + 'static,
    ) -> Self {
        self.dropped_item = DroppedItem::Dynamic(Box::new(closure));
        self
    }
    /// Sets that this block should drop nothing when dug
    pub fn set_no_drops(mut self) -> Self {
        self.dropped_item = DroppedItem::None;
        self
    }

    /// Adds a group to the list of groups for this block.
    /// These can affect diggability, dig speed, and other behavior in
    /// the map.
    ///
    /// See [crate::constants::block_groups] for useful values.
    pub fn add_block_group(mut self, group: impl Into<String>) -> Self {
        let group = group.into();
        if !self.client_info.groups.contains(&group) {
            self.client_info.groups.push(group);
        }
        self
    }

    /// Adds a group to the list of groups for the item corresponding to this block.
    /// These can affect crafting with this block (crafting API is TBD)
    ///
    /// See [crate::constants::block_groups] for useful values.
    pub fn add_item_group(mut self, group: impl Into<String>) -> Self {
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
    /// Set the appearance of the block to that specified by the given builder
    pub fn set_cube_appearance(mut self, appearance: CubeAppearanceBuilder) -> Self {
        self.variant_effect = match appearance.render_info.variant_effect() {
            CubeVariantEffect::None => VariantEffect::None,
            CubeVariantEffect::RotateNesw => VariantEffect::RotateNesw,
        };
        self.client_info.render_info = Some(RenderInfo::Cube(appearance.render_info));
        self.client_info.light_emission = appearance.light_emission;
        self.client_info.allow_light_propagation = appearance.allow_light_propagation;
        self
    }
    /// Sets the texture shown for this block's item in the inventory.
    pub fn set_inventory_texture(mut self, texture: impl Into<TextureReference>) -> Self {
        self.item.proto.inventory_texture = Some(texture.into());
        self
    }
    /// Convenience method that sets this block to a simple appearance as a cube with the same texture on all faces,
    /// no transparency/translucency, no light propagation or emission, and no additional appearance settings
    /// (which may be added in the future)
    pub fn set_cube_single_texture(mut self, texture: impl Into<TextureReference>) -> Self {
        self.set_cube_appearance(CubeAppearanceBuilder::new().set_single_texture(texture))
    }

    pub(crate) fn build_and_deploy_into(
        mut self,
        game_builder: &mut GameBuilder,
    ) -> Result<BlockTypeHandleWrapper> {
        let mut block = BlockType::default();
        block.client_info = self.client_info;
        block.dig_handler_inline = Some(self.dropped_item.build_dig_handler(game_builder));
        if let Some(modifier) = self.modifier {
            (modifier)(&mut block);
        }
        let block_handle = game_builder.inner.blocks_mut().register_block(block)?;

        let mut item = self.item;
        let air_block = game_builder.air_block;
        let extended_data_initializer = self.extended_data_initializer.take();
        // todo factor out a default place handler and export it to users of the API
        item.place_handler = Some(Box::new(move |ctx, coord, anchor, stack| {
            if stack.proto().quantity == 0 {
                return Ok(None);
            }
            let extended_data = match &extended_data_initializer {
                Some(x) => (x)(ctx.clone(), coord, anchor, stack)?,
                None => None,
            };
            let variant = match self.variant_effect {
                VariantEffect::None => 0,
                VariantEffect::RotateNesw => ctx
                    .initiator()
                    .position()
                    .map(|pos| variants::rotate_nesw_azimuth_to_variant(pos.face_direction.0))
                    .unwrap_or(0),
            };
            match ctx
                .game_map()
                // TODO be more flexible with placement (e.g. water)
                .compare_and_set_block(
                    coord,
                    air_block,
                    block_handle.with_variant(variant)?,
                    extended_data,
                    false,
                )?
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

pub struct CubeAppearanceBuilder {
    render_info: CubeRenderInfo,
    light_emission: u32,
    allow_light_propagation: bool,
}
impl CubeAppearanceBuilder {
    pub fn new() -> Self {
        Self {
            render_info: CubeRenderInfo {
                tex_left: make_texture_ref(FALLBACK_UNKNOWN_TEXTURE.to_string()),
                tex_right: make_texture_ref(FALLBACK_UNKNOWN_TEXTURE.to_string()),
                tex_top: make_texture_ref(FALLBACK_UNKNOWN_TEXTURE.to_string()),
                tex_bottom: make_texture_ref(FALLBACK_UNKNOWN_TEXTURE.to_string()),
                tex_front: make_texture_ref(FALLBACK_UNKNOWN_TEXTURE.to_string()),
                tex_back: make_texture_ref(FALLBACK_UNKNOWN_TEXTURE.to_string()),
                render_mode: CubeRenderMode::SolidOpaque.into(),
                variant_effect: CubeVariantEffect::None.into(),
            },
            light_emission: 0,
            allow_light_propagation: false,
        }
    }

    /// Sets this block to render as a cube and applies the same texture for all six faces.
    pub fn set_single_texture<T>(self, texture: T) -> Self
    where
        T: Into<TextureReference>,
    {
        let tex = texture.into();
        self.set_individual_textures(
            tex.clone(),
            tex.clone(),
            tex.clone(),
            tex.clone(),
            tex.clone(),
            tex,
        )
    }

    
    /// Sets the light emission of this block, causing it to glow and illuminate other blocks.
    /// The meaningful range of light emission is 0-15.
    ///
    /// Moving one block causes light to fall off by one.
    pub fn set_light_emission(mut self, light_emission: u32) -> Self {
        assert!(light_emission < 16);
        self.light_emission = light_emission;
        self
    }

    /// Sets whether the block allows light to propagate through it.
    /// 
    /// This is separate from set_needs_transparency and/or set_needs_translucency; those affect
    /// how the block is rasterized on the GPU, but not how map lighting is calculated.
    pub fn set_allow_light_propagation(mut self, allow_light_propagation: bool) -> Self {
        self.allow_light_propagation = allow_light_propagation;
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
    ) -> Self
    where
        T: Into<TextureReference>,
    {
        self.render_info.tex_left = Some(left.into());
        self.render_info.tex_right = Some(right.into());
        self.render_info.tex_top = Some(top.into());
        self.render_info.tex_bottom = Some(bottom.into());
        self.render_info.tex_front = Some(front.into());
        self.render_info.tex_back = Some(back.into());

        self
    }

    /// Indicates that the textures may have transparent pixels. This does not support translucency.
    ///
    /// This does not cause in-game light to propagate through the block; use set_allow_light_propagation for that.
    /// 
    /// Stability note: It's possible that we may start autodetecting transparent pixels in texture files.
    /// If that happens, this method will become a deprecated no-op.
    pub fn set_needs_transparency(mut self) -> Self {
        self.render_info
            .set_render_mode(CubeRenderMode::Transparent);
        self
    }

    /// Indicates that the textures may have translucent pixels. The behavior of this is still TBD.
    /// 
    /// This does not cause in-game light to propagate through the block; use set_allow_light_propagation for that.
    ///
    /// Stability note: The signature of this method will likely remain the same, but the render behavior may change.
    /// Additional controls might be added in the future (as separate methods)
    pub fn set_needs_translucency(mut self) -> Self {
        self.render_info
            .set_render_mode(CubeRenderMode::Translucent);
        self
    }

    /// Makes the block able to point in the four lateral directions, rotating it when placed
    pub fn set_rotate_laterally(mut self) -> Self {
        self.render_info.variant_effect = CubeVariantEffect::RotateNesw.into();
        self
    }
}
fn make_texture_ref(tex_name: String) -> Option<TextureReference> {
    Some(TextureReference {
        texture_name: tex_name,
    })
}

/// Contains utilities for block variant schemes that are built into the game engine
pub mod variants {
    /// Given an azimuth angle of a player (in degrees), returns the variant that makes the block face the player
    /// when placed.
    pub fn rotate_nesw_azimuth_to_variant(azimuth: f64) -> u16 {
        let azimuth = azimuth.rem_euclid(360.);
        match azimuth {
            x if x < 45.0 => 0,
            x if x < 135.0 => 1,
            x if x < 225.0 => 2,
            x if x < 315.0 => 3,
            x if x < 360.0 => 0,
            _ => 0,
        }
    }
}
