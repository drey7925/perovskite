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

use std::time::Duration;

use anyhow::Result;
use itertools::Itertools;
use perovskite_core::{
    block_id::{special_block_defs::AIR_ID, BlockId},
    constants::{
        block_groups::{DEFAULT_GAS, DEFAULT_LIQUID, DEFAULT_SOLID, TRIVIALLY_REPLACEABLE},
        items::default_item_interaction_rules,
        textures::FALLBACK_UNKNOWN_TEXTURE,
    },
    coordinates::{BlockCoordinate, ChunkCoordinate, ChunkOffset},
    protocol::{
        self,
        blocks::{
            block_type_def::{PhysicsInfo, RenderInfo},
            AxisAlignedBox, AxisAlignedBoxRotation, AxisAlignedBoxes, BlockTypeDef, CubeRenderInfo,
            CubeRenderMode, CubeVariantEffect, Empty, PlantLikeRenderInfo,
        },
        items::{self as items_proto, item_def::QuantityType, ItemDef},
        render::TextureReference,
    },
};
/// Unstable re-export of the raw blocks API. This API is subject to
/// breaking changes that do not follow semver, before 1.0
#[cfg(feature = "unstable_api")]
pub use perovskite_server::game_state::blocks as server_api;

use perovskite_server::game_state::{
    blocks::{BlockInteractionResult, BlockType, BlockTypeHandle, ExtendedData, InlineHandler},
    event::HandlerContext,
    game_map::{
        BulkUpdateCallback, CasOutcome, ChunkNeighbors, MapChunk, TimerState,
        VerticalNeighborTimerCallback,
    },
    items::{HasInteractionRules, InteractionRuleExt, Item, ItemStack},
};

use crate::{
    game_builder::{BlockName, GameBuilder, ItemName, StaticItemName},
    maybe_export,
};

pub mod custom_geometry;

/// The item obtained when the block is dug.
/// TODO stabilize item stack and expose a vec<itemstack> option
enum DroppedItem {
    None,
    Fixed(String, u32),
    Dynamic(Box<dyn Fn() -> (StaticItemName, u32) + Sync + Send + 'static>),
    NotDiggable,
}
impl DroppedItem {
    fn build_dig_handler_inner<F>(closure: F, _game_builder: &GameBuilder) -> Box<InlineHandler>
    where
        F: Fn() -> Vec<ItemStack> + Sync + Send + 'static,
    {
        Box::new(move |ctx, target_block, ext_data, stack| {
            let block_type = ctx.block_types().get_block(target_block)?.0;
            let rule = ctx
                .items()
                .from_stack(stack)
                .and_then(|item| item.get_interaction_rule(block_type).cloned());
            if rule.as_ref().and_then(|x| x.dig_time(block_type)).is_some() {
                *target_block = AIR_ID;
                ext_data.clear();

                Ok(BlockInteractionResult {
                    item_stacks: closure(),
                    tool_wear: match rule {
                        Some(rule) => rule.computed_tool_wear(block_type)?,
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
            DroppedItem::None => Self::build_dig_handler_inner(Vec::new, game_builder),
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
            DroppedItem::NotDiggable => Box::new(|ctx, block, _, _| {
                tracing::warn!(
                    "Block {:?} is not diggable, but {:?} tried to dig it",
                    block,
                    ctx.initiator()
                );
                Ok(Default::default())
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

#[derive(Debug, Clone, Copy)]
enum VariantEffect {
    None,
    RotateNesw,
    Liquid,
}

/// Represents a block built by [GameBuilder::add_block]
pub struct BuiltBlock {
    /// The ID of the block
    pub id: BlockId,
    /// The autogenerated item for this blocktype
    pub item_name: ItemName,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatterType {
    /// Most tools target this block, and can try to dig it.
    Solid,
    /// Most tools target *through* this block, and most placed blocks can displace this block (removing it if the placed block occupies the same space).
    /// Buckets can target this block.
    Liquid,
    /// Most tools target *through* this block, and most placed blocks can displace this block (removing it if the placed block occupies the same space).
    /// The item interactions of this block are not yet well-defined, but will likely follow game content/plugins that are created in the future.
    Gas,
}

/// Builder for simple blocks.
/// Note that there are behaviors that this builder cannot express, but
/// [crate::server_api::BlockType] (when used directly via the unstable API) can.
#[must_use = "Builders do nothing unless used; set_foo will return a new builder."]
pub struct BlockBuilder {
    block_name: String,
    item: Item,
    dropped_item: DroppedItem,
    modifiers: Vec<Box<dyn FnOnce(&mut BlockType)>>,
    item_modifiers: Vec<Box<dyn FnOnce(&mut Item)>>,
    /// Same parameters as [perovskite_server::game_state::items::PlaceHandler]
    extended_data_initializer: Option<ExtendedDataInitializer>,
    // Exposed within the crate while not all APIs are complete
    pub(crate) client_info: BlockTypeDef,
    variant_effect: VariantEffect,
    liquid_flow_period: Option<Duration>,
    falls_down: bool,
    matter_type: MatterType,
}
impl BlockBuilder {
    /// Create a new block builder that will build a block and a corresponding inventory
    /// item for it.
    pub fn new(name: impl Into<BlockName>) -> BlockBuilder {
        let name = name.into().0;
        let item = Item {
            proto: ItemDef {
                short_name: name.clone(),
                display_name: name.clone(),
                inventory_texture: None,
                groups: vec![],
                interaction_rules: default_item_interaction_rules(),
                quantity_type: Some(QuantityType::Stack(256)),
                block_apperance: name.clone(),
                sort_key: "".to_string(),
            },
            dig_handler: None,
            tap_handler: None,
            place_handler: Some(Box::new(|_, _, _, _| {
                panic!("Incomplete place_handler; item registration was not completed properly");
            })),
        };

        BlockBuilder {
            block_name: name.clone(),
            item,
            dropped_item: DroppedItem::Fixed(name.clone(), 1),
            modifiers: vec![],
            item_modifiers: vec![],
            extended_data_initializer: None,
            client_info: BlockTypeDef {
                id: 0,
                short_name: name.clone(),
                base_dig_time: 1.0,
                groups: vec![],
                wear_multiplier: 1.0,
                light_emission: 0,
                allow_light_propagation: false,
                footstep_sound: 0,
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
                tool_custom_hitbox: None,
                physics_info: Some(PhysicsInfo::Solid(Empty {})),
                sound_id: 0,
                sound_volume: 0.0,
            },
            variant_effect: VariantEffect::None,
            liquid_flow_period: None,
            falls_down: false,
            matter_type: MatterType::Solid,
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

    /// Sets the sort key for the item in inventories/menus
    pub fn set_item_sort_key(mut self, sort_key: impl Into<String>) -> Self {
        self.item.proto.sort_key = sort_key.into();
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
        closure: impl Fn() -> (StaticItemName, u32) + Send + Sync + 'static,
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
    /// See [block_groups] for useful values.
    pub fn add_block_group(mut self, group: impl Into<String>) -> Self {
        let group = group.into();
        if !self.client_info.groups.contains(&group) {
            self.client_info.groups.push(group);
        }
        self
    }

    /// Adds multiple block groups to the list of groups for this block.
    /// These can affect diggability, dig speed, and other behavior in
    /// the map.
    pub fn add_block_groups(mut self, grops: impl IntoIterator<Item = impl Into<String>>) -> Self {
        for group in grops {
            self = self.add_block_group(group);
        }
        self
    }

    /// Sets that this block should not be diggable
    pub fn set_not_diggable(mut self) -> Self {
        self.dropped_item = DroppedItem::NotDiggable;
        self
    }

    /// Adds a group to the list of groups for the item corresponding to this block.
    /// These can affect crafting with this block (crafting API is TBD)
    ///
    /// See [block_groups] for useful values.
    pub fn add_item_group(mut self, group: impl Into<String>) -> Self {
        self.item.proto.groups.push(group.into());
        self
    }
    /// Set the display name visible when hovering in the inventory.
    pub fn set_display_name(mut self, display_name: impl Into<String>) -> Self {
        self.item.proto.display_name = display_name.into();
        self
    }

    maybe_export!(
        /// Run arbitrary changes on the block definition just before it's registered
        fn add_modifier(mut self, modifier: Box<dyn FnOnce(&mut BlockType)>) -> Self {
            self.modifiers.push(modifier);
            self
        }
    );
    maybe_export!(
        /// Run arbitrary changes on the associated item definition just before it's registered
        fn add_item_modifier(mut self, modifier: Box<dyn FnOnce(&mut Item)>) -> Self {
            self.item_modifiers.push(modifier);
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
            CubeVariantEffect::Liquid => VariantEffect::Liquid,
        };
        self.client_info.render_info = Some(RenderInfo::Cube(appearance.render_info));
        self
    }
    /// Sets the texture shown for this block's item in the inventory.
    pub fn set_inventory_texture(mut self, texture: impl Into<TextureReference>) -> Self {
        self.item.proto.block_apperance = "".to_string();
        self.item.proto.inventory_texture = Some(texture.into());
        self
    }
    /// Convenience method that sets this block to a simple appearance as a cube with the same texture on all faces,
    /// no transparency/translucency, no light propagation or emission, and no additional appearance settings
    /// (which may be added in the future)
    pub fn set_cube_single_texture(self, texture: impl Into<TextureReference>) -> Self {
        self.set_cube_appearance(CubeAppearanceBuilder::new().set_single_texture(texture))
    }

    pub fn set_axis_aligned_boxes_appearance(
        mut self,
        appearance: AxisAlignedBoxesAppearanceBuilder,
    ) -> Self {
        self.variant_effect = if appearance
            .display_proto
            .boxes
            .iter()
            .any(|b| b.rotation() == AxisAlignedBoxRotation::Nesw)
        {
            VariantEffect::RotateNesw
        } else {
            VariantEffect::None
        };
        self.client_info.render_info = Some(RenderInfo::AxisAlignedBoxes(
            appearance.display_proto.clone(),
        ));
        self.client_info.physics_info = Some(PhysicsInfo::SolidCustomCollisionboxes(
            appearance.collision_proto,
        ));
        self.client_info.tool_custom_hitbox = Some(appearance.tool_proto);
        self
    }

    /// Sets the block to have a plant-like appearance
    pub fn set_plant_like_appearance(mut self, appearance: PlantLikeAppearanceBuilder) -> Self {
        self.client_info.render_info = Some(RenderInfo::PlantLike(appearance.render_info));
        // TODO: this should be separated out into its own control. For now, this is a reasonable default
        self.client_info.physics_info = Some(PhysicsInfo::Air(Empty {}));
        self
    }

    /// Sets the light emission of this block, causing it to glow and illuminate other blocks.
    /// The meaningful range of light emission is 0-15.
    ///
    /// Moving one block causes light to fall off by one.
    pub fn set_light_emission(mut self, light_emission: u32) -> Self {
        assert!(light_emission < 16);
        self.client_info.light_emission = light_emission;
        self
    }

    /// Sets whether the block allows light to propagate through it.
    ///
    /// This is separate from appearance builder calls like set_needs_transparency and/or
    /// set_needs_translucency; those affect how the block is rasterized on the GPU, but
    /// not how map lighting is calculated.
    pub fn set_allow_light_propagation(mut self, allow_light_propagation: bool) -> Self {
        self.client_info.allow_light_propagation = allow_light_propagation;
        self
    }

    /// Makes the block flow as a liquid on a regular basis. Setting period to None disables flow
    /// Note that you should also call set_matter_type(Liquid) if you want the block to interact
    /// as a liquid (e.g. tools/blocks point through it/can replace it when placing,
    /// buckets point at it/fill it)
    ///
    /// When multiple liquids share the same period, they can share a timer worker, which improves
    /// CPU efficiency.
    pub fn set_liquid_flow(mut self, period: Option<Duration>) -> Self {
        self.liquid_flow_period = period;
        self
    }

    /// Makes the block fall when there's nothing under it.
    ///
    /// The exact implementation is subject to change. At the moment,
    /// it is approximate, best-effort, and an abrupt, un-animated fall
    pub fn set_falls_down(mut self, falls_down: bool) -> Self {
        self.falls_down = falls_down;
        self
    }

    /// Sets the matter type of this block for tool interactions, and adds the corresponding block groun (see [block_groups]).
    /// The default is Solid
    pub fn set_matter_type(mut self, matter_type: MatterType) -> Self {
        self.matter_type = matter_type;
        self
    }

    /// Set the appearance of the block to that specified by the given builder
    pub(crate) fn build_and_deploy_into(
        mut self,
        game_builder: &mut GameBuilder,
    ) -> Result<BuiltBlock> {
        let mut block = BlockType::default();
        block.client_info = self.client_info;
        block.client_info.groups.push(match self.matter_type {
            MatterType::Solid => DEFAULT_SOLID.to_string(),
            MatterType::Liquid => DEFAULT_LIQUID.to_string(),
            MatterType::Gas => DEFAULT_GAS.to_string(),
        });
        if self.matter_type == MatterType::Solid {
            block.client_info.footstep_sound = game_builder.footstep_sound.0;
        }
        block.client_info.groups.sort();
        block.client_info.groups.dedup();
        block.dig_handler_inline = Some(self.dropped_item.build_dig_handler(game_builder));
        for modifier in self.modifiers {
            modifier(&mut block);
        }
        let block_handle = game_builder.inner.blocks_mut().register_block(block)?;

        let mut item = self.item;
        let item_name = ItemName(item.proto.short_name.clone());
        let extended_data_initializer = self.extended_data_initializer.take();
        // todo factor out a default place handler and export it to users of the API
        item.place_handler = Some(Box::new(move |ctx, coord, anchor, stack| {
            if stack.proto().quantity == 0 {
                return Ok(None);
            }
            let extended_data = match &extended_data_initializer {
                Some(x) => x(ctx.clone(), coord, anchor, stack)?,
                None => None,
            };
            let variant = match self.variant_effect {
                VariantEffect::None => 0,
                VariantEffect::RotateNesw => ctx
                    .initiator()
                    .position()
                    .map(|pos| variants::rotate_nesw_azimuth_to_variant(pos.face_direction.0))
                    .unwrap_or(0),
                VariantEffect::Liquid => 0xfff,
            };
            match ctx
                .game_map()
                .compare_and_set_block_predicate(
                    coord,
                    |block, _, block_types| {
                        // fast path
                        if block == AIR_ID {
                            return Ok(true);
                        };
                        let block_type = block_types.get_block(&block)?.0;
                        Ok(block_type
                            .client_info
                            .groups
                            .iter()
                            .any(|g| g == TRIVIALLY_REPLACEABLE))
                    },
                    block_handle.with_variant(variant)?,
                    extended_data,
                )?
                .0
            {
                CasOutcome::Match => Ok(stack.decrement()),
                CasOutcome::Mismatch => Ok(Some(stack.clone())),
            }
        }));
        for modifier in self.item_modifiers {
            modifier(&mut item);
        }
        game_builder.inner.items_mut().register_item(item)?;

        if let Some(period) = self.liquid_flow_period {
            if !matches!(self.variant_effect, VariantEffect::Liquid) {
                tracing::warn!(
                    "Liquid flow is only supported for blocks with a liquid variant effect"
                );
            } else {
                game_builder
                    .liquids_by_flow_time
                    .entry(period)
                    .or_default()
                    .push(block_handle);
            }
        }

        if self.falls_down {
            game_builder.falling_blocks.push(block_handle);
        }

        Ok(BuiltBlock {
            id: block_handle,
            item_name,
        })
    }
}

#[derive(Clone)]
pub struct CubeAppearanceBuilder {
    render_info: CubeRenderInfo,
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

    /// Makes the block act like a liquid during meshing. Variants 0-7 are partial (flowing)
    /// liquid, 0xfff is liquid source. Other values are undefined (but may be given a meaning
    /// in the future).
    pub(crate) fn set_liquid_shape(mut self) -> Self {
        self.render_info.variant_effect = CubeVariantEffect::Liquid.into();
        self
    }
}

impl Default for CubeAppearanceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct PlantLikeAppearanceBuilder {
    render_info: PlantLikeRenderInfo,
}
impl PlantLikeAppearanceBuilder {
    /// Constructs a new plant-like appearance builder, with a dummy texture and reasonable defaults.
    pub fn new() -> Self {
        Self {
            render_info: PlantLikeRenderInfo {
                tex: make_texture_ref(FALLBACK_UNKNOWN_TEXTURE.to_string()),
                wave_effect_scale: 0.1,
            },
        }
    }
    /// Sets the block's texture.
    pub fn set_texture<T>(mut self, texture: T) -> Self
    where
        T: Into<TextureReference>,
    {
        let tex = texture.into();
        self.render_info.tex = Some(tex);
        self
    }
    /// Sets the magnitude of the waving animation effect on the top of the block.
    pub fn set_wave_effect_scale(mut self, scale: f32) -> Self {
        self.render_info.wave_effect_scale = scale;
        self
    }
}
impl Default for PlantLikeAppearanceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub use protocol::render::TextureCrop;

fn make_texture_ref(tex_name: String) -> Option<TextureReference> {
    Some(TextureReference {
        texture_name: tex_name,
        crop: None,
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

pub(crate) struct FallingBlocksChunkEdgePropagator {
    pub(crate) blocks: Vec<BlockId>,
}
impl VerticalNeighborTimerCallback for FallingBlocksChunkEdgePropagator {
    fn vertical_neighbor_callback(
        &self,
        ctx: &HandlerContext,
        _upper: ChunkCoordinate,
        _lower: ChunkCoordinate,
        upper_chunk: &mut MapChunk,
        lower_chunk: &mut MapChunk,
        _timer_state: &TimerState,
    ) -> Result<()> {
        let replaceable_blocks = ctx
            .block_types()
            .fast_block_group(TRIVIALLY_REPLACEABLE)
            .expect("Missing replaceable block group");

        let blocks = self.blocks.iter().map(|&b| b.base_id()).collect::<Vec<_>>();
        // consider whether we might have enough falling blocks that a linear scan becomes slow
        for x in 0..16 {
            for z in 0..16 {
                // Upper chunk first
                for y in (0..15).rev() {
                    // Reverse iteration isn't the most efficient, but 16 entries of four bytes
                    // is 64 bytes, which fits in a cache line.
                    //
                    // Y covers 0 to 15. The block just above this Y covers 1 to 16.
                    let bottom = ChunkOffset { x, y, z };
                    let top = ChunkOffset { x, y: y + 1, z };
                    let bottom_block = upper_chunk.get_block(bottom);
                    let top_block = upper_chunk.get_block(top);
                    // TODO enhance this. we need some additional callbacks
                    // added to the block type that the falling block falls into, to
                    // actually damage/break the bottom block. For now we just replace it
                    // with the falling block and leave air where the falling block fell from
                    if top_block != AIR_ID
                        && replaceable_blocks.contains(bottom_block)
                        && blocks.contains(&top_block.base_id())
                    {
                        upper_chunk.swap_blocks(bottom, top);
                        upper_chunk.set_block(top, AIR_ID, None);
                    }
                }
                // then the seam
                let lower_coord = ChunkOffset { x, y: 15, z };
                let upper_coord = ChunkOffset { x, y: 0, z };
                let bottom_block = lower_chunk.get_block(lower_coord);
                let top_block = upper_chunk.get_block(upper_coord);
                if top_block != AIR_ID
                    && replaceable_blocks.contains(bottom_block)
                    && blocks.contains(&top_block.base_id())
                {
                    MapChunk::swap_blocks_across_chunks(
                        lower_chunk,
                        upper_chunk,
                        lower_coord,
                        upper_coord,
                    );
                    upper_chunk.set_block(upper_coord, AIR_ID, None);
                }

                // and now for the lower chunk
                for y in (0..15).rev() {
                    let bottom = ChunkOffset { x, y, z };
                    let top = ChunkOffset { x, y: y + 1, z };
                    let bottom_block: BlockId = lower_chunk.get_block(bottom);
                    let top_block = lower_chunk.get_block(top);

                    if top_block != AIR_ID
                        && replaceable_blocks.contains(bottom_block)
                        && blocks.contains(&top_block.base_id())
                    {
                        lower_chunk.swap_blocks(bottom, top);
                        lower_chunk.set_block(top, AIR_ID, None);
                    }
                }
            }
        }
        Ok(())
    }
}

pub(crate) struct LiquidPropagator {
    pub(crate) liquids: Vec<BlockTypeHandle>,
}
impl BulkUpdateCallback for LiquidPropagator {
    fn bulk_update_callback(
        &self,
        _ctx: &HandlerContext,
        chunk_coordinate: ChunkCoordinate,
        _timer_state: &TimerState,
        chunk: &mut MapChunk,
        neighbors: Option<&ChunkNeighbors>,
    ) -> Result<()> {
        let neighbors = neighbors.unwrap();
        for liquid_type in self.liquids.iter() {
            for x in 0..16 {
                for z in 0..16 {
                    for y in 0..16 {
                        let offset = ChunkOffset { x, y, z };
                        let coord = chunk_coordinate.with_offset(offset);
                        let block = chunk.get_block(offset);
                        if liquid_type.equals_ignore_variant(block) && block.variant() == 0xfff {
                            // liquid sources are invariant.
                            continue;
                        }
                        let is_air = AIR_ID.equals_ignore_variant(block);
                        let is_same_liquid = liquid_type.equals_ignore_variant(block);
                        if is_air || is_same_liquid {
                            // Apply the decay rule first
                            let variant_from_decay = match block.variant() {
                                // Air stays air, regardless of what variant it has
                                _ if is_air => -1,
                                // By default, decay one liquid level.
                                // Zero becomes air
                                0 => -1,
                                // Source stays source
                                0xfff if is_same_liquid => 0xfff,
                                // and flowing water drops by a level
                                x => x.saturating_sub(1).min(7) as i32,
                            };
                            let mut variant_from_flow = -1;
                            for (dx, dy, dz) in [(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)] {
                                if let Some(neighbor_liquid) = coord
                                    .try_delta(dx, dy, dz)
                                    .and_then(|x| neighbors.get_block(x))
                                {
                                    // if there's a matching liquid at the neighbor...
                                    if liquid_type.equals_ignore_variant(neighbor_liquid) {
                                        // and the material below it causes it to spread...
                                        if coord
                                            .try_delta(dx, dy - 1, dz)
                                            .and_then(|x| neighbors.get_block(x))
                                            .is_some_and(|x| {
                                                self.can_flow_laterally_over(x, liquid_type)
                                            })
                                        {
                                            variant_from_flow = variant_from_flow
                                                .max((neighbor_liquid.variant() as i32) - 1)
                                                .min(7);
                                        }
                                    }
                                }
                            }

                            if !coord
                                .try_delta(0, -1, 0)
                                .and_then(|x| neighbors.get_block(x))
                                .is_some_and(|x| self.can_flow_laterally_over(x, liquid_type))
                            {
                                // If this block is draining directly down, clamp its variant (from flow) to 0.
                                variant_from_flow = variant_from_flow.min(0);
                            }

                            if coord
                                .try_delta(0, 1, 0)
                                .and_then(|x| neighbors.get_block(x))
                                .is_some_and(|x| x.equals_ignore_variant(*liquid_type))
                            {
                                // If there's liquid above, let it flow into here. This happens after clamping-to-zero (on downflow)
                                // and overrides it.
                                variant_from_flow = 7;
                            }

                            let new_variant = variant_from_flow.max(variant_from_decay);

                            let new_block = if new_variant < 0 {
                                AIR_ID
                            } else {
                                liquid_type.with_variant_unchecked(new_variant as u16)
                            };
                            if block != new_block {
                                chunk.set_block(coord.offset(), new_block, None);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

impl LiquidPropagator {
    fn can_flow_laterally_over(&self, x: BlockId, liquid_type: &BlockTypeHandle) -> bool {
        if x.equals_ignore_variant(AIR_ID) {
            // if it's air below, don't let it flow laterally
            false
        } else if x.equals_ignore_variant(*liquid_type) {
            // if it's the same liquid below, don't let it flow laterally unless it's a source
            x.variant() == 0xfff
        } else {
            // It's a different block
            true
        }
    }
}

/// How textures for an axis-aligned box should be cropped
#[derive(Clone, Debug)]
pub enum TextureCropping {
    /// The texture is sized to span the entire 1x1x1 cube in world space. It is then cropped down to fit the axis-aligned box being drawn.
    /// This prevents visual artifacts (z-flickering) when two axis-aligned boxes overlap on a face, as long as both use the same texture.
    AutoCrop,
    /// The texture is scaled to fit the axis-aligned box.
    NoCrop,
}

#[derive(Clone, Copy, Debug)]
/// How axis-aligned boxes rotate
pub enum RotationMode {
    /// No rotation, the box always faces the same direction
    None,
    /// The box rotates horizontally around the center of the block.
    /// When placed, the player's facing direction sets the rotation.
    RotateHorizontally,
}

/// The textures to apply to an axis-aligned box.
#[derive(Clone, Debug)]
pub struct AaBoxProperties {
    left: TextureReference,
    right: TextureReference,
    top: TextureReference,
    bottom: TextureReference,
    front: TextureReference,
    back: TextureReference,
    crop_mode: TextureCropping,
    rotation_mode: RotationMode,
    is_visible: bool,
    is_colliding: bool,
    is_tool_hitbox: bool,
}
impl AaBoxProperties {
    pub fn new(
        left: impl Into<TextureReference>,
        right: impl Into<TextureReference>,
        top: impl Into<TextureReference>,
        bottom: impl Into<TextureReference>,
        front: impl Into<TextureReference>,
        back: impl Into<TextureReference>,
        crop_mode: TextureCropping,
        rotation_mode: RotationMode,
    ) -> Self {
        Self {
            left: left.into(),
            right: right.into(),
            top: top.into(),
            bottom: bottom.into(),
            front: front.into(),
            back: back.into(),
            crop_mode,
            rotation_mode,
            is_visible: true,
            is_colliding: true,
            is_tool_hitbox: true,
        }
    }
    pub fn new_custom_usage(
        left: impl Into<TextureReference>,
        right: impl Into<TextureReference>,
        top: impl Into<TextureReference>,
        bottom: impl Into<TextureReference>,
        front: impl Into<TextureReference>,
        back: impl Into<TextureReference>,
        crop_mode: TextureCropping,
        rotation_mode: RotationMode,
        is_visible: bool,
        is_colliding: bool,
        is_tool_hitbox: bool,
    ) -> Self {
        Self {
            left: left.into(),
            right: right.into(),
            top: top.into(),
            bottom: bottom.into(),
            front: front.into(),
            back: back.into(),
            crop_mode,
            rotation_mode,
            is_visible,
            is_colliding,
            is_tool_hitbox,
        }
    }
    pub fn new_single_tex(
        texture: impl Into<TextureReference>,
        crop_mode: TextureCropping,
        rotation_mode: RotationMode,
    ) -> Self {
        let tex = texture.into();
        Self {
            left: tex.clone(),
            right: tex.clone(),
            top: tex.clone(),
            bottom: tex.clone(),
            front: tex.clone(),
            back: tex,
            crop_mode,
            rotation_mode,
            is_visible: true,
            is_colliding: true,
            is_tool_hitbox: true,
        }
    }
}

/// Block appearance builder for blocks that have custom axis-aligned box geometry
pub struct AxisAlignedBoxesAppearanceBuilder {
    display_proto: AxisAlignedBoxes,
    collision_proto: AxisAlignedBoxes,
    tool_proto: AxisAlignedBoxes,
}
impl AxisAlignedBoxesAppearanceBuilder {
    pub fn new() -> Self {
        Self {
            display_proto: AxisAlignedBoxes::default(),
            collision_proto: AxisAlignedBoxes::default(),
            tool_proto: AxisAlignedBoxes::default(),
        }
    }

    /// Adds a box to the block appearance builder.
    ///
    /// x, y, z are given as (min, max) with the center of the cube at 0.0.
    /// A full cube would span from -0.5 to 0.5 in all directions.
    pub fn add_box(
        self,
        box_properties: AaBoxProperties,
        x: (f32, f32),
        y: (f32, f32),
        z: (f32, f32),
    ) -> Self {
        self.add_box_with_variant_mask(box_properties, x, y, z, 0)
    }

    /// Same as [`AxisAlignedBoxesAppearanceBuilder::add_box`] but with an additional variant mask.
    /// If the variant mask is 0, or variant_mask & block.variant() is nonzero,
    /// the box will be drawn.
    pub fn add_box_with_variant_mask(
        self,
        box_properties: AaBoxProperties,
        x: (f32, f32),
        y: (f32, f32),
        z: (f32, f32),
        variant_mask: u32,
    ) -> Self {
        self.add_box_with_variant_mask_and_slope(
            box_properties,
            x,
            y,
            z,
            variant_mask,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    }

    /// Same as [`AxisAlignedBoxesAppearanceBuilder::add_box_with_variant_mask`] but with an additional slope_x and slope_z.
    /// If the slopes are 0.0, the top face is flat. Otherwise, the top face slopes with the actual top y equal to
    /// y.1 + (x*slope_x) + (z*slope_z). If the axis-aligned box is configured to rotate, then rotation is applied after the slope calculation.
    pub fn add_box_with_variant_mask_and_slope(
        mut self,
        box_properties: AaBoxProperties,
        x: (f32, f32),
        y: (f32, f32),
        z: (f32, f32),
        variant_mask: u32,
        top_slope_x: f32,
        top_slope_z: f32,
        bottom_slope_x: f32,
        bottom_slope_z: f32,
    ) -> Self {
        // TODO: When slope is nonzero and the crop mode is set to autocrop, we need
        // to adjust the corners of the texture to account for the slope.
        let the_box = AxisAlignedBox {
            x_min: x.0,
            x_max: x.1,
            y_min: y.0,
            y_max: y.1,
            z_min: z.0,
            z_max: z.1,
            tex_left: Some(Self::maybe_crop(
                box_properties.left,
                &box_properties.crop_mode,
                (0.5 - z.1, 0.5 - z.0),
                (0.5 - y.1, 0.5 - y.0),
            )),
            tex_right: Some(Self::maybe_crop(
                box_properties.right,
                &box_properties.crop_mode,
                (z.0 + 0.5, z.1 + 0.5),
                (0.5 - y.1, 0.5 - y.0),
            )),
            tex_top: Some(Self::maybe_crop(
                box_properties.top,
                &box_properties.crop_mode,
                (0.5 - x.1, 0.5 - x.0),
                (z.0 + 0.5, z.1 + 0.5),
            )),
            tex_bottom: Some(Self::maybe_crop(
                box_properties.bottom,
                &box_properties.crop_mode,
                (x.0 + 0.5, x.1 + 0.5),
                (z.0 + 0.5, z.1 + 0.5),
            )),
            tex_front: Some(Self::maybe_crop(
                box_properties.front,
                &box_properties.crop_mode,
                (x.0 + 0.5, x.1 + 0.5),
                (0.5 - y.1, 0.5 - y.0),
            )),
            tex_back: Some(Self::maybe_crop(
                box_properties.back,
                &box_properties.crop_mode,
                (0.5 - x.1, 0.5 - x.0),
                (0.5 - y.1, 0.5 - y.0),
            )),
            rotation: match box_properties.rotation_mode {
                RotationMode::None => AxisAlignedBoxRotation::None.into(),
                RotationMode::RotateHorizontally => AxisAlignedBoxRotation::Nesw.into(),
            },
            variant_mask,
            top_slope_x,
            top_slope_z,
            bottom_slope_x,
            bottom_slope_z,
        };
        if box_properties.is_visible {
            self.display_proto.boxes.push(the_box.clone());
        }
        if box_properties.is_colliding {
            self.collision_proto.boxes.push(the_box.clone());
        }
        if box_properties.is_tool_hitbox {
            self.tool_proto.boxes.push(the_box);
        }
        self
    }

    fn maybe_crop(
        tex: TextureReference,
        crop_mode: &TextureCropping,
        extents_u: (f32, f32),
        extents_v: (f32, f32),
    ) -> TextureReference {
        match crop_mode {
            TextureCropping::AutoCrop => TextureReference {
                texture_name: tex.texture_name,
                crop: Some(TextureCrop {
                    left: extents_u.0,
                    right: extents_u.1,
                    top: extents_v.0,
                    bottom: extents_v.1,
                    dynamic: None,
                }),
            },
            TextureCropping::NoCrop => TextureReference {
                texture_name: tex.texture_name,
                crop: None,
            },
        }
    }
}

impl Default for AxisAlignedBoxesAppearanceBuilder {
    fn default() -> Self {
        Self::new()
    }
}
