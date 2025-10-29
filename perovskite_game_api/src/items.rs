use crate::blocks::variants::rotate_nesw_azimuth_to_variant;
use crate::blocks::CubeAppearanceBuilder;
use crate::blocks::{DroppedItem, RotationMode};
use crate::game_builder::{BlockName, GameBuilder, ItemName};
use crate::NonExhaustive;
use anyhow::{bail, Context, Result};
use perovskite_core::block_id::{BlockError, BlockId};
use perovskite_core::constants::block_groups::TRIVIALLY_REPLACEABLE;
use perovskite_core::constants::items::default_item_interaction_rules;
use perovskite_core::coordinates::BlockCoordinate;
use perovskite_core::protocol::items::item_def;
use perovskite_core::protocol::items::item_def::QuantityType;
use perovskite_core::protocol::render::TextureReference;
use perovskite_server::game_state::blocks::{
    BlockInteractionResult, BlockTypeManager, FastBlockName,
};
use perovskite_server::game_state::event::HandlerContext;
use perovskite_server::game_state::items::{
    BlockInteractionHandler, Item, ItemInteractionResult, ItemStack, PointeeBlockCoords,
};

/// If the item is pointing at this kind of block, apply this item action.
/// For now, this always ignores variants
#[derive(Debug, Clone)]
pub enum ItemActionTarget {
    Block(FastBlockName),
    Blocks(Vec<FastBlockName>),
    BlockGroup(String),
    Any,
}
impl ItemActionTarget {
    fn matches(&self, block_types: &BlockTypeManager, id: BlockId) -> bool {
        match &self {
            ItemActionTarget::Block(x) => {
                let x_id = block_types.resolve_name(&x);
                x_id.is_some_and(|x_id| x_id.equals_ignore_variant(id))
            }
            ItemActionTarget::Blocks(blocks) => blocks.iter().any(|x| {
                let x_id = block_types.resolve_name(&x);
                x_id.is_some_and(|x_id| x_id.equals_ignore_variant(id))
            }),
            ItemActionTarget::BlockGroup(g) => {
                let block_type = match block_types.get_block(id) {
                    Ok(x) => x,
                    Err(BlockError::IdNotFound(_)) => {
                        // false arther than erring since we might match an Any later
                        return false;
                    }
                    Err(e) => {
                        log::warn!("Unexpected error in item handler: {e:?}");
                        return false;
                    }
                };
                block_type.0.client_info.groups.contains(&g)
            }
            ItemActionTarget::Any => true,
        }
    }
}

pub enum ItemAction {
    /// Places a block.
    PlaceBlock {
        /// The block to be placed
        block: FastBlockName,
        /// If applicable, use the player's facing direction to change the variant of the block.
        /// Note that the block itself must be set up to rotate with the variant (e.g.
        /// [CubeAppearanceBuilder::set_rotate_laterally]). This is not currently checked
        /// automatically.
        rotation_mode: RotationMode,
        /// If true, place even if the block isn't one that you can normally wipe out by placing
        ignore_trivially_replaceable: bool,
        /// If true, place onto the selected block. If false, replace the selected block.
        place_onto: bool,
    },
    /// Digs out the existing block. Note that at this time, this does not have feature parity with
    /// [crate::default_game::tools::register_tool], and is probably less convenient for use-cases
    /// that are primarily tools
    DigBlock,
    /// No action is taken.
    DoNothing,
}

pub enum StackDecrement {
    /// Decrement the stack by this much. If the stack isn't large enough, cancel the whole action.
    /// This is recommended for item stacks
    FixedRequireSufficient(u32),
    /// Decrement the stack by this much. If the stack isn't large enough, destroy the stack but
    /// finish the action. This is recommended for tool wear.
    FixedDestroyIfInsufficient(u32),
    /// Tool wear scaled: respect the wear multiplier defined for the block being dug (multiplying
    /// the base wear specified here). If the tool health isn't high enough, break the tool.
    /// Always does at least 1 wear
    WearMultipliedBy(f64),
}

pub struct ItemHandler {
    /// What kind of block can be targeted for this action to occur? This always checks against the
    /// targeted block (even for placement actions)
    pub target: ItemActionTarget,
    /// What action is taken?
    pub action: ItemAction,
    /// What happens to the item stack when this action successfully occurs?
    pub stack_decrement: Option<StackDecrement>,
    /// What does the player get? (in addition to the things gained from DigBlock)
    pub dropped_item: DroppedItem,
    /// Marker that additional fields may be added in the future, construct with Default::default
    pub _ne: NonExhaustive,
}
impl ItemHandler {
    fn execute(
        &self,
        ctx: &HandlerContext,
        coord: PointeeBlockCoords,
        target_block_id: BlockId,
        stack: &ItemStack,
    ) -> Result<ItemInteractionResult> {
        let new_stack = match &self.stack_decrement {
            None => Some(stack.clone()),
            Some(StackDecrement::FixedRequireSufficient(x)) => {
                if stack.quantity_or_wear() < *x {
                    // Not enough
                    return Ok(ItemInteractionResult {
                        updated_stack: Some(stack.clone()),
                        obtained_items: vec![],
                    });
                } else {
                    stack.decrement_by(*x)
                }
            }
            Some(StackDecrement::FixedDestroyIfInsufficient(x)) => stack.decrement_by(*x),
            Some(StackDecrement::WearMultipliedBy(base)) => {
                let multiplier = ctx
                    .block_types()
                    .get_block(target_block_id)
                    .map(|x| x.0.client_info.wear_multiplier)
                    .unwrap_or(1.0)
                    .min(1.0);
                stack.decrement_by((base * multiplier) as u32)
            }
        };

        let (success, dig_items) = match &self.action {
            ItemAction::PlaceBlock {
                block,
                rotation_mode,
                ignore_trivially_replaceable,
                place_onto,
            } => {
                let action_coord = if *place_onto {
                    match coord.preceding {
                        Some(x) => x,
                        None => {
                            return Ok(ItemInteractionResult {
                                updated_stack: Some(stack.clone()),
                                obtained_items: vec![],
                            });
                        }
                    }
                } else {
                    coord.selected
                };
                let variant = match rotation_mode {
                    RotationMode::None => 0,
                    RotationMode::RotateHorizontally => rotate_nesw_azimuth_to_variant(
                        ctx.initiator()
                            .position()
                            .map(|x| x.face_direction.0)
                            .unwrap_or(0.0),
                    ),
                };
                let new_block = ctx
                    .block_types()
                    .resolve_name(&block)
                    .context("Unknown block in ItemAction::PlaceBlock")?
                    .with_variant_unchecked(variant);
                let success =
                    ctx.game_map()
                        .mutate_block_atomically(action_coord, |id, _ext| {
                            if !self.target.matches(ctx.block_types(), *id) {
                                // Someone raced with us
                                tracing::error!("mismatch inside atomic mutator");
                                return Ok(false);
                            }
                            let success = *ignore_trivially_replaceable
                                || ctx.block_types().get_block(*id).is_ok_and(|x| {
                                    x.0.client_info
                                        .groups
                                        .iter()
                                        .any(|x| x == TRIVIALLY_REPLACEABLE)
                                });
                            if success {
                                *id = new_block;
                            }
                            Ok(success)
                        })?;
                (success, vec![])
            }
            ItemAction::DigBlock => (
                true,
                ctx.game_map()
                    .dig_block(coord.selected, ctx.initiator(), Some(&stack))?
                    .item_stacks,
            ),
            ItemAction::DoNothing => (true, vec![]),
        };
        if !success {
            return Ok(ItemInteractionResult {
                updated_stack: Some(stack.clone()),
                obtained_items: vec![],
            });
        }
        let mut items = self
            .dropped_item
            .get_item(coord.selected, target_block_id.variant());
        items.extend(dig_items);
        Ok(ItemInteractionResult {
            updated_stack: dbg!(new_stack),
            obtained_items: items,
        })
    }
}

/// The defaults for an item handler, which does nothing, matching no blocks, etc. These defaults
/// should be overridden.
impl Default for ItemHandler {
    fn default() -> Self {
        Self {
            target: ItemActionTarget::Blocks(vec![]),
            action: ItemAction::DoNothing,
            stack_decrement: None,
            dropped_item: DroppedItem::None,
            _ne: NonExhaustive(()),
        }
    }
}

/// Builder for items, providing additional functionality (interactivity, wear, etc.) beyond
/// [crate::game_builder::GameBuilder::register_basic_item].
///
/// This is less featured than enabling the `unstable_api` feature and writing item definitions and
/// event handlers by hand using `crate::server_api`.
///
/// It's meant for common use-cases, primarily in content that features simple mechanics but needs
/// to quickly/easily expand the collection of blocks and items available to players. Note that as
/// requirements evolve, more features might be made available through the builder.
///
/// Entity interactions are currently not supported; the default dig and tap handlers (as defined on
/// the entity) will apply.
///
/// Custom interaction rules are currently not supported, items will target solid objects and ignore
/// liquids and air.
#[must_use = "Builders do nothing unless used; setters will return a new builder"]
pub struct ItemBuilder {
    modifiers: Vec<Box<dyn FnOnce(&mut Item)>>,
    dig_handlers: Vec<ItemHandler>,
    tap_handlers: Vec<ItemHandler>,
    right_click_handlers: Vec<ItemHandler>,
    item_obj: Item,
}
impl ItemBuilder {
    pub fn new(item_name: impl Into<ItemName>) -> Self {
        let item_name: ItemName = item_name.into();
        ItemBuilder {
            modifiers: vec![],
            dig_handlers: vec![],
            tap_handlers: vec![],
            right_click_handlers: vec![],
            item_obj: Item {
                proto: perovskite_core::protocol::items::ItemDef {
                    short_name: item_name.0,
                    display_name: "".to_string(),
                    groups: vec![],
                    interaction_rules: default_item_interaction_rules(),
                    sort_key: "".to_string(),
                    appearance: None,
                    quantity_type: None,
                },
                dig_handler: None,
                dig_entity_handler: None,
                tap_handler: None,
                tap_entity_handler: None,
                place_on_block_handler: None,
                place_on_entity_handler: None,
            },
        }
    }

    /// Makes this item have this texture in the inventory. It is not 3d projected or turned into
    /// a block
    pub fn set_inventory_texture(mut self, texture: impl Into<TextureReference>) -> Self {
        let tex: TextureReference = texture.into();
        self.item_obj.proto.appearance = Some(item_def::Appearance::InventoryTexture(tex.diffuse));
        self
    }

    /// Convenience function to make this item dig blocks without special effect or benefit
    pub fn add_default_dig_handler(mut self) -> Self {
        self.dig_handlers.push(ItemHandler {
            target: ItemActionTarget::Any,
            stack_decrement: None,
            dropped_item: DroppedItem::None,
            action: ItemAction::DigBlock,
            ..Default::default()
        });
        self
    }

    /// Sets the item to stack this many.
    pub fn set_max_stack(mut self, count: u32) -> Self {
        self.item_obj.proto.quantity_type = Some(QuantityType::Stack(count));
        self
    }

    /// Sets the item to have this much tool wear when fully repaired.
    pub fn set_max_wear(mut self, count: u32) -> Self {
        self.item_obj.proto.quantity_type = Some(QuantityType::Wear(count));
        self
    }

    pub fn set_display_name(mut self, name: impl Into<String>) -> Self {
        self.item_obj.proto.display_name = name.into();
        self
    }

    /// Makes this item look like this block, when placed in the inventory. Adds a bit of time to
    /// login/client startup since blocks must be rendered on the client (but this is cached)
    pub fn set_inventory_block_appearance(mut self, block: impl Into<BlockName>) -> Self {
        let block: BlockName = block.into();
        self.item_obj.proto.appearance = Some(item_def::Appearance::BlockApperance(block.0));
        self
    }

    /// Specifies where the block gets sorted (e.g. to provide richer categorization). e.g. while
    /// an iron pickaxe might be called `default:iron_pickaxe`, it'll have a sort key of
    /// `default:tools:pickaxes:iron` to sort with the default plugin, with tools, with pickaxes.
    pub fn set_sort_key(mut self, sort_key: impl Into<String>) -> Self {
        self.item_obj.proto.sort_key = sort_key.into();
        self
    }

    /// Adds a dig handler to the item. Currently, the first matching handler (based on its
    /// `target`) will apply.
    pub fn add_dig_handler(mut self, handler: ItemHandler) -> Self {
        self.dig_handlers.push(handler);
        self
    }

    /// Adds a tap handler to the item. Currently, the first matching handler (based on its
    /// `target`) will apply.
    pub fn add_tap_handler(mut self, handler: ItemHandler) -> Self {
        self.tap_handlers.push(handler);
        self
    }

    /// Adds a tap handler to the item. Currently, the first matching handler (based on its
    /// `target`) will apply.
    pub fn add_right_click_handler(mut self, handler: ItemHandler) -> Self {
        self.right_click_handlers.push(handler);
        self
    }

    pub fn build_into(mut self, builder: &mut GameBuilder) -> Result<()> {
        self.item_obj.dig_handler = Some(build_handler_chain(self.dig_handlers));
        self.item_obj.tap_handler = Some(build_handler_chain(self.tap_handlers));
        self.item_obj.place_on_block_handler = Some(build_handler_chain(self.right_click_handlers));
        builder.inner.items_mut().register_item(self.item_obj)
    }
}

fn build_handler_chain(handlers: Vec<ItemHandler>) -> Box<BlockInteractionHandler> {
    Box::new(move |ctx, coord, item_stack| {
        // todo: this would be great with an optimistic transaction API. In lieu of cobbling one
        // together here ad-hoc, this will just be weakly consistent instead
        let block = ctx.game_map().get_block(coord.selected)?;
        for handler in &handlers {
            tracing::info!("checking a handler for {:?}", handler.target);
            if handler.target.matches(ctx.block_types(), block) {
                tracing::info!("match");
                return handler.execute(ctx, coord, block, item_stack);
            }
        }
        // none matched
        Ok(ItemInteractionResult {
            updated_stack: Some(item_stack.clone()),
            obtained_items: vec![],
        })
    })
}
