//! A block that spawns a cart onto the track it faces when tapped, consuming the cart item from
//! its own inventory.
//!
//! Variant layout:
//! * bits 0-1: facing direction (rotate laterally)
//! * [SPAWNED_OCCUPIED_VARIANT_BIT]: the spawner is locked by the cart it spawned. Cleared when
//!   the cart releases it (see [super::PendingAction::SpawnerRelease])
//! * [SPAWNER_LOCKING_VARIANT_BIT]: short-lived lock held while a spawn attempt is in flight.
//!   We can't atomically update the map and the entity state, so this serializes spawn attempts
//!   until we know whether to transition to occupied or unoccupied.
use std::time::Duration;

use anyhow::{bail, Context, Result};
use perovskite_core::{
    block_id::BlockId,
    chat::{ChatMessage, SERVER_ERROR_COLOR},
    coordinates::BlockCoordinate,
    protocol::render::TextureTransform,
};
use perovskite_server::game_state::{
    blocks::{CompassDirection, ExtendedData},
    event::HandlerContext,
    items::{ItemStack, MaybeStack},
};

use super::{spawn_cart, CartsGameBuilderExtension, PendingAction};
use crate::{
    blocks::{BlockAppearanceBuilder, BlockBuilder, CubeAppearanceBuilder},
    colors::Color,
    default_game::block_groups::BRITTLE,
    game_builder::{GameBuilder, StaticBlockName, StaticTextureName, TextureRefExt},
    include_texture_bytes,
};
use perovskite_server::game_state::client_ui::UiElementContainer;

pub const CART_SPAWNER: StaticBlockName = StaticBlockName("carts:cart_spawner");

const SPAWNER_ARROW_TEX: StaticTextureName = StaticTextureName("carts:spawner_arrow_point_up");
const SPAWNER_FRONT_TEX: StaticTextureName = StaticTextureName("carts:spawner_front");
const SPAWNER_BACK_TEX: StaticTextureName = StaticTextureName("carts:spawner_back");

/// Set while a cart spawned by this spawner may still be on top of it.
pub(super) const SPAWNED_OCCUPIED_VARIANT_BIT: u16 = 4;
/// Short-lived lock held during a spawn attempt.
pub(super) const SPAWNER_LOCKING_VARIANT_BIT: u16 = 8;

const LOCK_SPIN_ATTEMPTS: u32 = 100;
const LOCK_SPIN_INTERVAL: Duration = Duration::from_millis(10);

const SPAWNER_INVENTORY_NAME: &str = "spawner_inv";
const SPAWNER_INVENTORY_SIZE: (u32, u32) = (1, 8);
const CART_ITEM_NAME: &str = "carts:minecart";
const SPAWNED_CART_LENGTH: u32 = 8;

// TODO: support filters on block inventories (e.g. an `allowed items` predicate checked on
// insertion by popups and by automation), so that this inventory can be restricted to carts.
// Until then, the spawner just ignores anything that isn't a cart.

pub(super) fn register_cart_spawner(game_builder: &mut GameBuilder) -> Result<()> {
    include_texture_bytes!(
        game_builder,
        SPAWNER_ARROW_TEX,
        "textures/spawner_arrow_point_up.png"
    )?;
    include_texture_bytes!(
        game_builder,
        SPAWNER_FRONT_TEX,
        "textures/spawner_front.png"
    )?;
    include_texture_bytes!(game_builder, SPAWNER_BACK_TEX, "textures/spawner_back.png")?;

    let appearance = point_facing_appearance(game_builder, Color::Orange)?;
    game_builder.add_block(
        BlockBuilder::new(CART_SPAWNER)
            .set_display_name("Cart spawner")
            .set_static_hover_text("Cart spawner")
            .add_block_group(BRITTLE)
            .set_appearance(appearance)
            // We use variant bits for the occupied/locking state
            .force_disable_track_placer()
            .add_interact_key_menu_entry("", "Open inventory")
            .set_extended_data_initializer(Box::new(|_, _, _| {
                let mut data = ExtendedData::default();
                data.inventory_mut(SPAWNER_INVENTORY_NAME.to_string(), SPAWNER_INVENTORY_SIZE);
                Ok(Some(data))
            }))
            .add_modifier(|block| {
                block.interact_key_handler = Some(Box::new(|ctx, coord, _| {
                    ctx.initiator().try_with_player(|p| {
                        Ok(ctx
                            .new_popup()
                            .title("Cart spawner")
                            .inventory_view_block(
                                SPAWNER_INVENTORY_NAME,
                                "Carts:",
                                SPAWNER_INVENTORY_SIZE,
                                coord,
                                SPAWNER_INVENTORY_NAME.to_string(),
                                true,
                                true,
                                false,
                            )?
                            .inventory_view_stored(
                                "player_inv",
                                "Player inventory:",
                                p.main_inventory(),
                                true,
                                true,
                            )?)
                    })
                }));
                block.tap_handler_full = Some(Box::new(|ctx, coord, _tool| {
                    if let Err(e) = tap_spawner(ctx, coord) {
                        ctx.initiator().send_chat_message(
                            ChatMessage::new_server_message(format!("Can't spawn cart: {e}"))
                                .with_color(SERVER_ERROR_COLOR),
                        )?;
                    }
                    Ok(Default::default())
                }));
            }),
    )?;
    Ok(())
}

// Copied from autobuild/machines.rs (point_facing_appearance), since the two will diverge.
fn point_facing_appearance(
    game_builder: &mut GameBuilder,
    color: Color,
) -> Result<BlockAppearanceBuilder> {
    let machine_side = Color::colorize_to_texture(&color, game_builder, SPAWNER_ARROW_TEX)?;
    let machine_front = Color::colorize_to_texture(&color, game_builder, SPAWNER_FRONT_TEX)?;
    let machine_back = Color::colorize_to_texture(&color, game_builder, SPAWNER_BACK_TEX)?;
    Ok(CubeAppearanceBuilder::new()
        .set_individual_textures(
            machine_side.with_transform(TextureTransform::RotateCounterClockwise),
            machine_side.with_transform(TextureTransform::RotateClockwise),
            machine_side.with_transform(TextureTransform::Rotate180),
            machine_side.with_transform(TextureTransform::Rotate180),
            // visual back and front are flipped - we want the spawner to face the direction
            // the player faces - the sign flip in CompassDirection is also correct as a result
            &machine_back,
            &machine_front,
        )
        .set_rotate_laterally()
        .into())
}

enum LockOutcome {
    /// We hold the locking bit and have removed a cart from the inventory. The block ID is the
    /// block as it was when locked (including its facing direction).
    Locked(BlockId, ItemStack),
    /// Another spawn attempt is in flight
    Busy,
    /// A previously spawned cart still holds this spawner
    Occupied,
    NoCart,
    /// The block at that coordinate is not the spawner we expected
    Invalid,
}

fn try_lock(
    ctx: &HandlerContext,
    coord: BlockCoordinate,
    expected: BlockId,
) -> Result<LockOutcome> {
    ctx.game_map().mutate_block_atomically(coord, |block, ext| {
        if !block.equals_ignore_variant(expected) {
            return Ok(LockOutcome::Invalid);
        }
        let variant = block.variant();
        if variant & SPAWNER_LOCKING_VARIANT_BIT != 0 {
            return Ok(LockOutcome::Busy);
        }
        if variant & SPAWNED_OCCUPIED_VARIANT_BIT != 0 {
            return Ok(LockOutcome::Occupied);
        }
        let cart = ext
            .as_mut()
            .and_then(|ext| ext.inventories.get_mut(SPAWNER_INVENTORY_NAME))
            .and_then(|inv| {
                inv.contents_mut()
                    .iter_mut()
                    .find(|slot| slot.try_item_name() == Some(CART_ITEM_NAME))
                    .and_then(|slot| slot.take_items(Some(1)))
            });
        let Some(cart) = cart else {
            return Ok(LockOutcome::NoCart);
        };
        *block = block.with_variant_unchecked(variant | SPAWNER_LOCKING_VARIANT_BIT);
        Ok(LockOutcome::Locked(*block, cart))
    })
}

/// Drops the locking bit, and either marks the spawner occupied (if a cart was spawned) or
/// returns the cart item to the inventory.
fn finish_spawn(
    ctx: &HandlerContext,
    coord: BlockCoordinate,
    expected: BlockId,
    spawned: bool,
    refund: Option<ItemStack>,
) -> Result<()> {
    ctx.game_map().mutate_block_atomically(coord, |block, ext| {
        if !block.equals_ignore_variant(expected) {
            // The spawner was replaced mid-spawn; there is nothing left to unlock
            tracing::warn!("Cart spawner at {coord:?} changed during spawn");
            return Ok(());
        }
        let mut variant = block.variant() & !SPAWNER_LOCKING_VARIANT_BIT;
        if spawned {
            variant |= SPAWNED_OCCUPIED_VARIANT_BIT;
        }
        *block = block.with_variant_unchecked(variant);
        if let Some(stack) = refund {
            let leftover = ext
                .get_or_insert_default()
                .inventory_mut(SPAWNER_INVENTORY_NAME.to_string(), SPAWNER_INVENTORY_SIZE)
                .try_insert(stack);
            if leftover.is_some() {
                tracing::warn!("Cart spawner at {coord:?} was full when refunding a cart");
            }
        }
        Ok(())
    })
}

fn tap_spawner(ctx: &HandlerContext, coord: BlockCoordinate) -> Result<()> {
    // Get everything fallible done before taking the lock, so we don't strand it
    let config = ctx
        .extension::<CartsGameBuilderExtension>()
        .context("Missing carts extension")?;
    let expected = ctx.game_map().get_block(coord)?;
    let direction = CompassDirection::from_rotation_variant(expected.variant());
    let (dx, dz) = direction.to_delta_xz();
    let rail_pos = coord
        .try_delta(dx, 0, dz)
        .context("Spawner is at the edge of the map")?;

    let mut outcome = LockOutcome::Busy;
    for _ in 0..LOCK_SPIN_ATTEMPTS {
        outcome = try_lock(ctx, coord, expected)?;
        if !matches!(outcome, LockOutcome::Busy) {
            break;
        }
        std::thread::sleep(LOCK_SPIN_INTERVAL);
    }
    let (locked_block, cart) = match outcome {
        LockOutcome::Locked(block, cart) => (block, cart),
        LockOutcome::Busy => bail!("Spawner is busy"),
        LockOutcome::Occupied => bail!("A cart is still on the spawner"),
        LockOutcome::NoCart => bail!("No cart in the spawner inventory"),
        LockOutcome::Invalid => bail!("Spawner changed"),
    };

    let spawn_result = spawn_cart(
        &config,
        ctx,
        "",
        SPAWNED_CART_LENGTH,
        rail_pos,
        direction,
        Some(PendingAction::SpawnerRelease(coord, locked_block)),
    );
    let spawned = matches!(spawn_result, Ok(Some(_)));
    let refund = if spawned { None } else { Some(cart) };
    // Always unlock, regardless of how spawning went
    finish_spawn(ctx, coord, expected, spawned, refund)?;
    match spawn_result? {
        Some(_) => Ok(()),
        None => bail!("No track in front of the spawner"),
    }
}

/// Clears the occupied bit of a spawner, once the cart that spawned from it has left.
pub(super) fn spawner_release(
    ctx: &HandlerContext,
    coord: BlockCoordinate,
    expected: BlockId,
) -> Result<()> {
    // If a spawn attempt is still in flight, it hasn't yet decided whether to mark the spawner as
    // occupied. Wait for that, or we'd clear a bit that's about to be set and wedge the spawner.
    for _ in 0..LOCK_SPIN_ATTEMPTS {
        let done = ctx.game_map().mutate_block_atomically(coord, |block, _| {
            if !block.equals_ignore_variant(expected) {
                return Ok(true);
            }
            let variant = block.variant();
            if variant & SPAWNER_LOCKING_VARIANT_BIT != 0 {
                return Ok(false);
            }
            *block = block.with_variant_unchecked(variant & !SPAWNED_OCCUPIED_VARIANT_BIT);
            Ok(true)
        })?;
        if done {
            return Ok(());
        }
        std::thread::sleep(LOCK_SPIN_INTERVAL);
    }
    bail!("Timed out waiting to release spawner at {coord:?}")
}
