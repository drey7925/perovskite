use crate::blocks::{BlockBuilder, CubeAppearanceBuilder};
use crate::default_game::basic_blocks::ores::IRON_INGOT;
use crate::default_game::recipes::RecipeSlot;
use crate::default_game::{item_groups, DefaultGameBuilderExtension};
use crate::game_builder::{GameBuilder, StaticBlockName, StaticTextureName};
use crate::include_texture_bytes;
use anyhow::Result;
use perovskite_core::chat::{ChatMessage, SERVER_ERROR_COLOR};
use perovskite_core::constants::permissions;
use perovskite_core::protocol;
use perovskite_core::protocol::items::item_stack::QuantityType;
use perovskite_server::game_state::blocks::ExtendedData;
use perovskite_server::game_state::client_ui::UiElementContainer;
use perovskite_server::game_state::items::ItemStack;

/// Unlocked chest.
pub const CHEST: StaticBlockName = StaticBlockName("default:chest");
/// Locked chest.
pub const LOCKED_CHEST: StaticBlockName = StaticBlockName("default:locked_chest");

const CHEST_SIDE_TEXTURE: StaticTextureName = StaticTextureName("default:chest_side");
const CHEST_TOP_TEXTURE: StaticTextureName = StaticTextureName("default:chest_top");
const CHEST_FRONT_TEXTURE: StaticTextureName = StaticTextureName("default:chest_front");
const LOCKED_CHEST_FRONT_TEXTURE: StaticTextureName =
    StaticTextureName("default:chest_front_locked");
pub(crate) fn register_chest(game_builder: &mut GameBuilder) -> Result<()> {
    include_texture_bytes!(game_builder, CHEST_SIDE_TEXTURE, "textures/chest_side.png")?;
    include_texture_bytes!(game_builder, CHEST_TOP_TEXTURE, "textures/chest_top.png")?;
    include_texture_bytes!(
        game_builder,
        CHEST_FRONT_TEXTURE,
        "textures/chest_front.png"
    )?;
    include_texture_bytes!(
        game_builder,
        LOCKED_CHEST_FRONT_TEXTURE,
        "textures/chest_front_locked.png"
    )?;

    game_builder.add_block(
        BlockBuilder::new(CHEST)
            .set_cube_appearance(
                CubeAppearanceBuilder::new()
                    .set_individual_textures(
                        CHEST_SIDE_TEXTURE,
                        CHEST_SIDE_TEXTURE,
                        CHEST_TOP_TEXTURE,
                        CHEST_TOP_TEXTURE,
                        CHEST_FRONT_TEXTURE,
                        CHEST_SIDE_TEXTURE,
                    )
                    .set_rotate_laterally(),
            )
            .set_display_name("Unlocked chest")
            .add_interact_key_menu_entry("", "Open chest")
            .add_modifier(Box::new(|bt| {
                bt.interact_key_handler = Some(Box::new(|ctx, coord, _| match ctx.initiator() {
                    perovskite_server::game_state::event::EventInitiator::Player(p) => {
                        Ok(Some(make_chest_popup(&ctx, coord, p)?))
                    }
                    _ => Ok(None),
                }));
            })),
    )?;

    const LOCKED_CHEST_OWNER: &str = "default:locked_chest:owner";

    game_builder.add_block(
        BlockBuilder::new(LOCKED_CHEST)
            .set_cube_appearance(
                CubeAppearanceBuilder::new()
                    .set_individual_textures(
                        CHEST_SIDE_TEXTURE,
                        CHEST_SIDE_TEXTURE,
                        CHEST_TOP_TEXTURE,
                        CHEST_TOP_TEXTURE,
                        LOCKED_CHEST_FRONT_TEXTURE,
                        CHEST_SIDE_TEXTURE,
                    )
                    .set_rotate_laterally(),
            )
            .set_display_name("Locked chest")
            .add_interact_key_menu_entry("", "Open chest")
            .add_modifier(Box::new(|bt| {
                bt.interact_key_handler = Some(Box::new(|ctx, coord, _| match ctx.initiator() {
                    perovskite_server::game_state::event::EventInitiator::Player(p) => {
                        let (_, owner) =
                            ctx.game_map().get_block_with_extended_data(coord, |data| {
                                Ok(data.simple_data.get(LOCKED_CHEST_OWNER).cloned())
                            })?;

                        let owner_matches = if let Some(owner) = &owner {
                            owner == p.player.name()
                        } else {
                            true
                        };

                        if owner_matches
                            || p.player
                                .has_permission(permissions::BYPASS_INVENTORY_CHECKS)
                        {
                            Ok(Some(make_chest_popup(&ctx, coord, p)?))
                        } else {
                            // unwrap is safe - if owner were none, we would have given access to the chest
                            p.player.send_chat_message(
                                ChatMessage::new_server_message(format!(
                                    "Only {} can open this chest",
                                    owner.unwrap()
                                ))
                                .with_color(SERVER_ERROR_COLOR),
                            )?;
                            Ok(None)
                        }
                    }
                    _ => Ok(None),
                }))
            }))
            .set_extended_data_initializer(Box::new(|ctx, _coord, _stack| {
                Ok(ctx.initiator().player_name().map(|name| {
                    let mut data = ExtendedData::default();
                    data.simple_data
                        .insert(LOCKED_CHEST_OWNER.to_string(), name.to_string());
                    data
                }))
            })),
    )?;
    game_builder
        .builder_extension_mut::<DefaultGameBuilderExtension>()
        .crafting_recipes
        .register_recipe(super::recipes::RecipeImpl {
            slots: [
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Empty,
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
            ],
            result: ItemStack {
                proto: protocol::items::ItemStack {
                    item_name: CHEST.0.to_string(),
                    quantity: 1,
                    current_wear: 0,
                    quantity_type: Some(QuantityType::Stack(256)),
                },
            },
            shapeless: false,
            metadata: (),
        });
    game_builder
        .builder_extension_mut::<DefaultGameBuilderExtension>()
        .crafting_recipes
        .register_recipe(super::recipes::RecipeImpl {
            slots: [
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Exact(IRON_INGOT.0.to_string()),
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
                RecipeSlot::Group(item_groups::WOOD_PLANKS.to_string()),
            ],
            result: ItemStack {
                proto: protocol::items::ItemStack {
                    item_name: LOCKED_CHEST.0.to_string(),
                    quantity: 1,
                    current_wear: 0,
                    quantity_type: Some(QuantityType::Stack(256)),
                },
            },
            shapeless: false,
            metadata: (),
        });
    Ok(())
}

fn make_chest_popup(
    ctx: &perovskite_server::game_state::event::HandlerContext<'_>,
    coord: perovskite_core::coordinates::BlockCoordinate,
    p: &perovskite_server::game_state::event::PlayerInitiator<'_>,
) -> Result<perovskite_server::game_state::client_ui::Popup, anyhow::Error> {
    Ok(ctx
        .new_popup()
        .title("Chest")
        .inventory_view_block(
            "chest_inv",
            "Chest contents:",
            (4, 8),
            coord,
            "chest_inv".to_string(),
            true,
            true,
            false,
        )?
        .inventory_view_stored(
            "player_inv",
            "Player inventory:",
            p.player.main_inventory(),
            true,
            true,
        )?)
}
