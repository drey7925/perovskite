use anyhow::Result;
use itertools::Itertools;
use parking_lot::RwLock;
use perovskite_core::{
    chat::{ChatMessage, SERVER_ERROR_COLOR},
    constants::{item_groups::HIDDEN_FROM_CREATIVE, permissions::CREATIVE},
    protocol::items::item_stack,
};
use perovskite_server::game_state::entities::EntityClassId;
use perovskite_server::game_state::inventory::{
    VirtualInputCallbacks, VirtualOutputReturnBehavior,
};
use perovskite_server::game_state::{
    client_ui::{Popup, PopupAction, PopupResponse, UiElementContainer},
    game_behaviors::InventoryPopupProvider,
    inventory::{InventoryKey, VirtualOutputCallbacks},
    items::ItemStack,
    GameState,
};
use std::sync::Arc;

use crate::game_builder::GameBuilder;

use super::{recipes::RecipeBook, DefaultGameBuilderExtension};

pub(crate) fn register_game_behaviors(
    game_builder: &mut GameBuilder,
    player_entity_id: EntityClassId,
) -> Result<()> {
    let extension = game_builder.builder_extension::<DefaultGameBuilderExtension>();
    let recipe_book = extension.crafting_recipes.clone();
    let spawn_location = extension.settings.spawn_location.into();
    let super_users = extension.settings.super_users.iter().cloned().collect();
    {
        let behaviors = game_builder.inner.game_behaviors_mut();

        behaviors.make_inventory_popup =
            Box::new(DefaultGameInventoryPopupProvider { recipe_book });
        behaviors.super_users = super_users;
        behaviors.spawn_location = Box::new(move |_| spawn_location);
        behaviors.player_entity_class = Box::new(move |_| player_entity_id);
    }
    Ok(())
}

struct CreativeInvState {
    all_items: Vec<ItemStack>,
    last_search: String,
    items: Vec<ItemStack>,
    offset: usize,
}
impl CreativeInvState {
    fn current_view(&self) -> Vec<Option<ItemStack>> {
        let mut items: Vec<_> = self
            .items
            .iter()
            .skip(self.offset)
            .take(32)
            .map(|x| Some(x.clone()))
            .collect();
        items.resize_with(32, || None);
        items
    }
}
struct DefaultGameInventoryPopupProvider {
    recipe_book: Arc<RecipeBook<9, ()>>,
}

impl InventoryPopupProvider for DefaultGameInventoryPopupProvider {
    #[allow(clippy::unnecessary_unwrap)]
    fn make_inventory_popup(
        &self,
        game_state: Arc<GameState>,
        _player_name: String,
        permissions: Vec<String>,
        main_inventory_key: InventoryKey,
    ) -> Result<Popup> {
        let has_creative = permissions.iter().any(|x| x == CREATIVE);
        let mut items = game_state
            .item_manager()
            .registered_items()
            .filter(|item| !item.proto.groups.iter().any(|x| x == HIDDEN_FROM_CREATIVE))
            .sorted_by_key(|x| {
                if x.proto.sort_key.is_empty() {
                    &x.proto.short_name
                } else {
                    &x.proto.sort_key
                }
            })
            .map(|x| x.make_max_stack())
            .collect::<Vec<ItemStack>>();

        let creative_state_for_peek = Arc::new(RwLock::new(CreativeInvState {
            all_items: items.clone(),
            last_search: String::new(),
            items,
            offset: 0,
        }));
        let creative_state_for_take = creative_state_for_peek.clone();
        let creative_state_for_update = creative_state_for_peek.clone();
        let creative_inv_callbacks = VirtualOutputCallbacks {
            peek: Box::new(move |_| creative_state_for_peek.read().current_view()),
            take: Box::new(move |_, slot, count| {
                let mut stack = creative_state_for_take
                    .read()
                    .current_view()
                    .get(slot)
                    .cloned()
                    .flatten();
                if stack.is_some() && count.is_some() {
                    stack.as_mut().unwrap().proto.quantity = count.unwrap();
                }
                stack.map(|x| (x, VirtualOutputReturnBehavior::Drop))
            }),
            // Creative items are essentially free; take them back
            return_borrowed: Box::new(|_, _, _, _| Ok(None)),
        };

        let craft_peek = {
            let crafting_recipes = self.recipe_book.clone();
            let game_state = game_state.clone();
            Box::new(move |ctx: &Popup| {
                let input = ctx
                    .inventory_views()
                    .get("craft_in")
                    .unwrap()
                    .peek(ctx)
                    .unwrap();
                let result = crafting_recipes
                    .find(
                        game_state.item_manager(),
                        input.iter().collect::<Vec<_>>().as_slice(),
                    )
                    .map(|x| x.result.clone());
                vec![result]
            })
        };
        let craft_take = {
            let game_state = game_state.clone();
            let recipe_book = self.recipe_book.clone();
            Box::new(move |ctx: &Popup, _slot, _count| {
                let source_view = ctx.inventory_views().get("craft_in").unwrap();
                let input = source_view.peek(ctx).unwrap();
                let result = recipe_book.find(
                    game_state.item_manager(),
                    input.iter().collect::<Vec<_>>().as_slice(),
                );
                if result.is_some() {
                    for (i, item) in input.iter().enumerate() {
                        if item.is_some() {
                            source_view.take(ctx, i, Some(1)).unwrap();
                        }
                    }
                }
                result.map(|x| {
                    (
                        x.result.clone(),
                        VirtualOutputReturnBehavior::ReturnToInventory,
                    )
                })
            })
        };

        let crafting_callbacks = VirtualOutputCallbacks {
            peek: craft_peek,
            take: craft_take,
            // Don't allow items to be returned
            return_borrowed: Box::new(|_, _, _, stack| Ok(Some(stack))),
        };

        let trash_bin_callbacks = VirtualInputCallbacks {
            put: Box::new(|_, _, _| Ok(None)),
        };

        let button_callback = {
            move |response: PopupResponse| {
                if let PopupAction::ButtonClicked(x) = &response.user_action {
                    match x.as_str() {
                        "update_btn" => {
                            if let Some(count) = response
                                .textfield_values
                                .get("count")
                                .and_then(|x| x.parse::<u32>().ok())
                            {
                                let mut lock = creative_state_for_update.write();

                                let query = response
                                    .textfield_values
                                    .get("search")
                                    .cloned()
                                    .unwrap_or_default()
                                    .trim()
                                    .to_string();
                                if lock.last_search != query {
                                    lock.last_search = query.to_string();

                                    let re = match regex::RegexBuilder::new(&query)
                                        .case_insensitive(true)
                                        .build()
                                    {
                                        Ok(x) => x,
                                        Err(regex::Error::Syntax(s)) => {
                                            response
                                                .ctx
                                                .initiator()
                                                .send_chat_message(
                                                    ChatMessage::new_server_message(format!(
                                                        "Invalid search query syntax: {}",
                                                        s
                                                    ))
                                                    .with_color(SERVER_ERROR_COLOR),
                                                )
                                                .unwrap();
                                            regex::RegexBuilder::new(".*").build().unwrap()
                                        }
                                        Err(regex::Error::CompiledTooBig(_)) => {
                                            response
                                                .ctx
                                                .initiator()
                                                .send_chat_message(
                                                    ChatMessage::new_server_message(
                                                        "Search query too large/complex"
                                                            .to_string(),
                                                    )
                                                    .with_color(SERVER_ERROR_COLOR),
                                                )
                                                .unwrap();
                                            regex::RegexBuilder::new(".*").build().unwrap()
                                        }
                                        Err(_) => {
                                            response
                                                .ctx
                                                .initiator()
                                                .send_chat_message(
                                                    ChatMessage::new_server_message(
                                                        "Unknown error parsing search query"
                                                            .to_string(),
                                                    )
                                                    .with_color(SERVER_ERROR_COLOR),
                                                )
                                                .unwrap();
                                            regex::RegexBuilder::new(".*").build().unwrap()
                                        }
                                    };
                                    lock.items = lock
                                        .all_items
                                        .iter()
                                        .filter(|x| {
                                            re.is_match(&x.proto.item_name)
                                                || response
                                                    .ctx
                                                    .item_manager()
                                                    .get_item(x.proto.item_name.as_str())
                                                    .map_or(false, |x| {
                                                        x.proto
                                                            .groups
                                                            .iter()
                                                            .any(|x| re.is_match(x))
                                                            || re.is_match(&x.proto.display_name)
                                                            || re.is_match(&x.proto.short_name)
                                                    })
                                        })
                                        .cloned()
                                        .collect();
                                    lock.offset = 0;
                                }

                                if count >= 1 {
                                    for item in lock.items.iter_mut() {
                                        item.proto.quantity = match item.proto.quantity_type {
                                            Some(item_stack::QuantityType::Stack(x)) => {
                                                count.min(x)
                                            }
                                            _ => 1,
                                        }
                                    }
                                }
                            }
                        }
                        "left" => {
                            let mut lock = creative_state_for_update.write();
                            lock.offset = lock.offset.saturating_sub(32);
                        }
                        "right" => {
                            let mut lock = creative_state_for_update.write();
                            if (lock.offset + 32) < lock.items.len() {
                                lock.offset += 32;
                            }
                        }
                        _ => {
                            tracing::warn!("Unknown button: {:?}", x);
                        }
                    }
                }
            }
        };
        if has_creative {
            Ok(Popup::new(game_state)
                .title("Inventory")
                .text_field("search", "Filter: ", "", true, false)
                .text_field("count", "Creative stack size: ", "256", true, false)
                .button("update_btn", "Update", true, false)
                .side_by_side_layout("Navigation", |p| {
                    Ok(p.button("left", "Prev. page", true, false).button(
                        "right",
                        "Next page",
                        true,
                        false,
                    ))
                })?
                .inventory_view_virtual_output(
                    "creative",
                    "Creative items:",
                    (4, 8),
                    creative_inv_callbacks,
                    false,
                    true,
                )?
                .side_by_side_layout("Crafting", |ui| {
                    ui.inventory_view_transient(
                        "craft_in",
                        "Crafting input:",
                        (3, 3),
                        vec![],
                        true,
                        true,
                    )?
                    .inventory_view_virtual_output(
                        "craft_out",
                        "Crafting output:",
                        (1, 1),
                        crafting_callbacks,
                        true,
                        false,
                    )?
                    .inventory_view_virtual_input(
                        "trash_bin",
                        "Trash bin:",
                        (1, 1),
                        trash_bin_callbacks,
                    )
                })?
                .inventory_view_stored("main", "Player inventory:", main_inventory_key, true, true)?
                .set_button_callback(button_callback))
        } else {
            Ok(Popup::new(game_state)
                .title("Inventory")
                .side_by_side_layout("Crafting", |ui| {
                    ui.inventory_view_transient(
                        "craft_in",
                        "Crafting input:",
                        (3, 3),
                        vec![],
                        true,
                        true,
                    )?
                    .inventory_view_virtual_output(
                        "craft_out",
                        "Crafting output:",
                        (1, 1),
                        crafting_callbacks,
                        true,
                        false,
                    )?
                    .inventory_view_virtual_input(
                        "trash_bin",
                        "Trash bin:",
                        (1, 1),
                        trash_bin_callbacks,
                    )
                })?
                .label("Player inventory:")
                .inventory_view_stored("main", "Player inventory:", main_inventory_key, true, true)?
                .set_button_callback(button_callback))
        }
    }
}
