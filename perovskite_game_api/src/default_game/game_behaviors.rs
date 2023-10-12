use anyhow::Result;
use parking_lot::RwLock;
use perovskite_core::{constants::permissions::CREATIVE, protocol::items::item_stack};
use perovskite_server::game_state::{
    client_ui::{Popup, PopupAction, PopupResponse},
    game_behaviors::InventoryPopupProvider,
    inventory::{InventoryKey, VirtualOutputCallbacks},
    items::{Item, ItemStack},
    GameState,
};
use std::{collections::HashSet, sync::Arc};

use super::{recipes::RecipeBook, DefaultGameBuilder};

pub(crate) fn register_game_behaviors(game_builder: &mut DefaultGameBuilder) -> Result<()> {
    let recipe_book = game_builder.crafting_recipes.clone();
    let behaviors = game_builder.inner.inner.game_behaviors_mut();
    behaviors.make_inventory_popup = Box::new(DefaultGameInventoryPopupProvider { recipe_book });
    behaviors.super_users = game_builder.settings.super_users.iter().cloned().collect();

    Ok(())
}

struct CreativeInvState {
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
            .map(|x| x.make_max_stack())
            .collect::<Vec<ItemStack>>();
        items.sort_by(|x, y| x.proto.item_name.cmp(&y.proto.item_name));

        let creative_state_for_peek = Arc::new(RwLock::new(CreativeInvState { items, offset: 0 }));
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
                stack
            }),
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
                result.map(|x| x.result.clone())
            })
        };

        let crafting_callbacks = VirtualOutputCallbacks {
            peek: craft_peek,
            take: craft_take,
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
                                if count >= 1 {
                                    let mut lock = creative_state_for_update.write();
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
                .text_field("count", "Item count: ", "256", true)
                .button("update_btn", "Update", true)
                .button("left", "<--", true)
                .button("right", "-->", true)
                .label("Creative items:")
                .inventory_view_virtual_output("creative", (4, 8), creative_inv_callbacks, false)?
                .label("Crafting input:")
                .inventory_view_transient("craft_in", (3, 3), vec![], true, true)?
                .label("Crafting output:")
                .inventory_view_virtual_output("craft_out", (1, 1), crafting_callbacks, true)?
                .label("Player inventory:")
                .inventory_view_stored("main", main_inventory_key, true, true)?
                .set_button_callback(button_callback))
        } else {
            Ok(Popup::new(game_state)
                .title("Inventory")
                .label("Crafting input:")
                .inventory_view_transient("craft_in", (3, 3), vec![], true, true)?
                .label("Crafting output:")
                .inventory_view_virtual_output("craft_out", (1, 1), crafting_callbacks, true)?
                .label("Player inventory:")
                .inventory_view_stored("main", main_inventory_key, true, true)?
                .set_button_callback(button_callback))
        }
    }
}
