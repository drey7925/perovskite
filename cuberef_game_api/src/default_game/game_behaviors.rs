use anyhow::Result;
use parking_lot::{Mutex, RwLock};
use std::sync::Arc;

use cuberef_server::game_state::{
    client_ui::{Popup, PopupAction, PopupResponse},
    inventory::{InventoryKey, VirtualOutputCallbacks},
    items::ItemStack,
    GameState,
};

use super::{recipes::RecipeBook, DefaultGameBuilder};

pub(crate) fn register_game_behaviors(game_builder: &mut DefaultGameBuilder) -> Result<()> {
    let recipe_book = game_builder.crafting_recipes.clone();
    game_builder
        .inner
        .inner
        .game_behaviors_mut()
        .make_inventory_popup = Box::new(move |game_state, _, inv_key| {
        make_inventory_popup(game_state, inv_key, recipe_book.clone())
    });

    Ok(())
}

fn make_inventory_popup(
    game_state: Arc<GameState>,
    main_inventory_key: InventoryKey,
    crafting_recipes: Arc<RecipeBook<9>>,
) -> Result<Popup> {
    // Generally all the callbacks would reference a more interesting arc that backs the inventory
    // For this example, we only need to store one vec
    let mut all_items = game_state
        .item_manager()
        .registered_items()
        .map(|x| Some(ItemStack::new(x, 256)))
        .collect::<Vec<Option<ItemStack>>>();

    // TODO pagination
    all_items.resize_with(32, || None);
    let all_items_for_peek = Arc::new(RwLock::new(all_items));
    let all_items_for_take = all_items_for_peek.clone();
    let all_items_for_update = all_items_for_peek.clone();
    let creative_inv_callbacks = VirtualOutputCallbacks {
        peek: Box::new(move |_| {
            log::info!("Debug only: peeking from creative");
            all_items_for_peek.read().clone()
        }),
        take: Box::new(move |_, slot, count| {
            log::info!("Debug only: taking from creative");
            let mut stack = all_items_for_take.read().get(slot).cloned().flatten();
            if stack.is_some() && count.is_some() {
                stack.as_mut().unwrap().proto.quantity = count.unwrap();
            }
            stack
        }),
    };

    let peek = {
        let crafting_recipes = crafting_recipes.clone();
        let game_state = game_state.clone();
        Box::new(move |ctx: &Popup| {
            let input = ctx
                .inventory_views()
                .get("craft_in")
                .unwrap()
                .peek(ctx)
                .unwrap();
            let result = crafting_recipes.find(&game_state, &input);
            vec![result]
        })
    };
    let take = {
        let game_state = game_state.clone();
        Box::new(move |ctx: &Popup, _slot, _count| {
            let source_view = ctx.inventory_views().get("craft_in").unwrap();
            let input = source_view.peek(ctx).unwrap();
            let result = crafting_recipes.find(&game_state, &input);
            if result.is_some() {
                for i in 0..9 {
                    if input[i].is_some() {
                        source_view.take(ctx, i, Some(1)).unwrap();
                    }
                }
            }
            result
        })
    };

    let crafting_callbacks = VirtualOutputCallbacks { peek, take };

    let button_callback = {
        let game_state = game_state.clone();
        move |response: PopupResponse| {
            if let PopupAction::ButtonClicked(x) = &response.user_action {
                if x == "update_btn" {
                    if let Some(count) = response
                        .textfield_values
                        .get("count")
                        .and_then(|x| x.parse::<u32>().ok())
                    {
                        if count >= 1 {
                            let mut all_items = game_state
                                .item_manager()
                                .registered_items()
                                .map(|x| Some(ItemStack::new(x, count)))
                                .collect::<Vec<Option<ItemStack>>>();
                            all_items.resize_with(32, || None);
                            *all_items_for_update.write() = all_items;
                        }
                    }
                }
            }
        }
    };

    Ok(Popup::new(game_state)
        .title("Inventory")
        .text_field("count", "Item count: ", "256", true)
        .button("update_btn", "Update", true)
        .label("Creative items:")
        .inventory_view_virtual_output("creative", (4, 8), creative_inv_callbacks, false)?
        .label("Crafting input:")
        .inventory_view_transient("craft_in", (3, 3), vec![], true, true)?
        .label("Crafting output:")
        .inventory_view_virtual_output("craft_out", (1, 1), crafting_callbacks, true)?
        .label("Player inventory:")
        .inventory_view_stored("main", main_inventory_key, true, true)?
        .set_button_callback(button_callback))
}
