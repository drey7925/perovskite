use std::sync::Arc;

use anyhow::Result;

use super::{client_ui::Popup, inventory::InventoryKey, GameState};

/// Contains various callbacks that can be used to configure the game, but don't
/// have a specific place elsewhere in the codebase
#[non_exhaustive]
pub struct GameBehaviors {
    /// Creates a [Popup] for the inventory of a player that's
    /// entering the game.
    pub make_inventory_popup:
        Box<dyn Fn(Arc<GameState>, String, InventoryKey) -> Result<Popup> + Send + Sync + 'static>,
}
impl Default for GameBehaviors {
    fn default() -> Self {
        Self {
            make_inventory_popup: Box::new(defaults::make_inventory_popup),
        }
    }
}

mod defaults {
    use super::*;
    /// A very minimal implementation of the inventory popup.
    pub(crate) fn make_inventory_popup(
        game_state: Arc<GameState>,
        _player_name: String,
        main_inventory_key: InventoryKey,
    ) -> Result<Popup> {
        Popup::new(game_state)
            .title("Inventory")
            .label("Player inventory:")
            .inventory_view_stored("main", main_inventory_key, true, true)
    }
}
