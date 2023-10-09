use std::{collections::HashSet, sync::Arc, time::Duration};

use anyhow::Result;
use perovskite_core::constants::permissions::ELIGIBLE_PREFIX;

use super::{client_ui::Popup, inventory::InventoryKey, GameState};

pub trait InventoryPopupProvider: Send + Sync + 'static {
    fn make_inventory_popup(
        &self,
        game_state: Arc<GameState>,
        player_name: String,
        player_permissions: Vec<String>,
        main_inventory: InventoryKey,
    ) -> Result<Popup>;
}

/// Contains various callbacks that can be used to configure the game, but don't
/// have a specific place elsewhere in the codebase
#[non_exhaustive]
pub struct GameBehaviors {
    /// Creates a [Popup] for the inventory of a player that's
    /// entering the game.
    pub make_inventory_popup: Box<dyn InventoryPopupProvider>,
    /// The length of an in-game day, in terms of real-world duration.
    pub day_length: Duration,
    /// Permissions that can be granted to a player.
    pub defined_permissions: HashSet<String>,
    /// Permissions that should be granted by default to new players.
    pub default_permissions: HashSet<String>,
    /// Permissions that are ambiently applied to all players. These are not
    /// stored in the player database; i.e. if a permission is removed from this
    /// list, then existing players will lose it.
    pub ambient_permissions: HashSet<String>,
}
impl GameBehaviors {
    pub(crate) fn has_defined_permission(&self, permission: &str) -> bool {
        if let Some(eligible_permission) = permission.strip_prefix(ELIGIBLE_PREFIX) {
            self.defined_permissions.contains(eligible_permission)            
        } else {
            self.defined_permissions.contains(permission)
        }
    }
}
impl Default for GameBehaviors {
    fn default() -> Self {
        use perovskite_core::constants::permissions::*;
        Self {
            make_inventory_popup: Box::new(defaults::DefaultInventoryPopupProvider),
            day_length: Duration::from_secs(24 * 60),
            defined_permissions: ALL_PERMISSIONS.iter().map(|x| x.to_string()).collect(),
            default_permissions: HashSet::from([
                DIG_PLACE.to_string(),
                PUNCH.to_string(),
                FAST_MOVE.to_string(),
                LOG_IN.to_string(),
            ]),
            // TODO clean these up - for testing only
            // TODO find a way to give the initial player administrative permissions
            ambient_permissions: HashSet::from([CREATIVE.to_string(), GRANT.to_string()]),
        }
    }
}

mod defaults {
    use super::*;

    pub(crate) struct DefaultInventoryPopupProvider;
    impl InventoryPopupProvider for DefaultInventoryPopupProvider {
        fn make_inventory_popup(
            &self,
            game_state: Arc<GameState>,
            _player_name: String,
            _player_permissions: Vec<String>,
            main_inventory_key: InventoryKey,
        ) -> Result<Popup> {
            Popup::new(game_state)
                .title("Inventory")
                .label("Player inventory:")
                .inventory_view_stored("main", main_inventory_key, true, true)
        }
    }

}
