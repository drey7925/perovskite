use std::{borrow::Cow, collections::HashSet, sync::Arc, time::Duration};

use anyhow::Result;
use cgmath::{vec3, Vector3};
use itertools::Itertools;
use perovskite_core::{
    chat::ChatMessage,
    constants::permissions::{ELIGIBLE_PREFIX, FAST_MOVE, FLY, NOCLIP},
};
use tonic::async_trait;

use super::{
    client_ui::Popup, event::HandlerContext, inventory::InventoryKey, player::Player, GameState,
};

/// Creates a [Popup] that will be shown when the player presses the inventory key
pub trait InventoryPopupProvider: Send + Sync + 'static {
    fn make_inventory_popup(
        &self,
        game_state: Arc<GameState>,
        player_name: String,
        player_permissions: Vec<String>,
        main_inventory: InventoryKey,
    ) -> Result<Popup>;
}

/// Generic trait used for various sundry handlers that need to be run in an async context
#[async_trait]
pub trait GenericAsyncHandler<T, U>: Send + Sync + 'static {
    async fn handle(&self, req: &T, context: HandlerContext<'_>) -> Result<U>;
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
    /// Users that will get all permissions
    pub super_users: HashSet<String>,

    pub on_player_join: Box<dyn GenericAsyncHandler<Player, ()>>,
    pub on_player_leave: Box<dyn GenericAsyncHandler<Player, ()>>,

    pub spawn_location: Box<dyn Fn(&str) -> Vector3<f64> + Send + Sync + 'static>,
}
impl GameBehaviors {
    pub(crate) fn has_defined_permission(&self, permission: &str) -> bool {
        if let Some(eligible_permission) = permission.strip_prefix(ELIGIBLE_PREFIX) {
            self.defined_permissions.contains(eligible_permission)
        } else {
            self.defined_permissions.contains(permission)
        }
    }
    pub(crate) fn ambient_permissions(&self, name: &str) -> Cow<HashSet<String>> {
        let mut permissions = Cow::Borrowed(&self.ambient_permissions);
        if self.super_users.contains(name) {
            Cow::to_mut(&mut permissions).extend(
                self.defined_permissions
                    .iter()
                    .map(|x| ELIGIBLE_PREFIX.to_owned() + x),
            );
            Cow::to_mut(&mut permissions).extend([
                FLY.to_owned(),
                FAST_MOVE.to_owned(),
                NOCLIP.to_owned(),
            ])
        }
        permissions
    }
}
impl Default for GameBehaviors {
    fn default() -> Self {
        use perovskite_core::constants::permissions::*;
        Self {
            make_inventory_popup: Box::new(defaults::DefaultInventoryPopupProvider),
            day_length: Duration::from_secs(24 * 60),
            defined_permissions: ALL_PERMISSIONS.iter().map(|x| x.to_string()).collect(),
            // All permissions are granted until the game is more polished.
            // In particular, we need to ensure that players don't spawn underground, and that
            // we have a way to bootstrap administrative permissions.
            default_permissions: HashSet::from([
                FAST_MOVE.to_string(),
                CREATIVE.to_string(),
                DIG_PLACE.to_string(),
                TAP_INTERACT.to_string(),
                LOG_IN.to_string(),
                WORLD_STATE.to_string(),
                INVENTORY.to_string(),
                CHAT.to_string(),
            ]),
            ambient_permissions: HashSet::from([]),
            super_users: HashSet::from([]),
            on_player_join: Box::new(PlayerJoinHandlerImpl),
            on_player_leave: Box::new(SendChatMessageHandlerImpl::new(|player: &Player, _| {
                ChatMessage::new_server_message(format!("{} left the game.", player.name()))
            })),
            spawn_location: Box::new(|_| vec3(0.0, 30.0, 0.0)),
        }
    }
}

pub struct SendChatMessageHandlerImpl<T> {
    message_generator: Box<dyn Fn(&T, &HandlerContext) -> ChatMessage + 'static + Send + Sync>,
}
impl<T> SendChatMessageHandlerImpl<T> {
    pub fn new(
        message_generator: impl Fn(&T, &HandlerContext) -> ChatMessage + 'static + Send + Sync,
    ) -> Self {
        Self {
            message_generator: Box::new(message_generator),
        }
    }
}
#[async_trait]
impl<T: Send + Sync + 'static> GenericAsyncHandler<T, ()> for SendChatMessageHandlerImpl<T> {
    async fn handle(&self, req: &T, context: HandlerContext<'_>) -> Result<()> {
        let message = (self.message_generator)(req, &context);
        context.game_state.chat().broadcast_chat_message(message)?;
        Ok(())
    }
}

pub struct PlayerJoinHandlerImpl;
#[async_trait]
impl GenericAsyncHandler<Player, ()> for PlayerJoinHandlerImpl {
    async fn handle(&self, req: &Player, context: HandlerContext<'_>) -> Result<()> {
        let broadcast_message =
            ChatMessage::new_server_message(format!("{} joined the game.", req.name()));
        context.chat().broadcast_chat_message(broadcast_message)?;

        let mut connected_players = vec![];
        context.player_manager().for_all_connected_players(|p| {
            connected_players.push(p.name().to_string());
            Ok(())
        })?;
        let individual_message = ChatMessage::new_server_message(format!(
            "Welcome! Currently connected players: {}",
            connected_players.iter().sorted().join(", ")
        ));
        req.send_chat_message_async(individual_message).await?;
        #[cfg(debug_assertions)]
        {
            req.send_chat_message_async(
                ChatMessage::new_server_message(
                    "Debug mode is enabled. Server will be slow and may be crashy.",
                )
                .with_color((255, 0, 0)),
            )
            .await?;
        }
        Ok(())
    }
}

mod defaults {
    use crate::game_state::client_ui::UiElementContainer;

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
                .inventory_view_stored("main", "Player inventory:", main_inventory_key, true, true)
        }
    }
}
