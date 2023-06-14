use std::{
    collections::{HashMap, HashSet},
    sync::{atomic::AtomicU64, Arc},
};

use super::{
    inventory::{
        BorrowedStack, InventoryKey, InventoryView, InventoryViewId, TypeErasedInventoryView,
    },
    GameState,
};
use anyhow::{bail, Result};
use cuberef_core::protocol::ui as proto;

pub type TextField = proto::TextField;
pub type Button = proto::Button;
/// Callbacks for inventory views will receive this hashmap,
/// mapping all inventory views using the form_key passed when calling
/// inventory_view_stored or inventory_view_transient
pub type CallbackContext<'a> = HashMap<String, &'a mut dyn TypeErasedInventoryView>;

/// Choice of a widget. This API is subject to change.
pub enum UiElement {
    /// A static label
    Label(String),
    /// A textfield with the given parameters
    TextField(TextField),
    /// A button that the user can click.
    /// The provided string is the name passed in the event
    Button(Button),
    /// An inventory view: key (within this popup), the inventory view id
    InventoryView(String, InventoryViewId),
}

/// A server-side representation of a custom UI drawn on the client's screen.
/// The client and server network code will serialize popups and display them on
/// the client's screen. Clicking buttons will lead to events firing at the server.
pub struct Popup<'a> {
    id: u64,
    game_state: Arc<GameState>,
    widgets: Vec<UiElement>,
    inventory_views: HashMap<String, InventoryView<CallbackContext<'a>>>,
    interested_stored_inventories: HashSet<InventoryKey>,
    inventory_update_callback: Option<Box<dyn Fn(CallbackContext) + Send + Sync>>,
}
impl<'a> Popup<'a> {
    /// Creates a new popup. Until it's sent to a player, it is inert and has no effects
    pub fn new(game_state: Arc<GameState>) -> Self {
        Popup {
            id: NEXT_POPUP_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            game_state,
            widgets: vec![],
            inventory_views: HashMap::new(),
            interested_stored_inventories: HashSet::new(),
            inventory_update_callback: None,
        }
    }
    /// Adds a new label to this popup. At the moment, the layout is still TBD.
    pub fn label(mut self, label: String) -> Self {
        self.widgets.push(UiElement::Label(label));
        self
    }
    /// Adds a new text field to this popup. At the moment, the layout is still TBD.
    pub fn text_field(
        mut self,
        key: String,
        label: String,
        initial: String,
        enabled: bool,
    ) -> Self {
        self.widgets.push(UiElement::TextField(TextField {
            key,
            label,
            initial,
            enabled,
        }));
        self
    }
    /// Adds a new button to this popup.
    pub fn button(mut self, key: String, label: String, enabled: bool) -> Self {
        self.widgets.push(UiElement::Button(Button {
            key,
            label,
            enabled,
        }));
        self
    }
    /// Adds an inventory view backed by a stored inventory managed in the inventory manager
    /// If can_place or can_take are true, then player actions are directly written back to the
    /// game database.
    pub fn inventory_view_stored(
        mut self,
        form_key: String,
        inventory_key: InventoryKey,
        can_place: bool,
        can_take: bool,
    ) -> Result<Self> {
        let view =
            InventoryView::new_stored(inventory_key, self.game_state.clone(), can_place, can_take)?;
        self.widgets
            .push(UiElement::InventoryView(form_key.clone(), view.id));
        self.interested_stored_inventories.insert(inventory_key);
        match self.inventory_views.insert(form_key.clone(), view) {
            Some(view) => {
                bail!(
                    "form_key {} already registered for view {:?}",
                    form_key,
                    view.id
                );
            }
            None => Ok(self),
        }
    }
    /// Adds a transient inventory view that does not store items anywhere (but only exists while the
    /// popup is open).
    ///
    /// initial_contents must either have length equal to dimensions.0 * dimensions.1, or be zero length
    pub fn inventory_view_transient(
        mut self,
        form_key: String,
        dimensions: (u32, u32),
        initial_contents: Vec<Option<BorrowedStack>>,
        can_place: bool,
        can_take: bool,
    ) -> Result<Self> {
        let view = InventoryView::new_transient(
            self.game_state.clone(),
            dimensions,
            initial_contents,
            can_place,
            can_take,
        )?;
        self.widgets
            .push(UiElement::InventoryView(form_key.clone(), view.id));
        match self.inventory_views.insert(form_key.clone(), view) {
            Some(view) => {
                bail!(
                    "form_key {} already registered for view {:?}",
                    form_key,
                    view.id
                );
            }
            None => Ok(self),
        }
    }
    /// Sets a function that will be called after any inventories are updated.
    ///
    /// Exact parameters are still tbd
    ///
    /// This replaces any previously set function
    pub fn set_inventory_action_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(CallbackContext) + Send + Sync + 'static,
    {
        self.inventory_update_callback = Some(Box::new(callback));
        self
    }

    pub(crate) fn invoke_inventory_action_callback(&mut self) {
        if let Some(callback) = &self.inventory_update_callback {
            let context = self
                .inventory_views
                .iter_mut()
                .map(|(k, v)| (k.clone(), v as &mut dyn TypeErasedInventoryView))
                .collect();
            callback(context);
        }
    }

    pub(crate) fn to_proto(&self) -> proto::PopupDescription {
        let elements = self
            .widgets
            .iter()
            .map(|x| {
                let element = match x {
                    UiElement::Label(label) => proto::ui_element::Element::Label(label.clone()),
                    UiElement::TextField(field) => {
                        proto::ui_element::Element::TextField(field.clone())
                    }
                    UiElement::Button(button) => proto::ui_element::Element::Button(button.clone()),
                    UiElement::InventoryView(key, view_id) => {
                        // we can expect() here since presence in the map is an invariant
                        let view = self
                            .inventory_views
                            .get(key)
                            .expect("No inventory view found for provided key");
                        proto::ui_element::Element::Inventory(proto::InventoryView {
                            inventory_key: view_id.0,
                            can_place: view.can_place,
                            can_take: view.can_take,
                        })
                    }
                };
                proto::UiElement {
                    element: Some(element),
                }
            })
            .collect();
        proto::PopupDescription {
            popup_id: self.id,
            element: elements,
        }
    }

    pub(crate) fn inventory_views(
        &self,
    ) -> impl Iterator<Item = &InventoryView<CallbackContext<'a>>> {
        self.inventory_views.values()
    }
}

static NEXT_POPUP_ID: AtomicU64 = AtomicU64::new(1);
