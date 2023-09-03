use std::{
    collections::{HashMap, HashSet},
    sync::{atomic::AtomicU64, Arc},
};

use super::{
    inventory::{
        BorrowedStack, InventoryKey, InventoryView, InventoryViewId, VirtualOutputCallbacks,
    },
    GameState,
};
use anyhow::{bail, Result};
use perovskite_core::{protocol::ui as proto, coordinates::BlockCoordinate};

pub type TextField = proto::TextField;
pub type Button = proto::Button;
/// Callbacks for inventory views will receive this hashmap,
/// mapping all inventory views using the form_key passed when calling
/// inventory_view_stored or inventory_view_transient

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

#[derive(Clone, PartialEq, Eq)]
pub enum PopupAction {
    /// The popup was closed
    PopupClosed,
    /// A button (added to the UI programmatically) was clicked; its key is given here
    ButtonClicked(String),
}

/// Passed when a callback is invoked
/// Contents TBD
pub struct PopupResponse {
    /// The key of the button that was clicked
    pub user_action: PopupAction,
    /// The values of all textfields
    pub textfield_values: HashMap<String, String>,
}

/// A server-side representation of a custom UI drawn on the client's screen.
/// The client and server network code will serialize popups and display them on
/// the client's screen. Clicking buttons will lead to events firing at the server.
pub struct Popup {
    id: u64,
    title: String,
    game_state: Arc<GameState>,
    widgets: Vec<UiElement>,
    inventory_views: HashMap<String, InventoryView<Popup>>,
    interested_stored_inventories: HashSet<InventoryKey>,
    interested_coordinates: HashSet<BlockCoordinate>,
    inventory_update_callback: Option<Box<dyn Fn(&Popup) + Send + Sync>>,
    button_callback: Option<Box<dyn Fn(PopupResponse) + Send + Sync>>,
}
impl Popup {
    /// Creates a new popup. Until it's sent to a player, it is inert and has no effects
    pub fn new(game_state: Arc<GameState>) -> Self {
        Popup {
            id: NEXT_POPUP_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            title: String::new(),
            game_state,
            widgets: vec![],
            inventory_views: HashMap::new(),
            interested_stored_inventories: HashSet::new(),
            interested_coordinates: HashSet::new(),
            inventory_update_callback: None,
            button_callback: None,
        }
    }
    /// Sets the popup title.
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }
    /// Adds a new label to this popup. At the moment, the layout is still TBD.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.widgets.push(UiElement::Label(label.into()));
        self
    }
    /// Adds a new text field to this popup. At the moment, the layout is still TBD.
    pub fn text_field(
        mut self,
        key: impl Into<String>,
        label: impl Into<String>,
        initial: impl Into<String>,
        enabled: bool,
    ) -> Self {
        self.widgets.push(UiElement::TextField(TextField {
            key: key.into(),
            label: label.into(),
            initial: initial.into(),
            enabled,
        }));
        self
    }
    /// Adds a new button to this popup.
    pub fn button(
        mut self,
        key: impl Into<String>,
        label: impl Into<String>,
        enabled: bool,
    ) -> Self {
        self.widgets.push(UiElement::Button(Button {
            key: key.into(),
            label: label.into(),
            enabled,
        }));
        self
    }
    /// Adds an inventory view backed by a stored inventory managed in the inventory manager
    /// If can_place or can_take are true, then player actions are directly written back to the
    /// game database.
    pub fn inventory_view_stored(
        mut self,
        form_key: impl Into<String>,
        inventory_key: InventoryKey,
        can_place: bool,
        can_take: bool,
    ) -> Result<Self> {
        let form_key = form_key.into();
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
        form_key: impl Into<String>,
        dimensions: (u32, u32),
        initial_contents: Vec<Option<BorrowedStack>>,
        can_place: bool,
        can_take: bool,
    ) -> Result<Self> {
        let form_key = form_key.into();
        let view = InventoryView::new_transient(
            self.game_state.clone(),
            dimensions,
            initial_contents,
            can_place,
            can_take,
            false,
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

    pub fn inventory_view_virtual_output(
        mut self,
        form_key: impl Into<String>,
        dimensions: (u32, u32),
        callbacks: VirtualOutputCallbacks<Popup>,
        take_exact: bool,
    ) -> Result<Self> {
        let form_key = form_key.into();
        let view = InventoryView::new_virtual_output(
            self.game_state.clone(),
            dimensions,
            callbacks,
            take_exact,
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

    pub fn inventory_view_block(
        mut self,
        form_key: impl Into<String>,
        default_dimensions: (u32, u32),
        coord: BlockCoordinate,
        inv_key: String,
        can_place: bool,
        can_take: bool,
        take_exact: bool
    ) -> Result<Self> {
        let form_key = form_key.into();
        let view = InventoryView::new_block(
            self.game_state.clone(),
            default_dimensions,
            coord,
            inv_key,
            can_place,
            can_take,
            take_exact,
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
        F: Fn(&Popup) + Send + Sync + 'static,
    {
        self.inventory_update_callback = Some(Box::new(callback));
        self
    }

    /// Sets a function that will be called when a button is clicked
    ///
    /// Exact parameters are still tbd
    ///
    /// This replaces any previously set function
    pub fn set_button_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(PopupResponse) + Send + Sync + 'static,
    {
        self.button_callback = Some(Box::new(callback));
        self
    }

    pub(crate) fn invoke_inventory_action_callback(&self) {
        if let Some(callback) = &self.inventory_update_callback {
            callback(self);
        }
    }

    pub(crate) fn handle_response(
        &mut self,
        response: PopupResponse,
        player_main_inv: InventoryKey,
    ) -> Result<()> {
        if let PopupAction::PopupClosed = response.user_action {
            for view in self.inventory_views.values_mut() {
                view.clear_if_transient(Some(player_main_inv))?;
            }
        }
        if let Some(callback) = &self.button_callback {
            callback(response)
        }
        Ok(())
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
            title: self.title.clone(),
            element: elements,
        }
    }

    pub fn inventory_views(&self) -> &HashMap<String, InventoryView<Popup>> {
        &self.inventory_views
    }

    pub fn id(&self) -> u64 {
        self.id
    }
}

static NEXT_POPUP_ID: AtomicU64 = AtomicU64::new(1);
