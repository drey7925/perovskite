use std::{
    collections::{HashMap, HashSet},
    sync::{atomic::AtomicU64, Arc},
};

use self::private::UiElementContainerPrivate;

use super::{
    event::HandlerContext,
    inventory::{
        BorrowedStack, InventoryKey, InventoryView, InventoryViewId, VirtualOutputCallbacks,
    },
    GameState,
};
use crate::game_state::inventory::{InventoryViewWithContext, VirtualInputCallbacks};
use anyhow::{bail, Result};
use perovskite_core::{coordinates::BlockCoordinate, protocol::ui as proto};

pub type TextField = proto::TextField;
pub type Button = proto::Button;
pub type Checkbox = proto::Checkbox;
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
    /// A checkbox that the user can click
    Checkbox(Checkbox),
    /// An inventory view: key (within this popup), label, the inventory view id
    InventoryView(String, String, InventoryViewId),
    /// A side-by-side layout
    SideBySideLayout(String, Vec<UiElement>),
}
impl UiElement {
    fn to_proto(
        &self,
        inventory_views: &HashMap<String, InventoryView<Popup>>,
    ) -> proto::UiElement {
        let element = match self {
            UiElement::Label(label) => proto::ui_element::Element::Label(label.clone()),
            UiElement::TextField(field) => proto::ui_element::Element::TextField(field.clone()),
            UiElement::Button(button) => proto::ui_element::Element::Button(button.clone()),
            UiElement::Checkbox(checkbox) => proto::ui_element::Element::Checkbox(checkbox.clone()),
            UiElement::InventoryView(key, label, view_id) => {
                // we can expect() here since presence in the map is an invariant
                let view = inventory_views
                    .get(key)
                    .expect("No inventory view found for provided key");
                proto::ui_element::Element::Inventory(proto::InventoryView {
                    inventory_key: view_id.0,
                    label: label.clone(),
                    can_place: view.can_place,
                    can_take: view.can_take,
                    place_without_swap: view.put_without_swap,
                })
            }
            UiElement::SideBySideLayout(header, elements) => {
                proto::ui_element::Element::SideBySide(proto::SideBySideLayout {
                    header: header.clone(),
                    element: elements
                        .iter()
                        .map(|x| x.to_proto(inventory_views))
                        .collect(),
                })
            }
        };
        proto::UiElement {
            element: Some(element),
        }
    }
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
pub struct PopupResponse<'a> {
    /// The key of the button that was clicked
    pub user_action: PopupAction,
    /// The values of all textfields
    pub textfield_values: HashMap<String, String>,
    /// The values of all checkboxes
    pub checkbox_values: HashMap<String, bool>,
    /// Handler context
    pub ctx: HandlerContext<'a>,
}

type InventoryUpdateCallback = Box<dyn Fn(&Popup) -> Result<()> + Send + Sync>;

type ButtonCallback = Box<dyn Fn(PopupResponse) -> Result<()> + Send + Sync>;

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
    inventory_update_callback: Option<InventoryUpdateCallback>,
    button_callback: Option<ButtonCallback>,
}

impl Popup {
    pub(crate) fn find_view(
        &self,
        id: InventoryViewId,
    ) -> Option<Box<InventoryViewWithContext<'_, Popup>>> {
        if let Some(view) = self.inventory_views().values().find(|x| x.id == id) {
            Some(Box::new(InventoryViewWithContext {
                view,
                context: self,
            }))
        } else {
            None
        }
    }

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

    pub(crate) fn invoke_inventory_action_callback(&self) -> Result<()> {
        if let Some(callback) = &self.inventory_update_callback {
            callback(self)?;
        }
        Ok(())
    }

    pub(crate) fn handle_response(
        &mut self,
        response: PopupResponse,
        player_main_inv: InventoryKey,
    ) -> Result<()> {
        if let PopupAction::PopupClosed = response.user_action {
            for view in self.deref_to_popup().inventory_views.values_mut() {
                view.clear_if_transient(Some(player_main_inv))?;
            }
        }
        if let Some(callback) = &self.button_callback {
            callback(response)?
        }
        Ok(())
    }

    pub(crate) fn to_proto(&self) -> proto::PopupDescription {
        let elements = self
            .widgets
            .iter()
            .map(|x| x.to_proto(&self.inventory_views))
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

// https://stackoverflow.com/a/53207767/1424875
// Waiting for https://github.com/rust-lang/rfcs/blob/master/text/2145-type-privacy.md
mod private {

    pub trait UiElementContainerPrivate {
        fn push_widget(&mut self, element: super::UiElement);
        fn deref_to_popup(&mut self) -> &mut super::Popup;
    }
}
impl UiElementContainerPrivate for Popup {
    fn push_widget(&mut self, element: UiElement) {
        self.widgets.push(element);
    }
    fn deref_to_popup(&mut self) -> &mut Popup {
        self
    }
}

impl UiElementContainer for Popup {}

pub struct SideBySideLayoutBuilder<'a> {
    popup: &'a mut Popup,
    widgets: Vec<UiElement>,
}

impl UiElementContainerPrivate for SideBySideLayoutBuilder<'_> {
    fn push_widget(&mut self, element: UiElement) {
        self.widgets.push(element);
    }
    fn deref_to_popup(&mut self) -> &mut Popup {
        self.popup
    }
}
impl UiElementContainer for SideBySideLayoutBuilder<'_> {}

pub struct TextFieldBuilder {
    field: TextField,
}
impl TextFieldBuilder {
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            field: TextField {
                key: key.into(),
                label: "".to_string(),
                initial: "".to_string(),
                enabled: true,
                multiline: false,
                hover_text: "".to_string(),
            },
        }
    }
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.field.label = label.into();
        self
    }

    pub fn initial(mut self, initial: impl Into<String>) -> Self {
        self.field.initial = initial.into();
        self
    }

    pub fn enabled(mut self, enabled: bool) -> Self {
        self.field.enabled = enabled;
        self
    }

    pub fn multiline(mut self, multiline: bool) -> Self {
        self.field.multiline = multiline;
        self
    }

    pub fn hover_text(mut self, text: impl Into<String>) -> Self {
        self.field.hover_text = text.into();
        self
    }
}

// blanket impls for adding elements, assuming that we're adding to something that can store
// elements
pub trait UiElementContainer: UiElementContainerPrivate + Sized {
    /// Adds a new label to this popup. At the moment, the layout is still TBD.
    fn label(mut self, label: impl Into<String>) -> Self {
        self.push_widget(UiElement::Label(label.into()));
        self
    }
    /// Adds a new text field to this popup. At the moment, the layout is still TBD.
    ///
    /// Deprecated: Use text_field_from_builder for all functionalities; this method has too many
    /// parameters that would get more and more unwieldy as features are added; hence it's limited
    /// to the current set of parameters, and will not be expanded as textfields increase in
    /// functionality
    #[deprecated]
    fn text_field(
        mut self,
        key: impl Into<String>,
        label: impl Into<String>,
        initial: impl Into<String>,
        enabled: bool,
        multiline: bool,
    ) -> Self {
        self.push_widget(UiElement::TextField(TextField {
            key: key.into(),
            label: label.into(),
            initial: initial.into(),
            enabled,
            multiline,
            hover_text: "".to_string(),
        }));
        self
    }

    fn text_field_from_builder(mut self, builder: TextFieldBuilder) -> Self {
        self.push_widget(UiElement::TextField(builder.field));
        self
    }

    /// Adds a new button to this popup.
    fn button(
        mut self,
        key: impl Into<String>,
        label: impl Into<String>,
        enabled: bool,
        will_close_popup: bool,
    ) -> Self {
        self.push_widget(UiElement::Button(Button {
            key: key.into(),
            label: label.into(),
            enabled,
            will_close_popup,
        }));
        self
    }
    /// Adds a checkbox to this popup
    fn checkbox(
        mut self,
        key: impl Into<String>,
        label: impl Into<String>,
        initial: bool,
        enabled: bool,
    ) -> Self {
        self.push_widget(UiElement::Checkbox(Checkbox {
            key: key.into(),
            label: label.into(),
            initial,
            enabled,
        }));
        self
    }
    /// Adds an inventory view backed by a stored inventory managed in the inventory manager
    /// If can_place or can_take are true, then player actions are directly written back to the
    /// game database.
    fn inventory_view_stored(
        mut self,
        form_key: impl Into<String>,
        label: impl Into<String>,
        inventory_key: InventoryKey,
        can_place: bool,
        can_take: bool,
    ) -> Result<Self> {
        let form_key = form_key.into();
        let view = InventoryView::new_stored(
            inventory_key,
            self.deref_to_popup().game_state.clone(),
            can_place,
            can_take,
        )?;
        self.push_widget(UiElement::InventoryView(
            form_key.clone(),
            label.into(),
            view.id,
        ));
        self.deref_to_popup()
            .interested_stored_inventories
            .insert(inventory_key);
        match self
            .deref_to_popup()
            .inventory_views
            .insert(form_key.clone(), view)
        {
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
    fn inventory_view_transient(
        mut self,
        form_key: impl Into<String>,
        label: impl Into<String>,
        dimensions: (u32, u32),
        initial_contents: Vec<Option<BorrowedStack>>,
        can_place: bool,
        can_take: bool,
    ) -> Result<Self> {
        let form_key = form_key.into();
        let view = InventoryView::new_transient(
            self.deref_to_popup().game_state.clone(),
            dimensions,
            initial_contents,
            can_place,
            can_take,
            false,
        )?;
        self.push_widget(UiElement::InventoryView(
            form_key.clone(),
            label.into(),
            view.id,
        ));
        match self
            .deref_to_popup()
            .inventory_views
            .insert(form_key.clone(), view)
        {
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

    fn inventory_view_virtual_output(
        mut self,
        form_key: impl Into<String>,
        label: impl Into<String>,
        dimensions: (u32, u32),
        callbacks: VirtualOutputCallbacks<Popup>,
        take_exact: bool,
        allow_return: bool,
    ) -> Result<Self> {
        let form_key = form_key.into();
        let view = InventoryView::new_virtual_output(
            self.deref_to_popup().game_state.clone(),
            dimensions,
            callbacks,
            take_exact,
            allow_return,
        )?;
        self.push_widget(UiElement::InventoryView(
            form_key.clone(),
            label.into(),
            view.id,
        ));
        match self
            .deref_to_popup()
            .inventory_views
            .insert(form_key.clone(), view)
        {
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

    fn inventory_view_virtual_input(
        mut self,
        form_key: impl Into<String>,
        label: impl Into<String>,
        dimensions: (u32, u32),
        callbacks: VirtualInputCallbacks<Popup>,
    ) -> Result<Self> {
        let form_key = form_key.into();
        let view = InventoryView::new_virtual_input(
            self.deref_to_popup().game_state.clone(),
            dimensions,
            callbacks,
        )?;
        self.push_widget(UiElement::InventoryView(
            form_key.clone(),
            label.into(),
            view.id,
        ));
        match self
            .deref_to_popup()
            .inventory_views
            .insert(form_key.clone(), view)
        {
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

    fn inventory_view_block(
        mut self,
        form_key: impl Into<String>,
        label: impl Into<String>,
        default_dimensions: (u32, u32),
        coord: BlockCoordinate,
        inv_key: String,
        can_place: bool,
        can_take: bool,
        take_exact: bool,
    ) -> Result<Self> {
        let form_key = form_key.into();
        let view = InventoryView::new_block(
            self.deref_to_popup().game_state.clone(),
            default_dimensions,
            coord,
            inv_key,
            can_place,
            can_take,
            take_exact,
        )?;
        self.push_widget(UiElement::InventoryView(
            form_key.clone(),
            label.into(),
            view.id,
        ));
        match self
            .deref_to_popup()
            .inventory_views
            .insert(form_key.clone(), view)
        {
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
    fn set_inventory_action_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&Popup) -> Result<()> + Send + Sync + 'static,
    {
        self.deref_to_popup().inventory_update_callback = Some(Box::new(callback));
        self
    }

    /// Sets a function that will be called when a button is clicked
    ///
    /// Exact parameters are still tbd
    ///
    /// This replaces any previously set function
    fn set_button_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(PopupResponse) -> Result<()> + Send + Sync + 'static,
    {
        self.deref_to_popup().button_callback = Some(Box::new(callback));
        self
    }

    /// Adds a side-by-side layout that will be displayed.
    /// The vertical alignment is still TBD
    fn side_by_side_layout(
        mut self,
        label: impl Into<String>,
        f: impl FnOnce(SideBySideLayoutBuilder) -> Result<SideBySideLayoutBuilder>,
    ) -> Result<Self> {
        let builder = SideBySideLayoutBuilder {
            popup: self.deref_to_popup(),
            widgets: vec![],
        };
        let result = f(builder)?;
        let widgets = result.widgets;
        self.push_widget(UiElement::SideBySideLayout(label.into(), widgets));
        Ok(self)
    }
}

static NEXT_POPUP_ID: AtomicU64 = AtomicU64::new(1);
