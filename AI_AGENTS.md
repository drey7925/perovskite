# To add a setting

* Read perovskite_client/src/client_state/settings.rs
* Pick a category from GameSettings (unless adding a new category)
* Add a field to the appropriate struct
* If adding a new type, be sure to include all of the derives seen on existing structs/enums
* Give a sensible default value in the default implementation
* In perovskite_client/src/main_menu.rs, find the corresponding function labeled `draw_...` and add an egui control, consistent with existing controls in the file.
* If the setting is likely needed for the Vulkan pipeline or swapchain (i.e. it affects a pipeline, layout, shader, etc), flag it for human attention by printing a message once you're done making changes.

# To bump the version:

* Run `ls -d */` in the root of the repository to see all of the crates. Disregard `target`
* Inside each of these, update Cargo.toml to update the version of the crate, and also any perovskite_ dependencies.

Current dependency order:
* Publish core, then server, then game_api, then client. If any additional crates are found other than these, please notify the user to update CLAUDE.local.md.
* For consistency, recommend using `cargo publish -p [crate name]` instead of changing directory then running `cargo publish`

# Adding New UI Controls to the Protocol

This section explains how to add a new UI control type to the perovskite UI system. The UI system uses protocol buffers for client-server communication, with server-side builders and client-side egui rendering.

## Architecture Overview

The UI system has three main components:

1. **Protocol Definition** (`perovskite_core/proto/ui.proto`) - Defines the wire format
2. **Server-Side API** (`perovskite_server/src/game_state/client_ui.rs`) - Provides builder APIs for creating UIs
3. **Client-Side Rendering** (`perovskite_client/src/game_ui/egui_ui.rs`) - Renders UI using egui

## Step-by-Step Guide

### 1. Define the Protocol Message

In `perovskite_core/proto/ui.proto`, add a new message for your control:

```protobuf
message YourControl {
  // The key used to identify this control in responses (if interactive)
  string key = 1;
  // Display text or label
  string label = 2;
  // Any other properties specific to your control
  bool some_property = 3;
  string another_property = 4;
}
```

**Guidelines:**
- Use a `key` field (string) for interactive controls that need to be identified in callbacks
- Use a `label` field for user-visible text
- Use `enabled` (bool) for controls that can be disabled
- Use `initial` for default values
- Follow naming conventions from existing controls (TextField, Checkbox, Button)

### 2. Add to UiElement Oneof

In the same proto file, add your control to the `UiElement` message's oneof:

```protobuf
message UiElement {
  oneof element {
    string label = 1;
    TextField text_field = 2;
    Button button = 3;
    InventoryView inventory = 4;
    SideBySideLayout side_by_side = 5;
    Checkbox checkbox = 6;
    YourControl your_control = 7;  // Add here with next available number
  }
}
```

### 3. Add Rust Enum Variant (Server-Side)

In `perovskite_server/src/game_state/client_ui.rs`, add a variant to the `UiElement` enum:

```rust
pub enum UiElement {
    Label(String),
    TextField(TextField),
    Button(Button),
    Checkbox(Checkbox),
    InventoryView(String, String, InventoryViewId),
    SideBySideLayout(String, Vec<UiElement>),
    YourControl(YourControl),  // Add here, using the proto type
}
```

### 4. Implement to_proto() Conversion

In the `UiElement::to_proto()` method, add a match arm for your new variant:

```rust
impl UiElement {
    fn to_proto(
        &self,
        inventory_views: &HashMap<String, InventoryView<Popup>>,
    ) -> proto::UiElement {
        let element = match self {
            // ... existing arms ...
            UiElement::YourControl(your_control) => {
                proto::ui_element::Element::YourControl(your_control.clone())
            }
        };
        proto::UiElement {
            element: Some(element),
        }
    }
}
```

**Note:** If your control needs special handling (like InventoryView does), implement custom conversion logic here.

### 5. Add Builder Method to UiElementContainer

Add a builder method to the `UiElementContainer` trait for ergonomic API:

```rust
pub trait UiElementContainer: UiElementContainerPrivate + Sized {
    // ... existing methods ...

    /// Adds your control to this popup
    fn your_control(
        mut self,
        key: impl Into<String>,
        label: impl Into<String>,
        some_property: bool,
        another_property: impl Into<String>,
    ) -> Self {
        self.push_widget(UiElement::YourControl(YourControl {
            key: key.into(),
            label: label.into(),
            some_property,
            another_property: another_property.into(),
        }));
        self
    }
}
```

**Guidelines:**
- Use `impl Into<String>` for string parameters for ergonomics
- Chain the builder pattern by returning `Self`
- Use descriptive parameter names
- Add doc comments explaining the control's behavior

### 6. Optional: Create a Builder Type

For complex controls (like TextField), create a dedicated builder:

```rust
pub struct YourControlBuilder {
    control: YourControl,
}

impl YourControlBuilder {
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            control: YourControl {
                key: key.into(),
                label: "".to_string(),
                some_property: false,
                another_property: "".to_string(),
            },
        }
    }

    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.control.label = label.into();
        self
    }

    pub fn some_property(mut self, value: bool) -> Self {
        self.control.some_property = value;
        self
    }

    // ... more builder methods ...
}
```

Then add a method to `UiElementContainer`:

```rust
fn your_control(mut self, builder: YourControlBuilder) -> Self {
    self.push_widget(UiElement::YourControl(builder.control));
    self
}
```

### 7. Implement Client-Side Rendering

In `perovskite_client/src/game_ui/egui_ui.rs`, add rendering logic in the `render_element()` method:

```rust
fn render_element(
    &mut self,
    ui: &mut egui::Ui,
    popup: &PopupDescription,
    element: &proto::UiElement,
    id: Id,
    atlas_texture_id: TextureId,
    client_state: &ClientState,
    clicked_button: &mut Option<(String, bool)>,
) {
    match &element.element {
        // ... existing arms ...
        Some(proto::ui_element::Element::YourControl(your_control)) => {
            // Render using egui widgets
            ui.label(your_control.label.clone());

            // For interactive controls, handle user input
            if ui.button("Click me").clicked() {
                // Store result or trigger callback
            }
        }
        // ... existing arms ...
    }
}
```

**Guidelines:**
- Use egui widgets appropriate to your control's purpose
- For stateful controls (like TextField, Checkbox), store state in HashMaps keyed by `(popup.popup_id, control.key)`
- Handle disabled state with `ui.add_enabled(control.enabled, widget)`
- Respect `self.allow_*_interaction` flags for buttons/interactive elements

### 8. Handle State Management (If Needed)

For controls that maintain state between renders:

**Add state storage in EguiUi struct:**
```rust
pub(crate) struct EguiUi {
    // ... existing fields ...
    your_control_values: FxHashMap<(u64, String), YourValueType>,
}
```

**Initialize in `new()`:**
```rust
your_control_values: FxHashMap::default(),
```

**Use in `render_element()`:**
```rust
let value = self
    .your_control_values
    .entry((popup.popup_id, your_control.key.clone()))
    .or_insert(your_control.initial_value.clone());
```

**Add getter method if values need to be sent back:**
```rust
fn get_your_control_values(&self, popup_id: u64) -> HashMap<String, YourValueType> {
    self.your_control_values
        .iter()
        .filter(|((popup, _), _)| popup == &popup_id)
        .map(|((_, form_key), value)| (form_key.clone(), value.clone()))
        .collect()
}
```

**Clear state in `clear_fields()`:**
```rust
fn clear_fields(&mut self, popup: &PopupDescription) {
    // ... existing clears ...
    self.your_control_values
        .retain(|(popup_id, _), _| popup_id != &popup.popup_id);
}
```

### 9. Update PopupResponse (If Sending Values Back)

If your control needs to send values back to the server:

**Update proto:**
```protobuf
message PopupResponse {
  uint64 popup_id = 1;
  bool closed = 2;
  string clicked_button = 3;
  map<string, string> text_fields = 4;
  map<string, bool> checkboxes = 5;
  map<string, YourValueType> your_controls = 6;  // Add here
}
```

**Update response construction in `draw_popup()`:**
```rust
GameAction::PopupResponse(PopupResponse {
    popup_id: popup.popup_id,
    closed: false,
    clicked_button: button_name,
    text_fields: self.get_text_fields(popup.popup_id),
    checkboxes: self.get_checkboxes(popup.popup_id),
    your_controls: self.get_your_control_values(popup.popup_id),  // Add here
})
```

## Examples

### Simple Non-Interactive Control (Label)

Label is the simplest control - it just displays text with no state or interaction.

**Proto:** Just a string in the oneof
**Server:** Single-line builder method
**Client:** Single `ui.label()` call

### Interactive Stateful Control (Checkbox)

Checkbox demonstrates state management and value return.

**Proto:** Defines `key`, `label`, `initial`, `enabled`
**Server:** Simple builder method
**Client:** Stores state in `checkboxes` HashMap, returns via `get_checkboxes()`

### Complex Control with Builder (TextField)

TextField uses a builder pattern for its many optional properties.

**Proto:** Many fields for configuration
**Server:** Dedicated `TextFieldBuilder` with chaining methods
**Client:** Different rendering for multiline vs single-line

### Container Control (SideBySideLayout)

SideBySideLayout contains other elements and uses a special builder.

**Proto:** Contains `repeated UiElement`
**Server:** `SideBySideLayoutBuilder` struct that implements `UiElementContainer`
**Client:** Recursive `render_element()` calls with layout wrapper

## Testing

After implementing your control:

1. Create a test popup in your game code
2. Verify the control renders correctly on the client
3. Test interaction (if applicable)
4. Verify state persists during popup lifetime
5. Verify state clears when popup closes
6. Test with disabled state (if applicable)

## Common Patterns

### Read-Only Display
If your control only displays information (like Label), you don't need state management or response handling.

### Simple Input
For basic user input (like Checkbox), store state in a HashMap and include in PopupResponse.

### Complex Configuration
For controls with many options (like TextField), use a builder pattern.

### Containers
For controls that contain other elements, implement a separate builder struct that also implements `UiElementContainer`.

### Special Integration
For controls that integrate with game systems (like InventoryView), add custom fields to Popup and special handling in to_proto().
