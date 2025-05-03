use std::sync::Arc;

use super::settings::GameSettings;
use rustc_hash::FxHashSet;
use serde::{Deserialize, Deserializer, Serialize};
use winit::event::{DeviceEvent, ElementState, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::platform::scancode::PhysicalKeyExtScancode;

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub(crate) enum BoundAction {
    MoveForward,
    MoveBackward,
    MoveLeft,
    MoveRight,
    Jump,
    Descend,
    FastMove,
    TogglePhysics,
    Interact,
    Dig,
    Place,
    MouseCapture,
    Inventory,
    Menu,
    Chat,
    ChatSlash,
    DebugPanel,
    PerfPanel,
    ViewRangeUp,
    ViewRangeDown,
}
impl std::fmt::Display for BoundAction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}
impl BoundAction {
    pub(crate) fn user_friendly_name(&self) -> &str {
        match self {
            BoundAction::MoveForward => "Move forward",
            BoundAction::MoveBackward => "Move backward",
            BoundAction::MoveLeft => "Move left",
            BoundAction::MoveRight => "Move right",
            BoundAction::Jump => "Jump",
            BoundAction::Descend => "Descend",
            BoundAction::FastMove => "Fast move",
            BoundAction::Interact => "Interact",
            BoundAction::Dig => "Dig",
            BoundAction::Place => "Place",
            BoundAction::MouseCapture => "Capture/release mouse",
            BoundAction::Inventory => "Open inventory",
            BoundAction::TogglePhysics => "Toggle physics mode",
            BoundAction::Menu => "Pause menu",
            BoundAction::Chat => "Chat",
            BoundAction::ChatSlash => "Chat slash command",
            BoundAction::DebugPanel => "Debug panel",
            BoundAction::PerfPanel => "Performance panel",
            BoundAction::ViewRangeUp => "Increase view range",
            BoundAction::ViewRangeDown => "Decrease view range",
        }
    }

    pub(crate) fn tooltip(&self) -> &str {
        match self {
            BoundAction::MoveForward => "Walk/fly/swim forward",
            BoundAction::MoveBackward => "Walk/fly/swim backward",
            BoundAction::MoveLeft => "Walk/fly/swim left",
            BoundAction::MoveRight => "Walk/fly/swim right",
            BoundAction::Jump => "Jump, climb ladders, swim upward",
            BoundAction::Descend => "Descend ladders, swim downward",
            BoundAction::FastMove => "Hold to move quickly, if you have the right permission",
            BoundAction::Interact => "Interact with blocks like chests, furnaces, etc.",
            BoundAction::Dig => "Hold to dig the block you're pointing at",
            BoundAction::Place => "Click/press to place the block in your hand",
            BoundAction::MouseCapture => "Toggle whether the mouse is captured in the window",
            BoundAction::Inventory => "Open the inventory view",
            BoundAction::TogglePhysics => {
                "Switch between normal, flying, and noclip (if your permissions allow)"
            }
            BoundAction::Menu => "Pause the game and open the pause menu",
            BoundAction::Chat => "View chat and/or type a chat message",
            BoundAction::ChatSlash => "Open chat with / prefilled for a command - it's highly recommended to keep this bound to your slash key in your choice of keyboard layout",
            BoundAction::DebugPanel => "Open/close the debug panel",
            BoundAction::PerfPanel => "Open/close the performance panel",
            BoundAction::ViewRangeUp => "Increase the max chunk view distance",
            BoundAction::ViewRangeDown => "Decrease the max chunk view distance",
        }
    }

    pub(crate) fn all_bound_actions() -> &'static [BoundAction] {
        const ALL: &[BoundAction] = &[
            BoundAction::MoveForward,
            BoundAction::MoveBackward,
            BoundAction::MoveLeft,
            BoundAction::MoveRight,
            BoundAction::Jump,
            BoundAction::Descend,
            BoundAction::FastMove,
            BoundAction::Interact,
            BoundAction::Dig,
            BoundAction::Place,
            BoundAction::MouseCapture,
            BoundAction::Inventory,
            BoundAction::TogglePhysics,
            BoundAction::Menu,
            BoundAction::Chat,
            BoundAction::ChatSlash,
            BoundAction::DebugPanel,
            BoundAction::PerfPanel,
            BoundAction::ViewRangeUp,
            BoundAction::ViewRangeDown,
        ];
        ALL
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(default)]
pub(crate) struct KeybindSettings {
    pub(crate) camera_sensitivity: f64,
    pub(crate) scroll_inverse_sensitivity: u32,

    pub(crate) move_forward: Keybind,
    pub(crate) move_backward: Keybind,
    pub(crate) move_left: Keybind,
    pub(crate) move_right: Keybind,
    pub(crate) jump: Keybind,
    pub(crate) descend: Keybind,
    pub(crate) fast_move: Keybind,
    pub(crate) toggle_physics: Keybind,

    pub(crate) interact_key: Keybind,
    pub(crate) dig: Keybind,
    pub(crate) place: Keybind,

    pub(crate) mouse_capture: Keybind,
    pub(crate) inventory: Keybind,
    pub(crate) menu: Keybind,

    pub(crate) chat: Keybind,
    pub(crate) chat_slash: Keybind,

    pub(crate) physics_debug: Keybind,
    pub(crate) performance_panel: Keybind,

    pub(crate) view_range_up: Keybind,
    pub(crate) view_range_down: Keybind,

    pub(crate) hotbar_slots: [Keybind; 8],
}
impl KeybindSettings {
    pub(crate) fn get(&self, action: BoundAction) -> Keybind {
        match action {
            BoundAction::MoveForward => self.move_forward,
            BoundAction::MoveBackward => self.move_backward,
            BoundAction::MoveLeft => self.move_left,
            BoundAction::MoveRight => self.move_right,
            BoundAction::Jump => self.jump,
            BoundAction::Descend => self.descend,
            BoundAction::FastMove => self.fast_move,
            BoundAction::Interact => self.interact_key,
            BoundAction::Dig => self.dig,
            BoundAction::Place => self.place,
            BoundAction::MouseCapture => self.mouse_capture,
            BoundAction::Inventory => self.inventory,
            BoundAction::TogglePhysics => self.toggle_physics,
            BoundAction::Menu => self.menu,
            BoundAction::Chat => self.chat,
            BoundAction::ChatSlash => self.chat_slash,
            BoundAction::DebugPanel => self.physics_debug,
            BoundAction::PerfPanel => self.performance_panel,
            BoundAction::ViewRangeUp => self.view_range_up,
            BoundAction::ViewRangeDown => self.view_range_down,
        }
    }

    pub(crate) fn set(&mut self, action: BoundAction, keybind: Keybind) {
        match action {
            BoundAction::MoveForward => self.move_forward = keybind,
            BoundAction::MoveBackward => self.move_backward = keybind,
            BoundAction::MoveLeft => self.move_left = keybind,
            BoundAction::MoveRight => self.move_right = keybind,
            BoundAction::Jump => self.jump = keybind,
            BoundAction::Descend => self.descend = keybind,
            BoundAction::FastMove => self.fast_move = keybind,
            BoundAction::Interact => self.interact_key = keybind,
            BoundAction::Dig => self.dig = keybind,
            BoundAction::Place => self.place = keybind,
            BoundAction::MouseCapture => self.mouse_capture = keybind,
            BoundAction::Inventory => self.inventory = keybind,
            BoundAction::TogglePhysics => self.toggle_physics = keybind,
            BoundAction::Menu => self.menu = keybind,
            BoundAction::Chat => self.chat = keybind,
            BoundAction::ChatSlash => self.chat_slash = keybind,
            BoundAction::DebugPanel => self.physics_debug = keybind,
            BoundAction::PerfPanel => self.performance_panel = keybind,
            BoundAction::ViewRangeUp => self.view_range_up = keybind,
            BoundAction::ViewRangeDown => self.view_range_down = keybind,
        }
    }

    pub(crate) fn migrate_scancodes(self) -> Self {
        fn migrate(keybind: Keybind) -> Keybind {
            match keybind {
                Keybind::ScanCode(x) => Keybind::Key(PhysicalKey::from_scancode(x)),
                x => x,
            }
        }
        KeybindSettings {
            camera_sensitivity: self.camera_sensitivity,
            scroll_inverse_sensitivity: self.scroll_inverse_sensitivity,
            move_forward: migrate(self.move_forward),
            move_backward: migrate(self.move_backward),
            move_left: migrate(self.move_left),
            move_right: migrate(self.move_right),
            jump: migrate(self.jump),
            descend: migrate(self.descend),
            fast_move: migrate(self.fast_move),
            toggle_physics: migrate(self.toggle_physics),
            interact_key: migrate(self.interact_key),
            dig: migrate(self.dig),
            place: migrate(self.place),
            mouse_capture: migrate(self.mouse_capture),
            inventory: migrate(self.inventory),
            menu: migrate(self.menu),
            chat: migrate(self.chat),
            chat_slash: migrate(self.chat_slash),
            physics_debug: migrate(self.physics_debug),
            performance_panel: migrate(self.performance_panel),
            view_range_up: migrate(self.view_range_up),
            view_range_down: migrate(self.view_range_down),
            hotbar_slots: self.hotbar_slots.map(|x| migrate(x)),
        }
    }
}
impl Default for KeybindSettings {
    fn default() -> Self {
        use Keybind::*;
        Self {
            camera_sensitivity: 0.3,
            scroll_inverse_sensitivity: 100,
            move_forward: Key(PhysicalKey::Code(KeyCode::KeyW)),
            move_backward: Key(PhysicalKey::Code(KeyCode::KeyS)),
            move_left: Key(PhysicalKey::Code(KeyCode::KeyA)),
            move_right: Key(PhysicalKey::Code(KeyCode::KeyD)),
            toggle_physics: Key(PhysicalKey::Code(KeyCode::KeyP)),
            jump: Key(PhysicalKey::Code(KeyCode::KeyJ)),
            descend: Key(PhysicalKey::Code(KeyCode::ShiftLeft)),
            fast_move: Key(PhysicalKey::Code(KeyCode::ControlLeft)),
            interact_key: Key(PhysicalKey::Code(KeyCode::KeyW)),
            dig: MouseButton(winit::event::MouseButton::Left),
            place: MouseButton(winit::event::MouseButton::Right),
            mouse_capture: Key(PhysicalKey::Code(KeyCode::KeyW)),
            inventory: Key(PhysicalKey::Code(KeyCode::KeyW)),
            menu: Key(PhysicalKey::Code(KeyCode::KeyW)),
            chat: Key(PhysicalKey::Code(KeyCode::KeyW)),
            chat_slash: Key(PhysicalKey::Code(KeyCode::KeyW)),
            physics_debug: Key(PhysicalKey::Code(KeyCode::KeyW)),
            performance_panel: Key(PhysicalKey::Code(KeyCode::KeyW)),
            view_range_up: Key(PhysicalKey::Code(KeyCode::KeyW)),
            view_range_down: Key(PhysicalKey::Code(KeyCode::KeyW)),
            hotbar_slots: [
                Key(PhysicalKey::Code(KeyCode::Digit1)),
                Key(PhysicalKey::Code(KeyCode::Digit2)),
                Key(PhysicalKey::Code(KeyCode::Digit3)),
                Key(PhysicalKey::Code(KeyCode::Digit4)),
                Key(PhysicalKey::Code(KeyCode::Digit5)),
                Key(PhysicalKey::Code(KeyCode::Digit6)),
                Key(PhysicalKey::Code(KeyCode::Digit7)),
                Key(PhysicalKey::Code(KeyCode::Digit8)),
            ],
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub(crate) enum Keybind {
    ScanCode(u32),
    Key(winit::keyboard::PhysicalKey),
    MouseButton(winit::event::MouseButton),
}

impl Keybind {
    pub(crate) fn to_ui_string(&self) -> String {
        match self {
            Keybind::ScanCode(sc) => {
                format!("ScanCode({})", sc)
            }
            Keybind::Key(PhysicalKey::Code(code)) => {
                format!("[{:?}]", code)
            }
            Keybind::Key(PhysicalKey::Unidentified(nkc)) => {
                format!("[{:?}]", nkc)
            }
            Keybind::MouseButton(btn) => {
                format!("Mouse({:?})", btn)
            }
        }
    }
}

pub(crate) struct InputState {
    pub(crate) settings: Arc<arc_swap::ArcSwap<GameSettings>>,

    active_keybinds: FxHashSet<Keybind>,
    new_presses: FxHashSet<Keybind>,
    new_releases: FxHashSet<Keybind>,

    // Used by physics controller to update the camera
    pending_camera_delta: (f64, f64),
    // Used by HUD to update the hotbar and tool controller
    pending_scroll_pixels: i32,
    pending_scroll_slots: i32,

    mouse_captured: bool,
    // If true, an egui modal window is up, and the user's input should be ignored
    // by everything except that egui window.
    modal_active: bool,
}
impl InputState {
    pub(crate) fn new(settings: Arc<arc_swap::ArcSwap<GameSettings>>) -> InputState {
        InputState {
            settings,
            active_keybinds: FxHashSet::default(),
            new_presses: FxHashSet::default(),
            new_releases: FxHashSet::default(),
            pending_camera_delta: (0.0, 0.0),
            pending_scroll_pixels: 0,
            pending_scroll_slots: 0,
            mouse_captured: true,
            modal_active: false,
        }
    }

    pub(crate) fn set_modal_active(&mut self, active: bool) {
        if active {
            self.active_keybinds.clear();
            self.new_presses.clear();
            self.new_releases.clear();
        }
        self.modal_active = active;
    }

    pub(crate) fn is_pressed(&self, action: BoundAction) -> bool {
        if self.modal_active {
            return false;
        }
        self.active_keybinds
            .contains(&self.settings.load().input.get(action))
    }

    pub(crate) fn take_just_pressed(&mut self, action: BoundAction) -> bool {
        if self.modal_active {
            return false;
        }
        self.new_presses
            .remove(&self.settings.load().input.get(action))
    }

    pub(crate) fn peek_just_pressed(&self, action: BoundAction) -> bool {
        if self.modal_active {
            return false;
        }
        self.new_presses
            .contains(&self.settings.load().input.get(action))
    }

    pub(crate) fn take_just_released(&mut self, action: BoundAction) -> bool {
        if self.modal_active {
            return false;
        }
        self.new_releases
            .remove(&self.settings.load().input.get(action))
    }

    pub(crate) fn window_event(&mut self, event: &WindowEvent) {
        if let WindowEvent::KeyboardInput { event: input, .. } = event {
            match input.state {
                ElementState::Pressed => {
                    self.active_keybinds
                        .insert(Keybind::Key(input.physical_key));
                    self.new_presses.insert(Keybind::Key(input.physical_key));
                    if Keybind::Key(input.physical_key) == self.settings.load().input.mouse_capture
                    {
                        self.mouse_captured = !self.mouse_captured;
                    }
                }
                ElementState::Released => {
                    self.active_keybinds
                        .remove(&Keybind::Key(input.physical_key));
                    self.new_releases.insert(Keybind::Key(input.physical_key));
                }
            }
        } else if let &WindowEvent::MouseInput { state, button, .. } = event {
            match state {
                ElementState::Pressed => {
                    if self.mouse_captured {
                        self.active_keybinds.insert(Keybind::MouseButton(button));
                        self.new_presses.insert(Keybind::MouseButton(button));
                    } else {
                        self.mouse_captured = true;
                    }
                }
                ElementState::Released => {
                    self.active_keybinds.remove(&Keybind::MouseButton(button));
                    self.new_releases.insert(Keybind::MouseButton(button));
                }
            }
        }
    }

    pub(crate) fn handle_device_event(&mut self, event: &DeviceEvent) {
        let settings = self.settings.load();
        if self.mouse_captured {
            if let DeviceEvent::MouseMotion { delta } = event {
                self.pending_camera_delta.0 += delta.0 * settings.input.camera_sensitivity;
                self.pending_camera_delta.1 += delta.1 * settings.input.camera_sensitivity;
            };
        }
        if let DeviceEvent::MouseWheel {
            delta: MouseScrollDelta::LineDelta(_, lines),
        } = event
        {
            self.pending_scroll_slots += lines.round() as i32;
            self.pending_scroll_pixels = 0;
        }
        if let DeviceEvent::MouseWheel {
            delta: MouseScrollDelta::PixelDelta(pixels),
        } = event
        {
            self.pending_scroll_pixels += pixels.y.round() as i32;
            self.pending_scroll_slots += self
                .pending_scroll_pixels
                .div_euclid(settings.input.scroll_inverse_sensitivity as i32);
            self.pending_scroll_pixels = self
                .pending_scroll_pixels
                .rem_euclid(settings.input.scroll_inverse_sensitivity as i32);
        }
    }

    pub(crate) fn take_mouse_delta(&mut self) -> (f64, f64) {
        let delta = self.pending_camera_delta;
        self.pending_camera_delta = (0., 0.);
        if self.modal_active {
            (0., 0.)
        } else {
            delta
        }
    }

    pub(crate) fn take_scroll_slots(&mut self) -> i32 {
        let delta = self.pending_scroll_slots;
        self.pending_scroll_slots = 0;
        if self.modal_active {
            0
        } else {
            delta
        }
    }

    /// Returns a zero-based index for the hotbar selection picked by the player
    pub(crate) fn take_hotbar_selection(&mut self) -> Option<u32> {
        if self.modal_active {
            return None;
        }
        let slots = self.settings.load().input.hotbar_slots;
        for (i, slot) in slots.iter().enumerate() {
            if self.new_presses.remove(&slot) {
                return Some(i as u32);
            }
        }
        None
    }
    pub(crate) fn is_mouse_captured(&self) -> bool {
        self.mouse_captured && !self.modal_active
    }
}
