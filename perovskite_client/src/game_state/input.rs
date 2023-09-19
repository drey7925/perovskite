use std::sync::Arc;

use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize};
use winit::event::{DeviceEvent, ElementState, MouseScrollDelta, WindowEvent};

use super::settings::GameSettings;

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub(crate) enum BoundAction {
    MoveForward,
    MoveBackward,
    MoveLeft,
    MoveRight,
    Jump,
    Descend,
    TogglePhysics,
    Interact,
    Dig,
    Place,
    MouseCapture,
    Inventory,
    Menu,
    Chat,
    ChatSlash,
    PhysicsDebug
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
    pub(crate) toggle_physics: Keybind,

    pub(crate) interact_key: Keybind,
    pub(crate) dig: Keybind,
    pub(crate) place: Keybind,

    pub(crate) mouse_capture: Keybind,
    pub(crate) inventory: Keybind,
    pub(crate) menu: Keybind,

    pub(crate) chat: Keybind,
    pub(crate) chat_slash: Keybind,

    pub(crate) physics_debug: Keybind
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
            BoundAction::Interact => self.interact_key,
            BoundAction::Dig => self.dig,
            BoundAction::Place => self.place,
            BoundAction::MouseCapture => self.mouse_capture,
            BoundAction::Inventory => self.inventory,
            BoundAction::TogglePhysics => self.toggle_physics,
            BoundAction::Menu => self.menu,
            BoundAction::Chat => self.chat,
            BoundAction::ChatSlash => self.chat_slash,
            BoundAction::PhysicsDebug => self.physics_debug
        }
    }
}
impl Default for KeybindSettings {
    fn default() -> Self {
        use Keybind::*;
        Self {
            camera_sensitivity: 1.0,
            scroll_inverse_sensitivity: 100,
            move_forward: ScanCode(0x11),
            move_backward: ScanCode(0x1f),
            move_left: ScanCode(0x1e),
            move_right: ScanCode(0x20),
            toggle_physics: ScanCode(0x19),
            jump: ScanCode(0x39),
            descend: ScanCode(0x2a),
            interact_key: ScanCode(0x21),
            dig: MouseButton(winit::event::MouseButton::Left),
            place: MouseButton(winit::event::MouseButton::Right),
            mouse_capture: ScanCode(0x38),
            inventory: ScanCode(0x17),
            menu: ScanCode(0x1),
            chat: ScanCode(0x14),
            chat_slash: ScanCode(0x35),
            physics_debug: ScanCode(0x2)
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub(crate) enum Keybind {
    ScanCode(u32),
    MouseButton(winit::event::MouseButton),
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

    pub(crate) fn event<T>(&mut self, event: &winit::event::Event<'_, T>) {
        match event {
            winit::event::Event::WindowEvent {
                window_id: _,
                event,
            } => self.handle_window_event(event),
            winit::event::Event::DeviceEvent {
                device_id: _,
                event,
            } => self.handle_device_event(event),
            _ => {}
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

    pub(crate) fn take_just_released(&mut self, action: BoundAction) -> bool {
        if self.modal_active {
            return false;
        }
        self.new_releases
            .remove(&self.settings.load().input.get(action))
    }

    fn handle_window_event(&mut self, event: &WindowEvent) {
        if let WindowEvent::KeyboardInput { input, .. } = event {
            match input.state {
                ElementState::Pressed => {
                    self.active_keybinds
                        .insert(Keybind::ScanCode(input.scancode));
                    self.new_presses.insert(Keybind::ScanCode(input.scancode));
                    if Keybind::ScanCode(input.scancode) == self.settings.load().input.mouse_capture
                    {
                        self.mouse_captured = !self.mouse_captured;
                    }
                }
                ElementState::Released => {
                    self.active_keybinds
                        .remove(&Keybind::ScanCode(input.scancode));
                    self.new_releases.insert(Keybind::ScanCode(input.scancode));
                }
            }
        } else if let &WindowEvent::MouseInput {
            state,
            button,
            ..
        } = event
        {
            match state {
                ElementState::Pressed => {
                    self.active_keybinds.insert(Keybind::MouseButton(button));
                    self.new_presses.insert(Keybind::MouseButton(button));
                },
                ElementState::Released => {
                    self.active_keybinds.remove(&Keybind::MouseButton(button));
                    self.new_releases.insert(Keybind::MouseButton(button));
                },
            }
        }
    }

    fn handle_device_event(&mut self, event: &DeviceEvent) {
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

    pub(crate) fn is_mouse_captured(&self) -> bool {
        self.mouse_captured && !self.modal_active
    }
}
