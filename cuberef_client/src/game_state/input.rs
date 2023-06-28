use std::collections::VecDeque;

use parking_lot::MutexGuard;
use winit::event::MouseScrollDelta;

use super::ClientState;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct KeybindSettings {
    move_forward: Keybind,
    move_backward: Keybind,
    move_left: Keybind,
    move_right: Keybind,
    jump: Keybind,
}
impl Default for KeybindSettings {
    fn default() -> Self {
        use Keybind::*;
        Self {
            move_forward: ScanCode(0x11),
            move_backward: ScanCode(0x1f),
            move_left: ScanCode(0x1e),
            move_right: ScanCode(0x20),
            jump: ScanCode(0x39)
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub(crate) enum Keybind {
    ScanCode(u32),
    MouseButton(winit::event::MouseButton),
}

struct InputState {
    // Used by physics controller to update the camera
    pending_camera_delta: (f64, f64),
    // Used by HUD to update the hotbar and tool controller
    pending_scroll_pixels: i32,
    pending_scroll_slots: i32,
}
impl InputState {
    pub(crate) fn event<T>(&mut self, event: winit::event::Event<'_, T>) {}
}
