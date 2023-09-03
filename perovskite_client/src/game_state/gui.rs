use egui_demo_lib::DemoWindows;

pub(crate) struct GameGui {
    is_active: bool,
    demo_windows: DemoWindows   
}
impl GameGui {
    pub(crate) fn set_active(&mut self) {
        self.is_active = true;
    }
    pub(crate) fn set_inactive(&mut self) {
        self.is_active = false;
    }
    pub(crate) fn toggle_active(&mut self) {
        self.is_active = !self.is_active
    }
    pub(crate) fn is_active(&self) -> bool {
        self.is_active
    }
    pub(crate) fn draw(&mut self, ctx: &egui::Context) {
        self.demo_windows.ui(ctx);
    }
}