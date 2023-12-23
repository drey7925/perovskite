use std::time::Instant;

/// Chat message stored in memory. Currently a string, but may be expanded to support metadata, rich text, etc
#[derive(Clone, Debug)]
pub struct ChatMessage {
    timestamp: Instant,
    origin: String,
    origin_color: (u8, u8, u8),
    text: String,
}

impl ChatMessage {
    pub fn text(&self) -> &str {
        self.text.as_ref()
    }
    pub fn origin_color(&self) -> (u8, u8, u8) {
        self.origin_color
    }
    pub fn origin_color_fixed32(&self) -> u32 {
        color_to_fixed32(self.origin_color)
    }

    pub fn timestamp(&self) -> Instant {
        self.timestamp
    }

    pub fn origin(&self) -> &str {
        self.origin.as_ref()
    }

    pub fn new(origin: impl ToString, text: impl ToString) -> Self {
        Self {
            origin: origin.to_string(),
            origin_color: (0xc8, 0xc8, 0xc8),
            timestamp: Instant::now(),
            text: text.to_string(),
        }
    }
    pub fn with_color(mut self, origin_color: (u8, u8, u8)) -> Self {
        self.origin_color = origin_color;
        self
    }
    pub fn with_color_fixed32(mut self, origin_color: u32) -> Self {
        self.origin_color = color_from_fixed32(origin_color);
        self
    }
    pub fn with_origin(mut self, origin: impl ToString) -> Self {
        self.origin = origin.to_string();
        self
    }

    pub fn new_server_message(text: impl ToString) -> Self {
        Self {
            origin: "[server]".to_string(),
            origin_color: SERVER_MESSAGE_COLOR,
            timestamp: Instant::now(),
            text: text.to_string(),
        }
    }
}

pub const SERVER_MESSAGE_COLOR: (u8, u8, u8) = (0, 255, 255);
pub const SERVER_WARNING_COLOR: (u8, u8, u8) = (255, 255, 0);
pub const SERVER_ERROR_COLOR: (u8, u8, u8) = (255, 0, 0);
pub const CLIENT_INTERNAL_MESSAGE_COLOR: (u8, u8, u8) = (255, 255, 0);

pub fn color_to_fixed32(color: (u8, u8, u8)) -> u32 {
    ((color.0 as u32) << 16) | ((color.1 as u32) << 8) | (color.2 as u32)
}
pub fn color_from_fixed32(mut color: u32) -> (u8, u8, u8) {
    color &= 0x00ffffff;
    ((color >> 16) as u8, (color >> 8) as u8, color as u8)
}
