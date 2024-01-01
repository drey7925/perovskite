use std::io::Cursor;

use crate::game_builder::{
    BlockName, GameBuilder, ItemName, StaticBlockName, StaticTextureName, TextureName,
};
use anyhow::{ensure, Result};
use image::{imageops::colorops, Rgba};

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// The in-game colors supported by this plugin.
///
/// Whenever a block or item can be colored, it will be colored with
/// the colors specified here (or a subset)
pub enum Color {
    Red,
    Orange,
    Yellow,
    Lime,
    Green,
    Teal,
    Blue,
    Purple,
    Pink,
    Black,
    White,
    Gray,
}

pub const ALL_COLORS: &[Color] = &[
    Color::Red,
    Color::Orange,
    Color::Yellow,
    Color::Lime,
    Color::Green,
    Color::Teal,
    Color::Blue,
    Color::Purple,
    Color::Pink,
    Color::Black,
    Color::White,
    Color::Gray,
];

pub mod item_groups {
    pub const DYES: &str = "dyes";
}

const DYE_BASE: &str = "colors:dye";

impl Color {
    /// Converts this color to a string suitable for suffixes, internal names, etc
    pub fn as_string(&self) -> &'static str {
        match self {
            Color::Red => "red",
            Color::Orange => "orange",
            Color::Yellow => "yellow",
            Color::Lime => "lime",
            Color::Green => "green",
            Color::Teal => "teal",
            Color::Blue => "blue",
            Color::Purple => "purple",
            Color::Pink => "pink",
            Color::Black => "black",
            Color::White => "white",
            Color::Gray => "gray",
        }
    }

    /// Converts this color to a string for display names
    pub fn as_display_string(&self) -> &'static str {
        match self {
            Color::Red => "Red",
            Color::Orange => "Orange",
            Color::Yellow => "Yellow",
            Color::Lime => "Lime",
            Color::Green => "Green",
            Color::Teal => "Teal",
            Color::Blue => "Blue",
            Color::Purple => "Purple",
            Color::Pink => "Pink",
            Color::Black => "Black",
            Color::White => "White",
            Color::Gray => "Gray",
        }
    }

    /// Converts this color to an RGB tuple.
    pub fn as_rgb(&self) -> (u8, u8, u8) {
        match self {
            Color::Red => (255, 0, 0),
            Color::Orange => (255, 128, 0),
            Color::Yellow => (255, 255, 0),
            Color::Lime => (0, 255, 0),
            Color::Green => (0, 128, 0),
            Color::Teal => (0, 255, 192),
            Color::Blue => (0, 0, 255),
            Color::Purple => (128, 0, 255),
            Color::Pink => (255, 0, 255),
            // For visual appeal, we actually encode black as (64, 64, 64)
            Color::Black => (64, 64, 64),
            Color::White => (255, 255, 255),
            Color::Gray => (128, 128, 128),
        }
    }
    /// Converts this color to an RGB tuple as floats.
    pub fn as_rgb_float(&self) -> (f32, f32, f32) {
        let (r, g, b) = self.as_rgb();
        (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0)
    }

    /// Returns the item name for the dye associated with this color
    pub fn dye_item_name(&self) -> String {
        format!("{}_{}", DYE_BASE, self.as_string())
    }

    /// Returns the name for the color group for this color
    pub fn color_group_name(&self) -> &'static str {
        match self {
            Color::Red => "colors:red",
            Color::Orange => "colors:orange",
            Color::Yellow => "colors:yellow",
            Color::Lime => "colors:lime",
            Color::Green => "colors:green",
            Color::Teal => "colors:teal",
            Color::Blue => "colors:blue",
            Color::Purple => "colors:purple",
            Color::Pink => "colors:pink",
            Color::Black => "colors:black",
            Color::White => "colors:white",
            Color::Gray => "colors:gray",
        }
    }

    pub fn colorize_to_png(&self, source_image: &image::DynamicImage) -> Result<Vec<u8>> {
        let color_multiplier = self.as_rgb_float();
        let mut colorized_image = source_image.to_rgba8();
        colorized_image.pixels_mut().for_each(|pixel| {
            *pixel = Rgba([
                (pixel.0[0] as f32 * color_multiplier.0) as u8,
                (pixel.0[1] as f32 * color_multiplier.1) as u8,
                (pixel.0[2] as f32 * color_multiplier.2) as u8,
                pixel.0[3],
            ]);
        });
        let mut bytes: Vec<u8> = Vec::new();
        colorized_image.write_to(&mut Cursor::new(&mut bytes), image::ImageOutputFormat::Png)?;
        Ok(bytes)
    }

    /// Applies the color to the source image and returns the result.
    /// The source image must be in a format supported by the `image` crate.
    /// The mask image sets where colorization is and isn't applied. Brighter values in the
    /// source image will result in more colorization.
    ///
    /// The source and mask must be the same size.
    pub fn colorize_to_png_with_mask(
        &self,
        colorized_image: &image::DynamicImage,
        source_mask: &image::GrayImage,
    ) -> Result<Vec<u8>> {
        ensure!(
            colorized_image.width() == source_mask.width(),
            "Source and mask must be the same size"
        );
        ensure!(
            colorized_image.height() == source_mask.height(),
            "Source and mask must be the same size"
        );

        let color_multiplier = self.as_rgb_float();
        let mut colorized_image = colorized_image.to_rgba8();
        colorized_image
            .pixels_mut()
            .zip(source_mask.pixels())
            .for_each(|(pixel, mask)| {
                let mask_value = mask.0[0] as f32 / 255.0;
                *pixel = Rgba([
                    (pixel.0[0] as f32 * lerp(1.0, color_multiplier.0, mask_value)) as u8,
                    (pixel.0[1] as f32 * lerp(1.0, color_multiplier.1, mask_value)) as u8,
                    (pixel.0[2] as f32 * lerp(1.0, color_multiplier.2, mask_value)) as u8,
                    pixel.0[3],
                ]);
            });
        let mut bytes: Vec<u8> = Vec::new();
        colorized_image
            .write_to(&mut Cursor::new(&mut bytes), image::ImageOutputFormat::Png)
            .unwrap();
        Ok(bytes)
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

/// A type that can have a color applied to it
pub trait Colorize {
    type Output;
    fn colorize(&self, color: Color) -> Self::Output;
}

impl Colorize for TextureName {
    type Output = TextureName;
    fn colorize(&self, color: Color) -> Self::Output {
        TextureName(format!("{}_{}", self.0, color.as_string()))
    }
}
impl Colorize for StaticTextureName {
    type Output = TextureName;
    fn colorize(&self, color: Color) -> Self::Output {
        TextureName(format!("{}_{}", self.0, color.as_string()))
    }
}
impl Colorize for BlockName {
    type Output = BlockName;
    fn colorize(&self, color: Color) -> Self::Output {
        BlockName(format!("{}_{}", self.0, color.as_string()))
    }
}
impl Colorize for StaticBlockName {
    type Output = BlockName;
    fn colorize(&self, color: Color) -> Self::Output {
        BlockName(format!("{}_{}", self.0, color.as_string()))
    }
}

pub fn register_dyes(game_builder: &mut GameBuilder) -> Result<()> {
    let base_texture = image::load_from_memory(include_bytes!("textures/dye.png"))?;
    for color in ALL_COLORS {
        let item_name = color.dye_item_name();
        let texture_bytes = color.colorize_to_png(&base_texture)?;
        let texture_name = TextureName(format!("colors:dye_{}.png", color.as_string()));
        let display_name = format!("{} dye", color.as_display_string());
        game_builder.register_texture_bytes(&texture_name, &texture_bytes)?;
        game_builder.register_basic_item(
            ItemName(item_name),
            display_name,
            texture_name,
            vec![
                item_groups::DYES.to_string(),
                color.color_group_name().to_string(),
            ],
        )?;
    }
    // TODO crafting recipes
    Ok(())
}
