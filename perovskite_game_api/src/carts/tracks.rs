// For now, hide unused warnings since they're distracting
#![allow(dead_code)]

use std::num::NonZeroU8;

use anyhow::{Context, Result};
use cgmath::{vec3, InnerSpace, Vector3};
use perovskite_core::constants::item_groups;
use perovskite_core::{
    block_id::{special_block_defs::AIR_ID, BlockId},
    chat::ChatMessage,
    constants::{block_groups, items::default_item_interaction_rules},
    coordinates::BlockCoordinate,
    protocol::{
        self,
        items::{item_def::QuantityType, item_stack},
        render::DynamicCrop,
    },
};
use perovskite_server::game_state::{
    client_ui::{PopupAction, PopupResponse, UiElementContainer},
    entities::{DeferrableResult, Deferral},
    event::EventInitiator,
    game_map::{CasOutcome, ServerGameMap},
    items::Item,
};

use crate::{
    blocks::{variants, AaBoxProperties, AxisAlignedBoxesAppearanceBuilder, BlockBuilder},
    default_game::{recipes::RecipeSlot, DefaultGameBuilder},
    game_builder::{BlockName, StaticBlockName, StaticTextureName},
    include_texture_bytes,
};

use super::{b2vec, CartsGameBuilderExtension};

#[rustfmt::skip]
pub(crate) mod c {
    // We only have 12 bits for the variant stored in the block...
    pub(crate) const ROTATION_BITS: u16 = 0b0000_0000_0000_0011;
    pub(crate) const X_SELECTOR: u16 = 0b0000_0000_0011_1100;
    pub(crate) const X_SELECTOR_DIV: u16 = 0b0000_0000_0000_0100;
    pub(crate) const Y_SELECTOR: u16 = 0b0000_0011_1100_0000;
    pub(crate) const Y_SELECTOR_DIV: u16 = 0b0000_0000_0100_0000;
    pub(crate) const FLIP_X_BIT: u16 = 0b0000_0100_0000_0000;
    // But we can still use the upper four bits for our own tracking, as long as
    // they don't need to be encoded on the game map

    /// If set on a secondary or tertiary, it means that the secondary/tertiary only
    /// applies when the diverging route is active
    ///
    /// If set on a prev/next, it means that we should enter that track
    pub(crate) const DIVERGING_ROUTE: u16 = 0b0001_0000_0000_0000;
    pub(crate) const REVERSE_SCAN: u16 = 0b0010_0000_0000_0000;
    pub(crate) const SLOPE_ENCODING: u16 = 0b0100_0000_0000_0000;
    pub(crate) const ENTRY_PRESENT: u16 = 0b1000_0000_0000_0000;
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct TileId(u16);

impl TileId {
    const fn new(
        x: u16,
        y: u16,
        rotation: u16,
        flip_x: bool,
        reverse: bool,
        diverging: bool,
    ) -> TileId {
        if x >= 16 {
            panic!("x too big, should be < 16");
        }
        if y >= 11 {
            panic!("y too big, should be < 11");
        }
        if rotation >= 4 {
            panic!("rotation too big, should be < 4");
        }
        TileId(
            c::ENTRY_PRESENT
                | x * c::X_SELECTOR_DIV
                | y * c::Y_SELECTOR_DIV
                | (rotation & c::ROTATION_BITS)
                | if flip_x { c::FLIP_X_BIT } else { 0 }
                | if reverse { c::REVERSE_SCAN } else { 0 }
                | if diverging { c::DIVERGING_ROUTE } else { 0 },
        )
    }
    fn new_slope(numerator: u16, denominator: u16, rotation: u16, reverse: bool) -> TileId {
        if denominator >= 16 {
            panic!("denominator too big, should be < 16");
        }
        if numerator > denominator {
            panic!("numerator too big, should be <= denominator");
        }
        TileId(
            c::ENTRY_PRESENT
                | c::SLOPE_ENCODING
                | numerator * c::X_SELECTOR_DIV
                | denominator * c::Y_SELECTOR_DIV
                | (rotation & c::ROTATION_BITS)
                | if reverse { c::REVERSE_SCAN } else { 0 },
        )
    }

    fn try_new(
        x: u16,
        y: u16,
        rotation: u16,
        flip_x: bool,
        reverse: bool,
        diverging: bool,
    ) -> Result<TileId, &'static str> {
        if x >= 16 {
            return Err("x too big, should be < 16");
        }
        if y >= 11 {
            return Err("y too big, should be < 11");
        }
        if rotation >= 4 {
            return Err("rotation too big, should be < 4");
        }
        Ok(TileId::new(x, y, rotation, flip_x, reverse, diverging))
    }
    const fn empty() -> TileId {
        TileId(0)
    }
    const fn present(&self) -> bool {
        (self.0 & c::ENTRY_PRESENT) != 0
    }
    const fn is_slope_encoding(&self) -> bool {
        (self.0 & c::SLOPE_ENCODING) != 0
    }
    const fn x(&self) -> u16 {
        (self.0 & c::X_SELECTOR) / c::X_SELECTOR_DIV
    }
    const fn y(&self) -> u16 {
        (self.0 & c::Y_SELECTOR) / c::Y_SELECTOR_DIV
    }
    const fn is_same_x_y(&self, other: TileId) -> bool {
        (self.0 & (c::X_SELECTOR | c::Y_SELECTOR | c::SLOPE_ENCODING))
            == (other.0 & (c::X_SELECTOR | c::Y_SELECTOR | c::SLOPE_ENCODING))
    }
    const fn same_variant(&self, other: TileId) -> bool {
        (self.0
            & (c::X_SELECTOR
                | c::Y_SELECTOR
                | c::ROTATION_BITS
                | c::FLIP_X_BIT
                | c::SLOPE_ENCODING))
            == (other.0
                & (c::X_SELECTOR
                    | c::Y_SELECTOR
                    | c::ROTATION_BITS
                    | c::FLIP_X_BIT
                    | c::SLOPE_ENCODING))
    }
    /// The rotation, clockwise, in units of 90 degrees. Always 0, 1, 2, or 3.
    pub(crate) const fn rotation(&self) -> u16 {
        self.0 & c::ROTATION_BITS
    }
    /// Whether X coordinates (in scanning, coordinate output, and drawing) are flipped
    const fn flip_x(&self) -> bool {
        (self.0 & c::FLIP_X_BIT) != 0
    }
    /// Whether to scan next->prev rather than prev->next
    const fn reverse(&self) -> bool {
        (self.0 & c::REVERSE_SCAN) != 0
    }
    const fn diverging(&self) -> bool {
        (self.0 & c::DIVERGING_ROUTE) != 0
    }
    const fn block_variant(&self) -> u16 {
        self.0 & (0b0000_1111_1111_1111)
    }

    const fn from_variant(variant: u16, reverse: bool, diverging: bool) -> Self {
        TileId(
            c::ENTRY_PRESENT
                | (variant & 0b0000_1111_1111_1111)
                | if reverse { c::REVERSE_SCAN } else { 0 }
                | if diverging { c::DIVERGING_ROUTE } else { 0 },
        )
    }

    const fn with_rotation_wrapped(&self, rotation: u16) -> TileId {
        TileId(self.0 & !c::ROTATION_BITS | (rotation & c::ROTATION_BITS))
    }

    const fn xor_flip_x(&self, flip_x: bool) -> TileId {
        TileId(self.0 ^ if flip_x { c::FLIP_X_BIT } else { 0 })
    }

    const fn correct_rotation_for_x_flip(&self, flip_x: bool) -> TileId {
        // If facing 1/3, then correct for x-flip
        if self.rotation() & 1 == 1 && flip_x {
            TileId(self.0 ^ 2)
        } else {
            *self
        }
    }

    const fn rotation_flip_bitset_mask(&self) -> u8 {
        1 << ((self.rotation() as u8) | (if self.flip_x() { 4 } else { 0 }))
    }

    fn with_reverse_diverge(&self, reverse: bool, diverge: bool) -> TileId {
        TileId(
            self.0 & (!c::REVERSE_SCAN & !c::DIVERGING_ROUTE)
                | if reverse { c::REVERSE_SCAN } else { 0 }
                | if diverge { c::DIVERGING_ROUTE } else { 0 },
        )
    }
}

impl std::fmt::Debug for TileId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.present() {
            f.debug_struct("TileId")
                .field("slope_encoding", &self.is_slope_encoding())
                .field("x", &self.x())
                .field("y", &self.y())
                .field("rotation", &self.rotation())
                .field("flip_x", &self.flip_x())
                .field("diverging", &self.diverging())
                .field("reverse", &self.reverse())
                .field("raw", &self.0)
                .finish()
        } else {
            f.write_str("TileId { .. }")
        }
    }
}

/// An offset in MAP coordinates
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct EncodedDelta(u8);

impl EncodedDelta {
    const PRESENT: u8 = 0b1000_0000;
    const Y_UP: u8 = 0b0100_0000;
    const fn new(x: i8, z: i8) -> EncodedDelta {
        if x < -1 || x > 1 {
            panic!();
        }
        if z < -1 || z > 1 {
            panic!();
        }
        let x_enc = (x + 1) as u8;
        let y_enc = (z + 1) as u8;
        EncodedDelta((x_enc << 2) | y_enc | Self::PRESENT)
    }
    const fn new_y_up(x: i8, z: i8) -> EncodedDelta {
        if x < -1 || x > 1 {
            panic!();
        }
        if z < -1 || z > 1 {
            panic!();
        }
        let x_enc = (x + 1) as u8;
        let y_enc = (z + 1) as u8;
        EncodedDelta((x_enc << 2) | y_enc | Self::PRESENT | Self::Y_UP)
    }
    const fn present(&self) -> bool {
        (self.0 & Self::PRESENT) != 0
    }
    const fn x(&self) -> i8 {
        ((self.0 & 0b1100) >> 2) as i8 - 1
    }
    const fn y(&self) -> i8 {
        if (self.0 & Self::Y_UP) != 0 {
            1
        } else {
            0
        }
    }
    const fn z(&self) -> i8 {
        (self.0 & 0b11) as i8 - 1
    }
    const fn empty() -> EncodedDelta {
        EncodedDelta(0)
    }
    fn eval_rotation(&self, flip_x: bool, rotation: u16) -> (i8, i8) {
        eval_rotation(self.x(), self.z(), flip_x, rotation)
    }
    fn eval_delta(
        &self,
        base: BlockCoordinate,
        flip_x: bool,
        rotation: u16,
    ) -> Option<BlockCoordinate> {
        let (x, z) = self.eval_rotation(flip_x, rotation);
        let y = self.y();
        base.try_delta(x as i32, y as i32, z as i32)
    }
}

impl std::fmt::Debug for EncodedDelta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.present() {
            f.debug_struct("EncodedDelta")
                .field("x", &self.x())
                .field("z", &self.z())
                .field("raw", &self.0)
                .finish()
        } else {
            f.write_str("EncodedDelta { .. }")
        }
    }
}

pub(crate) fn eval_rotation<T: Copy + std::ops::Neg<Output = T>>(
    x: T,
    z: T,
    flip_x: bool,
    rotation: u16,
) -> (T, T) {
    let x = if flip_x { -x } else { x };
    let z = z;
    match rotation & 3 {
        0 => (x, z),
        1 => (z, -x),
        2 => (-x, -z),
        3 => (-z, x),
        _ => unreachable!(),
    }
}

/* In texture atlas order

tile_x ->

    prev_tile
|------------------|------------------|     tile_y
|  #            #  |                  |      vvvv
|  #============#  |                  |
|  #            #  |  secondary_tile  |
|  #============#  |    (if needed)   |
|  #            #  |                  |
|  #============#  |                  |
|  #            #  |                  |
|------------------|------------------|
|                  |                  |
|                  |                  |
|  next_tile       |                  |
|                  |                  |
|                  |                  |
|                  |                  |
|                  |                  |
|------------------|------------------|

Note: Variant rotates CLOCKWISE

Game map coordinates for variant = 0
Top view
X <--+
     |
     V

     Z
*/

const TN_CONTROL_PRESENT: u8 = 1;
const TN_CONTROL_FLIP_X: u8 = 2;
/// 320 km/h or so, in m/s
const TRACK_INHERENT_MAX_SPEED: u8 = 90;

#[derive(Debug, Clone, Copy)]
struct TrackTile {
    /// The next tile that we will scan. This is given for variant = 0, and internally rotated by the engine when we're scanning other variants
    next_delta: EncodedDelta,
    /// The previous tile that we will scan. This is given for variant = 0, and internally rotated by the engine when we're scanning other variants
    prev_delta: EncodedDelta,
    /// The next tile that we will scan if we're diverging. This is given for variant = 0, and internally rotated by the engine when we're scanning other variants
    next_diverging_delta: EncodedDelta,
    /// The previous tile that we will scan if we're diverging. This is given for variant = 0, and internally rotated by the engine when we're scanning other variants
    prev_diverging_delta: EncodedDelta,
    /// Secondary tile that needs to be set. As with before, this is given for variant = 0, and internally rotated by the engine when we're scanning other variants
    secondary_coord: EncodedDelta,
    /// The tile we expect at the secondary tile. If any tile ID in this array is set to diverging, but we're not currently diverging, we'll accept it regardless of what the
    /// actual secondary tile is
    secondary_tile: [TileId; 2],
    /// Tertiary tile that needs to be set. As with before, this is given for variant = 0, and internally rotated by the engine when we're scanning other variants
    tertiary_coord: EncodedDelta,
    /// The tile we expect at the tertiary tile. If the tile ID says diverging, we only need this tile to be set if we're diverging
    tertiary_tile: TileId,
    /// The variants that can exist in the next tile if we're signalled straight through (or if there is no choice).
    /// The tile ID for (0,0) is special: it means any of the STRAIGHT_TRACK_ELIGIBLE_CONNECTIONS (i.e. the tracks that line up to a straight track)
    /// Note that even for that tile ID, the rotation bits in the tile ID are applied to figure out the direction in which we enter that straight-track-compatible
    /// connection, whatever it may be
    allowed_next_tiles: [TileId; 3],
    /// The variants that can exist in the next tile if we're signalled to diverge.
    allowed_next_diverging_tiles: [TileId; 2],
    /// The variants that can exist in the prev tile if we're signalled straight through (or if there is no choice).
    /// The tile ID for (0,0) is special: it means any of the STRAIGHT_TRACK_ELIGIBLE_CONNECTIONS (i.e. the tracks that line up to a straight track)
    allowed_prev_tiles: [TileId; 2],
    /// The variants that can exist in the prev tile if we're signalled to diverge.
    allowed_prev_diverging_tiles: [TileId; 2],
    /// Bitset representing the eligible straight track connections that arrive forward, non-diverging into this tile. See [TrackTile::rotation_flip_bitset_mask]
    straight_track_eligible_connections: u8,
    /// Bitset representing the eligible straight track connections that arrive reverse, non-diverging into this tile. See [TrackTile::rotation_flip_bitset_mask]
    reverse_straight_track_eligible_connections: u8,
    /// Bitset representing the eligible straight track connections that arrive forward, diverging into this tile. See [TrackTile::rotation_flip_bitset_mask]
    diverging_straight_track_eligible_connections: u8,
    /// Bitset representing the eligible straight track connections that arrive reverse, diverging into this tile. See [TrackTile::rotation_flip_bitset_mask]
    reverse_diverging_straight_track_eligible_connections: u8,
    /// The actual physical position of the cart on the tile, expressed in 1/128ths of a tile
    physical_x_offset: i8,
    /// The actual physical position of the cart on the tile, expressed in 1/128ths of a tile
    physical_z_offset: i8,
    /// When diverging, the actual physical position of the cart on the tile, expressed in 1/128ths of a tile
    diverging_physical_x_offset: i8,
    /// When diverging, the actual physical position of the cart on the tile, expressed in 1/128ths of a tile
    diverging_physical_z_offset: i8,
    /// The max speed, in meters per second, for a non-diverging move through the tile
    max_speed: u8,
    /// The max speed, in meters per second, for a diverging move through the tile
    diverging_max_speed: u8,
    /// Bitfield of which physical directions are considered straight-through, when spawning a cart
    /// Bits 0..3 are for forward movements, bits 4..7 are for reverse
    straight_through_spawn_dirs: u8,
    /// Bitfield of which physical directions are considered diverging, when spawning a cart
    /// Bits 0..3 are for forward movements, bits 4..7 are for reverse
    diverging_dirs_spawn_dirs: u8,
    /// If nonzero, this track tile can start/end a diverging move.
    ///
    /// When this is nonzero, the following additional conditions are imposed:
    ///     * In the forward and forward diverging directions, a single incoming path splits into the normal and diverging paths
    ///     * In the reverse and reverse diverging directions, the incoming diverging and normal paths converge onto a normal path
    /// If these preconditions are violated, the behavior of the interlocking scanner may be abnormal.
    ///
    /// When nonzero, the value indicates the length of the switch - when this counts down to zero, the switch can be released.
    /// This ensures that carts don't sideswipe each other at a switch while still diverging.
    ///
    /// Note that the upper bit may become reserved if we support slip switches; in that case, the required preconditions above will apply
    /// for the non-slip-switch case.
    switch_length: u8,
}

impl TrackTile {
    // This can't be in Default because it needs to be const
    const fn default() -> Self {
        TrackTile {
            next_delta: EncodedDelta::empty(),
            prev_delta: EncodedDelta::empty(),
            next_diverging_delta: EncodedDelta::empty(),
            prev_diverging_delta: EncodedDelta::empty(),
            secondary_coord: EncodedDelta::empty(),
            secondary_tile: [TileId::empty(); 2],
            tertiary_coord: EncodedDelta::empty(),
            tertiary_tile: TileId::empty(),
            allowed_next_tiles: [TileId::empty(); 3],
            allowed_next_diverging_tiles: [TileId::empty(); 2],
            allowed_prev_tiles: [TileId::empty(); 2],
            allowed_prev_diverging_tiles: [TileId::empty(); 2],
            straight_track_eligible_connections: 0,
            reverse_straight_track_eligible_connections: 0,
            diverging_straight_track_eligible_connections: 0,
            reverse_diverging_straight_track_eligible_connections: 0,
            physical_x_offset: 0,
            physical_z_offset: 0,
            diverging_physical_x_offset: 0,
            diverging_physical_z_offset: 0,
            max_speed: TRACK_INHERENT_MAX_SPEED,
            diverging_max_speed: TRACK_INHERENT_MAX_SPEED,
            straight_through_spawn_dirs: 0b0100_0001,
            diverging_dirs_spawn_dirs: 0,
            switch_length: 0,
        }
    }
}

fn make_physical_offset(num: impl Into<f32>, den: impl Into<f32>) -> i8 {
    let num: f32 = num.into();
    let den: f32 = den.into();
    let val = ((num / den) * 128.0).round();
    if !val.is_finite() {
        panic!("Invalid diverging offset (not finite): {:?}", val);
    }
    if val < -128.0 || val > 127.0 {
        panic!("Invalid diverging offset (overflows): {:?}", val);
    }
    val as i8
}

// The basic track at (0,0) has lots of connections, so they're listed here rather than making each TRACK_TILES larger.
// Note that because the straight track connection is symmetric, flip_x will be evaluated for both true and false.
/*
        |------------------|------------------|     tile_y
        |                  |  # straight   #  |      vvvv
        |                  |  # or         #  |
        |                  |  # compatible #  |
        |                  |  # track      #  |
        |                  |  #            #  |
        |                  |  #  scan dir  #  |
        |                  |  #   vvvvvv   #  |
        |------------------|------------------|
        |                  |  what tiles can  |
        |                  | we see in this   |
        |                  | spot?            |
        |                  |                  |
        |                  |                  |
        |                  |                  |
        |                  |                  |
        |------------------|------------------|
Game map coordinates for variant = 0
Top view
X <--+
     |
     V

     Z
*/

fn build_track_tiles() -> ([[Option<TrackTile>; 16]; 11], Vec<Template>) {
    let mut track_tiles = [[None; 16]; 11];
    let mut templates = Vec::new();

    // 8, 8 is the curve tile
    /*
           tile_x ->

           prev_tile
       |------------------|------------------|     tile_y
       |                  |                  |      vvvv
       |                  |                  |
       |                  |  prev_tile       |
       |                  |                  |
       |                  |                  |
       |                  |                  |
       |                  |                  |
       |------------------|------------------|
       |                  | /            |   |
       |                  |-      |      |   |
       |  next_tile       |       V      |   |
       |                  |      scan    /   |
       |                  | direction   /    |
       |                  |            /     |
       |                  |-----------/      |
       |------------------|------------------|
        Game map coordinates for variant = 0
        Top view
        X <--+
             |
             V

             Z
    */
    // [0][0] is the straight track, running in the +/- Z direction
    track_tiles[0][0] = Some(TrackTile {
        next_delta: EncodedDelta::new(0, 1),
        prev_delta: EncodedDelta::new(0, -1),
        allowed_next_tiles: [
            // Connects to a straight track (or straight track compatible) with the same rotation on its output side
            TileId::new(0, 0, 0, false, false, false),
            TileId::empty(),
            TileId::empty(),
        ],
        allowed_prev_tiles: [TileId::new(0, 0, 2, false, false, false), TileId::empty()],
        straight_track_eligible_connections: TileId::new(0, 0, 0, false, false, false)
            .rotation_flip_bitset_mask()
            | TileId::new(0, 0, 0, true, false, false).rotation_flip_bitset_mask(),
        reverse_straight_track_eligible_connections: TileId::new(0, 0, 2, false, true, false)
            .rotation_flip_bitset_mask()
            | TileId::new(0, 0, 2, true, true, false).rotation_flip_bitset_mask(),
        ..TrackTile::default()
    });
    // [8][8] is the 90 degree curve without a switch
    track_tiles[8][8] = Some(TrackTile {
        next_delta: EncodedDelta::new(1, 0),
        prev_delta: EncodedDelta::new(0, -1),
        allowed_next_tiles: [
            TileId::new(0, 0, 1, false, false, false),
            TileId::empty(),
            TileId::empty(),
        ],
        allowed_prev_tiles: [TileId::new(0, 0, 2, false, false, false), TileId::empty()],
        straight_track_eligible_connections: TileId::new(8, 8, 0, false, false, false).rotation_flip_bitset_mask()
            // including if X is flipped
            | TileId::new(8, 8, 0, true, false, false).rotation_flip_bitset_mask(),
        reverse_straight_track_eligible_connections: TileId::new(8, 8, 1, false, true, false)
            .rotation_flip_bitset_mask()
            | TileId::new(8, 8, 3, true, true, false).rotation_flip_bitset_mask(),
        max_speed: 10,
        ..TrackTile::default()
    });

    // CONTINUE HERE
    // [0][1] and [1][1] start the 8-block switch (folded in half to make a 4 block segment in the tile space)
    let (t1, t2, t3) = build_folded_switch(&mut track_tiles, 4, 0, 1, 6, 0, 1, 40);
    templates.push(t1);
    templates.push(t2);
    templates.push(t3);

    let (t1, t2, t3) = build_folded_switch(&mut track_tiles, 8, 2, 0, 8, 0, 2, 85);
    templates.push(t1);
    templates.push(t2);
    templates.push(t3);

    templates.sort_by(|a, b| {
        a.category
            .cmp(&b.category)
            .then(a.sort_subkey.cmp(&b.sort_subkey))
    });

    for len in [8, 16, 32, 64, 128, 256] {
        let template = build_straight_track_template(len);
        templates.push(template);
    }

    build_slope_templates(&mut templates);

    (track_tiles, templates)
}

fn build_slope_templates(templates: &mut Vec<Template>) {
    let mut template_tiles_8 = vec![];
    for i in 1..=8 {
        template_tiles_8.push(TemplateEntry {
            tile_id: TileId::new_slope(i, 8, 0, false),
            offset_x: 0,
            offset_y: 0,
            offset_z: (i - 1) as i32,
            tracks_consumed: 1,
        })
    }
    templates.push(Template {
        category: "Slopes".to_string(),
        sort_subkey: 8,
        name: "8-block slope up".to_string(),
        id: "slope_8".to_string(),
        entries: template_tiles_8.into_boxed_slice(),
        bifurcate: false,
    });

    let mut template_tiles_8_down = vec![];
    for i in 1..=8 {
        template_tiles_8_down.push(TemplateEntry {
            tile_id: TileId::new_slope(9 - i, 8, 2, false),
            offset_x: 0,
            offset_y: -1,
            offset_z: i as i32,
            tracks_consumed: 1,
        })
    }
    templates.push(Template {
        category: "Slopes".to_string(),
        sort_subkey: 9,
        name: "8-block slope down".to_string(),
        id: "slope_8_down".to_string(),
        entries: template_tiles_8_down.into_boxed_slice(),
        bifurcate: false,
    })
}

fn build_straight_track_template(len: u16) -> Template {
    let mut tiles = vec![];
    for y in 0..len {
        tiles.push(TemplateEntry {
            tile_id: TileId::new(0, 0, 0, false, false, false),
            offset_x: 0,
            offset_y: 0,
            offset_z: y as i32,
            tracks_consumed: 1,
        });
    }
    Template {
        category: "Straight Track".to_string(),
        sort_subkey: len,
        name: format!("{} blocks", len),
        id: format!("straight_track_{}", len),
        entries: tiles.into_boxed_slice(),
        bifurcate: false,
    }
}

fn build_folded_switch(
    track_tiles: &mut [[Option<TrackTile>; 16]; 11],
    switch_half_len: u16,
    switch_xmin: u16,
    switch_ymin: u16,
    diag_xmin: u16,
    diag_ymin: u16,
    skip_secondary_tiles: u16,
    max_turnout_speed: u8,
) -> (Template, Template, Template) {
    let mut switch_template_tiles = Vec::new();
    let mut half_switch_template_tiles = Vec::new();
    let mut diag_template_tiles = Vec::new();

    for y in 0..switch_half_len {
        // The tiles on column 0 only carry straight-through traffic. Diverging traffic is always on column 1 tiles,
        // with the column 0 tile being merely a secondary tile.
        // We can just copy the 0,0 straight track into here

        let switch_primary = TemplateEntry {
            tile_id: TileId::new(switch_xmin, switch_ymin + y, 0, false, false, false),
            offset_x: 1,
            offset_y: 0,
            offset_z: y as i32,
            tracks_consumed: 2,
        };
        let switch_secondary = TemplateEntry {
            tile_id: TileId::new(switch_xmin + 1, switch_ymin + y, 0, false, false, false),
            offset_x: 0,
            offset_y: 0,
            offset_z: y as i32,
            tracks_consumed: 2,
        };
        let diag_primary = TemplateEntry {
            tile_id: TileId::new(diag_xmin, diag_ymin + y, 0, false, false, false),
            offset_x: 1,
            offset_y: 0,
            offset_z: y as i32,
            tracks_consumed: 1,
        };
        let diag_secondary = TemplateEntry {
            tile_id: TileId::new(diag_xmin + 1, diag_ymin + y, 0, false, false, false),
            offset_x: 0,
            offset_y: 0,
            offset_z: y as i32,
            tracks_consumed: 1,
        };

        switch_template_tiles.push(switch_primary);
        switch_template_tiles.push(switch_secondary);

        half_switch_template_tiles.push(diag_primary);
        half_switch_template_tiles.push(switch_secondary);

        diag_template_tiles.push(diag_primary);
        diag_template_tiles.push(diag_secondary);

        let switch_farside = TemplateEntry {
            tile_id: TileId::new(switch_xmin + 1, switch_ymin + y, 2, false, false, false),
            offset_x: 1,
            offset_y: 0,
            offset_z: (2 * switch_half_len - 1 - y) as i32,
            tracks_consumed: 2,
        };
        let switch_farside_secondary = TemplateEntry {
            tile_id: TileId::new(switch_xmin, switch_ymin + y, 2, false, false, false),
            offset_x: 0,
            offset_y: 0,
            offset_z: (2 * switch_half_len - 1 - y) as i32,
            tracks_consumed: 2,
        };
        let diag_farside = TemplateEntry {
            tile_id: TileId::new(diag_xmin + 1, diag_ymin + y, 2, false, false, false),
            offset_x: 1,
            offset_y: 0,
            offset_z: (2 * switch_half_len - 1 - y) as i32,
            tracks_consumed: 1,
        };
        let diag_farside_secondary = TemplateEntry {
            tile_id: TileId::new(diag_xmin, diag_ymin + y, 2, false, false, false),
            offset_x: 0,
            offset_y: 0,
            offset_z: (2 * switch_half_len - 1 - y) as i32,
            tracks_consumed: 1,
        };

        switch_template_tiles.push(switch_farside);
        switch_template_tiles.push(switch_farside_secondary);
        half_switch_template_tiles.push(diag_farside);
        half_switch_template_tiles.push(switch_farside_secondary);
        diag_template_tiles.push(diag_farside);
        diag_template_tiles.push(diag_farside_secondary);

        track_tiles[switch_xmin as usize][(switch_ymin + y) as usize] = track_tiles[0][0];
        track_tiles[switch_xmin as usize + 1][(switch_ymin + y) as usize] = Some(TrackTile {
            next_delta: EncodedDelta::new(0, 1),
            // At y = 4, we are hitting the fold-in-half
            next_diverging_delta: if y == (switch_half_len - 1) {
                EncodedDelta::new(1, 1)
            } else {
                EncodedDelta::new(0, 1)
            },

            prev_delta: EncodedDelta::new(0, -1),
            // Prev always converges back to the main track
            prev_diverging_delta: EncodedDelta::new(0, -1),
            allowed_next_tiles: [
                // The next tile should be straight-track-compatible
                TileId::new(0, 0, 0, false, false, false),
                TileId::empty(),
                TileId::empty(),
            ],
            allowed_next_diverging_tiles: if y == (switch_half_len - 1) {
                // We're at the fold-in-half. We want the opposite diverging tile. We deflected halfway off our starting
                // track, and now need to deflect back to the main track (in the new tile)
                // This can also be a non-switch
                [
                    // The main tile is the one with the main track, so we want (1,4). (0,4) will only be visited for
                    // non-diverging moves in the half of a crossover where the switch track is on the other side
                    TileId::new(
                        switch_xmin + 1,
                        switch_ymin + switch_half_len - 1,
                        2,
                        false,
                        true,
                        true,
                    ),
                    // The non-switch diagonal tiles are never marked diverging
                    TileId::new(
                        diag_xmin + 1,
                        diag_ymin + switch_half_len - 1,
                        2,
                        false,
                        true,
                        false,
                    ),
                ]
            } else {
                // The next tile should be the next tile in the set, still diverging
                // Or the non-switch diagonal tile with the same slope
                [
                    TileId::new(
                        switch_xmin + 1,
                        switch_ymin + (y + 1),
                        0,
                        false,
                        false,
                        true,
                    ),
                    TileId::new(diag_xmin + 1, diag_ymin + (y + 1), 0, false, false, false),
                ]
            },
            allowed_prev_tiles: [TileId::new(0, 0, 2, false, false, false), TileId::empty()],
            allowed_prev_diverging_tiles: if y == 0 {
                // If 0, diverging just merges back onto the main track
                [TileId::new(0, 0, 2, false, false, false), TileId::empty()]
            } else {
                [
                    TileId::new(switch_xmin + 1, switch_ymin + (y - 1), 0, false, true, true),
                    TileId::new(diag_xmin + 1, diag_ymin + (y - 1), 0, false, true, true),
                ]
            },
            straight_track_eligible_connections: TileId::new(0, 0, 0, false, false, false)
                .rotation_flip_bitset_mask()
                | TileId::new(0, 0, 0, true, false, false).rotation_flip_bitset_mask(),
            reverse_straight_track_eligible_connections: TileId::new(0, 0, 2, false, true, false)
                .rotation_flip_bitset_mask()
                | TileId::new(0, 0, 2, true, true, false).rotation_flip_bitset_mask(),

            // At X+1, we need the counterpart to our main tile.
            secondary_coord: if y >= skip_secondary_tiles {
                EncodedDelta::new(1, 0)
            } else {
                EncodedDelta::empty()
            },
            secondary_tile: [
                // Only needed if diverging
                // Need either the corresponding switch tile...
                TileId::new(switch_xmin, switch_ymin + y, 0, false, false, true),
                // ...or the corresponding diagonal tile. We still set diverging=true, because for this case of TileId, it's
                // a matter of whether our *current* state is diverging. The fact that the diag tiles have no diverging
                // moves is immaterial
                TileId::new(diag_xmin, diag_ymin + y, 0, false, false, true),
            ],
            diverging_physical_x_offset: make_physical_offset(
                2 * y + 1,
                switch_half_len.checked_mul(4).unwrap(),
            ),
            switch_length: if y == 0 {
                switch_half_len.checked_mul(2).unwrap().try_into().unwrap()
            } else {
                0
            },
            max_speed: TRACK_INHERENT_MAX_SPEED,
            diverging_max_speed: max_turnout_speed,
            ..TrackTile::default()
        });

        // The left diagonal tile is only secondary and is never actually traversed. We don't need to fill it in

        // We do define the right diagonal tile here
        track_tiles[diag_xmin as usize + 1][(diag_ymin + y) as usize] = Some(TrackTile {
            next_delta: if y == (switch_half_len - 1) {
                EncodedDelta::new(1, 1)
            } else {
                EncodedDelta::new(0, 1)
            },

            prev_delta: EncodedDelta::new(0, -1),
            // No diverging
            prev_diverging_delta: EncodedDelta::empty(),
            allowed_next_tiles: if y == (switch_half_len - 1) {
                // We're at the fold-in-half. We want the opposite diverging tile. We deflected halfway off our starting
                // track, and now need to deflect back to the main track (in the new tile)
                // This can also be a non-switch
                [
                    // The main tile is the one with the main track, so we want (1,4). (0,4) will only be visited for
                    // non-diverging moves in the half of a crossover where the switch track is on the other side
                    TileId::new(
                        switch_xmin + 1,
                        switch_ymin + switch_half_len - 1,
                        2,
                        false,
                        true,
                        true,
                    ),
                    // The non-switch diagonal tiles are never marked diverging
                    TileId::new(
                        diag_xmin + 1,
                        diag_ymin + switch_half_len - 1,
                        2,
                        false,
                        true,
                        false,
                    ),
                    TileId::empty(),
                ]
            } else {
                // The next tile should be the next tile in the set, still diverging
                // Or the non-switch diagonal tile with the same slope
                [
                    TileId::new(
                        switch_xmin + 1,
                        switch_ymin + (y + 1),
                        0,
                        false,
                        false,
                        true,
                    ),
                    TileId::new(diag_xmin + 1, diag_ymin + (y + 1), 0, false, false, false),
                    TileId::empty(),
                ]
            },
            allowed_prev_tiles: if y == 0 {
                // If 0, diverging just merges back onto the main track
                [TileId::new(0, 0, 2, false, false, false), TileId::empty()]
            } else {
                [
                    TileId::new(switch_xmin + 1, switch_ymin + (y - 1), 0, false, true, true),
                    TileId::new(diag_xmin + 1, diag_ymin + (y - 1), 0, false, true, false),
                ]
            },
            // Only connects to straight track if y == 0, and only at the input side, regardless of flip_x
            straight_track_eligible_connections: if y == 0 {
                // We can enter a straight track with the same rotation, regardless of flip_x
                // Note that X and Y don't matter here; we're just getting the bitset mask.
                TileId::new(0, 0, 0, false, false, false).rotation_flip_bitset_mask()
                    | TileId::new(0, 0, 0, true, false, false).rotation_flip_bitset_mask()
            } else {
                0
            },
            // Does not connect to any straight track on its output side
            reverse_straight_track_eligible_connections: 0,
            // At X+1, we need the counterpart to our main tile.
            secondary_coord: if y >= skip_secondary_tiles {
                EncodedDelta::new(1, 0)
            } else {
                EncodedDelta::empty()
            },
            secondary_tile: [
                // The diagonal move is non-diverging in the case of the diagonal. So we need the corresponding tile
                // to be there unconditionally.
                // Need either the corresponding switch tile...
                TileId::new(switch_xmin, switch_ymin + y, 0, false, false, false),
                // ...or the corresponding diagonal tile.
                TileId::new(diag_xmin, diag_ymin + y, 0, false, false, false),
            ],
            physical_x_offset: make_physical_offset(
                2 * y + 1,
                switch_half_len.checked_mul(4).unwrap(),
            ),
            max_speed: max_turnout_speed,
            ..TrackTile::default()
        });
    }

    switch_template_tiles.sort_by_key(|t| t.offset_x.abs() + t.offset_z.abs());
    half_switch_template_tiles.sort_by_key(|t| t.offset_x.abs() + t.offset_z.abs());
    diag_template_tiles.sort_by_key(|t| t.offset_x.abs() + t.offset_z.abs());

    (
        Template {
            category: "Crossover".to_string(),
            name: (switch_half_len * 2).to_string() + "-block",
            sort_subkey: switch_half_len,
            id: format!("crossover_{}", switch_half_len),
            entries: switch_template_tiles.into_iter().collect(),
            bifurcate: true,
        },
        Template {
            category: "Parallel Turnout".to_string(),
            name: (switch_half_len * 2).to_string() + "-block",
            sort_subkey: switch_half_len,
            id: format!("parallel_turnout_{}", switch_half_len),
            entries: half_switch_template_tiles.into_iter().collect(),
            bifurcate: true,
        },
        Template {
            category: "Diagonal".to_string(),
            name: (switch_half_len * 2).to_string() + "-block",
            sort_subkey: switch_half_len,
            id: format!("diagonal_{}", switch_half_len),
            entries: diag_template_tiles.into_iter().collect(),
            bifurcate: true,
        },
    )
}

lazy_static::lazy_static! {
    static ref TRACK_TILES: [[Option<TrackTile>; 16]; 11] = build_track_tiles().0;
    pub(crate) static ref TRACK_TEMPLATES: Vec<Template> = build_track_tiles().1;
    static ref SLOPE_TRACKS: [Option<TrackTile>; 16] = build_slope_tiles();
}

const RAIL_TILES_TEX: StaticTextureName = StaticTextureName("carts:rail_tile");
const RAIL_SIMPLE_TEX: StaticTextureName = StaticTextureName("carts:rail_simple");
const TRANSPARENT_PIXEL: StaticTextureName = StaticTextureName("carts:transparent_pixel");

pub(crate) fn register_tracks(
    game_builder: &mut crate::game_builder::GameBuilder,
) -> Result<(BlockId, BlockId, [BlockId; 8])> {
    const SIGNAL_SIDE_TOP_TEX: StaticTextureName = StaticTextureName("carts:signal_side_top");
    include_texture_bytes!(game_builder, RAIL_TILES_TEX, "textures/rail_atlas.png")?;
    include_texture_bytes!(game_builder, RAIL_SIMPLE_TEX, "textures/rail.png")?;
    include_texture_bytes!(game_builder, TRANSPARENT_PIXEL, "textures/transparent.png")?;

    let rail_tile_box = AaBoxProperties::new(
        TRANSPARENT_PIXEL,
        TRANSPARENT_PIXEL,
        RAIL_TILES_TEX,
        RAIL_TILES_TEX,
        TRANSPARENT_PIXEL,
        TRANSPARENT_PIXEL,
        crate::blocks::TextureCropping::AutoCrop,
        crate::blocks::RotationMode::RotateHorizontally,
    );
    let rail_tile = game_builder.add_block(
        BlockBuilder::new(StaticBlockName("carts:rail_tile"))
            .set_axis_aligned_boxes_appearance(
                AxisAlignedBoxesAppearanceBuilder::new().add_box_with_variant_mask_and_slope(
                    rail_tile_box,
                    (-0.5, 0.5),
                    (-0.5, -0.4375),
                    (-0.5, 0.5),
                    0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ),
            )
            .set_allow_light_propagation(true)
            .set_display_name("Railway track")
            .add_modifier(Box::new(|bt| {
                bt.interact_key_handler = Some(Box::new(|ctx, coord| {
                    let block = ctx.game_map().get_block(coord)?;
                    let tile_id = TileId::from_variant(block.variant(), false, false);
                    ctx.initiator()
                        .send_chat_message(ChatMessage::new("[INFO]", format!("{:?}", tile_id)))?;

                    Ok(Some(
                        ctx.new_popup()
                            .title("Mr. Yellow")
                            .label("Update tile:")
                            .text_field("tile_x", "Tile X", tile_id.x().to_string(), true, false)
                            .text_field("tile_y", "Tile Y", tile_id.y().to_string(), true, false)
                            .text_field(
                                "rotation",
                                "Rotation",
                                tile_id.rotation().to_string(),
                                true,
                                false,
                            )
                            .checkbox("flip_x", "Flip X", tile_id.flip_x(), true)
                            .button("apply", "Apply", true, true)
                            .label("Scan tile:")
                            .checkbox("reverse", "Reverse", false, true)
                            .checkbox("diverging", "Diverging", false, true)
                            .button("scan", "Scan", true, false)
                            .button("multiscan", "Multi-Scan", true, false)
                            .set_button_callback(Box::new(move |response: PopupResponse<'_>| {
                                match handle_popup_response(
                                    &response,
                                    coord,
                                    response
                                        .ctx
                                        .extension::<CartsGameBuilderExtension>()
                                        .as_ref()
                                        .context("CartsGameBuilderExtension missing")?,
                                ) {
                                    Ok(_) => {}
                                    Err(e) => {
                                        response.ctx.initiator().send_chat_message(
                                            ChatMessage::new(
                                                "[ERROR]",
                                                "Failed to parse popup response: ".to_string()
                                                    + e.to_string().as_str(),
                                            ),
                                        )?;
                                    }
                                }
                                Ok(())
                            })),
                    ))
                }));
                let ri = bt.client_info.render_info.as_mut().unwrap();
                match ri {
                    protocol::blocks::block_type_def::RenderInfo::AxisAlignedBoxes(aabb) => {
                        aabb.boxes.iter_mut().for_each(|b| {
                            b.tex_top.as_mut().unwrap().crop.as_mut().unwrap().dynamic =
                                Some(DynamicCrop {
                                    x_selector_bits: 0b0000_0000_0011_1100,
                                    y_selector_bits: 0b0000_0011_1100_0000,
                                    x_cells: 16,
                                    y_cells: 11,
                                    flip_x_bit: 0b0000_0100_0000_0000,
                                    flip_y_bit: 0,
                                    extra_flip_x: false,
                                    extra_flip_y: false,
                                });
                            b.tex_bottom
                                .as_mut()
                                .unwrap()
                                .crop
                                .as_mut()
                                .unwrap()
                                .dynamic = Some(DynamicCrop {
                                x_selector_bits: 0b0000_0000_0011_1100,
                                y_selector_bits: 0b0000_0011_1100_0000,
                                x_cells: 16,
                                y_cells: 11,
                                flip_x_bit: 0b0000_0100_0000_0000,
                                flip_y_bit: 0,
                                extra_flip_x: true,
                                extra_flip_y: false,
                            })
                        })
                    }
                    _ => unreachable!(),
                }
            })),
    )?;

    const CURVED_RAIL_ITEM_TEX: StaticTextureName = StaticTextureName("carts:curved_rail_item");
    include_texture_bytes!(
        game_builder,
        CURVED_RAIL_ITEM_TEX,
        "textures/curved_rail_item.png"
    )?;
    let rail_tile_id = rail_tile.id;
    let mut rail_slopes_8 = [BlockId::from(0); 8];
    game_builder.inner.items_mut().register_item(Item {
        proto: protocol::items::ItemDef {
            short_name: "carts:rail_curve".to_string(),
            display_name: "Curved rail".to_string(),
            inventory_texture: Some(CURVED_RAIL_ITEM_TEX.into()),
            groups: vec![],
            block_apperance: String::new(),
            interaction_rules: default_item_interaction_rules(),
            quantity_type: Some(QuantityType::Stack(256)),
            sort_key: "carts:rail_curve".to_string(),
        },
        dig_handler: None,
        tap_handler: None,
        place_handler: Some(Box::new(move |ctx, coord, _anchor, stack| {
            if stack.proto().quantity == 0 {
                return Ok(None);
            }
            let rotation = ctx
                .initiator()
                .position()
                .map(|pos| variants::rotate_nesw_azimuth_to_variant(pos.face_direction.0))
                .unwrap_or(0);
            let variant = rotation | TileId::new(8, 8, 0, false, false, false).block_variant();
            match ctx
                .game_map()
                .compare_and_set_block_predicate(
                    coord,
                    |block, _, block_types| {
                        // fast path
                        if block == AIR_ID {
                            return Ok(true);
                        };
                        let block_type = block_types.get_block(&block)?.0;
                        Ok(block_type
                            .client_info
                            .groups
                            .iter()
                            .any(|g| g == block_groups::TRIVIALLY_REPLACEABLE))
                    },
                    rail_tile_id.with_variant(variant)?,
                    None,
                )?
                .0
            {
                CasOutcome::Match => Ok(stack.decrement()),
                CasOutcome::Mismatch => Ok(Some(stack.clone())),
            }
        })),
    })?;

    for i in 0..8 {
        rail_slopes_8[i] = register_rail_slope(game_builder, i as u8 + 1, 8)?;
    }
    let rail_slope_1 = register_rail_slope(game_builder, 1, 1)?;

    game_builder.register_crafting_recipe(
        [
            RecipeSlot::Empty,
            RecipeSlot::Empty,
            RecipeSlot::Empty,
            RecipeSlot::Empty,
            RecipeSlot::Exact("carts:rail_tile".to_string()),
            RecipeSlot::Exact("carts:rail_tile".to_string()),
            RecipeSlot::Empty,
            RecipeSlot::Exact("carts:rail_tile".to_string()),
            RecipeSlot::Empty,
        ],
        "carts:rail_curve".to_string(),
        3,
        Some(item_stack::QuantityType::Stack(256)),
        false,
    );

    Ok((rail_tile.id, rail_slope_1, rail_slopes_8))
}

fn register_rail_slope(
    game_builder: &mut crate::game_builder::GameBuilder,
    numerator: u8,
    denominator: u8,
) -> Result<BlockId> {
    let rail_tile_box = AaBoxProperties::new(
        TRANSPARENT_PIXEL,
        TRANSPARENT_PIXEL,
        RAIL_SIMPLE_TEX,
        RAIL_SIMPLE_TEX,
        TRANSPARENT_PIXEL,
        TRANSPARENT_PIXEL,
        crate::blocks::TextureCropping::AutoCrop,
        crate::blocks::RotationMode::RotateHorizontally,
    );
    let y_bottom = (numerator as f32 - 0.5) / (denominator as f32) - 0.5;
    let y_top = y_bottom + (1.0 / 16.0);

    game_builder
        .add_block(
            BlockBuilder::new(BlockName(format!("rail_slope_{numerator}_{denominator}")))
                .set_axis_aligned_boxes_appearance(
                    AxisAlignedBoxesAppearanceBuilder::new().add_box_with_variant_mask_and_slope(
                        rail_tile_box,
                        (-0.5, 0.5),
                        (y_bottom, y_top),
                        (-0.5, 0.5),
                        0,
                        0.0,
                        1.0 / (denominator as f32),
                        0.0,
                        1.0 / (denominator as f32),
                    ),
                )
                .add_block_groups(if denominator == 1 {
                    vec![]
                } else {
                    vec![item_groups::HIDDEN_FROM_CREATIVE]
                })
                .set_allow_light_propagation(true)
                .set_display_name(format!("Slope {numerator}/{denominator}"))
                .add_modifier(Box::new(|bt| {
                    bt.interact_key_handler = Some(Box::new(|ctx, coord| {
                        let block = ctx.game_map().get_block(coord)?;
                        let tile_id = TileId::from_variant(block.variant(), false, false);
                        ctx.initiator().send_chat_message(ChatMessage::new(
                            "[INFO]",
                            format!("{:?}", tile_id),
                        ))?;

                        Ok(Some(
                            ctx.new_popup()
                                .title("Mr. Yellow")
                                .label("Update tile:")
                                .text_field(
                                    "tile_x",
                                    "Tile X",
                                    tile_id.x().to_string(),
                                    true,
                                    false,
                                )
                                .text_field(
                                    "tile_y",
                                    "Tile Y",
                                    tile_id.y().to_string(),
                                    true,
                                    false,
                                )
                                .text_field(
                                    "rotation",
                                    "Rotation",
                                    tile_id.rotation().to_string(),
                                    true,
                                    false,
                                )
                                .checkbox("flip_x", "Flip X", tile_id.flip_x(), true)
                                .button("apply", "Apply", true, true)
                                .label("Scan tile:")
                                .checkbox("reverse", "Reverse", false, true)
                                .checkbox("diverging", "Diverging", false, true)
                                .button("scan", "Scan", true, false)
                                .button("multiscan", "Multi-Scan", true, false)
                                .set_button_callback(Box::new(
                                    move |response: PopupResponse<'_>| {
                                        match handle_popup_response(
                                            &response,
                                            coord,
                                            response.ctx.extension().as_ref().unwrap(),
                                        ) {
                                            Ok(_) => {}
                                            Err(e) => {
                                                response.ctx.initiator().send_chat_message(
                                                    ChatMessage::new(
                                                        "[ERROR]",
                                                        "Failed to parse popup response: "
                                                            .to_string()
                                                            + e.to_string().as_str(),
                                                    ),
                                                )?;
                                            }
                                        }
                                        Ok(())
                                    },
                                )),
                        ))
                    }));
                })),
        )
        .map(|b| b.id)
}

fn build_slope_tiles() -> [Option<TrackTile>; 16] {
    let mut result = [None; 16];
    // the 1/1 steep track
    result[0] = Some(TrackTile {
        next_delta: EncodedDelta::new_y_up(0, 1),
        prev_delta: EncodedDelta::new(0, -1),
        next_diverging_delta: EncodedDelta::empty(),
        prev_diverging_delta: EncodedDelta::empty(),
        secondary_coord: EncodedDelta::empty(),
        secondary_tile: [TileId::empty(); 2],
        tertiary_coord: EncodedDelta::empty(),
        tertiary_tile: TileId::empty(),
        allowed_next_tiles: [
            // connects to a straight track
            TileId::new(0, 0, 0, false, false, false),
            TileId::empty(),
            TileId::empty(),
        ],
        allowed_next_diverging_tiles: [TileId::empty(); 2],
        allowed_prev_tiles: [TileId::new(0, 0, 2, false, false, false), TileId::empty()],
        allowed_prev_diverging_tiles: [TileId::empty(); 2],
        straight_track_eligible_connections: TileId::new(0, 0, 0, false, false, false)
            .rotation_flip_bitset_mask()
            // including ff X is flipped
            | TileId::new(0, 0, 0, true, false, false).rotation_flip_bitset_mask(),
        // Slopes are special: we enter them in reverse from a straight track one level HIGHER
        // Therefore, this case cannot be encoded in this field.
        reverse_straight_track_eligible_connections: 0,
        diverging_straight_track_eligible_connections: 0,
        reverse_diverging_straight_track_eligible_connections: 0,
        physical_x_offset: 0,
        physical_z_offset: 0,
        diverging_physical_x_offset: 0,
        diverging_physical_z_offset: 0,
        max_speed: 6,
        diverging_max_speed: 6,
        straight_through_spawn_dirs: 0b0100_0001,
        diverging_dirs_spawn_dirs: 0,
        switch_length: 0,
    });
    // The 1/8 slope tiles will use slots 1-8
    for i in 1..=8 {
        result[i] = Some(TrackTile {
            next_delta: if i == 8 {
                EncodedDelta::new_y_up(0, 1)
            } else {
                EncodedDelta::new(0, 1)
            },
            prev_delta: EncodedDelta::new(0, -1),
            next_diverging_delta: EncodedDelta::empty(),
            prev_diverging_delta: EncodedDelta::empty(),
            secondary_coord: EncodedDelta::empty(),
            secondary_tile: [TileId::empty(); 2],
            tertiary_coord: EncodedDelta::empty(),
            tertiary_tile: TileId::empty(),
            allowed_next_tiles: [
                // connects to a straight track
                if i == 8 {
                    TileId::new(0, 0, 0, false, false, false)
                } else {
                    TileId::new_slope((i + 1) as u16, 8, 0, false)
                },
                TileId::empty(),
                TileId::empty(),
            ],
            allowed_next_diverging_tiles: [TileId::empty(); 2],
            allowed_prev_tiles: [
                if i == 1 {
                    TileId::new(0, 0, 2, false, false, false)
                } else {
                    TileId::new_slope((i - 1) as u16, 8, 0, true)
                },
                TileId::empty(),
            ],
            allowed_prev_diverging_tiles: [TileId::empty(); 2],
            straight_track_eligible_connections: if i == 1 {
                // We can enter this track in its forward scan direction
                TileId::new(0, 0, 0, false, false, false)
                    .rotation_flip_bitset_mask()
                    // including if X is flipped
                    | TileId::new(0, 0, 0, true, false, false).rotation_flip_bitset_mask()
            } else {
                0
            },
            // Slopes are special: we enter them in reverse from a straight track one level HIGHER
            // Therefore, this case cannot be encoded in this field. Instead, we check the
            // denominator accordingly.
            reverse_straight_track_eligible_connections: 0,
            diverging_straight_track_eligible_connections: 0,
            reverse_diverging_straight_track_eligible_connections: 0,
            physical_x_offset: 0,
            physical_z_offset: 0,
            diverging_physical_x_offset: 0,
            diverging_physical_z_offset: 0,
            max_speed: 60,
            diverging_max_speed: 6,
            straight_through_spawn_dirs: 0b0100_0001,
            diverging_dirs_spawn_dirs: 0,
            switch_length: 0,
        });
    }
    result
}

pub(crate) enum ScanOutcome {
    /// We advanced, here's the new state
    Success(ScanState),
    /// We were on a track before, but we cannot advance because the track is ending or incomplete
    CannotAdvance,
    /// We're not even on a track
    NotOnTrack,
    /// We got a deferral while reading the map
    Deferral(Deferral<Result<BlockId>, BlockCoordinate>),
}
impl std::fmt::Debug for ScanOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScanOutcome::Success(state) => state.fmt(f),
            ScanOutcome::CannotAdvance => f.write_str("CannotAdvance"),
            ScanOutcome::NotOnTrack => f.write_str("NotOnTrack"),
            ScanOutcome::Deferral(_deferral) => f.write_str("Deferral {..}"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ScanState {
    pub(crate) block_coord: BlockCoordinate,
    pub(crate) is_reversed: bool,
    pub(crate) is_diverging: bool,
    pub(crate) allowable_speed: f32,
    pub(crate) odometer: f64,
    current_tile_id: TileId,
}

impl ScanState {
    pub(crate) fn spawn_at(
        block_coord: BlockCoordinate,
        az_direction: u8,
        game_map: &ServerGameMap,
        cart_config: &CartsGameBuilderExtension,
    ) -> Result<Option<Self>> {
        anyhow::ensure!(az_direction < 4);
        let block = game_map.get_block(block_coord)?;

        let tile_id = if block.equals_ignore_variant(cart_config.rail_block) {
            TileId::from_variant(block.variant(), false, false)
        } else if let Some((num, den, rotation)) = cart_config.parse_slope(block) {
            TileId::new_slope(num as u16, den as u16, rotation, false)
        } else {
            return Ok(None);
        };
        let mut corrected_rotation = (tile_id.rotation() + 4 - az_direction as u16) % 4;
        if tile_id.flip_x() && corrected_rotation & 1 == 1 {
            // If the corrected rotation is horizontal w.r.t. the tile, flip it over.
            corrected_rotation ^= 2;
        }
        let tile = match get_track_tile(tile_id) {
            Some(tile) => tile,
            None => return Ok(None),
        };
        assert!(corrected_rotation < 4);
        tracing::debug!(
            "Corrected rotation: {} -> {}",
            tile_id.rotation(),
            corrected_rotation
        );
        let (is_reversed, is_diverging) =
            if tile.straight_through_spawn_dirs & (1 << corrected_rotation) == 0 {
                (false, false)
            } else if tile.straight_through_spawn_dirs & (1 << (corrected_rotation + 4)) == 0 {
                (true, false)
            } else if tile.diverging_dirs_spawn_dirs & (1 << corrected_rotation) == 0 {
                (false, true)
            } else if tile.diverging_dirs_spawn_dirs & (1 << (corrected_rotation + 4)) == 0 {
                (true, true)
            } else {
                tracing::info!(
                    "Spawn is not allowed on this tile in this direction. {:?}",
                    tile
                );
                return Ok(None);
            };
        Ok(Some(ScanState {
            block_coord,
            is_reversed,
            is_diverging,
            allowable_speed: if is_diverging {
                tile.diverging_max_speed as f32
            } else {
                tile.max_speed as f32
            },
            current_tile_id: tile_id,
            odometer: 0.0,
        }))
    }

    pub(crate) fn advance<const CHATTY: bool>(
        &self,
        get_block: impl Fn(BlockCoordinate) -> DeferrableResult<Result<BlockId>, BlockCoordinate>,
        cart_config: &CartsGameBuilderExtension,
    ) -> Result<ScanOutcome> {
        if CHATTY {
            tracing::info!("{:?}", cart_config);
        }

        let block = match get_block(self.block_coord) {
            DeferrableResult::AvailableNow(block) => block?,
            DeferrableResult::Deferred(deferral) => return Ok(ScanOutcome::Deferral(deferral)),
        };

        let current_tile_id = match self.parse_block_id::<CHATTY>(block, cart_config) {
            Some(value) => value,
            None => return Ok(ScanOutcome::NotOnTrack),
        };
        if !current_tile_id.present() {}
        let tile = get_track_tile(current_tile_id);
        if CHATTY {
            tracing::info!("{:?}:\n {:?}", current_tile_id, tile);
        }
        let current_tile_def = match tile {
            Some(tile) => tile,
            None => return Ok(ScanOutcome::NotOnTrack),
        };

        let eligible_tiles = match (self.is_reversed, self.is_diverging) {
            (false, false) => current_tile_def.allowed_next_tiles.as_slice(),
            (false, true) => current_tile_def.allowed_next_diverging_tiles.as_slice(),
            (true, false) => current_tile_def.allowed_prev_tiles.as_slice(),
            (true, true) => current_tile_def.allowed_prev_diverging_tiles.as_slice(),
        };

        // The next place we would scan

        let next_delta = match (self.is_reversed, self.is_diverging) {
            (false, false) => current_tile_def.next_delta,
            (true, false) => current_tile_def.prev_delta,
            (false, true) => current_tile_def.next_diverging_delta,
            (true, true) => current_tile_def.prev_diverging_delta,
        };
        if !next_delta.present() {
            if CHATTY {
                tracing::info!("Next delta absent");
            }
            return Ok(ScanOutcome::CannotAdvance);
        }
        let next_coord = next_delta
            .eval_delta(
                self.block_coord,
                current_tile_id.flip_x(),
                current_tile_id.rotation(),
            )
            .context("block coordinate overflow")?;
        if CHATTY {
            tracing::info!("Next coord: {:?}", next_coord);
        }

        let next_block = match get_block(next_coord) {
            DeferrableResult::AvailableNow(block) => block.unwrap(),
            DeferrableResult::Deferred(deferral) => return Ok(ScanOutcome::Deferral(deferral)),
        };
        let next_tile_id = self
            .parse_block_id::<CHATTY>(next_block, cart_config)
            .unwrap_or_else(|| TileId::empty());

        if CHATTY {
            tracing::info!("Next tile: {:?}", next_tile_id);
        }

        // When we check the tile IDs allowed, its rotation would be this
        // (we subtract the current tile ID's rotation from the next tile ID's real rotation, since
        // if that next tile is in fact eligible, it'll be encoded in the eligible_tiles array with this adjusted rotation)
        let rotation_corrected_next_tile_id = next_tile_id
            .with_rotation_wrapped(next_tile_id.rotation() + 4 - current_tile_id.rotation())
            .xor_flip_x(current_tile_id.flip_x())
            .correct_rotation_for_x_flip(current_tile_id.flip_x());
        if CHATTY {
            tracing::info!(
                "Corrected rotation: {} -> {:?}",
                next_tile_id.rotation(),
                rotation_corrected_next_tile_id
            );
        }

        let mut matching_tile_id = TileId::empty();
        for &proposed_tile_id in eligible_tiles {
            if !proposed_tile_id.present() {
                continue;
            }
            // (0,0) is special: it means any of the straight track eligible connections
            if proposed_tile_id.x() == 0 && proposed_tile_id.y() == 0 {
                // we need another rotation correction here...
                // This time, we want to subtract the adjustment made by the straight track entry, since we need
                // to correct for another rotation offset, specifically the one between the proposed ID and
                let straight_track_rotation_corrected_tile_id = rotation_corrected_next_tile_id
                    .with_rotation_wrapped(
                        rotation_corrected_next_tile_id.rotation() + 4
                            - proposed_tile_id.rotation(),
                    )
                    .xor_flip_x(proposed_tile_id.flip_x());
                if CHATTY {
                    tracing::info!(
                        "Trying the straight track connections. Corrected {} -> {:?}",
                        rotation_corrected_next_tile_id.rotation(),
                        straight_track_rotation_corrected_tile_id
                    );
                }

                if let Some(tile) = get_track_tile(next_tile_id) {
                    let mask =
                        straight_track_rotation_corrected_tile_id.rotation_flip_bitset_mask();
                    if tile.straight_track_eligible_connections & mask != 0 {
                        matching_tile_id = next_tile_id.with_reverse_diverge(false, false);
                        break;
                    }
                    if tile.reverse_straight_track_eligible_connections & mask != 0 {
                        matching_tile_id = next_tile_id.with_reverse_diverge(true, false);
                        break;
                    }
                    if tile.diverging_straight_track_eligible_connections & mask != 0 {
                        matching_tile_id = next_tile_id.with_reverse_diverge(false, true);
                        break;
                    }
                    if tile.reverse_diverging_straight_track_eligible_connections & mask != 0 {
                        matching_tile_id = next_tile_id.with_reverse_diverge(true, true);
                        break;
                    }
                }
                if let Some(neighbor_below) = next_coord.try_delta(0, -1, 0) {
                    if CHATTY {
                        tracing::info!("Checking downward slope at {:?}", neighbor_below);
                    }
                    let block_below = match get_block(neighbor_below) {
                        DeferrableResult::AvailableNow(block) => block.unwrap(),
                        DeferrableResult::Deferred(deferral) => {
                            return Ok(ScanOutcome::Deferral(deferral));
                        }
                    };

                    if let Some((num, den, rotation)) = cart_config.parse_slope(block_below) {
                        let mut slope_rotation_corrected_next_rotation = (rotation + 8
                            - current_tile_id.rotation()
                            - proposed_tile_id.rotation())
                            & 3;
                        if current_tile_id.flip_x() {
                            slope_rotation_corrected_next_rotation ^= 2;
                        }
                        if proposed_tile_id.flip_x() {
                            slope_rotation_corrected_next_rotation ^= 2;
                        }
                        if CHATTY {
                            tracing::info!("Found slope: {} / {} @ {}", num, den, rotation);
                            tracing::info!(
                                "rotation_corrected_next_tile_id rotation {}",
                                slope_rotation_corrected_next_rotation
                            );
                            tracing::info!(
                                "straight_track_rotation_corrected_tile_id {}",
                                straight_track_rotation_corrected_tile_id.rotation()
                            );
                            tracing::info!("proposed rotation {}", proposed_tile_id.rotation());
                        }
                        // To start a downward slope, we need the numerator and denominator to match
                        if num == den {
                            // we are going down the slope, so we need rotation 2 in the tile space
                            // (i.e. after correcting for placement rotation)
                            if slope_rotation_corrected_next_rotation == 2 {
                                if CHATTY {
                                    tracing::info!(
                                        "Corrected rotation: {} -> {} matches",
                                        rotation,
                                        slope_rotation_corrected_next_rotation
                                    );
                                }
                                return Ok(ScanOutcome::Success(self.build_next_state(
                                    neighbor_below,
                                    TileId::new_slope(num as u16, den as u16, rotation ^ 2, false),
                                    true,
                                    false,
                                )));
                            } else {
                                if CHATTY {
                                    tracing::info!(
                                        "No match found: {:?} -> {:?}",
                                        rotation,
                                        slope_rotation_corrected_next_rotation
                                    );
                                }
                            }
                        }
                    }
                }
            } else if rotation_corrected_next_tile_id.same_variant(proposed_tile_id) {
                matching_tile_id = proposed_tile_id;
            }
        }
        if !matching_tile_id.present() {
            if CHATTY {
                tracing::info!("No match found");
            }
            return Ok(ScanOutcome::CannotAdvance);
        }

        // Now check secondary and tertiary tiles
        let next_tile = match get_track_tile(next_tile_id) {
            Some(tile) => tile,
            None => {
                if CHATTY {
                    log::error!("Match found, but no tile.");
                }
                return Ok(ScanOutcome::CannotAdvance);
            }
        };
        if next_tile.secondary_coord.present() {
            let secondary_block_coord = match next_tile.secondary_coord.eval_delta(
                next_coord,
                next_tile_id.flip_x(),
                next_tile_id.rotation(),
            ) {
                Some(secondary_block_coord) => secondary_block_coord,
                None => {
                    if CHATTY {
                        tracing::info!("Secondary tile coord overflows");
                    }
                    return Ok(ScanOutcome::CannotAdvance);
                }
            };
            let secondary_block = match get_block(secondary_block_coord) {
                DeferrableResult::AvailableNow(block) => block.unwrap(),
                DeferrableResult::Deferred(deferral) => return Ok(ScanOutcome::Deferral(deferral)),
            };
            if !secondary_block.equals_ignore_variant(cart_config.rail_block) {
                if CHATTY {
                    tracing::info!(
                        "Secondary block: {:?} is not the base track",
                        secondary_block
                    );
                }
                return Ok(ScanOutcome::CannotAdvance);
            }

            let secondary_tile = TileId::from_variant(
                secondary_block.variant(),
                // Don't care about reverse or diverging
                false,
                false,
            );

            let rotation_corrected_secondary_tile_id = secondary_tile
                .with_rotation_wrapped(secondary_tile.rotation() + 4 - next_tile_id.rotation())
                .xor_flip_x(next_tile_id.flip_x())
                .correct_rotation_for_x_flip(next_tile_id.flip_x());

            let mut secondary_ok = false;
            for proposed_secondary_tile_id in next_tile.secondary_tile {
                if proposed_secondary_tile_id.diverging() && !self.is_diverging {
                    // This tile only needs to match if we're diverging, but we're not
                    secondary_ok = true;
                    break;
                }
                if proposed_secondary_tile_id.same_variant(rotation_corrected_secondary_tile_id) {
                    secondary_ok = true;
                    break;
                }
            }

            if !secondary_ok {
                if CHATTY {
                    tracing::info!(
                        "Secondary tile doesn't match: {:?}",
                        rotation_corrected_secondary_tile_id
                    );
                }
                return Ok(ScanOutcome::CannotAdvance);
            }
        }
        if next_tile.tertiary_coord.present() {
            let tertiary_block_coord = match next_tile.tertiary_coord.eval_delta(
                next_coord,
                next_tile_id.flip_x(),
                next_tile_id.rotation(),
            ) {
                Some(tertiary_block_coord) => tertiary_block_coord,
                None => {
                    if CHATTY {
                        tracing::info!("Tertiary tile coord overflows");
                    }
                    return Ok(ScanOutcome::CannotAdvance);
                }
            };
            let tertiary_block = match get_block(tertiary_block_coord) {
                DeferrableResult::AvailableNow(block) => block.unwrap(),
                DeferrableResult::Deferred(deferral) => return Ok(ScanOutcome::Deferral(deferral)),
            };
            if !tertiary_block.equals_ignore_variant(cart_config.rail_block) {
                if CHATTY {
                    tracing::info!("Tertiary block: {:?} is not the base track", tertiary_block);
                }
            }
            let tertiary_tile = TileId::from_variant(
                tertiary_block.variant(),
                // Don't care about reverse or diverging
                false,
                false,
            );

            let rotation_corrected_tertiary_tile_id = tertiary_tile
                .with_rotation_wrapped(tertiary_tile.rotation() + 4 - next_tile_id.rotation())
                .xor_flip_x(next_tile_id.flip_x())
                .correct_rotation_for_x_flip(next_tile_id.flip_x());

            if (!next_tile.tertiary_tile.diverging() || self.is_diverging)
                && !next_tile
                    .tertiary_tile
                    .same_variant(rotation_corrected_tertiary_tile_id)
            {
                if CHATTY {
                    tracing::info!(
                        "Tertiary tile doesn't match: {:?}",
                        rotation_corrected_tertiary_tile_id
                    );
                }
                return Ok(ScanOutcome::CannotAdvance);
            }
        }

        if CHATTY {
            tracing::info!("Found matching tile: {:?}", matching_tile_id,);
        }
        assert!(matching_tile_id.present());
        assert!(next_tile_id.present());
        Ok(ScanOutcome::Success(self.build_next_state(
            next_coord,
            next_tile_id,
            matching_tile_id.reverse(),
            matching_tile_id.diverging(),
        )))
    }

    #[inline]
    fn build_next_state(
        &self,
        next_coord: BlockCoordinate,
        next_tile_id: TileId,
        reverse: bool,
        diverging: bool,
    ) -> ScanState {
        let mut next_state = self.clone();
        let next_tile = get_track_tile(next_tile_id).unwrap();

        next_state.block_coord = next_coord;
        next_state.is_reversed = reverse;
        next_state.is_diverging = diverging;
        next_state.current_tile_id = next_tile_id;
        next_state.allowable_speed = if diverging {
            next_tile.diverging_max_speed as f32
        } else {
            next_tile.max_speed as f32
        };
        next_state.odometer =
            self.odometer + (self.vec_coord() - next_state.vec_coord()).magnitude();
        next_state
    }

    fn parse_block_id<const CHATTY: bool>(
        &self,
        next_block: BlockId,
        cart_config: &CartsGameBuilderExtension,
    ) -> Option<TileId> {
        if next_block.equals_ignore_variant(cart_config.rail_block) {
            let next_variant = next_block.variant();
            Some(TileId::from_variant(
                next_variant,
                self.is_reversed,
                self.is_diverging,
            ))
        } else if let Some((num, den, rotation)) = cart_config.parse_slope(next_block) {
            Some(TileId::new_slope(
                num as u16,
                den as u16,
                rotation,
                self.is_reversed,
            ))
        } else {
            if CHATTY {
                tracing::info!("block: {:?} is not the base track", next_block);
            }
            None
        }
    }

    pub(crate) fn signal_rotation_ok(&self, rotation: u16) -> bool {
        let rotation = rotation & 3;
        let tile = get_track_tile(self.current_tile_id);
        let tile = match tile {
            Some(tile) => tile,
            None => {
                return false;
            }
        };
        let corrected_rotation = self
            .current_tile_id
            .with_rotation_wrapped(self.current_tile_id.rotation() + 4 - rotation)
            .xor_flip_x(self.current_tile_id.flip_x())
            .correct_rotation_for_x_flip(self.current_tile_id.flip_x())
            .rotation();
        let bitmask = match (self.is_diverging, self.is_reversed) {
            (false, false) => tile.straight_through_spawn_dirs & 0xf,
            (false, true) => tile.straight_through_spawn_dirs >> 4,
            (true, false) => tile.diverging_dirs_spawn_dirs & 0xf,
            (true, true) => tile.diverging_dirs_spawn_dirs >> 4,
        };
        bitmask & (1 << corrected_rotation) != 0
    }

    pub(crate) fn signal_reversed_rotation_ok(&self, rotation: u16) -> bool {
        let rotation = rotation & 3;
        let tile = get_track_tile(self.current_tile_id);
        let tile = match tile {
            Some(tile) => tile,
            None => {
                return false;
            }
        };
        let corrected_rotation = self
            .current_tile_id
            .with_rotation_wrapped(self.current_tile_id.rotation() + 4 - rotation)
            .xor_flip_x(self.current_tile_id.flip_x())
            .correct_rotation_for_x_flip(self.current_tile_id.flip_x())
            .rotation();
        let bitmask = match (self.is_diverging, self.is_reversed) {
            (false, true) => tile.straight_through_spawn_dirs & 0xf,
            (false, false) => tile.straight_through_spawn_dirs >> 4,
            (true, true) => tile.diverging_dirs_spawn_dirs & 0xf,
            (true, false) => tile.diverging_dirs_spawn_dirs >> 4,
        };
        bitmask & (1 << corrected_rotation) != 0
    }

    pub(crate) fn get_switch_length(&self) -> Option<NonZeroU8> {
        let tile = get_track_tile(self.current_tile_id);
        let tile = match tile {
            Some(tile) => tile,
            None => {
                return None;
            }
        };
        NonZeroU8::new(tile.switch_length)
    }

    // Inline these because they share a lot of common subexpressions.
    #[inline]
    pub(crate) fn can_diverge_left(&self) -> bool {
        // The tile has to be switch eligible, it has to be a forward move facing the points, and the tile
        // has to be flipped along X (since unflipped tiles turn out to the right)
        self.get_switch_length().is_some() && !self.is_reversed && self.current_tile_id.flip_x()
    }
    #[inline]
    pub(crate) fn can_diverge_right(&self) -> bool {
        // The tile has to be switch eligible, it has to be a forward move facing the points, and the tile
        // cannot be flipped along X (since unflipped tiles turn out to the left)
        self.get_switch_length().is_some() && !self.is_reversed && !self.current_tile_id.flip_x()
    }

    #[inline]
    pub(crate) fn can_converge(&self) -> bool {
        // The tile has to be switch eligible, it has to be a backward move approaching against the points.
        self.get_switch_length().is_some() && self.is_reversed
    }

    pub(crate) fn vec_coord(&self) -> Vector3<f64> {
        let base_coord = b2vec(self.block_coord);
        if !self.current_tile_id.present() {
            return base_coord;
        }
        let (offset_x, offset_y, offset_z) = if self.current_tile_id.is_slope_encoding() {
            let num = self.current_tile_id.x() as u32;
            let denom = self.current_tile_id.y() as u32;
            (0, (128 * (2 * num - 1)) / (2 * denom), 0)
        } else {
            let tile = match get_track_tile(self.current_tile_id) {
                Some(tile) => tile,
                None => {
                    return base_coord;
                }
            };

            let offset_x = if self.is_diverging {
                tile.diverging_physical_x_offset
            } else {
                tile.physical_x_offset
            };
            let offset_z = if self.is_diverging {
                tile.diverging_physical_z_offset
            } else {
                tile.physical_z_offset
            };
            let (offset_x, offset_z) = eval_rotation(
                offset_x,
                offset_z,
                self.current_tile_id.flip_x(),
                self.current_tile_id.rotation(),
            );
            (offset_x, 0, offset_z)
        };
        base_coord
            + vec3(
                offset_x as f64 / 128.0,
                offset_y as f64 / 128.0,
                offset_z as f64 / 128.0,
            )
    }
}

fn get_track_tile(id: TileId) -> Option<TrackTile> {
    if !id.present() {
        return None;
    }
    if id.is_slope_encoding() {
        match id.y() {
            1 => {
                if id.x() == 1 {
                    return SLOPE_TRACKS[0];
                } else {
                    return None;
                }
            }
            8 => {
                if id.x() == 0 || id.x() > 8 {
                    return None;
                } else {
                    return SLOPE_TRACKS[id.x() as usize];
                }
            }
            _ => return None,
        }
    }
    TRACK_TILES[id.x() as usize][id.y() as usize]
}

const fn allowable_speed_for_slope(den: u8) -> f32 {
    match den {
        1 => 6.0,
        8 => 60.0,
        _ => unreachable!(),
    }
}

// test only
fn handle_popup_response(
    response: &PopupResponse,
    coord: BlockCoordinate,
    cart_config: &CartsGameBuilderExtension,
) -> Result<()> {
    match &response.user_action {
        PopupAction::ButtonClicked(x) => match x.as_str() {
            "apply" => {
                let tile_x = response
                    .textfield_values
                    .get("tile_x")
                    .context("missing tile_x")?
                    .parse::<u16>()?;
                let tile_y = response
                    .textfield_values
                    .get("tile_y")
                    .context("missing tile_y")?
                    .parse::<u16>()?;
                let rotation = response
                    .textfield_values
                    .get("rotation")
                    .context("missing rotation")?
                    .parse::<u16>()?;
                let flip_x = *response
                    .checkbox_values
                    .get("flip_x")
                    .context("missing flip_x")?;
                let tile_id = TileId::try_new(tile_x, tile_y, rotation, flip_x, false, false)
                    .map_err(|x| anyhow::anyhow!(x))?;
                response
                    .ctx
                    .game_map()
                    .mutate_block_atomically(coord, |b, _ext| {
                        *b = b.with_variant(tile_id.block_variant())?;
                        Ok(())
                    })?;
            }
            "scan" => {
                let state = ScanState {
                    block_coord: coord,
                    is_reversed: *response
                        .checkbox_values
                        .get("reverse")
                        .context("missing reverse")?,
                    is_diverging: *response
                        .checkbox_values
                        .get("diverging")
                        .context("missing diverging")?,
                    allowable_speed: 90.0,
                    // dummy
                    current_tile_id: TileId::empty(),
                    odometer: 0.0,
                };
                dbg!(state.advance::<true>(
                    |coord| response.ctx.game_map().get_block(coord).into(),
                    cart_config,
                ))?;
            }
            "multiscan" => {
                let mut state = ScanState {
                    block_coord: coord,
                    is_reversed: *response
                        .checkbox_values
                        .get("reverse")
                        .context("missing reverse")?,
                    is_diverging: *response
                        .checkbox_values
                        .get("diverging")
                        .context("missing diverging")?,
                    allowable_speed: 90.0,
                    // dummy
                    current_tile_id: TileId::empty(),
                    odometer: 0.0,
                };
                response.ctx.run_deferred(move |ctx| {
                    for _ in 0..20 {
                        let prev = state.vec_coord() + vec3(0.0, 1.0, 0.0);

                        match state.advance::<true>(
                            |coord| ctx.game_map().get_block(coord).into(),
                            ctx.extension().as_ref().unwrap(),
                        )? {
                            ScanOutcome::Success(s) => {
                                state = s;
                            }
                            _ => {
                                break;
                            }
                        }

                        let current = state.vec_coord() + vec3(0.0, 1.0, 0.0);
                        std::thread::sleep(std::time::Duration::from_millis(125));
                        if let EventInitiator::WeakPlayerRef(p) = ctx.initiator() {
                            p.try_to_run(|p| p.set_position_blocking(0.5 * (prev + current)));
                        }
                        std::thread::sleep(std::time::Duration::from_millis(125));
                        if let EventInitiator::WeakPlayerRef(p) = ctx.initiator() {
                            p.try_to_run(|p| p.set_position_blocking(current));
                        }
                    }
                    Ok(())
                });
            }
            _ => {
                return Err(anyhow::anyhow!("unknown button clicked: {}", x));
            }
        },
        _ => {}
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct TemplateEntry {
    pub(crate) tile_id: TileId,
    pub(crate) offset_x: i32,
    pub(crate) offset_y: i32,
    pub(crate) offset_z: i32,
    pub(crate) tracks_consumed: u8,
}

#[derive(Clone, Debug)]
pub(crate) struct Template {
    pub(crate) category: String,
    pub(crate) sort_subkey: u16,
    pub(crate) name: String,
    pub(crate) id: String,
    pub(crate) entries: Box<[TemplateEntry]>,
    pub(crate) bifurcate: bool,
}

pub(crate) fn build_block(
    config: &CartsGameBuilderExtension,
    tile_id: TileId,
    extra_rotation: u16,
    flip_x: bool,
) -> Option<BlockId> {
    let adjusted_rotation = ((tile_id.rotation()) + (extra_rotation & 3)) & 3;
    let mut adjusted_variant = (tile_id.block_variant() & !3) | adjusted_rotation;

    if tile_id.is_slope_encoding() {
        match tile_id.y() {
            1 => {
                if tile_id.x() == 1 {
                    Some(config.rail_slope_1.with_variant(adjusted_variant).unwrap())
                } else {
                    None
                }
            }
            8 => match tile_id.x() {
                1..=8 => Some(
                    config.rail_slopes_8[(tile_id.x() - 1) as usize]
                        .with_variant(adjusted_variant)
                        .unwrap(),
                ),
                _ => None,
            },
            _ => None,
        }
    } else {
        if flip_x {
            adjusted_variant |= c::FLIP_X_BIT;
        }
        Some(config.rail_block.with_variant(adjusted_variant).unwrap())
    }
}
