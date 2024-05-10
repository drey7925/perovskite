// For now, hide unused warnings since they're distracting
#![allow(dead_code)]

use anyhow::{Context, Result};
use cgmath::vec3;
use perovskite_core::{
    block_id::BlockId, chat::ChatMessage, coordinates::BlockCoordinate,
    protocol::render::DynamicCrop,
};
use perovskite_server::game_state::{
    client_ui::{PopupAction, PopupResponse, UiElementContainer},
    entities::{DeferrableResult, Deferral},
    event::EventInitiator,
    game_map::ServerGameMap,
};

use crate::{
    blocks::{AaBoxProperties, AxisAlignedBoxesAppearanceBuilder, BlockBuilder},
    game_builder::{StaticBlockName, StaticTextureName},
    include_texture_bytes,
};

use super::b2vec;

#[rustfmt::skip]
mod c {
    // We only have 12 bits for the variant stored in the block...
    pub(crate) const ROTATION_BITS: u16   = 0b0000_0000_0000_0011;
    pub(crate) const X_SELECTOR: u16      = 0b0000_0000_0011_1100;
    pub(crate) const X_SELECTOR_DIV: u16  = 0b0000_0000_0000_0100;
    pub(crate) const Y_SELECTOR: u16      = 0b0000_0011_1100_0000;
    pub(crate) const Y_SELECTOR_DIV: u16  = 0b0000_0000_0100_0000;
    pub(crate) const FLIP_X_BIT: u16      = 0b0000_0100_0000_0000;
    // But we can still use the upper four bits for our own tracking, as long as
    // they don't need to be encoded on the game map

    /// If set on a secondary or tertiary, it means that the secondary/tertiary only
    /// applies when the diverging route is active
    ///
    /// If set on a prev/next, it means that we shoud enter that track 
    pub(crate) const DIVERGING_ROUTE: u16 = 0b0001_0000_0000_0000;
    pub(crate) const REVERSE_SCAN: u16    = 0b0010_0000_0000_0000;
    pub(crate) const ENTRY_PRESENT: u16   = 0b1000_0000_0000_0000;
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct TileId(u16);
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
    const fn x(&self) -> u16 {
        (self.0 & c::X_SELECTOR) / c::X_SELECTOR_DIV
    }
    const fn y(&self) -> u16 {
        (self.0 & c::Y_SELECTOR) / c::Y_SELECTOR_DIV
    }
    const fn is_same_x_y(&self, other: TileId) -> bool {
        (self.0 & (c::X_SELECTOR | c::Y_SELECTOR)) == (other.0 & (c::X_SELECTOR | c::Y_SELECTOR))
    }
    const fn same_variant(&self, other: TileId) -> bool {
        (self.0 & (c::X_SELECTOR | c::Y_SELECTOR | c::ROTATION_BITS | c::FLIP_X_BIT))
            == (other.0 & (c::X_SELECTOR | c::Y_SELECTOR | c::ROTATION_BITS | c::FLIP_X_BIT))
    }
    /// The rotation, clockwise, in units of 90 degrees. Always 0, 1, 2, or 3.
    const fn rotation(&self) -> u16 {
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
    const fn present(&self) -> bool {
        (self.0 & Self::PRESENT) != 0
    }
    const fn x(&self) -> i8 {
        ((self.0 & 0b1100) >> 2) as i8 - 1
    }
    const fn z(&self) -> i8 {
        (self.0 & 0b11) as i8 - 1
    }
    const fn empty() -> EncodedDelta {
        EncodedDelta(0)
    }
    const fn eval_rotation(&self, flip_x: bool, rotation: u16) -> (i8, i8) {
        let x = if flip_x { -self.x() } else { self.x() };
        let z = self.z();
        match rotation & 3 {
            0 => (x, z),
            1 => (z, -x),
            2 => (-x, -z),
            3 => (-z, x),
            _ => unreachable!(),
        }
    }
    fn eval_delta(
        &self,
        base: BlockCoordinate,
        flip_x: bool,
        rotation: u16,
    ) -> Option<BlockCoordinate> {
        let (x, z) = self.eval_rotation(flip_x, rotation);
        base.try_delta(x as i32, 0, z as i32)
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

fn eval_rotation(x: i8, z: i8, flip_x: bool, rotation: u16) -> (i8, i8) {
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
    /// If true, this track tile can start/end a diverging move.
    ///
    /// When this is true, the following additional conditions are imposed:
    ///     * In the forward and forward diverging directions, a single incoming path splits into the normal and diverging paths
    ///     * In the reverse and reverse diverging directions, the incoming diverging and normal paths converge onto a normal path
    /// If these preconditions are violated, the behavior of the interlocking scanner may be abnormal.
    ///
    /// Note that this may become a small enum if we ever add slip switches; in that case, the required preconditions above will apply
    /// for the non-slip-switch case.
    switch_eligible: bool,
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
            switch_eligible: false,
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

// TODO this is going to grow. Use a bitset over 16 * 11 * 4 * 8 bits = 704 bytes
const STRAIGHT_TRACK_ELIGIBLE_CONNECTIONS: &[TileId] = &[
    // We can enter the 90-degree elbow at (8, 8) in its forward scan direction.
    TileId::new(8, 8, 0, false, false, false),
    // including if X is flipped
    TileId::new(8, 8, 0, true, false, false),
    // We can enter the 90-degree elbow at (8, 8) in its reverse scan direction.
    // For this to work, the elbow must be turned 90 degrees so we're seeing its end-of-path
    // side. We set reverse = true because we want the scan to continue in the reverse direction
    //
    // but if X is flipped on the elbow, we need the opposite rotation
    TileId::new(8, 8, 1, false, true, false),
    TileId::new(8, 8, 3, true, true, false),
];

fn build_track_tiles() -> [[Option<TrackTile>; 16]; 11] {
    let mut track_tiles = [[None; 16]; 11];

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
        straight_track_eligible_connections: (
            // We can enter a straight track with the same rotation, regardless of flip_x
            TileId::new(0, 0, 0, false, false, false).rotation_flip_bitset_mask()
                | TileId::new(0, 0, 0, true, false, false).rotation_flip_bitset_mask()
        ),
        reverse_straight_track_eligible_connections: (
            // We can enter a straight track from the other side, and do a reverse move through it
            TileId::new(0, 0, 2, false, true, false).rotation_flip_bitset_mask()
                | TileId::new(0, 0, 2, true, true, false).rotation_flip_bitset_mask()
        ),
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
        straight_track_eligible_connections: (
            // We can enter the 90-degree elbow at (8, 8) in its forward scan direction.
            TileId::new(8, 8, 0, false, false, false).rotation_flip_bitset_mask()
            // including if X is flipped
            | TileId::new(8, 8, 0, true, false, false).rotation_flip_bitset_mask()
        ),
        reverse_straight_track_eligible_connections: (
            // We can enter the 90-degree elbow at (8, 8) in its reverse scan direction.
            // For this to work, the elbow must be turned 90 degrees so we're seeing its end-of-path
            // side. We set reverse = true because we want the scan to continue in the reverse direction
            //
            // but if X is flipped on the elbow, we need the opposite rotation
            TileId::new(8, 8, 1, false, true, false).rotation_flip_bitset_mask()
                | TileId::new(8, 8, 3, true, true, false).rotation_flip_bitset_mask()
        ),
        ..TrackTile::default()
    });

    // CONTINUE HERE
    // [0][1] and [1][1] start the 8-block switch (folded in half to make a 4 block segment in the tile space)
    build_folded_switch(&mut track_tiles, 4, 0, 1, 6, 0, 1);

    track_tiles
}

fn build_folded_switch(
    track_tiles: &mut [[Option<TrackTile>; 16]; 11],
    switch_half_len: u16,
    switch_xmin: u16,
    switch_ymin: u16,
    diag_xmin: u16,
    diag_ymin: u16,
    skip_secondary_tiles: u16,
) {
    for y in 0..switch_half_len {
        // The tiles on column 0 only carry straight-through traffic. Diverging traffic is always on column 1 tiles,
        // with the column 0 tile being merely a secondary tile.
        // We can just copy the 0,0 straight track into here
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
                        switch_ymin + (y + 1) as u16,
                        0,
                        false,
                        false,
                        true,
                    ),
                    TileId::new(
                        diag_xmin + 1,
                        diag_ymin + (y + 1) as u16,
                        0,
                        false,
                        false,
                        false,
                    ),
                ]
            },
            allowed_prev_tiles: [TileId::new(0, 0, 2, false, false, false), TileId::empty()],
            allowed_prev_diverging_tiles: if y == 0 {
                // If 0, diverging just merges back onto the main track
                [TileId::new(0, 0, 2, false, false, false), TileId::empty()]
            } else {
                [
                    TileId::new(
                        switch_xmin + 1,
                        switch_ymin + (y - 1) as u16,
                        0,
                        false,
                        true,
                        true,
                    ),
                    TileId::new(
                        diag_xmin + 1,
                        diag_ymin + (y - 1) as u16,
                        0,
                        false,
                        true,
                        true,
                    ),
                ]
            },
            straight_track_eligible_connections: (
                // We can enter a straight track with the same rotation, regardless of flip_x
                // Note that X and Y don't matter here; we're just getting the bitset mask.
                TileId::new(0, 0, 0, false, false, false).rotation_flip_bitset_mask()
                    | TileId::new(0, 0, 0, true, false, false).rotation_flip_bitset_mask()
            ),
            reverse_straight_track_eligible_connections: (
                // We can enter a straight track from the other side, and do a reverse move through it
                TileId::new(0, 0, 2, false, true, false).rotation_flip_bitset_mask()
                    | TileId::new(0, 0, 2, true, true, false).rotation_flip_bitset_mask()
            ),

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
            switch_eligible: (y == 0),

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
                        switch_ymin + (y + 1) as u16,
                        0,
                        false,
                        false,
                        true,
                    ),
                    TileId::new(
                        diag_xmin + 1,
                        diag_ymin + (y + 1) as u16,
                        0,
                        false,
                        false,
                        false,
                    ),
                    TileId::empty(),
                ]
            },
            allowed_prev_tiles: if y == 0 {
                // If 0, diverging just merges back onto the main track
                [TileId::new(0, 0, 2, false, false, false), TileId::empty()]
            } else {
                [
                    TileId::new(
                        switch_xmin + 1,
                        switch_ymin + (y - 1) as u16,
                        0,
                        false,
                        true,
                        true,
                    ),
                    TileId::new(
                        diag_xmin + 1,
                        diag_ymin + (y - 1) as u16,
                        0,
                        false,
                        true,
                        false,
                    ),
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
            ..TrackTile::default()
        });
    }
}

lazy_static::lazy_static! {
    static ref TRACK_TILES: [[Option<TrackTile>; 16]; 11] = build_track_tiles();
}

pub(crate) fn register_tracks(
    game_builder: &mut crate::game_builder::GameBuilder,
) -> Result<BlockId> {
    const SIGNAL_SIDE_TOP_TEX: StaticTextureName = StaticTextureName("carts:signal_side_top");
    const RAIL_TILES_TEX: StaticTextureName = StaticTextureName("carts:rail_tile");
    const TRANSPARENT_PIXEL: StaticTextureName = StaticTextureName("carts:transparent_pixel");
    include_texture_bytes!(game_builder, RAIL_TILES_TEX, "textures/rail_atlas.png")?;
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
            .set_axis_aligned_boxes_appearance(AxisAlignedBoxesAppearanceBuilder::new().add_box(
                rail_tile_box,
                (-0.5, 0.5),
                (-0.5, -0.4),
                (-0.5, 0.5),
            ))
            .set_allow_light_propagation(true)
            .set_display_name("Railway track")
            .add_modifier(Box::new(|bt| {
                bt.interact_key_handler = Some(Box::new(|ctx, coord| {
                    let block = ctx.game_map().get_block(coord)?;
                    let tile_id = TileId::from_variant(block.variant(), false, false);
                    ctx.initiator().send_chat_message(ChatMessage::new(
                        "[INFO]",
                        format!("{:?}", tile_id),
                    ))?;

                    Ok(Some(ctx.new_popup()
                        .title("Mr. Yellow")
                        .label("Update tile:")
                        .text_field("tile_x", "Tile X", tile_id.x().to_string(), true, false)
                        .text_field("tile_y", "Tile Y", tile_id.y().to_string(), true, false)
                        .text_field("rotation", "Rotation", tile_id.rotation().to_string(), true, false)
                        .checkbox("flip_x", "Flip X", tile_id.flip_x(), true)
                        .button("apply", "Apply", true)
                        .label("Scan tile:")
                        .checkbox("reverse", "Reverse", false, true)
                        .checkbox("diverging", "Diverging", false, true)
                        .button("scan", "Scan", true)
                        .button("multiscan", "Multi-Scan", true)
                        .set_button_callback(Box::new(move |response: PopupResponse<'_>| {
                            match handle_popup_response(&response, coord) {
                                Ok(_) => {},
                                Err(e) => {
                                    response.ctx.initiator().send_chat_message(ChatMessage::new("[ERROR]", "Failed to parse popup response: ".to_string() + &e.to_string())).unwrap();
                                },
                            }
                        }))))
                }));
                let ri = bt.client_info.render_info.as_mut().unwrap();
                match ri {
                    perovskite_core::protocol::blocks::block_type_def::RenderInfo::AxisAlignedBoxes(aabb) => {
                        aabb.boxes.iter_mut().for_each(|b| {
                            b.tex_top.as_mut().unwrap().crop.as_mut().unwrap().dynamic = Some(
                                DynamicCrop {
                                    x_selector_bits: 0b0000_0000_0011_1100,
                                    y_selector_bits: 0b0000_0011_1100_0000,
                                    x_cells: 16,
                                    y_cells: 11,
                                    flip_x_bit: 0b0000_0100_0000_0000,
                                    flip_y_bit: 0,
                                    extra_flip_x: false,
                                    extra_flip_y: false,
                                }
                            );
                            b.tex_bottom.as_mut().unwrap().crop.as_mut().unwrap().dynamic = Some(
                                DynamicCrop {
                                    x_selector_bits: 0b0000_0000_0011_1100,
                                    y_selector_bits: 0b0000_0011_1100_0000,
                                    x_cells: 16,
                                    y_cells: 11,
                                    flip_x_bit: 0b0000_0100_0000_0000,
                                    flip_y_bit: 0,
                                    extra_flip_x: true,
                                    extra_flip_y: false,
                                }
                            )
                        })
                    },
                    _ => unreachable!()
                }
            })),
    )?;

    Ok(rail_tile.id)
}

pub(crate) enum ScanOutcome {
    /// We advanced, here's the new state
    Success(ScanState),
    /// We cannot advance
    CannotAdvance,
    /// We're not even on a track
    NotOnTrack,
    /// We got a deferral while reading the map
    Deferral(Deferral<Result<BlockId>, BlockCoordinate>),
}

#[derive(Debug, Clone)]
pub(crate) struct ScanState {
    pub(crate) block_coord: BlockCoordinate,
    pub(crate) vec_coord: cgmath::Vector3<f64>,
    pub(crate) is_reversed: bool,
    pub(crate) is_diverging: bool,
    pub(crate) base_track_block: BlockId,
    pub(crate) allowable_speed: f32,
    current_tile_id: TileId,
}
impl ScanState {
    pub(crate) fn spawn_at(
        block_coord: BlockCoordinate,
        az_direction: u8,
        base_track_block: BlockId,
        game_map: &ServerGameMap,
    ) -> Result<Option<Self>> {
        anyhow::ensure!(az_direction < 4);
        let block = game_map.get_block(block_coord)?;
        if !block.equals_ignore_variant(base_track_block) {
            // TODO detect slope tracks and other special tracks
            return Ok(None);
        }
        let tile_id = TileId::from_variant(block.variant(), false, false);
        let mut corrected_rotation = (tile_id.rotation() + 4 - az_direction as u16) % 4;
        if tile_id.flip_x() && corrected_rotation & 1 == 1 {
            // If the corrected rotation is horizontal w.r.t. the tile, flip it over.
            corrected_rotation ^= 2;
        }
        let tile = match TRACK_TILES[tile_id.x() as usize][tile_id.y() as usize] {
            Some(tile) => tile,
            None => return Ok(None),
        };
        assert!(corrected_rotation < 4);
        tracing::info!(
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
        let offset_x = if is_diverging {
            tile.diverging_physical_x_offset
        } else {
            tile.physical_x_offset
        };
        let offset_z = if is_diverging {
            tile.diverging_physical_z_offset
        } else {
            tile.physical_z_offset
        };
        let vec_coord =
            b2vec(block_coord) + vec3(offset_x as f64 / 128.0, 0.0, offset_z as f64 / 128.0);
        Ok(Some(ScanState {
            block_coord,
            vec_coord,
            is_reversed,
            is_diverging,
            base_track_block,
            allowable_speed: if is_diverging {
                tile.diverging_max_speed as f32
            } else {
                tile.max_speed as f32
            },
            current_tile_id: tile_id,
        }))
    }

    // Test only, prototype version that logs lots of details about each calculation
    pub(crate) fn advance<const CHATTY: bool>(
        &self,
        get_block: impl Fn(BlockCoordinate) -> DeferrableResult<Result<BlockId>, BlockCoordinate>,
    ) -> Result<ScanOutcome> {
        let block = match get_block(self.block_coord) {
            DeferrableResult::AvailableNow(block) => block,
            DeferrableResult::Deferred(deferral) => return Ok(ScanOutcome::Deferral(deferral)),
        };

        let variant = block.unwrap().variant();
        let current_tile_id = TileId::from_variant(variant, self.is_reversed, self.is_diverging);
        if !current_tile_id.present() {}
        let tile = TRACK_TILES[current_tile_id.x() as usize][current_tile_id.y() as usize];
        if CHATTY {
            tracing::info!("{:?}:\n {:?}", current_tile_id, tile);
        }
        let current_tile_def = match tile {
            Some(tile) => tile,
            None => return Ok(ScanOutcome::NotOnTrack),
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
        if !next_block.equals_ignore_variant(self.base_track_block) {
            if CHATTY {
                tracing::info!("Next block: {:?} is not the base track", next_block);
            }
            return Ok(ScanOutcome::CannotAdvance);
        }
        let next_variant = next_block.variant();

        // This is the tile we see at the next coordinate. Does it match?
        let next_tile_id = TileId::from_variant(next_variant, self.is_reversed, self.is_diverging);
        if CHATTY {
            tracing::info!("Next tile: {:?}", next_tile_id);
        }

        // TODO: Does this generate rancid assembly because of the mismatched lengths?
        let eligible_tiles = match (self.is_reversed, self.is_diverging) {
            (false, false) => current_tile_def.allowed_next_tiles.as_slice(),
            (false, true) => current_tile_def.allowed_next_diverging_tiles.as_slice(),
            (true, false) => current_tile_def.allowed_prev_tiles.as_slice(),
            (true, true) => current_tile_def.allowed_prev_diverging_tiles.as_slice(),
        };

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

                if let Some(tile) =
                    TRACK_TILES[next_tile_id.x() as usize][next_tile_id.y() as usize]
                {
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
        let next_tile = match TRACK_TILES[next_tile_id.x() as usize][next_tile_id.y() as usize] {
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
            if !secondary_block.equals_ignore_variant(self.base_track_block) {
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
            if !tertiary_block.equals_ignore_variant(self.base_track_block) {
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

        let mut next_state = self.clone();

        next_state.block_coord = next_coord;
        next_state.is_reversed = matching_tile_id.reverse();
        next_state.is_diverging = matching_tile_id.diverging();
        next_state.current_tile_id = next_tile_id;
        let base_coord = b2vec(next_coord);

        let offset_x = if next_state.is_diverging {
            next_tile.diverging_physical_x_offset
        } else {
            next_tile.physical_x_offset
        };
        let offset_z = if next_state.is_diverging {
            next_tile.diverging_physical_z_offset
        } else {
            next_tile.physical_z_offset
        };
        next_state.vec_coord =
            base_coord + vec3(offset_x as f64 / 128.0, 0.0, offset_z as f64 / 128.0);

        let (offset_x, offset_z) = eval_rotation(
            offset_x,
            offset_z,
            next_tile_id.flip_x(),
            next_tile_id.rotation(),
        );
        next_state.vec_coord =
            base_coord + vec3(offset_x as f64 / 128.0, 0.0, offset_z as f64 / 128.0);
        if CHATTY {
            tracing::info!(
                "Found matching tile: {:?}. Coords: {:?}.",
                matching_tile_id,
                self.vec_coord
            );
        }

        Ok(ScanOutcome::Success(next_state))
    }

    pub(crate) fn signal_rotation_ok(&self, rotation: u16) -> bool {
        let rotation = rotation & 3;
        let tile =
            TRACK_TILES[self.current_tile_id.x() as usize][self.current_tile_id.y() as usize];
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

    pub(crate) fn is_switch_eligible(&self) -> bool {
        let tile =
            TRACK_TILES[self.current_tile_id.x() as usize][self.current_tile_id.y() as usize];
        let tile = match tile {
            Some(tile) => tile,
            None => {
                return false;
            }
        };
        tile.switch_eligible
    }

    pub(crate) fn can_diverge_left(&self) -> bool {
        // The tile has to be switch eligible, it has to be a forward move facing the points, and the tile
        // has to be flipped along X (since unflipped tiles turn out to the right)
        self.is_switch_eligible() && !self.is_reversed && self.current_tile_id.flip_x()
    }
    pub(crate) fn can_diverge_right(&self) -> bool {
        // The tile has to be switch eligible, it has to be a forward move facing the points, and the tile
        // cannot be flipped along X (since unflipped tiles turn out to the left)
        self.is_switch_eligible() && !self.is_reversed && !self.current_tile_id.flip_x()
    }

    pub(crate) fn can_converge(&self) -> bool {
        // The tile has to be switch eligible, it has to be a backward move approaching against the points.
        self.is_switch_eligible() && self.is_reversed
    }
}

// test only
fn handle_popup_response(response: &PopupResponse, coord: BlockCoordinate) -> Result<()> {
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
                    vec_coord: b2vec(coord),
                    base_track_block: response
                        .ctx
                        .block_types()
                        .get_by_name("carts:rail_tile")
                        .unwrap(),
                    allowable_speed: 90.0,
                    // dummy
                    current_tile_id: TileId::empty(),
                };
                state.advance::<true>(|coord| response.ctx.game_map().get_block(coord).into())?;
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
                    vec_coord: b2vec(coord),
                    base_track_block: response
                        .ctx
                        .block_types()
                        .get_by_name("carts:rail_tile")
                        .unwrap(),
                    allowable_speed: 90.0,
                    // dummy
                    current_tile_id: TileId::empty(),
                };
                response.ctx.run_deferred(move |ctx| {
                    for _ in 0..20 {
                        let prev = state.vec_coord + vec3(0.0, 1.0, 0.0);

                        match state
                            .advance::<true>(|coord| ctx.game_map().get_block(coord).into())?
                        {
                            ScanOutcome::Success(s) => {
                                state = s;
                            }
                            _ => {
                                break;
                            }
                        }

                        let current = state.vec_coord + vec3(0.0, 1.0, 0.0);
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
