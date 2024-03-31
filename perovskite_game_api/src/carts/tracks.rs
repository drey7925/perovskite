// For now, hide unused warnings since they're distracting
#![allow(dead_code)]

use anyhow::{Context, Result};
use cgmath::vec3;
use perovskite_core::{
    chat::ChatMessage, coordinates::BlockCoordinate, protocol::render::DynamicCrop,
};
use perovskite_server::game_state::{
    client_ui::{PopupAction, PopupResponse, UiElementContainer},
    event::{EventInitiator, HandlerContext},
    game_map::ServerGameMap,
};
use rand::Rng;

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

    fn from_variant(variant: u16, reverse: bool, diverging: bool) -> Self {
        TileId(
            c::ENTRY_PRESENT
                | (variant & 0b0000_1111_1111_1111)
                | if reverse { c::REVERSE_SCAN } else { 0 }
                | if diverging { c::DIVERGING_ROUTE } else { 0 },
        )
    }

    fn with_rotation_wrapped(&self, rotation: u16) -> TileId {
        TileId(self.0 & !c::ROTATION_BITS | (rotation & c::ROTATION_BITS))
    }

    fn xor_flip_x(&self, flip_x: bool) -> TileId {
        TileId(self.0 ^ if flip_x { c::FLIP_X_BIT } else { 0 })
    }

    fn correct_rotation_for_x_flip(&self, flip_x: bool) -> TileId {
        // If facing 1/3, then correct for x-flip
        if self.rotation() & 1 == 1 && flip_x {
            TileId(self.0 ^ 2)
        } else {
            *self
        }
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

#[derive(Debug, Clone, Copy)]
struct TrackTile {
    /// The next tile that we will scan. This is given for variant = 0, and internally rotated by the engine when we're scanning other variants
    next_coord: EncodedDelta,
    /// The previous tile that we will scan. This is given for variant = 0, and internally rotated by the engine when we're scanning other variants
    prev_coord: EncodedDelta,
    /// Secondary tile that needs to be set. As with before, this is given for variant = 0, and internally rotated by the engine when we're scanning other variants
    secondary_coord: EncodedDelta,
    /// The tile we expect at the secondary tile.
    secondary_tile: TileId,
    /// Tertiary tile that needs to be set. As with before, this is given for variant = 0, and internally rotated by the engine when we're scanning other variants
    tertiary_coord: EncodedDelta,
    /// The tile we expect at the tertiary tile
    tertiary_tile: TileId,
    /// The variants that can exist in the next tile if we're signalled straight through (or if there is no choice).
    /// The tile ID for (0,0) is special: it means any of the STRAIGHT_TRACK_ELIGIBLE_CONNECTIONS (i.e. the tracks that line up to a straight track)
    /// Note that even for that tile ID, the rotation bits in the tile ID are applied to figure out the direction in which we enter that straight-track-compatible
    /// connection, whatever it may be
    allowed_next_tiles: [TileId; 3],
    /// The variants that can exist in the next tile if we're signalled to diverge.
    allowed_next_diverging_tiles: [TileId; 1],
    /// The variants that can exist in the prev tile if we're signalled straight through (or if there is no choice).
    /// The tile ID for (0,0) is special: it means any of the STRAIGHT_TRACK_ELIGIBLE_CONNECTIONS (i.e. the tracks that line up to a straight track)
    allowed_prev_tiles: [TileId; 1],
    /// The variants that can exist in the prev tile if we're signalled to diverge.
    allowed_prev_diverging_tiles: [TileId; 1],
}
impl TrackTile {
    // This can't be in Default because it needs to be const
    const fn default() -> Self {
        TrackTile {
            next_coord: EncodedDelta::empty(),
            prev_coord: EncodedDelta::empty(),
            secondary_coord: EncodedDelta::empty(),
            secondary_tile: TileId::empty(),
            tertiary_coord: EncodedDelta::empty(),
            tertiary_tile: TileId::empty(),
            allowed_next_tiles: [TileId::empty(); 3],
            allowed_next_diverging_tiles: [TileId::empty(); 1],
            allowed_prev_tiles: [TileId::empty(); 1],
            allowed_prev_diverging_tiles: [TileId::empty(); 1],
        }
    }
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
const STRAIGHT_TRACK_ELIGIBLE_CONNECTIONS: [TileId; 8] = [
    // We can enter a straight track with the same rotation, regardless of flip_x
    TileId::new(0, 0, 0, false, false, false),
    TileId::new(0, 0, 0, true, false, false),
    // We can enter a straight track from the other side, and do a reverse move through it
    TileId::new(0, 0, 2, false, true, false),
    TileId::new(0, 0, 2, true, true, false),
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
        next_coord: EncodedDelta::new(0, 1),
        prev_coord: EncodedDelta::new(0, -1),
        allowed_next_tiles: [
            // Connects to a straight track (or straight track compatible) with the same rotation on its output side
            TileId::new(0, 0, 0, false, false, false),
            TileId::empty(),
            TileId::empty(),
        ],
        allowed_prev_tiles: [TileId::new(0, 0, 2, false, false, false)],
        ..TrackTile::default()
    });
    // [8][8] is the 90 degree curve without a switch
    track_tiles[8][8] = Some(TrackTile {
        next_coord: EncodedDelta::new(1, 0),
        prev_coord: EncodedDelta::new(0, -1),
        allowed_next_tiles: [
            TileId::new(0, 0, 1, false, false, false),
            TileId::empty(),
            TileId::empty(),
        ],
        allowed_prev_tiles: [TileId::new(0, 0, 2, false, false, false)],
        ..TrackTile::default()
    });

    track_tiles
}

lazy_static::lazy_static! {
    static ref TRACK_TILES: [[Option<TrackTile>; 16]; 11] = build_track_tiles();
}

pub(crate) fn register_tracks(game_builder: &mut crate::game_builder::GameBuilder) -> Result<()> {
    const SIGNAL_SIDE_TOP_TEX: StaticTextureName = StaticTextureName("carts:signal_side_top");
    const RAIL_TEST_TEX: StaticTextureName = StaticTextureName("carts:rail_test");

    include_texture_bytes!(game_builder, RAIL_TEST_TEX, "textures/rail_atlas_test.png")?;

    let rail_test_box = AaBoxProperties::new(
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_SIDE_TOP_TEX,
        RAIL_TEST_TEX,
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_SIDE_TOP_TEX,
        SIGNAL_SIDE_TOP_TEX,
        crate::blocks::TextureCropping::AutoCrop,
        crate::blocks::RotationMode::RotateHorizontally,
    );
    game_builder.add_block(
        BlockBuilder::new(StaticBlockName("carts:rail_test"))
            .set_axis_aligned_boxes_appearance(AxisAlignedBoxesAppearanceBuilder::new().add_box(
                rail_test_box,
                (-0.5, 0.5),
                (-0.5, -0.4),
                (-0.5, 0.5),
            ))
            .add_modifier(Box::new(|bt| {
                bt.interact_key_handler = Some(Box::new(|ctx, coord| {
                    // let mut rng = rand::thread_rng();
                    // let variant = rng.gen_range(0..4096);
                    // ctx.game_map().mutate_block_atomically(coord, |b, _ext| {
                    //     // TODO this should check equals_ignore_variant
                    //     let old_variant = b.variant();
                    //     let new_variant = (variant & !3) | (old_variant & 3);
                    //     *b = b.with_variant(new_variant)?;

                    //     ctx.initiator().send_chat_message(ChatMessage::new("[RNG]", format!("{:b}", variant)))?;

                    //     Ok(())
                    // })?;

                    let block = ctx.game_map().get_block(coord)?;
                    ctx.initiator().send_chat_message(ChatMessage::new(
                        "[INFO]",
                        format!("{:?}", TileId::from_variant(block.variant(), false, false)),
                    ))?;

                    Ok(Some(ctx.new_popup()
                        .title("Rail test")
                        .label("Update tile:")
                        .text_field("tile_x", "Tile X", "0", true)
                        .text_field("tile_y", "Tile Y", "0", true)
                        .text_field("rotation", "Rotation", "0", true)
                        .text_field("flip_x", "Flip X", "false", true)
                        .button("apply", "Apply", true)
                        .label("Scan tile:")
                        .text_field("reverse", "Reverse", "false", true)
                        .text_field("diverging", "Diverging", "false", true)
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
                                }
                            )
                        })
                    },
                    _ => unreachable!()
                }
            })),
    )?;

    Ok(())
}

struct ScanState {
    block_coord: BlockCoordinate,
    is_reversed: bool,
    is_diverging: bool,
}
impl ScanState {
    // Test only, prototype version that logs lots of details about each calculation
    fn advance_verbose(&mut self, ctx: &HandlerContext) -> Result<()> {
        let block = ctx.game_map().get_block(self.block_coord)?;
        let variant = block.variant();
        let current_tile_id = TileId::from_variant(variant, self.is_reversed, self.is_diverging);
        if !current_tile_id.present() {}
        let tile = TRACK_TILES[current_tile_id.x() as usize][current_tile_id.y() as usize];
        ctx.initiator().send_chat_message(ChatMessage::new(
            "[SCAN]",
            format!("{:?}:\n {:?}", current_tile_id, tile),
        ))?;
        let current_tile_def = match tile {
            Some(tile) => tile,
            None => return Ok(()),
        };
        // The next place we would scan
        let next_coord = if self.is_reversed {
            current_tile_def.prev_coord.eval_delta(
                self.block_coord,
                current_tile_id.flip_x(),
                current_tile_id.rotation(),
            )
        } else {
            current_tile_def.next_coord.eval_delta(
                self.block_coord,
                current_tile_id.flip_x(),
                current_tile_id.rotation(),
            )
        }
        .context("block coordinate overflow")?;
        ctx.initiator().send_chat_message(ChatMessage::new(
            "[SCAN]",
            format!("Next coord: {:?}", next_coord),
        ))?;

        let next_block = ctx.game_map().get_block(next_coord)?;
        // TODO check that the block is actually a rail
        let next_variant = next_block.variant();

        // This is the tile we see at the next coordinate. Does it match?
        let next_tile_id = TileId::from_variant(next_variant, self.is_reversed, self.is_diverging);
        ctx.initiator().send_chat_message(ChatMessage::new(
            "[SCAN]",
            format!("Next tile: {:?}", next_tile_id),
        ))?;

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
        ctx.initiator().send_chat_message(ChatMessage::new(
            "[SCAN]",
            format!(
                "Corrected rotation: {} -> {}",
                next_tile_id.rotation(),
                rotation_corrected_next_tile_id.rotation()
            ),
        ))?;

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
                ctx.initiator().send_chat_message(ChatMessage::new(
                    "[SCAN]",
                    format!(
                        "Trying the straight track connections. Corrected {} -> {:?}",
                        rotation_corrected_next_tile_id.rotation(),
                        straight_track_rotation_corrected_tile_id
                    ),
                ))?;
                // try all the straight track connections
                for &proposed_straight_track_id in STRAIGHT_TRACK_ELIGIBLE_CONNECTIONS.iter() {
                    if straight_track_rotation_corrected_tile_id
                        .same_variant(proposed_straight_track_id)
                    {
                        matching_tile_id = proposed_straight_track_id;
                    }
                }
            } else if rotation_corrected_next_tile_id.same_variant(proposed_tile_id) {
                matching_tile_id = proposed_tile_id;
            }
        }
        if !matching_tile_id.present() {
            ctx.initiator().send_chat_message(ChatMessage::new(
                "[SCAN]",
                "No matching tile found".to_string(),
            ))?;
            return Ok(());
        }
        self.block_coord = next_coord;
        self.is_reversed = matching_tile_id.reverse();
        self.is_diverging = matching_tile_id.diverging();

        ctx.initiator().send_chat_message(ChatMessage::new(
            "[SCAN]",
            format!("Found matching tile: {:?}", matching_tile_id),
        ))?;

        Ok(())
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
                let flip_x = response
                    .textfield_values
                    .get("flip_x")
                    .context("missing flip_x")?
                    .parse::<bool>()?;
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
                let mut state = ScanState {
                    block_coord: coord,
                    is_reversed: response
                        .textfield_values
                        .get("reverse")
                        .unwrap()
                        .parse::<bool>()?,
                    is_diverging: response
                        .textfield_values
                        .get("diverging")
                        .unwrap()
                        .parse::<bool>()?,
                };
                state.advance_verbose(&response.ctx)?;
            }
            "multiscan" => {
                let mut state = ScanState {
                    block_coord: coord,
                    is_reversed: response
                        .textfield_values
                        .get("reverse")
                        .unwrap()
                        .parse::<bool>()?,
                    is_diverging: response
                        .textfield_values
                        .get("diverging")
                        .unwrap()
                        .parse::<bool>()?,
                };
                response.ctx.run_deferred(move |ctx| {
                    for _ in 0..20 {
                        let prev = b2vec(state.block_coord) + vec3(0.0, 1.0, 0.0);
                        state.advance_verbose(ctx)?;

                        let current = b2vec(state.block_coord) + vec3(0.0, 1.0, 0.0);
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
