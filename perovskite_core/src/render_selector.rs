// Copyright 2023 drey7925
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

//! The *render selector* is a 32-bit word that the client builds for each block while
//! meshing (and when evaluating collision/tool hitboxes), and which conditional block
//! geometry is matched against. It is a superset of the block variant; the extra bits are
//! computed locally on the client and never appear on the wire or in the map database.
//!
//! Bit layout:
//! * bits `[0, 12)` — the block variant, exactly as sent by the server.
//! * bits `[12, 18)` — neighbor-presence bits, one per face, in the order
//!   X+, X-, Y+ (up), Y- (down), Z+, Z-. A bit is set if the neighboring block in that
//!   direction is considered "present" for connection purposes (currently: solid opaque,
//!   or the same base block as the block being rendered). These are computed by the
//!   client from its local copy of the map.
//! * bits `[18, 31)` — reserved, always zero.
//! * bit 31 — always set. This allows an "always render" mask to be expressed as
//!   all-ones, so the hot rendering path only needs a single `mask & selector != 0`
//!   test with no special case for an empty mask.
//!
//! A conditional geometry element (e.g. an [`AxisAlignedBox`](crate::protocol::blocks::AxisAlignedBox)
//! via its `variant_mask` field) is active when `mask & selector != 0`. A mask of zero is
//! normalized to all-ones ("always active") when block definitions are parsed.

use crate::block_id::BLOCK_VARIANT_MASK;

/// Bits of the render selector holding the server-assigned block variant.
pub const VARIANT_BITS: u32 = BLOCK_VARIANT_MASK;

/// Set when the neighbor in the X+ direction is present.
pub const NEIGHBOR_XPLUS: u32 = 1 << 12;
/// Set when the neighbor in the X- direction is present.
pub const NEIGHBOR_XMINUS: u32 = 1 << 13;
/// Set when the neighbor in the Y+ (up) direction is present.
pub const NEIGHBOR_YPLUS: u32 = 1 << 14;
/// Set when the neighbor in the Y- (down) direction is present.
pub const NEIGHBOR_YMINUS: u32 = 1 << 15;
/// Set when the neighbor in the Z+ direction is present.
pub const NEIGHBOR_ZPLUS: u32 = 1 << 16;
/// Set when the neighbor in the Z- direction is present.
pub const NEIGHBOR_ZMINUS: u32 = 1 << 17;

/// All six neighbor-presence bits.
pub const ALL_NEIGHBOR_BITS: u32 = NEIGHBOR_XPLUS
    | NEIGHBOR_XMINUS
    | NEIGHBOR_YPLUS
    | NEIGHBOR_YMINUS
    | NEIGHBOR_ZPLUS
    | NEIGHBOR_ZMINUS;

/// Always set in a render selector; see the module docs for rationale.
pub const ALWAYS: u32 = 1 << 31;

/// The bits that a geometry mask (e.g. `AxisAlignedBox::variant_mask`) may meaningfully
/// contain. Masks with bits outside this range are clamped (with a warning) by the client.
pub const VALID_MASK_BITS: u32 = VARIANT_BITS | ALL_NEIGHBOR_BITS;

/// The neighbor offsets corresponding to the neighbor-presence bits, in bit order
/// starting at [`NEIGHBOR_XPLUS`]. Offsets are in world coordinates with Y up.
pub const NEIGHBOR_OFFSETS: [(i8, i8, i8); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

/// Builds a render selector with the given variant and no neighbor bits.
#[inline(always)]
pub const fn from_variant(variant: u16) -> u32 {
    ALWAYS | (variant as u32 & VARIANT_BITS)
}
