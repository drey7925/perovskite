//! Templates are contiguous sets of blocks that can be placed in the world as a unit.
//! They consist of a 3D array of block IDs, with a special-cased block ID (`MASK`) that
//! indicates that existing blocks should be preserved when placing the template.
//!
//! Templates have two formats:
//! * An in-memory format, which uses numeric block IDs and hence must be used with a block type manager
//!   and/or game map. Transporting these between game boundaries will lead to unpredictable behavior.
//! * A portable format, which includes a mapping table, and can be freely serialized (through its public API),
//!   deserialized (same API), and used across game boundaries. This also has an in-memory format, for the benefit
//!   of making hardcoded templates in Rust code while getting to reuse the block ID resolution logic.
//!
//! This module does not know anything about where templates are stored; you may freely store serialized templates
//! on disk, in the game database, somewhere on the network, etc.
//!
//! All templates are rectangular prisms of blocks; a template has its origin at the minimum x, y, and z coordinates
//! of the template. The rotation of the template is always normalized to variant 0, and

use perovskite_core::block_id::BlockId;

/// The mask block ID is used to indicate that existing blocks should be preserved when placing the template.
/// This is a variant of the `air` block, whose definition and variants are fully controlled by the engine rather
/// than content. We choose a small ID so that protobuf encoding is efficient.
pub const MASK: BlockId = BlockId(1);

pub struct InMemTemplate {
    /// The size of the template in each dimension.
    size: (u32, u32, u32),
    /// The block IDs in the template. The layout is X, Z, Y matching the layout used by `GameMap`.
    blocks: Vec<BlockId>,
}
