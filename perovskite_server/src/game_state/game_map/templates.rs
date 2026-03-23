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

use std::collections::BTreeMap;

use perovskite_core::block_id::BlockId;

use crate::game_state::blocks::ExtendedData;

/// The mask block ID is used to indicate that existing blocks should be preserved when placing the template.
/// This is a variant of the `air` block, whose definition and variants are fully controlled by the engine rather
/// than content. We choose a small ID so that protobuf encoding is efficient.
pub const MASK: BlockId = BlockId(1);

/// A template that is stored in memory, using numeric block IDs. It can be applied to the GameMap that it came from;
/// it should not be sent across process boundaries, or used across two instances of GameMap from two Server instances
/// in the same process.
///
/// A reasonable size
pub struct InMemTemplate {
    /// The size of the template in each dimension.
    size: (i32, i32, i32),
    /// The block IDs in the template. The layout is X, Z, Y matching the layout used by `GameMap`.
    blocks: Vec<BlockId>,
    /// The extended data attached to blocks within the template.
    extended_data: BTreeMap<usize, ExtendedData>,
}
impl InMemTemplate {
    /// Creates a new template of the given size, filled with mask blocks, meaning that it will not
    /// change the map at all, if applied.
    pub fn new_empty(size: (i32, i32, i32)) -> Self {
        assert!(size.0 > 0 && size.1 > 0 && size.2 > 0);
        // Avoid stack-allocating a large array just to move it into the vector's storage
        let mut blocks = bytemuck::zeroed_vec(size.0 as usize * size.1 as usize * size.2 as usize);
        blocks.fill(MASK);
        Self {
            size,
            blocks,
            extended_data: Default::default(),
        }
    }

    pub fn size(&self) -> (i32, i32, i32) {
        self.size
    }

    #[inline]
    pub fn block_at(&self, x: i32, y: i32, z: i32) -> BlockId {
        self.blocks[self.index(x, y, z)]
    }

    #[inline]
    pub fn ext_data_at(&self, x: i32, y: i32, z: i32) -> Option<&ExtendedData> {
        self.extended_data.get(&self.index(x, y, z))
    }

    #[inline]
    fn index(&self, x: i32, y: i32, z: i32) -> usize {
        assert!(x >= 0 && x < self.size.0);
        assert!(y >= 0 && y < self.size.1);
        assert!(z >= 0 && z < self.size.2);
        x as usize * self.size.1 as usize * self.size.2 as usize
            + z as usize * self.size.2 as usize
            + y as usize
    }

    #[inline]
    pub fn set_block_at(
        &mut self,
        x: i32,
        y: i32,
        z: i32,
        block_id: BlockId,
        extended_data: Option<ExtendedData>,
    ) {
        let index = self.index(x, y, z);
        self.blocks[index] = block_id;
        if let Some(ext_data) = extended_data {
            self.extended_data.insert(index, ext_data);
        } else {
            self.extended_data.remove(&index);
        }
    }
}
