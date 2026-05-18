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

use std::collections::{BTreeMap, HashMap};

use anyhow::{Context, Result};
use bytemuck::cast_slice;
use itertools::Itertools;
use perovskite_core::{block_id::BlockId, coordinates::BlockCoordinate};

use crate::game_state::blocks::{BlockTypeManager, ExtendedData};

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
    pub fn new_empty(sx: i32, sy: i32, sz: i32) -> Self {
        let size = (sx, sz, sy);
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

    /// Returns the size of the template in (x, y, z) blocks.
    pub fn size(&self) -> (i32, i32, i32) {
        (self.size.0, self.size.2, self.size.1)
    }

    /// Returns the block ID at the given template-local coordinate.
    #[inline]
    pub fn block_at(&self, x: i32, y: i32, z: i32) -> BlockId {
        self.blocks[self.index(x, y, z)]
    }

    /// Returns the extended data at the given template-local coordinate, if any.
    #[inline]
    pub fn ext_data_at(&self, x: i32, y: i32, z: i32) -> Option<&ExtendedData> {
        self.extended_data.get(&self.index(x, y, z))
    }

    #[inline]
    fn index(&self, x: i32, y: i32, z: i32) -> usize {
        assert!(x >= 0 && x < self.size.0);
        assert!(z >= 0 && z < self.size.1);
        assert!(y >= 0 && y < self.size.2);
        x as usize * self.size.1 as usize * self.size.2 as usize
            + z as usize * self.size.2 as usize
            + y as usize
    }

    /// Sets the block and optional extended data at the given template-local coordinate.
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

#[derive(prost::Message)]
pub struct SerializedTemplate {
    #[prost(uint32, tag = "1")]
    pub sx: u32,
    #[prost(uint32, tag = "2")]
    pub sy: u32,
    #[prost(uint32, tag = "3")]
    pub sz: u32,
    #[prost(uint32, repeated, tag = "4")]
    pub blocks: Vec<u32>,
    #[prost(map = "uint64,message", tag = "5")]
    pub extended_data: HashMap<u64, perovskite_core::protocol::map::ExtendedData>,
    #[prost(map = "uint32,string", tag = "6")]
    pub block_type_mapping: HashMap<u32, String>,
}
impl SerializedTemplate {
    /// Converts a `InMemTemplate` to a `SerializedTemplate`.
    pub fn from_in_mem(in_mem: &InMemTemplate, blocks: &BlockTypeManager) -> Result<Self> {
        let (sx, sy, sz) = in_mem.size();
        let mut extended_data = HashMap::new();
        let mut block_type_mapping = HashMap::new();
        for (index, ext_data) in &in_mem.extended_data {
            let (block_type, _) = blocks.get_block_by_id(in_mem.blocks[*index])?;

            let ext_data_proto = super::serialize_ext_data_for_server(
                0,                             // The offset is bogus; we will reconstruct it later.
                BlockCoordinate::new(0, 0, 0), // only used for error reporting
                ext_data,
                block_type,
            )?;
            if let Some(ext_data_proto) = ext_data_proto {
                extended_data.insert(*index as u64, ext_data_proto);
            }
        }
        for block_id in in_mem
            .blocks
            .iter()
            .map(|x| x.with_variant_unchecked(0))
            .filter(|x| *x != MASK)
            .unique()
        {
            block_type_mapping.insert(
                block_id.into(),
                blocks.get_block_by_id(block_id)?.0.short_name().to_string(),
            );
        }
        Ok(Self {
            sx: sx as u32,
            sy: sy as u32,
            sz: sz as u32,
            blocks: cast_slice(&in_mem.blocks).to_vec(),
            extended_data,
            block_type_mapping,
        })
    }

    pub fn to_in_mem(&self, blocks: &BlockTypeManager) -> Result<InMemTemplate> {
        if self.blocks.len() != self.sx as usize * self.sy as usize * self.sz as usize {
            return Err(anyhow::anyhow!("Invalid block count"));
        }
        if self.blocks.is_empty() {
            return Ok(InMemTemplate::new_empty(0, 0, 0));
        }
        let mut in_mem = InMemTemplate::new_empty(self.sx as i32, self.sy as i32, self.sz as i32);

        let max_block_index = (self.blocks.iter().max().unwrap() >> 12) as usize;
        let mut remapping: Vec<BlockId> = Vec::with_capacity(max_block_index + 1);
        remapping.resize_with(max_block_index + 1, || MASK);

        for (their_id, name) in &self.block_type_mapping {
            let block_type = blocks
                .get_by_name(name)
                .with_context(|| format!("unknown block {}", name))?;
            remapping[*their_id as usize >> 12] = block_type;
        }

        let mut remapped_blocks = Vec::with_capacity(self.blocks.len());
        for their_id in &self.blocks {
            let block_index = their_id >> 12;
            if block_index as usize >= remapping.len() {
                return Err(anyhow::anyhow!("Invalid block ID"));
            }
            let our_id = remapping[block_index as usize].with_variant_of(BlockId(*their_id));
            remapped_blocks.push(our_id);
        }

        for (index, ext_data_proto) in &self.extended_data {
            let our_id = remapped_blocks[*index as usize];
            let our_block_type = blocks.get_block_by_id(our_id)?.0;
            let ext_data = super::deserialize_server_ext_data(
                ext_data_proto.clone(),
                BlockCoordinate::new(0, 0, 0), // offset is bogus; only used for debugging
                &our_block_type,
            )?;
            in_mem.extended_data.insert(*index as usize, ext_data);
        }
        in_mem.blocks = remapped_blocks;
        Ok(in_mem)
    }
}

#[test]
fn test_round_trip() {
    use crate::game_state::blocks::testonly_make_dummy_block;
    let mut src_blocks = BlockTypeManager::new();
    let src_a = src_blocks
        .register_block(testonly_make_dummy_block(String::from("block_a")))
        .unwrap();
    let src_b = src_blocks
        .register_block(testonly_make_dummy_block(String::from("block_b")))
        .unwrap();

    let mut dst_blocks = BlockTypeManager::new();
    // register in the opposite order
    let dst_b = dst_blocks
        .register_block(testonly_make_dummy_block(String::from("block_b")))
        .unwrap();
    let dst_a = dst_blocks
        .register_block(testonly_make_dummy_block(String::from("block_a")))
        .unwrap();

    let mut in_mem = InMemTemplate::new_empty(1, 1, 2);
    in_mem.set_block_at(0, 0, 0, src_a.with_variant_unchecked(1), None);
    in_mem.set_block_at(0, 0, 1, src_b.with_variant_unchecked(2), None);

    let serialized = SerializedTemplate::from_in_mem(&in_mem, &src_blocks).unwrap();
    let round_tripped = serialized.to_in_mem(&dst_blocks).unwrap();
    assert_eq!(round_tripped.blocks[0], dst_a.with_variant_unchecked(1));
    assert_eq!(round_tripped.blocks[1], dst_b.with_variant_unchecked(2));
}
