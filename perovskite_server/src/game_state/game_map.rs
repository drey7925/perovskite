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

use anyhow::Error;
use perovskite_core::block_id::BlockId;
use perovskite_core::lighting::{ChunkColumn, Lightfield};
use rand::distributions::Bernoulli;
use rand::prelude::Distribution;
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use smallvec::{smallvec, SmallVec};
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::{
    collections::HashSet,
    fmt::Debug,
    sync::{Arc, Weak},
    time::{Duration, Instant},
};
use tokio_util::sync::CancellationToken;
use tracy_client::{plot, span};

use crate::sync::{AtomicInstant, RwCondvar};
use crate::{
    database::database_engine::{GameDatabase, KeySpace},
    game_state::inventory::Inventory,
};
use crate::{run_handler, CachelineAligned};

use super::blocks::BlockInteractionResult;
use super::event::log_trace;
use super::{
    blocks::{
        self, BlockTypeManager, ExtDataHandling, ExtendedData, ExtendedDataHolder, InlineContext,
        TryAsHandle,
    },
    event::{EventInitiator, HandlerContext},
    items::ItemStack,
    GameState,
};

use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use perovskite_core::{
    coordinates::{BlockCoordinate, ChunkCoordinate, ChunkOffset},
    protocol::map as mapchunk_proto,
};
use prost::Message;

use log::{error, info, warn};

use anyhow::{bail, ensure, Context, Result};
use integer_encoding::{VarIntReader, VarIntWriter};
use tokio::{
    sync::{broadcast, mpsc},
    task::JoinHandle,
};

trait AsDbKey
where
    Self: Sized,
{
    fn as_bytes(&self) -> Vec<u8>;
    fn from_bytes(bytes: &[u8]) -> Result<Self>;
}

#[derive(Clone, Copy, Debug)]
pub enum CasOutcome {
    /// The block compared equal to the expected and was set accordingly.
    Match,
    /// The block did not match the desired one, and no change was made to the map.
    Mismatch,
}

impl AsDbKey for BlockCoordinate {
    fn as_bytes(&self) -> Vec<u8> {
        let mut result = Vec::new();
        result
            .write_varint(self.x)
            .expect("varint write should not fail");
        result
            .write_varint(self.y)
            .expect("varint write should not fail");
        result
            .write_varint(self.z)
            .expect("varint write should not fail");
        result
    }

    fn from_bytes(bytes: &[u8]) -> Result<BlockCoordinate> {
        let len = bytes.len();
        let mut bytes = std::io::Cursor::new(bytes);
        let x = bytes.read_varint()?;
        let y = bytes.read_varint()?;
        let z = bytes.read_varint()?;
        ensure!(
            bytes.position() == len as u64,
            "Trailing data after provided bytes"
        );
        Ok(BlockCoordinate { x, y, z })
    }
}

impl AsDbKey for ChunkCoordinate {
    fn as_bytes(&self) -> Vec<u8> {
        let mut result = Vec::new();
        result
            .write_varint(self.x)
            .expect("varint write should not fail");
        result
            .write_varint(self.y)
            .expect("varint write should not fail");
        result
            .write_varint(self.z)
            .expect("varint write should not fail");
        result
    }

    fn from_bytes(bytes: &[u8]) -> Result<ChunkCoordinate> {
        let len = bytes.len();
        let mut bytes = std::io::Cursor::new(bytes);
        let x = bytes.read_varint()?;
        let y = bytes.read_varint()?;
        let z = bytes.read_varint()?;
        ensure!(
            bytes.position() == len as u64,
            "Trailing data after provided bytes"
        );
        Ok(ChunkCoordinate { x, y, z })
    }
}

#[derive(PartialEq, Eq)]
pub(crate) enum ChunkUsage {
    // Chunk is for the client. Extended data is not present (or client-specific ext data will be present
    // if that is implemented)
    Client,
    // Chunk is for the server and its database. Extended data is present.
    Server,
}

/// Representation of a single chunk in memory. Most game logic should instead use the game map directly,
/// which abstracts over chunk boundaries and presents a unified interface that does not require being aware
/// of how chunks are divided.
///
/// This class is meant for use by map generators and bulk map accesses that would be inefficient when done
/// block-by-block.
pub struct MapChunk {
    coord: ChunkCoordinate,
    // TODO: this was exposed for the temporary mapgen API. Lock this down and refactor
    // more calls similar to mutate_block_atomically
    pub(crate) block_ids: Arc<[AtomicU32; 4096]>,
    extended_data: Box<FxHashMap<u16, ExtendedData>>,
    dirty: bool,
}
impl MapChunk {
    fn new(coord: ChunkCoordinate, storage: Arc<[AtomicU32; 4096]>) -> Self {
        Self {
            coord,
            block_ids: storage,
            extended_data: Default::default(),
            dirty: false,
        }
    }

    fn serialize(
        &self,
        usage: ChunkUsage,
        game_state: &GameState,
    ) -> Result<mapchunk_proto::StoredChunk> {
        let _span = span!("serialize chunk");
        let mut extended_data = Vec::new();

        if usage == ChunkUsage::Server {
            for index in 0..4096 {
                let offset = ChunkOffset::from_index(index);
                let block_coord = self.coord.with_offset(offset);

                if let Some(ext_data) = self.extended_data.get(&index.try_into().unwrap()) {
                    if let Some(ext_data_proto) =
                        self.extended_data_to_proto(index, block_coord, ext_data, game_state)?
                    {
                        extended_data.push(ext_data_proto);
                    }
                }
            }
        }

        let proto = mapchunk_proto::StoredChunk {
            chunk_data: Some(mapchunk_proto::stored_chunk::ChunkData::V1(
                mapchunk_proto::ChunkV1 {
                    block_ids: self
                        .block_ids
                        .iter()
                        .map(|x| x.load(Ordering::Relaxed))
                        .collect(),
                    extended_data,
                },
            )),
            startup_counter: game_state.startup_counter,
        };

        Ok(proto)
    }

    fn extended_data_to_proto(
        &self,
        block_index: usize,
        block_coord: BlockCoordinate,
        ext_data: &ExtendedData,
        game_state: &GameState,
    ) -> Result<Option<mapchunk_proto::ExtendedData>> {
        // We're in MapChunk so we have at least a read-level mutex, meaning we can use a relaxed load
        let id = self.block_ids[block_index].load(Ordering::Relaxed);
        let (block_type, _) = game_state
            .game_map()
            .block_type_manager()
            .get_block_by_id(id.into())?;
        if block_type.extended_data_handling == ExtDataHandling::NoExtData
            && ext_data.custom_data.is_some()
        {
            error!(
            "Found extended data, but block {} doesn't support extended data, while serializing {:?}, ",
            block_type.client_info.short_name, block_coord,
        );
        }

        if ext_data.custom_data.is_some() || !ext_data.inventories.is_empty() {
            let serialized_custom_data = match ext_data.custom_data.as_ref() {
                Some(x) => match &block_type.serialize_extended_data_handler {
                    Some(serializer) => {
                        let handler_context = InlineContext {
                            tick: game_state.tick(),
                            initiator: EventInitiator::Engine,
                            location: block_coord,
                            block_types: game_state.game_map().block_type_manager(),
                            items: game_state.item_manager(),
                        };

                        serializer(handler_context, x)?
                    }
                    None => {
                        if ext_data.custom_data.is_some() {
                            error!(
                                    "Block at {:?}, type {} indicated extended data, but had no serialize handler",
                                    block_coord, block_type.client_info.short_name
                                );
                        }
                        None
                    }
                },
                None => None,
            };
            let inventories = ext_data
                .inventories
                .iter()
                .map(|(k, v)| (k.clone(), v.to_proto()))
                .collect();

            Ok(Some(mapchunk_proto::ExtendedData {
                offset_in_chunk: block_index.try_into().unwrap(),
                serialized_data: serialized_custom_data.unwrap_or_default(),
                simple_storage: ext_data.simple_data.clone(),
                inventories,
            }))
        } else {
            Ok(None)
        }
    }

    fn deserialize(
        coordinate: ChunkCoordinate,
        bytes: &[u8],
        game_state: Arc<GameState>,
        storage: Arc<[AtomicU32; 4096]>,
    ) -> Result<MapChunk> {
        let _span = span!("parse chunk");
        let proto = mapchunk_proto::StoredChunk::decode(bytes)
            .with_context(|| "MapChunk proto serialization failed")?;
        let run_cold_load_postprocessors = proto.startup_counter != game_state.startup_counter;
        match proto.chunk_data {
            Some(mapchunk_proto::stored_chunk::ChunkData::V1(chunk_data)) => parse_v1(
                chunk_data,
                coordinate,
                game_state,
                storage,
                run_cold_load_postprocessors,
            ),
            None => bail!("Missing chunk_data or unrecognized format"),
        }
    }
    /// Sets the block at the given coordinate within the chunk.
    /// This function is intended to be used from map generators and bulk timer callbacks
    ///
    /// TODO refactor and fix
    pub fn set_block(
        &mut self,
        coordinate: ChunkOffset,
        block: BlockId,
        extended_data: Option<ExtendedData>,
    ) {
        // We have a &mut MapChunk, meaning we can use relaxed loads and stores
        let old_block = BlockId(self.block_ids[coordinate.as_index()].load(Ordering::Relaxed));
        let extended_data_was_some = extended_data.is_some();
        self.block_ids[coordinate.as_index()].store(block.into(), Ordering::Relaxed);
        let old_ext_data = if let Some(extended_data) = extended_data {
            self.extended_data
                .insert(coordinate.as_index().try_into().unwrap(), extended_data)
        } else {
            self.extended_data
                .remove(&coordinate.as_index().try_into().unwrap())
        };

        if old_block != block || extended_data_was_some || old_ext_data.is_some() {
            self.dirty = true;
        }
    }
    /// Swaps blocks at two offsets, with extended data moved along with them
    pub fn swap_blocks(&mut self, a: ChunkOffset, b: ChunkOffset) {
        if a == b {
            return;
        }

        // .swap() would have been nice, but it won't borrowck.
        let a_id = self.block_ids[a.as_index()].load(Ordering::Relaxed);
        let b_id = self.block_ids[b.as_index()].load(Ordering::Relaxed);
        self.block_ids[a.as_index()].store(b_id, Ordering::Relaxed);
        self.block_ids[b.as_index()].store(a_id, Ordering::Relaxed);

        let a_ext = self.extended_data.remove(&(a.as_index() as u16));
        let b_ext = self.extended_data.remove(&(b.as_index() as u16));
        if let Some(a_ext) = a_ext {
            self.extended_data.insert(b.as_index() as u16, a_ext);
        }
        if let Some(b_ext) = b_ext {
            self.extended_data.insert(a.as_index() as u16, b_ext);
        }
        self.dirty = true;
    }

    pub fn swap_blocks_across_chunks(
        a_chunk: &mut Self,
        b_chunk: &mut Self,
        a_coord: ChunkOffset,
        b_coord: ChunkOffset,
    ) {
        // std::mem::swap would have been nice, but it won't borrowck.
        let a_id = a_chunk.block_ids[a_coord.as_index()].load(Ordering::Relaxed);
        let b_id = b_chunk.block_ids[b_coord.as_index()].load(Ordering::Relaxed);
        a_chunk.block_ids[a_coord.as_index()].store(b_id, Ordering::Relaxed);
        b_chunk.block_ids[b_coord.as_index()].store(a_id, Ordering::Relaxed);

        // a_chunk and b_chunk cannot alias each other, so no need for an equality check
        let a_ext = a_chunk.extended_data.remove(&(a_coord.as_index() as u16));
        let b_ext = b_chunk.extended_data.remove(&(b_coord.as_index() as u16));
        if let Some(a_ext) = a_ext {
            b_chunk
                .extended_data
                .insert(b_coord.as_index() as u16, a_ext);
        }
        if let Some(b_ext) = b_ext {
            a_chunk
                .extended_data
                .insert(a_coord.as_index() as u16, b_ext);
        }
        a_chunk.dirty = true;
        b_chunk.dirty = true;
    }

    #[inline]
    pub fn get_block(&self, coordinate: ChunkOffset) -> BlockId {
        // We have a &MapChunk, meaning we can use relaxed loads
        BlockId(self.block_ids[coordinate.as_index()].load(Ordering::Relaxed))
    }
}

fn parse_v1(
    mut chunk_data: mapchunk_proto::ChunkV1,
    coordinate: ChunkCoordinate,
    game_state: Arc<GameState>,
    storage: Arc<[AtomicU32; 4096]>,
    run_cold_load_postprocessors: bool,
) -> std::result::Result<MapChunk, Error> {
    let mut extended_data = FxHashMap::default();
    ensure!(
        chunk_data.block_ids.len() == 4096,
        "Block IDs length != 4096"
    );
    if run_cold_load_postprocessors {
        // The length should be right so this should be safe and zero or low cost
        for processor in game_state.block_types().cold_load_postprocessors() {
            let cast: &mut [BlockId] =
                bytemuck::cast_slice_mut(chunk_data.block_ids.as_mut_slice());
            let sized: &mut [BlockId; 4096] =
                cast.try_into().expect("Failed to convert to BlockId array");
            processor(sized);
        }
    }

    for mapchunk_proto::ExtendedData {
        offset_in_chunk,
        serialized_data,
        inventories,
        simple_storage,
    } in chunk_data.extended_data.into_iter()
    {
        ensure!(offset_in_chunk < 4096);
        let offset = ChunkOffset::from_index(offset_in_chunk as usize);
        let block_coord = coordinate.with_offset(offset);
        let block_id = *chunk_data.block_ids.get(offset_in_chunk as usize).expect(
            "Block IDs vec lookup failed, even though bounds check passed. This should not happen.",
        );
        let (block_def, _) = game_state
            .game_map()
            .block_type_manager()
            .get_block_by_id(block_id.into())?;
        if block_def.extended_data_handling == ExtDataHandling::NoExtData {
            error!(
                "Block at {:?}, type {} cannot handle extended data, but serialized chunk contained extended data",
                block_coord,
                block_def.client_info.short_name
            );
            continue;
        }
        let handler_context = InlineContext {
            tick: game_state.tick(),
            initiator: EventInitiator::Engine,
            location: block_coord,
            block_types: game_state.game_map().block_type_manager(),
            items: game_state.item_manager(),
        };
        if let Some(ref deserialize) = block_def.deserialize_extended_data_handler {
            extended_data.insert(
                offset_in_chunk.try_into().unwrap(),
                ExtendedData {
                    custom_data: deserialize(handler_context, &serialized_data)?,
                    simple_data: simple_storage,
                    inventories: inventories
                        .iter()
                        .map(|(k, v)| Ok((k.clone(), Inventory::from_proto(v.clone(), None)?)))
                        .collect::<Result<hashbrown::HashMap<_, _>>>()?,
                },
            );
        } else {
            if !serialized_data.is_empty() {
                warn!(
                    "Block at {:?}, type {} has extended data, but had no deserialize handler",
                    block_coord, block_def.client_info.short_name
                );
            }
            extended_data.insert(
                offset_in_chunk.try_into().unwrap(),
                ExtendedData {
                    custom_data: None,
                    simple_data: simple_storage,
                    inventories: inventories
                        .iter()
                        .map(|(k, v)| Ok((k.clone(), Inventory::from_proto(v.clone(), None)?)))
                        .collect::<Result<hashbrown::HashMap<_, _>>>()?,
                },
            );
        }
    }
    for (i, block_id) in chunk_data.block_ids.iter().enumerate() {
        storage[i].store(*block_id, Ordering::Relaxed);
    }
    Ok(MapChunk {
        coord: coordinate,
        // Unwrap is safe - this should only fail if the length is wrong, but we checked the length above.
        block_ids: storage.clone(),
        extended_data: Box::new(extended_data),
        dirty: false,
    })
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockUpdate {
    pub(crate) location: BlockCoordinate,
    pub(crate) new_value: BlockId,
}

struct MapChunkInnerReadGuard<'a> {
    guard: RwLockReadGuard<'a, HolderState>,
}
impl<'a> Deref for MapChunkInnerReadGuard<'a> {
    type Target = MapChunk;
    fn deref(&self) -> &Self::Target {
        self.guard.unwrap()
    }
}

struct MapChunkInnerWriteGuard<'a> {
    guard: RwLockWriteGuard<'a, HolderState>,
}
impl<'a> MapChunkInnerWriteGuard<'a> {
    fn downgrade(self) -> MapChunkInnerReadGuard<'a> {
        MapChunkInnerReadGuard {
            guard: RwLockWriteGuard::downgrade(self.guard),
        }
    }

    fn clone_block_ids(&self) -> Box<[u32; 4096]> {
        self.block_ids
            .iter()
            .map(|x| x.load(Ordering::Relaxed))
            .collect::<Vec<_>>()
            .into_boxed_slice()
            .try_into()
            .unwrap()
    }
}
impl<'a> Deref for MapChunkInnerWriteGuard<'a> {
    type Target = MapChunk;
    fn deref(&self) -> &Self::Target {
        self.guard.unwrap()
    }
}
impl<'a> DerefMut for MapChunkInnerWriteGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guard.unwrap_mut()
    }
}

enum HolderState {
    Empty,
    Err(Error),
    Ok(MapChunk),
}
impl HolderState {
    fn unwrap_mut(&mut self) -> &mut MapChunk {
        if let HolderState::Ok(x) = self {
            x
        } else {
            panic!("HolderState is not Ok");
        }
    }
    fn unwrap(&self) -> &MapChunk {
        if let HolderState::Ok(x) = self {
            x
        } else {
            panic!("HolderState is not Ok");
        }
    }
}

struct MapChunkHolder {
    chunk: RwLock<HolderState>,
    condition: RwCondvar,
    // TODO - should these go into their own cache line to avoid ping-ponging?
    //
    // At this time, benchmarking shows unclear evidence regarding this.
    // TODO: write a multithreaded benchmark with contention.
    last_accessed: AtomicInstant,
    last_written: AtomicInstant,
    block_bloom_filter: cbloom::Filter,
    // A bit hacky - there are two arcs to the same data - one in the chunk, and one here.
    // The one in MapChunk is used for strong reads/writes under the control of a RwLock.
    // This one here is used for weak reads that don't require any consistency guarantees.
    atomic_storage: Arc<[AtomicU32; 4096]>,
    // This is set only if chunk's state is ready.
    fast_path_read_ready: AtomicBool,
}
impl MapChunkHolder {
    fn new_empty() -> Self {
        // We're OK with allowing this lint; we really do want this atomic copied 4096 times.
        #[allow(clippy::declare_interior_mutable_const)]
        const ATOMIC_INITIALIZER: AtomicU32 = AtomicU32::new(0);
        Self {
            chunk: RwLock::new(HolderState::Empty),
            condition: RwCondvar::new(),
            last_accessed: AtomicInstant::new(),
            last_written: AtomicInstant::new(),
            // TODO: make bloom filter configurable or adaptive
            // For now, 128 bytes is a reasonable overhead (a chunk contains 4096 u32s which is 16 KiB already + extended data)
            // 5 hashers is a reasonable tradeoff - it's not the best false positive rate, but less hashers means cheaper insertion
            block_bloom_filter: cbloom::Filter::with_size_and_hashers(128, 5),
            atomic_storage: Arc::new([ATOMIC_INITIALIZER; 4096]),
            fast_path_read_ready: AtomicBool::new(false),
        }
    }

    /// Get the chunk, blocking until it's loaded
    fn wait_and_get_for_read(&self) -> Result<MapChunkInnerReadGuard<'_>> {
        tokio::task::block_in_place(|| {
            let mut guard = self.chunk.read();
            let _span = span!("wait_and_get");
            loop {
                match &*guard {
                    HolderState::Empty => self.condition.wait_reader(&mut guard),
                    HolderState::Err(e) => {
                        return Err(Error::msg(format!("Chunk load failed: {e:?}")))
                    }
                    HolderState::Ok(_) => return Ok(MapChunkInnerReadGuard { guard }),
                }
            }
        })
    }
    /// Get the chunk, returning None if it's not loaded yet
    /// However, this WILL wait for the mutex (just not for a database load/mapgen)
    fn try_get_read(&self) -> Result<Option<MapChunkInnerReadGuard<'_>>> {
        tokio::task::block_in_place(|| {
            let guard = self.chunk.read();
            match &*guard {
                HolderState::Empty => Ok(None),
                HolderState::Err(e) => Err(Error::msg(format!("Chunk load failed: {e:?}"))),
                HolderState::Ok(_) => Ok(Some(MapChunkInnerReadGuard { guard })),
            }
        })
    }

    /// Get the chunk, blocking until it's loaded
    fn wait_and_get_for_write(&self) -> Result<MapChunkInnerWriteGuard<'_>> {
        tokio::task::block_in_place(|| {
            let mut guard = self.chunk.write();
            let _span = span!("wait_and_get");
            loop {
                match &*guard {
                    HolderState::Empty => self.condition.wait_writer(&mut guard),
                    HolderState::Err(e) => {
                        return Err(Error::msg(format!("Chunk load failed: {e:?}")))
                    }
                    HolderState::Ok(_) => return Ok(MapChunkInnerWriteGuard { guard }),
                }
            }
        })
    }

    /// Get the chunk, returning None if it's not loaded yet
    /// However, this WILL wait for the mutex (just not for a database load/mapgen)
    fn try_get_write(&self) -> Result<Option<MapChunkInnerWriteGuard<'_>>> {
        tokio::task::block_in_place(|| {
            let guard = self.chunk.write();
            match &*guard {
                HolderState::Empty => Ok(None),
                HolderState::Err(e) => Err(Error::msg(format!("Chunk load failed: {e:?}"))),
                HolderState::Ok(_) => Ok(Some(MapChunkInnerWriteGuard { guard })),
            }
        })
    }

    /// Set the chunk, and notify any threads waiting in wait_and_get
    fn fill(
        &self,
        chunk: MapChunk,
        light_columns: &FxHashMap<(i32, i32), ChunkColumn>,
        block_types: &BlockTypeManager,
    ) {
        let mut seen_blocks = FxHashSet::default();
        for block_id in chunk
            .block_ids
            .iter()
            .map(|x| BlockId::from(x.load(Ordering::Relaxed)).base_id())
        {
            if seen_blocks.insert(block_id) {
                // this generates expensive `LOCK OR %rax(%r8,%rdx,8) as well as an expensive DIV
                // on x86_64.
                // Only insert unique block ids (still need to test this optimization)
                // TODO fork the bloom filter library and extend it to support constant lengths
                self.block_bloom_filter.insert(block_id as u64);
            }
        }
        let _span = span!("game_map waiting to fill chunk");
        let mut guard = self.chunk.write();
        assert!(matches!(*guard, HolderState::Empty));

        self.fill_lighting_for_load(&chunk, light_columns, block_types);
        *guard = HolderState::Ok(chunk);
        self.fast_path_read_ready.store(true, Ordering::Release);
        self.condition.notify_all();
    }

    fn update_lighting_after_edit(
        &self,
        outer_guard: &MapChunkOuterGuard,
        chunk: &MapChunk,
        offset: ChunkOffset,
        block_types: &BlockTypeManager,
    ) {
        let light_column = outer_guard
            .read_guard
            .light_columns
            .get(&(chunk.coord.x, chunk.coord.z))
            .unwrap();
        let mut occluded = false;
        for y in 0..16 {
            let id = chunk.get_block(ChunkOffset {
                x: offset.x,
                y,
                z: offset.z,
            });
            if !block_types.allows_light_propagation(id) {
                occluded = true;
                break;
            }
        }

        let mut cursor = light_column.cursor_into(chunk.coord.y);
        assert!(cursor.current_valid());
        cursor
            .current_occlusion_mut()
            .set(offset.x, offset.z, occluded);
        cursor.propagate_lighting();
    }

    fn fill_lighting_for_load(
        &self,
        chunk: &MapChunk,
        light_columns: &FxHashMap<(i32, i32), ChunkColumn>,
        block_types: &BlockTypeManager,
    ) {
        // Unwrap is ok - the light column should exist by the time the chunk was inserted.
        let light_column = light_columns.get(&(chunk.coord.x, chunk.coord.z)).unwrap();

        let mut occlusion = Lightfield::zero();
        for x in 0..16 {
            for z in 0..16 {
                'inner: for y in 0..16 {
                    let id = chunk.get_block(ChunkOffset { x, y, z });
                    if !block_types.allows_light_propagation(id) {
                        occlusion.set(x, z, true);
                        break 'inner;
                    }
                }
            }
        }

        let mut cursor = light_column.cursor_into(chunk.coord.y);
        *cursor.current_occlusion_mut() = occlusion;
        // We need to set this before the loop, since the cursor will advance. Technically, the
        // lighting state isn't valid until the incoming light is updated, but we will do so before
        // releasing any locks.
        cursor.mark_valid();
        cursor.propagate_lighting();
    }

    /// Set the chunk to an error, and notify any waiting threads so they can propagate the error
    fn set_err(&self, err: Error) {
        let mut guard = self.chunk.write();
        assert!(matches!(*guard, HolderState::Empty));
        *guard = HolderState::Err(err);
        self.condition.notify_all();
    }

    fn bump_access_time(&self) {
        self.last_accessed.update_now_release();
    }
}

struct MapChunkOuterGuard<'a> {
    read_guard: RwLockReadGuard<'a, MapShard>,
    coord: ChunkCoordinate,
    writeback_permit: Option<mpsc::Permit<'a, WritebackReq>>,
    force_writeback: bool,
}
impl<'a> MapChunkOuterGuard<'a> {
    // Actually submits a writeback request to the writeback queue.
    // Naming note - _inner because ServerGameMap first does some error checking and
    // monitoring/stats before calling this.
    fn write_back_inner(mut self) {
        self.last_written.update_now_release();
        self.writeback_permit
            .take()
            .unwrap()
            .send(WritebackReq::Chunk(self.coord));
    }
}
impl<'a> Drop for MapChunkOuterGuard<'a> {
    fn drop(&mut self) {
        if self.force_writeback {
            // If the permit is still there, we didn't write this chunk back.
            // Do so now.
            // If the permit was already taken, we don't need to do anything.
            if let Some(permit) = self.writeback_permit.take() {
                self.last_written.update_now_release();
                permit.send(WritebackReq::Chunk(self.coord));
            }
        }
    }
}
impl<'a> Deref for MapChunkOuterGuard<'a> {
    type Target = MapChunkHolder;
    fn deref(&self) -> &Self::Target {
        // This unwrap should always succeed - get_chunk and friends ensure that the chunk is inserted
        // before MapChunkOuterGuard is created
        self.read_guard.chunks.get(&self.coord).as_ref().unwrap()
    }
}

const NUM_CHUNK_SHARDS: usize = 16;
// todo scale up
const NUM_ENTITY_SHARDS: usize = 1;

fn shard_id(coord: ChunkCoordinate) -> usize {
    (coord.coarse_hash_no_y() % NUM_CHUNK_SHARDS as u64) as usize
}

struct MapShard {
    chunks: FxHashMap<ChunkCoordinate, MapChunkHolder>,
    light_columns: FxHashMap<(i32, i32), ChunkColumn>,
}
impl MapShard {
    fn new(_shard: usize) -> MapShard {
        MapShard {
            chunks: FxHashMap::default(),
            light_columns: FxHashMap::default(),
        }
    }
}

/// Represents the entire map of the world.
/// This struct provides safe interior mutability - a shared reference
/// is sufficient to read and write it. Locking is done at an implementation-defined
/// granularity.
pub struct ServerGameMap {
    game_state: Weak<GameState>,
    database: Arc<dyn GameDatabase>,
    // sharded 16 ways based on the coarse hash of the chunk
    // Each one is 72 bytes, so we pad them out to a whole cacheline. This wastes a bit of space, unfortunately
    // However, it means that one shard's lock control word isn't aliasing with the data of other shards.
    live_chunks: [CachelineAligned<RwLock<MapShard>>; NUM_CHUNK_SHARDS],
    block_type_manager: Arc<BlockTypeManager>,
    block_update_sender: broadcast::Sender<BlockUpdate>,
    writeback_senders: [mpsc::Sender<WritebackReq>; NUM_CHUNK_SHARDS],
    shutdown: CancellationToken,
    writeback_handles: [Mutex<Option<JoinHandle<Result<()>>>>; NUM_CHUNK_SHARDS],
    cleanup_handles: [Mutex<Option<JoinHandle<Result<()>>>>; NUM_CHUNK_SHARDS],
    timer_controller: Mutex<Option<TimerController>>,
}
impl ServerGameMap {
    pub(crate) fn new(
        game_state: Weak<GameState>,
        database: Arc<dyn GameDatabase>,
        block_type_manager: Arc<BlockTypeManager>,
    ) -> Result<Arc<ServerGameMap>> {
        let (block_update_sender, _) = broadcast::channel(BROADCAST_CHANNEL_SIZE);
        let mut writeback_senders = vec![];
        let mut writeback_receivers = vec![];
        for _ in 0..NUM_CHUNK_SHARDS {
            let (sender, receiver) = mpsc::channel(WRITEBACK_QUEUE_SIZE);
            writeback_senders.push(sender);
            writeback_receivers.push(receiver);
        }

        let cancellation = CancellationToken::new();

        let result = Arc::new(ServerGameMap {
            game_state: game_state.clone(),
            database: database.clone(),
            live_chunks: std::array::from_fn(|shard| {
                CachelineAligned(RwLock::new(MapShard::new(shard)))
            }),
            block_type_manager,
            block_update_sender,
            writeback_senders: writeback_senders.try_into().unwrap(),
            shutdown: cancellation.clone(),
            writeback_handles: std::array::from_fn(|_| Mutex::new(None)),
            cleanup_handles: std::array::from_fn(|_| Mutex::new(None)),
            timer_controller: None.into(),
        });
        for (i, receiver) in writeback_receivers.into_iter().enumerate() {
            let game_state_clone = game_state.clone();
            let mut writeback = GameMapWriteback {
                map: result.clone(),
                receiver,
                cancellation: cancellation.clone(),
                shard_id: i,
            };
            let writeback_handle = crate::spawn_async(
                &format!("map_writeback_{}", i),
                async move {
                    let result = writeback.run_loop().await;
                    match result {
                        Ok(()) => {
                            if !writeback.cancellation.is_cancelled() {
                                tracing::error!(
                                    "Map writeback for shard {} exited while map wasn't shut down.",
                                    i
                                );
                            }
                            Ok(())
                        }
                        Err(e) => {
                            tracing::error!(
                                "Map writeback for shard {} exited with an error: {:?}",
                                i,
                                e
                            );
                            if let Some(gs) = game_state_clone.upgrade() {
                                gs.crash_shutdown(e);
                            } else {
                                tracing::error!("Game state is gone, cannot start crash shutdown. Did it shut down before us?")
                            }
                            bail!("Crash initiated");
                        }
                    }
                },
            )?;
            *result.writeback_handles[i].lock() = Some(writeback_handle);

            let mut cache_cleanup = MapCacheCleanup {
                map: result.clone(),
                cancellation: cancellation.clone(),
                shard_id: i,
            };

            let game_state_clone = game_state.clone();
            let cleanup_handle = crate::spawn_async(&format!("map_cleanup_{}", i), async move {
                let result = cache_cleanup.run_loop().await;
                match result {
                    Ok(()) => {
                        if !cache_cleanup.cancellation.is_cancelled() {
                            tracing::error!(
                                "Map cache cleanup for shard {} exited while map wasn't shut down.",
                                i
                            );
                        }
                        Ok(())
                    }
                    Err(e) => {
                        tracing::error!(
                            "Map cache cleanup for shard {} exited with an error: {:?}",
                            i,
                            e
                        );
                        if let Some(gs) = game_state_clone.upgrade() {
                            gs.crash_shutdown(e);
                        } else {
                            tracing::error!("Game state is gone, cannot start crash shutdown. Did it shut down before us?")
                        }
                        bail!("Crash initiated");
                    }
                }
            })?;
            *result.cleanup_handles[i].lock() = Some(cleanup_handle);
        }

        let timer_controller = TimerController {
            map: result.clone(),
            game_state,
            // This needs to be separate from the cancellation token above, since we need the
            // timer controller to shut down before the rest of the map (since the timer controller
            // relies on the writeback thread to actually write back)
            cancellation: CancellationToken::new(),
            timers: FxHashMap::default(),
        };
        *result.timer_controller.lock() = Some(timer_controller);

        Ok(result)
    }

    pub(crate) fn bump_chunk(&self, coord: ChunkCoordinate) -> bool {
        let _span = span!("bump_access_time");
        if let Some(chunk) = self.live_chunks[shard_id(coord)].read().chunks.get(&coord) {
            chunk.bump_access_time();
            true
        } else {
            false
        }
    }

    /// Gets a block from the map + its variant + its extended data.
    ///
    /// Because the extended data is not necessarily Clone or Copy, it is provided by calling the provided
    /// callback; if it matches your desired type (using Any::downcast_ref or similar), you can retrieve/copy
    /// any relevant data out.
    ///
    /// Warning: If your extended data includes interior mutability (mutex, refcell, etc), it is important
    /// to note that this function does NOT set the dirty bit on the chunk, and any changes made via that
    /// interior mutability may be lost.
    pub fn get_block_with_extended_data<F, T>(
        &self,
        coord: BlockCoordinate,
        extended_data_callback: F,
    ) -> Result<(BlockId, Option<T>)>
    where
        F: FnOnce(&ExtendedData) -> Result<Option<T>>,
    {
        let chunk_guard = self.get_chunk(coord.chunk())?;
        let chunk = chunk_guard.wait_and_get_for_read()?;

        let id = chunk.block_ids[coord.offset().as_index()].load(Ordering::Relaxed);
        let ext_data = match chunk
            .extended_data
            .get(&coord.offset().as_index().try_into().unwrap())
        {
            Some(x) => extended_data_callback(x)?,
            None => None,
        };

        Ok((id.into(), ext_data))
    }

    fn get_block_with_extended_data_no_load<F, T>(
        &self,
        coord: BlockCoordinate,
        extended_data_callback: F,
    ) -> Result<Option<(BlockId, Option<T>)>>
    where
        F: FnOnce(&ExtendedData) -> Option<T>,
    {
        let chunk_guard = match self.try_get_chunk(coord.chunk(), false) {
            Some(x) => x,
            None => return Ok(None),
        };
        let chunk = match chunk_guard.try_get_read()? {
            Some(x) => x,
            None => return Ok(None),
        };

        let id = chunk.block_ids[coord.offset().as_index()].load(Ordering::Relaxed);
        let ext_data = match chunk
            .extended_data
            .get(&coord.offset().as_index().try_into().unwrap())
        {
            Some(x) => extended_data_callback(x),
            None => None,
        };

        Ok(Some((BlockId(id), ext_data)))
    }

    /// Gets a block + variant without its extended data. This will perform a data load if the chunk
    /// is not loaded
    pub fn get_block(&self, coord: BlockCoordinate) -> Result<BlockId> {
        tokio::task::block_in_place(|| {
            let chunk_guard = self.get_chunk(coord.chunk())?;
            let chunk = chunk_guard.wait_and_get_for_read()?;

            let id = chunk.block_ids[coord.offset().as_index()].load(Ordering::Relaxed);
            Ok(id.into())
        })
    }

    /// Attempts to get a block without its extended data. Will not attempt to load blocks on cache misses
    ///
    /// This avoids taking write locks or doing expensive IO, at the expense of sometimes not being able to
    /// actually get the block
    ///
    /// However, it will still wait to get a read lock for the chunk map itself. The only circumstances
    /// where this should fail is if the chunk is unloaded.
    pub fn try_get_block(&self, coord: BlockCoordinate) -> Option<BlockId> {
        let chunk_guard = self.try_get_chunk(coord.chunk(), false)?;
        // We don't have a mapchunk lock, so we need actual atomic ordering here
        if chunk_guard.fast_path_read_ready.load(Ordering::Acquire) {
            Some(
                chunk_guard.atomic_storage[coord.offset().as_index()]
                    .load(Ordering::Acquire)
                    .into(),
            )
        } else {
            None
        }
    }

    /// Sets a block on the map. No handlers are run, and the block is updated unconditionally.
    /// The old block is returned along with its extended data, if any.
    pub fn set_block<T: TryAsHandle>(
        &self,
        coord: BlockCoordinate,
        block: T,
        new_data: Option<ExtendedData>,
    ) -> Result<(BlockId, Option<ExtendedData>)> {
        let new_id = block
            .as_handle(&self.block_type_manager)
            .with_context(|| "Block not found")?;

        let chunk_guard = self.get_chunk(coord.chunk())?;
        let mut chunk = chunk_guard.wait_and_get_for_write()?;

        let old_id = chunk.block_ids[coord.offset().as_index()].load(Ordering::Relaxed);
        let old_block = old_id.into();
        let old_data = match new_data {
            Some(new_data) => chunk
                .extended_data
                .insert(coord.offset().as_index().try_into().unwrap(), new_data),
            None => chunk
                .extended_data
                .remove(&coord.offset().as_index().try_into().unwrap()),
        };
        chunk.block_ids[coord.offset().as_index()].store(new_id.into(), Ordering::Relaxed);

        chunk.dirty = true;
        let light_change = self
            .block_type_manager()
            .allows_light_propagation(old_id.into())
            ^ self.block_type_manager().allows_light_propagation(new_id);
        if light_change {
            chunk_guard.update_lighting_after_edit(
                &chunk_guard,
                &chunk,
                coord.offset(),
                self.block_type_manager(),
            );
        }
        drop(chunk);
        chunk_guard
            .block_bloom_filter
            .insert(new_id.base_id() as u64);

        self.enqueue_writeback(chunk_guard)?;
        self.broadcast_block_change(BlockUpdate {
            location: coord,
            new_value: new_id,
        });
        Ok((old_block, old_data))
    }

    /// Sets a block on the map. No handlers are run, and the block is updated unconditionally.
    /// The old block is returned along with its extended data, if any.
    pub fn compare_and_set_block<T: TryAsHandle, U: TryAsHandle>(
        &self,
        coord: BlockCoordinate,
        expected: T,
        block: U,
        extended_data: Option<ExtendedData>,
        check_variant: bool,
    ) -> Result<(CasOutcome, BlockId, Option<ExtendedData>)> {
        if check_variant {
            self.compare_and_set_block_predicate(
                coord,
                |x, _, _| {
                    Ok(x.as_handle(&self.block_type_manager)
                        .zip(expected.as_handle(&self.block_type_manager))
                        .map_or(false, |(x, y)| x == y))
                },
                block,
                extended_data,
            )
        } else {
            self.compare_and_set_block_predicate(
                coord,
                |x, _, _| {
                    Ok(x.as_handle(&self.block_type_manager)
                        .zip(expected.as_handle(&self.block_type_manager))
                        .map_or(false, |(x, y)| x.equals_ignore_variant(y)))
                },
                block,
                extended_data,
            )
        }
    }

    /// Sets a block on the map. No handlers are run, and the block is updated unconditionally.
    /// The old block is returned along with its extended data, if any.
    pub fn compare_and_set_block_predicate<F, T: TryAsHandle>(
        &self,
        coord: BlockCoordinate,
        predicate: F,
        block: T,
        new_extended_data: Option<ExtendedData>,
    ) -> Result<(CasOutcome, BlockId, Option<ExtendedData>)>
    where
        F: FnOnce(BlockId, Option<&ExtendedData>, &BlockTypeManager) -> Result<bool>,
    {
        let new_id = block
            .as_handle(&self.block_type_manager)
            .with_context(|| "Block not found")?;

        let chunk_guard = self.get_chunk(coord.chunk())?;
        let mut chunk = chunk_guard.wait_and_get_for_write()?;

        let old_id = BlockId(chunk.block_ids[coord.offset().as_index()].load(Ordering::Relaxed));
        let old_block = old_id.into();
        if !predicate(
            old_block,
            chunk
                .extended_data
                .get(&coord.offset().as_index().try_into().unwrap()),
            self.block_type_manager(),
        )? {
            return Ok((CasOutcome::Mismatch, old_block, None));
        }
        let new_data_was_some = new_extended_data.is_some();
        let old_data = match new_extended_data {
            Some(new_data) => chunk
                .extended_data
                .insert(coord.offset().as_index().try_into().unwrap(), new_data),
            None => chunk
                .extended_data
                .remove(&coord.offset().as_index().try_into().unwrap()),
        };
        chunk.block_ids[coord.offset().as_index()].store(new_id.into(), Ordering::Relaxed);
        if old_id != new_id || old_data.is_some() || new_data_was_some {
            chunk.dirty = true;
        }

        let light_change = self.block_type_manager().allows_light_propagation(old_id)
            ^ self.block_type_manager().allows_light_propagation(new_id);
        if light_change {
            chunk_guard.update_lighting_after_edit(
                &chunk_guard,
                &chunk,
                coord.offset(),
                self.block_type_manager(),
            );
        }
        drop(chunk);
        chunk_guard
            .block_bloom_filter
            .insert(new_id.base_id() as u64);
        self.enqueue_writeback(chunk_guard)?;
        self.broadcast_block_change(BlockUpdate {
            location: coord,
            new_value: new_id,
        });
        Ok((CasOutcome::Match, old_block, old_data))
    }

    /// Runs the given mutator on the block and its extended data.
    /// The function may mutate the data, or leave it as-is, and it may return a value to the caller
    /// through its own return value.
    ///
    /// Notes:
    /// * If the mutator changes the block type, it should set the extended data to
    /// None, or to an extended data object that the new block type can handle.
    /// * If the mutator returns a non-Ok status, any changes it made will still be applied.
    ///
    /// It is not safe to call other GameMap functions (e.g. get/set blocks) from the handler - they may deadlock.
    ///
    /// Warning: If the mutator panics, the extended data may be lost.
    pub fn mutate_block_atomically<F, T>(&self, coord: BlockCoordinate, mutator: F) -> Result<T>
    where
        F: FnOnce(&mut BlockId, &mut ExtendedDataHolder) -> Result<T>,
    {
        tokio::task::block_in_place(|| {
            let chunk_guard = self.get_chunk(coord.chunk())?;
            let mut chunk = chunk_guard.wait_and_get_for_write()?;

            let (result, block_changed) = self.mutate_block_atomically_locked(
                &chunk_guard,
                &mut chunk,
                coord.offset(),
                mutator,
                self,
            )?;
            if block_changed {
                // mutate_block_atomically_locked already sent a broadcast.
                chunk_guard.update_lighting_after_edit(
                    &chunk_guard,
                    &chunk,
                    coord.offset(),
                    self.block_type_manager(),
                );
                drop(chunk);
                self.enqueue_writeback(chunk_guard)?;
            }
            Ok(result)
        })
    }

    /// Same as [mutate_block_atomically], but returns None if it cannot immediately run the mutator without
    /// having to block (for a mutex, writeback permit, or IO).
    ///
    /// Args:
    ///   `coord` - the block coordinate
    ///   `mutator` - the function to run. Same signature and warnings apply as
    ///               `mutate_block_atomically`
    ///   `wait_for_inner` - If true, we're willing to wait for the inner mutex; this is useful if
    ///                      the caller is OK with blocking but doesn't want to load unloaded chunks
    pub fn try_mutate_block_atomically<F, T>(
        &self,
        coord: BlockCoordinate,
        mutator: F,
        wait_for_inner: bool,
    ) -> Result<Option<T>>
    where
        F: FnOnce(&mut BlockId, &mut ExtendedDataHolder) -> Result<T>,
    {
        let chunk_guard = match self.try_get_chunk(coord.chunk(), true) {
            Some(x) => x,
            None => return Ok(None),
        };
        let mut chunk = if wait_for_inner {
            chunk_guard.wait_and_get_for_write()?
        } else {
            match chunk_guard.try_get_write()? {
                Some(x) => x,
                None => return Ok(None),
            }
        };

        let (result, block_changed) = self.mutate_block_atomically_locked(
            &chunk_guard,
            &mut chunk,
            coord.offset(),
            mutator,
            self,
        )?;
        if block_changed {
            // mutate_block_atomically_locked already sent a broadcast.
            chunk_guard.update_lighting_after_edit(
                &chunk_guard,
                &chunk,
                coord.offset(),
                self.block_type_manager(),
            );
            drop(chunk);
            self.enqueue_writeback(chunk_guard)?;
        }
        Ok(Some(result))
    }

    /// Internal impl detail of mutate_block_atomically and timers
    fn mutate_block_atomically_locked<F, T>(
        &self,
        holder: &MapChunkHolder,
        chunk: &mut MapChunkInnerWriteGuard<'_>,
        offset: ChunkOffset,
        mutator: F,
        game_map: &ServerGameMap,
    ) -> Result<(T, bool)>
    where
        F: FnOnce(&mut BlockId, &mut ExtendedDataHolder) -> Result<T>,
    {
        let mut extended_data = chunk
            .extended_data
            .remove(&offset.as_index().try_into().unwrap());

        let mut data_holder = ExtendedDataHolder::new(&mut extended_data);

        let old_id = chunk.block_ids[offset.as_index()]
            .load(Ordering::Relaxed)
            .into();
        let mut new_id = old_id;
        let closure_result = mutator(&mut new_id, &mut data_holder);

        // This would have been a nice optimization, but I can't even consistently be sure to set
        // the dirty bit in my own code. Do this for now, until there's a better design.
        let extended_data_dirty = true; //data_holder.dirty();
        if let Some(new_data) = extended_data {
            chunk
                .extended_data
                .insert(offset.as_index().try_into().unwrap(), new_data);
        }

        if new_id != old_id {
            chunk.block_ids[offset.as_index()].store(new_id.into(), Ordering::Relaxed);
            chunk.dirty = true;
            holder.block_bloom_filter.insert(new_id.base_id() as u64)
        }
        if extended_data_dirty {
            chunk.dirty = true;
            // data holder is dirty, so inventories might have been updated
            self.game_state
                .upgrade()
                .unwrap()
                .inventory_manager()
                .broadcast_block_update(chunk.coord.with_offset(offset));
        }
        if new_id != old_id {
            game_map.broadcast_block_change(BlockUpdate {
                location: chunk.coord.with_offset(offset),
                new_value: new_id,
            });
        }

        Ok((closure_result?, new_id != old_id))
    }

    /// Digs a block, running its on-dig event handler. The items it drops are returned.
    ///
    /// This does not check whether the tool is able to dig the block.
    pub fn dig_block(
        &self,
        coord: BlockCoordinate,
        initiator: &EventInitiator,
        tool: Option<&ItemStack>,
    ) -> Result<BlockInteractionResult> {
        self.run_block_interaction(
            coord,
            initiator,
            tool,
            |block| block.dig_handler_inline.as_deref(),
            |block| block.dig_handler_full.as_deref(),
        )
    }

    /// Taps a block, as if it were hit without being fully dug. The items it drops are returned.
    pub fn tap_block(
        &self,
        coord: BlockCoordinate,
        initiator: &EventInitiator,
        tool: Option<&ItemStack>,
    ) -> Result<BlockInteractionResult> {
        self.run_block_interaction(
            coord,
            initiator,
            tool,
            |block| block.tap_handler_inline.as_deref(),
            |block| block.tap_handler_full.as_deref(),
        )
    }

    /// Runs the given dig or interact handler for the block at the specified coordinate.
    ///
    /// This method will retrieve the block at the provided coordinate, look up its dig or interact
    /// handler based on the provided callbacks, and invoke it with the given context.
    ///
    /// Any items dropped by the handler will be returned. Note that if a dig handler is provided,
    /// it is assumed the block's dig handler has already been run separately.
    ///
    /// # Arguments
    ///
    /// * `coord` - The block coordinate to run the handler for
    /// * `initiator` - The initiator of this event, for handler context  
    /// * `tool` - Optional tool item stack, if this is a tool interaction
    /// * `get_block_inline_handler` - Callback to retrieve the desired inline handler
    /// * `get_block_full_handler` - Callback to retrieve the desired full handler
    ///
    /// # Returns
    ///
    /// A vector containing any item stacks dropped by the invoked handlers.
    ///
    /// # Errors
    ///
    /// Returns any error encountered while retrieving the block or invoking handlers.
    pub fn run_block_interaction<F, G>(
        &self,
        coord: BlockCoordinate,
        initiator: &EventInitiator,
        tool: Option<&ItemStack>,
        get_block_inline_handler: F,
        get_block_full_handler: G,
    ) -> Result<BlockInteractionResult>
    where
        F: Fn(&blocks::BlockType) -> Option<&blocks::InlineHandler>,
        G: Fn(&blocks::BlockType) -> Option<&blocks::FullHandler>,
    {
        let game_state = self.game_state();
        let tick = game_state.tick();
        let (blocktype, mut result) = self.mutate_block_atomically(coord, |block, ext_data| {
            let (blocktype, _) = self.block_type_manager().get_block(block)?;

            if let Some(ref inline_handler) = get_block_inline_handler(blocktype) {
                let ctx = InlineContext {
                    tick,
                    initiator: initiator.clone(),
                    location: coord,
                    block_types: self.block_type_manager(),
                    items: game_state.item_manager(),
                };
                let result = run_handler!(
                    || (inline_handler)(ctx, block, ext_data, tool),
                    "block_inline",
                    initiator,
                )?;
                Ok((blocktype, result))
            } else {
                Ok((blocktype, Default::default()))
            }
        })?;

        // possible future optimization: If we don't have an inline handler, don't bother locking, just use try_get_block
        // we need this to happen outside of mutate_block_atomically (which holds a chunk lock) to avoid a deadlock.
        if let Some(full_handler) = get_block_full_handler(blocktype) {
            let ctx = HandlerContext {
                tick,
                initiator: initiator.clone(),
                game_state: self.game_state(),
            };
            result += run_handler!(
                || (full_handler)(&ctx, coord, tool),
                "block_full",
                initiator,
            )?;
        }

        Ok(result)
    }

    pub fn make_inline_context<'a>(
        &'a self,
        coord: BlockCoordinate,
        initiator: &EventInitiator<'a>,
        game_state: &'a GameState,
        tick: u64,
    ) -> InlineContext<'a> {
        InlineContext {
            tick,
            initiator: initiator.clone(),
            location: coord,
            block_types: self.block_type_manager(),
            items: game_state.item_manager(),
        }
    }

    pub(crate) fn block_type_manager(&self) -> &BlockTypeManager {
        &self.block_type_manager
    }

    // Gets a chunk, loading it from database/generating it if it is not in memory
    #[tracing::instrument(level = "trace", name = "get_chunk", skip(self))]
    fn get_chunk(&self, coord: ChunkCoordinate) -> Result<MapChunkOuterGuard> {
        tokio::task::block_in_place(|| {
            log_trace("get_chunk starting");
            let writeback_permit = self.get_writeback_permit(shard_id(coord))?;
            log_trace("get_chunk acquired writeback permit");
            let shard = shard_id(coord);
            let mut load_chunk_tries = 0;
            let result = loop {
                let read_guard = {
                    let _span = span!("acquire game_map read lock");
                    self.live_chunks[shard].read()
                };
                log_trace("get_chunk acquired read lock");
                // All good. The chunk is loaded.
                if read_guard.chunks.contains_key(&coord) {
                    log_trace("get_chunk chunk loaded");
                    return Ok(MapChunkOuterGuard {
                        read_guard,
                        coord,
                        writeback_permit: Some(writeback_permit),
                        force_writeback: false,
                    });
                }

                load_chunk_tries += 1;
                drop(read_guard);
                log_trace("get_chunk read lock released");
                // The chunk is not loaded. Give up the lock, get a write lock, get an entry into the map, and then fill it
                // under a read lock
                // This can't be done with an upgradable lock due to a deadlock risk.
                let mut write_guard = {
                    let _span = span!("acquire game_map write lock");
                    self.live_chunks[shard].write()
                };
                log_trace("get_chunk acquired write lock");
                if write_guard.chunks.contains_key(&coord) {
                    // Someone raced with us. Try looping again.
                    info!("Race while upgrading in get_chunk; retrying two-phase lock");
                    log_trace("get_chunk race detected");
                    drop(write_guard);
                    continue;
                }

                // We still hold the write lock. Insert, downgrade back to a read lock, and fill the chunk before returning.
                // Since we still hold the write lock, our check just above is still correct, and we can safely insert.
                write_guard
                    .chunks
                    .insert(coord, MapChunkHolder::new_empty());
                write_guard
                    .light_columns
                    .entry((coord.x, coord.z))
                    .or_insert_with(ChunkColumn::empty)
                    .insert_empty(coord.y);

                // Now we downgrade the write lock.
                // If another thread races ahead of us and does the same lookup before we manage to fill the chunk,
                // they'll get an empty chunk holder and will wait for the condition variable to be signalled
                // (when that thread waits on the condition variable, it atomically releases the inner lock)
                let read_guard = RwLockWriteGuard::downgrade(write_guard);
                log_trace("get_chunk downgraded write lock");
                // We had a write lock and downgraded it atomically. No other thread could have removed the entry.
                let chunk_holder = read_guard.chunks.get(&coord).unwrap();
                match self
                    .load_uncached_or_generate_chunk(coord, chunk_holder.atomic_storage.clone())
                {
                    Ok((chunk, force_writeback)) => {
                        log_trace("get_chunk chunk loaded, filling");
                        chunk_holder.fill(
                            chunk,
                            &read_guard.light_columns,
                            self.block_type_manager(),
                        );
                        log_trace("get_chunk chunk filled");
                        let outer_guard = MapChunkOuterGuard {
                            read_guard,
                            coord,
                            writeback_permit: Some(writeback_permit),
                            force_writeback,
                        };
                        break Ok(outer_guard);
                    }
                    Err(e) => {
                        chunk_holder
                            .set_err(Error::msg(format!("Chunk load/generate failed: {e:?}")));
                        // Unfortunate duplication, anyhow::Error is not Clone
                        break Err(Error::msg(format!("Chunk load/generate failed: {e:?}")));
                    }
                }
            };
            if load_chunk_tries > 1 {
                warn!("Took {load_chunk_tries} tries to load {coord:?}");
            }
            result
        })
    }

    #[tracing::instrument(level = "trace", name = "try_get_chunk", skip(self))]
    fn try_get_chunk(
        &self,
        coord: ChunkCoordinate,
        want_permit: bool,
    ) -> Option<MapChunkOuterGuard> {
        let shard = shard_id(coord);
        let mut permit = None;
        if want_permit {
            match self.try_get_writeback_permit(shard) {
                Ok(Some(p)) => permit = Some(p),
                Ok(None) => return None,
                Err(e) => {
                    tracing::error!("Failed to get writeback permit: {:?}", e);
                    return None;
                }
            }
        }
        let guard = self.live_chunks[shard].read();
        // We need this check - if the chunk isn't in memory, we cannot construct a MapChunkOuterGuard for it
        // (unwrapping will panic)
        //
        // As long as the guard lives, nobody can remove the entry from the map
        match guard.chunks.get(&coord) {
            None => return None,
            Some(holder) => {
                holder.last_accessed.update_now_relaxed();
            }
        }
        return Some(MapChunkOuterGuard {
            read_guard: guard,
            coord,
            writeback_permit: permit,
            force_writeback: false,
        });
    }

    fn get_writeback_permit(&self, shard: usize) -> Result<mpsc::Permit<'_, WritebackReq>> {
        let _span = span!("get writeback permit");
        Ok(tokio::runtime::Handle::current().block_on(self.writeback_senders[shard].reserve())?)
    }

    fn try_get_writeback_permit(
        &self,
        shard: usize,
    ) -> Result<Option<mpsc::Permit<'_, WritebackReq>>> {
        match self.writeback_senders[shard].try_reserve() {
            Ok(permit) => Ok(Some(permit)),
            Err(mpsc::error::TrySendError::Full(_)) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    // Loads the chunk from the DB, or generates it if missing, REGARDLESS of whether the chunk
    // is in memory.
    // If the chunk is already in memory, this may cause data loss by reading a stale instance
    // from the DB/mapgen
    // Returns the chunk and a boolean - if true, the chunk was generated and should be written
    // back unconditionally.
    fn load_uncached_or_generate_chunk(
        &self,
        coord: ChunkCoordinate,
        storage: Arc<[AtomicU32; 4096]>,
    ) -> Result<(MapChunk, bool)> {
        let data = self
            .database
            .get(&KeySpace::MapchunkData.make_key(&coord.as_bytes()))?;
        if let Some(data) = data {
            return Ok((
                MapChunk::deserialize(coord, &data, self.game_state(), storage)?,
                false,
            ));
        }

        let mut chunk = MapChunk::new(coord, storage);
        chunk.dirty = true;
        {
            let _span = span!("mapgen running");
            run_handler!(
                || {
                    self.game_state().mapgen().fill_chunk(coord, &mut chunk);
                    Ok(())
                },
                "mapgen",
                &EventInitiator::Engine
            )?;
        }
        Ok((chunk, true))
    }

    pub fn game_state(&self) -> Arc<GameState> {
        Weak::upgrade(&self.game_state).unwrap()
    }

    fn unload_chunk_locked(
        &self,
        lock: &mut RwLockWriteGuard<MapShard>,
        coord: ChunkCoordinate,
    ) -> Result<()> {
        let _span = span!("unload single chunk");
        let chunk = lock.chunks.remove(&coord);
        if let Some(chunk) = chunk {
            // The read/write type doesn't matter here. We already have a write lock on the mapshard.
            // If this is contended, something else has already gone terribly wrong.
            match chunk.chunk.into_inner() {
                HolderState::Empty => {
                    panic!("chunk unload got a chunk with an empty holder. This should never happen - please file a bug");
                }
                HolderState::Err(e) => {
                    warn!("chunk unload trying to unload a chunk with an error: {e:?}");
                }
                HolderState::Ok(chunk) => {
                    if chunk.dirty {
                        self.database.put(
                            &KeySpace::MapchunkData.make_key(&coord.as_bytes()),
                            &chunk
                                .serialize(ChunkUsage::Server, &self.game_state())?
                                .encode_to_vec(),
                        )?;
                    }
                    let light_column = lock.light_columns.get_mut(&(coord.x, coord.z)).unwrap();
                    light_column.remove(coord.y);
                    if light_column.is_empty() {
                        lock.light_columns.remove(&(coord.x, coord.z));
                    }

                    // TODO(lighting) caller should recalc lighting from just above the chunk that was removed
                    // (this should be batched)
                }
            }
        }
        Ok(())
    }

    // Broadcasts when a block on the map changes
    fn broadcast_block_change(&self, update: BlockUpdate) {
        // The only error we expect is that there are no receivers. This is fine; we might
        // be running a timer for a block that's still loaded after all players log out
        let _ = self.block_update_sender.send(update);
    }
    /// Enqueues this chunk to be written back to the database.
    fn enqueue_writeback(&self, chunk: MapChunkOuterGuard<'_>) -> Result<()> {
        if self.shutdown.is_cancelled() {
            return Err(Error::msg("Writeback thread was shut down"));
        }
        plot!(
            "writeback queue capacity",
            self.writeback_senders[shard_id(chunk.coord)].capacity() as f64
        );
        chunk.write_back_inner();
        Ok(())
    }

    /// Create a receiver that is notified of changes to all block IDs (including variant changes).
    /// This receiver will not obtain messages for changes to extended data.
    pub(crate) fn subscribe(&self) -> broadcast::Receiver<BlockUpdate> {
        self.block_update_sender.subscribe()
    }

    /// Get a chunk from the map and return its client proto if either load_if_missing is true, or
    /// the chunk is already cached in memory. If this loads a chunk, the chunk will stay in memory.
    ///
    /// This is only made visible for benchmarking, and otherwise isn't useful or stable for most use-cases.
    ///
    /// mark_action is called with the chunk already loaded, before its lock is taken. It should
    /// still be fast, as it holds the read lock on the map shard.
    #[doc(hidden)]
    pub fn serialize_for_client(
        &self,
        coord: ChunkCoordinate,
        load_if_missing: bool,
        mark_action: impl FnOnce(),
    ) -> Result<Option<mapchunk_proto::StoredChunk>> {
        if load_if_missing {
            (mark_action)();
            let chunk_guard = self.get_chunk(coord)?;
            let chunk = chunk_guard.wait_and_get_for_read()?;
            Ok(Some(
                chunk.serialize(ChunkUsage::Client, &self.game_state())?,
            ))
        } else {
            let chunk_guard = self.try_get_chunk(coord, false);
            if let Some(chunk) = chunk_guard {
                (mark_action)();
                let chunk = chunk.wait_and_get_for_read()?;
                Ok(Some(
                    chunk.serialize(ChunkUsage::Client, &self.game_state())?,
                ))
            } else {
                Ok(None)
            }
        }
    }

    pub(crate) async fn do_shutdown(&self) -> Result<()> {
        tracing::info!("Shutting down game map");
        let timers = self.timer_controller.lock().take();
        match timers {
            Some(timers) => {
                match tokio::time::timeout(Duration::from_secs(10), timers.shutdown()).await {
                    Ok(Ok(())) => {
                        tracing::info!("Shut down timer controller");
                    }
                    Ok(Err(e)) => {
                        tracing::error!("Error shutting down timer controller: {e:?}");
                    }
                    Err(_) => {
                        tracing::error!("Timed out waiting for timer controller to shut down");
                    }
                }
            }
            None => {
                tracing::warn!("Tried to shutdown timer controller but it was never brought up");
            }
        }
        // We need to stop the timers before we stop the writeback threads, since the timers might
        // try to write to the map, which requires a writeback thread to actually write back.
        self.shutdown.cancel();
        let await_task = async {
            for i in 0..NUM_CHUNK_SHARDS {
                let writeback_handle = self.writeback_handles[i].lock().take();
                writeback_handle.unwrap().await??;

                let cleanup_handle = self.cleanup_handles[i].lock().take();
                cleanup_handle.unwrap().await??;
            }
            Ok::<_, Error>(())
        };
        match tokio::time::timeout(Duration::from_secs(10), await_task).await {
            Ok(Ok(())) => {
                tracing::info!("Shut down async map workers");
            }
            Ok(Err(e)) => {
                tracing::error!("Error shutting down async map workers: {e:?}");
            }
            Err(_) => {
                tracing::error!("Timed out waiting for async map workers to shut down");
            }
        }
        self.flush();

        Ok(())
    }

    pub(crate) fn flush(&self) {
        for shard in 0..NUM_CHUNK_SHARDS {
            let mut lock = self.live_chunks[shard].write();
            let coords: Vec<_> = lock.chunks.keys().copied().collect();
            info!(
                "ServerGameMap shard {} being flushed: Writing back {} chunks",
                shard,
                coords.len()
            );
            for coord in coords {
                match self.unload_chunk_locked(&mut lock, coord) {
                    Ok(_) => { /* pass */ }
                    Err(e) => {
                        log::error!("Writeback error for {:?}: {:?}", coord, e);
                    }
                }
            }
        }
        match self.database.flush() {
            Ok(_) => { /* pass */ }
            Err(e) => {
                log::error!("Flushing DB failed: {:?}", e);
            }
        }
    }

    /// Rudimentary backpressure/flow control.
    pub(crate) fn in_pushback(&self) -> bool {
        self.writeback_senders
            .iter()
            .map(mpsc::Sender::capacity)
            .max()
            .unwrap()
            < (WRITEBACK_QUEUE_SIZE / 2)
    }

    pub(crate) fn debug_shard_sizes(&self) -> Vec<usize> {
        self.live_chunks
            .iter()
            .map(|x| x.read().chunks.len())
            .collect()
    }
}
impl Drop for ServerGameMap {
    fn drop(&mut self) {
        self.flush();
        tracing::info!("ServerGameMap shut down");
    }
}

enum WritebackReq {
    Chunk(ChunkCoordinate),
    Flush,
}

// TODO expose as flags or configs
pub(crate) const CACHE_CLEAN_MIN_AGE: Duration = Duration::from_secs(10);
const CACHE_CLEAN_INTERVAL: Duration = Duration::from_secs(3);
const CACHE_CLEANUP_KEEP_N_RECENTLY_USED: usize = 128;
const CACHE_CLEANUP_RELOCK_EVERY_N: usize = 32;

struct MapCacheCleanup {
    map: Arc<ServerGameMap>,
    cancellation: CancellationToken,
    shard_id: usize,
}
impl MapCacheCleanup {
    async fn run_loop(&mut self) -> Result<()> {
        let mut interval = tokio::time::interval(CACHE_CLEAN_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        while !self.cancellation.is_cancelled() {
            tokio::select! {
                _ = interval.tick() => {
                    tokio::task::block_in_place(|| self.do_cleanup())?;
                }
                _ = self.cancellation.cancelled() => {
                    info!("Map cache cleanup thread shutting down");
                    break;
                }
            }
        }
        Ok(())
    }
    fn do_cleanup(&self) -> Result<()> {
        let _span = span!("map cache cleanup");
        loop {
            let _span = span!("map cache cleanup iteration");
            let now = Instant::now();
            let mut lock = {
                let _span = span!("game_map cleanup write lock");
                self.map.live_chunks[self.shard_id].write()
            };
            if lock.chunks.len() <= CACHE_CLEANUP_KEEP_N_RECENTLY_USED {
                return Ok(());
            }
            let mut entries: Vec<_> = lock
                .chunks
                .iter()
                .map(|(k, v)| (v.last_accessed.get_acquire(), *k))
                .filter(|&entry| (now - entry.0) >= CACHE_CLEAN_MIN_AGE)
                .collect();
            entries.sort_unstable_by_key(|entry| entry.0);
            entries.reverse();
            let num_entries = entries.len();
            for entry in entries
                .into_iter()
                .skip(CACHE_CLEANUP_KEEP_N_RECENTLY_USED)
                .take(CACHE_CLEANUP_RELOCK_EVERY_N)
            {
                assert_eq!(self.shard_id, shard_id(entry.1));
                self.map.unload_chunk_locked(&mut lock, entry.1)?;
            }
            if num_entries > (CACHE_CLEANUP_RELOCK_EVERY_N + CACHE_CLEANUP_KEEP_N_RECENTLY_USED) {
                continue;
            } else {
                return Ok(());
            }
        }
    }
}

// TODO expose as flags or configs
const BROADCAST_CHANNEL_SIZE: usize = 8192;
const WRITEBACK_QUEUE_SIZE: usize = 256;
const WRITEBACK_COALESCE_TIME: Duration = Duration::from_secs(3);
const WRITEBACK_COALESCE_MAX_SIZE: usize = 8;

struct GameMapWriteback {
    map: Arc<ServerGameMap>,
    receiver: mpsc::Receiver<WritebackReq>,
    cancellation: CancellationToken,
    shard_id: usize,
}
impl GameMapWriteback {
    async fn run_loop(&mut self) -> Result<()> {
        while !self.cancellation.is_cancelled() {
            let writebacks = self.gather().await.unwrap();

            if writebacks.len() >= (WRITEBACK_COALESCE_MAX_SIZE) * 4 {
                warn!("Writeback backlog of {} chunks is unusually high; is the writeback thread falling behind?", writebacks.len());
            }
            tokio::task::block_in_place(|| self.do_writebacks(writebacks))?;
        }

        info!("Map writeback exiting");
        Ok(())
    }

    fn do_writebacks(&self, writebacks: Vec<ChunkCoordinate>) -> Result<()> {
        let _span = span!("game_map do_writebacks");
        tracing::trace!("Writing back {} chunks", writebacks.len());
        let lock = {
            let _span = span!("acquire game_map read lock for writeback");
            // This lock needs to barge ahead of writers to avoid starvation.
            // If writeback is overloaded, writers may as well wait a bit.
            self.map.live_chunks[self.shard_id].read_recursive()
        };
        for coord in writebacks {
            assert_eq!(self.shard_id, shard_id(coord));
            match lock.chunks.get(&coord) {
                Some(chunk_holder) => {
                    if let Some(mut chunk) = chunk_holder.try_get_write()? {
                        if chunk.dirty {
                            self.map.database.put(
                                &KeySpace::MapchunkData.make_key(&coord.as_bytes()),
                                &chunk
                                    .serialize(ChunkUsage::Server, &self.map.game_state())?
                                    .encode_to_vec(),
                            )?;
                        }
                        chunk.dirty = false;
                    } else {
                        warn!(
                            "Writeback thread got chunk {:?} but it wasn't loaded yet",
                            coord
                        );
                    }
                }
                None => {
                    warn!(
                        "Writeback thread got chunk {:?} but it wasn't in memory",
                        coord
                    );
                }
            }
        }
        Ok(())
    }

    async fn gather(&mut self) -> Option<Vec<ChunkCoordinate>> {
        let mut entries = Vec::new();
        let should_coalesce = tokio::select! {
            x = self.receiver.recv() => {
                match x {
                    Some(WritebackReq::Chunk(x)) => {
                        entries.push(x);
                        true
                    }
                    Some(WritebackReq::Flush) => {
                        // We're asked to flush but we're already caught up
                        return Some(vec![]);
                    }
                    None => return None,
                }
            },
            _ = self.cancellation.cancelled() => {
                log::info!("Map writeback detected cancellation");
                false
            }
        };
        if should_coalesce {
            let mut count = 1;
            let start = Instant::now();
            let deadline = start + WRITEBACK_COALESCE_TIME;
            while count < WRITEBACK_COALESCE_MAX_SIZE {
                tokio::select! {
                    _ = tokio::time::sleep_until(deadline.into()) => {
                        break;
                    }
                    _ = self.cancellation.cancelled() => {
                        break;
                    }
                    chunk = self.receiver.recv() => {
                        match chunk {
                            Some(WritebackReq::Chunk(chunk)) => {
                                count += 1;
                                entries.push(chunk);
                            }
                            Some(WritebackReq::Flush) => {
                                // Asked to flush. Stop coalescing and return now.
                                break;
                            }
                            None => {
                                // Other end is gone; we'll deal with that after writing back
                                break;
                            }
                        }

                    }
                }
            }
        }
        // Catch up anything already queued up after coalescing (but don't wait for more requests to arrive)
        loop {
            let maybe_chunk = self.receiver.try_recv();
            match maybe_chunk {
                Ok(WritebackReq::Chunk(x)) => entries.push(x),
                Ok(WritebackReq::Flush) => {
                    // pass, we're flushing right now
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    warn!("unexpected disconnected writeback queue receiver");
                    break;
                }
            }
        }
        entries.reverse();
        let entries_len = entries.len();
        let mut dedup = Vec::new();
        let mut seen = HashSet::new();
        for entry in entries {
            if seen.insert(entry) {
                dedup.push(entry)
            }
        }
        dedup.reverse();
        plot!("writeback_coalesce_items", entries_len as f64);
        plot!(
            "writeback_dedup_ratio",
            dedup.len() as f64 / entries_len as f64
        );
        Some(dedup)
    }
}
impl Drop for GameMapWriteback {
    fn drop(&mut self) {
        self.cancellation.cancel()
    }
}

pub struct TimerState {
    pub prev_tick_time: Instant,
    pub current_tick_time: Instant,
}

struct ShardState {
    timer_state: TimerState,
    // pre-allocated here to avoid allocations in the timer path
    neighbor_buffer: ChunkNeighbors,
}

pub trait TimerInlineCallback: Send + Sync {
    /// Called once for each block on the map that matched the block type configured for the timer
    /// This may be invoked concurrenty for multiple blocks and multiple chunks.
    ///
    /// Args:
    /// * coordinate: Location of the block this is being called for
    /// * state: The ShardState for this run of the timer.
    /// * block_type: Mutable reference to the block type in the block.
    /// * data: Mutable reference to the extended data in the block.
    /// * ctx: Context for the callback
    fn inline_callback(
        &self,
        coordinate: BlockCoordinate,
        timer_state: &TimerState,
        block_type: &mut BlockId,
        data: &mut ExtendedDataHolder,
        ctx: &InlineContext,
    ) -> Result<()>;
}

pub struct ChunkNeighbors {
    center: BlockCoordinate,
    presence_bitmap: u32,
    blocks: Box<[u32; 48 * 48 * 48]>,
}
impl ChunkNeighbors {
    /// Get the neighbors of a chunk.
    /// Note: This is guaranteed to return the neighbors of the chunk in question, assuming that the chunk is loaded.
    ///
    /// Chunks that are *not* the neighbor may or may not be returned arbitrarily (due to optimizations in the timer engine). Do not
    /// rely on their presence. They may also be returned inconsistently (i.e. either before or after the timer callback's effects)
    pub fn get_block(&self, coord: BlockCoordinate) -> Option<BlockId> {
        let dx = coord.x - self.center.x;
        let dz = coord.z - self.center.z;
        let dy = coord.y - self.center.y;
        if !(-16..=32).contains(&dx) || !(-16..=32).contains(&dz) || !(-16..=32).contains(&dy) {
            return None;
        }
        let cx = dx >> 4;
        let cz = dz >> 4;
        let cy = dy >> 4;
        if self.presence_bitmap & (1 << ((cx + 1) * 9 + (cz + 1) * 3 + (cy + 1))) == 0 {
            return None;
        }
        let index = ((dx + 16) * 48 * 48) as usize + (dz + 16) as usize * 48 + (dy + 16) as usize;
        Some(self.blocks[index].into())
    }
}

pub trait BulkUpdateCallback: Send + Sync {
    /// Called once for each chunk that *might* contain one of the block types configured for this timer.
    /// *This is a probabilistic check, and the callback may be called even if a configured block type is not actually
    ///    present.*
    ///
    /// Performance tip: Iterating in x/z/y (y on the innermost loop) order is the most cache-friendly order possible.
    ///
    /// **Warning**: Trying to access the map via ctx can cause deadlocks
    fn bulk_update_callback(
        &self,
        ctx: &HandlerContext<'_>,
        chunk_coordinate: ChunkCoordinate,
        timer_state: &TimerState,
        chunk: &mut MapChunk,
        neighbors: Option<&ChunkNeighbors>,
    ) -> Result<()>;
}

pub trait VerticalNeighborTimerCallback: Send + Sync {
    /// Called once for each chunk that *might* contain one of the block types configured for this timer.
    ///
    /// In particular, this will be called when either upper or lower contains the block type in question.
    ///
    /// *This is a probabilistic check, and the callback may be called even if a configured block type is not actually
    ///    present.*
    ///
    /// Performance tip: Iterating in x/z/y (y on the innermost loop) order is the most cache-friendly order possible.
    fn vertical_neighbor_callback(
        &self,
        ctx: &HandlerContext<'_>,
        upper: ChunkCoordinate,
        lower: ChunkCoordinate,
        upper_chunk: &mut MapChunk,
        lower_chunk: &mut MapChunk,
        timer_state: &TimerState,
    ) -> Result<()>;
}

pub enum TimerCallback {
    /// Callback operating on one block at a time. The engine may call it concurrently for multiple
    /// blocks in a chunk, or for multiple chunks. The timing of the callback may be changed between versions,
    /// but the engine will call it once per matching block per timer cycle.
    PerBlockLocked(Box<dyn TimerInlineCallback>),
    /// Callback operating on entire chunks. The engine may call it concurrently for multiple chunks.
    /// This callback will be called once per timer cycle, with the chunk locked for edit.
    ///
    /// No neighbor data is passed.
    BulkUpdate(Box<dyn BulkUpdateCallback>),
    /// Callback operating on entire chunks. The engine may call it concurrently for multiple chunks.
    /// This callback will be called once per timer cycle, with the chunk locked for edit.
    ///
    /// Neighbor data is passed.
    BulkUpdateWithNeighbors(Box<dyn BulkUpdateCallback>),
    /// A fast callback that gives you locked access to *two* vertically contiguous chunks
    /// at a time. The iteration order is top-to-bottom, and only vertically contiguous chunks
    /// are supported. This takes advantage of lighting-related acceleration structures.
    ///
    /// Note that the sharding policy of this timer may be significantly different from other
    /// timer types. In the current implementation, sharding is done based on vertical slices
    /// of the loaded map, rather than on a chunk-by-chunk basis.
    ///
    /// Experimental, subject to change (even more so than everything else in this crate)
    LockedVerticalNeighors(Box<dyn VerticalNeighborTimerCallback>),
}

/// Marker that a struct may be extended in the future
pub struct NonExhaustive(pub(crate) ());

/// Control for a map timer.
pub struct TimerSettings {
    /// The time between ticks of the timer
    pub interval: Duration,
    /// The number of shards
    pub shards: usize,
    /// How strongly to stagger the shards.
    /// 0.0 - fire all shards at approximately the same time
    /// 1.0 - spread the shards as far apart as possible
    /// Intermediate values - spread the shards out, but in a smaller span of time
    pub spreading: f64,
    /// The set of block types (*not* including variant types) that this timer should act on
    pub block_types: Vec<BlockId>,
    /// If set, do *not* use block bloom filters to determine whether a block is present.
    /// This is useful for bulk update callbacks that might need to run in all chunks.
    pub ignore_block_type_presence_check: bool,
    /// The probability that the action will be taken for each matching block. Each matching block is
    /// sampled independently, using an unspecified RNG that does not derive from the game seed.
    ///
    /// **Warning:** Ignored for handlers that act on entire chunks (e.g. BulkUpdate or BulkUpdateWithNeighbors)
    pub per_block_probability: f64,
    /// For *bulk handlers only*, if the bulk handler leaves a chunk unchanged, do not run the bulk handler for that chunk
    /// again until the next time the chunk is modified by external means.
    ///
    /// For bulk handlers with neighbors, the handler will run if the chunk or any neighbors have been modified.
    pub idle_chunk_after_unchanged: bool,
    pub _ne: NonExhaustive,
}
impl Default for TimerSettings {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(1),
            shards: 256,
            spreading: 1.0,
            block_types: Default::default(),
            ignore_block_type_presence_check: false,
            per_block_probability: 1.0,
            idle_chunk_after_unchanged: false,
            _ne: NonExhaustive(()),
        }
    }
}

struct GameMapTimer {
    // The name of the timer; in the future this will be used for catching up tasks on unloaded chunks
    name: String,
    // The action to take on each block when the timer fires
    callback: TimerCallback,
    settings: TimerSettings,
    cancellation: CancellationToken,
    // This mutex must be held across an await (for each handle), so we use a tokio async mutex here.
    tasks: tokio::sync::Mutex<Vec<JoinHandle<Result<()>>>>,
}
impl GameMapTimer {
    fn spawn_shards(
        self: &Arc<Self>,
        game_state: Arc<GameState>,
        fine_shards_per_coarse: usize,
    ) -> Result<()> {
        ensure!(!self.cancellation.is_cancelled());
        let mut tasks = self.tasks.blocking_lock();
        // Hacky: Delay 0.1 seconds to allow shards to start up
        let first_run = Instant::now() + Duration::from_millis(100);
        for coarse_shard in 0..NUM_CHUNK_SHARDS {
            for fine_shard in 0..fine_shards_per_coarse {
                let start_time = first_run
                    + (self.settings.interval.mul_f64(
                        self.settings.spreading
                            * (fine_shard + fine_shards_per_coarse * coarse_shard) as f64
                            / (fine_shards_per_coarse * NUM_CHUNK_SHARDS) as f64,
                    ));
                {
                    let cloned_self = self.clone();
                    let cloned_game_state = game_state.clone();
                    tasks.push(crate::spawn_async(
                        &format!("timer_{}_shard_{}", self.name, fine_shard),
                        // TODO error-check this
                        // It's brittle on shutdown due to closed channels
                        // We should probaly shut down the timers before shutting down the rest of the map
                        async move {
                            cloned_self
                                .run_shard(
                                    start_time,
                                    coarse_shard,
                                    fine_shard,
                                    fine_shards_per_coarse,
                                    cloned_game_state,
                                )
                                .await
                        },
                    )?);
                }
            }
        }

        Ok(())
    }
    #[tracing::instrument(
        name = "timer_shard",
        level = "trace",
        skip(self, start_time, coarse_shard, fine_shard, fine_shards_per_coarse, game_state),
        fields(
            timer_name = %self.name,
        )
    )]
    async fn run_shard(
        &self,
        start_time: Instant,
        coarse_shard: usize,
        fine_shard: usize,
        fine_shards_per_coarse: usize,
        game_state: Arc<GameState>,
    ) -> Result<()> {
        let block_types =
            FxHashSet::from_iter(self.settings.block_types.iter().map(|x| x.base_id()));

        let mut interval = tokio::time::interval_at(start_time.into(), self.settings.interval);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut shard_state = ShardState {
            timer_state: TimerState {
                prev_tick_time: start_time,
                current_tick_time: Instant::now(),
            },
            neighbor_buffer: ChunkNeighbors {
                center: BlockCoordinate::new(0, 0, 0),
                presence_bitmap: 0,
                blocks: Box::new([0; 48 * 48 * 48]),
            },
        };
        // Todo detect skipped ticks and adjust accordingly
        while !self.cancellation.is_cancelled() {
            tokio::select! {
                _ = interval.tick() => {
                    let current_tick_start = Instant::now();
                    shard_state.timer_state.current_tick_time = current_tick_start;

                    tokio::task::block_in_place(|| self.delegate_locking_path(coarse_shard, fine_shard, fine_shards_per_coarse, game_state.clone(), &block_types, &mut shard_state))?;
                    shard_state.timer_state.prev_tick_time = current_tick_start;
                }
            }
        }
        Ok(())
    }

    fn do_vertical_neighbor_locking(
        &self,
        coarse_shard: usize,
        fine_shard: usize,
        fine_shards_per_coarse: usize,
        game_state: Arc<GameState>,
        state: &mut ShardState,
    ) -> Result<()> {
        let _span = span!("timer tick hand-over-hand");
        let mut writeback_permit = Some(game_state.game_map().get_writeback_permit(coarse_shard)?);
        let mut writeback_permit2 = Some(game_state.game_map().get_writeback_permit(coarse_shard)?);

        // Read the 2D slice coords we will work on, and then unlock
        let mut read_lock = {
            let _span = span!("acquire game_map read lock");
            game_state.game_map().live_chunks[coarse_shard].read()
        };
        let mut coords = {
            let _span = span!("read and filter chunks");
            read_lock
                .light_columns
                .keys()
                .filter(|&coord| {
                    let mut hasher = FxHasher::default();
                    coord.hash(&mut hasher);
                    hasher.finish() % fine_shards_per_coarse as u64 == fine_shard as u64
                })
                .copied()
                .collect::<Vec<_>>()
        };
        // Basic sort to try to increase locality a bit
        coords.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        for (x, z) in coords.into_iter() {
            {
                // we have a light column
                let light_col = match read_lock.light_columns.get(&(x, z)) {
                    Some(light_col) => light_col.copy_keys(),
                    None => {
                        continue;
                    }
                };
                // It's a bit of a pain to get safe deadlock-free borrows to work correctly if we actually use the chunk cursor.
                // To be clear, this is not a limitation of the borrow-checker; it is a limitation of the logic I managed to
                // come up with when I tried the nifty approach here

                // array_windows is unstable, so we have to do this manually
                for i in (0..light_col.len() - 1).rev() {
                    let upper_y = light_col[i + 1];
                    let lower_y = light_col[i];
                    if upper_y == (lower_y + 1) {
                        // We can actually work on a chunk
                        if writeback_permit.is_none() {
                            writeback_permit = Some(reacquire_writeback_permit(
                                &game_state,
                                coarse_shard,
                                &mut read_lock,
                            )?);
                        }
                        if writeback_permit2.is_none() {
                            writeback_permit2 = Some(reacquire_writeback_permit(
                                &game_state,
                                coarse_shard,
                                &mut read_lock,
                            )?);
                        }
                        // Lock ordering: It's important that we lock these upper-to-lower
                        // TODO: consider whether true hand-over-hand improves performance
                        let upper_coord = ChunkCoordinate::new(x, upper_y, z);
                        let lower_coord = ChunkCoordinate::new(x, lower_y, z);
                        let upper_chunk = match read_lock.chunks.get(&upper_coord) {
                            Some(upper_chunk) => upper_chunk,
                            None => {
                                continue;
                            }
                        };
                        let lower_chunk = match read_lock.chunks.get(&lower_coord) {
                            Some(lower_chunk) => lower_chunk,
                            None => {
                                continue;
                            }
                        };
                        let passed_block_presence = self.settings.ignore_block_type_presence_check
                            || self.settings.block_types.iter().any(|x| {
                                upper_chunk
                                    .block_bloom_filter
                                    .maybe_contains(x.base_id() as u64)
                            })
                            || self.settings.block_types.iter().any(|x| {
                                lower_chunk
                                    .block_bloom_filter
                                    .maybe_contains(x.base_id() as u64)
                            });
                        if !passed_block_presence {
                            continue;
                        }
                        let last_update = upper_chunk
                            .last_written
                            .get_acquire()
                            .max(lower_chunk.last_written.get_acquire());
                        let should_run = !self.settings.idle_chunk_after_unchanged
                            || last_update >= state.timer_state.prev_tick_time;
                        if should_run {
                            self.handle_chunk_vertical_pairs(
                                upper_coord,
                                lower_coord,
                                upper_chunk,
                                lower_chunk,
                                &game_state,
                                &mut writeback_permit,
                                &mut writeback_permit2,
                                state,
                            )?;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn do_tick_locking_with_neighbors(
        &self,
        coarse_shard: usize,
        fine_shard: usize,
        fine_shards_per_coarse: usize,
        game_state: Arc<GameState>,
        state: &mut ShardState,
    ) -> Result<()> {
        let _span = span!("timer tick with neighbors");
        let mut writeback_permit = Some(game_state.game_map().get_writeback_permit(coarse_shard)?);

        // Read the coords, and then unlock.
        let mut coords = {
            let _span = span!("read and filter chunks");
            game_state.game_map().live_chunks[coarse_shard]
                .read()
                .chunks
                .keys()
                .filter(|&coord| {
                    coord.hash_u64() % fine_shards_per_coarse as u64 == fine_shard as u64
                })
                .copied()
                .collect::<Vec<_>>()
        };
        // Basic sort to try to increase locality a bit
        coords.sort_unstable_by(|a, b| {
            a.x.cmp(&b.x)
                .then_with(|| a.z.cmp(&b.z))
                .then_with(|| a.y.cmp(&b.y))
        });
        plot!("timer tick coords", coords.len() as f64);

        for coord in coords.into_iter() {
            if writeback_permit.is_none() {
                // We don't hold a read lock, so this doesn't risk deadlock
                writeback_permit = Some(game_state.game_map().get_writeback_permit(coarse_shard)?);
            }
            // this does the locking twice, with the benefit that it elides all of the memory copying
            // associated with build_neighbors if we don't end up actually using that neighbor data
            let (matches, latest_update) = self.build_neighbors(
                &mut state.neighbor_buffer,
                coord,
                game_state.game_map(),
                false,
            )?;
            let should_run = (!self.settings.idle_chunk_after_unchanged
                || latest_update.is_some_and(|x| x >= state.timer_state.prev_tick_time))
                && matches;

            if should_run {
                let (_, _) = self.build_neighbors(
                    &mut state.neighbor_buffer,
                    coord,
                    game_state.game_map(),
                    true,
                )?;
                let shard = game_state.game_map().live_chunks[coarse_shard].read();
                if let Some(holder) = shard.chunks.get(&coord) {
                    if let Some(mut chunk) = holder.try_get_write()? {
                        match &self.callback {
                            TimerCallback::BulkUpdateWithNeighbors(_) => {
                                self.run_bulk_handler(
                                    &game_state,
                                    holder,
                                    &mut chunk,
                                    coord,
                                    Some(&state.neighbor_buffer),
                                    state,
                                    &mut writeback_permit,
                                )?;
                            }
                            _ => unreachable!(),
                        }
                        // if chunk.dirty {
                        //     // This has a small missed optimization - until the chunk is written back, this
                        //     // will keep firing. If further optimizations is needed, track whether
                        //     // *this* bulk updater modified the chunk, and use that for setting
                        //     // the last_written timestamp
                        //     *holder.last_written.lock() = state.timer_state.current_tick_time;
                        //     writeback_permit
                        //         .take()
                        //         .unwrap()
                        //         .send(WritebackReq::Chunk(coord));
                        // }
                    }
                }
            }
        }

        Ok(())
    }

    // Picks the locking path based on the type of callback
    fn delegate_locking_path(
        &self,
        coarse_shard: usize,
        fine_shard: usize,
        fine_shards_per_coarse: usize,
        game_state: Arc<GameState>,
        block_types: &FxHashSet<u32>,
        state: &mut ShardState,
    ) -> Result<()> {
        match &self.callback {
            TimerCallback::PerBlockLocked(_) | TimerCallback::BulkUpdate(_) => self
                .do_tick_fast_lock_path(
                    coarse_shard,
                    fine_shard,
                    fine_shards_per_coarse,
                    game_state,
                    block_types,
                    state,
                ),
            TimerCallback::BulkUpdateWithNeighbors(_) => self.do_tick_locking_with_neighbors(
                coarse_shard,
                fine_shard,
                fine_shards_per_coarse,
                game_state,
                state,
            ),
            TimerCallback::LockedVerticalNeighors(_) => self.do_vertical_neighbor_locking(
                coarse_shard,
                fine_shard,
                fine_shards_per_coarse,
                game_state,
                state,
            ),
        }
    }

    // Fast lock path - no inter-chunk interactions allowed
    fn do_tick_fast_lock_path(
        &self,
        coarse_shard: usize,
        fine_shard: usize,
        fine_shards_per_coarse: usize,
        game_state: Arc<GameState>,
        block_types: &FxHashSet<u32>,
        state: &ShardState,
    ) -> Result<()> {
        let _span = span!("timer tick fast");
        let mut writeback_permit = Some(game_state.game_map().get_writeback_permit(coarse_shard)?);
        let mut read_lock = {
            let _span = span!("acquire game_map read lock");
            game_state.game_map().live_chunks[coarse_shard].read()
        };
        let coords = read_lock
            .chunks
            .keys()
            .filter(|x| x.hash_u64() % fine_shards_per_coarse as u64 == fine_shard as u64)
            .cloned()
            .collect::<Vec<_>>();
        plot!("timer tick coords", coords.len() as f64);
        for (i, coord) in coords.into_iter().enumerate() {
            if writeback_permit.is_none() {
                writeback_permit = Some(reacquire_writeback_permit(
                    &game_state,
                    coarse_shard,
                    &mut read_lock,
                )?);
            } else if i % 100 == 0 {
                let _span = span!("timer bumping");
                RwLockReadGuard::bump(&mut read_lock);
            }
            if let Some(chunk) = read_lock.chunks.get(&coord) {
                if self.settings.ignore_block_type_presence_check
                    || self
                        .settings
                        .block_types
                        .iter()
                        .any(|x| chunk.block_bloom_filter.maybe_contains(x.base_id() as u64))
                {
                    let last_update = chunk.last_written.get_acquire();
                    let should_run = !self.settings.idle_chunk_after_unchanged
                        || last_update >= state.timer_state.prev_tick_time;
                    if should_run {
                        self.handle_chunk_no_neighbors(
                            coord,
                            chunk,
                            &game_state,
                            block_types,
                            &mut writeback_permit,
                            state,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    fn handle_chunk_no_neighbors(
        &self,
        coord: ChunkCoordinate,
        holder: &MapChunkHolder,
        game_state: &Arc<GameState>,
        block_types: &FxHashSet<u32>,
        writeback_permit: &mut Option<mpsc::Permit<'_, WritebackReq>>,
        state: &ShardState,
    ) -> Result<()> {
        assert!(writeback_permit.is_some());

        if let Some(mut chunk) = holder.try_get_write()? {
            match &self.callback {
                TimerCallback::PerBlockLocked(_) => {
                    self.run_per_block_handler(
                        game_state,
                        chunk,
                        holder,
                        block_types,
                        coord,
                        state,
                    )?;
                }
                TimerCallback::BulkUpdate(_) => {
                    let chunk_update = holder.last_written.get_acquire();
                    if !self.settings.idle_chunk_after_unchanged
                        || chunk_update >= state.timer_state.prev_tick_time
                    {
                        self.run_bulk_handler(
                            game_state,
                            holder,
                            &mut chunk,
                            coord,
                            None,
                            state,
                            writeback_permit,
                        )?;
                    }
                }
                TimerCallback::BulkUpdateWithNeighbors(_) => {
                    unreachable!()
                }
                TimerCallback::LockedVerticalNeighors(_) => {
                    unreachable!()
                }
            }
        }

        Ok(())
    }

    fn handle_chunk_vertical_pairs(
        &self,
        upper_coord: ChunkCoordinate,
        lower_coord: ChunkCoordinate,
        upper_holder: &MapChunkHolder,
        lower_holder: &MapChunkHolder,
        game_state: &Arc<GameState>,
        upper_writeback_permit: &mut Option<mpsc::Permit<'_, WritebackReq>>,
        lower_writeback_permit: &mut Option<mpsc::Permit<'_, WritebackReq>>,
        state: &ShardState,
    ) -> Result<()> {
        assert!(upper_writeback_permit.is_some());
        assert!(lower_writeback_permit.is_some());
        let mut upper_chunk = match upper_holder.try_get_write()? {
            Some(x) => x,
            None => {
                return Ok(());
            }
        };
        let mut lower_chunk = match lower_holder.try_get_write()? {
            Some(x) => x,
            None => {
                return Ok(());
            }
        };
        let upper_old_block_ids = upper_chunk.clone_block_ids();
        let lower_old_block_ids = lower_chunk.clone_block_ids();

        let ctx = HandlerContext {
            tick: 0,
            initiator: EventInitiator::Engine,
            game_state: game_state.clone(),
        };

        match &self.callback {
            TimerCallback::LockedVerticalNeighors(x) => {
                run_handler!(
                    || x.vertical_neighbor_callback(
                        &ctx,
                        upper_coord,
                        lower_coord,
                        &mut upper_chunk,
                        &mut lower_chunk,
                        &state.timer_state,
                    ),
                    "vertical_neighbor_timer",
                    &EventInitiator::Engine
                )?;
            }
            _ => {
                unreachable!()
            }
        }
        reconcile_after_bulk_handler(
            &upper_old_block_ids,
            &mut upper_chunk,
            upper_holder,
            game_state,
            upper_coord,
            upper_writeback_permit,
            state.timer_state.current_tick_time,
        );
        reconcile_after_bulk_handler(
            &lower_old_block_ids,
            &mut lower_chunk,
            lower_holder,
            game_state,
            lower_coord,
            lower_writeback_permit,
            state.timer_state.current_tick_time,
        );

        Ok(())
    }

    fn run_per_block_handler(
        &self,
        game_state: &Arc<GameState>,
        mut chunk: MapChunkInnerWriteGuard<'_>,
        holder: &MapChunkHolder,
        block_types: &FxHashSet<u32>,
        coord: ChunkCoordinate,
        state: &ShardState,
    ) -> Result<(), Error> {
        let mut rng = rand::thread_rng();
        let sampler = Bernoulli::new(self.settings.per_block_probability)?;
        let map = game_state.game_map();
        for i in 0..4096 {
            let block_id = BlockId(chunk.block_ids[i].load(Ordering::Relaxed));
            assert!(holder
                .block_bloom_filter
                .maybe_contains(block_id.base_id() as u64));
            if block_types.contains(&block_id.base_id()) && sampler.sample(&mut rng) {
                match self.run_per_block_callback(
                    holder,
                    &mut chunk,
                    ChunkOffset::from_index(i),
                    map,
                    coord,
                    game_state,
                    state,
                ) {
                    Ok(()) => {
                        // continue
                    }
                    Err(e) => {
                        log::error!("Timer callback {} failed: {:?}", self.name, e);
                    }
                }
            }
        }
        Ok(())
    }

    fn run_per_block_callback(
        &self,
        holder: &MapChunkHolder,
        chunk: &mut MapChunkInnerWriteGuard<'_>,
        offset: ChunkOffset,
        map: &ServerGameMap,
        coord: ChunkCoordinate,
        game_state: &Arc<GameState>,
        state: &ShardState,
    ) -> Result<()> {
        match &self.callback {
            TimerCallback::PerBlockLocked(cb) => {
                map.mutate_block_atomically_locked(
                    holder,
                    chunk,
                    offset,
                    |block_id, extended_data| {
                        let ctx = InlineContext {
                            // todo actual ticks
                            tick: 0,
                            initiator: EventInitiator::Engine,
                            location: coord.with_offset(offset),
                            block_types: game_state.game_map().block_type_manager(),
                            items: game_state.item_manager(),
                        };
                        run_handler!(
                            || cb.inline_callback(
                                coord.with_offset(offset),
                                &state.timer_state,
                                block_id,
                                extended_data,
                                &ctx
                            ),
                            "timer_inline_locked",
                            &EventInitiator::Engine
                        )?;
                        Ok(())
                    },
                    map,
                )?;
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    async fn await_shutdown(&self) -> Result<()> {
        for task in self.tasks.lock().await.drain(..) {
            task.await??;
        }
        Ok(())
    }

    fn run_bulk_handler(
        &self,
        game_state: &Arc<GameState>,
        holder: &MapChunkHolder,
        chunk: &mut MapChunkInnerWriteGuard<'_>,
        coord: ChunkCoordinate,
        neighbor_data: Option<&ChunkNeighbors>,
        state: &ShardState,
        permit: &mut Option<mpsc::Permit<'_, WritebackReq>>,
    ) -> Result<()> {
        let old_block_ids: Box<[u32; 4096]> = chunk.clone_block_ids();
        let ctx = HandlerContext {
            tick: 0,
            initiator: EventInitiator::Engine,
            game_state: game_state.clone(),
        };
        match &self.callback {
            TimerCallback::BulkUpdate(cb) => {
                assert!(neighbor_data.is_none());
                run_handler!(
                    || cb.bulk_update_callback(
                        &ctx,
                        coord,
                        &state.timer_state,
                        chunk,
                        neighbor_data
                    ),
                    "timer_bulk_update",
                    &EventInitiator::Engine
                )?;
            }
            TimerCallback::BulkUpdateWithNeighbors(cb) => {
                assert!(neighbor_data.is_some());
                run_handler!(
                    || cb.bulk_update_callback(
                        &ctx,
                        coord,
                        &state.timer_state,
                        chunk,
                        neighbor_data
                    ),
                    "timer_bulk_update_with_neighbors",
                    &EventInitiator::Engine
                )?;
            }
            _ => unreachable!(),
        };
        reconcile_after_bulk_handler(
            &old_block_ids,
            chunk,
            holder,
            game_state,
            coord,
            permit,
            state.timer_state.current_tick_time,
        );
        Ok(())
    }

    fn build_neighbors(
        &self,
        neighbor_data: &mut ChunkNeighbors,
        center_coord: ChunkCoordinate,
        game_map: &ServerGameMap,
        copy_data: bool,
    ) -> Result<(bool, Option<Instant>)> {
        let buf = &mut neighbor_data.blocks;
        let mut presence_bitmap = 0u32;
        let mut any_blooms_match = false;
        let mut update_times: SmallVec<[_; 27]> = smallvec![];
        for cx in -1..=1 {
            for cz in -1..=1 {
                for cy in -1..=1 {
                    if let Some(neighbor_coord) = center_coord.try_delta(cx, cy, cz) {
                        let shard = game_map.live_chunks[shard_id(neighbor_coord)].read();
                        if let Some(neighbor_holder) = shard.chunks.get(&neighbor_coord) {
                            if self.settings.block_types.iter().any(|x| {
                                neighbor_holder
                                    .block_bloom_filter
                                    .maybe_contains(x.base_id() as u64)
                            }) {
                                any_blooms_match = true;
                            }
                            update_times.push(neighbor_holder.last_written.get_acquire());

                            if let Some(contents) = neighbor_holder.try_get_read()? {
                                presence_bitmap |= 1 << ((cx + 1) * 9 + (cz + 1) * 3 + (cy + 1));
                                if copy_data {
                                    for dx in 0..16 {
                                        for dz in 0..16 {
                                            for dy in 0..16 {
                                                let x = (16 * cx) + dx;
                                                let z = (16 * cz) + dz;
                                                let y = (16 * cy) + dy;
                                                let o_index = (x + 16) as usize * 48 * 48
                                                    + (z + 16) as usize * 48
                                                    + (y + 16) as usize;
                                                let i_index = dx * 16 * 16 + dz * 16 + dy;
                                                buf[o_index] = contents.block_ids[i_index as usize]
                                                    .load(Ordering::Relaxed);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        neighbor_data.center = center_coord.with_offset(ChunkOffset { x: 0, y: 0, z: 0 });
        neighbor_data.presence_bitmap = presence_bitmap;
        Ok((any_blooms_match, update_times.into_iter().max()))
    }
}

fn reconcile_after_bulk_handler(
    old_block_ids: &[u32; 4096],
    chunk: &mut MapChunkInnerWriteGuard<'_>,
    holder: &MapChunkHolder,
    game_state: &Arc<GameState>,
    coord: ChunkCoordinate,
    permit: &mut Option<mpsc::Permit<'_, WritebackReq>>,
    _update_time: Instant,
) {
    let mut seen_blocks = FxHashSet::default();
    let mut any_updated = false;
    for i in 0..4096 {
        let old_block_id = BlockId::from(old_block_ids[i]);
        let new_block_id = BlockId::from(chunk.block_ids[i].load(Ordering::Relaxed));
        if old_block_id != new_block_id {
            if seen_blocks.insert(new_block_id.base_id()) {
                // this generates expensive `LOCK OR %rax(%r8,%rdx,8) as well as an expensive DIV
                // on x86_64.
                // Only insert unique block ids (still need to test this optimization)
                // TODO fork the bloom filter library and extend it to support constant lengths
                holder
                    .block_bloom_filter
                    .insert(new_block_id.base_id() as u64);
            }
            chunk.dirty = true;
            any_updated = true;
            game_state.game_map().broadcast_block_change(BlockUpdate {
                location: coord.with_offset(ChunkOffset::from_index(i)),
                new_value: new_block_id,
            });
        }
    }

    if any_updated {
        permit.take().unwrap().send(WritebackReq::Chunk(coord));
        holder.last_written.update_now_release();
    }
}

fn reacquire_writeback_permit<'a, 'b>(
    game_state: &'a Arc<GameState>,
    coarse_shard: usize,
    read_lock: &mut RwLockReadGuard<'_, MapShard>,
) -> Result<mpsc::Permit<'b, WritebackReq>, Error>
where
    'a: 'b,
{
    if let Some(permit) = game_state
        .game_map()
        .try_get_writeback_permit(coarse_shard)?
    {
        Ok(permit)
    } else {
        // We need to release the read lock to get a permit, as the writeback thread needs to get a write lock
        // to make progress.
        let permit_or = RwLockReadGuard::unlocked(read_lock, || {
            game_state.game_map().get_writeback_permit(coarse_shard)
        });
        Ok(permit_or?)
    }
}

struct TimerController {
    map: Arc<ServerGameMap>,
    game_state: Weak<GameState>,
    cancellation: CancellationToken,
    timers: FxHashMap<String, Arc<GameMapTimer>>,
}
impl TimerController {
    async fn shutdown(&self) -> Result<()> {
        self.cancellation.cancel();
        for timer in self.timers.values() {
            timer.await_shutdown().await?;
        }

        Ok(())
    }

    fn spawn_timer(
        &mut self,
        gs: Arc<GameState>,
        name: String,
        settings: TimerSettings,
        callback: TimerCallback,
    ) -> Result<()> {
        let shards = (settings.shards - 1) / NUM_CHUNK_SHARDS + 1;
        let timer = Arc::new(GameMapTimer {
            name: name.clone(),
            callback,
            settings,
            // This is a bug-prone code smell. The server game map shouldn't
            // be rummaging around in the timer controller to get the cancellation token.
            // TODO: fix it.
            cancellation: self.cancellation.clone(),
            tasks: tokio::sync::Mutex::new(vec![]),
        });
        timer.spawn_shards(gs, shards)?;
        self.timers.insert(name, timer);
        Ok(())
    }
}

impl ServerGameMap {
    /// Registers a timer to run on the map with the specified settings.
    pub fn register_timer(
        &self,
        name: String,
        settings: TimerSettings,
        callback: TimerCallback,
    ) -> Result<()> {
        let mut guard = self.timer_controller.lock();
        guard
            .as_mut()
            .unwrap()
            .spawn_timer(self.game_state().clone(), name, settings, callback)
    }
}

#[cfg(fuzzing)]
pub mod fuzz {
    use super::BlockCoordinate;

    pub fn fuzz_coordinate_serialization(data: &[u8]) {
        let block_coord = match BlockCoordinate::from_bytes(data) {
            Ok(x) => x,
            Err(_) => return,
        };
        let data2 = block_coord.to_bytes();
        assert_eq!(data, data2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let bc = BlockCoordinate {
            x: 1,
            y: -2,
            z: 2100000000,
        };
        let bytes = bc.as_bytes();
        let bc2 = BlockCoordinate::from_bytes(&bytes).unwrap();
        assert_eq!(bc, bc2);
    }
}
