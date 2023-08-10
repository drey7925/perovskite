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
use cuberef_core::block_id::BlockId;
use rand::distributions::Bernoulli;
use rand::prelude::Distribution;
use rustc_hash::{FxHashMap, FxHashSet};
use std::ops::{Deref, DerefMut};
use std::{
    collections::HashSet,
    fmt::Debug,
    sync::{Arc, Weak},
    time::{Duration, Instant},
};
use tokio_util::sync::CancellationToken;
use tracy_client::{plot, span};

use crate::run_handler;
use crate::{
    database::database_engine::{GameDatabase, KeySpace},
    game_state::inventory::Inventory,
};

use super::blocks::BlockInteractionResult;
use super::{
    blocks::{
        self, BlockTypeHandle, BlockTypeManager, ExtDataHandling, ExtendedData, ExtendedDataHolder,
        InlineContext, TryAsHandle,
    },
    event::{EventInitiator, HandlerContext},
    items::ItemStack,
    GameState,
};

use cuberef_core::{
    coordinates::{BlockCoordinate, ChunkCoordinate, ChunkOffset},
    protocol::map as mapchunk_proto,
};
use parking_lot::{Condvar, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};
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
    // once that is implemented)
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
    own_coord: ChunkCoordinate,
    // TODO: this was exposed for the temporary mapgen API. Lock this down and refactor
    // more calls similar to mutate_block_atomically
    pub(crate) block_ids: Vec<u32>,
    extended_data: FxHashMap<u16, ExtendedData>,

    game_state: Weak<GameState>,
    dirty: bool,
}
impl MapChunk {
    fn new(own_coord: ChunkCoordinate, game_state: Arc<GameState>) -> Self {
        let extended_data = FxHashMap::default();
        Self {
            own_coord,
            block_ids: vec![0; 4096],
            extended_data,
            game_state: Arc::downgrade(&game_state),
            dirty: false,
        }
    }

    fn serialize(&self, usage: ChunkUsage) -> Result<mapchunk_proto::StoredChunk> {
        let _span = span!("serialize chunk");
        let mut extended_data = Vec::new();

        for index in 0..4096 {
            let offset = ChunkOffset::from_index(index);
            let block_coord = self.own_coord.with_offset(offset);

            if usage == ChunkUsage::Server {
                if let Some(ext_data) = self.extended_data.get(&index.try_into().unwrap()) {
                    if let Some(ext_data_proto) =
                        self.extended_data_to_proto(index, block_coord, ext_data)?
                    {
                        extended_data.push(ext_data_proto);
                    }
                }
            }
        }

        let proto = mapchunk_proto::StoredChunk {
            chunk_data: Some(mapchunk_proto::stored_chunk::ChunkData::V1(
                mapchunk_proto::ChunkV1 {
                    block_ids: self.block_ids.to_vec(),
                    extended_data,
                },
            )),
        };

        Ok(proto)
    }

    fn extended_data_to_proto(
        &self,
        block_index: usize,
        block_coord: BlockCoordinate,
        ext_data: &ExtendedData,
    ) -> Result<Option<mapchunk_proto::ExtendedData>> {
        let id = self.block_ids[block_index];
        let game_state = self
            .game_state
            .upgrade()
            .with_context(|| "GameState weakref unexpectedly gone")?;
        let (block_type, _) = game_state
            .map()
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
                            block_types: game_state.map().block_type_manager(),
                            items: game_state.item_manager(),
                        };

                        (serializer)(handler_context, x)?
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
    ) -> Result<MapChunk> {
        let _span = span!("parse chunk");
        let proto = mapchunk_proto::StoredChunk::decode(bytes)
            .with_context(|| "MapChunk proto serialization failed")?;
        match proto.chunk_data {
            Some(mapchunk_proto::stored_chunk::ChunkData::V1(chunk_data)) => {
                parse_v1(chunk_data, coordinate, game_state)
            }
            None => bail!("Missing chunk_data or unrecognized format"),
        }
    }
    /// Sets the block at the given coordinate within the chunk.
    /// This function is intended to be used from map generators, and may
    /// lead to data loss or inconsistency when used elsewhere
    ///
    /// TODO refactor and fix
    pub fn set_block(
        &mut self,
        coordinate: ChunkOffset,
        block: BlockTypeHandle,
        extended_data: Option<ExtendedData>,
    ) {
        self.block_ids[coordinate.as_index()] = block.id().into();
        if let Some(extended_data) = extended_data {
            self.extended_data
                .insert(coordinate.as_index().try_into().unwrap(), extended_data);
        } else {
            self.extended_data
                .remove(&coordinate.as_index().try_into().unwrap());
        }
    }
}

fn parse_v1(
    chunk_data: mapchunk_proto::ChunkV1,
    coordinate: ChunkCoordinate,
    game_state: Arc<GameState>,
) -> std::result::Result<MapChunk, anyhow::Error> {
    let mut extended_data = FxHashMap::default();
    ensure!(
        chunk_data.block_ids.len() == 4096,
        "Block IDs length != 4096"
    );

    for mapchunk_proto::ExtendedData {
        offset_in_chunk,
        serialized_data,
        inventories,
    } in chunk_data.extended_data.iter()
    {
        ensure!(*offset_in_chunk < 4096);
        let offset = ChunkOffset::from_index(*offset_in_chunk as usize);
        let block_coord = coordinate.with_offset(offset);
        let block_id = *chunk_data.block_ids.get(*offset_in_chunk as usize).expect(
            "Block IDs vec lookup failed, even though bounds check passed. This should not happen.",
        );
        let (block_def, _) = game_state
            .map()
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
            block_types: game_state.map().block_type_manager(),
            items: game_state.item_manager(),
        };
        if let Some(ref deserialize) = block_def.deserialize_extended_data_handler {
            extended_data.insert(
                (*offset_in_chunk).try_into().unwrap(),
                ExtendedData {
                    custom_data: deserialize(handler_context, serialized_data)?,
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
                (*offset_in_chunk).try_into().unwrap(),
                ExtendedData {
                    custom_data: None,
                    inventories: inventories
                        .iter()
                        .map(|(k, v)| Ok((k.clone(), Inventory::from_proto(v.clone(), None)?)))
                        .collect::<Result<hashbrown::HashMap<_, _>>>()?,
                },
            );
        }
    }
    Ok(MapChunk {
        own_coord: coordinate,
        block_ids: chunk_data.block_ids,
        extended_data,
        game_state: Arc::downgrade(&game_state),
        dirty: false,
    })
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockUpdate {
    pub(crate) location: BlockCoordinate,
    pub(crate) new_value: BlockTypeHandle,
}

struct MapChunkInnerGuard<'a> {
    guard: MutexGuard<'a, HolderState>,
}
impl<'a> Deref for MapChunkInnerGuard<'a> {
    type Target = MapChunk;
    fn deref(&self) -> &Self::Target {
        self.guard.unwrap()
    }
}
impl<'a> DerefMut for MapChunkInnerGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guard.unwrap_mut()
    }
}

enum HolderState {
    Empty,
    Err(anyhow::Error),
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
    chunk: Mutex<HolderState>,
    condition: Condvar,
    last_accessed: Mutex<Instant>,
    block_bloom_filter: cbloom::Filter,
}
impl MapChunkHolder {
    fn new_empty() -> Self {
        Self {
            chunk: Mutex::new(HolderState::Empty),
            condition: Condvar::new(),
            last_accessed: Mutex::new(Instant::now()),
            // TODO: make bloom filter configurable or adaptive
            // For now, 128 bytes is a reasonable overhead (a chunk contains 4096 u32s which is 16 KiB already + extended data)
            // 5 hashers is a reasonable tradeoff - it's not the best false positive rate, but less hashers means cheaper insertion
            block_bloom_filter: cbloom::Filter::with_size_and_hashers(128, 5),
        }
    }

    /// Get the chunk, blocking until it's loaded
    fn wait_and_get(&self) -> Result<MapChunkInnerGuard<'_>> {
        let mut guard = self.chunk.lock();
        let _span = span!("wait_and_get");
        loop {
            match &*guard {
                HolderState::Empty => self.condition.wait(&mut guard),
                HolderState::Err(e) => return Err(Error::msg(format!("Chunk load failed: {e:?}"))),
                HolderState::Ok(_) => return Ok(MapChunkInnerGuard { guard }),
            }
        }
    }
    /// Get the chunk, returning None if it's not loaded yet
    fn try_get(&self) -> Result<Option<MapChunkInnerGuard<'_>>> {
        let guard = self.chunk.lock();
        match &*guard {
            HolderState::Empty => Ok(None),
            HolderState::Err(e) => Err(Error::msg(format!("Chunk load failed: {e:?}"))),
            HolderState::Ok(_) => Ok(Some(MapChunkInnerGuard { guard })),
        }
    }
    /// Set the chunk, and notify any threads waiting in wait_and_get
    fn fill(&self, chunk: MapChunk) {
        for block_id in chunk.block_ids.iter() {
            self.block_bloom_filter
                .insert(BlockId::from(*block_id).base_id() as u64);
        }
        let _span = span!("game_map waiting to fill chunk");
        let mut guard = self.chunk.lock();
        assert!(matches!(*guard, HolderState::Empty));
        *guard = HolderState::Ok(chunk);
        self.condition.notify_all();
    }
    /// Set the chunk to an error, and notify any waiting threads so they can propagate the error
    fn set_err(&self, err: anyhow::Error) {
        let mut guard = self.chunk.lock();
        assert!(matches!(*guard, HolderState::Empty));
        *guard = HolderState::Err(err);
        self.condition.notify_all();
    }

    fn bump_access_time(&self) {
        *self.last_accessed.lock() = Instant::now();
    }
}

struct MapChunkOuterGuard<'a> {
    // TODO: This has a slight inefficiency - we have to call get() on the map an extra time
    read_guard: RwLockReadGuard<'a, FxHashMap<ChunkCoordinate, MapChunkHolder>>,
    coord: ChunkCoordinate,
    writeback_permit: Option<mpsc::Permit<'a, WritebackReq>>,
    force_writeback: bool,
}
impl<'a> MapChunkOuterGuard<'a> {
    // Actually submits a writeback request to the writeback queue.
    // Naming note - _inner because ServerGameMap first does some error checking and
    // monitoring/stats before calling this.
    fn write_back_inner(mut self) {
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
        self.read_guard.get(&self.coord).as_ref().unwrap()
    }
}

/// Represents the entire map of the world.
/// This struct provides safe interior mutability - a shared reference
/// is sufficient to read and write it. Locking is done at an implementation-defined
/// granularity.
pub struct ServerGameMap {
    game_state: Weak<GameState>,
    database: Arc<dyn GameDatabase>,
    live_chunks: RwLock<FxHashMap<ChunkCoordinate, MapChunkHolder>>,
    block_type_manager: Arc<BlockTypeManager>,
    block_update_sender: broadcast::Sender<BlockUpdate>,
    writeback_sender: mpsc::Sender<WritebackReq>,
    shutdown: CancellationToken,
    timer_handle: Mutex<Option<JoinHandle<Result<()>>>>,
    writeback_handle: Mutex<Option<JoinHandle<Result<()>>>>,
    cleanup_handle: Mutex<Option<JoinHandle<Result<()>>>>,
    timer_controller: Mutex<Option<TimerController>>,
}
impl ServerGameMap {
    pub(crate) fn new(
        game_state: Weak<GameState>,
        database: Arc<dyn GameDatabase>,
        block_type_manager: Arc<BlockTypeManager>,
    ) -> Result<Arc<ServerGameMap>> {
        let (block_update_sender, _) = broadcast::channel(BROADCAST_CHANNEL_SIZE);
        let (writeback_sender, writeback_receiver) = mpsc::channel(WRITEBACK_QUEUE_SIZE);

        let cancellation = CancellationToken::new();

        let result = Arc::new(ServerGameMap {
            game_state: game_state.clone(),
            database,
            live_chunks: RwLock::new(FxHashMap::default()),
            block_type_manager,
            block_update_sender,
            writeback_sender,
            shutdown: cancellation.clone(),
            timer_handle: None.into(),
            writeback_handle: None.into(),
            cleanup_handle: None.into(),
            timer_controller: None.into(),
        });

        let mut writeback = GameMapWriteback {
            map: result.clone(),
            receiver: writeback_receiver,
            cancellation: cancellation.clone(),
        };

        let mut cache_cleanup = MapCacheCleanup {
            map: result.clone(),
            cancellation: cancellation.clone(),
        };

        let writeback_handle =
            crate::spawn_async("map_writeback", async move { writeback.run_loop().await })?;
        *result.writeback_handle.lock() = Some(writeback_handle);
        let cleanup_handle =
            crate::spawn_async("map_cleanup", async move { cache_cleanup.run_loop().await })?;
        *result.cleanup_handle.lock() = Some(cleanup_handle);

        let timer_controller = TimerController {
            map: result.clone(),
            game_state,
            cancellation,
            timers: FxHashMap::default(),
        };
        *result.timer_controller.lock() = Some(timer_controller);

        Ok(result)
    }

    pub(crate) fn bump_access_time(&self, coord: ChunkCoordinate) {
        let _span = span!("game_map read lock");
        if let Some(chunk) = self.live_chunks.read().get(&coord) {
            chunk.bump_access_time();
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
    ) -> Result<(BlockTypeHandle, Option<T>)>
    where
        F: FnOnce(&ExtendedData) -> Option<T>,
    {
        let chunk_guard = self.get_chunk(coord.chunk())?;
        let chunk = chunk_guard.wait_and_get()?;

        let id = chunk.block_ids[coord.offset().as_index()];
        let ext_data = match chunk
            .extended_data
            .get(&coord.offset().as_index().try_into().unwrap())
        {
            Some(x) => extended_data_callback(x),
            None => None,
        };

        Ok((
            self.block_type_manager().make_blockref(id.into())?,
            ext_data,
        ))
    }

    /// Gets a block + variant without its extended data
    pub fn get_block(&self, coord: BlockCoordinate) -> Result<BlockTypeHandle> {
        let chunk_guard = self.get_chunk(coord.chunk())?;
        let chunk = chunk_guard.wait_and_get()?;

        let id = chunk.block_ids[coord.offset().as_index()];

        self.block_type_manager().make_blockref(id.into())
    }

    /// Sets a block on the map. No handlers are run, and the block is updated unconditionally.
    /// The old block is returned along with its extended data, if any.
    pub fn set_block<T: TryAsHandle>(
        &self,
        coord: BlockCoordinate,
        block: T,
        new_data: Option<ExtendedData>,
    ) -> Result<(BlockTypeHandle, Option<ExtendedData>)> {
        let block = block
            .as_handle(&self.block_type_manager)
            .with_context(|| "Block not found")?;

        let chunk_guard = self.get_chunk(coord.chunk())?;
        let mut chunk = chunk_guard.wait_and_get()?;

        let old_id = chunk.block_ids[coord.offset().as_index()];
        let old_block = self.block_type_manager().make_blockref(old_id.into())?;
        let old_data = match new_data {
            Some(new_data) => chunk
                .extended_data
                .insert(coord.offset().as_index().try_into().unwrap(), new_data),
            None => chunk
                .extended_data
                .remove(&coord.offset().as_index().try_into().unwrap()),
        };

        chunk.block_ids[coord.offset().as_index()] = block.id().into();
        chunk.dirty = true;

        drop(chunk);
        chunk_guard
            .block_bloom_filter
            .insert(block.id().base_id() as u64);
        self.enqueue_writeback(chunk_guard)?;
        self.broadcast_block_change(BlockUpdate {
            location: coord,
            new_value: block,
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
    ) -> Result<(CasOutcome, BlockTypeHandle, Option<ExtendedData>)> {
        if check_variant {
            self.compare_and_set_block_predicate(
                coord,
                |x, _| {
                    x.as_handle(&self.block_type_manager)
                        .zip(expected.as_handle(&self.block_type_manager))
                        .map_or(false, |(x, y)| x == y)
                },
                block,
                extended_data,
            )
        } else {
            self.compare_and_set_block_predicate(
                coord,
                |x, _| {
                    x.as_handle(&self.block_type_manager)
                        .zip(expected.as_handle(&self.block_type_manager))
                        .map_or(false, |(x, y)| x.equals_ignore_variant(y))
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
        extended_data: Option<ExtendedData>,
    ) -> Result<(CasOutcome, BlockTypeHandle, Option<ExtendedData>)>
    where
        F: FnOnce(BlockTypeHandle, Option<&ExtendedData>) -> bool,
    {
        let block = block
            .as_handle(&self.block_type_manager)
            .with_context(|| "Block not found")?;

        let chunk_guard = self.get_chunk(coord.chunk())?;
        let mut chunk = chunk_guard.wait_and_get()?;

        let old_id = chunk.block_ids[coord.offset().as_index()];
        let old_block = self.block_type_manager().make_blockref(old_id.into())?;
        if !predicate(
            old_block,
            chunk
                .extended_data
                .get(&coord.offset().as_index().try_into().unwrap()),
        ) {
            return Ok((CasOutcome::Mismatch, old_block, None));
        }
        let old_data = match extended_data {
            Some(new_data) => chunk
                .extended_data
                .insert(coord.offset().as_index().try_into().unwrap(), new_data),
            None => chunk
                .extended_data
                .remove(&coord.offset().as_index().try_into().unwrap()),
        };
        chunk.block_ids[coord.offset().as_index()] = block.id().into();
        chunk.dirty = true;
        drop(chunk);
        chunk_guard
            .block_bloom_filter
            .insert(block.id().base_id() as u64);
        chunk_guard.write_back_inner();
        self.broadcast_block_change(BlockUpdate {
            location: coord,
            new_value: block,
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
        F: FnOnce(&mut BlockTypeHandle, &mut ExtendedDataHolder) -> Result<T>,
    {
        let chunk_guard = self.get_chunk(coord.chunk())?;
        let mut chunk = chunk_guard.wait_and_get()?;

        self.mutate_block_atomically_locked(
            &chunk_guard,
            &mut chunk,
            coord.offset(),
            mutator,
            self,
        )
    }

    /// Internal impl detail of mutate_block_atomically and timers
    fn mutate_block_atomically_locked<F, T>(
        &self,
        holder: &MapChunkHolder,
        chunk: &mut MapChunkInnerGuard<'_>,
        offset: ChunkOffset,
        mutator: F,
        game_map: &ServerGameMap,
    ) -> anyhow::Result<T>
    where
        F: FnOnce(&mut BlockTypeHandle, &mut ExtendedDataHolder) -> Result<T>,
    {
        let mut extended_data = chunk
            .extended_data
            .remove(&offset.as_index().try_into().unwrap());

        let mut data_holder = ExtendedDataHolder::new(&mut extended_data);

        let old_id = chunk.block_ids[offset.as_index()].into();
        let mut block_type = game_map.block_type_manager().make_blockref(old_id)?;
        let closure_result = mutator(&mut block_type, &mut data_holder);

        let extended_data_dirty = data_holder.dirty();
        if let Some(new_data) = extended_data {
            chunk
                .extended_data
                .insert(offset.as_index().try_into().unwrap(), new_data);
        }

        let new_id = block_type.id();
        if new_id != old_id {
            chunk.block_ids[offset.as_index()] = new_id.into();
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
                .broadcast_block_update(chunk.own_coord.with_offset(offset));
        }

        if new_id != old_id {
            game_map.broadcast_block_change(BlockUpdate {
                location: chunk.own_coord.with_offset(offset),
                new_value: block_type,
            });
        }

        closure_result
    }

    /// Digs a block, running its on-dig event handler. The items it drops are returned.
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
        F: FnOnce(&blocks::BlockType) -> Option<&blocks::InlineHandler>,
        G: FnOnce(&blocks::BlockType) -> Option<&blocks::FullHandler>,
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
                    initiator.clone(),
                )?;
                Ok((blocktype, result))
            } else {
                Ok((blocktype, Default::default()))
            }
            
        })?;
        // we need this to happen outside of mutate_block_atomically (which holds a chunk lock) to avoid a deadlock.
        if let Some(full_handler) = get_block_full_handler(blocktype) {
            let ctx = HandlerContext {
                tick,
                initiator: initiator.clone(),
                game_state: self.game_state(),
            };
            result += run_handler!(
                || (full_handler)(ctx, coord, tool),
                "block_full",
                initiator.clone(),
            )?;
        }

        Ok(result)
    }

    pub(crate) fn block_type_manager(&self) -> &BlockTypeManager {
        &self.block_type_manager
    }

    // Gets a chunk, loading it from database/generating it if it is not in memory
    #[tracing::instrument(level = "trace", name = "get_chunk", skip(self))]
    fn get_chunk<'a>(&'a self, coord: ChunkCoordinate) -> Result<MapChunkOuterGuard<'a>> {
        let writeback_permit = self.get_writeback_permit()?;

        let mut load_chunk_tries = 0;
        let result = loop {
            let read_guard = {
                let _span = span!("acquire game_map read lock");
                self.live_chunks.read()
            };
            // All good. The chunk is loaded.
            if read_guard.contains_key(&coord) {
                return Ok(MapChunkOuterGuard {
                    read_guard,
                    coord,
                    writeback_permit: Some(writeback_permit),
                    force_writeback: false,
                });
            }
            load_chunk_tries += 1;
            drop(read_guard);

            // The chunk is not loaded. Give up the lock, get a write lock, get an entry into the map, and then fill it
            // under a read lock
            // This can't be done with an upgradable lock due to a deadlock risk.
            let mut write_guard = {
                let _span = span!("acquire game_map write lock");
                self.live_chunks.write()
            };
            if write_guard.contains_key(&coord) {
                // Someone raced with us. Try looping again.
                log::info!("Race while upgrading in get_chunk; retrying two-phase lock");
                drop(write_guard);
                continue;
            }

            // We still hold the write lock. Insert, downgrade back to a read lock, and fill the chunk before returning.
            // Since we still hold the write lock, our check just above is still correct, and we can safely insert.
            write_guard.insert(coord, MapChunkHolder::new_empty());

            // Now we downgrade the read lock.
            // If another thread races ahead of us and does the same lookup before we manage to fill the chunk,
            // they'll get an empty chunk holder and will wait for the condition variable to be signalled
            // (when that thread waits on the condition variable, it atomically releases the inner lock)
            let read_guard = RwLockWriteGuard::downgrade(write_guard);
            // We had a write lock and downgraded it atomically. No other thread could have removed the entry.
            let chunk_guard = read_guard.get(&coord).unwrap();
            match self.load_uncached_or_generate_chunk(coord) {
                Ok((chunk, force_writeback)) => {
                    chunk_guard.fill(chunk);
                    break Ok(MapChunkOuterGuard {
                        read_guard,
                        coord,
                        writeback_permit: Some(writeback_permit),
                        force_writeback,
                    });
                }
                Err(e) => {
                    chunk_guard.set_err(Error::msg(format!("Chunk load/generate failed: {e:?}")));
                    // Unfortunate duplication, anyhow::Error is not Clone
                    break Err(Error::msg(format!("Chunk load/generate failed: {e:?}")));
                }
            }
        };
        if load_chunk_tries > 1 {
            log::warn!("Took {load_chunk_tries} tries to load {coord:?}");
        }
        result
    }

    fn get_writeback_permit(&self) -> Result<mpsc::Permit<'_, WritebackReq>> {
        let _span = span!("get writeback permit");
        Ok(tokio::runtime::Handle::current().block_on(self.writeback_sender.reserve())?)
    }

    fn try_get_writeback_permit(&self) -> Result<Option<mpsc::Permit<'_, WritebackReq>>> {
        match self.writeback_sender.try_reserve() {
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
    fn load_uncached_or_generate_chunk(&self, coord: ChunkCoordinate) -> Result<(MapChunk, bool)> {
        let data = self
            .database
            .get(&KeySpace::MapchunkData.make_key(&coord.as_bytes()))?;
        if let Some(data) = data {
            return Ok((
                MapChunk::deserialize(coord, &data, self.game_state())?,
                false,
            ));
        }

        let mut chunk = MapChunk::new(coord, self.game_state());
        chunk.dirty = true;
        {
            let _span = span!("mapgen running");
            run_handler!(
                || {
                    self.game_state().mapgen().fill_chunk(coord, &mut chunk);
                    Ok(())
                },
                "mapgen",
                EventInitiator::Engine
            )?;
        }
        Ok((chunk, true))
    }

    pub fn game_state(&self) -> Arc<GameState> {
        Weak::upgrade(&self.game_state).unwrap()
    }

    fn unload_chunk_locked(
        &self,
        lock: &mut RwLockWriteGuard<FxHashMap<ChunkCoordinate, MapChunkHolder>>,
        coord: ChunkCoordinate,
    ) -> Result<()> {
        let _span = span!("unload single chunk");
        let chunk = lock.remove(&coord);
        if let Some(chunk) = chunk {
            match &*chunk.chunk.lock() {
                HolderState::Empty => {
                    log::warn!("chunk unload got a chunk with an empty holder. This should never happen - please file a bug");
                }
                HolderState::Err(e) => {
                    log::warn!("chunk unload trying to unload a chunk with an error: {e:?}");
                }
                HolderState::Ok(chunk) => {
                    if chunk.dirty {
                        self.database.put(
                            &KeySpace::MapchunkData.make_key(&coord.as_bytes()),
                            &chunk.serialize(ChunkUsage::Server)?.encode_to_vec(),
                        )?;
                    }
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
    fn enqueue_writeback(&self, chunk: MapChunkOuterGuard<'_>) -> Result<()> {
        if self.shutdown.is_cancelled() {
            return Err(Error::msg("Writeback thread was shut down"));
        }
        plot!(
            "writeback queue capacity",
            self.writeback_sender.capacity() as f64
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
    pub(crate) fn get_chunk_client_proto(
        &self,
        coord: ChunkCoordinate,
        load_if_missing: bool,
    ) -> Result<Option<mapchunk_proto::StoredChunk>> {
        if load_if_missing {
            let chunk_guard = self.get_chunk(coord)?;
            let chunk = chunk_guard.wait_and_get()?;
            Ok(Some(chunk.serialize(ChunkUsage::Client)?))
        } else {
            Ok(None)
            // let guard = {
            //     let _span = span!("acquire gamemap read lock");
            //     self.live_chunks.read()
            // };
            // let chunk_holder = guard.get(&coord);
            // if let Some(chunk_holder) = chunk_holder {
            //     match chunk_holder.try_get()? {
            //         Some(x) => Ok(Some(x.serialize(ChunkUsage::Client)?)),
            //         None => Ok(None),
            //     }
            // } else {
            //     Ok(None)
            // }
        }
    }

    pub(crate) fn request_shutdown(&self) {
        self.shutdown.cancel();
    }

    pub(crate) async fn await_shutdown(&self) -> Result<()> {
        let writeback_handle = self.writeback_handle.lock().take();
        writeback_handle.unwrap().await??;

        let cleanup_handle = self.cleanup_handle.lock().take();
        cleanup_handle.unwrap().await??;
        self.flush();
        Ok(())
    }

    pub(crate) fn flush(&self) {
        let mut lock = self.live_chunks.write();
        let coords: Vec<_> = lock.keys().copied().collect();
        log::info!(
            "ServerGameMap being flushed: Writing back {} chunks",
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
        match self.database.flush() {
            Ok(_) => { /* pass */ }
            Err(e) => {
                log::error!("Flushing DB failed: {:?}", e);
            }
        }
    }

    /// Rudimentary backpressure/flow control.
    pub(crate) fn in_pushback(&self) -> bool {
        self.writeback_sender.capacity() < (WRITEBACK_QUEUE_SIZE / 2)
    }
}
impl Drop for ServerGameMap {
    fn drop(&mut self) {
        self.flush()
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
                self.map.live_chunks.write()
            };
            if lock.len() <= CACHE_CLEANUP_KEEP_N_RECENTLY_USED {
                return Ok(());
            }
            let mut entries: Vec<_> = lock
                .iter()
                .map(|(k, v)| (*v.last_accessed.lock(), *k))
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
const BROADCAST_CHANNEL_SIZE: usize = 1024;
const WRITEBACK_QUEUE_SIZE: usize = 256;
const WRITEBACK_COALESCE_TIME: Duration = Duration::from_secs(3);
const WRITEBACK_COALESCE_MAX_SIZE: usize = 8;

struct GameMapWriteback {
    map: Arc<ServerGameMap>,
    receiver: mpsc::Receiver<WritebackReq>,
    cancellation: CancellationToken,
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

        log::info!("Map writeback exiting");
        Ok(())
    }

    fn do_writebacks(&self, writebacks: Vec<ChunkCoordinate>) -> Result<()> {
        let _span = span!("game_map do_writebacks");
        tracing::trace!("Writing back {} chunks", writebacks.len());
        let lock = {
            let _span = span!("acquire game_map read lock for writeback");
            // This lock needs to barge ahead of writers to avoid starvation.
            // If writeback is overloaded, writers may as well wait a bit.
            self.map.live_chunks.read_recursive()
        };
        for coord in writebacks {
            match lock.get(&coord) {
                Some(chunk_holder) => {
                    if let Some(mut chunk) = chunk_holder.try_get()? {
                        if !chunk.dirty {
                            warn!("Writeback thread got chunk {:?} but it wasn't dirty", coord);
                        }
                        self.map.database.put(
                            &KeySpace::MapchunkData.make_key(&coord.as_bytes()),
                            &chunk.serialize(ChunkUsage::Server)?.encode_to_vec(),
                        )?;
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

pub trait TimerInlineCallback: Send + Sync {
    /// Called once for each block on the map that matched the block type configured for the timer
    /// This may be invoked concurrenty for multiple blocks and multiple chunks.
    ///
    /// Args:
    /// * coordinate: Location of the block this is being called for
    /// * missed_timers: The number of timer cycles to run. This may be greater than 0 if chunks are unloaded
    ///     or the game engine is overloaded. *Currently unimplemented, always 0*.
    /// * block_type: Mutable reference to the block type in the block.
    /// * data: Mutable reference to the extended data in the block.
    /// * ctx: Context for the callback
    fn inline_callback(
        &self,
        coordinate: BlockCoordinate,
        missed_timers: u64,
        block_type: &mut BlockTypeHandle,
        data: &mut ExtendedDataHolder,
        ctx: &InlineContext,
    ) -> Result<()>;
}

pub trait TimerUnlockedCallback: Send + Sync {
    /// Called once for each block on the map that matched the block type configured for the timer
    /// This may be invoked concurrenty for multiple chunks.
    ///
    /// Args:
    /// * coordinate: Location of the block this is being called for
    /// * missed_timers: The number of timer cycles to run. This may be greater than 1 if chunks are unloaded
    ///     or the game engine is overloaded. *Currently unimplemented, always 1*.
    /// * game_state: Access to the game state to perform whatever action is required.
    fn inline_callback(
        &self,
        coordinate: BlockCoordinate,
        missed_timers: u64,
        game_state: &Arc<GameState>,
    ) -> Result<()>;
}

pub enum TimerCallback {
    /// Callback operating on one block at a time. The engine may call it concurrently for multiple
    /// blocks in a chunk, or for multiple chunks. The timing of the callback may be changed between versions,
    /// but the engine will call it once per matching block per timer cycle.
    InlineLocked(Box<dyn TimerInlineCallback>),
    /// Callback that may need to access other blocks. The engine may call it concurrently for multiple
    /// blocks in a chunk, or for multiple chunks, but the callback may be subject to locking in practice.
    /// The callback will be called once per matching block per timer cycle.
    InlineUnlocked(Box<dyn TimerUnlockedCallback>),
}

/// Marker that a struct may be extended in the future
pub struct NonExhaustive(pub(crate) ());

/// Control for a map timer.
pub struct TimerSettings {
    /// The time between ticks of the timer
    pub interval: Duration,
    /// The number of shards
    pub shards: u32,
    /// How strongly to stagger the shards.
    /// 0.0 - fire all shards at approximately the same time
    /// 1.0 - spread the shards as far apart as possible
    /// Intermediate values - spread the shards out, but in a smaller span of time
    pub spreading: f64,
    /// The set of block types (*not* including variant types) that this timer should act on
    pub block_types: Vec<BlockTypeHandle>,
    // The probability that the action will be taken for each matching block. Each matching block is
    // sampled independently, using an unspecified RNG that does not derive from the game seed.
    pub per_block_probability: f64,
    pub _ne: NonExhaustive,
}
impl Default for TimerSettings {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(1),
            shards: 256,
            spreading: 1.0,
            block_types: Default::default(),
            per_block_probability: 1.0,
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
    fn spawn_shards(self: &Arc<Self>, game_state: Arc<GameState>, shard_count: u32) -> Result<()> {
        ensure!(!self.cancellation.is_cancelled());
        let mut tasks = self.tasks.blocking_lock();
        // Hacky: Delay 0.1 seconds to allow shards to start up
        let first_run = Instant::now() + Duration::from_millis(100);
        for shard_id in 0..shard_count {
            let start_time = first_run
                + (self
                    .settings
                    .interval
                    .mul_f64(self.settings.spreading * shard_id as f64 / shard_count as f64));
            {
                let cloned_self = self.clone();
                let cloned_game_state = game_state.clone();
                tasks.push(crate::spawn_async(
                    &format!("timer_{}_shard_{}", self.name, shard_id),
                    async move {
                        cloned_self
                            .run_shard(start_time, shard_id, shard_count, cloned_game_state)
                            .await
                    },
                )?);
            }
        }
        Ok(())
    }
    #[tracing::instrument(
        name = "timer_shard",
        level = "trace",
        skip(self, shard_id, total_shards, game_state),
        fields(
            timer_name = %self.name,
        )
    )]
    async fn run_shard(
        &self,
        start_time: Instant,
        shard_id: u32,
        total_shards: u32,
        game_state: Arc<GameState>,
    ) -> Result<()> {
        let block_types =
            FxHashSet::from_iter(self.settings.block_types.iter().map(|x| x.id().base_id()));

        let mut interval = tokio::time::interval_at(start_time.into(), self.settings.interval);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // Todo detect skipped ticks and adjust accordingly
        while !self.cancellation.is_cancelled() {
            tokio::select! {
                _ = interval.tick() => {
                    tokio::task::block_in_place(|| self.do_tick(shard_id, total_shards, game_state.clone(), &block_types))?;
                }
            }
        }
        Ok(())
    }

    fn do_tick(
        &self,
        shard_id: u32,
        total_shards: u32,
        game_state: Arc<GameState>,
        block_types: &FxHashSet<u32>,
    ) -> Result<()> {
        let _span = span!("timer tick");
        let mut writeback_permit = Some(game_state.map().get_writeback_permit()?);
        let mut read_lock = {
            let _span = span!("acquire game_map read lock");
            game_state.map().live_chunks.read()
        };
        let coords = read_lock.keys().cloned().collect::<Vec<_>>();
        plot!("timer tick coords", coords.len() as f64);
        for (i, coord) in coords.into_iter().enumerate() {
            if writeback_permit.is_none() {
                // A chunk used our writeback permit. Get a new one.
                // Fast path - if we can get a permit without blocking, take it
                if let Some(permit) = game_state.map().try_get_writeback_permit()? {
                    writeback_permit = Some(permit);
                } else {
                    drop(read_lock);

                    writeback_permit = Some(game_state.map().get_writeback_permit()?);

                    read_lock = {
                        let _span = span!("reacquire game_map read lock (need permit)");
                        game_state.map().live_chunks.read()
                    };
                }
            } else if i % 100 == 0 {
                let _span = span!("timer bumping");
                RwLockReadGuard::bump(&mut read_lock);
            }
            if coord.hash_u64() % total_shards as u64 == shard_id as u64 {
                if let Some(chunk) = read_lock.get(&coord) {
                    if self.settings.block_types.iter().any(|x| {
                        chunk
                            .block_bloom_filter
                            .maybe_contains(x.id().base_id() as u64)
                    }) {
                        self.handle_chunk(
                            coord,
                            chunk,
                            &game_state,
                            block_types,
                            &mut writeback_permit,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    fn handle_chunk(
        &self,
        coord: ChunkCoordinate,
        holder: &MapChunkHolder,
        game_state: &Arc<GameState>,
        block_types: &FxHashSet<u32>,
        writeback_permit: &mut Option<mpsc::Permit<'_, WritebackReq>>,
    ) -> Result<()> {
        assert!(writeback_permit.is_some());
        let mut rng = rand::thread_rng();
        let sampler = Bernoulli::new(self.settings.per_block_probability)?;
        let map = game_state.map();
        if let Some(mut chunk) = holder.try_get()? {
            assert!(chunk.block_ids.len() == 4096);
            for i in 0..4096 {
                let block_id = BlockId(chunk.block_ids[i]);
                assert!(holder
                    .block_bloom_filter
                    .maybe_contains(block_id.base_id() as u64));
                if block_types.contains(&block_id.base_id()) && sampler.sample(&mut rng) {
                    match self.run_callback(
                        holder,
                        &mut chunk,
                        ChunkOffset::from_index(i),
                        map,
                        coord,
                        game_state,
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
            if chunk.dirty {
                writeback_permit
                    .take()
                    .unwrap()
                    .send(WritebackReq::Chunk(chunk.own_coord));
            }
        }

        Ok(())
    }

    fn run_callback(
        &self,
        holder: &MapChunkHolder,
        chunk: &mut MapChunkInnerGuard<'_>,
        offset: ChunkOffset,
        map: &ServerGameMap,
        coord: ChunkCoordinate,
        game_state: &Arc<GameState>,
    ) -> Result<(), Error> {
        match &self.callback {
            TimerCallback::InlineLocked(cb) => map.mutate_block_atomically_locked(
                holder,
                chunk,
                offset,
                |block_id, extended_data| {
                    let ctx = InlineContext {
                        // todo actual ticks
                        tick: 0,
                        initiator: EventInitiator::Engine,
                        location: coord.with_offset(offset),
                        block_types: game_state.map().block_type_manager(),
                        items: game_state.item_manager(),
                    };
                    run_handler!(
                        || cb.inline_callback(
                            coord.with_offset(offset),
                            0,
                            block_id,
                            extended_data,
                            &ctx
                        ),
                        "timer_inline_locked",
                        EventInitiator::Engine
                    )?;
                    Ok(())
                },
                map,
            )?,
            TimerCallback::InlineUnlocked(cb) => run_handler!(
                || cb.inline_callback(coord.with_offset(offset), 0, game_state),
                "timer_inline_locked",
                EventInitiator::Engine
            )?,
        }
        Ok(())
    }

    async fn await_shutdown(&self) -> Result<()> {
        for task in self.tasks.lock().await.drain(..) {
            task.await??;
        }
        Ok(())
    }
}

struct TimerController {
    map: Arc<ServerGameMap>,
    game_state: Weak<GameState>,
    cancellation: CancellationToken,
    timers: FxHashMap<String, Arc<GameMapTimer>>,
}
impl TimerController {
    async fn await_shutdown(&self) -> Result<()> {
        assert!(self.cancellation.is_cancelled());
        for timer in self.timers.values() {
            timer.await_shutdown().await?;
        }

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
        let timer_controller = guard.as_mut().unwrap();
        let shards = settings.shards;
        let timer = Arc::new(GameMapTimer {
            name: name.clone(),
            callback,
            settings,
            cancellation: self.shutdown.clone(),
            tasks: tokio::sync::Mutex::new(vec![]),
        });
        timer.spawn_shards(self.game_state().clone(), shards)?;
        timer_controller.timers.insert(name, timer);
        Ok(())
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

pub(crate) mod lighting;

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
