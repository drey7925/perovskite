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
use rustc_hash::FxHashMap;
use std::ops::{Deref, DerefMut};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    sync::{Arc, Weak},
    time::{Duration, Instant},
};
use tokio_util::sync::CancellationToken;

use crate::{
    database::database_engine::{GameDatabase, KeySpace},
    game_state::inventory::Inventory,
};

use super::{
    blocks::{
        self, BlockTypeHandle, BlockTypeManager, ExtDataHandling, ExtendedData, ExtendedDataHolder,
        InlineContext, TryAsHandle,
    },
    event::{EventInitiator, HandlerContext},
    handlers::run_handler,
    items::ItemStack,
    GameState,
};

use cuberef_core::{
    coordinates::{BlockCoordinate, ChunkCoordinate, ChunkOffset},
    protocol::map as mapchunk_proto,
};
use parking_lot::{
    Condvar, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockUpgradableReadGuard,
    RwLockWriteGuard,
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
    // once that is implemented)
    Client,
    // Chunk is for the server and its database. Extended data is present.
    Server,
}

// todo figure out the mapgen API and lock this down if necessary
pub struct MapChunk {
    own_coord: ChunkCoordinate,
    // TODO: this was exposed for the temporary mapgen API. Lock this down and refactor
    // set_block and related to allow an API to set blocks when given a &mut MapChunk
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
        let proto = mapchunk_proto::StoredChunk::decode(bytes)
            .with_context(|| "MapChunk proto serialization failed")?;
        match proto.chunk_data {
            Some(mapchunk_proto::stored_chunk::ChunkData::V1(chunk_data)) => {
                parse_v1(chunk_data, coordinate, game_state)
            }
            None => bail!("Missing chunk_data or unrecognized format"),
        }
    }
    /// Sets the block at th given coordinate within the chunk.
    /// This function is intended to be used from map generators, and may
    /// lead to data loss or inconsistency when used elsewhere
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
                        .collect::<Result<HashMap<_, _>>>()?,
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
                        .collect::<Result<HashMap<_, _>>>()?,
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
}
impl MapChunkHolder {
    fn new_empty() -> Self {
        Self {
            chunk: Mutex::new(HolderState::Empty),
            condition: Condvar::new(),
            last_accessed: Mutex::new(Instant::now()),
        }
    }
    /// Get the chunk, blocking until it's loaded
    fn wait_and_get(&self) -> Result<MapChunkInnerGuard<'_>> {
        let mut guard = self.chunk.lock();

        loop {
            self.bump_access_time();
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
        self.bump_access_time();
        match &*guard {
            HolderState::Empty => Ok(None),
            HolderState::Err(e) => Err(Error::msg(format!("Chunk load failed: {e:?}"))),
            HolderState::Ok(_) => Ok(Some(MapChunkInnerGuard { guard })),
        }
    }
    /// Set the chunk, and notify any threads waiting in wait_and_get
    fn fill(&self, chunk: MapChunk) {
        *self.chunk.lock() = HolderState::Ok(chunk);
        self.condition.notify_all();
    }
    /// Set the chunk to an error, and notify any waiting threads so they can propagate the error
    fn set_err(&self, err: anyhow::Error) {
        *self.chunk.lock() = HolderState::Err(err);
    }

    fn bump_access_time(&self) {
        *self.last_accessed.lock() = Instant::now();
    }
}

struct MapChunkOuterGuard<'a> {
    // TODO: This has a slight inefficiency - we have to call get() on the map an extra time
    read_guard: RwLockReadGuard<'a, HashMap<ChunkCoordinate, MapChunkHolder>>,
    coord: ChunkCoordinate,
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
    // Initial unoptimized version: A global mutex for the whole map.
    // Possible optimizations:
    // * Sharding
    // * DashMap
    // * Inner vs outer locks https://gist.github.com/drey7925/c4b89ff1d15c635875619ce19a749914
    live_chunks: RwLock<HashMap<ChunkCoordinate, MapChunkHolder>>,
    block_type_manager: Arc<BlockTypeManager>,
    block_update_sender: broadcast::Sender<BlockUpdate>,
    writeback_sender: mpsc::Sender<WritebackReq>,
    shutdown: CancellationToken,
    writeback_handle: Mutex<Option<JoinHandle<Result<()>>>>,
    cleanup_handle: Mutex<Option<JoinHandle<Result<()>>>>,
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
            game_state,
            database,
            live_chunks: RwLock::new(HashMap::new()),
            block_type_manager,
            block_update_sender,
            writeback_sender,
            shutdown: cancellation.clone(),
            writeback_handle: None.into(),
            cleanup_handle: None.into(),
        });

        let mut writeback = GameMapWriteback {
            map: result.clone(),
            receiver: writeback_receiver,
            cancellation: cancellation.clone(),
        };
        let writeback_handle = tokio::spawn(async move { writeback.run_loop().await });
        *result.writeback_handle.lock() = Some(writeback_handle);

        let mut cache_cleanup = MapCacheCleanup {
            map: result.clone(),
            cancellation,
        };
        let cleanup_handle = tokio::spawn(async move { cache_cleanup.run_loop().await });
        *result.cleanup_handle.lock() = Some(cleanup_handle);
        Ok(result)
    }

    pub(crate) fn bump_access_time(&self, coord: ChunkCoordinate) {
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
        self.enqueue_writeback(coord.chunk())?;
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
        self.enqueue_writeback(coord.chunk())?;
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
    pub fn mutate_block_atomically<F, T>(&self, coord: BlockCoordinate, mutator: F) -> Result<T>
    where
        F: FnOnce(&mut BlockTypeHandle, &mut ExtendedDataHolder) -> Result<T>,
    {
        let chunk_guard = self.get_chunk(coord.chunk())?;
        let mut chunk = chunk_guard.wait_and_get()?;

        let mut extended_data = chunk
            .extended_data
            .remove(&coord.offset().as_index().try_into().unwrap());

        let mut data_holder = ExtendedDataHolder::new(&mut extended_data);
        let old_id = chunk.block_ids[coord.offset().as_index()].into();
        let mut block_type = self.block_type_manager().make_blockref(old_id)?;
        let closure_result = mutator(&mut block_type, &mut data_holder);
        let dirty = data_holder.dirty();

        if let Some(new_data) = extended_data {
            chunk
                .extended_data
                .insert(coord.offset().as_index().try_into().unwrap(), new_data);
        }
        let new_id = block_type.id();
        if new_id != old_id {
            chunk.block_ids[coord.offset().as_index()] = new_id.into();
            chunk.dirty = true;
        }
        if dirty {
            chunk.dirty = true;
        }
        if chunk.dirty {
            self.enqueue_writeback(coord.chunk())?;
        }
        if new_id != old_id {
            self.broadcast_block_change(BlockUpdate {
                location: coord,
                new_value: block_type,
            });
        }

        closure_result
    }

    /// Digs a block, running its on-dig event handler. The items it drops are returned.
    /// Note that while tool is passed to this function, the tool's dig handler has *already*
    /// run.
    pub fn dig_block(
        &self,
        coord: BlockCoordinate,
        initiator: EventInitiator,
        tool: Option<&ItemStack>,
    ) -> Result<Vec<ItemStack>> {
        self.run_block_interaction(
            coord,
            initiator,
            tool,
            |block| block.dig_handler_inline.as_deref(),
            |block| block.dig_handler_full.as_deref(),
        )
    }

    pub fn run_block_interaction<F, G>(
        &self,
        coord: BlockCoordinate,
        initiator: EventInitiator,
        tool: Option<&ItemStack>,
        get_block_inline_handler: F,
        get_block_full_handler: G,
    ) -> Result<Vec<ItemStack>>
    where
        F: FnOnce(&blocks::BlockType) -> Option<&blocks::InlineHandler>,
        G: FnOnce(&blocks::BlockType) -> Option<&blocks::FullHandler>,
    {
        let game_state = self.game_state();
        let tick = game_state.tick();
        let (blocktype, mut drops) = self.mutate_block_atomically(coord, |block, ext_data| {
            let (blocktype, _) = self.block_type_manager().get_block(block)?;

            let mut drops = Vec::new();
            if let Some(ref inline_handler) = get_block_inline_handler(blocktype) {
                let ctx = InlineContext {
                    tick,
                    initiator: initiator.clone(),
                    location: coord,
                    block_types: self.block_type_manager(),
                    items: game_state.item_manager(),
                };
                drops.append(&mut run_handler(
                    || (inline_handler)(ctx, block, ext_data, tool),
                    "block_inline",
                    initiator.clone(),
                )?);
            };
            Ok((blocktype, drops))
        })?;
        // we need this to happen outside of mutate_block_atomically (which holds a chunk lock) to avoid a deadlock.
        if let Some(full_handler) = get_block_full_handler(blocktype) {
            let ctx = HandlerContext {
                tick,
                initiator: initiator.clone(),
                game_state: self.game_state(),
            };
            drops.append(&mut run_handler(
                || (full_handler)(ctx, coord, tool),
                "block_full",
                initiator.clone(),
            )?);
        }

        Ok(drops)
    }

    pub(crate) fn block_type_manager(&self) -> &BlockTypeManager {
        &self.block_type_manager
    }

    // Gets a chunk, loading it from database/generating it if it is not in memory
    fn get_chunk<'a>(&'a self, coord: ChunkCoordinate) -> Result<MapChunkOuterGuard<'a>> {
        // Start with an upgradable read
        let lock = self.live_chunks.upgradable_read();

        if lock.contains_key(&coord) {
            return Ok(MapChunkOuterGuard {
                read_guard: RwLockUpgradableReadGuard::downgrade(lock),
                coord,
            });
        }

        // The chunk is not loaded. Upgrade the lock temporarily, get an entry into the map, and then fill it
        // under a read lock
        let mut write_lock = RwLockUpgradableReadGuard::upgrade(lock);
        write_lock.insert(coord, MapChunkHolder::new_empty());

        // We are the thread that inserted the chunk. Downgrade back to a read lock, and fill the chunk before returning.
        let read_lock = RwLockWriteGuard::downgrade(write_lock);
        // Unwrap is safe, since we just inserted the chunk
        let chunk_guard = read_lock.get(&coord).unwrap();

        // We assert this is unwind-safe: it only interacts with the database and mapgen, which themselves shouldn't have
        // weird unwinding behavior. TODO validate this
        match catch_unwind(AssertUnwindSafe(|| {
            self.load_uncached_or_generate_chunk(coord)
        })) {
            Ok(Ok(chunk)) => {
                chunk_guard.fill(chunk);
                Ok(MapChunkOuterGuard {
                    read_guard: read_lock,
                    coord,
                })
            }
            Ok(Err(e)) => {
                let message = Error::msg(format!("Failed to load chunk: {e:?}"));
                chunk_guard.set_err(e);
                Err(message)
            }
            Err(e) => {
                chunk_guard.set_err(Error::msg(format!("Chunk load/generate panicked: {e:?}")));
                // Unfortunate duplication, anyhow::Error is not Clone
                Err(Error::msg(format!("Chunk load/generate panicked: {e:?}")))
            }
        }
    }

    // Loads the chunk from the DB, or generates it if missing, REGARDLESS of whether the chunk
    // is in memory.
    // If the chunk is already in memory, this may cause data loss by reading a stale instance
    // from the DB.
    fn load_uncached_or_generate_chunk(&self, coord: ChunkCoordinate) -> Result<MapChunk> {
        let data = self
            .database
            .get(&KeySpace::MapchunkData.make_key(&coord.as_bytes()))?;
        if let Some(data) = data {
            return MapChunk::deserialize(coord, &data, self.game_state());
        }

        let mut chunk = MapChunk::new(coord, self.game_state());
        chunk.dirty = true;
        self.game_state().mapgen().fill_chunk(coord, &mut chunk);
        // This will only execute after the mutex is released.
        self.enqueue_writeback(coord)?;
        Ok(chunk)
    }

    pub fn game_state(&self) -> Arc<GameState> {
        Weak::upgrade(&self.game_state).unwrap()
    }

    fn unload_chunk_locked(
        &self,
        lock: &mut RwLockWriteGuard<HashMap<ChunkCoordinate, MapChunkHolder>>,
        coord: ChunkCoordinate,
    ) -> Result<()> {
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
        match self.block_update_sender.send(update) {
            Ok(_) => {}
            Err(_) => warn!("No receivers for block update {:?}", update),
        }
    }
    fn enqueue_writeback(&self, chunk: ChunkCoordinate) -> Result<()> {
        if self.shutdown.is_cancelled() {
            return Err(Error::msg("Writeback thread was shut down"));
        }
        match self
            .writeback_sender
            .blocking_send(WritebackReq::Chunk(chunk))
        {
            Ok(_) => Ok(()),
            Err(_) => {
                log::error!("Couldn't enqueue the writeback; data loss may occur");
                Err(Error::msg("enqueue_writeback failed"))
            }
        }
    }

    /// Create a receiver that is notified of changes to all block IDs (including variant changes).
    /// This receiver will not obtain messages for changes to extended data.
    pub(crate) fn subscribe(&self) -> broadcast::Receiver<BlockUpdate> {
        self.block_update_sender.subscribe()
    }

    /// Get a chunk from the map and return its client proto if either load_if_missing is true, or
    /// the chunk is already cached in memory. If load_if_missing is true, the chunk will be kept in
    /// memory after this function returns.
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
            let guard = self.live_chunks.read();
            let chunk_holder = guard.get(&coord);
            if let Some(chunk_holder) = chunk_holder {
                match chunk_holder.try_get()? {
                    Some(x) => Ok(Some(x.serialize(ChunkUsage::Client)?)),
                    None => Ok(None),
                }
            } else {
                Ok(None)
            }
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
const CACHE_CLEAN_MIN_AGE: Duration = Duration::from_secs(15);
const CACHE_CLEAN_INTERVAL: Duration = Duration::from_secs(3);
const CACHE_CLEANUP_KEEP_N_RECENTLY_USED: usize = 128;
const CACHE_CLEANUP_RELOCK_EVERY_N: usize = 32;

struct MapCacheCleanup {
    map: Arc<ServerGameMap>,
    cancellation: CancellationToken,
}
impl MapCacheCleanup {
    async fn run_loop(&mut self) -> Result<()> {
        while !self.cancellation.is_cancelled() {
            let deadline = Instant::now() + CACHE_CLEAN_INTERVAL;
            tokio::select! {
                _ = tokio::time::sleep_until(deadline.into()) => {
                    let start = Instant::now();
                    tokio::task::block_in_place(|| self.do_cleanup())?;
                    let _time = Instant::now() - start;
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
        loop {
            let now = Instant::now();
            let mut lock = self.map.live_chunks.write();
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
const BROADCAST_CHANNEL_SIZE: usize = 128;
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
            info!("Writing back {} chunks", writebacks.len());
            if writebacks.len() >= (WRITEBACK_COALESCE_MAX_SIZE) * 4 {
                warn!("Writeback backlog of {} chunks is unusually high; is the writeback thread falling behind?", writebacks.len());
            }
            tokio::task::block_in_place(|| self.do_writebacks(writebacks))?;
        }

        log::info!("Map writeback exiting");
        Ok(())
    }

    fn do_writebacks(&self, writebacks: Vec<ChunkCoordinate>) -> Result<()> {
        let lock = self.map.live_chunks.read();
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
        let mut dedup = Vec::new();
        let mut seen = HashSet::new();
        for entry in entries {
            if seen.insert(entry) {
                dedup.push(entry)
            }
        }
        dedup.reverse();
        Some(dedup)
    }
}
impl Drop for GameMapWriteback {
    fn drop(&mut self) {
        self.cancellation.cancel()
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
