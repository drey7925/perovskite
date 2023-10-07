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

use std::collections::HashSet;
use std::iter::once;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use crate::game_state::client_ui::PopupAction;
use crate::game_state::client_ui::PopupResponse;
use crate::game_state::event::EventInitiator;
use crate::game_state::event::HandlerContext;

use crate::game_state::event::PlayerInitiator;
use crate::game_state::game_map::BlockUpdate;
use crate::game_state::game_map::CACHE_CLEAN_MIN_AGE;
use crate::game_state::inventory::InventoryKey;
use crate::game_state::inventory::InventoryViewWithContext;
use crate::game_state::inventory::TypeErasedInventoryView;
use crate::game_state::inventory::UpdatedInventory;
use crate::game_state::items;

use crate::game_state::items::Item;

use crate::game_state::player::PlayerContext;
use crate::game_state::player::PlayerEventReceiver;
use crate::game_state::player::PlayerState;
use crate::game_state::GameState;
use crate::run_handler;

use anyhow::bail;
use anyhow::Context;
use anyhow::Error;
use anyhow::Result;
use cgmath::InnerSpace;
use cgmath::Vector3;
use cgmath::Zero;
use parking_lot::MutexGuard;
use perovskite_core::chat::ChatMessage;
use perovskite_core::coordinates::{BlockCoordinate, ChunkCoordinate, PlayerPositionUpdate};

use itertools::iproduct;
use log::error;
use log::info;
use log::warn;
use parking_lot::Mutex;
use parking_lot::RwLock;
use perovskite_core::protocol::coordinates::Angles;
use perovskite_core::protocol::game_rpc as proto;
use perovskite_core::protocol::game_rpc::stream_to_client::ServerMessage;
use perovskite_core::protocol::game_rpc::MapDeltaUpdateBatch;
use perovskite_core::protocol::game_rpc::PlayerPosition;
use perovskite_core::protocol::game_rpc::StreamToClient;
use prost::Message;
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;
use tokio::sync::{broadcast, mpsc, watch};
use tokio::task::block_in_place;
use tokio_util::sync::CancellationToken;
use tracy_client::plot;
use tracy_client::span;

static CLIENT_CONTEXT_ID_COUNTER: AtomicUsize = AtomicUsize::new(1);

pub(crate) struct PlayerCoroutinePack {
    context: Arc<SharedContext>,
    chunk_sender: MapChunkSender,
    block_event_sender: BlockEventSender,
    inventory_event_sender: InventoryEventSender,
    inbound_worker: InboundWorker,
    misc_outbound_worker: MiscOutboundWorker,
}
impl PlayerCoroutinePack {
    pub(crate) async fn run_all(mut self) -> Result<()> {
        let username = self.context.player_context.name().to_string();
        initialize_protocol_state(&self.context, &self.inbound_worker.outbound_tx).await?;

        tracing::info!("Starting workers for {}...", username);
        crate::spawn_async(&format!("inbound_worker_{}", username), async move {
            if let Err(e) = self.inbound_worker.run_inbound_loop().await {
                log::error!("Error running inbound loop: {:?}", e);
            }
        })?;
        crate::spawn_async(&format!("chunk_sender_{}", username), async move {
            if let Err(e) = self.chunk_sender.run_loop().await {
                log::error!("Error running chunk sender: {:?}", e);
            }
        })?;
        crate::spawn_async(&format!("block_event_sender_{}", username), async move {
            if let Err(e) = self.block_event_sender.run_outbound_loop().await {
                log::error!("Error running block event sender loop: {:?}", e);
            }
        })?;
        crate::spawn_async(&format!("inv_event_sender_{}", username), async move {
            if let Err(e) = self.inventory_event_sender.run_outbound_loop().await {
                log::error!("Error running inventory event sender loop: {:?}", e);
            }
        })?;
        crate::spawn_async(&format!("misc_outbound_worker_{}", username), async move {
            if let Err(e) = self.misc_outbound_worker.run_outbound_loop().await {
                log::error!("Error running misc outbound worker loop: {:?}", e);
            }
        })?;

        Ok(())
    }
}

pub(crate) async fn make_client_contexts(
    game_state: Arc<GameState>,
    player_context: PlayerContext,
    player_event_receiver: PlayerEventReceiver,
    inbound_rx: tonic::Streaming<proto::StreamToServer>,
    outbound_tx: mpsc::Sender<tonic::Result<StreamToClient>>,
) -> Result<PlayerCoroutinePack> {
    let id = CLIENT_CONTEXT_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

    let initial_position = PlayerPositionUpdate {
        position: player_context.last_position().position,
        velocity: cgmath::Vector3::zero(),
        face_direction: (0., 0.),
    };
    let (pos_send, pos_recv) = watch::channel(PositionAndPacing {
        position: initial_position,
        chunks_to_send: 16,
    });

    let cancellation = CancellationToken::new();
    // TODO add other inventories from the inventory UI layer
    let mut interested_inventories = HashSet::new();
    interested_inventories.insert(player_context.main_inventory());

    let player_context = Arc::new(player_context);

    let block_events = game_state.game_map().subscribe();
    let inventory_events = game_state.inventory_manager().subscribe();
    let chunk_tracker = Arc::new(ChunkTracker::new(
        game_state.clone(),
        player_context.clone(),
    ));

    let context = Arc::new(SharedContext {
        game_state,
        player_context,
        id,
        cancellation,
    });

    let inbound_worker = InboundWorker {
        context: context.clone(),
        inbound_rx,
        own_positions: pos_send,
        outbound_tx: outbound_tx.clone(),
        next_pos_writeback: Instant::now(),
        chunk_pacing: Aimd {
            val: INITIAL_CHUNKS_PER_UPDATE as f64,
            floor: 0.,
            ceiling: MAX_CHUNKS_PER_UPDATE as f64,
            additive_increase: 64.,
            multiplicative_decrease: 0.5,
        },
    };
    let chunk_limit_aimd = Arc::new(Mutex::new(Aimd {
        val: LOAD_LAZY_SORTED_COORDS.len() as f64,
        floor: 1024.0f64.min(LOAD_LAZY_SORTED_COORDS.len() as f64),
        ceiling: LOAD_LAZY_SORTED_COORDS.len() as f64,
        additive_increase: 256.0,
        multiplicative_decrease: 0.75,
    }));
    let chunk_sender = MapChunkSender {
        context: context.clone(),
        outbound_tx: outbound_tx.clone(),
        chunk_tracker: chunk_tracker.clone(),
        player_position: pos_recv,
        skip_if_near: Vector3::zero(),
        elements_to_skip: 0,
        snappy_encoder: snap::raw::Encoder::new(),
        snappy_input_buffer: vec![],
        snappy_output_buffer: vec![],
        chunk_limit_aimd: chunk_limit_aimd.clone(),
    };
    let block_event_sender = BlockEventSender {
        context: context.clone(),
        outbound_tx: outbound_tx.clone(),
        block_events,
        chunk_tracker,
        chunk_limit_aimd,
    };
    let inventory_event_sender = InventoryEventSender {
        context: context.clone(),
        outbound_tx: outbound_tx.clone(),
        inventory_events,
        interested_inventories,
    };

    let misc_outbound_worker = MiscOutboundWorker {
        context: context.clone(),
        player_event_receiver,
        outbound_tx: outbound_tx.clone(),
    };

    Ok(PlayerCoroutinePack {
        context,
        chunk_sender,
        block_event_sender,
        inventory_event_sender,
        inbound_worker,
        misc_outbound_worker,
    })
}

#[tracing::instrument(
    name = "initialize_protocol_state",
    skip_all,
    fields(
        player = %context.player_context.name(),
    )
)]
async fn initialize_protocol_state(
    context: &SharedContext,
    outbound_tx: &mpsc::Sender<tonic::Result<StreamToClient>>,
) -> Result<()> {
    let (hotbar_update, inv_manipulation_update) = {
        let player_state = context.player_context.state.lock();
        (
            make_inventory_update(&context.game_state, &&player_state.hotbar_inventory_view)?,
            make_inventory_update(
                &context.game_state,
                &&player_state.inventory_manipulation_view,
            )?,
        )
    };

    outbound_tx.send(Ok(hotbar_update)).await?;
    outbound_tx.send(Ok(inv_manipulation_update)).await?;
    send_all_popups(context, outbound_tx).await?;

    let message = {
        let player_state = context.player_context.state.lock();
        make_client_state_update_message(&context, player_state)?
    };
    outbound_tx
        .send(Ok(message))
        .await
        .map_err(|_| Error::msg("Could not send outbound message (initial state)"))?;
    Ok(())
}

async fn send_all_popups(
    context: &SharedContext,
    outbound_tx: &mpsc::Sender<tonic::Result<StreamToClient>>,
) -> Result<()> {
    let updates = {
        let player_state = context.player_context.state.lock();
        let mut updates = vec![];

        for popup in player_state
            .active_popups
            .iter()
            .chain(once(&player_state.inventory_popup))
        {
            for view in popup.inventory_views().values() {
                updates.push(make_inventory_update(
                    &context.game_state,
                    &InventoryViewWithContext {
                        view,
                        context: popup,
                    },
                )?);
            }
        }
        updates
    };
    for update in updates {
        outbound_tx
            .send(Ok(update))
            .await
            .with_context(|| "Could not send outbound message (inventory update)")?;
    }

    Ok(())
}

pub struct SharedContext {
    game_state: Arc<GameState>,
    player_context: Arc<PlayerContext>,
    id: usize,
    cancellation: CancellationToken,
}
impl Drop for SharedContext {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

/// Tracks what chunks are close enough to the player to be of interest
pub(crate) struct ChunkTracker {
    player_context: Arc<PlayerContext>,
    game_state: Arc<GameState>,
    loaded_chunks_bloom: cbloom::Filter,
    // The client knows about these chunks, and we should send updates to them
    loaded_chunks: RwLock<FxHashSet<ChunkCoordinate>>,
    // todo later - chunks that the client might have unloaded, but might be worth sending on a best-effort basis
    // todo - move AIMD pacing into this
}
impl ChunkTracker {
    pub(crate) fn new(game_state: Arc<GameState>, player_context: Arc<PlayerContext>) -> Self {
        Self {
            game_state,
            player_context,
            // Bloom filter size is given in bytes, not bits - but many formulae and calculators use bits.
            // According to https://hur.st/bloomfilter/?n=50000&p=&m=64%20KiB&k=, we can expect a reasonable false positive rate here
            // for our current supported draw distances.
            //
            // We'll probably get more false positives from chunks being unloaded.
            //
            // TODO consider clearing and regenerating the bloom filter when false positive rate goes up due to unloaded chunks still
            // present in the filter
            loaded_chunks_bloom: cbloom::Filter::new(
                65536,
                ((LOAD_LAZY_DISTANCE as usize).pow(3) / 2).max(1),
            ),
            loaded_chunks: RwLock::new(FxHashSet::default()),
        }
    }
    fn is_loaded(&self, coord: ChunkCoordinate) -> bool {
        if !self.loaded_chunks_bloom.maybe_contains(coord.hash_u64()) {
            return false;
        }
        self.loaded_chunks.read().contains(&coord)
    }
    // Marks a chunk as loaded. This must be called before the chunk is actually loaded and sent to the client
    fn mark_chunk_loaded(&self, coord: ChunkCoordinate) {
        self.loaded_chunks_bloom.insert(coord.hash_u64());
        self.loaded_chunks.write().insert(coord);
    }

    // Marks a chunk as unloaded. This must be called after the chunk is actually unloaded and the corresponding message is sent
    // to the client
    fn mark_chunk_unloaded(&self, player_coord: ChunkCoordinate) {
        self.loaded_chunks.write().remove(&player_coord);
    }

    fn mark_chunks_unloaded(&self, chunks: impl Iterator<Item = ChunkCoordinate>) {
        let mut write_lock = self.loaded_chunks.write();
        for coord in chunks {
            write_lock.remove(&coord);
        }
    }

    fn clear(&self) {
        self.loaded_chunks_bloom.clear();
        self.loaded_chunks.write().clear();
    }
}

// Loads and unloads player chunks
pub(crate) struct MapChunkSender {
    context: Arc<SharedContext>,

    outbound_tx: mpsc::Sender<tonic::Result<StreamToClient>>,
    chunk_tracker: Arc<ChunkTracker>,

    player_position: watch::Receiver<PositionAndPacing>,
    skip_if_near: cgmath::Vector3<f64>,
    elements_to_skip: usize,

    snappy_encoder: snap::raw::Encoder,
    snappy_input_buffer: Vec<u8>,
    snappy_output_buffer: Vec<u8>,

    chunk_limit_aimd: Arc<Mutex<Aimd>>,
}
const ACCESS_TIME_BUMP_SHARDS: u32 = 32;
impl MapChunkSender {
    #[tracing::instrument(
        name = "MapChunkSender loop",
        level = "trace",
        skip(self),
        fields(
            player_name = %self.context.player_context.name(),
        ),
    )]
    pub(crate) async fn run_loop(&mut self) -> Result<()> {
        assert!(CACHE_CLEAN_MIN_AGE > (Duration::from_secs_f64(0.2) * ACCESS_TIME_BUMP_SHARDS));
        let mut access_time_bump_idx = 0;
        while !self.context.cancellation.is_cancelled() {
            tokio::select! {
                _ = self.player_position.changed() => {
                    access_time_bump_idx = (access_time_bump_idx + 1) % ACCESS_TIME_BUMP_SHARDS;
                    let update = *self.player_position.borrow_and_update();
                    self.handle_position_update(update, access_time_bump_idx).await?;
                }
                _ = self.context.cancellation.cancelled() => {
                    info!("Client outbound loop {} detected cancellation and shutting down", self.context.id)
                }
                _ = self.context.game_state.await_start_shutdown() => {
                    info!("Game shutting down, disconnecting {}", self.context.id);
                    // cancel the inbound context as well
                    self.context.cancellation.cancel();
                }
            };
        }
        Ok(())
    }

    fn snappy_encode<T>(&mut self, msg: &T) -> Result<Vec<u8>>
    where
        T: Message,
    {
        self.snappy_input_buffer.clear();
        msg.encode(&mut self.snappy_input_buffer)?;
        let compressed_len_bound = snap::raw::max_compress_len(self.snappy_input_buffer.len());
        if compressed_len_bound == 0 {
            bail!("Input is too long to compress");
        }
        if self.snappy_output_buffer.len() < compressed_len_bound {
            self.snappy_output_buffer.resize(compressed_len_bound, 0);
        }
        let actual_compressed_len = self
            .snappy_encoder
            .compress(&self.snappy_input_buffer, &mut self.snappy_output_buffer)?;
        Ok(self.snappy_output_buffer[0..actual_compressed_len].to_vec())
    }

    #[tracing::instrument(
        name = "HandlePositionUpdate",
        level = "trace",
        skip(self, update, bump_index),
        fields(
        player_name = %self.context.player_context.name(),
            budget = %update.chunks_to_send,
            player_chunk
        ),
    )]
    async fn handle_position_update(
        &mut self,
        update: PositionAndPacing,
        bump_index: u32,
    ) -> Result<()> {
        let position = update.position;
        // TODO anticheat/safety checks
        // TODO consider caching more in the player's movement direction as a form of prefetch???
        let player_block_coord: BlockCoordinate = match position.position.try_into() {
            Ok(x) => x,
            Err(e) => {
                log::warn!(
                    "Player had invalid position: {:?}, error {:?}",
                    position.position,
                    e
                );
                // TODO teleport the player
                BlockCoordinate::new(0, 0, 0)
            }
        };
        let player_chunk = player_block_coord.chunk();
        tracing::Span::current().record("player_chunk", format!("{:?}", player_chunk));

        // Phase 1: Unload chunks that are too far away
        let chunks_to_unsubscribe: Vec<_> = self
            .chunk_tracker
            .loaded_chunks
            .read()
            .iter()
            .filter(|&x| self.should_unload(player_chunk, *x))
            .cloned()
            .collect();
        let message = proto::StreamToClient {
            tick: self.context.game_state.tick(),
            server_message: Some(proto::stream_to_client::ServerMessage::MapChunkUnsubscribe(
                proto::MapChunkUnsubscribe {
                    chunk_coord: chunks_to_unsubscribe.iter().map(|&x| x.into()).collect(),
                },
            )),
        };
        self.outbound_tx
            .send(Ok(message))
            .await
            .map_err(|_| Error::msg("Could not send outbound message (mapchunk unsubscribe)"))?;
        self.chunk_tracker
            .mark_chunks_unloaded(chunks_to_unsubscribe.into_iter());

        // Phase 2: Load chunks that are close enough.
        // Chunks are expensive to load and send, so we keep track of
        let mut sent_chunks = 0;

        let _skip = if (self.skip_if_near - position.position).magnitude2() < 256.0 {
            self.elements_to_skip
        } else {
            0
        };

        let start_time = Instant::now();
        for (i, &(dx, dy, dz)) in LOAD_LAZY_SORTED_COORDS.iter().enumerate().skip(0) {
            let coord = ChunkCoordinate {
                x: player_chunk.x.saturating_add(dx),
                y: player_chunk.y.saturating_add(dy),
                z: player_chunk.z.saturating_add(dz),
            };
            let distance = dx.abs() + dy.abs() + dz.abs();
            if !coord.is_in_bounds() {
                continue;
            }
            if self.player_position.has_changed().unwrap_or(false) && distance > FORCE_LOAD_DISTANCE
            {
                // If we already have a new position update, restart the process with the new position
                tracing::event!(
                    tracing::Level::TRACE,
                    "Got a new position update, restarting the load process"
                );
                break;
            }
            let mut chunk_needs_reload = false;
            if coord.hash_u64() % (ACCESS_TIME_BUMP_SHARDS as u64) == (bump_index as u64)
                && !block_in_place(|| self.context.game_state.game_map().bump_chunk(coord))
            {
                // chunk wasn't in the map, so we need to reload it
                chunk_needs_reload = true;
            }

            if self.chunk_tracker.is_loaded(coord) && !chunk_needs_reload {
                continue;
            }
            // We load chunks as long as they're close enough and the map system
            // isn't overloaded. If the map system is overloaded, we'll only load
            // chunks that are close enough to the player to really matter.
            let should_load = distance <= LOAD_EAGER_DISTANCE
                && (distance <= FORCE_LOAD_DISTANCE
                    || !self.context.game_state.game_map().in_pushback());
            if distance > LOAD_EAGER_DISTANCE && start_time.elapsed() > Duration::from_millis(250) {
                self.elements_to_skip = i;
                break;
            }
            let chunk_data = tokio::task::block_in_place(|| {
                self.context
                    .game_state
                    .game_map()
                    .serialize_for_client(coord, should_load)
            })?;
            if let Some(chunk_data) = chunk_data {
                let chunk_bytes = self.snappy_encode(&chunk_data)?;
                let message = proto::StreamToClient {
                    tick: self.context.game_state.tick(),
                    server_message: Some(proto::stream_to_client::ServerMessage::MapChunk(
                        proto::MapChunk {
                            chunk_coord: Some(coord.into()),
                            snappy_encoded_bytes: chunk_bytes,
                        },
                    )),
                };
                self.outbound_tx.send(Ok(message)).await.map_err(|_| {
                    self.context.cancellation.cancel();
                    Error::msg("Could not send outbound message (full mapchunk)")
                })?;
                self.chunk_tracker.mark_chunk_loaded(coord);
                sent_chunks += 1;
                if sent_chunks > update.chunks_to_send {
                    break;
                }
            }
        }
        self.skip_if_near = position.position;
        self.chunk_limit_aimd.lock().increase();
        Ok(())
    }

    fn should_unload(&self, player_chunk: ChunkCoordinate, chunk: ChunkCoordinate) -> bool {
        player_chunk.manhattan_distance(chunk) > UNLOAD_DISTANCE as u32
    }
}

// Sends block updates to the client
pub(crate) struct BlockEventSender {
    context: Arc<SharedContext>,

    // RPC stream is sent via this channel
    outbound_tx: mpsc::Sender<tonic::Result<proto::StreamToClient>>,

    // All updates to the map from all sources, not yet filtered by location (BlockEventSender is
    // responsible for filtering)
    block_events: broadcast::Receiver<BlockUpdate>,
    chunk_tracker: Arc<ChunkTracker>,

    // AIMD tracker used to control chunks to be sent to the client
    chunk_limit_aimd: Arc<Mutex<Aimd>>,
}
impl BlockEventSender {
    #[tracing::instrument(
        name = "BlockEventOutboundLoop",
        level = "info",
        skip(self),
        fields(
            player_name = %self.context.player_context.name(),
        )
    )]
    pub(crate) async fn run_outbound_loop(&mut self) -> Result<()> {
        while !self.context.cancellation.is_cancelled() {
            tokio::select! {
                block_event = self.block_events.recv() => {
                    self.handle_block_update(block_event).await?;
                }
                _ = self.context.cancellation.cancelled() => {
                    info!("Client outbound loop {} detected cancellation and shutting down", self.context.id)
                    // pass
                }
            };
        }
        Ok(())
    }

    #[tracing::instrument(
        name = "HandleBlockUpdate",
        level = "trace",
        skip(self, update),
        fields(
            player_name = %self.context.player_context.name(),
        )
    )]
    async fn handle_block_update(
        &mut self,
        update: Result<BlockUpdate, broadcast::error::RecvError>,
    ) -> Result<()> {
        // Sleep 10 msec to allow some more updates to be aggregated
        tokio::time::sleep(Duration::from_millis(10)).await;
        let update = match update {
            Err(broadcast::error::RecvError::Lagged(num_pending)) => {
                tracing::warn!(
                    "Client {} is lagged, {} pending",
                    self.context.id,
                    num_pending
                );
                // This client context is lagging behind and lost block updates.
                // Fall back and get it resynced
                return self.handle_block_update_lagged().await;
            }
            Err(broadcast::error::RecvError::Closed) => {
                self.context.cancellation.cancel();
                return Ok(());
            }
            Ok(x) => x,
        };
        let mut updates = FxHashMap::default();
        if self.chunk_tracker.is_loaded(update.location.chunk()) {
            updates.insert(update.location, update);
        }
        // Drain and batch as many updates as possible
        // TODO coalesce
        while updates.len() < MAX_UPDATE_BATCH_SIZE {
            match self.block_events.try_recv() {
                Ok(update) => {
                    if self.chunk_tracker.is_loaded(update.location.chunk()) {
                        // last update wins
                        updates.insert(update.location, update);
                    }
                }
                Err(broadcast::error::TryRecvError::Empty) => break,
                Err(e) => {
                    // we'll deal with it the next time the main loop runs
                    warn!("Unexpected error from block events broadcast: {:?}", e);
                }
            }
        }
        plot!("block updates", updates.len() as f64);
        tracing::event!(
            tracing::Level::TRACE,
            "Sending {} block updates",
            updates.len()
        );

        let mut update_protos = Vec::new();
        for update in updates.values() {
            update_protos.push(proto::MapDeltaUpdate {
                block_coord: Some(update.location.into()),
                new_id: update.new_value.into(),
            })
        }

        if !update_protos.is_empty() {
            let message = proto::StreamToClient {
                tick: self.context.game_state.tick(),
                server_message: Some(proto::stream_to_client::ServerMessage::MapDeltaUpdate(
                    MapDeltaUpdateBatch {
                        updates: update_protos,
                    },
                )),
            };
            self.outbound_tx
                .send(Ok(message))
                .await
                .with_context(|| "Could not send outbound message (block update)")?;
        }
        Ok(())
    }

    async fn handle_block_update_lagged(&mut self) -> Result<()> {
        // Decrease the number of chunks we're willing to send
        self.chunk_limit_aimd.lock().decrease();

        // this ends up racy. resubscribe first, so we get duplicate/pointless events after the
        // resubscribe, rather than missing events if we resubscribe to the broadcast after sending current
        // chunk states
        self.block_events.resubscribe();
        self.chunk_tracker.clear();
        Ok(())
    }

    pub(crate) async fn teleport_player(&mut self, location: Vector3<f64>) -> Result<()> {
        let mut player_state = self.context.player_context.state.lock();
        player_state.last_position = PlayerPositionUpdate {
            position: location,
            velocity: Vector3::zero(),
            face_direction: (0., 0.),
        };
        let message = make_client_state_update_message(&self.context, player_state);
        self.outbound_tx
            .send(Ok(message?))
            .await
            .map_err(|_| Error::msg("Could not send outbound message (updated state)"))
    }
}

fn make_client_state_update_message(
    ctx: &SharedContext,
    player_state: MutexGuard<'_, PlayerState>,
) -> Result<StreamToClient> {
    let message = {
        let position = player_state.last_position.position;

        let (time_now, day_len) = {
            let time_state = ctx.game_state.time_state().lock();
            (time_state.time_of_day(), time_state.day_length())
        };
        StreamToClient {
            tick: ctx.game_state.tick(),
            server_message: Some(proto::stream_to_client::ServerMessage::ClientState(
                proto::SetClientState {
                    position: Some(PlayerPosition {
                        position: Some(position.try_into()?),
                        velocity: Some(Vector3::zero().try_into()?),
                        face_direction: Some(Angles {
                            deg_azimuth: 0.,
                            deg_elevation: 0.,
                        }),
                    }),
                    hotbar_inventory_view: player_state.hotbar_inventory_view.id.0,
                    inventory_popup: Some(player_state.inventory_popup.to_proto()),
                    inventory_manipulation_view: player_state.inventory_manipulation_view.id.0,
                    time_of_day: time_now,
                    day_length_sec: day_len.as_secs_f64(),
                },
            )),
        }
    };

    Ok(message)
}

pub(crate) struct InventoryEventSender {
    context: Arc<SharedContext>,
    // RPC stream is sent via this channel
    outbound_tx: mpsc::Sender<tonic::Result<proto::StreamToClient>>,

    // TODO consider delta updates for this
    inventory_events: broadcast::Receiver<UpdatedInventory>,
    interested_inventories: HashSet<InventoryKey>,
}
impl InventoryEventSender {
    #[tracing::instrument(
        name = "InventoryOutboundLoop",
        level = "info",
        skip(self),
        fields(
            player_name = %self.context.player_context.name(),
        )
    )]
    pub(crate) async fn run_outbound_loop(&mut self) -> Result<()> {
        while !self.context.cancellation.is_cancelled() {
            tokio::select! {
                inventory_update = self.inventory_events.recv() => {
                    self.handle_inventory_update(inventory_update).await?;
                }
                _ = self.context.cancellation.cancelled() => {
                    info!("Client outbound loop {} detected cancellation and shutting down", self.context.id)
                    // pass
                }
            };
        }
        Ok(())
    }

    async fn handle_inventory_update(
        &mut self,
        update: Result<UpdatedInventory, broadcast::error::RecvError>,
    ) -> Result<()> {
        let key = match update {
            Err(broadcast::error::RecvError::Lagged(x)) => {
                log::error!("Client {} is lagged, {} pending", self.context.id, x);
                // TODO resync in the future? Right now we just kick the client off
                // A client that's desynced on inventory updates is struggling, so not sure
                // what we can do
                tracing::error!("Client {} is lagged, {} pending", self.context.id, x);
                self.inventory_events.resubscribe();
                return Ok(());
            }
            Err(broadcast::error::RecvError::Closed) => {
                self.context.cancellation.cancel();
                return Ok(());
            }
            Ok(x) => x,
        };
        let mut update_keys = FxHashSet::default();
        update_keys.insert(key);

        tracing::span!(tracing::Level::TRACE, "gather inventory updates").in_scope(|| {
            for _ in 0..256 {
                match self.inventory_events.try_recv() {
                    Ok(update) => {
                        update_keys.insert(update);
                    }
                    Err(broadcast::error::TryRecvError::Empty) => break,
                    Err(e) => {
                        // we'll deal with it the next time the main loop runs
                        warn!("Unexpected error from inventory events broadcast: {:?}", e);
                    }
                }
            }
        });
        let update_messages = tracing::span!(
            tracing::Level::TRACE,
            "filter and serialize inventory updates"
        )
        .in_scope(|| -> anyhow::Result<_> {
            let player_state = self.context.player_context.state.lock();
            let mut update_messages = vec![];
            for key in update_keys {
                if player_state.hotbar_inventory_view.wants_update_for(&key) {
                    update_messages.push(make_inventory_update(
                        &self.context.game_state,
                        &&player_state.hotbar_inventory_view,
                    )?);
                }

                for popup in player_state
                    .active_popups
                    .iter()
                    .chain(once(&player_state.inventory_popup))
                {
                    for view in popup.inventory_views().values() {
                        if view.wants_update_for(&key) {
                            update_messages.push(make_inventory_update(
                                &self.context.game_state,
                                &InventoryViewWithContext {
                                    view,
                                    context: popup,
                                },
                            )?);
                        }
                    }
                }
            }
            Ok(update_messages)
        })?;

        for update in update_messages {
            self.outbound_tx
                .send(Ok(update))
                .await
                .with_context(|| "Could not send outbound message (inventory update)")?;
        }

        Ok(())
    }
}

// State/structure backing a gRPC GameStream on the inbound side
pub(crate) struct InboundWorker {
    context: Arc<SharedContext>,

    // RPC stream is received via this channel
    inbound_rx: tonic::Streaming<proto::StreamToServer>,
    // Some inbound loop operations don't actually require the outbound loop.
    // Note that there are some issues with race conditions here, especially around
    // handled_sequence messages getting sent befoe the outbound loop can send the actual
    // in-game effects for them. If this poses an issue, refactor later.
    outbound_tx: mpsc::Sender<tonic::Result<proto::StreamToClient>>,
    // The client's self-reported position
    own_positions: watch::Sender<PositionAndPacing>,
    next_pos_writeback: Instant,

    chunk_pacing: Aimd,
}
impl InboundWorker {
    // Poll for world events and send them through outbound_tx
    pub(crate) async fn run_inbound_loop(&mut self) -> Result<()> {
        while !self.context.cancellation.is_cancelled() {
            tokio::select! {
                message = self.inbound_rx.message() => {
                    match message {
                        Err(e) => {
                            warn!("Client {}, Failure reading inbound message: {:?}", self.context.id, e)
                        },
                        Ok(None) => {
                            info!("Client {} disconnected", self.context.id);
                            self.context.cancellation.cancel();
                            return Ok(());
                        }
                        Ok(Some(message)) => {
                            match self.handle_message(&message).await {
                                Ok(_) => {},
                                Err(e) => {
                                    warn!("Client {} failed to handle message: {:?}, error: {:?}", self.context.id, message, e);
                                    // TODO notify the client once there's a chat or server->client error handling message
                                },
                            }
                        }
                    }

                }
                _ = self.context.cancellation.cancelled() => {
                    info!("Client inbound context {} detected cancellation and shutting down", self.context.id)
                    // pass
                }
            };
        }
        Ok(())
    }

    async fn handle_message(&mut self, message: &proto::StreamToServer) -> Result<()> {
        // todo do something with the client tick once we define ticks
        match &message.client_message {
            None => {
                warn!(
                    "Client context {} got empty/unknown message from client",
                    self.context.id
                );
            }
            Some(proto::stream_to_server::ClientMessage::Dig(dig_message)) => {
                // TODO check whether the current item can dig this block, and whether
                // it's been long enough since the last dig
                let coord: BlockCoordinate = dig_message
                    .block_coord
                    .as_ref()
                    .map(|x| x.into())
                    .context("Missing block_coord")?;
                self.run_map_handlers(
                    coord,
                    dig_message.item_slot,
                    |item| {
                        item.and_then(|x| x.dig_handler.as_deref())
                            .unwrap_or(&items::default_dig_handler)
                    },
                    dig_message
                        .position
                        .as_ref()
                        .context("Missing player position")?
                        .try_into()?,
                )
                .await?;
            }
            Some(proto::stream_to_server::ClientMessage::Tap(tap_message)) => {
                let coord: BlockCoordinate = tap_message
                    .block_coord
                    .as_ref()
                    .map(|x| x.into())
                    .context("Missing block_coord")?;
                self.run_map_handlers(
                    coord,
                    tap_message.item_slot,
                    |item| {
                        item.and_then(|x| x.tap_handler.as_deref())
                            .unwrap_or(&items::default_tap_handler)
                    },
                    tap_message
                        .position
                        .as_ref()
                        .context("Missing player position")?
                        .try_into()?,
                )
                .await?;
            }
            Some(proto::stream_to_server::ClientMessage::PositionUpdate(pos_update)) => {
                self.handle_pos_update(message.client_tick, pos_update)?;
            }
            Some(proto::stream_to_server::ClientMessage::BugCheck(bug_check)) => {
                error!("Client bug check: {:?}", bug_check);
            }
            Some(proto::stream_to_server::ClientMessage::Place(place_message)) => {
                self.handle_place(place_message).await?;
            }
            Some(proto::stream_to_server::ClientMessage::Inventory(inventory_message)) => {
                self.handle_inventory_action(inventory_message).await?;
            }
            Some(proto::stream_to_server::ClientMessage::PopupResponse(response)) => {
                self.handle_popup_response(response).await?;
            }
            Some(proto::stream_to_server::ClientMessage::InteractKey(interact_key)) => {
                self.handle_interact_key(interact_key).await?;
            }
            Some(proto::stream_to_server::ClientMessage::ChatMessage(message)) => {
                self.context
                    .game_state
                    .chat()
                    .handle_inbound_chat_message(
                        self.context.player_context.make_initiator(),
                        self.context.game_state.clone(),
                        message,
                    )
                    .await?;
            }
            Some(_) => {
                warn!(
                    "Unimplemented client->server message {:?} on context {}",
                    message, self.context.id
                );
            }
        }
        // todo decide whether we should send nacks on error, or just complain about them
        // to the local log
        // also decide whether position updates merit an ack
        self.send_ack(message.sequence).await?;
        Ok(())
    }

    #[tracing::instrument(
        name = "map_handler",
        level = "trace",
        skip(self, coord, selected_inv_slot, get_item_handler),
        fields(
            player_name = %self.context.player_context.name(),
        ),
    )]
    async fn run_map_handlers<F>(
        &mut self,
        coord: BlockCoordinate,
        selected_inv_slot: u32,
        get_item_handler: F,
        player_position: PlayerPositionUpdate,
    ) -> Result<()>
    where
        F: FnOnce(Option<&Item>) -> &items::BlockInteractionHandler,
    {
        tokio::task::block_in_place(|| {
            self.map_handler_sync(selected_inv_slot, get_item_handler, coord, player_position)
        })
    }

    fn map_handler_sync<F>(
        &mut self,
        selected_inv_slot: u32,
        get_item_handler: F,
        coord: BlockCoordinate,
        player_position: PlayerPositionUpdate,
    ) -> std::result::Result<(), anyhow::Error>
    where
        F: FnOnce(Option<&Item>) -> &items::BlockInteractionHandler,
    {
        let game_state = &self.context.game_state;

        game_state.inventory_manager().mutate_inventory_atomically(
            &self.context.player_context.main_inventory(),
            |inventory| {
                let stack = inventory
                    .contents_mut()
                    .get_mut(selected_inv_slot as usize)
                    .with_context(|| "Item slot was out of bounds")?;

                let initiator = EventInitiator::Player(PlayerInitiator {
                    player: &self.context.player_context,
                    position: player_position,
                });

                let item_handler =
                    get_item_handler(game_state.item_manager().from_stack(stack.as_ref()));

                let result = {
                    let ctx = HandlerContext {
                        tick: game_state.tick(),
                        initiator: initiator.clone(),
                        game_state: game_state.clone(),
                    };
                    run_handler!(
                        || {
                            item_handler(
                                ctx,
                                coord,
                                stack.as_ref().unwrap_or(&items::NO_TOOL_STACK),
                            )
                        },
                        "item dig handler",
                        &initiator,
                    )?
                };
                *stack = result.updated_tool;
                let mut leftover = vec![];
                for stack in result.obtained_items {
                    if let Some(x) = inventory.try_insert(stack) {
                        leftover.push(x)
                    }
                }
                if !leftover.is_empty() {
                    // This will require support for non-block things in the world
                    error!("Dropped items from dig_block not empty; TODO/FIXME handle those items");
                }
                Ok(())
            },
        )
    }

    async fn send_ack(&mut self, sequence: u64) -> Result<()> {
        if sequence == 0 {
            return Ok(());
        };
        let message = proto::StreamToClient {
            tick: self.context.game_state.tick(),
            server_message: Some(proto::stream_to_client::ServerMessage::HandledSequence(
                sequence,
            )),
        };

        self.outbound_tx.send(Ok(message)).await.map_err(|_| {
            self.context.cancellation.cancel();
            Error::msg("Could not send outbound message (sequence ack)")
        })?;

        Ok(())
    }

    fn handle_pos_update(&mut self, _tick: u64, update: &proto::ClientUpdate) -> Result<()> {
        match &update.position {
            Some(pos_update) => {
                let (az, el) = match &pos_update.face_direction {
                    Some(x) => (x.deg_azimuth, x.deg_elevation),
                    None => {
                        log::warn!("No angles in update from client");
                        (0., 0.)
                    }
                };
                let pos = PlayerPositionUpdate {
                    position: pos_update
                        .position
                        .as_ref()
                        .context("Missing position")?
                        .try_into()?,
                    velocity: pos_update
                        .velocity
                        .as_ref()
                        .context("Missing velocity")?
                        .try_into()?,
                    face_direction: (az, el),
                };

                if let Some(pacing) = &update.pacing {
                    if pacing.pending_chunks < 256 {
                        self.chunk_pacing.increase();
                    } else if pacing.pending_chunks > 1024 {
                        self.chunk_pacing.decrease();
                    }
                }

                self.own_positions.send_replace(PositionAndPacing {
                    position: pos,
                    chunks_to_send: self.chunk_pacing.get(),
                });
                self.context.player_context.update_client_position_state(pos, update.hotbar_slot);
            }
            None => {
                warn!("No position in update message from client");
            }
        }
        Ok(())
    }

    #[tracing::instrument(
        name = "place_action",
        level = "trace",
        skip(self),
        fields(
            player_name = %self.context.player_context.name(),
        ),
    )]
    async fn handle_place(&mut self, place_message: &proto::PlaceAction) -> Result<()> {
        tokio::task::block_in_place(|| {
            let _span = span!("handle_place");
            self.context
                .game_state
                .inventory_manager()
                .mutate_inventory_atomically(
                    &self.context.player_context.main_inventory(),
                    |inventory| {
                        let stack = inventory
                            .contents_mut()
                            .get_mut(place_message.item_slot as usize)
                            .with_context(|| "Item slot was out of bounds")?;

                        let initiator = EventInitiator::Player(PlayerInitiator {
                            player: &self.context.player_context,
                            position: place_message
                                .position
                                .as_ref()
                                .context("Missing position")?
                                .try_into()?,
                        });

                        let handler = self
                            .context
                            .game_state
                            .item_manager()
                            .from_stack(stack.as_ref())
                            .and_then(|x| x.place_handler.as_deref());
                        if let Some(handler) = handler {
                            let ctx = HandlerContext {
                                tick: self.context.game_state.tick(),
                                initiator: initiator.clone(),
                                game_state: self.context.game_state.clone(),
                            };
                            let new_stack = run_handler!(
                                || {
                                    handler(
                                        ctx,
                                        place_message
                                            .block_coord
                                            .clone()
                                            .with_context(|| {
                                                "Missing block_coord in place message"
                                            })?
                                            .into(),
                                        place_message
                                            .anchor
                                            .clone()
                                            .with_context(|| "Missing anchor in place message")?
                                            .into(),
                                        stack.as_ref().unwrap(),
                                    )
                                },
                                "item_place",
                                &initiator,
                            )?;
                            *stack = new_stack;
                        }
                        Ok(())
                    },
                )
        })
    }

    #[tracing::instrument(
        name = "inventory_action",
        level = "trace",
        skip(self),
        fields(
            player_name = %self.context.player_context.name(),
        ),
    )]
    async fn handle_inventory_action(&mut self, action: &proto::InventoryAction) -> Result<()> {
        if action.source_view == action.destination_view {
            log::error!(
                "Cannot handle an inventory action with the same source and destination view"
            );
            // todo send the user an error
            return Ok(());
        }
        let updates =
            {
                let mut views_to_send = HashSet::new();
                block_in_place(|| -> anyhow::Result<Vec<StreamToClient>> {
                    let mut player_state = self.context.player_context.player.state.lock();

                    player_state.handle_inventory_action(action)?;

                    for popup in player_state
                        .active_popups
                        .iter()
                        .chain(once(&player_state.inventory_popup))
                    {
                        if popup.inventory_views().values().any(|x| {
                            x.id.0 == action.source_view || x.id.0 == action.destination_view
                        }) {
                            for view in popup.inventory_views().values() {
                                views_to_send.insert(view.id);
                            }
                        }
                    }

                    let mut updates = vec![];
                    for view in views_to_send {
                        updates.push(make_inventory_update(
                            &self.context.game_state,
                            player_state.find_inv_view(view)?.as_ref(),
                        )?);
                    }

                    anyhow::Result::Ok(updates)
                })?
            };
        for update in updates {
            self.outbound_tx
                .send(Ok(update))
                .await
                .map_err(|_| Error::msg("Could not send outbound message (inventory update)"))?;
        }

        Ok(())
    }

    #[tracing::instrument(
        name = "popup_response",
        level = "trace",
        skip(self, action),
        fields(
            player_name = %self.context.player_context.name(),
            id = %action.popup_id,
            was_closed = %action.closed,
            button = %action.clicked_button,
        ),
    )]
    async fn handle_popup_response(
        &mut self,
        action: &perovskite_core::protocol::ui::PopupResponse,
    ) -> Result<()> {
        let user_action = if action.closed {
            PopupAction::PopupClosed
        } else {
            PopupAction::ButtonClicked(action.clicked_button.clone())
        };
        let updates = tokio::task::block_in_place(|| -> anyhow::Result<_> {
            let _span = span!("handle_popup_response");
            let mut player_state = self.context.player_context.state.lock();
            let mut updates = vec![];
            if action.closed {
                player_state
                    .inventory_manipulation_view
                    .clear_if_transient(Some(self.context.player_context.main_inventory_key))?;
                updates.push(make_inventory_update(
                    &self.context.game_state,
                    &&player_state.inventory_manipulation_view,
                )?);
            }
            if let Some(popup) = player_state
                .active_popups
                .iter_mut()
                .find(|x| x.id() == action.popup_id)
            {
                popup.handle_response(
                    PopupResponse {
                        user_action,
                        textfield_values: action.text_fields.clone(),
                    },
                    self.context.player_context.main_inventory(),
                )?;
                for view in popup.inventory_views().values() {
                    updates.push(make_inventory_update(
                        &self.context.game_state,
                        &InventoryViewWithContext {
                            view,
                            context: popup,
                        },
                    )?);
                }
            } else if player_state.inventory_popup.id() == action.popup_id {
                player_state.inventory_popup.handle_response(
                    PopupResponse {
                        user_action,
                        textfield_values: action.text_fields.clone(),
                    },
                    self.context.player_context.main_inventory(),
                )?;
                for view in player_state.inventory_popup.inventory_views().values() {
                    updates.push(make_inventory_update(
                        &self.context.game_state,
                        &InventoryViewWithContext {
                            view,
                            context: &player_state.inventory_popup,
                        },
                    )?);
                }
            } else {
                log::error!(
                    "Got popup response for nonexistent popup {:?}",
                    action.popup_id
                );
            }
            if action.closed {
                player_state
                    .active_popups
                    .retain(|x| x.id() != action.popup_id)
            }
            // drop before async calls
            drop(player_state);
            anyhow::Result::Ok(updates)
        })?;
        for update in updates {
            self.outbound_tx
                .send(Ok(update))
                .await
                .map_err(|_| Error::msg("Could not send outbound message (popup update)"))?;
        }

        Ok(())
    }

    #[tracing::instrument(
        name = "interact_key",
        level = "trace",
        skip(self),
        fields(
            player_name = %self.context.player_context.name(),
        ),
    )]
    async fn handle_interact_key(
        &mut self,
        interact_message: &proto::InteractKeyAction,
    ) -> Result<()> {
        let messages = tokio::task::block_in_place(|| -> anyhow::Result<_> {
            self.handle_interact_key_sync(interact_message)
        })?;
        for message in messages {
            self.outbound_tx
                .send(Ok(message))
                .await
                .map_err(|_| Error::msg("Could not send outbound message (interact key)"))?;
        }
        Ok(())
    }

    fn handle_interact_key_sync(
        &mut self,
        interact_message: &proto::InteractKeyAction,
    ) -> Result<Vec<StreamToClient>> {
        let coord: BlockCoordinate = interact_message
            .block_coord
            .as_ref()
            .map(|x| x.into())
            .with_context(|| "Missing block_coord")?;
        let initiator = EventInitiator::Player(PlayerInitiator {
            player: &self.context.player_context,
            position: interact_message
                .position
                .as_ref()
                .context("Missing position")?
                .try_into()?,
        });
        let ctx = HandlerContext {
            tick: self.context.game_state.tick(),
            initiator: initiator.clone(),
            game_state: self.context.game_state.clone(),
        };
        let block = self.context.game_state.game_map().get_block(coord)?;
        let mut messages = vec![];
        if let Some(handler) = &self
            .context
            .game_state
            .game_map()
            .block_type_manager()
            .get_block(&block)
            .unwrap()
            .0
            .interact_key_handler
        {
            if let Some(popup) =
                run_handler!(|| (handler)(ctx, coord), "interact_key", &initiator,)?
            {
                messages.push(StreamToClient {
                    tick: self.context.game_state.tick(),
                    server_message: Some(ServerMessage::ShowPopup(popup.to_proto())),
                });
                for view in popup.inventory_views().values() {
                    messages.push(make_inventory_update(
                        &self.context.game_state,
                        &InventoryViewWithContext {
                            view,
                            context: &popup,
                        },
                    )?)
                }
                self.context
                    .player_context
                    .player
                    .state
                    .lock()
                    .active_popups
                    .push(popup);
            }
        }
        Ok(messages)
    }
}
impl Drop for InboundWorker {
    fn drop(&mut self) {
        self.context.cancellation.cancel();
    }
}

/// Handles non-performance-critical messages that don't require a dedicated
/// coroutine, e.g. chat
struct MiscOutboundWorker {
    outbound_tx: mpsc::Sender<tonic::Result<StreamToClient>>,
    player_event_receiver: PlayerEventReceiver,
    context: Arc<SharedContext>,
}
impl MiscOutboundWorker {
    async fn run_outbound_loop(&mut self) -> Result<()> {
        let mut broadcast_messages = self.context.game_state.chat().subscribe();
        let mut resync_player_state_global =
            self.context.game_state.subscribe_player_state_resyncs();
        while !self.context.cancellation.is_cancelled() {
            tokio::select! {
                message = self.player_event_receiver.chat_messages.recv() => {
                    match message {
                        Some(message) => {
                        self.transmit_chat_message(message).await?;
                        },
                        None => {
                            tracing::warn!("Chat message sender disconnected")
                        }
                    }
                },
                message = self.player_event_receiver.disconnection_message.recv() => {
                    match message {
                        Some(message) => {
                            self.outbound_tx.send(Ok(StreamToClient {
                                tick: self.context.game_state.tick(),
                                server_message: Some(ServerMessage::ShutdownMessage(message)),
                            })).await?;
                        },
                        None => {
                            tracing::warn!("Disconnection message sender disconnected")
                        }
                    }
                    // Even if the player is using a hacked client, we will disconnect
                    // them on our end
                    self.context.cancellation.cancel();
                },
                _ = resync_player_state_global.changed() => {
                    let lock = self.context.player_context.player.state.lock();
                    self.outbound_tx.send(Ok(make_client_state_update_message(&self.context, lock)?)).await?;
                },
                _ = self.player_event_receiver.reinit_player_state.changed() => {
                    let lock = self.context.player_context.player.state.lock();
                    self.outbound_tx.send(Ok(make_client_state_update_message(&self.context, lock)?)).await?;
                }
                message = broadcast_messages.recv() => {
                    match message {
                        Ok(message) => {
                            self.transmit_chat_message(message).await?;
                        },
                        Err(broadcast::error::RecvError::Lagged(_)) => {
                            self.transmit_chat_message(ChatMessage::new("[system]", "Your chat connection is lagging severely. Some chat messages may be lost").with_color((255, 0, 0))).await?;
                            broadcast_messages = broadcast_messages.resubscribe();
                        },
                        Err(broadcast::error::RecvError::Closed) => {
                            self.transmit_chat_message(ChatMessage::new("[system]", "Internal error - chat connection closed").with_color((255, 0, 0))).await?;
                            tracing::error!("Chat broadcast channel closed");
                            break
                        }
                    }
                }
                _ = self.context.game_state.await_start_shutdown() => {
                    info!("Game shutting down, disconnecting {}", self.context.id);
                    self.outbound_tx.send(Ok(StreamToClient {
                        tick: self.context.game_state.tick(),
                        server_message: Some(ServerMessage::ShutdownMessage("Game server is shutting down.".to_string())),
                    })).await?;
                    // cancel the inbound context as well
                    self.context.cancellation.cancel();
                },
                _ = self.context.cancellation.cancelled() => {
                    info!("Client misc outbound context {} detected cancellation and shutting down", self.context.id)
                    // pass
                }
            }
        }
        Ok(())
    }
    async fn transmit_chat_message(&mut self, message: ChatMessage) -> Result<()> {
        self.outbound_tx
            .send(Ok(StreamToClient {
                tick: self.context.game_state.tick(),
                server_message: Some(ServerMessage::ChatMessage(proto::ChatMessage {
                    origin: message.origin().to_string(),
                    color_argb: message.origin_color_fixed32(),
                    message: message.text().to_string(),
                })),
            }))
            .await?;
        Ok(())
    }
}

// TODO tune these and make them adjustable via settings
// Units of chunks
// Chunks within this distance will be loaded into memory if not yet loaded
const LOAD_EAGER_DISTANCE: i32 = 20;
// Chunks within this distance will be sent if they are already loaded into memory
const LOAD_LAZY_DISTANCE: i32 = 25;
const UNLOAD_DISTANCE: i32 = 50;
// Chunks within this distance will be sent, even if flow control would otherwise prevent them from being sent
const FORCE_LOAD_DISTANCE: i32 = 3;

const MAX_UPDATE_BATCH_SIZE: usize = 256;

const INITIAL_CHUNKS_PER_UPDATE: usize = 128;
const MAX_CHUNKS_PER_UPDATE: usize = 256;

lazy_static::lazy_static! {
    static ref LOAD_LAZY_ZIGZAG_VEC: Vec<i32> = {
        let mut v = vec![0];
        for i in 1..=LOAD_LAZY_DISTANCE {
            v.push(i);
            v.push(-i);
        }
        v
    };

}
lazy_static::lazy_static! {
    static ref LOAD_LAZY_SORTED_COORDS: Vec<(i32, i32, i32)> = {
        let mut v = vec![];
        for (&x, &y, &z) in iproduct!(LOAD_LAZY_ZIGZAG_VEC.iter(), LOAD_LAZY_ZIGZAG_VEC.iter(), LOAD_LAZY_ZIGZAG_VEC.iter()) {
            v.push((x, y, z));
        }
        v.sort_by_key(|(x, y, z)| x.abs() + (4 * y.abs()) + z.abs());
        v.retain(|x| (x.0 + x.1 + x.2) <= LOAD_LAZY_DISTANCE);
        v
    };
}

#[derive(Copy, Clone, Debug)]
struct PositionAndPacing {
    position: PlayerPositionUpdate,
    chunks_to_send: usize,
}

struct Aimd {
    val: f64,
    floor: f64,
    ceiling: f64,
    additive_increase: f64,
    multiplicative_decrease: f64,
}
impl Aimd {
    fn increase(&mut self) {
        self.val = (self.val + self.additive_increase).clamp(self.floor, self.ceiling);
    }
    fn decrease(&mut self) {
        debug_assert!(self.multiplicative_decrease < 1.);
        self.val = (self.val * self.multiplicative_decrease).clamp(self.floor, self.ceiling);
    }
    fn get(&self) -> usize {
        self.val as usize
    }
}

fn make_inventory_update(
    game_state: &GameState,
    view: &dyn TypeErasedInventoryView,
) -> Result<StreamToClient> {
    block_in_place(|| {
        Ok(StreamToClient {
            tick: game_state.tick(),
            server_message: Some(ServerMessage::InventoryUpdate(view.to_client_proto()?)),
        })
    })
}
