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
use std::ops::RangeInclusive;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use crate::game_state::client_ui::PopupAction;
use crate::game_state::client_ui::PopupResponse;
use crate::game_state::entities::IterEntity;
use crate::game_state::entities::Movement;
use crate::game_state::event::EventInitiator;
use crate::game_state::event::HandlerContext;
use crate::game_state::event::{log_trace, WeakPlayerRef};
use crate::game_state::event::{run_traced, run_traced_sync};

use crate::game_state::event::PlayerInitiator;
use crate::game_state::game_map::BlockUpdate;
use crate::game_state::game_map::CACHE_CLEAN_MIN_AGE;
use crate::game_state::inventory::InventoryKey;
use crate::game_state::inventory::InventoryViewWithContext;
use crate::game_state::inventory::TypeErasedInventoryView;
use crate::game_state::inventory::UpdatedInventory;
use crate::game_state::items;

use crate::game_state::items::{default_generic_handler, Item};

use crate::game_state::player::PlayerState;
use crate::game_state::player::{Player, PlayerEventReceiver};
use crate::game_state::player::{PlayerContext, PlayerEvent};
use crate::game_state::GameState;
use crate::{run_handler, spawn_async};

use anyhow::bail;
use anyhow::Context;
use anyhow::Error;
use anyhow::Result;
use cgmath::Vector3;
use cgmath::Zero;
use parking_lot::MutexGuard;
use perovskite_core::chat::ChatMessage;
use perovskite_core::constants::permissions;
use perovskite_core::coordinates::{BlockCoordinate, ChunkCoordinate, PlayerPositionUpdate};

use crate::game_state::audio_crossbar::{AudioCrossbarReceiver, AudioEvent, AudioInstruction};
use either::Either;
use itertools::iproduct;
use parking_lot::Mutex;
use parking_lot::RwLock;
use perovskite_core::game_actions::ToolTarget;
use perovskite_core::protocol::audio::{SampledSoundPlayback, SoundSource};
use perovskite_core::protocol::coordinates as coords_proto;
use perovskite_core::protocol::coordinates::Vec3D;
use perovskite_core::protocol::entities as entities_proto;
use perovskite_core::protocol::game_rpc as proto;
use perovskite_core::protocol::game_rpc::dig_tap_action::ActionTarget;
use perovskite_core::protocol::game_rpc::interact_key_action::InteractionTarget;
use perovskite_core::protocol::game_rpc::stream_to_client::ServerMessage;
use perovskite_core::protocol::game_rpc::PlayerPosition;
use perovskite_core::protocol::game_rpc::StreamToClient;
use perovskite_core::protocol::game_rpc::{MapDeltaUpdateBatch, ServerPerformanceMetrics};
use perovskite_core::util::{LogInspect, TraceBuffer};
use prost::Message;
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{broadcast, mpsc, watch};
use tokio::task::block_in_place;
use tokio_util::sync::CancellationToken;
use tracing::error;
use tracing::info;
use tracing::warn;
use tracy_client::plot;
use tracy_client::span;

use super::grpc_service::SERVER_MAX_PROTOCOL_VERSION;

static CLIENT_CONTEXT_ID_COUNTER: AtomicUsize = AtomicUsize::new(1);

pub(crate) struct PlayerCoroutinePack {
    context: Arc<SharedContext>,
    chunk_senders: [MapChunkSender; 2],
    net_prioritizer: NetPrioritizer,
    block_event_sender: BlockEventSender,
    inventory_event_sender: InventoryEventSender,
    inbound_worker: InboundWorker,
    misc_outbound_worker: MiscOutboundWorker,
    entity_sender: EntityEventSender,
    audio_sender: AudioSender,
}
impl PlayerCoroutinePack {
    pub(crate) async fn run_all(mut self) -> Result<()> {
        let username = self.context.player_context.name().to_string();
        initialize_protocol_state(&self.context, &self.inbound_worker.outbound_tx).await?;
        self.context
            .game_state
            .game_behaviors()
            .on_player_join
            .handle(
                &self.context.player_context,
                HandlerContext {
                    tick: self.context.game_state.tick(),
                    initiator: EventInitiator::Engine,
                    game_state: self.context.game_state.clone(),
                },
            )
            .await?;

        tracing::info!("Starting workers for {}...", username);

        let context_clone = self.context.clone();
        let kick_closure = move |reason: &'static str| async move {
            let result = context_clone.player_context.kick_player(reason).await;
            context_clone.cancellation.cancel();
            match result {
                Ok(()) => {}
                Err(e) => {
                    tracing::warn!("Error kicking player: {:?}", e);
                }
            }
        };
        let kick = kick_closure.clone();
        crate::spawn_async(&format!("net_prio_{}", username), async move {
            if let Err(e) = self.net_prioritizer.run_loop().await {
                tracing::error!("Error running outbound network priotizier: {:?}", e);
                kick("net prio crashed").await;
            }
        })?;

        let kick = kick_closure.clone();
        crate::spawn_async(&format!("inbound_worker_{}", username), async move {
            if let Err(e) = self.inbound_worker.inbound_worker_loop().await {
                tracing::error!("Error running inbound loop: {:?}", e);
                kick("Inbound loop crashed").await;
            }
        })?;

        for (i, mut sender) in self.chunk_senders.into_iter().enumerate() {
            let kick = kick_closure.clone();
            crate::spawn_async(&format!("chunk_sender_{i}_{username}"), async move {
                if let Err(e) = sender.chunk_sender_loop().await {
                    tracing::error!("Error running chunk sender: {:?}", e);
                    kick("Chunk sender loop crashed").await;
                }
            })?;
        }

        let kick = kick_closure.clone();
        crate::spawn_async(&format!("block_event_sender_{}", username), async move {
            if let Err(e) = self.block_event_sender.block_sender_loop().await {
                tracing::error!("Error running block event sender loop: {:?}", e);
                kick("Chunk sender loop crashed").await;
            }
        })?;

        let kick = kick_closure.clone();
        crate::spawn_async(&format!("inv_event_sender_{}", username), async move {
            if let Err(e) = self.inventory_event_sender.inv_sender_loop().await {
                tracing::error!("Error running inventory event sender loop: {:?}", e);
                kick("inventory event loop crashed").await;
            }
        })?;

        let kick = kick_closure.clone();
        crate::spawn_async(&format!("misc_outbound_worker_{}", username), async move {
            if let Err(e) = self.misc_outbound_worker.misc_outbound_worker_loop().await {
                tracing::error!("Error running misc outbound worker loop: {:?}", e);
                // cancellations are handled in the misc outbound worker
                kick("Misc outbound worker crashed").await;
            }
        })?;

        let kick = kick_closure.clone();
        crate::spawn_async(&format!("entity_sender_{}", username), async move {
            if let Err(e) = self.entity_sender.entity_sender_loop().await {
                tracing::error!("Error running entity sender worker loop: {:?}", e);
                kick("Entity sender crashed").await;
            }
        })?;

        let kick = kick_closure.clone();
        crate::spawn_async(&format!("audio_sender_{}", username), async move {
            if let Err(e) = self.audio_sender.audio_sender_loop().await {
                tracing::error!("Error running audio sender worker loop: {:?}", e);
                kick("Audio sender crashed").await;
            }
        })?;

        // No shutdown actions at the moment
        // let cancellation = self.context.cancellation.clone();
        // crate::spawn_async(&format!("shutdown_actions_{}", username), async move {
        //     cancellation.cancelled().await;
        //     self.context
        //         .game_state
        //         .entities()
        //         .remove(self.context.player_context.entity_id)
        //         .await;
        // })?;

        Ok(())
    }
}

pub(crate) async fn make_client_contexts(
    game_state: Arc<GameState>,
    player_context: PlayerContext,
    player_event_receiver: PlayerEventReceiver,
    inbound_rx: tonic::Streaming<proto::StreamToServer>,
    outbound_tx: mpsc::Sender<tonic::Result<StreamToClient>>,
    effective_protocol_version: u32,
) -> Result<PlayerCoroutinePack> {
    let id = CLIENT_CONTEXT_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

    let initial_position = PlayerPositionUpdate {
        position: player_context.last_position().position,
        velocity: Vector3::zero(),
        face_direction: (0., 0.),
    };
    let (pos_send, pos_recv) = watch::channel(PositionAndPacing {
        position: initial_position,
        chunks_to_send: 16,
        client_requested_render_distance: 20,
    });
    let pos_recv_clone = pos_recv.clone();

    let cancellation = CancellationToken::new();
    // TODO add other inventories from the inventory UI layer
    let mut interested_inventories = HashSet::new();
    interested_inventories.insert(player_context.main_inventory());

    let player_context = Arc::new(player_context);

    let block_events = game_state.game_map().subscribe();
    let inventory_events = game_state.inventory_manager().subscribe();
    let chunk_tracker = Arc::new(ChunkTracker::new());
    let chunk_aimd = Arc::new(Mutex::new(Aimd {
        val: INITIAL_CHUNKS_PER_UPDATE as f64,
        floor: 0.,
        ceiling: MAX_CHUNKS_PER_UPDATE as f64,
        additive_increase: 64.,
        multiplicative_decrease: 0.5,
    }));
    let context = Arc::new(SharedContext {
        game_state: game_state.clone(),
        player_context,
        id,
        cancellation,
        effective_protocol_version,
        chunk_aimd,
        enable_performance_metrics: AtomicBool::new(false),
    });

    let inbound_worker = InboundWorker {
        context: context.clone(),
        inbound_rx,
        own_positions: pos_send,
        outbound_tx: outbound_tx.clone(),
        next_pos_writeback: Instant::now(),
    };
    let chunk_sender_0 = MapChunkSender {
        context: context.clone(),
        outbound_tx: outbound_tx.clone(),
        chunk_tracker: chunk_tracker.clone(),
        player_position: pos_recv.clone(),
        skip_if_near: ChunkCoordinate { x: 0, y: 0, z: 0 },
        processed_elements: 0,
        snappy: SnappyEncoder {
            snappy_encoder: snap::raw::Encoder::new(),
            snappy_input_buffer: vec![],
            snappy_output_buffer: vec![],
        },
        range_hint_lookaside: Default::default(),
        hinted_chunks: vec![],
        shard_parity: 0,
    };
    let chunk_sender_1 = MapChunkSender {
        context: context.clone(),
        outbound_tx: outbound_tx.clone(),
        chunk_tracker: chunk_tracker.clone(),
        player_position: pos_recv,
        skip_if_near: ChunkCoordinate { x: 0, y: 0, z: 0 },
        processed_elements: 0,
        snappy: SnappyEncoder {
            snappy_encoder: snap::raw::Encoder::new(),
            snappy_input_buffer: vec![],
            snappy_output_buffer: vec![],
        },
        range_hint_lookaside: Default::default(),
        hinted_chunks: vec![],
        shard_parity: 1,
    };

    let (entity_tx, entity_rx) = mpsc::channel(16);
    let (map_tx, map_rx) = mpsc::channel(16);
    let (prio_tx, prio_rx) = mpsc::channel(16);

    let net_prioritizer = NetPrioritizer {
        context: context.clone(),
        map_rx,
        entity_rx,
        prio_rx,
        tx: outbound_tx,
    };

    let block_event_sender = BlockEventSender {
        context: context.clone(),
        outbound_tx: map_tx.clone(),
        block_events,
        chunk_tracker,
    };
    let inventory_event_sender = InventoryEventSender {
        context: context.clone(),
        outbound_tx: prio_tx.clone(),
        inventory_events,
        interested_inventories,
    };

    let misc_outbound_worker = MiscOutboundWorker {
        context: context.clone(),
        player_event_receiver,
        outbound_tx: prio_tx.clone(),
    };

    let entity_sender = EntityEventSender {
        context: context.clone(),
        outbound_tx: entity_tx.clone(),
        sent_entities: FxHashMap::default(),
        last_update: 0,
    };

    let audio_sender = AudioSender {
        context: context.clone(),
        outbound_tx: prio_tx.clone(),
        events: context.game_state.audio().subscribe(),
        position_watch: pos_recv_clone,
    };

    Ok(PlayerCoroutinePack {
        context,
        chunk_senders: [chunk_sender_0, chunk_sender_1],
        net_prioritizer,
        block_event_sender,
        inventory_event_sender,
        inbound_worker,
        misc_outbound_worker,
        entity_sender,
        audio_sender,
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
    if !context.player_context.has_permission(permissions::LOG_IN) {
        context.cancellation.cancel();
        outbound_tx
            .send(Ok(StreamToClient {
                tick: context.game_state.tick(),
                server_message: Some(ServerMessage::ShutdownMessage(
                    "You don't have permission to log in".to_string(),
                )),
                performance_metrics: context.maybe_get_performance_metrics(),
            }))
            .await?;
    }
    let (hotbar_update, inv_manipulation_update) = {
        let player_state = context.player_context.state.lock();
        (
            make_inventory_update(&context, &&player_state.hotbar_inventory_view)?,
            make_inventory_update(&context, &&player_state.inventory_manipulation_view)?,
        )
    };

    outbound_tx.send(Ok(hotbar_update)).await?;
    outbound_tx.send(Ok(inv_manipulation_update)).await?;
    send_all_popups(context, outbound_tx).await?;

    let message = {
        let player_state = context.player_context.state.lock();
        make_client_state_update_message(context, player_state, true)?
    };
    outbound_tx
        .send(Ok(message))
        .await
        .map_err(|_| Error::msg("Could not send outbound message (initial state)"))?;

    if context.effective_protocol_version != SERVER_MAX_PROTOCOL_VERSION {
        context.player_context.send_chat_message_async(ChatMessage::new_server_message(
            "Your client is out of date and you may not be able to use all server features. Please consider updating.".to_string()
        )).await?;
    }
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
            .chain(player_state.inventory_popup.iter())
        {
            for view in popup.inventory_views().values() {
                updates.push(make_inventory_update(
                    &context,
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
    effective_protocol_version: u32,
    chunk_aimd: Arc<Mutex<Aimd>>,
    enable_performance_metrics: AtomicBool,
}
impl SharedContext {
    fn maybe_get_performance_metrics(&self) -> Option<ServerPerformanceMetrics> {
        if self.enable_performance_metrics.load(Ordering::Relaxed) {
            Some(self.game_state.performance_metrics_proto())
        } else {
            None
        }
    }
}
impl Drop for SharedContext {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

/// Tracks what chunks are close enough to the player to be of interest
pub(crate) struct ChunkTracker {
    loaded_chunks_bloom: cbloom::Filter,
    // The client knows about these chunks, and we should send updates to them
    loaded_chunks: RwLock<FxHashSet<ChunkCoordinate>>,
    // todo later - chunks that the client might have unloaded, but might be worth sending on a best-effort basis
    // todo - move AIMD pacing into this
}
impl ChunkTracker {
    pub(crate) fn new() -> Self {
        Self {
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
        self.read().contains(&coord)
    }
    // Marks a chunk as loaded. This must be called before the chunk is actually loaded and sent to the client
    fn mark_chunk_loaded(&self, coord: ChunkCoordinate) {
        self.loaded_chunks_bloom.insert(coord.hash_u64());
        self.write().insert(coord);
    }

    // Marks a chunk as unloaded. This must be called after the chunk is actually unloaded and the corresponding message is sent
    // to the client
    fn mark_chunk_unloaded(&self, player_coord: ChunkCoordinate) {
        self.write().remove(&player_coord);
    }

    fn write(&self) -> parking_lot::RwLockWriteGuard<FxHashSet<ChunkCoordinate>> {
        tokio::task::block_in_place(|| self.loaded_chunks.write())
    }
    fn read(&self) -> parking_lot::RwLockReadGuard<FxHashSet<ChunkCoordinate>> {
        tokio::task::block_in_place(|| self.loaded_chunks.read())
    }

    fn mark_chunks_unloaded(&self, chunks: impl Iterator<Item = ChunkCoordinate>) {
        let mut write_lock = self.write();
        for coord in chunks {
            write_lock.remove(&coord);
        }
    }

    fn clear(&self) {
        self.loaded_chunks_bloom.clear();
        self.write().clear();
    }
}

struct SnappyEncoder {
    snappy_encoder: snap::raw::Encoder,
    snappy_input_buffer: Vec<u8>,
    snappy_output_buffer: Vec<u8>,
}
impl SnappyEncoder {
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
}

struct NetPrioritizer {
    context: Arc<SharedContext>,
    map_rx: mpsc::Receiver<tonic::Result<StreamToClient>>,
    entity_rx: mpsc::Receiver<tonic::Result<StreamToClient>>,
    prio_rx: mpsc::Receiver<tonic::Result<StreamToClient>>,
    tx: mpsc::Sender<tonic::Result<StreamToClient>>,
}
impl NetPrioritizer {
    async fn run_loop(&mut self) -> Result<()> {
        while !self.context.cancellation.is_cancelled() {
            let message = tokio::select! {
                biased;
                _ = self.context.cancellation.cancelled() => break,
                msg = self.prio_rx.recv() => msg,
                msg = self.entity_rx.recv() => msg,
                msg = self.map_rx.recv() => msg,
            };
            let message = match message {
                Some(msg) => msg,
                None => break,
            };
            self.tx.send(message).await?;
        }
        tracing::info!("Prioritized outbound sender exiting");
        Ok(())
    }
}

// Loads and unloads player chunks
pub(crate) struct MapChunkSender {
    context: Arc<SharedContext>,

    outbound_tx: mpsc::Sender<tonic::Result<StreamToClient>>,
    chunk_tracker: Arc<ChunkTracker>,

    player_position: watch::Receiver<PositionAndPacing>,
    skip_if_near: ChunkCoordinate,
    processed_elements: usize,

    snappy: SnappyEncoder,

    range_hint_lookaside: FxHashMap<(i32, i32), Option<RangeInclusive<i32>>>,
    hinted_chunks: Vec<(i32, i32, i32, bool)>,
    shard_parity: usize,
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
    pub(crate) async fn chunk_sender_loop(&mut self) -> Result<()> {
        assert!(CACHE_CLEAN_MIN_AGE > (Duration::from_secs_f64(0.2) * ACCESS_TIME_BUMP_SHARDS));
        let mut access_time_bump_idx = 0;
        while !self.context.cancellation.is_cancelled() {
            tokio::select! {
                _ = self.player_position.changed() => {
                    access_time_bump_idx = (access_time_bump_idx + 1) % ACCESS_TIME_BUMP_SHARDS;
                    let update = *self.player_position.borrow_and_update();
                    self.send_chunks_for_position_update(update, access_time_bump_idx).await?;
                }
                _ = self.context.cancellation.cancelled() => {
                    info!("Client outbound loop {} detected cancellation and shutting down", self.context.id)
                }
                _ = self.context.game_state.await_start_shutdown() => {
                    info!("Game shutting down, disconnecting {}", self.context.id);
                    // cancel the inbound context as well
                    self.context.cancellation.cancel();
                }
            }
        }
        Ok(())
    }

    #[tracing::instrument(
        name = "HandlePositionUpdate",
        level = "trace",
        skip(self, update, bump_index),
        fields(
            player_name = %self.context.player_context.name()
        ),
    )]
    async fn send_chunks_for_position_update(
        &mut self,
        update: PositionAndPacing,
        bump_index: u32,
    ) -> Result<()> {
        let trace = TraceBuffer::new(false);
        trace.log("starting hpu");
        let position = update.position;

        let mut load_deadline = Instant::now();

        // TODO anticheat/safety checks
        // TODO consider caching more in the player's movement direction as a form of prefetch???
        let true_position: BlockCoordinate = match position.position.try_into() {
            Ok(x) => x,
            Err(e) => {
                warn!(
                    "Player had invalid position: {:?}, error {:?}",
                    position.position, e
                );
                let fixed_position = self.fix_position_and_notify(position.position).await?;
                self.context
                    .player_context
                    .set_position(fixed_position)
                    .await?;
                fixed_position.try_into().unwrap_or_else(|e| {
                    tracing::error!(
                        "Player had invalid position even after fix: {:?} {:?}",
                        position.position,
                        e
                    );
                    BlockCoordinate::new(0, 0, 0)
                })
            }
        };
        let prefetch_position = if let Some(position_override) = self.get_position_override() {
            load_deadline += Duration::from_millis(500);
            position_override.try_into().unwrap_or(true_position)
        } else {
            true_position
        };
        let true_player_chunk = true_position.chunk();
        let prefetch_player_chunk = prefetch_position.chunk();

        tracing::Span::current().record("player_chunk", format!("{:?}", prefetch_player_chunk));

        trace.log("player position resolved");

        // Phase 1: Unload chunks that are too far away
        let unsub_message = tokio::task::block_in_place(|| {
            let chunks_to_unsubscribe: Vec<_> = self
                .chunk_tracker
                .loaded_chunks
                .read()
                .trace_point(&trace, "chunk tracker read acquired")
                .iter()
                .filter(|&x| crate::game_state::game_map::shard_id(*x) % 2 == self.shard_parity)
                .filter(|&x| {
                    Self::should_unload(
                        prefetch_player_chunk,
                        true_player_chunk,
                        *x,
                        update.client_requested_render_distance,
                    )
                })
                .cloned()
                .collect();

            trace.log("unsub chunks msg ready");
            let message = StreamToClient {
                tick: self.context.game_state.tick(),
                server_message: Some(ServerMessage::MapChunkUnsubscribe(
                    proto::MapChunkUnsubscribe {
                        chunk_coord: chunks_to_unsubscribe.iter().map(|&x| x.into()).collect(),
                    },
                )),
                performance_metrics: self.context.maybe_get_performance_metrics(),
            };

            self.chunk_tracker
                .mark_chunks_unloaded(chunks_to_unsubscribe.into_iter());
            trace.log("unsub chunks sent");

            message
        });
        self.outbound_tx
            .send(Ok(unsub_message))
            .await
            .map_err(|_| Error::msg("Could not send outbound message (mapchunk unsubscribe)"))?;

        let effective_terrain_distance = LOAD_TERRAIN_DISTANCE
            .clamp(FORCE_LOAD_DISTANCE, update.client_requested_render_distance);

        // Phase 2: Load chunks that are close enough.
        tokio::task::block_in_place(|| {
            self.hinted_chunks.clear();
            trace.log("clean hints");
            self.range_hint_lookaside.retain(|&(cx_, cz_), _| {
                (cx_ - prefetch_player_chunk.x).abs() + (cz_ - prefetch_player_chunk.z).abs()
                    < (effective_terrain_distance * 2)
            });
            trace.log("start LTSC");
            for &(dx, dz) in LOAD_TERRAIN_SORTED_COORDS.iter() {
                if (dx.abs() + dz.abs()) > effective_terrain_distance {
                    break;
                }
                let cx = prefetch_player_chunk.x.saturating_add(dx);
                let cz = prefetch_player_chunk.z.saturating_add(dz);
                if !ChunkCoordinate::bounds_check(cx, 0, cz) {
                    continue;
                }
                let coord = ChunkCoordinate { x: cx, y: 0, z: cz };
                if crate::game_state::game_map::shard_id(coord) % 2 != self.shard_parity {
                    continue;
                }

                // Tricky: Need a clone since otherwise iterating will exhaust the range
                let hint = self
                    .range_hint_lookaside
                    .entry((cx, cz))
                    .or_insert_with(|| self.context.game_state.mapgen().terrain_range_hint(cx, cz))
                    .clone();
                if let Some(hint) = hint {
                    if dx.abs().max(dz.abs()) <= FORCE_LOAD_DISTANCE {
                        for y in -FORCE_LOAD_DISTANCE..=FORCE_LOAD_DISTANCE {
                            self.hinted_chunks.push((dx, y, dz, true))
                        }
                    }

                    for y in hint {
                        if (prefetch_player_chunk.y - y).abs() <= effective_terrain_distance {
                            self.hinted_chunks
                                .push((dx, y - prefetch_player_chunk.y, dz, false))
                        }
                    }
                }
            }
        });
        trace.log("LTSC hints ready");
        let llsc_slice = &LOAD_LAZY_SORTED_COORDS;
        let (llsc_before, llsc_after) =
            llsc_slice.split_at(MIN_INDEX_FOR_DISTANCE[FORCE_LOAD_DISTANCE as usize]);

        let start_time = Instant::now();

        // TODO: Actually dedupe the generated lists
        let mut dupes = FxHashSet::default();

        let effective_eager_distance =
            LOAD_EAGER_DISTANCE.clamp(FORCE_LOAD_DISTANCE, update.client_requested_render_distance);
        let effective_lazy_distance =
            LOAD_LAZY_DISTANCE.clamp(FORCE_LOAD_DISTANCE, update.client_requested_render_distance);
        let effective_vertical_cap = VERTICAL_DISTANCE_CAP
            .clamp(FORCE_LOAD_DISTANCE, update.client_requested_render_distance);

        let mut sent_chunks = 0;
        for (i, (dx, dy, dz, force)) in llsc_before
            .iter()
            .map(|&(x, y, z)| (x, y, z, false))
            .chain(self.hinted_chunks.drain(..))
            .chain(llsc_after.iter().map(|&(x, y, z)| (x, y, z, false)))
            .enumerate()
        {
            let coord = ChunkCoordinate {
                x: prefetch_player_chunk.x.saturating_add(dx),
                y: prefetch_player_chunk.y.saturating_add(dy),
                z: prefetch_player_chunk.z.saturating_add(dz),
            };
            if Self::should_unload(
                prefetch_player_chunk,
                true_player_chunk,
                coord,
                update.client_requested_render_distance,
            ) {
                continue;
            }

            if !coord.is_in_bounds() {
                continue;
            }
            if crate::game_state::game_map::shard_id(coord) % 2 != self.shard_parity {
                continue;
            }
            if !dupes.insert((dx, dy, dz)) {
                continue;
            }
            let distance = dx.abs() + dy.abs() + dz.abs();
            if self.player_position.has_changed().unwrap_or(false)
                && distance > FORCE_LOAD_DISTANCE
                && Instant::now() > load_deadline
            {
                trace.log("player pos changed, past FLD");
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
            // // We do need to bump chunk access times, even in the skip range :(
            // // Otherwise nearby chunks get unloaded and timers never fire
            // if i < skip {
            //     continue;
            // }

            if self.chunk_tracker.is_loaded(coord) && !chunk_needs_reload {
                continue;
            }
            // We load chunks as long as they're close enough and the map system
            // isn't overloaded. If the map system is overloaded, we'll only load
            // chunks that are close enough to the player to really matter.
            let should_load = (force
                || (distance <= effective_eager_distance && dy.abs() <= effective_vertical_cap))
                && (distance <= effective_lazy_distance
                    || !self.context.game_state.game_map().in_pushback());

            self.processed_elements = i;
            if distance > effective_eager_distance
                && start_time.elapsed() > Duration::from_millis(250)
            {
                break;
            }
            // avoid starving other tasks
            if i % 100 == 0 {
                trace.log("Yielding");
                tokio::task::yield_now().await;
            }

            trace.log("serialize_for_client start");
            let chunk_message = block_in_place(|| -> anyhow::Result<Option<StreamToClient>> {
                let chunk_data = run_traced_sync(trace.clone(), || {
                    self.context.game_state.game_map().serialize_for_client(
                        coord,
                        should_load,
                        || self.chunk_tracker.mark_chunk_loaded(coord),
                    )
                })?;

                if let Some(chunk_data) = chunk_data {
                    trace.log("snappy_encode start");
                    let chunk_bytes = self.snappy.snappy_encode(&chunk_data)?;
                    let message = StreamToClient {
                        tick: self.context.game_state.tick(),
                        server_message: Some(ServerMessage::MapChunk(proto::MapChunk {
                            chunk_coord: Some(coord.into()),
                            snappy_encoded_bytes: chunk_bytes,
                        })),
                        performance_metrics: self.context.maybe_get_performance_metrics(),
                    };
                    trace.log("message ready done");
                    Ok(Some(message))
                } else {
                    Ok(None)
                }
            })?;
            if let Some(message) = chunk_message {
                self.outbound_tx.send(Ok(message)).await.map_err(|_| {
                    self.context.cancellation.cancel();
                    Error::msg("Could not send outbound message (full mapchunk)")
                })?;

                trace.log("message sent");
                sent_chunks += 1;
                if sent_chunks > update.chunks_to_send {
                    break;
                }
            }
        }
        self.context.chunk_aimd.lock().increase();
        Ok(())
    }

    async fn fix_position_and_notify(&self, position: Vector3<f64>) -> Result<Vector3<f64>> {
        let new_position =
            if !position.x.is_finite() || !position.y.is_finite() || !position.z.is_finite() {
                // The player somehow got an inf/nan position, so respawn them
                self.context
                    .player_context
                    .send_chat_message_async(ChatMessage::new_server_message(
                        "Your position is borked. Respawning you.",
                    ))
                    .await?;
                (self.context.game_state.game_behaviors().spawn_location)(
                    &self.context.player_context.name,
                )
            } else {
                // The position is finite but it's out of bounds
                self.context
                    .player_context
                    .send_chat_message_async(ChatMessage::new_server_message(
                        "You've hit the edge of the map.",
                    ))
                    .await?;
                Vector3::new(
                    position.x.clamp(-2147483640.0, 2147483640.0),
                    position.y.clamp(-2147483640.0, 2147483640.0),
                    position.z.clamp(-2147483640.0, 2147483640.0),
                )
            };
        tracing::warn!(
            "Fixing position for {}: {position:?} -> {new_position:?}",
            self.context.player_context.name
        );
        Ok(new_position)
    }

    fn should_unload(
        prefetch_player_chunk: ChunkCoordinate,
        true_player_chunk: ChunkCoordinate,
        chunk: ChunkCoordinate,
        client_distance_req: i32,
    ) -> bool {
        let distance =
            UNLOAD_DISTANCE.clamp(FORCE_LOAD_DISTANCE * 3 / 2, client_distance_req * 3 / 2);
        let x_min = prefetch_player_chunk.x.min(true_player_chunk.x) - distance;
        let x_max = prefetch_player_chunk.y.max(true_player_chunk.x) + distance;
        let z_min = prefetch_player_chunk.z.min(true_player_chunk.z) - distance;
        let z_max = prefetch_player_chunk.z.max(true_player_chunk.z) + distance;
        let y_min = prefetch_player_chunk.y.min(true_player_chunk.y) - distance;
        let y_max = prefetch_player_chunk.y.max(true_player_chunk.y) + distance;
        !((chunk.x >= x_min && chunk.x <= x_max)
            && (chunk.z >= z_min && chunk.z <= z_max)
            && (chunk.y >= y_min && chunk.y <= y_max))
    }
    fn get_position_override(&self) -> Option<Vector3<f64>> {
        tokio::task::block_in_place(|| {
            let entity_id = self
                .context
                .player_context
                .state
                .lock()
                .attached_to_entity?;
            // Add 2 seconds of predictive lookahead
            let tick = self.context.game_state.tick().saturating_add(2_000_000_000);
            self.context
                .game_state
                .entities()
                .predictive_position(entity_id, tick)
        })
    }
}

// Sends block updates to the client
pub(crate) struct BlockEventSender {
    context: Arc<SharedContext>,

    // RPC stream is sent via this channel
    outbound_tx: mpsc::Sender<tonic::Result<StreamToClient>>,

    // All updates to the map from all sources, not yet filtered by location (BlockEventSender is
    // responsible for filtering)
    block_events: broadcast::Receiver<BlockUpdate>,
    chunk_tracker: Arc<ChunkTracker>,
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
    pub(crate) async fn block_sender_loop(&mut self) -> Result<()> {
        while !self.context.cancellation.is_cancelled() {
            tokio::select! {
                block_event = self.block_events.recv() => {
                    self.handle_block_update(block_event).await?;
                }
                _ = self.context.cancellation.cancelled() => {
                    info!("Client outbound loop {} detected cancellation and shutting down", self.context.id)
                    // pass
                }
            }
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
        let mut have_slept = false;
        while updates.len() < MAX_UPDATE_BATCH_SIZE {
            match self.block_events.try_recv() {
                Ok(update) => {
                    if self.chunk_tracker.is_loaded(update.location.chunk()) {
                        // last update wins
                        updates.insert(update.location, update);
                    }
                }
                Err(broadcast::error::TryRecvError::Empty) => {
                    // If we have no events, try sleeping *once* to get some more events into
                    // a batch
                    // TODO consider making this a select so we can immediately fill a batch
                    // without the 10-msec wait.
                    if !have_slept {
                        have_slept = true;
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    } else {
                        break;
                    }
                }
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
        for (coord, update) in updates {
            update_protos.push(proto::MapDeltaUpdate {
                block_coord: Some(coord.into()),
                new_id: update.id.into(),
                new_client_ext_data: update.new_ext_data,
            })
        }

        if !update_protos.is_empty() {
            let message = StreamToClient {
                tick: self.context.game_state.tick(),
                server_message: Some(ServerMessage::MapDeltaUpdate(MapDeltaUpdateBatch {
                    updates: update_protos,
                })),
                performance_metrics: self.context.maybe_get_performance_metrics(),
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
        self.context.chunk_aimd.lock().decrease_to_floor();

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
        let message = make_client_state_update_message(&self.context, player_state, true);
        self.outbound_tx
            .send(Ok(message?))
            .await
            .map_err(|_| Error::msg("Could not send outbound message (updated state)"))
    }
}

fn make_client_state_update_message(
    ctx: &SharedContext,
    mut player_state: MutexGuard<'_, PlayerState>,
    update_location: bool,
) -> Result<StreamToClient> {
    let message = {
        let position = if update_location {
            Some(PlayerPosition {
                position: Some(player_state.last_position.position.try_into()?),
                velocity: Some(Vector3::zero().try_into()?),
                face_direction: Some(coords_proto::Angles {
                    deg_azimuth: 0.,
                    deg_elevation: 0.,
                }),
            })
        } else {
            None
        };
        let (time_now, day_len) = {
            let time_state = ctx.game_state.time_state().lock();
            (time_state.time_of_day(), time_state.day_length())
        };
        player_state.repair_inventory_popup(
            &ctx.player_context.name,
            ctx.player_context.player.main_inventory_key,
            &ctx.game_state,
        )?;
        StreamToClient {
            tick: ctx.game_state.tick(),
            server_message: Some(ServerMessage::ClientState(proto::SetClientState {
                position,
                hotbar_inventory_view: player_state.hotbar_inventory_view.id.0,
                inventory_popup: Some(player_state.inventory_popup.as_ref().unwrap().to_proto()),
                inventory_manipulation_view: player_state.inventory_manipulation_view.id.0,
                time_of_day: time_now,
                day_length_sec: day_len.as_secs_f64(),
                permission: player_state
                    .effective_permissions(&ctx.game_state, ctx.player_context.name())
                    .iter()
                    .cloned()
                    .collect(),
                attached_to_entity: player_state.attached_to_entity,
            })),
            performance_metrics: ctx.maybe_get_performance_metrics(),
        }
    };

    Ok(message)
}

pub(crate) struct InventoryEventSender {
    context: Arc<SharedContext>,
    // RPC stream is sent via this channel
    outbound_tx: mpsc::Sender<tonic::Result<StreamToClient>>,

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
    pub(crate) async fn inv_sender_loop(&mut self) -> Result<()> {
        while !self.context.cancellation.is_cancelled() {
            tokio::select! {
                inventory_update = self.inventory_events.recv() => {
                    self.handle_inventory_update(inventory_update).await?;
                }
                _ = self.context.cancellation.cancelled() => {
                    info!("Client outbound loop {} detected cancellation and shutting down", self.context.id)
                    // pass
                }
            }
        }
        Ok(())
    }

    async fn handle_inventory_update(
        &mut self,
        update: Result<UpdatedInventory, broadcast::error::RecvError>,
    ) -> Result<()> {
        let key = match update {
            Err(broadcast::error::RecvError::Lagged(x)) => {
                tracing::error!("Client {} is lagged, {} pending", self.context.id, x);
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
        .in_scope(|| -> Result<_> {
            let player_state = self.context.player_context.state.lock();
            let mut update_messages = vec![];
            for key in update_keys {
                if player_state.hotbar_inventory_view.wants_update_for(&key) {
                    update_messages.push(make_inventory_update(
                        &self.context,
                        &&player_state.hotbar_inventory_view,
                    )?);
                }

                for popup in player_state
                    .active_popups
                    .iter()
                    .chain(player_state.inventory_popup.iter())
                {
                    for view in popup.inventory_views().values() {
                        if view.wants_update_for(&key) {
                            update_messages.push(make_inventory_update(
                                &self.context,
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
    outbound_tx: mpsc::Sender<tonic::Result<StreamToClient>>,
    // The client's self-reported position
    own_positions: watch::Sender<PositionAndPacing>,
    next_pos_writeback: Instant,
}
impl InboundWorker {
    fn process_footstep(player: Arc<Player>, ctx: &SharedContext, coord: BlockCoordinate) {
        let ppu = player.last_position();
        let res = Self::map_handler_sync(
            ctx,
            None,
            move |_| {
                &move |ctx, coord, stack| {
                    default_generic_handler(ctx, coord, stack, {
                        move |gs| {
                            gs.game_map().run_block_interaction(
                                coord,
                                &ctx.initiator,
                                None,
                                |bt| bt.step_on_handler_inline.as_deref(),
                                |bt| bt.step_on_handler_full.as_deref(),
                            )
                        }
                    })
                }
            },
            coord,
            ppu,
        );
        if let Err(e) = res {
            tracing::error!("Footstep handler failed: {e:?}");
        }
    }

    async fn off_loop_footsteps_worker(
        ctx: Arc<SharedContext>,
        mut rx: mpsc::Receiver<BlockCoordinate>,
    ) -> Result<()> {
        let player = Arc::downgrade(&ctx.player_context.player);
        while !ctx.cancellation.is_cancelled() {
            tokio::select! {
                msg = rx.recv() => {
                    match msg {
                        None => {break}
                        Some(coord) => {
                            match player.upgrade() {
                                Some(p) => block_in_place(|| Self::process_footstep(p, &ctx, coord)),
                                None => { tracing::warn!("Can't upgrade player weak ref in footstep worker"); break; }
                            }
                        }
                    }
                }
                _ = ctx.cancellation.cancelled() => {
                    break;
                }
            }
        }
        Ok(())
    }

    // Poll for world events and send them through outbound_tx
    pub(crate) async fn inbound_worker_loop(&mut self) -> Result<()> {
        let (mut step_tx, step_rx) = mpsc::channel(32);
        let mut step_worker_handle = spawn_async(
            &format!("{}_footstep_worker", &self.context.player_context.name),
            Self::off_loop_footsteps_worker(self.context.clone(), step_rx),
        )?;

        const INBOUND_TIMEOUT: Duration = Duration::from_secs(10);
        while !self.context.cancellation.is_cancelled() {
            let timeout = tokio::time::sleep(INBOUND_TIMEOUT);
            tokio::pin!(timeout);
            let trace_buffer = TraceBuffer::new(false);
            trace_buffer.log("Waiting for inbound message");
            tokio::select! {
                message = self.inbound_rx.message() => {
                    match message {
                        Err(e) => {
                            error!("Client {}, Failure reading inbound message: {:?}", self.context.id, e);
                            self.context.cancellation.cancel();
                            return Ok(());
                        },
                        Ok(None) => {
                            info!("Client {} disconnected", self.context.id);
                            self.context.cancellation.cancel();
                            return Ok(());
                        }
                        Ok(Some(message)) => {
                            match run_traced(trace_buffer, async { self.handle_message(&message, &mut step_tx).await }).await {
                                Ok(_) => {},
                                Err(e) => {
                                    warn!("Client {} failed to handle message: {:?}, error: {:?}", self.context.id, message, e);
                                    // TODO notify the client once there's a chat or server->client error handling message
                                },
                            }
                        }
                    }

                },
                _ = &mut step_worker_handle => {
                    info!("Off-thread footstep worker finished")
                }
                _ = self.context.cancellation.cancelled() => {
                    info!("Client inbound context {} detected cancellation and shutting down", self.context.id)
                    // pass
                },
                _ = timeout => {
                    warn!("Client inbound context {} timed out and shutting down", self.context.id);
                    self.context.cancellation.cancel();
                    self.context.game_state.game_behaviors().on_player_err.handle(
                        &self.context.player_context,
                        HandlerContext {
                            tick: self.context.game_state.tick(),
                            initiator: EventInitiator::Engine,
                            game_state: self.context.game_state.clone(),
                        },
                    ).await?;
                    // Do not send the generic player-leave message
                    return Ok(());
                }
            }
        }
        self.context
            .game_state
            .game_behaviors()
            .on_player_leave
            .handle(
                &self.context.player_context,
                HandlerContext {
                    tick: self.context.game_state.tick(),
                    initiator: EventInitiator::Engine,
                    game_state: self.context.game_state.clone(),
                },
            )
            .await?;
        Ok(())
    }

    fn check_player_permission(&self, permission: &str) -> Result<()> {
        if !self.context.player_context.has_permission(permission) {
            log_trace("Player permission check failed");
            return Err(Error::msg("Player does not have permission"));
        }
        log_trace("Player permission check succeeded");
        Ok(())
    }

    async fn handle_message(
        &mut self,
        message: &proto::StreamToServer,
        step_tx: &mut mpsc::Sender<BlockCoordinate>,
    ) -> Result<()> {
        let enable_perf_stats = message.want_performance_metrics
            && self
                .context
                .player_context
                .has_permission(permissions::PERFORMANCE_METRICS);
        self.context
            .enable_performance_metrics
            .store(enable_perf_stats, Ordering::Relaxed);

        log_trace("Handling inbound message");
        // todo do something with the client tick once we define ticks
        match &message.client_message {
            None => {
                warn!(
                    "Client context {} got empty/unknown message from client",
                    self.context.id
                );
            }
            Some(proto::stream_to_server::ClientMessage::Dig(dig_message)) => {
                self.check_player_permission(permissions::DIG_PLACE)?;
                let position = dig_message
                    .position
                    .as_ref()
                    .context("Missing player position")?
                    .try_into()?;
                match dig_message
                    .action_target
                    .as_ref()
                    .context("missing target")?
                {
                    ActionTarget::BlockCoord(coord) => {
                        // TODO check whether the current item can dig this block, and whether
                        // it's been long enough since the last dig
                        Self::run_map_handlers(
                            &self.context,
                            coord.into(),
                            dig_message.item_slot,
                            |item| {
                                item.and_then(|x| x.dig_handler.as_deref())
                                    .unwrap_or(&items::default_dig_handler)
                            },
                            position,
                        )
                        .await?;
                    }
                    ActionTarget::EntityTarget(id) => {
                        Self::run_map_handlers(
                            &self.context,
                            *id,
                            dig_message.item_slot,
                            |item| {
                                item.and_then(|x| x.dig_entity_handler.as_deref())
                                    .unwrap_or(&items::default_entity_dig_handler)
                            },
                            position,
                        )
                        .await?
                    }
                }
            }
            Some(proto::stream_to_server::ClientMessage::Tap(tap_message)) => {
                self.check_player_permission(permissions::TAP_INTERACT)?;
                let position = tap_message
                    .position
                    .as_ref()
                    .context("Missing player position")?
                    .try_into()?;
                match tap_message
                    .action_target
                    .as_ref()
                    .context("missing target")?
                {
                    ActionTarget::BlockCoord(coord) => {
                        Self::run_map_handlers(
                            &self.context,
                            coord.into(),
                            tap_message.item_slot,
                            |item| {
                                item.and_then(|x| x.tap_handler.as_deref())
                                    .unwrap_or(&items::default_tap_handler)
                            },
                            position,
                        )
                        .await?
                    }
                    ActionTarget::EntityTarget(id) => {
                        Self::run_map_handlers(
                            &self.context,
                            *id,
                            tap_message.item_slot,
                            |item| {
                                item.and_then(|x| x.dig_entity_handler.as_deref())
                                    .unwrap_or(&items::default_entity_tap_handler)
                            },
                            position,
                        )
                        .await?
                    }
                }
            }
            Some(proto::stream_to_server::ClientMessage::PositionUpdate(pos_update)) => {
                self.handle_pos_update(message.client_tick, pos_update, step_tx)
                    .await?;
            }
            Some(proto::stream_to_server::ClientMessage::BugCheck(bug_check)) => {
                error!("Client bug check: {:?}", bug_check);
            }
            Some(proto::stream_to_server::ClientMessage::Place(place_message)) => {
                self.check_player_permission(permissions::DIG_PLACE)?;
                self.handle_place(place_message).await?;
            }
            Some(proto::stream_to_server::ClientMessage::Inventory(inventory_message)) => {
                self.check_player_permission(permissions::INVENTORY)?;
                self.handle_inventory_action(inventory_message).await?;
            }
            Some(proto::stream_to_server::ClientMessage::PopupResponse(response)) => {
                self.check_player_permission(permissions::INVENTORY)?;
                self.handle_popup_response(response).await?;
            }
            Some(proto::stream_to_server::ClientMessage::InteractKey(interact_key)) => {
                self.check_player_permission(permissions::TAP_INTERACT)?;
                self.handle_interact_key(interact_key).await?;
            }
            Some(proto::stream_to_server::ClientMessage::ChatMessage(message)) => {
                let gs = self.context.game_state.clone();
                let player_handle = self.context.player_context.clone();
                let message = message.clone();
                tokio::task::spawn(async move {
                    if let Err(e) = gs
                        .chat()
                        .handle_inbound_chat_message(
                            player_handle.make_initiator(),
                            gs.clone(),
                            &message,
                        )
                        .await
                    {
                        warn!("Error handling inbound chat message: {:?}", e);
                    }
                });
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
        log_trace("Sending ack");
        self.send_ack(message.sequence).await?;
        log_trace("Done handling message");
        Ok(())
    }

    #[tracing::instrument(
        name = "map_handler",
        level = "trace",
        skip(ctx, coord, selected_inv_slot, get_item_handler),
        fields(
            player_name = %ctx.player_context.name(),
        ),
    )]
    async fn run_map_handlers<F, T>(
        ctx: &SharedContext,
        coord: T,
        selected_inv_slot: u32,
        get_item_handler: F,
        player_position: PlayerPositionUpdate,
    ) -> Result<()>
    where
        F: FnOnce(Option<&Item>) -> &items::GenericInteractionHandler<T>,
    {
        block_in_place(|| {
            Self::map_handler_sync(
                ctx,
                Some(selected_inv_slot),
                get_item_handler,
                coord,
                player_position,
            )
        })
    }

    fn map_handler_sync<F, T>(
        ctx: &SharedContext,
        selected_inv_slot: Option<u32>,
        get_item_handler: F,
        coord: T,
        player_position: PlayerPositionUpdate,
    ) -> std::result::Result<(), Error>
    where
        F: FnOnce(Option<&Item>) -> &items::GenericInteractionHandler<T>,
    {
        log_trace("Running map handlers");
        let game_state = &ctx.game_state;

        game_state.inventory_manager().mutate_inventory_atomically(
            &ctx.player_context.main_inventory(),
            |inventory| {
                let mut none_stack = None;
                log_trace("In inventory mutator");
                let stack = match selected_inv_slot {
                    Some(x) => inventory
                        .contents_mut()
                        .get_mut(x as usize)
                        .with_context(|| "Item slot was out of bounds")?,
                    None => &mut none_stack,
                };

                let initiator = EventInitiator::Player(PlayerInitiator {
                    player: &ctx.player_context.player,
                    weak: Arc::downgrade(&ctx.player_context.player),
                    position: player_position,
                });

                let item_handler =
                    get_item_handler(game_state.item_manager().get_stack_item(stack.as_ref()));

                let result = {
                    let ctx = HandlerContext {
                        tick: game_state.tick(),
                        initiator: initiator.clone(),
                        game_state: game_state.clone(),
                    };
                    run_handler!(
                        || {
                            item_handler(
                                &ctx,
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
                log_trace("Done running map handlers");
                Ok(())
            },
        )
    }

    async fn send_ack(&mut self, sequence: u64) -> Result<()> {
        if sequence == 0 {
            return Ok(());
        };
        let message = StreamToClient {
            tick: self.context.game_state.tick(),
            server_message: Some(ServerMessage::HandledSequence(sequence)),
            performance_metrics: self.context.maybe_get_performance_metrics(),
        };

        self.outbound_tx.send(Ok(message)).await.map_err(|_| {
            self.context.cancellation.cancel();
            Error::msg("Could not send outbound message (sequence ack)")
        })?;

        Ok(())
    }

    async fn handle_pos_update(
        &mut self,
        _tick: u64,
        update: &proto::ClientUpdate,
        step_tx: &mut mpsc::Sender<BlockCoordinate>,
    ) -> Result<()> {
        match &update.position {
            Some(pos_update) => {
                let (az, el) = match &pos_update.face_direction {
                    Some(x) => (x.deg_azimuth, x.deg_elevation),
                    None => {
                        warn!("No angles in update from client");
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

                let mut max_distance = i32::MAX;

                if let Some(pacing) = &update.pacing {
                    if pacing.pending_chunks < 256 {
                        self.context.chunk_aimd.lock().increase();
                    } else if pacing.pending_chunks > 1024 {
                        self.context.chunk_aimd.lock().decrease();
                    }
                    max_distance = pacing.distance_limit as i32
                }

                self.own_positions.send_replace(PositionAndPacing {
                    position: pos,
                    chunks_to_send: self.context.chunk_aimd.lock().get(),
                    client_requested_render_distance: max_distance,
                });
                self.context
                    .player_context
                    .update_client_position_state(pos, update.hotbar_slot);
                self.context
                    .game_state
                    .entities()
                    .set_kinematics(
                        self.context.player_context.entity_id,
                        pos_update
                            .position
                            .as_ref()
                            .context("Missing position")?
                            .try_into()?,
                        // TODO: Player movements use degrees, entity movements use radians.
                        // This is inconsistent and should be fixed in a later protocol version
                        Movement::stop_and_stay(
                            pos.face_direction.0 as f32 * std::f32::consts::PI / 180.0,
                            0.0,
                            f32::MAX,
                        ),
                        crate::game_state::entities::InitialMoveQueue::SingleMove(None),
                    )
                    .await;

                let mut dropped = 0;
                for step in &update.footstep_coordinate {
                    if let Some(coord) = step.coord {
                        if let Err(_) = step_tx.try_send(coord.try_into()?) {
                            dropped += 1;
                        }
                    }
                }
                if dropped > 0 {
                    tracing::warn!("Dropped {} footstep events", dropped);
                }

                if let Some(footstep) = update.footstep_coordinate.last() {
                    let coord = footstep.coord.context("Missing coord")?.into();
                    let footstep_sound_id = self
                        .context
                        .game_state
                        .game_map()
                        .try_get_block(coord)
                        .and_then(|id| {
                            self.context
                                .game_state
                                .block_types()
                                .get_block_by_id(id)
                                .ok()
                        })
                        .and_then(|block_def| match block_def.0.client_info.footstep_sound {
                            0 => None,
                            x => Some(x),
                        });
                    // TODO: Wire up the stepped-on block handler here, _in addition_ to the coord
                    // that the player is on based on their *current* position (they may not have
                    // moved enough to add it to the footstep list, and may be standing on it
                    // indefinitely without ever adding it)
                    if let Some(sound_id) = footstep_sound_id {
                        self.context.game_state.audio().send_event(AudioEvent {
                            initiating_context_id: self.context.id,
                            instruction: AudioInstruction::PlaySampledSound(SampledSoundPlayback {
                                tick: 0,
                                sound_id,
                                position: Some(Vec3D {
                                    x: coord.x as f64,
                                    y: coord.y as f64,
                                    z: coord.z as f64,
                                }),
                                disable_doppler: false,
                                disable_falloff: false,
                                disable_balance: false,
                                source: SoundSource::SoundsourcePlayer.into(),
                                volume: 1.0,
                            }),
                        })
                    }
                }
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
        block_in_place(|| {
            log_trace("Running place handlers");
            let _span = span!("handle_place");
            self.context
                .game_state
                .inventory_manager()
                .mutate_inventory_atomically(
                    &self.context.player_context.main_inventory(),
                    |inventory| {
                        log_trace("In inventory mutator");
                        let stack = inventory
                            .contents_mut()
                            .get_mut(place_message.item_slot as usize)
                            .with_context(|| "Item slot was out of bounds")?;

                        let initiator = EventInitiator::Player(PlayerInitiator {
                            player: &self.context.player_context,
                            weak: Arc::downgrade(&self.context.player_context.player),
                            position: place_message
                                .position
                                .as_ref()
                                .context("Missing position")?
                                .try_into()?,
                        });
                        let anchor: ToolTarget = place_message
                            .place_anchor
                            .with_context(|| "Missing anchor in place message")?
                            .into();
                        match anchor {
                            ToolTarget::Block(anchor) => {
                                let handler = self
                                    .context
                                    .game_state
                                    .item_manager()
                                    .get_stack_item(stack.as_ref())
                                    .and_then(|x| x.place_on_block_handler.as_deref());
                                // TODO: check if block has a place-upon handler and invoke it here
                                // Consider including a locked, mutate_block_atomically-eligible
                                // place-upon handler
                                if let Some(handler) = handler {
                                    let ctx = HandlerContext {
                                        tick: self.context.game_state.tick(),
                                        initiator: initiator.clone(),
                                        game_state: self.context.game_state.clone(),
                                    };
                                    let new_stack = run_handler!(
                                        || {
                                            handler(
                                                &ctx,
                                                place_message
                                                    .block_coord
                                                    .with_context(|| {
                                                        "Missing block_coord in place message"
                                                    })?
                                                    .into(),
                                                anchor,
                                                stack.as_ref().unwrap(),
                                            )
                                        },
                                        "item_place",
                                        &initiator,
                                    )?;
                                    *stack = new_stack;
                                }
                            }
                            ToolTarget::Entity(target) => {
                                let handler = self
                                    .context
                                    .game_state
                                    .item_manager()
                                    .get_stack_item(stack.as_ref())
                                    .and_then(|x| x.place_on_entity_handler.as_deref());
                                if let Some(handler) = handler {
                                    let ctx = HandlerContext {
                                        tick: self.context.game_state.tick(),
                                        initiator: initiator.clone(),
                                        game_state: self.context.game_state.clone(),
                                    };
                                    let new_stack = run_handler!(
                                        || { handler(&ctx, target, stack.as_ref().unwrap(),) },
                                        "item_place",
                                        &initiator,
                                    )?;
                                    *stack = new_stack;
                                } else {
                                    tracing::error!(
                                        "TODO: Run the entity's on-place handler instead"
                                    )
                                }
                            }
                        }

                        log_trace("Done running place handlers");
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
            tracing::error!(
                "Cannot handle an inventory action with the same source and destination view"
            );
            // todo send the user an error
            return Ok(());
        }
        let updates =
            {
                let mut views_to_send = HashSet::new();
                block_in_place(|| -> Result<Vec<StreamToClient>> {
                    log_trace("Locking player for inventory action");
                    let mut player_state = self.context.player_context.player.state.lock();
                    log_trace("Locked player for inventory action");
                    player_state.repair_inventory_popup(
                        &self.context.player_context.name,
                        self.context.player_context.player.main_inventory_key,
                        &self.context.game_state,
                    )?;
                    player_state.handle_inventory_action(action)?;

                    for popup in player_state
                        .active_popups
                        .iter()
                        .chain(player_state.inventory_popup.iter())
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
                            &self.context,
                            player_state.find_inv_view(view)?.as_ref(),
                        )?);
                    }

                    updates.push(make_inventory_update(
                        &self.context,
                        &&player_state.inventory_manipulation_view,
                    )?);
                    log_trace("Done running inventory action");
                    Ok(updates)
                })?
            };
        for update in updates {
            self.outbound_tx
                .send(Ok(update))
                .await
                .map_err(|_| Error::msg("Could not send outbound message (inventory update)"))?;
        }
        log_trace("Done sending inventory updates");

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
        let updates = block_in_place(|| -> Result<_> {
            let _span = span!("handle_popup_response");
            let ctx = HandlerContext {
                tick: self.context.game_state.tick(),
                initiator: self.context.player_context.make_initiator(),
                game_state: self.context.game_state.clone(),
            };
            let mut player_state = self.context.player_context.state.lock();
            player_state.repair_inventory_popup(
                &self.context.player_context.name,
                self.context.player_context.player.main_inventory_key,
                &self.context.game_state,
            )?;

            let mut updates = vec![];
            if action.closed {
                player_state
                    .inventory_manipulation_view
                    .clear_if_transient(Some(self.context.player_context.main_inventory_key))?;
                updates.push(make_inventory_update(
                    &self.context,
                    &&player_state.inventory_manipulation_view,
                )?);
            }
            if let Some(popup_pos) = player_state
                .active_popups
                .iter_mut()
                .position(|x| x.id() == action.popup_id)
            {
                let mut popup = player_state.active_popups.swap_remove(popup_pos);
                let handling_result = (|| -> Result<()> {
                    MutexGuard::unlocked(&mut player_state, || {
                        run_handler!(
                            || popup.handle_response(
                                PopupResponse {
                                    user_action,
                                    textfield_values: action.text_fields.clone(),
                                    checkbox_values: action.checkboxes.clone(),
                                    ctx: ctx.clone(),
                                },
                                self.context.player_context.main_inventory(),
                            ),
                            "popup_response",
                            &ctx.initiator
                        )
                    })?;
                    for view in popup.inventory_views().values() {
                        updates.push(make_inventory_update(
                            &self.context,
                            &InventoryViewWithContext {
                                view,
                                context: &popup,
                            },
                        )?);
                    }
                    Ok(())
                })();
                player_state.active_popups.push(popup);
                handling_result?;
            } else if player_state
                .inventory_popup
                .as_ref()
                .is_some_and(|x| x.id() == action.popup_id)
            {
                let mut inventory_popup = player_state.inventory_popup.take().unwrap();
                tracing::info!("Taking inventory popup");
                let handling_result = (|| -> Result<()> {
                    MutexGuard::unlocked(&mut player_state, || {
                        run_handler!(
                            || inventory_popup.handle_response(
                                PopupResponse {
                                    user_action,
                                    textfield_values: action.text_fields.clone(),
                                    checkbox_values: action.checkboxes.clone(),
                                    ctx: ctx.clone(),
                                },
                                self.context.player_context.main_inventory(),
                            ),
                            "popup_response",
                            &ctx.initiator
                        )
                    })?;
                    for view in inventory_popup.inventory_views().values() {
                        updates.push(make_inventory_update(
                            &self.context,
                            &InventoryViewWithContext {
                                view,
                                context: &inventory_popup,
                            },
                        )?);
                    }

                    Ok(())
                })();
                if player_state.inventory_popup.is_none() {
                    tracing::info!("Returning inventory popup");
                    player_state.inventory_popup = Some(inventory_popup);
                } else {
                    tracing::info!("Not returning inventory popup since a new one was added");
                }

                handling_result?;
            } else {
                tracing::error!(
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
            Ok(updates)
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
        let messages =
            block_in_place(|| -> Result<_> { self.handle_interact_key_sync(interact_message) })?;
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
        let initiator = EventInitiator::Player(PlayerInitiator {
            player: &self.context.player_context,
            weak: Arc::downgrade(&self.context.player_context.player),
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
        let menu_entry = interact_message.menu_entry.clone();
        let handler = match interact_message
            .interaction_target
            .as_ref()
            .context("missing target")?
        {
            // Interact keys always bypass player inventory and fire on the target regardless of the
            // held tool
            InteractionTarget::BlockCoord(coord) => {
                let coord = coord.into();

                let block = self.context.game_state.game_map().get_block(coord)?;
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
                    Some(Either::Left(move || (handler)(ctx, coord, &menu_entry)))
                } else {
                    None
                }
            }
            InteractionTarget::EntityTarget(target) => {
                let mut class = None;
                let tei = target.trailing_entity_index as usize;
                self.context.game_state.entities().visit_entity_by_id(
                    target.entity_id,
                    |entity| {
                        if tei == 0 {
                            class = Some(entity.class)
                        } else {
                            class = entity
                                .trailing_entities
                                .and_then(|x| x.get(tei - 1))
                                .map(|x| x.class_id);
                        }
                        Ok(())
                    },
                )?;
                let class = class.context("Entity not found")?;
                let class_def = self
                    .context
                    .game_state
                    .entities()
                    .types()
                    .get_type(class)
                    .with_context(|| format!("Class {} not found", class))?;
                Some(Either::Right(move || {
                    class_def
                        .handlers
                        .on_interact_key(&ctx, *target, &menu_entry)
                }))
            }
        };
        let handler = match handler {
            None => return Ok(vec![]),
            Some(x) => x,
        };
        let mut messages = vec![];
        if let Some(popup) =
            either::for_both!(handler, h => run_handler!(h, "interact_key", &initiator))?
        {
            for view in popup.inventory_views().values() {
                messages.push(make_inventory_update(
                    &self.context,
                    &InventoryViewWithContext {
                        view,
                        context: &popup,
                    },
                )?)
            }
            messages.push(StreamToClient {
                tick: self.context.game_state.tick(),
                server_message: Some(ServerMessage::ShowPopup(popup.to_proto())),
                performance_metrics: self.context.maybe_get_performance_metrics(),
            });
            self.context
                .player_context
                .player
                .state
                .lock()
                .active_popups
                .push(popup);
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
    async fn misc_outbound_worker_loop(&mut self) -> Result<()> {
        let mut broadcast_messages = self.context.game_state.chat().subscribe();
        let mut resync_player_state_global =
            self.context.game_state.subscribe_player_state_resyncs();
        while !self.context.cancellation.is_cancelled() {
            tokio::select! {
                event = self.player_event_receiver.rx.recv() => {
                    match event {
                        Some(event) => {
                        self.handle_player_event(event).await?;
                        },
                        None => {
                            warn!("Player event sender disconnected")
                        }
                    }
                }
                _ = resync_player_state_global.changed() => {
                    let message = make_client_state_update_message(&self.context, self.context.player_context.player.state.lock(), false)?;
                    self.outbound_tx.send(Ok(message)).await?;
                },
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
                        performance_metrics: self.context.maybe_get_performance_metrics(),
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

    async fn handle_player_event(&mut self, event: PlayerEvent) -> Result<()> {
        match event {
            PlayerEvent::ChatMessage(message) => {
                self.transmit_chat_message(message).await?;
            }
            PlayerEvent::DisconnectionMessage(reason) => {
                self.outbound_tx
                    .send(Ok(StreamToClient {
                        tick: self.context.game_state.tick(),
                        server_message: Some(ServerMessage::ShutdownMessage(reason)),
                        performance_metrics: self.context.maybe_get_performance_metrics(),
                    }))
                    .await?;
            }
            PlayerEvent::ReinitPlayerState(want_location_update) => {
                let message = make_client_state_update_message(
                    &self.context,
                    self.context.player_context.player.state.lock(),
                    want_location_update,
                )?;
                self.outbound_tx.send(Ok(message)).await?;
            }
            PlayerEvent::UpdatedPopup(popup_id) => {
                let popup_proto = {
                    self.context
                        .player_context
                        .state
                        .lock()
                        .active_popups
                        .iter()
                        .find(|x| x.id() == popup_id)
                        .map(|x| x.to_proto())
                };
                match popup_proto {
                    Some(x) => {
                        self.outbound_tx
                            .send(Ok(StreamToClient {
                                tick: self.context.game_state.tick(),
                                server_message: Some(ServerMessage::ShowPopup(x)),
                                performance_metrics: self.context.maybe_get_performance_metrics(),
                            }))
                            .await?;
                    }
                    None => {
                        warn!("Updated popup not found: {}", popup_id);
                    }
                };
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
                performance_metrics: self.context.maybe_get_performance_metrics(),
            }))
            .await?;
        Ok(())
    }
}

/// Handles sending entities. Currently unoptimized, and not very scalable.
struct EntityEventSender {
    outbound_tx: mpsc::Sender<tonic::Result<StreamToClient>>,
    context: Arc<SharedContext>,
    // entity id -> last sequence number we've sent
    sent_entities: FxHashMap<u64, u64>,
    last_update: u64,
}
impl EntityEventSender {
    async fn entity_sender_loop(&mut self) -> Result<()> {
        if self.context.effective_protocol_version < 4 {
            tracing::info!(
                "Client {} is using an unsupported protocol version, ignoring entity events",
                self.context.id
            );
            self.context.cancellation.cancelled().await;
            return Ok(());
        }

        // TODO - optimize this with actual awakenings based on relevant entities
        let mut interval = tokio::time::interval(Duration::from_millis(10));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        while !self.context.cancellation.is_cancelled() {
            tokio::select! {
                _ = self.context.cancellation.cancelled() => {
                    info!("Entity event sender context {} detected cancellation and shutting down", self.context.id);
                }
                _ = interval.tick() => {
                    self.do_tick().await?;
                }
            }
        }
        Ok(())
    }
    async fn do_tick(&mut self) -> Result<()> {
        // todo - this requires optimization

        let update_time = self.context.game_state.tick();
        let messages = {
            let mut messages = vec![];

            let entities = self.context.game_state.entities();

            // TODO this is suboptimal
            let known_to_us = self.sent_entities.keys().cloned().collect::<HashSet<_>>();
            let mut still_valid = HashSet::new();

            let tick = self.context.game_state.tick();
            tokio::task::block_in_place(|| -> anyhow::Result<()> {
                for shard in entities.shards() {
                    shard.for_each_entity(None, tick, |entity: IterEntity| {
                        still_valid.insert(entity.id);
                        if entity.id == self.context.player_context.entity_id {
                            return Ok(());
                        }
                        if entity.last_nontrivial_modification < self.last_update {
                            return Ok(());
                        }

                        let last_sent_sequence = self.sent_entities.get(&entity.id).cloned();
                        let last_available_sequence = entity
                            .next_moves
                            .last()
                            .map(|m| m.sequence)
                            .unwrap_or(entity.current_move.sequence);
                        // This check is too lenient - we have various modifications that don't change
                        // the last sent sequence, but are otherwise important (e.g. force-advancing out of
                        // a wait-forever movement with DEQUEUE_WHEN_READY in the entity engine)
                        self.sent_entities
                            .insert(entity.id, last_available_sequence);

                        let mut moves_to_send = Vec::with_capacity(8);
                        let mut pos = entity.starting_position;
                        for m in
                            std::iter::once(&entity.current_move).chain(entity.next_moves.iter())
                        {
                            if m.sequence > last_sent_sequence.unwrap_or(0) {
                                moves_to_send.push(entities_proto::EntityMove {
                                    sequence: m.sequence,
                                    start_position: Some(pos.try_into()?),
                                    velocity: Some(m.movement.velocity.try_into()?),
                                    acceleration: Some(m.movement.acceleration.try_into()?),
                                    face_direction: m.movement.face_direction,
                                    pitch: m.movement.pitch,
                                    start_tick: m.start_tick,
                                    time_ticks: (m.movement.move_time * 1_000_000_000.0) as u64,
                                });
                            }

                            pos = m.movement.pos_after_move(pos);
                        }
                        let message = entities_proto::EntityUpdate {
                            id: entity.id,
                            entity_class: entity.class,
                            planned_move: moves_to_send,
                            remove: false,
                            trailing_entity: entity
                                .trailing_entities
                                .unwrap_or(&[])
                                .iter()
                                .map(|e| entities_proto::TrailingEntity {
                                    class: e.class_id,
                                    distance: e.trailing_distance,
                                })
                                .collect(),
                        };
                        messages.push(message);
                        Ok(())
                    })?;
                }
                Ok(())
            })?;
            let to_remove = known_to_us.difference(&still_valid).collect::<Vec<_>>();

            for &entity_id in to_remove.into_iter() {
                messages.push(entities_proto::EntityUpdate {
                    id: entity_id,
                    planned_move: vec![],
                    remove: true,
                    // entity class doesn't matter since we're going to remove it
                    entity_class: 0,
                    trailing_entity: vec![],
                });
                self.sent_entities.remove(&entity_id);
            }

            messages
        };
        let tick = self.context.game_state.tick();
        for message in messages {
            self.outbound_tx
                .send(Ok(StreamToClient {
                    tick,
                    server_message: Some(ServerMessage::EntityUpdate(message)),
                    performance_metrics: self.context.maybe_get_performance_metrics(),
                }))
                .await?;
        }
        self.last_update = update_time;
        Ok(())
    }
}

// TODO tune these and make them adjustable via settings
// Units of chunks
// Chunks within this distance will be loaded into memory if not yet loaded
const LOAD_EAGER_DISTANCE: i32 = 25;
// Chunks within this distance will be sent if they are already loaded into memory
// TODO: This has to equal LOAD_EAGER_DISTANCE to avoid an issue where a chunk is lazily processed,
// and then never eagerly re-processed as it gets closer
const LOAD_LAZY_DISTANCE: i32 = 25;
// How wide of a range we'll look at with mapgen assist only at terrain level
const LOAD_TERRAIN_DISTANCE: i32 = 50;
const UNLOAD_DISTANCE: i32 = 200;
// Chunks within this distance will be sent, even if flow control would otherwise prevent them from being sent
const FORCE_LOAD_DISTANCE: i32 = 4;

const MAX_UPDATE_BATCH_SIZE: usize = 256;

const INITIAL_CHUNKS_PER_UPDATE: usize = 128;
const MAX_CHUNKS_PER_UPDATE: usize = 4096;
const VERTICAL_DISTANCE_CAP: i32 = 10;

lazy_static::lazy_static! {
    static ref LOAD_LAZY_ZIGZAG_VEC: Vec<i32> = {
        let mut v = vec![0];
        for i in 1..=LOAD_LAZY_DISTANCE {
            v.push(i);
            v.push(-i);
        }
        v
    };
    static ref LOAD_TERRAIN_VEC_2D: Vec<i32> = {
        let mut v = vec![0];
        for i in 1..=LOAD_TERRAIN_DISTANCE {
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
        v.sort_by_key(|(x, y, z)| x.abs() + y.abs() + z.abs());
        v.retain(|x| (x.0.abs() + x.1.abs() + x.2.abs()) <= LOAD_LAZY_DISTANCE);
        v
    };
    static ref LOAD_TERRAIN_SORTED_COORDS: Vec<(i32, i32)> = {
        let mut v = vec![];
        for (&x, &z) in iproduct!(LOAD_TERRAIN_VEC_2D.iter(), LOAD_TERRAIN_VEC_2D.iter()) {
            v.push((x, z));
        }
        v.sort_by_key(|(x, z)| x.abs() + z.abs());
        v.retain(|x| (x.0.abs() + x.1.abs()) <= LOAD_TERRAIN_DISTANCE);
        v
    };
    // chunk distance -> min index where we would encounter it
    static ref MIN_INDEX_FOR_DISTANCE: Vec<usize> = {
        let mut v = vec![];
        v.resize((LOAD_LAZY_DISTANCE + 1) as usize, usize::MAX);
        for (i, &(x, y, z)) in LOAD_LAZY_SORTED_COORDS.iter().enumerate() {
            let distance = (x.abs() + y.abs() + z.abs()) as usize;
            v[distance] = v[distance].min(i);
        }
        for x in &v {
            assert_ne!(*x, usize::MAX);
        }
        v
    };
    // chunk distance -> max index where we would encounter it
    static ref MAX_INDEX_FOR_DISTANCE: Vec<usize> = {
        let mut v = vec![0; (LOAD_LAZY_DISTANCE + 1) as usize];
        for (i, &(x, y, z)) in LOAD_LAZY_SORTED_COORDS.iter().enumerate() {
            let distance = (x.abs() + y.abs() + z.abs()) as usize;
            v[distance] = v[distance].max(i);
        }
        v
    };
}

#[derive(Copy, Clone, Debug)]
struct PositionAndPacing {
    position: PlayerPositionUpdate,
    chunks_to_send: usize,
    client_requested_render_distance: i32,
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

    fn decrease_to_floor(&mut self) {
        self.val = self.floor;
    }
    fn get(&self) -> usize {
        self.val as usize
    }
}

fn make_inventory_update(
    context: &SharedContext,
    view: &dyn TypeErasedInventoryView,
) -> Result<StreamToClient> {
    block_in_place(|| {
        Ok(StreamToClient {
            tick: context.game_state.tick(),
            server_message: Some(ServerMessage::InventoryUpdate(view.to_client_proto()?)),
            // TODO: make them available here, although we'd need to refactor a lot of call sites
            performance_metrics: context.maybe_get_performance_metrics(),
        })
    })
}

pub(crate) struct AudioSender {
    context: Arc<SharedContext>,

    // RPC stream is sent via this channel
    outbound_tx: mpsc::Sender<tonic::Result<StreamToClient>>,
    events: AudioCrossbarReceiver,
    position_watch: watch::Receiver<PositionAndPacing>,
}
impl AudioSender {
    #[tracing::instrument(
        name = "AudioSenderLoop",
        level = "info",
        skip(self),
        fields(
            player_name=%self.context.player_context.name(),
        )
    )]
    pub(crate) async fn audio_sender_loop(&mut self) -> Result<()> {
        while !self.context.cancellation.is_cancelled() {
            tokio::select! {
                event = self.events.recv(self.position_watch.borrow().position.position) => {
                    self.handle_event(event?).await?;
                }
                _ = self.context.cancellation.cancelled() => {
                    info!("Client outbound loop {} detected cancellation and shutting down", self.context.id)
                    // pass
                }
            }
        }
        Ok(())
    }

    pub(crate) async fn handle_event(&mut self, event: AudioEvent) -> Result<()> {
        if event.initiating_context_id == self.context.id {
            // Don't send player-initiated events back to themselves
            return Ok(());
        }
        match event.instruction {
            AudioInstruction::PlaySampledSound(x) => {
                self.outbound_tx
                    .send(Ok(StreamToClient {
                        tick: self.context.game_state.tick(),
                        server_message: Some(ServerMessage::PlaySampledSound(x)),
                        performance_metrics: self.context.maybe_get_performance_metrics(),
                    }))
                    .await?
            }
        }

        Ok(())
    }
}
