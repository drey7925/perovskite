use crate::{
    client_state::{
        entities::GameEntity, items::ClientInventory, ClientState, FastChunkNeighbors, GameAction,
    },
    net_client::{MAX_PROTOCOL_VERSION, MIN_PROTOCOL_VERSION},
};
use anyhow::{anyhow, ensure, Context, Result};
use cgmath::{vec3, InnerSpace, Vector3, Zero};
use futures::StreamExt;
use log::warn;
use parking_lot::Mutex;
use perovskite_core::{
    chat::ChatMessage,
    coordinates::{BlockCoordinate, ChunkCoordinate, PlayerPositionUpdate},
    protocol::entities as entities_proto,
    protocol::game_rpc::{self as rpc, InteractKeyAction, StreamToClient, StreamToServer},
};
use prost::Message;
use std::ops::Deref;
use std::sync::atomic::Ordering;
use std::{
    backtrace,
    collections::{hash_map::Entry, HashMap},
    sync::Arc,
    time::{Duration, Instant},
};

use super::mesh_worker::{
    propagate_neighbor_data, MeshBatcher, MeshWorker, NeighborPropagationScratchpad,
    NeighborPropagator,
};
use crate::audio::{
    EvictedAudioHealer, SimpleSoundControlBlock, SOUND_MOVESPEED_ENABLED, SOUND_PRESENT,
    SOUND_SQUARELAW_ENABLED,
};
use perovskite_core::block_id::BlockId;
use perovskite_core::protocol::game_rpc::Footstep;
use perovskite_core::protocol::map::StoredChunk;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tonic::Streaming;
use tracy_client::{plot, span};

struct SharedState {
    protocol_version: u32,
    // Some messages are sent straight from the inbound context, namely protocol bugchecks
    outbound_tx: mpsc::Sender<StreamToServer>,
    client_state: Arc<ClientState>,
    ack_map: Mutex<HashMap<u64, Instant>>,
    mesh_workers: Vec<Arc<MeshWorker>>,
    neighbor_propagators: Vec<Arc<NeighborPropagator>>,
    audio_healer: Arc<EvictedAudioHealer>,

    batcher: Arc<MeshBatcher>,
    initial_state_notification: Arc<tokio::sync::Notify>,
    cancellation: CancellationToken,
}
impl SharedState {
    async fn send_bugcheck(&self, description: String) -> Result<()> {
        log::error!("Protocol bugcheck: {}", description);
        self.outbound_tx
            .send(StreamToServer {
                sequence: 0,
                client_tick: 0,
                client_message: Some(rpc::stream_to_server::ClientMessage::BugCheck(
                    rpc::ClientBugCheck {
                        description,
                        backtrace: backtrace::Backtrace::force_capture().to_string(),
                        protocol_version: self.protocol_version,
                        min_protocol_version: MIN_PROTOCOL_VERSION,
                        max_protocol_version: MAX_PROTOCOL_VERSION,
                    },
                )),
                want_performance_metrics: self
                    .client_state
                    .want_server_perf
                    .load(Ordering::Relaxed),
            })
            .await?;
        Ok(())
    }
}

pub(crate) async fn make_contexts(
    client_state: Arc<ClientState>,
    tx_send: mpsc::Sender<StreamToServer>,
    stream: Streaming<StreamToClient>,
    action_receiver: mpsc::Receiver<GameAction>,
    // Not yet used, only one protocol version is supported
    protocol_version: u32,
    initial_state_notification: Arc<tokio::sync::Notify>,
) -> Result<(InboundContext, OutboundContext)> {
    let cancellation = client_state.shutdown.clone();

    let mut mesh_workers = vec![];
    let mesh_worker_handles = futures::stream::FuturesUnordered::new();
    for _ in 0..client_state.settings.load().render.num_mesh_workers {
        let (worker, handle) = MeshWorker::new(client_state.clone());
        mesh_workers.push(worker);
        mesh_worker_handles.push(handle);
    }
    let mut neighbor_propagators = vec![];
    let neighbor_propagator_handles = futures::stream::FuturesUnordered::new();
    for _ in 0..client_state.settings.load().render.num_neighbor_propagators {
        let (worker, handle) = NeighborPropagator::new(client_state.clone(), mesh_workers.clone());
        neighbor_propagators.push(worker);
        neighbor_propagator_handles.push(handle);
    }

    let (batcher, batcher_handle) = MeshBatcher::new(client_state.clone());

    let (audio_healer, audio_healer_handle) = EvictedAudioHealer::new(client_state.clone());

    let shared_state = Arc::new(SharedState {
        protocol_version,
        outbound_tx: tx_send,
        client_state,
        ack_map: Mutex::new(HashMap::new()),
        mesh_workers,
        neighbor_propagators,
        batcher,
        initial_state_notification,
        audio_healer,
        cancellation,
    });

    let inbound = InboundContext {
        inbound_rx: stream,
        shared_state: shared_state.clone(),
        mesh_worker_handles,
        neighbor_propagator_handles,
        batcher_handle,

        snappy_helper: SnappyDecodeHelper::new(),
        inline_nprop_scratchpad: NeighborPropagationScratchpad::default(),
        inline_fcn_scratchpad: FastChunkNeighbors::default(),
        audio_healer_handle,
    };

    let outbound = OutboundContext {
        shared_state,
        sequence: 1,
        action_receiver,
        last_pos_update_seq: None,
    };

    Ok((inbound, outbound))
}

pub(crate) struct OutboundContext {
    sequence: u64,

    action_receiver: mpsc::Receiver<GameAction>,
    last_pos_update_seq: Option<u64>,
    shared_state: Arc<SharedState>,
}
impl OutboundContext {
    async fn send_sequenced_message(
        &mut self,
        message: rpc::stream_to_server::ClientMessage,
    ) -> Result<u64> {
        // todo tick
        let start_time = Instant::now();
        self.sequence += 1;
        self.shared_state
            .ack_map
            .lock()
            .insert(self.sequence, Instant::now());
        self.shared_state
            .outbound_tx
            .send(StreamToServer {
                sequence: self.sequence,
                client_tick: 0,
                client_message: Some(message),
                want_performance_metrics: self
                    .shared_state
                    .client_state
                    .want_server_perf
                    .load(Ordering::Relaxed),
            })
            .await?;
        let now = Instant::now();
        plot!("send_sequnced wait time", (now - start_time).as_secs_f64());
        Ok(self.sequence)
    }

    /// Send a position update to the server
    async fn send_position_update(&mut self, pos: PlayerPositionUpdate) -> Result<()> {
        if let Some(last_pos_send) = self
            .last_pos_update_seq
            .and_then(|x| self.shared_state.ack_map.lock().get(&x).copied())
        {
            // We haven't gotten an ack for the last pos update; withhold the current one
            let delay = Instant::now() - last_pos_send;
            if delay > Duration::from_secs_f64(0.25) {
                log::warn!("Waiting {delay:?} for a position update");
            }
            plot!("pos_update_wait", delay.as_secs_f64());
            // todo send an update but signal that we don't want chunks as far as AIMD/pacing on
            // the server
            return Ok(());
        } else {
            plot!("pos_update_wait", 0.);
        }

        // If this overflows, the client is severely behind (by 4 billion chunks!) and may as well crash
        let pending_chunks = self
            .shared_state
            .mesh_workers
            .iter()
            .map(|worker| worker.queue_len())
            .sum::<usize>()
            .max(
                self.shared_state
                    .neighbor_propagators
                    .iter()
                    .map(|x| x.queue_len())
                    .sum::<usize>(),
            )
            .try_into()
            .unwrap();
        let hotbar_slot = self.shared_state.client_state.hud.lock().hotbar_slot;

        let animation_state = self
            .shared_state
            .client_state
            .physics_state
            .lock()
            .take_animation_state();
        let sequence = self
            .send_sequenced_message(rpc::stream_to_server::ClientMessage::PositionUpdate(
                rpc::ClientUpdate {
                    position: Some(pos.to_proto()?),
                    pacing: Some(rpc::ClientPacing {
                        pending_chunks,
                        distance_limit: self
                            .shared_state
                            .client_state
                            .render_distance
                            .load(Ordering::Relaxed),
                    }),
                    hotbar_slot,
                    footstep_coordinate: animation_state
                        .footstep_coord
                        .into_iter()
                        .map(|(tick, coord)| Footstep {
                            coord: Some(coord.into()),
                            tick,
                        })
                        .collect(),
                },
            ))
            .await?;
        self.last_pos_update_seq = Some(sequence);
        Ok(())
    }

    async fn handle_game_action(&mut self, action: GameAction) -> Result<()> {
        self.send_position_update(self.shared_state.client_state.last_position())
            .await?;
        match action {
            GameAction::Dig(action) => {
                self.send_sequenced_message(rpc::stream_to_server::ClientMessage::Dig(
                    rpc::DigTapAction {
                        action_target: Some(action.target.into()),
                        prev_coord: action.prev.map(|x| x.into()),
                        item_slot: action.item_slot,
                        position: Some(action.player_pos.to_proto()?),
                    },
                ))
                .await?;
            }
            GameAction::Tap(action) => {
                self.send_sequenced_message(rpc::stream_to_server::ClientMessage::Tap(
                    rpc::DigTapAction {
                        action_target: Some(action.target.into()),
                        prev_coord: action.prev.map(|x| x.into()),
                        item_slot: action.item_slot,
                        position: Some(action.player_pos.to_proto()?),
                    },
                ))
                .await?;
            }
            GameAction::Place(action) => {
                self.send_sequenced_message(rpc::stream_to_server::ClientMessage::Place(
                    rpc::PlaceAction {
                        block_coord: action.target.map(Into::into),
                        place_anchor: Some(action.anchor.into()),
                        item_slot: action.item_slot,
                        position: Some(action.player_pos.to_proto()?),
                    },
                ))
                .await?;
            }
            GameAction::Inventory(action) => {
                self.send_sequenced_message(rpc::stream_to_server::ClientMessage::Inventory(
                    rpc::InventoryAction {
                        source_view: action.source_view,
                        source_slot: action.source_slot.try_into()?,
                        destination_view: action.destination_view,
                        destination_slot: action.destination_slot.try_into()?,
                        count: action.count,
                        swap: action.swap,
                    },
                ))
                .await?;
            }
            GameAction::PopupResponse(popup_response) => {
                self.send_sequenced_message(rpc::stream_to_server::ClientMessage::PopupResponse(
                    popup_response,
                ))
                .await?;
            }
            GameAction::InteractKey(action) => {
                self.send_sequenced_message(rpc::stream_to_server::ClientMessage::InteractKey(
                    InteractKeyAction {
                        interaction_target: Some(action.target.into()),
                        position: Some(action.player_pos.to_proto()?),
                        item_slot: action.item_slot,
                    },
                ))
                .await?;
            }
            GameAction::ChatMessage(message) => {
                self.send_sequenced_message(rpc::stream_to_server::ClientMessage::ChatMessage(
                    message,
                ))
                .await?;
            }
        }
        Ok(())
    }

    pub(crate) async fn run_outbound_loop(&mut self) -> Result<()> {
        let mut position_tx_timer = tokio::time::interval(Duration::from_secs_f64(0.1));
        position_tx_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        while !self.shared_state.cancellation.is_cancelled() {
            tokio::select! {
                _ = position_tx_timer.tick() => {
                    // TODO send updates at a lower rate if substantially unchanged
                    self.send_position_update(self.shared_state.client_state.last_position()).await?;
                },

                _ = self.shared_state.cancellation.cancelled() => {
                    log::info!("Outbound stream context detected cancellation and shutting down")
                    // pass
                }

                action = self.action_receiver.recv() => {
                    match action {
                        Some(x) => self.handle_game_action(x).await?,
                        None => {
                            log::warn!("Action sender closed, shutting down outbound loop");
                            self.shared_state.cancellation.cancel();
                        }
                    }
                }
            }
        }
        log::warn!("Exiting outbound loop");
        Ok(())
    }
}

pub(crate) struct InboundContext {
    inbound_rx: Streaming<StreamToClient>,
    shared_state: Arc<SharedState>,

    mesh_worker_handles: futures::stream::FuturesUnordered<tokio::task::JoinHandle<Result<()>>>,
    neighbor_propagator_handles:
        futures::stream::FuturesUnordered<tokio::task::JoinHandle<Result<()>>>,
    batcher_handle: tokio::task::JoinHandle<Result<()>>,

    snappy_helper: SnappyDecodeHelper,
    inline_nprop_scratchpad: NeighborPropagationScratchpad,
    inline_fcn_scratchpad: FastChunkNeighbors,

    audio_healer_handle: tokio::task::JoinHandle<Result<()>>,
}
impl InboundContext {
    pub(crate) async fn run_inbound_loop(&mut self) -> Result<()> {
        while !self.shared_state.cancellation.is_cancelled() {
            tokio::select! {
                message = self.inbound_rx.message() => {
                    match message {
                        Err(e) => {
                            log::warn!("Server sent an error: {:?}", e);
                            *self.shared_state.client_state.pending_error.lock() = Some(anyhow::Error::from(e.clone()));
                            return Err(e.into());
                        },
                        Ok(None) => {
                            log::info!("Server disconnected");
                            let mut pending_error = self.shared_state.client_state.pending_error.lock();
                            if pending_error.is_none() {
                                *pending_error = Some(anyhow!("Server disconnected unexpectedly without sending a detailed error message"));
                            }

                            self.shared_state.cancellation.cancel();
                        }
                        Ok(Some(mut message)) => {
                            match self.handle_message(&mut message).await {
                                Ok(_) => {},
                                Err(e) => {
                                    log::warn!("Client failed to handle message: {:?}, error: {:?}", message, e);
                                },
                            }
                        }
                    }

                }
                _ = self.shared_state.cancellation.cancelled() => {
                    log::info!("Inbound stream context detected cancellation and shutting down")
                    // pass
                },
                result = &mut self.mesh_worker_handles.next() => {
                    match &result {
                        Some(Err(e)) => {
                            log::error!("Error awaiting mesh worker: {e:?}");
                        }
                        Some(Ok(Err(e))) => {
                            log::error!("Mesh worker crashed: {e:?}");
                        }
                        Some(Ok(_)) => {
                            log::info!("Mesh worker exiting");
                        }
                        None => {
                            log::error!("Mesh worker exited unexpectedly");
                        }
                    }
                    break;
                }
                result = &mut self.neighbor_propagator_handles.next() => {
                    match &result {
                        Some(Err(e)) => {
                            log::error!("Error awaiting neighbor propagator: {e:?}");
                        }
                        Some(Ok(Err(e))) => {
                            log::error!("Neighbor propagator crashed: {e:?}");
                        }
                        Some(Ok(_)) => {
                            log::info!("Neighbor propagator exiting");
                        }
                        None => {
                            log::error!("Neighbor propagator exited unexpectedly");
                        }
                    }
                    break;
                },
                result = &mut self.batcher_handle => {
                    match &result {
                        Ok(Err(e)) => {
                            log::error!("Mesh batcher crashed: {e:?}");
                        },
                        Ok(Ok(_)) => {
                            log::info!("Mesh batcher exiting");
                        },
                        Err(e) => {
                            log::error!("Error awaiting mesh batcher: {e:?}");
                        }
                    }
                },
                result = &mut self.audio_healer_handle => {
                    match &result {
                        Ok(Err(e)) => {
                            log::error!("Audio healer crashed: {e:?}");
                        },
                        Ok(Ok(_)) => {
                            log::info!("Audio healer exiting");
                        },
                        Err(e) => {
                            log::error!("Error awaiting audio healer: {e:?}");
                        }
                    }
                }
            }
        }
        log::warn!("Exiting inbound loop");
        // Notify the mesh worker so it can exit soon
        for worker in self.shared_state.mesh_workers.iter() {
            worker.cancel();
        }
        for worker in self.shared_state.neighbor_propagators.iter() {
            worker.cancel();
        }
        self.shared_state.batcher.cancel();
        Ok(())
    }
    async fn handle_message(&mut self, message: &StreamToClient) -> Result<()> {
        if message.tick == 0 {
            log::warn!("Got message with tick 0");
        } else {
            self.shared_state
                .client_state
                .timekeeper
                .update_error(message.tick);
        }
        *self.shared_state.client_state.server_perf.lock() = message.performance_metrics.clone();
        match &message.server_message {
            None => {
                log::warn!("Got empty message from server");
            }
            Some(rpc::stream_to_client::ServerMessage::HandledSequence(sequence)) => {
                self.handle_ack(*sequence).await?;
            }
            Some(rpc::stream_to_client::ServerMessage::MapChunk(chunk)) => {
                self.handle_mapchunk(chunk).await?;
            }
            Some(rpc::stream_to_client::ServerMessage::MapChunkUnsubscribe(unsub)) => {
                self.handle_unsubscribe(unsub).await?;
            }
            Some(rpc::stream_to_client::ServerMessage::MapDeltaUpdate(delta_update)) => {
                self.handle_map_delta_update(delta_update).await?;
            }
            Some(rpc::stream_to_client::ServerMessage::InventoryUpdate(inventory_update)) => {
                self.handle_inventory_update(inventory_update).await?;
            }
            Some(rpc::stream_to_client::ServerMessage::ClientState(client_state)) => {
                self.handle_client_state_update(client_state).await?;
            }
            Some(rpc::stream_to_client::ServerMessage::ShowPopup(popup_desc)) => {
                self.shared_state
                    .client_state
                    .egui
                    .lock()
                    .show_popup(popup_desc);
            }
            Some(rpc::stream_to_client::ServerMessage::ChatMessage(message)) => self
                .shared_state
                .client_state
                .chat
                .lock()
                .message_history
                .push(
                    ChatMessage::new(&message.origin, &message.message)
                        .with_color_fixed32(message.color_argb),
                ),
            Some(rpc::stream_to_client::ServerMessage::ShutdownMessage(msg)) => {
                *self.shared_state.client_state.pending_error.lock() =
                    Some(anyhow!("Server shutdown message: {}", &msg));
            }
            Some(rpc::stream_to_client::ServerMessage::EntityUpdate(update)) => {
                let estimated_send_time = self
                    .shared_state
                    .client_state
                    .timekeeper
                    .adjust_server_tick(message.tick);
                self.handle_entity_update(update, estimated_send_time)
                    .await?;
            }
            Some(rpc::stream_to_client::ServerMessage::PlaySampledSound(sound)) => {
                let position = match sound.position {
                    Some(p) => p.try_into()?,
                    None => {
                        ensure!(sound.disable_doppler && sound.disable_falloff);
                        Vector3::zero()
                    }
                };
                let tick_now = self.shared_state.client_state.timekeeper.now();
                let tick = if sound.tick == 0 {
                    tick_now
                } else {
                    sound.tick
                };

                let mut flags = SOUND_PRESENT;
                if !sound.disable_doppler {
                    flags |= SOUND_MOVESPEED_ENABLED;
                };
                if !sound.disable_falloff {
                    flags |= SOUND_SQUARELAW_ENABLED;
                };
                let control_block = SimpleSoundControlBlock {
                    flags,
                    position,
                    volume: sound.volume.clamp(0.0, 1.0),
                    start_tick: tick,
                    id: sound.sound_id,
                    end_tick: tick
                        + self
                            .shared_state
                            .client_state
                            .audio
                            .sampled_sound_length(sound.sound_id)
                            .unwrap_or(0),
                    source: sound.source(),
                };
                self.shared_state
                    .client_state
                    .audio
                    .insert_or_update_simple_sound(
                        tick_now,
                        self.shared_state
                            .client_state
                            .weakly_ordered_last_position()
                            .position,
                        control_block,
                        None,
                    );
            }
            Some(_) => {
                log::warn!("Unimplemented server->client message {:?}", message);
            }
        }
        Ok(())
    }

    fn enqueue_for_nprop(&self, coord: ChunkCoordinate) {
        self.shared_state.neighbor_propagators
            [coord.hash_u64() as usize % self.shared_state.neighbor_propagators.len()]
        .enqueue(coord);
    }

    fn enqueue_for_meshing(&self, coord: ChunkCoordinate) {
        self.shared_state.mesh_workers
            [coord.hash_u64() as usize % self.shared_state.mesh_workers.len()]
        .enqueue(coord);
    }

    async fn handle_mapchunk(&mut self, chunk: &rpc::MapChunk) -> Result<()> {
        match &chunk.chunk_coord {
            Some(coord) => {
                tokio::task::block_in_place(|| {
                    let _span = span!("handle_mapchunk");
                    let coord = coord.into();

                    let data = self
                        .snappy_helper
                        .decode::<StoredChunk>(&chunk.snappy_encoded_bytes)?
                        .chunk_data
                        .with_context(|| "inner chunk_data missing")?;
                    let block_ids: &[u32; 4096] = match &data {
                        perovskite_core::protocol::map::stored_chunk::ChunkData::V1(v1_data) => {
                            ensure!(v1_data.block_ids.len() == 4096);
                            v1_data.block_ids.deref().try_into().unwrap()
                        }
                    };
                    let extra_chunks = self.shared_state.client_state.chunks.insert_or_update(
                        &self.shared_state.client_state,
                        coord,
                        block_ids,
                        &self.shared_state.client_state.block_types,
                    )?;

                    self.enqueue_for_nprop(coord);

                    for i in -1..=1 {
                        for j in (-1 - extra_chunks as i32)..=1 {
                            for k in -1..=1 {
                                if let Some(neighbor) = coord.try_delta(i, j, k) {
                                    self.enqueue_for_nprop(neighbor);
                                }
                            }
                        }
                    }
                    Ok::<(), anyhow::Error>(())
                })?;
            }
            None => {
                self.shared_state
                    .send_bugcheck("Got chunk without a coordinate".to_string())
                    .await?;
            }
        };
        Ok(())
    }

    async fn handle_ack(&mut self, seq: u64) -> Result<()> {
        let send_time = self.shared_state.ack_map.lock().remove(&seq);
        match send_time {
            Some(time) => {
                // todo track in a histogram
                //log::info!("Seq {} took {:?}", seq, Instant::now() - time)
                plot!("ack_rtt", (Instant::now() - time).as_secs_f64());
            }
            None => {
                let desc = format!("got ack for seq {} which we didn't send", seq);
                self.shared_state.send_bugcheck(desc).await?;
            }
        }
        Ok(())
    }
    async fn handle_unsubscribe(&mut self, unsub: &rpc::MapChunkUnsubscribe) -> Result<()> {
        // TODO hold more old chunks (possibly LRU) to provide a higher render distance
        let mut bad_coords = vec![];
        let mut chunk_lock = self.shared_state.client_state.chunks.chunk_lock();
        for coord in unsub.chunk_coord.iter() {
            let coord = coord.into();
            match self
                .shared_state
                .client_state
                .chunks
                .remove_locked(&coord, &mut chunk_lock)
            {
                Some(_x) => {
                    self.shared_state
                        .client_state
                        .world_audio
                        .lock()
                        .remove_chunk(coord);
                }
                None => {
                    bad_coords.push(coord.clone());
                }
            }
        }
        if !bad_coords.is_empty() {
            self.shared_state
                .send_bugcheck(format!(
                    "Asked to unsubscribe to chunks we never subscribed to: {:?}",
                    bad_coords
                ))
                .await?
        }
        Ok(())
    }

    async fn handle_map_delta_update(&mut self, batch: &rpc::MapDeltaUpdateBatch) -> Result<()> {
        let (missing_coord, unknown_coords) =
            tokio::task::block_in_place(|| self.map_delta_update_sync(batch))?;
        if missing_coord {
            self.shared_state
                .send_bugcheck("Missing block_coord in delta update".to_string())
                .await?;
        }
        // In the latest concurrency model, this is not a bug - it's a more conservative/harmless
        // outcome of a race condition.
        for coord in unknown_coords {
            warn!("Received coord {coord:?} is not in cache. This is fine if it happens occasionally, but if it happens constantly, it suggests either a bug or severe server overload.")
        }

        Ok(())
    }

    fn apply_map_audio_delta_batch(&self, batch: &rpc::MapDeltaUpdateBatch) {
        let tick = self.shared_state.client_state.timekeeper.now();
        let pos = self
            .shared_state
            .client_state
            .weakly_ordered_last_position();
        let block_types = self.shared_state.client_state.block_types.deref();
        let mut map_sound = self.shared_state.client_state.world_audio.lock();

        for entry in &batch.updates {
            let coord: BlockCoordinate = match entry.block_coord {
                None => {
                    // The 3d rendering map will complain again, we can be silent here
                    continue;
                }
                Some(c) => c.into(),
            };
            match block_types.block_sound(BlockId::from(entry.new_id)) {
                None => map_sound.remove(coord),
                Some((id, vol)) => {
                    map_sound.insert_or_update(tick, pos.position, coord, id.get(), vol)
                }
            }
        }
    }

    fn map_delta_update_sync(
        &mut self,
        batch: &rpc::MapDeltaUpdateBatch,
    ) -> Result<(bool, Vec<BlockCoordinate>), anyhow::Error> {
        self.apply_map_audio_delta_batch(batch);

        let (needs_remesh, unknown_coords, missing_coord) =
            self.shared_state
                .client_state
                .chunks
                .apply_delta_batch(batch, &self.shared_state.client_state.block_types)?;

        {
            let _span = span!("remesh for delta");

            // Only do inline meshing for chunks within 3 chunks of the current player location
            // Otherwise, we tie up the network thread for too long
            let current_position = self.shared_state.client_state.last_position().position;
            let eligible_for_inline = |coord: ChunkCoordinate| {
                let base = vec3(
                    coord.x as f64 * 16.0 + 8.0,
                    coord.y as f64 * 16.0 + 8.0,
                    coord.z as f64 * 16.0 + 8.0,
                );
                (base - current_position).magnitude2() < (48.0 * 48.0)
            };
            for &coord in needs_remesh.iter() {
                if eligible_for_inline(coord) {
                    self.shared_state
                        .client_state
                        .chunks
                        .cloned_neighbors_fast(coord, &mut self.inline_fcn_scratchpad);
                    propagate_neighbor_data(
                        &self.shared_state.client_state.block_types,
                        &self.inline_fcn_scratchpad,
                        &mut self.inline_nprop_scratchpad,
                    )?;
                } else {
                    self.enqueue_for_nprop(coord);
                }
            }
            for coord in needs_remesh {
                if eligible_for_inline(coord) {
                    self.shared_state
                        .client_state
                        .chunks
                        .maybe_mesh_and_maybe_promote(
                            coord,
                            &self.shared_state.client_state.block_renderer,
                        )?;
                } else {
                    self.enqueue_for_meshing(coord);
                }
            }
        }
        Ok((missing_coord, unknown_coords))
    }

    async fn handle_inventory_update(
        &mut self,
        inventory_update: &rpc::InventoryUpdate,
    ) -> Result<()> {
        if inventory_update.inventory.is_none() {
            return self
                .shared_state
                .send_bugcheck("Missing inventory in InventoryUpdate".to_string())
                .await;
        };
        let inv = ClientInventory::from_proto(inventory_update);
        let mut hud_lock = self.shared_state.client_state.hud.lock();
        if Some(inventory_update.view_id) == hud_lock.hotbar_view_id {
            hud_lock.invalidate_hotbar();
            let slot = hud_lock.hotbar_slot();
            drop(hud_lock);
            self.shared_state
                .client_state
                .tool_controller
                .lock()
                .change_held_item(
                    &self.shared_state.client_state,
                    slot,
                    inv.contents()[slot as usize]
                        .clone()
                        .and_then(|x| self.shared_state.client_state.items.get(&x.item_name))
                        .cloned(),
                )
        }

        self.shared_state
            .client_state
            .inventories
            .lock()
            .inventory_views
            .insert(inventory_update.view_id, inv);
        Ok(())
    }

    async fn handle_client_state_update(
        &mut self,
        state_update: &rpc::SetClientState,
    ) -> Result<()> {
        self.shared_state.initial_state_notification.notify_one();
        self.shared_state
            .client_state
            .handle_server_update(state_update)
    }

    async fn handle_entity_update(
        &mut self,
        update: &entities_proto::EntityUpdate,
        estimated_send_tick: u64,
    ) -> Result<()> {
        if update.remove {
            if self
                .shared_state
                .client_state
                .entities
                .lock()
                .remove_entity(update.id, &self.shared_state.client_state)
                .is_none()
            {
                self.shared_state
                    .send_bugcheck(format!("Got remove for non-existent entity {}", update.id))
                    .await?;
            }
            return Ok(());
        }

        let outcome = match self
            .shared_state
            .client_state
            .entities
            .lock()
            .entities
            .entry(update.id)
        {
            Entry::Occupied(mut entry) => entry.get_mut().update(update, estimated_send_tick),
            Entry::Vacant(entry) => {
                match GameEntity::from_proto(update, self.shared_state.client_state.deref()) {
                    Ok(x) => {
                        entry.insert(x);
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            }
        };

        if let Err(e) = outcome {
            self.shared_state
                .send_bugcheck(format!("Failed to handle entity update: {:?}", e))
                .await?;
        }

        Ok(())
    }
}

pub(crate) struct SnappyDecodeHelper {
    snappy_decoder: snap::raw::Decoder,
    snappy_output_buffer: Vec<u8>,
}
impl SnappyDecodeHelper {
    fn decode<T>(&mut self, data: &[u8]) -> Result<T>
    where
        T: Message + Default,
    {
        let decode_len = snap::raw::decompress_len(data)?;
        if self.snappy_output_buffer.len() < decode_len {
            self.snappy_output_buffer.resize(decode_len, 0);
        }
        let decompressed_len = self
            .snappy_decoder
            .decompress(data, &mut self.snappy_output_buffer)?;
        Ok(T::decode(&self.snappy_output_buffer[0..decompressed_len])?)
    }

    pub(crate) fn new() -> SnappyDecodeHelper {
        SnappyDecodeHelper {
            snappy_decoder: snap::raw::Decoder::new(),
            snappy_output_buffer: Vec::new(),
        }
    }
}
