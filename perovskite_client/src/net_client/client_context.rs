use std::{
    backtrace,
    collections::{hash_map::Entry, HashMap},
    sync::Arc,
    time::{Duration, Instant},
};

use crate::{
    game_state::{
        chunk::SnappyDecodeHelper, entities::GameEntity, items::ClientInventory, ClientState,
        GameAction,
    },
    net_client::{MAX_PROTOCOL_VERSION, MIN_PROTOCOL_VERSION},
};
use anyhow::Result;
use cgmath::{vec3, InnerSpace};
use futures::StreamExt;
use parking_lot::Mutex;
use perovskite_core::{
    chat::ChatMessage,
    coordinates::{BlockCoordinate, ChunkCoordinate, PlayerPositionUpdate},
    protocol::game_rpc::{self as rpc, InteractKeyAction, StreamToClient, StreamToServer},
};

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tonic::Streaming;
use tracy_client::{plot, span};

use super::mesh_worker::{propagate_neighbor_data, MeshBatcher, MeshWorker, NeighborPropagator};

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

    let ack_map = Arc::new(Mutex::new(HashMap::new()));
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

    let inbound = InboundContext {
        outbound_tx: tx_send.clone(),
        inbound_rx: stream,
        client_state: client_state.clone(),
        cancellation: cancellation.clone(),
        ack_map: ack_map.clone(),
        mesh_workers: mesh_workers.clone(),
        mesh_worker_handles,
        neighbor_propagators: neighbor_propagators.clone(),
        neighbor_propagator_handles,
        batcher,
        batcher_handle,
        snappy_helper: SnappyDecodeHelper::new(),
        protocol_version,
        initial_state_notification,
    };

    let outbound = OutboundContext {
        outbound_tx: tx_send,
        sequence: 1,
        client_state,
        cancellation,
        ack_map,
        action_receiver,
        last_pos_update_seq: None,
        mesh_workers,
        neighbor_propagators,
        protocol_version,
    };

    Ok((inbound, outbound))
}

pub(crate) struct OutboundContext {
    outbound_tx: mpsc::Sender<rpc::StreamToServer>,
    sequence: u64,
    client_state: Arc<ClientState>,
    // Cancellation, shared with the inbound context
    pub(crate) cancellation: CancellationToken,
    ack_map: Arc<Mutex<HashMap<u64, Instant>>>,

    action_receiver: mpsc::Receiver<GameAction>,
    last_pos_update_seq: Option<u64>,

    // Used only for pacing
    mesh_workers: Vec<Arc<MeshWorker>>,
    neighbor_propagators: Vec<Arc<NeighborPropagator>>,

    protocol_version: u32,
}
impl OutboundContext {
    async fn send_sequenced_message(
        &mut self,
        message: rpc::stream_to_server::ClientMessage,
    ) -> Result<u64> {
        // todo tick
        let start_time = Instant::now();
        self.sequence += 1;
        self.ack_map.lock().insert(self.sequence, Instant::now());
        self.outbound_tx
            .send(rpc::StreamToServer {
                sequence: self.sequence,
                client_tick: 0,
                client_message: Some(message),
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
            .and_then(|x| self.ack_map.lock().get(&x).copied())
        {
            // We haven't gotten an ack for the last pos update; withhold the current one
            let delay = Instant::now() - last_pos_send;
            if delay > Duration::from_secs_f64(0.25) {
                log::warn!("Waiting {delay:?} for a position update");
            }
            plot!("pos_update_wait", delay.as_secs_f64());
            // todo send an update but signal that we don't want chunks
            return Ok(());
        } else {
            plot!("pos_update_wait", 0.);
        }

        // If this overflows, the client is severely behind (by 4 billion chunks!) and may as well crash
        let pending_chunks = self
            .mesh_workers
            .iter()
            .map(|worker| worker.queue_len())
            .sum::<usize>()
            .max(
                self.neighbor_propagators
                    .iter()
                    .map(|x| x.queue_len())
                    .sum::<usize>(),
            )
            .try_into()
            .unwrap();
        let hotbar_slot = self.client_state.hud.lock().hotbar_slot;
        let sequence = self
            .send_sequenced_message(rpc::stream_to_server::ClientMessage::PositionUpdate(
                rpc::ClientUpdate {
                    position: Some(pos.to_proto()?),
                    pacing: Some(rpc::ClientPacing { pending_chunks }),
                    hotbar_slot,
                },
            ))
            .await?;
        self.last_pos_update_seq = Some(sequence);
        Ok(())
    }

    async fn handle_game_action(&mut self, action: GameAction) -> Result<()> {
        self.send_position_update(self.client_state.last_position())
            .await?;
        match action {
            GameAction::Dig(action) => {
                self.send_sequenced_message(rpc::stream_to_server::ClientMessage::Dig(
                    rpc::DigTapAction {
                        block_coord: Some(action.target.into()),
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
                        block_coord: Some(action.target.into()),
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
                        block_coord: Some(action.target.into()),
                        anchor: action.anchor.map(|x| x.into()),
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
                        block_coord: Some(action.target.into()),
                        position: Some(action.player_pos.to_proto()?),
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
        while !self.cancellation.is_cancelled() {
            tokio::select! {
                _ = position_tx_timer.tick() => {
                    // TODO send updates at a lower rate if substantially unchanged
                    self.send_position_update(self.client_state.last_position()).await?;
                },

                _ = self.cancellation.cancelled() => {
                    log::info!("Outbound stream context detected cancellation and shutting down")
                    // pass
                }

                action = self.action_receiver.recv() => {
                    match action {
                        Some(x) => self.handle_game_action(x).await?,
                        None => {
                            log::warn!("Action sender closed, shutting down outbound loop");
                            self.cancellation.cancel();
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
    // Some messages are sent straight from the inbound context, namely protocol bugchecks
    outbound_tx: mpsc::Sender<rpc::StreamToServer>,
    inbound_rx: tonic::Streaming<rpc::StreamToClient>,
    client_state: Arc<ClientState>,
    // Cancellation, shared with the inbound context
    pub(crate) cancellation: CancellationToken,

    ack_map: Arc<Mutex<HashMap<u64, Instant>>>,
    mesh_workers: Vec<Arc<MeshWorker>>,
    mesh_worker_handles: futures::stream::FuturesUnordered<tokio::task::JoinHandle<Result<()>>>,
    neighbor_propagators: Vec<Arc<NeighborPropagator>>,
    neighbor_propagator_handles:
        futures::stream::FuturesUnordered<tokio::task::JoinHandle<Result<()>>>,

    batcher: Arc<MeshBatcher>,
    batcher_handle: tokio::task::JoinHandle<Result<()>>,

    snappy_helper: SnappyDecodeHelper,
    protocol_version: u32,

    initial_state_notification: Arc<tokio::sync::Notify>,
}
impl InboundContext {
    pub(crate) async fn run_inbound_loop(&mut self) -> Result<()> {
        while !self.cancellation.is_cancelled() {
            tokio::select! {
                message = self.inbound_rx.message() => {
                    match message {
                        Err(e) => {
                            log::warn!("Server sent an error: {:?}", e);
                            *self.client_state.pending_error.lock() = Some(format!("{:?}", e));
                            return Err(e.into());
                        },
                        Ok(None) => {
                            log::info!("Server disconnected");
                            let mut pending_error = self.client_state.pending_error.lock();
                            if pending_error.is_none() {
                                *pending_error = Some("Server disconnected unexpectedly without sending a detailed error message".to_string());
                            }

                            self.cancellation.cancel();
                        }
                        Ok(Some(message)) => {
                            match self.handle_message(&message).await {
                                Ok(_) => {},
                                Err(e) => {
                                    log::warn!("Client failed to handle message: {:?}, error: {:?}", message, e);
                                },
                            }
                        }
                    }

                }
                _ = self.cancellation.cancelled() => {
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
                    return result.unwrap()?;
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
                    return result.unwrap()?;
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
                }
            };
        }
        log::warn!("Exiting inbound loop");
        // Notify the mesh worker so it can exit soon
        for worker in self.mesh_workers.iter() {
            worker.cancel();
        }
        for worker in self.neighbor_propagators.iter() {
            worker.cancel();
        }
        self.batcher.cancel();
        Ok(())
    }
    async fn handle_message(&mut self, message: &rpc::StreamToClient) -> Result<()> {
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
                self.client_state.egui.lock().show_popup(popup_desc);
            }
            Some(rpc::stream_to_client::ServerMessage::ChatMessage(message)) => {
                self.client_state.chat.lock().message_history.push(
                    ChatMessage::new(&message.origin, &message.message)
                        .with_color_fixed32(message.color_argb),
                )
            }
            Some(rpc::stream_to_client::ServerMessage::ShutdownMessage(msg)) => {
                *self.client_state.pending_error.lock() = Some(msg.clone());
            }
            Some(rpc::stream_to_client::ServerMessage::EntityMovement(movement)) => {
                self.handle_entity_movement(movement).await?;
            }
            Some(_) => {
                log::warn!("Unimplemented server->client message {:?}", message);
            }
        }
        Ok(())
    }

    fn enqueue_for_nprop(&self, coord: ChunkCoordinate) {
        self.neighbor_propagators[coord.hash_u64() as usize % self.neighbor_propagators.len()]
            .enqueue(coord);
    }
    fn enqueue_for_meshing(&self, coord: ChunkCoordinate) {
        self.mesh_workers[coord.hash_u64() as usize % self.mesh_workers.len()].enqueue(coord);
    }

    async fn handle_mapchunk(&mut self, chunk: &rpc::MapChunk) -> Result<()> {
        match &chunk.chunk_coord {
            Some(coord) => {
                tokio::task::block_in_place(|| {
                    let _span = span!("handle_mapchunk");
                    let coord = coord.into();
                    let extra_chunks = self.client_state.chunks.insert_or_update(
                        coord,
                        chunk.clone(),
                        &mut self.snappy_helper,
                        &self.client_state.block_types,
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
                self.send_bugcheck("Got chunk without a coordinate".to_string())
                    .await?;
            }
        };
        Ok(())
    }

    async fn handle_ack(&mut self, seq: u64) -> Result<()> {
        let send_time = self.ack_map.lock().remove(&seq);
        match send_time {
            Some(time) => {
                // todo track in a histogram
                //log::info!("Seq {} took {:?}", seq, Instant::now() - time)
                plot!("ack_rtt", (Instant::now() - time).as_secs_f64());
            }
            None => {
                let desc = format!("got ack for seq {} which we didn't send", seq);
                self.send_bugcheck(desc).await?;
            }
        }
        Ok(())
    }

    async fn send_bugcheck(&mut self, description: String) -> Result<()> {
        log::error!("Protocol bugcheck: {}", description);
        self.outbound_tx
            .send(rpc::StreamToServer {
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
            })
            .await?;
        Ok(())
    }

    async fn handle_unsubscribe(&mut self, unsub: &rpc::MapChunkUnsubscribe) -> Result<()> {
        // TODO hold more old chunks (possibly LRU) to provide a higher render distance
        let mut bad_coords = vec![];
        for coord in unsub.chunk_coord.iter() {
            match self.client_state.chunks.remove(&coord.into()) {
                Some(_x) => {}
                None => {
                    bad_coords.push(coord.clone());
                }
            }

            // TODO - do we need to do this?
            // tokio::task::block_in_place(|| {
            //     self.mesh_worker.queue.lock().remove(&coord.into());
            // });
        }
        if !bad_coords.is_empty() {
            self.send_bugcheck(format!(
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
            self.send_bugcheck("Missing block_coord in delta update".to_string())
                .await?;
        }
        for coord in unknown_coords {
            self.send_bugcheck(format!(
                "Got delta at {:?} but chunk is not in cache",
                coord
            ))
            .await?;
        }

        Ok(())
    }

    fn map_delta_update_sync(
        &mut self,
        batch: &rpc::MapDeltaUpdateBatch,
    ) -> Result<(bool, Vec<BlockCoordinate>), anyhow::Error> {
        let (needs_remesh, unknown_coords, missing_coord) = self
            .client_state
            .chunks
            .apply_delta_batch(batch, &self.client_state.block_types)?;

        {
            let _span = span!("remesh for delta");
            let mut scratchpad = Box::new([0; 48 * 48 * 48]);

            // Only do inline meshing for chunks within 3 chunks of the current player location
            // Otherwise, we tie up the network thread for too long
            let current_position = self.client_state.last_position().position;
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
                    let neighbors = self.client_state.chunks.cloned_neighbors_fast(coord);
                    propagate_neighbor_data(
                        &self.client_state.block_types,
                        &neighbors,
                        &mut scratchpad,
                    )?;
                } else {
                    self.enqueue_for_nprop(coord);
                }
            }
            for coord in needs_remesh {
                if eligible_for_inline(coord) {
                    self.client_state
                        .chunks
                        .maybe_mesh_and_maybe_promote(coord, &self.client_state.block_renderer)?;
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
                .send_bugcheck("Missing inventory in InventoryUpdate".to_string())
                .await;
        };
        let inv = ClientInventory::from_proto(inventory_update);
        let mut hud_lock = self.client_state.hud.lock();
        if Some(inventory_update.view_id) == hud_lock.hotbar_view_id {
            hud_lock.invalidate_hotbar();
            let slot = hud_lock.hotbar_slot();
            drop(hud_lock);
            self.client_state.tool_controller.lock().update_item(
                &self.client_state,
                slot,
                inv.contents()[slot as usize]
                    .clone()
                    .and_then(|x| self.client_state.items.get(&x.item_name))
                    .cloned(),
            )
        }

        self.client_state
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
        self.initial_state_notification.notify_one();
        self.client_state.handle_server_update(state_update)
    }

    async fn handle_entity_movement(&mut self, movement: &rpc::EntityMovement) -> Result<()> {
        if movement.remove {
            if self
                .client_state
                .entities
                .lock()
                .entities
                .remove(&movement.entity_id)
                .is_none()
            {
                self.send_bugcheck(format!(
                    "Got remove for non-existent entity {}",
                    movement.entity_id
                ))
                .await?;
            }
            return Ok(());
        }

        let position = match &movement.position {
            Some(position) => position.try_into()?,
            None => {
                return self
                    .send_bugcheck(format!(
                        "Got move for entity {} with no position",
                        movement.entity_id
                    ))
                    .await
            }
        };
        match self
            .client_state
            .entities
            .lock()
            .entities
            .entry(movement.entity_id)
        {
            Entry::Occupied(mut entry) => {
                entry.get_mut().position = position;
            }
            Entry::Vacant(entry) => {
                entry.insert(GameEntity { position });
            }
        }

        Ok(())
    }
}
