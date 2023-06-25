use std::{sync::Arc, collections::{HashMap, HashSet}, time::{Instant, Duration}, backtrace};

use anyhow::Result;
use parking_lot::{Mutex, Condvar};
use rustc_hash::FxHashSet;
use tokio::{sync::mpsc, task::spawn_blocking};
use tokio_util::sync::CancellationToken;
use tonic::Streaming;
use cuberef_core::{
    coordinates::{ChunkCoordinate, PlayerPositionUpdate, BlockCoordinate},
    protocol::{
        game_rpc::{
            self as rpc, cuberef_game_client::CuberefGameClient, GetBlockDefsRequest,
            GetItemDefsRequest, GetMediaRequest, StreamToClient, StreamToServer,
        }, coordinates::Angles,
    },
};
use tracy_client::{plot, span};
use crate::game_state::{ClientState, GameAction, chunk::{ClientChunk, mesh_chunk}, items::ClientInventory};

use super::mesh_worker::MeshWorker;


pub(crate) async fn make_contexts(
    client_state: Arc<ClientState>,
    tx_send: mpsc::Sender<StreamToServer>,
    stream: Streaming<StreamToClient>,
    action_receiver: mpsc::Receiver<GameAction>,
) -> Result<(InboundContext, OutboundContext)> {
    let cancellation = client_state.shutdown.clone();

    let ack_map = Arc::new(Mutex::new(HashMap::new()));
    let mesh_worker = Arc::new(MeshWorker {
        client_state: client_state.clone(),
        queue: Mutex::new(HashSet::new()),
        cond: Condvar::new(),
        shutdown: cancellation.clone(),
    });
    let inbound = InboundContext {
        outbound_tx: tx_send.clone(),
        inbound_rx: stream,
        client_state: client_state.clone(),
        cancellation: cancellation.clone(),
        ack_map: ack_map.clone(),
        mesh_worker: mesh_worker.clone(),
    };

    let outbound = OutboundContext {
        outbound_tx: tx_send,
        sequence: 1,
        client_state,
        cancellation,
        ack_map,
        action_receiver,
        last_pos_update_seq: None,
        mesh_worker,
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
    mesh_worker: Arc<MeshWorker>,
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
        let pending_chunks = self.mesh_worker.queue.lock().len().try_into().unwrap();
        let sequence = self
            .send_sequenced_message(rpc::stream_to_server::ClientMessage::PositionUpdate(
                rpc::ClientUpdate {
                    position: Some(rpc::PositionUpdate {
                        position: Some(pos.position.try_into()?),
                        velocity: Some(pos.velocity.try_into()?),
                        face_direction: Some(Angles {
                            deg_azimuth: pos.face_direction.0,
                            deg_elevation: pos.face_direction.1,
                        }),
                    }),
                    pacing: Some(rpc::ClientPacing { pending_chunks }),
                },
            ))
            .await?;
        self.last_pos_update_seq = Some(sequence);
        Ok(())
    }

    async fn handle_game_action(&mut self, action: GameAction) -> Result<()> {
        match action {
            GameAction::Dig(action) => {
                self.send_sequenced_message(rpc::stream_to_server::ClientMessage::Dig(
                    rpc::DigAction {
                        block_coord: Some(action.target.into()),
                        prev_coord: action.prev.map(|x| x.into()),
                        item_slot: action.item_slot,
                    },
                ))
                .await?;
            }
            GameAction::Tap(action) => {
                self.send_sequenced_message(rpc::stream_to_server::ClientMessage::Tap(
                    rpc::TapAction {
                        block_coord: Some(action.target.into()),
                        prev_coord: action.prev.map(|x| x.into()),
                        item_slot: action.item_slot,
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
                        destnation_slot: action.destination_slot.try_into()?,
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
        }
        Ok(())
    }

    pub(crate) async fn run_outbound_loop(&mut self) -> Result<()> {
        self.outbound_tx
            .send(StreamToServer {
                sequence: 0,
                client_tick: 0,
                client_message: Some(rpc::stream_to_server::ClientMessage::ClientInitialReady(
                    rpc::Nop {},
                )),
            })
            .await?;

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
    mesh_worker: Arc<MeshWorker>,
}
impl InboundContext {
    pub(crate) async fn run_inbound_loop(&mut self) -> Result<()> {
        let mesh_worker_clone = self.mesh_worker.clone();
        let mut mesh_worker_handle = spawn_blocking(move || mesh_worker_clone.run_mesh_worker());
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
                result = &mut mesh_worker_handle => {
                    match &result {
                        Err(e) => {
                            log::error!("Error awaiting mesh worker: {e:?}");
                        }
                        Ok(Err(e)) => {
                            log::error!("Mesh worker crashed: {e:?}");
                        }
                        Ok(_) => {
                            log::info!("Mesh worker exiting");
                        }
                    }
                    return result?;
                }
            };
        }
        log::warn!("Exiting inbound loop");
        // Notify the mesh worker so it can exit soon
        self.mesh_worker.cond.notify_all();
        Ok(())
    }
    async fn handle_message(&mut self, message: &rpc::StreamToClient) -> Result<()> {
        // todo (microoptimization) take the message by value
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
            Some(_) => {
                log::warn!("Unimplemented server->client message {:?}", message);
            }
        }
        Ok(())
    }

    fn enqueue_for_meshing(&self, coord: ChunkCoordinate) {
        let mut lock = self.mesh_worker.queue.lock();
        lock.insert(coord);
        self.mesh_worker.cond.notify_one();
    }

    async fn handle_mapchunk(&mut self, chunk: &rpc::MapChunk) -> Result<()> {
        // TODO offload meshing to another thread
        match &chunk.chunk_coord {
            Some(coord) => {
                tokio::task::block_in_place(|| {
                    let _span = span!("handle_mapchunk");
                    let coord = coord.into();
                    self.client_state
                        .chunks
                        .insert(coord, ClientChunk::from_proto(chunk.clone())?);

                    self.enqueue_for_meshing(coord);

                    if let Some(neighbor) = coord.try_delta(-1, 0, 0) {
                        self.enqueue_for_meshing(neighbor);
                    }
                    if let Some(neighbor) = coord.try_delta(1, 0, 0) {
                        self.enqueue_for_meshing(neighbor);
                    }
                    if let Some(neighbor) = coord.try_delta(0, -1, 0) {
                        self.enqueue_for_meshing(neighbor);
                    }
                    if let Some(neighbor) = coord.try_delta(0, 1, 0) {
                        self.enqueue_for_meshing(neighbor);
                    }
                    if let Some(neighbor) = coord.try_delta(0, 0, -1) {
                        self.enqueue_for_meshing(neighbor);
                    }
                    if let Some(neighbor) = coord.try_delta(0, 0, 1) {
                        self.enqueue_for_meshing(neighbor);
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
            self.mesh_worker.queue.lock().remove(&coord.into());
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
        let chunk_manager_read_lock = self.client_state.chunks.read_lock();

        let mut missing_coord = false;
        let mut unknown_coords = Vec::new();
        let mut needs_remesh = FxHashSet::default();
        for update in batch.updates.iter() {
            let block_coord: BlockCoordinate = match &update.block_coord {
                Some(x) => x.into(),
                None => {
                    missing_coord = true;
                    continue;
                }
            };
            let chunk = chunk_manager_read_lock.get_mut(&block_coord.chunk());
            let mut chunk = match chunk {
                Some(x) => x,
                None => {
                    unknown_coords.push(block_coord);
                    continue;
                }
            };
            // unwrap because we expect all errors to be internal.
            if chunk.apply_delta(update).unwrap() {
                needs_remesh.insert(block_coord.chunk());
                if let Some(neighbor) = block_coord.try_delta(-1, 0, 0) {
                    if neighbor.chunk() != block_coord.chunk() {
                        needs_remesh.insert(neighbor.chunk());
                    }
                }
                if let Some(neighbor) = block_coord.try_delta(1, 0, 0) {
                    if neighbor.chunk() != block_coord.chunk() {
                        needs_remesh.insert(neighbor.chunk());
                    }
                }
                if let Some(neighbor) = block_coord.try_delta(0, -1, 0) {
                    if neighbor.chunk() != block_coord.chunk() {
                        needs_remesh.insert(neighbor.chunk());
                    }
                }
                if let Some(neighbor) = block_coord.try_delta(0, 1, 0) {
                    if neighbor.chunk() != block_coord.chunk() {
                        needs_remesh.insert(neighbor.chunk());
                    }
                }
                if let Some(neighbor) = block_coord.try_delta(0, 0, -1) {
                    if neighbor.chunk() != block_coord.chunk() {
                        needs_remesh.insert(neighbor.chunk());
                    }
                }
                if let Some(neighbor) = block_coord.try_delta(0, 0, 1) {
                    if neighbor.chunk() != block_coord.chunk() {
                        needs_remesh.insert(neighbor.chunk());
                    }
                }
            }
        }
        drop(chunk_manager_read_lock);
        {
            let _span = span!("remesh for delta");
            for chunk in needs_remesh {
                mesh_chunk(
                    chunk,
                    &self.client_state.chunks.read_lock(),
                    &self.client_state.cube_renderer,
                )?;
            }
        }
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
        let x = state_update
            .position
            .as_ref()
            .and_then(|x| x.position.clone());
        let Some(pos_vector) = x else {
            return self.send_bugcheck("Missing position in ClientState update".to_string()).await;
        };
        self.client_state
            .physics_state
            .lock()
            .set_position(pos_vector.try_into()?);

        let mut egui_lock = self.client_state.egui.lock();
        egui_lock.inventory_view = state_update.inventory_popup.clone();
        egui_lock.inventory_manipulation_view_id = Some(state_update.inventory_manipulation_view);

        let mut hud_lock = self.client_state.hud.lock();
        hud_lock.hotbar_view_id = Some(state_update.hotbar_inventory_view);
        hud_lock.invalidate_hotbar();
        Ok(())
    }
}
