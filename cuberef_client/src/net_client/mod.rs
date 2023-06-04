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

use std::{
    backtrace,
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use cuberef_core::{
    coordinates::{BlockCoordinate, PlayerPositionUpdate},
    protocol::{
        coordinates::Angles,
        game_rpc::{
            self as rpc, cuberef_game_client::CuberefGameClient, GetBlockDefsRequest,
            GetItemDefsRequest, GetMediaRequest,
        },
    },
};
use image::DynamicImage;
use parking_lot::Mutex;
use rustc_hash::{FxHashMap, FxHashSet};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tonic::{async_trait, codegen::CompressionEncoding, transport::Channel, Request};

use crate::{
    cube_renderer::{AsyncTextureLoader, BlockRenderer, ClientBlockTypeManager},
    game_state::{
        chunk::{maybe_mesh_chunk, mesh_chunk, ClientChunk},
        items::{ClientInventory, ClientItemManager, InventoryManager},
        physics::PhysicsState,
        tool_controller::ToolController,
        ClientState, GameAction,
    },
    game_ui::GameUi,
    vulkan::VulkanContext,
};

async fn connect_grpc(
    server_addr: String,
) -> Result<rpc::cuberef_game_client::CuberefGameClient<Channel>> {
    CuberefGameClient::connect(server_addr)
        .await
        .with_context(|| "Failed to connect")
}

pub(crate) async fn connect_game(
    server_addr: String,
    cloned_context: VulkanContext,
) -> Result<Arc<ClientState>> {
    log::info!("Connecting to {}...", &server_addr);
    let mut connection = connect_grpc(server_addr.clone())
        .await?
        .max_decoding_message_size(1024 * 1024 * 256)
        .accept_compressed(CompressionEncoding::Gzip)
        .send_compressed(CompressionEncoding::Gzip);
    log::info!("Connection to {} established.", &server_addr);
    let block_defs_proto = connection.get_block_defs(GetBlockDefsRequest {}).await?;
    log::info!(
        "{} block defs loaded from server",
        block_defs_proto.get_ref().block_types.len()
    );

    let texture_loader = GrpcTextureLoader {
        connection: connection.clone(),
    };

    let block_types = Arc::new(ClientBlockTypeManager::new(
        block_defs_proto.into_inner().block_types,
    )?);

    let cube_renderer = Arc::new(
        BlockRenderer::new(block_types.clone(), texture_loader.clone(), &cloned_context).await?,
    );

    let item_defs_proto = connection.get_item_defs(GetItemDefsRequest {}).await?;
    log::info!(
        "{} item defs loaded from server",
        item_defs_proto.get_ref().item_defs.len()
    );
    let items = Arc::new(ClientItemManager::new(
        item_defs_proto.into_inner().item_defs,
    )?);

    let game_ui = Arc::new(Mutex::new(
        GameUi::new(items.clone(), texture_loader, &cloned_context).await?,
    ));
    // TODO clean up this hacky cloning of the context.
    // We need to clone it to start up the game ui without running into borrow checker issues,
    // since it provides access to the allocators. We then drop it early to ensure that it's not
    // used from these coroutines
    drop(cloned_context);

    let (action_sender, action_receiver) = mpsc::channel(4);
    let client_state = Arc::new(ClientState {
        block_types,
        items,
        last_update: Mutex::new(Instant::now()),
        physics_state: Mutex::new(PhysicsState::new()),
        tool_controller: Mutex::new(ToolController::new()),
        chunks: Mutex::new(FxHashMap::default()),
        inventories: Mutex::new(InventoryManager {
            inventories: FxHashMap::default(),
            main_inv_key: vec![],
        }),
        shutdown: CancellationToken::new(),
        actions: action_sender,
        cube_renderer,
        game_ui,
    });

    let (mut inbound, mut outbound) =
        make_contexts(client_state.clone(), connection, action_receiver).await?;
    // todo track exit status and catch errors?
    // Right now we just print a panic message, but don't actually exit because of it
    tokio::spawn(async move { inbound.run_inbound_loop().await.unwrap() });
    tokio::spawn(async move { outbound.run_outbound_loop().await.unwrap() });

    Ok(client_state)
}

async fn make_contexts(
    client_state: Arc<ClientState>,
    mut connection: CuberefGameClient<Channel>,
    action_receiver: mpsc::Receiver<GameAction>,
) -> Result<(InboundContext, OutboundContext)> {
    // todo tune depth
    let (tx_send, tx_recv) = mpsc::channel(4);

    let mut request = Request::new(ReceiverStream::new(tx_recv));
    let metadata = request.metadata_mut();
    metadata.append("x-cuberef-username", "fake".parse()?);
    metadata.append("x-cuberef-token", "fake".parse()?);

    let stream = connection.game_stream(request).await?.into_inner();
    let cancellation = client_state.shutdown.clone();

    let ack_map = Arc::new(Mutex::new(HashMap::new()));

    let inbound = InboundContext {
        tx_send: tx_send.clone(),
        inbound_rx: stream,
        client_state: client_state.clone(),
        cancellation: cancellation.clone(),
        ack_map: ack_map.clone(),
    };

    let outbound = OutboundContext {
        tx_send,
        sequence: 1,
        client_state,
        cancellation,
        ack_map,
        action_receiver,
    };

    Ok((inbound, outbound))
}

struct OutboundContext {
    tx_send: mpsc::Sender<rpc::StreamToServer>,
    sequence: u64,
    client_state: Arc<ClientState>,
    // Cancellation, shared with the inbound context
    cancellation: CancellationToken,
    ack_map: Arc<Mutex<HashMap<u64, Instant>>>,

    action_receiver: mpsc::Receiver<GameAction>,
}
impl OutboundContext {
    async fn send_sequenced_message(
        &mut self,
        message: rpc::stream_to_server::ClientMessage,
    ) -> Result<()> {
        // todo tick
        self.sequence += 1;
        self.ack_map.lock().insert(self.sequence, Instant::now());
        self.tx_send
            .send(rpc::StreamToServer {
                sequence: self.sequence,
                client_tick: 0,
                client_message: Some(message),
            })
            .await?;
        Ok(())
    }

    /// Send a position update to the server
    async fn send_position_update(&mut self, pos: PlayerPositionUpdate) -> Result<()> {
        self.send_sequenced_message(rpc::stream_to_server::ClientMessage::PositionUpdate(
            rpc::PositionUpdate {
                position: Some(pos.position.try_into()?),
                velocity: Some(pos.velocity.try_into()?),
                face_direction: Some(Angles {
                    deg_azimuth: pos.face_direction.0,
                    deg_elevation: pos.face_direction.1,
                }),
            },
        ))
        .await
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
                .await
            }
            GameAction::Tap(action) => {
                self.send_sequenced_message(rpc::stream_to_server::ClientMessage::Tap(
                    rpc::TapAction {
                        block_coord: Some(action.target.into()),
                        prev_coord: action.prev.map(|x| x.into()),
                        item_slot: action.item_slot,
                    },
                ))
                .await
            }
            GameAction::Place(action) => {
                self.send_sequenced_message(rpc::stream_to_server::ClientMessage::Place(
                    rpc::PlaceAction {
                        block_coord: Some(action.target.into()),
                        anchor: action.anchor.map(|x| x.into()),
                        item_slot: action.item_slot,
                    },
                ))
                .await
            }
        }
    }

    async fn run_outbound_loop(&mut self) -> Result<()> {
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

struct InboundContext {
    // Some messages are sent straight from the inbound context, namely protocol bugchecks
    tx_send: mpsc::Sender<rpc::StreamToServer>,
    inbound_rx: tonic::Streaming<rpc::StreamToClient>,
    client_state: Arc<ClientState>,
    // Cancellation, shared with the inbound context
    cancellation: CancellationToken,

    ack_map: Arc<Mutex<HashMap<u64, Instant>>>,
}
impl InboundContext {
    pub(crate) async fn run_inbound_loop(&mut self) -> Result<()> {
        while !self.cancellation.is_cancelled() {
            tokio::select! {
                message = self.inbound_rx.message() => {
                    match message {
                        Err(e) => {
                            log::warn!("Failure reading inbound message: {:?}", e)
                        },
                        Ok(None) => {
                            log::info!("Server disconnected cleanly");
                            self.cancellation.cancel();
                            return Ok(());
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
                }
            };
        }
        log::warn!("Exiting inbound loop");
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
                self.handle_delta_update(delta_update).await?;
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

    async fn handle_mapchunk(&mut self, chunk: &rpc::MapChunk) -> Result<()> {
        // TODO offload meshing to another thread
        match &chunk.chunk_coord {
            Some(coord) => {
                tokio::task::block_in_place(|| {
                    let coord = coord.into();
                    let mut lock = self.client_state.chunks.lock();
                    lock.insert(coord, ClientChunk::from_proto(chunk.clone())?);
                    mesh_chunk(coord, &mut lock, &self.client_state.cube_renderer)?;
                    drop(lock);

                    if let Some(neighbor) = coord.try_delta(-1, 0, 0) {
                        maybe_mesh_chunk(
                            neighbor,
                            &mut self.client_state.chunks.lock(),
                            &self.client_state.cube_renderer,
                        )?;
                    }
                    if let Some(neighbor) = coord.try_delta(1, 0, 0) {
                        maybe_mesh_chunk(
                            neighbor,
                            &mut self.client_state.chunks.lock(),
                            &self.client_state.cube_renderer,
                        )?;
                    }
                    if let Some(neighbor) = coord.try_delta(0, -1, 0) {
                        maybe_mesh_chunk(
                            neighbor,
                            &mut self.client_state.chunks.lock(),
                            &self.client_state.cube_renderer,
                        )?;
                    }
                    if let Some(neighbor) = coord.try_delta(0, 1, 0) {
                        maybe_mesh_chunk(
                            neighbor,
                            &mut self.client_state.chunks.lock(),
                            &self.client_state.cube_renderer,
                        )?;
                    }
                    if let Some(neighbor) = coord.try_delta(0, 0, -1) {
                        maybe_mesh_chunk(
                            neighbor,
                            &mut self.client_state.chunks.lock(),
                            &self.client_state.cube_renderer,
                        )?;
                    }
                    if let Some(neighbor) = coord.try_delta(0, 0, 1) {
                        maybe_mesh_chunk(
                            neighbor,
                            &mut self.client_state.chunks.lock(),
                            &self.client_state.cube_renderer,
                        )?;
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
            Some(_time) => {
                // todo track in a histogram
                // log::info!("Seq {} took {:?}", seq, Instant::now() - time)
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
        self.tx_send
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
            match self.client_state.chunks.lock().remove(&coord.into()) {
                Some(_x) => {}
                None => {
                    bad_coords.push(coord.clone());
                }
            }
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

    async fn handle_delta_update(&mut self, batch: &rpc::MapDeltaUpdateBatch) -> Result<()> {
        let mut lock = self.client_state.chunks.lock();

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
            let chunk = lock.get_mut(&block_coord.chunk());
            let chunk = match chunk {
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
        for chunk in needs_remesh {
            mesh_chunk(chunk, &mut lock, &self.client_state.cube_renderer)?;
        }

        drop(lock);
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
        let Some(inv_proto) = &inventory_update.inventory else {
            return self.send_bugcheck("Missing inventory in InventoryUpdate".to_string()).await;
        };
        let mut lock = self.client_state.inventories.lock();
        let inv = ClientInventory::from_proto(inv_proto.clone());

        // TODO remove inventories from the tracked set somehow
        // needs changes on the server as well
        if inv_proto.inventory_key == lock.main_inv_key {
            let mut ui_lock = self.client_state.game_ui.lock();
            ui_lock.invalidate_hotbar();
            let slot = ui_lock.hotbar_slot();
            drop(ui_lock);
            self.client_state.tool_controller.lock().update_item(
                &self.client_state,
                slot,
                inv.contents()[slot as usize]
                    .clone()
                    .and_then(|x| self.client_state.items.get(&x.item_name))
                    .cloned(),
            )
        }

        lock.inventories
            .insert(inv_proto.inventory_key.clone(), inv);
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
        let mut lock = self.client_state.inventories.lock();
        lock.main_inv_key = state_update.main_inventory_id.clone();
        self.client_state.game_ui.lock().invalidate_hotbar();
        Ok(())
    }
}

#[derive(Clone)]
struct GrpcTextureLoader {
    connection: CuberefGameClient<Channel>,
}
#[async_trait]
impl AsyncTextureLoader for GrpcTextureLoader {
    async fn load_texture(&mut self, tex_name: &str) -> Result<DynamicImage> {
        // TODO caching - right now we fetch all resources every time we start the game
        // Some resources are even fetched twice (once for blocks, once for items)
        log::info!("Loading resource {}", tex_name);
        let resp = self
            .connection
            .get_media(GetMediaRequest {
                media_name: tex_name.to_string(),
            })
            .await?;
        image::load_from_memory(&resp.into_inner().media)
            .with_context(|| "Image was fetched, but parsing failed")
    }
}
