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
use std::time::Instant;

use crate::game_state::blocks;
use crate::game_state::blocks::BlockType;
use crate::game_state::client_ui::PopupAction;
use crate::game_state::client_ui::PopupResponse;
use crate::game_state::event::EventInitiator;
use crate::game_state::event::HandlerContext;

use crate::game_state::game_map::BlockUpdate;
use crate::game_state::handlers;
use crate::game_state::inventory::InventoryKey;
use crate::game_state::inventory::InventoryViewWithContext;
use crate::game_state::inventory::TypeErasedInventoryView;
use crate::game_state::items;
use crate::game_state::items::DigResult;
use crate::game_state::items::Item;
use crate::game_state::player::PlayerContext;
use crate::game_state::GameState;
use crate::run_handler;

use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use cgmath::Vector3;
use cgmath::Zero;
use cuberef_core::coordinates::{BlockCoordinate, ChunkCoordinate, PlayerPositionUpdate};

use cuberef_core::protocol::coordinates::Angles;
use cuberef_core::protocol::game_rpc as proto;
use cuberef_core::protocol::game_rpc::stream_to_client::ServerMessage;
use cuberef_core::protocol::game_rpc::MapDeltaUpdateBatch;
use cuberef_core::protocol::game_rpc::PositionUpdate;
use cuberef_core::protocol::game_rpc::StreamToClient;
use itertools::iproduct;
use log::error;
use log::info;
use log::warn;
use tokio::sync::{broadcast, mpsc, watch};
use tokio::task::block_in_place;
use tokio_util::sync::CancellationToken;
use tracy_client::plot;
use tracy_client::span;

static CLIENT_CONTEXT_ID_COUNTER: AtomicUsize = AtomicUsize::new(1);

pub(crate) async fn make_client_contexts(
    game_state: Arc<GameState>,
    player_context: PlayerContext,
    inbound_rx: tonic::Streaming<proto::StreamToServer>,
    outbound_tx: mpsc::Sender<tonic::Result<StreamToClient>>,
) -> Result<(ClientInboundContext, ClientOutboundContext)> {
    let id = CLIENT_CONTEXT_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

    let initial_position = PlayerPositionUpdate {
        tick: game_state.tick(),
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

    let inbound = ClientInboundContext {
        context_id: id,
        game_state: game_state.clone(),
        player_context: player_context.clone(),
        cancellation: cancellation.clone(),
        inbound_rx,
        own_positions: pos_send,
        outbound_tx: outbound_tx.clone(),
        next_pos_writeback: Instant::now(),
        chunk_pacing: Aimd {
            val: INITIAL_CHUNKS_PER_UPDATE as f64,
            floor: 0.,
            ceiling: MAX_CHUNKS_PER_UPDATE as f64,
            additive_increase: 16.,
            multiplicative_decrease: 0.5,
        },
    };
    let block_events = game_state.map().subscribe();
    let inventory_events = game_state.inventory_manager().subscribe();

    let outbound = ClientOutboundContext {
        context_id: id,
        game_state,
        player_context,
        outbound_tx,
        cancellation,
        block_events,
        inventory_events,
        own_positions: pos_recv,
        interested_chunks: HashSet::new(),
        interested_inventories,
        chunks_known_to_client: HashSet::new(),
    };
    Ok((inbound, outbound))
}

// State/structure backing a gRPC GameStream on the outbound side
pub(crate) struct ClientOutboundContext {
    // Core fields
    context_id: usize,
    game_state: Arc<GameState>,
    player_context: Arc<PlayerContext>,

    // RPC stream is sent via this channel
    outbound_tx: mpsc::Sender<tonic::Result<proto::StreamToClient>>,

    // Inbound channels/flags
    // Cancellation, shared with the inbound context
    cancellation: CancellationToken,
    // All updates to the map from all sources, not yet filtered by location (ClientOutboundContext is
    // responsible for filtering)
    block_events: broadcast::Receiver<BlockUpdate>,
    // TODO consider delta updates for this
    inventory_events: broadcast::Receiver<InventoryKey>,
    // This character's own movement, coming from their client (forwarded by the ClientInboundContext to here
    // and to elsewhere)
    // In the future, anticheat might check for shenanigans involving these, probably not as part of ClientOutboundContext
    // coroutines
    own_positions: watch::Receiver<PositionAndPacing>,

    // Server-side state per-client state
    // Chunks that are close enough to the player to be of interest.
    // This will have hysteresis to avoid flapping unsubscribes/resubscribes;
    // the distance at which a client subscribes to a chunk is smaller than the distance
    // at which they will unsubscribe from the chunk
    interested_chunks: HashSet<ChunkCoordinate>,
    // The client should have these chunks cached already. If a chunk is missing from this set
    // we'll need to send the full chunk to the client first.
    chunks_known_to_client: HashSet<ChunkCoordinate>,

    interested_inventories: HashSet<InventoryKey>,
}
impl ClientOutboundContext {
    // Poll for world events and send relevant messages to the client through outbound_tx
    pub(crate) async fn run_outbound_loop(&mut self) -> Result<()> {
        self.initialize_outbound_loop().await?;

        while !self.cancellation.is_cancelled() {
            tokio::select! {
                block_event = self.block_events.recv() => {
                    self.handle_block_update(block_event).await?;
                }
                inv_key = self.inventory_events.recv() => {
                    self.handle_inventory_update(inv_key).await?;
                }
                _ = self.own_positions.changed() => {
                    let update = *self.own_positions.borrow_and_update();
                    self.handle_position_update(update).await?;
                }
                _ = self.cancellation.cancelled() => {
                    info!("Client outbound loop {} detected cancellation and shutting down", self.context_id)
                    // pass
                }
                _ = self.game_state.await_start_shutdown() => {
                    info!("Game shutting down, disconnecting {}", self.context_id);
                    // cancel the inbound context as well
                    self.cancellation.cancel();
                }
            };
        }
        Ok(())
    }

    async fn send_popup_updates(&mut self) -> Result<()> {
        let updates = {
            let player_state = self.player_context.state.lock();
            let mut updates = vec![];

            for popup in player_state
                .active_popups
                .iter()
                .chain(once(&player_state.inventory_popup))
            {
                for view in popup.inventory_views().values() {
                    updates.push(make_inventory_update(
                        &self.game_state,
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
            self.outbound_tx
                .send(Ok(update))
                .await
                .with_context(|| "Could not send outbound message (inventory update)")?;
        }

        Ok(())
    }

    async fn handle_inventory_update(
        &mut self,
        update: Result<InventoryKey, broadcast::error::RecvError>,
    ) -> Result<()> {
        let key = match update {
            Err(broadcast::error::RecvError::Lagged(x)) => {
                log::error!("Client {} is lagged, {} pending", self.context_id, x);
                // TODO resync in the future? Right now we just kick the client off
                // A client that's desynced on inventory updates is struggling, so not sure
                // what we can do
                bail!("Client {} is lagged, {} pending", self.context_id, x);
            }
            Err(broadcast::error::RecvError::Closed) => return self.shut_down_connected_client(),
            Ok(x) => x,
        };

        let updates = {
            let player_state = self.player_context.state.lock();
            let mut updates = vec![];
            if player_state.hotbar_inventory_view.wants_update_for(&key) {
                updates.push(make_inventory_update(
                    &self.game_state,
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
                        updates.push(make_inventory_update(
                            &self.game_state,
                            &InventoryViewWithContext {
                                view,
                                context: popup,
                            },
                        )?);
                    }
                }
            }
            updates
        };

        for update in updates {
            self.outbound_tx
                .send(Ok(update))
                .await
                .with_context(|| "Could not send outbound message (inventory update)")?;
        }

        Ok(())
    }

    async fn handle_block_update(
        &mut self,
        update: Result<BlockUpdate, broadcast::error::RecvError>,
    ) -> Result<()> {
        let update = match update {
            Err(broadcast::error::RecvError::Lagged(x)) => {
                log::warn!("Client {} is lagged, {} pending", self.context_id, x);
                // This client context is lagging behind and lost block updates.
                // Fall back and get it resynced
                return self.handle_block_update_lagged().await;
            }
            Err(broadcast::error::RecvError::Closed) => return self.shut_down_connected_client(),
            Ok(x) => x,
        };
        let mut updates = vec![update];
        // Drain and batch as many updates as possible
        while updates.len() < MAX_UPDATE_BATCH_SIZE {
            match self.block_events.try_recv() {
                Ok(update) => {
                    if self.wants_block_update(update.location) {
                        updates.push(update)
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

        let mut update_protos = Vec::new();
        for update in updates {
            if !self
                .chunks_known_to_client
                .contains(&update.location.chunk())
            {
                // The client doesn't know this chunk, but we think it should be interested in it
                // It was probably not in server cache when it became interesting, and wasn't
                // interesting enough to load into memory then. However, it's being updated, so we
                // may as well get it to the client
                self.maybe_send_full_chunk(update.location.chunk(), true)
                    .await?;
            }
            update_protos.push(proto::MapDeltaUpdate {
                block_coord: Some(update.location.into()),
                new_id: update.new_value.id().into(),
            })
        }

        if !update_protos.is_empty() {
            let message = proto::StreamToClient {
                tick: self.game_state.tick(),
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

    fn wants_block_update(&self, coord: BlockCoordinate) -> bool {
        self.interested_chunks.contains(&coord.chunk())
    }

    async fn handle_block_update_lagged(&mut self) -> Result<()> {
        // this ends up racy. resubscribe first, so we get duplicate/pointless events after the
        // resubscribe, rather than missing events if we resubscribe to the broadcast after sending current
        // chunk states
        self.block_events.resubscribe();
        self.chunks_known_to_client.clear();
        self.interested_chunks.clear();
        // TODO back off with the number of chunks to subscribe to
        Ok(())
    }

    fn shut_down_connected_client(&mut self) -> Result<()> {
        log::error!("Unimplemented: clean shutdown for connected client");
        self.cancellation.cancel();
        Ok(())
    }

    // Add a chunk to the list of interesting chunks - the client will receive notifications for these
    // chunks
    async fn subscribe_to_chunk(
        &mut self,
        coord: ChunkCoordinate,
        load_if_missing: bool,
    ) -> Result<bool> {
        self.interested_chunks.insert(coord);
        // Add the chunk to the interesting set first, then send its state
        // This ensures that we err on the side of extraneous block updates, rather than missing ones
        // TODO future optimization/cleanup add a block update seqnum of some kind???
        // Otherwise, inline maybe_send_full_chunk
        self.maybe_send_full_chunk(coord, load_if_missing).await
    }

    // Sends a full chunk
    async fn maybe_send_full_chunk(
        &mut self,
        coord: ChunkCoordinate,
        load_if_missing: bool,
    ) -> std::result::Result<bool, anyhow::Error> {
        if self.chunks_known_to_client.contains(&coord) {
            // The client already has this chunk and we think they have an up to date copy.
            return Ok(true);
        }
        let chunk_proto = tokio::task::block_in_place(|| {
            self.game_state
                .map()
                .get_chunk_client_proto(coord, load_if_missing)
        })?;
        match chunk_proto {
            None => Ok(false),
            Some(chunk_data) => {
                self.chunks_known_to_client.insert(coord);
                let message = proto::StreamToClient {
                    tick: self.game_state.tick(),
                    server_message: Some(proto::stream_to_client::ServerMessage::MapChunk(
                        proto::MapChunk {
                            chunk_coord: Some(coord.into()),
                            chunk_data: Some(chunk_data),
                        },
                    )),
                };
                self.outbound_tx
                    .send(Ok(message))
                    .await
                    .with_context(|| "Could not send outbound message (full mapchunk)")?;
                Ok(true)
            }
        }
    }

    async fn teleport_player(&mut self, location: Vector3<f64>) -> Result<()> {
        let message = {
            let player_state = self.player_context.state.lock();
            StreamToClient {
                tick: self.game_state.tick(),
                server_message: Some(proto::stream_to_client::ServerMessage::ClientState(
                    proto::SetClientState {
                        position: Some(PositionUpdate {
                            position: Some(location.try_into()?),
                            velocity: Some(Vector3::zero().try_into()?),
                            face_direction: Some(Angles {
                                deg_azimuth: 0.,
                                deg_elevation: 0.,
                            }),
                        }),
                        hotbar_inventory_view: player_state.hotbar_inventory_view.id.0,
                        inventory_popup: Some(player_state.inventory_popup.to_proto()),
                        inventory_manipulation_view: player_state.inventory_manipulation_view.id.0,
                    },
                )),
            }
        };
        self.outbound_tx
            .send(Ok(message))
            .await
            .with_context(|| "Could not send outbound message (initial state)")
    }

    async fn handle_position_update(&mut self, update: PositionAndPacing) -> Result<()> {
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
                self.teleport_player(Vector3::zero()).await?;
                BlockCoordinate::new(0, 0, 0)
            }
        };
        let player_chunk = player_block_coord.chunk();

        // These chunks are far enough away to unsubscribe, but the client doesn't even know about them yet
        let chunks_to_silently_unsubscribe = self
            .interested_chunks
            .iter()
            .copied()
            .filter(|&chunk| player_chunk.manhattan_distance(chunk) > UNLOAD_DISTANCE as u32)
            .collect::<Vec<_>>();

        for chunk in chunks_to_silently_unsubscribe.iter() {
            self.interested_chunks.remove(chunk);
        }

        // These chunks are far enough away to unsubscribe, and the client knows about them
        let chunks_to_unsubscribe = self
            .chunks_known_to_client
            .iter()
            .copied()
            .filter(|&chunk| player_chunk.manhattan_distance(chunk) > UNLOAD_DISTANCE as u32)
            .collect::<Vec<_>>();

        for chunk in chunks_to_unsubscribe.iter() {
            self.chunks_known_to_client.remove(chunk);
        }

        let message = proto::StreamToClient {
            tick: self.game_state.tick(),
            server_message: Some(proto::stream_to_client::ServerMessage::MapChunkUnsubscribe(
                proto::MapChunkUnsubscribe {
                    chunk_coord: chunks_to_unsubscribe.iter().map(|&x| x.into()).collect(),
                },
            )),
        };

        let mut candidate_chunks = vec![];
        for &(dx, dy, dz) in LOAD_LAZY_SORTED_COORDS.iter() {
            let chunk = ChunkCoordinate {
                x: player_chunk.x.saturating_add(dx),
                y: player_chunk.y.saturating_add(dy),
                z: player_chunk.z.saturating_add(dz),
            };
            if !chunk.is_in_bounds() {
                continue;
            }
            candidate_chunks.push(chunk);
        }
        let mut num_added = 0;
        for chunk in candidate_chunks {
            // TODO: consider whether we want an L(infinity) metric (i.e. max) rather than L1 (manhattan distance)
            let load_if_missing =
                chunk.manhattan_distance(player_chunk) <= LOAD_EAGER_DISTANCE as u32;

            if chunk.manhattan_distance(player_chunk) <= LOAD_LAZY_DISTANCE as u32 {
                self.game_state.map().bump_access_time(chunk);
                if !self.chunks_known_to_client.contains(&chunk) {
                    self.subscribe_to_chunk(chunk, load_if_missing).await?;
                    num_added += 1;
                    if num_added >= update.chunks_to_send {
                        break;
                    }
                }
            }
        }
        if !chunks_to_unsubscribe.is_empty() {
            self.outbound_tx
                .send(Ok(message))
                .await
                .with_context(|| "Could not send outbound message (chunk unsubscribe)")?;
        }
        Ok(())
    }

    async fn initialize_outbound_loop(&mut self) -> Result<()> {
        let initial_position = PlayerPositionUpdate {
            tick: self.game_state.tick(),
            position: self.player_context.last_position().position,
            velocity: cgmath::Vector3::zero(),
            face_direction: (0., 0.),
        };

        let (hotbar_update, inv_manipulation_update) = {
            let player_state = self.player_context.state.lock();
            (
                make_inventory_update(&self.game_state, &&player_state.hotbar_inventory_view)?,
                make_inventory_update(
                    &self.game_state,
                    &&player_state.inventory_manipulation_view,
                )?,
            )
        };

        self.outbound_tx.send(Ok(hotbar_update)).await?;
        self.outbound_tx.send(Ok(inv_manipulation_update)).await?;
        self.send_popup_updates().await?;

        let message = {
            let player_state = self.player_context.state.lock();
            StreamToClient {
                tick: self.game_state.tick(),
                server_message: Some(proto::stream_to_client::ServerMessage::ClientState(
                    proto::SetClientState {
                        position: Some(PositionUpdate {
                            position: Some(initial_position.position.try_into()?),
                            velocity: Some(Vector3::zero().try_into()?),
                            face_direction: Some(Angles {
                                deg_azimuth: 0.,
                                deg_elevation: 0.,
                            }),
                        }),
                        // TODO pull these out (or better yet, the whole proto) into a player_state method
                        hotbar_inventory_view: player_state.hotbar_inventory_view.id.0,
                        inventory_popup: Some(player_state.inventory_popup.to_proto()),
                        inventory_manipulation_view: player_state.inventory_manipulation_view.id.0,
                    },
                )),
            }
        };
        self.outbound_tx
            .send(Ok(message))
            .await
            .with_context(|| "Could not send outbound message (initial state)")?;
        Ok(())
    }
}
impl Drop for ClientOutboundContext {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}
// State/structure backing a gRPC GameStream on the inbound side
pub(crate) struct ClientInboundContext {
    // Core fields
    // Matches the ID of the outbound context
    context_id: usize,
    game_state: Arc<GameState>,
    player_context: Arc<PlayerContext>,

    // Cancellation, shared with the outbound context
    cancellation: CancellationToken,
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
impl ClientInboundContext {
    // Poll for world events and send them through outbound_tx
    pub(crate) async fn run_inbound_loop(&mut self) -> Result<()> {
        while !self.cancellation.is_cancelled() {
            tokio::select! {
                message = self.inbound_rx.message() => {
                    match message {
                        Err(e) => {
                            warn!("Client {}, Failure reading inbound message: {:?}", self.context_id, e)
                        },
                        Ok(None) => {
                            info!("Client {} disconnected cleanly", self.context_id);
                            self.cancellation.cancel();
                            return Ok(());
                        }
                        Ok(Some(message)) => {
                            match self.handle_message(&message).await {
                                Ok(_) => {},
                                Err(e) => {
                                    warn!("Client {} failed to handle message: {:?}, error: {:?}", self.context_id, message, e);
                                    // TODO notify the client once there's a chat or server->client error handling message
                                },
                            }
                        }
                    }

                }
                _ = self.cancellation.cancelled() => {
                    info!("Client inbound context {} detected cancellation and shutting down", self.context_id)
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
                    self.context_id
                );
            }
            Some(proto::stream_to_server::ClientMessage::Dig(dig_message)) => {
                // TODO check whether the current item can dig this block, and whether
                // it's been long enough since the last dig
                let coord: BlockCoordinate = dig_message
                    .block_coord
                    .as_ref()
                    .map(|x| x.into())
                    .with_context(|| "Missing block_coord")?;
                self.run_map_handlers(
                    coord,
                    dig_message.item_slot,
                    |item| item.dig_handler.as_deref(),
                    |block| block.dig_handler_inline.as_deref(),
                    |block| block.dig_handler_full.as_deref(),
                )
                .await?;
            }
            Some(proto::stream_to_server::ClientMessage::Tap(tap_message)) => {
                let coord: BlockCoordinate = tap_message
                    .block_coord
                    .as_ref()
                    .map(|x| x.into())
                    .with_context(|| "Missing block_coord")?;
                self.run_map_handlers(
                    coord,
                    tap_message.item_slot,
                    |item| item.tap_handler.as_deref(),
                    |block| block.tap_handler_inline.as_deref(),
                    |block| block.tap_handler_full.as_deref(),
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
            Some(_) => {
                warn!(
                    "Unimplemented client->server message {:?} on context {}",
                    message, self.context_id
                );
            }
        }
        // todo decide whether we should send nacks on error, or just complain about them
        // to the local log
        // also decide whether position updates merit an ack
        self.send_ack(message.sequence).await?;
        Ok(())
    }

    async fn run_map_handlers<F, G, H>(
        &mut self,
        coord: BlockCoordinate,
        selected_inv_slot: u32,
        get_item_handler: F,
        get_block_inline_handler: G,
        get_block_full_handler: H,
    ) -> Result<()>
    where
        F: FnOnce(&Item) -> Option<&items::BlockInteractionHandler>,
        G: FnOnce(&BlockType) -> Option<&blocks::InlineHandler>,
        H: FnOnce(&BlockType) -> Option<&blocks::FullHandler>,
    {
        tokio::task::block_in_place(|| {
            self.map_handler_sync(
                selected_inv_slot,
                get_item_handler,
                coord,
                get_block_inline_handler,
                get_block_full_handler,
            )
        })
    }

    fn map_handler_sync<F, G, H>(
        &mut self,
        selected_inv_slot: u32,
        get_item_handler: F,
        coord: BlockCoordinate,
        get_block_inline_handler: G,
        get_block_full_handler: H,
    ) -> std::result::Result<(), anyhow::Error>
    where
        F: FnOnce(&Item) -> Option<&items::BlockInteractionHandler>,
        G: FnOnce(&BlockType) -> Option<&blocks::InlineHandler>,
        H: FnOnce(&BlockType) -> Option<&blocks::FullHandler>,
    {
        self.game_state
            .inventory_manager()
            .mutate_inventory_atomically(&self.player_context.main_inventory(), |inventory| {
                let stack = inventory
                    .contents_mut()
                    .get_mut(selected_inv_slot as usize)
                    .with_context(|| "Item slot was out of bounds")?;

                let initiator = EventInitiator::Player(&self.player_context);

                let item_dig_handler = stack
                    .as_ref()
                    .and_then(|x| self.game_state.item_manager().from_stack(x))
                    .and_then(get_item_handler);

                let result = if let Some(handler) = item_dig_handler {
                    let ctx = HandlerContext {
                        tick: self.game_state.tick(),
                        initiator: initiator.clone(),
                        game_state: self.game_state.clone(),
                    };
                    run_handler!(
                        || {
                            handler(
                                ctx,
                                coord,
                                self.game_state.map().get_block(coord)?,
                                stack.as_ref().unwrap(),
                            )
                        },
                        "item dig handler",
                        initiator.clone(),
                    )?
                } else {
                    // This is blocking code, not async code (because of the mutex ops)
                    let obtained_items = self.game_state.map().run_block_interaction(
                        coord,
                        initiator,
                        stack.as_ref(),
                        get_block_inline_handler,
                        get_block_full_handler,
                    )?;
                    DigResult {
                        updated_tool: stack.clone(),
                        obtained_items,
                    }
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
            })
    }

    async fn send_ack(&mut self, sequence: u64) -> Result<()> {
        if sequence == 0 {
            return Ok(());
        };
        let message = proto::StreamToClient {
            tick: self.game_state.tick(),
            server_message: Some(proto::stream_to_client::ServerMessage::HandledSequence(
                sequence,
            )),
        };

        self.outbound_tx
            .send(Ok(message))
            .await
            .with_context(|| "Could not send outbound message (sequence ack)")?;

        Ok(())
    }

    fn handle_pos_update(&mut self, tick: u64, update: &proto::ClientUpdate) -> Result<()> {
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
                    tick,
                    position: pos_update
                        .position
                        .as_ref()
                        .with_context(|| "Missing position")?
                        .try_into()?,
                    velocity: pos_update
                        .velocity
                        .as_ref()
                        .with_context(|| "Missing velocity")?
                        .try_into()?,
                    face_direction: (az, el),
                };

                if let Some(pacing) = &update.pacing {
                    if pacing.pending_chunks < 16 {
                        self.chunk_pacing.increase();
                    } else if pacing.pending_chunks > 64 {
                        self.chunk_pacing.decrease();
                    }
                }

                self.own_positions.send_replace(PositionAndPacing {
                    position: pos,
                    chunks_to_send: self.chunk_pacing.get(),
                });
                self.player_context.update_position(pos);
            }
            None => {
                warn!("No position in update message from client");
            }
        }
        Ok(())
    }

    async fn handle_place(&mut self, place_message: &proto::PlaceAction) -> Result<()> {
        tokio::task::block_in_place(|| {
            let _span = span!("handle_place");
            self.game_state
                .inventory_manager()
                .mutate_inventory_atomically(&self.player_context.main_inventory(), |inventory| {
                    let stack = inventory
                        .contents_mut()
                        .get_mut(place_message.item_slot as usize)
                        .with_context(|| "Item slot was out of bounds")?;

                    let initiator = EventInitiator::Player(&self.player_context);

                    let handler = stack
                        .as_ref()
                        .and_then(|x| self.game_state.item_manager().from_stack(x))
                        .and_then(|x| x.place_handler.as_deref());
                    if let Some(handler) = handler {
                        let ctx = HandlerContext {
                            tick: self.game_state.tick(),
                            initiator: initiator.clone(),
                            game_state: self.game_state.clone(),
                        };
                        let new_stack = run_handler!(
                            || {
                                handler(
                                    ctx,
                                    place_message
                                        .block_coord
                                        .clone()
                                        .with_context(|| "Missing block_coord in place message")?
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
                            initiator,
                        )?;
                        *stack = new_stack;
                    }
                    Ok(())
                })
        })
    }

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
                    let mut player_state = self.player_context.player.state.lock();

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
                            &self.game_state,
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
                .with_context(|| "Could not send outbound message (inventory update)")?;
        }

        Ok(())
    }

    async fn handle_popup_response(
        &mut self,
        action: &cuberef_core::protocol::ui::PopupResponse,
    ) -> Result<()> {
        let user_action = if action.closed {
            PopupAction::PopupClosed
        } else {
            PopupAction::ButtonClicked(action.clicked_button.clone())
        };
        let updates = tokio::task::block_in_place(|| -> anyhow::Result<_> {
            let _span = span!("handle_popup_response");
            let mut player_state = self.player_context.state.lock();
            let mut updates = vec![];
            if action.closed {
                player_state
                    .inventory_manipulation_view
                    .clear_if_transient(Some(self.player_context.main_inventory_key))?;
                updates.push(make_inventory_update(
                    &self.game_state,
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
                    self.player_context.main_inventory(),
                )?;
                for view in popup.inventory_views().values() {
                    updates.push(make_inventory_update(
                        &self.game_state,
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
                    self.player_context.main_inventory(),
                )?;
                for view in player_state.inventory_popup.inventory_views().values() {
                    updates.push(make_inventory_update(
                        &self.game_state,
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
                .with_context(|| "Could not send outbound message (inventory update)")?;
        }

        Ok(())
    }

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
                .with_context(|| "Could not send outbound message (inventory update)")?;
        }
        Ok(())
    }

    fn handle_interact_key_sync(&mut self, interact_message: &proto::InteractKeyAction) -> Result<Vec<StreamToClient>> {
        let coord: BlockCoordinate = interact_message
            .block_coord
            .as_ref()
            .map(|x| x.into())
            .with_context(|| "Missing block_coord")?;
        let ctx = HandlerContext {
            tick: self.game_state.tick(),
            initiator: EventInitiator::Player(&self.player_context),
            game_state: self.game_state.clone(),
        };
        let block = self.game_state.map().get_block(coord)?;
        let mut messages = vec![];
        if let Some(handler) = &self
            .game_state
            .map()
            .block_type_manager()
            .get_block(&block)
            .unwrap()
            .0
            .interact_key_handler
        {
            if let Some(popup) = run_handler!(
                || (handler)(ctx, coord),
                "interact_key",
                EventInitiator::Player(&self.player_context.player),
            )? {
                messages.push(StreamToClient {
                    tick: self.game_state.tick(),
                    server_message: Some(ServerMessage::ShowPopup(popup.to_proto())),
                });
                for view in popup.inventory_views().values() {
                    messages.push(make_inventory_update(
                        &self.game_state,
                        &InventoryViewWithContext {
                            view,
                            context: &popup,
                        },
                    )?)
                }
                self.player_context
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
impl Drop for ClientInboundContext {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

// TODO tune these and make them adjustable via settings
// Units of chunks
const LOAD_EAGER_DISTANCE: i32 = 30;
const LOAD_LAZY_DISTANCE: i32 = 40;
const UNLOAD_DISTANCE: i32 = 50;
const MAX_UPDATE_BATCH_SIZE: usize = 256;

const INITIAL_CHUNKS_PER_UPDATE: usize = 16;
const MAX_CHUNKS_PER_UPDATE: usize = 512;

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
        v.sort_by_key(|(x, y, z)| x.abs() + y.abs() + z.abs());
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
