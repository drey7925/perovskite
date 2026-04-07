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

use anyhow::{bail, ensure};
use perovskite_core::protocol::game_rpc::perovskite_game_server::PerovskiteGame;
use perovskite_core::protocol::game_rpc::stream_to_client::ServerMessage;
use perovskite_core::protocol::game_rpc::stream_to_server::ClientMessage;
use perovskite_core::protocol::game_rpc::{
    self as proto, AuthSuccess, StreamToClient, StreamToServer,
};
use std::{pin::Pin, sync::Arc};

use crate::game_state::GameState;
use log::{error, info, warn};
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, Stream};
use tonic::metadata::MetadataMap;
use tonic::{Request, Response, Result, Status, Streaming};

use super::client_context::make_client_contexts;

fn get_metadata(req_metadata: &MetadataMap, key: &str) -> Result<String, Status> {
    match req_metadata.get(key).map(|x| x.to_str()) {
        Some(Ok(x)) => Ok(x.to_string()),
        Some(Err(e)) => {
            warn!("Could not decode string metadata for {}: {:?}", key, e);
            Err(Status::invalid_argument(format!(
                "String decode error for {}",
                key
            )))
        }
        None => Err(Status::invalid_argument(format!("Missing {}", key))),
    }
}

pub struct PerovskiteGameServerImpl {
    game_state: Arc<GameState>,
}

impl PerovskiteGameServerImpl {
    pub fn new(game_state: Arc<GameState>) -> Self {
        Self { game_state }
    }
}

#[tonic::async_trait]
impl PerovskiteGame for PerovskiteGameServerImpl {
    type GameStreamStream = Pin<Box<dyn Stream<Item = Result<StreamToClient, Status>> + Send>>;

    async fn game_stream(
        &self,
        req: Request<Streaming<StreamToServer>>,
    ) -> Result<Response<Self::GameStreamStream>> {
        info!("Stream established from {:?}", req.remote_addr());
        let (outbound_tx, outbound_rx) = mpsc::channel(4);
        let game_state = self.game_state.clone();
        crate::spawn_async("game_stream", async move {
            match game_stream_impl(game_state, req.into_inner(), outbound_tx).await {
                Ok(()) => (),
                Err(e) => tracing::error!("Error running game stream: {:?}", e),
            }
        })
        .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(Box::pin(ReceiverStream::new(outbound_rx))))
    }

    async fn get_block_defs(
        &self,
        req: Request<proto::GetBlockDefsRequest>,
    ) -> Result<Response<proto::GetBlockDefsResponse>> {
        let all: Vec<_> = self
            .game_state
            .game_map()
            .block_type_manager()
            .to_client_protos();
        let (page, next_token) = apply_pagination(&all, req.get_ref().pagination_token);
        Ok(Response::new(proto::GetBlockDefsResponse {
            block_types: page.to_vec(),
            next_pagination_token: next_token,
        }))
    }

    async fn get_item_defs(
        &self,
        req: Request<proto::GetItemDefsRequest>,
    ) -> Result<Response<proto::GetItemDefsResponse>> {
        let all: Vec<_> = self
            .game_state
            .item_manager()
            .registered_items()
            .map(|x| x.proto.clone())
            .collect();
        let (page, next_token) = apply_pagination(&all, req.get_ref().pagination_token);
        Ok(Response::new(proto::GetItemDefsResponse {
            item_defs: page.to_vec(),
            next_pagination_token: next_token,
        }))
    }

    async fn list_media(
        &self,
        req: Request<proto::ListMediaRequest>,
    ) -> Result<Response<proto::ListMediaResponse>> {
        let all: Vec<_> = self
            .game_state
            .media_resources()
            .entries()
            .map(|(k, v)| proto::ListMediaEntry {
                media_name: k.clone(),
                sha256: v.hash().to_vec(),
            })
            .collect();
        let (page, next_token) = apply_pagination(&all, req.get_ref().pagination_token);
        Ok(Response::new(proto::ListMediaResponse {
            media: page.to_vec(),
            next_pagination_token: next_token,
        }))
    }

    async fn get_media(
        &self,
        req: Request<proto::GetMediaRequest>,
    ) -> Result<Response<proto::GetMediaResponse>> {
        match self
            .game_state
            .media_resources()
            .get(&req.get_ref().media_name)
        {
            Some(resource) => match resource.data() {
                Ok(data) => Ok(Response::new(proto::GetMediaResponse { media: data })),
                Err(e) => Err(Status::internal(e.to_string())),
            },
            None => Err(Status::not_found(format!(
                "media item `{}` missing",
                &req.get_ref().media_name
            ))),
        }
    }

    async fn get_entity_defs(
        &self,
        req: Request<proto::GetEntityDefsRequest>,
    ) -> Result<Response<proto::GetEntityDefsResponse>> {
        let all: Vec<_> = self.game_state.entities().types().to_client_protos();
        let (page, next_token) = apply_pagination(&all, req.get_ref().pagination_token);
        Ok(Response::new(proto::GetEntityDefsResponse {
            entity_defs: page.to_vec(),
            next_pagination_token: next_token,
        }))
    }

    async fn get_audio_defs(
        &self,
        req: Request<proto::GetAudioDefsRequest>,
    ) -> Result<Response<proto::GetAudioDefsResponse>> {
        let all: Vec<_> = self
            .game_state
            .media_resources()
            .sampled_sound_client_protos();
        let (page, next_token) = apply_pagination(&all, req.get_ref().pagination_token);
        Ok(Response::new(proto::GetAudioDefsResponse {
            sampled_sounds: page.to_vec(),
            next_pagination_token: next_token,
        }))
    }
}

pub(crate) const SERVER_MIN_PROTOCOL_VERSION: u32 = 11;
pub(crate) const SERVER_MAX_PROTOCOL_VERSION: u32 = 11;

/// Maximum number of items returned in a single paginated response (protocol 11+).
#[cfg(not(test))]
const PAGE_SIZE: usize = 1024;
#[cfg(test)]
const PAGE_SIZE: usize = 3;

/// Returns `(page_slice, next_token)`.  `next_token` is 0 when this is the last page.
fn apply_pagination<T>(items: &[T], token: u64) -> (&[T], u64) {
    let start = token as usize;
    let end = (start + PAGE_SIZE).min(items.len());
    let next_token = if end < items.len() { end as u64 } else { 0 };
    (&items[start..end], next_token)
}

async fn game_stream_impl(
    game_state: Arc<GameState>,
    mut inbound_rx: Streaming<StreamToServer>,
    outbound_tx: mpsc::Sender<Result<StreamToClient>>,
) -> anyhow::Result<()> {
    let auth_outcome = match game_state
        .auth()
        .do_auth_flow(&mut inbound_rx, &outbound_tx)
        .await
    {
        Ok(outcome) => {
            info!(
                "Player {} successfully authenticated, protocol range {}..={}",
                outcome.username, outcome.min_protocol_version, outcome.max_protocol_version
            );
            outcome
        }
        Err(e) => {
            log::error!("Player failed to authenticate: {e:?}");
            return outbound_tx
                .send(Err(e))
                .await
                .map_err(|_| anyhow::Error::msg("Failed to send auth error"));
        }
    };

    let effective_min_protocol = SERVER_MIN_PROTOCOL_VERSION.max(auth_outcome.min_protocol_version);
    let effective_max_protocol = SERVER_MAX_PROTOCOL_VERSION.min(auth_outcome.max_protocol_version);
    ensure!(effective_min_protocol <= effective_max_protocol);

    outbound_tx
        .send(Ok(StreamToClient {
            tick: game_state.tick(),
            server_message: Some(ServerMessage::AuthSuccess(AuthSuccess {
                effective_protocol_version: effective_max_protocol,
            })),
            performance_metrics: None,
        }))
        .await?;

    let (player_context, player_event_receiver) = match game_state
        .player_manager()
        .clone()
        .connect(&auth_outcome.username)
    {
        Ok(x) => x,
        Err(e) => {
            log::error!("Failed to establish player context: {:?}", e);
            outbound_tx
                .send(Ok(StreamToClient {
                    tick: game_state.tick(),
                    server_message: Some(ServerMessage::ShutdownMessage(format!(
                        "Failed to establish player context: {e:?}"
                    ))),
                    performance_metrics: None,
                }))
                .await?;
            return Err(Status::internal("Failed to establish player context").into());
        }
    };

    // Wait for the initial ready message from the client
    match inbound_rx.message().await? {
        Some(StreamToServer {
            client_message: Some(ClientMessage::ClientInitialReady(_)),
            ..
        }) => {
            // all OK
            info!(
                "Client for {} reports ready; starting up client's context on server",
                auth_outcome.username
            );
        }
        Some(_) => {
            let err_response = Err(Status::invalid_argument(
                "Client did not send a ClientInitialReady as its first message after authenticating.",
            ));
            return outbound_tx.send(err_response).await.map_err(|_| {
                anyhow::Error::msg("Failed to send ClientInitialReady-expected error")
            });
        }
        None => {
            let err_response = Err(Status::unavailable(
                "Client did not send a ClientInitialReady and disconnected instead.",
            ));
            return outbound_tx
                .send(err_response)
                .await
                .map_err(|_| anyhow::Error::msg("Failed to send auth error"));
        }
    }

    match make_client_contexts(
        game_state.clone(),
        player_context,
        player_event_receiver,
        inbound_rx,
        outbound_tx.clone(),
        effective_max_protocol,
    )
    .await
    {
        Ok(coroutines) => coroutines.run_all().await?,
        Err(e) => {
            error!("Error setting up client context: {:?}", e);
            return outbound_tx
                .send(Err(Status::internal(format!(
                    "Failure setting up client contexts: {e:?}"
                ))))
                .await
                .map_err(|_| anyhow::Error::msg("Failed to send context setup error"));
        }
    };

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{apply_pagination, PAGE_SIZE};

    // Under cfg(test), PAGE_SIZE == 3.

    #[test]
    fn pagination_empty() {
        let items: Vec<u32> = vec![];
        let (page, next) = apply_pagination(&items, 0);
        assert!(page.is_empty());
        assert_eq!(next, 0);
    }

    #[test]
    fn pagination_fits_in_one_page() {
        let items: Vec<u32> = (0..PAGE_SIZE as u32).collect();
        let (page, next) = apply_pagination(&items, 0);
        assert_eq!(page, &items[..]);
        assert_eq!(next, 0, "should be last page");
    }

    #[test]
    fn pagination_exactly_one_more_than_page() {
        // PAGE_SIZE + 1 items → first page has PAGE_SIZE, second has 1.
        let items: Vec<u32> = (0..(PAGE_SIZE as u32 + 1)).collect();

        let (page0, next0) = apply_pagination(&items, 0);
        assert_eq!(page0.len(), PAGE_SIZE);
        assert_eq!(next0, PAGE_SIZE as u64, "token should point to next item");

        let (page1, next1) = apply_pagination(&items, next0);
        assert_eq!(page1.len(), 1);
        assert_eq!(next1, 0, "should be last page");
    }

    #[test]
    fn pagination_multiple_full_pages() {
        // 2 * PAGE_SIZE items → two full pages.
        let n = (2 * PAGE_SIZE) as u32;
        let items: Vec<u32> = (0..n).collect();

        let (page0, next0) = apply_pagination(&items, 0);
        assert_eq!(page0.len(), PAGE_SIZE);
        assert_eq!(next0, PAGE_SIZE as u64);

        let (page1, next1) = apply_pagination(&items, next0);
        assert_eq!(page1.len(), PAGE_SIZE);
        assert_eq!(next1, 0, "should be last page");

        // Collected items should match the original.
        let mut collected: Vec<u32> = page0.to_vec();
        collected.extend_from_slice(page1);
        assert_eq!(collected, items.as_slice());
    }

    #[test]
    fn pagination_three_pages() {
        // 2 * PAGE_SIZE + 2 items → two full pages + one partial page.
        let extra = 2u32;
        let n = (2 * PAGE_SIZE) as u32 + extra;
        let items: Vec<u32> = (0..n).collect();

        let (page0, next0) = apply_pagination(&items, 0);
        assert_eq!(page0.len(), PAGE_SIZE);

        let (page1, next1) = apply_pagination(&items, next0);
        assert_eq!(page1.len(), PAGE_SIZE);

        let (page2, next2) = apply_pagination(&items, next1);
        assert_eq!(page2.len(), extra as usize);
        assert_eq!(next2, 0);

        let mut collected: Vec<u32> = page0.to_vec();
        collected.extend_from_slice(page1);
        collected.extend_from_slice(page2);
        assert_eq!(collected, items.as_slice());
    }
}
