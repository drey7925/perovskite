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

use std::{pin::Pin, sync::Arc};

use cuberef_core::protocol::game_rpc::cuberef_game_server::CuberefGame;
use cuberef_core::protocol::game_rpc::stream_to_server::ClientMessage;
use cuberef_core::protocol::game_rpc::{self as proto, StreamToClient, StreamToServer};

use log::{error, info, warn};
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, Stream};
use tonic::metadata::MetadataMap;
use tonic::{Request, Response, Result, Status, Streaming};

use crate::game_state::GameState;

use super::client_context::make_client_contexts;

fn get_metadata(req_metadata: &MetadataMap, key: &str) -> Result<String, tonic::Status> {
    match req_metadata.get(key).map(|x| x.to_str()) {
        Some(Ok(x)) => Ok(x.to_string()),
        Some(Err(e)) => {
            warn!("Could not decode string metadata for {}: {:?}", key, e);
            Err(tonic::Status::invalid_argument(format!(
                "String decode error for {}",
                key
            )))
        }
        None => Err(tonic::Status::invalid_argument(format!("Missing {}", key))),
    }
}

pub struct CuberefGameServerImpl {
    game_state: Arc<GameState>,
}

impl CuberefGameServerImpl {
    pub fn new(game_state: Arc<GameState>) -> Self {
        Self { game_state }
    }
}

#[tonic::async_trait]
impl CuberefGame for CuberefGameServerImpl {
    type GameStreamStream =
        Pin<Box<dyn Stream<Item = Result<proto::StreamToClient, Status>> + Send>>;

    async fn game_stream(
        &self,
        req: Request<Streaming<proto::StreamToServer>>,
    ) -> Result<Response<Self::GameStreamStream>> {
        info!("Stream established from {:?}", req.remote_addr());
        let (outbound_tx, outbound_rx) = mpsc::channel(4);
        let game_state = self.game_state.clone();
        tokio::spawn(async move { game_stream_impl(game_state, req.into_inner(), outbound_tx).await.unwrap() });

        Result::Ok(Response::new(Box::pin(ReceiverStream::new(outbound_rx))))
    }

    async fn get_block_defs(
        &self,
        _req: Request<proto::GetBlockDefsRequest>,
    ) -> Result<Response<proto::GetBlockDefsResponse>> {
        Result::Ok(Response::new(proto::GetBlockDefsResponse {
            block_types: self
                .game_state
                .map()
                .block_type_manager()
                .to_client_protos(),
        }))
    }

    async fn get_item_defs(
        &self,
        _req: Request<proto::GetItemDefsRequest>,
    ) -> Result<Response<proto::GetItemDefsResponse>> {
        Result::Ok(Response::new(proto::GetItemDefsResponse {
            item_defs: self
                .game_state
                .item_manager()
                .registered_items()
                .map(|x| x.proto.clone())
                .collect(),
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
                Ok(data) => Result::Ok(Response::new(proto::GetMediaResponse { media: data })),
                Err(e) => Result::Err(Status::internal(e.to_string())),
            },
            None => Result::Err(Status::not_found(format!(
                "{} unimplemented",
                &req.get_ref().media_name
            ))),
        }
    }

    async fn list_media(
        &self,
        _req: Request<proto::ListMediaRequest>,
    ) -> Result<Response<proto::ListMediaResponse>> {
        Result::Ok(Response::new(proto::ListMediaResponse {
            media: self
                .game_state
                .media_resources()
                .entries()
                .map(|(k, v)| proto::ListMediaEntry {
                    media_name: k.clone(),
                    sha256: v.hash().to_vec(),
                })
                .collect(),
        }))
    }
}

async fn game_stream_impl(
    game_state: Arc<GameState>,
    mut inbound_rx: tonic::Streaming<StreamToServer>,
    outbound_tx: mpsc::Sender<tonic::Result<StreamToClient>>,
) -> anyhow::Result<()> {
    let username = match game_state
        .auth()
        .do_auth_flow(&mut inbound_rx, &outbound_tx)
        .await
    {
        Ok(x) => x,
        Err(e) => {
            return outbound_tx
                .send(Err(e))
                .await
                .map_err(|_| anyhow::Error::msg("Failed to send auth error"));
        }
    };
    let player_context = game_state
        .player_manager()
        .clone()
        .connect(&username)
        .map_err(|_x| Status::internal("Failed to establish player context"))?;

    // Wait for the initial ready message from the client
    match inbound_rx.message().await? {
        Some(StreamToServer {
            client_message: Some(ClientMessage::ClientInitialReady(_)),
            ..
        }) => {
            // all OK
            log::info!("Client reports ready; starting up client's context on server");
        }
        Some(_) => {
            let err_response = Err(tonic::Status::invalid_argument(
                "Client did not send a ClientInitialReady as its first message after authenticating.",
            ));
            return outbound_tx
                .send(err_response)
                .await
                .map_err(|_| anyhow::Error::msg("Failed to send auth error"));
        }
        None => {
            let err_response = Err(tonic::Status::unavailable(
            "Client did not send a ClientInitialReady and disconnected instead.",
        ));
            return outbound_tx
                .send(err_response)
                .await
                .map_err(|_| anyhow::Error::msg("Failed to send auth error"));
        }
    }

    let (mut inbound, mut outbound) = match make_client_contexts(
        game_state.clone(),
        player_context,
        inbound_rx,
        outbound_tx.clone(),
    )
    .await
    {
        Ok((inbound, outbound)) => (inbound, outbound),
        Err(e) => {
            error!("Error setting up client context: {:?}", e);
            return outbound_tx
                .send(Err(tonic::Status::internal(format!(
                    "Failure setting up client contexts: {e:?}"
                ))))
                .await
                .map_err(|_| anyhow::Error::msg("Failed to send context setup error"));
        }
    };

    // TODO handle the result rather than just quietly shutting down
    tokio::spawn(async move { inbound.run_inbound_loop().await.unwrap() });
    tokio::spawn(async move { outbound.run_outbound_loop().await.unwrap() });
    Ok(())
}
