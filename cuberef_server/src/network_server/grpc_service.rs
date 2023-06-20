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

use std::net::SocketAddr;
use std::{pin::Pin, sync::Arc};

use cuberef_core::protocol::game_rpc::cuberef_game_server::CuberefGame;
use cuberef_core::protocol::game_rpc::{self as proto};

use log::{error, info, warn};
use tokio_stream::{wrappers::ReceiverStream, Stream};
use tonic::metadata::MetadataMap;
use tonic::{Request, Response, Result, Status, Streaming};

use crate::game_state::GameState;

use super::auth::AuthService;
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
    auth_service: Arc<dyn AuthService>,
}

impl CuberefGameServerImpl {
    pub fn new(game_state: Arc<GameState>, auth_service: Arc<dyn AuthService>) -> Self {
        Self {
            game_state,
            auth_service,
        }
    }

    fn check_auth_token(
        &self,
        req_metadata: &MetadataMap,
        remote_addr: Option<SocketAddr>,
    ) -> Result<String, tonic::Status> {
        let username = get_metadata(req_metadata, "x-cuberef-username")?;
        let token = get_metadata(req_metadata, "x-cuberef-token")?;
        match self
            .auth_service
            .check_token(&username, &token, remote_addr)
        {
            Ok(super::auth::TokenOutcome::Success) => Ok(username),
            Ok(super::auth::TokenOutcome::Failure) => {
                Err(tonic::Status::unauthenticated("Bad token"))
            }
            Err(e) => {
                error!("check_token failed: {:?}", e);
                Err(tonic::Status::internal("Internal auth error"))
            }
        }
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
        let username = self.check_auth_token(req.metadata(), req.remote_addr())?;
        info!("Stream established from {:?}", req.remote_addr());

        let player_context = self
            .game_state
            .player_manager()
            .clone()
            .connect(&username)
            .map_err(|_x| Status::internal("Failed to establish player context"))?;

        let inbound = req.into_inner();

        let (mut inbound, mut outbound, outbound_rx) =
            match make_client_contexts(self.game_state.clone(), player_context, inbound).await {
                Ok((inbound, outbound, rx)) => (inbound, outbound, rx),
                Err(e) => {
                    error!("Error setting up client context: {:?}", e);
                    return Result::Err(Status::internal(
                        "Failed to establish client context on server",
                    ));
                }
            };

        // TODO handle the result rather than just quietly shutting down
        tokio::spawn(async move { inbound.run_inbound_loop().await.unwrap() });
        tokio::spawn(async move { outbound.run_outbound_loop().await.unwrap() });

        Result::Ok(Response::new(Box::pin(ReceiverStream::new(outbound_rx))))
    }

    async fn authenticate(
        &self,
        _req: Request<proto::AuthRequest>,
    ) -> Result<Response<proto::AuthResponse>> {
        Result::Err(Status::unimplemented("authenticate unimplemented"))
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
