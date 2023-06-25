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
    sync::Arc,
    time::{Duration, Instant},
};

use self::client_context::*;
use anyhow::{bail, Context, Error, Result};

use cuberef_core::{
    auth::CuberefOpaqueAuth,
    protocol::game_rpc::{
        self as rpc, cuberef_game_client::CuberefGameClient, stream_to_client::ServerMessage,
        stream_to_server::ClientMessage, GetBlockDefsRequest, GetItemDefsRequest, GetMediaRequest,
        StartAuth, StreamToClient, StreamToServer,
    },
};
use image::DynamicImage;
use opaque_ke::{
    ClientLoginFinishParameters, ClientRegistrationFinishParameters, CredentialResponse,
    RegistrationResponse,
};
use parking_lot::{Mutex};
use rand::rngs::OsRng;
use rustc_hash::{FxHashMap};
use tokio::{
    sync::{mpsc, watch},
};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tonic::{async_trait, codegen::CompressionEncoding, transport::Channel, Request, Streaming};


use crate::{
    cube_renderer::{AsyncTextureLoader, BlockRenderer, ClientBlockTypeManager},
    game_state::{
        items::{ClientItemManager, InventoryViewManager},
        physics::PhysicsState,
        tool_controller::ToolController,
        ChunkManager, ClientState,
    },
    vulkan::VulkanContext,
};

mod client_context;
pub(crate) mod mesh_worker;

async fn connect_grpc(
    server_addr: String,
) -> Result<rpc::cuberef_game_client::CuberefGameClient<Channel>> {
    CuberefGameClient::connect(server_addr)
        .await
        .map_err(|e| anyhow::Error::msg(e.to_string()))
}

pub(crate) async fn connect_game(
    server_addr: String,
    username: String,
    password: String,
    register: bool,
    ctx: &VulkanContext,
    progress: &mut watch::Sender<(f32, String)>,
) -> Result<Arc<ClientState>> {
    progress.send((0.1, "Connecting to server...".to_string()))?;
    log::info!("Connecting to {}...", &server_addr);
    let mut connection = connect_grpc(server_addr.clone())
        .await?
        .max_decoding_message_size(1024 * 1024 * 256)
        .accept_compressed(CompressionEncoding::Gzip)
        .send_compressed(CompressionEncoding::Gzip);

    // todo tune depth
    let (tx_send, tx_recv) = mpsc::channel(4);

    let request = Request::new(ReceiverStream::new(tx_recv));
    let mut stream = connection.game_stream(request).await?.into_inner();
    progress.send((0.2, "Logging into server...".to_string()))?;
    do_auth_handshake(&tx_send, &mut stream, username, password, register).await?;

    log::info!("Connection to {} established.", &server_addr);
    progress.send((0.3, "Loading block definitions...".to_string()))?;
    let block_defs_proto = connection.get_block_defs(GetBlockDefsRequest {}).await?;
    log::info!(
        "{} block defs loaded from server",
        block_defs_proto.get_ref().block_types.len()
    );

    progress.send((0.4, "Loading block textures...".to_string()))?;
    let texture_loader = GrpcTextureLoader {
        connection: connection.clone(),
    };

    let block_types = Arc::new(ClientBlockTypeManager::new(
        block_defs_proto.into_inner().block_types,
    )?);

    progress.send((0.5, "Setting up block renderer...".to_string()))?;
    let cube_renderer =
        Arc::new(BlockRenderer::new(block_types.clone(), texture_loader.clone(), ctx).await?);

    progress.send((0.6, "Loading item definitions...".to_string()))?;
    let item_defs_proto = connection.get_item_defs(GetItemDefsRequest {}).await?;
    log::info!(
        "{} item defs loaded from server",
        item_defs_proto.get_ref().item_defs.len()
    );
    let items = Arc::new(ClientItemManager::new(
        item_defs_proto.into_inner().item_defs,
    )?);

    progress.send((0.7, "Loading item textures...".to_string()))?;
    let (hud, egui) = crate::game_ui::make_uis(items.clone(), texture_loader, ctx).await?;

    // TODO clean up this hacky cloning of the context.
    // We need to clone it to start up the game ui without running into borrow checker issues,
    // since it provides access to the allocators.

    let (action_sender, action_receiver) = mpsc::channel(4);
    let client_state = Arc::new(ClientState {
        block_types,
        items,
        last_update: Mutex::new(Instant::now()),
        physics_state: Mutex::new(PhysicsState::new()),
        tool_controller: Mutex::new(ToolController::new()),
        chunks: ChunkManager::new(),
        inventories: Mutex::new(InventoryViewManager {
            inventory_views: FxHashMap::default(),
        }),
        shutdown: CancellationToken::new(),
        actions: action_sender,
        cube_renderer,
        hud: Arc::new(Mutex::new(hud)),
        egui: Arc::new(Mutex::new(egui)),
        pending_error: Mutex::new(None),
    });

    let (mut inbound, mut outbound) =
        make_contexts(client_state.clone(), tx_send, stream, action_receiver).await?;
    // todo track exit status and catch errors?
    // Right now we just print a panic message, but don't actually exit because of it
    tokio::spawn(async move {
        match inbound.run_inbound_loop().await {
            Ok(_) => log::info!("Inbound loop shut down normally"),
            Err(e) => log::info!("Inbound loop crashed: {e:?}"),
        }
    });
    tokio::spawn(async move {
        match outbound.run_outbound_loop().await {
            Ok(_) => log::info!("Outbound loop shut down normally"),
            Err(e) => log::info!("Outbound loop crashed: {e:?}"),
        }
    });

    progress.send((1.0, "Connected!".to_string()))?;
    tokio::time::sleep(Duration::from_secs_f64(0.25)).await;
    Ok(client_state)
}

async fn do_auth_handshake(
    tx: &mpsc::Sender<StreamToServer>,
    rx: &mut Streaming<StreamToClient>,
    username: String,
    password: String,
    register: bool,
) -> Result<()> {
    if register {
        do_register_handshake(tx, rx, username, password).await
    } else {
        do_login_handshake(tx, rx, username, password).await
    }
}

async fn do_register_handshake(
    tx: &mpsc::Sender<StreamToServer>,
    rx: &mut Streaming<StreamToClient>,
    username: String,
    password: String,
) -> Result<()> {
    let mut client_rng = OsRng;
    let client_state = opaque_ke::ClientRegistration::<CuberefOpaqueAuth>::start(
        &mut client_rng,
        password.as_bytes(),
    )
    .map_err(|e| Error::msg(format!("OPAQUE ClientRegistration start failed: {e:?}")))?;
    tx.send(StreamToServer {
        sequence: 0,
        client_tick: 0,
        client_message: Some(ClientMessage::StartAuthentication(StartAuth {
            username,
            register: true,
            opaque_request: client_state.message.serialize().to_vec(),
        })),
    })
    .await?;
    let registration_response = match rx.message().await? {
        Some(StreamToClient {
            server_message: Some(ServerMessage::ServerRegistrationResponse(resp)),
            ..
        }) => RegistrationResponse::deserialize(&resp).map_err(|e| {
            Error::msg(format!(
                "OPAQUE RegistrationResponse couldn't be decoded: {e:?}"
            ))
        })?,
        Some(x) => bail!(
            "Server sent an unexpected message in response to the registration request: {x:?}"
        ),
        None => bail!("Server disconnected before finishing registration"),
    };
    let finish_registration_result = client_state
        .state
        .finish(
            &mut client_rng,
            password.as_bytes(),
            registration_response,
            ClientRegistrationFinishParameters::default(),
        )
        .map_err(|e| Error::msg(format!("OPAQUE ClientRegistration finish failed: {e:?}")))?;
    tx.send(StreamToServer {
        sequence: 0,
        client_tick: 0,
        client_message: Some(ClientMessage::ClientRegistrationUpload(
            finish_registration_result.message.serialize().to_vec(),
        )),
    })
    .await?;
    match rx.message().await? {
        Some(StreamToClient {
            server_message: Some(ServerMessage::AuthSuccess(_)),
            ..
        }) => Ok(()),
        Some(x) => {
            bail!("Server sent an unexpected message instead of confirming auth success: {x:?}")
        }
        None => bail!("Server disconnected before confirming auth success"),
    }
}

async fn do_login_handshake(
    tx: &mpsc::Sender<StreamToServer>,
    rx: &mut Streaming<StreamToClient>,
    username: String,
    password: String,
) -> Result<()> {
    let mut client_rng = OsRng;
    let client_state =
        opaque_ke::ClientLogin::<CuberefOpaqueAuth>::start(&mut client_rng, password.as_bytes())
            .map_err(|e| Error::msg(format!("OPAQUE ClientLogin start failed: {e:?}")))?;
    tx.send(StreamToServer {
        sequence: 0,
        client_tick: 0,
        client_message: Some(ClientMessage::StartAuthentication(StartAuth {
            username,
            register: false,
            opaque_request: client_state.message.serialize().to_vec(),
        })),
    })
    .await?;
    let registration_response = match rx.message().await? {
        Some(StreamToClient {
            server_message: Some(ServerMessage::ServerLoginResponse(resp)),
            ..
        }) => CredentialResponse::deserialize(&resp).map_err(|e| {
            Error::msg(format!(
                "OPAQUE login CredentialResponse couldn't be decoded: {e:?}"
            ))
        })?,
        Some(x) => {
            bail!("Server sent an unexpected message in response to the login request: {x:?}")
        }
        None => bail!("Server disconnected before finishing login"),
    };
    let finish_login_result = client_state
        .state
        .finish(
            password.as_bytes(),
            registration_response,
            ClientLoginFinishParameters::default(),
        )
        .map_err(|e| Error::msg(format!("OPAQUE ClientLogin finish failed: {e:?}")))?;
    tx.send(StreamToServer {
        sequence: 0,
        client_tick: 0,
        client_message: Some(ClientMessage::ClientLoginCredential(
            finish_login_result.message.serialize().to_vec(),
        )),
    })
    .await?;

    match rx.message().await? {
        Some(StreamToClient {
            server_message: Some(ServerMessage::AuthSuccess(_)),
            ..
        }) => Ok(()),
        Some(x) => {
            bail!("Server sent an unexpected message instead of confirming auth success: {x:?}")
        }
        None => bail!("Server disconnected before confirming auth success"),
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
