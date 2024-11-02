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

use std::{sync::Arc, time::Duration};

use self::client_context::*;
use anyhow::{bail, Error, Result};

use arc_swap::ArcSwap;
use opaque_ke::{
    ClientLoginFinishParameters, ClientRegistrationFinishParameters, CredentialResponse,
    RegistrationResponse,
};
use parking_lot::Mutex;
use perovskite_core::protocol::game_rpc::GetAudioDefsRequest;
use perovskite_core::{
    auth::PerovskiteOpaqueAuth,
    protocol::game_rpc::{
        self as rpc, perovskite_game_client::PerovskiteGameClient, stream_to_client::ServerMessage,
        stream_to_server::ClientMessage, GetBlockDefsRequest, GetEntityDefsRequest,
        GetItemDefsRequest, GetMediaRequest, ListMediaRequest, StartAuth, StreamToClient,
        StreamToServer,
    },
};
use rand::rngs::OsRng;
use tokio::sync::{mpsc, watch};
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::{Certificate, ClientTlsConfig};
use tonic::{async_trait, transport::Channel, Request, Streaming};
use unicode_normalization::UnicodeNormalization;

use crate::vulkan::VulkanContext;
use crate::{
    audio,
    cache::CacheManager,
    game_state::{
        block_types::ClientBlockTypeManager, items::ClientItemManager, settings::GameSettings,
        timekeeper::Timekeeper, ClientState,
    },
    vulkan::{block_renderer::BlockRenderer, entity_renderer::EntityRenderer, VulkanWindow},
};

mod client_context;
pub(crate) mod mesh_worker;

const MIN_PROTOCOL_VERSION: u32 = 5;
const MAX_PROTOCOL_VERSION: u32 = 5;

async fn connect_grpc(server_addr: String) -> Result<PerovskiteGameClient<Channel>> {
    let tls = ClientTlsConfig::new()
        .with_native_roots()
        .with_webpki_roots();
    let channel = Channel::from_shared(server_addr)?
        .tls_config(tls)?
        .connect_timeout(Duration::from_secs(10))
        .connect()
        .await?;

    Ok(PerovskiteGameClient::new(channel))
}

const TOTAL_STEPS: f32 = 12.0;

pub(crate) async fn connect_game(
    server_addr: String,
    username: String,
    password: String,
    register: bool,
    settings: Arc<ArcSwap<GameSettings>>,
    vk_ctx: Arc<VulkanContext>,
    progress: &mut watch::Sender<(f32, String)>,
) -> Result<Arc<ClientState>> {
    progress.send((1.0 / TOTAL_STEPS, "Connecting to server...".to_string()))?;
    log::info!("Connecting to {}...", &server_addr);
    let mut connection = connect_grpc(server_addr.clone())
        .await?
        .max_decoding_message_size(1024 * 1024 * 256);

    // todo tune depth
    let (tx_send, tx_recv) = mpsc::channel(4);

    let request = Request::new(ReceiverStream::new(tx_recv));
    let mut stream = connection.game_stream(request).await?.into_inner();
    progress.send((2.0 / TOTAL_STEPS, "Logging into server...".to_string()))?;

    let AuthSuccess {
        protocol_version,
        tick: initial_tick,
    } = do_auth_handshake(&tx_send, &mut stream, username, password, register).await?;

    let timekeeper = Arc::new(Timekeeper::new(initial_tick));

    if protocol_version < MIN_PROTOCOL_VERSION {
        bail!(
            "Client is too new to handle protocol version {}",
            protocol_version
        )
    }
    if protocol_version > MAX_PROTOCOL_VERSION {
        bail!(
            "Client is too old to handle protocol version {}",
            protocol_version
        )
    }
    log::info!("Protocol version: {}", protocol_version);
    log::info!("Connection to {} established.", &server_addr);

    progress.send((3.0 / TOTAL_STEPS, "Fetching media list...".to_string()))?;
    let media_list = connection.list_media(ListMediaRequest::default()).await?;
    log::info!(
        "{} media items loaded from server",
        media_list.get_ref().media.len()
    );
    let texture_loader = GrpcTextureLoader {
        connection: connection.clone(),
    };
    let mut cache_manager = CacheManager::new(media_list.into_inner(), Box::new(texture_loader))?;

    progress.send((
        4.0 / TOTAL_STEPS,
        "Loading block definitions...".to_string(),
    ))?;
    let block_defs_proto = connection.get_block_defs(GetBlockDefsRequest {}).await?;
    log::info!(
        "{} block defs loaded from server",
        block_defs_proto.get_ref().block_types.len()
    );

    progress.send((5.0 / TOTAL_STEPS, "Loading block textures...".to_string()))?;

    let block_types = Arc::new(tokio::task::block_in_place(|| {
        ClientBlockTypeManager::new(block_defs_proto.into_inner().block_types)
    })?);

    progress.send((
        6.0 / TOTAL_STEPS,
        "Setting up block renderer...".to_string(),
    ))?;
    let block_renderer =
        { BlockRenderer::new(block_types.clone(), &mut cache_manager, vk_ctx.clone()).await? };

    progress.send((7.0 / TOTAL_STEPS, "Loading item definitions...".to_string()))?;
    let item_defs_proto = connection.get_item_defs(GetItemDefsRequest {}).await?;
    log::info!(
        "{} item defs loaded from server",
        item_defs_proto.get_ref().item_defs.len()
    );
    let items = Arc::new(ClientItemManager::new(
        item_defs_proto.into_inner().item_defs,
    )?);

    progress.send((8.0 / TOTAL_STEPS, "Loading item textures...".to_string()))?;
    let (hud, egui) = crate::game_ui::make_uis(
        items.clone(),
        &mut cache_manager,
        vk_ctx.clone(),
        &block_renderer,
        settings.clone(),
    )
    .await?;

    progress.send((
        9.0 / TOTAL_STEPS,
        "Loading entity definitions...".to_string(),
    ))?;
    let entity_defs = connection.get_entity_defs(GetEntityDefsRequest {}).await?;
    log::info!(
        "{} entity defs loaded from server",
        entity_defs.get_ref().entity_defs.len()
    );
    let entitity_renderer = EntityRenderer::new(
        entity_defs.into_inner().entity_defs,
        &mut cache_manager,
        &vk_ctx,
    )
    .await?;
    progress.send((10.0 / TOTAL_STEPS, "Loading audio files...".to_string()))?;
    let audio_defs = connection
        .get_audio_defs(GetAudioDefsRequest {})
        .await?
        .into_inner();

    let audio = audio::start_engine(
        settings.clone(),
        timekeeper.clone(),
        &audio_defs.sampled_sounds,
        &mut cache_manager,
    )
    .await?;

    let (action_sender, action_receiver) = mpsc::channel(4);

    let client_state = Arc::new(ClientState::new(
        settings,
        block_types,
        items,
        action_sender,
        hud,
        egui,
        block_renderer,
        entitity_renderer,
        timekeeper,
        audio,
    )?);
    tx_send
        .send(StreamToServer {
            sequence: 0,
            client_tick: 0,
            client_message: Some(ClientMessage::ClientInitialReady(rpc::Nop {})),
        })
        .await?;
    let initial_state_notification = Arc::new(tokio::sync::Notify::new());
    let (mut inbound, mut outbound) = make_contexts(
        client_state.clone(),
        tx_send,
        stream,
        action_receiver,
        protocol_version,
        initial_state_notification.clone(),
    )
    .await?;
    let client_state_clone = client_state.clone();
    tokio::spawn(async move {
        match inbound.run_inbound_loop().await {
            Ok(_) => log::info!("Inbound loop shut down normally"),
            Err(e) => {
                log::error!("Inbound loop crashed: {e:?}");
                *client_state_clone.pending_error.lock() = Some(e);
            }
        }
    });

    progress.send((
        11.0 / TOTAL_STEPS,
        "Waiting for initial game state...".to_string(),
    ))?;
    initial_state_notification.notified().await;
    tokio::spawn(async move {
        match outbound.run_outbound_loop().await {
            Ok(_) => log::info!("Outbound loop shut down normally"),
            Err(e) => log::error!("Outbound loop crashed: {e:?}"),
        }
    });

    progress.send((12.0 / TOTAL_STEPS, "Connected!".to_string()))?;
    // Sleep for a moment so the user can see the connected message
    tokio::time::sleep(Duration::from_secs_f64(0.25)).await;
    Ok(client_state)
}

// To avoid inconsistencies between operating systems and environments, attempt to normalize any
// unicode in the password.
fn normalize_password(password: String) -> String {
    password.nfkc().to_string()
}

struct AuthSuccess {
    protocol_version: u32,
    tick: u64,
}

async fn do_auth_handshake(
    tx: &mpsc::Sender<StreamToServer>,
    rx: &mut Streaming<StreamToClient>,
    username: String,
    password: String,
    register: bool,
) -> Result<AuthSuccess> {
    if register {
        do_register_handshake(tx, rx, username, normalize_password(password)).await
    } else {
        do_login_handshake(tx, rx, username, normalize_password(password)).await
    }
}

async fn do_register_handshake(
    tx: &mpsc::Sender<StreamToServer>,
    rx: &mut Streaming<StreamToClient>,
    username: String,
    password: String,
) -> Result<AuthSuccess> {
    let mut client_rng = OsRng;
    let client_state = opaque_ke::ClientRegistration::<PerovskiteOpaqueAuth>::start(
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
            min_protocol_version: MIN_PROTOCOL_VERSION,
            max_protocol_version: MAX_PROTOCOL_VERSION,
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
            server_message: Some(ServerMessage::AuthSuccess(success)),
            tick,
        }) => Ok(AuthSuccess {
            protocol_version: success.effective_protocol_version,
            tick,
        }),
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
) -> Result<AuthSuccess> {
    let mut client_rng = OsRng;
    let client_state =
        opaque_ke::ClientLogin::<PerovskiteOpaqueAuth>::start(&mut client_rng, password.as_bytes())
            .map_err(|e| Error::msg(format!("OPAQUE ClientLogin start failed: {e:?}")))?;
    tx.send(StreamToServer {
        sequence: 0,
        client_tick: 0,
        client_message: Some(ClientMessage::StartAuthentication(StartAuth {
            username,
            register: false,
            opaque_request: client_state.message.serialize().to_vec(),
            min_protocol_version: MIN_PROTOCOL_VERSION,
            max_protocol_version: MAX_PROTOCOL_VERSION,
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
            server_message: Some(ServerMessage::AuthSuccess(success)),
            tick,
        }) => Ok(AuthSuccess {
            protocol_version: success.effective_protocol_version,
            tick,
        }),
        Some(x) => {
            bail!("Server sent an unexpected message instead of confirming auth success: {x:?}")
        }
        None => bail!("Server disconnected before finishing login"),
    }
}

#[derive(Clone)]
struct GrpcTextureLoader {
    connection: PerovskiteGameClient<Channel>,
}
#[async_trait]
impl AsyncMediaLoader for GrpcTextureLoader {
    async fn load_media(&mut self, tex_name: &str) -> Result<Vec<u8>> {
        log::info!("Loading resource {}", tex_name);
        let resp = self
            .connection
            .get_media(GetMediaRequest {
                media_name: tex_name.to_string(),
            })
            .await?;
        Ok(resp.into_inner().media)
    }
}

#[async_trait]
pub(crate) trait AsyncMediaLoader {
    async fn load_media(&mut self, tex_name: &str) -> Result<Vec<u8>>;
}
