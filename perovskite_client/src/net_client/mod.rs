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

use self::client_workers::*;
use anyhow::{anyhow, bail, Context, Error, Result};

use arc_swap::ArcSwap;

use perovskite_core::protocol::game_rpc::{
    server_login_response, GetAudioDefsRequest, ServerLoginResponse,
};
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
use tokio::task::block_in_place;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tonic::transport::ClientTlsConfig;
use tonic::IntoRequest;
use tonic::{async_trait, transport::Channel, Request, Streaming};
use unicode_normalization::UnicodeNormalization;

use crate::client_state::ChunkManager;
use crate::vulkan::far_geometry::FarGeometryState;
use crate::vulkan::raytrace_buffer::RaytraceBufferManager;
use crate::vulkan::VulkanContext;
use crate::{
    audio,
    client_state::{
        block_types::ClientBlockTypeManager, items::ClientItemManager, settings::GameSettings,
        timekeeper::Timekeeper, ClientState,
    },
    media::CacheManager,
    vulkan::{block_renderer::BlockRenderer, entity_renderer::EntityRenderer},
};

mod client_workers;
pub(crate) mod mesh_worker;

const MIN_PROTOCOL_VERSION: u32 = 9;
const MAX_PROTOCOL_VERSION: u32 = 11;

async fn connect_grpc(server_addr: String) -> Result<PerovskiteGameClient<Channel>> {
    let tls = ClientTlsConfig::new()
        .with_native_roots()
        .with_webpki_roots();
    let channel = Channel::from_shared(server_addr)?
        .tls_config(tls)?
        .connect_timeout(Duration::from_secs(10))
        .user_agent("PerovskiteClient")?
        .connect()
        .await?;

    Ok(PerovskiteGameClient::new(channel))
}

const TOTAL_STEPS: f32 = 12.0;

pub(crate) trait WithVersionHeaders {
    fn with_version_headers(self) -> Self;
}

impl<T> WithVersionHeaders for Request<T> {
    fn with_version_headers(self) -> Self {
        let mut request = self;
        request.metadata_mut().append(
            perovskite_core::protocol::MIN_VERSION_HEADER,
            MIN_PROTOCOL_VERSION.to_string().try_into().unwrap(),
        );
        request.metadata_mut().append(
            perovskite_core::protocol::MAX_VERSION_HEADER,
            MAX_PROTOCOL_VERSION.to_string().try_into().unwrap(),
        );
        request
    }
}
pub(crate) async fn connect_game(
    server_addr: String,
    username: String,
    password: String,
    register: bool,
    settings: Arc<ArcSwap<GameSettings>>,
    vk_ctx: Arc<VulkanContext>,
    progress: &mut watch::Sender<(f32, String)>,
    cancel: CancellationToken,
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
    log::info!("Logging into server...");

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
    log::info!(
        "Connection to {} established with protocol version {}",
        &server_addr,
        protocol_version
    );

    progress.send((3.0 / TOTAL_STEPS, "Fetching media list...".to_string()))?;
    log::info!("Fetching media list...");
    let media_list = {
        let mut all_media = Vec::new();
        let mut pagination_token = 0u64;
        loop {
            let resp = connection
                .list_media(
                    ListMediaRequest { pagination_token }
                        .into_request()
                        .with_version_headers(),
                )
                .await?
                .into_inner();
            all_media.extend(resp.media);
            pagination_token = resp.next_pagination_token;
            if pagination_token == 0 {
                break;
            }
        }
        log::info!("{} media items loaded from server", all_media.len());
        perovskite_core::protocol::game_rpc::ListMediaResponse {
            media: all_media,
            next_pagination_token: 0,
        }
    };
    let texture_loader = GrpcTextureLoader {
        connection: connection.clone(),
    };
    let mut cache_manager = CacheManager::new(media_list, Box::new(texture_loader))?;

    progress.send((
        4.0 / TOTAL_STEPS,
        "Loading block definitions...".to_string(),
    ))?;
    log::info!("Loading block definitions...");
    let block_types_vec = {
        let mut all = Vec::new();
        let mut pagination_token = 0u64;
        loop {
            let resp = connection
                .get_block_defs(
                    GetBlockDefsRequest { pagination_token }
                        .into_request()
                        .with_version_headers(),
                )
                .await?
                .into_inner();
            all.extend(resp.block_types);
            pagination_token = resp.next_pagination_token;
            if pagination_token == 0 {
                break;
            }
        }
        log::info!("{} block defs loaded from server", all.len());
        all
    };

    progress.send((5.0 / TOTAL_STEPS, "Loading block textures...".to_string()))?;
    log::info!("Loading block textures...");
    let block_types = Arc::new(block_in_place(|| {
        ClientBlockTypeManager::new(block_types_vec)
    })?);

    progress.send((
        6.0 / TOTAL_STEPS,
        "Setting up block renderer...".to_string(),
    ))?;
    log::info!("Setting up block renderer...");
    let block_renderer =
        { BlockRenderer::new(block_types.clone(), &mut cache_manager, vk_ctx.clone()).await? };

    let chunks = block_in_place(|| -> Result<_> {
        Ok(ChunkManager::new(Arc::new(RaytraceBufferManager::new(
            vk_ctx.clone(),
        )?)))
    })?;

    progress.send((7.0 / TOTAL_STEPS, "Loading item definitions...".to_string()))?;
    log::info!("Loading item definitions...");
    let items = Arc::new(ClientItemManager::new({
        let mut all = Vec::new();
        let mut pagination_token = 0u64;
        loop {
            let resp = connection
                .get_item_defs(
                    GetItemDefsRequest { pagination_token }
                        .into_request()
                        .with_version_headers(),
                )
                .await?
                .into_inner();
            all.extend(resp.item_defs);
            pagination_token = resp.next_pagination_token;
            if pagination_token == 0 {
                break;
            }
        }
        log::info!("{} item defs loaded from server", all.len());
        all
    })?);

    progress.send((8.0 / TOTAL_STEPS, "Loading item textures...".to_string()))?;
    log::info!("Loading item textures...");
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
    log::info!("Loading entity definitions...");
    let entitity_renderer = EntityRenderer::new(
        {
            let mut all = Vec::new();
            let mut pagination_token = 0u64;
            loop {
                let resp = connection
                    .get_entity_defs(
                        GetEntityDefsRequest { pagination_token }
                            .into_request()
                            .with_version_headers(),
                    )
                    .await?
                    .into_inner();
                all.extend(resp.entity_defs);
                pagination_token = resp.next_pagination_token;
                if pagination_token == 0 {
                    break;
                }
            }
            log::info!("{} entity defs loaded from server", all.len());
            all
        },
        &mut cache_manager,
        &vk_ctx,
    )
    .await?;
    progress.send((10.0 / TOTAL_STEPS, "Loading audio files...".to_string()))?;
    log::info!("Loading audio files...");
    let audio = Arc::new(
        audio::start_engine(
            settings.clone(),
            timekeeper.clone(),
            &{
                let mut all = Vec::new();
                let mut pagination_token = 0u64;
                loop {
                    let resp = connection
                        .get_audio_defs(
                            GetAudioDefsRequest { pagination_token }
                                .into_request()
                                .with_version_headers(),
                        )
                        .await?
                        .into_inner();
                    all.extend(resp.sampled_sounds);
                    pagination_token = resp.next_pagination_token;
                    if pagination_token == 0 {
                        break;
                    }
                }
                log::info!("{} audio defs loaded from server", all.len());
                all
            },
            &mut cache_manager,
        )
        .await
        .context("Failed to start audio engine")?,
    );

    let far_geometry = FarGeometryState::new(block_types.clone());

    let (action_sender, action_receiver) = mpsc::channel(4);

    let client_state = Arc::new(ClientState::new(
        settings,
        // Do not allow cancellation within the client state itself (e.g. from a crash) to cancel
        // the token representing the user's attempt to make a connection
        cancel.child_token(),
        block_types,
        chunks,
        items,
        action_sender,
        hud,
        egui,
        block_renderer,
        entitity_renderer,
        far_geometry,
        timekeeper,
        audio,
    )?);
    tx_send
        .send(StreamToServer {
            sequence: 0,
            client_tick: 0,
            client_message: Some(ClientMessage::ClientInitialReady(rpc::Nop {})),
            want_performance_metrics: false,
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
    spawn_and_monitor_loop("inbound", client_state.clone(), async move {
        inbound.run_inbound_loop().await
    });

    progress.send((
        11.0 / TOTAL_STEPS,
        "Waiting for initial game state...".to_string(),
    ))?;
    initial_state_notification.notified().await;
    spawn_and_monitor_loop("outbound", client_state.clone(), async move {
        outbound.run_outbound_loop().await
    });

    progress.send((12.0 / TOTAL_STEPS, "Connected!".to_string()))?;
    // Sleep for a moment so the user can see the connected message
    tokio::time::sleep(Duration::from_secs_f64(0.25)).await;
    Ok(client_state)
}

fn spawn_and_monitor_loop(
    description: &'static str,
    client_state: Arc<ClientState>,
    future: impl std::future::Future<Output = Result<()>> + Send + 'static,
) {
    let client_state_clone = client_state.clone();
    tokio::spawn(async move {
        let handle = tokio::spawn(future);
        match handle.await {
            Ok(Ok(_)) => log::info!("{description} loop shut down normally"),
            Ok(Err(e)) => {
                log::error!("{description} loop crashed: {e:?}");
                *client_state_clone.pending_error.lock() = Some(e);
            }
            Err(e) => {
                if e.is_panic() {
                    log::error!("{description} loop panicked: {e:?}");
                    *client_state_clone.pending_error.lock() = Some(anyhow!(
                        "{description} loop panicked; please check the log for further details"
                    ));
                } else {
                    log::error!("{description} loop disconnected: {e:?}");
                    *client_state_clone.pending_error.lock() =
                        Some(anyhow!("{description} loop unexpectedly cancelled"));
                }
            }
        }
    });
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
    let client_state = opaque4::ClientRegistration::<PerovskiteOpaqueAuth>::start(
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
        want_performance_metrics: false,
    })
    .await?;
    let registration_response = match rx.message().await? {
        Some(StreamToClient {
            server_message: Some(ServerMessage::ServerRegistrationResponse(resp)),
            ..
        }) => opaque4::RegistrationResponse::deserialize(&resp).map_err(|e| {
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
            opaque4::ClientRegistrationFinishParameters::default(),
        )
        .map_err(|e| Error::msg(format!("OPAQUE ClientRegistration finish failed: {e:?}")))?;
    tx.send(StreamToServer {
        sequence: 0,
        client_tick: 0,
        client_message: Some(ClientMessage::ClientRegistrationUpload(
            finish_registration_result.message.serialize().to_vec(),
        )),
        want_performance_metrics: false,
    })
    .await?;
    match rx.message().await? {
        Some(StreamToClient {
            server_message: Some(ServerMessage::AuthSuccess(success)),
            tick,
            ..
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
        opaque4::ClientLogin::<PerovskiteOpaqueAuth>::start(&mut client_rng, password.as_bytes())
            .map_err(|e| Error::msg(format!("OPAQUE ClientLogin start failed: {e:?}")))?;
    tx.send(StreamToServer {
        sequence: 0,
        client_tick: 0,
        client_message: Some(ClientMessage::StartAuthentication(StartAuth {
            username: username.clone(),
            register: false,
            opaque_request: client_state.message.serialize().to_vec(),
            min_protocol_version: MIN_PROTOCOL_VERSION,
            max_protocol_version: MAX_PROTOCOL_VERSION,
        })),
        want_performance_metrics: false,
    })
    .await?;
    let registration_response = match rx.message().await? {
        Some(StreamToClient {
            server_message:
                Some(ServerMessage::ServerLoginResponse(ServerLoginResponse {
                    login_result: Some(server_login_response::LoginResult::RawOpaque4Response(resp)),
                })),
            ..
        }) => opaque4::CredentialResponse::deserialize(&resp).map_err(|e| {
            Error::msg(format!(
                "OPAQUE login CredentialResponse couldn't be decoded: {e:?}"
            ))
        })?,
        Some(StreamToClient {
            server_message:
                Some(ServerMessage::ServerLoginResponse(ServerLoginResponse {
                    login_result: Some(server_login_response::LoginResult::DoLegacyOpaque2(true)),
                })),
            ..
        }) => return legacy_opaque2::do_legacy_login_handshake(tx, rx, username, password).await,
        Some(x) => {
            bail!("Server sent an unexpected message in response to the login request: {x:?}")
        }
        None => bail!("Server disconnected before finishing login"),
    };
    let finish_login_result = client_state
        .state
        .finish(
            &mut client_rng,
            password.as_bytes(),
            registration_response,
            opaque4::ClientLoginFinishParameters::default(),
        )
        .map_err(|e| Error::msg(format!("OPAQUE ClientLogin finish failed: {e:?}")))?;
    tx.send(StreamToServer {
        sequence: 0,
        client_tick: 0,
        client_message: Some(ClientMessage::ClientLoginCredential(
            finish_login_result.message.serialize().to_vec(),
        )),
        want_performance_metrics: false,
    })
    .await?;

    match rx.message().await? {
        Some(StreamToClient {
            server_message: Some(ServerMessage::AuthSuccess(success)),
            tick,
            ..
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

mod legacy_opaque2 {
    use perovskite_core::{
        auth::LegacyPerovskiteOpaqueAuth,
        protocol::game_rpc::{
            server_login_response, stream_to_client::ServerMessage,
            stream_to_server::ClientMessage, ServerLoginResponse, StreamToClient, StreamToServer,
        },
    };

    use crate::net_client::AuthSuccess;
    use anyhow::{bail, Error, Result};
    use rand::rngs::OsRng;
    use tokio::sync::mpsc;
    use tonic::Streaming;

    pub(super) async fn do_legacy_login_handshake(
        tx: &mpsc::Sender<StreamToServer>,
        rx: &mut Streaming<StreamToClient>,
        username: String,
        password: String,
    ) -> Result<AuthSuccess> {
        log::info!("Starting legacy OPAQUE2 login for user {}", username);
        let mut client_rng = OsRng;
        let client_state = opaque2::ClientLogin::<LegacyPerovskiteOpaqueAuth>::start(
            &mut client_rng,
            password.as_bytes(),
        )
        .map_err(|e| Error::msg(format!("OPAQUE ClientLogin start failed: {e:?}")))?;
        tx.send(StreamToServer {
            sequence: 0,
            client_tick: 0,
            client_message: Some(ClientMessage::LegacyOpaque2InitialRequest(
                client_state.message.serialize().to_vec(),
            )),
            want_performance_metrics: false,
        })
        .await?;
        let registration_response = match rx.message().await? {
            Some(StreamToClient {
                server_message:
                    Some(ServerMessage::ServerLoginResponse(ServerLoginResponse {
                        login_result:
                            Some(server_login_response::LoginResult::RawOpaque2Response(resp)),
                    })),
                ..
            }) => opaque2::CredentialResponse::deserialize(&resp).map_err(|e| {
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
                opaque2::ClientLoginFinishParameters::default(),
            )
            .map_err(|e| Error::msg(format!("OPAQUE ClientLogin finish failed: {e:?}")))?;
        tx.send(StreamToServer {
            sequence: 0,
            client_tick: 0,
            client_message: Some(ClientMessage::ClientLoginCredential(
                finish_login_result.message.serialize().to_vec(),
            )),
            want_performance_metrics: false,
        })
        .await?;

        match rx.message().await? {
            Some(StreamToClient {
                server_message: Some(ServerMessage::AuthSuccess(success)),
                tick,
                ..
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
}
