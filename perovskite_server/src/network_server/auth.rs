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

use std::{sync::Arc, u8};

use opaque_ke::{
    CredentialFinalization, CredentialRequest, RegistrationRequest, RegistrationUpload,
    ServerLogin, ServerLoginStartParameters, ServerLoginStartResult, ServerRegistration,
    ServerSetup,
};
use perovskite_core::{
    auth::PerovskiteOpaqueAuth,
    protocol::game_rpc::{
        stream_to_client::ServerMessage, stream_to_server::ClientMessage, Nop, StreamToClient,
        StreamToServer,
    },
};
use rand::rngs::OsRng;
use tokio::sync::mpsc;

use crate::database::database_engine::{GameDatabase, KeySpace};

fn db_key_from_username(username: &str) -> Vec<u8> {
    let mut key_builder = Vec::new();
    key_builder.append(&mut b"user_auth_opaque_".to_vec());
    key_builder.append(&mut hex::encode(username).as_bytes().to_vec());
    KeySpace::UserMeta.make_key(&key_builder)
}

pub(crate) struct AuthOutcome {
    pub username: String,
    pub min_protocol_version: u32,
    pub max_protocol_version: u32,
}

pub struct AuthService {
    db: Arc<dyn GameDatabase>,
    server_setup: ServerSetup<PerovskiteOpaqueAuth>,
}
impl AuthService {
    pub(crate) fn create(db: Arc<dyn GameDatabase>) -> anyhow::Result<AuthService> {
        const DB_KEY: &[u8] = b"auth_opaque_serversetup";
        let db_key = &KeySpace::Metadata.make_key(DB_KEY);
        let server_setup = match db.get(db_key)? {
            Some(x) => ServerSetup::<PerovskiteOpaqueAuth>::deserialize(&x)
                .map_err(|e| anyhow::Error::msg(format!("OPAQUE ServerSetup error: {e:?}")))?,
            None => {
                let mut rng = OsRng;
                let server_setup = ServerSetup::new(&mut rng);
                db.put(db_key, &server_setup.serialize())?;
                server_setup
            }
        };
        Ok(AuthService {
            db: db.clone(),
            server_setup,
        })
    }

    fn start_registration(&self, username: &str, client_request: &[u8]) -> tonic::Result<Vec<u8>> {
        let server_registration_start_result = ServerRegistration::<PerovskiteOpaqueAuth>::start(
            &self.server_setup,
            RegistrationRequest::deserialize(client_request).map_err(|e| {
                log::error!("OPAQUE start_registration parse error: {e:?}");
                tonic::Status::invalid_argument(
                    "OPAQUE start_registration message could not be parsed",
                )
            })?,
            username.as_bytes(),
        )
        .map_err(|e| {
            log::error!("OPAQUE start_registration error: {e:?}");
            tonic::Status::unauthenticated("OPAQUE start_registration failed")
        })?;
        Ok(server_registration_start_result
            .message
            .serialize()
            .to_vec())
    }

    fn finish_registration(&self, username: &str, client_upload: &[u8]) -> tonic::Result<()> {
        let server_registration_finish_result = ServerRegistration::<PerovskiteOpaqueAuth>::finish(
            RegistrationUpload::deserialize(client_upload).map_err(|e| {
                log::error!("OPAQUE finish_registration parse error: {e:?}");
                tonic::Status::invalid_argument(
                    "OPAQUE finish_registration message could not be parsed",
                )
            })?,
        );
        let db_key = &db_key_from_username(username);
        let user_registration = self.db.get(db_key).map_err(|e| {
            log::error!("Internal DB lookup error: {e:?}");
            tonic::Status::internal("Internal finish_registration error")
        })?;
        if user_registration.is_some() {
            return Err(tonic::Status::already_exists(
                "This username is already taken.",
            ));
        }
        self.db
            .put(db_key, &server_registration_finish_result.serialize())
            .map_err(|e| {
                log::error!("Internal DB store error: {e:?}");
                tonic::Status::internal("Internal finish_registration error")
            })?;
        Ok(())
    }

    fn start_login(
        &self,
        username: &str,
        client_request: &[u8],
    ) -> tonic::Result<ServerLoginStartResult<PerovskiteOpaqueAuth>> {
        let pw_file = self.db.get(&db_key_from_username(username)).map_err(|e| {
            log::error!("Internal DB lookup error: {e:?}");
            tonic::Status::internal("Internal start_login error")
        })?;
        let pw_file = match pw_file {
            Some(x) => x,
            None => {
                return Err(tonic::Status::unauthenticated(
                    "No such user found; please register",
                ));
            }
        };
        let pw_file = ServerRegistration::deserialize(&pw_file).map_err(|e| {
            log::error!("OPAQUE start_login parse error: {e:?}");
            tonic::Status::invalid_argument(
                "OPAQUE start_login data corrupted; please contact server admin",
            )
        })?;

        let mut rng = OsRng;
        ServerLogin::start(
            &mut rng,
            &self.server_setup,
            Some(pw_file),
            CredentialRequest::deserialize(client_request).map_err(|e| {
                log::error!("OPAQUE start_login parse error: {e:?}");
                tonic::Status::invalid_argument("OPAQUE start_login message could not be parsed")
            })?,
            username.as_bytes(),
            ServerLoginStartParameters::default(),
        )
        .map_err(|e| {
            log::error!("OPAQUE start_login step failed: {e:?}");
            tonic::Status::unauthenticated("OPAQUE start_login failed")
        })
    }

    fn finish_login(
        &self,
        prior_phase: ServerLoginStartResult<PerovskiteOpaqueAuth>,
        client_login_credential: &[u8],
    ) -> tonic::Result<()> {
        let result = prior_phase.state.finish(
            CredentialFinalization::deserialize(client_login_credential).map_err(|e| {
                log::error!("OPAQUE finish_login parse error: {e:?}");
                tonic::Status::invalid_argument("OPAQUE finish_login message could not be parsed")
            })?,
        );
        match result {
            Ok(_) => Ok(()),
            Err(opaque_ke::errors::ProtocolError::InvalidLoginError) => {
                Err(tonic::Status::unauthenticated("Invalid password"))
            }
            e @ Err(_) => {
                log::error!("OPAQUE finish_login step failed: {e:?}");
                Err(tonic::Status::unauthenticated("OPAQUE finish_login failed"))
            }
        }
    }

    pub(crate) async fn do_auth_flow(
        &self,
        inbound: &mut tonic::Streaming<StreamToServer>,
        outbound: &mpsc::Sender<tonic::Result<StreamToClient>>,
    ) -> tonic::Result<AuthOutcome> {
        match inbound.message().await {
            Ok(Some(StreamToServer {
                client_message: Some(ClientMessage::StartAuthentication(req)),
                ..
            })) => {
                let username = validate_username(&req.username)?;
                if req.register {
                    self.do_registration_flow(&req.opaque_request, &username, inbound, outbound)
                        .await?;
                    Ok(AuthOutcome {
                        username,
                        min_protocol_version: req.min_protocol_version,
                        max_protocol_version: req.max_protocol_version,
                    })
                } else {
                    self.do_login_flow(&req.opaque_request, &username, inbound, outbound)
                        .await?;
                    Ok(AuthOutcome {
                        username,
                        min_protocol_version: req.min_protocol_version,
                        max_protocol_version: req.max_protocol_version,
                    })
                }
            }
            Ok(Some(_)) => Err(tonic::Status::unauthenticated(
                "Client's first message wasn't StartAuthentication",
            )),
            Ok(None) => Err(tonic::Status::unauthenticated(
                "Client disconnected before authenticating",
            )),
            Err(e) => Err(e),
        }
    }

    async fn do_registration_flow(
        &self,
        opaque_request: &[u8],
        username: &str,
        inbound: &mut tonic::Streaming<StreamToServer>,
        outbound: &mpsc::Sender<Result<StreamToClient, tonic::Status>>,
    ) -> tonic::Result<()> {
        log::info!("Starting registration for {}", username);
        outbound
            .send(Ok(StreamToClient {
                tick: 0,
                server_message: Some(ServerMessage::ServerRegistrationResponse(
                    self.start_registration(username, opaque_request)?,
                )),
            }))
            .await
            .map_err(|_| tonic::Status::unavailable("Error sending error to client"))?;

        let response = inbound.message().await.map_err(|_| {
            tonic::Status::unavailable("Error reading registration response from client")
        })?;
        match response {
            Some(StreamToServer {
                client_message: Some(ClientMessage::ClientRegistrationUpload(data)),
                ..
            }) => {
                self.finish_registration(username, &data)?;
                Ok(())
            }
            Some(_) => Err(tonic::Status::unauthenticated(
                "Client did not send a registration response",
            )),
            None => Err(tonic::Status::unauthenticated(
                "Client disconnected before finishing registration",
            )),
        }
    }

    async fn do_login_flow(
        &self,
        opaque_request: &[u8],
        username: &str,
        inbound: &mut tonic::Streaming<StreamToServer>,
        outbound: &mpsc::Sender<Result<StreamToClient, tonic::Status>>,
    ) -> tonic::Result<()> {
        log::info!("Starting login for {}", username);
        let login_state = self.start_login(username, opaque_request)?;
        outbound
            .send(Ok(StreamToClient {
                tick: 0,
                server_message: Some(ServerMessage::ServerLoginResponse(
                    login_state.message.serialize().to_vec(),
                )),
            }))
            .await
            .map_err(|_| tonic::Status::unavailable("Error sending error to client"))?;

        let response = inbound
            .message()
            .await
            .map_err(|_| tonic::Status::unavailable("Error reading login response from client"))?;
        match response {
            Some(StreamToServer {
                client_message: Some(ClientMessage::ClientLoginCredential(data)),
                ..
            }) => {
                self.finish_login(login_state, &data)?;
                Ok(())
            }
            Some(_) => Err(tonic::Status::unauthenticated(
                "Client did not send a login credential",
            )),
            None => Err(tonic::Status::unauthenticated(
                "Client disconnected before finishing registration",
            )),
        }
    }
}

fn validate_username(username: &str) -> tonic::Result<String> {
    let trimmed = username.trim();
    if trimmed.is_empty() {
        return Err(tonic::Status::invalid_argument(
            "Username must not be blank",
        ));
    };
    // Limits to make it easier to identify players
    // There's no technical limitation requiring these, but usernames that don't follow these rules
    // are error-prone and hard to type (including for server admins)
    if trimmed.len() > 16 {
        return Err(tonic::Status::invalid_argument(
            "Username is too long (max 16 characters)",
        ));
    }
    if !trimmed
        .chars()
        .all(|x| x.is_ascii_alphanumeric() || x == '_' || x == '.')
    {
        return Err(tonic::Status::invalid_argument(
            "Username must only contain letters, numbers, underscores, and dots",
        ));
    }
    if trimmed.contains("._")
        || trimmed.contains("|.")
        || trimmed.contains("..")
        || trimmed.contains("__")
    {
        return Err(tonic::Status::invalid_argument(
            "Username can't contain consecutive dots or underscores",
        ));
    }
    Ok(trimmed.to_string())
}
