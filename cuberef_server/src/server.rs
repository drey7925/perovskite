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
    net::{IpAddr, SocketAddr},
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
};

use anyhow::{bail, Context, Result};
use clap::Parser;
use cuberef_core::protocol::game_rpc::cuberef_game_server::CuberefGameServer;
use tonic::codegen::CompressionEncoding;

use crate::{
    database::{database_engine::GameDatabase, rocksdb::RocksDbBackend},
    game_state::{
        blocks::BlockTypeManager, items::ItemManager, mapgen::MapgenInterface, testutils::FakeAuth,
        GameState,
    },
    media::MediaManager,
    network_server::{auth::AuthService, grpc_service::CuberefGameServerImpl},
};

#[derive(Parser, Debug, Clone)]
pub struct ServerArgs {
    /// The directory to use to store the world's data
    #[arg(short, long, value_name = "DATA_DIR")]
    data_dir: PathBuf,

    /// If true, the data dir will be created if it doesn't exist.
    #[arg(long)]
    create: bool,

    /// The interface address to bind to. By default, bind all interfaces.
    #[arg(long)]
    bind_addr: Option<IpAddr>,

    #[arg(short, long, default_value_t = 28273)]
    port: u16,
}

pub struct Server {
    runtime: tokio::runtime::Runtime,
    game_state: Arc<GameState>,
    auth: Arc<dyn AuthService>,
    bind_address: SocketAddr,
}
impl Server {
    fn new(
        runtime: tokio::runtime::Runtime,
        game_state: Arc<GameState>,
        auth: Arc<dyn AuthService>,
        bind_address: SocketAddr,
    ) -> Result<Server> {
        Ok(Server {
            runtime,
            game_state,
            auth,
            bind_address,
        })
    }

    pub fn game_state(&self) -> &GameState {
        self.game_state.as_ref()
    }

    /// Starts the network server, and blocks until the game
    /// is shut down with Ctrl+C or start_shutdown is called on the game state.
    pub fn serve(&self) -> Result<()> {
        self.runtime.block_on(self.serve_async())
    }

    async fn serve_async(&self) -> Result<()> {
        let cuberef_service = CuberefGameServer::new(CuberefGameServerImpl::new(
            self.game_state.clone(),
            self.auth.clone(),
        ))
        .accept_compressed(CompressionEncoding::Gzip)
        .send_compressed(CompressionEncoding::Gzip);
        let reflection_service = tonic_reflection::server::Builder::configure()
            .register_encoded_file_descriptor_set(cuberef_core::protocol::DESCRIPTOR_SET)
            .build()
            .unwrap();
        tonic::transport::Server::builder()
        .add_service(cuberef_service)
        .add_service(reflection_service)
        .serve_with_shutdown(self.bind_address, async {
            tokio::select! {
                result = tokio::signal::ctrl_c() => {
                    result.unwrap();
                    log::info!("Ctrl+C received, shutting down network service");
                },
                _ = self.game_state().await_start_shutdown() => {
                    log::info!("Game shutdown requested programmatically; shutting down network serivce");
                }
            }
            self.game_state.start_shutdown();
        }).await.with_context(||"Tonic server error")
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.runtime.block_on(self.game_state.finish_shutdown());
    }
}

pub struct ServerBuilder {
    runtime: tokio::runtime::Runtime,
    db: Arc<dyn GameDatabase>,
    blocks: BlockTypeManager,
    items: ItemManager,
    mapgen: Option<Box<dyn FnOnce(Arc<BlockTypeManager>, u32) -> Arc<dyn MapgenInterface>>>,
    media: MediaManager,
    auth: Arc<dyn AuthService>,
    args: ServerArgs,
}
impl ServerBuilder {
    pub fn from_cmdline() -> Result<ServerBuilder> {
        Self::from_args(&ServerArgs::parse())
    }
    pub fn from_args(args: &ServerArgs) -> Result<ServerBuilder> {
        if !Path::exists(&args.data_dir) {
            std::fs::create_dir(&args.data_dir)?;
            log::info!("Created new data directory at {:?}", args.data_dir);
        } else {
            if !Path::is_dir(&args.data_dir) {
                bail!("Specified data directory is not a directory.");
            }
            log::info!("Loaded existing data directory at {:?}", args.data_dir);
        }
        
        let mut db_dir = args.data_dir.clone();
        db_dir.push("database");
        let db = Arc::new(RocksDbBackend::new(db_dir)?);

        let blocks = BlockTypeManager::create_or_load(db.as_ref(), true)?;
        let items = ItemManager::new();
        let media = MediaManager::new();
        Ok(ServerBuilder {
            runtime: tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?,
            db,
            blocks,
            items,
            mapgen: None,
            media,
            // TODO real auth
            auth: Arc::new(FakeAuth {}),
            args: args.clone(),
        })
    }
    pub fn blocks(&mut self) -> &mut BlockTypeManager {
        &mut self.blocks
    }
    pub fn items(&mut self) -> &mut ItemManager {
        &mut self.items
    }
    pub fn media(&mut self) -> &mut MediaManager {
        &mut self.media
    }
    /// Sets the mapgen for this game.
    /// Stability note: The mapgen API is a WIP, and has not been stabilized yet.
    pub fn set_mapgen<F>(&mut self, mapgen: F)
    where
        F: (FnOnce(Arc<BlockTypeManager>, u32) -> Arc<dyn MapgenInterface>) + 'static,
    {
        self.mapgen = Some(Box::new(mapgen))
    }
    pub fn set_auth(&mut self, auth: Arc<dyn AuthService>) {
        self.auth = auth;
    }

    pub fn build(self) -> Result<Server> {
        let addr = SocketAddr::new(
            // Bind to all interfaces (v4 and v6) by default
            self.args.bind_addr.unwrap_or(IpAddr::from_str("::")?),
            self.args.port,
        );

        let blocks = Arc::new(self.blocks);
        blocks.save_to(self.db.as_ref())?;
        let _rt_guard = self.runtime.enter();
        Server::new(
            self.runtime,
            GameState::new(
                self.db,
                blocks,
                self.items,
                self.media,
                self.mapgen.with_context(|| "Mapgen not specified")?,
            )?,
            self.auth,
            addr,
        )
    }
}
