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
        blocks::BlockTypeManager, game_behaviors::GameBehaviors, items::ItemManager,
        mapgen::MapgenInterface, GameState, game_map::{TimerSettings, TimerCallback},
    },
    media::MediaManager,
    network_server::{grpc_service::CuberefGameServerImpl},
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
    bind_address: SocketAddr,
}
impl Server {
    fn new(
        runtime: tokio::runtime::Runtime,
        game_state: Arc<GameState>,
        bind_address: SocketAddr,
    ) -> Result<Server> {
        Ok(Server {
            runtime,
            game_state,
            bind_address,
        })
    }

    pub fn game_state(&self) -> &GameState {
        self.game_state.as_ref()
    }

    /// Starts the network server, and blocks until the game
    /// is shut down with Ctrl+C or start_shutdown is called on the game state.
    pub fn serve(&self) -> Result<()> {
        let _tracy_client = tracy_client::Client::start();
        self.runtime.block_on(self.serve_async())
    }

    async fn serve_async(&self) -> Result<()> {
        #[cfg(feature = "deadlock_detection")]
        {
            use parking_lot::deadlock;
            use std::thread;
            use std::time::Duration;

            thread::spawn(move || loop {
                thread::sleep(Duration::from_secs(10));
                let deadlocks = deadlock::check_deadlock();
                if deadlocks.is_empty() {
                    continue;
                }

                println!("{} deadlocks detected", deadlocks.len());
                for (i, threads) in deadlocks.iter().enumerate() {
                    println!("Deadlock #{}", i);
                    for t in threads {
                        println!("Thread Id {:#?}", t.thread_id());
                        println!("{:#?}", t.backtrace());
                    }
                }
            });
        }

        let cuberef_service = CuberefGameServer::new(CuberefGameServerImpl::new(
            self.game_state.clone(),
        ));
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
    map_timers: Vec<(String, TimerSettings, TimerCallback)>,
    args: ServerArgs,
    game_behaviors: GameBehaviors,
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
        let items = ItemManager::new()?;
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
            map_timers: Vec::new(),
            args: args.clone(),
            game_behaviors: Default::default(),
        })
    }
    pub fn blocks_mut(&mut self) -> &mut BlockTypeManager {
        &mut self.blocks
    }
    pub fn blocks(&self) -> &BlockTypeManager {
        &self.blocks
    }
    pub fn items_mut(&mut self) -> &mut ItemManager {
        &mut self.items
    }
    pub fn items(&self) -> &ItemManager {
        &self.items
    }
    pub fn media_mut(&mut self) -> &mut MediaManager {
        &mut self.media
    }
    pub fn media(&self) -> &MediaManager {
        &self.media
    }
    pub fn add_timer(&mut self, name: impl Into<String>, settings: TimerSettings, callback: TimerCallback) {
        self.map_timers.push((name.into(), settings, callback));
    }
    /// Sets the mapgen for this game.
    /// Stability note: The mapgen API is a WIP, and has not been stabilized yet.
    pub fn set_mapgen<F>(&mut self, mapgen: F)
    where
        F: (FnOnce(Arc<BlockTypeManager>, u32) -> Arc<dyn MapgenInterface>) + 'static,
    {
        self.mapgen = Some(Box::new(mapgen))
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
        let game_state = 
        GameState::new(
            self.db,
            blocks,
            self.items,
            self.media,
            self.mapgen.with_context(|| "Mapgen not specified")?,
            self.game_behaviors,
        )?;
        for (name, settings, callback) in self.map_timers {
            game_state.map().register_timer(name, settings, callback)?;
        }
        Server::new(
            self.runtime,
            game_state,
            addr,
        )
    }

    pub fn game_behaviors_mut(&mut self) -> &mut GameBehaviors {
        &mut self.game_behaviors
    }
}
