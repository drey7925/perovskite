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

use std::env::temp_dir;
use std::fmt::Debug;
use std::{
    net::{IpAddr, SocketAddr},
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
};

use anyhow::{bail, Context, Result};
use clap::Parser;
use integer_encoding::VarInt;
use perovskite_core::coordinates::ChunkCoordinate;
use perovskite_core::{
    protocol::game_rpc::perovskite_game_server::PerovskiteGameServer,
    util::set_trace_rate_denominator,
};
use rocksdb::Options;
use serde::{Deserialize, Serialize};
use tonic::transport::{Identity, ServerTlsConfig};
use type_map::concurrent::TypeMap;

use crate::database::rocksdb::RocksdbOptions;
pub use crate::database::GameDatabase;
use crate::database::{InMemGameDatabase, KeySpace};
use crate::game_state::game_map::MapChunk;
use crate::{
    database::rocksdb::RocksDbBackend,
    game_state::{
        blocks::BlockTypeManager,
        chat::commands::{ChatCommand, ChatCommandHandler, CommandManager},
        entities::EntityTypeManager,
        game_behaviors::GameBehaviors,
        game_map::{TimerCallback, TimerSettings},
        items::ItemManager,
        mapgen::MapgenInterface,
        GameState, GameStateExtension,
    },
    media::MediaManager,
    network_server::grpc_service::PerovskiteGameServerImpl,
};

#[derive(Parser, Debug, Clone)]
pub struct ServerArgs {
    /// The directory to use to store the world's data
    #[arg(short, long, value_name = "DATA_DIR")]
    pub data_dir: PathBuf,

    /// The interface address to bind to. By default, bind all interfaces.
    #[arg(long)]
    pub bind_addr: Option<IpAddr>,

    /// The port to listen on. (Currently TCP, could become QUIC-over-UDP in a future version)
    #[arg(short, long, default_value_t = 28273)]
    pub port: u16,

    /// How often to print trace buffers. The higher the value, the lower the chance of printing
    /// one
    #[arg(long, default_value_t = 65536)]
    pub trace_rate_denominator: usize,

    /// Size of Rocksdb's underlying cache; passed directly to the database library.
    #[arg(long, default_value_t = 128)]
    pub rocksdb_point_lookup_cache_mib: u64,

    /// Number of file descriptors to use for the underlying game database. On Linux, the program
    /// will attempt to increase RLIMIT_NOFILE to 133% of this value; if it cannot set the limit
    /// high enough, the Rocksdb file descriptor count will automatically be lowered to 75% of the
    /// detected limit.
    #[arg(long, default_value_t = 512)]
    pub rocksdb_num_fds: std::os::raw::c_int,

    /// How many map prefetchers to use. Note that they currently don't do anything in the default
    /// game, and will sit idle.
    #[arg(long, default_value_t = 8)]
    pub num_map_prefetchers: usize,
}

pub struct Server {
    runtime: tokio::runtime::Runtime,
    game_state: Arc<GameState>,
    bind_address: SocketAddr,
    tls_config: LoadedTlsConfig,
}
impl Server {
    fn new(
        runtime: tokio::runtime::Runtime,
        game_state: Arc<GameState>,
        bind_address: SocketAddr,
        tls_config: LoadedTlsConfig,
    ) -> Result<Server> {
        Ok(Server {
            runtime,
            game_state,
            bind_address,
            tls_config,
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

                eprintln!("{} deadlocks detected", deadlocks.len());
                for (i, threads) in deadlocks.iter().enumerate() {
                    eprintln!("Deadlock #{}", i);
                    for t in threads {
                        eprintln!("Thread Id {:#?}", t.thread_id());
                        eprintln!("{:#?}", t.backtrace());
                    }
                }
            });
        }

        let perovskite_service =
            PerovskiteGameServer::new(PerovskiteGameServerImpl::new(self.game_state.clone()));
        let reflection_service = tonic_reflection::server::Builder::configure()
            .register_encoded_file_descriptor_set(perovskite_core::protocol::DESCRIPTOR_SET)
            .build_v1()?;

        let mut server_builder = tonic::transport::Server::builder();

        let shutdown_task = async {
            tokio::select! {
                result = tokio::signal::ctrl_c() => {
                    result.unwrap();
                    tracing::info!("Ctrl+C received, shutting down network service");
                },
                _ = self.game_state().await_start_shutdown() => {
                    tracing::info!("Game shutdown requested programmatically; shutting down network serivce");
                }
            }
            self.game_state.start_shutdown();
        };
        tracing::info!(
            "Serving on {:?} with TLS {:?}",
            self.bind_address,
            self.tls_config
        );
        match &self.tls_config {
            LoadedTlsConfig::NoTls => server_builder
                .add_service(perovskite_service)
                .add_service(reflection_service)
                .serve_with_shutdown(self.bind_address, shutdown_task)
                .await
                .with_context(|| "Tonic server error"),
            LoadedTlsConfig::Identity(i) => server_builder
                .tls_config(ServerTlsConfig::new().identity(i.clone()))?
                .add_service(perovskite_service)
                .add_service(reflection_service)
                .serve_with_shutdown(self.bind_address, shutdown_task)
                .await
                .with_context(|| "Tonic server error"),
        }
    }

    pub fn run_task_in_server<T>(&self, task: impl FnOnce(&GameState) -> Result<T>) -> Result<T> {
        let _enter_guard = self.runtime.enter();
        task(self.game_state())
    }
}

pub fn testonly_in_memory() -> Result<Server> {
    testonly_in_memory_with_db(Arc::new(InMemGameDatabase::new()))
}

/// A simple server, with nothing registered, for unit tests and doctests
pub fn testonly_in_memory_with_db(db: Arc<dyn GameDatabase>) -> Result<Server> {
    let mut blocks = BlockTypeManager::create_or_load(db.as_ref())?;
    let entities = EntityTypeManager::create_or_load(db.as_ref())?;
    blocks.pre_build()?;
    let blocks = Arc::new(blocks);
    let startup_counter = advance_startup_counter(db.as_ref())?;
    blocks.save_to(db.as_ref(), startup_counter)?;
    entities.pre_build()?;
    entities.save_to(db.as_ref(), startup_counter)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .max_blocking_threads(1)
        .worker_threads(1)
        .enable_all()
        .build()?;
    let _rt_guard = runtime.enter();

    let game_behaviors = GameBehaviors::dummy_game_behaviors();

    let args = ServerArgs {
        data_dir: temp_dir().join("perovskite_inmem_dummy"),
        bind_addr: None,
        port: 0,
        trace_rate_denominator: 1024,
        rocksdb_point_lookup_cache_mib: 32,
        rocksdb_num_fds: 32,
        num_map_prefetchers: 8,
    };

    let gs = GameState::new(
        args,
        db,
        blocks,
        entities,
        ItemManager::new()?,
        MediaManager::new(),
        Box::new(|_, _| Arc::new(DummyMapgen)),
        game_behaviors,
        CommandManager::new(),
        TypeMap::new(),
        0,
        // Force seed to 0 for testing
        Some(0),
    )?;
    let bind_address = SocketAddr::new(IpAddr::from_str("::").unwrap(), 0);
    Server::new(runtime, gs, bind_address, LoadedTlsConfig::NoTls)
}

/// Returns a new value each time the function is called.
fn advance_startup_counter(db: &dyn GameDatabase) -> Result<u64> {
    let key = b"startup_counter".to_vec();
    let key = KeySpace::Metadata.make_key(&key);
    match db.get(&key)? {
        Some(x) => match u64::decode_var(&x) {
            Some((val, read)) => {
                if read != x.len() {
                    tracing::warn!(
                        "Saved startup counter was {} bytes but only {} bytes were decoded (to {:?})",
                        x.len(),
                        read,
                        val
                    )
                }
                let next_counter = val + 1;
                tracing::info!(
                    "Previous startup counter {:?} advanced to {:?}",
                    val,
                    next_counter
                );
                db.put(&key, &next_counter.encode_var_vec())?;
                Ok(next_counter)
            }
            None => {
                bail!("Decoding varint for startup counter failed",);
            }
        },
        None => {
            let counter = 1;
            db.put(&key, &counter.encode_var_vec())?;
            tracing::info!("Initialized startup counter to {:?}", counter,);
            Ok(counter)
        }
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        tracing::info!("Server dropped, starting shutdown");
        match self.runtime.block_on(self.game_state.shut_down()) {
            Ok(_) => {
                tracing::info!("Server shutdown complete.");
            }
            Err(e) => {
                tracing::error!("Server shutdown was unclean: {e:?}");
            }
        };
    }
}

pub struct ServerBuilder {
    runtime: tokio::runtime::Runtime,
    db: Arc<dyn GameDatabase>,
    blocks: BlockTypeManager,
    entities: EntityTypeManager,
    items: ItemManager,
    mapgen: Option<Box<dyn FnOnce(Arc<BlockTypeManager>, u32) -> Arc<dyn MapgenInterface>>>,
    media: MediaManager,
    map_timers: Vec<(String, TimerSettings, TimerCallback)>,
    args: ServerArgs,
    game_behaviors: GameBehaviors,
    commands: CommandManager,
    extensions: type_map::concurrent::TypeMap,
    startup_actions: Vec<Box<dyn FnOnce(&Arc<GameState>) -> Result<()> + Send + Sync + 'static>>,
    force_seed: Option<u32>,
}

struct DummyMapgen;

impl MapgenInterface for DummyMapgen {
    fn fill_chunk(&self, _coord: ChunkCoordinate, _chunk: &mut MapChunk) {
        // pass
    }
}

impl ServerBuilder {
    pub fn from_cmdline() -> Result<ServerBuilder> {
        Self::from_args(&ServerArgs::parse())
    }

    pub fn from_args(args: &ServerArgs) -> Result<ServerBuilder> {
        tracing::info!(
            "Build info: {}",
            include_str!(concat!(env!("OUT_DIR"), "/build_info.txt"))
        );
        if !Path::exists(&args.data_dir) {
            std::fs::create_dir(&args.data_dir)?;
            tracing::info!("Created new data directory at {:?}", args.data_dir);
        } else {
            if !Path::is_dir(&args.data_dir) {
                bail!("Specified data directory is not a directory.");
            }
            tracing::info!("Loaded existing data directory at {:?}", args.data_dir);
        }

        let mut db_dir = args.data_dir.clone();
        db_dir.push("database");
        let mut options = RocksdbOptions::default();
        options.create_if_missing(true);
        options.optimize_for_point_lookup(args.rocksdb_point_lookup_cache_mib);
        let rocksdb_fds = set_rlimit_for_rocksdb(args.rocksdb_num_fds)?;
        tracing::info!(
            "Using up to {} open file descriptors for rocksdb",
            rocksdb_fds
        );
        options.set_max_open_files(rocksdb_fds);

        let db = Self::make_rocksdb_backend(db_dir, options)?;

        let blocks = BlockTypeManager::create_or_load(db.as_ref())?;
        let entities = EntityTypeManager::create_or_load(db.as_ref())?;
        let items = ItemManager::new()?;
        let media = MediaManager::new();
        Ok(ServerBuilder {
            runtime: tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?,
            db,
            blocks,
            entities,
            items,
            mapgen: None,
            media,
            map_timers: Vec::new(),
            args: args.clone(),
            game_behaviors: Default::default(),
            commands: CommandManager::new(),
            extensions: type_map::concurrent::TypeMap::new(),
            startup_actions: Vec::new(),
            force_seed: None,
        })
    }

    #[cfg(not(feature = "db_failure_injection"))]
    fn make_rocksdb_backend(db_dir: PathBuf, options: Options) -> Result<Arc<dyn GameDatabase>> {
        Ok(Arc::new(RocksDbBackend::new(db_dir, options)?))
    }

    pub fn force_seed(&mut self, force_seed: Option<u32>) {
        self.force_seed = force_seed
    }

    #[cfg(feature = "db_failure_injection")]
    fn make_rocksdb_backend(db_dir: PathBuf, options: Options) -> Result<Arc<dyn GameDatabase>> {
        use crate::database::failure_injection::FailureInjectedDbWrapper;
        tracing::warn!("This server is running with DB failure injection on. This is DANGEROUS and meant only for development.");
        Ok(Arc::new(FailureInjectedDbWrapper::new(
            RocksDbBackend::new(db_dir, options)?,
        )))
    }

    pub fn blocks_mut(&mut self) -> &mut BlockTypeManager {
        &mut self.blocks
    }
    pub fn blocks(&self) -> &BlockTypeManager {
        &self.blocks
    }
    pub fn entities_mut(&mut self) -> &mut EntityTypeManager {
        &mut self.entities
    }
    pub fn entities(&self) -> &EntityTypeManager {
        &self.entities
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
    pub fn add_timer(
        &mut self,
        name: impl Into<String>,
        settings: TimerSettings,
        callback: TimerCallback,
    ) {
        self.map_timers.push((name.into(), settings, callback));
    }
    pub fn register_command(
        &mut self,
        name: &str,
        command: Box<dyn ChatCommandHandler>,
        help_text: &str,
    ) -> Result<()> {
        self.commands.add_command(
            name.to_string(),
            ChatCommand::new(command, help_text.to_string()),
        )
    }
    /// Sets the mapgen for this game.
    /// Stability note: The mapgen API is a WIP, and has not been stabilized yet.
    pub fn set_mapgen<F>(&mut self, mapgen: F)
    where
        F: (FnOnce(Arc<BlockTypeManager>, u32) -> Arc<dyn MapgenInterface>) + 'static,
    {
        self.mapgen = Some(Box::new(mapgen))
    }

    pub fn build(mut self) -> Result<Server> {
        set_trace_rate_denominator(self.args.trace_rate_denominator);
        let addr = SocketAddr::new(
            // Bind to all interfaces (v4 and v6) by default
            self.args.bind_addr.unwrap_or(IpAddr::from_str("::")?),
            self.args.port,
        );
        self.blocks.pre_build()?;
        let blocks = Arc::new(self.blocks);
        let startup_counter = advance_startup_counter(self.db.as_ref())?;
        blocks.save_to(self.db.as_ref(), startup_counter)?;

        self.entities.pre_build()?;
        self.entities.save_to(self.db.as_ref(), startup_counter)?;

        let _rt_guard = self.runtime.enter();
        let game_state = GameState::new(
            self.args.clone(),
            self.db,
            blocks,
            self.entities,
            self.items,
            self.media,
            self.mapgen.with_context(|| "Mapgen not specified")?,
            self.game_behaviors,
            self.commands,
            self.extensions,
            startup_counter,
            self.force_seed,
        )?;
        for (name, settings, callback) in self.map_timers {
            game_state
                .game_map()
                .register_timer(name, settings, callback)?;
        }
        for action in self.startup_actions {
            action(&game_state)?;
        }

        let tls_config = Self::build_tls_config(self.args.data_dir.clone())?;
        Server::new(self.runtime, game_state, addr, tls_config)
    }

    pub fn game_behaviors_mut(&mut self) -> &mut GameBehaviors {
        &mut self.game_behaviors
    }

    pub fn data_dir(&self) -> &PathBuf {
        &self.args.data_dir
    }

    /// Adds an extension to the server's game state.
    /// This extension can be accessed through [GameState::extension].
    pub fn add_extension<T: GameStateExtension>(&mut self, extension: T) {
        if self.extensions.contains::<T>() {
            panic!("Extension already added");
        }
        self.extensions.insert(extension);
    }

    pub fn register_startup_action(
        &mut self,
        action: impl FnOnce(&Arc<GameState>) -> Result<()> + Send + Sync + 'static,
    ) {
        self.startup_actions.push(Box::new(action));
    }

    fn build_tls_config(data_dir: PathBuf) -> Result<LoadedTlsConfig> {
        let tls_config_file = data_dir.join("tls.ron");

        if !tls_config_file.exists() {
            log::info!(
                "{} not present; using defaults (no TLS)",
                tls_config_file.display()
            );
            return Ok(LoadedTlsConfig::NoTls);
        }
        log::info!("Loading TLS settings from {}", tls_config_file.display());
        let config = ron::from_str::<TlsSettings>(&std::fs::read_to_string(&tls_config_file)?)?;
        match config.mode {
            TlsMode::NoTls => Ok(LoadedTlsConfig::NoTls),
            TlsMode::CertificateFiles => {
                let certificate_path = config.certificate_pem_filename.as_ref().context(
                    "If TLS mode is CertificateFiles, certificate_pem_filename must be specified",
                )?;
                let key_pair_path = config.keypair_pem_filename.as_ref().context(
                    "If TLS mode is CertificateFiles, keypair_pem_filename must also be specified",
                )?;
                tracing::info!(
                "Loading TLS certificates and keys from {certificate_path:?} and {key_pair_path:?}"
            );
                let certificate_bytes = std::fs::read(certificate_path)?;
                let key_pair_bytes = std::fs::read(key_pair_path)?;
                Ok(LoadedTlsConfig::Identity(Identity::from_pem(
                    certificate_bytes,
                    key_pair_bytes,
                )))
            }
        }
    }
}

pub enum LoadedTlsConfig {
    NoTls,
    Identity(tonic::transport::Identity),
}

impl Debug for LoadedTlsConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadedTlsConfig::NoTls => f.debug_tuple("NoTls").finish(),
            // Refrain from dumping the private key
            LoadedTlsConfig::Identity(i) => f.debug_tuple("Identity(...)").field(i).finish(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum TlsMode {
    NoTls,
    CertificateFiles,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct TlsSettings {
    pub mode: TlsMode,
    pub certificate_pem_filename: Option<String>,
    pub keypair_pem_filename: Option<String>,
}

pub const FILENAME: &str = "settings.ron";

impl Default for TlsSettings {
    fn default() -> Self {
        Self {
            mode: TlsMode::NoTls,
            certificate_pem_filename: None,
            keypair_pem_filename: None,
        }
    }
}

fn set_rlimit_for_rocksdb(rocksdb_desired_fds: std::os::raw::c_int) -> Result<std::os::raw::c_int> {
    let rocksdb_desired_fds = rocksdb_desired_fds.clamp(32, 524288);
    let intended_limit = rocksdb_desired_fds as u64 * 4 / 3;
    let actual_limit = rlimit::increase_nofile_limit(intended_limit)?;
    tracing::info!(
        "File descriptor limit updated to {} (wanted at least {})",
        actual_limit,
        intended_limit
    );
    if actual_limit >= intended_limit {
        Ok(rocksdb_desired_fds)
    } else {
        let rocksdb_allowed_fds = actual_limit * 3 / 4;
        if rocksdb_allowed_fds < 64 {
            tracing::warn!(
                "The rocksdb file descriptor count is very low. DB performance may be slow."
            );
            Ok(rocksdb_allowed_fds
                .max(16)
                .try_into()
                .context("Integer overflow converting number of file descriptors")?)
        } else {
            Ok(rocksdb_allowed_fds
                .try_into()
                .context("Integer overflow converting number of file descriptors")?)
        }
    }
}
