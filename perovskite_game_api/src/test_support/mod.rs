use anyhow::bail;
use googletest::prelude::*;
use parking_lot::Mutex;
use perovskite_core::{block_id::BlockId, coordinates::BlockCoordinate};
use rand::RngCore;
use std::{path::PathBuf, sync::Arc};

use crate::game_builder::GameBuilder;
use perovskite_server::{
    database::InMemGameDatabase,
    game_state::{mapgen::MapgenInterface, GameState},
    server::{Server, ServerArgs, ServerBuilder},
};

macro_rules! assert_ok_and_assign {
    (let $var:ident = $result:expr) => {
        #[allow(unused)]
        let $var = match $result {
            Ok(v) => v,
            Err(e) => return googletest::fail!("{:?}", e),
        };
    };
    (let $var:ident : $ty:ty = $result:expr) => {
        #[allow(unused)]
        let $var: $ty = match $result {
            Ok(v) => v,
            Err(e) => return googletest::fail!("{:?}", e),
        };
    };
}

/// A static, global world used during testing. This is private to the framework, but
/// some helpers are provided to access its contents in controlled ways.
struct TestFixtureBacking {
    database: Arc<InMemGameDatabase>,
    temp_dir: Option<PathBuf>,
    server: Option<Arc<Server>>,
}

thread_local! {
    static CURRENT_FIXTURE: Mutex<Option<Arc<Mutex<TestFixtureBacking>>>> = Mutex::new(None);
}
static LOG_INIT: std::sync::Once = std::sync::Once::new();

/// A test fixture that provides a global in-memory world for testing.
///
/// The server starts out stopped; it must be started with the game content under test
/// using [TestFixture::start_server] before it can be used.
pub struct TestFixture;
impl googletest::fixtures::Fixture for TestFixture {
    fn set_up() -> googletest::Result<Self> {
        LOG_INIT.call_once(|| {
            const DEFAULT_LOG_FILTER: &str = "info,perovskite_server=warn,perovskite_game_api=warn";

            let env_value =
                std::env::var("RUST_LOG").unwrap_or_else(|_| DEFAULT_LOG_FILTER.to_string());

            let env_value = if env_value.is_empty() {
                DEFAULT_LOG_FILTER.to_string()
            } else {
                env_value
            };

            tracing_subscriber::fmt()
                .with_env_filter(tracing_subscriber::EnvFilter::try_new(&env_value).unwrap())
                .init();
        });
        CURRENT_FIXTURE.with(|f| {
            let mut fixture = f.lock();
            match fixture.as_mut() {
                Some(_) => {
                    eprintln!("Existing test fixture not properly torn down; this is unexpected");
                }
                None => {
                    let temp_dir = std::env::temp_dir()
                        .join(format!("perovskite-{}", rand::thread_rng().next_u64()));
                    *fixture = Some(Arc::new(Mutex::new(TestFixtureBacking {
                        database: Arc::new(InMemGameDatabase::new()),
                        temp_dir: Some(temp_dir),
                        server: None,
                    })));
                }
            }
        });
        Ok(Self)
    }

    fn tear_down(self) -> googletest::Result<()> {
        CURRENT_FIXTURE.with(|f| {
            let mut fixture = f.lock();
            match fixture.take() {
                Some(backing) => {
                    drop(backing);
                }
                None => {
                    panic!("No test fixture found");
                }
            }
        });
        Ok(())
    }
}
impl TestFixture {
    fn inner() -> Arc<Mutex<TestFixtureBacking>> {
        CURRENT_FIXTURE.with(|f| {
            f.lock()
                .clone()
                .expect("Missing fixture; was set_up called?")
        })
    }

    /// Tries to start the server with the given game content.
    ///
    /// If the server is already running, this will return an error.
    /// If the server fails to start, this will return an error.
    ///
    /// By default, this will create a mapgen that fills the entire world with air; this
    /// can be overridden using [GameBuilderTestExt::set_flatland_mapgen] (or other mapgen
    /// helpers if added later). Note that the mapgen API itself is unstable, so `set_flatland_mapgen`
    /// is offered as a stable test-specific API.
    pub fn start_server(
        &self,
        content_init: impl FnOnce(&mut GameBuilder),
    ) -> googletest::Result<()> {
        let fixture = TestFixture::inner();
        let mut fixture = fixture.lock();
        match fixture.server {
            Some(_) => {
                fail!("Server is already running")
            }
            None => {
                let builder = ServerBuilder::from_args_and_db(
                    &ServerArgs {
                        data_dir: fixture.temp_dir.clone().unwrap(),
                        bind_addr: None,
                        port: 0,
                        trace_rate_denominator: usize::MAX,
                        rocksdb_num_fds: 512,
                        rocksdb_point_lookup_cache_mib: 128,
                        num_map_prefetchers: 8,
                    },
                    fixture.database.clone(),
                )
                .expect("Failed to create server builder");
                let mut game_builder = GameBuilder::from_serverbuilder(builder)
                    .expect("Failed to create game builder");
                game_builder.set_flatland_mapgen(BlockId::AIR);
                content_init(&mut game_builder);
                let server = Arc::new(game_builder.into_server().expect("Failed to create server"));
                fixture.server = Some(server.clone());
                Ok(())
            }
        }
    }

    pub fn stop_server(&self) -> anyhow::Result<()> {
        let fixture = TestFixture::inner();
        let mut fixture = fixture.lock();
        match fixture.server.take() {
            Some(server) => {
                drop(server);
                tracing::info!("Server stopped");
                Ok(())
            }
            None => {
                bail!("Server is not running");
            }
        }
    }

    pub fn run_assertions_in_server(
        &self,
        task: impl FnOnce(&GameState) -> googletest::Result<()>,
    ) -> googletest::Result<()> {
        let fixture = TestFixture::inner();
        let result = fixture
            .lock()
            .server
            .as_ref()
            .expect("Server not running")
            .run_task_in_server(task);
        result
    }
}

const ZERO_COORD: BlockCoordinate = BlockCoordinate::new(0, 0, 0);

/// A simple map generator that can be used for testing.
///
/// Negative Y values are filled with the given block ID, positive and zero Y values are filled with air.
pub struct FlatlandMapgen {
    pub block_id: BlockId,
}
impl MapgenInterface for FlatlandMapgen {
    fn fill_chunk(
        &self,
        coord: perovskite_core::coordinates::ChunkCoordinate,
        chunk: &mut perovskite_server::game_state::game_map::MapChunk,
    ) {
        let id = if coord.y < 0 {
            self.block_id
        } else {
            BlockId::AIR
        };
        chunk.fill(id);
    }

    fn far_mesh_estimate(
        &self,
        _x: f64,
        _z: f64,
    ) -> perovskite_server::game_state::mapgen::FarMeshPoint {
        perovskite_server::game_state::mapgen::FarMeshPoint {
            height: 0.0,
            block_type: self.block_id,
            water_height: 0.0,
        }
    }
}

pub trait GameBuilderTestExt {
    fn set_flatland_mapgen(&mut self, block_id: BlockId) -> &mut Self;
}
impl GameBuilderTestExt for GameBuilder {
    fn set_flatland_mapgen(&mut self, block_id: BlockId) -> &mut Self {
        self.inner
            .set_mapgen(move |_, _| Arc::new(FlatlandMapgen { block_id }));
        self
    }
}

#[cfg(test)]
#[gtest]
fn sample_test(fixture: &TestFixture) -> googletest::Result<()> {
    assert_ok_and_assign!(let server = fixture.start_server(|builder| {
        builder.set_flatland_mapgen(BlockId(4096));
    }));
    fixture.run_assertions_in_server(|gs| {
        use googletest::expect_that;

        // verify that the mapgen did its job
        expect_that!(gs.game_map().get_block(ZERO_COORD), ok(eq(&BlockId(0))));
        expect_that!(
            gs.game_map().get_block(BlockCoordinate::new(0, -1, 0)),
            ok(eq(&BlockId(4096)))
        );

        gs.game_map()
            .set_block(ZERO_COORD, BlockId(0), None)
            .expect("set_block");
        expect_that!(gs.game_map().get_block(ZERO_COORD), ok(eq(&BlockId(0))));
        gs.game_map()
            .set_block(ZERO_COORD, BlockId(1), None)
            .expect("set_block");
        expect_that!(gs.game_map().get_block(ZERO_COORD), ok(eq(&BlockId(1))));
        std::thread::sleep(std::time::Duration::from_secs(1));
        Ok(())
    })?;
    Ok(())
}
