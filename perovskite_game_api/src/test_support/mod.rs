use anyhow::bail;
use arc_swap::ArcSwapOption;
use googletest::{description::Description, matcher::MatcherResult, prelude::*};
use perovskite_core::{block_id::BlockId, coordinates::BlockCoordinate};
use rand::RngCore;
use std::{borrow::Borrow, path::PathBuf, sync::Arc};

use crate::game_builder::GameBuilder;
use perovskite_server::{
    database::InMemGameDatabase,
    game_state::{
        blocks::TryToBlockId, event::EventInitiator, items::ItemStack, mapgen::MapgenInterface,
        GameState,
    },
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
    server: ArcSwapOption<Server>,
}

thread_local! {
    static CURRENT_FIXTURE: ArcSwapOption<TestFixtureBacking> = ArcSwapOption::new(None);
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
            const DEFAULT_LOG_FILTER: &str =
                "info,perovskite_server=error,perovskite_game_api=error";

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
            let temp_dir =
                std::env::temp_dir().join(format!("perovskite-{}", rand::thread_rng().next_u64()));
            let new_fixture = Arc::new(TestFixtureBacking {
                database: Arc::new(InMemGameDatabase::new()),
                temp_dir: Some(temp_dir),
                server: ArcSwapOption::new(None),
            });
            let old = f.swap(Some(new_fixture));
            if old.is_some() {
                eprintln!("Existing test fixture not properly torn down; this is unexpected");
            }
        });
        Ok(Self)
    }

    fn tear_down(self) -> googletest::Result<()> {
        CURRENT_FIXTURE.with(|f| {
            let fixture = f.swap(None);
            match fixture {
                Some(backing) => {
                    log::warn!("Tearing down test fixture");
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
    fn inner() -> Arc<TestFixtureBacking> {
        CURRENT_FIXTURE.with(|f| {
            f.load()
                .clone()
                .expect("Missing fixture; was set_up called?")
        })
    }

    fn server() -> Arc<Server> {
        Self::inner()
            .server
            .load()
            .as_ref()
            .expect("Server not running")
            .clone()
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
        content_init: impl FnOnce(&mut GameBuilder) -> anyhow::Result<()>,
    ) -> googletest::Result<()> {
        let fixture = TestFixture::inner();
        let new_server = {
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
            let mut game_builder =
                GameBuilder::from_serverbuilder(builder).expect("Failed to create game builder");
            game_builder.set_flatland_mapgen(BlockId::AIR);
            content_init(&mut game_builder).or_fail()?;
            Arc::new(game_builder.into_server().expect("Failed to create server"))
        };
        if fixture.server.swap(Some(new_server)).is_some() {
            fail!("Server is already running")
        } else {
            Ok(())
        }
    }

    pub fn stop_server(&self) -> anyhow::Result<()> {
        let fixture = TestFixture::inner();
        let old_server = fixture.server.swap(None);
        match old_server {
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
            .server
            .load()
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

#[derive(MatcherBase, Debug)]
/// A matcher that matches a block id ignoring the variant.
struct IsBlock<T: TryToBlockId>(T);
impl<T: TryToBlockId> IsBlock<T> {
    fn describe_block(server: &Server, block_type: Option<BlockId>) -> String {
        block_type
            .map(|id| {
                server
                    .game_state()
                    .block_types()
                    .get_block(id)
                    .ok()
                    .map_or_else(
                        || format!("an unknown block with ID 0x{:x}", id.0),
                        |x| x.0.short_name().to_string(),
                    )
            })
            .unwrap_or_else(|| "an unknown block that will never match".to_string())
    }
}
impl<T: TryToBlockId, U: Borrow<BlockId> + Copy + std::fmt::Debug> Matcher<U> for IsBlock<T> {
    fn matches(&self, actual: U) -> MatcherResult {
        let server = TestFixture::server();
        let expected_id = self.0.try_to_block_id(server.game_state().block_types());
        match expected_id {
            Some(expected_id) => {
                if expected_id.equals_ignore_variant(*actual.borrow()) {
                    MatcherResult::Match
                } else {
                    MatcherResult::NoMatch
                }
            }
            None => MatcherResult::NoMatch,
        }
    }

    fn describe(&self, matcher_result: MatcherResult) -> Description {
        let server = TestFixture::server();
        let expected_type = self.0.try_to_block_id(server.game_state().block_types());

        let block_name = Self::describe_block(&server, expected_type);

        if matcher_result == MatcherResult::Match {
            format!("is {}", block_name).into()
        } else {
            format!("is not {}", block_name).into()
        }
    }

    fn explain_match(&self, actual: U) -> Description {
        let server = TestFixture::server();
        let actual_name = Self::describe_block(&server, Some(*actual.borrow()));
        let expected_id = self.0.try_to_block_id(server.game_state().block_types());
        let expected_name = Self::describe_block(&server, expected_id);
        if self.matches(actual).is_match() {
            Description::new().text(format!(
                "({}), which matches the expected block",
                actual_name
            ))
        } else if expected_id.is_some() {
            Description::new().text(format!(
                "({}), which does not match the expected block {}",
                actual_name, expected_name
            ))
        } else {
            Description::new().text(format!(
                "({}), in a matcher that will never match",
                actual_name
            ))
        }
    }
}

#[derive(MatcherBase, Debug)]
struct IsBlockWithVariant<T: TryToBlockId>(T);
impl<T: TryToBlockId> IsBlockWithVariant<T> {
    fn describe_block(server: &Server, block_type: Option<BlockId>) -> String {
        block_type
            .map(|id| {
                server
                    .game_state()
                    .block_types()
                    .get_block(id)
                    .ok()
                    .map_or_else(
                        || format!("an unknown block with ID 0x{:x}", id.0),
                        |x| format!("{:?}:0x{:x}", x.0.short_name(), id.variant()),
                    )
            })
            .unwrap_or_else(|| "an unknown block that will never match".to_string())
    }
}
impl<T: TryToBlockId, U: Borrow<BlockId> + Copy + std::fmt::Debug> Matcher<U>
    for IsBlockWithVariant<T>
{
    fn matches(&self, actual: U) -> MatcherResult {
        let expected_id = self
            .0
            .try_to_block_id(TestFixture::server().game_state().block_types());

        if expected_id == Some(*actual.borrow()) {
            MatcherResult::Match
        } else {
            MatcherResult::NoMatch
        }
    }

    fn describe(&self, matcher_result: MatcherResult) -> Description {
        let server = TestFixture::server();
        let expected_type = self.0.try_to_block_id(server.game_state().block_types());

        if matcher_result == MatcherResult::Match {
            format!("is {}", Self::describe_block(&server, expected_type)).into()
        } else {
            format!("is not {}", Self::describe_block(&server, expected_type)).into()
        }
    }

    fn explain_match(&self, actual: U) -> Description {
        let server = TestFixture::server();
        let actual_id = *actual.borrow();
        let actual_name = Self::describe_block(&server, Some(actual_id));
        let expected_id = self
            .0
            .try_to_block_id(TestFixture::server().game_state().block_types());
        let expected_name = Self::describe_block(&server, expected_id);
        if Some(actual_id) == expected_id {
            Description::new().text(format!(
                "({}), which matches the expected block {}",
                actual_name, expected_name
            ))
        } else if expected_id.is_some_and(|id| id.equals_ignore_variant(actual_id)) {
            Description::new().text(format!(
                "({}), which matches the expected block type {} but has the wrong variant",
                actual_name, expected_name
            ))
        } else if expected_id.is_some() {
            Description::new().text(format!(
                "({}), which does not match the expected block {}",
                actual_name, expected_name
            ))
        } else {
            Description::new().text(format!(
                "({}), in a matcher that will never match",
                actual_name
            ))
        }
    }
}

#[derive(MatcherBase, Debug)]
struct IsItemStack<T, U>(T, U)
where
    T: AsRef<str>,
    U: Matcher<u32>;
impl<T, U, V> Matcher<V> for IsItemStack<T, U>
where
    T: AsRef<str>,
    U: Matcher<u32>,
    V: Borrow<ItemStack> + Copy + std::fmt::Debug,
{
    fn matches(&self, actual: V) -> MatcherResult {
        let actual = actual.borrow();
        if self.0.as_ref() == actual.item_name() {
            if self.1.matches(actual.quantity_or_wear()).is_match() {
                MatcherResult::Match
            } else {
                MatcherResult::NoMatch
            }
        } else {
            MatcherResult::NoMatch
        }
    }

    fn describe(&self, matcher_result: MatcherResult) -> Description {
        let item_name = self.0.as_ref();
        if matcher_result == MatcherResult::Match {
            format!(
                "is {} with a quantity/wear that {}",
                item_name,
                self.1.describe(MatcherResult::Match)
            )
            .into()
        } else {
            format!(
                "is not {} with a quantity/wear that {}",
                item_name,
                self.1.describe(MatcherResult::Match)
            )
            .into()
        }
    }

    fn explain_match(&self, actual: V) -> Description {
        let actual = actual.borrow();
        if self.0.as_ref() == actual.item_name() {
            Description::new()
                .text("Is an itemstack of".to_string())
                .text(self.0.as_ref().to_string())
                .text("with a quantity/wear that")
                .nested(self.1.explain_match(actual.quantity_or_wear()))
        } else {
            Description::new()
                .text("is not an itemstack of")
                .text(self.0.as_ref().to_string())
                .text("with a quantity/wear that")
                .nested(self.1.explain_match(actual.quantity_or_wear()))
        }
    }
}

#[cfg(test)]
#[gtest]
fn sample_test(fixture: &TestFixture) -> googletest::Result<()> {
    assert_ok_and_assign!(let server = fixture.start_server(|builder| {
        builder.set_flatland_mapgen(BlockId(4096));
        Ok(())
    }));
    fixture.run_assertions_in_server(|gs| {
        use googletest::expect_that;
        use perovskite_core::block_id::special_block_defs::AIR_ID;

        // verify that the mapgen did its job
        expect_that!(gs.game_map().get_block(ZERO_COORD), ok(IsBlock(AIR_ID)));
        expect_that!(
            gs.game_map().get_block(BlockCoordinate::new(0, -1, 0)),
            ok(IsBlock(BlockId(4096)))
        );

        gs.game_map()
            .set_block(ZERO_COORD, BlockId(0), None)
            .expect("set_block");
        expect_that!(
            gs.game_map().get_block(ZERO_COORD).or_fail()?,
            IsBlock(BlockId(0))
        );

        Ok(())
    })?;
    Ok(())
}

#[cfg(test)]
#[gtest]
fn sample_test_real_game(fixture: &TestFixture) -> googletest::Result<()> {
    fixture.start_server(|builder| crate::configure_default_game(builder))?;
    fixture.run_assertions_in_server(|gs| {
        use googletest::expect_that;
        use perovskite_core::block_id::special_block_defs::AIR_ID;

        use crate::default_game::basic_blocks::{DIRT, DIRT_WITH_GRASS};

        gs.game_map()
            .set_block(ZERO_COORD, DIRT_WITH_GRASS, None)
            .or_fail()?;
        expect_that!(
            gs.game_map().get_block(ZERO_COORD).or_fail()?,
            IsBlock(DIRT_WITH_GRASS)
        );

        let dig_result = gs
            .game_map()
            .dig_block(ZERO_COORD, &EventInitiator::Engine, None)
            .or_fail()?;
        // Digging dirt with grass should yield dirt
        expect_that!(
            dig_result.item_stacks,
            elements_are![IsItemStack(DIRT.0, eq(1))]
        );

        expect_that!(
            gs.game_map().get_block(ZERO_COORD).or_fail()?,
            IsBlock(AIR_ID)
        );

        Ok(())
    })?;
    Ok(())
}
