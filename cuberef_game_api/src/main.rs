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

use cuberef_game_api::default_game::DefaultGameBuilder;
use tracing::metadata::LevelFilter;
use tracing_subscriber::{prelude::*, EnvFilter};

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    #[cfg(feature = "dhat-heap")]
    {
        let _profiler = dhat::Profiler::new_heap();
    }

    #[cfg(feature = "tokio-console")]
    {
        let console_layer = console_subscriber::spawn();
        tracing_subscriber::registry()
            .with(console_layer)
            .with(tracing_subscriber::fmt::layer().with_filter(
                tracing_subscriber::EnvFilter::builder()
                    .with_default_directive(LevelFilter::INFO.into())
                    .from_env_lossy()))
            .with(tracing_tracy::TracyLayer::new().with_filter(tracing_subscriber::filter::filter_fn(|x| {
                x.module_path().is_some_and(|x| x.contains("cuberef"))
            })))
            .init();
    }
    #[cfg(not(feature = "tokio-console"))]
    {
        tracing_subscriber::fmt::init();
    }

    let game = DefaultGameBuilder::new_from_commandline().unwrap();
    game.build_and_run().unwrap();
}
