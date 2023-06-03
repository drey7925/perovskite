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

use std::sync::Arc;

use clap::Parser;
use cuberef_server::{
    game_state::testutils::{register_test_blocks_and_items, FakeMapgen},
    server::{ServerArgs, ServerBuilder},
};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Eventually this will be re-exported and called from the user of cuberef_plugin_api
    let args = ServerArgs::parse();
    let mut builder = ServerBuilder::from_args(&args).unwrap();
    register_test_blocks_and_items(&mut builder);
    builder.set_mapgen(|blocks, _| {
        Arc::new(FakeMapgen {
            block_type_manager: blocks,
        })
    });
    let server = builder.build().unwrap();

    server.serve().unwrap();
}
