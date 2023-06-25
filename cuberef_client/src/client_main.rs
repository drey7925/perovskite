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



use anyhow::Result;
use clap::Parser;
use winit::event_loop::EventLoop;

use crate::vulkan::game_renderer::GameRenderer;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct ClientArgs {
    /// The server host:port to connect to
    server: String,
}

pub fn run_client(_args: &ClientArgs) -> Result<()> {
    let _tracy_client = tracy_client::Client::start();
    let event_loop = EventLoop::new();

    let window = GameRenderer::create(&event_loop)?;
    window.run_loop(event_loop);
    Ok(())
}
