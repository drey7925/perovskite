// Copyright 2026 drey7925
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

//! Logs into a server headlessly, waits for chunks around a fixed camera to load, renders one
//! frame, and writes it as a PNG.
//!
//! Works without a display or GPU via Mesa's software Vulkan driver (lavapipe), e.g.:
//!
//! ```text
//! VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
//!   cargo run -p perovskite_client --bin headless_render -- \
//!   --server grpc://localhost:28275 --user agent --register \
//!   --pos 0 10 0 --az 45 --el -20 --out frame.png
//! ```

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use cgmath::vec3;
use clap::Parser;
use perovskite_client::vulkan::headless::{render_headless, HeadlessRenderRequest};

#[derive(Parser, Debug)]
#[command(about = "Render a single frame from a Perovskite server to a PNG, without a window")]
struct Args {
    /// Server address
    #[arg(long, default_value = "grpc://localhost:28275")]
    server: String,
    #[arg(long)]
    user: String,
    /// Password. If omitted, read from the PEROVSKITE_PASSWORD environment variable, falling back
    /// to the username.
    #[arg(long)]
    pass: Option<String>,
    /// Register a new account instead of logging in
    #[arg(long)]
    register: bool,

    /// Camera (eye) position: X Y Z
    #[arg(long, num_args = 3, allow_negative_numbers = true, value_names = ["X", "Y", "Z"])]
    pos: Vec<f64>,
    /// Camera azimuth in degrees
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    az: f64,
    /// Camera elevation in degrees (positive looks up)
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    el: f64,

    #[arg(long, default_value_t = 1280)]
    width: u32,
    #[arg(long, default_value_t = 720)]
    height: u32,
    #[arg(long, default_value = "frame.png")]
    out: PathBuf,

    /// Seconds without any change in the renderable chunk count (after the chunks near the camera
    /// have loaded) before rendering
    #[arg(long, default_value_t = 2.0)]
    settle_secs: f64,
    /// Radius in chunks (per axis) around the camera that must be fully loaded before rendering.
    /// The server always sends chunks within 3 of the player, so keep this at most 3.
    #[arg(long, default_value_t = 2)]
    near_radius: i32,
    /// Maximum seconds to wait for chunks before rendering anyway
    #[arg(long, default_value_t = 60.0)]
    timeout_secs: f64,
    /// Frames to render (and discard) before the captured frame
    #[arg(long, default_value_t = 1)]
    warmup_frames: u32,
    /// Don't draw the HUD (crosshair, hotbar)
    #[arg(long)]
    no_hud: bool,
    /// Use the interactive client's settings file instead of defaults
    #[arg(long)]
    settings_from_disk: bool,
    /// Render distance, in chunks. Defaults to 8 (the interactive default of 20 is very slow on
    /// a software rasterizer), or to the settings file's value with --settings-from-disk.
    #[arg(long)]
    render_distance: Option<u32>,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();
    anyhow::ensure!(args.pos.len() == 3, "--pos takes exactly three values");
    let password = match args.pass {
        Some(pass) => pass,
        None => std::env::var("PEROVSKITE_PASSWORD").unwrap_or_else(|_| args.user.clone()),
    };

    render_headless(HeadlessRenderRequest {
        server: args.server,
        username: args.user,
        password,
        register: args.register,
        position: vec3(args.pos[0], args.pos[1], args.pos[2]),
        azimuth: args.az,
        elevation: args.el,
        width: args.width,
        height: args.height,
        output: args.out,
        settle_time: Duration::from_secs_f64(args.settle_secs),
        near_radius: args.near_radius,
        max_wait: Duration::from_secs_f64(args.timeout_secs),
        warmup_frames: args.warmup_frames,
        render_hud: !args.no_hud,
        settings_from_disk: args.settings_from_disk,
        render_distance: match (args.render_distance, args.settings_from_disk) {
            (Some(distance), _) => Some(distance),
            (None, true) => None,
            (None, false) => Some(8),
        },
    })
    .context("Headless render failed")
}
