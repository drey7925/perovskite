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

//! Headless, single-shot rendering: log in, wait for chunks around a fixed camera to load, render
//! one frame into an offscreen image, and write it out as a PNG.
//!
//! There is no window, surface, swapchain, egui, or audio output, so this works with a software
//! Vulkan implementation (e.g. Mesa's lavapipe/llvmpipe) on a machine without a display.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use arc_swap::ArcSwap;
use cgmath::Vector3;
use perovskite_core::coordinates::BlockCoordinate;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use vulkano::sync::GpuFuture;
use winit::dpi::PhysicalSize;

use super::game_renderer::{connect_impl, make_active_game, FrameCommandBuffers};
use super::VulkanWindow;
use crate::client_state::settings::GameSettings;
use crate::main_menu::InputCapture;

pub struct HeadlessRenderRequest {
    /// Server address, e.g. `grpc://localhost:28273`
    pub server: String,
    pub username: String,
    pub password: String,
    /// Register a new account rather than logging into an existing one
    pub register: bool,

    /// Camera (eye) position in world coordinates
    pub position: Vector3<f64>,
    /// Camera azimuth, in degrees
    pub azimuth: f64,
    /// Camera elevation, in degrees. Positive looks up.
    pub elevation: f64,

    pub width: u32,
    pub height: u32,
    pub output: PathBuf,

    /// Chunk loading is considered done once every chunk within `near_radius` of the camera has
    /// arrived, and after that the number of renderable (i.e. meshed, non-empty) chunks has not
    /// changed for this long. The server sends, and the mesh workers process, chunks nearest to
    /// the camera first, so this is a reasonable proxy for "the view is loaded".
    ///
    /// The total number of *loaded* chunks is not used, since the server keeps streaming distant
    /// (often all-air, hence never renderable) chunks long after the visible ones arrive.
    /// A stable renderable count alone isn't enough either: on a slow client/server, chunks
    /// can trickle in slowly enough to look settled before the nearby ones have arrived.
    pub settle_time: Duration,
    /// Radius, in chunks (per axis), around the camera that must be fully loaded before
    /// rendering. The server always sends chunks within 3 chunks of the player, so values up to 3
    /// are safe; larger values may never be satisfied (and will wait until `max_wait`).
    pub near_radius: i32,
    /// Render anyway (with a warning) if chunks haven't settled after this long.
    pub max_wait: Duration,
    /// Number of frames to render before the captured one. Some state (e.g. raytrace buffers,
    /// text, HUD contents) is only uploaded during a frame, so a warmup frame helps.
    pub warmup_frames: u32,
    pub render_hud: bool,
    /// Use the interactive client's settings file (e.g. to match render settings) instead of
    /// defaults. Audio is disabled either way.
    pub settings_from_disk: bool,
    /// Overrides the render distance (in chunks) from settings.
    pub render_distance: Option<u32>,
}

/// Runs a full headless render as described by the request, blocking until the PNG is written.
pub fn render_headless(request: HeadlessRenderRequest) -> Result<()> {
    let mut settings = if request.settings_from_disk {
        GameSettings::load_from_disk()?
    } else {
        GameSettings::default()
    };
    settings.audio.enable_audio = false;
    if let Some(render_distance) = request.render_distance {
        settings.render.render_distance = render_distance;
    }
    let settings = Arc::new(ArcSwap::new(Arc::new(settings)));
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let vk_wnd = VulkanWindow::create_headless(&settings, request.width, request.height)
        .context("Failed to create headless Vulkan context")?;
    log::info!("Headless rendering on {}", vk_wnd.current_gpu_name());

    let cancellation = CancellationToken::new();
    let (progress_tx, mut progress_rx) = watch::channel((0.0, String::new()));
    let connection = rt.spawn(connect_impl(
        vk_wnd.clone_context(),
        settings.clone(),
        request.server.clone(),
        request.username.clone(),
        request.password.clone(),
        request.register,
        progress_tx,
        cancellation.clone(),
    ));
    rt.spawn(async move {
        while progress_rx.changed().await.is_ok() {
            let (fraction, message) = progress_rx.borrow_and_update().clone();
            log::info!("Connecting: {:.0}% {message}", fraction * 100.0);
        }
    });
    let client_state = rt
        .block_on(connection)
        .context("Connection task panicked")??
        .context("Connection was cancelled")?;

    client_state.physics_state.lock().pin_camera(
        request.position,
        request.azimuth,
        request.elevation,
    );

    let mut game = make_active_game(&vk_wnd, client_state.clone())?;
    game.render_hud = request.render_hud;

    wait_for_chunks(
        &mut game,
        &client_state,
        request.position,
        request.near_radius,
        request.settle_time,
        request.max_wait,
    )?;

    let size = PhysicalSize::new(request.width, request.height);
    let framebuffer = &vk_wnd.framebuffers[0];
    let mut input_capture = InputCapture::NotCapturing;
    for i in 0..=request.warmup_frames {
        let FrameCommandBuffers {
            primary, prework, ..
        } = game
            .build_command_buffers(size, &vk_wnd, framebuffer, &mut input_capture)
            .with_context(|| format!("Failed to build command buffers for frame {i}"))?;
        let mut future = vulkano::sync::now(vk_wnd.vk_device.clone()).boxed();
        if let Some(prework) = prework {
            future = future
                .then_execute(vk_wnd.graphics_queue.clone(), prework)?
                .boxed();
        }
        future
            .then_execute(vk_wnd.graphics_queue.clone(), primary)?
            .then_signal_fence_and_flush()?
            .wait(None)?;
    }
    if let Some(err) = client_state.pending_error.lock().take() {
        return Err(err.context("Client failed while rendering"));
    }

    let mut image = vk_wnd.read_back_target(0)?;
    // The output image may have garbage alpha (e.g. from the sky or UI blending); a screenshot
    // should be opaque.
    for pixel in image.pixels_mut() {
        pixel.0[3] = 255;
    }
    image
        .save(&request.output)
        .with_context(|| format!("Failed to write {}", request.output.display()))?;
    log::info!("Wrote {}", request.output.display());

    drop(game);
    client_state.shutdown.cancel();
    cancellation.cancel();
    drop(client_state);
    rt.shutdown_timeout(Duration::from_secs(2));
    Ok(())
}

fn wait_for_chunks(
    game: &mut super::game_renderer::ActiveGame,
    client_state: &crate::client_state::ClientState,
    camera_position: Vector3<f64>,
    near_radius: i32,
    settle_time: Duration,
    max_wait: Duration,
) -> Result<()> {
    let camera_chunk = BlockCoordinate::try_from(camera_position)
        .context("Camera position out of bounds")?
        .chunk();
    let start = Instant::now();
    let mut near_loaded_at = None;
    let mut last_renderable = 0;
    let mut last_change = Instant::now();
    loop {
        if let Some(err) = client_state.pending_error.lock().take() {
            return Err(err.context("Client failed while waiting for chunks"));
        }
        if client_state.cancel_requested() {
            bail!("Client shut down while waiting for chunks");
        }
        // Runs the per-frame client state update (with the camera pinned, so no physics). This
        // also publishes the camera position to the mesh workers and the network.
        game.advance_without_rendering();

        let (loaded, renderable) = client_state.chunks.chunk_counts();
        if renderable != last_renderable {
            last_renderable = renderable;
            last_change = Instant::now();
        }
        let (near_loaded, near_total) =
            client_state.chunks.loaded_within(camera_chunk, near_radius);
        if near_loaded_at.is_none() && near_loaded == near_total {
            log::info!(
                "All {near_total} chunks near the camera loaded after {:.1}s",
                start.elapsed().as_secs_f64()
            );
            near_loaded_at = Some(Instant::now());
        }
        let settled = near_loaded_at.is_some_and(|t| t.elapsed() >= settle_time)
            && last_change.elapsed() >= settle_time;
        if settled {
            log::info!(
                "Chunks settled after {:.1}s: {loaded} loaded, {renderable} renderable",
                start.elapsed().as_secs_f64()
            );
            return Ok(());
        }
        if start.elapsed() >= max_wait {
            log::warn!(
                "Chunks did not settle within {:.1}s ({near_loaded}/{near_total} near the camera, \
                 {loaded} loaded, {renderable} renderable); rendering anyway",
                max_wait.as_secs_f64()
            );
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}
