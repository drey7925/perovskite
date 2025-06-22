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

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};

use arc_swap::ArcSwap;
use cgmath::{vec3, InnerSpace};
use log::info;

use super::{
    shaders::{
        cube_geometry::{self, BlockRenderPass, CubeGeometryDrawCall},
        egui_adapter::EguiAdapter,
        entity_geometry, flat_texture,
    },
    FramebufferHolder, VulkanContext, VulkanWindow,
};
use crate::client_state::input::Keybind;
use crate::main_menu::InputCapture;
use crate::vulkan::raytrace_buffer::{RaytraceBuffer, RenderThreadAction, RtFrameData};
use crate::vulkan::shaders::flat_texture::FlatPipelineConfig;
use crate::vulkan::shaders::raytracer::{
    ChunkMapHeader, RaytracingBindings, RaytracingPerFrameData,
};
use crate::vulkan::shaders::{raytracer, sky, LiveRenderConfig};
use crate::{
    client_state::{settings::GameSettings, ClientState, FrameState},
    main_menu::MainMenu,
    net_client,
};
use parking_lot::Mutex;
use tokio::sync::{oneshot, watch};
use tracy_client::{plot, span, Client};
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferExecFuture, CopyBufferInfo, SubpassBeginInfo,
    SubpassContents, SubpassEndInfo,
};
use vulkano::render_pass::Subpass;
use vulkano::swapchain::{PresentFuture, SwapchainAcquireFuture};
use vulkano::sync::future::JoinFuture;
use vulkano::{
    command_buffer::PrimaryAutoCommandBuffer,
    swapchain::{self, SwapchainPresentInfo},
    sync::{future::FenceSignalFuture, GpuFuture},
    Validated, VulkanError,
};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState};
use winit::event_loop::ActiveEventLoop;
use winit::window::WindowId;
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

pub(crate) struct ActiveGame {
    cube_provider: cube_geometry::CubePipelineProvider,
    cube_pipeline: cube_geometry::CubePipelineWrapper,

    flat_provider: flat_texture::FlatTexPipelineProvider,
    flat_pipeline: flat_texture::FlatTexPipelineWrapper,

    entities_provider: entity_geometry::EntityPipelineProvider,
    entities_pipeline: entity_geometry::EntityPipelineWrapper,

    sky_provider: sky::SkyPipelineProvider,
    sky_pipeline: sky::SkyPipelineWrapper,

    raytraced_provider: raytracer::RaytracedPipelineProvider,
    raytraced_pipeline: raytracer::RaytracedPipelineWrapper,

    egui_adapter: Option<EguiAdapter>,

    client_state: Arc<ClientState>,
    // Held across frames to avoid constant reallocations. Only the allocation itself is important
    cube_draw_calls: Vec<CubeGeometryDrawCall>,

    raytrace_data: Option<RaytraceBuffer>,
}

impl ActiveGame {
    fn advance_without_rendering(&mut self) {
        let _ = self
            .client_state
            .next_frame(1.0, self.client_state.timekeeper.now());
    }
    fn build_command_buffers(
        &mut self,
        window_size: PhysicalSize<u32>,
        ctx: &VulkanWindow,
        framebuffer: &FramebufferHolder,
        input_capture: &mut InputCapture,
    ) -> Result<(
        Arc<PrimaryAutoCommandBuffer>,
        Option<Arc<PrimaryAutoCommandBuffer>>,
    )> {
        let _span = span!("build renderer buffers");
        let mut command_buf_builder = ctx.start_command_buffer()?;
        let start_tick = self.client_state.timekeeper.now();
        {
            let mut lock = self.client_state.entities.lock();

            lock.advance_all_states(
                start_tick,
                &self.client_state.audio,
                // This position is only needed for audio
                self.client_state.weakly_ordered_last_position().position,
            );
        }
        let FrameState {
            scene_state,
            mut player_position,
            tool_state,
            ime_enabled,
        } = self.client_state.next_frame(
            (window_size.width as f64) / (window_size.height as f64),
            start_tick,
        );
        ctx.window.set_ime_allowed(ime_enabled);

        ctx.reclaim_u32.unsequester(framebuffer.image_i);
        let prework_buffer = if self.client_state.settings.load().render.raytracing {
            let RtFrameData {
                new_buffer,
                update_steps,
            } = self.client_state.chunks.raytrace_buffers().acquire(&ctx)?;
            if let Some(new_buffer) = new_buffer {
                if let Some(old) = self.raytrace_data.replace(new_buffer) {
                    ctx.reclaim_u32.give_buffer(
                        old.data,
                        Some(framebuffer.image_i),
                        Duration::from_secs(5),
                    );
                }
            }
            if update_steps.is_empty() {
                None
            } else {
                let mut buf = ctx.start_command_buffer()?;
                for step in update_steps {
                    match step {
                        RenderThreadAction::Incremental {
                            staging_buffer,
                            scatter_gather_list,
                        } => {
                            let dst_buffer = self
                                .raytrace_data
                                .as_ref()
                                .context(
                                    "Got raytrace incremental update without a raytrace buffer",
                                )?
                                .data
                                .buffer
                                .clone();

                            buf.copy_buffer(CopyBufferInfo {
                                regions: scatter_gather_list.into_iter().collect(),
                                ..CopyBufferInfo::buffers(staging_buffer.buffer.clone(), dst_buffer)
                            })?;
                            ctx.reclaim_u32.give_buffer(
                                staging_buffer,
                                Some(framebuffer.image_i),
                                Duration::from_secs(5),
                            );
                        }
                    }
                }

                Some(buf)
            }
        } else {
            if let Some(buf) = self.raytrace_data.take() {
                ctx.reclaim_u32.give_buffer(
                    buf.data,
                    Some(framebuffer.image_i),
                    Duration::from_secs(5),
                );
            }
            None
        };

        ctx.start_raster_render_pass(
            &mut command_buf_builder,
            &framebuffer,
            scene_state.clear_color,
        )?;

        self.sky_pipeline
            .bind_and_draw(ctx, scene_state, &mut command_buf_builder)?;

        self.cube_draw_calls.clear();

        {
            let entity_lock = self.client_state.entities.lock();
            if let Some(entity_target) = entity_lock.attached_to_entity {
                if let Some(entity) = entity_lock.entities.get(&entity_target.entity_id) {
                    if let Some(position) = entity.attach_position(
                        start_tick,
                        &self.client_state.entity_renderer,
                        entity_target.trailing_entity_index,
                    ) {
                        player_position = position;
                        let debug_speed = entity.debug_speed(start_tick);
                        self.client_state
                            .physics_state
                            .lock()
                            .set_position(player_position);

                        plot!(
                            "entity_buffer",
                            entity.estimated_buffer(start_tick).max(0.0) as f64
                        );
                        plot!(
                            "entity_buffer_count",
                            entity.estimated_buffer_count() as f64
                        );
                        // Most test tracks run in the Z direction, so this is an easy metric to debug with.
                        plot!("entity_z_coord", player_position.z);
                        plot!("entity_cms", entity.debug_cms() as f64);
                        plot!("entity_cme", entity.debug_cme(start_tick) as f64);
                        plot!("entity_speed", debug_speed as f64);
                    }
                }
            }
        }

        if let Some(pointee) = tool_state.pointee {
            let pointee_cube = self.client_state.block_renderer.make_pointee_cube(
                player_position,
                pointee.target(),
                &self.client_state,
                start_tick,
            )?;
            if let Some(pointee_cube) = pointee_cube {
                self.cube_draw_calls.push(pointee_cube);
            }
        }

        let chunks = {
            let _span = span!("Waiting for chunk_lock");
            self.client_state.chunks.renderable_chunks_cloned_view()
        };
        let mut chunks = chunks.into_iter().collect::<Vec<_>>();
        plot!("total_chunks", chunks.len() as f64);

        let (batched_calls, batched_handled) = self
            .client_state
            .chunks
            .make_batched_draw_calls(player_position, scene_state.vp_matrix);

        self.cube_draw_calls.extend(batched_calls);

        // Sort by expected draw order, closest to farthest
        chunks.sort_unstable_by_key(|(coord, _)| {
            let world_coord = vec3(
                coord.x as f64 * 16.0 + 8.0,
                coord.y as f64 * 16.0 + 8.0,
                coord.z as f64 * 16.0 + 8.0,
            );
            -1 * ((player_position - world_coord).magnitude2() as i64)
        });

        self.cube_draw_calls
            .extend(chunks.iter().filter_map(|(coord, chunk)| {
                if !batched_handled.contains(coord) {
                    chunk.make_draw_call(*coord, player_position, scene_state.vp_matrix)
                } else {
                    None
                }
            }));
        plot!(
            "chunk_rate",
            self.cube_draw_calls.len() as f64 / chunks.len() as f64
        );

        if !self.cube_draw_calls.is_empty() {
            if self.raytrace_data.is_none() {
                self.cube_pipeline
                    .draw(
                        ctx,
                        &mut command_buf_builder,
                        scene_state,
                        &mut self.cube_draw_calls,
                        BlockRenderPass::Opaque,
                    )
                    .context("Opaque pipeline draw failed")?;
                self.cube_pipeline
                    .draw(
                        ctx,
                        &mut command_buf_builder,
                        scene_state,
                        &mut self.cube_draw_calls,
                        BlockRenderPass::Transparent,
                    )
                    .context("Transparent pipeline draw failed")?;
                self.cube_pipeline
                    .draw(
                        ctx,
                        &mut command_buf_builder,
                        scene_state,
                        &mut self.cube_draw_calls,
                        BlockRenderPass::Translucent,
                    )
                    .context("Translucent pipeline draw failed")?;
            } else {
                self.cube_pipeline
                    .draw(
                        ctx,
                        &mut command_buf_builder,
                        scene_state,
                        &mut self.cube_draw_calls,
                        BlockRenderPass::RaytraceFallback,
                    )
                    .context("RaytraceFallback pipeline draw failed")?;
            }
        }
        let entity_draw_calls = {
            let lock = self.client_state.entities.lock();
            lock.render_calls(
                player_position,
                start_tick,
                &self.client_state.entity_renderer,
            )
        };
        self.entities_pipeline
            .draw(
                ctx,
                &mut command_buf_builder,
                scene_state,
                entity_draw_calls,
            )
            .context("Entities pipeline draw failed")?;

        {
            if let Some(buf) = self.raytrace_data.as_ref() {
                self.raytraced_pipeline.draw_rt_primary(
                    ctx,
                    RaytracingBindings {
                        scene_state,
                        data: buf.data.clone(),
                        header: buf.header.clone(),
                        depth_view: framebuffer.rt_primary_framebuffer.attachments()[1].clone(),
                        deferred_buffers: framebuffer
                            .deferred_specular_buffers
                            .clone()
                            .context("Missing deferred buffers but trying to raytrace")?,
                        player_pos: player_position,
                        render_distance: self.client_state.render_distance.load(Ordering::Relaxed),
                    },
                    &mut command_buf_builder,
                )?;
            }
        }

        command_buf_builder.end_render_pass(SubpassEndInfo {
            ..Default::default()
        })?;

        ctx.start_non_depth_render_pass(
            &mut command_buf_builder,
            framebuffer.pre_blit_nondepth_framebuffer.clone(),
            SubpassContents::Inline,
        )?;

        self.flat_pipeline
            .bind(ctx, &mut command_buf_builder)
            .context("Flat pipeline bind failed")?;
        // HUD update-and-render must happen after the frame state is completed, since other layers
        // (especially tool controller) may want to consume input at higher priority than HUD
        self.flat_pipeline
            .draw(
                &mut command_buf_builder,
                &self
                    .client_state
                    .hud
                    .lock()
                    .update_and_render(ctx, &self.client_state)
                    .context("HUD update-and-render failed")?,
            )
            .context("Flat pipeline draw failed")?;

        command_buf_builder.end_render_pass(SubpassEndInfo {
            ..Default::default()
        })?;

        framebuffer
            .blit_supersampling(&mut command_buf_builder)
            .context("Supersampling blit failed")?;
        ctx.start_non_depth_render_pass(
            &mut command_buf_builder,
            framebuffer.post_blit_framebuffer.clone(),
            SubpassContents::SecondaryCommandBuffers,
        )
        .context("Start post-blit render pass failed")?;

        // Then draw the UI
        self.egui_adapter
            .as_mut()
            .context("Internal error: egui adapter missing")?
            .draw(
                ctx,
                &mut command_buf_builder,
                &self.client_state,
                input_capture,
                &tool_state,
            )
            .context("Egui draw failed")?;

        command_buf_builder.end_render_pass(SubpassEndInfo {
            ..Default::default()
        })?;
        let primary = command_buf_builder
            .build()
            .with_context(|| "Command buffer build failed")?;
        let prework = match prework_buffer {
            None => None,
            Some(x) => Some(
                x.build()
                    .with_context(|| "Command buffer build failed for prework buffer")?,
            ),
        };

        Ok((primary, prework))
    }

    fn handle_resize(&mut self, ctx: &mut VulkanWindow) -> Result<()> {
        let global_config = self
            .client_state
            .settings
            .load()
            .render
            .build_global_config();
        self.cube_pipeline = self.cube_provider.make_pipeline(
            ctx,
            self.client_state.block_renderer.atlas(),
            &global_config,
        )?;
        self.entities_pipeline = self.entities_provider.make_pipeline(
            ctx,
            self.client_state.entity_renderer.atlas(),
            &global_config,
        )?;
        self.flat_pipeline = {
            let hud_lock = self.client_state.hud.lock();
            self.flat_provider.make_pipeline(
                ctx,
                FlatPipelineConfig {
                    atlas: hud_lock.texture_atlas.as_ref(),
                    subpass: Subpass::from(ctx.non_depth_render_pass.clone(), 0)
                        .context("nondepth renderpass 1 missing")?,
                    enable_depth_stencil: false,
                    enable_supersampling: true,
                },
                &global_config,
            )?
        };
        self.sky_pipeline = self.sky_provider.make_pipeline(ctx, &global_config)?;
        self.raytraced_pipeline = self.raytraced_provider.make_pipeline(
            ctx,
            &self.client_state.block_renderer,
            &global_config,
        )?;
        self.egui_adapter
            .as_mut()
            .context("Missing egui adapter")?
            .notify_resize(ctx)?;
        Ok(())
    }
}

pub(crate) struct ConnectionState {
    pub(crate) progress: watch::Receiver<(f32, String)>,
    pub(crate) result: oneshot::Receiver<Result<Arc<ClientState>>>,
}

pub(crate) enum GameState {
    MainMenu,
    Connecting(ConnectionState),
    Error(anyhow::Error),
    Active(ActiveGame),
}
impl GameState {
    fn as_mut(&mut self) -> GameStateMutRef<'_> {
        match self {
            GameState::MainMenu => GameStateMutRef::MainMenu,
            GameState::Connecting(x) => GameStateMutRef::Connecting(x),
            GameState::Active(x) => GameStateMutRef::Active(x),
            GameState::Error(x) => GameStateMutRef::ConnectError(x),
        }
    }
    fn update_if_connected(&mut self, ctx: &VulkanWindow, event_loop: &ActiveEventLoop) {
        if let GameState::Connecting(state) = self {
            match state.result.try_recv() {
                Ok(Ok(client_state)) => {
                    let mut game = match make_active_game(ctx, client_state) {
                        Ok(game) => game,
                        Err(e) => {
                            *self = GameState::Error(e);
                            return;
                        }
                    };
                    let egui_adapter =
                        EguiAdapter::new(ctx, event_loop, game.client_state.egui.clone());
                    match egui_adapter {
                        Ok(x) => {
                            game.egui_adapter = Some(x);
                            *self = GameState::Active(game);
                        }
                        Err(x) => *self = GameState::Error(x),
                    };
                }
                Ok(Err(e)) => {
                    *self = GameState::Error(e);
                }
                Err(oneshot::error::TryRecvError::Closed) => {
                    *self = GameState::Error(anyhow!(
                        "Connection thread crashed without details".to_string()
                    ))
                }
                Err(oneshot::error::TryRecvError::Empty) => {
                    // pass
                }
            }
        } else if let GameState::Active(game) = self {
            let pending_error = game.client_state.pending_error.lock().take();
            if let Some(err) = pending_error {
                *self = GameState::Error(err)
            } else if game.client_state.cancel_requested() {
                if *game.client_state.wants_exit_from_game.lock() {
                    event_loop.exit();
                }
                *self = GameState::MainMenu;
            }
        }
    }
}

fn make_active_game(vk_wnd: &VulkanWindow, client_state: Arc<ClientState>) -> Result<ActiveGame> {
    let global_render_config = client_state.settings.load().render.build_global_config();

    let cube_provider = cube_geometry::CubePipelineProvider::new(vk_wnd.vk_device.clone())?;
    let cube_pipeline = cube_provider.make_pipeline(
        &vk_wnd,
        client_state.block_renderer.atlas(),
        &global_render_config,
    )?;

    let entities_provider = entity_geometry::EntityPipelineProvider::new(vk_wnd.vk_device.clone())?;
    let entities_pipeline = entities_provider.make_pipeline(
        &vk_wnd,
        client_state.entity_renderer.atlas(),
        &global_render_config,
    )?;

    let flat_provider = flat_texture::FlatTexPipelineProvider::new(vk_wnd.vk_device.clone())?;
    let flat_pipeline = {
        let hud_lock = client_state.hud.lock();
        flat_provider.make_pipeline(
            &vk_wnd,
            FlatPipelineConfig {
                atlas: hud_lock.texture_atlas.as_ref(),
                subpass: Subpass::from(vk_wnd.non_depth_render_pass.clone(), 0)
                    .context("non-depth renderpass missing")?,
                enable_depth_stencil: false,
                enable_supersampling: true,
            },
            &global_render_config,
        )?
    };

    let sky_provider = sky::SkyPipelineProvider::new(vk_wnd.vk_device.clone())?;
    let sky_pipeline = sky_provider.make_pipeline(&vk_wnd, &global_render_config)?;
    let raytraced_provider = raytracer::RaytracedPipelineProvider::new(vk_wnd.vk_device.clone())?;
    let raytraced_pipeline = raytraced_provider.make_pipeline(
        &vk_wnd,
        &client_state.block_renderer,
        &global_render_config,
    )?;

    let game = ActiveGame {
        cube_provider,
        cube_pipeline,
        flat_provider,
        flat_pipeline,
        entities_provider,
        entities_pipeline,
        sky_provider,
        sky_pipeline,
        raytraced_provider,
        raytraced_pipeline,
        client_state,
        egui_adapter: None,
        cube_draw_calls: vec![],
        raytrace_data: None,
    };

    Ok(game)
}

pub(crate) enum GameStateMutRef<'a> {
    MainMenu,
    Connecting(&'a mut ConnectionState),
    ConnectError(&'a mut anyhow::Error),
    Active(&'a mut ActiveGame),
}

type FutureType = FenceSignalFuture<
    PresentFuture<CommandBufferExecFuture<JoinFuture<Box<dyn GpuFuture>, SwapchainAcquireFuture>>>,
>;

pub struct GameRenderer {
    ctx: VulkanWindow,
    settings: Arc<ArcSwap<GameSettings>>,
    game: Mutex<GameState>,

    main_menu: Mutex<MainMenu>,
    rt: Arc<tokio::runtime::Runtime>,

    warned_about_capture_issue: bool,

    fences: Vec<Option<Arc<FutureType>>>,
    previous_fence_i: usize,
    input_capture: InputCapture,
    need_swapchain_recreate: bool,
}
impl GameRenderer {
    pub(crate) fn create(event_loop: &ActiveEventLoop) -> Result<GameRenderer> {
        let settings = Arc::new(ArcSwap::new(GameSettings::load_from_disk()?.into()));

        let ctx = VulkanWindow::create(event_loop, &settings)?;
        let rt = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?,
        );
        let main_menu = MainMenu::new(&ctx, event_loop, settings.clone());
        let frames_in_flight = ctx.swapchain_images.len();
        let fences = vec![None; frames_in_flight];
        Ok(GameRenderer {
            ctx,
            settings,
            game: Mutex::new(GameState::MainMenu),
            main_menu: Mutex::new(main_menu),
            rt,
            warned_about_capture_issue: false,
            fences,
            previous_fence_i: 0,
            input_capture: InputCapture::NotCapturing,
            need_swapchain_recreate: false,
        })
    }
}

pub struct GameApplication {
    renderer: Option<GameRenderer>,
}
impl GameApplication {
    pub fn new() -> Self {
        Self { renderer: None }
    }
}

impl ApplicationHandler for GameApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_none() {
            self.renderer = Some(GameRenderer::create(event_loop).unwrap());
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(renderer) = &mut self.renderer {
            renderer.window_event(event_loop, window_id, event);
        }
    }

    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(renderer) = &mut self.renderer {
            renderer.device_event(event_loop, device_id, event);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(renderer) = &mut self.renderer {
            renderer.ctx.window.request_redraw();
        }
    }
}

impl GameRenderer {
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if event == WindowEvent::RedrawRequested {
            return self.redraw(event_loop);
        }

        if let InputCapture::Capturing(action) = self.input_capture {
            match &event {
                WindowEvent::KeyboardInput { event, .. } => {
                    self.input_capture =
                        InputCapture::Captured(action, Keybind::Key(event.physical_key));
                    return;
                }
                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button,
                    ..
                } => {
                    self.input_capture =
                        InputCapture::Captured(action, Keybind::MouseButton(*button));
                    return;
                }
                _ => {}
            }
        }
        if let WindowEvent::Resized(_) = &event {
            self.need_swapchain_recreate = true;
        }

        let mut game_lock = self.game.lock();
        game_lock.update_if_connected(&self.ctx, event_loop);

        if let GameStateMutRef::Active(game) = game_lock.as_mut() {
            if event == WindowEvent::CloseRequested || event == WindowEvent::Destroyed {
                game.client_state.shutdown.cancel();
                event_loop.exit();
                return;
            }
            let consumed = game.egui_adapter.as_mut().unwrap().window_event(&event);

            if !consumed {
                game.client_state.window_event(&event);
            }
        } else {
            if event == WindowEvent::CloseRequested || event == WindowEvent::Destroyed {
                event_loop.exit();
                return;
            }
            self.main_menu.lock().update(&event);
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        let mut game_lock = self.game.lock();

        if let GameStateMutRef::Active(game) = game_lock.as_mut() {
            game.client_state.device_event(&event);
        }
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        let mut game_lock = self.game.lock();

        game_lock.update_if_connected(&self.ctx, event_loop);
        if let GameStateMutRef::Active(game) = game_lock.as_mut() {
            if self.ctx.want_recreate.swap(false, Ordering::AcqRel) {
                self.need_swapchain_recreate = true;
            }

            let _span = span!("MainEventsCleared");
            if self.ctx.window.has_focus() && game.client_state.input.lock().is_mouse_captured() {
                let size = self.ctx.window.inner_size();
                match self
                    .ctx
                    .window
                    .set_cursor_position(PhysicalPosition::new(size.width / 2, size.height / 2))
                {
                    Ok(_) => (),
                    Err(e) => {
                        if !self.warned_about_capture_issue {
                            self.warned_about_capture_issue = true;
                            log::warn!(
                                            "Failed to move cursor position to center of window. This message will only display once per session. {:?}", e
                                        )
                        }
                    }
                }
                let mode = if cfg!(target_os = "macos") {
                    winit::window::CursorGrabMode::Locked
                } else {
                    winit::window::CursorGrabMode::Confined
                };
                match self.ctx.window.set_cursor_grab(mode) {
                    Ok(_) => (),
                    Err(e) => {
                        if !self.warned_about_capture_issue {
                            self.warned_about_capture_issue = true;
                            log::warn!(
                                            "Failed to grab cursor, falling back to non-confined. This message will only display once per session. {:?}", e
                                        )
                        }
                    }
                }
                self.ctx.window.set_cursor_visible(false);
            } else {
                self.ctx.window.set_cursor_visible(true);
                // todo this is actually fallible, and some window managers get unhappy
                match self
                    .ctx
                    .window
                    .set_cursor_grab(winit::window::CursorGrabMode::None)
                {
                    Ok(_) => (),
                    Err(_) => {
                        if !self.warned_about_capture_issue {
                            self.warned_about_capture_issue = true;
                            log::warn!(
                                            "Failed to release cursor, falling back to non-confined. This message will only display once per session."
                                        )
                        }
                    }
                }
            }
        } else {
            self.ctx
                .window
                .set_cursor_grab(winit::window::CursorGrabMode::None)
                .unwrap();
            self.ctx.window.set_cursor_visible(true);
        }

        if self.need_swapchain_recreate {
            let _span = span!("Recreate swapchain");
            let size = self.ctx.window.inner_size();

            if size.height == 0 || size.width == 0 {
                if let GameStateMutRef::Active(game) = game_lock.as_mut() {
                    game.advance_without_rendering();
                    // Hacky, otherwise we spin really fast here, while also messing
                    // with physics state thanks to tiny updates.
                    //
                    // This has to hold together long enough until vulkano releases the
                    // taskgraph and this whole module is rewritten anyway.
                    std::thread::sleep(Duration::from_millis(100));
                }
                return;
            }

            self.need_swapchain_recreate = false;
            self.ctx
                .recreate_swapchain(size, self.settings.load().render.build_global_config())
                .unwrap();
            self.ctx.viewport.extent = size.into();
            if let GameStateMutRef::Active(game) = game_lock.as_mut() {
                if let Err(e) = game.handle_resize(&mut self.ctx) {
                    *game_lock = GameState::Error(e)
                };
            }
        }

        let _swapchain_span = span!("Acquire swapchain image");
        let (image_i, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(self.ctx.swapchain.clone(), None) {
                Ok(r) => r,
                Err(Validated::Error(VulkanError::OutOfDate)) => {
                    info!("Swapchain out of date");
                    self.need_swapchain_recreate = true;
                    if let GameStateMutRef::Active(game) = game_lock.as_mut() {
                        game.advance_without_rendering();
                    }
                    return;
                }
                Err(e) => panic!("failed to acquire next image: {:?}", e),
            };
        Client::running()
            .expect("tracy client must be running")
            .frame_mark();
        if suboptimal {
            self.need_swapchain_recreate = true;
        }
        if let Some(image_fence) = &self.fences[image_i as usize] {
            // TODO: This sometimes stalls for a long time. Figure out why.
            image_fence.wait(None).unwrap();
        }
        let mut previous_future = match self.fences[self.previous_fence_i].clone() {
            // Create a NowFuture
            None => {
                let mut now = vulkano::sync::now(self.ctx.vk_device.clone());
                now.cleanup_finished();
                now.boxed()
            }
            // Use the existing FenceSignalFuture
            Some(fence) => fence.boxed(),
        };
        drop(_swapchain_span);
        let window_size = self.ctx.window.inner_size();

        let fb_holder = &self.ctx.framebuffers[image_i as usize];

        let game_command_buffers = if let GameStateMutRef::Active(game) = game_lock.as_mut() {
            match game.build_command_buffers(
                window_size,
                &self.ctx,
                fb_holder,
                &mut self.input_capture,
            ) {
                Ok(x) => Some(x),
                Err(e) => {
                    *game_lock = GameState::Error(e);
                    None
                }
            }
        } else {
            None
        };
        let command_buffers = if let Some(command_buffers) = game_command_buffers {
            command_buffers
        } else {
            // either we're in the main menu or we got an error building the real
            // game buffers
            let mut command_buf_builder = self.ctx.start_command_buffer().unwrap();
            // we're not in the active game, allow the IME
            self.ctx.window.set_ime_allowed(true);

            // self.ctx
            //     .start_ssaa_raster_render_pass(
            //         &mut command_buf_builder,
            //         &fb_holder,
            //         [0.0, 0.0, 0.0, 0.0],
            //     )
            //     .unwrap();
            // // There are two subpasses in the SSAA render pass
            // command_buf_builder
            //     .next_subpass(SubpassEndInfo::default(), SubpassBeginInfo::default())
            //     .unwrap();
            // command_buf_builder
            //     .end_render_pass(SubpassEndInfo {
            //         ..Default::default()
            //     })
            //     .unwrap();
            // // With the render pass done, blit to the final framebuffer
            // fb_holder
            //     .blit_supersampling(&mut command_buf_builder)
            //     .unwrap();
            self.ctx
                .start_non_depth_render_pass(
                    &mut command_buf_builder,
                    fb_holder.post_blit_framebuffer.clone(),
                    SubpassContents::SecondaryCommandBuffers,
                )
                .unwrap();

            if let Some(connection_settings) = self.main_menu.lock().draw(
                &mut self.ctx,
                &mut game_lock,
                &mut command_buf_builder,
                &mut self.input_capture,
            ) {
                self.start_connection(connection_settings);
            }
            command_buf_builder
                .end_render_pass(SubpassEndInfo {
                    ..Default::default()
                })
                .unwrap();

            (
                command_buf_builder
                    .build()
                    .with_context(|| "Command buffer build failed")
                    .unwrap(),
                None,
            )
        };

        let (primary, prework) = command_buffers;

        if let Some(prework) = prework {
            previous_future = Box::new(
                previous_future
                    .then_execute(self.ctx.graphics_queue.clone(), prework)
                    .unwrap()
                    .then_signal_fence_and_flush()
                    .unwrap(),
            );
        }

        {
            let _span = span!("submit to Vulkan");
            let future = previous_future
                .join(acquire_future)
                .then_execute(self.ctx.graphics_queue.clone(), primary)
                .unwrap()
                .then_swapchain_present(
                    self.ctx.graphics_queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(
                        self.ctx.swapchain.clone(),
                        image_i,
                    ),
                )
                .then_signal_fence_and_flush();
            self.fences[image_i as usize] = match future {
                // Holding this in an Arc makes this easier
                #[allow(clippy::arc_with_non_send_sync)]
                Ok(value) => Some(Arc::new(value)),
                Err(Validated::Error(VulkanError::OutOfDate)) => {
                    self.need_swapchain_recreate = true;
                    None
                }
                Err(e) => {
                    log::error!("failed to flush future: {e}");
                    None
                }
            };
        }
        self.previous_fence_i = image_i as usize;
    }

    fn start_connection(&self, connection: ConnectionSettings) {
        let game_settings = self.settings.clone();

        {
            let ctx = self.ctx.clone_context();
            self.rt.spawn(async move {
                let active_game = connect_impl(
                    ctx,
                    game_settings,
                    connection.host,
                    connection.user,
                    connection.pass,
                    connection.register,
                    connection.progress,
                )
                .await;
                match connection.result.send(active_game) {
                    Ok(_) => {}
                    Err(_) => {
                        panic!("Failed to hand off the active game to the renderer.")
                    }
                }
            });
        }
    }
}

impl Drop for ActiveGame {
    fn drop(&mut self) {
        self.client_state.shutdown.cancel()
    }
}

pub(crate) struct ConnectionSettings {
    pub(crate) host: String,
    pub(crate) user: String,
    pub(crate) pass: String,
    pub(crate) register: bool,

    pub(crate) progress: watch::Sender<(f32, String)>,
    pub(crate) result: oneshot::Sender<Result<Arc<ClientState>>>,
}

async fn connect_impl(
    ctx: Arc<VulkanContext>,
    settings: Arc<ArcSwap<GameSettings>>,
    server_addr: String,
    username: String,
    password: String,
    register: bool,
    mut progress: watch::Sender<(f32, String)>,
) -> Result<Arc<ClientState>> {
    progress.send((0.1, format!("Connecting to {}", server_addr)))?;
    let client_state = net_client::connect_game(
        server_addr,
        username,
        password,
        register,
        settings,
        ctx,
        &mut progress,
    )
    .await?;
    Ok(client_state)
}
