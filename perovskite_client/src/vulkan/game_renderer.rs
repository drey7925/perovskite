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

use parking_lot::Mutex;
use tokio::sync::{oneshot, watch};
use tracy_client::{plot, span, Client};
use vulkano::command_buffer::SubpassEndInfo;
use vulkano::render_pass::Subpass;
use vulkano::{
    command_buffer::PrimaryAutoCommandBuffer,
    swapchain::{self, SwapchainPresentInfo},
    sync::{future::FenceSignalFuture, GpuFuture},
    Validated, VulkanError,
};

use winit::event::ElementState;
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget},
};

use crate::game_state::input::Keybind;
use crate::main_menu::InputCapture;
use crate::vulkan::shaders::flat_texture::FlatPipelineConfig;
use crate::vulkan::shaders::{sky, LiveRenderConfig};
use crate::{
    game_state::{settings::GameSettings, ClientState, FrameState},
    main_menu::MainMenu,
    net_client,
};

use super::{
    shaders::{
        cube_geometry::{self, BlockRenderPass, CubeGeometryDrawCall},
        egui_adapter::EguiAdapter,
        entity_geometry, flat_texture, PipelineProvider, PipelineWrapper,
    },
    FramebufferHolder, VulkanContext, VulkanWindow,
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

    egui_adapter: Option<EguiAdapter>,

    client_state: Arc<ClientState>,
    // Held across frames to avoid constant reallocations
    cube_draw_calls: Vec<CubeGeometryDrawCall>,
}

impl ActiveGame {
    fn advance_without_rendering(&mut self) {
        let _ = self.client_state.next_frame(1.0);
    }
    fn build_command_buffers(
        &mut self,
        window_size: PhysicalSize<u32>,
        ctx: &VulkanWindow,
        framebuffer: &FramebufferHolder,
        input_capture: &mut InputCapture,
    ) -> Result<Arc<PrimaryAutoCommandBuffer>> {
        let _span = span!("build renderer buffers");
        let mut command_buf_builder = ctx.start_command_buffer()?;
        let FrameState {
            scene_state,
            mut player_position,
            tool_state,
            ime_enabled,
        } = self
            .client_state
            .next_frame((window_size.width as f64) / (window_size.height as f64));
        ctx.window.set_ime_allowed(ime_enabled);

        ctx.start_ssaa_render_pass(
            &mut command_buf_builder,
            framebuffer.ssaa_framebuffer.clone(),
            scene_state.clear_color,
        )?;

        self.sky_pipeline
            .bind(ctx, scene_state, &mut command_buf_builder, ())?;
        self.sky_pipeline.draw(&mut command_buf_builder, (), ())?;

        self.cube_draw_calls.clear();
        let start_tick = self.client_state.timekeeper.now();

        {
            let mut entity_lock = self.client_state.entities.lock();
            entity_lock.advance_all_states(start_tick, &self.client_state.audio, player_position);
            if let Some(entity_id) = entity_lock.attached_to_entity {
                if let Some(entity) = entity_lock.entities.get(&entity_id) {
                    player_position =
                        entity.attach_position(start_tick, &self.client_state.entity_renderer);
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

        if let Some(pointee) = tool_state.pointee {
            self.cube_draw_calls.push(
                self.client_state
                    .block_renderer
                    .make_pointee_cube(player_position, pointee.target())?,
            );
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
            self.cube_pipeline
                .bind(
                    ctx,
                    scene_state,
                    &mut command_buf_builder,
                    BlockRenderPass::Opaque,
                )
                .context("Opaque pipeline bind failed")?;
            self.cube_pipeline
                .draw(
                    &mut command_buf_builder,
                    &mut self.cube_draw_calls,
                    BlockRenderPass::Opaque,
                )
                .context("Opaque pipeline draw failed")?;
            self.cube_pipeline
                .bind(
                    ctx,
                    scene_state,
                    &mut command_buf_builder,
                    BlockRenderPass::Transparent,
                )
                .context("Transparent pipeline bind failed")?;
            self.cube_pipeline
                .draw(
                    &mut command_buf_builder,
                    &mut self.cube_draw_calls,
                    BlockRenderPass::Transparent,
                )
                .context("Transparent pipeline draw failed")?;
            self.cube_pipeline
                .bind(
                    ctx,
                    scene_state,
                    &mut command_buf_builder,
                    BlockRenderPass::Translucent,
                )
                .context("Translucent bind failed")?;
            self.cube_pipeline
                .draw(
                    &mut command_buf_builder,
                    &mut self.cube_draw_calls,
                    BlockRenderPass::Translucent,
                )
                .context("Translucent pipeline draw failed")?;
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
            .bind(ctx, scene_state, &mut command_buf_builder, ())
            .context("Entities pipeline bind failed")?;
        self.entities_pipeline
            .draw(&mut command_buf_builder, entity_draw_calls, ())
            .context("Entities pipeline draw failed")?;

        self.flat_pipeline
            .bind(ctx, (), &mut command_buf_builder, ())
            .context("Flat pipeline bind failed")?;
        self.flat_pipeline
            .draw(
                &mut command_buf_builder,
                &self
                    .client_state
                    .hud
                    .lock()
                    .update_and_render(ctx, &self.client_state)
                    .context("HUD update-and-render failed")?,
                (),
            )
            .context("Flat pipeline draw failed")?;

        command_buf_builder.end_render_pass(SubpassEndInfo {
            ..Default::default()
        })?;
        // With the render pass done, blit to the final framebuffer
        framebuffer
            .blit_supersampling(&mut command_buf_builder)
            .context("Supersampling blit failed")?;
        ctx.start_post_blit_render_pass(
            &mut command_buf_builder,
            framebuffer.post_blit_framebuffer.clone(),
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
            )
            .context("Egui draw failed")?;

        command_buf_builder.end_render_pass(SubpassEndInfo {
            ..Default::default()
        })?;
        command_buf_builder
            .build()
            .with_context(|| "Command buffer build failed")
    }

    fn handle_resize(&mut self, ctx: &mut VulkanWindow) -> Result<()> {
        let global_config = LiveRenderConfig {
            supersampling: self.client_state.settings.load().render.supersampling,
        };
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
                    subpass: Subpass::from(ctx.ssaa_render_pass.clone(), 0)
                        .context("SSAA subpass 0 missing")?,
                    enable_depth_stencil: true,
                    enable_supersampling: true,
                },
                &global_config,
            )?
        };
        self.sky_pipeline = self.sky_provider.make_pipeline(ctx, (), &global_config)?;
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
    fn update_if_connected(
        &mut self,
        ctx: &VulkanWindow,
        event_loop: &EventLoopWindowTarget<()>,
        control_flow: &mut ControlFlow,
    ) {
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
                    *control_flow = ControlFlow::ExitWithCode(0);
                }
                *self = GameState::MainMenu;
            }
        }
    }
}

fn make_active_game(vk_wnd: &VulkanWindow, client_state: Arc<ClientState>) -> Result<ActiveGame> {
    let global_render_config = LiveRenderConfig {
        supersampling: client_state.settings.load().render.supersampling,
    };

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
                subpass: Subpass::from(vk_wnd.ssaa_render_pass.clone(), 0)
                    .context("SSAA subpass 0 missing")?,
                enable_depth_stencil: true,
                enable_supersampling: true,
            },
            &global_render_config,
        )?
    };

    let sky_provider = sky::SkyPipelineProvider::new(vk_wnd.vk_device.clone())?;
    let sky_pipeline = sky_provider.make_pipeline(&vk_wnd, (), &global_render_config)?;

    let game = ActiveGame {
        cube_provider,
        cube_pipeline,
        flat_provider,
        flat_pipeline,
        entities_provider,
        entities_pipeline,
        sky_provider,
        sky_pipeline,
        client_state,
        egui_adapter: None,
        cube_draw_calls: vec![],
    };

    Ok(game)
}

pub(crate) enum GameStateMutRef<'a> {
    MainMenu,
    Connecting(&'a mut ConnectionState),
    ConnectError(&'a mut anyhow::Error),
    Active(&'a mut ActiveGame),
}

pub struct GameRenderer {
    ctx: VulkanWindow,
    settings: Arc<ArcSwap<GameSettings>>,
    game: Mutex<GameState>,

    main_menu: Mutex<MainMenu>,
    rt: Arc<tokio::runtime::Runtime>,

    warned_about_capture_issue: bool,
}
impl GameRenderer {
    pub(crate) fn create(event_loop: &EventLoop<()>) -> Result<GameRenderer> {
        let settings = Arc::new(ArcSwap::new(GameSettings::load_from_disk()?.into()));

        let ctx = VulkanWindow::create(event_loop, &settings).unwrap();
        let rt = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap(),
        );
        let main_menu = MainMenu::new(&ctx, event_loop, settings.clone());
        Ok(GameRenderer {
            ctx,
            settings,
            game: Mutex::new(GameState::MainMenu),
            main_menu: Mutex::new(main_menu),
            rt,
            warned_about_capture_issue: false,
        })
    }

    pub fn run_loop(mut self, event_loop: EventLoop<()>) {
        let mut recreate_swapchain = false;

        let frames_in_flight = self.ctx.swapchain_images.len();
        let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
        let mut previous_fence_i = 0;

        let mut input_capture = InputCapture::NotCapturing;

        event_loop.run(move |event, event_loop, control_flow| {
            let mut game_lock = self.game.lock();
            game_lock.update_if_connected(&self.ctx, event_loop, control_flow);
            {
                let _span = span!("client_state handling window event");

                if let InputCapture::Capturing(action) = input_capture {
                    if let Event::WindowEvent {
                        window_id: _,
                        event
                    } = &event {
                        match event {WindowEvent::KeyboardInput { input, .. } => {
                            input_capture =
                                InputCapture::Captured(action, Keybind::ScanCode(input.scancode));
                            return;
                        }
                        WindowEvent::MouseInput {
                            state: ElementState::Pressed,
                            button,
                            ..
                        } => {
                            input_capture =
                                InputCapture::Captured(action, Keybind::MouseButton(*button));
                            return;
                        }
                        _ => {}
                    }}
                }

                if let GameStateMutRef::Active(game) = game_lock.as_mut() {
                    let consumed = if let Event::WindowEvent {
                        window_id: _,
                        event,
                    } = &event
                    {
                        game.egui_adapter.as_mut().unwrap().window_event(event)
                    } else {
                        false
                    };

                    if !consumed {
                        game.client_state.window_event(&event);
                    }
                } else if let Event::WindowEvent {
                    window_id: _,
                    event,
                } = &event
                {
                    self.main_menu.lock().update(event);
                }
            }
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    if let GameStateMutRef::Active(game) = game_lock.as_mut() {
                        game.client_state.shutdown.cancel();
                    }
                    *control_flow = ControlFlow::Exit;
                }
                Event::WindowEvent {
                    event: WindowEvent::Resized(_),
                    ..
                } => {
                    recreate_swapchain = true;
                }
                Event::MainEventsCleared => {
                    if let GameStateMutRef::Active(game) = game_lock.as_mut() {
                        if self.ctx.want_recreate.swap(false, Ordering::AcqRel) {
                            recreate_swapchain = true;
                        }

                        let _span = span!("MainEventsCleared");
                        if self.ctx.window.has_focus()
                            && game.client_state.input.lock().is_mouse_captured()
                        {
                            let size = self.ctx.window.inner_size();
                            match self.ctx
                                .window
                                .set_cursor_position(PhysicalPosition::new(
                                    size.width / 2,
                                    size.height / 2,
                                )) {
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
                            match self
                                .ctx
                                .window
                                .set_cursor_grab(mode)
                            {
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
                            match self.ctx
                                .window
                                .set_cursor_grab(winit::window::CursorGrabMode::None) {
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

                    if recreate_swapchain {
                        let _span = span!("Recreate swapchain");
                        let size = self.ctx.window.inner_size();

                        if size.height == 0 || size.width == 0 {
                            if let GameStateMutRef::Active(game) = game_lock.as_mut()
                            {
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

                        recreate_swapchain = false;
                        self.ctx.recreate_swapchain(size, self.settings.load().render.supersampling).unwrap();
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
                                recreate_swapchain = true;
                                if let GameStateMutRef::Active(game) = game_lock.as_mut()
                                {
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
                        recreate_swapchain = true;
                    }
                    if let Some(image_fence) = &fences[image_i as usize] {
                        // TODO: This sometimes stalls for a long time. Figure out why.
                        image_fence.wait(None).unwrap();
                    }
                    let previous_future = match fences[previous_fence_i as usize].clone() {
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

                    let game_command_buffers = if let GameStateMutRef::Active(game) = game_lock.as_mut()
                    {
                        match game.build_command_buffers(
                            window_size,
                            &self.ctx,
                            fb_holder,
                            &mut input_capture,
                        ) {
                            Ok(x) => {Some(x)}
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
                        self.ctx
                            .start_ssaa_render_pass(
                                &mut command_buf_builder,
                                fb_holder.ssaa_framebuffer.clone(),
                                [0.0, 0.0, 0.0, 0.0],
                            )
                            .unwrap();
                        command_buf_builder.end_render_pass(SubpassEndInfo {
                            ..Default::default()
                        }).unwrap();
                        // With the render pass done, blit to the final framebuffer
                        fb_holder.blit_supersampling(&mut command_buf_builder).unwrap();
                        self.ctx.start_post_blit_render_pass(
                            &mut command_buf_builder,
                            fb_holder.post_blit_framebuffer.clone(),
                        ).unwrap();

                        if let Some(connection_settings) = self.main_menu.lock().draw(
                            &mut self.ctx,
                            &mut game_lock,
                            &mut command_buf_builder,
                            &mut input_capture
                        ) {
                            self.start_connection(connection_settings);
                        }
                        command_buf_builder.end_render_pass(SubpassEndInfo {
                            ..Default::default()
                        }).unwrap();

                        command_buf_builder
                            .build()
                            .with_context(|| "Command buffer build failed")
                            .unwrap()
                    };

                    {
                        let _span = span!("submit to Vulkan");
                        let future = previous_future
                            .join(acquire_future)
                            .then_execute(self.ctx.graphics_queue.clone(), command_buffers)
                            .unwrap()
                            .then_swapchain_present(
                                self.ctx.graphics_queue.clone(),
                                SwapchainPresentInfo::swapchain_image_index(
                                    self.ctx.swapchain.clone(),
                                    image_i,
                                ),
                            )
                            .then_signal_fence_and_flush();
                        fences[image_i as usize] = match future {
                            // Holding this in an Arc makes this easier
                            #[allow(clippy::arc_with_non_send_sync)]
                            Ok(value) => Some(Arc::new(value)),
                            Err(Validated::Error(VulkanError::OutOfDate)) => {
                                recreate_swapchain = true;
                                None
                            }
                            Err(e) => {
                                log::error!("failed to flush future: {e}");
                                None
                            }
                        };
                    }
                    previous_fence_i = image_i;
                }
                _ => {}
            }
        })
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
