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

use anyhow::{Context, Result};

use log::info;

use parking_lot::Mutex;
use tokio::sync::{oneshot, watch};
use tracy_client::{plot, span, Client};
use vulkano::{
    command_buffer::PrimaryAutoCommandBuffer,
    image::SampleCount,
    render_pass::Subpass,
    swapchain::{self, AcquireError, SwapchainPresentInfo},
    sync::{future::FenceSignalFuture, FlushError, GpuFuture},
};

use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget},
};

use crate::{
    game_state::{ClientState, FrameState},
    main_menu::MainMenu,
    net_client,
};

use super::{
    shaders::{
        cube_geometry::{self, BlockRenderPass},
        egui_adapter::{self, EguiAdapter},
        flat_texture, PipelineProvider, PipelineWrapper,
    },
    CommandBufferBuilder, VulkanContext,
};

pub(crate) struct ActiveGame {
    cube_provider: cube_geometry::CubePipelineProvider,
    cube_pipeline: cube_geometry::CubePipelineWrapper,

    flat_provider: flat_texture::FlatTexPipelineProvider,
    flat_pipeline: flat_texture::FlatTexPipelineWrapper,

    egui_adapter: Option<egui_adapter::EguiAdapter>,

    client_state: Arc<ClientState>,
}

impl ActiveGame {
    fn build_command_buffers(
        &mut self,
        window_size: PhysicalSize<u32>,
        ctx: &VulkanContext,
        mut command_buf_builder: vulkano::command_buffer::AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<vulkano::command_buffer::allocator::StandardCommandBufferAllocator>,
        >,
    ) -> PrimaryAutoCommandBuffer {
        let _span = span!("build renderer buffers");
        let FrameState {
            view_proj_matrix,
            player_position,
            tool_state,
        } = self
            .client_state
            .next_frame((window_size.width as f64) / (window_size.height as f64));
        let mut cube_draw_calls = vec![];
        if let Some(pointee) = tool_state.pointee {
            cube_draw_calls.push(
                self.client_state
                    .cube_renderer
                    .make_pointee_cube(player_position, pointee)
                    .unwrap(),
            );
        }
        // test only
        if let Some(neighbor) = tool_state.neighbor {
            cube_draw_calls.push(
                self.client_state
                    .cube_renderer
                    .make_pointee_cube(player_position, neighbor)
                    .unwrap(),
            );
        }

        let chunk_lock = {
            let _span = span!("Waiting for chunk_lock");
            self.client_state.chunks.cloned_view()
        };
        plot!("total_chunks", chunk_lock.len() as f64);
        cube_draw_calls.extend(
            chunk_lock
                .values()
                .filter_map(|chunk| chunk.lock().make_draw_call(player_position)),
        );
        plot!(
            "chunk_rate",
            cube_draw_calls.len() as f64 / chunk_lock.len() as f64
        );

        if !cube_draw_calls.is_empty() {
            self.cube_pipeline
                .bind(
                    ctx,
                    view_proj_matrix,
                    &mut command_buf_builder,
                    BlockRenderPass::Opaque,
                )
                .unwrap();
            self.cube_pipeline
                .draw(
                    &mut command_buf_builder,
                    &cube_draw_calls,
                    BlockRenderPass::Opaque,
                )
                .unwrap();
            self.cube_pipeline
                .bind(
                    ctx,
                    view_proj_matrix,
                    &mut command_buf_builder,
                    BlockRenderPass::Transparent,
                )
                .unwrap();
            self.cube_pipeline
                .draw(
                    &mut command_buf_builder,
                    &cube_draw_calls,
                    BlockRenderPass::Transparent,
                )
                .unwrap();
            self.cube_pipeline
                .bind(
                    ctx,
                    view_proj_matrix,
                    &mut command_buf_builder,
                    BlockRenderPass::Translucent,
                )
                .unwrap();
            self.cube_pipeline
                .draw(
                    &mut command_buf_builder,
                    &cube_draw_calls,
                    BlockRenderPass::Translucent,
                )
                .unwrap();
        }

        self.flat_pipeline
            .bind(ctx, (), &mut command_buf_builder, ())
            .unwrap();
        self.flat_pipeline
            .draw(
                &mut command_buf_builder,
                &self
                    .client_state
                    .hud
                    .lock()
                    .render(ctx, &self.client_state)
                    .unwrap(),
                (),
            )
            .unwrap();
        self.egui_adapter
            .as_mut()
            .unwrap()
            .draw(ctx, &mut command_buf_builder, &self.client_state)
            .unwrap();

        finish_command_buffer(command_buf_builder).unwrap()
    }

    fn handle_resize(&mut self, ctx: &mut VulkanContext, size: PhysicalSize<u32>) -> Result<()> {
        ctx.viewport.dimensions = size.into();
        self.cube_pipeline = self
            .cube_provider
            .make_pipeline(ctx, self.client_state.cube_renderer.atlas())
            .unwrap();
        self.flat_pipeline = self
            .flat_provider
            .make_pipeline(ctx, (self.client_state.hud.lock().texture_atlas(), 0))?;
        self.egui_adapter.as_mut().unwrap().notify_resize(ctx)?;
        Ok(())
    }
}

pub(crate) struct ConnectionState {
    pub(crate) progress: watch::Receiver<(f32, String)>,
    pub(crate) result: oneshot::Receiver<Result<ActiveGame>>,
}

pub(crate) enum GameState {
    MainMenu,
    Connecting(ConnectionState),
    ConnectError(String),
    Active(ActiveGame),
}
impl GameState {
    fn as_mut<'a>(&'a mut self) -> GameStateMutRef<'a> {
        match self {
            GameState::MainMenu => GameStateMutRef::MainMenu,
            GameState::Connecting(x) => GameStateMutRef::Connecting(x),
            GameState::Active(x) => GameStateMutRef::Active(x),
            GameState::ConnectError(x) => GameStateMutRef::ConnectError(x),
        }
    }
    fn update_if_connected(&mut self, ctx: &VulkanContext, event_loop: &EventLoopWindowTarget<()>) {
        if let GameState::Connecting(state) = self {
            match state.result.try_recv() {
                Ok(Ok(mut game)) => {
                    let egui_adapter =
                        EguiAdapter::new(ctx, event_loop, game.client_state.egui.clone());
                    match egui_adapter {
                        Ok(x) => {
                            game.egui_adapter = Some(x);
                            *self = GameState::Active(game);
                        }
                        Err(x) => *self = GameState::ConnectError(x.to_string()),
                    };
                }
                Ok(Err(e)) => {
                    *self = GameState::ConnectError(e.to_string());
                }
                Err(oneshot::error::TryRecvError::Closed) => {
                    *self = GameState::ConnectError("Connection thread crashed".to_string())
                }
                Err(oneshot::error::TryRecvError::Empty) => {
                    // pass
                }
            }
        }
    }
}

pub(crate) enum GameStateMutRef<'a> {
    MainMenu,
    Connecting(&'a mut ConnectionState),
    ConnectError(&'a mut String),
    Active(&'a mut ActiveGame),
}

pub struct GameRenderer {
    ctx: VulkanContext,
    game: Mutex<GameState>,

    main_menu: Mutex<MainMenu>,
    rt: Arc<tokio::runtime::Runtime>,
}
impl GameRenderer {
    pub(crate) fn create(event_loop: &EventLoop<()>) -> Result<GameRenderer> {
        let ctx = VulkanContext::create(event_loop).unwrap();
        let rt = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap(),
        );
        let main_menu = MainMenu::new(&ctx, event_loop);
        Ok(GameRenderer {
            ctx,
            game: Mutex::new(GameState::MainMenu),
            main_menu: Mutex::new(main_menu),
            rt,
        })
    }

    async fn connect_impl(
        ctx: VulkanContext,
        server_addr: String,
        username: String,
        password: String,
        mut progress: watch::Sender<(f32, String)>,
    ) -> Result<ActiveGame> {
        progress.send((0.1, format!("Connecting to {}", server_addr)))?;
        let client_state =
            net_client::connect_game(server_addr.to_string(), &ctx, &mut progress)
                .await?;

        let cube_provider = cube_geometry::CubePipelineProvider::new(ctx.vk_device.clone())?;
        let cube_pipeline =
            cube_provider.make_pipeline(&ctx, client_state.cube_renderer.atlas())?;

        let flat_provider = flat_texture::FlatTexPipelineProvider::new(ctx.vk_device.clone())?;
        let flat_pipeline =
            flat_provider.make_pipeline(&ctx, (client_state.hud.lock().texture_atlas(), 0))?;

        let game = ActiveGame {
            cube_provider,
            cube_pipeline,
            flat_provider,
            flat_pipeline,
            client_state,
            egui_adapter: None,
        };

        Ok(game)
    }

    pub fn run_loop(mut self, event_loop: EventLoop<()>) {
        let mut resized = false;
        let mut recreate_swapchain = false;
        let frames_in_flight = self.ctx.swapchain_images.len();
        let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
        let mut previous_fence_i = 0;

        event_loop.run(move |event, event_loop, control_flow| {
            let mut game_lock = self.game.lock();
            game_lock.update_if_connected(&self.ctx, &event_loop);
            {
                let _span = span!("client_state handling window event");

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
                } else {
                    if let Event::WindowEvent {
                        window_id: _,
                        event,
                    } = &event
                    {
                        self.main_menu.lock().update(&event);
                    }
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
                    resized = true;
                }
                Event::MainEventsCleared => {
                    if let GameStateMutRef::Active(game) = game_lock.as_mut() {

                        let _span = span!("MainEventsCleared");
                        if self.ctx.window.has_focus() && game.client_state.should_capture() {
                            let size = self.ctx.window.inner_size();
                            self.ctx
                                .window
                                .set_cursor_position(PhysicalPosition::new(
                                    size.width / 2,
                                    size.height / 2,
                                ))
                                .unwrap();
                            self.ctx
                                .window
                                .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                                .unwrap();
                            self.ctx.window.set_cursor_visible(false);
                        } else {
                            self.ctx.window.set_cursor_visible(true);
                            self.ctx
                                .window
                                .set_cursor_grab(winit::window::CursorGrabMode::None)
                                .unwrap();
                        }
                    }
                    if resized || recreate_swapchain {
                        let _span = span!("Recreate swapchain");
                        let size = self.ctx.window.inner_size();
                        recreate_swapchain = false;
                        self.ctx.recreate_swapchain(size).unwrap();
                        if resized {
                            resized = false;
                            if let GameStateMutRef::Active(game) = game_lock.as_mut() {
                                game.handle_resize(&mut self.ctx, size).unwrap();
                            }
                        }
                    }

                    let _swapchain_span = span!("Acquire swapchain image");
                    let (image_i, suboptimal, acquire_future) =
                        match swapchain::acquire_next_image(self.ctx.swapchain.clone(), None) {
                            Ok(r) => r,
                            Err(AcquireError::OutOfDate) => {
                                info!("Swapchain out of date");
                                recreate_swapchain = true;
                                return;
                            }
                            Err(e) => panic!("failed to acquire next image: {e}"),
                        };
                    Client::running()
                        .expect("tracy client must be running")
                        .frame_mark();
                    if suboptimal {
                        recreate_swapchain = true;
                    }
                    if let Some(image_fence) = &fences[image_i as usize] {
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

                    // From https://vulkano.rs/compute_pipeline/descriptor_sets.html:
                    // Once you have created a descriptor set, you may also use it with other pipelines,
                    // as long as the bindings' types match those the pipelines' shaders expect.
                    // But Vulkan requires that you provide a pipeline whenever you create a descriptor set;
                    // you cannot create one independently of any particular pipeline.
                    let mut command_buf_builder = self.ctx.start_command_buffer().unwrap();
                    self.ctx
                        .start_render_pass(
                            &mut command_buf_builder,
                            self.ctx.framebuffers[image_i as usize].clone(),
                        )
                        .unwrap();
                    let command_buffers = if let GameStateMutRef::Active(game) = game_lock.as_mut()
                    {
                        game.build_command_buffers(window_size, &self.ctx, command_buf_builder)
                    } else {
                        let rt = self.rt.clone();
                        let ctx = self.ctx.clone();
                        self.main_menu.lock().draw(
                            &self.ctx,
                            &mut game_lock,
                            &mut command_buf_builder,
                            |host, user, pass| {
                                let progress =
                                    watch::channel((0.0, "Starting connection...".to_string()));
                                let result = oneshot::channel();
                                rt.spawn(async move {
                                    let active_game =
                                        Self::connect_impl(ctx, host, user, pass, progress.0).await;
                                    match result.0.send(active_game) {
                                        Ok(_) => {},
                                        Err(_) => {
                                            panic!("Failed to hand off the active game to the renderer.")
                                        },
                                    }
                                });
                                ConnectionState {
                                    progress: progress.1,
                                    result: result.1,
                                }
                            },
                        );
                        finish_command_buffer(command_buf_builder).unwrap()
                    };

                    {
                        let _span = span!("submit to Vulkan");
                        let future = previous_future
                            .join(acquire_future)
                            .then_execute(self.ctx.queue.clone(), command_buffers)
                            .unwrap()
                            .then_swapchain_present(
                                self.ctx.queue.clone(),
                                SwapchainPresentInfo::swapchain_image_index(
                                    self.ctx.swapchain.clone(),
                                    image_i,
                                ),
                            )
                            .then_signal_fence_and_flush();
                        fences[image_i as usize] = match future {
                            Ok(value) => Some(Arc::new(value)),
                            Err(FlushError::OutOfDate) => {
                                recreate_swapchain = true;
                                None
                            }
                            Err(e) => {
                                println!("failed to flush future: {e}");
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
}

impl Drop for ActiveGame {
    fn drop(&mut self) {
        self.client_state.shutdown.cancel()
    }
}

fn finish_command_buffer(
    mut builder: CommandBufferBuilder<PrimaryAutoCommandBuffer>,
) -> Result<PrimaryAutoCommandBuffer> {
    let _span = span!();
    builder.end_render_pass()?;
    builder
        .build()
        .with_context(|| "Command buffer build failed")
}
