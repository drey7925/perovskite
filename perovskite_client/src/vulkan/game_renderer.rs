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

use std::{sync::Arc, time::Duration};

use anyhow::{Context, Result};

use arc_swap::ArcSwap;
use futures::stream::repeat_with;
use log::info;

use parking_lot::{Condvar, Mutex, RwLock};
use tokio::sync::{oneshot, watch, Semaphore};
use tracy_client::{plot, span, Client};
use vulkano::{
    command_buffer::PrimaryAutoCommandBuffer,
    pipeline::graphics::viewport::Viewport,
    render_pass::Framebuffer,
    swapchain::{self, AcquireError, SwapchainAcquireFuture, SwapchainPresentInfo},
    sync::{future::FenceSignalFuture, FlushError, GpuFuture},
};

use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget},
};

use crate::{
    block_renderer::{VkChunkPass, VkChunkVertexData},
    game_state::{settings::GameSettings, ClientState, FrameState},
    main_menu::MainMenu,
    net_client,
};

use super::{
    shaders::{
        cube_geometry::{self, BlockRenderPass, CubeGeometryDrawCall},
        egui_adapter::{self, EguiAdapter},
        flat_texture, PipelineProvider, PipelineWrapper,
    },
    CommandBufferBuilder, VulkanWindow,
};

pub(crate) struct ActiveGame {
    cube_provider: cube_geometry::CubePipelineProvider,
    cube_pipeline: cube_geometry::CubePipelineWrapper,

    flat_provider: flat_texture::FlatTexPipelineProvider,
    flat_pipeline: flat_texture::FlatTexPipelineWrapper,

    egui_adapter: Mutex<Option<egui_adapter::EguiAdapter>>,

    client_state: Arc<ClientState>,
}

impl ActiveGame {
    fn build_command_buffers(
        &self,
        window_size: PhysicalSize<u32>,
        ctx: &VulkanWindow,
        framebuffer: Arc<Framebuffer>,
        mut command_buf_builder: vulkano::command_buffer::AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<vulkano::command_buffer::allocator::StandardCommandBufferAllocator>,
        >,
    ) -> PrimaryAutoCommandBuffer {
        let _span = span!("build renderer buffers");

        let mut cube_draw_calls = vec![];

        let FrameState {
            scene_state,
            player_position,
            tool_state,
        } = self
            .client_state
            .next_frame((window_size.width as f64) / (window_size.height as f64));
        ctx.start_render_pass(
            &mut command_buf_builder,
            framebuffer,
            scene_state.clear_color,
        )
        .unwrap();
        cube_draw_calls.clear();
        if let Some(pointee) = tool_state.pointee {
            cube_draw_calls.push(
                self.client_state
                    .block_renderer
                    .make_pointee_cube(player_position, pointee)
                    .unwrap(),
            );
        }
        // test only
        if let Some(neighbor) = tool_state.neighbor {
            if self
                .client_state
                .settings
                .load()
                .render
                .show_placement_guide
            {
                cube_draw_calls.push(
                    self.client_state
                        .block_renderer
                        .make_pointee_cube(player_position, neighbor)
                        .unwrap(),
                );
            }
        }

        let (entity_translations, vtx, idx) = {
            let mut entity_coords = vec![];
            let entity_lock = self.client_state.entities.lock();
            for entity in entity_lock.entities.values() {
                entity_coords.push(entity.as_transform(player_position))
            }
            (
                entity_coords,
                entity_lock.fake_entity_vtx.clone(),
                entity_lock.fake_entity_idx.clone(),
            )
        };

        for translation in entity_translations {
            cube_draw_calls.push(CubeGeometryDrawCall {
                models: VkChunkVertexData {
                    solid_opaque: None,
                    transparent: Some(VkChunkPass {
                        vtx: vtx.clone(),
                        idx: idx.clone(),
                    }),
                    translucent: None,
                },
                model_matrix: translation,
            })
        }

        let chunk_lock = {
            let _span = span!("Waiting for chunk_lock");
            self.client_state.chunks.renderable_chunks_cloned_view()
        };
        plot!("total_chunks", chunk_lock.len() as f64);
        cube_draw_calls.extend(chunk_lock.iter().filter_map(|(coord, chunk)| {
            chunk.make_draw_call(*coord, player_position, scene_state.vp_matrix)
        }));
        plot!(
            "chunk_rate",
            cube_draw_calls.len() as f64 / chunk_lock.len() as f64
        );

        if !cube_draw_calls.is_empty() {
            self.cube_pipeline
                .bind(
                    ctx,
                    scene_state,
                    &mut command_buf_builder,
                    BlockRenderPass::Opaque,
                )
                .unwrap();
            self.cube_pipeline
                .draw(
                    &mut command_buf_builder,
                    &mut cube_draw_calls,
                    BlockRenderPass::Opaque,
                )
                .unwrap();
            self.cube_pipeline
                .bind(
                    ctx,
                    scene_state,
                    &mut command_buf_builder,
                    BlockRenderPass::Transparent,
                )
                .unwrap();
            self.cube_pipeline
                .draw(
                    &mut command_buf_builder,
                    &mut cube_draw_calls,
                    BlockRenderPass::Transparent,
                )
                .unwrap();
            self.cube_pipeline
                .bind(
                    ctx,
                    scene_state,
                    &mut command_buf_builder,
                    BlockRenderPass::Translucent,
                )
                .unwrap();
            self.cube_pipeline
                .draw(
                    &mut command_buf_builder,
                    &mut cube_draw_calls,
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
                    .update_and_render(ctx, &self.client_state)
                    .unwrap(),
                (),
            )
            .unwrap();
        self.egui_adapter
            .lock()
            .as_mut()
            .unwrap()
            .draw(ctx, &mut command_buf_builder, &self.client_state)
            .unwrap();

        finish_command_buffer(command_buf_builder).unwrap()
    }

    fn handle_resize(&mut self, ctx: &VulkanWindow) -> Result<()> {
        self.cube_pipeline = self
            .cube_provider
            .make_pipeline(ctx, self.client_state.block_renderer.atlas())
            .unwrap();
        self.flat_pipeline = self
            .flat_provider
            .make_pipeline(ctx, (self.client_state.hud.lock().texture_atlas(), 0))?;
        self.egui_adapter
            .lock()
            .as_mut()
            .unwrap()
            .notify_resize(ctx)?;
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
    Active(Arc<RwLock<ActiveGame>>),
}
impl GameState {
    fn as_mut(&mut self) -> GameStateMutRef<'_> {
        match self {
            GameState::MainMenu => GameStateMutRef::MainMenu,
            GameState::Connecting(x) => GameStateMutRef::Connecting(x),
            GameState::Active(x) => GameStateMutRef::Active(x),
            GameState::ConnectError(x) => GameStateMutRef::ConnectError(x),
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
                Ok(Ok(mut game)) => {
                    let egui_adapter =
                        EguiAdapter::new(ctx, event_loop, game.client_state.egui.clone());
                    match egui_adapter {
                        Ok(x) => {
                            game.egui_adapter = Mutex::new(Some(x));
                            *self = GameState::Active(Arc::new(RwLock::new(game)));
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
        } else if let GameState::Active(game) = self {
            let game = game.read();
            let pending_error = game.client_state.pending_error.lock().take();
            if let Some(err) = pending_error {
                drop(game);
                *self = GameState::ConnectError(err)
            } else if game.client_state.cancel_requested() {
                if *game.client_state.wants_exit_from_game.lock() {
                    *control_flow = ControlFlow::ExitWithCode(0);
                }
                drop(game);
                *self = GameState::MainMenu;
            }
        }
    }
}

pub(crate) enum GameStateMutRef<'a> {
    MainMenu,
    Connecting(&'a mut ConnectionState),
    ConnectError(&'a mut String),
    Active(&'a mut Arc<RwLock<ActiveGame>>),
}

fn ensure_send_sync<T: Send + Sync>(_: T) {}

enum FenceState {
    Fence(Arc<FenceSignalFuture<Box<dyn GpuFuture + Send + Sync>>>),
    Pristine,
    AwaitingFence,
}

struct FenceHolder {
    state: Mutex<FenceState>,
    condvar: Condvar,
}
impl FenceHolder {
    fn take(&self) -> Option<Arc<FenceSignalFuture<Box<dyn GpuFuture + Send + Sync>>>> {
        let mut lock = self.state.lock();
        loop {
            match std::mem::replace(&mut *lock, FenceState::AwaitingFence) {
                FenceState::Fence(x) => return Some(x),
                FenceState::Pristine => return None,
                FenceState::AwaitingFence => self.condvar.wait(&mut lock),
            }
        }
    }
    fn clone_current(&self) -> Option<Arc<FenceSignalFuture<Box<dyn GpuFuture + Send + Sync>>>> {
        let mut lock = self.state.lock();
        loop {
            match &*lock {
                FenceState::Fence(x) => return Some(x.clone()),
                FenceState::Pristine => return None,
                FenceState::AwaitingFence => self.condvar.wait(&mut lock),
            }
        }
    }
    fn put(&self, x: Arc<FenceSignalFuture<Box<dyn GpuFuture + Send + Sync>>>) {
        let mut lock = self.state.lock();
        *lock = FenceState::Fence(x);
        self.condvar.notify_one();
    }

    fn new() -> FenceHolder {
        FenceHolder {
            state: Mutex::new(FenceState::Pristine),
            condvar: Condvar::new(),
        }
    }

    fn set_pristine(&self) {
        let mut lock = self.state.lock();
        *lock = FenceState::Pristine;
        self.condvar.notify_one();
    }
}

pub struct GameRenderer {
    ctx: Arc<VulkanWindow>,
    settings: Arc<ArcSwap<GameSettings>>,
    game: Mutex<GameState>,

    main_menu: Mutex<MainMenu>,
    rt: Arc<tokio::runtime::Runtime>,
    render_threadpool: rayon::ThreadPool,
}
impl GameRenderer {
    pub(crate) fn create(event_loop: &EventLoop<()>) -> Result<GameRenderer> {
        let settings = Arc::new(ArcSwap::new(
            GameSettings::load_from_disk()?.unwrap_or_default().into(),
        ));

        let ctx = VulkanWindow::create(event_loop, &settings).unwrap();
        let rt = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap(),
        );
        let main_menu = MainMenu::new(&ctx, event_loop, settings.clone());
        Ok(GameRenderer {
            ctx: Arc::new(ctx),
            settings,
            game: Mutex::new(GameState::MainMenu),
            main_menu: Mutex::new(main_menu),
            rt,
            render_threadpool: rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap(),
        })
    }

    pub fn run_loop(self, event_loop: EventLoop<()>) {
        let mut resized = false;
        let mut recreate_swapchain = false;
        let frames_in_flight = self.ctx.swapchain_images.load().len();
        let fences: Arc<Vec<FenceHolder>> = Arc::new(Vec::from_iter(
            std::iter::repeat_with(|| FenceHolder::new()).take(frames_in_flight),
        ));
        let mut previous_fence_i = 0;
        let mut fence_queue = (u32::MAX, u32::MAX, u32::MAX);
        event_loop.run(move |event, event_loop, control_flow| {
            let mut game_lock = self.game.lock();
            game_lock.update_if_connected(&self.ctx, event_loop, control_flow);
            {
                let _span = span!("client_state handling window event");

                if let GameStateMutRef::Active(game) = game_lock.as_mut() {
                    let game = game.read();
                    let consumed = if let Event::WindowEvent {
                        window_id: _,
                        event,
                    } = &event
                    {
                        game.egui_adapter
                            .lock()
                            .as_mut()
                            .unwrap()
                            .window_event(event)
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
                        game.read().client_state.shutdown.cancel();
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
                        if self.ctx.window.has_focus()
                            && game.read().client_state.input.lock().is_mouse_captured()
                        {
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
                    } else {
                        self.ctx
                            .window
                            .set_cursor_grab(winit::window::CursorGrabMode::None)
                            .unwrap();
                        self.ctx.window.set_cursor_visible(true);
                    }

                    if resized || recreate_swapchain {
                        let _span = span!("Recreate swapchain");
                        let size = self.ctx.window.inner_size();
                        recreate_swapchain = false;
                        self.ctx.recreate_swapchain(size).unwrap();
                        if resized {
                            resized = false;

                            let new_viewport = Viewport {
                                dimensions: (size.width as f32, size.height as f32).into(),
                                ..self.ctx.viewport.load().as_ref().clone()
                            };

                            self.ctx.viewport.store(Arc::new(new_viewport));
                            if let GameStateMutRef::Active(game) = game_lock.as_mut() {
                                game.write().handle_resize(&self.ctx).unwrap();
                            }
                        }
                    }

                    let _swapchain_span = span!("Acquire swapchain image");
                    println!("acquiring swapchain image, previous {}, fence_queue_0 {}, fence_queue_1 {}", previous_fence_i, fence_queue.0, fence_queue.1);
                    if fence_queue.0 != u32::MAX {
                        println!("fenceQ {}", fence_queue.1);
                        if let Some(image_fence) = fences[fence_queue.1 as usize].clone_current() {
                            image_fence.wait(None).unwrap();
                        }
                    }
                    let (image_i, suboptimal, acquire_future) = match swapchain::acquire_next_image(
                        self.ctx.swapchain.load().clone(),
                        Some(Duration::from_secs(5)),
                    ) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            info!("Swapchain out of date");
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };
                    println!("acquired swapchain image, current {}, previous {}", image_i, previous_fence_i);
                    fence_queue = (fence_queue.1, fence_queue.2, image_i);
                    Client::running()
                        .expect("tracy client must be running")
                        .frame_mark();
                    if suboptimal {
                        recreate_swapchain = true;
                    }
                    println!("clone current {}", image_i);
                    if let Some(image_fence) = fences[image_i as usize].clone_current() {
                        image_fence.wait(None).unwrap();
                    }
                    println!("clone previous {}", previous_fence_i);
                    let previous_future = match fences[previous_fence_i as usize].take() {
                        // Create a NowFuture
                        None => {
                            let mut now = vulkano::sync::now(self.ctx.vk_device.clone());
                            now.cleanup_finished();
                            now.boxed_send_sync()
                        }
                        // Use the existing FenceSignalFuture
                        Some(fence) => fence.boxed_send_sync(),
                    };
                    drop(_swapchain_span);
                    let window_size = self.ctx.window.inner_size();

                    // From https://vulkano.rs/compute_pipeline/descriptor_sets.html:
                    // Once you have created a descriptor set, you may also use it with other pipelines,
                    // as long as the bindings' types match those the pipelines' shaders expect.
                    // But Vulkan requires that you provide a pipeline whenever you create a descriptor set;
                    // you cannot create one independently of any particular pipeline.
                    let framebuffer = self.ctx.framebuffers.load()[image_i as usize].clone();
                    if let GameStateMutRef::Active(game) = game_lock.as_mut() {
                        let game_clone = game.clone();
                        let ctx_clone = self.ctx.clone();
                        let fences_clone = fences.clone();
                        let work = move || {
                            let command_buf_builder = ctx_clone.start_command_buffer().unwrap();
                            let command_buffers = game_clone.read().build_command_buffers(
                                window_size,
                                &ctx_clone,
                                framebuffer.clone(),
                                command_buf_builder,
                            );
                            Self::submit_to_vulkan(
                                previous_future,
                                acquire_future,
                                &ctx_clone,
                                image_i,
                                &fences_clone,
                                command_buffers,
                            );
                        };
                        self.render_threadpool.spawn(work);
                    } else {
                        let mut command_buf_builder = self.ctx.start_command_buffer().unwrap();
                        self.ctx
                            .start_render_pass(
                                &mut command_buf_builder,
                                framebuffer,
                                [0.0, 0.0, 0.0, 0.0],
                            )
                            .unwrap();
                        if let Some(connection_settings) = self.main_menu.lock().draw(
                            &self.ctx,
                            &mut game_lock,
                            &mut command_buf_builder,
                        ) {
                            self.start_connection(connection_settings);
                        }
                        let command_buffers = finish_command_buffer(command_buf_builder).unwrap();
                        Self::submit_to_vulkan(
                            previous_future,
                            acquire_future,
                            &self.ctx,
                            image_i,
                            &fences,
                            command_buffers,
                        );
                    };

                    previous_fence_i = image_i;
                }
                _ => {}
            }
        })
    }

    fn submit_to_vulkan(
        previous_future: Box<dyn GpuFuture + Send + Sync>,
        acquire_future: SwapchainAcquireFuture,
        ctx: &VulkanWindow,
        image_i: u32,
        fences: &Vec<FenceHolder>,
        command_buffers: PrimaryAutoCommandBuffer,
    ) -> SubmitResult {
        let mut result = SubmitResult::Success;
        let _span = span!("submit to Vulkan");
        let future = previous_future
            .join(acquire_future)
            .then_execute(ctx.queue.clone(), command_buffers)
            .unwrap()
            .then_swapchain_present(
                ctx.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(ctx.swapchain.load().clone(), image_i),
            )
            .boxed_send_sync()
            .then_signal_fence_and_flush();
        let new_fence = match future {
            Ok(value) => Some(Arc::new(value)),
            Err(FlushError::OutOfDate) => {
                result = SubmitResult::NeedsRecreateSwapchain;
                None
            }
            Err(e) => {
                log::error!("failed to flush future: {e}");
                None
            }
        };
        if let Some(new_fence) = new_fence {
            println!("set fence {}", image_i);
            fences[image_i as usize].put(new_fence);
        } else {
            println!("set pristine {}", image_i);
            fences[image_i as usize].set_pristine();
        }
        result
    }

    fn start_connection(&self, connection: ConnectionSettings) {
        let game_settings = self.settings.clone();
        {
            let ctx = self.ctx.clone();
            self.rt.spawn(async move {
                let active_game = connect_impl(
                    &ctx,
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

enum SubmitResult {
    Success,
    NeedsRecreateSwapchain,
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

pub(crate) struct ConnectionSettings {
    pub(crate) host: String,
    pub(crate) user: String,
    pub(crate) pass: String,
    pub(crate) register: bool,

    pub(crate) progress: watch::Sender<(f32, String)>,
    pub(crate) result: oneshot::Sender<Result<ActiveGame>>,
}

async fn connect_impl(
    ctx: &VulkanWindow,
    settings: Arc<ArcSwap<GameSettings>>,
    server_addr: String,
    username: String,
    password: String,
    register: bool,
    mut progress: watch::Sender<(f32, String)>,
) -> Result<ActiveGame> {
    progress.send((0.1, format!("Connecting to {}", server_addr)))?;
    let client_state = net_client::connect_game(
        server_addr,
        username,
        password,
        register,
        settings,
        &ctx,
        &mut progress,
    )
    .await?;

    let cube_provider = cube_geometry::CubePipelineProvider::new(ctx.vk_device.clone())?;
    let cube_pipeline = cube_provider.make_pipeline(&ctx, client_state.block_renderer.atlas())?;

    let flat_provider = flat_texture::FlatTexPipelineProvider::new(ctx.vk_device.clone())?;
    let flat_pipeline =
        flat_provider.make_pipeline(&ctx, (client_state.hud.lock().texture_atlas(), 0))?;

    let game = ActiveGame {
        cube_provider,
        cube_pipeline,
        flat_provider,
        flat_pipeline,
        client_state,
        egui_adapter: Mutex::new(None),
    };

    Ok(game)
}
