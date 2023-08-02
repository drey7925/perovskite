use arc_swap::ArcSwap;
use cgmath::{vec3, Vector3, Zero};
use cuberef_core::coordinates::{BlockCoordinate, PlayerPositionUpdate};
use rand::{self, Rng};
use std::{sync::Arc, time::Duration};
use tokio::{sync::watch, task::JoinHandle};
use tokio_util::sync::CancellationToken;

use winit::event_loop::EventLoop;

use crate::{
    game_state::{settings::GameSettings, DigTapAction},
    net_client,
    vulkan::{VulkanContext},
};

pub struct Loadtester {
    runtime: tokio::runtime::Runtime,
    workers: Vec<JoinHandle<()>>,
    vk_ctx: Arc<VulkanContext>,
    settings: Arc<ArcSwap<GameSettings>>,
    cancel: CancellationToken,
}
impl Loadtester {
    pub fn new() -> Self {
        Self {
            runtime: tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap(),
            workers: vec![],
            vk_ctx: Arc::new(
                VulkanContext::create(Box::<EventLoop<()>>::leak(Box::new(EventLoop::new())))
                    .unwrap(),
            ),
            settings: Arc::new(ArcSwap::new(
                GameSettings::load_from_disk()
                    .unwrap()
                    .unwrap_or_default()
                    .into(),
            )),
            cancel: CancellationToken::new(),
        }
    }
    pub fn start_worker(&mut self, addr: String, username: String) {
        let vk_ctx = self.vk_ctx.clone();
        let cancel = self.cancel.clone();
        let settings = self.settings.clone();
        self.workers.push(self.runtime.spawn(async move {
            let mut progress = watch::channel((0.0, "Connecting to server...".to_string()));
            let cs = net_client::connect_game(
                addr,
                username,
                "test_pass".to_string(),
                true,
                settings,
                &vk_ctx,
                &mut progress.0,
            )
            .await
            .unwrap();
            log::info!("Loadtest worker connected");
            {
                let mut rng = rand::thread_rng();
                cs.physics_state.lock().set_position(vec3(
                    rng.gen_range(-1000000.0..1000000.0),
                    rng.gen_range(-1000000.0..1000000.0),
                    rng.gen_range(-1000000.0..1000000.0),
                ));
            }
            while !cancel.is_cancelled() {
                tokio::time::sleep(Duration::from_secs_f32(0.1)).await;
                let pos = {
                    let mut phys = cs.physics_state.lock();
                    let mut rng = rand::thread_rng();
                    let pos = phys.pos();
                    let new_pos = pos
                        + vec3(
                            rng.gen_range(-0.5..0.5),
                            rng.gen_range(-0.5..0.5),
                            rng.gen_range(-0.5..0.5),
                        );
                    phys.set_position(new_pos);
                    pos
                };
                cs.actions.send(crate::game_state::GameAction::Dig(DigTapAction {
                    target: BlockCoordinate::new(pos.x as i32, pos.y as i32, pos.z as i32),
                    prev: None,
                    item_slot: 0,
                    player_pos: PlayerPositionUpdate {
                        position: pos,
                        velocity: Vector3::zero(),
                        face_direction: (0., 0.),
                    },
                })).await.unwrap();
            }
            panic!();
        }))
    }
}
impl Default for Loadtester {
    fn default() -> Self {
        Self::new()
    }
}
