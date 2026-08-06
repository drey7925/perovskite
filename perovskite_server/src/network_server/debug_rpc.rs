use std::{sync::Arc, time::Duration};

use circular_buffer::CircularBuffer;
use perovskite_core::{
    coordinates::{BlockCoordinate, ChunkOffset},
    protocol::debug::{
        perovskite_debug_server::PerovskiteDebug, DebugBlockDef, DebugItemDef, FindBlockDefsReq,
        FindBlockDefsResp, FindItemDefsReq, FindItemDefsResp, LastEventsReq, LastEventsResp,
        SetWorkingCoordReq, SetWorkingCoordResp,
    },
};
use tonic::{Response, Status};

use crate::game_state::{
    player::{PlayerContext, PlayerEventReceiver},
    GameState,
};

const FAKE_PLAYER_NAME: &str = "internal:debug_client";
const CHUNK_LOAD_DISTANCE: i32 = 6;

pub(crate) struct DebugServer {
    player_ctx: PlayerContext,
    game_state: Arc<GameState>,
    background_work: tokio::task::JoinHandle<()>,
    last_events: Arc<tokio::sync::Mutex<CircularBuffer<64, String>>>,
    working_coord: Arc<tokio::sync::Mutex<BlockCoordinate>>,
}
impl DebugServer {
    pub(crate) fn new(game_state: Arc<GameState>) -> anyhow::Result<Self> {
        let (player_ctx, event_rx) = game_state
            .player_manager()
            .clone()
            .connect(FAKE_PLAYER_NAME)
            .unwrap();

        let event_buf = Arc::new(tokio::sync::Mutex::new(CircularBuffer::new()));

        let working_coord = Arc::new(tokio::sync::Mutex::new(BlockCoordinate {
            x: 0,
            y: 0,
            z: 0,
        }));
        let background_work = tokio::spawn(do_background_work(
            game_state.clone(),
            event_rx,
            event_buf.clone(),
            working_coord.clone(),
        ));

        Ok(DebugServer {
            player_ctx,
            game_state,
            background_work,
            last_events: event_buf,
            working_coord,
        })
    }
}

async fn do_background_work(
    game_state: Arc<GameState>,
    mut event_rx: PlayerEventReceiver,
    last_events: Arc<tokio::sync::Mutex<CircularBuffer<64, String>>>,
    working_coord: Arc<tokio::sync::Mutex<BlockCoordinate>>,
) {
    let mut keepalive_interval = tokio::time::interval(Duration::from_millis(100));
    while !game_state.is_shutting_down() {
        tokio::select! {
            _ = keepalive_interval.tick() => {
                let center = working_coord.lock().await.chunk();
                for dx in -CHUNK_LOAD_DISTANCE..=CHUNK_LOAD_DISTANCE {
                    for dz in -CHUNK_LOAD_DISTANCE..=CHUNK_LOAD_DISTANCE {
                        for dy in -CHUNK_LOAD_DISTANCE..=CHUNK_LOAD_DISTANCE {
                            if let Some(chunk) = center.try_delta(dx, dy, dz) {
                                game_state.game_map().get_block(chunk.with_offset(ChunkOffset::new(0, 0, 0))).unwrap();
                            }
                        }
                    }
                }
            },
            Some(event) = event_rx.rx.recv() => {
                last_events.lock().await.push_back(format!("PlayerEvent {event:?}")).unwrap();
            },
        }
    }
}

#[tonic::async_trait]
impl PerovskiteDebug for DebugServer {
    async fn find_block_defs(
        &self,
        request: tonic::Request<FindBlockDefsReq>,
    ) -> Result<Response<FindBlockDefsResp>, Status> {
        let re = regex::Regex::new(&request.into_inner().name_regex)
            .map_err(|x| Status::invalid_argument(format!("Invalid regex: {:?}", x)))?;
        let mut response = FindBlockDefsResp::default();
        for block_type in self.game_state.block_types().all_types() {
            if re.is_match(&block_type.short_name()) {
                response.entries.push(DebugBlockDef {
                    name: Some(block_type.short_name().to_string()),
                    client_info: Some(block_type.client_info.clone()),
                    present_handlers: block_type
                        .debug_handler_list()
                        .into_iter()
                        .map(|x| x.to_string())
                        .collect(),
                });
            }
        }
        Ok(response.into())
    }
    async fn find_item_defs(
        &self,
        request: tonic::Request<FindItemDefsReq>,
    ) -> Result<Response<FindItemDefsResp>, Status> {
        let re = regex::Regex::new(&request.into_inner().name_regex)
            .map_err(|x| Status::invalid_argument(format!("Invalid regex: {:?}", x)))?;
        let mut response = FindItemDefsResp::default();
        for item in self.game_state.item_manager().registered_items() {
            if re.is_match(&item.proto.short_name) {
                response.entries.push(DebugItemDef {
                    name: Some(item.proto.short_name.to_string()),
                    client_info: Some(item.proto.clone()),
                    present_handlers: item
                        .debug_handler_list()
                        .into_iter()
                        .map(|x| x.to_string())
                        .collect(),
                });
            }
        }
        Ok(response.into())
    }

    async fn last_events(
        &self,
        request: tonic::Request<LastEventsReq>,
    ) -> Result<Response<LastEventsResp>, Status> {
        let req = request.into_inner();
        let n = req.n.unwrap_or(64) as usize;
        let mut events = self
            .last_events
            .lock()
            .await
            .iter()
            .rev()
            .take(n)
            .cloned()
            .collect::<Vec<_>>();
        // we reversed to take the last N, reverse again so they're in order
        events.reverse();
        Ok(Response::new(LastEventsResp { events }).into())
    }

    async fn set_working_coord(
        &self,
        request: tonic::Request<SetWorkingCoordReq>,
    ) -> Result<Response<SetWorkingCoordResp>, Status> {
        let req = request.into_inner();
        let mut working_coord = self.working_coord.lock().await;
        *working_coord = BlockCoordinate {
            x: req.x,
            y: req.y,
            z: req.z,
        };
        Ok(Response::new(SetWorkingCoordResp {}).into())
    }
}
