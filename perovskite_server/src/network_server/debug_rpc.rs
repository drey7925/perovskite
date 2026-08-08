use std::{sync::Arc, time::Duration};

use cgmath::Vector3;
use circular_buffer::CircularBuffer;
use perovskite_core::{
    block_id::BlockId,
    coordinates::{BlockCoordinate, ChunkOffset, PlayerPositionUpdate},
    protocol::debug::{
        perovskite_debug_server::PerovskiteDebug, DebugBlockDef, DebugItemDef, FindBlockDefsReq,
        FindBlockDefsResp, FindItemDefsReq, FindItemDefsResp, GetBlockByIdReq, GetBlockByIdResp,
        GetBlockReq, GetBlockResp, LastEventsReq, LastEventsResp, SetBlockReq, SetBlockResp,
        SetWorkingCoordReq, SetWorkingCoordResp,
    },
};
use tonic::{Response, Status};

use crate::game_state::{
    blocks::BlockType,
    game_map::serialize_single_client_extended_data,
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

/// Builds the debug-facing summary of a block type, shared by RPCs that look up block types
/// either by name (`find_block_defs`) or by numeric ID (`get_block_by_id`, `get_block`).
fn debug_block_def(block_type: &BlockType) -> DebugBlockDef {
    DebugBlockDef {
        name: Some(block_type.short_name().to_string()),
        client_info: Some(block_type.client_info.clone()),
        present_handlers: block_type
            .debug_handler_list()
            .into_iter()
            .map(|x| x.to_string())
            .collect(),
    }
}

/// Resolves a `BlockId` (type + variant) into its `DebugBlockDef`, variant, and a human-readable
/// one-line description (e.g. "default:dirt with variant 0x4"), or a `NotFound` status if the
/// block type doesn't exist.
fn describe_block_id(
    game_state: &GameState,
    block_id: BlockId,
) -> Result<(DebugBlockDef, u16, String), Status> {
    let (block_type, variant) = game_state
        .block_types()
        .get_block_by_id(block_id)
        .map_err(|e| Status::not_found(format!("Block id 0x{:x}: {e:#}", block_id.0)))?;
    let description = format!("{} with variant 0x{:x}", block_type.short_name(), variant);
    Ok((debug_block_def(block_type), variant, description))
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
                response.entries.push(debug_block_def(block_type));
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
        tokio::task::block_in_place(|| {
            self.player_ctx.update_client_position_state(
                PlayerPositionUpdate {
                    position: Vector3::new(req.x as f64, req.y as f64, req.z as f64),
                    velocity: Vector3::new(0.0, 0.0, 0.0),
                    face_direction: (0.0, 0.0),
                },
                0,
            )
        });
        Ok(Response::new(SetWorkingCoordResp {}).into())
    }

    async fn get_block_by_id(
        &self,
        request: tonic::Request<GetBlockByIdReq>,
    ) -> Result<Response<GetBlockByIdResp>, Status> {
        let req = request.into_inner();
        let (def, variant, description) =
            describe_block_id(&self.game_state, BlockId(req.block_id))?;
        Ok(Response::new(GetBlockByIdResp {
            def: Some(def),
            variant: variant as u32,
            description,
        }))
    }

    async fn get_block(
        &self,
        request: tonic::Request<GetBlockReq>,
    ) -> Result<Response<GetBlockResp>, Status> {
        let req = request.into_inner();
        let coord = BlockCoordinate {
            x: req.x,
            y: req.y,
            z: req.z,
        };
        let (block_id, extended_data) =
            tokio::task::block_in_place(|| self.get_block_for_debug(coord, req.extended_data))
                .map_err(|e| {
                    Status::internal(format!("Failed to get block at {coord:?}: {e:#}"))
                })?;
        let (def, variant, description) = describe_block_id(&self.game_state, block_id)?;
        Ok(Response::new(GetBlockResp {
            block_id: block_id.0,
            def: Some(def),
            variant: variant as u32,
            description,
            extended_data,
        }))
    }

    async fn set_block(
        &self,
        request: tonic::Request<SetBlockReq>,
    ) -> Result<Response<SetBlockResp>, Status> {
        let req = request.into_inner();
        let coord = BlockCoordinate {
            x: req.x,
            y: req.y,
            z: req.z,
        };
        let variant = req.variant as u16;
        let block_id = self
            .game_state
            .block_types()
            .get_by_name(&req.block_name)
            .ok_or_else(|| Status::not_found(format!("No block type named {:?}", req.block_name)))?
            .with_variant(variant)
            .map_err(|e| {
                Status::invalid_argument(format!(
                    "Invalid variant 0x{:x} for block {:?}: {e:#}",
                    req.variant, req.block_name
                ))
            })?;
        let (prev_id, _prev_ext) = tokio::task::block_in_place(|| {
            self.game_state.game_map().set_block(coord, block_id, None)
        })
        .map_err(|e| Status::internal(format!("Failed to set block at {coord:?}: {e:#}")))?;
        Ok(Response::new(SetBlockResp {
            previous_block_id: prev_id.0,
            previous_description: self.game_state.block_types().human_short_name(prev_id),
        }))
    }
}

impl DebugServer {
    fn get_block_for_debug(
        &self,
        coord: BlockCoordinate,
        extended_data: bool,
    ) -> anyhow::Result<(BlockId, Option<String>)> {
        if extended_data {
            self.game_state
                .game_map()
                .get_block_with_extended_data(coord, |id, ext| {
                    let inventories: Vec<_> = ext
                        .inventories
                        .iter()
                        .map(|(name, inventory)| format!("{name}: {inventory:?}"))
                        .collect();
                    let custom = ext
                        .custom_data
                        .as_ref()
                        .map(|x| x.debug_as_string())
                        .unwrap_or_else(|| "None".to_string());
                    let client = match serialize_single_client_extended_data(
                        coord,
                        ext,
                        self.game_state.block_types(),
                        id,
                    ) {
                        Ok(Some(x)) => format!(
                            "Hover text: {:?}, block texts: {:?}",
                            x.block_text,
                            x.rendered_text
                                .iter()
                                .map(|x| x.spans.iter().map(|x| &x.text))
                                .flatten()
                                .collect::<Vec<_>>()
                        ),
                        Ok(None) => "None".to_string(),
                        Err(e) => format!("Error: {e:?}"),
                    };
                    Ok(Some(format!(
                        "Simple: {:?}\nInventories: {}\nCustom: {}\nClient: {}",
                        &ext.simple_data,
                        inventories.join(", "),
                        custom,
                        client
                    )))
                })
        } else {
            let id = self.game_state.game_map().get_block(coord)?;
            Ok((id, None))
        }
    }
}
