use std::sync::Arc;

use perovskite_core::protocol::debug::{
    perovskite_debug_server::PerovskiteDebug, DebugBlockDef, FindBlockDefsReq, FindBlockDefsResp,
    FindItemDefsReq, FindItemDefsResp,
};
use tonic::{Response, Status};

use crate::game_state::GameState;

pub(crate) struct DebugServer {
    game_state: Arc<GameState>,
}
impl DebugServer {
    pub(crate) fn new(game_state: Arc<GameState>) -> Self {
        DebugServer { game_state }
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
        todo!()
    }
}
