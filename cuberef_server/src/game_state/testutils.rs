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

use std::{net::SocketAddr, sync::Arc};

use anyhow::Context;
use cuberef_core::{
    constants::{
        block_groups::{DEFAULT_SOLID, GRANULAR},
        items::default_item_interaction_rules,
    },
    coordinates::{BlockCoordinate, ChunkCoordinate, ChunkOffset},
    protocol::render::TextureReference,
    protocol::{
        blocks::{
            block_type_def::{PhysicsInfo, RenderInfo},
            BlockTypeDef, CubeRenderInfo, Empty,
        },
        items::item_def::QuantityType,
    },
};
use log::warn;


use crate::{
    game_state::{
        blocks::{BlockType, BlockTypeManager},
        game_map::{CasOutcome, MapChunk},
        items::{DigResult, ItemStack},
        mapgen::MapgenInterface,
        GameState,
    },
    network_server::auth::{AuthOutcome, AuthService, RegisterOutcome, TokenOutcome},
    server::ServerBuilder,
};



const EMPTY: Empty = Empty {};

pub struct FakeAuth {}
impl AuthService for FakeAuth {
    fn create_account(
        &self,
        username: &str,
        _password_hash: &[u8],
        remote_addr: Option<SocketAddr>,
    ) -> anyhow::Result<crate::network_server::auth::RegisterOutcome> {
        warn!(
            "FakeAuth allowing account creation for {} from {:?}",
            username, remote_addr
        );
        Ok(RegisterOutcome::Success("faketoken".to_string()))
    }

    fn authenticate_user(
        &self,
        username: &str,
        _password_hash: &[u8],
        remote_addr: Option<SocketAddr>,
    ) -> anyhow::Result<crate::network_server::auth::AuthOutcome> {
        warn!(
            "FakeAuth allowing authentication for {} from {:?}",
            username, remote_addr
        );
        Ok(AuthOutcome::Success("faketoken".to_string()))
    }

    fn check_token(
        &self,
        username: &str,
        _token: &str,
        remote_addr: Option<SocketAddr>,
    ) -> anyhow::Result<TokenOutcome> {
        warn!(
            "FakeAuth allowing fake token for {} from {:?}",
            username, remote_addr
        );
        Ok(TokenOutcome::Success)
    }
}

pub struct FakeMapgen {
    pub block_type_manager: Arc<BlockTypeManager>,
}

impl MapgenInterface for FakeMapgen {
    fn fill_chunk(&self, coord: ChunkCoordinate, chunk: &mut MapChunk) {
        let grass = self
            .block_type_manager
            .make_block_name("test:grass".to_string());
        let dirt = self
            .block_type_manager
            .make_block_name("test:dirt".to_string());
        let air = self
            .block_type_manager
            .make_block_name("test:air".to_string());

        for x in 0..16 {
            for y in 0..16 {
                for z in 0..16 {
                    let offset = ChunkOffset { x, y, z };
                    let coord2 = coord.with_offset(offset);
                    assert!(coord2.chunk() == coord);

                    let landscape =
                        0.002 * (coord2.x as f64).powi(3) + 0.55 * (coord2.z as f64).cbrt();
                    let ii = coord2.y - (landscape.clamp(-10000000., 1000000.) as i32);
                    let block_name = match ii {
                        8 => &grass,
                        i32::MIN..=7 => &dirt,
                        _ => &air,
                    };

                    chunk.block_ids[offset.as_index()] = self
                        .block_type_manager
                        .resolve_name(block_name)
                        .unwrap()
                        .id()
                        .into()
                }
            }
        }
    }
}

fn testonly_make_tex(name: &str) -> Option<TextureReference> {
    Some(TextureReference {
        texture_name: name.to_string(),
    })
}

pub fn register_test_blocks_and_items(builder: &mut ServerBuilder) {
    let grass = builder.blocks().make_block_name("test:grass".to_string());
    let air = builder.blocks().make_block_name("test:air".to_string());
    let air2 = builder.blocks().make_block_name("test:air".to_string());

    builder
        .items()
        .register_item(super::items::Item {
            proto: cuberef_core::protocol::items::ItemDef {
                short_name: "test:dirt".to_string(),
                display_name: "Dirt".to_string(),
                inventory_texture: testonly_make_tex("dirt.png"),
                groups: vec![],
                interaction_rules: default_item_interaction_rules(),
                quantity_type: Some(QuantityType::Stack(256)),
            },
            dig_handler: Some(Box::new(move |ctx, coord, bth, stack| {
                println!("dirt dig: {:?}", bth);
                Ok(DigResult {
                    updated_tool: Some(stack.clone()),
                    obtained_items: ctx.game_map().dig_block(
                        coord,
                        ctx.initiator.clone(),
                        Some(stack),
                    )?,
                })
            })),
            tap_handler: None,
            place_handler: Some(Box::new(move |ctx, coord, _, stack| {
                if stack.proto().quantity == 0 {
                    return Ok(None);
                }
                match ctx
                    .game_map()
                    .compare_and_set_block(coord, "test:air", "test:dirt", None, false)?
                    .0
                {
                    super::game_map::CasOutcome::Match => Ok(stack.decrement()),
                    super::game_map::CasOutcome::Mismatch => Ok(Some(stack.clone())),
                }
            })),
        })
        .unwrap();

    let dirt_block = builder
        .blocks()
        .register_block(BlockType {
            client_info: BlockTypeDef {
                id: 0,
                short_name: "test:dirt".to_string(),
                render_info: Some(RenderInfo::Cube(CubeRenderInfo {
                    tex_left: testonly_make_tex("dirt.png"),
                    tex_right: testonly_make_tex("dirt.png"),
                    tex_top: testonly_make_tex("dirt.png"),
                    tex_bottom: testonly_make_tex("dirt.png"),
                    tex_front: testonly_make_tex("dirt.png"),
                    tex_back: testonly_make_tex("dirt.png"),
                })),
                physics_info: Some(PhysicsInfo::Solid(EMPTY)),
                base_dig_time: 1.0,
                groups: vec![String::from(DEFAULT_SOLID), String::from(GRANULAR)],
            },
            extended_data_handling: crate::game_state::blocks::ExtDataHandling::NoExtData,
            deserialize_extended_data_handler: None,
            serialize_extended_data_handler: None,
            extended_data_to_client_side: None,
            dig_handler_full: None,
            dig_handler_inline: Some(Box::new(move |ctx, bth, _ext, _tool| {
                println!("dirt dig");
                *bth = ctx
                    .block_types()
                    .resolve_name(&air)
                    .with_context(|| "couldn't find test:air")?;
                let item = ctx
                    .items()
                    .get_item("test:dirt")
                    .with_context(|| "dirt item not found")?;
                Ok(vec![ItemStack::new(item, 1)])
            })),
            tap_handler_full: None,
            tap_handler_inline: None,
            place_upon_handler: None,
            interact_key_handlers: vec![],
            is_unknown_block: false,
            block_type_manager_id: None,
        })
        .unwrap();
    let _dirt_grass_block = builder
        .blocks()
        .register_block(BlockType {
            client_info: BlockTypeDef {
                id: 0,
                short_name: "test:grass".to_string(),
                render_info: Some(RenderInfo::Cube(CubeRenderInfo {
                    tex_left: testonly_make_tex("dirt_grass.png"),
                    tex_right: testonly_make_tex("dirt_grass.png"),
                    tex_top: testonly_make_tex("grass.png"),
                    tex_bottom: testonly_make_tex("dirt.png"),
                    tex_front: testonly_make_tex("dirt_grass.png"),
                    tex_back: testonly_make_tex("dirt_grass.png"),
                })),
                physics_info: Some(PhysicsInfo::Solid(EMPTY)),
                base_dig_time: 1.0,
                groups: vec![String::from(DEFAULT_SOLID), String::from(GRANULAR)],
            },
            extended_data_handling: crate::game_state::blocks::ExtDataHandling::NoExtData,
            deserialize_extended_data_handler: None,
            serialize_extended_data_handler: None,
            extended_data_to_client_side: None,
            dig_handler_full: Some(Box::new(move |ctx, coord, _tool| {
                let grass = ctx
                    .game_map()
                    .block_type_manager()
                    .resolve_name(&grass)
                    .with_context(|| "can't touch grass")
                    .unwrap();
                let air = ctx
                    .game_map()
                    .block_type_manager()
                    .resolve_name(&air2)
                    .with_context(|| "can't find air")
                    .unwrap();

                println!("grass dig: cas");
                let (cas_outcome, _, _) = ctx
                    .game_map()
                    .compare_and_set_block(coord, grass, air, None, false)?;

                println!("grass dig: start atomic mutation");
                ctx.game_map().mutate_block_atomically(
                    BlockCoordinate {
                        x: coord.x,
                        y: coord.y - 1,
                        z: coord.z,
                    },
                    |bth, _ext| {
                        println!("grass dig: block below: {bth:?}");
                        if bth.equals_ignore_variant(dirt_block) {
                            *bth = grass;
                        }
                        Ok(())
                    },
                )?;

                println!("grass dig: end atomic mutation");
                match cas_outcome {
                    CasOutcome::Match => {
                        let item = ctx
                            .items()
                            .get_item("test:dirt")
                            .with_context(|| "dirt item not found")?;
                        Ok(vec![ItemStack::new(item, 1)])
                    }
                    CasOutcome::Mismatch => Ok(vec![]),
                }
            })),
            dig_handler_inline: None,
            tap_handler_full: None,
            tap_handler_inline: Some(Box::new(move |_, block_type, _, _| {
                *block_type = dirt_block;
                Ok(vec![])
            })),
            place_upon_handler: None,
            interact_key_handlers: vec![],
            is_unknown_block: false,
            block_type_manager_id: None,
        })
        .unwrap();
    let _air_block = builder
        .blocks()
        .register_block(BlockType {
            client_info: BlockTypeDef {
                id: 0,
                short_name: "test:air".to_string(),
                render_info: Some(RenderInfo::Empty(EMPTY)),
                physics_info: Some(PhysicsInfo::Air(EMPTY)),
                base_dig_time: 1.0,
                groups: vec![],
            },
            extended_data_handling: crate::game_state::blocks::ExtDataHandling::NoExtData,
            deserialize_extended_data_handler: None,
            serialize_extended_data_handler: None,
            extended_data_to_client_side: None,
            dig_handler_full: None,
            dig_handler_inline: Some(Box::new(|_, _, _, _| {
                println!("air dig handler");
                Ok(vec![])
            })),
            tap_handler_full: None,
            tap_handler_inline: None,
            place_upon_handler: None,
            interact_key_handlers: vec![],
            is_unknown_block: false,
            block_type_manager_id: None,
        })
        .unwrap();

    builder
        .media()
        .register_from_memory("dirt.png", include_bytes!("testonly_media/test_dirt.png"))
        .unwrap();
    builder
        .media()
        .register_from_memory(
            "dirt_grass.png",
            include_bytes!("testonly_media/test_dirt_grass.png"),
        )
        .unwrap();
    builder
        .media()
        .register_from_memory("grass.png", include_bytes!("testonly_media/test_grass.png"))
        .unwrap();
}

pub async fn testonly_finish_shutdown(gs: &GameState) {
    gs.finish_shutdown().await
}
