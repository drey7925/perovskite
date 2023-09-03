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

use std::{time::Duration};


use perovskite_core::{coordinates::{BlockCoordinate}};
use log::{warn, info};
use tokio::{runtime::Runtime, time::sleep};

use crate::{
    game_state::{
        testutils::testonly_make_gamestate,
    },
};

#[cfg(test)]
#[ctor::ctor]
fn init() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
}


#[test]
fn end_to_end_1() {
    let game_state = testonly_make_gamestate();
    
    let rt = Runtime::new().unwrap();
    let mut receiver = game_state.map().subscribe();
    let _recv_handler = rt.spawn(async move {
        loop {
            match receiver.recv().await {
                Ok(x) => info!("BlockUpdate {:?}", x),
                Err(e) => warn!("RecvError: {:?}", e),
            }
        }
    });


    println!(
        "at 0 0 -5 {}",
        game_state
            .map()
            .block_type_manager()
            .get_block(
                &game_state
                    .map()
                    .get_block(BlockCoordinate::new(0, 0, -5))
                    .unwrap()
            )
            .unwrap()
            .0
            .short_name()
    );
    println!(
        "at 0 0 8 {}",
        game_state
            .map()
            .block_type_manager()
            .get_block(
                &game_state
                    .map()
                    .get_block(BlockCoordinate::new(0, 0, 8))
                    .unwrap()
            )
            .unwrap()
            .0
            .short_name()
    );
    game_state
        .map()
        .unload_chunk(BlockCoordinate::new(0, 0, 8).chunk())
        .unwrap();
    println!(
        "at 0 0 9 {}",
        game_state
            .map()
            .block_type_manager()
            .get_block(
                &game_state
                    .map()
                    .get_block(BlockCoordinate::new(0, 0, 9))
                    .unwrap()
            )
            .unwrap()
            .0
            .short_name()
    );
    game_state
        .map()
        .dig_block(
            BlockCoordinate::new(0, 0, 8),
            crate::game_state::event::EventInitiator::Engine,
            None,
        )
        .unwrap();
    println!("dig");
    println!(
        "at 0 0 8 {}",
        game_state
            .map()
            .block_type_manager()
            .get_block(
                &game_state
                    .map()
                    .get_block(BlockCoordinate::new(0, 0, 8))
                    .unwrap()
            )
            .unwrap()
            .0
            .short_name()
    );
    println!(
        "at 0 0 7 {}",
        game_state
            .map()
            .block_type_manager()
            .get_block(
                &game_state
                    .map()
                    .get_block(BlockCoordinate::new(0, 0, 7))
                    .unwrap()
            )
            .unwrap()
            .0
            .short_name()
    );
    println!(
        "at 0 0 6 {}",
        game_state
            .map()
            .block_type_manager()
            .get_block(
                &game_state
                    .map()
                    .get_block(BlockCoordinate::new(0, 0, 6))
                    .unwrap()
            )
            .unwrap()
            .0
            .short_name()
    );
    game_state
        .map()
        .dig_block(
            BlockCoordinate::new(0, 1, 3),
            crate::game_state::event::EventInitiator::Engine,
            None,
        )
        .unwrap();
    println!("dig!");
    println!(
        "at 0 1 4 {}",
        game_state
            .map()
            .block_type_manager()
            .get_block(
                &game_state
                    .map()
                    .get_block(BlockCoordinate::new(0, 1, 4))
                    .unwrap()
            )
            .unwrap()
            .0
            .short_name()
    );
    println!(
        "at 0 1 3 {}",
        game_state
            .map()
            .block_type_manager()
            .get_block(
                &game_state
                    .map()
                    .get_block(BlockCoordinate::new(0, 1, 3))
                    .unwrap()
            )
            .unwrap()
            .0
            .short_name()
    );
    println!(
        "at 0 1 2 {}",
        game_state
            .map()
            .block_type_manager()
            .get_block(
                &game_state
                    .map()
                    .get_block(BlockCoordinate::new(0, 1, 2))
                    .unwrap()
            )
            .unwrap()
            .0
            .short_name()
    );
}

#[test]
fn benchmarks() {
 
    let game_state = testonly_make_gamestate();

    let rt = Runtime::new().unwrap();
    let mut receiver = game_state.map().subscribe();
    let _recv_handler = rt.spawn(async move {
        loop {
            match receiver.recv().await {
                Ok(_) => {sleep(Duration::from_millis(100)).await;},
                Err(_) => {} // this was spammy hence commented out: warn!("RecvError: {:?}", e),
            }
        }
    });
    let options = microbench::Options::default();
    microbench::bench(&options, "get_block", || {
        game_state
            .map()
            .get_block(BlockCoordinate::new(0, 1, 2))
            .unwrap()
    });

    let handle = game_state
        .map()
        .block_type_manager()
        .get_by_name("test:dirt")
        .unwrap();
    microbench::bench(&options, "set_block_handle", || {
        game_state
            .map()
            .set_block(BlockCoordinate::new(0, 1, 2), handle, None)
            .unwrap()
    });

    let handle = game_state
        .map()
        .block_type_manager()
        .get_by_name("test:dirt")
        .unwrap();
    microbench::bench(&options, "cas_block_handle", || {
        game_state
            .map()
            .compare_and_set_block(BlockCoordinate::new(0, 1, 2), handle, handle, None, false)
            .unwrap()
    });

    let _handle = game_state
        .map()
        .block_type_manager()
        .get_by_name("test:dirt")
        .unwrap();
    microbench::bench(&options, "mutate_empty_closure", || {
        game_state
            .map()
            .mutate_block_atomically(BlockCoordinate::new(0, 1, 2), |_, _| Ok(()))
            .unwrap()
    });
    let options = microbench::Options::default();
    microbench::bench(&options, "get_with_unloads", || {
        game_state
            .map()
            .get_block(BlockCoordinate::new(0, 1, 2))
            .unwrap();
        game_state
            .map()
            .unload_chunk(BlockCoordinate::new(0, 1, 2).chunk())
    });
}
