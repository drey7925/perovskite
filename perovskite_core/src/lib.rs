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

pub mod perovskite {
    pub mod protocol {
        pub mod blocks {
            tonic::include_proto!("perovskite.protocol.blocks");
        }
        pub mod map {
            tonic::include_proto!("perovskite.protocol.map");
        }
        pub mod game_rpc {
            tonic::include_proto!("perovskite.protocol.game_rpc");
        }
        pub mod coordinates {
            tonic::include_proto!("perovskite.protocol.coordinates");
        }
        pub mod render {
            tonic::include_proto!("perovskite.protocol.render");
        }
        pub mod items {
            tonic::include_proto!("perovskite.protocol.items");
        }
        pub mod players {
            tonic::include_proto!("perovskite.protocol.players");
        }
        pub mod ui {
            tonic::include_proto!("perovskite.protocol.ui");
        }
        pub mod entities {
            tonic::include_proto!("perovskite.protocol.entities");
        }
        pub const DESCRIPTOR_SET: &[u8] =
            tonic::include_file_descriptor_set!("perovskite_descriptor");
    }
}
pub use perovskite::protocol;

pub mod auth;
pub mod block_id;
pub mod chat;
pub mod constants;
pub mod coordinates;
pub mod items;
pub mod lighting;
pub mod time;
