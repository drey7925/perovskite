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

use std::{env, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let descriptor_path =
        PathBuf::from(env::var("OUT_DIR").unwrap()).join("perovskite_descriptor.bin");
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .file_descriptor_set_path(descriptor_path)
        .compile(
            &[
                "proto/coordinates.proto",
                "proto/blocks.proto",
                "proto/game_rpc.proto",
                "proto/items.proto",
                "proto/mapchunk.proto",
                "proto/players.proto",
                "proto/render.proto",
                "proto/ui.proto",
                "proto/entities.proto",
            ],
            &["proto"],
        )?;
    Ok(())
}
