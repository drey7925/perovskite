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
use perovskite_client::client_main;

use mimalloc::MiMalloc;

#[cfg(not(all(feature = "tracy", feature = "tracy_malloc")))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[cfg(all(feature = "tracy", feature = "tracy_malloc"))]
#[global_allocator]
static GLOBAL: tracy_client::ProfiledAllocator<MiMalloc> =
    tracy_client::ProfiledAllocator::new(MiMalloc, 100);

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    client_main::run_client().unwrap();
}
