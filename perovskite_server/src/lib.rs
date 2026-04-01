#![allow(dead_code)]
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

pub mod database;
/// Various utilities to convert between formats.
pub mod formats;
/// The state of the actual game world and all the things inside it
pub mod game_state;
/// Media files that the client will need
pub mod media;
/// The server that powers the multiplayer and player interaction experience
pub mod network_server;
/// The actual builders and lifecycle of a game server
pub mod server;

use anyhow::Result;
use bytemuck::Zeroable;
use std::{
    future::Future,
    ops::{Deref, DerefMut},
};
use tokio::task::JoinHandle;

#[cfg(tokio_unstable)]
pub(crate) fn spawn_async<T>(name: &str, task: T) -> Result<JoinHandle<T::Output>>
where
    T: Future + Send + 'static,
    T::Output: Send + 'static,
{
    use std::thread::JoinHandle;

    Ok(tokio::task::Builder::new().name(name).spawn(task)?)
}

#[cfg(not(tokio_unstable))]
pub(crate) fn spawn_async<T>(_name: &str, task: T) -> Result<JoinHandle<T::Output>>
where
    T: Future + Send + 'static,
    T::Output: Send + 'static,
{
    Ok(tokio::task::spawn(task))
}

#[derive(Zeroable)]
#[zeroable(bound = "T: Zeroable")]
#[repr(align(64))]
struct CachelineAligned<T>(T);
impl<T> Deref for CachelineAligned<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T> DerefMut for CachelineAligned<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Internal token to statically prove that we're inside a tokio blocking region.
///
/// This is meant as a best-effort tool to avoid bugs, not as a hard safety feature.
/// The token is trivially launderable by copying it, so don't rely on it for security.
#[derive(Debug)]
struct BlockingRegionToken;

#[inline]

fn block_in_place<F, T>(f: F) -> T
where
    F: FnOnce(&BlockingRegionToken) -> T,
{
    tokio::task::block_in_place(|| f(&BlockingRegionToken))
}
