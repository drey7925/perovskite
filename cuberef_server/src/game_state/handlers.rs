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

use tracy_client::{span, span_location};

use super::event::EventInitiator;

/// Wrapper for handlers, eventually used for accounting, error handling, etc.
/// Currently a no-op
#[inline]
pub(crate) fn run_handler_impl<T, F>(closure: F, _name: &'static str, _initiator: EventInitiator) -> T
where
    F: FnOnce() -> T,
{
    // TODO tracing, etc
    (closure)()
}

#[macro_export]
macro_rules! run_handler {
    ($closure:expr, $name:literal, $initiator:expr $(,)?) => {
        {
            let _span = span!(concat!($name, " handler"));
            $crate::game_state::handlers::run_handler_impl($closure, $name, $initiator)
        }
    };
}
