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

use cgmath::Vector3;
pub(crate) struct RaycastIterator {
    // Current position of the iterator
    current: Vector3<f64>,
    // Unit vector in the direction we are casting, such that all three components
    // are positive
    direction: Vector3<f64>,
    // Vector of -1s and 1s, indicating which components we flipped in BOTH current
    // and direction
    correction: Vector3<f64>
}