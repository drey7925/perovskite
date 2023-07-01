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

use std::{collections::HashSet, ops::RangeInclusive, time::Duration};

use cgmath::{vec3, Angle, Deg, InnerSpace, Vector3};
use cuberef_core::{
    coordinates::BlockCoordinate,
    protocol::blocks::{
        block_type_def::{self, PhysicsInfo},
        BlockTypeDef,
    },
};

use tracy_client::{plot, span};

use crate::cube_renderer::ClientBlockTypeManager;

use super::{
    input::{BoundAction, InputState},
    ChunkManagerView, ClientState,
};

const PLAYER_WIDTH: f64 = 0.75;
const EYE_TO_TOP: f64 = 0.2;
const EYE_TO_BTM: f64 = 1.5;
// opposite corners of an axis-aligned-bounding-box
const PLAYER_COLLISIONBOX_CORNER_POS: Vector3<f64> =
    vec3(PLAYER_WIDTH / 2., EYE_TO_TOP, PLAYER_WIDTH / 2.);
const PLAYER_COLLISIONBOX_CORNER_NEG: Vector3<f64> =
    vec3(-PLAYER_WIDTH / 2., -EYE_TO_BTM, -PLAYER_WIDTH / 2.);
const _PLAYER_COLLISIONBOX_CORNERS: [Vector3<f64>; 8] = [
    vec3(PLAYER_WIDTH / 2., EYE_TO_TOP, PLAYER_WIDTH / 2.),
    vec3(PLAYER_WIDTH / 2., EYE_TO_TOP, -PLAYER_WIDTH / 2.),
    vec3(PLAYER_WIDTH / 2., EYE_TO_BTM, PLAYER_WIDTH / 2.),
    vec3(PLAYER_WIDTH / 2., EYE_TO_BTM, -PLAYER_WIDTH / 2.),
    vec3(-PLAYER_WIDTH / 2., -EYE_TO_TOP, PLAYER_WIDTH / 2.),
    vec3(-PLAYER_WIDTH / 2., -EYE_TO_TOP, -PLAYER_WIDTH / 2.),
    vec3(-PLAYER_WIDTH / 2., -EYE_TO_BTM, PLAYER_WIDTH / 2.),
    vec3(-PLAYER_WIDTH / 2., -EYE_TO_BTM, -PLAYER_WIDTH / 2.),
];
const TRAVERSABLE_BUMP_HEIGHT_LANDED: f64 = 0.501;
const TRAVERSABLE_BUMP_HEIGHT_MIDAIR: f64 = 0.2;
const WALK_SPEED: f64 = 3.0;
const JUMP_VELOCITY: f64 = 6.0;
// Not quite earth gravity; tuned for a natural feeling
const GRAVITY_ACCEL: f64 = 16.;
const TERMINAL_VELOCITY: f64 = 90.;
const DAMPING: f64 = GRAVITY_ACCEL / (TERMINAL_VELOCITY * TERMINAL_VELOCITY);

// Two levels of epsilon. physics_eps is used to break out of
// physics-level edge cases with minimal visual jankiness
const COLLISION_EPS: f64 = 0.000005;
// Use twice the collision epsilon for detecting surfaces/blocks (when not colliding with them)
const _DETECTION_EPS: f64 = COLLISION_EPS * 2.;
// float_eps is used to avoid issues related to exact float comparisions
const _FLOAT_EPS: f64 = 0.000001;

trait BumpMagnitude {
    fn bias_with(self, bias: f64) -> Self;
}
impl BumpMagnitude for f64 {
    fn bias_with(self, bias: f64) -> Self {
        self + (bias * self.signum())
    }
}
impl BumpMagnitude for Vector3<f64> {
    fn bias_with(self, bias: f64) -> Self {
        vec3(
            self.x.bias_with(bias),
            self.y.bias_with(bias),
            self.z.bias_with(bias),
        )
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum PhysicsMode {
    Standard,
    FlyingCollisions,
    Noclip,
}
impl PhysicsMode {
    fn next(&self) -> PhysicsMode {
        match self {
            PhysicsMode::Standard => PhysicsMode::FlyingCollisions,
            PhysicsMode::FlyingCollisions => PhysicsMode::Noclip,
            PhysicsMode::Noclip => PhysicsMode::Standard,
        }
    }
}

pub(crate) struct PhysicsState {
    pos: cgmath::Vector3<f64>,
    az: Deg<f64>,
    el: Deg<f64>,
    target_az: Deg<f64>,
    target_el: Deg<f64>,
    physics_mode: PhysicsMode,
    // Used for the standard physics mode
    y_velocity: f64,
    landed_last_frame: bool,
    last_land_height: f64,
}

impl PhysicsState {
    pub(crate) fn new() -> Self {
        Self {
            pos: cgmath::vec3(0., 0., -4.),
            az: Deg(0.),
            el: Deg(0.),
            target_az: Deg(0.),
            target_el: Deg(0.),
            physics_mode: PhysicsMode::Standard,
            y_velocity: 0.,
            landed_last_frame: false,
            last_land_height: 0.,
        }
    }

    // view matrix, position
    pub(crate) fn update_and_get(
        &mut self,
        client_state: &ClientState,
        _aspect_ratio: f64,
        delta: Duration,
    ) -> (cgmath::Vector3<f64>, (f64, f64)) {
        let mut input = client_state.input.lock();
        if input.take_just_pressed(BoundAction::TogglePhysics) {
            self.physics_mode = self.physics_mode.next();
        }
        let (x, y) = input.take_mouse_delta();
        self.target_az = Deg((self.target_az.0 + (x)).rem_euclid(360.0));
        self.target_el = Deg((self.target_el.0 - (y)).clamp(-90.0, 90.0));

        self.update_smooth_angles(delta);
        match self.physics_mode {
            PhysicsMode::Standard => self.update_standard(&mut input, delta, client_state),
            PhysicsMode::FlyingCollisions => {
                self.update_flying(&mut input, delta, true, client_state)
            }
            PhysicsMode::Noclip => self.update_flying(&mut input, delta, false, client_state),
        }

        (self.pos, (self.angle()))
    }

    fn update_standard(
        &mut self,
        input: &mut InputState,
        delta: std::time::Duration,
        client_state: &ClientState,
    ) {
        let _span = span!("physics_standard");
        // todo handle long deltas without falling through the ground
        let delta_secs = delta.as_secs_f64();

        let chunks = client_state.chunks.read_lock();
        let block_types = &client_state.block_types;

        // The block that the player's foot is in
        let surrounding_block = get_block(
            BlockCoordinate {
                x: self.pos.x.round() as i32,
                y: (self.pos.y - EYE_TO_BTM).round() as i32,
                z: self.pos.z.round() as i32,
            },
            &chunks,
            block_types,
        );

        let block_physics = surrounding_block.and_then(|x| x.physics_info.as_ref());
        let (mut new_yv, target) = match block_physics {
            Some(PhysicsInfo::Air(_)) => {
                self.update_target_air(input, self.pos, delta_secs * 3.0, delta)
            }
            Some(PhysicsInfo::Fluid(fluid_data)) => {
                let surface_test_block = get_block(
                    BlockCoordinate {
                        x: self.pos.x.round() as i32,
                        y: (self.pos.y - EYE_TO_BTM + fluid_data.surface_thickness).round() as i32,
                        z: self.pos.z.round() as i32,
                    },
                    &chunks,
                    block_types,
                );
                let is_surface = matches!(
                    surface_test_block.and_then(|x| x.physics_info.as_ref()),
                    Some(PhysicsInfo::Air(_))
                );
                self.update_target_fluid(input, self.pos, fluid_data, delta, is_surface)
            }
            Some(PhysicsInfo::Solid(_)) => {
                // We're in a block, possibly one we just placed. This shouldn't happen, so allow
                // the user to jump or walk out of it (unless they would run into other solid blocks
                // in the process)
                self.update_target_air(input, self.pos, delta_secs * 3.0, delta)
            }
            None => self.update_target_air(input, self.pos, delta_secs * 3.0, delta),
        };

        plot!("physics_delta", (target - self.pos).magnitude());

        let mut new_pos = clamp_collisions_loop(self.pos, target, &chunks, block_types);
        // If we hit a floor or ceiling
        if (new_pos.y - target.y) > COLLISION_EPS {
            self.landed_last_frame = true;
            self.last_land_height = new_pos.y;
            new_yv = 0.;
        } else if (new_pos.y - target.y) < -COLLISION_EPS {
            new_yv = 0.;
            self.landed_last_frame = false;
        } else {
            self.landed_last_frame = false;
        }

        // If we're landed now, we're able to bump up to the current height + about half a block or so (enough for a stair)
        // Otherwise, we'll only bump up no more than 0.1 blocks or so, and never more than our current height

        let bump_height = if self.landed_last_frame {
            TRAVERSABLE_BUMP_HEIGHT_LANDED
        } else {
            (self.last_land_height + TRAVERSABLE_BUMP_HEIGHT_MIDAIR - new_pos.y)
                .clamp(0., TRAVERSABLE_BUMP_HEIGHT_MIDAIR)
        };

        if bump_height > 0.
            && ((new_pos.x - target.x).abs() > COLLISION_EPS
                || (new_pos.z - target.z).abs() > COLLISION_EPS)
        {
            // We hit something in the horizontal direction. Try to bump up by some distance no larger than TRAVERSABLE_BUMP_HEIGHT...
            let bump_target = vec3(new_pos.x, new_pos.y + bump_height, new_pos.z);
            // This is where we end up as we try to go up the bump
            let bump_outcome = clamp_collisions_loop(new_pos, bump_target, &chunks, block_types);
            // and this is where we want to end up after the bump
            let post_bump_target = vec3(target.x, bump_outcome.y, target.z);
            let post_bump_outcome =
                clamp_collisions(bump_outcome, post_bump_target, &chunks, block_types);
            // did we end up making it anywhere horizontally?
            if (post_bump_outcome.x - bump_target.x).abs() > COLLISION_EPS
                || (post_bump_outcome.z - bump_target.z).abs() > COLLISION_EPS
            {
                // If so, bump back down to the surface
                let down_bump_target = vec3(
                    post_bump_outcome.x,
                    post_bump_outcome.y - bump_height,
                    post_bump_outcome.z,
                );
                new_pos =
                    clamp_collisions(post_bump_outcome, down_bump_target, &chunks, block_types);
                if cfg!(debug_assertions) {
                    println!("bump success {bump_height} \ntarget   : {bump_target:?},\noutcome  : {bump_outcome:?}\nptarget : {post_bump_target:?}\npoutcome: {post_bump_outcome:?} \ndtarget : {down_bump_target:?}\nnewpos  : {new_pos:?}");
                }
            }
        }

        self.pos = new_pos;
        self.y_velocity = new_yv;
    }

    fn update_target_air(
        &mut self,
        input: &mut InputState,
        pos: Vector3<f64>,
        distance: f64,
        time_delta: Duration,
    ) -> (f64, Vector3<f64>) {
        // TODO stop using raw scancodes here
        // TODO make these configurable by the user
        let mut target = pos;
        if input.is_pressed(BoundAction::MoveForward) {
            target.z += self.az.cos() * distance;
            target.x -= self.az.sin() * distance;
        } else if input.is_pressed(BoundAction::MoveBackward) {
            target.z -= self.az.cos() * distance;
            target.x += self.az.sin() * distance;
        }

        if input.is_pressed(BoundAction::MoveLeft) {
            target.z += self.az.sin() * distance;
            target.x += self.az.cos() * distance;
        } else if input.is_pressed(BoundAction::MoveRight) {
            target.z -= self.az.sin() * distance;
            target.x -= self.az.cos() * distance;
        }
        // velocity if unperturbed
        let mut new_yv = self.y_velocity - (time_delta.as_secs_f64() * GRAVITY_ACCEL)
            + (DAMPING * self.y_velocity.min(0.).powi(2) * time_delta.as_secs_f64());
        // Jump key
        if input.is_pressed(BoundAction::Jump) && self.landed_last_frame {
            self.landed_last_frame = false;
            new_yv = JUMP_VELOCITY;
        }
        target.y += new_yv * time_delta.as_secs_f64();
        (new_yv, target)
    }

    fn update_target_fluid(
        &self,
        input: &mut InputState,
        pos: Vector3<f64>,
        fluid_data: &cuberef_core::protocol::blocks::FluidPhysicsInfo,
        delta: Duration,
        is_surface: bool,
    ) -> (f64, Vector3<f64>) {
        let delta = delta.as_secs_f64();
        let (horizontal_speed, vertical_velocity, jump_velocity, sink_velocity) = if is_surface {
            (
                fluid_data.surf_horizontal_speed,
                fluid_data.surf_vertical_speed,
                fluid_data.surf_jump_speed,
                fluid_data.surf_sink_speed,
            )
        } else {
            (
                fluid_data.horizontal_speed,
                fluid_data.vertical_speed,
                fluid_data.jump_speed,
                fluid_data.sink_speed,
            )
        };
        let mut target = pos;
        if input.is_pressed(BoundAction::MoveForward) {
            target.z += self.az.cos() * horizontal_speed * delta;
            target.x -= self.az.sin() * horizontal_speed * delta;
        } else if input.is_pressed(BoundAction::MoveBackward) {
            target.z -= self.az.cos() * horizontal_speed * delta;
            target.x += self.az.sin() * horizontal_speed * delta;
        }

        if input.is_pressed(BoundAction::MoveLeft) {
            target.z += self.az.sin() * horizontal_speed * delta;
            target.x += self.az.cos() * horizontal_speed * delta;
        } else if input.is_pressed(BoundAction::MoveRight) {
            target.z -= self.az.sin() * horizontal_speed * delta;
            target.x -= self.az.cos() * horizontal_speed * delta;
        }
        let vy = if input.is_pressed(BoundAction::Jump) {
            jump_velocity
        } else if input.is_pressed(BoundAction::Descend) {
            sink_velocity
        } else {
            vertical_velocity
        };
        target.y += vy * delta;
        (vy, target)
    }

    fn update_flying(
        &mut self,
        input: &mut InputState,
        delta: std::time::Duration,
        collisions: bool,
        client_state: &ClientState,
    ) {
        let distance = delta.as_secs_f64() * WALK_SPEED * 5.0;

        let mut new_pos = self.pos;

        if input.is_pressed(BoundAction::MoveForward) {
            new_pos.z += self.az.cos() * distance;
            new_pos.x -= self.az.sin() * distance;
        } else if input.is_pressed(BoundAction::MoveBackward) {
            new_pos.z -= self.az.cos() * distance;
            new_pos.x += self.az.sin() * distance;
        }

        if input.is_pressed(BoundAction::MoveLeft) {
            new_pos.z += self.az.sin() * distance;
            new_pos.x += self.az.cos() * distance;
        } else if input.is_pressed(BoundAction::MoveRight) {
            new_pos.z -= self.az.sin() * distance;
            new_pos.x -= self.az.cos() * distance;
        }
        if input.is_pressed(BoundAction::Jump) {
            new_pos.y += distance;
        } else if input.is_pressed(BoundAction::Descend) {
            new_pos.y -= distance;
        }

        if collisions {
            let chunks = client_state.chunks.read_lock();
            let block_types = &client_state.block_types;
            new_pos = clamp_collisions_loop(self.pos, new_pos, &chunks, block_types);
        }

        self.pos = new_pos
    }

    pub(crate) fn pos(&self) -> Vector3<f64> {
        self.pos
    }
    pub(crate) fn set_position(&mut self, pos: Vector3<f64>) {
        self.pos = pos;
    }
    pub(crate) fn angle(&self) -> (f64, f64) {
        (self.az.0, self.el.0)
    }

    fn update_smooth_angles(&mut self, delta: std::time::Duration) {
        let factor =
            ANGLE_SMOOTHING_FACTOR.powf(delta.as_secs_f64() / ANGLE_SMOOTHING_REFERENCE_DELTA);
        self.el = (self.el * factor) + (self.target_el * (1.0 - factor));

        let az_x = (self.az.cos() * factor) + (self.target_az.cos() * (1.0 - factor));
        let az_y = (self.az.sin() * factor) + (self.target_az.sin() * (1.0 - factor));
        self.az = Deg::atan2(az_y, az_x);
    }
}

const ANGLE_SMOOTHING_FACTOR: f64 = 0.8;
const ANGLE_SMOOTHING_REFERENCE_DELTA: f64 = 1.0 / 165.0;

fn clamp_collisions_loop(
    old_pos: Vector3<f64>,
    target: Vector3<f64>,
    chunks: &ChunkManagerView,
    block_types: &ClientBlockTypeManager,
) -> Vector3<f64> {
    if (target - old_pos).magnitude2() < (COLLISION_EPS * COLLISION_EPS) {
        return old_pos;
    }
    let mut prev = old_pos;
    loop {
        let intended_delta = target - prev;
        let shortened_mag = intended_delta.magnitude().min(0.05);
        if shortened_mag < COLLISION_EPS {
            return prev;
        }
        let shortened_delta = (intended_delta / intended_delta.magnitude()) * shortened_mag;
        let shortened_target = prev + shortened_delta;
        let current = prev;
        let current = clamp_collisions(
            current,
            vec3(shortened_target.x, current.y, current.z),
            chunks,
            block_types,
        );
        let current = clamp_collisions(
            current,
            vec3(current.x, shortened_target.y, current.z),
            chunks,
            block_types,
        );
        let current = clamp_collisions(
            current,
            vec3(current.x, current.y, shortened_target.z),
            chunks,
            block_types,
        );
        if (prev - current).magnitude2() < (COLLISION_EPS * COLLISION_EPS) {
            return current;
        }
        prev = current;
    }
}

fn clamp_collisions(
    old_pos: Vector3<f64>,
    mut new_pos: Vector3<f64>,
    chunks: &ChunkManagerView,
    block_types: &ClientBlockTypeManager,
) -> Vector3<f64> {
    // Determine all the interacting blocks for old_pos and new_pos (i.e. those blocks that the player intersects with)
    // For old_active, bias against including a block when we're right on the edge
    let old_active = compute_bbox_intersections(old_pos, 0.);
    // For new_active, bias toward including a block when we're right on the edge
    let new_active = compute_bbox_intersections(new_pos, 0.);

    let new_blocks: Vec<_> = new_active.difference(&old_active).collect();

    if new_blocks.is_empty() {
        return new_pos;
    }

    for &coord in new_blocks {
        if !is_collision(get_block(coord, chunks, block_types)) {
            continue;
        }
        new_pos = clamp_single_block(old_pos, new_pos, coord);
    }
    new_pos
}

#[inline]
fn clamp_single_block(
    old: Vector3<f64>,
    new: Vector3<f64>,
    coord: BlockCoordinate,
) -> Vector3<f64> {
    vec3(
        clamp_single_axis(
            old.x,
            new.x,
            coord.x,
            PLAYER_COLLISIONBOX_CORNER_POS.x,
            PLAYER_COLLISIONBOX_CORNER_NEG.x,
        ),
        clamp_single_axis(
            old.y,
            new.y,
            coord.y,
            PLAYER_COLLISIONBOX_CORNER_POS.y,
            PLAYER_COLLISIONBOX_CORNER_NEG.y,
        ),
        clamp_single_axis(
            old.z,
            new.z,
            coord.z,
            PLAYER_COLLISIONBOX_CORNER_POS.z,
            PLAYER_COLLISIONBOX_CORNER_NEG.z,
        ),
    )
}

fn clamp_single_axis(old: f64, new: f64, obstacle: i32, pos_bias: f64, neg_bias: f64) -> f64 {
    debug_assert!(pos_bias > 0.);
    debug_assert!(neg_bias < 0.);
    // // Check if we're already within the obstacle block in the current dimension
    // if (old + neg_bias) < (obstacle as f64 + 0.5) && (old + pos_bias) > (obstacle as f64 - 0.5) {
    //     return new;
    // }
    match new.total_cmp(&old) {
        std::cmp::Ordering::Less => {
            let boundary = obstacle as f64 + 0.5 - neg_bias;
            if old + COLLISION_EPS > boundary && new - COLLISION_EPS < boundary {
                // back away from the boundary by FLOAT_EPS
                boundary + COLLISION_EPS
            } else {
                new
            }
        }
        std::cmp::Ordering::Equal => new,
        std::cmp::Ordering::Greater => {
            let boundary = obstacle as f64 - 0.5 - pos_bias;
            if old - COLLISION_EPS < boundary && new + COLLISION_EPS > boundary {
                boundary - COLLISION_EPS
            } else {
                new
            }
        }
    }
}

fn is_collision(block: Option<&BlockTypeDef>) -> bool {
    match block {
        Some(def) => match def.physics_info {
            Some(block_type_def::PhysicsInfo::Solid(_)) => true,
            Some(block_type_def::PhysicsInfo::Fluid(_)) => false,
            Some(block_type_def::PhysicsInfo::Air(_)) => false,
            // no physics info -> no interaction
            None => false,
        },
        // no block def -> unknown block -> true
        None => true,
    }
}

fn get_block<'a>(
    coord: BlockCoordinate,
    chunks: &ChunkManagerView,
    block_types: &'a ClientBlockTypeManager,
) -> Option<&'a BlockTypeDef> {
    chunks
        .get(&coord.chunk())
        .and_then(|chunk| block_types.get_blockdef(chunk.get(coord.offset())))
}

#[inline]
fn inclusive<T: Ord>(a: T, b: T) -> RangeInclusive<T> {
    if b >= a {
        a..=b
    } else {
        b..=a
    }
}

fn compute_bbox_intersections(pos: Vector3<f64>, bias: f64) -> HashSet<BlockCoordinate> {
    let corner1 = pos + PLAYER_COLLISIONBOX_CORNER_POS.bias_with(bias);
    let corner2 = pos + PLAYER_COLLISIONBOX_CORNER_NEG.bias_with(bias);
    let mut output = HashSet::new();
    for x in inclusive(corner1.x.round() as i32, corner2.x.round() as i32) {
        for y in inclusive(corner1.y.round() as i32, corner2.y.round() as i32) {
            for z in inclusive(corner1.z.round() as i32, corner2.z.round() as i32) {
                output.insert(BlockCoordinate { x, y, z });
            }
        }
    }
    output
}
