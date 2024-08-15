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

use std::{ops::RangeInclusive, sync::Arc, time::Duration};

use arc_swap::ArcSwap;
use cgmath::{vec3, Angle, Deg, InnerSpace, Matrix3, Vector3, Zero};
use perovskite_core::{
    constants::permissions,
    coordinates::BlockCoordinate,
    protocol::blocks::{
        block_type_def::{self, PhysicsInfo},
        BlockTypeDef,
    },
};

use tracy_client::{plot, span};

use super::{
    input::{BoundAction, InputState},
    settings::GameSettings,
    ChunkManagerView, ClientBlockTypeManager, ClientState,
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
const TRAVERSABLE_BUMP_HEIGHT_LANDED: f64 = 0.51;
const TRAVERSABLE_BUMP_HEIGHT_MIDAIR: f64 = 0.2;
const FLY_SPEED: f64 = 24.0;
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
    fn next(&self, allow_fly: bool, allow_noclip: bool) -> PhysicsMode {
        match self {
            PhysicsMode::Standard => {
                if allow_fly {
                    PhysicsMode::FlyingCollisions
                } else {
                    PhysicsMode::Standard
                }
            }
            PhysicsMode::FlyingCollisions => {
                if allow_noclip {
                    PhysicsMode::Noclip
                } else {
                    PhysicsMode::Standard
                }
            }
            PhysicsMode::Noclip => PhysicsMode::Standard,
        }
    }
}

pub(crate) struct PhysicsState {
    pos: Vector3<f64>,
    last_velocity: Vector3<f64>,
    az: Deg<f64>,
    el: Deg<f64>,
    target_az: Deg<f64>,
    target_el: Deg<f64>,
    physics_mode: PhysicsMode,
    landed_last_frame: bool,
    last_land_height: f64,
    walk_sound_odometer: f64,
    bump_decay: f64,
    settings: Arc<ArcSwap<GameSettings>>,

    can_fly: bool,
    can_fast: bool,
    can_noclip: bool,
}

impl PhysicsState {
    pub(crate) fn new(settings: Arc<ArcSwap<GameSettings>>) -> Self {
        Self {
            pos: vec3(0., 0., -4.),
            last_velocity: vec3(0., 0., 0.),
            az: Deg(0.),
            el: Deg(0.),
            target_az: Deg(0.),
            target_el: Deg(0.),
            physics_mode: PhysicsMode::Standard,
            landed_last_frame: false,
            last_land_height: 0.,
            walk_sound_odometer: 0.,
            bump_decay: 0.,
            settings,
            can_fly: false,
            can_fast: false,
            can_noclip: false,
        }
    }

    // view matrix, position
    pub(crate) fn update_and_get(
        &mut self,
        client_state: &ClientState,
        _aspect_ratio: f64,
        delta: Duration,
    ) -> (Vector3<f64>, Vector3<f64>, (f64, f64)) {
        let mut input = client_state.input.lock();
        if input.take_just_pressed(BoundAction::TogglePhysics) {
            self.physics_mode = self.physics_mode.next(self.can_fly, self.can_noclip);
            client_state.chat.lock().show_client_message(format!(
                "Physics mode changed to {}",
                match self.physics_mode {
                    PhysicsMode::Standard => "standard",
                    PhysicsMode::FlyingCollisions => "flying",
                    PhysicsMode::Noclip => "noclip",
                }
            ));
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
        let adjusted_for_bump = vec3(self.pos.x, self.pos.y - self.bump_decay, self.pos.z);
        (adjusted_for_bump, self.last_velocity, self.angle())
    }

    fn update_standard(
        &mut self,
        input: &mut InputState,
        // todo handle long deltas without falling through the ground
        delta: Duration,
        client_state: &ClientState,
    ) {
        let _span = span!("physics_standard");

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

        let block_physics = surrounding_block.and_then(|(x, _)| x.physics_info.as_ref());

        let delta = delta.as_secs_f64();
        let (target, mut velocity) = match block_physics {
            Some(PhysicsInfo::Air(_)) => self.update_target_air(input, self.pos, 3.0, delta),
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
                    surface_test_block.and_then(|(x, _)| x.physics_info.as_ref()),
                    Some(PhysicsInfo::Air(_))
                );
                self.update_target_fluid(input, self.pos, fluid_data, delta, is_surface)
            }
            Some(PhysicsInfo::Solid(_)) => {
                // We're in a block, possibly one we just placed. This shouldn't happen, so allow
                // the user to jump or walk out of it (unless they would run into other solid blocks
                // in the process)
                self.update_target_air(input, self.pos, 3.0, delta)
            }
            Some(PhysicsInfo::SolidCustomCollisionboxes(_)) => {
                self.update_target_air(input, self.pos, 3.0, delta)
            }
            None => self.update_target_air(input, self.pos, 3.0, delta),
        };

        plot!("physics_delta", (target - self.pos).magnitude());

        let new_pos = clamp_collisions_loop(self.pos, target, &chunks, block_types);
        // If we hit a floor or ceiling
        if (new_pos.y - target.y) > COLLISION_EPS {
            self.last_land_height = new_pos.y;
            self.walk_sound_odometer += (new_pos - self.pos).magnitude();
            if self.walk_sound_odometer > 1.0 || !self.landed_last_frame {
                client_state
                    .audio
                    .testonly_play_footstep(client_state.timekeeper.now(), new_pos);
                self.walk_sound_odometer = 0.0;
                log::info!("landed at {new_pos:?}");
            }

            self.landed_last_frame = true;
            velocity.y = 0.;
        } else if (new_pos.y - target.y) < -COLLISION_EPS {
            velocity.y = 0.;
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

        self.pos =
            self.update_with_bumping(new_pos, bump_height, target, chunks, block_types, delta);
        self.last_velocity = velocity;
    }

    fn update_with_bumping(
        &mut self,
        mut new_pos: Vector3<f64>,
        bump_height: f64,
        target: Vector3<f64>,
        chunks: ChunkManagerView<'_>,
        block_types: &Arc<ClientBlockTypeManager>,
        delta_secs: f64,
    ) -> Vector3<f64> {
        let pre_bump_y = new_pos.y;
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
                clamp_collisions_loop(bump_outcome, post_bump_target, &chunks, block_types);
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
                new_pos = clamp_collisions_loop(
                    post_bump_outcome,
                    down_bump_target,
                    &chunks,
                    block_types,
                );
                self.bump_decay = self.bump_decay.max(new_pos.y - pre_bump_y);
                if self.settings.load().render.physics_debug {
                    log::info!("bump success {bump_height} \ntarget   : {bump_target:?},\noutcome  : {bump_outcome:?}\nptarget : {post_bump_target:?}\npoutcome: {post_bump_outcome:?} \ndtarget : {down_bump_target:?}\nnewpos  : {new_pos:?}");
                    log::info!("bump decay: {}", self.bump_decay);
                }
            }
        }
        self.bump_decay = (self.bump_decay - (delta_secs * BUMP_DECAY_SPEED)).max(0.0);

        new_pos
    }

    fn update_target_air(
        &mut self,
        input: &mut InputState,
        pos: Vector3<f64>,
        speed: f64,
        delta_seconds: f64,
    ) -> (Vector3<f64>, Vector3<f64>) {
        let mut velocity = self.apply_movement_input(input, speed);
        // velocity if unperturbed
        velocity.y = self.last_velocity.y - (delta_seconds * GRAVITY_ACCEL)
            + (DAMPING * self.last_velocity.y.min(0.).powi(2) * delta_seconds);
        // Jump key
        if input.is_pressed(BoundAction::Jump) && self.landed_last_frame {
            self.landed_last_frame = false;
            velocity.y = JUMP_VELOCITY;
        }
        let target = pos + velocity * delta_seconds;
        (target, velocity)
    }

    fn apply_movement_input(&self, input: &mut InputState, mut multiplier: f64) -> Vector3<f64> {
        if self.can_fast && input.is_pressed(BoundAction::FastMove) {
            multiplier *= Self::FAST_MOVE_RATIO;
        }
        let mut result = Vector3::zero();
        if input.is_pressed(BoundAction::MoveForward) {
            result.z += self.az.cos() * multiplier;
            result.x += self.az.sin() * multiplier;
        } else if input.is_pressed(BoundAction::MoveBackward) {
            result.z -= self.az.cos() * multiplier;
            result.x -= self.az.sin() * multiplier;
        }

        if input.is_pressed(BoundAction::MoveLeft) {
            result.z += self.az.sin() * multiplier;
            result.x -= self.az.cos() * multiplier;
        } else if input.is_pressed(BoundAction::MoveRight) {
            result.z -= self.az.sin() * multiplier;
            result.x += self.az.cos() * multiplier;
        }
        result
    }

    const FAST_MOVE_RATIO: f64 = 3.0;

    fn update_target_fluid(
        &self,
        input: &mut InputState,
        pos: Vector3<f64>,
        fluid_data: &perovskite_core::protocol::blocks::FluidPhysicsInfo,
        delta: f64,
        is_surface: bool,
    ) -> (Vector3<f64>, Vector3<f64>) {
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
        let mut velocity = self.apply_movement_input(input, horizontal_speed);
        velocity.y = if input.is_pressed(BoundAction::Jump) {
            jump_velocity
        } else if input.is_pressed(BoundAction::Descend) {
            sink_velocity
        } else {
            vertical_velocity
        };
        let target = pos + velocity * delta;
        (target, velocity)
    }

    fn update_flying(
        &mut self,
        input: &mut InputState,
        delta: Duration,
        collisions: bool,
        client_state: &ClientState,
    ) {
        let mut velocity = self.apply_movement_input(input, FLY_SPEED);
        if input.is_pressed(BoundAction::Jump) {
            velocity.y += FLY_SPEED;
        } else if input.is_pressed(BoundAction::Descend) {
            velocity.y -= FLY_SPEED;
        }
        let target_pos = self.pos + velocity * delta.as_secs_f64();

        if collisions {
            let chunks = client_state.chunks.read_lock();
            let block_types = &client_state.block_types;
            let new_pos = clamp_collisions_loop(self.pos, target_pos, &chunks, block_types);
            self.pos = self.update_with_bumping(
                new_pos,
                TRAVERSABLE_BUMP_HEIGHT_LANDED,
                target_pos,
                chunks,
                block_types,
                delta.as_secs_f64(),
            );
        } else {
            self.pos = target_pos
        }
    }

    pub(crate) fn pos(&self) -> Vector3<f64> {
        self.pos
    }
    pub(crate) fn set_position(&mut self, pos: Vector3<f64>) {
        self.pos = pos;
    }
    pub(crate) fn update_permissions(
        &mut self,
        permissions: &[String],
        client_state: &ClientState,
    ) {
        self.can_fly = permissions.iter().any(|p| p == permissions::FLY);
        self.can_fast = permissions.iter().any(|p| p == permissions::FAST_MOVE);
        self.can_noclip = permissions.iter().any(|p| p == permissions::NOCLIP);

        let new_physics_mode = if !self.can_noclip {
            match self.physics_mode {
                PhysicsMode::Standard => PhysicsMode::Standard,
                PhysicsMode::FlyingCollisions => PhysicsMode::FlyingCollisions,
                PhysicsMode::Noclip => PhysicsMode::FlyingCollisions,
            }
        } else if !self.can_fly {
            PhysicsMode::Standard
        } else {
            self.physics_mode
        };
        if new_physics_mode != self.physics_mode {
            client_state.chat.lock().show_client_message(format!(
                "Physics mode changed to {}",
                match new_physics_mode {
                    PhysicsMode::Standard => "standard",
                    PhysicsMode::FlyingCollisions => "flying",
                    PhysicsMode::Noclip => "noclip",
                }
            ));
        }
        self.physics_mode = new_physics_mode;
    }
    pub(crate) fn angle(&self) -> (f64, f64) {
        (self.az.0, self.el.0)
    }

    fn update_smooth_angles(&mut self, delta: Duration) {
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
const BUMP_DECAY_SPEED: f64 = 3.0;

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

struct CollisionBox {
    min: Vector3<f64>,
    max: Vector3<f64>,
}
impl CollisionBox {
    fn full_cube(coord: Vector3<f64>) -> CollisionBox {
        CollisionBox {
            min: vec3(coord.x - 0.5, coord.y - 0.5, coord.z - 0.5),
            max: vec3(coord.x + 0.5, coord.y + 0.5, coord.z + 0.5),
        }
    }

    fn from_aabb(
        coord: Vector3<f64>,
        aa_box: &perovskite_core::protocol::blocks::AxisAlignedBox,
        variant: u16,
    ) -> CollisionBox {
        let (min, max) = apply_aabox_transformation(aa_box, coord, variant);
        CollisionBox { min, max }
    }
}

pub(crate) fn apply_aabox_transformation(
    aa_box: &perovskite_core::protocol::blocks::AxisAlignedBox,
    coord: Vector3<f64>,
    variant: u16,
) -> (Vector3<f64>, Vector3<f64>) {
    let min = vec3(
        aa_box.x_min as f64,
        aa_box.y_min as f64,
        aa_box.z_min as f64,
    );
    let max = vec3(
        aa_box.x_max as f64,
        aa_box.y_max as f64,
        aa_box.z_max as f64,
    );
    let (min, max) = match aa_box.rotation() {
        perovskite_core::protocol::blocks::AxisAlignedBoxRotation::None => {
            (coord + min, coord + max)
        }
        perovskite_core::protocol::blocks::AxisAlignedBoxRotation::Nesw => {
            let r_matrix = Matrix3::from_angle_y(Deg(90.0 * (variant % 4) as f64));
            let v1 = coord + r_matrix * min;
            let v2 = coord + r_matrix * max;
            let min = vec3(v1.x.min(v2.x), v1.y.min(v2.y), v1.z.min(v2.z));
            let max = vec3(v1.x.max(v2.x), v1.y.max(v2.y), v1.z.max(v2.z));
            (min, max)
        }
    };
    (min, max)
}

fn clamp_collisions(
    old_pos: Vector3<f64>,
    mut new_pos: Vector3<f64>,
    chunks: &ChunkManagerView,
    block_types: &ClientBlockTypeManager,
) -> Vector3<f64> {
    // For new_active, bias toward including a block when we're right on the edge
    let collision_boxes = get_collision_boxes(new_pos, chunks, block_types);
    let player_bbox_min = vec3(
        PLAYER_COLLISIONBOX_CORNER_NEG.x + new_pos.x,
        PLAYER_COLLISIONBOX_CORNER_NEG.y + new_pos.y,
        PLAYER_COLLISIONBOX_CORNER_NEG.z + new_pos.z,
    );
    let player_bbox_max = vec3(
        PLAYER_COLLISIONBOX_CORNER_POS.x + new_pos.x,
        PLAYER_COLLISIONBOX_CORNER_POS.y + new_pos.y,
        PLAYER_COLLISIONBOX_CORNER_POS.z + new_pos.z,
    );

    #[inline]
    fn overlaps(min1: f64, max1: f64, min2: f64, max2: f64) -> bool {
        min1 <= max2 && min2 <= max1
    }

    for cbox in collision_boxes {
        if overlaps(cbox.min.x, cbox.max.x, player_bbox_min.x, player_bbox_max.x)
            && overlaps(cbox.min.y, cbox.max.y, player_bbox_min.y, player_bbox_max.y)
            && overlaps(cbox.min.z, cbox.max.z, player_bbox_min.z, player_bbox_max.z)
        {
            new_pos = clamp_single_block(old_pos, new_pos, cbox);
        }
    }
    new_pos
}

#[inline]
fn clamp_single_block(old: Vector3<f64>, new: Vector3<f64>, cbox: CollisionBox) -> Vector3<f64> {
    vec3(
        clamp_single_axis(
            old.x,
            new.x,
            cbox.min.x,
            cbox.max.x,
            PLAYER_COLLISIONBOX_CORNER_POS.x,
            PLAYER_COLLISIONBOX_CORNER_NEG.x,
        ),
        clamp_single_axis(
            old.y,
            new.y,
            cbox.min.y,
            cbox.max.y,
            PLAYER_COLLISIONBOX_CORNER_POS.y,
            PLAYER_COLLISIONBOX_CORNER_NEG.y,
        ),
        clamp_single_axis(
            old.z,
            new.z,
            cbox.min.z,
            cbox.max.z,
            PLAYER_COLLISIONBOX_CORNER_POS.z,
            PLAYER_COLLISIONBOX_CORNER_NEG.z,
        ),
    )
}

fn clamp_single_axis(
    old: f64,
    new: f64,
    obstacle_min: f64,
    obstacle_max: f64,
    pos_bias: f64,
    neg_bias: f64,
) -> f64 {
    debug_assert!(pos_bias > 0.);
    debug_assert!(neg_bias < 0.);

    match new.total_cmp(&old) {
        std::cmp::Ordering::Less => {
            let boundary = obstacle_max - neg_bias;
            if old + COLLISION_EPS > boundary && new - COLLISION_EPS < boundary {
                // back away from the boundary by FLOAT_EPS
                boundary + COLLISION_EPS
            } else {
                new
            }
        }
        std::cmp::Ordering::Equal => new,
        std::cmp::Ordering::Greater => {
            let boundary = obstacle_min - pos_bias;
            if old - COLLISION_EPS < boundary && new + COLLISION_EPS > boundary {
                boundary - COLLISION_EPS
            } else {
                new
            }
        }
    }
}

fn get_block<'a>(
    coord: BlockCoordinate,
    chunks: &ChunkManagerView,
    block_types: &'a ClientBlockTypeManager,
) -> Option<(&'a BlockTypeDef, u16)> {
    let chunk = chunks.get(&coord.chunk())?;
    let id = chunk.get_single(coord.offset());
    let block = block_types.get_blockdef(id)?;
    Some((block, id.variant()))
}

#[inline]
fn inclusive<T: Ord>(a: T, b: T) -> RangeInclusive<T> {
    if b >= a {
        a..=b
    } else {
        b..=a
    }
}

fn get_collision_boxes(
    pos: Vector3<f64>,
    chunks: &ChunkManagerView,
    block_types: &ClientBlockTypeManager,
) -> Vec<CollisionBox> {
    let corner1 = pos + PLAYER_COLLISIONBOX_CORNER_POS;
    let corner2 = pos + PLAYER_COLLISIONBOX_CORNER_NEG;
    let mut output = vec![];
    for x in inclusive(corner1.x.round() as i32, corner2.x.round() as i32) {
        for y in inclusive(corner1.y.round() as i32, corner2.y.round() as i32) {
            for z in inclusive(corner1.z.round() as i32, corner2.z.round() as i32) {
                let coord = BlockCoordinate { x, y, z };
                match get_block(coord, chunks, block_types) {
                    Some((block, variant)) => {
                        push_collision_boxes(coord.into(), block, variant, &mut output);
                    }
                    None => output.push(CollisionBox::full_cube(coord.into())),
                }
            }
        }
    }
    output
}

fn push_collision_boxes(
    coord: Vector3<f64>,
    block: &BlockTypeDef,
    variant: u16,
    output: &mut Vec<CollisionBox>,
) {
    match &block.physics_info {
        Some(PhysicsInfo::Solid(_)) => output.push(CollisionBox::full_cube(coord)),
        Some(PhysicsInfo::Fluid(_)) => {}
        Some(PhysicsInfo::Air(_)) => {}
        Some(PhysicsInfo::SolidCustomCollisionboxes(boxes)) => {
            for box_ in &boxes.boxes {
                if box_.variant_mask == 0 || (variant & box_.variant_mask as u16) != 0 {
                    output.push(CollisionBox::from_aabb(coord, box_, variant));
                }
            }
        }
        None => {}
    }
}
