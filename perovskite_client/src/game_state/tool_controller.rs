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

use std::time::{Duration, Instant};

use cgmath::num_traits::Float;
use cgmath::{vec3, Vector3};
use lazy_static::lazy_static;
use perovskite_core::constants::permissions;
use perovskite_core::protocol::blocks::block_type_def::PhysicsInfo;
use perovskite_core::{block_id::BlockId, coordinates::PlayerPositionUpdate};

use perovskite_core::constants::items::default_item_interaction_rules;
use perovskite_core::coordinates::BlockCoordinate;

use line_drawing::WalkVoxels;
use perovskite_core::game_actions::ToolTarget;
use perovskite_core::protocol::blocks::BlockTypeDef;
use perovskite_core::protocol::game_rpc as proto;
use perovskite_core::protocol::game_rpc::EntityTarget;
use perovskite_core::protocol::items::interaction_rule::DigBehavior;
use perovskite_core::protocol::items::{InteractionRule, ItemDef};
use proto::interact_key_action::InteractionTarget;
use rustc_hash::FxHashSet;

use super::physics::apply_aabox_transformation;
use super::{input::BoundAction, make_fallback_blockdef, ClientState, GameAction};

/// The thing which is being dug/tapped or was dug/tapped
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) enum ToolTargetWithId {
    /// Block coordinate, and the ID at that coordinate at the time the target was calculated
    Block(BlockCoordinate, BlockId),
    /// Selected entity (entity id, trailing entity index), entity class
    Entity(EntityTarget, u32),
}

impl ToolTargetWithId {
    pub(crate) fn target(&self) -> ToolTarget {
        match self {
            ToolTargetWithId::Block(x, id) => ToolTarget::Block(*x),
            ToolTargetWithId::Entity(id, class) => ToolTarget::Entity(*id),
        }
    }
}

struct TargetProperties<'a> {
    /// Block groups (also used for entities) that will be matched against the current item's
    /// interaction rules
    target_groups: &'a [String],
    /// The base dig time; see blocks.proto
    base_dig_time: f64,
}

#[derive(Clone, Copy)]
struct DigState {
    // 0.0 to 1.0
    progress: f64,
    // Coordinate being dug
    target: ToolTargetWithId,
    item_dig_behavior: Option<DigBehavior>,
    // Current durability (of selected block) for scaled time
    base_durability: f64,
}

fn target_properties(state: &ClientState, target: ToolTargetWithId) -> TargetProperties {
    match target {
        ToolTargetWithId::Block(coord, id) => state
            .block_types
            .get_blockdef(id)
            .map(|x| TargetProperties {
                target_groups: &x.groups,
                base_dig_time: x.base_dig_time,
            })
            .unwrap_or(TargetProperties {
                target_groups: &FALLBACK_GROUPS,
                base_dig_time: 1.0,
            }),
        ToolTargetWithId::Entity(_target, _) => {
            todo!()
        }
    }
}

lazy_static! {
    static ref FALLBACK_GROUPS: Box<[String]> = { Box::new([String::new()]) };
}

// Computes the block that the player is pointing at, tracks the progress of digging, etc.
pub(crate) struct ToolController {
    dig_progress: Option<DigState>,
    current_item: ItemDef,
    current_slot: u32,
    current_item_interacting_groups: Vec<Vec<String>>,
    fallback_blockdef: BlockTypeDef,
    can_dig_place: bool,
    can_tap_interact: bool,
    futile_dig_since: Option<Instant>,
}
impl ToolController {
    pub(crate) fn new() -> ToolController {
        ToolController {
            dig_progress: None,
            current_item: default_item(),
            current_slot: 0,
            current_item_interacting_groups: get_dig_interacting_groups(&default_item()),
            fallback_blockdef: make_fallback_blockdef(),
            can_dig_place: false,
            can_tap_interact: false,
            futile_dig_since: None,
        }
    }

    pub(crate) fn update_permissions(&mut self, permissions: &[String]) {
        self.can_dig_place = permissions
            .iter()
            .any(|p| p.contains(permissions::DIG_PLACE));
        self.can_tap_interact = permissions
            .iter()
            .any(|p| p.contains(permissions::TAP_INTERACT));
    }

    // Update the current item
    pub(crate) fn change_held_item(
        &mut self,
        client_state: &ClientState,
        slot: u32,
        item: Option<ItemDef>,
    ) {
        let item = item.unwrap_or(default_item());
        self.current_slot = slot;
        self.current_item_interacting_groups = get_dig_interacting_groups(&item);
        if let Some(progress) = &mut self.dig_progress {
            progress.item_dig_behavior = Self::compute_dig_behavior(
                &item,
                target_properties(client_state, progress.target).target_groups,
            );
        }
        self.current_item = item;
    }

    // Compute a new selected block.
    pub(crate) fn update(&mut self, client_state: &ClientState, delta: Duration) -> ToolState {
        let player_pos = client_state.last_position();
        let (pointee, neighbor, target_properties) =
            match self.compute_pointee(client_state, &player_pos) {
                Some(x) => x,
                None => {
                    // Ensure that we don't leave pending input events that fire when we finally do get a pointee
                    let mut input = client_state.input.lock();
                    input.take_just_pressed(BoundAction::Dig);
                    input.take_just_released(BoundAction::Dig);
                    input.take_just_pressed(BoundAction::Place);
                    input.take_just_pressed(BoundAction::Interact);
                    self.dig_progress = None;
                    return ToolState {
                        pointee: None,
                        neighbor: None,
                        action: None,
                    };
                }
            };
        let mut action = None;

        let mut input = client_state.input.lock();

        if input.is_pressed(BoundAction::Dig) && self.can_dig_place {
            if self
                .dig_progress
                .as_ref()
                .map_or(true, |x: &DigState| x.target != pointee)
            {
                // The pointee changed
                let behavior =
                    Self::compute_dig_behavior(&self.current_item, target_properties.target_groups);
                if let Some(DigBehavior::InstantDig(_)) = behavior {
                    // Only do this when the pointee changes
                    action = Some(GameAction::Dig(super::DigTapAction {
                        target: pointee.target(),
                        prev: neighbor,
                        item_slot: self.current_slot,
                        player_pos,
                    }));
                }
                self.dig_progress = behavior.map(|behavior| DigState {
                    progress: 0.,
                    target: pointee,
                    item_dig_behavior: Some(behavior),
                    base_durability: target_properties.base_dig_time,
                });
            } else if let Some(dig_progress) = &mut self.dig_progress {
                // The pointee is the same, update dig progress
                let delta_progress = match dig_progress.item_dig_behavior {
                    None => 0.,
                    Some(DigBehavior::InstantDig(_)) => 0.,
                    Some(DigBehavior::InstantDigOneshot(_)) => {
                        if input.take_just_pressed(BoundAction::Dig) {
                            action = Some(GameAction::Dig(super::DigTapAction {
                                target: pointee.target(),
                                prev: neighbor,
                                item_slot: self.current_slot,
                                player_pos,
                            }));
                        }
                        0.
                    }
                    Some(DigBehavior::ConstantTime(dig_seconds)) => {
                        delta.as_secs_f64() / dig_seconds
                    }
                    Some(DigBehavior::ScaledTime(scale)) => {
                        delta.as_secs_f64() / (scale * dig_progress.base_durability)
                    }
                    Some(DigBehavior::Undiggable(_)) => 0.,
                };

                dig_progress.progress += delta_progress;
                if dig_progress.progress >= 1.0 {
                    // Lock out digging until progress is reset by changing the pointee or pointed-to blockid
                    dig_progress.progress = f64::NEG_INFINITY;
                    action = Some(GameAction::Dig(super::DigTapAction {
                        target: pointee.target(),
                        prev: neighbor,
                        item_slot: self.current_slot,
                        player_pos,
                    }));
                }
            }
        } else {
            self.dig_progress = None;
        }

        if input.take_just_released(BoundAction::Dig) && self.can_tap_interact {
            action = Some(GameAction::Tap(super::DigTapAction {
                target: pointee.target(),
                prev: neighbor,
                item_slot: self.current_slot,
                player_pos,
            }))
        } else if input.take_just_pressed(BoundAction::Place) && self.can_dig_place {
            action = Some(GameAction::Place(super::PlaceAction {
                target: neighbor,
                anchor: pointee.target(),
                item_slot: self.current_slot,
                player_pos,
            }))
        } else if input.take_just_pressed(BoundAction::Interact) && self.can_tap_interact {
            action = Some(GameAction::InteractKey(super::InteractKeyAction {
                target: pointee.target(),
                item_slot: self.current_slot,
                player_pos,
            }))
        }

        if !self.can_dig_place {
            if input.peek_just_pressed(BoundAction::Dig) {
                self.futile_dig_since = Some(Instant::now());
            } else if !input.is_pressed(BoundAction::Dig) {
                self.futile_dig_since = None;
            }
            if self.futile_dig_since.map_or(false, |x: Instant| {
                x.elapsed() > Duration::from_secs_f64(0.5)
            }) {
                self.futile_dig_since = None;
                client_state
                    .chat
                    .lock()
                    .show_client_message("You don't have permission to dig".to_string());
            }
            if input.peek_just_pressed(BoundAction::Place) {
                client_state
                    .chat
                    .lock()
                    .show_client_message("You don't have permission to place".to_string());
            }
        }

        if let Some(action) = &action {
            log::info!("Sending player action: {:?}", action);
        }

        ToolState {
            pointee: Some(pointee),
            neighbor,
            action,
        }
    }

    pub(crate) fn pointee_block_id(&self) -> Option<BlockId> {
        match self.dig_progress {
            None => None,
            Some(x) => match x.target {
                ToolTargetWithId::Block(_, id) => Some(id),
                ToolTargetWithId::Entity(_, _) => None, // TODO
            },
        }
    }

    fn compute_dig_behavior(item: &ItemDef, target_groups: &[String]) -> Option<DigBehavior> {
        let block_groups = target_groups.iter().collect::<FxHashSet<_>>();
        for rule in item.interaction_rules.iter() {
            if rule.block_group.iter().all(|x| block_groups.contains(x)) {
                return rule.dig_behavior.clone();
            }
        }
        None
    }

    /// Determines what the player is pointing at.
    ///
    /// Returns: (pointed target, if applicable the preceding block, target info)
    fn compute_pointee<'a>(
        &'a self,
        client_state: &'a ClientState,
        last_pos: &PlayerPositionUpdate,
    ) -> Option<(
        ToolTargetWithId,
        Option<BlockCoordinate>,
        TargetProperties<'a>,
    )> {
        // TODO handle custom collision boxes
        let pos = last_pos.position;
        let end = pos + (POINTEE_DISTANCE * last_pos.face_unit_vector());

        let delta_inv = vec3(
            1.0 / (end.x - pos.x),
            1.0 / (end.y - pos.y),
            1.0 / (end.z - pos.z),
        );

        let chunks = client_state.chunks.read_lock();
        let mut prev = None;
        let start_pos = match (pos + RAYCAST_FUDGE_VEC).try_into() {
            Ok(x) => x,
            Err(_) => {
                log::error!("pos: {pos:?} corrupt");
                return None;
            }
        };
        let end_pos = match (end + RAYCAST_FUDGE_VEC).try_into() {
            Ok(x) => x,
            Err(_) => {
                log::error!("end: {end:?} corrupt");
                return None;
            }
        };

        for (x, y, z) in
            WalkVoxels::<f64, i64>::new(start_pos, end_pos, &line_drawing::VoxelOrigin::Corner)
        {
            let coord = BlockCoordinate {
                x: x.try_into().ok()?,
                y: y.try_into().ok()?,
                z: z.try_into().ok()?,
            };
            let chunk = chunks.get(&coord.chunk());
            if let Some(chunk) = chunk {
                let id = chunk.get_single(coord.offset());
                let block_def = client_state
                    .block_types
                    .get_blockdef(id)
                    .unwrap_or(&self.fallback_blockdef);
                if check_intersection(
                    vec3(coord.x as f64, coord.y as f64, coord.z as f64),
                    block_def,
                    pos,
                    delta_inv,
                    id.variant(),
                ) {
                    for rule in self.current_item_interacting_groups.iter() {
                        if rule.iter().all(|x| block_def.groups.contains(x)) {
                            return Some((
                                ToolTargetWithId::Block(coord, id),
                                prev,
                                TargetProperties {
                                    target_groups: &block_def.groups,
                                    base_dig_time: block_def.base_dig_time,
                                },
                            ));
                        }
                    }
                }
                prev = Some(coord);
            }
        }
        None
    }
}

fn check_intersection(
    block_coord: cgmath::Vector3<f64>,
    block_def: &BlockTypeDef,
    pos: cgmath::Vector3<f64>,
    delta_inv: cgmath::Vector3<f64>,
    variant: u16,
) -> bool {
    match &block_def.tool_custom_hitbox {
        Some(boxes) => {
            boxes.boxes.is_empty()
                || boxes.boxes.iter().any(|aa_box| {
                    if aa_box.variant_mask != 0 && aa_box.variant_mask & variant as u32 == 0 {
                        return false;
                    }

                    let (min, max) = apply_aabox_transformation(aa_box, block_coord, variant);

                    // Disregard T itself, just check whether it's a hit
                    check_intersection_core(pos, delta_inv, min, max).0
                })
        }
        _ => true,
    }
}

#[inline]
pub(crate) fn check_intersection_core<F: Float>(
    pos: Vector3<F>,
    delta_inv: Vector3<F>,
    min: Vector3<F>,
    max: Vector3<F>,
) -> (bool, F) {
    let tx1 = (min.x - pos.x) * delta_inv.x;
    let tx2 = (max.x - pos.x) * delta_inv.x;

    let t_min = tx1.min(tx2);
    let t_max = tx1.max(tx2);

    let ty1 = (min.y - pos.y) * delta_inv.y;
    let ty2 = (max.y - pos.y) * delta_inv.y;

    let t_min = t_min.max(ty1.min(ty2));
    let t_max = t_max.min(ty1.max(ty2));

    let tz1 = (min.z - pos.z) * delta_inv.z;
    let tz2 = (max.z - pos.z) * delta_inv.z;

    let t_min = t_min.max(tz1.min(tz2));
    let t_max = t_max.min(tz1.max(tz2));

    let hit = t_max >= t_min && t_max >= F::zero();
    (hit, t_min)
}

fn get_dig_interacting_groups(item: &ItemDef) -> Vec<Vec<String>> {
    let mut result = vec![];
    for rule in &item.interaction_rules {
        if rule.dig_behavior.is_some() {
            result.push(rule.block_group.clone())
        }
    }
    result
}

fn default_item() -> ItemDef {
    let rules = default_item_interaction_rules();
    ItemDef {
        short_name: String::new(),
        display_name: String::new(),
        inventory_texture: None,
        groups: vec![],
        interaction_rules: rules,
        quantity_type: None,
        block_apperance: "".to_string(),
        // Sort key is irrelevant
        sort_key: String::new(),
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ToolState {
    // The block being pointed at
    pub(crate) pointee: Option<ToolTargetWithId>,
    // The previous block coordinate, where we would place a block
    // if the place button were pressed
    pub(crate) neighbor: Option<BlockCoordinate>,
    // The action taken during this frame
    pub(crate) action: Option<GameAction>,
}

const POINTEE_DISTANCE: f64 = 6.;
// line_drawing seems to have problems when using Center rather than Corner
// Fudge it manually
// TODO file a bug for that crate
const RAYCAST_FUDGE_VEC: cgmath::Vector3<f64> = vec3(0.5, 0.5, 0.5);
