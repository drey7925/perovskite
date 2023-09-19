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

use std::time::Duration;


use perovskite_core::{block_id::BlockId, coordinates::PlayerPositionUpdate};

use perovskite_core::constants::items::default_item_interaction_rules;
use perovskite_core::coordinates::BlockCoordinate;

use perovskite_core::protocol::blocks::BlockTypeDef;
use perovskite_core::protocol::items::interaction_rule::DigBehavior;
use perovskite_core::protocol::items::ItemDef;
use line_drawing::WalkVoxels;
use rustc_hash::FxHashSet;

use super::{input::BoundAction, make_fallback_blockdef, ClientState, GameAction};

struct DigState {
    // 0.0 to 1.0
    progress: f64,
    // Coordinate being dug
    coord: BlockCoordinate,
    // Current block ID
    id: BlockId,
    item_dig_behavior: Option<DigBehavior>,
    // Current durability (of selected block) for scaled time
    base_durability: f64,
}

// Computes the block that the player is pointing at, tracks the progress of digging, etc.
pub(crate) struct ToolController {
    dig_progress: Option<DigState>,
    current_item: ItemDef,
    current_slot: u32,
    current_item_interacting_groups: Vec<Vec<String>>,
    fallback_blockdef: BlockTypeDef,
}
impl ToolController {
    pub(crate) fn new() -> ToolController {
        ToolController {
            dig_progress: None,
            current_item: default_item(),
            current_slot: 0,
            current_item_interacting_groups: get_dig_interacting_groups(&default_item()),
            fallback_blockdef: make_fallback_blockdef(),
        }
    }

    // Update the current item
    pub(crate) fn update_item(
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
                client_state.block_types.get_blockdef(progress.id).unwrap(),
            );
        }
        self.current_item = item;
    }

    // Compute a new selected block.
    pub(crate) fn update(&mut self, client_state: &ClientState, delta: Duration) -> ToolState {
        let player_pos = client_state.last_position();
        let (pointee, neighbor, block_def) = match self.compute_pointee(client_state, &player_pos) {
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

        if input.is_pressed(BoundAction::Dig) {
            if self.dig_progress.as_ref().map_or(true, |x: &DigState| {
                x.coord != pointee || !x.id.equals_ignore_variant(BlockId(block_def.id))
            }) {
                // The pointee changed
                let behavior = Self::compute_dig_behavior(&self.current_item, block_def);
                if let Some(DigBehavior::InstantDig(_)) = behavior {
                    // Only do this when the pointee changes
                    action = Some(GameAction::Dig(super::DigTapAction {
                        target: pointee,
                        prev: neighbor,
                        item_slot: self.current_slot,
                        player_pos,
                    }));
                }
                self.dig_progress = behavior.map(|behavior| DigState {
                    progress: 0.,
                    coord: pointee,
                    id: BlockId(block_def.id),
                    item_dig_behavior: Some(behavior),
                    base_durability: block_def.base_dig_time,
                });
            } else if let Some(dig_progress) = &mut self.dig_progress {
                // The pointee is the same, update dig progress
                let delta_progress = match dig_progress.item_dig_behavior {
                    None => 0.,
                    Some(DigBehavior::InstantDig(_)) => 0.,
                    Some(DigBehavior::InstantDigOneshot(_)) => {
                        if input.take_just_pressed(BoundAction::Dig) {
                            action = Some(GameAction::Dig(super::DigTapAction {
                                target: pointee,
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
                        target: pointee,
                        prev: neighbor,
                        item_slot: self.current_slot,
                        player_pos,
                    }));
                }
            }
        } else {
            self.dig_progress = None;
        }
        
        if input.take_just_released(BoundAction::Dig) {
            action = Some(GameAction::Tap(super::DigTapAction {
                target: pointee,
                prev: neighbor,
                item_slot: self.current_slot,
                player_pos,
            }))
        } else if input.take_just_pressed(BoundAction::Place) && neighbor.is_some() {
            action = Some(GameAction::Place(super::PlaceAction {
                target: neighbor.unwrap(),
                anchor: Some(pointee),
                item_slot: self.current_slot,
                player_pos,
            }))
        } else if input.take_just_pressed(BoundAction::Interact) {
            action = Some(GameAction::InteractKey(super::InteractKeyAction {
                target: pointee,
                item_slot: self.current_slot,
                player_pos,
            }))
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
    fn compute_dig_behavior(item: &ItemDef, block_def: &BlockTypeDef) -> Option<DigBehavior> {
        let block_groups = block_def.groups.iter().collect::<FxHashSet<_>>();
        for rule in item.interaction_rules.iter() {
            if rule.block_group.iter().all(|x| block_groups.contains(x)) {
                return rule.dig_behavior.clone();
            }
        }
        None
    }

    fn compute_pointee<'a>(
        &'a self,
        client_state: &'a ClientState,
        last_pos: &PlayerPositionUpdate,
    ) -> Option<(BlockCoordinate, Option<BlockCoordinate>, &'a BlockTypeDef)> {
        // TODO handle custom collision boxes
        let pos = last_pos.position;
        let end = pos + (POINTEE_DISTANCE * last_pos.face_unit_vector());
        let chunks = client_state.chunks.read_lock();
        let mut prev = None;
        for (x, y, z) in WalkVoxels::<f64, i64>::new(
            (pos + RAYCAST_FUDGE_VEC).into(),
            (end + RAYCAST_FUDGE_VEC).into(),
            &line_drawing::VoxelOrigin::Corner,
        ) {
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
                for rule in self.current_item_interacting_groups.iter() {
                    if rule.iter().all(|x| block_def.groups.contains(x)) {
                        return Some((coord, prev, block_def));
                    }
                }
                prev = Some(coord);
            }
        }
        None
    }
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
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ToolState {
    // The block being pointed at
    pub(crate) pointee: Option<BlockCoordinate>,
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
const RAYCAST_FUDGE_VEC: cgmath::Vector3<f64> = cgmath::vec3(0.5, 0.5, 0.5);
