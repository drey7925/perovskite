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

use std::borrow::Cow;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cgmath::num_traits::Float;
use cgmath::{vec3, Vector3};
use lazy_static::lazy_static;
use perovskite_core::constants::permissions;
use perovskite_core::{block_id::BlockId, coordinates::PlayerPositionUpdate};

use perovskite_core::constants::items::default_item_interaction_rules;
use perovskite_core::coordinates::BlockCoordinate;

use super::physics::apply_aabox_transformation;
use super::{input::BoundAction, make_fallback_blockdef, ClientState, GameAction};
use line_drawing::WalkVoxels;
use perovskite_core::game_actions::ToolTarget;
use perovskite_core::protocol::blocks::{BlockTypeDef, InteractKeyOption};
use perovskite_core::protocol::game_rpc::EntityTarget;
use perovskite_core::protocol::items::interaction_rule::DigBehavior;
use perovskite_core::protocol::items::ItemDef;
use perovskite_core::protocol::map::ClientExtendedData;
use rustc_hash::FxHashSet;

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
            ToolTargetWithId::Block(x, _id) => ToolTarget::Block(*x),
            ToolTargetWithId::Entity(id, _class) => ToolTarget::Entity(*id),
        }
    }
    pub(crate) fn without_trailing_index(&self) -> ToolTargetWithId {
        match self {
            ToolTargetWithId::Block(x, id) => ToolTargetWithId::Block(*x, *id),
            ToolTargetWithId::Entity(id, class) => ToolTargetWithId::Entity(
                EntityTarget {
                    entity_id: id.entity_id,
                    trailing_entity_index: 0,
                },
                *class,
            ),
        }
    }
}

struct TargetProperties<'a> {
    /// Block groups (also used for entities) that will be matched against the current item's
    /// interaction rules
    target_groups: &'a [String],
    /// The base dig time; see blocks.proto
    base_dig_time: f64,
    /// For entities, whether trailing entities should be merged into the main entity as far as
    /// progress tracking. if true, only reset the progress if the pointee changes ignoring trailing
    /// entity ID, otherwise reset the progress if the pointee changes even with the same main
    /// entity
    ///
    /// For blocks, always false and irrelevant
    merge_trailing_entities: bool,
    interact_key_options: &'a [InteractKeyOption],
    hover_text: Option<Cow<'a, str>>,
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
    // If false, reset the progress if the pointee changes to a different trailing entity of the
    // same main entity. If true, only reset the progress if the pointee changes ignoring trailing
    // entity ID
    merge_trailing_entities: bool,
}
impl DigState {
    fn should_reset_for(&self, target: &ToolTargetWithId) -> bool {
        if self.merge_trailing_entities {
            if let ToolTargetWithId::Entity(target_id, _) = &self.target {
                if let ToolTargetWithId::Entity(id, _) = target {
                    return id.entity_id != target_id.entity_id;
                }
            }
        }
        self.target != *target
    }
}

fn target_properties(
    state: &ClientState,
    target: ToolTargetWithId,
    ext: Option<ClientExtendedData>,
) -> TargetProperties<'_> {
    match target {
        ToolTargetWithId::Block(coord, id) => {
            if let Some(blockdef) = state.block_types.get_blockdef(id) {
                let hover_text: Option<Cow<str>> = if blockdef.has_client_extended_data {
                    ext.map(|x| {
                        x.block_text
                            .iter()
                            .map(|x| x.text.clone())
                            // needless allocation but oh well, temporary impl
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .map(|x| Cow::Owned(x))
                } else {
                    None
                };

                TargetProperties {
                    target_groups: &blockdef.groups,
                    base_dig_time: blockdef.base_dig_time,
                    merge_trailing_entities: false,
                    interact_key_options: &blockdef.interact_key_option,
                    hover_text,
                }
            } else {
                TargetProperties {
                    target_groups: &FALLBACK_GROUPS,
                    base_dig_time: 1.0,
                    merge_trailing_entities: false,
                    interact_key_options: &[],
                    hover_text: Some(Cow::Borrowed("")),
                }
            }
        }
        ToolTargetWithId::Entity(_target, class) => state
            .entity_renderer
            .client_info(class)
            .map(|x| TargetProperties {
                target_groups: &x.tool_interaction_groups,
                base_dig_time: x.base_dig_time,
                merge_trailing_entities: x.merge_trailing_entities_for_dig,
                interact_key_options: &x.interact_key_options,
                // TODO: hover text for entities
                hover_text: None,
            })
            .unwrap_or(TargetProperties {
                // Undiggable - for now
                target_groups: &[],
                base_dig_time: 1.0,
                merge_trailing_entities: false,
                interact_key_options: &[],
                hover_text: None,
            }),
    }
}

lazy_static! {
    static ref FALLBACK_GROUPS: Box<[String]> = Box::new([String::new()]);
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
    menu_entries: Arc<Vec<InteractKeyOption>>,
    selected_menu_entry: Option<(ToolTargetWithId, usize)>,
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
            menu_entries: Arc::new(vec![]),
            selected_menu_entry: None,
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
                target_properties(client_state, progress.target, None).target_groups,
            );
        }
        self.current_item = item;
    }

    // Compute a new selected block.
    pub(crate) fn update(
        &mut self,
        client_state: &ClientState,
        player_pos: PlayerPositionUpdate,
        delta_nanos: u64,
        tick: u64,
    ) -> ToolState {
        let (_dist, pointee, neighbor, target_properties) = match Self::compute_pointee(
            client_state,
            &player_pos,
            tick,
            &self.fallback_blockdef,
            &self.current_item_interacting_groups,
        ) {
            Some(x) => x,
            None => {
                // Ensure that we don't leave pending input events that fire when we finally do get a pointee
                let mut input = client_state.input.lock();
                input.take_just_pressed(BoundAction::Dig);
                input.take_just_released(BoundAction::Dig);
                input.take_just_pressed(BoundAction::Place);
                input.take_just_pressed(BoundAction::Interact);
                self.dig_progress = None;
                self.selected_menu_entry = None;
                return ToolState {
                    pointee: None,
                    neighbor: None,
                    action: None,
                    interact_key_options: None,
                    selected_interact_option: 0,
                    hover_text: None,
                };
            }
        };

        match self.selected_menu_entry {
            Some((target, _)) => {
                // menu is up, if pointee changed, reset it
                if target.without_trailing_index() != pointee.without_trailing_index() {
                    if target_properties.interact_key_options.is_empty() {
                        self.menu_entries = Arc::new(vec![]);
                        self.selected_menu_entry = None;
                    } else {
                        self.menu_entries =
                            Arc::new(target_properties.interact_key_options.to_vec());
                        self.selected_menu_entry = Some((pointee, 0));
                    }
                }
            }
            None => {
                // No current menu up, bring one up
                if !target_properties.interact_key_options.is_empty() {
                    self.menu_entries = Arc::new(target_properties.interact_key_options.to_vec());
                    self.selected_menu_entry = Some((pointee, 0))
                }
            }
        }

        let mut input = client_state.input.lock();
        let mut keyed_command = None;
        if let Some((_, slot)) = &mut self.selected_menu_entry {
            // Take these away, so the HUD can't have them
            let scroll_slots = input.take_scroll_slots();
            let hotbar_input = input.take_hotbar_selection();
            if let Some(hotbar_input) = hotbar_input {
                if (hotbar_input as usize) < self.menu_entries.len() {
                    *slot = hotbar_input as usize;
                    keyed_command = Some(hotbar_input);
                }
            }
            if self.menu_entries.len() > 0 {
                *slot = ((*slot as i32) - scroll_slots).rem_euclid(self.menu_entries.len() as i32)
                    as usize;
            }
        }

        let mut action = None;
        if input.is_pressed(BoundAction::Dig) && self.can_dig_place {
            if self
                .dig_progress
                .as_ref()
                .map_or(true, |x: &DigState| x.should_reset_for(&pointee))
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
                    merge_trailing_entities: target_properties.merge_trailing_entities,
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
                        (delta_nanos as f64 / 1_000_000_000.0) / dig_seconds
                    }
                    Some(DigBehavior::ScaledTime(scale)) => {
                        (delta_nanos as f64 / 1_000_000_000.0)
                            / (scale * dig_progress.base_durability)
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
        } else if let Some(cmd) = keyed_command {
            if self.can_tap_interact {
                action = Some(GameAction::InteractKey(super::InteractKeyAction {
                    target: pointee.target(),
                    item_slot: self.current_slot,
                    player_pos,
                    menu_entry: self
                        .menu_entries
                        .get(cmd as usize)
                        .map(|x| x.id.clone())
                        .unwrap_or(String::new()),
                }))
            }
        } else if input.take_just_pressed(BoundAction::Interact) && self.can_tap_interact {
            action = Some(GameAction::InteractKey(super::InteractKeyAction {
                target: pointee.target(),
                item_slot: self.current_slot,
                player_pos,
                menu_entry: self
                    .selected_menu_entry
                    .and_then(|(_, index)| self.menu_entries.get(index))
                    .map(|x| x.id.clone())
                    .unwrap_or(String::new()),
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
                client_state.egui.lock().push_status_bar(
                    Duration::from_secs(5),
                    "You don't have permission to dig".to_string(),
                );
            }
            if input.peek_just_pressed(BoundAction::Place) {
                client_state.egui.lock().push_status_bar(
                    Duration::from_secs(5),
                    "You don't have permission to place".to_string(),
                );
            }
        }

        if let Some(action) = &action {
            log::info!("Sending player action: {:?}", action);
        }

        ToolState {
            pointee: Some(pointee),
            neighbor,
            action,
            interact_key_options: if self.menu_entries.is_empty()
                || self.selected_menu_entry.is_none()
            {
                None
            } else {
                Some(self.menu_entries.clone())
            },
            selected_interact_option: self.selected_menu_entry.map(|x| x.1).unwrap_or(0),
            hover_text: target_properties.hover_text.map(|x| x.to_string()),
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
        client_state: &'a ClientState,
        last_pos: &PlayerPositionUpdate,
        tick: u64,
        fallback_blockdef: &BlockTypeDef,
        current_item_interacting_groups: &[Vec<String>],
    ) -> Option<(
        f64,
        ToolTargetWithId,
        Option<BlockCoordinate>,
        TargetProperties<'a>,
    )> {
        let block_pointee = Self::compute_block_pointee(
            client_state,
            last_pos,
            fallback_blockdef,
            current_item_interacting_groups,
        );
        let entity_pointee = Self::compute_entity_pointee(client_state, last_pos, tick);
        if let Some(blk) = block_pointee {
            if let Some(ent) = entity_pointee {
                if blk.0 > ent.0 {
                    Some(ent)
                } else {
                    Some(blk)
                }
            } else {
                Some(blk)
            }
        } else if let Some(x) = entity_pointee {
            Some(x)
        } else {
            None
        }
    }

    fn compute_block_pointee<'a>(
        client_state: &'a ClientState,
        last_pos: &PlayerPositionUpdate,
        fallback_blockdef: &BlockTypeDef,
        current_item_interacting_groups: &[Vec<String>],
    ) -> Option<(
        f64,
        ToolTargetWithId,
        Option<BlockCoordinate>,
        TargetProperties<'a>,
    )> {
        let pos = last_pos.position;
        let face = last_pos.face_unit_vector();
        let end = pos + (POINTEE_DISTANCE * face);

        let delta_inv = vec3(1.0 / (face.x), 1.0 / (face.y), 1.0 / (face.z));

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
            let chunk = client_state.chunks.chunks.get(&coord.chunk());
            if let Some(chunk) = chunk {
                let (id, ext) = chunk.get_single_with_extended_data(coord.offset());
                let block_def = client_state
                    .block_types
                    .get_blockdef(id)
                    .unwrap_or(fallback_blockdef);
                if let Some(t) = check_intersection(
                    vec3(coord.x as f64, coord.y as f64, coord.z as f64),
                    block_def,
                    pos,
                    delta_inv,
                    id.variant(),
                ) {
                    let target = ToolTargetWithId::Block(coord, id);
                    for rule in current_item_interacting_groups.iter() {
                        if rule.iter().all(|x| block_def.groups.contains(x)) {
                            return Some((
                                t,
                                target,
                                prev,
                                target_properties(client_state, target, ext),
                            ));
                        }
                    }
                }
                prev = Some(coord);
            }
        }
        None
    }

    fn compute_entity_pointee<'a>(
        client_state: &'a ClientState,
        last_pos: &PlayerPositionUpdate,
        tick: u64,
    ) -> Option<(
        f64,
        ToolTargetWithId,
        Option<BlockCoordinate>,
        TargetProperties<'a>,
    )> {
        let pos = last_pos.position;
        let face = last_pos.face_unit_vector();
        client_state
            .entities
            .lock()
            .raycast(
                pos,
                face,
                tick,
                &client_state.entity_renderer,
                POINTEE_DISTANCE as f32,
            )
            .map(|(entity_id, trail_index, class, distance)| {
                let target = ToolTargetWithId::Entity(
                    EntityTarget {
                        entity_id,
                        trailing_entity_index: trail_index,
                    },
                    class,
                );

                (
                    distance as f64,
                    target,
                    None,
                    target_properties(client_state, target, None),
                )
            })
    }
}

fn check_intersection(
    block_coord: cgmath::Vector3<f64>,
    block_def: &BlockTypeDef,
    pos: cgmath::Vector3<f64>,
    delta_inv: cgmath::Vector3<f64>,
    variant: u16,
) -> Option<f64> {
    let bcv = vec3(
        block_coord.x as f64,
        block_coord.y as f64,
        block_coord.z as f64,
    );
    match &block_def.tool_custom_hitbox {
        Some(boxes) => {
            if boxes.boxes.is_empty() {
                check_intersection_core(
                    pos,
                    delta_inv,
                    vec3(-0.5, -0.5, -0.5) + bcv,
                    vec3(0.5, 0.5, 0.5) + bcv,
                )
            } else {
                let mut best_t = None;
                for aa_box in &boxes.boxes {
                    if aa_box.variant_mask != 0 && aa_box.variant_mask & variant as u32 == 0 {
                        continue;
                    }

                    let (min, max) = apply_aabox_transformation(aa_box, block_coord, variant);

                    // Disregard T itself, just check whether it's a hit
                    if let Some(t) = check_intersection_core(pos, delta_inv, min, max) {
                        if best_t.is_none() || t < best_t? {
                            best_t = Some(t);
                        }
                    }
                }
                best_t
            }
        }
        _ => {
            // We need to compute the hit distance. If we're already here, and the block has a
            // normal hitbox, then we *have* to hit it. However, we don't have a better way then to
            // run check_intersection_core anyway, so let's sanity check that it returns true.
            //
            // This will provide a consistency check that our two collision algorithms (WalkVoxels
            // and check_intersection_core) are consistent.
            let t = check_intersection_core(
                pos,
                delta_inv,
                vec3(-0.5, -0.5, -0.5) + bcv,
                vec3(0.5, 0.5, 0.5) + bcv,
            );
            if t.is_none() {
                log::warn!("Expected to hit a block that takes up its full extent, but didn't");
            }
            t
        }
    }
}

#[inline]
pub(crate) fn check_intersection_core<F: Float>(
    pos: Vector3<F>,
    delta_inv: Vector3<F>,
    min: Vector3<F>,
    max: Vector3<F>,
) -> Option<F> {
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
    if hit {
        Some(t_min)
    } else {
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
        appearance: None,
        groups: vec![],
        interaction_rules: rules,
        quantity_type: None,
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
    // The action taken during this frame, if there was one
    pub(crate) action: Option<GameAction>,
    // The menu entries, if a menu is up (None vec otherwise)
    pub(crate) interact_key_options: Option<Arc<Vec<InteractKeyOption>>>,
    // Index of selected menu item, irrelevant if interact_key_options is empty
    pub(crate) selected_interact_option: usize,
    // The header for the menu, if a menu is up
    pub(crate) hover_text: Option<String>,
}

impl ToolState {
    pub(crate) fn pointee_debug(&self, client_state: &ClientState) -> Option<String> {
        match self.pointee.as_ref()? {
            ToolTargetWithId::Block(_, id) => client_state
                .block_types
                .get_blockdef(*id)
                .map(|x| {
                    format!(
                        "Block: {id:x?}: {} {}",
                        x.short_name,
                        client_state.block_types.debug_properties(*id)
                    )
                })
                .or_else(|| {
                    Some(format!(
                        "Block: {id:x?}: ??? {}",
                        client_state.block_types.debug_properties(*id)
                    ))
                }),
            ToolTargetWithId::Entity(_, class) => client_state
                .entity_renderer
                .class_name(*class)
                .map(|x| format!("Entity: {class:x?}: {x}"))
                .or_else(|| Some(format!("Entity: {class:x?}: ???"))),
        }
    }
}

const POINTEE_DISTANCE: f64 = 6.;
// line_drawing seems to have problems when using Center rather than Corner
// Fudge it manually
// TODO file a bug for that crate
const RAYCAST_FUDGE_VEC: cgmath::Vector3<f64> = vec3(0.5, 0.5, 0.5);
