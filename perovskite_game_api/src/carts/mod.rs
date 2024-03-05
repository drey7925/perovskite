use std::collections::VecDeque;

// This is a temporary implementation used while developing the entity system
use anyhow::Result;
use async_trait::async_trait;
use cgmath::vec3;
use perovskite_core::{block_id::BlockId, chat::ChatMessage, coordinates::BlockCoordinate};
use perovskite_server::game_state::{
    chat::commands::ChatCommandHandler,
    entities::{EntityCoroutine, Movement},
    event::{EventInitiator, HandlerContext},
    GameStateExtension,
};

use crate::{
    blocks::{BlockBuilder, CubeAppearanceBuilder},
    game_builder::{GameBuilderExtension, StaticBlockName, StaticTextureName},
    include_texture_bytes,
};

#[derive(Clone)]
struct CartsGameBuilderExtension {
    rail_block: BlockId,
    speedpost_1: BlockId,
    speedpost_2: BlockId,
    speedpost_3: BlockId,
}
impl Default for CartsGameBuilderExtension {
    fn default() -> Self {
        CartsGameBuilderExtension {
            rail_block: 0.into(),
            speedpost_1: 0.into(),
            speedpost_2: 0.into(),
            speedpost_3: 0.into(),
        }
    }
}

pub fn register_carts(game_builder: &mut crate::game_builder::GameBuilder) -> Result<()> {
    let rail_tex = StaticTextureName("carts:rail");
    let speedpost1_tex = StaticTextureName("carts:speedpost1");
    let speedpost2_tex = StaticTextureName("carts:speedpost2");
    let speedpost3_tex = StaticTextureName("carts:speedpost3");
    include_texture_bytes!(game_builder, rail_tex, "textures/rail.png")?;
    include_texture_bytes!(game_builder, speedpost1_tex, "textures/speedpost_1.png")?;
    include_texture_bytes!(game_builder, speedpost2_tex, "textures/speedpost_2.png")?;
    include_texture_bytes!(game_builder, speedpost3_tex, "textures/speedpost_3.png")?;

    let rail = game_builder.add_block(
        BlockBuilder::new(StaticBlockName("carts:rail"))
            .set_cube_appearance(CubeAppearanceBuilder::new().set_single_texture(rail_tex)),
    )?;

    let speedpost1 = game_builder.add_block(
        BlockBuilder::new(StaticBlockName("carts:speedpost1"))
            .set_cube_appearance(CubeAppearanceBuilder::new().set_single_texture(speedpost1_tex)),
    )?;
    let speedpost2 = game_builder.add_block(
        BlockBuilder::new(StaticBlockName("carts:speedpost2"))
            .set_cube_appearance(CubeAppearanceBuilder::new().set_single_texture(speedpost2_tex)),
    )?;
    let speedpost3 = game_builder.add_block(
        BlockBuilder::new(StaticBlockName("carts:speedpost3"))
            .set_cube_appearance(CubeAppearanceBuilder::new().set_single_texture(speedpost3_tex)),
    )?;

    let ext = game_builder.builder_extension::<CartsGameBuilderExtension>();
    ext.rail_block = rail.id;
    ext.speedpost_1 = speedpost1.id;
    ext.speedpost_2 = speedpost2.id;
    ext.speedpost_3 = speedpost3.id;

    Ok(())
}

impl GameBuilderExtension for CartsGameBuilderExtension {
    fn pre_run(&mut self, server_builder: &mut perovskite_server::server::ServerBuilder) {
        server_builder.add_extension(self.clone());
        server_builder
            .register_command(
                "spawn_cart",
                Box::new(CartSpawner::new(self.clone())),
                "Spawn a cart at the current location",
            )
            .unwrap();
    }
}
impl GameStateExtension for CartsGameBuilderExtension {}

struct CartSpawner {
    config: CartsGameBuilderExtension,
}
impl CartSpawner {
    fn new(config: CartsGameBuilderExtension) -> Self {
        Self { config }
    }
}
#[async_trait]
impl ChatCommandHandler for CartSpawner {
    async fn handle(&self, _message: &str, context: &HandlerContext<'_>) -> Result<()> {
        let position = if let Some(location) = context.initiator().position() {
            location.position
        } else {
            context
                .initiator()
                .send_chat_message_async(ChatMessage::new_server_message("No location"))
                .await?;
            return Ok(());
        };
        let quantized = BlockCoordinate::new(
            position.x.floor() as i32,
            position.y.floor() as i32,
            position.z.floor() as i32,
        );
        // find the nearest rail the same X and Z location
        let mut rail_pos = None;
        for dy in [0, 1, -1, 2, -2, 3, -3] {
            if let Some(coord) = quantized.try_delta(0, dy, 0) {
                if context.game_map().try_get_block(coord) == Some(self.config.rail_block) {
                    rail_pos = Some(coord);
                }
            }
        }
        let rail_pos = if let Some(rail_pos) = rail_pos {
            rail_pos
        } else {
            context
                .initiator()
                .send_chat_message_async(ChatMessage::new_server_message("No rail found"))
                .await?;
            return Ok(());
        };
        context
            .entities()
            .new_entity(
                b2vec(rail_pos.try_delta(0, 2, 0).unwrap()),
                Some(Box::pin(CartCoroutine {
                    config: self.config.clone(),
                    scheduled_segments: VecDeque::new(),
                    unplanned_segments: VecDeque::new(),
                    scan_position: rail_pos,
                    // Until we encounter a speed post, proceed at minimal safe speed
                    last_speed_post_indication: 0.5,
                    last_speed: 0.0,
                })),
            )
            .await;

        Ok(())
    }
}

fn b2vec(b: BlockCoordinate) -> cgmath::Vector3<f64> {
    cgmath::Vector3::new(b.x as f64, b.y as f64, b.z as f64)
}
fn vec2b(v: cgmath::Vector3<f64>) -> BlockCoordinate {
    BlockCoordinate::new(v.x as i32, v.y as i32, v.z as i32)
}

/// A segment of track where we know we can run
#[derive(Copy, Clone, Debug)]
struct TrackClearance {
    from: BlockCoordinate,
    to: BlockCoordinate,
    // The maximum speed we can run in this track segment
    max_speed: f32,
}
impl TrackClearance {
    fn manhattan_dist(&self) -> u32 {
        (self.from.x - self.to.x).abs() as u32
            + (self.from.y - self.to.y).abs() as u32
            + (self.from.z - self.to.z).abs() as u32
    }
}

const MAX_ACCEL: f32 = 4.5;
struct CartCoroutine {
    config: CartsGameBuilderExtension,
    // Segments where we've already calculated a braking curve. (clearance, starting speed, acceleration, time)
    // currently unused and empty
    scheduled_segments: VecDeque<(TrackClearance, f32, f32, f32)>,
    // Segments where we don't yet have a braking curve
    unplanned_segments: VecDeque<TrackClearance>,
    // The current coordinate where we're going to scan next
    scan_position: BlockCoordinate,
    // The last speed post we encountered while scanning
    last_speed_post_indication: f32,
    // The last velocity we already delivered to the entity system
    last_speed: f32,
}
impl EntityCoroutine for CartCoroutine {
    fn plan_move(
        mut self: std::pin::Pin<&mut Self>,
        services: &perovskite_server::game_state::entities::EntityCoroutineServices<'_>,
        current_position: cgmath::Vector3<f64>,
        whence: cgmath::Vector3<f64>,
        when: f32,
    ) -> perovskite_server::game_state::entities::CoroutineResult {
        // Fill the unplanned segment queue unless we've scanned too far ahead or it's full
        let mut step_count = 0;
        for seg in self.scheduled_segments.iter() {
            step_count += seg.0.manhattan_dist();
        }
        for seg in self.unplanned_segments.iter() {
            step_count += seg.manhattan_dist();
        }
        println!("==========");
        println!(
            "cart coro: step_count = {}. {} in braking curve, {} unplanned",
            step_count,
            self.scheduled_segments.len(),
            self.unplanned_segments.len()
        );

        // hacky clean up
        let mut force_speed_post = false;
        // todo tune
        while step_count < 256
            && (self.scheduled_segments.len() + self.unplanned_segments.len()) < 16
        {
            if self.unplanned_segments.is_empty()
                || self.unplanned_segments.back().unwrap().max_speed
                    != self.last_speed_post_indication
                || force_speed_post
            {
                let empty_segment = TrackClearance {
                    from: self.scan_position,
                    to: self.scan_position,
                    max_speed: self.last_speed_post_indication,
                };
                self.unplanned_segments.push_back(empty_segment);
            }
            // TODO: self should store direction of movement.

            // On the test track, we proceed in the Z+ direction. Speed posts are in the +X direction relative to the track
            step_count += 1;
            // These should be checked, but they only fail at the end of the map.
            let next_scan = self.scan_position.try_delta(0, 0, 1).unwrap();
            let next_speed_post = next_scan.try_delta(1, 0, 0).unwrap();

            let scan_block = services.try_get_block(next_scan);
            let speed_post_block = services.try_get_block(next_speed_post);
            if scan_block != Some(self.config.rail_block) {
                // We're at the end of the track. The last segment already includes the last bit of track
                break;
            } else {
                self.scan_position = next_scan;
                self.unplanned_segments.back_mut().unwrap().to = next_scan;
            }

            if speed_post_block == Some(self.config.speedpost_1) {
                self.last_speed_post_indication = 1.0;
                force_speed_post = true;
            } else if speed_post_block == Some(self.config.speedpost_2) {
                self.last_speed_post_indication = 2.0;
                force_speed_post = true;
            } else if speed_post_block == Some(self.config.speedpost_3) {
                self.last_speed_post_indication = 30.0;
                force_speed_post = true;
            }
        }

        // Now, calculate the braking curve for the planned segment. For now, we'll rebuild the braking curve each time
        // Scan backwards through all unplanned segments, then all planned segments.
        // todo optimize this later

        // The maximum speed we can be moving at the end of the current track segment.
        let mut max_exit_speed = 0.0;
        // If true, we hit the max track speed for at least one segment. Otherwise, our braking curve is affected by
        // unknown track segments that we haven't scanned yet, and hence is not certain yet.
        let mut segments_schedulable = false;
        let mut schedulable_segments = VecDeque::new();

        // Iterate in reverse of track order
        for (idx, seg) in self.unplanned_segments.iter().enumerate().rev() {
            let distance = seg.manhattan_dist();
            // vf^2 = vi^2 + 2a(d)
            let mut max_entry_speed =
                (max_exit_speed * max_exit_speed + 2.0 * MAX_ACCEL * distance as f32).sqrt();

            if max_entry_speed > seg.max_speed {
                max_entry_speed = seg.max_speed;
            }
            println!("> unplanned segment, enter at {max_entry_speed}, exit at {max_exit_speed}, len {distance}");
            // Record the segment if it's schedulable, or if it's the only segment we have
            if segments_schedulable || idx == 0 {
                // Record the actual speed we need to stay under as we exit this segment
                // We push_front, so as we iterate, we get them in track order (which is what we need to now build the curve segment by segment)
                schedulable_segments.push_front((*seg, max_exit_speed));
                println!(">> planning it");
            }
            // This segment could be entered at maximum track speed and we would still stop in time. Any segments before it (in track order)
            // are now schedulable. This one is not yet schedulable, since we don't know how early in it we would need to start slowing down.
            if max_entry_speed >= seg.max_speed {
                segments_schedulable = true;
            }
            // The exit speed of the previous segment is the entry speed of the current segment
            max_exit_speed = max_entry_speed;
        }

        for _ in 0..schedulable_segments.len() {
            self.unplanned_segments.pop_front();
        }

        let mut last_speed = self
            .scheduled_segments
            .back()
            .map(|x| x.1)
            .unwrap_or(self.last_speed);
        for (seg, max_exit_speed) in schedulable_segments.into_iter() {
            // TODO - we should slice up the segment, rather than having a slow acceleration over the whole segment
            // This takes casework; the case where we have time to get to max speed, hold max speed, then slow down is easy
            // The case where we just slow down or just speed up is a trivial subset of that
            // The case where we speed up, don't reach max speed, and have to start slowing down, requires some annoying algebra.
            //
            // accelerating from last_velocity to max_exit_speed
            // vf^2 = vi^2 + 2a(d)
            let distance = seg.manhattan_dist();
            if distance == 0 {
                continue;
            }
            let desired_accel = (max_exit_speed * max_exit_speed - last_speed * last_speed)
                / (2.0 * distance as f32);
            let actual_accel = desired_accel.clamp(-MAX_ACCEL, MAX_ACCEL);
            let actual_speed =
                (last_speed * last_speed + 2.0 * actual_accel * distance as f32).sqrt();
            let time = (actual_speed - last_speed) / actual_accel;
            println!("> planned segment, enter at {last_speed}, want exit at {max_exit_speed}, got exit at {actual_speed}, len {distance}, desired accel {desired_accel}, actual accel {actual_accel}, time {time}");

            self.scheduled_segments
                .push_back((seg, last_speed, actual_accel, time));
            last_speed = actual_speed;
        }

        println!("scan position: {:?}", self.scan_position);

        if self.scheduled_segments.is_empty() {
            // No schedulable segments
            // Scan every second, trying to start again.
            println!("No schedulable segments");
            self.last_speed = 0.0;
            self.last_speed_post_indication = 0.5;
            return perovskite_server::game_state::entities::CoroutineResult::Successful(
                perovskite_server::game_state::entities::EntityMoveDecision::QueueUpMovement(
                    Movement {
                        velocity: cgmath::Vector3::new(0.0, 0.0, 0.0),
                        acceleration: cgmath::Vector3::new(0.0, 0.0, 0.0),
                        face_direction: 0.0,
                        move_time: 1.0,
                    },
                ),
            );
        } else {
            let segment = self.scheduled_segments.pop_front().unwrap();
            self.last_speed = segment.1 + (segment.2 * segment.3);

            println!(
                "returning a movement, speed is {}, accel is {}",
                self.last_speed, segment.2
            );
            return perovskite_server::game_state::entities::CoroutineResult::Successful(
                perovskite_server::game_state::entities::EntityMoveDecision::QueueUpMovement(
                    Movement {
                        velocity: cgmath::Vector3::new(0.0, 0.0, segment.1),
                        acceleration: cgmath::Vector3::new(0.0, 0.0, segment.2),
                        face_direction: 0.0,
                        move_time: segment.3,
                    },
                ),
            );
        }
    }
}
