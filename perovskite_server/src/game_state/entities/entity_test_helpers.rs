use anyhow::{bail, Result};
use cgmath::Vector3;
use perovskite_core::coordinates::{BlockCoordinate, ChunkOffset};
use std::pin::Pin;

use crate::game_state::{
    entities::{EntityCoroutine, EntityCoroutineServices, MoveQueue, MoveQueueType},
    GameState,
};

pub struct CoroutineTester {
    coro: Pin<Box<dyn EntityCoroutine>>,
    coro_engaged: bool,
    move_queue: MoveQueue,
    last_tick: u64,
    last_pos: Vector3<f64>,
    last_sequence: u64,
    current_move: Option<super::StoredMovement>,
}
impl CoroutineTester {
    /// Advances the coroutine by one planned move, with as many deferrals as needed.
    /// Deferrals run inline; this must be called from within an async runtime (e.g. run_in_server or within a handler)
    pub fn advance(&mut self, gs: &GameState) -> Result<()> {
        // warm up chunks around the current position
        let chunk = BlockCoordinate::try_from(self.last_pos)?.chunk();

        // warm up some chunks around the entity under test
        // (todo? do we need a way to avoid this for any tests?)
        for dx in -1..=1 {
            for dz in -1..=1 {
                for dy in -1..=1 {
                    if let Some(neighbor) = chunk.try_delta(dx, dy, dz) {
                        let _ = gs
                            .game_map()
                            .get_block(neighbor.with_offset(ChunkOffset::from_index(0)))?;
                    }
                }
            }
        }

        if !self.coro_engaged {
            bail!("Coroutine was previously disengaged with StopCoroutineControl");
        }
        let services = EntityCoroutineServices {
            game_state: &gs.clone_self(),
        };

        let mut whence = self.last_pos;
        let mut queue_time = 0.0;
        for mv in self.move_queue.as_slice() {
            whence = mv.movement.pos_after_move(whence);
            queue_time += mv.movement.move_time;
        }
        let mut last_result = self.coro.as_mut().plan_move(
            &services,
            self.last_pos,
            whence,
            queue_time,
            self.move_queue.remaining_capacity(),
        );
        let move_result = loop {
            match last_result {
                super::CoroutineResult::Successful(entity_move_decision) => {
                    break entity_move_decision
                }
                super::CoroutineResult::_DeferredMoveResult(deferral) => {
                    break tokio::runtime::Handle::current().block_on(deferral.deferred_call);
                }
                super::CoroutineResult::_DeferredReenterCoroutine(deferral) => {
                    let deferred_result =
                        tokio::runtime::Handle::current().block_on(deferral.deferred_call);
                    last_result = self.coro.as_mut().continuation(
                        &services,
                        self.last_pos,
                        whence,
                        queue_time,
                        self.move_queue.remaining_capacity(),
                        deferred_result,
                        deferral.trace_buffer,
                    )
                }
            }
        };
        if let Some(old_move) = self.current_move.take() {
            self.last_pos = old_move.movement.pos_after_move(self.last_pos);
            self.last_tick += old_move.movement.time_in_ticks();
        }
        match move_result {
            super::EntityMoveDecision::QueueSingleMovement(movement) => {
                self.last_sequence = self.last_sequence.wrapping_add(1);
                self.current_move = Some(super::StoredMovement {
                    sequence: self.last_sequence,
                    start_tick: self.last_tick,
                    movement,
                });
            }
            super::EntityMoveDecision::QueueUpMultiple(mut movements) => {
                if movements.len() > self.move_queue.remaining_capacity() + 1 {
                    bail!(
                        "Coroutine returned {} movements but only {} capacity remains in queue",
                        movements.len(),
                        self.move_queue.remaining_capacity()
                    );
                }
                if movements.len() > 0 {
                    let first = movements.remove(0);
                    self.current_move = Some(super::StoredMovement {
                        sequence: self.last_sequence,
                        start_tick: self.last_tick,
                        movement: first,
                    });
                    self.last_sequence = self.last_sequence.wrapping_add(1);
                }
                for mv in movements {
                    self.move_queue.push(super::StoredMovement {
                        sequence: self.last_sequence,
                        start_tick: self.last_tick,
                        movement: mv,
                    });
                    self.last_sequence = self.last_sequence.wrapping_add(1);
                }
            }
            super::EntityMoveDecision::ResetMovement(start_pos, movement, initial_move_queue) => {
                self.last_pos = start_pos;
                self.last_sequence = 0;
                self.last_tick = 0;
                self.move_queue =
                    initial_move_queue.into_move_queue(self.last_tick, self.last_sequence);
                self.current_move = Some(super::StoredMovement {
                    sequence: self.last_sequence,
                    start_tick: self.last_tick,
                    movement,
                });
                self.last_sequence = self.last_sequence.wrapping_add(1);
            }
            super::EntityMoveDecision::ImmediateDespawn => {
                self.coro_engaged = false;
                self.current_move = None;
            }
            super::EntityMoveDecision::AskAgainLater(time) => {
                self.last_tick += (time * 1_000_000_000.0) as u64;
            }
            super::EntityMoveDecision::AskAgainLaterFlexible(range) => {
                self.last_tick += (range.start * 1_000_000_000.0) as u64;
            }
            super::EntityMoveDecision::StopCoroutineControl => {
                self.coro_engaged = false;
                self.current_move = None;
            }
        }
        Ok(())
    }

    /// Return the total duration of moves currently in the queue, including the current move.
    pub fn move_buffer(&self) -> f32 {
        if !self.coro_engaged {
            return 0.0;
        }
        let mut total_seconds = 0.0;
        if let Some(mv) = &self.current_move {
            total_seconds += mv.movement.move_time;
        }
        for mv in self.move_queue.as_slice() {
            total_seconds += mv.movement.move_time;
        }
        total_seconds
    }

    /// Return the position the entity would be in after the current move completes.
    pub fn current_position(&self) -> Vector3<f64> {
        self.last_pos
    }

    /// Return the position the entity would be in after all moves in the queue complete.
    pub fn post_queue_position(&self) -> Vector3<f64> {
        let mut pos = self.last_pos;
        if let Some(mv) = &self.current_move {
            pos = mv.movement.pos_after_move(pos);
        }
        for mv in self.move_queue.as_slice() {
            pos = mv.movement.pos_after_move(pos);
        }
        pos
    }

    pub fn is_engaged(&self) -> bool {
        self.coro_engaged
    }

    pub fn new(
        coro: Pin<Box<dyn EntityCoroutine>>,
        queue_type: MoveQueueType,
        start_pos: Vector3<f64>,
        gs: &GameState,
    ) -> Result<Self> {
        let mut res = CoroutineTester {
            coro,
            coro_engaged: true,
            move_queue: match queue_type {
                MoveQueueType::SingleMove => MoveQueue::SingleMove(None),
                MoveQueueType::Buffer8 => MoveQueue::Buffer8(Default::default()),
                MoveQueueType::Buffer64 => MoveQueue::Buffer64(Default::default()),
            },
            last_tick: 0,
            last_pos: start_pos,
            last_sequence: 0,
            current_move: None,
        };
        res.advance(gs)?;
        Ok(res)
    }
}
