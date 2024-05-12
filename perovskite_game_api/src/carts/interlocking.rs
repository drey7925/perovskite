use std::sync::atomic::AtomicUsize;

use anyhow::{bail, Result};
use perovskite_core::{block_id::BlockId, coordinates::BlockCoordinate};
use perovskite_server::game_state::event::HandlerContext;
use rand::Rng;

use crate::carts::signals;

use super::{
    signals::{
        automatic_signal_acquire, interlocking_signal_preacquire, query_interlocking_signal,
        SignalConfig, SignalParseOutcome,
    },
    tracks::ScanState,
    CartsGameBuilderExtension,
};

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub(super) enum SwitchState {
    Straight,
    Diverging,
}
// A RAII wrapper that releases signals and switches when it goes out of scope
#[derive(Clone)]
struct SignalTransaction<'a> {
    // The coordinate, the expected block ID, the block ID to roll back to, and the block ID to commit to
    map_changes: Vec<(BlockCoordinate, BlockId, BlockId, BlockId)>,
    handler_context: &'a HandlerContext<'a>,
}
impl<'a> SignalTransaction<'a> {
    /// Commits the intended changes to the interlocking
    ///
    /// To be used when we've successfully found a path through the interlocking
    fn commit(mut self) {
        for (coord, _expected, _rollback, commit) in self.map_changes.drain(..) {
            self.handler_context
                .game_map()
                .mutate_block_atomically(coord, |block, _ext| {
                    *block = commit;
                    Ok(())
                })
                .log_error();
        }
        self.map_changes.clear();
        drop(self);
    }

    fn insert(
        &mut self,
        coord: BlockCoordinate,
        expected: BlockId,
        rollback: BlockId,
        commit: BlockId,
    ) {
        self.map_changes.push((coord, expected, rollback, commit));
    }

    fn rollback(self) {
        drop(self);
    }

    fn new(handler_context: &'a HandlerContext<'a>) -> SignalTransaction<'a> {
        SignalTransaction {
            map_changes: Vec::new(),
            handler_context,
        }
    }
}
impl Drop for SignalTransaction<'_> {
    fn drop(&mut self) {
        for (coord, expected, rollback, _commit) in &self.map_changes {
            self.handler_context
                .game_map()
                .mutate_block_atomically(*coord, |b, _ext| {
                    if *b != *expected {
                        tracing::warn!(
                            "expected {:?} to be {:?}, but it was actually {:?}",
                            coord,
                            expected,
                            b
                        );
                    }
                    *b = *rollback;
                    Ok(())
                })
                .log_error();
        }
    }
}

/// Attempts to find a path through an interlocking.
///
/// Args:
///     initial_state: The initial state of the cart. This should be at the same block where
///         the initial interlocking signal was encountered
///     max_scan_distance: The maximum distance (in tiles) to scan.
/// Side Effects (in the game map):
///     * Sets all relevant signals to permissive aspects with the correct turnout indicators lit
///     * Sets all switches to the relevant state
/// Returns:
///     Some(InterlockingResolution) if a path was found, None otherwise
///     Once this returns a valid resolution, the cart should begin moving, following the given path.
///         As the cart moves, it should verify that all signals remain set to permissive, flip them to
///         occupied as it enters them, release them back to restrictive-unoccupied as it leaves blocks,
///         verify that all switches are set to the relevant state, and release the switches back to
///         an unset state as the cart passes.
pub(super) async fn interlock_cart(
    handler_context: HandlerContext<'_>,
    initial_state: ScanState,
    max_scan_distance: usize,
    cart_config: CartsGameBuilderExtension,
) -> Result<Option<Vec<InterlockingStep>>> {
    while !handler_context.is_shutting_down() {
        let resolution = single_pathfind_attempt(
            &handler_context,
            initial_state.clone(),
            max_scan_distance,
            &cart_config,
        )?;
        if resolution.is_some() {
            return Ok(dbg!(resolution));
        } else {
            tracing::debug!("No path found, trying again");
            // Randomized backoff, 500-1000 msec
            //
            // A "smarter" deadlock resolution strategy like wound-wait or wait-die would
            // require too many bits of block state. We could put that state into extended data,
            // but that makes extended data error handling more tricky.
            let backoff = rand::thread_rng().gen_range(500..1000);
            tokio::time::sleep(std::time::Duration::from_millis(backoff)).await;
            return Ok(None);
        }
    }
    Ok(None)
}

fn single_pathfind_attempt(
    handler_context: &HandlerContext<'_>,
    initial_state: ScanState,
    max_scan_distance: usize,
    cart_config: &CartsGameBuilderExtension,
) -> Result<Option<Vec<InterlockingStep>>> {
    let mut steps = vec![];
    let mut state = initial_state;
    let mut transaction = SignalTransaction::new(handler_context);

    let mut left_pending = false;
    let mut right_pending = false;

    // Scan forward through tracks until we either:
    // 1. Reach the end of the track (which means that we can finish the path)
    // 2. Reach a correctly-facing automatic signal (which means that we're returning to simple non-interlocking tracks
    //      in the correct direction)
    // 3. Reach a starting signal (TODO implement this)
    // 4. Run out of steps (in which case we clean up and give up)
    // 5. Encounter an automatic signal facing the wrong way (in which case we clean up and give up)
    //      Note that interlocking signals facing the wrong way are OK
    //
    // Each time we encounter an interlocking signal facing the correct way, we'll query it to determine the direction
    //      in which the current cart should go. We then store that until a later move consumes that decision, since the signal might be
    //      slightly upstream of the switch, and there might be both a left and a right switch under control of the signal.
    //      In particular, if we see a switch resolve to the left, we'll take the first left-capable switch, unless we either see another interlocking signal
    //        before then.
    //      e.g. in the case below, if we are signalled left, we will take the second switch in the chain, since it's the first left-capable switch
    //                                    -------------------->
    //                                   /
    //      [signal]-----[switch]----[switch]------------------>
    //                        \
    //                         ----------->
    //
    // TODO later: If we are asked to fork, we will take up to one switch in each direction as we fan out and explore multiple paths.
    while steps.len() < max_scan_distance {
        let mut acquired_signal = BlockId::from(0);

        let track_coord = state.block_coord;
        let signal_position = track_coord.try_delta(0, 2, 0);
        // first parse the signal we see...
        if let Some(signal_coord) = signal_position {
            let query_result = handler_context.game_map().mutate_block_atomically(
                signal_coord,
                |block, ext| {
                    // if we have to give up and roll back, we'll restore to this
                    let rollback_block = *block;
                    if block.equals_ignore_variant(cart_config.interlocking_signal)
                        && state.signal_rotation_ok(block.variant())
                    {
                        match interlocking_signal_preacquire(signal_coord, block, *block) {
                            signals::SignalLockOutcome::Contended
                            | signals::SignalLockOutcome::InvalidSignal => {
                                Ok(SignalParseOutcome::Deny)
                            }
                            signals::SignalLockOutcome::Acquired => {
                                // We didn't conflict with anyone else for the signal, and we now hold the lock for it.
                                let query_result = query_interlocking_signal(ext.as_ref(), "test")?;
                                if let Some(variant) = query_result.adjust_variant(block.variant())
                                {
                                    // The signal's parsing outcome led to a new variant.
                                    // If we commit the transaction, we'll apply this block to the map
                                    let commit_block = block.with_variant(variant).unwrap();
                                    // *block was already updated when we called preacquire. If we have to commit or roll back,
                                    // we expect to see the preacquired block.
                                    transaction.insert(
                                        signal_coord,
                                        *block,
                                        rollback_block,
                                        commit_block,
                                    );
                                    acquired_signal = commit_block;
                                }
                                Ok(query_result)
                            }
                        }
                    } else if block.equals_ignore_variant(cart_config.automatic_signal)
                        && state.signal_rotation_ok(block.variant())
                    {
                        match automatic_signal_acquire(signal_coord, block, *block) {
                            signals::SignalLockOutcome::InvalidSignal => {
                                Ok(SignalParseOutcome::Deny)
                            }
                            signals::SignalLockOutcome::Acquired => {
                                // We didn't conflict with anyone else for the signal, and we now hold the lock for it.
                                acquired_signal = *block;
                                Ok(SignalParseOutcome::AutomaticSignal)
                            }
                            signals::SignalLockOutcome::Contended => {
                                // We can't get the lock for the signal so we can't leave the interlocking.
                                Ok(SignalParseOutcome::Deny)
                            }
                        }
                    } else {
                        Ok(SignalParseOutcome::NotASignal)
                    }
                },
            )?;

            match query_result {
                SignalParseOutcome::Straight => {
                    left_pending = false;
                    right_pending = false;
                }
                SignalParseOutcome::DivergingLeft => {
                    left_pending = true;
                    right_pending = false;
                }
                SignalParseOutcome::DivergingRight => {
                    left_pending = false;
                    right_pending = true;
                }
                SignalParseOutcome::Fork => {
                    // forking is not yet supported.
                    // In principle it should be simple - we just have to recursively call this function and ensure that on success,
                    // we commit the right set of changes by passing a transaction with the correct outcome of the branch added to it.
                    todo!()
                }
                SignalParseOutcome::Deny => {
                    return Ok(None);
                }
                SignalParseOutcome::NotASignal => {
                    // do nothing, keep scanning
                }
                SignalParseOutcome::AutomaticSignal => {
                    steps.push(InterlockingStep {
                        scan_state: state.clone(),
                        enter_signal: acquired_signal,
                        pass_switch: BlockId::from(0),
                    });
                    transaction.commit();
                    return Ok(Some(steps));
                }
            }
        }
        // then set the switch accordingly.
        let mut set_switch = BlockId::from(0);
        if state.is_switch_eligible() {
            let switch_coord = track_coord.try_delta(0, -1, 0);

            if let Some(switch_coord) = switch_coord {
                let switch_decision = handler_context.game_map().mutate_block_atomically(
                    switch_coord,
                    |switch_block, _ext| {
                        if switch_block.equals_ignore_variant(cart_config.switch_diverging)
                            || switch_block.equals_ignore_variant(cart_config.switch_straight)
                        {
                            // The switch is already set, so we can't pass
                            return Ok(None);
                        } else if switch_block.equals_ignore_variant(cart_config.switch_unset) {
                            // The switch is unset, so we can pass.
                            let mut will_diverge = false;
                            if state.can_diverge_left() && left_pending {
                                left_pending = false;
                                will_diverge = true;
                            } else if state.can_diverge_right() && right_pending {
                                right_pending = false;
                                will_diverge = true;
                            } else if state.can_converge() && state.is_diverging {
                                will_diverge = true;
                            }
                            let new_block = if will_diverge {
                                cart_config.switch_diverging.with_variant_of(*switch_block)
                            } else {
                                cart_config.switch_straight.with_variant_of(*switch_block)
                            };
                            transaction.insert(switch_coord, new_block, *switch_block, new_block);
                            *switch_block = new_block;
                            set_switch = new_block;

                            if will_diverge {
                                Ok(Some(SwitchState::Diverging))
                            } else {
                                Ok(Some(SwitchState::Straight))
                            }
                        } else {
                            // not a switch block. We can go straight through it, but we can't
                            // diverge or approach from the trailing diverging side
                            if state.is_diverging {
                                Ok(None)
                            } else {
                                Ok(Some(SwitchState::Straight))
                            }
                        }
                    },
                )?;
                match switch_decision {
                    Some(SwitchState::Diverging) => {
                        // if we're approaching from the leading side, we need to set the diverging bit
                        // If we're approaching from the reverse side, we are done - the track patterns will
                        // guide us onto the correct track.
                        state.is_diverging = true;
                    }
                    Some(SwitchState::Straight) => {
                        // Nothing to do, the tracks already guide us straight unless we set the diverging bit.
                    }
                    None => {
                        // We can't pass through the switch
                        return Ok(None);
                    }
                }
            }
        }

        steps.push(InterlockingStep {
            scan_state: state.clone(),
            enter_signal: acquired_signal,
            pass_switch: set_switch,
        });
        let new_state =
            state.advance::<false>(|coord| handler_context.game_map().get_block(coord).into())?;
        match new_state {
            super::tracks::ScanOutcome::Success(new_state) => {
                state = new_state;
            }
            super::tracks::ScanOutcome::CannotAdvance => {
                transaction.commit();
                return Ok(Some(steps));
            }
            super::tracks::ScanOutcome::NotOnTrack => {
                return Ok(None);
            }
            super::tracks::ScanOutcome::Deferral(_) => {
                panic!("Got a deferral from track scan, but we're not in a deferrable context and the block getter shouldn't defer");
            }
        }
    }
    Ok(None)
}
trait LogError {
    fn log_error(self);
}
impl LogError for Result<(), anyhow::Error> {
    fn log_error(self) {
        if let Err(e) = self {
            tracing::error!("{}", e);
        }
    }
}

#[derive(Debug, Clone)]
pub struct InterlockingStep {
    pub scan_state: ScanState,
    pub enter_signal: BlockId,
    pub pass_switch: BlockId,
}
