use anyhow::Result;
use perovskite_core::{block_id::BlockId, coordinates::BlockCoordinate};
use perovskite_server::game_state::event::HandlerContext;
use rand::Rng;
use smallvec::SmallVec;

use crate::carts::signals::{
    self, starting_signal_acquire_back, starting_signal_depart_forward,
    starting_signal_preacquire_front,
};

use super::{
    signals::{
        automatic_signal_acquire, interlocking_signal_preacquire, query_interlocking_signal,
        query_starting_signal, SignalParseOutcome,
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
    cart_name: &str,
    max_scan_distance: usize,
    cart_config: CartsGameBuilderExtension,
    resume: Option<InterlockingResumeState>,
    last_speed_post: f32,
) -> Result<Option<InterlockingRoute>> {
    if !handler_context.is_shutting_down() {
        let resolution = single_pathfind_attempt(
            &handler_context,
            cart_name,
            initial_state.clone(),
            max_scan_distance,
            &cart_config,
            resume.clone(),
        )?;
        if let Some(resolution) = resolution {
            // TODO: Run this only when contention is detected. This requires tracking
            // contention in the cart coroutine state itself, since we can't loop in this function
            // for some reason.
            //
            // if any_contention {
            //     // fudge factor to avoid the case where cart speeds cycle up and down as the
            //     // trajectory optimizer runs into the back of the previous cart
            //     let exit_speed = resolution
            //         .steps
            //         .iter()
            //         .rev()
            //         .find_map(|step| step.speed_post)
            //         .unwrap_or(last_speed_post);

            //     tokio::time::sleep(std::time::Duration::from_secs_f64(
            //         0.5 * exit_speed as f64 / super::MAX_ACCEL,
            //     ))
            //     .await;
            // }
            return Ok(Some(resolution));
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
    cart_name: &str,
    initial_state: ScanState,
    max_scan_distance: usize,
    cart_config: &CartsGameBuilderExtension,
    resume: Option<InterlockingResumeState>,
) -> Result<Option<InterlockingRoute>> {
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
        let mut acquired_signal_was_reverse_starting = false;

        let track_coord = state.block_coord;
        let signal_position = track_coord.try_delta(0, 2, 0);

        let mut speed_post = None;

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
                                let query_result =
                                    query_interlocking_signal(ext.as_ref(), cart_name)?;
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
                                Ok(query_result.to_parse_outcome())
                            }
                        }
                    } else if resume.as_ref().is_some_and(|resume| {
                        resume
                            .starting_signal
                            .is_some_and(|(coord, _)| coord == signal_coord)
                    }) {
                        // This is the starting signal that this cart stopped in front of.
                        match starting_signal_depart_forward(signal_coord, block, *block) {
                            signals::SignalLockOutcome::InvalidSignal
                            | signals::SignalLockOutcome::Contended => Ok(SignalParseOutcome::Deny),
                            signals::SignalLockOutcome::Acquired => {
                                tracing::debug!("Starting signal acquired at {:?}", signal_coord);
                                // We didn't conflict with anyone else for the signal, and we now hold the lock for it.
                                // We use the interlocking signal query here since we know we're allowed to pass it now
                                let query_result =
                                    query_interlocking_signal(ext.as_ref(), cart_name)?;
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
                                Ok(query_result.to_parse_outcome())
                            }
                        }
                    } else if block.equals_ignore_variant(cart_config.starting_signal)
                        && state.signal_rotation_ok(block.variant())
                    {
                        match starting_signal_preacquire_front(signal_coord, block, *block) {
                            signals::SignalLockOutcome::InvalidSignal
                            | signals::SignalLockOutcome::Contended => Ok(SignalParseOutcome::Deny),
                            signals::SignalLockOutcome::Acquired => {
                                tracing::info!("Starting signal preacquired at {:?}", signal_coord);
                                // We didn't conflict with anyone else for the signal, and we now hold the lock for it.
                                // Furthermore, we have acquired the entire route (including all signals and switches) leading up
                                // to this signal (in previous iterations).
                                // However, we haven't yet determined whether we're going to stop at it, or pass through it.
                                acquired_signal = *block;

                                let query_result = query_starting_signal(
                                    block.variant(),
                                    ext.as_ref(),
                                    cart_name,
                                )?;
                                if let Some(variant) = query_result.adjust_variant(block.variant())
                                {
                                    let variant = variant;
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
                                Ok(query_result.to_parse_outcome())
                            }
                        }
                    } else if block.equals_ignore_variant(cart_config.starting_signal)
                        && state.signal_reversed_rotation_ok(block.variant())
                    {
                        // Starting signal, approaching it from the back

                        match starting_signal_acquire_back(signal_coord, block, *block) {
                            signals::SignalLockOutcome::InvalidSignal => {
                                Ok(SignalParseOutcome::Deny)
                            }
                            signals::SignalLockOutcome::Acquired => {
                                // We didn't conflict with anyone else for the signal, and we now hold the lock for it.
                                acquired_signal_was_reverse_starting = true;
                                acquired_signal = *block;
                                transaction.insert(signal_coord, *block, rollback_block, *block);
                                // No indication - we don't want to wipe out pending left/right divergence decisions.
                                Ok(SignalParseOutcome::NoIndication)
                            }
                            signals::SignalLockOutcome::Contended => {
                                // The signal is contended, and we cannot safely make this move. It's important that we don't simply
                                // roll up to it and stop, because chances are the signal is contended because a cart is sitting at it,
                                // waiting for its clearance to leave. In that case, if we do approach it and stop short of it,
                                // we'll just produce a deadlock on the tracks.
                                Ok(SignalParseOutcome::Deny)
                            }
                        }
                    } else if block.equals_ignore_variant(cart_config.automatic_signal) {
                        if state.signal_rotation_ok(block.variant()) {
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
                            // Automatic signals facing the wrong way can break the invariants of interlockings
                            // We cannot admit moves through them
                            Ok(SignalParseOutcome::Deny)
                        }
                    } else if let Some(speed) = cart_config.parse_speedpost(*block) {
                        speed_post = Some(speed);
                        Ok(SignalParseOutcome::NoIndication)
                    } else {
                        Ok(SignalParseOutcome::NoIndication)
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
                SignalParseOutcome::NoIndication => {
                    // do nothing, keep scanning
                }
                SignalParseOutcome::AutomaticSignal => {
                    steps.push(InterlockingStep {
                        scan_state: state.clone(),
                        acquired_signal,
                        acquired_signal_was_reverse_starting,
                        clear_switch: None,
                        speed_post: None,
                    });
                    transaction.commit();
                    return Ok(Some(InterlockingRoute {
                        steps,
                        resume: None,
                    }));
                }
                SignalParseOutcome::StartingSignalApproachThenStop => {
                    // We do NOT push the step that gets us past the signal,
                    // we just commit the transaction right away
                    transaction.commit();
                    return Ok(Some(InterlockingRoute {
                        steps,
                        resume: Some(InterlockingResumeState {
                            starting_signal: Some((signal_coord, acquired_signal)),
                        }),
                    }));
                }
            }
        }
        // then set the switch accordingly.
        let mut clear_switch = None;
        if let Some(switch_len) = state.get_switch_length() {
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
                            let will_diverge;
                            let defer_switch;
                            if state.can_diverge_left() && left_pending {
                                left_pending = false;
                                will_diverge = true;
                                // Facing the points, defer the switch until we clear its full length
                                defer_switch = true;
                            } else if state.can_diverge_right() && right_pending {
                                right_pending = false;
                                will_diverge = true;
                                // Facing the points, defer the switch until we clear its full length
                                defer_switch = true;
                            } else if state.can_converge() && state.is_diverging {
                                // Trailing the points, once we get through it we can clear it immediately without deferring
                                will_diverge = true;
                                defer_switch = false;
                            } else if state.can_converge() && !state.is_diverging {
                                // Trailing the points, once we get through we can clear it immediately without deferring
                                will_diverge = false;
                                defer_switch = false;
                            } else if state.can_diverge_left() || state.can_diverge_right() {
                                // Facing the points, proceeding straight, still need to defer the switch to avoid fouling
                                defer_switch = true;
                                will_diverge = false;
                            } else {
                                // Just a random switch block without a switch
                                defer_switch = false;
                                will_diverge = false;
                            }
                            let new_block = if will_diverge {
                                cart_config.switch_diverging.with_variant_of(*switch_block)
                            } else {
                                cart_config.switch_straight.with_variant_of(*switch_block)
                            };
                            transaction.insert(switch_coord, new_block, *switch_block, new_block);
                            *switch_block = new_block;
                            if defer_switch {
                                clear_switch =
                                    Some((switch_coord, *switch_block, switch_len.get()));
                            } else {
                                tracing::debug!("Clearing switch at {:?}", switch_coord);
                                clear_switch = Some((switch_coord, *switch_block, 0));
                            }

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
            acquired_signal,
            acquired_signal_was_reverse_starting,
            clear_switch,
            speed_post,
        });
        let new_state =
            state.advance::<false>(|coord| handler_context.game_map().get_block(coord).into())?;
        match new_state {
            super::tracks::ScanOutcome::Success(new_state) => {
                state = new_state;
            }
            super::tracks::ScanOutcome::CannotAdvance => {
                transaction.commit();
                return Ok(Some(InterlockingRoute {
                    steps,
                    resume: None,
                }));
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
pub(crate) struct InterlockingStep {
    pub(crate) scan_state: ScanState,
    pub(crate) acquired_signal: BlockId,
    pub(crate) clear_switch: Option<(BlockCoordinate, BlockId, u8)>,
    pub(crate) speed_post: Option<f32>,
    pub(crate) acquired_signal_was_reverse_starting: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct InterlockingRoute {
    pub(crate) steps: Vec<InterlockingStep>,
    pub(crate) resume: Option<InterlockingResumeState>,
}

/// Potentially returned as part of a successful interlocking route; should be passed
/// back into the next attempt to pass through the interlocking
#[derive(Debug, Clone)]
pub(crate) struct InterlockingResumeState {
    starting_signal: Option<(BlockCoordinate, BlockId)>,
}
