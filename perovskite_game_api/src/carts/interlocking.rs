// For now, hide unused warnings since they're distracting
#![allow(dead_code)]

use anyhow::{Context, Result};
use perovskite_core::{block_id::BlockId, coordinates::BlockCoordinate};
use perovskite_server::game_state::{
    blocks::{CompassDirection, ExtendedData},
    event::HandlerContext,
    game_map::ServerGameMap,
};
use rand::Rng;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::collections::HashMap;

use super::network::{AdjacencyHit, AdjacencyHitKind};

use crate::carts::signals::{
    self, get_infra_block_name, starting_signal_acquire_back, starting_signal_depart_forward,
    starting_signal_preacquire_front, SignalConfig, SIGNAL_BLOCK_CONNECTIVITY,
};
use crate::carts::{network::CachedHit, station::StationRoute};
use crate::circuits::{BusMessage, PinState};

use super::{
    signals::{
        automatic_signal_acquire, interlocking_signal_preacquire, query_interlocking_signal,
        query_starting_signal, SignalInstruction,
    },
    tracks::ScanState,
    CartsGameBuilderExtension,
};

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub(super) enum SwitchState {
    Straight,
    Diverging,
}
/// Holder of a pending set of signal and switch changes that will set up a path through an interlocking.
///
/// While this object is alove, the signals in question are locked and cannot be used by other carts.
///
/// Dropping this will roll back the state, allowing a different cart to acquire the interlocking.
/// Calling commit will finalize the changes.
///
/// This object is intended to be short-lived; don't store it for long.
#[derive(Clone)]
#[must_use = "Caller should explicitly commit or roll back as soon as possible."]
pub(crate) struct SignalTransaction<'a> {
    // The coordinate, the expected block ID, the block ID to roll back to, and the block ID to commit to
    map_changes: Vec<(BlockCoordinate, BlockId, BlockId, BlockId)>,
    handler_context: &'a HandlerContext<'a>,
}
impl<'a> SignalTransaction<'a> {
    /// Commits the intended changes to the interlocking
    ///
    /// To be used when we've successfully found a path through the interlocking
    pub(crate) fn commit(mut self) {
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

    // Made available to show intention, not necessarily used in the current impl
    #[allow(unused)]
    pub(crate) fn rollback(self) {
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

impl<'a> std::fmt::Debug for SignalTransaction<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SignalTransaction")
            .field("map_changes", &self.map_changes)
            .finish()
    }
}

/// Attempts to find a path through an interlocking.
///
/// Args:
///   * initial_state: The initial state of the cart. This should be at the same block where
///         the initial interlocking signal was encountered
///   * max_scan_distance: The maximum distance (in tiles) to scan.
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
    cart_id: u32,
    max_scan_distance: usize,
    cart_config: CartsGameBuilderExtension,
    resume: Option<InterlockingResumeState>,
    last_speed_post: f32,
    starting_from_standstill: bool,
    buffer_time_estimate: f32,
    signal_coord: BlockCoordinate,
) -> Result<Option<InterlockingRoute>> {
    if !handler_context.is_shutting_down() {
        let resolution = single_pathfind_attempt(
            &handler_context,
            cart_name,
            (handler_context.startup_counter(), cart_id),
            initial_state.clone(),
            max_scan_distance,
            &cart_config,
            resume.clone(),
            buffer_time_estimate,
        )?;
        if let Some(resolution) = resolution {
            let pending_route = resolution.commit();
            send_signal_bus_message(
                &handler_context,
                signal_coord,
                cart_name,
                cart_id,
                "success",
                buffer_time_estimate,
            )?;
            if starting_from_standstill {
                let exit_speed = pending_route
                    .steps
                    .iter()
                    .rev()
                    .find_map(|step| step.speed_post)
                    .unwrap_or(last_speed_post);
                let delay_time = 0.5 * exit_speed as f64 / super::MAX_ACCEL;
                tracing::info!(
                    "Sleeping for {} seconds since starting from standstill",
                    delay_time
                );
                tokio::time::sleep(std::time::Duration::from_secs_f64(delay_time)).await;
            }
            return Ok(Some(pending_route));
        } else {
            tracing::debug!("No path found, trying again");
            // Randomized backoff, 500-1000 msec
            //
            // A "smarter" deadlock resolution strategy like wound-wait or wait-die would
            // require too many bits of block state. We could put that state into extended data,
            // but that makes extended data error handling more tricky.
            let backoff = rand::thread_rng().gen_range(500..1000);
            tokio::time::sleep(std::time::Duration::from_millis(backoff)).await;

            send_signal_bus_message(
                &handler_context,
                signal_coord,
                cart_name,
                cart_id,
                "failed",
                buffer_time_estimate,
            )?;
            return Ok(None);
        }
    }
    Ok(None)
}

fn send_signal_bus_message(
    ctx: &HandlerContext,
    signal_coord: BlockCoordinate,
    cart_name: &str,
    cart_id: u32,
    outcome: &str,
    buffer_time_estimate: f32,
) -> Result<()> {
    let mut data = HashMap::new();

    let (block, nickname) =
        ctx.game_map()
            .get_block_with_extended_data(signal_coord, |_, ext| match ext.custom_data {
                Some(ref custom_data) => match custom_data.downcast_ref::<SignalConfig>() {
                    Some(config) => Ok(Some(config.signal_nickname.clone())),
                    _ => {
                        tracing::warn!("expected SignalConfig, got a different type");
                        Ok(None)
                    }
                },
                None => Ok(None),
            })?;

    data.insert("signal_coord".to_string(), signal_coord.to_string());
    data.insert("signal_nickname".to_string(), nickname.unwrap_or_default());
    data.insert("cart_name".to_string(), cart_name.to_string());
    data.insert("cart_id".to_string(), cart_id.to_string());
    data.insert("outcome".to_string(), outcome.to_string());
    data.insert(
        "at_signal_now".to_string(),
        (buffer_time_estimate < 0.01).to_string(),
    );
    let bus_message = BusMessage {
        sender: signal_coord,
        data,
    };
    let cctx = crate::circuits::events::make_root_context(ctx);
    for connectivity in SIGNAL_BLOCK_CONNECTIVITY {
        if let Some(target) = connectivity.eval(signal_coord, block.variant()) {
            crate::circuits::events::transmit_bus_message(
                &cctx,
                target,
                signal_coord,
                PinState::Low,
                bus_message.clone(),
            )?
        }
    }
    Ok(())
}

pub(crate) fn single_pathfind_attempt<'a>(
    ctx: &'a HandlerContext<'a>,
    cart_name: &str,
    cart_id: (u64, u32),
    initial_state: ScanState,
    max_scan_distance: usize,
    cart_config: &CartsGameBuilderExtension,
    resume: Option<InterlockingResumeState>,
    _buffer_time_estimate: f32,
) -> Result<Option<PendingRoute<'a>>> {
    let mut steps = vec![];
    let mut state = initial_state;
    let mut transaction = SignalTransaction::new(ctx);

    let mut station_route: Option<StationRoute> = None;

    let mut left_pending = false;
    let mut right_pending = false;

    // Scan forward through tracks until we either:
    // 1. Reach the end of the track (which means that we can finish the path)
    // 2. Reach a correctly-facing automatic signal (which means that we're returning to simple non-interlocking tracks
    //      in the correct direction)
    // 3. Reach a starting signal
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
            let query_result =
                ctx.game_map()
                    .mutate_block_atomically(signal_coord, |block, ext| {
                        // if we have to give up and roll back, we'll restore to this
                        let rollback_block = *block;

                        let direction = CompassDirection::from_rotation_variant(block.variant());
                        if let Some(route) = station_route.as_mut() {
                            let next_hop = route.next_hop();
                            if next_hop.is_some_and(|hop| {
                                hop.track_coord == track_coord
                                    && hop.travel_direction == Some(direction)
                            }) {
                                route.drop_next_hop();
                            }
                        }

                        if block.equals_ignore_variant(cart_config.interlocking_signal)
                            && state.signal_rotation_ok(block.variant())
                        {
                            match interlocking_signal_preacquire(signal_coord, block) {
                                signals::SignalLockOutcome::Contended
                                | signals::SignalLockOutcome::InvalidSignal => {
                                    Ok(SignalInstruction::Deny)
                                }
                                signals::SignalLockOutcome::Acquired => {
                                    // We didn't conflict with anyone else for the signal, and we now hold the lock for it.
                                    let query_result = query_interlocking_signal(
                                        ctx,
                                        ext.as_ref(),
                                        cart_name,
                                        station_route.as_ref(),
                                        cart_id,
                                    )?;
                                    if let Some(route) = query_result.station_route {
                                        station_route = Some(route);
                                    }
                                    if let Some(variant) =
                                        query_result.instruction.adjust_variant(block.variant())
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
                                    } else {
                                        *block = rollback_block;
                                    }
                                    Ok(query_result.instruction.to_parse_outcome())
                                }
                            }
                        } else if resume.as_ref().is_some_and(|resume| {
                            resume
                                .starting_signal
                                .is_some_and(|(coord, _)| coord == signal_coord)
                        }) {
                            // This is the starting signal that this cart stopped in front of.
                            match starting_signal_depart_forward(signal_coord, block) {
                                signals::SignalLockOutcome::InvalidSignal
                                | signals::SignalLockOutcome::Contended => {
                                    Ok(SignalInstruction::Deny)
                                }
                                signals::SignalLockOutcome::Acquired => {
                                    tracing::debug!(
                                        "Starting signal acquired at {:?}",
                                        signal_coord
                                    );
                                    // We didn't conflict with anyone else for the signal, and we now hold the lock for it.
                                    // We use the interlocking signal query here since we know we're allowed to pass it now
                                    let query_result = query_interlocking_signal(
                                        ctx,
                                        ext.as_ref(),
                                        cart_name,
                                        station_route.as_ref(),
                                        cart_id,
                                    )?;

                                    if let Some(route) = query_result.station_route {
                                        station_route = Some(route);
                                    }
                                    if let Some(variant) =
                                        query_result.instruction.adjust_variant(block.variant())
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
                                    Ok(query_result.instruction.to_parse_outcome())
                                }
                            }
                        } else if block.equals_ignore_variant(cart_config.starting_signal)
                            && state.signal_rotation_ok(block.variant())
                        {
                            match starting_signal_preacquire_front(signal_coord, block) {
                                signals::SignalLockOutcome::InvalidSignal
                                | signals::SignalLockOutcome::Contended => {
                                    Ok(SignalInstruction::Deny)
                                }
                                signals::SignalLockOutcome::Acquired => {
                                    tracing::info!(
                                        "Starting signal preacquired at {:?}",
                                        signal_coord
                                    );
                                    // We didn't conflict with anyone else for the signal, and we now hold the lock for it.
                                    // Furthermore, we have acquired the entire route (including all signals and switches) leading up
                                    // to this signal (in previous iterations).
                                    // However, we haven't yet determined whether we're going to stop at it, or pass through it.
                                    acquired_signal = *block;

                                    let query_result = query_starting_signal(
                                        block.variant(),
                                        ctx,
                                        ext.as_ref(),
                                        cart_name,
                                        station_route.as_ref(),
                                        cart_id,
                                    )?;
                                    if let Some(variant) =
                                        query_result.instruction.adjust_variant(block.variant())
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
                                    Ok(query_result.instruction.to_parse_outcome())
                                }
                            }
                        } else if block.equals_ignore_variant(cart_config.starting_signal)
                            && state.signal_reversed_rotation_ok(block.variant())
                        {
                            // Starting signal, approaching it from the back

                            match starting_signal_acquire_back(signal_coord, block) {
                                signals::SignalLockOutcome::InvalidSignal => {
                                    Ok(SignalInstruction::Deny)
                                }
                                signals::SignalLockOutcome::Acquired => {
                                    // We didn't conflict with anyone else for the signal, and we now hold the lock for it.
                                    acquired_signal_was_reverse_starting = true;
                                    acquired_signal = *block;
                                    transaction.insert(
                                        signal_coord,
                                        *block,
                                        rollback_block,
                                        *block,
                                    );
                                    // No indication - we don't want to wipe out pending left/right divergence decisions.
                                    Ok(SignalInstruction::NoIndication)
                                }
                                signals::SignalLockOutcome::Contended => {
                                    // The signal is contended, and we cannot safely make this move. It's important that we don't simply
                                    // roll up to it and stop, because chances are the signal is contended because a cart is sitting at it,
                                    // waiting for its clearance to leave. In that case, if we do approach it and stop short of it,
                                    // we'll just produce a deadlock on the tracks.
                                    Ok(SignalInstruction::Deny)
                                }
                            }
                        } else if block.equals_ignore_variant(cart_config.automatic_signal) {
                            if state.signal_rotation_ok(block.variant()) {
                                match automatic_signal_acquire(signal_coord, block, *block) {
                                    signals::SignalLockOutcome::InvalidSignal => {
                                        Ok(SignalInstruction::Deny)
                                    }
                                    signals::SignalLockOutcome::Acquired => {
                                        // We didn't conflict with anyone else for the signal, and we now hold the lock for it.
                                        acquired_signal = *block;
                                        Ok(SignalInstruction::AutomaticSignal)
                                    }
                                    signals::SignalLockOutcome::Contended => {
                                        // We can't get the lock for the signal so we can't leave the interlocking.
                                        Ok(SignalInstruction::Deny)
                                    }
                                }
                            } else {
                                // Automatic signals facing the wrong way can break the invariants of interlockings
                                // We cannot admit moves through them
                                Ok(SignalInstruction::Deny)
                            }
                        } else if let Some(speed) = cart_config.parse_speedpost(*block) {
                            speed_post = Some(speed);
                            Ok(SignalInstruction::NoIndication)
                        } else {
                            Ok(SignalInstruction::NoIndication)
                        }
                    })?;

            match query_result {
                SignalInstruction::Straight => {
                    left_pending = false;
                    right_pending = false;
                }
                SignalInstruction::DivergingLeft => {
                    left_pending = true;
                    right_pending = false;
                }
                SignalInstruction::DivergingRight => {
                    left_pending = false;
                    right_pending = true;
                }
                SignalInstruction::Deny => {
                    return Ok(None);
                }
                SignalInstruction::NoIndication => {
                    // do nothing, keep scanning
                }
                SignalInstruction::AutomaticSignal => {
                    steps.push(InterlockingStep {
                        scan_state: state.clone(),
                        acquired_signal,
                        acquired_signal_was_reverse_starting,
                        clear_switch: None,
                        speed_post: None,
                    });
                    return Ok(Some(PendingRoute {
                        transaction,
                        route: InterlockingRoute {
                            steps,
                            resume: None,
                        },
                    }));
                }
                SignalInstruction::StartingSignalApproachThenStop => {
                    // We do NOT push the step that gets us past the signal,
                    // we just commit the transaction right away
                    return Ok(Some(PendingRoute {
                        transaction,
                        route: InterlockingRoute {
                            steps,
                            resume: Some(InterlockingResumeState {
                                starting_signal: Some((signal_coord, acquired_signal)),
                            }),
                        },
                    }));
                }
            }
        }
        // then set the switch accordingly.
        let mut clear_switch = None;
        if let Some(switch_len) = state.get_switch_length() {
            let switch_coord = track_coord.try_delta(0, -1, 0);

            if let Some(switch_coord) = switch_coord {
                let switch_decision = ctx.game_map().mutate_block_atomically(
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
        let new_state = state.advance::<false>(ctx.game_map(), cart_config)?;
        match new_state {
            super::tracks::ScanOutcome::Success(new_state) => {
                state = new_state;
            }
            super::tracks::ScanOutcome::NotOnTrack => {
                return Ok(None);
            }
            super::tracks::ScanOutcome::Deferral(_) => {
                panic!("Got a deferral from track scan, but we're not in a deferrable context and the block getter shouldn't defer");
            }
            // TODO: make explicit the cases that hit this
            _e => {
                return Ok(Some(PendingRoute {
                    transaction,
                    route: InterlockingRoute {
                        steps,
                        resume: None,
                    },
                }));
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

#[derive(Debug)]
pub(crate) struct PendingRoute<'a> {
    transaction: SignalTransaction<'a>,
    route: InterlockingRoute,
}
impl<'a> PendingRoute<'a> {
    pub(crate) fn commit(self) -> InterlockingRoute {
        self.transaction.commit();
        self.route
    }
    #[cfg(test)]
    pub(crate) fn inner(&self) -> &InterlockingRoute {
        &self.route
    }

    #[cfg(test)]
    pub(crate) fn pending_change_count(&self) -> usize {
        self.transaction.map_changes.len()
    }

    pub(crate) fn rollback(self) {
        self.transaction.rollback();
    }
}

#[derive(Debug, Clone)]
pub(crate) struct InterlockingRoute {
    pub(crate) steps: Vec<InterlockingStep>,
    pub(crate) resume: Option<InterlockingResumeState>,
}

/// Potentially returned as part of a successful interlocking route; should be passed
/// back into the next attempt to pass through the interlocking. Used for starting signals,
/// where a cart stops within the bounds of the interlocking.
#[derive(Debug, Clone)]
pub(crate) struct InterlockingResumeState {
    starting_signal: Option<(BlockCoordinate, BlockId)>,
}

/// One possible path through an interlocking discovered by passive topological scanning.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct InterlockingPathResult {
    /// Where this path terminates (automatic-signal exit, dead end, or step limit).
    pub(crate) endpoint: AdjacencyHit,
    /// Starting signals and waypoints encountered along the path, in forward order.
    pub(crate) via: SmallVec<[AdjacencyHit; 4]>,
}

#[derive(Clone, prost::Message)]
pub(crate) struct CachedPathResult {
    #[prost(message, tag = "1")]
    pub(crate) endpoint: Option<CachedHit>,
    #[prost(message, repeated, tag = "2")]
    pub(crate) via: Vec<CachedHit>,
}
impl From<InterlockingPathResult> for CachedPathResult {
    fn from(path_result: InterlockingPathResult) -> CachedPathResult {
        CachedPathResult {
            endpoint: Some(path_result.endpoint.into()),
            via: path_result.via.into_iter().map(|hit| hit.into()).collect(),
        }
    }
}

impl TryFrom<CachedPathResult> for InterlockingPathResult {
    type Error = anyhow::Error;
    fn try_from(value: CachedPathResult) -> Result<Self, Self::Error> {
        Ok(InterlockingPathResult {
            endpoint: value.endpoint.context("Missing endpoint")?.try_into()?,
            via: value
                .via
                .into_iter()
                .map(|hit| hit.try_into())
                .collect::<Result<SmallVec<[AdjacencyHit; 4]>, _>>()?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PathDecision {
    Forward,
    Left,
    Right,
}

/// One interlocking signal encountered on a scan path, with its routing decision.
/// `waypoints_start` is the index into the associated `InterlockingPathResult::via`
/// slice from which this signal's downstream waypoints begin.
#[derive(Debug, Clone)]
pub(crate) struct SignalStep {
    pub(crate) signal_coord: BlockCoordinate,
    pub(crate) facing: CompassDirection,
    pub(crate) decision: PathDecision,
    pub(crate) waypoints_start: usize,
}

pub(crate) type RoutingPath = Vec<SignalStep>;

/// Cached routing-table entry: waypoints + terminal destination for one path direction.
/// The last item is always the terminal (EndOfInterlockingSignal, EndOfTrack, etc.).
#[derive(Clone, prost::Message)]
pub(crate) struct RoutingTablePath {
    #[prost(message, repeated, tag = "1")]
    pub(crate) items: Vec<CachedHit>,
}

impl From<AdjacencyHit> for CachedPathResult {
    fn from(hit: AdjacencyHit) -> CachedPathResult {
        CachedPathResult {
            endpoint: Some(hit.into()),
            via: Vec::new(),
        }
    }
}

struct ScanItem {
    state: ScanState,
    left_pending: bool,
    right_pending: bool,
    /// (block_coord, is_reversed, is_diverging) visited to detect loops.
    visited: FxHashSet<(BlockCoordinate, bool, bool)>,
    waypoints: SmallVec<[AdjacencyHit; 4]>,
    steps: usize,
    /// Signals encountered so far: (signal_coord_y2, facing, decision, waypoints_start_at_encounter).
    signals: Vec<(BlockCoordinate, CompassDirection, PathDecision, usize)>,
}

/// Passively scan all topologically distinct routes through an interlocking.
///
/// No signals or switches are acquired or mutated. Every time a pending diverge
/// direction meets a real switch, both the diverging and the straight-through
/// branches are explored independently. Returns one `InterlockingPathResult`
/// per distinct (endpoint, intermediate-waypoints) pair.
///
/// `initial_state` should be positioned at the first interlocking signal tile
/// (the same coordinate that would be passed to `single_pathfind_attempt`).
pub(crate) fn scan_interlocking_routes(
    initial_state: ScanState,
    step_limit: usize,
    game_map: &ServerGameMap,
    cart_config: &CartsGameBuilderExtension,
) -> Result<FxHashMap<InterlockingPathResult, Vec<RoutingPath>>> {
    let mut results: FxHashMap<InterlockingPathResult, Vec<RoutingPath>> = FxHashMap::default();
    // Both pending start true: the initial interlocking signal can route either way.
    let mut stack: Vec<ScanItem> = vec![ScanItem {
        state: initial_state,
        left_pending: true,
        right_pending: true,
        visited: FxHashSet::default(),
        waypoints: SmallVec::new(),
        steps: 0,
        signals: Vec::new(),
    }];

    'pop: while let Some(mut item) = stack.pop() {
        loop {
            if item.steps >= step_limit {
                let path_result = InterlockingPathResult {
                    endpoint: AdjacencyHit {
                        kind: AdjacencyHitKind::StepLimitExhausted,
                        track_coord: item.state.block_coord,
                        travel_direction: None,
                        step_count: item.steps,
                        name: None,
                    },
                    via: item.waypoints,
                };
                let routing_path: RoutingPath = item
                    .signals
                    .into_iter()
                    .map(|(sc, facing, decision, ws)| SignalStep {
                        signal_coord: sc,
                        facing,
                        decision,
                        waypoints_start: ws,
                    })
                    .collect();
                results.entry(path_result).or_default().push(routing_path);
                continue 'pop;
            }

            let track_coord = item.state.block_coord;
            let visit_key = (track_coord, item.state.is_reversed, item.state.is_diverging);
            if item.visited.contains(&visit_key) {
                continue 'pop;
            }
            item.visited.insert(visit_key);

            // Check the signal/waypoint slot at Y+2 above the current track tile.
            if let Some(signal_coord) = track_coord.try_delta(0, 2, 0) {
                let (block, name) = get_infra_block_name(signal_coord, game_map, cart_config)?;
                if block != BlockId::AIR {
                    let variant = block.variant();
                    if block.equals_ignore_variant(cart_config.interlocking_signal) {
                        if item.state.signal_rotation_ok(variant) {
                            // Every correctly-facing interlocking signal opens both diverge directions.
                            // Record it as a Forward signal step (decision may be updated to Left/Right at fork time).
                            let facing_dir = CompassDirection::from_rotation_variant(variant);
                            item.signals.push((
                                signal_coord,
                                facing_dir,
                                PathDecision::Forward,
                                item.waypoints.len(),
                            ));
                            item.left_pending = true;
                            item.right_pending = true;
                        }
                        // Backwards interlocking signal: ignored (NoIndication), like single_pathfind_attempt.
                        // A backwards interlocking signal in the body of an interlocking simply means we
                        // passed it going the other way; it does not invalidate the path.
                    } else if block.equals_ignore_variant(cart_config.automatic_signal) {
                        if item.state.signal_rotation_ok(variant) {
                            // Correctly-facing automatic signal marks the interlocking exit.
                            let facing_dir = CompassDirection::from_rotation_variant(variant);
                            let path_result = InterlockingPathResult {
                                endpoint: AdjacencyHit {
                                    kind: AdjacencyHitKind::EndOfInterlockingSignal,
                                    track_coord,
                                    travel_direction: Some(facing_dir),
                                    step_count: item.steps,
                                    name,
                                },
                                via: item.waypoints,
                            };
                            let routing_path: RoutingPath = item
                                .signals
                                .into_iter()
                                .map(|(sc, f, decision, ws)| SignalStep {
                                    signal_coord: sc,
                                    facing: f,
                                    decision,
                                    waypoints_start: ws,
                                })
                                .collect();
                            results.entry(path_result).or_default().push(routing_path);
                        }
                        // Backwards automatic signal: also an invalid path per interlocking rules.
                        continue 'pop;
                    } else if block.equals_ignore_variant(cart_config.starting_signal)
                        && item.state.signal_rotation_ok(variant)
                    {
                        // Correctly-facing starting signal: record as intermediate and reset diverge state.
                        let facing_dir = CompassDirection::from_rotation_variant(variant);
                        item.waypoints.push(AdjacencyHit {
                            kind: AdjacencyHitKind::StartingSignal,
                            track_coord,
                            travel_direction: Some(facing_dir),
                            step_count: item.steps,
                            name,
                        });
                        item.left_pending = true;
                        item.right_pending = true;
                    } else if block.equals_ignore_variant(cart_config.waypoint)
                        && item.state.signal_rotation_ok(variant)
                    {
                        let facing_dir = CompassDirection::from_rotation_variant(variant);
                        item.waypoints.push(AdjacencyHit {
                            kind: AdjacencyHitKind::WaypointBlock,
                            track_coord,
                            travel_direction: Some(facing_dir),
                            step_count: item.steps,
                            name,
                        });
                    }
                    // Speedposts, backwards starting signals, wrong-way waypoints: ignored.
                }
            }

            // Handle switch at Y-1 below the current track tile.
            if item.state.get_switch_length().is_some() {
                if let Some(switch_coord) = track_coord.try_delta(0, -1, 0) {
                    let switch_block = game_map.get_block(switch_coord)?;
                    let has_switch = switch_block.equals_ignore_variant(cart_config.switch_unset)
                        || switch_block.equals_ignore_variant(cart_config.switch_straight)
                        || switch_block.equals_ignore_variant(cart_config.switch_diverging);

                    if has_switch {
                        // Branch for each applicable pending direction.
                        // Guard !is_diverging: if we're already on the diverging path through
                        // this switch tile, do not re-branch; just advance along that path.
                        if item.state.can_diverge_left()
                            && item.left_pending
                            && !item.state.is_diverging
                        {
                            let mut div_state = item.state;
                            div_state.is_diverging = true;
                            // Clone signals and update the last entry to Left.
                            let mut new_signals = item.signals.clone();
                            if let Some(last) = new_signals.last_mut() {
                                last.2 = PathDecision::Left;
                            }
                            stack.push(ScanItem {
                                state: div_state,
                                left_pending: false,
                                right_pending: false,
                                visited: item.visited.clone(),
                                waypoints: item.waypoints.clone(),
                                steps: item.steps,
                                signals: new_signals,
                            });
                            item.left_pending = false;
                        }
                        if item.state.can_diverge_right()
                            && item.right_pending
                            && !item.state.is_diverging
                        {
                            let mut div_state = item.state;
                            div_state.is_diverging = true;
                            // Clone signals and update the last entry to Right.
                            let mut new_signals = item.signals.clone();
                            if let Some(last) = new_signals.last_mut() {
                                last.2 = PathDecision::Right;
                            }
                            stack.push(ScanItem {
                                state: div_state,
                                left_pending: false,
                                right_pending: false,
                                visited: item.visited.clone(),
                                waypoints: item.waypoints.clone(),
                                steps: item.steps,
                                signals: new_signals,
                            });
                            item.right_pending = false;
                        }
                        // The current item continues straight through.
                    }
                }
            }

            // Advance to the next track tile.
            match item.state.advance::<false>(game_map, cart_config)? {
                super::tracks::ScanOutcome::Success(new_state) => {
                    item.state = new_state;
                    item.steps += 1;
                }
                _ => {
                    let path_result = InterlockingPathResult {
                        endpoint: AdjacencyHit {
                            kind: AdjacencyHitKind::EndOfTrack,
                            track_coord,
                            travel_direction: None,
                            step_count: item.steps,
                            name: None,
                        },
                        via: item.waypoints,
                    };
                    let routing_path: RoutingPath = item
                        .signals
                        .into_iter()
                        .map(|(sc, facing, decision, ws)| SignalStep {
                            signal_coord: sc,
                            facing,
                            decision,
                            waypoints_start: ws,
                        })
                        .collect();
                    results.entry(path_result).or_default().push(routing_path);
                    continue 'pop;
                }
            }
        }
    }

    Ok(results)
}

/// Applies the routing tables discovered by `scan_interlocking_routes` to the signal blocks in the map.
///
/// For each signal encountered during the scan, this writes `left_paths`, `right_paths`, and `forward_paths`
/// into its `SignalConfig` extended data, preserving all other fields.
pub(crate) fn apply_interlocking_routes_to_signals(
    routes: &FxHashMap<InterlockingPathResult, Vec<RoutingPath>>,
    game_map: &ServerGameMap,
) -> Result<()> {
    // Build routing tables keyed by signal_coord: (left_paths, right_paths, forward_paths)
    let mut tables: FxHashMap<
        BlockCoordinate,
        (
            Vec<RoutingTablePath>,
            Vec<RoutingTablePath>,
            Vec<RoutingTablePath>,
        ),
    > = FxHashMap::default();

    for (path_result, routing_paths) in routes {
        for routing_path in routing_paths {
            for step in routing_path {
                let items: Vec<CachedHit> = path_result.via[step.waypoints_start..]
                    .iter()
                    .map(|h| CachedHit::from(h.clone()))
                    .chain(std::iter::once(CachedHit::from(
                        path_result.endpoint.clone(),
                    )))
                    .collect();
                let table_path = RoutingTablePath { items };
                let entry = tables.entry(step.signal_coord).or_default();
                match step.decision {
                    PathDecision::Left => entry.0.push(table_path),
                    PathDecision::Right => entry.1.push(table_path),
                    PathDecision::Forward => entry.2.push(table_path),
                }
            }
        }
    }

    fn dedup_paths(paths: &mut Vec<RoutingTablePath>) {
        use prost::Message;
        let mut seen = FxHashSet::<Vec<u8>>::default();
        paths.retain(|p| seen.insert(p.encode_to_vec()));
    }

    for (_, (ref mut l, ref mut r, ref mut f)) in &mut tables {
        dedup_paths(l);
        dedup_paths(r);
        dedup_paths(f);
    }

    for (signal_coord, (left_paths, right_paths, forward_paths)) in tables {
        game_map.mutate_block_atomically(signal_coord, |_block, ext| {
            let ext_inner = ext.get_or_insert_with(ExtendedData::default);
            match ext_inner.custom_data.as_mut() {
                Some(data) => match data.downcast_mut::<crate::carts::signals::SignalConfig>() {
                    Some(sc) => {
                        sc.left_paths = left_paths;
                        sc.right_paths = right_paths;
                        sc.forward_paths = forward_paths;
                    }
                    _ => {
                        tracing::warn!(
                            "expected SignalConfig at {:?}, got different type",
                            signal_coord
                        );
                    }
                },
                None => {
                    let _ = ext_inner.custom_data.insert(Box::new(
                        crate::carts::signals::SignalConfig {
                            left_paths,
                            right_paths,
                            forward_paths,
                            ..Default::default()
                        },
                    ));
                }
            }
            ext.set_dirty();
            Ok(())
        })?;
    }

    Ok(())
}
