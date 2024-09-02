use std::ops::{Add, Range};
use std::sync::atomic::AtomicU8;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use arc_swap::ArcSwap;
use cgmath::{InnerSpace, Vector3, Zero};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Host, OutputCallbackInfo};
use parking_lot::Mutex;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use seqlock::SeqLock;
use tokio_util::sync::{CancellationToken, DropGuard};
use tracy_client::{plot, span};

use crate::audio::testdata::FOOTSTEP_WAV_BYTES;
use crate::game_state::entities::{ElapsedOrOverflow, EntityMove};
use crate::game_state::settings::GameSettings;
use crate::game_state::timekeeper::Timekeeper;

// Public for testing
pub struct EngineHandle {
    control: Arc<SharedControl>,
    drop_guard: DropGuard,
    simple_sounds_bitmap: Mutex<AllocatorState>,
}

/// An opaque token used to modify/remove a sound in a slot.
/// Resilient to evictions, since it keeps a sequence number.
///
/// Note: PartialOrd/Ord derived implementation is only for the benefit of
/// ord-based datastructures (e.g. BTreeMap). No semantic meaning should
/// be ascribed to the result of these comparisons.
#[derive(PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct SimpleSoundToken(u64);

struct AllocatorState {
    simple_sound_tokens: [u64; NUM_SIMPLE_SOUND_SLOTS],
    simple_sounds_next_sequence: u64,
}

impl AllocatorState {
    fn new() -> AllocatorState {
        AllocatorState {
            simple_sound_tokens: [0; NUM_SIMPLE_SOUND_SLOTS],
            // This must start at a nonzero value, since 0 is a sentinel
            simple_sounds_next_sequence: NUM_SIMPLE_SOUND_SLOTS as u64,
        }
    }
}

impl EngineHandle {
    pub(crate) fn alloc_simple_sound(
        &self,
        tick_now: u64,
        player_position: Vector3<f64>,
        sound: SimpleSoundControlBlock,
    ) -> Option<SimpleSoundToken> {
        assert_ne!(sound.flags & SOUND_PRESENT, 0);
        let mut alloc_lock = self.simple_sounds_bitmap.lock();

        let seqnum = alloc_lock.simple_sounds_next_sequence;
        alloc_lock.simple_sounds_next_sequence += NUM_SIMPLE_SOUND_SLOTS as u64;

        if let Some(index) = alloc_lock.simple_sound_tokens.iter().position(|x| *x == 0) {
            let token = seqnum + (index as u64);
            alloc_lock.simple_sound_tokens[index] = token;
            {
                let mut lock = self.control.simple_sounds[index].lock_write();
                *lock = sound;
            }
            return Some(SimpleSoundToken(token));
        } else {
            let candidate_score = sound.compute_score(tick_now, player_position);
            let mut scores = [0; NUM_SIMPLE_SOUND_SLOTS];
            for i in 0..NUM_SIMPLE_SOUND_SLOTS {
                let control_block = self.control.simple_sounds[i].read();
                scores[i] = control_block.compute_score(tick_now, player_position);
            }
            let (min_index, min_score) = scores
                .iter()
                .copied()
                .enumerate()
                .min_by_key(|&(i, score)| score)
                .unwrap();
            if min_score < candidate_score {
                let token = seqnum + (min_index as u64);
                alloc_lock.simple_sound_tokens[min_index] = token;
                {
                    let mut lock = self.control.simple_sounds[min_index].lock_write();
                    *lock = sound;
                }
                return Some(SimpleSoundToken(token));
            }
            return None;
        }
    }

    pub(crate) fn update_position(
        &self,
        tick: u64,
        pos: Vector3<f64>,
        velocity: Vector3<f32>,
        azimuth: f64,
    ) {
        {
            let mut lock = self.control.player_state.lock_write();
            lock.position = pos;
            lock.velocity = velocity;
            lock.azimuth_radian = azimuth;
            lock.position_timebase_tick = tick;
        }
    }

    pub(crate) fn testonly_update_entity(
        &self,
        entity_id: u64,
        mv: EntityMove,
        mv2: Option<EntityMove>,
        len: f32,
        volume: f32,
    ) {
        if entity_id < NUM_TURBULENCE_ENTITY_SLOTS as u64 {
            let mut lock = self.control.turbulence_sources[entity_id as usize].lock_write();
            *lock = EntityTurbulenceControlBlock {
                flags: SOUND_PRESENT,
                entity_id,
                leading: mv,
                second: mv2,
                entity_len: len,
                volume,
            }
        }
    }

    pub(crate) fn testonly_play_footstep(&self, tick: u64, position: Vector3<f64>) {
        {
            if self
                .alloc_simple_sound(
                    tick,
                    position,
                    SimpleSoundControlBlock {
                        // This sounds awful when directionality is enabled
                        flags: SOUND_PRESENT | SOUND_SQUARELAW_ENABLED | SOUND_MOVESPEED_ENABLED,
                        position,
                        volume: 1.0,
                        start_tick: tick,
                        id: 0,
                        end_tick: u64::MAX,
                    },
                )
                .is_none()
            {
                log::warn!("Failed to allocate a sound slot for footstep")
            }
        }
    }
}

/// Starts an engine and returns the handle that controls it
pub(crate) async fn start_engine(
    settings: Arc<ArcSwap<GameSettings>>,
    timekeeper: Arc<Timekeeper>,
    initial_position: Vector3<f64>,
) -> Result<EngineHandle> {
    let control = Arc::new(SharedControl::new(settings.clone(), initial_position));
    let control_clone = control.clone();
    let (tx, rx) = tokio::sync::oneshot::channel::<Result<()>>();
    let cancellation_token = tokio_util::sync::CancellationToken::new();
    let token_clone = cancellation_token.clone();
    tokio::task::block_in_place(move || {
        let host = cpal::default_host();
        // TODO use a selected device from the settings
        let output_device =
            select_output_device(&host, &settings.load().audio.preferred_output_device)?;
        let mut device_configs: Vec<_> = output_device.supported_output_configs()?.collect();
        device_configs.sort_by(cpal::SupportedStreamConfigRange::cmp_default_heuristics);
        log::info!("Available device configs: {:?}", device_configs);
        // TODO use a selected config from the settings
        let selected_config_range = device_configs.last().context("no supported config")?;
        let selected_config = selected_config_range.with_max_sample_rate();
        log::info!("Using config: {:?}", selected_config);

        let mut engine_state =
            EngineState::new(control_clone, selected_config.clone(), timekeeper)?;

        let handle = tokio::runtime::Handle::current();

        let _ = std::thread::spawn(move || {
            let startup_work = move || {
                let stream = output_device
                    .build_output_stream(
                        &selected_config.into(),
                        move |data: &mut [f32], info: &cpal::OutputCallbackInfo| {
                            engine_state.callback(data, info);
                        },
                        move |err| {
                            log::error!("Output stream error: {}", err);
                        },
                        None,
                    )
                    .context("error creating output stream")?;

                stream.play().context("error playing stream")?;
                Ok(stream)
            };
            // We need to not drop the stream until after the cancellation token is cancelled
            let stream = match startup_work() {
                Ok(stream) => {
                    tx.send(Ok(()))
                        .expect("Audio startup notification: Listener dropped");
                    stream
                }
                Err(e) => {
                    log::error!("Failed to start audio: {}", e);
                    tx.send(Err(e)).unwrap();
                    return;
                }
            };
            handle.block_on(token_clone.cancelled());
            stream.pause().unwrap();
            drop(stream);
        });
        Ok::<(), anyhow::Error>(())
    })?;

    rx.await??;
    log::info!("Audio engine started successfully");

    Ok(EngineHandle {
        control,
        drop_guard: cancellation_token.drop_guard(),
        simple_sounds_bitmap: Mutex::new(AllocatorState::new()),
    })
}

fn select_output_device(host: &Host, prefix: &str) -> Result<cpal::Device> {
    if prefix.is_empty() {
        log::info!("Using default output device");
        return host
            .default_output_device()
            .context("no default output device");
    }
    let devices = host.output_devices()?;
    for device in devices {
        if device.name().unwrap().starts_with(prefix) {
            log::info!("Using output device: {}", device.name().unwrap());
            return Ok(device);
        }
    }
    log::warn!("No output device with name starting with {} found", prefix);
    host.default_output_device()
        .context("no default output device")
}

pub async fn start_engine_for_standalone_test() -> Result<EngineHandle> {
    let settings = Arc::new(ArcSwap::from_pointee(GameSettings::default()));
    let handle = start_engine(
        settings,
        Arc::new(Timekeeper::new(0)),
        Vector3::new(0.0, 0.0, 0.0),
    )
    .await?;

    {
        let mut lock = handle.control.turbulence_sources[15].lock_write();
        *lock = EntityTurbulenceControlBlock {
            flags: SOUND_PRESENT,
            entity_id: 0,
            leading: EntityMove::zero(),
            second: None,
            entity_len: 0.0,
            volume: 1.0,
        }
    }

    Ok(handle)
}

pub(crate) async fn start_engine_for_testing(
    settings: Arc<ArcSwap<GameSettings>>,
    timekeeper: Arc<Timekeeper>,
) -> Result<EngineHandle> {
    let handle = start_engine(settings, timekeeper, Vector3::new(0.0, 0.0, 0.0)).await?;
    Ok(handle)
}

struct EngineState {
    control: Arc<SharedControl>,
    private_control: PrivateControl,
    buffer_counter: u64,
    stream_config: cpal::SupportedStreamConfig,
    initial_callback_timing: Option<(Instant, cpal::StreamInstant)>,
    initial_playback_timing: Option<(Instant, cpal::StreamInstant)>,
    game_timekeeper: Arc<Timekeeper>,
    previous_tick: u64,

    nanos_per_sample: f64,
    total_samples_seen: u64,

    test_sampler: PrerecordedSampler,
    test_turbulence_sampler: ApproximateTurbulenceSampler,

    rng: rand::rngs::SmallRng,
}

// Const rather than using log! because I don't know what locks we might have to deal when the
// logger checks whether it's enabled
const ENABLE_DEBUG_PRINT: bool = false;

impl EngineState {
    pub(crate) fn callback(&mut self, data: &mut [f32], info: &OutputCallbackInfo) {
        data.fill(0.0);
        let now = Instant::now();

        let jitter = if self.initial_callback_timing.is_none() {
            self.initial_callback_timing = Some((now, info.timestamp().callback));
            self.initial_playback_timing = Some((now, info.timestamp().playback));
            0
        } else {
            let time_delta = now - self.initial_callback_timing.unwrap().0;
            let estimated_callback = self
                .initial_callback_timing
                .unwrap()
                .1
                .add(time_delta)
                .unwrap();

            if info.timestamp().callback < estimated_callback {
                -1 * (estimated_callback
                    .duration_since(&info.timestamp().callback)
                    .unwrap()
                    .as_nanos() as i64)
            } else {
                info.timestamp()
                    .callback
                    .duration_since(&estimated_callback)
                    .unwrap()
                    .as_nanos() as i64
            }
        };

        let margin = info
            .timestamp()
            .playback
            .duration_since(&info.timestamp().callback)
            .unwrap_or(Duration::ZERO);

        let elapsed = info
            .timestamp()
            .playback
            .duration_since(&self.initial_playback_timing.unwrap().1)
            .unwrap_or(Duration::ZERO);

        let buffer_start_tick = self
            .game_timekeeper
            .time_to_tick_estimate(self.initial_playback_timing.unwrap().0.add(elapsed));

        if data.len() % self.stream_config.channels() as usize != 0 {
            panic!(
                "Output callback length {} not divisible by number of channels {}",
                data.len(),
                self.stream_config.channels()
            );
        }
        let samples = data.len() / self.stream_config.channels() as usize;

        let buffer_end_tick =
            buffer_start_tick + (data.len() as f64 * self.nanos_per_sample) as u64;

        if ENABLE_DEBUG_PRINT && self.buffer_counter % 30 == 0 {
            log::info!(
                "Output callback: {:?}, diff {:?}. Buffer length: {} samples ({} seconds). Tick {:?},  Jitter {:?}",
                info,
                margin,
                data.len(),
                data.len() as f64 / self.stream_config.sample_rate().0 as f64,
                buffer_start_tick,
                jitter
            );
            log::info!(
                "samples seen: {}, elapsed: {:?}. Effective sample rate {}",
                self.total_samples_seen,
                elapsed,
                self.total_samples_seen as f64 / elapsed.as_secs_f64()
            );
        }
        self.buffer_counter += 1;

        self.total_samples_seen += samples as u64;

        let player_state = self.control.player_state.read();

        for i in 0..NUM_SIMPLE_SOUND_SLOTS {
            let control_block = self.control.simple_sounds[i].read();
            let private_state = &mut self.private_control.simple_sounds[i];

            if control_block.flags & SOUND_PRESENT == 0 {
                continue;
            }
            if control_block.start_tick >= buffer_end_tick {
                continue;
            }

            // Compute the start and end samples, and the corresponding ticks.
            let (eval_start_tick, sample_range) =
                if control_block.start_tick != private_state.start_tick {
                    if ENABLE_DEBUG_PRINT {
                        log::info!("Sound {} start @{}", i, control_block.start_tick,);
                    }

                    private_state.start_tick = control_block.start_tick;
                    private_state.elapsed_samples = 0;
                    private_state.last_distance.clear();
                    private_state.last_balance.clear();
                    // A new sound is starting at some point in ths current buffer.
                    // This is how many samples into the buffer.
                    let start_offset_samples =
                    // saturating sub - we might underflow if either the sound was delivered
                    // after its scheduled start time, or if we missed it due to a tiny floating or
                    // rounding error
                    ((control_block.start_tick.saturating_sub(buffer_start_tick) as f64)
                        / self.nanos_per_sample) as usize;

                    let remaining_samples = (samples.saturating_sub(start_offset_samples)) as u64;
                    // TODO: If a sound is truly short (fitting entirely into this buffer), we don't
                    // handle it quite right in the current prototype.

                    // The sound starts at the control block's start tick (game tick time since we
                    // haven't had a chance to accumulate any skew yet)
                    //
                    // The buffer will be filled from start_offset_samples to the end

                    (control_block.start_tick, start_offset_samples..samples)
                } else {
                    // The starting tick, in the reference frame of the audio source, but with skew
                    // correction (i.e. based on elapsed samples)
                    let start_tick = private_state.start_tick
                        + (private_state.elapsed_samples as f64 * self.nanos_per_sample) as u64;
                    (start_tick, 0..samples)
                };

            private_state.elapsed_samples += sample_range.len() as u64;
            if ENABLE_DEBUG_PRINT && self.buffer_counter % 30 == 0 {
                log::info!(
                    "ID {:?}, sample range: {:?}, start tick: {:?}, elapsed samples: {:?}",
                    control_block.id,
                    sample_range,
                    eval_start_tick,
                    private_state.elapsed_samples
                );
            }

            let sampler = &self.test_sampler;

            Self::sample_simple_sound(
                self.stream_config.channels() as usize,
                self.nanos_per_sample,
                data,
                sample_range,
                &player_state,
                &control_block,
                private_state,
                eval_start_tick,
                sampler,
            );
        }
        for i in 0..NUM_TURBULENCE_ENTITY_SLOTS {
            let control_block = self.control.turbulence_sources[i].read();
            if control_block.flags & SOUND_PRESENT == 0 {
                continue;
            }

            let private_state = &mut self.private_control.turbulence_sounds[i];

            Self::sample_entity_turbulence(
                self.stream_config.channels() as usize,
                self.nanos_per_sample,
                data,
                control_block,
                &player_state,
                &control_block,
                private_state,
                &mut self.rng,
                buffer_start_tick,
                samples,
            )
        }
    }

    fn new(
        control: Arc<SharedControl>,
        stream_config: cpal::SupportedStreamConfig,
        timekeeper: Arc<Timekeeper>,
    ) -> Result<EngineState> {
        Ok(EngineState {
            buffer_counter: 0,
            control,
            stream_config: stream_config.clone(),
            private_control: PrivateControl::default(),
            initial_callback_timing: None,
            initial_playback_timing: None,
            game_timekeeper: timekeeper,
            // what the heck? Extra factor of 2?
            nanos_per_sample: 1_000_000_000.0 / stream_config.sample_rate().0 as f64,
            previous_tick: 0,
            total_samples_seen: 0,

            test_sampler: PrerecordedSampler::from_wav(
                FOOTSTEP_WAV_BYTES,
                stream_config.sample_rate().0,
            )?,
            test_turbulence_sampler: ApproximateTurbulenceSampler {},
            rng: rand::rngs::SmallRng::from_rng(&mut rand::thread_rng())?,
        })
    }
    fn sample_simple_sound(
        channels: usize,
        nanos_per_sample: f64,
        buffer: &mut [f32],
        sample_range: Range<usize>,
        player_state: &PlayerState,
        control: &SimpleSoundControlBlock,
        state: &mut SimpleSoundState,
        start_tick: u64,
        sampler: &PrerecordedSampler,
    ) {
        let left_ear_azimuth = player_state.azimuth_radian + std::f64::consts::FRAC_PI_2;
        let left_ear_z = left_ear_azimuth.cos();
        let left_ear_x = left_ear_azimuth.sin();
        let left_ear_vec = Vector3::new(left_ear_x, 0.0, left_ear_z);

        for sample in sample_range {
            let tick = start_tick + (sample as f64 * nanos_per_sample) as u64;
            let tick_delta = tick - control.start_tick;

            let position = player_state.position(tick);
            let distance = state
                .last_distance
                .update((position - control.position).magnitude(), 0.001);
            let travel_time = if control.flags & SOUND_MOVESPEED_ENABLED != 0 {
                distance / SPEED_OF_SOUND_METER_PER_SECOND
            } else {
                0.0
            };
            let effective_time_delta = (tick_delta as f64 / 1_000_000_000.0) - travel_time;
            if effective_time_delta < 0.0 {
                // Sound not yet started
                continue;
            }
            if effective_time_delta
                > (control.end_tick - control.start_tick) as f64 / 1_000_000_000.0
            {
                // Sound finished
                break;
            }
            let amplitude = if control.flags & SOUND_SQUARELAW_ENABLED != 0 {
                let clamped_distance = distance.max(MIN_DISTANCE) as f32;
                // this affects amplitude, not power, so it should be non-squared
                control.volume / clamped_distance
            } else {
                control.volume
            };

            let balance_left = left_ear_vec.dot(position - control.position);
            let balance_left = state.last_balance.update(balance_left, 0.001);
            let l_amplitude = amplitude * (1.0 + 0.8 * balance_left as f32);

            let r_amplitude = amplitude * (1.0 - 0.8 * balance_left as f32);

            let (l, r) = sampler.sample(effective_time_delta / nanos_per_sample * 1_000_000_000.0);
            let (l, r) = if channels == 1 {
                (l + r / 2.0, 0.0)
            } else {
                (l, r)
            };

            for chan in 0..channels {
                // let sample_value =
                //     (2.0 * std::f64::consts::PI * effective_time_delta * (control.id as f64)).sin();
                let sample_value = if chan == 0 { l } else { r };
                let amplitude = if channels == 2 && control.flags & SOUND_DIRECTIONAL != 0 {
                    if chan == 0 {
                        l_amplitude
                    } else {
                        r_amplitude
                    }
                } else {
                    amplitude
                };
                let index = sample * channels + chan;
                buffer[index] += amplitude * sample_value / 4000.0;
            }
        }
    }
    fn sample_entity_turbulence(
        channels: usize,
        nanos_per_sample: f64,
        data: &mut [f32],
        control_block: EntityTurbulenceControlBlock,
        player_state: &PlayerState,
        entity_turbulence_control_block: &EntityTurbulenceControlBlock,
        scratchpad: &mut TurbulenceScratchpad,
        rng: &mut SmallRng,
        buffer_start_tick: u64,
        samples: usize,
    ) {
        let left_ear_azimuth = player_state.azimuth_radian + std::f64::consts::FRAC_PI_2;
        let left_ear_z = left_ear_azimuth.cos();
        let left_ear_x = left_ear_azimuth.sin();
        let left_ear_vec = Vector3::new(left_ear_x, 0.0, left_ear_z);

        if control_block.flags & SOUND_MOVESPEED_ENABLED != 0 {
            panic!("Doppler effect support not yet implemented, requires larger move queues")
        }
        if control_block.entity_id != scratchpad.entity_id {
            *scratchpad = TurbulenceScratchpad {
                entity_id: control_block.entity_id,
                ..TurbulenceScratchpad::default()
            }
        }

        for sample in 0..samples {
            let tick = buffer_start_tick + (sample as f64 * nanos_per_sample) as u64;

            let elapsed_or_overflow = control_block.leading.elapsed_or_overflow(tick);
            let (entity_move, entity_move_elapsed) = match elapsed_or_overflow {
                ElapsedOrOverflow::ElapsedThisMove(time) => (control_block.leading, time),
                ElapsedOrOverflow::OverflowIntoNext(time) => {
                    if let Some(second) = control_block.second {
                        (second, time.min(second.total_time_sec()))
                    } else {
                        (
                            control_block.leading,
                            control_block.leading.total_time_sec(),
                        )
                    }
                }
            };

            let entity_pos = entity_move.qproj(entity_move_elapsed);
            let player_pos = player_state.position;
            // distance this sample
            let distance = (player_pos - entity_pos).magnitude();
            // distance in this sample if the player hadn't moved
            let distance_unmoved_player = (scratchpad.last_player_pos - entity_pos).magnitude();
            let distance_diff = (distance_unmoved_player - scratchpad.last_distance);

            scratchpad.last_distance = distance;
            scratchpad.last_player_pos = player_pos;

            let current_velocity = entity_move.instantaneous_velocity(entity_move_elapsed);

            scratchpad.approach_state = match scratchpad.approach_state {
                ApproachState::Initial(acc) => {
                    // Looking for a decrease
                    let new_acc = (acc - distance_diff).max(0.0);
                    if new_acc > 0.1 {
                        ApproachState::DecreaseDetected(0.0)
                    } else {
                        ApproachState::Initial(new_acc)
                    }
                }
                ApproachState::DecreaseDetected(acc) => {
                    let new_acc = (acc + distance_diff).max(0.0);
                    if new_acc > 0.1 {
                        let _span = span!("start passing");
                        scratchpad.edge_cycle = 0.0;
                        // This is a local minimum
                        ApproachState::Passing(
                            0.0,
                            entity_pos,
                            player_pos - entity_pos,
                            entity_move
                                .instantaneous_velocity(entity_move_elapsed)
                                .normalize(),
                        )
                    } else {
                        ApproachState::DecreaseDetected(new_acc)
                    }
                }
                x => x,
            };

            let (effective_displacement, start_recovery) = match &mut scratchpad.approach_state {
                ApproachState::Initial(_) => (player_pos - entity_pos, false),
                ApproachState::DecreaseDetected(_) => (player_pos - entity_pos, false),
                ApproachState::Passing(
                    travel_since,
                    captured_entity_location,
                    captured_e2p,
                    captured_velocity,
                ) => {
                    let additional_travel =
                        (current_velocity.magnitude() as f64) * nanos_per_sample / 1_000_000_000.0;

                    let new_travel = *travel_since + additional_travel;
                    if (new_travel / 32.0).floor() != (*travel_since / 32.0).floor() {
                        scratchpad.edge_cycle = 0.0;
                    }

                    *travel_since = new_travel;
                    let current_displacement = player_pos - *captured_entity_location;
                    let captured_velocity = captured_velocity.cast().unwrap();
                    let parallel = current_displacement.project_on(captured_velocity);
                    let perpendicular = current_displacement - parallel;

                    // How far the player has moved *along* with the entity
                    let parallel_displacement = current_displacement.dot(captured_velocity);
                    (
                        perpendicular,
                        *travel_since > (control_block.entity_len as f64 + parallel_displacement),
                    )
                }
            };
            let effective_distance = effective_displacement.magnitude();
            let clamped_distance = effective_distance.max(MIN_ENTITY_DISTANCE) as f32;
            // multiplier affects amplitude, not power
            let multiplier = 1.0 / (clamped_distance);
            if start_recovery {
                scratchpad.approach_state = ApproachState::Initial(0.0);
                scratchpad.trailing_edge_blend_factor = 1.0;
                scratchpad.trailing_edge_blend_value = multiplier;
                scratchpad.trailing_edge_blend_source_direction =
                    effective_displacement.normalize();
                scratchpad.edge_cycle = -1.0;
            }
            let edge_multiplier = if scratchpad.edge_cycle < 0.0 {
                1.0
            } else if scratchpad.edge_cycle < 0.25 {
                scratchpad.edge_cycle += nanos_per_sample as f32 / 250_000_000.0;
                1.0 + (2.0 * scratchpad.edge_cycle)
            } else if scratchpad.edge_cycle < 1.0 {
                scratchpad.edge_cycle += nanos_per_sample as f32 / 250_000_000.0;
                (5.0 / 3.0) - (2.0 / 3.0 * scratchpad.edge_cycle)
            } else {
                scratchpad.edge_cycle = -1.0;
                1.0
            };

            // Theoretically should be N^5 or N^6, but this is subjectively better
            let speed_multiplier = (current_velocity.magnitude() / 70.0).powi(4);

            let multiplier = scratchpad.trailing_edge_blend_value
                * scratchpad.trailing_edge_blend_factor
                + multiplier * (1.0 - scratchpad.trailing_edge_blend_factor);
            scratchpad.trailing_edge_blend_factor = (scratchpad.trailing_edge_blend_factor
                - nanos_per_sample as f32 / 2_000_000_000.0)
                .clamp(0.0, 1.0);

            let effective_displacement_smoothed = scratchpad.trailing_edge_blend_source_direction
                * scratchpad.trailing_edge_blend_factor as f64
                + effective_displacement.normalize()
                    * (1.0 - scratchpad.trailing_edge_blend_factor as f64);
            let effective_dir = effective_displacement_smoothed.normalize();

            let amplitude = control_block.volume * multiplier * edge_multiplier * speed_multiplier;

            let mut balance_left = left_ear_vec.dot(effective_dir);
            if balance_left.is_nan() {
                balance_left = 0.0;
            }
            let balance_left = scratchpad.last_balance.update(balance_left, 0.001);
            let l_amplitude = amplitude * (1.0 + 0.8 * balance_left as f32);

            let r_amplitude = amplitude * (1.0 - 0.8 * balance_left as f32);
            let (l, r) = ApproximateTurbulenceSampler::sample(scratchpad, rng);
            let (l, r) = if channels == 1 {
                (l + r / 2.0, 0.0)
            } else {
                (l, r)
            };
            for chan in 0..channels {
                // let sample_value =
                //     (2.0 * std::f64::consts::PI * effective_time_delta * (control.id as f64)).sin();
                let sample_value = if chan == 0 { l } else { r };
                let amplitude = if channels == 2 {
                    if chan == 0 {
                        l_amplitude
                    } else {
                        r_amplitude
                    }
                } else {
                    amplitude
                };
                let index = sample * channels + chan;
                data[index] += amplitude * sample_value / 800.0;
            }
        }
    }
}

/// The player's movement state
/// This must be small and copiable since it's accessed via a seqlock
#[derive(Clone, Copy, Debug)]
#[repr(align(64))]
pub(crate) struct PlayerState {
    position: Vector3<f64>,
    velocity: Vector3<f32>,
    acceleration: Vector3<f32>,
    azimuth_radian: f64,
    position_timebase_tick: u64,
}

impl PlayerState {
    #[inline]
    pub(crate) fn position(&self, tick: u64) -> Vector3<f64> {
        let dt = tick.saturating_sub(self.position_timebase_tick) as f64 / 1_000_000_000.0;
        let x = self.position.x
            + (self.velocity.x as f64 * dt)
            + 0.5 * (self.acceleration.x as f64 * dt * dt);
        let y = self.position.y
            + (self.velocity.y as f64 * dt)
            + 0.5 * (self.acceleration.y as f64 * dt * dt);
        let z = self.position.z
            + (self.velocity.z as f64 * dt)
            + 0.5 * (self.acceleration.z as f64 * dt * dt);
        Vector3::new(x, y, z)
    }
}

pub const NUM_SIMPLE_SOUND_SLOTS: usize = 64;
pub const NUM_TURBULENCE_ENTITY_SLOTS: usize = 64;

/// The private control state, used internally by the audio engine to track the state of
/// the player and sounds
struct PrivateControl {
    simple_sounds: Box<[SimpleSoundState; NUM_SIMPLE_SOUND_SLOTS]>,
    turbulence_sounds: Box<[TurbulenceScratchpad; NUM_TURBULENCE_ENTITY_SLOTS]>,
}
impl Default for PrivateControl {
    fn default() -> PrivateControl {
        PrivateControl {
            simple_sounds: Box::new([SimpleSoundState::default(); NUM_SIMPLE_SOUND_SLOTS]),
            turbulence_sounds: Box::new(
                [TurbulenceScratchpad::default(); NUM_TURBULENCE_ENTITY_SLOTS],
            ),
        }
    }
}

/// The shared control state, via which sounds can be controlled
pub(crate) struct SharedControl {
    player_state: SeqLock<PlayerState>,
    simple_sounds: Box<[SeqLock<SimpleSoundControlBlock>; NUM_SIMPLE_SOUND_SLOTS]>,
    turbulence_sources: Box<[SeqLock<EntityTurbulenceControlBlock>; NUM_SIMPLE_SOUND_SLOTS]>,
    settings: Arc<ArcSwap<GameSettings>>,
}
impl SharedControl {
    fn new(settings: Arc<ArcSwap<GameSettings>>, initial_position: Vector3<f64>) -> SharedControl {
        const SIMPLE_SOUND_CONST_INIT: SeqLock<SimpleSoundControlBlock> =
            SeqLock::new(SimpleSoundControlBlock::const_default());
        const TURBULENCE_SOURCE_CONST_INIT: SeqLock<EntityTurbulenceControlBlock> =
            SeqLock::new(EntityTurbulenceControlBlock::const_default());
        SharedControl {
            player_state: SeqLock::new(PlayerState {
                position: initial_position,
                velocity: Vector3::zero(),
                acceleration: Vector3::zero(),
                azimuth_radian: 0.0,
                position_timebase_tick: 0,
            }),
            simple_sounds: Box::new([SIMPLE_SOUND_CONST_INIT; NUM_SIMPLE_SOUND_SLOTS]),
            turbulence_sources: Box::new(
                [TURBULENCE_SOURCE_CONST_INIT; NUM_TURBULENCE_ENTITY_SLOTS],
            ),
            // Start all volumes at 0 until they are set by the game loop
            settings,
        }
    }
}

// Note: This must be small and copiable, since it's accessed via a seqlock
#[derive(Clone, Copy, Debug)]
#[repr(align(64))]
/// The state of a sound that plays at some time and place.
pub(crate) struct SimpleSoundControlBlock {
    /// The flags for the sound. See SOUND_* constants
    ///
    /// Note that the type of this field is subject to change if more than 8 flags are needed.
    pub flags: u8,
    /// The position of the sound in 3D space. The sound is assumed to be stationary,
    /// but the player's motion may be considered for Doppler effect.
    pub position: Vector3<f64>,
    /// The volume of the sound. If this is set to 0.0, the sound is silenced but continues to be
    /// active.
    pub volume: f32,
    /// The starting tick of the sound
    pub start_tick: u64,
    /// The sound ID (not yet used, *current test will play a sine with this frequency*)
    pub id: u32,
    /// The ending tick of the sound. If the span from start_tick to end_tick is greater than
    /// the length of the sound, it will loop. u64::MAX is treated as effectively infinite.
    ///
    /// Skew policy: If the tick clock and codec sample clock go out of skew, the sound will stop
    /// based on
    pub end_tick: u64,
}

impl SimpleSoundControlBlock {
    const fn const_default() -> Self {
        SimpleSoundControlBlock {
            flags: 0,
            position: Vector3::new(0.0, 0.0, 0.0),
            volume: 0.0,
            start_tick: 0,
            id: 0,
            end_tick: 0,
        }
    }

    /// Heuristically identify which sounds are most important. Higher values are considered more
    /// important.
    pub(crate) fn compute_score(&self, tick_now: u64, player_position: Vector3<f64>) -> u64 {
        if self.flags & SOUND_PRESENT == 0 {
            return 0;
        }
        if self.flags & SOUND_STICKY != 0 {
            return u64::MAX;
        }

        // We assume that the player won't move faster than the speed of sound
        let distance = (self.position - player_position).magnitude();
        let estimated_end_tick = if self.flags & SOUND_MOVESPEED_ENABLED != 0 {
            let delay_ticks =
                ((distance / SPEED_OF_SOUND_METER_PER_SECOND) * 1_000_000_000.0) as u64;
            // The estimated end tick in the *player* reference frame
            self.end_tick.saturating_add(delay_ticks)
        } else {
            self.end_tick
        };
        if estimated_end_tick < tick_now {
            return 0;
        }

        const BASE_SCORE: u64 = 1 << 60;
        const BASE_SCORE_F64: f64 = BASE_SCORE as f64;
        return if self.flags & SOUND_SQUARELAW_ENABLED != 0 {
            let dropoff = (distance * distance).max(1.0);
            let score = BASE_SCORE_F64 / dropoff;
            if !score.is_finite() || !score.is_sign_positive() {
                log::warn!(
                    "Nonsensical score for control block {:?}, tick {}, pos {:?}",
                    self,
                    tick_now,
                    player_position
                );
                return 0;
            }
            score as u64
        } else {
            BASE_SCORE
        };
    }
}

#[derive(Clone, Copy)]
struct EntityTurbulenceControlBlock {
    flags: u8,
    entity_id: u64,
    leading: EntityMove,
    second: Option<EntityMove>,
    entity_len: f32,
    volume: f32,
}

impl EntityTurbulenceControlBlock {
    const fn const_default() -> EntityTurbulenceControlBlock {
        EntityTurbulenceControlBlock {
            flags: 0,
            entity_id: 0,
            leading: EntityMove::zero(),
            second: None,
            entity_len: 0.0,
            volume: 0.0,
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct SimpleSoundState {
    start_tick: u64,
    elapsed_samples: u64,
    last_distance: SmoothedVar,
    last_balance: SmoothedVar,
}
impl Default for SimpleSoundState {
    fn default() -> Self {
        SimpleSoundState {
            start_tick: 0,
            elapsed_samples: 0,
            last_distance: SmoothedVar::default(),
            last_balance: SmoothedVar::default(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct SmoothedVar {
    val: f64,
}
impl Default for SmoothedVar {
    fn default() -> Self {
        SmoothedVar { val: 0.0 }
    }
}
impl SmoothedVar {
    #[inline]
    fn update(&mut self, val: f64, mix: f64) -> f64 {
        if self.val.is_nan() {
            self.val = val;
        } else {
            self.val = self.val * (1.0 - mix) + val * mix;
        }
        self.val
    }
    fn reset(&mut self, val: f64) {
        self.val = val;
    }
    fn get(&self) -> f64 {
        self.val
    }
    fn clear(&mut self) {
        self.val = f64::NAN;
    }
}

/// If set, the entry in the control block is valid and enabled
pub const SOUND_PRESENT: u8 = 0x1;

/// If set, the sound will be subject to speed-of-sound and doppler effects
pub const SOUND_MOVESPEED_ENABLED: u8 = 0x2;
/// If set, the sound will be subject to square-law effects affecting its amplitude
pub const SOUND_SQUARELAW_ENABLED: u8 = 0x4;
/// If set, the sound undergoes directionality effects
pub const SOUND_DIRECTIONAL: u8 = 0x8;
/// If set, the sound is considered sticky, and will never be deallocated
pub const SOUND_STICKY: u8 = 0x10;

/// The minimum distance considered for square-law effects. By enforcing this, we avoid
/// excessive amplitudes for near-zero distances.
pub const MIN_DISTANCE: f64 = 1.0;
pub const MIN_ENTITY_DISTANCE: f64 = 5.0;

pub const SPEED_OF_SOUND_METER_PER_SECOND: f64 = 343.0;

const OVERSAMPLE_FACTOR: u32 = 1;

struct PrerecordedSampler {
    data: Vec<(f32, f32)>,
}
impl PrerecordedSampler {
    fn from_wav(wav_bytes: &[u8], target_sample_rate: u32) -> Result<Self> {
        let wav = hound::WavReader::new(wav_bytes).unwrap();
        let source_rate = dbg!(wav.spec().sample_rate);
        let source_channels = wav.spec().channels as usize;
        let internal_target_rate = dbg!(target_sample_rate * OVERSAMPLE_FACTOR);
        if source_channels != 1 && source_channels != 2 {
            bail!(
                "Unsupported number of channels: {}; only 1 or 2 are supported",
                source_channels
            );
        }

        let mut left = Vec::new();
        let mut right = Vec::new();
        for (i, sample) in wav.into_samples::<i32>().enumerate() {
            let sample = sample.unwrap();
            if source_channels == 1 {
                left.push(sample as f32);
                right.push(sample as f32);
            } else if source_channels == 2 && i % 2 == 0 {
                left.push(sample as f32);
            } else if source_channels == 2 && i % 2 == 1 {
                right.push(sample as f32);
            }
        }

        const CHUNK_SIZE: usize = 4096;

        let mut resampler = rubato::FftFixedInOut::new(
            source_rate as usize,
            internal_target_rate as usize,
            CHUNK_SIZE,
            2,
        )?;
        let mut left_output = Vec::new();
        let mut right_output = Vec::new();

        let mut pos = 0;

        loop {
            use rubato::Resampler;
            let input_frames = resampler.input_frames_next();
            let remaining = left.len() - pos;
            if remaining < input_frames {
                if remaining > 0 {
                    let outputs =
                        resampler.process_partial(Some(&[&left[pos..], &right[pos..]]), None)?;
                    assert_eq!(outputs.len(), 2);
                    let mut outputs = outputs.into_iter();
                    left_output.extend(outputs.next().unwrap());
                    right_output.extend(outputs.next().unwrap());
                }

                let leftover_outputs = resampler.process_partial::<Vec<f32>>(None, None)?;
                assert_eq!(leftover_outputs.len(), 2);
                let mut leftover_outputs = leftover_outputs.into_iter();
                left_output.extend(leftover_outputs.next().unwrap());
                right_output.extend(leftover_outputs.next().unwrap());
                break;
            } else {
                let outputs = resampler.process(
                    &[
                        &left[pos..pos + input_frames],
                        &right[pos..pos + input_frames],
                    ],
                    None,
                )?;
                assert_eq!(outputs.len(), 2);
                let mut outputs = outputs.into_iter();
                left_output.extend(outputs.next().unwrap());
                right_output.extend(outputs.next().unwrap());
                pos += input_frames;
            }
        }

        let data = left_output
            .into_iter()
            .zip(right_output.into_iter())
            .map(|(l, r)| (l, r))
            .collect();
        Ok(Self { data })
    }

    #[inline]
    fn sample(&self, sample: f64) -> (f32, f32) {
        let upscaled = sample * OVERSAMPLE_FACTOR as f64;
        let preceding = upscaled.floor();
        let following = upscaled.ceil();
        let fpart = (upscaled - preceding) as f32;

        let preceding_sample = if preceding < 0.0 {
            (0.0, 0.0)
        } else if preceding as usize >= self.data.len() {
            (0.0, 0.0)
        } else {
            self.data[preceding as usize]
        };

        let following_sample = if following < 0.0 {
            (0.0, 0.0)
        } else if following as usize >= self.data.len() {
            (0.0, 0.0)
        } else {
            self.data[following as usize]
        };

        (
            (1.0 - fpart) * preceding_sample.0 + fpart * following_sample.0,
            (1.0 - fpart) * preceding_sample.1 + fpart * following_sample.1,
        )
    }
}

struct ApproximateTurbulenceSampler {}
impl ApproximateTurbulenceSampler {
    fn sample(scratchpad: &mut TurbulenceScratchpad, rng: &mut SmallRng) -> (f32, f32) {
        let rng_val = rng.gen_range(-1.0..1.0) * 4000.0;
        let filtered = rng_val * 0.1 + scratchpad.iir_state * 0.9;
        scratchpad.iir_state = filtered;
        (filtered as f32, filtered as f32)
    }
}

trait Sampler {
    fn sample(
        &self,
        sample: f64,
        player_state: &PlayerState,
        scratchpad: &mut TurbulenceScratchpad,
        rng: &mut SmallRng,
    ) -> (f32, f32);
}

#[derive(Clone, Copy, Debug)]
enum ApproachState {
    // Haven't seen a large enough decrease yet, value is accumulated decrease
    Initial(f64),
    // Haven't seen a large enough increase yet, value is accumulated increase
    DecreaseDetected(f64),
    // Fields:
    // * Distance of the approach
    // * Entity travel since that approach
    // * Entity location at capture time
    // * Entity->player displacement at capture time, unnormalized
    // * Entity movement direction at capture time, normalized
    Passing(f64, Vector3<f64>, Vector3<f64>, Vector3<f32>),
}

#[derive(Copy, Clone, Debug)]
struct TurbulenceScratchpad {
    // Really simple first-order IIR filter, single state
    iir_state: f64,
    // The entity ID, used to make sure that we're looking at the same entity
    entity_id: u64,
    // Last player position, used for close-approach detection
    last_player_pos: Vector3<f64>,
    // Last distance
    last_distance: f64,
    approach_state: ApproachState,
    // Overview of algorithm:
    // Track distance from last_player_pos vs current player pos
    // Once it hits a local minimum, reset acc_since_last_approach,
    // and start counting it up. Snapshot the actual approach distance
    // into last_approach_distance
    last_balance: SmoothedVar,
    trailing_edge_blend_factor: f32,
    trailing_edge_blend_value: f32,
    trailing_edge_blend_source_direction: Vector3<f64>,
    // Used to provide an extra burst of sound at edges
    edge_cycle: f32,
}
impl Default for TurbulenceScratchpad {
    fn default() -> Self {
        TurbulenceScratchpad {
            iir_state: 0.0,
            entity_id: u64::MAX,
            last_player_pos: Vector3::zero(),
            last_distance: 0.0,
            approach_state: ApproachState::Initial(0.0),
            last_balance: Default::default(),
            trailing_edge_blend_value: 0.0,
            trailing_edge_blend_factor: 0.0,
            trailing_edge_blend_source_direction: Vector3::zero(),
            edge_cycle: -1.0,
        }
    }
}

// Exposed for benchmarks
#[doc(hidden)]
pub mod generated_eqns {
    #[inline]
    #[doc(hidden)]
    pub fn travel_time_newton_raphson(
        dt: f64,
        rt: f64,
        px: f64,
        py: f64,
        pz: f64,
        vx: f64,
        vy: f64,
        vz: f64,
        ax: f64,
        ay: f64,
        az: f64,
    ) -> f64 {
        /*
        See attached ipynb for symbolic derivation.

        It's tempting to optimize this using Herbie. Sadly, the results are nonsensical.
        e.g. there are results with higher reported accuracy that disregard dt entirely.

        Nevertheless, here's the FPCore form for reference:

        FPCore form:
        (FPCore (dt SPEED_OF_SOUND_METER_PER_SECOND ax rt px vx ay py vy az pz vz)
        :precision binary64
        (let* ((t_0 (- (* 2.0 dt) (* 2.0 rt)))
               (t_1 (+ (- dt) rt))
               (t_2 (pow t_1 2.0))
               (t_3 (+ (+ (* (* 0.5 ay) t_2) py) (* vy t_1)))
               (t_4 (+ (+ (* (* 0.5 ax) t_2) px) (* vx t_1)))
               (t_5 (+ (+ (* (* 0.5 az) t_2) pz) (* vz t_1)))
               (t_6 (sqrt (+ (+ (pow t_4 2.0) (pow t_3 2.0)) (pow t_5 2.0)))))
          (-
           dt
           (/
            (- (* dt SPEED_OF_SOUND_METER_PER_SECOND) t_6)
            (-
             SPEED_OF_SOUND_METER_PER_SECOND
             (/
              (+
               (+
                (* (* (/ 1.0 2.0) (- (* ax t_0) (* 2.0 vx))) t_4)
                (* (* (/ 1.0 2.0) (- (* ay t_0) (* 2.0 vy))) t_3))
               (* (* (/ 1.0 2.0) (- (* az t_0) (* 2.0 vz))) t_5))
              t_6))))))
        */
        //
        // Naive Rust code from sympy:
        dt - (dt * super::SPEED_OF_SOUND_METER_PER_SECOND
            - ((0.5 * ax * (-dt + rt).powi(2) + px + vx * (-dt + rt)).powi(2)
                + (0.5 * ay * (-dt + rt).powi(2) + py + vy * (-dt + rt)).powi(2)
                + (0.5 * az * (-dt + rt).powi(2) + pz + vz * (-dt + rt)).powi(2))
            .sqrt())
            / (super::SPEED_OF_SOUND_METER_PER_SECOND
                - ((1_f64 / 2.0)
                    * (ax * (2.0 * dt - 2.0 * rt) - 2.0 * vx)
                    * (0.5 * ax * (-dt + rt).powi(2) + px + vx * (-dt + rt))
                    + (1_f64 / 2.0)
                        * (ay * (2.0 * dt - 2.0 * rt) - 2.0 * vy)
                        * (0.5 * ay * (-dt + rt).powi(2) + py + vy * (-dt + rt))
                    + (1_f64 / 2.0)
                        * (az * (2.0 * dt - 2.0 * rt) - 2.0 * vz)
                        * (0.5 * az * (-dt + rt).powi(2) + pz + vz * (-dt + rt)))
                    / ((0.5 * ax * (-dt + rt).powi(2) + px + vx * (-dt + rt)).powi(2)
                        + (0.5 * ay * (-dt + rt).powi(2) + py + vy * (-dt + rt)).powi(2)
                        + (0.5 * az * (-dt + rt).powi(2) + pz + vz * (-dt + rt)).powi(2))
                    .sqrt())
    }
}

mod testdata {
    pub(super) const FOOTSTEP_WAV_BYTES: &[u8] = include_bytes!("footstep_temp.wav");
}
