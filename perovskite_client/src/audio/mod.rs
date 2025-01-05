use std::collections::hash_map::Entry;
use std::io::Cursor;
use std::num::{NonZero, NonZeroU64};
use std::ops::{Add, Deref, DerefMut, Range};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use arc_swap::ArcSwap;
use cgmath::{InnerSpace, Vector3, Zero};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Host, OutputCallbackInfo};
use hound::WavReader;
use parking_lot::{Condvar, Mutex};
use perovskite_core::coordinates::{BlockCoordinate, ChunkCoordinate, ChunkOffset};
use perovskite_core::protocol::audio::{SampledSound, SoundSource};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rubato::{FftFixedInOut, Resampler};
use rustc_hash::{FxHashMap, FxHashSet};
use seqlock::SeqLock;
use smallvec::SmallVec;
use tokio_util::sync::{CancellationToken, DropGuard};
use tracy_client::{plot, span};

use crate::cache::CacheManager;
use crate::game_state::entities::{ElapsedOrOverflow, EntityMove};
use crate::game_state::settings::GameSettings;
use crate::game_state::timekeeper::Timekeeper;
use crate::game_state::ClientState;

// Public for testing
pub struct EngineHandle {
    control: Arc<SharedControl>,
    drop_guard: DropGuard,
    allocator_state: Mutex<AllocatorState>,
    simple_sound_lengths: FxHashMap<u32, u64>,
}

/// An opaque token used to modify/remove a sound in a slot.
/// Resilient to evictions, since it keeps a sequence number.
///
/// Note: PartialOrd/Ord derived implementation is only for the benefit of
/// ord-based datastructures (e.g. BTreeMap). No semantic meaning should
/// be ascribed to the result of these comparisons.
#[derive(PartialEq, Eq, Hash, Debug, PartialOrd, Ord, Clone, Copy)]
pub struct SimpleSoundToken(NonZeroU64);
#[derive(PartialEq, Eq, Hash, Debug, PartialOrd, Ord, Clone, Copy)]
pub struct ProceduralEntityToken(NonZeroU64);

struct AllocatorState {
    simple_sound_tokens: [u64; NUM_SIMPLE_SOUND_SLOTS],
    simple_sounds_next_sequence: u64,
    entity_slot_tokens: [u64; NUM_PROCEDURAL_ENTITY_SLOTS],
    entity_slots_next_sequence: u64,
}

impl AllocatorState {
    fn new() -> AllocatorState {
        AllocatorState {
            simple_sound_tokens: [0; NUM_SIMPLE_SOUND_SLOTS],
            // This must start at a nonzero value, since 0 is a sentinel
            simple_sounds_next_sequence: NUM_SIMPLE_SOUND_SLOTS as u64,
            entity_slot_tokens: [0; NUM_PROCEDURAL_ENTITY_SLOTS],
            // Likewise
            entity_slots_next_sequence: NUM_PROCEDURAL_ENTITY_SLOTS as u64,
        }
    }
}

impl EngineHandle {
    pub(crate) fn is_token_evicted(&self, token: SimpleSoundToken) -> bool {
        let index = token.0.get() % (NUM_SIMPLE_SOUND_SLOTS as u64);
        let mut alloc_lock = self.allocator_state.lock();
        alloc_lock.simple_sound_tokens[index as usize] != token.0.get()
    }

    pub(crate) fn insert_or_update_simple_sound(
        &self,
        tick_now: u64,
        player_position: Vector3<f64>,
        sound: SimpleSoundControlBlock,
        previous_token: Option<SimpleSoundToken>,
    ) -> Option<SimpleSoundToken> {
        assert_ne!(sound.flags & SOUND_PRESENT, 0);
        let mut alloc_lock = self.allocator_state.lock();

        let seqnum = alloc_lock.simple_sounds_next_sequence;

        if let Some(token) = previous_token {
            let index = token.0.get() % (NUM_SIMPLE_SOUND_SLOTS as u64);
            // If the entity hasn't been evicted, put it back in the same slot
            if alloc_lock.entity_slot_tokens[index as usize] == token.0.get() {
                let mut lock = self.control.simple_sounds[index as usize].lock_write();
                *lock = sound;
                return Some(token);
            }
        }
        alloc_lock.simple_sounds_next_sequence += NUM_SIMPLE_SOUND_SLOTS as u64;

        return if let Some(index) = alloc_lock.simple_sound_tokens.iter().position(|x| *x == 0) {
            let token = seqnum + (index as u64);
            alloc_lock.simple_sound_tokens[index] = token;
            {
                let mut lock = self.control.simple_sounds[index].lock_write();
                *lock = sound;
            }
            Some(SimpleSoundToken(NonZeroU64::new(token).unwrap()))
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
                return Some(SimpleSoundToken(NonZeroU64::new(token).unwrap()));
            }
            None
        };
    }

    pub(crate) fn remove_simple_sound(&self, token: SimpleSoundToken) -> bool {
        let mut alloc_lock = self.allocator_state.lock();

        let index = token.0.get() % (NUM_SIMPLE_SOUND_SLOTS as u64);
        if alloc_lock.simple_sound_tokens[index as usize] == token.0.get() {
            alloc_lock.simple_sound_tokens[index as usize] = 0;
            let mut lock = self.control.simple_sounds[index as usize].lock_write();
            *lock = SimpleSoundControlBlock::const_default();
            true
        } else {
            false
        }
    }

    pub(crate) fn update_entity_state(
        &self,
        tick_now: u64,
        player_position: Vector3<f64>,
        sound: ProceduralEntitySoundControlBlock,
        previous_token: Option<ProceduralEntityToken>,
    ) -> Option<ProceduralEntityToken> {
        assert_ne!(sound.flags & SOUND_PRESENT, 0);
        let mut alloc_lock = self.allocator_state.lock();

        if let Some(token) = previous_token {
            let index = token.0.get() % (NUM_PROCEDURAL_ENTITY_SLOTS as u64);
            // If the entity hasn't been evicted, put it back in the same slot
            if alloc_lock.entity_slot_tokens[index as usize] == token.0.get() {
                let mut lock = self.control.entity_slots[index as usize].lock_write();
                *lock = sound;
                return Some(token);
            }
        }

        let seqnum = alloc_lock.entity_slots_next_sequence;
        alloc_lock.entity_slots_next_sequence += NUM_PROCEDURAL_ENTITY_SLOTS as u64;

        if let Some(index) = alloc_lock.entity_slot_tokens.iter().position(|x| *x == 0) {
            let token = seqnum + (index as u64);
            alloc_lock.entity_slot_tokens[index] = token;
            {
                let mut lock = self.control.entity_slots[index].lock_write();
                *lock = sound;
            }
            return Some(ProceduralEntityToken(NonZeroU64::new(token).unwrap()));
        } else {
            let candidate_score = sound.compute_score(tick_now, player_position);
            let mut scores = [0; NUM_PROCEDURAL_ENTITY_SLOTS];
            for i in 0..NUM_PROCEDURAL_ENTITY_SLOTS {
                let control_block = self.control.entity_slots[i].read();
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
                alloc_lock.entity_slot_tokens[min_index] = token;
                {
                    let mut lock = self.control.entity_slots[min_index].lock_write();
                    *lock = sound;
                }
                return Some(ProceduralEntityToken(NonZeroU64::new(token).unwrap()));
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
        testonly_entity_attached: bool,
    ) {
        {
            let mut lock = self.control.player_state.lock_write();
            lock.position = pos;
            lock.velocity = velocity;
            lock.azimuth_radian = azimuth;
            lock.position_timebase_tick = tick;
            lock.entity_filter_state = if testonly_entity_attached {
                EntityFilterState {
                    cutoff_hz: 250,
                    degree: 2,
                }
            } else {
                EntityFilterState {
                    cutoff_hz: 0,
                    degree: 0,
                }
            };
            lock.entity_filter_extra_gain = if testonly_entity_attached { 5.0 } else { 1.0 };
        }
        self.control.enabled.store(true, Ordering::Release);
    }

    pub(crate) fn sampled_sound_length(&self, id: u32) -> Option<u64> {
        self.simple_sound_lengths.get(&id).copied()
    }
}

/// Starts an engine and returns the handle that controls it
pub(crate) async fn start_engine(
    settings: Arc<ArcSwap<GameSettings>>,
    timekeeper: Arc<Timekeeper>,
    sampled_sounds: &[SampledSound],
    cache_manager: &mut CacheManager,
) -> Result<EngineHandle> {
    let mut wavs = FxHashMap::default();
    let mut wav_lengths = FxHashMap::default();
    for sound in sampled_sounds {
        let bytes = cache_manager
            .load_media_by_name(&sound.media_filename)
            .await?;
        let wav = WavReader::new(Cursor::new(bytes))?;
        let length_nanos = wav_len_nanos(&wav);

        if wavs.insert(sound.sound_id, wav).is_some() {
            log::warn!(
                "Duplicate sound with ID {}, name {}",
                sound.sound_id,
                sound.media_filename
            );
        };
        wav_lengths.insert(sound.sound_id, length_nanos);
    }

    let control = Arc::new(SharedControl::new(settings.clone(), Vector3::zero()));
    let cancellation_token = tokio_util::sync::CancellationToken::new();
    if settings.load().audio.enable_audio {
        let control_clone = control.clone();
        let (tx, rx) = tokio::sync::oneshot::channel::<Result<()>>();
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

            let handle = tokio::runtime::Handle::current();
            let mut engine_state =
                EngineState::new(control_clone, selected_config.clone(), timekeeper, wavs)?;

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
                if let Err(e) = stream.pause() {
                    log::warn!("Failure pausing audio stream: {:?}", e)
                }
                drop(stream);
            });
            Ok::<(), anyhow::Error>(())
        })?;
        rx.await
            .context("Waiting for audio engine startup failed")?
            .context("Audio engine startup")?;
        log::info!("Audio engine started successfully");
    } else {
        log::info!("Returning an empty audio engine because audio was disabled");
    }

    Ok(EngineHandle {
        control,
        drop_guard: cancellation_token.drop_guard(),
        allocator_state: Mutex::new(AllocatorState::new()),
        simple_sound_lengths: wav_lengths,
    })
}

fn wav_len_nanos<R: std::io::Read>(wav: &WavReader<R>) -> u64 {
    let samples = wav.duration() as f64;
    let rate = wav.spec().sample_rate as f64;
    let length_nanos = (1_000_000_000.0 * samples / rate) as u64;
    length_nanos
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

    rng: rand::rngs::SmallRng,

    entity_filter_state: EntityFilterState,
    entity_filter_left: IirLpfCascade<8>,
    entity_filter_right: IirLpfCascade<8>,
    entity_filter_input_buffer: Vec<f32>,

    // HashMap: While the server has the IDs as consecutive, this isn't a protocol guarantee
    samplers: FxHashMap<u32, PrerecordedSampler>,
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
        if !self.control.enabled.load(Ordering::Acquire) {
            return;
        }

        let player_state = self.control.player_state.read();

        if player_state.entity_filter_state != self.entity_filter_state {
            self.entity_filter_state = player_state.entity_filter_state;
            let degree = self.entity_filter_state.degree;
            // Recreate the entity filter
            self.entity_filter_left = if degree == 0 {
                IirLpfCascade::default()
            } else {
                IirLpfCascade::new_for_degree(
                    self.entity_filter_state.degree as usize,
                    self.entity_filter_state.cutoff_hz as f32,
                    (1_000_000_000.0 / self.nanos_per_sample) as f32,
                )
            };
            self.entity_filter_right = self.entity_filter_left;
        }
        self.entity_filter_input_buffer.clear();
        self.entity_filter_input_buffer.resize(data.len(), 0.0);

        let volumes = self.control.settings.load().audio.volumes;

        'slots: for i in 0..NUM_SIMPLE_SOUND_SLOTS {
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

                    // let remaining_samples = (samples.saturating_sub(start_offset_samples)) as u64;
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

            let sampler = match self.samplers.get(&control_block.id) {
                Some(x) => x,
                None => {
                    continue 'slots;
                }
            };
            let buf = if control_block.flags & SOUND_BYPASS_ATTACHED_ENTITY_FILTER == 0 {
                self.entity_filter_input_buffer.deref_mut()
            } else {
                &mut *data
            };

            Self::sample_simple_sound(
                self.stream_config.channels() as usize,
                self.nanos_per_sample,
                buf,
                sample_range,
                &player_state,
                &control_block,
                private_state,
                eval_start_tick,
                sampler,
                volumes.volume_for(control_block.source),
            );
        }
        for i in 0..NUM_PROCEDURAL_ENTITY_SLOTS {
            let control_block = self.control.entity_slots[i].read();
            if control_block.flags & SOUND_PRESENT == 0 {
                continue;
            }

            let private_state = &mut self.private_control.procedural_entity_sounds[i];

            let buf = if control_block.flags & SOUND_BYPASS_ATTACHED_ENTITY_FILTER == 0 {
                self.entity_filter_input_buffer.deref_mut()
            } else {
                &mut *data
            };

            Self::sample_entity_turbulence(
                self.stream_config.channels() as usize,
                self.nanos_per_sample,
                buf,
                control_block,
                &player_state,
                private_state,
                &mut self.rng,
                buffer_start_tick,
                samples,
                volumes.volume_for(control_block.sound_source),
            )
        }
        match self.stream_config.channels() {
            1 => self.downmix_mono(data, player_state.entity_filter_extra_gain),
            2 => self.downmix_stereo(data, player_state.entity_filter_extra_gain),
            // TODO handle the messed-up sample rate here
            _ => self.downmix_mono(data, player_state.entity_filter_extra_gain),
        }
    }

    fn new(
        control: Arc<SharedControl>,
        stream_config: cpal::SupportedStreamConfig,
        timekeeper: Arc<Timekeeper>,
        sampled_sounds: FxHashMap<u32, WavReader<Cursor<Vec<u8>>>>,
    ) -> Result<EngineState> {
        let mut samplers = FxHashMap::default();
        let mut resampler_cache = FxHashMap::default();
        for (k, v) in sampled_sounds {
            println!("{}", k);
            samplers.insert(
                k,
                PrerecordedSampler::from_wav(
                    v,
                    stream_config.sample_rate().0,
                    &mut resampler_cache,
                )?,
            );
        }

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

            rng: rand::rngs::SmallRng::from_rng(&mut rand::thread_rng())?,
            entity_filter_state: EntityFilterState {
                cutoff_hz: 0,
                degree: 0,
            },
            entity_filter_left: Default::default(),
            entity_filter_right: Default::default(),
            // Generously allocate memory for one second
            entity_filter_input_buffer: Vec::with_capacity(
                stream_config.sample_rate().0 as usize * 2,
            ),
            samplers,
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
        volume_multiplier: f32,
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
                ((distance / SPEED_OF_SOUND_METER_PER_SECOND) * 1_000_000_000.0) as i64
            } else {
                0
            };
            // Design note: We're pretty lax with possible i64 overflows. This is fine, since this
            // happens at around 292 years of uptime. If we wanted to get the full 584 years of u64,
            // we could do i128 math, at the slight expense of performance for all other reasonable
            // uses
            let effective_time_delta = tick_delta as i64 - travel_time;
            if effective_time_delta < 0 {
                // Sound not yet started
                continue;
            }
            if effective_time_delta as u64 > control.end_tick - control.start_tick {
                // Sound finished
                break;
            }
            let effective_time_delta = effective_time_delta as u64 % sampler.len_nanos;

            let effective_time_delta = effective_time_delta as f64 / 1_000_000_000.0;

            let amplitude = if control.flags & SOUND_SQUARELAW_ENABLED != 0 {
                let clamped_distance = distance.max(MIN_DISTANCE) as f32;
                // this affects amplitude, not power, so it should be non-squared
                // However, using something like 1.5th power sounds better in-game (although is
                // nonphysical)
                // Worth seeing if sqrt and float mul is indeed faster than powf
                volume_multiplier * control.volume / (clamped_distance * clamped_distance.sqrt())
            } else {
                volume_multiplier * control.volume
            };

            let balance_left = left_ear_vec.dot((position - control.position).normalize());
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
        control_block: ProceduralEntitySoundControlBlock,
        player_state: &PlayerState,
        scratchpad: &mut EntityScratchpad,
        rng: &mut SmallRng,
        buffer_start_tick: u64,
        samples: usize,
        volume_multiplier: f32,
    ) {
        let left_ear_azimuth = player_state.azimuth_radian + std::f64::consts::FRAC_PI_2;
        let left_ear_z = left_ear_azimuth.cos();
        let left_ear_x = left_ear_azimuth.sin();
        let left_ear_vec = Vector3::new(left_ear_x, 0.0, left_ear_z);

        if control_block.entity_id != scratchpad.entity_id {
            *scratchpad = EntityScratchpad {
                entity_id: control_block.entity_id,
                turbulence_iir_state: IirLpfCascade::new_equal_cascade(
                    control_block.turbulence.lpf_cutoff_hz,
                    (1_000_000_000.0 / nanos_per_sample) as f32,
                ),
                ..EntityScratchpad::default()
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
            let current_velocity = entity_move.instantaneous_velocity(entity_move_elapsed);

            let (distance_falloff_multiplier, edge_multiplier, balance_left) = if control_block
                .flags
                & SOUND_ENTITY_SPATIAL
                != 0
            {
                let entity_pos = entity_move.qproj(entity_move_elapsed);
                let player_pos = player_state.position;
                // distance this sample
                let distance = (player_pos - entity_pos).magnitude();
                // distance in this sample if the player hadn't moved
                let distance_unmoved_player = (scratchpad.last_player_pos - entity_pos).magnitude();
                let distance_diff = distance_unmoved_player - scratchpad.last_distance;

                scratchpad.last_distance = distance;
                scratchpad.last_player_pos = player_pos;

                scratchpad.approach_state = match scratchpad.approach_state {
                    ApproachState::Initial(acc) => {
                        // Looking for a decrease
                        let new_acc = (acc - distance_diff).max(0.0);
                        if new_acc > 10.0 {
                            ApproachState::DecreaseDetected(0.0)
                        } else {
                            ApproachState::Initial(new_acc)
                        }
                    }
                    ApproachState::DecreaseDetected(acc) => {
                        let new_acc = (acc + distance_diff).max(0.0);
                        if new_acc > 0.1 {
                            scratchpad.edge_cycle = 0.0;
                            // This is a local minimum
                            // Is it close enough?
                            if distance < (2.0 * control_block.entity_len as f64) {
                                ApproachState::Passing(
                                    0.0,
                                    entity_pos,
                                    player_pos - entity_pos,
                                    entity_move
                                        .instantaneous_velocity(entity_move_elapsed)
                                        .normalize(),
                                )
                            } else {
                                // Nope, we're probably getting some artifact far away, and we might
                                // end up stuck in Passing for a while until it actually passes by
                                ApproachState::Initial(0.0)
                            }
                        } else {
                            ApproachState::DecreaseDetected(new_acc)
                        }
                    }
                    x => x,
                };

                let (effective_displacement, start_recovery) = match &mut scratchpad.approach_state
                {
                    ApproachState::Initial(_) => (player_pos - entity_pos, false),
                    ApproachState::DecreaseDetected(_) => (player_pos - entity_pos, false),
                    ApproachState::Passing(
                        travel_since,
                        captured_entity_location,
                        _captured_e2p,
                        captured_velocity,
                    ) => {
                        let additional_travel = (current_velocity.magnitude() as f64)
                            * nanos_per_sample
                            / 1_000_000_000.0;

                        let new_travel = *travel_since + additional_travel;
                        if (new_travel / 32.0).floor() != (*travel_since / 32.0).floor() {
                            scratchpad.edge_cycle = 0.0;
                        }

                        *travel_since = new_travel;
                        let player_displacement = player_pos - *captured_entity_location;
                        let ent_displacement = entity_pos - *captured_entity_location;
                        let captured_velocity = captured_velocity.cast().unwrap();
                        let parallel = player_displacement.project_on(captured_velocity);
                        let perpendicular = player_displacement - parallel;

                        // How far the player has moved *along* with the entity
                        let parallel_displacement = player_displacement.dot(captured_velocity);
                        let parallel_met = *travel_since
                            > (control_block.entity_len as f64 + parallel_displacement);

                        // Heuristic/hacky solution
                        let entity_own_parallel = ent_displacement.project_on(captured_velocity);
                        let entity_own_perpendicular = ent_displacement - entity_own_parallel;
                        let perpendicular_met =
                            entity_own_perpendicular.magnitude() > control_block.entity_len as f64;

                        (perpendicular, parallel_met | perpendicular_met)
                    }
                };
                // max is a hacky fix for spurious sounds; TODO fix it properly or rewrite this mess
                let effective_distance = effective_displacement
                    .magnitude()
                    .max(distance - control_block.entity_len as f64);

                let clamped_distance = effective_distance.max(MIN_ENTITY_DISTANCE) as f32;
                // multiplier affects amplitude, not power
                let distance_falloff_multiplier = 1.0 / (clamped_distance);
                if start_recovery {
                    scratchpad.approach_state = ApproachState::Initial(0.0);
                    scratchpad.trailing_edge_blend_factor = 1.0;
                    scratchpad.trailing_edge_blend_value = distance_falloff_multiplier;
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

                let distance_falloff_multiplier = scratchpad.trailing_edge_blend_value
                    * scratchpad.trailing_edge_blend_factor
                    + distance_falloff_multiplier * (1.0 - scratchpad.trailing_edge_blend_factor);
                scratchpad.trailing_edge_blend_factor = (scratchpad.trailing_edge_blend_factor
                    - nanos_per_sample as f32 / 2_000_000_000.0)
                    .clamp(0.0, 1.0);

                let effective_displacement_smoothed = scratchpad
                    .trailing_edge_blend_source_direction
                    * scratchpad.trailing_edge_blend_factor as f64
                    + effective_displacement.normalize()
                        * (1.0 - scratchpad.trailing_edge_blend_factor as f64);
                let effective_dir = effective_displacement_smoothed.normalize();
                let mut balance_left = left_ear_vec.dot(effective_dir);
                if balance_left.is_nan() {
                    balance_left = 0.0;
                }
                let edge_strength = MIN_ENTITY_DISTANCE as f32 / clamped_distance;
                let edge_multiplier = (edge_multiplier * edge_strength) + (1.0 - edge_strength);
                (distance_falloff_multiplier, edge_multiplier, balance_left)
            } else {
                (1.0, 1.0, 0.0)
            };

            // Theoretically should be N^5 or N^6, but this is subjectively better
            let speed_multiplier = (current_velocity.magnitude() / 70.0).powi(4);

            let turbulence_amplitude = control_block.turbulence.volume
                * distance_falloff_multiplier
                * edge_multiplier
                * speed_multiplier
                * volume_multiplier;

            let balance_left = scratchpad.last_balance.update(balance_left, 0.001);
            let l_amplitude = turbulence_amplitude * (1.0 + 0.8 * balance_left as f32);

            let r_amplitude = turbulence_amplitude * (1.0 - 0.8 * balance_left as f32);
            let (l, r) = sample_turbulence(scratchpad, rng);
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
                    turbulence_amplitude
                };
                let index = sample * channels + chan;
                data[index] += amplitude * sample_value / 800.0;
            }
        }
    }
    fn downmix_mono(&mut self, dst: &mut [f32], gain: f32) {
        assert_eq!(self.entity_filter_input_buffer.len(), dst.len());
        for (in_sample, out_sample) in self.entity_filter_input_buffer.iter().zip(dst.iter_mut()) {
            *out_sample = gain * self.entity_filter_left.sample(*in_sample);
        }
    }
    fn downmix_stereo(&mut self, dst: &mut [f32], gain: f32) {
        assert_eq!(self.entity_filter_input_buffer.len(), dst.len());
        assert_eq!(self.entity_filter_input_buffer.len() % 2, 0);
        for (idx, (in_sample, out_sample)) in self
            .entity_filter_input_buffer
            .iter()
            .zip(dst.iter_mut())
            .enumerate()
        {
            if idx % 2 == 0 {
                *out_sample = gain * self.entity_filter_left.sample(*in_sample);
            }
        }
        for (idx, (in_sample, out_sample)) in self
            .entity_filter_input_buffer
            .iter()
            .zip(dst.iter_mut())
            .enumerate()
        {
            if idx % 2 == 1 {
                *out_sample = gain * self.entity_filter_right.sample(*in_sample);
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
    entity_filter_state: EntityFilterState,
    entity_filter_extra_gain: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EntityFilterState {
    cutoff_hz: u32,
    degree: u32,
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

pub const NUM_SIMPLE_SOUND_SLOTS: usize = 256;
pub const NUM_PROCEDURAL_ENTITY_SLOTS: usize = 64;

/// The private control state, used internally by the audio engine to track the state of
/// the player and sounds
struct PrivateControl {
    simple_sounds: Box<[SimpleSoundState; NUM_SIMPLE_SOUND_SLOTS]>,
    procedural_entity_sounds: Box<[EntityScratchpad; NUM_PROCEDURAL_ENTITY_SLOTS]>,
}
impl Default for PrivateControl {
    fn default() -> PrivateControl {
        PrivateControl {
            simple_sounds: Box::new([SimpleSoundState::default(); NUM_SIMPLE_SOUND_SLOTS]),
            procedural_entity_sounds: Box::new(
                [EntityScratchpad::default(); NUM_PROCEDURAL_ENTITY_SLOTS],
            ),
        }
    }
}

/// The shared control state, via which sounds can be controlled
pub(crate) struct SharedControl {
    enabled: AtomicBool,
    player_state: SeqLock<PlayerState>,
    simple_sounds: Box<[SeqLock<SimpleSoundControlBlock>; NUM_SIMPLE_SOUND_SLOTS]>,
    entity_slots: Box<[SeqLock<ProceduralEntitySoundControlBlock>; NUM_PROCEDURAL_ENTITY_SLOTS]>,
    settings: Arc<ArcSwap<GameSettings>>,
}
impl SharedControl {
    fn new(settings: Arc<ArcSwap<GameSettings>>, initial_position: Vector3<f64>) -> SharedControl {
        const SIMPLE_SOUND_CONST_INIT: SeqLock<SimpleSoundControlBlock> =
            SeqLock::new(SimpleSoundControlBlock::const_default());
        const PROCEDURAL_ENTITY_CONST_INIT: SeqLock<ProceduralEntitySoundControlBlock> =
            SeqLock::new(ProceduralEntitySoundControlBlock::const_default());
        SharedControl {
            enabled: AtomicBool::new(false),
            player_state: SeqLock::new(PlayerState {
                position: initial_position,
                velocity: Vector3::zero(),
                acceleration: Vector3::zero(),
                azimuth_radian: 0.0,
                position_timebase_tick: 0,
                entity_filter_state: EntityFilterState {
                    cutoff_hz: 0,
                    degree: 0,
                },
                entity_filter_extra_gain: 1.0,
            }),
            simple_sounds: Box::new([SIMPLE_SOUND_CONST_INIT; NUM_SIMPLE_SOUND_SLOTS]),
            entity_slots: Box::new([PROCEDURAL_ENTITY_CONST_INIT; NUM_PROCEDURAL_ENTITY_SLOTS]),
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
    pub flags: FlagsType,
    /// The position of the sound in 3D space. The sound is assumed to be stationary,
    /// but the player's motion may be considered for Doppler effect.
    pub position: Vector3<f64>,
    /// The volume of the sound. If this is set to 0.0, the sound is silenced.
    pub volume: f32,
    /// The starting tick of the sound
    pub start_tick: u64,
    /// The sound ID (per the media manager)
    pub id: u32,
    /// The ending tick of the sound. If the span from start_tick to end_tick is greater than
    /// the length of the sound, it will loop. u64::MAX is treated as effectively infinite.
    ///
    /// Skew policy: If the tick clock and codec sample clock go out of skew, the sound will stop
    /// based on TBD
    pub end_tick: u64,
    /// The source to attribute the sound to
    pub source: SoundSource,
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
            source: SoundSource::SoundsourceUnspecified,
        }
    }

    /// Heuristically identify which sounds are most important. Higher values are considered more
    /// important.
    pub(crate) fn compute_score(&self, tick_now: u64, player_position: Vector3<f64>) -> u64 {
        if self.flags & SOUND_PRESENT == 0 {
            return 0;
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
        // Check if the sound is over first; we want sticky sounds to still get cleaned up, just not
        // while they're still running.
        if estimated_end_tick < tick_now {
            return 0;
        }
        if self.flags & SOUND_STICKY != 0 {
            return u64::MAX;
        }
        const BASE_SCORE: u64 = 1 << 60;
        const BASE_SCORE_F64: f64 = BASE_SCORE as f64;
        return if self.flags & SOUND_SQUARELAW_ENABLED != 0 {
            let dropoff = distance.max(1.0);
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

#[derive(Clone, Copy, Debug)]
pub(crate) struct ProceduralEntitySoundControlBlock {
    pub(crate) flags: FlagsType,
    pub(crate) entity_id: u64,
    pub(crate) leading: EntityMove,
    pub(crate) second: Option<EntityMove>,
    pub(crate) entity_len: f32,
    pub(crate) turbulence: TurbulenceSourceControlBlock,
    pub(crate) sound_source: SoundSource,
}

impl ProceduralEntitySoundControlBlock {
    pub(crate) fn compute_score(&self, tick_now: u64, player_position: Vector3<f64>) -> u64 {
        if self.flags & SOUND_PRESENT == 0 {
            return 0;
        }
        // TODO: What does sticky mean for an entity that exhausts its moves?
        if self.flags & SOUND_STICKY != 0 {
            return u64::MAX;
        }

        let (overflow, position) = match self.leading.elapsed_or_overflow(tick_now) {
            ElapsedOrOverflow::ElapsedThisMove(t) => (0.0, self.leading.qproj(t)),
            ElapsedOrOverflow::OverflowIntoNext(t) => match self.second {
                None => (t, self.leading.qproj(self.leading.total_time_sec())),
                Some(mv) => match mv.elapsed_or_overflow(tick_now) {
                    ElapsedOrOverflow::ElapsedThisMove(t) => (0.0, mv.qproj(t)),
                    ElapsedOrOverflow::OverflowIntoNext(t) => (t, mv.qproj(mv.total_time_sec())),
                },
            },
        };
        let distance = (position - player_position).magnitude();

        const BASE_SCORE: u64 = 1 << 60;
        const BASE_SCORE_F64: f64 = BASE_SCORE as f64;
        let mut distance_score = if self.flags & SOUND_SQUARELAW_ENABLED != 0 {
            let dropoff = distance.max(1.0);
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

        let mut vague_speed_estimate = self
            .leading
            .instantaneous_velocity(0.0)
            .magnitude()
            .max(
                self.leading
                    .instantaneous_velocity(self.leading.total_time_sec() / 2.0)
                    .magnitude(),
            )
            .max(
                self.leading
                    .instantaneous_velocity(self.leading.total_time_sec())
                    .magnitude(),
            );
        if let Some(second) = self.second {
            vague_speed_estimate = vague_speed_estimate
                .max(second.instantaneous_velocity(0.0).magnitude())
                .max(
                    second
                        .instantaneous_velocity(second.total_time_sec() / 2.0)
                        .magnitude(),
                )
                .max(
                    second
                        .instantaneous_velocity(second.total_time_sec())
                        .magnitude(),
                )
        }

        let mut volume_estimate = 0.0;
        volume_estimate += self.turbulence.volume * (vague_speed_estimate / 70.0).powi(4);

        if volume_estimate < 0.001 {
            distance_score /= 10;
        }

        let overflow_demerit = (overflow as f64 / 10.0) * BASE_SCORE_F64;
        return distance_score.saturating_sub(overflow_demerit as u64);
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct TurbulenceSourceControlBlock {
    pub(crate) volume: f32,
    pub(crate) lpf_cutoff_hz: f32,
}

impl ProceduralEntitySoundControlBlock {
    const fn const_default() -> ProceduralEntitySoundControlBlock {
        ProceduralEntitySoundControlBlock {
            flags: 0,
            entity_id: 0,
            leading: EntityMove::zero(),
            second: None,
            entity_len: 0.0,
            turbulence: TurbulenceSourceControlBlock {
                volume: 0.0,
                lpf_cutoff_hz: 1000.0,
            },
            sound_source: SoundSource::SoundsourceUnspecified,
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

pub type FlagsType = u8;

/// If set, the entry in the control block is valid and enabled
pub const SOUND_PRESENT: FlagsType = 0x1;

/// If set, the sound will be subject to speed-of-sound and doppler effects
/// Applicable for simple sounds
pub const SOUND_MOVESPEED_ENABLED: FlagsType = 0x2;
/// If set, the sound will be subject to square-law effects affecting its amplitude
/// Applicable for simple sounds
pub const SOUND_SQUARELAW_ENABLED: FlagsType = 0x4;
/// If set, the sound undergoes directionality effects
/// Applicable for simple sounds
pub const SOUND_DIRECTIONAL: FlagsType = 0x8;

/// If set, the entity sound has spatial effects
/// This should not be set for an entity to which the player is attached.
pub const SOUND_ENTITY_SPATIAL: FlagsType = 0x2;

/// If set, the sound is considered sticky, and will never be deallocated
pub const SOUND_STICKY: FlagsType = 0x10;
/// If set, the sound bypasses the attached entity filter. Otherwise, the attached entity filter
/// is applied.
pub const SOUND_BYPASS_ATTACHED_ENTITY_FILTER: FlagsType = 0x20;

/// The minimum distance considered for square-law effects. By enforcing this, we avoid
/// excessive amplitudes for near-zero distances.
pub const MIN_DISTANCE: f64 = 1.0;
pub const MIN_ENTITY_DISTANCE: f64 = 5.0;

pub const SPEED_OF_SOUND_METER_PER_SECOND: f64 = 343.0;

const OVERSAMPLE_FACTOR: u32 = 1;

struct PrerecordedSampler {
    len_nanos: u64,
    data: Vec<(f32, f32)>,
}
impl PrerecordedSampler {
    fn from_wav<R: std::io::Read>(
        wav: WavReader<R>,
        target_sample_rate: u32,
        resampler_cache: &mut FxHashMap<(u32, u32), FftFixedInOut<f32>>,
    ) -> Result<Self> {
        let len_nanos = wav_len_nanos(&wav);
        let source_rate = wav.spec().sample_rate;
        let source_channels = wav.spec().channels as usize;
        let internal_target_rate = target_sample_rate * OVERSAMPLE_FACTOR;
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

        let mut resampler = match resampler_cache.entry((source_rate, internal_target_rate)) {
            Entry::Occupied(mut v) => {
                v.get_mut().reset();
                v.into_mut()
            }
            Entry::Vacant(v) => v.insert(rubato::FftFixedInOut::new(
                source_rate as usize,
                internal_target_rate as usize,
                CHUNK_SIZE,
                2,
            )?),
        };
        let output_delay = resampler.output_delay();
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
            .skip(output_delay)
            .map(|(l, r)| (l, r))
            .collect();
        Ok(Self { len_nanos, data })
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
    pub fn len_nanos(&self) -> u64 {
        self.len_nanos
    }
}

#[inline]
fn sample_turbulence(scratchpad: &mut EntityScratchpad, rng: &mut SmallRng) -> (f32, f32) {
    let rng_val = rng.gen_range(-1.0..1.0) * 4000.0;
    let filtered = scratchpad.turbulence_iir_state.sample(rng_val);
    (filtered, filtered)
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
struct EntityScratchpad {
    // Really simple first-order IIR filter, single state
    turbulence_iir_state: IirLpfCascade<1>,
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
    edge_strength: f32,
}
impl Default for EntityScratchpad {
    fn default() -> Self {
        EntityScratchpad {
            turbulence_iir_state: IirLpfCascade::default(),
            entity_id: u64::MAX,
            last_player_pos: Vector3::zero(),
            last_distance: 0.0,
            approach_state: ApproachState::Initial(0.0),
            last_balance: Default::default(),
            trailing_edge_blend_value: 0.0,
            trailing_edge_blend_factor: 0.0,
            trailing_edge_blend_source_direction: Vector3::zero(),
            edge_cycle: -1.0,
            edge_strength: 0.0,
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

/// A cascade of *single-pole* filters.
#[derive(Clone, Copy, Debug)]
struct IirLpfCascade<const N: usize> {
    gains: [f32; N],
    states: [f32; N],
}
impl<const N: usize> IirLpfCascade<N> {
    fn new_equal_cascade(cutoff_freq: f32, sample_rate: f32) -> Self {
        let decay = f32::exp(-std::f32::consts::TAU * (cutoff_freq / sample_rate));
        let gain = 1.0 - decay;
        IirLpfCascade {
            gains: [gain; N],
            states: [0.0; N],
        }
    }

    fn new_for_degree(degree: usize, cutoff_freq: f32, sample_rate: f32) -> Self {
        let decay = f32::exp(-std::f32::consts::TAU * (cutoff_freq / sample_rate));
        let gain = 1.0 - decay;
        let mut gains = [1.0; N];
        for i in 0..degree {
            gains[i] = gain;
        }
        IirLpfCascade {
            gains,
            states: [0.0; N],
        }
    }

    fn sample(&mut self, input: f32) -> f32 {
        let mut val = input;
        for i in 0..N {
            self.states[i] += self.gains[i] * (val - self.states[i]);
            val = self.states[i];
        }
        val
    }
}
impl<const N: usize> Default for IirLpfCascade<N> {
    fn default() -> Self {
        IirLpfCascade {
            gains: [1.0; N],
            states: [0.0; N],
        }
    }
}

struct MapSoundEmitter {
    token: Option<SimpleSoundToken>,
    sound_id: u32,
    volume: f32,
}

pub(crate) struct MapSoundState {
    emitters: FxHashMap<BlockCoordinate, MapSoundEmitter>,
    engine: Arc<EngineHandle>,
}
impl MapSoundState {
    pub(crate) fn new(engine: Arc<EngineHandle>) -> Self {
        Self {
            emitters: FxHashMap::default(),
            engine,
        }
    }

    fn split_borrow(
        &mut self,
    ) -> (
        &mut FxHashMap<BlockCoordinate, MapSoundEmitter>,
        &EngineHandle,
    ) {
        (&mut self.emitters, &self.engine)
    }

    fn make_control_block(
        coord: BlockCoordinate,
        tick_now: u64,
        sound_id: u32,
        volume: f32,
    ) -> SimpleSoundControlBlock {
        SimpleSoundControlBlock {
            flags: SOUND_PRESENT
                | SOUND_MOVESPEED_ENABLED
                | SOUND_SQUARELAW_ENABLED
                | SOUND_DIRECTIONAL,
            position: coord.into(),
            volume,
            start_tick: tick_now,
            id: sound_id,
            end_tick: u64::MAX,
            source: SoundSource::SoundsourceWorld,
        }
    }

    pub(crate) fn insert_or_update(
        &mut self,
        tick_now: u64,
        player_pos: Vector3<f64>,
        coord: BlockCoordinate,
        sound_id: u32,
        volume: f32,
    ) {
        let control_block = Self::make_control_block(coord, tick_now, sound_id, volume);
        let mut entry = self.emitters.entry(coord);
        match entry {
            Entry::Occupied(e) => {
                let v = e.into_mut();
                {
                    v.token = self.engine.insert_or_update_simple_sound(
                        tick_now,
                        player_pos,
                        control_block,
                        v.token,
                    );
                    v.sound_id = sound_id;
                    v.volume = volume;
                }
            }
            Entry::Vacant(v) => {
                v.insert(MapSoundEmitter {
                    token: self.engine.insert_or_update_simple_sound(
                        tick_now,
                        player_pos,
                        control_block,
                        None,
                    ),
                    sound_id,
                    volume,
                });
            }
        };
    }

    pub(crate) fn remove(&mut self, coord: BlockCoordinate) {
        if let Some(v) = self.emitters.remove(&coord) {
            if let Some(tok) = v.token {
                self.engine.remove_simple_sound(tok);
            }
        }
    }

    pub(crate) fn remove_chunk(&mut self, chunk: ChunkCoordinate) {
        let mut to_remove: SmallVec<[BlockCoordinate; 16]> = smallvec::SmallVec::new();
        for k in self.emitters.keys() {
            if k.chunk() == chunk {
                to_remove.push(*k);
            }
        }
        for k in to_remove {
            self.remove(k)
        }
    }

    fn heal(
        engine: &EngineHandle,
        tick_now: u64,
        player_pos: Vector3<f64>,
        k: BlockCoordinate,
        v: &mut MapSoundEmitter,
    ) -> HealResult {
        if let Some(tok) = v.token {
            if !engine.is_token_evicted(tok) {
                return HealResult::NoAction;
            }
        }
        let control = Self::make_control_block(k, tick_now, v.sound_id, v.volume);
        v.token = engine.insert_or_update_simple_sound(tick_now, player_pos, control, v.token);
        if v.token.is_some() {
            HealResult::Healed
        } else {
            HealResult::Unhealed
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum HealResult {
    NoAction,
    Healed,
    Unhealed,
}

pub(crate) struct EvictedAudioHealer {
    client_state: Arc<ClientState>,
    shutdown: CancellationToken,
}
impl EvictedAudioHealer {
    pub(crate) fn new(
        client_state: Arc<ClientState>,
    ) -> (Arc<Self>, tokio::task::JoinHandle<Result<()>>) {
        let worker = Arc::new(Self {
            shutdown: client_state.shutdown.clone(),
            client_state,
        });
        let handle = {
            let worker_clone = worker.clone();
            tokio::task::spawn(worker_clone.run_healing_loop())
        };
        (worker, handle)
    }

    pub(crate) async fn run_healing_loop(self: Arc<Self>) -> Result<()> {
        tracy_client::set_thread_name!("world_audio_healer");
        while !self.shutdown.is_cancelled() {
            let sleep_deadline = std::time::Instant::now() + Duration::from_secs(1);
            tokio::select! {
                _ = tokio::time::sleep_until(tokio::time::Instant::from(sleep_deadline)) => {
                    // pass
                },
                _ = self.shutdown.cancelled() => {
                    break;
                }
            };
            tokio::task::block_in_place(|| {
                // Consider pre-allocating if malloc becomes a pain
                let mut heal_vec = vec![];
                let pos = self.client_state.weakly_ordered_last_position().position;
                let tick_now = self.client_state.timekeeper.now();
                {
                    let mut lock = self.client_state.world_audio.lock();

                    let (emitters, engine) = lock.split_borrow();

                    heal_vec.extend(emitters.iter_mut());
                    heal_vec.sort_by_key(|(k, _)| {
                        (Vector3::<f64>::from(**k) - pos).magnitude2() as u64
                    });
                    for (&k, v) in heal_vec.into_iter() {
                        match MapSoundState::heal(engine, tick_now, pos, k, v) {
                            HealResult::NoAction => continue,
                            HealResult::Healed => continue,
                            HealResult::Unhealed => break,
                        }
                    }
                }
            });
        }
        log::info!("Spatial audio heal loop exiting");

        Ok(())
    }
}
