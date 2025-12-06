use std::{
    ops::{Deref, DerefMut},
    sync::atomic::{AtomicI64, Ordering},
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use tracy_client::plot;

//  server timebase             server time
//  |--------------[tick]------>|
//                              |--[initial net delay]->|
//                                                      when we observed [tick]
//                              |--[new net delay]-->|
//                                                   when we observe [tick] now
//                                                -->|  |<--error (negative)
//                              |--[new net delay]-------->|
//                                                         when we observe [tick] now
//                                   error (positive)-->|  |<--
struct TimekeeperInner {
    /// The actual error between the server's measured timebase and our own timebase.
    error: i64,
    /// The smoothed error that we have in current_skew.
    smoothed_error: i64,
    /// The last time we had a frame
    last_frame_time: Instant,
}

// One millsecond per second
const MAX_SLEW_UP: f64 = 0.001;
// 250 microseconds per second
const MAX_SLEW_DOWN: f64 = 0.00025;

#[repr(align(64))]
struct CachelineAligned<T>(T);
impl<T> Deref for CachelineAligned<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T> DerefMut for CachelineAligned<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Default)]
struct Skew {
    smoothed: AtomicI64,
    raw: AtomicI64,
}

pub(crate) struct Timekeeper {
    /// The initial estimate for what the server is using as its timebase,
    /// given in terms of our own system clock.
    ///
    /// Note that we can't just subtract the tick from the instant at which we start the timekeeper:
    /// On some systems like Windows, the actual value inside the Duration might underflow if the
    /// server has been up longer than the current system uptime.
    initial_timebase_estimate: Instant,
    /// The tick at which we observed initial_timebase_estimate.
    initial_timebase_tick: u64,
    inner: CachelineAligned<Mutex<TimekeeperInner>>,
    current_skew: CachelineAligned<Skew>,
}
impl Timekeeper {
    pub(crate) fn new(initial_tick: u64) -> Self {
        let now = Instant::now();

        Self {
            initial_timebase_estimate: now,
            initial_timebase_tick: initial_tick,
            inner: CachelineAligned(Mutex::new(TimekeeperInner {
                error: 0,
                smoothed_error: 0,
                last_frame_time: now,
            })),
            current_skew: CachelineAligned(Skew::default()),
        }
    }

    pub(crate) fn now(&self) -> u64 {
        self.time_to_tick_estimate(Instant::now())
    }

    pub(crate) fn update_error(&self, server_tick: u64) {
        if server_tick < self.initial_timebase_tick {
            panic!(
                "Server time ran backwards: {:?} vs {:?}",
                server_tick, self.initial_timebase_tick
            );
        }
        let now = Instant::now();
        let new_timebase_estimate =
            now - Duration::from_nanos(server_tick - self.initial_timebase_tick);
        let error = new_timebase_estimate - self.initial_timebase_estimate;
        self.inner.lock().error = error.as_nanos().try_into().unwrap();
    }

    pub(crate) fn update_frame(&self) {
        let mut lock = self.inner.lock();
        let now = Instant::now();
        let last_frame_nanos: i64 = (now - lock.last_frame_time).as_nanos().try_into().unwrap();
        let last_frame_nanos = last_frame_nanos as f64;
        plot!("lock_error", lock.error as f64);
        plot!("lock_sm_error", lock.smoothed_error as f64);

        if lock.error > lock.smoothed_error {
            // Error is positive. Therefore the server clock is ahead of where it was before
            // i.e.
            let diff_nanos = lock.error - lock.smoothed_error;
            let cap = (last_frame_nanos * MAX_SLEW_UP) as i64;
            let new_smoothed_error = lock.smoothed_error + std::cmp::min(diff_nanos, cap);

            lock.smoothed_error = new_smoothed_error;
            self.current_skew.raw.store(lock.error, Ordering::Relaxed);
            self.current_skew
                .smoothed
                .store(new_smoothed_error, Ordering::Relaxed);
        } else if lock.error < lock.smoothed_error {
            let diff_nanos = lock.smoothed_error - lock.error;
            let cap = (last_frame_nanos * MAX_SLEW_DOWN) as i64;
            let new_smoothed_error = lock.smoothed_error - std::cmp::min(diff_nanos, cap);
            lock.smoothed_error = new_smoothed_error;
            self.current_skew.raw.store(lock.error, Ordering::Relaxed);
            self.current_skew
                .smoothed
                .store(new_smoothed_error, Ordering::Relaxed);
        }
        lock.last_frame_time = now;
    }

    /// Returns the current timebase offset in nanoseconds.
    /// Positive means that the network delay has increased,
    /// so we should do things slightly earlier (where possible due to prefetching)
    /// to keep them in sync with the server.
    ///
    /// Negative means that the network delay has decreased, so we should do things slightly
    /// later (where possible due to available buffers/delays) to keep them in sync with the server.
    pub(crate) fn get_smoothed_offset(&self) -> i64 {
        self.current_skew.smoothed.load(Ordering::Relaxed)
    }
    pub(crate) fn get_raw_offset(&self) -> i64 {
        self.current_skew.raw.load(Ordering::Relaxed)
    }

    /// Adjusts the server tick based on the offset.
    pub(crate) fn adjust_server_tick(&self, server_tick: u64) -> u64 {
        (server_tick as i128 + self.get_smoothed_offset() as i128)
            .try_into()
            .unwrap()
    }

    #[allow(dead_code)]
    pub(crate) fn estimated_send_time(&self, server_tick: u64) -> Instant {
        // Note that ticks can run negative relative to each other, due to queueing in the server.
        // However, ticks should never run backwards compared to the first tick we received from the
        // server.
        if server_tick < self.initial_timebase_tick {
            panic!(
                "Server time ran backwards: {:?} vs {:?}",
                server_tick, self.initial_timebase_tick
            );
        }
        self.initial_timebase_estimate
            + Duration::from_nanos(
                self.adjust_server_tick(server_tick - self.initial_timebase_tick),
            )
    }

    pub(crate) fn time_to_tick_estimate(&self, time: Instant) -> u64 {
        let raw_nanos: i128 = (time - self.initial_timebase_estimate).as_nanos() as i128
            + self.initial_timebase_tick as i128;
        (raw_nanos - self.get_smoothed_offset() as i128) as u64
    }
}
