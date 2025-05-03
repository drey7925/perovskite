use anyhow::ensure;
use std::time::{Duration, Instant};

pub struct TimeState {
    realtime_start: Instant,
    game_time_start_days: f64,
    day_length: Duration,
}
impl TimeState {
    pub fn new(day_length: Duration, game_time_days: f64) -> Self {
        Self {
            realtime_start: Instant::now(),
            game_time_start_days: game_time_days,
            day_length,
        }
    }
    pub fn set_time(&mut self, game_time_days: f64) -> anyhow::Result<()> {
        self.realtime_start = Instant::now();
        ensure!(game_time_days.is_finite());
        self.game_time_start_days = game_time_days;
        Ok(())
    }
    pub fn set_day_length(&mut self, day_length: Duration) {
        self.day_length = day_length;
    }
    pub fn time_of_day(&self) -> f64 {
        (self.realtime_start.elapsed().as_secs_f64() / self.day_length.as_secs_f64()
            + self.game_time_start_days)
            % 1.0
    }
    pub fn day_length(&self) -> Duration {
        self.day_length
    }
}
