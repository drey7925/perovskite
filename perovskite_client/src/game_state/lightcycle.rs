use cgmath::{Vector3, Deg, Angle};
use perovskite_core::time::TimeState;
use splines::{Interpolation, Key, Spline};

pub(crate) struct LightCycle {
    sky_color: Spline<f32, Vector3<f32>>,
    light_color: Spline<f32, Vector3<f32>>,
    pub(crate) time_state: TimeState,
}
impl LightCycle {
    pub(crate) fn new(time_state: TimeState) -> Self {
        Self {
            sky_color: Spline::from_vec(vec![
                // Midnight - allow some wiggle room
                Key::new(-0.1, Vector3::new(0.0, 0.0, 0.0), Interpolation::Linear),
                // Dawn starts
                Key::new(0.21, Vector3::new(0.0, 0.0, 0.0), Interpolation::Linear),
                // Bright orange dawn sky
                Key::new(0.25, Vector3::new(0.9, 0.25, 0.0), Interpolation::Linear),
                // Sunrise
                Key::new(0.28, Vector3::new(0.5, 0.9, 0.5), Interpolation::Linear),
                // Midday
                Key::new(0.5, Vector3::new(0.25, 0.6, 1.0), Interpolation::Linear),
                // Sunset
                Key::new(0.71, Vector3::new(0.5, 0.9, 0.5), Interpolation::Linear),
                // Bright orange sunset sky
                Key::new(0.75, Vector3::new(0.9, 0.25, 0.0), Interpolation::Linear),
                // Dusk ends
                Key::new(0.78, Vector3::new(0.0, 0.0, 0.0), Interpolation::Linear),
                // Night - allow some wiggle room
                Key::new(1.1, Vector3::new(0.0, 0.0, 0.0), Interpolation::Linear),
            ]),
            light_color: Spline::from_vec(vec![
                // Midnight - allow some wiggle room
                Key::new(-0.1, Vector3::new(0.0, 0.0, 0.0), Interpolation::Linear),
                // Dawn starts
                Key::new(0.21, Vector3::new(0.0, 0.0, 0.0), Interpolation::Linear),
                // Bright orange dawn sky
                Key::new(0.25, Vector3::new(0.9, 0.45, 0.4), Interpolation::Linear),
                // Sunrise
                Key::new(0.28, Vector3::new(0.9, 0.9, 0.7), Interpolation::Linear),
                // Midday
                Key::new(0.5, Vector3::new(0.9, 0.9, 1.0), Interpolation::Linear),
                // Sunset
                Key::new(0.72, Vector3::new(0.9, 0.9, 0.7), Interpolation::Linear),
                // Bright orange sunset sky
                Key::new(0.75, Vector3::new(0.9, 0.45, 0.4), Interpolation::Linear),
                // Dusk ends
                Key::new(0.79, Vector3::new(0.0, 0.0, 0.0), Interpolation::Linear),
                // Night - allow some wiggle room
                Key::new(1.1, Vector3::new(0.0, 0.0, 0.0), Interpolation::Linear),
            ]),
            time_state,
        }
    }
    pub(crate) fn get_sky_color(&self) -> Vector3<f32> {
        self.sky_color.clamped_sample(self.time_state.time_of_day() as f32).unwrap_or(Vector3::new(0.0, 0.0, 0.0))
    }
    pub(crate) fn get_light_color(&self) -> Vector3<f32> {
        self.light_color.clamped_sample(self.time_state.time_of_day() as f32).unwrap_or(Vector3::new(0.0, 0.0, 0.0))
    }
    pub(crate) fn get_light_direction(&self) -> Vector3<f32> {
        let sun_angle = Deg(360.) * (self.time_state.time_of_day() as f32);
        Vector3::new(
            // At midnight, 0. Moves toward the easy, then during the day, decreases from east to west
            sun_angle.sin(),
            // At midnight, 1.0 (down in vulkan)
            sun_angle.cos(),
            0.
        )
    }
    pub(crate) fn get_colors(&self) -> (Vector3<f32>, Vector3<f32>, Vector3<f32>) {
        (self.get_sky_color(), self.get_light_color(), self.get_light_direction())
    }

    pub(crate) fn time_state_mut(&mut self) -> &mut TimeState {
        &mut self.time_state
    }
}

