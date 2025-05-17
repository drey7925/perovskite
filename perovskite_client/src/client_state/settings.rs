use std::{fs::create_dir_all, path::PathBuf};

use super::input::KeybindSettings;
use crate::vulkan::shaders::LiveRenderConfig;
use anyhow::{Context, Result};
use perovskite_core::protocol::audio::SoundSource;
use serde::{Deserialize, Serialize};

const SETTINGS_RON_FILE: &str = "settings.ron";

#[derive(Clone, Serialize, Deserialize, Debug, Copy, PartialEq, Eq)]
pub(crate) enum Supersampling {
    None,
    X2,
    X4,
    X8,
}
impl Supersampling {
    pub(crate) fn to_int(self) -> u32 {
        match self {
            Supersampling::None => 1,
            Supersampling::X2 => 2,
            Supersampling::X4 => 4,
            Supersampling::X8 => 8,
        }
    }
    pub(crate) fn to_float(self) -> f32 {
        self.to_int() as f32
    }
    pub(crate) fn blit_steps(self) -> usize {
        match self {
            Supersampling::None => 0,
            Supersampling::X2 => 1,
            Supersampling::X4 => 2,
            Supersampling::X8 => 3,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(default)]
pub(crate) struct RenderSettings {
    pub(crate) num_mesh_workers: usize,
    pub(crate) num_neighbor_propagators: usize,
    pub(crate) show_placement_guide: bool,
    pub(crate) testonly_noop_meshing: bool,
    pub(crate) physics_debug: bool,
    pub(crate) preferred_gpu: String,
    pub(crate) scale_inventories_with_high_dpi: bool,
    pub(crate) fov_degrees: f64,
    pub(crate) supersampling: Supersampling,
    pub(crate) render_distance: u32,
    pub(crate) raytracing: bool,
    pub(crate) raytraced_reflections: bool,
}

impl RenderSettings {
    pub(crate) fn build_global_config(&self) -> LiveRenderConfig {
        LiveRenderConfig {
            supersampling: self.supersampling,
            raytracing: self.raytracing,
            raytracing_reflections: self.raytraced_reflections,
        }
    }
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            num_mesh_workers: 2,
            num_neighbor_propagators: 1,
            show_placement_guide: false,
            testonly_noop_meshing: false,
            physics_debug: false,
            preferred_gpu: String::from(""),
            scale_inventories_with_high_dpi: false,
            fov_degrees: 75.0,
            supersampling: Supersampling::X2,
            render_distance: 50,
            raytracing: false,
            raytraced_reflections: true,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(default)]
pub(crate) struct DisplaySettings {
    pub(crate) hover_text_on_bottom_panel: bool,
}

impl Default for DisplaySettings {
    fn default() -> Self {
        Self {
            hover_text_on_bottom_panel: false,
        }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize, Debug)]
#[serde(default)]
pub(crate) struct AudioVolumes {
    pub(crate) global_volume: f32,
    pub(crate) background_volume: f32,
    pub(crate) self_volume: f32,
    pub(crate) other_players_volume: f32,
    pub(crate) world_volume: f32,
}

impl AudioVolumes {
    pub(crate) fn volume_for(&self, source: SoundSource) -> f32 {
        let specific_volume = match source {
            SoundSource::SoundsourceUnspecified => 1.0,
            SoundSource::SoundsourceBackground => self.background_volume,
            SoundSource::SoundsourceSelf => self.self_volume,
            SoundSource::SoundsourcePlayer => self.other_players_volume,
            SoundSource::SoundsourceWorld => self.world_volume,
        };
        self.global_volume * specific_volume
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(default)]
pub(crate) struct AudioSettings {
    pub(crate) enable_audio: bool,
    pub(crate) volumes: AudioVolumes,
    pub(crate) preferred_output_device: String,
    // The detected audio devices
    #[serde(skip)]
    pub(crate) output_devices: Vec<String>,
}

impl AudioSettings {
    pub(crate) fn fill_audio_devices(&mut self) -> Result<()> {
        use cpal::traits::DeviceTrait;
        use cpal::traits::HostTrait;
        let host = cpal::default_host();

        self.output_devices = host
            .output_devices()?
            .map(|d| d.name().unwrap_or(String::from("")))
            .collect();
        Ok(())
    }
}

impl Default for AudioVolumes {
    fn default() -> Self {
        Self {
            global_volume: 1.0,
            background_volume: 1.0,
            self_volume: 1.0,
            other_players_volume: 1.0,
            world_volume: 1.0,
        }
    }
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            enable_audio: true,
            volumes: AudioVolumes::default(),
            preferred_output_device: String::from(""),
            output_devices: vec![],
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(default)]
pub(crate) struct DebugSettings {
    pub(crate) extra_entity_debug: bool,
}

impl Default for DebugSettings {
    fn default() -> Self {
        Self {
            extra_entity_debug: false,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
#[serde(default)]
pub(crate) struct GameSettings {
    pub(crate) input: KeybindSettings,
    pub(crate) display: DisplaySettings,
    pub(crate) render: RenderSettings,
    pub(crate) audio: AudioSettings,
    pub(crate) last_hostname: String,
    pub(crate) last_username: String,
    pub(crate) previous_servers: Vec<String>,
    pub(crate) debug: DebugSettings,
    // If set, the settings couldn't be parsed and shouldn't be saved
    #[serde(skip)]
    pub(crate) internal_parsing_failure_message: Option<String>,
}
impl GameSettings {
    pub(crate) fn save_to_disk(&self) -> Result<()> {
        if let Some(err) = self.internal_parsing_failure_message.as_ref() {
            log::warn!("Not saving settings because parsing failed: {}", err);
            return Ok(());
        }
        let project_dirs = project_dirs().context("couldn't find config dir")?;
        let config_dir = project_dirs.config_dir();
        if !config_dir.exists() {
            create_dir_all(config_dir)?;
        }
        let config_file = config_dir.join(SETTINGS_RON_FILE);
        let config = ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default()).unwrap();
        std::fs::write(&config_file, config)?;
        log::info!("Saved settings to {}", clean_path(config_file));
        Ok(())
    }

    pub(crate) fn load_from_disk() -> Result<Self> {
        let config_file = directories::ProjectDirs::from("foo", "drey7925", "perovskite")
            .context("couldn't find config dir")?
            .config_dir()
            .join(SETTINGS_RON_FILE);
        if !config_file.exists() {
            log::warn!("No settings found at {}", clean_path(config_file));
            return Ok(Default::default());
        }
        let config = std::fs::read_to_string(&config_file)?;
        log::info!("Loaded settings from {}", clean_path(config_file));
        let mut parsed: GameSettings = match ron::from_str(&config) {
            Ok(parsed) => parsed,
            Err(err) => {
                log::error!("Failed to parse settings: {}", err);

                return Ok(GameSettings {
                    internal_parsing_failure_message: Some(err.to_string()),
                    ..Default::default()
                });
            }
        };
        parsed.input = parsed.input.migrate_scancodes();
        Ok(parsed)
    }

    pub(crate) fn push_hostname(&mut self, hostname: String) {
        self.previous_servers.retain(|h| h != &hostname);
        self.previous_servers.push(hostname);
    }
}

pub(crate) fn project_dirs() -> Option<directories::ProjectDirs> {
    directories::ProjectDirs::from("foo", "drey7925", "perovskite")
}

/// Replaces the user's home directory with an environment variable.
/// Selfishly, I use this to avoid showing my username when publicly streaming
/// a development session that might show the client log.
pub fn clean_path(path: PathBuf) -> String {
    let user_dirs = directories::UserDirs::new();
    if let Some(user_dirs) = user_dirs {
        if cfg!(target_os = "windows") {
            path.strip_prefix(user_dirs.home_dir())
                .map(|p| format!("%UserProfile%\\{}", p.display()))
                .unwrap_or(path.display().to_string())
        } else if cfg!(target_os = "linux") {
            path.strip_prefix(user_dirs.home_dir())
                .map(|p| format!("$HOME/{}", p.display()))
                .unwrap_or(path.display().to_string())
        } else {
            path.display().to_string()
        }
    } else {
        path.display().to_string()
    }
}
