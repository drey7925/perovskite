use std::{fs::create_dir_all, path::PathBuf};

use super::input::KeybindSettings;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const SETTINGS_RON_FILE: &str = "settings.ron";


#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(default)]
pub(crate) struct RenderSettings {
    pub(crate) num_mesh_workers: usize,
    pub(crate) num_neighbor_propagators: usize,
    pub(crate) show_placement_guide: bool,
    pub(crate) testonly_noop_meshing: bool
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            num_mesh_workers: 2,
            num_neighbor_propagators: 1,
            show_placement_guide: false,
            testonly_noop_meshing: false
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
#[serde(default)]
pub(crate) struct GameSettings {
    pub(crate) input: KeybindSettings,
    pub(crate) render: RenderSettings,
    pub(crate) last_hostname: String,
    pub(crate) last_username: String,

}
impl GameSettings {
    pub(crate) fn save_to_disk(&self) -> Result<()> {
        let project_dirs = directories::ProjectDirs::from("foo", "drey7925", "perovskite")
            .context("couldn't find config dir")?;
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

    pub(crate) fn load_from_disk() -> Result<Option<Self>> {
        let config_file = directories::ProjectDirs::from("foo", "drey7925", "perovskite")
            .context("couldn't find config dir")?
            .config_dir()
            .join(SETTINGS_RON_FILE);
        if !config_file.exists() {
            log::warn!("No settings found at {}", clean_path(config_file));
            return Ok(None);
        }
        let config = std::fs::read_to_string(&config_file)?;
        log::info!("Loaded settings from {}", clean_path(config_file));
        let parsed = ron::from_str(&config)?;
        Ok(Some(parsed))
    }
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
