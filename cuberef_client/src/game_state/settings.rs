use std::fs::create_dir_all;

use super::input::KeybindSettings;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const SETTINGS_RON_FILE: &str = "settings.ron";

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
#[serde(default)]
pub(crate) struct GameSettings {
    pub(crate) input: KeybindSettings,
    pub(crate) last_hostname: String,
    pub(crate) last_username: String,
}
impl GameSettings {
    pub(crate) fn save_to_disk(&self) -> Result<()> {
        let project_dirs = directories::ProjectDirs::from("foo", "drey7925", "cuberef")
            .context("couldn't find config dir")?;
        print!("Saving settings to {}", project_dirs.config_dir().display());
        let config_dir = project_dirs.config_dir();
        if !config_dir.exists() {
            create_dir_all(config_dir)?;
        }
        let config_file = config_dir.join(SETTINGS_RON_FILE);
        let config = ron::ser::to_string_pretty(
            self, ron::ser::PrettyConfig::default()).unwrap();
        std::fs::write(&config_file, config)?;
        log::info!("Saved settings to {}", config_file.display());
        Ok(())
    }

    pub(crate) fn load_from_disk() -> Result<Option<Self>> {
        let config_file = directories::ProjectDirs::from("foo", "drey7925", "cuberef")
            .context("couldn't find config dir")?
            .config_dir()
            .join(SETTINGS_RON_FILE);
        if !config_file.exists() {
            return Ok(None);
        }
        let config = std::fs::read_to_string(&config_file)?;
        let parsed = ron::from_str(&config)?;
        Ok(Some(parsed))
    }
}
