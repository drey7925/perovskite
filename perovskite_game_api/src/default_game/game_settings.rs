use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct GameSettings {
    /// Users that are eligible for all permissions
    pub super_users: Vec<String>,
}

pub const FILENAME: &str = "settings.ron";

impl Default for GameSettings {
    fn default() -> Self {
        Self {
            super_users: Vec::new(),
        }
    }
}

pub(crate) fn load(data_dir: &PathBuf) -> Result<GameSettings> {
    let config_file = data_dir.join(FILENAME);
    log::info!("Loading settings from {}", config_file.display());
    if !config_file.exists() {
        log::info!("No settings found; using defaults");
        return Ok(Default::default());
    }
    let config = ron::from_str::<GameSettings>(&std::fs::read_to_string(&config_file)?).unwrap();
    log::info!("Loaded settings from {}", config_file.display());
    Ok(config)
}
