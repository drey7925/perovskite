use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct DefaultGameSettings {
    /// Users that are eligible for all permissions
    pub super_users: Vec<String>,
    pub spawn_location: (f64, f64, f64),
}

pub const FILENAME: &str = "settings.ron";

impl Default for DefaultGameSettings {
    fn default() -> Self {
        Self {
            super_users: Vec::new(),
            // arbitrary, and not always meaningful
            // todo ask the mapgen?
            spawn_location: (5.0, 10.0, 5.0),
        }
    }
}

pub(crate) fn load(data_dir: &Path) -> Result<DefaultGameSettings> {
    let config_file = data_dir.join(FILENAME);
    log::info!("Loading settings from {}", config_file.display());
    if !config_file.exists() {
        log::info!("No settings found; using defaults");
        return Ok(Default::default());
    }
    let config =
        ron::from_str::<DefaultGameSettings>(&std::fs::read_to_string(&config_file)?).unwrap();
    log::info!("Loaded settings from {}", config_file.display());
    Ok(config)
}
