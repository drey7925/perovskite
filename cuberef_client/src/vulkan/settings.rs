use std::num::NonZeroU32;

use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(default)]
pub(crate) struct GraphicsSettings {
    pub(crate) supersampling: NonZeroU32
}
impl Default for GraphicsSettings {
    fn default() -> Self {
        Self {
            supersampling: NonZeroU32::try_from(1).unwrap(),
        }
    }
}