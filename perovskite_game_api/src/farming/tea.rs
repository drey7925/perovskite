//! A module allowing players to grow, process, and enjoy various varieties of tea. Eventually, a
//! possible testbed for additional item functionality (e.g. in teapots and rare varieties of leaves
//! obtained by chance)
//!
//! Roadmap:
//! * [ ] Basic tea growth
//! * [ ] Basic tea processing (e.g., withering, oxidation, fixing, drying)
//! * [ ] ...

use crate::game_builder::{GameBuilder, FALLBACK_UNKNOWN_TEXTURE_NAME};
use anyhow::Result;

pub(crate) fn register_tea(builder: &mut GameBuilder) -> Result<()> {
    builder.register_basic_item(
        "farming:tea_leaves_fresh",
        "Fresh tea leaves",
        FALLBACK_UNKNOWN_TEXTURE_NAME,
        vec![],
        "farming:tea:leaves:fresh",
    )?;

    register_tea_plant_stages(builder)?;

    Ok(())
}

fn register_tea_plant_stages(builder: &mut GameBuilder) -> Result<()> {
    Ok(())
}
