use crate::blocks::{BlockBuilder, CubeAppearanceBuilder};
use crate::default_game::basic_blocks::{DIRT_TEXTURE, WATER};
use crate::farming::FarmingGameStateExtension;
use crate::game_builder::{GameBuilder, OwnedTextureName, StaticTextureName};
use crate::include_texture_bytes;
use anyhow::{Context, Result};
use perovskite_core::block_id::BlockId;
use perovskite_core::constants::item_groups::HIDDEN_FROM_CREATIVE;
use perovskite_core::coordinates::{ChunkCoordinate, ChunkOffset};
use perovskite_server::game_state::blocks::FastBlockName;
use perovskite_server::game_state::event::HandlerContext;
use perovskite_server::game_state::game_map::{
    BulkUpdateCallback, ChunkNeighbors, MapChunk, TimerCallback, TimerSettings, TimerState,
};
use std::time::Duration;

#[derive(Clone, Debug)]
struct SoilTimerCallback {
    dry: BlockId,
    wet: BlockId,
    water: FastBlockName,
    allow_downflow: bool,
}

impl BulkUpdateCallback for SoilTimerCallback {
    fn bulk_update_callback(
        &self,
        ctx: &HandlerContext<'_>,
        chunk_coordinate: ChunkCoordinate,
        _timer_state: &TimerState,
        chunk: &mut MapChunk,
        neighbors: Option<&ChunkNeighbors>,
        _lights: Option<&perovskite_core::lighting::LightScratchpad>,
    ) -> Result<()> {
        let neighbors = neighbors.context("Missing neighbors in soil timer callback")?;
        let water = ctx
            .block_types()
            .resolve_name(&self.water)
            .context("water block missing")?;
        for x in 0..16 {
            for z in 0..16 {
                for y in 0..16 {
                    let offset = ChunkOffset::new(x as u8, z as u8, y as u8);
                    let current_block = chunk.get_block(offset);
                    if current_block.equals_ignore_variant(self.dry)
                        || current_block.equals_ignore_variant(self.wet)
                    {
                        let base_coord = chunk_coordinate.with_offset(offset);
                        let mut best_variant = 0;

                        // Include 0, 0, 0 for slow decay of moisture
                        for dx in -1..=1 {
                            for dz in -1..=1 {
                                for dy in -1..=1 {
                                    if let Some(coord) = base_coord.try_delta(dx, dz, dy) {
                                        let block =
                                            neighbors.get_block(coord).unwrap_or(BlockId(0));
                                        if block.equals_ignore_variant(self.wet) {
                                            if self.allow_downflow && y == 1 {
                                                // Water cascades down paddies and other downflow-
                                                // capable types of soil
                                                best_variant = 3;
                                            } else {
                                                best_variant = best_variant
                                                    .max(block.variant().saturating_sub(1).min(4));
                                            }
                                        } else if block.equals_ignore_variant(water) {
                                            best_variant = 3;
                                        }
                                    }
                                }
                            }
                        }
                        let new_block = if best_variant == 0 {
                            self.dry
                        } else {
                            self.wet.with_variant_unchecked(best_variant)
                        };
                        chunk.set_block(offset, new_block, None);
                    }
                }
            }
        }
        Ok(())
    }
}

const TILLED_SOIL_TOP: StaticTextureName = StaticTextureName("farming:tilled_soil");
const TILLED_SOIL_SIDE: StaticTextureName = DIRT_TEXTURE;
const TILLED_SOIL_WET_TOP: StaticTextureName = StaticTextureName("farming:tilled_soil_wet");
const TILLED_SOIL_WET_SIDE: StaticTextureName = StaticTextureName("farming:tilled_soil_wet_side");

pub(super) fn register_soil_blocks(builder: &mut GameBuilder) -> Result<()> {
    include_texture_bytes!(builder, TILLED_SOIL_TOP, "textures/tilled_dirt.png")?;
    include_texture_bytes!(builder, TILLED_SOIL_WET_TOP, "textures/tilled_dirt_wet.png")?;
    include_texture_bytes!(builder, TILLED_SOIL_WET_SIDE, "textures/dirt_side_wet.png")?;

    let paddy_dry = BlockBuilder::new(
        &builder
            .builder_extension_mut::<FarmingGameStateExtension>()
            .paddy_dry,
    )
    .set_cube_appearance(
        CubeAppearanceBuilder::new()
            .set_single_texture(OwnedTextureName::from_css_color("#000030")),
    )
    .build_and_deploy_into(builder)?;

    let paddy_wet = BlockBuilder::new(
        &builder
            .builder_extension_mut::<FarmingGameStateExtension>()
            .paddy_wet,
    )
    .set_cube_appearance(
        CubeAppearanceBuilder::new()
            .set_single_texture(OwnedTextureName::from_css_color("#000080")),
    )
    .add_item_group(HIDDEN_FROM_CREATIVE)
    .build_and_deploy_into(builder)?;

    // TODO: hide this from creative once we have a suitable tool
    let soil_dry = BlockBuilder::new(
        &builder
            .builder_extension_mut::<FarmingGameStateExtension>()
            .soil_dry,
    )
    .set_cube_appearance(CubeAppearanceBuilder::new().set_individual_textures(
        TILLED_SOIL_SIDE,
        TILLED_SOIL_SIDE,
        TILLED_SOIL_TOP,
        TILLED_SOIL_TOP,
        TILLED_SOIL_SIDE,
        TILLED_SOIL_SIDE,
    ))
    .build_and_deploy_into(builder)?;

    let soil_wet = BlockBuilder::new(
        &builder
            .builder_extension_mut::<FarmingGameStateExtension>()
            .soil_wet,
    )
    .set_cube_appearance(CubeAppearanceBuilder::new().set_individual_textures(
        TILLED_SOIL_WET_SIDE,
        TILLED_SOIL_WET_SIDE,
        TILLED_SOIL_WET_TOP,
        TILLED_SOIL_WET_TOP,
        TILLED_SOIL_WET_SIDE,
        TILLED_SOIL_WET_SIDE,
    ))
    .add_item_group(HIDDEN_FROM_CREATIVE)
    .build_and_deploy_into(builder)?;

    for (name, wet, dry, allow_downflow) in [
        ("SoilMoisture", soil_wet.id, soil_dry.id, false),
        ("PaddyMoisture", paddy_wet.id, paddy_dry.id, true),
    ] {
        let callback = SoilTimerCallback {
            dry,
            wet,
            water: FastBlockName::new(WATER.0),
            allow_downflow,
        };

        builder.inner.add_timer(
            name,
            TimerSettings {
                interval: Duration::from_secs(5),
                shards: 16,
                spreading: 1.0,
                block_types: vec![dry, wet],
                ignore_block_type_presence_check: false,
                per_block_probability: 1.0,
                idle_chunk_after_unchanged: false,
                ..Default::default()
            },
            TimerCallback::BulkUpdateWithNeighbors(Box::new(callback)),
        );
    }

    Ok(())
}
