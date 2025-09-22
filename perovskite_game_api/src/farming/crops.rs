use crate::blocks::{BlockBuilder, BuiltBlock, CubeAppearanceBuilder};
use crate::game_builder::{BlockName, GameBuilder, OwnedTextureName};
use anyhow::{ensure, Context, Result};
use itertools::Itertools;
use perovskite_core::block_id::special_block_defs::AIR_ID;
use perovskite_core::block_id::BlockId;
use perovskite_core::coordinates::{BlockCoordinate, ChunkCoordinate, ChunkOffset};
use perovskite_core::protocol::items::item_stack::QuantityType;
use perovskite_core::protocol::render::TextureReference;
use perovskite_server::game_state::blocks::{
    BlockInteractionResult, BlockType, ExtendedDataHolder, FastBlockName, InlineContext,
    InlineHandler,
};
use perovskite_server::game_state::event::HandlerContext;
use perovskite_server::game_state::game_map::{
    BulkUpdateCallback, ChunkNeighbors, MapChunk, TimerCallback, TimerInlineCallback,
    TimerSettings, TimerState, VerticalNeighborTimerCallback,
};
use perovskite_server::game_state::items::ItemStack;
use rand::{thread_rng, Rng};
use rustc_hash::FxHashMap;
use std::ops::RangeInclusive;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NonExhaustive(pub(crate) ());

#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum InteractionTransition {
    /// When dug, jump to a stage in this range, selected randomly
    JumpToStage(Vec<usize>),
    /// Remove the crop altogether
    Remove,
}

/// What happens when a crop is interacted with, dug, or tapped

#[derive(Debug, Clone)]
pub struct InteractionEffect {
    /// Items and counts dropped when this is dug
    pub item_drops: Vec<(String, RangeInclusive<u32>)>,
    /// What happens to the block itself
    pub transition: InteractionTransition,
    /// Probability of the transition happening; item drops happen regardless
    pub transition_probability: f64,

    /// Non-exhaustive - this struct must be constructed with functional update syntax, i.e.
    /// ```
    /// # use perovskite_game_api::farming::crops::InteractionEffect;
    /// InteractionEffect { item_drops: vec![("foo:bar".to_string(), 3..=5)], ..Default::default() }
    /// ```
    pub _ne: NonExhaustive,
}

impl InteractionEffect {
    fn build_handler(self, stage_fbns: Vec<FastBlockName>) -> Result<Box<InlineHandler>> {
        ensure!(self.transition_probability >= 0.0 && self.transition_probability <= 1.0);
        if let InteractionTransition::JumpToStage(stages) = &self.transition {
            ensure!(!stages.is_empty());
            for &stage in stages {
                ensure!(stage < stage_fbns.len());
            }
        }

        Ok(Box::new(
            move |ctx: InlineContext, id: &mut BlockId, _ext, _itemstack| {
                let mut rng = thread_rng();
                let drops = self
                    .item_drops
                    .iter()
                    .filter_map(|(name, counts)| match rng.gen_range(counts.clone()) {
                        0 => None,
                        x => Some(ItemStack {
                            proto: perovskite_core::protocol::items::ItemStack {
                                item_name: name.to_string(),
                                quantity: x,
                                current_wear: 0,
                                quantity_type: Some(QuantityType::Stack(256)),
                            },
                        }),
                    })
                    .collect();

                if rng.gen_bool(self.transition_probability) {
                    *id = match &self.transition {
                        InteractionTransition::JumpToStage(stages) => {
                            let stage = stages[rng.gen_range(0..stages.len())];
                            ctx.block_types()
                                .resolve_name(&stage_fbns[stage])
                                .unwrap_or(*id)
                        }
                        InteractionTransition::Remove => AIR_ID,
                    }
                }

                Ok(BlockInteractionResult {
                    item_stacks: drops,
                    tool_wear: 0,
                })
            },
        ))
    }
}

impl Default for InteractionEffect {
    fn default() -> Self {
        InteractionEffect {
            item_drops: Vec::new(),
            transition: InteractionTransition::Remove,
            transition_probability: 1.0,
            _ne: NonExhaustive(()),
        }
    }
}

/// A single growth stage of a crop. As the crop grows, it will transition to
/// the next growth stage whenever the map timer fires, subject to the probability specified
///
/// Default behavior: `dig_effect` removes block without dropping anything, no tap effects, no
/// interaction effects, 1.0 probability of growth (except for the last stage which is forced to 0
/// probability since there is no further stage)
#[derive(Clone, Debug)]
pub struct GrowthStage {
    /// What happens when the crop at this stage is dug
    pub dig_effect: Option<InteractionEffect>,
    /// What happens when the crop at this stage is tapped
    pub tap_effect: Option<InteractionEffect>,
    /// What happens when the crop at this stage is interacted
    /// The string indicates the menu text for the interaction effect
    pub interaction_effects: FxHashMap<String, InteractionEffect>,

    /// Extra block groups, that govern how tools dig this
    pub extra_block_groups: Vec<String>,
    /// How likely the stage is to increment when the timer fires
    pub grow_probability: f32,

    /// The block texture
    pub texture_name: OwnedTextureName,

    /// Non-exhaustive - this struct must be constructed with functional update syntax, i.e.
    /// ```
    /// # use perovskite_game_api::farming::crops::GrowthStage;
    /// GrowthStage { grow_probability: 0.5, ..Default::default() }
    /// ```
    pub _ne: NonExhaustive,
}
impl Default for GrowthStage {
    fn default() -> Self {
        GrowthStage {
            dig_effect: Some(Default::default()),
            tap_effect: None,
            interaction_effects: Default::default(),
            extra_block_groups: vec![],
            grow_probability: 1.0,
            texture_name: OwnedTextureName(String::new()),
            _ne: NonExhaustive(()),
        }
    }
}

/// The definition of a crop, passed to [`define_crop`]
#[derive(Clone, Debug)]
pub struct CropDefinition {
    /// The prefix for all blocks (and corresponding hidden internal items) for this crop
    /// Should follow block name syntax, e.g. `plugin_name:tomatoes`.
    /// The default for this is empty, which must be overridden.
    pub base_name: String,
    /// Definition for each growth stage. The default for this is empty, which must be overridden.
    pub stages: Vec<GrowthStage>,
    /// The kinds of soil blocks this will grow on. This can be either one of the blocks in
    /// [crate::farming::FarmingGameStateExtension], or any other block (e.g.
    /// [crate::default_game::basic_blocks::DIRT] converted with `.into()`). The default for this is
    /// empty, which should be overridden unless your crop should only grow with manual interactions
    pub eligible_soil_blocks: Vec<FastBlockName>,
    /// How often the timer fires. Each `grow_probability` is evaluated once per this duration.
    pub timer_period: Duration,
    pub _ne: NonExhaustive,
}
impl Default for CropDefinition {
    fn default() -> Self {
        Self {
            base_name: "".to_string(),
            stages: vec![],
            eligible_soil_blocks: vec![],
            timer_period: Duration::from_millis(60),
            _ne: NonExhaustive(()),
        }
    }
}

/// The result of defining a crop
#[derive(Clone, Debug)]
pub struct BuiltCrop {
    /// The blocks defined for each growth stage of the crop. Guaranteed to be the same length as
    /// `stages` in the crop definition.
    pub built_stages: Vec<BuiltBlock>,
}

pub const CROPS_GROUP: &str = "farming:crops";

pub fn define_crop(game_builder: &mut GameBuilder, def: CropDefinition) -> Result<BuiltCrop> {
    ensure!(!def.stages.is_empty());
    ensure!(!def.base_name.is_empty());

    let stage_names: Vec<String> = (0..def.stages.len())
        .map(|x| format!("{}:{}", &def.base_name, x))
        .collect();

    let stage_fbns: Vec<FastBlockName> = stage_names.iter().map(FastBlockName::new).collect();

    let mut built_stages = Vec::with_capacity(def.stages.len());
    for (stage, name) in def.stages.into_iter().zip(stage_names) {
        let mut builder = BlockBuilder::new(BlockName(name))
            .add_block_group(CROPS_GROUP)
            .add_block_groups(stage.extra_block_groups)
            .add_block_group(format!("crops:auto_group:{}", &def.base_name))
            .set_cube_appearance(
                CubeAppearanceBuilder::new().set_single_texture(stage.texture_name),
            );

        if let Some(effect) = stage.dig_effect {
            let handler = effect.build_handler(stage_fbns.clone())?;
            builder = builder.add_modifier(Box::new(|bt| {
                bt.dig_handler_inline = Some(handler);
            }));
        }
        if let Some(effect) = stage.tap_effect {
            let handler = effect.build_handler(stage_fbns.clone())?;
            builder = builder.add_modifier(Box::new(|bt| {
                bt.tap_handler_inline = Some(handler);
            }));
        }

        let built_block = builder.build_and_deploy_into(game_builder)?;

        built_stages.push((built_block, stage.grow_probability));
    }

    let timer_block_types = built_stages.iter().map(|x| x.0.id).collect();

    let timer_impl = GrowTimerImpl {
        stages: built_stages.clone(),
        eligible_soil_blocks: def.eligible_soil_blocks.clone(),
    };

    game_builder.inner.add_timer(
        format!("crops:autogen:growth_{}", &def.base_name),
        TimerSettings {
            interval: def.timer_period,
            shards: 16,
            spreading: 1.0,
            block_types: timer_block_types,
            ignore_block_type_presence_check: false,
            per_block_probability: 1.0,
            // must be false, we have a possibility of not acting on any blocks due to RNG
            idle_chunk_after_unchanged: false,
            ..Default::default()
        },
        TimerCallback::BulkUpdateWithNeighbors(Box::new(timer_impl)),
    );

    Ok(BuiltCrop {
        built_stages: built_stages.into_iter().map(|x| x.0).collect(),
    })
}

struct GrowTimerImpl {
    stages: Vec<(BuiltBlock, f32)>,
    eligible_soil_blocks: Vec<FastBlockName>,
}
// TODO: introduce lighting! This will require some engine improvements
impl GrowTimerImpl {
    fn act(&self, ctx: &HandlerContext<'_>, plant: BlockId, soil: BlockId) -> BlockId {
        let stage = self
            .stages
            .iter()
            .find_position(|x| x.0.id.equals_ignore_variant(plant));
        if let Some((i, (_, grow_probability))) = stage {
            tracing::debug!("gti stage matched");
            if self.eligible_soil_blocks.iter().any(|x| {
                ctx.block_types()
                    .resolve_name(&x)
                    .is_some_and(|x| x.equals_ignore_variant(soil))
            }) {
                tracing::debug!("gti act() matched soil");
                if i < (self.stages.len() - 1) {
                    tracing::debug!("gti found stage {}", i);
                    if rand::thread_rng().gen_bool(*grow_probability as f64) {
                        tracing::debug!("gti rng passed");
                        return self.stages[i + 1].0.id;
                    }
                }
            }
        }

        plant
    }
}
impl BulkUpdateCallback for GrowTimerImpl {
    fn bulk_update_callback(
        &self,
        ctx: &HandlerContext<'_>,
        chunk_coordinate: ChunkCoordinate,
        _timer_state: &TimerState,
        chunk: &mut MapChunk,
        neighbors: Option<&ChunkNeighbors>,
    ) -> Result<()> {
        let neighbors = neighbors.context("Crops growth update callback missing neighbors")?;
        tracing::debug!("gti timer invoked");
        for x in 0..16 {
            for z in 0..16 {
                for y in 0..16 {
                    let offset = ChunkOffset::new(x as u8, y as u8, z as u8);
                    let coord = chunk_coordinate.with_offset(offset);
                    let below = match coord.try_delta(0, -1, 0) {
                        Some(x) => x,
                        None => continue,
                    };
                    if let Some(block_below) = neighbors.get_block(below) {
                        let new_block = self.act(ctx, chunk.get_block(offset), block_below);
                        chunk.set_block(offset, new_block, None);
                    }
                }
            }
        }
        Ok(())
    }
}
