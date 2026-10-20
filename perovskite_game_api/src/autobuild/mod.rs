//! Tools for players to quickly build structures. Initial focus is on roads, to prove
//! out the UX and get some initial experimentation.

use std::any::Any;

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use itertools::Itertools;
use perovskite_core::{
    block_id::{special_block_defs::AIR_ID, BlockId},
    chat::{ChatMessage, SERVER_ERROR_COLOR},
    constants::{
        block_groups::{DEFAULT_SOLID, TRIVIALLY_REPLACEABLE},
        permissions::WORLD_STATE,
    },
    coordinates::{BlockCoordinate, ChunkCoordinate, ChunkOffset},
    protocol::{
        items::{self as items_proto, interaction_rule::DigBehavior, InteractionRule},
        ui::ToolHint,
    },
};
use perovskite_server::game_state::{
    blocks::{ExtendedData, FastBlockGroup},
    chat::commands::ChatCommandHandler,
    client_ui::{Popup, UiElementContainer},
    event::{EventInitiator, HandlerContext},
    items::{Item, ItemInteractionResult, ItemStack, PointeeBlockCoords},
    player::{self, Player},
};

use crate::{
    blocks::{
        self,
        variants::{VARIANT_PLACED_AUTOBUILD, VARIANT_PLACER_PLAYER},
    },
    default_game::block_groups::{
        NATURAL_AND_STRUCTURAL, NATURAL_GROUND, PLACER_NEVER_RELEVANT, VARIANT_ENCODES_PLACER,
    },
    game_builder::{GameBuilder, OwnedTextureName},
};

pub mod machines;
mod road;

pub fn initialize_autobuild(game: &mut GameBuilder) -> anyhow::Result<()> {
    game.inner.register_command(
        "undo",
        Box::new(UndoAutobuildCommand),
        "Undo the last autobuild operation",
    )?;

    game.inner
        .blocks_mut()
        .register_fast_block_group(NATURAL_GROUND);
    game.inner
        .blocks_mut()
        .register_fast_block_group(NATURAL_AND_STRUCTURAL);
    game.inner
        .blocks_mut()
        .register_fast_block_group(VARIANT_ENCODES_PLACER);
    game.inner
        .blocks_mut()
        .register_fast_block_group(PLACER_NEVER_RELEVANT);

    road::initialize_road_tool(game)?;

    let fill_texture = OwnedTextureName::from_css_color("#4080ff");
    let mut fill_tool = Item {
        ..Item::default_with_proto(items_proto::ItemDef {
            short_name: "autobuild:fill_tool".to_string(),
            display_name: "Fill tool".to_string(),
            appearance: fill_texture.into(),
            groups: vec![],
            interaction_rules: vec![],
            quantity_type: None,
            sort_key: "autobuild:fill_tool".to_string(),
            // Autobuild tools are meant for citybuilding, so we give them a large tool range
            tool_range: 50.0,
        })
    };
    configure_item::<FillTool>(&mut fill_tool);
    game.inner.items_mut().register_item(fill_tool)?;

    let clear_texture = OwnedTextureName::from_css_color("#ff7070");
    let mut clear_tool = Item {
        ..Item::default_with_proto(items_proto::ItemDef {
            short_name: "autobuild:clear_tool".to_string(),
            display_name: "Clear tool".to_string(),
            appearance: clear_texture.into(),
            groups: vec![],
            interaction_rules: vec![],
            quantity_type: None,
            sort_key: "autobuild:clear_tool".to_string(),
            // Autobuild tools are meant for citybuilding, so we give them a large tool range
            tool_range: 50.0,
        })
    };
    configure_item::<ClearTool>(&mut clear_tool);
    game.inner.items_mut().register_item(clear_tool)?;

    machines::register_machines(game)?;

    Ok(())
}

pub fn configure_item<T: Autobuilder>(item: &mut Item) {
    let item_name = item.proto.short_name.clone();
    item.place_on_block_handler = Some(Box::new(move |ctx, coord, stack| {
        place_on_block_interaction::<T>(ctx, coord, stack, &item_name)
    }));
    let item_name = item.proto.short_name.clone();
    item.dig_handler = Some(Box::new(move |ctx, coord, stack| {
        dig_interaction::<T>(ctx, coord, stack, &item_name)
    }));
    let item_name = item.proto.short_name.clone();
    item.tap_handler = Some(Box::new(move |ctx, coord, stack| {
        tap_interaction::<T>(ctx, coord, stack, &item_name)
    }));
    item.proto.interaction_rules = vec![InteractionRule {
        block_group: vec![DEFAULT_SOLID.to_string()],
        tool_wear: 1,
        dig_behavior: Some(DigBehavior::ConstantTime(0.3)),
    }];
}

struct UndoAutobuildCommand;
#[async_trait]
impl ChatCommandHandler for UndoAutobuildCommand {
    async fn handle(&self, _message: &str, context: &HandlerContext<'_>) -> Result<()> {
        if !context.initiator().check_permission_if_player(WORLD_STATE) {
            context.initiator().send_chat_message(
                ChatMessage::new_server_message("You don't have permission to use this command")
                    .with_color(SERVER_ERROR_COLOR),
            )?;
            return Ok(());
        }
        match context.initiator() {
            EventInitiator::Player(p) => {
                let undo = p.player.with_transient_data::<BatchedUndo, _>(|x| std::mem::take(x));
                undo.undo(context)?;
                Ok(())
            }
            EventInitiator::WeakPlayerRef(weak) => {
                weak.try_to_run(|p| {
                    let undo = p.with_transient_data::<BatchedUndo, _>(|x| std::mem::take(x));
                    undo.undo(context)?;
                    Ok(())
                })
                .ok_or_else(|| anyhow::anyhow!("Player not found"))?
            }
            _ => bail!("Autobuild undo command can only be used interactively by players; call methods on the undo object directly to undo (or better yet, file a feature request for proper database transactions; undo is meant as a player aid)"),
        }
    }
}

/// A helper for writing many blocks to the game map efficiently.
#[derive(Default)]
pub struct BatchedWrite {
    /// The blocks to write.
    blocks: Vec<PointWrite>,
}

/// What to do when a block cannot be overwritten during a [BatchedWrite].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverwriteFailureAction {
    /// Stop the write operation and return an error.
    Stop,
    /// Skip the block and continue with the next one.
    Skip,
}

/// Parameters for a [BatchedWrite].
pub struct WriteParameters<'a> {
    /// Whether to detect blocks placed by players and avoid overwriting them.
    pub detect_player_placed: bool,
    /// Whether to allow overwriting blocks that were placed by an earlier autobuild operation.
    pub allow_overwrite_autobuilds: bool,
    /// What to do if a block cannot be overwritten.
    pub overwrite_failure_action: OverwriteFailureAction,
    /// Additional block groups that should be considered trivially replaceable for this operation.
    pub additional_replaceable_groups: Vec<FastBlockGroup<'a>>,
}
impl<'a> Default for WriteParameters<'a> {
    fn default() -> Self {
        Self {
            detect_player_placed: true,
            allow_overwrite_autobuilds: false,
            overwrite_failure_action: OverwriteFailureAction::Skip,
            additional_replaceable_groups: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverwriteBehavior {
    /// If a conflicting block (based on the write settings) is found,
    /// the write will either stop (if [OverwriteFailureAction::Stop] is set)
    /// or skip the block (if [OverwriteFailureAction::Skip] is set).
    DetectConflicts,
    /// If a conflicting block is found, it will be overwritten regardless of
    /// [OverwriteFailureAction].
    ForceOverwrite,
    /// If a conflicting block is found, it will be skipped regardless of
    /// [OverwriteFailureAction].
    SilentlySkip,
}

#[derive(Clone, Debug, Copy)]
struct PointWrite {
    coord: BlockCoordinate,
    id: BlockId,
    overwrite_behavior: OverwriteBehavior,
}

impl BatchedWrite {
    /// Creates a new [BatchedWrite].
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a block to the write operation with [OverwriteBehavior::DetectConflicts].
    pub fn add_block(&mut self, coord: BlockCoordinate, id: BlockId) {
        self.add_block_with_behavior(coord, id, OverwriteBehavior::DetectConflicts)
    }

    /// Adds a block to the write operation with a specified overwrite behavior.
    pub fn add_block_with_behavior(
        &mut self,
        coord: BlockCoordinate,
        id: BlockId,
        overwrite_behavior: OverwriteBehavior,
    ) {
        self.blocks.push(PointWrite {
            coord,
            id,
            overwrite_behavior,
        });
    }

    /// Writes the blocks to the game map, and returns an undo object.
    ///
    /// This is more efficient than writing blocks one by one, because it allows
    /// the game map to batch the writes.
    pub fn commit(
        mut self,
        ctx: &HandlerContext,
        write_params: WriteParameters,
    ) -> Result<BatchedUndo> {
        // todo: bulk fixup, maybe move this into the template functionality?
        let mut undo = BatchedUndo::default();
        let start = std::time::Instant::now();
        let block_count = self.blocks.len();
        self.blocks
            .sort_unstable_by_key(|w| (w.coord.chunk().x, w.coord.chunk().z, w.coord.chunk().y));
        let sort_time = start.elapsed();

        let mut blocks_to_write = Vec::new();

        let read_start = std::time::Instant::now();
        if write_params.detect_player_placed {
            let trivially_replaceable_group = ctx
                .block_types()
                .fast_block_group(TRIVIALLY_REPLACEABLE)
                .context("TRIVIALY_REPLACEABLE fast block group not registered")?
                .to_owned();
            let trivially_replaceable = write_params
                .additional_replaceable_groups
                .iter()
                .fold(trivially_replaceable_group, |acc, group| {
                    acc.or(group.to_owned())
                });

            let tracks_variant_group =
                ctx.block_types()
                    .fast_block_group(VARIANT_ENCODES_PLACER)
                    .context("VARIANT_ENCODES_PLACER fast block group not registered")?;

            let placer_never_relevant_group = ctx
                .block_types()
                .fast_block_group(PLACER_NEVER_RELEVANT)
                .context("PLACER_NEVER_RELEVANT fast block group not registered")?;

            let natural_ground_group = ctx
                .block_types()
                .fast_block_group(NATURAL_GROUND)
                .context("NATURAL_GROUND fast block group not registered")?;

            let natural_and_structural = ctx
                .block_types()
                .fast_block_group(NATURAL_AND_STRUCTURAL)
                .context("NATURAL_AND_STRUCTURAL fast block group not registered")?;

            for (key, group) in self.blocks.iter().chunk_by(|w| w.coord.chunk()).into_iter() {
                ctx.game_map().bulk_read_chunk(key, |chunk| {
                    for w in group {
                        let old_block = chunk.get_block(w.coord.offset());

                        let is_natural_ground = natural_ground_group.contains(old_block);
                        let is_natural_and_structural = natural_and_structural.contains(old_block);
                        let natural_nonstructural = is_natural_ground && !is_natural_and_structural;
                        let placer_never_relevant = placer_never_relevant_group.contains(old_block);

                        let mask_matches = old_block.variant() & VARIANT_PLACER_PLAYER != 0;
                        let is_autobuilt = old_block.variant() & VARIANT_PLACED_AUTOBUILD != 0;

                        let tracked = tracks_variant_group.contains(old_block);

                        let flagged = if old_block == AIR_ID {
                            false
                        } else if trivially_replaceable.contains(old_block) {
                            false
                        } else if !write_params.detect_player_placed {
                            // we don't care about whether the player placed this
                            false
                        } else if !tracked {
                            // It's not tracked. Likely high-value, don't overwrite.
                            true
                        } else if placer_never_relevant {
                            false
                        } else if mask_matches {
                            // player placed it directly
                            // If it's natural nonstructural, we don't care.
                            !natural_nonstructural
                        } else if is_autobuilt {
                            // it's autobuild, so only conflict if autobuilds aren't allowed
                            !write_params.allow_overwrite_autobuilds
                        } else {
                            // it's not natural nonstructural, it's not player placed, it's not autobuild...
                            // we don't care about it
                            false
                        };

                        let effective_flag = match w.overwrite_behavior {
                            OverwriteBehavior::DetectConflicts => flagged,
                            OverwriteBehavior::ForceOverwrite => false,
                            OverwriteBehavior::SilentlySkip => {
                                if flagged {
                                    continue;
                                } else {
                                    false
                                }
                            }
                        };
                        if effective_flag {
                            if write_params.overwrite_failure_action == OverwriteFailureAction::Stop
                            {
                                let block_name = ctx.block_types().human_short_name(old_block);
                                bail!(
                                    "Found a conflicting block ({}) at {:?}",
                                    block_name,
                                    w.coord
                                );
                            } else {
                                // continue, but don't write it
                            }
                        } else {
                            blocks_to_write.push(w);
                        }
                    }
                    Ok(())
                })?;
            }
        } else {
            for (_key, group) in self.blocks.iter().chunk_by(|w| w.coord.chunk()).into_iter() {
                blocks_to_write.extend(group);
            }
        }
        let read_time = read_start.elapsed();

        let write_start = std::time::Instant::now();
        for (key, group) in blocks_to_write
            .iter()
            .chunk_by(|w| w.coord.chunk())
            .into_iter()
        {
            let undos = ctx.game_map().bulk_write_chunk(key, |chunk| {
                let mut undos = Vec::new();
                for w in group {
                    let (old_block, old_ext_data) = chunk.set_block(w.coord.offset(), w.id, None);
                    undos.push((w.coord.offset(), old_block, old_ext_data));
                }
                Ok(undos)
            })?;
            undo.blocks.push((key, undos));
        }
        let write_time = write_start.elapsed();
        let total_time = start.elapsed();
        tracing::info!(
            "BatchedWrite::write: sort_time: {:?} micros, read_time: {:?} micros, write_time: {:?} micros, total_time: {:?} micros ({:.2} nanoseconds per block)",
            sort_time.as_micros(),
            read_time.as_micros(),
            write_time.as_micros(),
            total_time.as_micros(),
            total_time.as_nanos() as f64 / block_count as f64
        );

        Ok(undo)
    }
}

/// Stores information needed to undo a [BatchedWrite].
#[derive(Default)]
pub struct BatchedUndo {
    /// The blocks that were overwritten, grouped by chunk.
    pub blocks: Vec<(
        ChunkCoordinate,
        Vec<(ChunkOffset, BlockId, Option<ExtendedData>)>,
    )>,
}

impl BatchedUndo {
    /// Undoes the write operation.
    pub fn undo(self, ctx: &HandlerContext) -> Result<()> {
        for (key, undos) in self.blocks {
            ctx.game_map().bulk_write_chunk(key, |chunk| {
                for (coord, block_id, ext_data) in undos {
                    chunk.set_block(coord, block_id, ext_data);
                }
                Ok(())
            })?;
        }
        Ok(())
    }
}

/// Trait for autobuilder tools.
///
/// This is essentially a type-level trait for ease of registration and to have less data to carry around.
///
/// If this is too un-ergonomic for a use-case, please open an issue - I am open to changing it with a good reason.
pub trait Autobuilder {
    type Settings: player::PersistentData + Default + Clone;
    type SelectionState: Any + Send + Sync + Default + Clone + 'static;
    /// Creates a popup for configuring the tool. If None, no popup will be shown (the side effect of the function may change the settings)
    fn make_settings_popup(
        ctx: &HandlerContext,
        coord: PointeeBlockCoords,
        settings: &Self::Settings,
    ) -> Option<Popup>;
    /// Saves the settings to the player's persistent data. This is provided as a convenience,
    /// to be used in the callback of the settings popup.
    fn save_settings(
        ctx: &HandlerContext,
        settings: Self::Settings,
        item_name: &str,
    ) -> Result<()> {
        let work = |p: &Player| {
            let transient_data = p.with_transient_data::<Self::SelectionState, _>(|x| x.clone());

            p.update_tool_hint(
                item_name.to_string(),
                Self::current_hint(&settings, &transient_data, ctx),
            )?;

            p.with_persistent_data(Self::TOOL_ID, |x| {
                *x = settings.clone();
                Ok(())
            })
        };
        match ctx.initiator() {
            EventInitiator::Player(p) => work(&p.player),
            EventInitiator::WeakPlayerRef(weak) => weak
                .try_to_run(work)
                .ok_or_else(|| anyhow::anyhow!("Player not found"))?,
            _ => bail!("Autobuild can only be used interactively by players"),
        }
    }
    /// Creates a popup for showing advice to the user, e.v. when the state is incomplete or
    /// the user's usage must be corrected.
    fn make_advice_popup(ctx: &HandlerContext, state: &Self::SelectionState) -> Popup;
    /// Returns true if the advice popup should be shown.
    fn should_show_advice(state: &Self::SelectionState) -> bool;

    /// Called when the player taps the tool on the ground.
    fn tap(
        ctx: &HandlerContext,
        coord: PointeeBlockCoords,
        settings: &mut Self::Settings,
        state: &mut Self::SelectionState,
    ) -> Result<()>;

    /// Derives the current tool hint from the selection state. Called after every
    /// action that may change state (tap and build), so the client always sees a
    /// hint that is consistent with the current state. Returns None (no hint) by
    /// default; override to provide per-state hints.
    fn current_hint(
        _settings: &Self::Settings,
        _state: &Self::SelectionState,
        _ctx: &HandlerContext,
    ) -> Option<ToolHint> {
        None
    }

    /// Actually builds. Invoked when the player right-clicks. Returns an undo object that
    /// can be used to undo the build.
    fn build(
        ctx: &HandlerContext,
        pointee_coord: PointeeBlockCoords,
        settings: &mut Self::Settings,
        state: &mut Self::SelectionState,
    ) -> Result<Option<BatchedUndo>>;
    /// A unique identifier for this tool, used to store settings in persistent data.
    /// This should be unique across all autobuilders, and also include `autobuild:` to
    /// avoid collisions with other plugins.
    const TOOL_ID: &'static str;
}

fn place_on_block_interaction<T: Autobuilder>(
    ctx: &HandlerContext,
    coord: PointeeBlockCoords,
    stack: &ItemStack,
    item_name: &str,
) -> Result<ItemInteractionResult> {
    let work = |p: &Player| -> Result<()> {
        let mut data = p.with_transient_data::<T::SelectionState, _>(|x| x.clone());
        let mut settings =
            p.with_persistent_data::<T::Settings, _>(T::TOOL_ID, |x| Ok(x.clone()))?;

        if T::should_show_advice(&data) {
            p.show_popup_blocking(T::make_advice_popup(ctx, &data))
        } else {
            match T::build(ctx, coord, &mut settings, &mut data) {
                Ok(Some(undo)) => {
                    p.send_chat_message(ChatMessage::new_server_message(
                        "Build successful! Use /undo to undo (only one undo at a time)",
                    ))?;
                    p.with_transient_data::<BatchedUndo, _>(|x| *x = undo);
                }
                Ok(None) => {}
                Err(e) => {
                    p.send_chat_message(
                        ChatMessage::new_server_message(e.to_string())
                            .with_color(SERVER_ERROR_COLOR),
                    )?;
                }
            }
            p.update_tool_hint(
                item_name.to_string(),
                T::current_hint(&settings, &data, ctx),
            )?;
            p.with_transient_data::<T::SelectionState, _>(|x| *x = data);
            p.with_persistent_data::<T::Settings, _>(T::TOOL_ID, |x| {
                *x = settings;
                Ok(())
            })?;

            Ok(())
        }
    };

    match ctx.initiator() {
        perovskite_server::game_state::event::EventInitiator::Player(p) => {
            work(p.player)?;
        }
        perovskite_server::game_state::event::EventInitiator::WeakPlayerRef(weak) => {
            match weak.try_to_run(|p| work(p)) {
                None => {}
                Some(Ok(_)) => {}
                Some(Err(e)) => return Err(e),
            }
        }
        _ => {
            bail!("Autobuild can only be used interactively by players; consider calling build functions directly");
        }
    };
    // give the user back their tool
    Ok(Some(stack.clone()).into())
}

fn dig_interaction<T: Autobuilder>(
    ctx: &HandlerContext,
    coord: PointeeBlockCoords,
    stack: &ItemStack,
    _item_name: &str,
) -> Result<ItemInteractionResult> {
    let work = |p: &Player| -> Result<()> {
        let settings = p.with_persistent_data::<T::Settings, _>(T::TOOL_ID, |x| Ok(x.clone()))?;
        if let Some(popup) = T::make_settings_popup(ctx, coord, &settings) {
            p.show_popup_blocking(popup)?;
        }
        Ok(())
    };

    match ctx.initiator() {
        perovskite_server::game_state::event::EventInitiator::Player(p) => {
            work(p.player)?;
        }
        perovskite_server::game_state::event::EventInitiator::WeakPlayerRef(weak) => {
            match weak.try_to_run(|p| work(p)) {
                None => {}
                Some(Ok(_)) => {}
                Some(Err(e)) => return Err(e),
            }
        }
        _ => {
            bail!("Autobuild can only be used interactively by players; consider calling build functions directly");
        }
    };
    // give the user back their tool
    Ok(Some(stack.clone()).into())
}

fn tap_interaction<T: Autobuilder>(
    ctx: &HandlerContext,
    coord: PointeeBlockCoords,
    stack: &ItemStack,
    item_name: &str,
) -> Result<ItemInteractionResult> {
    let work = |p: &Player| -> Result<()> {
        let mut state = p.with_transient_data::<T::SelectionState, _>(|x| x.clone());
        let mut settings =
            p.with_persistent_data::<T::Settings, _>(T::TOOL_ID, |x| Ok(x.clone()))?;

        if let Err(e) = T::tap(ctx, coord, &mut settings, &mut state) {
            p.send_chat_message(
                ChatMessage::new_server_message(format!("Error using {}: {}", item_name, e))
                    .with_color(SERVER_ERROR_COLOR),
            )?;
        }
        p.update_tool_hint(
            item_name.to_string(),
            T::current_hint(&settings, &state, ctx),
        )?;
        p.with_transient_data::<T::SelectionState, _>(|x| *x = state.clone());
        p.with_persistent_data::<T::Settings, _>(T::TOOL_ID, |x| {
            *x = settings;
            Ok(())
        })?;
        Ok(())
    };

    match ctx.initiator() {
        perovskite_server::game_state::event::EventInitiator::Player(p) => {
            work(p.player)?;
        }
        perovskite_server::game_state::event::EventInitiator::WeakPlayerRef(weak) => {
            match weak.try_to_run(|p| work(p)) {
                None => {}
                Some(Ok(_)) => {}
                Some(Err(e)) => return Err(e),
            }
        }
        _ => {
            bail!("Autobuild can only be used interactively by players; consider calling build functions directly");
        }
    };
    // give the user back their tool
    Ok(Some(stack.clone()).into())
}

pub trait AutobuildExt {
    fn set_autobuild_undo(&self, undo: BatchedUndo) -> Result<()>;
}
impl<'a> AutobuildExt for HandlerContext<'a> {
    fn set_autobuild_undo(&self, undo: BatchedUndo) -> Result<()> {
        match self.initiator() {
            perovskite_server::game_state::event::EventInitiator::Player(p) => {
                p.player
                    .with_transient_data::<BatchedUndo, _>(|x| *x = undo);
            }
            perovskite_server::game_state::event::EventInitiator::WeakPlayerRef(weak) => {
                weak.try_to_run(|p| {
                    p.with_transient_data::<BatchedUndo, _>(|x| *x = undo);
                    Ok::<(), anyhow::Error>(())
                })
                .ok_or_else(|| anyhow::anyhow!("Player not found"))??;
            }
            _ => {
                bail!("Autobuild can only be used interactively by players; consider calling build functions directly");
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
struct FillToolSelectionState {
    start: Option<BlockCoordinate>,
}

#[derive(Clone, prost::Message)]
struct FillToolSettings {
    #[prost(uint32, tag = "1")]
    block: u32,
}

struct FillTool;
impl Autobuilder for FillTool {
    type Settings = FillToolSettings;
    type SelectionState = FillToolSelectionState;

    fn make_settings_popup(
        ctx: &HandlerContext,
        coord: PointeeBlockCoords,
        _settings: &Self::Settings,
    ) -> Option<Popup> {
        let block_id = match ctx.game_map().get_block(coord.selected) {
            Ok(id) => id,
            Err(e) => {
                let _ = ctx.initiator().send_chat_message(
                    ChatMessage::new_server_message(format!("Error reading block: {e}"))
                        .with_color(SERVER_ERROR_COLOR),
                );
                return None;
            }
        };
        let block_name = ctx
            .block_types()
            .get_block(block_id)
            .map(|(b, _)| b.short_name().to_string())
            .unwrap_or_else(|_| format!("{:?}", block_id));
        let new_settings = FillToolSettings { block: block_id.0 };
        if let Err(e) = Self::save_settings(ctx, new_settings, "autobuild:fill_tool") {
            let _ = ctx.initiator().send_chat_message(
                ChatMessage::new_server_message(format!("Error saving settings: {e}"))
                    .with_color(SERVER_ERROR_COLOR),
            );
            return None;
        }
        let _ = ctx
            .initiator()
            .send_chat_message(ChatMessage::new_server_message(format!(
                "Will fill with {:?}",
                block_name
            )));
        None
    }

    fn make_advice_popup(ctx: &HandlerContext, _state: &Self::SelectionState) -> Popup {
        ctx.new_popup()
            .title("Fill tool")
            .label("Tap the tool on a block to set the first corner of the region.")
            .label("Then right-click on the opposite corner to fill the rectangular prism.")
            .label("Hold dig on a block to select it as the fill block.")
            .button("acknowledge", "Got it!", true, true)
    }

    fn should_show_advice(state: &Self::SelectionState) -> bool {
        state.start.is_none()
    }

    fn tap(
        ctx: &HandlerContext,
        coord: PointeeBlockCoords,
        settings: &mut Self::Settings,
        state: &mut Self::SelectionState,
    ) -> Result<()> {
        let block_name = ctx.block_types().human_short_name(BlockId(settings.block));
        ctx.initiator()
            .send_chat_message(ChatMessage::new_server_message(format!(
                "Fill corner set. Will fill with {}",
                block_name
            )))?;
        state.start = Some(coord.selected);
        Ok(())
    }

    fn current_hint(
        settings: &Self::Settings,
        state: &Self::SelectionState,
        ctx: &HandlerContext,
    ) -> Option<ToolHint> {
        Some(ToolHint {
            edit_delta_from: Some(BlockCoordinate::from(state.start?)),
            static_string: format!(
                "Fill with {}",
                ctx.block_types().human_short_name(BlockId(settings.block))
            ),
        })
    }

    fn build(
        ctx: &HandlerContext,
        right_click_coord: PointeeBlockCoords,
        settings: &mut Self::Settings,
        state: &mut Self::SelectionState,
    ) -> Result<Option<BatchedUndo>> {
        let start = state.start.context("No start corner set")?;
        let end = right_click_coord.selected;

        let x_min = start.x.min(end.x);
        let x_max = start.x.max(end.x);
        let y_min = start.y.min(end.y);
        let y_max = start.y.max(end.y);
        let z_min = start.z.min(end.z);
        let z_max = start.z.max(end.z);

        let volume = (x_max - x_min + 1)
            .saturating_mul(y_max - y_min + 1)
            .saturating_mul(z_max - z_min + 1);
        if volume > 1_048_576 {
            bail!("Region too large ({} blocks); maximum is 1048576", volume);
        }

        let block_id = BlockId(settings.block);
        if block_id.equals_ignore_variant(AIR_ID) {
            bail!("No block selected; please hold dig on a block to select it as the fill block.");
        }
        let encodes_placer_group = ctx
            .block_types()
            .fast_block_group(VARIANT_ENCODES_PLACER)
            .context("VARIANT_ENCODES_PLACER fast block group not registered")?;
        let block_id = if encodes_placer_group.contains(block_id) {
            block_id.with_variant_unchecked(blocks::variants::VARIANT_PLACED_AUTOBUILD as u16)
        } else {
            block_id
        };

        let mut writes = BatchedWrite::default();
        for x in x_min..=x_max {
            for y in y_min..=y_max {
                for z in z_min..=z_max {
                    writes.add_block(BlockCoordinate::new(x, y, z), block_id);
                }
            }
        }

        Ok(Some(writes.commit(
            ctx,
            WriteParameters {
                detect_player_placed: false,
                allow_overwrite_autobuilds: true,
                overwrite_failure_action: OverwriteFailureAction::Skip,
                ..Default::default()
            },
        )?))
    }

    const TOOL_ID: &'static str = "autobuild:fill_tool";
}

struct ClearTool;
#[derive(Clone, Debug, Default)]
struct ClearToolSelectionState {
    start: Option<BlockCoordinate>,
}

#[derive(Clone, prost::Message)]
struct ClearToolSettings {
    #[prost(bool, tag = "1")]
    destroy_all: bool,
}

impl Autobuilder for ClearTool {
    type Settings = ClearToolSettings;
    type SelectionState = ClearToolSelectionState;

    fn make_settings_popup(
        ctx: &HandlerContext,
        _coord: PointeeBlockCoords,
        settings: &Self::Settings,
    ) -> Option<Popup> {
        Some(
            ctx.new_popup()
                .title("Clear tool")
                .checkbox("destroy_all", "Destroy all", settings.destroy_all, true)
                .button("save", "Save", true, true)
                .set_button_callback(|resp| {
                    let destroy_all = resp
                        .checkbox_values
                        .get("destroy_all")
                        .copied()
                        .unwrap_or(false);
                    let new_settings = ClearToolSettings { destroy_all };
                    Self::save_settings(&resp.ctx, new_settings, "autobuild:clear_tool")
                }),
        )
    }

    fn make_advice_popup(ctx: &HandlerContext, _state: &Self::SelectionState) -> Popup {
        ctx.new_popup()
            .title("Clear tool")
            .label("Tap the tool on a block to set the first corner of the region.")
            .label("Then right-click on the opposite corner to clear the rectangular prism.")
            .button("acknowledge", "Got it!", true, true)
    }

    const TOOL_ID: &'static str = "autobuild:clear_tool";

    fn should_show_advice(state: &Self::SelectionState) -> bool {
        state.start.is_none()
    }

    fn tap(
        ctx: &HandlerContext,
        coord: PointeeBlockCoords,
        _settings: &mut Self::Settings,
        state: &mut Self::SelectionState,
    ) -> Result<()> {
        ctx.initiator()
            .send_chat_message(ChatMessage::new_server_message("Clear corner set"))?;
        state.start = Some(coord.selected);
        Ok(())
    }

    fn current_hint(
        settings: &Self::Settings,
        state: &Self::SelectionState,

        _ctx: &HandlerContext,
    ) -> Option<ToolHint> {
        Some(ToolHint {
            edit_delta_from: Some(BlockCoordinate::from(state.start?)),
            static_string: if settings.destroy_all {
                "Destroying all blocks!".to_string()
            } else {
                "Clearing naturally spawned and autoplaced blocks".to_string()
            },
        })
    }

    fn build(
        ctx: &HandlerContext,
        right_click_coord: PointeeBlockCoords,
        settings: &mut Self::Settings,
        state: &mut Self::SelectionState,
    ) -> Result<Option<BatchedUndo>> {
        let start = state.start.context("No start corner set")?;
        let end = right_click_coord.selected;

        let x_min = start.x.min(end.x);
        let x_max = start.x.max(end.x);
        let y_min = start.y.min(end.y);
        let y_max = start.y.max(end.y);
        let z_min = start.z.min(end.z);
        let z_max = start.z.max(end.z);

        let volume = (x_max - x_min + 1)
            .saturating_mul(y_max - y_min + 1)
            .saturating_mul(z_max - z_min + 1);
        if volume > 1_048_576 {
            bail!("Region too large ({} blocks); maximum is 1048576", volume);
        }

        let mut writes = BatchedWrite::default();
        for x in x_min..=x_max {
            for y in y_min..=y_max {
                for z in z_min..=z_max {
                    writes.add_block(BlockCoordinate::new(x, y, z), AIR_ID);
                }
            }
        }

        Ok(Some(writes.commit(
            ctx,
            WriteParameters {
                detect_player_placed: !settings.destroy_all,
                allow_overwrite_autobuilds: true,
                overwrite_failure_action: OverwriteFailureAction::Skip,
                ..Default::default()
            },
        )?))
    }
}
