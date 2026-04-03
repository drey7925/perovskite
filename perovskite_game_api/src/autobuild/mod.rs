//! Tools for players to quickly build structures. Initial focus is on roads, to prove
//! out the UX and get some initial experimentation.

use anyhow::{bail, Context, Result};
use perovskite_core::{
    block_id::BlockId, constants::items::default_item_interaction_rules,
    coordinates::BlockCoordinate, protocol::items as items_proto,
};
use perovskite_server::game_state::{
    client_ui::{Popup, TextFieldBuilder, UiElementContainer},
    event::{EventInitiator, HandlerContext},
    items::{Item, ItemInteractionResult, ItemStack, PointeeBlockCoords},
    player::Player,
};

use crate::{
    default_game::block_groups::{NATURAL_AND_STRUCTURAL, NATURAL_GROUND, VARIANT_ENCODES_PLACER},
    game_builder::{GameBuilder, OwnedTextureName},
};

pub fn initialize_autobuild(game: &mut GameBuilder) -> anyhow::Result<()> {
    game.inner
        .blocks_mut()
        .register_fast_block_group(NATURAL_GROUND);
    game.inner
        .blocks_mut()
        .register_fast_block_group(NATURAL_AND_STRUCTURAL);
    game.inner
        .blocks_mut()
        .register_fast_block_group(VARIANT_ENCODES_PLACER);

    let texture = OwnedTextureName::from_css_color("#808080");
    let road_tool = Item {
        place_on_block_handler: Some(Box::new(|ctx, coord, stack| {
            place_on_block_interaction(ctx, coord, stack)
        })),
        dig_handler: Some(Box::new(|ctx, coord, stack| {
            dig_interaction(ctx, coord, stack)
        })),

        ..Item::default_with_proto(items_proto::ItemDef {
            short_name: "autobuild:road_tool".to_string(),
            display_name: "Road tool".to_string(),
            appearance: texture.into(),
            groups: vec![],
            interaction_rules: default_item_interaction_rules(),
            quantity_type: None,
            sort_key: "autobuild:road_tool".to_string(),
        })
    };
    game.inner.items_mut().register_item(road_tool)?;

    Ok(())
}

#[derive(Clone, Debug, Default)]
struct RoadToolSelectionState {
    start: Option<BlockCoordinate>,
}

#[derive(Clone, Debug)]
struct RoadToolSettings {
    block: BlockId,
}
impl Default for RoadToolSettings {
    fn default() -> Self {
        Self { block: BlockId(0) }
    }
}

fn place_on_block_interaction(
    ctx: &HandlerContext,
    coord: PointeeBlockCoords,
    stack: &ItemStack,
) -> Result<ItemInteractionResult> {
    let work = |p: &Player| -> Result<()> {
        let data = p.with_transient_data::<RoadToolSelectionState, _>(|x| Ok(x.clone()))?;
        let settings = p.with_transient_data::<RoadToolSettings, _>(|x| Ok(x.clone()))?;
        if let Some(start) = data.start {
            // Road settings (once implemented) will come from the player's transient data
            ctx.run_deferred(move |ctx: &HandlerContext| {
                build_road(ctx.initiator(), start, coord.selected, settings)
            });
            Ok(())
        } else {
            p.show_popup_blocking(make_advice_popup(ctx))
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
            bail!("Road tool can only be used interactively by players; consider calling road builder functions directly");
        }
    };
    // give the user back their tool
    Ok(Some(stack.clone()).into())
}

fn dig_interaction(
    ctx: &HandlerContext,
    _coord: PointeeBlockCoords,
    stack: &ItemStack,
) -> Result<ItemInteractionResult> {
    let work = |p: &Player| -> Result<()> {
        // todo: move settings to persistent data (and make the underlying type a proto)
        let data = p.with_transient_data::<RoadToolSettings, _>(|x| Ok(x.clone()))?;
        p.show_popup_blocking(make_settings_popup(ctx, data))
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
            bail!("Road tool can only be used interactively by players; consider calling road builder functions directly");
        }
    };
    // give the user back their tool
    Ok(Some(stack.clone()).into())
}

fn make_advice_popup(ctx: &HandlerContext) -> Popup {
    ctx.new_popup()
        .title("Road tool")
        .label("Tap the tool on the ground to set the starting point. Then, right-click to finish building.")
        .label("Hold the dig/tap button to change road settings.")
        .button("acknowledge", "Got it!", true, true)
}

fn make_settings_popup(ctx: &HandlerContext, settings: RoadToolSettings) -> Popup {
    ctx.new_popup()
        .title("Road settings")
        .text_field(
            TextFieldBuilder::new("block_id").label("Block").initial(
                ctx.block_types()
                    .get_block(settings.block)
                    .map_or("???", |(b, _)| b.short_name()),
            ),
        )
        .button("save", "Save", true, true)
        .set_button_callback(|resp| {
            let block = resp.ctx.block_types().get_by_name(&resp.textfield_values.get("block_id").context("Missing block_id")?).context("Block not found")?;
            match resp.ctx.initiator() {
                perovskite_server::game_state::event::EventInitiator::Player(p) => {
                    p.player.with_transient_data::<RoadToolSettings, _>(|x| {
                        x.block = block;
                        Ok(())
                    })?;
                }
                perovskite_server::game_state::event::EventInitiator::WeakPlayerRef(weak) => {
                    match weak.try_to_run(|p| {
                        p.with_transient_data::<RoadToolSettings, _>(|x| {
                            x.block = block;
                            Ok(())
                        })
                    }) {
                        None => {}
                        Some(Ok(_)) => {}
                        Some(Err(e)) => return Err(e),
                    }
                }
                _ => {
                    bail!("Road tool can only be used interactively by players; consider calling road builder functions directly");
                }
            };
            Ok(())
        })
}

fn build_road(
    initiator: &EventInitiator,
    start: BlockCoordinate,
    end: BlockCoordinate,
    settings: RoadToolSettings,
) -> Result<()> {
    dbg!(initiator, start, end, settings);
    Ok(())
}
