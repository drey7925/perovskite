use crate::blocks::variants::rotate_nesw_azimuth_to_variant;
use crate::blocks::{
    AaBoxProperties, AxisAlignedBoxesAppearanceBuilder, BlockBuilder, RotationMode, TextureCropping,
};
use crate::game_builder::{
    GameBuilder, OwnedTextureName, StaticBlockName, StaticItemName, StaticTextureName,
};
use crate::include_texture_bytes;
use anyhow::{bail, ensure, Context};
use perovskite_core::block_id::BlockId;
use perovskite_core::constants::item_groups::HIDDEN_FROM_CREATIVE;
use perovskite_core::constants::items::default_item_interaction_rules;
use perovskite_core::coordinates::{BlockCoordinate, ChunkCoordinate, ChunkOffset};
use perovskite_core::lighting::LightScratchpad;
use perovskite_core::protocol::items::item_def::QuantityType;
use perovskite_core::protocol::items::ItemDef;
use perovskite_core::protocol::map::ClientExtendedData;
use perovskite_core::protocol::render::BlockText;
use perovskite_server::game_state::blocks::{ExtendedData, FastBlockName};
use perovskite_server::game_state::client_ui::{
    Popup, PopupAction, TextFieldBuilder, UiElementContainer,
};
use perovskite_server::game_state::event::{EventInitiator, HandlerContext};
use perovskite_server::game_state::game_map::{
    BulkUpdateCallback, ChunkNeighbors, MapChunk, TimerCallback, TimerSettings, TimerState,
};
use perovskite_server::game_state::items::Item;
use std::time::Duration;

const SIGN_TEXTURE_TEMP: StaticTextureName = StaticTextureName("default:sign_tmp");
const SIGN_ITEM_TEX: StaticTextureName = StaticTextureName("default:sign_item");

const TEXT_KEY: &str = "sign_text";
const WALL_SIGN: StaticBlockName = StaticBlockName("default:sign_wall");
const STANDING_SIGN: StaticBlockName = StaticBlockName("default:sign_standing");
const LIGHTPROBE: StaticBlockName = StaticBlockName("default:testonly_lightprobe");
const SIGN_ITEM: StaticItemName = StaticItemName("default:wooden_sign");

pub(crate) fn register_sign(game_builder: &mut GameBuilder) -> anyhow::Result<()> {
    include_texture_bytes!(game_builder, SIGN_TEXTURE_TEMP, "textures/maple_planks.png")?;
    include_texture_bytes!(game_builder, SIGN_ITEM_TEX, "textures/testonly_sign.png")?;

    let mut wall_sign = BlockId(0);
    let mut standing_sign = BlockId(0);
    let mut lightprobe = BlockId(0);
    for (name, aa_box, id_mut) in [
        (
            WALL_SIGN,
            AxisAlignedBoxesAppearanceBuilder::new().add_box(
                AaBoxProperties::new_single_tex(
                    SIGN_TEXTURE_TEMP,
                    TextureCropping::NoCrop,
                    RotationMode::RotateHorizontally,
                ),
                (-0.4, 0.4),
                (-0.3, 0.45),
                (0.4, 0.5),
            ),
            &mut wall_sign,
        ),
        (
            STANDING_SIGN,
            AxisAlignedBoxesAppearanceBuilder::new()
                .add_box(
                    AaBoxProperties::new_single_tex(
                        SIGN_TEXTURE_TEMP,
                        TextureCropping::NoCrop,
                        RotationMode::RotateHorizontally,
                    ),
                    (-0.4, 0.4),
                    (-0.3, 0.45),
                    (-0.3, -0.2),
                )
                .add_box(
                    AaBoxProperties::new_single_tex(
                        SIGN_TEXTURE_TEMP,
                        TextureCropping::NoCrop,
                        RotationMode::RotateHorizontally,
                    ),
                    (-0.05, 0.05),
                    (-0.5, 0.475),
                    (-0.2, -0.15),
                ),
            &mut standing_sign,
        ),
        (
            LIGHTPROBE,
            AxisAlignedBoxesAppearanceBuilder::new()
                .add_box(
                    AaBoxProperties::new_single_tex(
                        OwnedTextureName::from_css_color("green"),
                        TextureCropping::NoCrop,
                        RotationMode::RotateHorizontally,
                    ),
                    (-0.4, 0.4),
                    (-0.3, 0.45),
                    (-0.3, -0.2),
                )
                .add_box(
                    AaBoxProperties::new_single_tex(
                        OwnedTextureName::from_css_color("green"),
                        TextureCropping::NoCrop,
                        RotationMode::RotateHorizontally,
                    ),
                    (-0.05, 0.05),
                    (-0.5, 0.475),
                    (-0.2, -0.15),
                ),
            &mut lightprobe,
        ),
    ] {
        let block = game_builder.add_block(
            BlockBuilder::new(name)
                .set_axis_aligned_boxes_appearance(aa_box)
                .set_display_name("Wooden Sign")
                .add_item_group(HIDDEN_FROM_CREATIVE)
                .set_simple_dropped_item(SIGN_ITEM.0, 1)
                .set_allow_light_propagation(true)
                .add_interact_key_menu_entry("", "Set text...")
                .add_modifier(|bt| {
                    let fbn_standing = FastBlockName::new(STANDING_SIGN.0);
                    let fbn_wall = FastBlockName::new(WALL_SIGN.0);
                    let fbn_lightprobe = FastBlockName::new(LIGHTPROBE.0);
                    bt.interact_key_handler =
                        Some(Box::new(move |ctx, coord, _| match ctx.initiator() {
                            EventInitiator::Player(_p) => Ok(make_sign_popup(
                                &ctx,
                                coord,
                                &fbn_standing,
                                &fbn_wall,
                                &fbn_lightprobe,
                            )?),
                            _ => Ok(None),
                        }));
                    bt.client_info.has_client_extended_data = true;
                    bt.make_client_extended_data = Some(Box::new(|_, ext_data| {
                        Ok(Some(ClientExtendedData {
                            offset_in_chunk: 0,
                            block_text: ext_data
                                .simple_data
                                .get(TEXT_KEY)
                                .map(|x| BlockText {
                                    text: x.to_string(),
                                })
                                .into_iter()
                                .collect(),
                        }))
                    }))
                }),
        )?;
        *id_mut = block.id;
    }
    game_builder.inner.items_mut().register_item(Item {
        place_on_block_handler: Some(Box::new(move |ctx, coord, stack| {
            let new_block = if coord.selected.try_delta(0, 1, 0) == coord.preceding {
                standing_sign
            } else {
                wall_sign
            }
            .with_variant_unchecked(rotate_nesw_azimuth_to_variant(
                ctx.initiator()
                    .position()
                    .map(|x| x.face_direction.0)
                    .unwrap_or(0.0),
            ));
            let placed =
                ctx.game_map()
                    .mutate_block_atomically(coord.selected, |block, _ext| {
                        if ctx.block_types().is_trivially_replaceable(*block) {
                            *block = new_block;
                            Ok(true)
                        } else {
                            Ok(false)
                        }
                    })?;
            if placed {
                Ok(stack.decrement().into())
            } else {
                Ok(Some(stack.clone()).into())
            }
        })),
        ..Item::default_with_proto(ItemDef {
            short_name: SIGN_ITEM.0.to_string(),
            display_name: "Wooden Sign".to_string(),
            appearance: SIGN_ITEM_TEX.into(),
            groups: vec![],
            interaction_rules: default_item_interaction_rules(),
            sort_key: "default:sign:wood".to_string(),
            quantity_type: Some(QuantityType::Stack(256)),
        })
    })?;

    game_builder.inner.add_timer(
        "default:tmp_lightprobe",
        TimerSettings {
            interval: Duration::from_secs(1),
            spreading: 1.0,
            block_types: vec![lightprobe],
            idle_chunk_after_unchanged: false,
            populate_lighting: true,
            ..Default::default()
        },
        TimerCallback::BulkUpdateWithNeighbors(Box::new(LightprobeTimer(lightprobe))),
    );

    Ok(())
}

struct LightprobeTimer(BlockId);
impl BulkUpdateCallback for LightprobeTimer {
    fn bulk_update_callback(
        &self,
        _ctx: &HandlerContext<'_>,
        _chunk_coordinate: ChunkCoordinate,
        _timer_state: &TimerState,
        chunk: &mut MapChunk,
        _neighbors: Option<&ChunkNeighbors>,
        lights: Option<&LightScratchpad>,
    ) -> anyhow::Result<()> {
        let lights = lights.context("Lightprobe timer callback missing lights")?;
        for x in 0..16 {
            for z in 0..16 {
                for y in 0..16 {
                    let offset = ChunkOffset::new(x, y, z);
                    let block = chunk.get_block(offset);
                    if block.equals_ignore_variant(self.0) {
                        let mut data = ExtendedData::default();
                        let light = lights.get_packed_u4_u4(x as i32, y as i32, z as i32);
                        let upper = light >> 4;
                        let lower = light & 0xF;
                        let formatted = format!("upper: {}, lower: {}", upper, lower);
                        data.simple_data.insert(TEXT_KEY.to_string(), formatted);
                        chunk.set_block(offset, block, Some(data))
                    }
                }
            }
        }
        Ok(())
    }
}

fn make_sign_popup(
    ctx: &HandlerContext,
    coord: BlockCoordinate,
    fbn_standing: &FastBlockName,
    fbn_wall: &FastBlockName,
    fbn_lightprobe: &FastBlockName,
) -> anyhow::Result<Option<Popup>> {
    let (id, initial) = ctx
        .game_map()
        .get_block_with_extended_data(coord, |ext| Ok(ext.simple_data.get(TEXT_KEY).cloned()))?;
    if id.equals_ignore_variant(
        ctx.block_types()
            .resolve_name(fbn_standing)
            .context("Missing standing sign")?,
    ) || id.equals_ignore_variant(
        ctx.block_types()
            .resolve_name(fbn_wall)
            .context("Missing wall sign")?,
    ) || id.equals_ignore_variant(
        ctx.block_types()
            .resolve_name(fbn_lightprobe)
            .context("Missing lightprobe sign")?,
    ) {
        Ok(Some(
            ctx.new_popup()
                .title("Sign")
                .text_field(
                    TextFieldBuilder::new(TEXT_KEY)
                        .label("Text")
                        .multiline(true)
                        .initial(initial.unwrap_or_else(String::new)),
                )
                .button("save", "Save", true, true)
                .set_button_callback(move |resp| {
                    match resp.user_action {
                        PopupAction::PopupClosed => {
                            bail!("Button callback, but no button clicked");
                        }
                        PopupAction::ButtonClicked(btn) => {
                            ensure!(btn == "save");
                        }
                    }
                    let text = resp.textfield_values.get(TEXT_KEY);
                    resp.ctx
                        .game_map()
                        .mutate_block_atomically(coord, |_block, ext| {
                            let ext_inner = ext.get_or_insert_with(Default::default);
                            ext_inner.simple_data.insert(
                                TEXT_KEY.to_string(),
                                text.cloned().unwrap_or_else(String::new),
                            );
                            Ok(())
                        })
                }),
        ))
    } else {
        Ok(None)
    }
}
