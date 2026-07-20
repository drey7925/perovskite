use crate::blocks::variants::rotate_nesw_azimuth_to_variant;
use crate::blocks::{
    AaBoxProperties, AxisAlignedBoxesAppearanceBuilder, BlockBuilder, RotationMode, TextureCropping,
};
use crate::game_builder::{
    GameBuilder, OwnedTextureName, StaticBlockName, StaticItemName, StaticTextureName,
};
use crate::include_texture_bytes;
use anyhow::{bail, ensure, Context};
use cgmath::{vec3, Vector3};
use perovskite_core::block_id::BlockId;
use perovskite_core::constants::item_groups::HIDDEN_FROM_CREATIVE;
use perovskite_core::constants::items::default_item_interaction_rules;
use perovskite_core::constants::CHUNK_SIZE_U8;
use perovskite_core::coordinates::{BlockCoordinate, ChunkCoordinate, ChunkOffset};
use perovskite_core::protocol::items::item_def::QuantityType;
use perovskite_core::protocol::items::ItemDef;
use perovskite_core::protocol::map::ClientExtendedData;
use perovskite_core::protocol::render::{BlockHoverText, RenderedText, RichTextSpan};
use perovskite_core::vertical_occlusion::LightScratchpad;
use perovskite_server::game_state::blocks::{CompassDirection, ExtendedData, FastBlockName};
use perovskite_server::game_state::client_ui::{
    Popup, PopupAction, TextFieldBuilder, UiElementContainer,
};
use perovskite_server::game_state::event::{EventInitiator, HandlerContext};
use perovskite_server::game_state::game_map::timers::{
    BulkUpdateCallback, ChunkNeighbors, TimerCallback, TimerSettings, TimerState,
};
use perovskite_server::game_state::game_map::MapChunk;
use perovskite_server::game_state::items::Item;
use std::time::Duration;

const SIGN_TEXTURE_TEMP: StaticTextureName = StaticTextureName("default:sign_tmp");
const SIGN_ITEM_TEX: StaticTextureName = StaticTextureName("default:sign_item");

const TEXT_KEY: &str = "sign_text";
const LARGE_TEXT_VARIANT_BIT: u16 = 4;
const WALL_SIGN: StaticBlockName = StaticBlockName("default:sign_wall");
const STANDING_SIGN: StaticBlockName = StaticBlockName("default:sign_standing");
const LIGHTPROBE: StaticBlockName = StaticBlockName("default:testonly_lightprobe");
const SIGN_ITEM: StaticItemName = StaticItemName("default:wooden_sign");

#[derive(Clone, Copy, Debug)]
struct TextPlacement {
    u: Vector3<f32>,
    v: Vector3<f32>,
    offset: Vector3<f32>,
    u_texels: i32,
    v_texels: i32,
    u_texels_large: i32,
    v_texels_large: i32,
}
impl TextPlacement {
    fn rotate(self, d: CompassDirection) -> Self {
        TextPlacement {
            u: d.rotate_vec(self.u),
            v: d.rotate_vec(self.v),
            offset: d.rotate_vec(self.offset),
            u_texels: self.u_texels,
            v_texels: self.v_texels,
            u_texels_large: self.u_texels_large,
            v_texels_large: self.v_texels_large,
        }
    }
}

pub(crate) fn register_sign(game_builder: &mut GameBuilder) -> anyhow::Result<()> {
    include_texture_bytes!(game_builder, SIGN_TEXTURE_TEMP, "textures/maple_planks.png")?;
    include_texture_bytes!(game_builder, SIGN_ITEM_TEX, "textures/testonly_sign.png")?;

    let mut wall_sign = BlockId(0);
    let mut standing_sign = BlockId(0);
    let mut lightprobe = BlockId(0);
    for (name, aa_box, id_mut, text_placement) in [
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
            TextPlacement {
                offset: vec3(-0.4, 0.45, 0.3875),
                u: vec3(0.8, 0.0, 0.0),
                v: vec3(0.0, -0.75, 0.0),
                u_texels: 320,
                v_texels: 256,
                u_texels_large: 80,
                v_texels_large: 64,
            },
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
            TextPlacement {
                offset: vec3(-0.4, 0.45, -0.3125),
                u: vec3(0.8, 0.0, 0.0),
                v: vec3(0.0, -0.75, 0.0),
                u_texels: 320,
                v_texels: 256,
                u_texels_large: 80,
                v_texels_large: 64,
            },
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
            TextPlacement {
                offset: vec3(-0.4, 0.45, -0.3125),
                u: vec3(0.8, 0.0, 0.0),
                v: vec3(0.0, -0.75, 0.0),
                u_texels: 320,
                v_texels: 256,
                u_texels_large: 80,
                v_texels_large: 64,
            },
        ),
    ] {
        let block = game_builder.add_block(
            BlockBuilder::new(name)
                .set_axis_aligned_boxes_appearance(aa_box)
                .set_display_name("Wooden Sign")
                .add_item_group(HIDDEN_FROM_CREATIVE)
                .set_simple_dropped_item(SIGN_ITEM.0, 1)
                .set_allow_light_propagation(true)
                .set_allow_weather_propagation(true)
                .add_interact_key_menu_entry("", "Set text...")
                .add_modifier(move |bt| {
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
                    bt.make_client_extended_data = Some(Box::new(move |id, ext_data| {
                        let sign_text = ext_data
                            .simple_data
                            .get(TEXT_KEY)
                            .map(|x| BlockHoverText {
                                text: x.to_string(),
                            })
                            .and_then(|x| if x.text.is_empty() { None } else { Some(x) });
                        let placement = text_placement
                            .rotate(CompassDirection::from_rotation_variant(id.variant()));
                        let large = (id.variant() & LARGE_TEXT_VARIANT_BIT) != 0;
                        Ok(Some(ClientExtendedData {
                            offset_in_chunk: 0,
                            block_text: sign_text.clone().into_iter().collect(),
                            rendered_text: sign_text
                                .map(|x| RenderedText {
                                    spans: vec![RichTextSpan {
                                        text: x.text.clone(),
                                        texel_height: 64.0,
                                        color_rgb: 0x00000000,
                                        emissive_color_rgb: 0x80001c18,
                                    }],
                                    top_left_corner: Some(placement.offset.try_into().unwrap()),
                                    u_extent: Some(placement.u.try_into().unwrap()),
                                    v_extent: Some(placement.v.try_into().unwrap()),
                                    u_texels: if large {
                                        placement.u_texels_large
                                    } else {
                                        placement.u_texels
                                    },
                                    v_texels: if large {
                                        placement.v_texels_large
                                    } else {
                                        placement.v_texels
                                    },
                                })
                                .into_iter()
                                .collect(),
                            ..Default::default()
                        }))
                    }))
                }),
        )?;
        *id_mut = block.id;
    }
    game_builder.inner.items_mut().register_item(Item {
        place_on_block_handler: Some(Box::new(move |ctx, coord, stack| {
            let preceding_coord = match coord.preceding {
                Some(x) => x,
                None => return Ok(Some(stack.clone()).into()),
            };
            let new_block = if coord.selected.try_delta(0, 1, 0) == Some(preceding_coord) {
                standing_sign
            } else {
                wall_sign
            }
            .with_variant_unchecked(rotate_nesw_azimuth_to_variant(
                ctx.initiator_state()
                    .position()
                    .map(|x| x.face_direction.0)
                    .unwrap_or(0.0),
            ));
            let placed =
                ctx.game_map()
                    .mutate_block_atomically(preceding_coord, |block, _ext| {
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
            display_name: "Light probe (test only)".to_string(),
            appearance: SIGN_ITEM_TEX.into(),
            groups: vec![],
            interaction_rules: default_item_interaction_rules(),
            sort_key: "default:sign:wood".to_string(),
            quantity_type: Some(QuantityType::Stack(256)),
            tool_range: 6.0,
        })
    })?;

    game_builder.inner.add_timer(
        "default:tmp_lightprobe",
        TimerSettings {
            interval: Duration::from_secs(1),
            spreading: 1.0,
            shards: 1,
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
        for x in 0..CHUNK_SIZE_U8 {
            for z in 0..CHUNK_SIZE_U8 {
                for y in 0..CHUNK_SIZE_U8 {
                    let offset = ChunkOffset::new(x, y, z);
                    let block = chunk.get_block(offset);
                    if block.equals_ignore_variant(self.0) {
                        let mut data = ExtendedData::default();
                        let light = lights.get_packed_u4_u4(x as i32, y as i32, z as i32);
                        let upper = light >> 4;
                        let lower = light & 0xF;
                        let formatted = format!("upper: {}, lower: {}", upper, lower);
                        data.simple_data.insert(TEXT_KEY.to_string(), formatted);
                        chunk.set_block(offset, block, Some(data));
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
        .get_block_with_extended_data(coord, |_, ext| Ok(ext.simple_data.get(TEXT_KEY).cloned()))?;
    let id_standing = ctx
        .block_types()
        .resolve_name(fbn_standing)
        .context("Missing standing sign")?;
    let id_wall = ctx
        .block_types()
        .resolve_name(fbn_wall)
        .context("Missing wall sign")?;
    let id_lightprobe = ctx
        .block_types()
        .resolve_name(fbn_lightprobe)
        .context("Missing lightprobe sign")?;
    if id.equals_ignore_variant(id_standing)
        || id.equals_ignore_variant(id_wall)
        || id.equals_ignore_variant(id_lightprobe)
    {
        Ok(Some(
            ctx.new_popup()
                .title("Sign")
                .text_field(
                    TextFieldBuilder::new(TEXT_KEY)
                        .label("Text")
                        .multiline(true)
                        .initial(initial.unwrap_or_else(String::new)),
                )
                .checkbox(
                    "large",
                    "Large text",
                    (id.variant() & LARGE_TEXT_VARIANT_BIT) != 0,
                    true,
                )
                .button("save", "Save", true, true)
                .set_button_callback(move |resp| {
                    match resp.user_action {
                        PopupAction::PopupClosed => {
                            // do nothing, escape key hit
                            return Ok(());
                        }
                        PopupAction::ButtonClicked(btn) => {
                            ensure!(btn == "save");
                        }
                    }
                    let text = resp.textfield_values.get(TEXT_KEY);
                    let large = resp.checkbox_values.get("large").copied().unwrap_or(false);
                    resp.ctx
                        .game_map()
                        .mutate_block_atomically(coord, |block, ext| {
                            if !block.equals_ignore_variant(id_standing)
                                && !block.equals_ignore_variant(id_wall)
                                && !block.equals_ignore_variant(id_lightprobe)
                            {
                                bail!("Sign disappeared while editing");
                            }
                            let ext_inner = ext.get_or_insert_with(Default::default);
                            ext_inner.simple_data.insert(
                                TEXT_KEY.to_string(),
                                text.cloned().unwrap_or_else(String::new),
                            );

                            if large {
                                *block = block.or_variant(LARGE_TEXT_VARIANT_BIT);
                            } else {
                                *block = block.clear_variant_bits(LARGE_TEXT_VARIANT_BIT);
                            }

                            Ok(())
                        })
                }),
        ))
    } else {
        Ok(None)
    }
}
