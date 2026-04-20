use crate::game_ui::UNKNOWN_TEXTURE;
use crate::game_ui::egui_ui::RefinementsCtx;
use egui::load::SizedTexture;
use egui::Context;
use itertools::Itertools;
use perovskite_core::block_id::BlockId;
use perovskite_core::protocol::blocks::BlockTypeDef;
use perovskite_core::protocol::items::ItemDef;
use std::ops::ControlFlow;

use super::RefinementState;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
enum CategorySelection {
    #[default]
    None,
    Group(String),
    PluginPrefix(String),
}
impl CategorySelection {
    fn make_filter<T: RefinementItem>(&self) -> Box<dyn Fn(&T) -> bool> {
        match self {
            CategorySelection::None => Box::new(|_| true),
            CategorySelection::Group(group) => {
                let group = group.clone();
                Box::new(move |item| item.groups().contains(&group))
            }
            CategorySelection::PluginPrefix(prefix) => {
                let prefix_with_colon = prefix.to_string() + ":";
                Box::new(move |item| item.short_name().starts_with(&prefix_with_colon))
            }
        }
    }
}

#[derive(Default, Debug)]
pub(crate) struct CategoryPickerRefinement<T: RefinementItem> {
    state: PickerState,
    _phantom: std::marker::PhantomData<fn(T)>,
}

impl<T: RefinementItem> RefinementState for CategoryPickerRefinement<T> {
    fn draw(
        &mut self,
        _provoking_response: &egui::Response,
        ctx: &Context,
        refinements_ctx: &RefinementsCtx<'_>,
        base_id: egui::Id,
    ) -> ControlFlow<Option<String>> {
        draw_refinement_picker::<T>(&mut self.state, ctx, refinements_ctx, base_id)
    }

    fn open_button_text(&self) -> &str {
        "🔎"
    }
}

#[derive(Default, Debug)]
struct PickerState {
    selected_category: CategorySelection,
    selected_item: Option<String>,
    typing_filter: String,
}

pub(crate) trait RefinementItem: Sized {
    fn short_name(&self) -> &str;
    fn groups(&self) -> &[String];
    fn descriptive_name(&self) -> &str;
    fn render_details(&self, ui: &mut egui::Ui, ctx: &RefinementsCtx<'_>);

    /// Returns all items; `'a` is the lifetime of the data held inside `RefinementsCtx`.
    fn get_all_items<'a>(refinements_ctx: &RefinementsCtx<'a>) -> impl Iterator<Item = &'a Self>
    where
        Self: 'a;
    fn get_all_groups<'a>(refinements_ctx: &RefinementsCtx<'a>) -> &'a [String];
    fn get_all_plugin_prefixes<'a>(refinements_ctx: &RefinementsCtx<'a>) -> &'a [String];
    fn get_by_name<'a>(refinements_ctx: &RefinementsCtx<'a>, name: &str) -> Option<&'a Self>;
}
impl RefinementItem for ItemDef {
    fn short_name(&self) -> &str {
        &self.short_name
    }
    fn groups(&self) -> &[String] {
        &self.groups
    }
    fn descriptive_name(&self) -> &str {
        &self.display_name
    }
    fn render_details(&self, ui: &mut egui::Ui, ctx: &RefinementsCtx<'_>) {
        let fallback = ctx.atlas.coords.get(UNKNOWN_TEXTURE).copied();
        if let Some(pixel_rect) = ctx
            .atlas
            .coords
            .get(&self.short_name)
            .copied()
            .or(fallback)
        {
            let uv = ctx.atlas.egui_uv(pixel_rect);
            let image = egui::Image::from_texture(SizedTexture {
                id: ctx.atlas_texture_id,
                size: egui::vec2(64.0, 64.0),
            })
            .uv(uv);
            ui.add(image);
        }
        ui.label(&self.display_name);
    }

    fn get_all_items<'a>(refinements_ctx: &RefinementsCtx<'a>) -> impl Iterator<Item = &'a Self>
    where
        Self: 'a,
    {
        refinements_ctx.client_state.items.sorted_items()
    }
    fn get_all_groups<'a>(refinements_ctx: &RefinementsCtx<'a>) -> &'a [String] {
        refinements_ctx.client_state.items.groups()
    }
    fn get_all_plugin_prefixes<'a>(refinements_ctx: &RefinementsCtx<'a>) -> &'a [String] {
        refinements_ctx.client_state.items.plugin_prefixes()
    }
    fn get_by_name<'a>(refinements_ctx: &RefinementsCtx<'a>, name: &str) -> Option<&'a Self> {
        refinements_ctx.client_state.items.get(name)
    }
}
impl RefinementItem for BlockTypeDef {
    fn short_name(&self) -> &str {
        &self.short_name
    }
    fn groups(&self) -> &[String] {
        &self.groups
    }
    fn descriptive_name(&self) -> &str {
        &self.short_name
    }

    fn get_all_items<'a>(refinements_ctx: &RefinementsCtx<'a>) -> impl Iterator<Item = &'a Self>
    where
        Self: 'a,
    {
        refinements_ctx
            .client_state
            .block_types
            .sorted_blocks_by_name()
    }
    fn get_all_groups<'a>(refinements_ctx: &RefinementsCtx<'a>) -> &'a [String] {
        refinements_ctx.client_state.block_types.groups()
    }
    fn get_all_plugin_prefixes<'a>(refinements_ctx: &RefinementsCtx<'a>) -> &'a [String] {
        refinements_ctx.client_state.block_types.plugin_prefixes()
    }
    fn get_by_name<'a>(refinements_ctx: &RefinementsCtx<'a>, name: &str) -> Option<&'a Self> {
        let block_id = refinements_ctx
            .client_state
            .block_types
            .get_block_by_name(name)?;
        refinements_ctx
            .client_state
            .block_types
            .get_blockdef(block_id)
    }
    fn render_details(&self, ui: &mut egui::Ui, _ctx: &RefinementsCtx<'_>) {
        ui.label(format!("Block ID: {:?}", BlockId(self.id)));
        ui.label(format!(
            "Groups: {}",
            self.groups.iter().sorted().join(", ")
        ));
    }
}

fn draw_refinement_picker<T: RefinementItem>(
    state: &mut PickerState,
    ctx: &Context,
    refinements_ctx: &RefinementsCtx<'_>,
    base_id: egui::Id,
) -> ControlFlow<Option<String>> {
    let response = egui::Modal::new(base_id.with("refinement_picker")).show(ctx, |ui| {
        ui.set_min_width(480.0);
        if ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
            return ControlFlow::Break(state.selected_item.clone());
        }
        let bottom_panel_response = egui::TopBottomPanel::bottom("refinement_picker_bottom")
            .show_inside(ui, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("OK").clicked() {
                        return ControlFlow::Break(state.selected_item.clone());
                    } else if ui.button("Cancel").clicked() {
                        return ControlFlow::Break(None);
                    } else {
                        return ControlFlow::Continue(());
                    }
                })
                .inner
            });
        if bottom_panel_response.inner.is_break() {
            return bottom_panel_response.inner;
        }

        egui::TopBottomPanel::top("refinement_picker_top").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Filter");
                ui.text_edit_singleline(&mut state.typing_filter);
            });
        });

        egui::SidePanel::left("refinement_picker_categories")
            .default_width(120.0)
            .min_width(80.0)
            .show_inside(ui, |ui| {
                // Render category picker
                egui::ScrollArea::both()
                    .min_scrolled_width(80.0)
                    .max_width(f32::INFINITY)
                    .show(ui, |ui| {
                        egui::Grid::new(base_id.with("category_grid"))
                            .num_columns(1)
                            .striped(true)
                            .show(ui, |ui| {
                                if ui
                                    .selectable_value(
                                        &mut state.selected_category,
                                        CategorySelection::None,
                                        "All items",
                                    )
                                    .clicked()
                                {
                                    state.selected_category = CategorySelection::None;
                                    state.typing_filter.clear();
                                }
                                ui.end_row();

                                for group in T::get_all_groups(refinements_ctx) {
                                    if ui
                                        .selectable_value(
                                            &mut state.selected_category,
                                            CategorySelection::Group(group.to_string()),
                                            format!("[group] {}", group),
                                        )
                                        .clicked()
                                    {
                                        state.selected_category =
                                            CategorySelection::Group(group.to_string());
                                        state.typing_filter.clear();
                                    }
                                    ui.end_row();
                                }
                                for prefix in T::get_all_plugin_prefixes(refinements_ctx) {
                                    if ui
                                        .selectable_value(
                                            &mut state.selected_category,
                                            CategorySelection::PluginPrefix(prefix.to_string()),
                                            format!("[plugin] {}", prefix),
                                        )
                                        .clicked()
                                    {
                                        state.selected_category =
                                            CategorySelection::PluginPrefix(prefix.to_string());
                                        state.typing_filter.clear();
                                    }
                                    ui.end_row();
                                }
                            });
                    });
            });

        egui::SidePanel::right("refinement_picker_item_details")
            .default_width(160.0)
            .min_width(160.0)
            .show_inside(ui, |ui| {
                if let Some(item_name) = &state.selected_item {
                    if let Some(item) = T::get_by_name(refinements_ctx, item_name) {
                        item.render_details(ui, refinements_ctx);
                    } else {
                        ui.label("Unknown selection");
                    }
                } else {
                    ui.label("Nothing selected");
                }
            });

        egui::CentralPanel::default()
            .show_inside(ui, |ui| {
                // Render item list
                egui::ScrollArea::vertical()
                    .max_width(f32::INFINITY)
                    .auto_shrink(false)
                    .min_scrolled_height(320.0)
                    .show(ui, |ui| {
                        egui::Grid::new(base_id.with("item_grid"))
                            .num_columns(1)
                            .striped(true)
                            .show(ui, |ui| {
                                let filter = state.selected_category.make_filter::<T>();
                                for item in
                                    T::get_all_items(refinements_ctx).filter(|item| {
                                        filter(item)
                                            && item.short_name().contains(&state.typing_filter)
                                    })
                                {
                                    let response = ui
                                        .selectable_value(
                                            &mut state.selected_item,
                                            Some(item.short_name().to_string()),
                                            item.descriptive_name(),
                                        )
                                        .interact(egui::Sense::click());
                                    if response.double_clicked() {
                                        return ControlFlow::Break(Some(
                                            item.short_name().to_string(),
                                        ));
                                    } else if response.clicked() {
                                        state.selected_item = Some(item.short_name().to_string());
                                    }
                                    ui.end_row();
                                }
                                ControlFlow::Continue(())
                            })
                            .inner
                    })
                    .inner
            })
            .inner
    });
    response.inner
}
