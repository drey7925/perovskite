use crate::client_state::ClientState;
use crate::game_ui::egui_ui::RefinementState;
use egui::Context;
use perovskite_core::protocol::blocks::BlockTypeDef;
use perovskite_core::protocol::items::ItemDef;
use std::ops::ControlFlow;

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
        client_state: &ClientState,
        base_id: egui::Id,
    ) -> ControlFlow<Option<String>> {
        draw_refinement_picker::<T>(&mut self.state, ctx, client_state, base_id)
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
    fn render_details(&self, ui: &mut egui::Ui, client_state: &ClientState);

    fn get_all_items<'a>(client_state: &'a ClientState) -> impl Iterator<Item = &'a Self>
    where
        Self: 'a;
    fn get_all_groups(client_state: &ClientState) -> &[String];
    fn get_all_plugin_prefixes(client_state: &ClientState) -> &[String];
    fn get_by_name<'a>(client_state: &'a ClientState, name: &str) -> Option<&'a Self>;
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
    fn render_details(&self, ui: &mut egui::Ui, client_state: &ClientState) {
        ui.label("TODO: More info...");
    }

    fn get_all_items<'a>(client_state: &'a ClientState) -> impl Iterator<Item = &'a Self>
    where
        Self: 'a,
    {
        client_state.items.sorted_items()
    }
    fn get_all_groups(client_state: &ClientState) -> &[String] {
        client_state.items.groups()
    }
    fn get_all_plugin_prefixes(client_state: &ClientState) -> &[String] {
        client_state.items.plugin_prefixes()
    }
    fn get_by_name<'a>(client_state: &'a ClientState, name: &str) -> Option<&'a Self> {
        client_state.items.get(name)
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

    fn get_all_items<'a>(client_state: &'a ClientState) -> impl Iterator<Item = &'a Self>
    where
        Self: 'a,
    {
        client_state.block_types.sorted_blocks_by_name()
    }
    fn get_all_groups(client_state: &ClientState) -> &[String] {
        client_state.block_types.groups()
    }
    fn get_all_plugin_prefixes(client_state: &ClientState) -> &[String] {
        client_state.block_types.plugin_prefixes()
    }
    fn get_by_name<'a>(client_state: &'a ClientState, name: &str) -> Option<&'a Self> {
        let block_id = client_state.block_types.get_block_by_name(name)?;
        client_state.block_types.get_blockdef(block_id)
    }
    fn render_details(&self, ui: &mut egui::Ui, _client_state: &ClientState) {
        ui.label(format!("Block ID: {:?}", self.id));
        ui.label("TODO: More info...");
    }
}

fn draw_refinement_picker<T: RefinementItem>(
    state: &mut PickerState,
    ctx: &Context,
    client_state: &ClientState,
    base_id: egui::Id,
) -> ControlFlow<Option<String>> {
    let response = egui::Modal::new(base_id.with("refinement_picker")).show(ctx, |ui| {
        ui.set_min_width(320.0);
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

        egui::SidePanel::left("refinement_picker_categories").show_inside(ui, |ui| {
            // Render category picker
            egui::ScrollArea::vertical().show(ui, |ui| {
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

                        for group in T::get_all_groups(client_state) {
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
                        for prefix in T::get_all_plugin_prefixes(client_state) {
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
            .min_width(160.0)
            .show_inside(ui, |ui| {
                if let Some(item_name) = &state.selected_item {
                    if let Some(item) =
                        T::get_all_items(client_state).find(|item| item.short_name() == item_name)
                    {
                        item.render_details(ui, client_state);
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
                    .show(ui, |ui| {
                        egui::Grid::new(base_id.with("item_grid"))
                            .num_columns(1)
                            .striped(true)
                            .show(ui, |ui| {
                                let filter = state.selected_category.make_filter::<T>();
                                for item in T::get_all_items(client_state).filter(|item| {
                                    filter(item) && item.short_name().contains(&state.typing_filter)
                                }) {
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
