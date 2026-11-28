// Copyright 2026 drey7925
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

//! Architectural block-diagram loading screen for client initialization.
//!
//! Tracks each client subsystem as it transitions from `Pending` -> `Initializing` (amber) ->
//! `Initialized` (green) and renders a live schematic block diagram with data-flow traces as
//! the loading screen.

// Authorship disclosure: This was produced by Gemini Neon in Antigravity, then simplified by
// hand. An offline renderer was used by both Gemini and myself to quickly validate the appeaance
// of the screen.

use std::time::Instant;

use egui::{Align2, Color32, CornerRadius, FontId, Key, Painter, Pos2, Rect, Sense, Stroke, Vec2};
use enum_map::{enum_map, Enum, EnumMap};

/// Client subsystems initialized during server connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Enum)]
pub(crate) enum LoadingComponent {
    GrpcTransport,
    AuthHandshake,
    MediaCache,
    BlockDefs,
    BlockRenderer,
    ChunkBuffers,
    ItemDefs,
    GameUi,
    EntityRenderer,
    AudioEngine,
    FarGeometry,
    ClientStateBus,
    WorldSync,
}

impl LoadingComponent {
    pub(crate) const ALL: [LoadingComponent; 13] = [
        LoadingComponent::GrpcTransport,
        LoadingComponent::AuthHandshake,
        LoadingComponent::MediaCache,
        LoadingComponent::BlockDefs,
        LoadingComponent::BlockRenderer,
        LoadingComponent::ChunkBuffers,
        LoadingComponent::ItemDefs,
        LoadingComponent::GameUi,
        LoadingComponent::EntityRenderer,
        LoadingComponent::AudioEngine,
        LoadingComponent::FarGeometry,
        LoadingComponent::ClientStateBus,
        LoadingComponent::WorldSync,
    ];

    pub(crate) fn title(self) -> &'static str {
        match self {
            Self::GrpcTransport => "Network connection",
            Self::AuthHandshake => "Login",
            Self::MediaCache => "Media Cache",
            Self::BlockDefs => "Block Definitions",
            Self::BlockRenderer => "Block Renderer",
            Self::ChunkBuffers => "Chunk & RT Buffers",
            Self::ItemDefs => "Item Registry",
            Self::GameUi => "HUD & Game UI",
            Self::EntityRenderer => "Entity Renderer",
            Self::AudioEngine => "Audio Engine",
            Self::FarGeometry => "Far Geometry",
            Self::ClientStateBus => "Client State Bus",
            Self::WorldSync => "Initial World Sync",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ComponentStatus {
    Pending,
    Initializing,
    Initialized,
}

#[derive(Debug, Clone)]
pub(crate) struct ComponentEntry {
    pub(crate) status: ComponentStatus,
    pub(crate) detail: Option<String>,
    pub(crate) started_at: Option<Instant>,
    pub(crate) duration_ms: Option<u64>,
}

impl Default for ComponentEntry {
    fn default() -> Self {
        Self {
            status: ComponentStatus::Pending,
            detail: None,
            started_at: None,
            duration_ms: None,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct LoadingProgress {
    pub(crate) server_addr: String,
    pub(crate) status_message: String,
    pub(crate) started_at: Instant,
    pub(crate) components: EnumMap<LoadingComponent, ComponentEntry>,
}

impl LoadingProgress {
    pub(crate) fn new(server_addr: impl Into<String>) -> Self {
        Self {
            server_addr: server_addr.into(),
            status_message: "Starting connection...".to_string(),
            started_at: Instant::now(),
            components: enum_map! { _ => ComponentEntry::default() },
        }
    }

    pub(crate) fn begin(&mut self, comp: LoadingComponent, message: impl Into<String>) {
        let msg = message.into();
        let entry = &mut self.components[comp];
        entry.status = ComponentStatus::Initializing;
        entry.started_at = Some(Instant::now());
        entry.duration_ms = None;
        self.status_message = msg;
    }

    pub(crate) fn finish(&mut self, comp: LoadingComponent, detail: Option<String>) {
        let entry = &mut self.components[comp];
        entry.status = ComponentStatus::Initialized;
        if let Some(started) = entry.started_at {
            entry.duration_ms = Some(started.elapsed().as_millis() as u64);
        }
        if detail.is_some() {
            entry.detail = detail;
        }
    }

    pub(crate) fn set_status(&mut self, message: impl Into<String>) {
        self.status_message = message.into();
    }

    pub(crate) fn completed_count(&self) -> usize {
        self.components
            .values()
            .filter(|e| e.status == ComponentStatus::Initialized)
            .count()
    }

    pub(crate) fn total_count(&self) -> usize {
        LoadingComponent::ALL.len()
    }

    pub(crate) fn fraction(&self) -> f32 {
        self.completed_count() as f32 / self.total_count() as f32
    }

    pub(crate) fn any_initializing(&self) -> bool {
        self.components
            .values()
            .any(|e| e.status == ComponentStatus::Initializing)
    }

    /// Builds a deterministic snapshot at `step` (0..=13) for headless preview/testing.
    ///
    /// At step `k`:
    /// - components `0..k` are `Initialized` (green) with representative metadata
    /// - component `k` (if `< 13`) is `Initializing` (amber)
    /// - remaining components are `Pending`
    #[cfg(feature = "headless-menu")]
    pub(crate) fn demo_at_step(step: usize) -> Self {
        use std::time::Duration;
        let mut progress = Self::new("grpc://play.perovskite.local:28273");
        let sample_details: [(LoadingComponent, &str, &str, u64); 13] = [
            (
                LoadingComponent::GrpcTransport,
                "Connecting to server...",
                "HTTP/2 TLS",
                42,
            ),
            (
                LoadingComponent::AuthHandshake,
                "Logging into server...",
                "OPAQUE v11",
                318,
            ),
            (
                LoadingComponent::MediaCache,
                "Fetching media list...",
                "214 assets",
                64,
            ),
            (
                LoadingComponent::BlockDefs,
                "Loading block definitions...",
                "128 blocks",
                29,
            ),
            (
                LoadingComponent::BlockRenderer,
                "Setting up block renderer...",
                "2048px atlas",
                245,
            ),
            (
                LoadingComponent::ChunkBuffers,
                "Allocating chunk & raytrace buffers...",
                "BVH ready",
                18,
            ),
            (
                LoadingComponent::ItemDefs,
                "Loading item definitions...",
                "96 items",
                14,
            ),
            (
                LoadingComponent::GameUi,
                "Loading item textures & HUD...",
                "HUD + egui",
                87,
            ),
            (
                LoadingComponent::EntityRenderer,
                "Loading entity definitions...",
                "24 meshes",
                53,
            ),
            (
                LoadingComponent::AudioEngine,
                "Loading audio files...",
                "38 samples",
                112,
            ),
            (
                LoadingComponent::FarGeometry,
                "Preparing distant terrain preview...",
                "LOD tables",
                9,
            ),
            (
                LoadingComponent::ClientStateBus,
                "Starting client state & workers...",
                "Workers live",
                12,
            ),
            (
                LoadingComponent::WorldSync,
                "Waiting for initial game state...",
                "Synced",
                195,
            ),
        ];

        let clamped = step.min(sample_details.len());
        let mut total_ms: u64 = 0;
        for (idx, (comp, active_msg, done_detail, ms)) in sample_details.iter().enumerate() {
            let entry = &mut progress.components[*comp];
            if idx < clamped {
                entry.status = ComponentStatus::Initialized;
                entry.detail = Some((*done_detail).to_string());
                entry.duration_ms = Some(*ms);
                total_ms += *ms;
            } else if idx == clamped {
                entry.status = ComponentStatus::Initializing;
                entry.started_at = Some(Instant::now());
                progress.status_message = (*active_msg).to_string();
                total_ms += ms / 2;
            }
        }
        progress.started_at = Instant::now() - Duration::from_millis(total_ms);
        if clamped >= sample_details.len() {
            progress.status_message = "Connected! Launching renderer...".to_string();
        }
        progress
    }
}

// Color palette — dark systems blueprint with high-contrast amber/green status states
const BG_DEEP: Color32 = Color32::from_rgb(13, 18, 26);
const ZONE_BG: Color32 = Color32::from_rgba_premultiplied(20, 28, 40, 170);
const ZONE_BORDER: Color32 = Color32::from_rgb(38, 52, 72);
const ZONE_TITLE: Color32 = Color32::from_rgb(112, 134, 162);

const PENDING_FILL: Color32 = Color32::from_rgb(20, 27, 38);
const PENDING_BORDER: Color32 = Color32::from_rgb(56, 72, 94);
const PENDING_TEXT: Color32 = Color32::from_rgb(138, 154, 176);
const PENDING_SUBTEXT: Color32 = Color32::from_rgb(92, 108, 130);
const PENDING_LED: Color32 = Color32::from_rgb(66, 82, 104);

const AMBER_FILL: Color32 = Color32::from_rgb(44, 32, 14);
const AMBER_BORDER: Color32 = Color32::from_rgb(245, 158, 11);
const AMBER_TEXT: Color32 = Color32::from_rgb(254, 243, 199);
const AMBER_SUBTEXT: Color32 = Color32::from_rgb(251, 191, 36);

const GREEN_FILL: Color32 = Color32::from_rgb(14, 38, 28);
const GREEN_BORDER: Color32 = Color32::from_rgb(34, 197, 94);
const GREEN_TEXT: Color32 = Color32::from_rgb(220, 252, 231);
const GREEN_SUBTEXT: Color32 = Color32::from_rgb(134, 239, 172);

const TRACE_PENDING: Color32 = Color32::from_rgb(44, 58, 78);
const TRACE_AMBER: Color32 = Color32::from_rgb(245, 158, 11);
const TRACE_GREEN: Color32 = Color32::from_rgb(34, 176, 88);

const HEADER_TEXT: Color32 = Color32::from_rgb(226, 236, 248);
const MUTED_TEXT: Color32 = Color32::from_rgb(140, 158, 182);

/// Directed architectural data-flow edges between client components.
const EDGES: &[(LoadingComponent, LoadingComponent)] = &[
    // Column 1 vertical session pipeline
    (
        LoadingComponent::GrpcTransport,
        LoadingComponent::AuthHandshake,
    ),
    (
        LoadingComponent::AuthHandshake,
        LoadingComponent::MediaCache,
    ),
    // Column 1 -> Column 2 (server definitions & asset manifests)
    (LoadingComponent::AuthHandshake, LoadingComponent::BlockDefs),
    (LoadingComponent::AuthHandshake, LoadingComponent::ItemDefs),
    (
        LoadingComponent::MediaCache,
        LoadingComponent::EntityRenderer,
    ),
    (LoadingComponent::MediaCache, LoadingComponent::AudioEngine),
    // Column 2 -> Column 3 (GPU atlases, buffers, LOD)
    (LoadingComponent::BlockDefs, LoadingComponent::BlockRenderer),
    (LoadingComponent::BlockDefs, LoadingComponent::FarGeometry),
    (LoadingComponent::ItemDefs, LoadingComponent::GameUi),
    // Column 3 adjacent vertical dependency (make_uis takes &block_renderer)
    (LoadingComponent::BlockRenderer, LoadingComponent::GameUi),
    // Column 3 -> Column 4 (unified client state)
    (
        LoadingComponent::BlockRenderer,
        LoadingComponent::ClientStateBus,
    ),
    (LoadingComponent::GameUi, LoadingComponent::ClientStateBus),
    (
        LoadingComponent::ChunkBuffers,
        LoadingComponent::ClientStateBus,
    ),
    (
        LoadingComponent::FarGeometry,
        LoadingComponent::ClientStateBus,
    ),
    // Column 2 -> Column 4 via lower bypass bus (EntityRenderer & AudioEngine -> ClientStateBus)
    (
        LoadingComponent::EntityRenderer,
        LoadingComponent::ClientStateBus,
    ),
    (
        LoadingComponent::AudioEngine,
        LoadingComponent::ClientStateBus,
    ),
    // Column 4 vertical sync
    (
        LoadingComponent::ClientStateBus,
        LoadingComponent::WorldSync,
    ),
];

struct ColumnSpec {
    label: &'static str,
    nodes: &'static [LoadingComponent],
}

const COLUMNS: [ColumnSpec; 4] = [
    ColumnSpec {
        label: "Initial connection",
        nodes: &[
            LoadingComponent::GrpcTransport,
            LoadingComponent::AuthHandshake,
            LoadingComponent::MediaCache,
        ],
    },
    ColumnSpec {
        label: "Game content",
        nodes: &[
            LoadingComponent::BlockDefs,
            LoadingComponent::ItemDefs,
            LoadingComponent::EntityRenderer,
            LoadingComponent::AudioEngine,
        ],
    },
    ColumnSpec {
        label: "Rendering engine",
        nodes: &[
            LoadingComponent::BlockRenderer,
            LoadingComponent::GameUi,
            LoadingComponent::ChunkBuffers,
            LoadingComponent::FarGeometry,
        ],
    },
    ColumnSpec {
        label: "Live gameplay",
        nodes: &[
            LoadingComponent::ClientStateBus,
            LoadingComponent::WorldSync,
        ],
    },
];

/// Renders the full-screen block-diagram loading screen.
///
/// Returns `true` if the user requested to cancel the connection.
pub(crate) fn draw_loading_screen(ctx: &egui::Context, progress: &LoadingProgress) -> bool {
    if progress.any_initializing() {
        ctx.request_repaint();
    }

    let mut cancel_requested = ctx.input(|i| i.key_pressed(Key::Escape));
    let time_secs = ctx.input(|i| i.time);
    // Smooth 0.0..=1.0 pulse for amber initializing highlights
    let pulse = ((time_secs * 3.5).sin() * 0.5 + 0.5) as f32;

    let frame = egui::Frame::NONE.fill(BG_DEEP).inner_margin(0.0);
    egui::CentralPanel::default().frame(frame).show(ctx, |ui| {
        let full_rect = ui.max_rect();

        let margin_x = (full_rect.width() * 0.035).clamp(18.0, 44.0);
        let header_h = (full_rect.height() * 0.115).clamp(54.0, 78.0);
        let footer_h = (full_rect.height() * 0.105).clamp(48.0, 68.0);

        let header_rect = Rect::from_min_max(
            Pos2::new(full_rect.left() + margin_x, full_rect.top() + 10.0),
            Pos2::new(full_rect.right() - margin_x, full_rect.top() + header_h),
        );
        let footer_rect = Rect::from_min_max(
            Pos2::new(full_rect.left() + margin_x, full_rect.bottom() - footer_h),
            Pos2::new(full_rect.right() - margin_x, full_rect.bottom() - 10.0),
        );
        let diagram_rect = Rect::from_min_max(
            Pos2::new(full_rect.left() + margin_x, header_rect.bottom() + 12.0),
            Pos2::new(full_rect.right() - margin_x, footer_rect.top() - 12.0),
        );

        let button_size = Vec2::new(116.0, 32.0);
        let button_rect = Rect::from_min_size(
            Pos2::new(
                footer_rect.right() - button_size.x - 12.0,
                footer_rect.center().y - button_size.y * 0.5,
            ),
            button_size,
        );
        let response = ui.allocate_rect(button_rect, Sense::click());
        let hovered = response.hovered();
        if response.clicked() {
            cancel_requested = true;
        }

        let painter = ui.painter();
        draw_header(painter, header_rect, progress);
        draw_diagram(painter, diagram_rect, progress, pulse);
        draw_footer(painter, footer_rect, progress);

        let btn_fill = if hovered {
            Color32::from_rgb(76, 32, 36)
        } else {
            Color32::from_rgb(44, 24, 28)
        };
        let btn_border = if hovered {
            Color32::from_rgb(248, 113, 113)
        } else {
            Color32::from_rgb(165, 68, 72)
        };
        painter.rect(
            button_rect,
            CornerRadius::same(5),
            btn_fill,
            Stroke::new(1.2_f32, btn_border),
            egui::StrokeKind::Inside,
        );
        painter.text(
            button_rect.center(),
            Align2::CENTER_CENTER,
            "Cancel [Esc]",
            FontId::monospace(13.0),
            Color32::from_rgb(254, 226, 226),
        );
    });

    cancel_requested
}

fn draw_header(painter: &Painter, rect: Rect, progress: &LoadingProgress) {
    // Bottom rule under header
    painter.line_segment(
        [
            Pos2::new(rect.left(), rect.bottom()),
            Pos2::new(rect.right(), rect.bottom()),
        ],
        Stroke::new(1.0_f32, ZONE_BORDER),
    );

    let title_size = (rect.height() * 0.33).clamp(15.0, 21.0);
    let sub_size = (rect.height() * 0.21).clamp(11.0, 14.0);

    painter.text(
        Pos2::new(rect.left(), rect.top() + rect.height() * 0.28),
        Align2::LEFT_CENTER,
        "Game loading...",
        FontId::monospace(title_size),
        HEADER_TEXT,
    );

    let elapsed_secs = progress.started_at.elapsed().as_secs_f32();
    let server_line = format!(
        "Server: {} | Loading: {} / {} {:.1}s",
        progress.server_addr,
        progress.completed_count(),
        progress.total_count(),
        elapsed_secs,
    );
    painter.text(
        Pos2::new(rect.left(), rect.top() + rect.height() * 0.72),
        Align2::LEFT_CENTER,
        server_line,
        FontId::monospace(sub_size),
        MUTED_TEXT,
    );
}

fn draw_diagram(painter: &Painter, rect: Rect, progress: &LoadingProgress, pulse: f32) {
    let col_count = COLUMNS.len();
    let col_gap = (rect.width() * 0.042).clamp(22.0, 56.0);
    let col_w = (rect.width() - col_gap * (col_count as f32 - 1.0)) / col_count as f32;

    let mut node_rects: EnumMap<LoadingComponent, Rect> = enum_map! { _ => Rect::NOTHING };
    let mut node_cols: EnumMap<LoadingComponent, usize> = enum_map! { _ => 0 };
    let mut zone_rects: [Rect; 4] = [Rect::NOTHING; 4];

    // 1. Draw architectural column zones & compute node rects
    let zone_header_h = 26.0;
    let zone_pad_x = (col_w * 0.075).clamp(8.0, 16.0);
    let zone_pad_top = 12.0;
    // Leave a clean lower bus corridor inside the diagram for cross-column bypass traces
    let zone_pad_bottom = 26.0;

    for (col_idx, col) in COLUMNS.iter().enumerate() {
        let x0 = rect.left() + col_idx as f32 * (col_w + col_gap);
        let zone_rect =
            Rect::from_min_size(Pos2::new(x0, rect.top()), Vec2::new(col_w, rect.height()));
        zone_rects[col_idx] = zone_rect;

        painter.rect(
            zone_rect,
            CornerRadius::same(8),
            ZONE_BG,
            Stroke::new(1.0_f32, ZONE_BORDER),
            egui::StrokeKind::Inside,
        );

        // Column header strip
        let label_font = FontId::monospace((col_w * 0.048).clamp(9.5, 12.0));
        painter.text(
            Pos2::new(zone_rect.left() + zone_pad_x, zone_rect.top() + 14.0),
            Align2::LEFT_CENTER,
            col.label,
            label_font,
            ZONE_TITLE,
        );
        painter.line_segment(
            [
                Pos2::new(zone_rect.left() + 6.0, zone_rect.top() + zone_header_h),
                Pos2::new(zone_rect.right() - 6.0, zone_rect.top() + zone_header_h),
            ],
            Stroke::new(1.0_f32, Color32::from_rgba_premultiplied(56, 72, 94, 110)),
        );

        let inner_top = zone_rect.top() + zone_header_h + zone_pad_top;
        let inner_h = (zone_rect.bottom() - zone_pad_bottom - inner_top).max(40.0);
        let n = col.nodes.len();
        let node_gap = (inner_h * 0.065).clamp(10.0, 22.0);
        let max_node_h = 82.0;
        let slot_h = ((inner_h - node_gap * (n.saturating_sub(1) as f32)) / n as f32)
            .min(max_node_h)
            .max(36.0);
        let total_stack_h = slot_h * n as f32 + node_gap * (n.saturating_sub(1) as f32);
        let stack_top = inner_top + (inner_h - total_stack_h) * 0.5;

        for (row_idx, comp) in col.nodes.iter().enumerate() {
            let ny = stack_top + row_idx as f32 * (slot_h + node_gap);
            node_rects[*comp] = Rect::from_min_size(
                Pos2::new(zone_rect.left() + zone_pad_x, ny),
                Vec2::new((col_w - zone_pad_x * 2.0).max(60.0), slot_h),
            );
            node_cols[*comp] = col_idx;
        }
    }

    // 2. Draw orthogonal data-flow traces underneath the nodes (pending first, then green, then amber on top)
    let lower_bus_y = rect.bottom() - 12.0;
    for pass in [0u8, 1, 2] {
        for (idx, &(from, to)) in EDGES.iter().enumerate() {
            let from_status = progress.components[from].status;
            let to_status = progress.components[to].status;

            // Light incoming traces amber while target is initializing; green once both are ready.
            let (edge_pass, color, width) = match (from_status, to_status) {
                (ComponentStatus::Initialized, ComponentStatus::Initializing) => {
                    (2, TRACE_AMBER, 2.2_f32)
                }
                (ComponentStatus::Initialized, ComponentStatus::Initialized) => {
                    (1, TRACE_GREEN, 1.8_f32)
                }
                _ => (0, TRACE_PENDING, 1.3_f32),
            };
            if edge_pass != pass {
                continue;
            }
            draw_edge(
                painter,
                node_rects[from],
                node_rects[to],
                node_cols[from],
                node_cols[to],
                &zone_rects,
                lower_bus_y,
                color,
                width,
                idx,
            );
        }
    }

    // 3. Draw each component block
    for comp in LoadingComponent::ALL {
        draw_node(
            painter,
            node_rects[comp],
            comp,
            &progress.components[comp],
            pulse,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_edge(
    painter: &Painter,
    from: Rect,
    to: Rect,
    from_col: usize,
    to_col: usize,
    zone_rects: &[Rect; 4],
    lower_bus_y: f32,
    color: Color32,
    width: f32,
    edge_index: usize,
) {
    let stroke = Stroke::new(width, color);

    if from_col == to_col {
        let start = Pos2::new(from.center().x, from.bottom());
        let end = Pos2::new(to.center().x, to.top() - 2.0);
        painter.line_segment([start, end], stroke);
        draw_arrowhead(painter, end, Vec2::new(0.0, 1.0), color);
    } else if to_col > from_col + 1 {
        // Cross-column bypass: drop into the lower bus rail below intermediate columns,
        // then rise cleanly in the gutter immediately before the target column.
        let start = Pos2::new(from.right(), from.center().y);
        let end = Pos2::new(to.left() - 2.0, to.center().y + 10.0);
        let left_gutter_x = (zone_rects[from_col].right() + zone_rects[from_col + 1].left()) * 0.5
            + ((edge_index % 3) as f32 - 1.0) * 4.0;
        let right_gutter_x = (zone_rects[to_col - 1].right() + zone_rects[to_col].left()) * 0.5
            + ((edge_index % 3) as f32 - 1.0) * 4.0;
        let bus_y = lower_bus_y - (edge_index % 2) as f32 * 5.0;

        let p1 = Pos2::new(left_gutter_x, start.y);
        let p2 = Pos2::new(left_gutter_x, bus_y);
        let p3 = Pos2::new(right_gutter_x, bus_y);
        let p4 = Pos2::new(right_gutter_x, end.y);

        painter.line_segment([start, p1], stroke);
        painter.line_segment([p1, p2], stroke);
        painter.line_segment([p2, p3], stroke);
        painter.line_segment([p3, p4], stroke);
        painter.line_segment([p4, end], stroke);
        draw_arrowhead(painter, end, Vec2::new(1.0, 0.0), color);
    } else {
        let start = Pos2::new(from.right(), from.center().y);
        let end = Pos2::new(to.left() - 2.0, to.center().y);
        let gutter_center = (zone_rects[from_col].right() + zone_rects[to_col].left()) * 0.5;
        let lane_offset = ((edge_index % 5) as f32 - 2.0) * 3.0;
        let mid_x = gutter_center + lane_offset;
        let p1 = Pos2::new(mid_x, start.y);
        let p2 = Pos2::new(mid_x, end.y);

        painter.line_segment([start, p1], stroke);
        painter.line_segment([p1, p2], stroke);
        painter.line_segment([p2, end], stroke);
        draw_arrowhead(painter, end, Vec2::new(1.0, 0.0), color);
    }
}

fn draw_arrowhead(painter: &Painter, tip: Pos2, dir: Vec2, color: Color32) {
    let len = 6.5;
    let half_w = 3.6;
    let perp = Vec2::new(-dir.y, dir.x);
    let base = tip - dir * len;
    painter.add(egui::Shape::convex_polygon(
        vec![tip, base + perp * half_w, base - perp * half_w],
        color,
        Stroke::NONE,
    ));
}

fn draw_node(
    painter: &Painter,
    rect: Rect,
    comp: LoadingComponent,
    entry: &ComponentEntry,
    pulse: f32,
) {
    let (fill, border, border_w, title_col, sub_col, led_col) = match entry.status {
        ComponentStatus::Pending => (
            PENDING_FILL,
            PENDING_BORDER,
            1.2_f32,
            PENDING_TEXT,
            PENDING_SUBTEXT,
            PENDING_LED,
        ),
        ComponentStatus::Initializing => (
            AMBER_FILL,
            lerp_color(AMBER_BORDER, AMBER_TEXT, pulse * 0.35),
            2.2_f32,
            AMBER_TEXT,
            AMBER_SUBTEXT,
            lerp_color(AMBER_BORDER, Color32::WHITE, pulse * 0.45),
        ),
        ComponentStatus::Initialized => (
            GREEN_FILL,
            GREEN_BORDER,
            1.8_f32,
            GREEN_TEXT,
            GREEN_SUBTEXT,
            GREEN_BORDER,
        ),
    };

    painter.rect(
        rect,
        0,
        fill,
        Stroke::new(border_w, border),
        egui::StrokeKind::Inside,
    );

    let title_font_size = (rect.height() * 0.22)
        .min(rect.width() * 0.086)
        .clamp(10.5, 15.0);
    let sub_font_size = (rect.height() * 0.175)
        .min(rect.width() * 0.068)
        .clamp(9.0, 12.0);

    let led_pos = Pos2::new(rect.right() - 11.0, rect.top() + 12.0);
    painter.circle_filled(led_pos, 3.8, led_col);

    // Component title
    painter.text(
        Pos2::new(rect.left() + 12.0, rect.top() + rect.height() * 0.34),
        Align2::LEFT_CENTER,
        comp.title(),
        FontId::proportional(title_font_size),
        title_col,
    );

    // Live detail + timing if available
    let detail_line = match (&entry.detail, entry.duration_ms) {
        (Some(detail), Some(ms)) => format!("{detail}  ({ms} ms)"),
        (Some(detail), None) => detail.clone(),
        (None, Some(ms)) => format!("{ms} ms..."),
        (None, None) => String::new(),
    };
    painter.text(
        Pos2::new(rect.left() + 12.0, rect.top() + rect.height() * 0.72),
        Align2::LEFT_CENTER,
        detail_line,
        FontId::monospace(sub_font_size),
        sub_col,
    );
}

fn draw_footer(painter: &Painter, rect: Rect, progress: &LoadingProgress) {
    painter.rect(
        rect,
        CornerRadius::same(6),
        Color32::from_rgb(18, 25, 36),
        Stroke::new(1.0_f32, ZONE_BORDER),
        egui::StrokeKind::Inside,
    );

    const PROGRESS_BAR_LEN: usize = 32;
    let bars = ((PROGRESS_BAR_LEN as f32) * progress.fraction()) as usize;
    let empty = PROGRESS_BAR_LEN - bars;
    let progress_string: String = std::iter::repeat('=')
        .take(bars)
        .chain(std::iter::repeat(' ').take(empty))
        .collect();

    let status_text = format!("[{}] {}", progress_string, progress.status_message);
    painter.text(
        Pos2::new(rect.left() + 14.0, rect.center().y),
        Align2::LEFT_CENTER,
        status_text,
        FontId::monospace(13.5),
        if progress.any_initializing() {
            AMBER_SUBTEXT
        } else {
            GREEN_SUBTEXT
        },
    );
}

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let lerp = |u: u8, v: u8| ((u as f32) + (v as f32 - u as f32) * t).round() as u8;
    Color32::from_rgba_unmultiplied(
        lerp(a.r(), b.r()),
        lerp(a.g(), b.g()),
        lerp(a.b(), b.b()),
        lerp(a.a(), b.a()),
    )
}
