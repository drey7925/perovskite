use std::{
    hash::{Hash, Hasher},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use anyhow::{bail, ensure, Context, Result};
use cgmath::Vector3;

use glyph_brush::{
    ab_glyph::{FontRef, PxScale},
    BrushAction, BrushError, GlyphVertex, Section,
};
use image::GenericImage;
use log::error;
use perovskite_core::{
    coordinates::{BlockCoordinate, ChunkCoordinate},
    protocol::render::{RenderedText, RichTextSpan},
};
use rustc_hash::FxHashMap;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use vulkano::{
    buffer::{BufferUsage, Subbuffer},
    format::Format,
    image::sampler::{Filter, SamplerCreateInfo},
    pipeline::graphics::vertex_input::Vertex,
    DeviceSize,
};

use crate::{
    fonts::MPLUS1_LIGHT_BYTES,
    vulkan::{
        shaders::x5y5z5_pack_vec, BufferReclaim, ReclaimableBuffer, Texture2DHolder, VulkanContext,
    },
};
use tracy_client::span;

trait F32x3Ext {
    fn to_f64(self) -> Vector3<f64>;
    fn to_array(self) -> [f32; 3];
}
impl F32x3Ext for Vector3<f32> {
    fn to_f64(self) -> Vector3<f64> {
        Vector3::new(self.x as f64, self.y as f64, self.z as f64)
    }
    fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
}
trait F64x3Ext {
    fn to_f32(self) -> Vector3<f32>;
}
impl F64x3Ext for Vector3<f64> {
    fn to_f32(self) -> Vector3<f32> {
        Vector3::new(self.x as f32, self.y as f32, self.z as f32)
    }
}
fn color_to_565(color: u32) -> u16 {
    // R's 5 upper bytes are [23..=19] (18..=16 are ignored), and must
    // land in [15..=11].
    let r_component = (color & 0xf80000) >> 8;
    // G's 6 upper bytes are [15..=10] (9 and 8 are ignored) and must land in [10..=5].
    let g_component = (color & 0x00fc00) >> 5;
    // B's 5 upper bytes are [7..=3] (2..=0 are ignored) and must land in [4..=0].
    let b_component = (color & 0xf8) >> 3;
    (r_component | g_component | b_component) as u16
}
// Note that BufferContents got auto-derived via a blanket impl that applies for bytemuck'd
// types.
#[derive(Vertex, Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
// Not yet optimized for space/membw... :( - A lot to consider, e.g. given that
// VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT + R5G6B5 is not guaranteed, so we spend ALU
// on bit-twiddling out the color
pub(crate) struct TextVertex {
    /// Position, 3d space, relative to the renderer's origin in world space
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
    /// Texture coordinate in tex space (given as texels)
    #[format(R16G16_UINT)]
    uv_texcoord: [u16; 2],
    /// Diffuse color as R5G6B5
    #[format(R16_UINT)]
    diffuse_color: u16,

    /// Emissive color as R5G6B5
    #[format(R16_UINT)]
    emissive_color: u16,
    /// Encoded normal, same as CubeGeometryVertex, using [x5y5z5_pack_vec]
    #[format(R16_UINT)]
    encoded_normal: u16,
    // Flags. Low four bits are a very coarse version of the emissive alpha.
    // 0 = tightly focused on normal, 15 = nearly isotropic.
    // High four bits are for future use.
    #[format(R8_UINT)]
    flags: u8,
    // This pads from 23 to 24 bytes. Explicit padding needed to make bytemuck
    // pod work.
    #[format(R8_UINT)]
    padding: u8,
}
impl TextVertex {
    /// # Arguments
    ///
    /// * `v` - The glyph vertex to process
    /// * `tex_size` - The size of the texture atlas
    /// * `draw_origin` - The origin of the text renderer in draw space. The difference between the glyph's
    ///    position in world space, minus the renderer's, will be used to avoid a catastrophic cancellation
    ///    in fp32.
    fn make_glyph(
        v: GlyphVertex<PvExtra>,
        tex_size: (u32, u32),
        draw_origin: Vector3<f64>,
    ) -> [Self; 4] {
        dbg!(&v);
        let du = v.extra.text.u / v.bounds.width();
        let dv = v.extra.text.v / v.bounds.height();
        let origin = (v.extra.text.origin_world
            + (du * (v.pixel_coords.min.x - v.bounds.min.x)).to_f64()
            + (dv * (v.pixel_coords.min.y - v.bounds.min.y)).to_f64()
            + Vector3::from(v.extra.block_coord)
            - draw_origin)
            .to_f32();
        let world_du = du * v.pixel_coords.width();
        let world_dv = dv * v.pixel_coords.height();

        let tex_min = [
            (v.tex_coords.min.x * tex_size.0 as f32) as u16,
            (v.tex_coords.min.y * tex_size.1 as f32) as u16,
        ];
        let tex_size = [
            (v.tex_coords.width() * tex_size.0 as f32) as u16,
            (v.tex_coords.height() * tex_size.1 as f32) as u16,
        ];

        let prototype = TextVertex {
            position: [0.0, 0.0, 0.0],
            uv_texcoord: [0, 0],
            diffuse_color: v.extra.span.diffuse_565,
            emissive_color: v.extra.span.emissive_565,
            encoded_normal: v.extra.text.encoded_normal,
            flags: v.extra.span.flags,
            padding: 0,
        };
        [
            TextVertex {
                position: origin.to_array(),
                uv_texcoord: tex_min,
                ..prototype
            },
            TextVertex {
                position: (origin + world_du).to_array(),
                uv_texcoord: [tex_min[0] + tex_size[0], tex_min[1]],
                ..prototype
            },
            // order is important here since it has to correspond to the index buffer
            TextVertex {
                position: (origin + world_du + world_dv).to_array(),
                uv_texcoord: [tex_min[0] + tex_size[0], tex_min[1] + tex_size[1]],
                ..prototype
            },
            TextVertex {
                position: (origin + world_dv).to_array(),
                uv_texcoord: [tex_min[0], tex_min[1] + tex_size[1]],
                ..prototype
            },
        ]
    }
}

#[derive(Clone, Copy, Debug, Default, Hash, PartialEq)]
struct PvExtra {
    // TODO: In the future, consider canonicalizing to improve scalability, if
    // we think we can do better than glyph_brush for our use-case.
    // For now, just store the draw parameters straight up.
    text: TextDrawParams,
    // TODO: should we be hashing it? glyph_brush shouldn't care, only our shader
    // cares about these.
    span: SpanDrawParams,
    block_coord: BlockCoordinate,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct TextDrawParams {
    encoded_normal: u16,
    u: Vector3<f32>,
    v: Vector3<f32>,
    origin_world: Vector3<f64>,
}
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq)]
struct SpanDrawParams {
    diffuse_565: u16,
    emissive_565: u16,
    flags: u8,
}
impl Default for TextDrawParams {
    fn default() -> Self {
        Self {
            u: Vector3::new(1.0, 0.0, 0.0),
            v: Vector3::new(0.0, 1.0, 0.0),
            origin_world: Vector3::new(0.0, 0.0, 0.0),
            encoded_normal: 0,
        }
    }
}
impl Hash for TextDrawParams {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.u.map(f32::to_bits).hash(state);
        self.v.map(f32::to_bits).hash(state);
        self.origin_world.map(f64::to_bits).hash(state);
    }
}
impl TryFrom<&RenderedText> for TextDrawParams {
    type Error = anyhow::Error;
    fn try_from(text: &RenderedText) -> Result<Self> {
        let u: Vector3<f32> = text.u_extent.context("Missing u_extent")?.try_into()?;
        let v: Vector3<f32> = text.v_extent.context("Missing v_extent")?.try_into()?;
        let normal = u.cross(v);
        let origin_world: Vector3<f64> = text
            .top_left_corner
            .context("Missing top_left_corner")?
            .try_into()?;
        Ok(Self {
            origin_world,
            u,
            v,
            encoded_normal: x5y5z5_pack_vec(normal),
            ..Default::default()
        })
    }
}
impl From<&RichTextSpan> for SpanDrawParams {
    fn from(span: &RichTextSpan) -> Self {
        let emissive_alpha_upper = span.emissive_color_rgb & 0xf0000000 >> 28;
        Self {
            diffuse_565: color_to_565(span.color_rgb),
            emissive_565: color_to_565(span.emissive_color_rgb),
            flags: (emissive_alpha_upper & 0xf) as u8, // No other flags yet
        }
    }
}

struct CpuVertexData {
    vertices: Vec<TextVertex>,
    origin: Vector3<f64>,
}

pub(crate) struct TextOutput {
    /// The vertex buffer we will pass to vkCmdDrawIndexed.
    ///
    /// It is the responsibility of the caller to ensure that a large-enough
    /// index buffer is available to draw all of the vertices.
    pub(crate) vertices: ReclaimableBuffer<TextVertex>,
    pub(crate) indices: Subbuffer<[u32]>,
    pub(crate) index_count: u32,
    /// For now, we just hand over the texture. Later on, we'll provide a
    /// queue of incremental updates if necessary for performance; however,
    /// there are lots of implementation details that an incremental upload
    /// system has to get right, including sequencing changes to vertices after
    /// the atlas updates they depend on, draining out old incremental updates
    /// if a new full update or resize occurs, etc. Getting it wrong will probably
    /// produce hard-to-debug glitches. We can revisit if this proves to be a
    /// bottleneck.
    pub(crate) atlas: Texture2DHolder,
    /// Counter for atlas rebuilds. At some point if we do incremental atlas updates,
    /// we'll also have a minor generation.
    pub(crate) atlas_major_generation: usize,
    /// The origin that the renderer used. All coordinates in the vertex buffer are
    /// relative to this origin; the caller should also consider the player's camera
    /// position and adjust further (e.g. via the push constant in the shader) to transform
    /// positions accordingly.
    pub(crate) origin: Vector3<f64>,
}

pub(crate) enum MaybeUnchanged<T> {
    Unchanged,
    Changed(T),
}
impl<T> MaybeUnchanged<T> {
    pub(crate) fn take_change(&mut self) -> MaybeUnchanged<T> {
        std::mem::replace(self, MaybeUnchanged::Unchanged)
    }
}

pub(crate) struct TextRenderer {
    brush: glyph_brush::GlyphBrush<[TextVertex; 4], PvExtra, FontRef<'static>>,
    // The working texture. For now, we'll upload the whole texture to the GPU
    // but we could optimize by copying only updated regions.
    atlas: image::GrayImage,
    // todo: intervalset of dirty regions
}
static ERROR_COUNT: AtomicUsize = AtomicUsize::new(0);

impl TextRenderer {
    fn do_work(
        &mut self,
        block_texts: &FxHashMap<BlockCoordinate, BlockText>,
    ) -> (
        MaybeUnchanged<Option<CpuVertexData>>,
        MaybeUnchanged<image::GrayImage>,
    ) {
        let mut origin_sum = Vector3::new(0.0, 0.0, 0.0);
        {
            let _span = span!("text rendering build glyph brush queue");
            for (coord, block_text) in block_texts {
                origin_sum += Vector3::from(*coord);
                for panel in &block_text.protos {
                    let mut s =
                        Section::new().with_bounds((panel.u_texels as f32, panel.v_texels as f32));
                    let tdp = match TextDrawParams::try_from(panel) {
                        Ok(tdp) => tdp,
                        Err(e) => {
                            let error_count = ERROR_COUNT.fetch_add(1, Ordering::Relaxed);
                            if error_count < 20 {
                                error!("Error creating text params for block {:?}: {:?}", coord, e);
                            }
                            if error_count == 19 {
                                error!(
                                "To avoid logspam, further text param errors will be suppressed."
                            );
                            }
                            continue;
                        }
                    };
                    for span in panel.spans.iter() {
                        let sdp = SpanDrawParams::from(span);
                        let extra = PvExtra {
                            text: tdp,
                            span: sdp,
                            block_coord: *coord,
                        };
                        s.text.push(glyph_brush::Text {
                            text: &span.text,
                            scale: PxScale::from(span.texel_height),
                            extra,
                            ..Default::default()
                        });
                    }
                    self.brush.queue(s);
                }
            }
        }

        let origin = origin_sum / block_texts.len() as f64;

        let mut atlas_changed = false;
        let vertices = loop {
            let _span = span!("glyph_brush process_queued");
            let dims = self.atlas.dimensions();
            match self.brush.process_queued(
                |rect, pixels| {
                    atlas_changed = true;
                    self.atlas
                        .copy_from(
                            &image::GrayImage::from_vec(
                                rect.width() as u32,
                                rect.height() as u32,
                                pixels.to_vec(),
                            )
                            .expect("glyph brush pixels should be valid gray pixels"),
                            rect.min[0],
                            rect.min[1],
                        )
                        .unwrap();
                },
                |v| TextVertex::make_glyph(v, dims, origin),
            ) {
                Ok(BrushAction::ReDraw) => {
                    return (MaybeUnchanged::Unchanged, MaybeUnchanged::Unchanged)
                }
                Ok(BrushAction::Draw(vertices)) => break vertices,
                Err(BrushError::TextureTooSmall { suggested }) => {
                    log::info!(
                        "Glyph brush texture too small, suggested resize to {}x{}",
                        suggested.0,
                        suggested.1
                    );
                    if suggested.0 * suggested.1 > (4096 * 4096) {
                        panic!("Texture atlas too small: {}x{}", suggested.0, suggested.1);
                    } else {
                        // if neither coordinate is over 4096, use both rounded up to the next power of 2
                        if suggested.0 <= 4096 && suggested.1 <= 4096 {
                            let new_width = suggested.0.next_power_of_two();
                            let new_height = suggested.1.next_power_of_two();
                            log::info!("Resizing glyph texture to {}x{}", new_width, new_height);
                            self.atlas = image::GrayImage::new(new_width, new_height);
                        } else {
                            // Use the suggested area as a proxy for the right size
                            let raw_width = (suggested.0 * suggested.1).isqrt() as u32;
                            let effective_width = (raw_width + 1).next_power_of_two().min(4096);
                            log::info!(
                                    "Resizing glyph texture to {}x{} (not following suggested aspect ratio because it exceeds 4096 in at least one dimension)",
                                    effective_width,
                                    effective_width,
                                );
                            self.atlas = image::GrayImage::new(effective_width, effective_width);
                        }
                        continue;
                    }
                }
            }
        };

        log::info!("Generated {} vertices", vertices.len());

        if vertices.is_empty() {
            return (MaybeUnchanged::Changed(None), MaybeUnchanged::Unchanged);
        }

        let vtx_data = MaybeUnchanged::Changed(Some(CpuVertexData {
            vertices: bytemuck::cast_vec(vertices),
            origin,
        }));

        let atlas_data = if atlas_changed {
            MaybeUnchanged::Changed(self.atlas.clone())
        } else {
            MaybeUnchanged::Unchanged
        };

        (vtx_data, atlas_data)
    }
}

pub(crate) struct TextEngine {
    renderer: tokio::sync::Mutex<TextRenderer>,
    output: parking_lot::Mutex<MaybeUnchanged<Option<TextOutput>>>,
    text_objects: tokio::sync::Mutex<(usize, FxHashMap<BlockCoordinate, BlockText>)>, // modcount, map
    notify_work: tokio::sync::Notify,
    cancel_token: CancellationToken,
}
impl TextEngine {
    pub(crate) async fn insert_or_update(&self, coord: BlockCoordinate, text: Vec<RenderedText>) {
        let mut guard = self.text_objects.lock().await;
        if text.is_empty() {
            guard.1.remove(&coord);
            guard.0 = guard.0.wrapping_add(1);
        } else {
            guard.1.insert(coord, BlockText { protos: text });
            guard.0 = guard.0.wrapping_add(1);
        }
        self.notify_work.notify_one();
        // Same as implicit drop, but make intent clear that we drop _after_
        // notifying.
        drop(guard);
    }
    pub(crate) async fn remove_block(&self, coord: BlockCoordinate) {
        let mut guard = self.text_objects.lock().await;
        guard.1.remove(&coord);
        guard.0 = guard.0.wrapping_add(1);
        self.notify_work.notify_one();
        drop(guard);
    }
    pub(crate) async fn remove_chunk(&self, chunk_coord: ChunkCoordinate) {
        let mut guard = self.text_objects.lock().await;
        guard.1.retain(|coord, _| coord.chunk() != chunk_coord);
        guard.0 = guard.0.wrapping_add(1);
        self.notify_work.notify_one();
        drop(guard);
    }

    pub(crate) fn run_worker(
        self: Arc<Self>,
        vk_ctx: Arc<VulkanContext>,
    ) -> JoinHandle<Result<()>> {
        // there's only one of these workers, by design.
        tokio::task::spawn(async move {
            // A subbuffer large enough for the largest current text output.
            // When an output is too large, storing into index_buffer happens-before
            // storing to output.vertices and output.index_count.
            let mut index_buffer = Self::make_index_buffer(256, &vk_ctx)?;
            let mut last_mod_count = 0;
            let mut atlas_major_generation = 1;
            let mut atlas = None;

            while !self.cancel_token.is_cancelled() {
                let mut guard = self.text_objects.lock().await;
                while guard.0 == last_mod_count {
                    drop(guard);
                    self.notify_work.notified().await;
                    guard = self.text_objects.lock().await;
                }
                let mut renderer = self.renderer.lock().await;
                let (new_vertices, new_atlas) = renderer.do_work(&guard.1);
                last_mod_count = guard.0;
                drop(renderer);
                drop(guard);

                match new_vertices {
                    MaybeUnchanged::Changed(Some(vtx_data)) => {
                        let _span = span!("text_upload_gpu");

                        let mut transfer_buffer = vk_ctx.start_transfer_buffer()?;
                        let len = vtx_data.vertices.len() as u64;
                        assert_eq!(
                            len % 4,
                            0,
                            "Number of vertices must be a multiple of 4, got {}",
                            len
                        );
                        let index_count = len / 4 * 6;
                        ensure!(index_count < u32::MAX as u64, "Too many vertices");
                        if index_count > index_buffer.len() {
                            index_buffer = Self::make_index_buffer(
                                index_count
                                    .checked_next_power_of_two()
                                    .context("Capacity overflow for index buffer")?,
                                &vk_ctx,
                            )?;
                        }
                        let gpu_buffer = vk_ctx.iter_to_device_via_staging_with_reclaim(
                            vtx_data.vertices.into_iter(),
                            super::ReclaimType::GpuVtxTransferDst,
                            vk_ctx.txv_reclaimer().clone(),
                            BufferReclaim::<TextVertex>::size_class(len as DeviceSize),
                            &mut transfer_buffer,
                        )?;

                        if let MaybeUnchanged::Changed(new_atlas) = new_atlas {
                            atlas_major_generation += 1;
                            atlas = Some(Texture2DHolder::from_image(
                                &vk_ctx,
                                new_atlas,
                                Format::R8_UNORM,
                                SamplerCreateInfo {
                                    min_filter: Filter::Linear,
                                    mag_filter: Filter::Nearest,
                                    ..Default::default()
                                },
                                Some(&mut transfer_buffer),
                            )?);
                        }

                        vk_ctx.finish_transfer_buffer(transfer_buffer)?;

                        *self.output.lock() = MaybeUnchanged::Changed(Some(TextOutput {
                            vertices: gpu_buffer,
                            atlas: atlas
                                .clone()
                                .expect("Must have atlas after texture_upload_gpu"),
                            indices: index_buffer.clone(),
                            index_count: index_count as u32,
                            atlas_major_generation,
                            origin: vtx_data.origin,
                        }));
                    }
                    MaybeUnchanged::Unchanged => {
                        // pass
                    }
                    MaybeUnchanged::Changed(None) => {
                        *self.output.lock() = MaybeUnchanged::Changed(None);
                    }
                }
            }

            Ok(())
        })
    }

    fn make_index_buffer(cap: u64, vk_ctx: &VulkanContext) -> Result<Subbuffer<[u32]>> {
        let _span = span!("make_index_buffer");
        assert!(cap % 4 == 0);
        let mut indices = Vec::with_capacity(cap as usize);
        for quad in 0..cap / 4 {
            let base = quad * 4;
            if base > (u32::MAX as u64 - 4) {
                bail!("Capacity overflow for index buffer");
            }
            let base = base as u32;
            indices.extend([base, base + 1, base + 2, base + 2, base + 3, base]);
        }
        // Making this work with reclaim is tricky; just use a normal buffer instead.
        let gpu_buffer =
            vk_ctx.iter_to_device_via_staging(indices.into_iter(), BufferUsage::INDEX_BUFFER)?;
        Ok(gpu_buffer)
    }

    pub(crate) fn take_text_data(&self) -> MaybeUnchanged<Option<TextOutput>> {
        self.output.lock().take_change()
    }

    pub(crate) fn new() -> Self {
        let builder = glyph_brush::GlyphBrushBuilder::using_font(
            FontRef::try_from_slice(MPLUS1_LIGHT_BYTES).unwrap(),
        )
        .initial_cache_size((256, 256));

        Self {
            renderer: tokio::sync::Mutex::new(TextRenderer {
                brush: builder.build(),
                atlas: image::GrayImage::new(256, 256),
            }),
            output: parking_lot::Mutex::new(MaybeUnchanged::Unchanged),
            text_objects: tokio::sync::Mutex::new((0, FxHashMap::default())),
            notify_work: tokio::sync::Notify::new(),
            cancel_token: CancellationToken::new(),
        }
    }

    pub(crate) fn shut_down(&self) {
        self.cancel_token.cancel();
        self.notify_work.notify_waiters();
    }
}

struct BlockText {
    protos: Vec<RenderedText>,
    // possibly more in the future?
}

impl TextEngine {}

#[test]
fn simple_test_text_renderer() {
    let mut tr = TextRenderer {
        brush: glyph_brush::GlyphBrushBuilder::using_font(
            FontRef::try_from_slice(MPLUS1_LIGHT_BYTES).unwrap(),
        )
        .initial_cache_size((256, 256))
        .build(),
        atlas: image::GrayImage::new(256, 256),
    };
    let sample_text = Section::<'_, PvExtra>::new()
        .with_bounds((120.0, 120.0))
        .add_text(glyph_brush::Text::new("Hello world").with_scale(16.0));
    tr.brush.queue(sample_text);
    let processed = tr.brush.process_queued(
        |rect, tex_data| {
            println!("Glyph updated: {:?}", rect);
        },
        |v| {
            // sample output:
            // Glyph produced: GlyphVertex {
            //     tex_coords: Rect {
            //         min: point(0.1328125, 0.00390625),
            //         max: point(0.15625, 0.04296875),
            //     },
            //     pixel_coords: Rect {
            //         min: point(0.0, 3.0),
            //         max: point(6.0, 13.0),
            //     },
            //     bounds: Rect {
            //         min: point(0.0, 0.0),
            //         max: point(120.0, 120.0),
            //     },
            //     extra: PvExtra {
            //         draw_params: TextDrawParams {
            //             ...
            //         },
            //     },
            // }
            let text_vertex = TextVertex::make_glyph(v, (256, 256), tr.origin);
            println!("Triangle produced: {:#?}", text_vertex);
            text_vertex
        },
    );
    println!("=======");
    let sample_text = Section::<'_, PvExtra>::new()
        .with_bounds((120.0, 120.0))
        .add_text(glyph_brush::Text::new("Hello").with_scale(16.0));
    tr.brush.queue(sample_text);
    let processed = tr.brush.process_queued(
        |rect, tex_data| {
            println!("Glyph updated: {:?}", rect);
        },
        |v| {
            println!("Glyph produced: {:?}", v);
            [TextVertex {
                position: [0.0, 0.0, 0.0],
                uv_texcoord: [0, 0],
                diffuse_color: 0,
                emissive_color: 0,
                encoded_normal: 0,
                flags: 0,
                padding: 0,
            }; 4]
        },
    );
}
