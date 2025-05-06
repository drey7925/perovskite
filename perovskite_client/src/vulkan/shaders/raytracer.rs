use crate::client_state::settings::Supersampling;
use crate::vulkan::shaders::{LiveRenderConfig, PipelineProvider, PipelineWrapper, SceneState};
use crate::vulkan::{CommandBufferBuilder, VulkanContext, VulkanWindow};
use anyhow::Context;
use cgmath::{vec3, SquareMatrix, Vector3};
use perovskite_core::coordinates::BlockCoordinate;
use smallvec::smallvec;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, ColorBlendAttachmentState, ColorBlendState, ColorComponents,
};
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, FrontFace, RasterizationState};
use vulkano::pipeline::graphics::subpass::PipelineSubpassType;
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::graphics::viewport::{Scissor, Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::Subpass;
use vulkano::shader::ShaderModule;

vulkano_shaders::shader! {
    shaders: {
        raytraced_vtx: {
        ty: "vertex",
        src: r"
            #version 460
                const vec2 vertices[6] = vec2[](
                    vec2(-1.0, -1.0),
                    vec2(-1.0, 1.0),
                    vec2(1.0, 1.0),

                    vec2(-1.0, -1.0),
                    vec2(1.0, 1.0),
                    vec2(1.0, -1.0)
                );

                layout(set = 0, binding = 0) uniform RaytracedUniformData {
                    // Takes an NDC position and transforms it *back* to world space
                    mat4 inverse_vp_matrix;
                    ivec3 coarse_pos;
                    vec3 fine_pos;
                    // vec3 sun_direction;
                    // Used for dither
                    float supersampling;
                };

                layout(location = 0) out vec4 global_coord_facedir;

                void main() {
                    vec4 pos_ndc = vec4(vertices[gl_VertexIndex], 0.5, 1.0);
                    gl_Position = pos_ndc;
                    global_coord_facedir = inverse_vp_matrix * pos_ndc;
                }
            "
        },
        raytraced_frag: {
        ty: "fragment",
        src: r"#version 460
layout(location = 0) in vec4 global_coord_facedir;

layout(set = 0, binding = 0) uniform RaytracedUniformData {
    // Takes an NDC position and transforms it *back* to world space
    mat4 inverse_vp_matrix;
    ivec3 coarse_pos;
    vec3 fine_pos;
    // vec3 sun_direction;
    // Used for dither
    float supersampling;
};
layout(set = 0, binding = 1) readonly buffer chunk_map {
    uint chunks[];
};
layout(location = 0) out vec4 f_color;

uint phash(uvec3 coord, uvec3 k, uint n) {
    uvec3 products = coord * k;
    uint sum = products.x + products.y + products.z;
    return (sum % 1610612741) & (n - 1);
}

uint map_lookup(uvec3 coord, uvec3 k, uint n, uint mx) {
    uvec3 products = coord * k;
    uint sum = products.x + products.y + products.z;
    uint slot = (sum % 1610612741) & (n - 1);
    for (int s = 0; s <= mx; s++) {
        uint base = slot * 4 + 32;
        if ((chunks[base + 3] & 1) == 0) {
          return 0xffffffff;
        }
        if (uvec3(chunks[base], chunks[base + 1], chunks[base + 2]) == coord) {
            return slot;
        }
        slot = (slot + 1) & (n - 1);
    }
    return 0xffffffff;
}

// Raytraces through a single chunk, returns true if hit, false if no hit.
// (tentative signature, to be updated later)
bool traverse_chunk(ivec3 chunk, vec3 g0, vec3 g1, uvec3 k, uint n, uint mx) {
     uvec3 chk = uvec3(chunk + coarse_pos);
     uint res = map_lookup(chk, k, n, mx);
     if (res == 0xffffffff) {
         return false;
     }
	//f_color = vec4(g1 / 16, 1.0);
//return true;

    vec3 g0idx = floor(g0);
    vec3 g1idx = floor(g1);
    vec3 sgns = sign(g1idx - g0idx);

    vec3 g = g0idx;
    vec3 gpd = vec3(
        (g1idx.x > g0idx.x ? 1 : 0),
        (g1idx.y > g0idx.y ? 1 : 0),
        (g1idx.z > g0idx.z ? 1 : 0)
    );
    vec3 gp = g0idx + gpd;

    // Will this vectorize?
    vec3 v = vec3(
        g0.x == g1.x ? 1 : g1.x - g0.x,
        g0.y == g1.y ? 1 : g1.y - g0.y,
        g0.z == g1.z ? 1 : g1.z - g0.z
    );
    vec3 v2 = vec3(
        v.y * v.z,
        v.x * v.z,
        v.x * v.y
    );
    vec3 v2d = v2 / (v.x * v.y * v.z);
    vec3 err = (gp - g0) * v2;
    vec3 derr = sgns * v2;
    int i = 0;
    vec3 prev_int = g0;
//
//        if ((g0idx.y > 2.5) && (g0idx.y < 3.5)) {
//           f_color = vec4(1.0, 0.0, 0.0, 1.0);
//           return true;
//        }
//        if ((g1idx.y == 4)) {
//           f_color = vec4(1.0, 1.0, 0.0, 1.0);
//           return true;
//        }
if ((g.y < 0 || g.y > 16)) {
            f_color = vec4(0.0, 1.0, 0.0, 1.0);
            return true;
        }
if ((g.z < 0 || g.z > 16)) {
            f_color = vec4(0.0, 0.0, 1.0, 1.0);
            return true;
        }

    for (; i < 60; i++) {
        if ((g.x == 0 || g.x == 1) && (g.y == 1 || g.y == 0) && (g.z == 1 || g.z == 0) ) {
            f_color = vec4(gpd / 2.0 + vec3(0.5, 0.5, 0.5), 1.0);
            return true;
        }

//        if ((g.y == 3) && (g.x == 4)) {
//            f_color = vec4(0.0, 1.0, 1.0, 1.0);
//            return true;
//        }

        if (g == g1idx) {
            break;
        }
        vec3 r = abs(err);

        if (sgns.x != 0 && (sgns.y == 0 || r.x < r.y) && (sgns.z == 0 || r.x < r.z)) {
            g.x += sgns.x;
            err.x += derr.x;
        }
        else if (sgns.y != 0 && (sgns.z == 0 || r.y < r.z)) {
            g.y += sgns.y;
            err.y += derr.y;
        }
        else if (sgns.z != 0) {
            g.z += sgns.z;
            err.z += derr.z;

        } else {
            f_color = vec4(1.0, 1.0, 0.0, 1.0);
            return true;
        }


    }
    if (i >= 59) {
        f_color = vec4(1.0, 0.0, 0.0, 1.0);
    } else {
        f_color = vec4(0.0, 0.0, 0.0, 1.0);
    }
    return false;
}

void main() {
    vec2 pix3 = gl_FragCoord.xy / (16.0 * supersampling);
    int ix3 = int(pix3.x);
    int iy3 = int(pix3.y);
    if (iy3 == 1) {
      if (ix3 <= 160 && ix3 >= 1) {
        int idx = ix3 - 1;
        int idx0 = idx >> 5;
        uint idx1 = uint(idx & 31);
        uint bit = chunks[idx0] & (1 << idx1);
        if (bit != 0) {
            f_color = vec4(1.0, 0.0, 0.0, 1.0);
            return;
        } else {
            f_color = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }
      }
    }
    if (iy3 == 2) {
      if (ix3 <= 160 && ix3 >= 1) {
        int idx = ix3 - 1;
        int idx0 = idx >> 5;
        if (idx0 == 0) {
            f_color = vec4(1.0, 0.0, 0.0, 1.0);
            return;
        } else if (idx0 ==1 ) {
            f_color = vec4(1.0, 1.0, 0.0, 1.0);
            return;
        } else if (idx0 ==2 ) {
            f_color = vec4(0.0, 1.0, 0.0, 1.0);
            return;
        } else if (idx0 ==3 ) {
            f_color = vec4(0.0, 0.0, 1.0, 1.0);
            return;
        } else if (idx0 ==4 ) {
            f_color = vec4(1.0, 1.0, 0.0, 1.0);
            return;
        }
      }
    }

    if (iy3 == 3) {
      if (ix3 <= 160 && ix3 >= 1) {
        int idx = ix3 - 1;
        if ((idx % 2) == 0) {
            f_color = vec4(0.2, 1.0, 0.4, 1.0);
            return;
        }
        else {
            f_color = vec4(1.0, 1.0, 0.0, 1.0);
            return;
        }
      }
    }

    // Raytracing experiment
    vec2 pix2 = gl_FragCoord.xy / (2.0 * supersampling);
    int ix2 = int(pix2.x) % 2;
    int iy2 = int(pix2.y) % 2;

    if ((ix2 ^ iy2) == 0) {
        // Complementary to the control on the raster code
        discard;
    }
    vec3 facedir = normalize(global_coord_facedir.xyz);

    vec3 facedir_world = vec3(facedir.x, -facedir.y, facedir.z);
    uint n = chunks[0];
    uint mx = chunks[1];
    uint k1 = uint(chunks[2]);
    uint k2 = uint(chunks[3]);
    uint k3 = uint(chunks[4]);
    uvec3 k = uvec3(k1, k2, k3);


    // All raster geometry, as well as non-rendering calcs, assume that blocks have
    // their *centers* at integer coordinates.
    // However, we have the *edges* as axis-aligned. Fix this here.
    // Note that this shader works in world space, with Y up throughout.
    // The Y axis orientation has been flipped in the calculation of facedir_world.
    vec3 fine_pos_fixed = fine_pos + vec3(0.5, 0.5, 0.5);
    vec3 g0 = fine_pos_fixed / 16.0;
    vec3 g1 = (fine_pos_fixed + (500 * facedir_world)) / 16.0;

    vec3 g0idx = floor(g0);
    vec3 gfrac = g0 - g0idx;
    vec3 slope = g1 - g0;
    vec3 g1idx = floor(g1);
    vec3 sgns = sign(g1idx - g0idx);

    vec3 g = g0idx;
    vec3 gpd = vec3(
        (g1idx.x > g0idx.x ? 1 : 0),
        (g1idx.y > g0idx.y ? 1 : 0),
        (g1idx.z > g0idx.z ? 1 : 0)
    );
    vec3 gp = g0idx + gpd;

    // Will this vectorize?
    vec3 v = vec3(
        g0.x == g1.x ? 1 : g1.x - g0.x,
        g0.y == g1.y ? 1 : g1.y - g0.y,
        g0.z == g1.z ? 1 : g1.z - g0.z
    );
    vec3 v2 = vec3(
        v.y * v.z,
        v.x * v.z,
        v.x * v.y
    );
    vec3 v2d = v2 / (v.x * v.y * v.z);
    vec3 err = (gp - g0) * v2;
    vec3 derr = sgns * v2;
    int i = 0;
    vec3 prev_int = g0;
    vec3 col = vec3(1.0, 1.0, 1.0);
    for (; i < 60; i++) {
        if (g == g1idx) {
            break;
        }
        vec3 r = abs(err);

        vec3 start_cc = gfrac;
		vec3 end_cc;
        vec3 old_g = g;
        //col = gfrac;
        if (sgns.x != 0 && (sgns.y == 0 || r.x < r.y) && (sgns.z == 0 || r.x < r.z)) {
            g.x += sgns.x;
            float diff = gpd.x - gfrac.x;
            gfrac += diff * (slope / slope.x);
			end_cc = gfrac;
            err.x += derr.x;
            gfrac.x -= sgns.x;
            col = gfrac;
        }
        else if (sgns.y != 0 && (sgns.z == 0 || r.y < r.z)) {
            g.y += sgns.y;
            float diff = gpd.y - gfrac.y;
            gfrac += diff * (slope / slope.y);
			end_cc = gfrac;
            err.y += derr.y;
            gfrac.y -= sgns.y;
            col = gfrac;
        }
        else if (sgns.z != 0) {
            g.z += sgns.z;
            float diff = gpd.z - gfrac.z;
            gfrac += diff * (slope / slope.z);
			end_cc = gfrac;
            err.z += derr.z;
            gfrac.z -= sgns.z;
            col = gfrac;

        } else {
            f_color = vec4(1.0, 1.0, 0.0, 1.0);
            return;
        }

        if (traverse_chunk(ivec3(old_g), start_cc * 16.0, end_cc * 16.0, k, n, mx)) {
            return;
        }
//        ivec3 t = ivec3(g);
//        uvec3 chk = uvec3(t + cpos);
//        uint res = map_lookup(chk, k, n, mx);
//        if (res != 0xffffffff) {
//            if ((col.x < 0.0625 || col.x > 0.125) && (col.y < 0.0625 || col.y > 0.125) && (col.z < 0.0625 || col.z > 0.125)) {
//                f_color = vec4(col, 1.0);
//                return;
//            } else {
//                f_color = vec4(0.0, 0.0, 0.0, 1.0);
//                return;
//            }
//        }
    }
    if (i >= 59) {
        f_color = vec4(1.0, 0.0, 0.0, 1.0);
    } else {
        f_color = vec4(0.0, 0.0, 0.0, 1.0);
    }
}"
        }
    }
}

pub(crate) struct RaytracedPipelineWrapper {
    pipeline: Arc<GraphicsPipeline>,
    supersampling: Supersampling,
}

impl PipelineWrapper<(), (SceneState, Subbuffer<[u32]>, Vector3<f64>)>
    for RaytracedPipelineWrapper
{
    type PassIdentifier = ();

    fn draw<L>(
        &mut self,
        builder: &mut CommandBufferBuilder<L>,
        _draw_calls: (),
        _pass: Self::PassIdentifier,
    ) -> anyhow::Result<()> {
        unsafe {
            // Safety: TODO
            builder.draw(6, 1, 0, 0)?;
        }
        Ok(())
    }

    fn bind<L>(
        &mut self,
        ctx: &VulkanContext,
        per_frame_config: (SceneState, Subbuffer<[u32]>, Vector3<f64>),
        command_buf_builder: &mut CommandBufferBuilder<L>,
        _pass: Self::PassIdentifier,
    ) -> anyhow::Result<()> {
        let (per_frame_config, ssbo, player_pos) = per_frame_config;
        command_buf_builder.bind_pipeline_graphics(self.pipeline.clone())?;
        let layout = self.pipeline.layout().clone();

        let player_chunk = BlockCoordinate::try_from(player_pos)?.chunk();
        let fine = player_pos
            - vec3(
                (player_chunk.x * 16) as f64,
                (player_chunk.y * 16) as f64,
                (player_chunk.z * 16) as f64,
            );

        let per_frame_data = RaytracedUniformData {
            inverse_vp_matrix: per_frame_config
                .vp_matrix
                .invert()
                .with_context(|| {
                    format!("VP matrix was singular: {:?}", per_frame_config.vp_matrix)
                })?
                .into(),
            supersampling: self.supersampling.to_float(),
            coarse_pos: [player_chunk.x, player_chunk.y, player_chunk.z].into(),
            fine_pos: [fine.x as f32, fine.y as f32, fine.z as f32].into(),
        };
        let per_frame_set_layout = layout
            .set_layouts()
            .get(0)
            .with_context(|| "Raytraced layout missing set 0")?;
        let uniform_buffer = Buffer::from_data(
            ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                    | MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            per_frame_data,
        )?;
        let per_frame_set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            per_frame_set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, uniform_buffer),
                WriteDescriptorSet::buffer(1, ssbo),
            ],
            [],
        )?;
        command_buf_builder.bind_descriptor_sets(
            vulkano::pipeline::PipelineBindPoint::Graphics,
            layout,
            0,
            vec![per_frame_set],
        )?;
        Ok(())
    }
}

pub(crate) struct RaytracedPipelineProvider {
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
}
impl PipelineProvider for RaytracedPipelineProvider {
    type DrawCall<'a> = ();
    type PerPipelineConfig<'a> = ();
    type PerFrameConfig = (SceneState, Subbuffer<[u32]>, Vector3<f64>);
    type PipelineWrapperImpl = RaytracedPipelineWrapper;

    fn make_pipeline(
        &self,
        ctx: &VulkanWindow,
        _config: Self::PerPipelineConfig<'_>,
        global_config: &LiveRenderConfig,
    ) -> anyhow::Result<Self::PipelineWrapperImpl> {
        let vs = self
            .vs
            .entry_point("main")
            .context("Missing vertex shader")?;
        let fs = self
            .fs
            .entry_point("main")
            .context("Missing fragment shader")?;
        let stages = smallvec![
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];
        let layout = PipelineLayout::new(
            self.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(self.device.clone())?,
        )?;
        let pipeline_info = GraphicsPipelineCreateInfo {
            stages,
            // No bindings or attributes
            vertex_input_state: Some(VertexInputState::new()),
            input_assembly_state: Some(InputAssemblyState::default()),
            rasterization_state: Some(RasterizationState {
                cull_mode: CullMode::Back,
                front_face: FrontFace::CounterClockwise,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            viewport_state: Some(ViewportState {
                viewports: smallvec![Viewport {
                    offset: [0.0, 0.0],
                    depth_range: 0.0..=1.0,
                    extent: [
                        ctx.viewport.extent[0] * global_config.supersampling.to_float(),
                        ctx.viewport.extent[1] * global_config.supersampling.to_float()
                    ],
                }],
                scissors: smallvec![Scissor {
                    offset: [0, 0],
                    extent: [
                        ctx.viewport.extent[0] as u32 * global_config.supersampling.to_int(),
                        ctx.viewport.extent[1] as u32 * global_config.supersampling.to_int()
                    ],
                }],
                ..Default::default()
            }),
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState {
                    // No depth test whatsoever
                    compare_op: CompareOp::Always,
                    write_enable: false,
                }),
                depth_bounds: Default::default(),
                stencil: Default::default(),
                ..Default::default()
            }),
            color_blend_state: Some(ColorBlendState {
                attachments: vec![ColorBlendAttachmentState {
                    blend: Some(AttachmentBlend::alpha()),
                    color_write_mask: ColorComponents::all(),
                    color_write_enable: true,
                }],
                ..Default::default()
            }),
            subpass: Some(PipelineSubpassType::BeginRenderPass(
                Subpass::from(ctx.ssaa_render_pass.clone(), 0).context("Missing subpass")?,
            )),
            ..GraphicsPipelineCreateInfo::layout(layout.clone())
        };
        let pipeline = GraphicsPipeline::new(self.device.clone(), None, pipeline_info)?;
        Ok(RaytracedPipelineWrapper {
            pipeline,
            supersampling: global_config.supersampling,
        })
    }
}

impl RaytracedPipelineProvider {
    pub(crate) fn new(device: Arc<Device>) -> anyhow::Result<Self> {
        Ok(RaytracedPipelineProvider {
            vs: load_raytraced_vtx(device.clone())?,
            fs: load_raytraced_frag(device.clone())?,
            device,
        })
    }
}
