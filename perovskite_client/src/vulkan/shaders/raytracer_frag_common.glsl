layout(location = 0) in vec3 facedir_world_in;

#include "raytracer_spec_constants.glsl"

// Performance note: It is tempting to optimize this shader to use float16. It produces some small artifacts, but
// furthermore it actually causes a small performance loss (at least on Ampere) despite SM occupancy increasing.

struct TexRef {
    vec2 top_left;
    vec2 width_height;
};

struct SimpleCubeInfo {
    uint flags;
    TexRef tex[6];
};

layout(set = 0, binding = 0) uniform sampler2D diffuse_tex;
layout(set = 0, binding = 1) uniform sampler2D specular_tex;
layout(set = 0, binding = 2) readonly buffer RaytraceControl {
    SimpleCubeInfo cube_info[];
};


layout(set = 1, binding = 1) uniform ChunkMapHeader {
    uint n_minus_one;
    uint mxc;
    uvec3 k;
    ivec3 min_chunk;
    ivec3 max_chunk;
};
layout(set = 1, binding = 2) readonly buffer chunk_map {
    uint chunks[];
};

// Each shader derived from this common file should declare its own storage image bindings
// with the appropriate binding indices and readonly/writeonly

//layout (set = 1, binding = 3, rgba8) uniform restrict image2D deferred_specular_color;
//layout (set = 1, binding = 4, rgba32f) uniform restrict image2D deferred_specular_ray_dir;
//layout(input_attachment_index = 0, set = 1, binding = 5) uniform subpassInput f_depth_in;

#include "raytracer_bindings.glsl"
#include "sky.glsl"

const mat2x3 face_swizzlers[] = {
// X+
mat2x3(
vec3(0, 0, 1), vec3(0, -1, 0)
),
// X-
mat2x3(
vec3(0, 0, -1), vec3(0, -1, 0)
),
// Y+
mat2x3(
vec3(-1, 0, 0), vec3(0, 0, 1)
),
// Y-
mat2x3(
vec3(1, 0, 0), vec3(0, 0, 1)
),
// Z+
mat2x3(
vec3(-1, 0, 0), vec3(0, -1, 0)
),
// Z-
mat2x3(
vec3(1, 0, 0), vec3(0, -1, 0)
),
};

const float global_brightness_table[] = {
0.0, 0.015625, 0.044194173, 0.08118988,
0.125, 0.17469281, 0.22963966, 0.28937906,
0.35355338, 0.421875, 0.49410588, 0.5700449,
0.649519, 0.7323776, 0.8184875, 0.90773046
};

const float brightness_table[] = {
0.03125, 0.07432544, 0.123381935, 0.17677669,
0.23364824, 0.29345337, 0.35581362, 0.4204482,
0.48713928, 0.55571234, 0.6260238, 0.69795364,
0.77139926, 0.8462722, 0.9224952, 1.0
};

const int face_backoffs_offset[] = {
18 * 18, -18 * 18,
1, -1,
18, -18
};

const vec3 face_reflectors[] = {
vec3(-1, 1, 1), vec3(-1, 1, 1),
vec3(1, -1, 1), vec3(1, -1, 1),
vec3(1, 1, -1), vec3(1, 1, -1),
};

const vec3 debug_face_colors[] = {
vec3(1, 0, 0), vec3(1, 1, 0),
vec3(0, 1, 0), vec3(0, 1, 1),
vec3(0, 0, 1), vec3(1, 0, 1),
};

uint phash(uvec3 coord, uvec3 k, uint n_minus_one) {
    uvec3 products = coord * k;
    uint sum = products.x + products.y + products.z;
    return (sum % 1610612741) & n_minus_one;
}

//uint map_lookup(uvec3 coord, uvec3 k, uint n, uint mx) {
//    uvec3 products = coord * k;
//    uint sum = products.x + products.y + products.z;
//    uint slot = (sum % 1610612741) & (n - 1);
//    for (int s = 0; s <= mx; s++) {
//        uint base = slot * 4;
//        if ((chunks[base + 3] & 1) == 0) {
//            return 0xffffffff;
//        }
//        if (uvec3(chunks[base], chunks[base + 1], chunks[base + 2]) == coord) {
//            return slot;
//        }
//        slot = (slot + 1) & (n - 1);
//    }
//    return 0xffffffff;
//}

struct HitInfo {
    ivec3 hit_block;
    vec3 start_cc;
    vec3 end_cc;
    uint block_id;
    uint face_light;
};

// Raytraces through a single chunk, returns true if hit, false if no hit.
// (tentative signature, to be updated later)
//
// input: hit_info's start_cc/end_cc represent the hit state of this chunk when we start traversing
// output: start_cc/end_cc represent the hit state of the block we hit, if there was a hit. Otherwise, undefined
bool traverse_chunk(uint slot, inout HitInfo info) {
    info.start_cc *= 16;
    info.end_cc *= 16;
    // 4n ints for packed keys table, 5832 ints per chunk, and 343 ints
    // to account for the fact that offset [0,0,0] is partway in chunk (lighting/neighbor data)
    // 343 + 4 = 347 because n-1 rather than n
    uint base = 4 * n_minus_one + 347 + (7328 * slot);
    // 5860 is 5856 (length of block data) + 4 (n-1 compensation)
    uint light_base = 4 * n_minus_one + 5860 + (7328 * slot);

    ivec3 g = ivec3(floor(info.start_cc));
    ivec3 g1idx = ivec3(floor(info.end_cc));
    ivec3 sgns = ivec3(sign(g1idx - g));

    uvec3 gpd = uvec3(
    (g1idx.x > g.x ? 1 : 0),
    (g1idx.y > g.y ? 1 : 0),
    (g1idx.z > g.z ? 1 : 0)
    );

    vec3 gfrac = info.start_cc - g;
    vec3 slope = info.end_cc - info.start_cc;

    vec3 v = mix(info.end_cc - info.start_cc, vec3(1), equal(info.start_cc, info.end_cc));
    vec3 derr = vec3(
    v.y * v.z,
    v.x * v.z,
    v.x * v.y
    );
    vec3 err = (gpd - gfrac) * derr;
    derr *= sgns;
    // CubeFace is structured such that gpd can be used to get the correct side
    for (int i = 0; i < 60; i++) {
        info.start_cc = gfrac;

        //        if ((g.y == 3) && (g.x == 4)) {
        //            f_color = vec4(0.0, 1.0, 1.0, 1.0);
        //            return true;
        //        }

        vec3 r = abs(err);

        // Hide latency by overlaying fetch with next-block calc
        uint block_id;
        bool should_break = g == g1idx;

        info.hit_block = g;
        ivec3 mask = ivec3(~15);
        uint offset;
        if ((g & mask) == ivec3(0)) {
            offset = g.x * 324 + g.y + g.z * 18;
            block_id = chunks[base + offset];
        } else {
            block_id = 0xffffffff;
        }
        uint n_face;
        if (sgns.x != 0 && (sgns.y == 0 || r.x <= r.y) && (sgns.z == 0 || r.x <= r.z)) {
            g.x += sgns.x;
            float diff = gpd.x - gfrac.x;
            // can we absorb gfrac into end_cc?
            gfrac += diff * (slope / slope.x);
            info.end_cc = gfrac;
            err.x += derr.x;
            gfrac.x -= sgns.x;
            n_face = gpd.x;
        }
        else if (sgns.y != 0 && (sgns.z == 0 || r.y <= r.z)) {
            g.y += sgns.y;
            float diff = gpd.y - gfrac.y;
            gfrac += diff * (slope / slope.y);
            info.end_cc = gfrac;
            err.y += derr.y;
            gfrac.y -= sgns.y;
            n_face = 2 + gpd.y;
        }
        else if (sgns.z != 0) {
            g.z += sgns.z;
            float diff = gpd.z - gfrac.z;
            gfrac += diff * (slope / slope.z);
            info.end_cc = gfrac;
            err.z += derr.z;
            gfrac.z -= sgns.z;
            n_face = 4 + gpd.z;
        }

        if ((block_id != 0)
        && ((block_id & 0xfffff000u) != (info.block_id & 0xfffff000u))
        && (block_id != 0xffffffff)
        && ((cube_info[block_id >> 12].flags & SKIP_MASK) == 0)) {
            info.block_id = block_id;
            uint l_offset = 343+(offset) + face_backoffs_offset[info.face_light & 7u];
            uint raw_light = chunks[light_base + (l_offset / 4)] >> (8 * (l_offset & 3u));
            info.face_light |= (raw_light << 8);
            return true;
        }
        if (block_id != 0xffffffff) {
            info.block_id = block_id;
        }
        info.face_light = n_face;
        if (should_break) {
            return false;
        }
    }
    return false;
}

vec3 decode_normal(uint index) {
    const float sqrt_half = sqrt(0.5);
    // Matches CubeFace in BlockRenderer
    const vec3 normals[10] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(-1.0, 0.0, 0.0),
    // Warning: CubeFace Y+ then Y- is in world coords, not Vk coords
    // This has Vk coords, to match up with the global light direction in Vk coords
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, -1.0, 0.0),
    vec3(0.0, 0.0, 1.0),
    vec3(0.0, 0.0, -1.0),
    vec3(sqrt_half, 0.0, sqrt_half),
    vec3(sqrt_half, 0.0, -sqrt_half),
    vec3(-sqrt_half, 0.0, sqrt_half),
    vec3(-sqrt_half, 0.0, -sqrt_half)
    );
    return normals[index];
}

bool traverse_space(vec3 g0, vec3 g1, inout HitInfo info) {
    ivec3 g0idx = ivec3(floor(g0));
    vec3 gfrac = g0 - g0idx;
    vec3 slope = g1 - g0;
    ivec3 g1idx = ivec3(floor(g1));
    ivec3 sgns = sign(g1idx - g0idx);
    ivec3 g = g0idx;
    uvec3 gpd = uvec3(
    (g1idx.x > g0idx.x ? 1 : 0),
    (g1idx.y > g0idx.y ? 1 : 0),
    (g1idx.z > g0idx.z ? 1 : 0)
    );

    vec3 v = mix(g1 - g0, vec3(1), equal(g1, g0));

    vec3 derr = vec3(
    v.y * v.z,
    v.x * v.z,
    v.x * v.y
    );
    vec3 err = (gpd - gfrac) * derr;
    derr *= sgns;
    uint slot_base;
    uint try_slot;
    for (uint i = 0; i < render_distance; i++) {
        uvec3 chk = uvec3(g + coarse_pos);
        // Hide latency by interleaving map lookup with next-chunk calc
        // We do this by doing the first lookup now, and hoping that we have a prefetched cacheline
        // by the time we finish the math for next chunk

        // This is inlined from the old map lookup function and rearranged
        uvec3 products = chk * k;
        uint sum = products.x + products.y + products.z;
        uint try_slot = (sum % 1610612741) & n_minus_one;
        uint slot_base = try_slot * 4;
        uint slot_flag = chunks[slot_base];

        vec3 r = abs(err);

        info.start_cc = gfrac;
        bool should_break = g == g1idx;

        uint n_face;
        if (sgns.x != 0 && (sgns.y == 0 || r.x <= r.y) && (sgns.z == 0 || r.x <= r.z)) {
            g.x += sgns.x;
            float diff = gpd.x - gfrac.x;
            gfrac += diff * (slope / slope.x);
            info.end_cc = gfrac;
            err.x += derr.x;
            gfrac.x -= sgns.x;
            n_face = gpd.x;
        }
        else if (sgns.y != 0 && (sgns.z == 0 || r.y <= r.z)) {
            g.y += sgns.y;
            float diff = gpd.y - gfrac.y;
            gfrac += diff * (slope / slope.y);
            info.end_cc = gfrac;
            err.y += derr.y;
            gfrac.y -= sgns.y;
            n_face = 2 + gpd.y;
        }
        else if (sgns.z != 0) {
            g.z += sgns.z;
            float diff = gpd.z - gfrac.z;
            gfrac += diff * (slope / slope.z);
            info.end_cc = gfrac;
            err.z += derr.z;
            gfrac.z -= sgns.z;
            n_face = 4 + gpd.z;
        } else {
            return false;
        }

        uint slot = 0xffffffff;
        for (int s = 0; s <= mxc; s++) {
            if (slot_flag == 0) {
                break;
            }
            if (uvec3(chunks[slot_base + 1], chunks[slot_base + 2], chunks[slot_base + 3]) == chk) {
                slot = try_slot;
                break;
            }
            try_slot = (try_slot + 1) & n_minus_one;
            slot_base = try_slot * 4;
            slot_flag = chunks[slot_base];
        }

        if (slot != 0xffffffff && traverse_chunk(slot, info) && ((slot_flag & 4u) == 0)) {
            // We need to restore the old value of g prior to updates.
            // Holding it in a register will increase register pressure.
            // However, we know that chk = old_g + coarse_pos, and it's OK for hits to be mildly expensive
            info.hit_block += 16 * (ivec3(chk) - coarse_pos);
            return true;
        }
        info.face_light = n_face;

        if (should_break) {
            return false;
        }
    }
    return false;
}

// Compute the range of distances that could possibly hit a chunk
// returns min, max
vec2 t_range(vec3 start, vec3 dir) {
    dir = normalize(dir);
    vec3 c_min = min_chunk - coarse_pos;
    vec3 c_max = max_chunk - coarse_pos;
    vec3 t_for_min = (c_min - start) / dir;
    vec3 t_for_max = (c_max + 1 - start) / dir;
    vec3 t_min = min(t_for_min, t_for_max);
    vec3 t_max = max(t_for_min, t_for_max);
    return vec2(
    max(0, max(t_min.x, max(t_min.y, t_min.z))),
    min(render_distance, min(t_max.x, min(t_max.y, t_max.z)))
    );
}

struct SampleResult {
    vec4 diffuse;
    vec4 specular;
};

const mat2x2 rotations[] = {
mat2x2(vec2(1, 0), vec2(0, 1)),
mat2x2(vec2(0, -1), vec2(1, 0)),
mat2x2(vec2(-1, 0), vec2(0, -1)),
mat2x2(vec2(0, 1), vec2(-1, 0)),
};
const uint face_remaps[6][4] = {
{ 0, 4, 1, 5 },
{ 1, 5, 0, 4 },
{ 2, 2, 2, 2 },
{ 3, 3, 3, 3 },
{ 4, 1, 5, 0 },
{ 5, 0, 4, 1 },
};

SampleResult sample_simple(HitInfo info, uint idx, bool want_spec) {
    uint face = info.face_light & 7u;
    vec3 start_cc = info.start_cc;

    if ((cube_info[idx].flags & 2u) != 0) {
        uint variant = info.block_id & 3u;
        start_cc.xz = ((start_cc.xz - 0.5) * rotations[variant]) + 0.5;
        face = face_remaps[face][variant];
    }

    vec2 tl = cube_info[idx].tex[face].top_left;
    vec2 wh = cube_info[idx].tex[face].width_height;
    vec2 uv = ((start_cc - 0.5) * face_swizzlers[face]) + 0.5;
    vec2 texel = tl + (uv * wh);

    vec4 diffuse = texture(diffuse_tex, texel);
    vec4 specular = vec4(0);

    if (SPECULAR) {
        if (want_spec) {
            specular = texture(specular_tex, texel);
        }
    }

    // For debugging
    // allow seeing some of the texture, plus avoid the sampler disappearing from the final shader program
    //vec4 tex_color = vec4(debug_face_colors[info.face], 1.0) + 0.05 * texture(tex, texel);

    float global_brightness_contribution = global_brightness_table[bitfieldExtract(info.face_light, 12, 4)];
    float gbc_adjustment = 0.5 + 0.5 * max(0, dot(sun_direction, decode_normal(face)));
    // TODO: Do a ray query to the sun instead
    vec3 global_light = global_brightness_color * global_brightness_contribution * gbc_adjustment;
    vec4 final_diffuse = vec4((brightness_table[bitfieldExtract(info.face_light, 8, 4)] + global_light) * diffuse.rgb, diffuse.a);

    return SampleResult(
    final_diffuse,
    specular
    );
}

float random (vec2 st, float f) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * f);
}

