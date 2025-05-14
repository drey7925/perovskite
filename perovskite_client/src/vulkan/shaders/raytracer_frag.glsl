#version 460
#extension GL_KHR_shader_subgroup_vote: enable

layout(location = 0) in vec4 global_coord_facedir;
layout(location = 0) out vec4 f_color;

#include "raytracer_bindings.glsl"

const mat2x4 face_swizzlers[] = {
// X+
mat2x4(
vec4(0, 0, 1, 0), vec4(0, -1, 0, 1)
),
// X-
mat2x4(
vec4(0, 0, -1, 1), vec4(0, -1, 0, 1)
),
// Y+
mat2x4(
vec4(-1, 0, 0, 1), vec4(0, 0, 1, 0)
),
// Y-
mat2x4(
vec4(1, 0, 0, 0), vec4(0, 0, 1, 0)
),
// Z+
mat2x4(
vec4(-1, 0, 0, 1), vec4(0, -1, 0, 1)
),
// Z-
mat2x4(
vec4(1, 0, 0, 0), vec4(0, -1, 0, 1)
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
18 * 18, 18 * 18,
1, -1,
18, -18
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
    uint face;
    uint light_data;
};

// Raytraces through a single chunk, returns true if hit, false if no hit.
// (tentative signature, to be updated later)
bool traverse_chunk(uint slot, vec3 g0, vec3 g1, inout HitInfo info) {
    // 4n ints for packed keys table, 5832 ints per chunk, and 343 ints
    // to account for the fact that offset [0,0,0] is partway in chunk (lighting/neighbor data)
    // 343 + 4 = 347 because n-1 rather than n
    uint base = 4 * n_minus_one + 347 + (7328 * slot);
    // 5860 is 5856 (length of block data) + 4 (n-1 compensation)
    uint light_base = 4 * n_minus_one + 5860 + (7328 * slot);

    ivec3 g0idx = ivec3(floor(g0));
    ivec3 g1idx = ivec3(floor(g1));
    ivec3 sgns = ivec3(sign(g1idx - g0idx));

    ivec3 g = g0idx;
    uvec3 gpd = uvec3(
    (g1idx.x > g0idx.x ? 1 : 0),
    (g1idx.y > g0idx.y ? 1 : 0),
    (g1idx.z > g0idx.z ? 1 : 0)
    );

    vec3 gfrac = g0 - g0idx;
    vec3 slope = g1 - g0;

    // Will this vectorize?
    vec3 v = vec3(
    g0.x == g1.x ? 1 : g1.x - g0.x,
    g0.y == g1.y ? 1 : g1.y - g0.y,
    g0.z == g1.z ? 1 : g1.z - g0.z
    );
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
        ivec3 old_g = g;
        ivec3 mask = ivec3(~15);
        uint offset;
        if ((g & mask) == ivec3(0)) {
            offset = g.x * 324 + g.y + g.z * 18;
            block_id = chunks[base + offset];
        } else {
            block_id = 0;
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

        if (block_id != 0) {
            info.block_id = block_id;
            uint l_offset = 343+(offset) + face_backoffs_offset[info.face];
            uint raw_light = chunks[light_base + (l_offset / 4)];
            info.light_data = (raw_light >> (8 * (l_offset & 3u)));
            return true;
        }
        info.face = n_face;
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
    vec3(0.0, -1.0, 0.0),
    vec3(0.0, 1.0, 0.0),
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
    info.face = 0;
    ivec3 g = g0idx;
    uvec3 gpd = uvec3(
    (g1idx.x > g0idx.x ? 1 : 0),
    (g1idx.y > g0idx.y ? 1 : 0),
    (g1idx.z > g0idx.z ? 1 : 0)
    );

    // Will this vectorize?
    vec3 v = vec3(
    g0.x == g1.x ? 1 : g1.x - g0.x,
    g0.y == g1.y ? 1 : g1.y - g0.y,
    g0.z == g1.z ? 1 : g1.z - g0.z
    );
    vec3 derr = vec3(
    v.y * v.z,
    v.x * v.z,
    v.x * v.y
    );
    vec3 err = (gpd - gfrac) * derr;
    derr *= sgns;
    for (;;) {
        vec3 r = abs(err);

        vec3 start_cc = gfrac;
        vec3 end_cc;
        bool should_break = g == g1idx;
        ivec3 old_g = g;


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

        uint n_face;
        if (sgns.x != 0 && (sgns.y == 0 || r.x <= r.y) && (sgns.z == 0 || r.x <= r.z)) {
            g.x += sgns.x;
            float diff = gpd.x - gfrac.x;
            gfrac += diff * (slope / slope.x);
            end_cc = gfrac;
            err.x += derr.x;
            gfrac.x -= sgns.x;
            n_face = gpd.x;
        }
        else if (sgns.y != 0 && (sgns.z == 0 || r.y <= r.z)) {
            g.y += sgns.y;
            float diff = gpd.y - gfrac.y;
            gfrac += diff * (slope / slope.y);
            end_cc = gfrac;
            err.y += derr.y;
            gfrac.y -= sgns.y;
            n_face = 2 + gpd.y;
        }
        else if (sgns.z != 0) {
            g.z += sgns.z;
            float diff = gpd.z - gfrac.z;
            gfrac += diff * (slope / slope.z);
            end_cc = gfrac;
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

        if (slot != 0xffffffff && traverse_chunk(slot, start_cc * 16.0, end_cc * 16.0, info)) {
            return true;
        }
        info.face = n_face;

        if (should_break) {
            return false;
        }
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
            uint dbg_data;
            if (idx0 == 0) {
                dbg_data = n_minus_one;
            }
            if (idx0 == 1) {
                dbg_data = mxc;
            }
            if (idx0 == 2) {
                dbg_data = k.x;
            }
            if (idx0 == 3) {
                dbg_data = k.y;
            }
            if (idx0 == 4) {
                dbg_data = k.z;
            }
            uint idx1 = uint(idx & 31);
            uint bit = dbg_data & (1u << idx1);
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
            } else if (idx0 ==1) {
                f_color = vec4(1.0, 1.0, 0.0, 1.0);
                return;
            } else if (idx0 ==2) {
                f_color = vec4(0.0, 1.0, 0.0, 1.0);
                return;
            } else if (idx0 ==3) {
                f_color = vec4(0.0, 0.0, 1.0, 1.0);
                return;
            } else if (idx0 ==4) {
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
    vec2 pix2 = gl_FragCoord.xy / (2.0 * supersampling);
    int ix2 = int(pix2.x) % 2;
    int iy2 = int(pix2.y) % 2;

    //    if ((ix2 ^ iy2) == 0) {
    //        // Complementary to the control on the raster code
    //        discard;
    //    }
    vec3 facedir = normalize(global_coord_facedir.xyz);

    vec3 facedir_world = vec3(facedir.x, -facedir.y, facedir.z);


    // All raster geometry, as well as non-rendering calcs, assume that blocks have
    // their *centers* at integer coordinates.
    // However, we have the *edges* as axis-aligned. Fix this here.
    // Note that this shader works in world space, with Y up throughout.
    // The Y axis orientation has been flipped in the calculation of facedir_world.
    vec3 fine_pos_fixed = fine_pos + vec3(0.5, 0.5, 0.5);
    vec3 g0 = fine_pos_fixed / 16.0;
    vec3 g1 = (fine_pos_fixed + (500 * facedir_world)) / 16.0;

    HitInfo info = {
    ivec3(0),
    vec3(0),
    vec3(0),
    0,
    0,
    0,
    };
    if (traverse_space(g0, g1, info)) {
        uint idx = info.block_id >> 12;
        if (idx >= max_cube_info_idx) {
            // TODO proper fallback
            idx = 1;
        }
        if ((cube_info[idx].flags & 1) != 0) {
            // TODO: cube rotation :(
            vec2 tl = cube_info[idx].tex[info.face].top_left;
            vec2 wh = cube_info[idx].tex[info.face].width_height;
            vec2 uv = vec4(info.start_cc, 1) * face_swizzlers[info.face];
            vec4 tex_color = texture(tex, tl + (uv * wh));
            float global_brightness_contribution = global_brightness_table[(info.light_data & 0xf0u) >> 4];
            float gbc_adjustment = 0.5 + 0.5 * max(0, dot(global_light_direction, decode_normal(info.face)));
            // TODO: Do a ray query to the sun instead
            vec3 global_light = global_brightness_color * global_brightness_contribution * gbc_adjustment;
            f_color = vec4(
                (brightness_table[info.light_data & 0x0fu] + global_light) * tex_color.rgb, 1.0
            );
            return;
        }
    }
    discard;

}