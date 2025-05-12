#version 460
#extension GL_KHR_shader_subgroup_vote: enable

layout(location = 0) in vec4 global_coord_facedir;
layout(location = 0) out vec4 f_color;

#include "raytracer_bindings.glsl"

const mat2x4 face_swizzlers[] = {
// X+
mat2x4(
vec4(0, 0, -1, 1), vec4(0, -1, 0, 1)
),
// X-
mat2x4(
vec4(0, 0, 1, 0), vec4(0, -1, 0, 1)
),
// Y+
mat2x4(
vec4(1, 0, 0, 0), vec4(0, 0, 1, 0)
),
// Y-
mat2x4(
vec4(-1, 0, 0, 1), vec4(0, 0, -1, 1)
),
// Z+
mat2x4(
vec4(1, 0, 0, 0), vec4(0, -1, 0, 1)
),
// Z-
mat2x4(
vec4(-1, 0, 0, 1), vec4(0, -1, 0, 1)
),
};

const int face_backoffs_offset[] = {
18 * 18, -18 * 18,
-1, -1,
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
//        uint base = slot * 4 + 32;
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

// Raytraces through a single chunk, returns true if hit, false if no hit.
// (tentative signature, to be updated later)
bool traverse_chunk(uint slot, vec3 g0, vec3 g1, uvec3 k, uint n_minus_one, uint mx, uint face) {
    // 32 ints for header (may move to a UBO later), 4n ints for packed keys table, 5832 ints per chunk, and 343 ints
    // to account for the fact that offset [0,0,0] is partway in chunk (lighting/neighbor data)
    // 32 + 343 = 375
    // but also 379 because n-1 rather than n
    uint base = 4 * n_minus_one + 379 + (7328 * slot);

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
    vec3 v2 = vec3(
    v.y * v.z,
    v.x * v.z,
    v.x * v.y
    );
    vec3 v2d = v2 / (v.x * v.y * v.z);
    vec3 err = (gpd - gfrac) * v2;
    vec3 derr = sgns * v2;
    int i = 0;

    // CubeFace is structured such that gpd can be used to get the correct side
    for (; i < 60; i++) {


        vec3 start_cc = gfrac;
        vec3 end_cc;

        //        if ((g.y == 3) && (g.x == 4)) {
        //            f_color = vec4(0.0, 1.0, 1.0, 1.0);
        //            return true;
        //        }

        vec3 r = abs(err);

        // Hide latency by overlaying fetch with next-block calc
        uint block_id;
        ivec3 old_g = g;
        ivec3 mask = ivec3(~15);
        if ((g & mask) == ivec3(0)) {
            uint offset = g.x * 324 + g.y + g.z * 18;
            block_id = chunks[base + offset];
        } else {
            block_id = 0;
        }
        uint n_face;
        vec2 luv;
        if (sgns.x != 0 && (sgns.y == 0 || r.x < r.y) && (sgns.z == 0 || r.x < r.z)) {
            g.x += sgns.x;
            float diff = gpd.x - gfrac.x;
            gfrac += diff * (slope / slope.x);
            end_cc = gfrac;
            err.x += derr.x;
            gfrac.x -= sgns.x;
            n_face = 1 - gpd.x;
            luv = gfrac.yz;
        }
        else if (sgns.y != 0 && (sgns.z == 0 || r.y < r.z)) {
            g.y += sgns.y;
            float diff = gpd.y - gfrac.y;
            gfrac += diff * (slope / slope.y);
            end_cc = gfrac;
            err.y += derr.y;
            gfrac.y -= sgns.y;
            n_face = 2 + gpd.y;
            luv = gfrac.xz;
        }
        else if (sgns.z != 0) {
            g.z += sgns.z;
            float diff = gpd.z - gfrac.z;
            gfrac += diff * (slope / slope.z);
            end_cc = gfrac;
            err.z += derr.z;
            gfrac.z -= sgns.z;
            n_face = 5 - gpd.z;
            luv = gfrac.xy;
        }

        if (block_id != 0) {
            uint idx = block_id >> 12;
            if (idx >= max_cube_info_idx) {
                // TODO proper fallback
                idx = 1;
            }
            if ((cube_info[idx].flags & 1) != 0) {
                // TODO: cube rotation :(
                vec2 tl = cube_info[idx].tex[face].top_left;
                vec2 wh = cube_info[idx].tex[face].width_height;
                vec2 uv = vec4(start_cc, 1) * face_swizzlers[face];
                f_color = texture(tex, tl + (uv * wh));
                return true;
            }
        }
        face = n_face;
        if (old_g == g1idx) {
            break;
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
    uint last_face = 0;
    // Raytracing experiment
    vec2 pix2 = gl_FragCoord.xy / (2.0 * supersampling);
    int ix2 = int(pix2.x) % 2;
    int iy2 = int(pix2.y) % 2;

    //    if ((ix2 ^ iy2) == 0) {
    //        // Complementary to the control on the raster code
    //        discard;
    //    }
    vec3 facedir = normalize(global_coord_facedir.xyz);

    vec3 facedir_world = vec3(facedir.x, -facedir.y, facedir.z);
    uint n_minus_one = chunks[0] - 1;
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
    vec3 err = (gpd - gfrac) * v2;
    vec3 derr = sgns * v2;
    int i = 0;
    vec3 col = vec3(1.0, 1.0, 1.0);
    for (; i < 60; i++) {
        vec3 r = abs(err);

        vec3 start_cc = gfrac;
        vec3 end_cc;
        ivec3 old_g = g;

        uvec3 chk = uvec3(g + coarse_pos);
        // Hide latency by interleaving map lookup with next-chunk calc
        // We do this by doing the first lookup now, and hoping that we have a prefetched cacheline
        // by the time we finish the math for next chunk

        // This is inlined from the old map lookup function and rearranged
        uvec3 products = chk * k;
        uint sum = products.x + products.y + products.z;
        uint try_slot = (sum % 1610612741) & n_minus_one;
        uint slot_base = try_slot * 4 + 32;
        uint slot_flag = chunks[slot_base];

        uint n_face;
        //col = gfrac;
        if (sgns.x != 0 && (sgns.y == 0 || r.x < r.y) && (sgns.z == 0 || r.x < r.z)) {
            g.x += sgns.x;
            float diff = gpd.x - gfrac.x;
            gfrac += diff * (slope / slope.x);
            end_cc = gfrac;
            err.x += derr.x;
            gfrac.x -= sgns.x;
            col = gfrac;
            n_face = 1 - gpd.x;
        }
        else if (sgns.y != 0 && (sgns.z == 0 || r.y < r.z)) {
            g.y += sgns.y;
            float diff = gpd.y - gfrac.y;
            gfrac += diff * (slope / slope.y);
            end_cc = gfrac;
            err.y += derr.y;
            gfrac.y -= sgns.y;
            col = gfrac;
            n_face = 2 + gpd.y;
        }
        else if (sgns.z != 0) {
            g.z += sgns.z;
            float diff = gpd.z - gfrac.z;
            gfrac += diff * (slope / slope.z);
            end_cc = gfrac;
            err.z += derr.z;
            gfrac.z -= sgns.z;
            col = gfrac;
            n_face = 5 - gpd.z;

        } else {
            f_color = vec4(1.0, 1.0, 0.0, 1.0);
            return;
        }

        uint slot = 0xffffffff;
        for (int s = 0; s <= mx; s++) {
            if (slot_flag == 0) {
                break;
            }
            if (uvec3(chunks[slot_base + 1], chunks[slot_base + 2], chunks[slot_base + 3]) == chk) {
                slot = try_slot;
                break;
            }
            try_slot = (try_slot + 1) & n_minus_one;
            slot_base = try_slot * 4 + 32;
            slot_flag = chunks[slot_base];
        }

        if (slot != 0xffffffff && traverse_chunk(slot, start_cc * 16.0, end_cc * 16.0, k, n_minus_one, mx, last_face)) {
            return;
        }
        last_face = n_face;

        if (old_g == g1idx) {
            break;
        }
    }
    if (i >= 59) {
        f_color = vec4(1.0, 0.0, 0.0, 1.0);
    } else {
        f_color = vec4(0.0, 0.0, 0.0, 1.0);
    }
}