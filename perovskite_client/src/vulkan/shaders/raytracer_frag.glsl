#version 460
#extension GL_KHR_shader_subgroup_vote: enable

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
    uint base = 4 * n + 32 + 4096 * res;
    //f_color = vec4(g1 / 16, 1.0);
    //return true;
    vec3 hit_color = normalize(g1 - g0) / 2 + 0.5;

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
    vec3 err = (gp - g0) * v2;
    vec3 derr = sgns * v2;
    int i = 0;


    for (; i < 60; i++) {


        vec3 start_cc = gfrac;
        vec3 end_cc;
        vec3 old_g = g;


        //        if ((g.y == 3) && (g.x == 4)) {
        //            f_color = vec4(0.0, 1.0, 1.0, 1.0);
        //            return true;
        //        }

        vec3 r = abs(err);


        if (sgns.x != 0 && (sgns.y == 0 || r.x < r.y) && (sgns.z == 0 || r.x < r.z)) {
            g.x += sgns.x;
            float diff = gpd.x - gfrac.x;
            gfrac += diff * (slope / slope.x);
            end_cc = gfrac;
            err.x += derr.x;
            gfrac.x -= sgns.x;
        }
        else if (sgns.y != 0 && (sgns.z == 0 || r.y < r.z)) {
            g.y += sgns.y;
            float diff = gpd.y - gfrac.y;
            gfrac += diff * (slope / slope.y);
            end_cc = gfrac;
            err.y += derr.y;
            gfrac.y -= sgns.y;
        }
        else if (sgns.z != 0) {
            g.z += sgns.z;
            float diff = gpd.z - gfrac.z;
            gfrac += diff * (slope / slope.z);
            end_cc = gfrac;
            err.z += derr.z;
            gfrac.z -= sgns.z;

        }

        //        if(gfrac.x < 0.05 || gfrac.x > 0.95 || gfrac.y < 0.05 || gfrac.y > 0.95 || gfrac.z < 0.05 || gfrac.z > 0.95) {
        //            hit_color = vec3(0, 0, 0);
        //        }

        //        if (offset < 0 || offset >= 4096) {
        //            return true;
        //        }
        if (old_g.x < 0 || old_g.y < 0 || old_g.z < 0) { continue; }
        if (old_g.x >= 16 || old_g.y >= 16 || old_g.z >= 16) { continue; }

        uint offset = 256 * uint(old_g.x) + 16 * uint(old_g.z) + uint(old_g.y);

        if (offset >= 0 && offset < 4096) {
            bool is_nonzero = chunks[base + offset] != 0;
            if (subgroupAny(is_nonzero) && !subgroupAll(is_nonzero)) {
                f_color = vec4(1.0, 0.0, 0.0, 1.0);
                return true;
            }
            if (is_nonzero) {
                f_color = vec4(start_cc, 1.0);
            }
            return true;
        }

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

//    if ((ix2 ^ iy2) == 0) {
//        // Complementary to the control on the raster code
//        discard;
//    }
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
    vec3 col = vec3(1.0, 1.0, 1.0);
    for (; i < 60; i++) {
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