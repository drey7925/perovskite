#version 460
#define SKIP_MASK 0u
#include "raytracer_frag_common.glsl"

layout (set = 1, binding = 3, rg32f) uniform restrict image2D shadow_in;
layout (set = 1, binding = 4, r8) uniform restrict image2D shadow_out;

void main() {
    int half_offset = int(SPECULAR_DOWNSAMPLING) / 2;
    vec2 dist_contrib = imageLoad(shadow_in, int(SPECULAR_DOWNSAMPLING) * ivec2(gl_FragCoord.xy) + half_offset).xy;
    vec3 spec_dir = normalize(sun_direction);
    vec4 ndc_depth = vec4(pos_ndc_xy, 0.5, 1.0);
    vec3 facedir_world_in = (inverse_vp_matrix * ndc_depth).xyz  * vec3(1.0, -1.0, 1.0);
    vec3 g0 = normalize(facedir_world_in) * (dist_contrib.x - 0.001) + fine_pos;

    vec2 t_min_max = t_range(g0, spec_dir);

    float transmission = 1.0;

    if (t_min_max.x > t_min_max.y) {
        imageStore(shadow_out, ivec2(gl_FragCoord.xy), vec4(1.0, 0, 0, 0));
        return;
    }
    vec3 g1 = g0 + (spec_dir * t_min_max.y);
    vec4 spec_rgba = vec4(0);
    for (int i = 0; i < 3; i++) {
        HitInfo info = {
        ivec3(0),
        vec3(0),
        vec3(0),
        0,
        0,
        };
        if (!traverse_space(g0, g1, info)) {
            break;
        }
        // traverse_space will fill info w/ valid data iff it returns true
        uint idx = info.block_id >> 12;
        if (idx >= max_cube_info_idx) {
            // CPU-side code should fix this and we should never enter this branch
            break;
        }
        if ((cube_info[idx].flags & 1) != 0) {
            vec4 diffuse = sample_simple(info, idx, false).diffuse;
            transmission *= (1.0 - diffuse.a);
        }
        vec3 midpoint = (info.start_cc + info.end_cc) / 2;
        g0 = (vec3(info.hit_block) + midpoint) / 16.0;
        if (transmission < 0.1) {
            break;
        }
    }

    imageStore(shadow_out, ivec2(gl_FragCoord.xy), vec4(transmission, 0, 0, 0));
}
