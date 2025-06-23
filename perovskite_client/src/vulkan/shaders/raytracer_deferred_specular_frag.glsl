#version 460
#include "raytracer_frag_common.glsl"

layout (set = 1, binding = 4, rgba8) uniform restrict readonly image2D deferred_specular_color;
layout (set = 1, binding = 5, rgba32ui) uniform restrict readonly uimage2D deferred_specular_ray_dir;


void main() {
    vec3 spec_color = imageLoad(deferred_specular_color, ivec2(gl_FragCoord.xy)).rgb;
    uvec4 spec_ray_dir = imageLoad(deferred_specular_ray_dir, ivec2(gl_FragCoord.xy));
    vec3 facedir_world = normalize(facedir_world_in);

    float prev_depth = subpassLoad(f_depth_in).r;
    vec4 rpos = inverse_vp_matrix * vec4(gl_FragCoord.xy, prev_depth, gl_FragCoord.w);
    rpos /= rpos.w;
    f_color = vec4(0.0, 0.0, 0.0, 1.0);
    vec3 spec_dir = normalize(uintBitsToFloat(spec_ray_dir.rgb));
    uint spec_block = spec_ray_dir.a;
    vec3 spec_start_vec = (rpos.xyz * vec3(1, -1, 1) + fine_pos) / 16.0;
    vec2 t_min_max = t_range(spec_start_vec, spec_dir);

    if (t_min_max.x > t_min_max.y) {
        return;
    }
    // we just had a specular, so t0 is 0 and start is just spec_start;
    vec3 g1 = spec_start_vec + (spec_dir * t_min_max.y);
    vec4 spec_rgba = vec4(0);
    for (int i = 0; i < 3; i++) {
        HitInfo info = {
        ivec3(0),
        vec3(0),
        vec3(0),
        spec_block,
        0,
        };
        if (traverse_space(spec_start_vec, g1, info)) {
            // traverse_space will fill info w/ valid data iff it returns true
            uint idx = info.block_id >> 12;
            if (idx >= max_cube_info_idx) {
                // CPU-side code should fix this and we should never enter this branch
                f_color = vec4(1.0, 0.0, 0.0, 1.0);
                return;
            }
            if ((cube_info[idx].flags & 1) != 0) {
                vec4 sampled = sample_simple(info, idx, false).diffuse;
                float alpha_contrib = sampled.a * (1 - spec_rgba.a);
                spec_rgba.rgb += alpha_contrib * sampled.rgb;
                spec_rgba.a += alpha_contrib;
            }

            vec3 midpoint = (info.start_cc + info.end_cc) / 2;
            spec_start_vec = (vec3(info.hit_block) + midpoint) / 16.0 - fine_pos;
        }
    }

    if (spec_rgba.a < 0.99) {
        spec_rgba += (1 - spec_rgba.a) * vec4(sky_rgb(spec_dir, sun_direction), 1.0);
    }
    f_color.rgb += spec_color * spec_rgba.rgb * spec_rgba.a;

}
