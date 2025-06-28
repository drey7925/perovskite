#version 460
#define SKIP_MASK 0u
#include "raytracer_frag_common.glsl"

//layout (set = 1, binding = 3, rgba8) uniform restrict readonly image2D deferred_specular_color;
layout (set = 1, binding = 4, rgba32ui) uniform restrict readonly uimage2D deferred_specular_ray_dir;
layout (set = 1, binding = 5, rgba8) uniform restrict writeonly image2D specular_result;

void main() {
    uvec4 spec_ray_dir = imageLoad(deferred_specular_ray_dir, ivec2(gl_FragCoord.xy));

    vec4 f_color = vec4(0.0, 0.0, 0.0, 1.0);
    vec3 spec_dir = uintBitsToFloat(spec_ray_dir.rgb);
    uint spec_block = spec_ray_dir.a;
    vec3 g0 = normalize(facedir_world_in) * length(spec_dir) + fine_pos;
    spec_dir = normalize(spec_dir);

    vec2 t_min_max = t_range(g0, spec_dir);

    if (t_min_max.x > t_min_max.y) {
        f_color = vec4(0.25, 0.0, 0.0, 1.0);
        return;
    }
    vec3 g1 = g0 + (spec_dir * t_min_max.y);
    vec4 spec_rgba = vec4(0);
    for (int i = 0; i < 3; i++) {
        HitInfo info = {
        ivec3(0),
        vec3(0),
        vec3(0),
        spec_block,
        0,
        };
        if (!traverse_space(g0, g1, info)) {
            break;
        }
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
        spec_block = info.block_id;
        vec3 midpoint = (info.start_cc + info.end_cc) / 2;
        g0 = (vec3(info.hit_block) + midpoint) / 16.0;
        if (spec_rgba.a > 0.75) {
            break;
        }
    }

    if (spec_rgba.a < 0.99) {
        spec_rgba += (1 - spec_rgba.a) * vec4(sky_rgb(spec_dir, sun_direction), 1.0);
    }
    f_color.rgb = spec_rgba.rgb * spec_rgba.a;
    imageStore(specular_result, ivec2(gl_FragCoord.xy), f_color);
}
