#version 460
layout (set = 0, binding = 0, rgba8) uniform restrict readonly image2D deferred_specular_color;
layout (set = 0, binding = 1, rgba32ui) uniform restrict readonly uimage2D deferred_specular_ray_dir;
layout (set = 0, binding = 2, rgba32ui) uniform restrict writeonly uimage2D deferred_specular_ray_dir_out;

#include "raytracer_spec_constants.glsl"

void main() {
    ivec2 max_coord = imageSize(deferred_specular_color).xy - ivec2(1, 1);

    uint best_distance = SPECULAR_DOWNSAMPLING + 10;
    uvec4 best_dsrd = uvec4(0);
    bool keep = false;
    ivec2 base = (ivec2(gl_FragCoord.xy) * int(SPECULAR_DOWNSAMPLING));
    for (int i = 0; i < SPECULAR_DOWNSAMPLING; i++) {
        for (int j = 0; j < SPECULAR_DOWNSAMPLING; j++) {
            ivec2 coord = clamp(base + ivec2(i, j), ivec2(0, 0), ivec2(max_coord));
            vec4 spec_color = imageLoad(deferred_specular_color, coord);
            uvec4 dsrd = imageLoad(deferred_specular_ray_dir, coord);
            uint distance = abs(i - int(SPECULAR_DOWNSAMPLING) / 2) + abs(j - int(SPECULAR_DOWNSAMPLING / 2));
            if ((dsrd.a != 0) && (distance < best_distance)) {
                best_dsrd = dsrd;
                best_distance = distance;
            }
            if (spec_color.a > 0.5) {
                keep = true;
            }
        }
    }
    imageStore(deferred_specular_ray_dir_out, ivec2(gl_FragCoord.xy), best_dsrd);

    if (!keep) {
        discard;
    }
}