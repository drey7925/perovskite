#version 460
layout (set = 0, binding = 0, rgba8) uniform restrict readonly image2D deferred_specular_color;

#include "raytracer_spec_constants.glsl"

void main() {
    int down_right = int(SPECULAR_DOWNSAMPLING - 1) / 2;
    int up_left = int(SPECULAR_DOWNSAMPLING - 1) - down_right;

    ivec2 max_coord = imageSize(deferred_specular_color).xy - ivec2(1, 1);

    bool keep = false;
    ivec2 base = (ivec2(gl_FragCoord.xy) * int(SPECULAR_DOWNSAMPLING)) + ivec2(up_left, up_left);
    for (int i = -up_left; i <= down_right; i++) {
        for (int j = -up_left; j <= down_right; j++) {
            ivec2 coord = clamp(base + ivec2(i, j), ivec2(0, 0), ivec2(max_coord));
            vec4 spec_color = imageLoad(deferred_specular_color, coord);
            if (spec_color.a > 0.5) {
                keep = true;
            }
        }
    }

    if (!keep) {
        discard;
    }
}