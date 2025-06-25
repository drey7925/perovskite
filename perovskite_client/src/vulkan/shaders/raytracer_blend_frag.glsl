#version 460
layout (set = 0, binding = 0, rgba8) uniform restrict readonly image2D deferred_specular_color;
layout (set = 0, binding = 1, rgba32ui) uniform restrict readonly uimage2D deferred_specular_ray_dir;
layout (set = 0, binding = 2, rgba8) uniform restrict readonly image2D specular_result;

layout(location = 0) out vec4 f_color;

#include "raytracer_spec_constants.glsl"

void main() {
    ivec2 max_coord = imageSize(specular_result).xy - ivec2(1, 1);
    ivec2 coord = clamp(ivec2(gl_FragCoord.xy / SPECULAR_DOWNSAMPLING), ivec2(0, 0), ivec2(max_coord));

    f_color = vec4(0, 0, 0, 0);
    // avoid optimizing out deferred_specular_ray_dir for now
    if (imageLoad(deferred_specular_ray_dir, coord).a == 0xffffffff) {
        f_color.x += 0.0001;
    }

    f_color += imageLoad(specular_result, coord) * imageLoad(deferred_specular_color, ivec2(gl_FragCoord.xy));
}