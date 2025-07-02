#version 460
layout (set = 0, binding = 0, rgba8) uniform restrict readonly image2D main_color;
layout (set = 0, binding = 1, rgba8) uniform restrict readonly image2D deferred_specular_color;
layout (set = 0, binding = 2, rgba32ui) uniform restrict readonly uimage2D deferred_specular_ray_dir;
layout (set = 0, binding = 3, rgba8) uniform restrict readonly image2D specular_result;
layout (set = 0, binding = 4, rg32f) uniform restrict readonly image2D shadow_distance;
layout (set = 0, binding = 5, r8) uniform restrict readonly image2D shadow_out;

layout(location = 0) out vec4 f_color;

#include "raytracer_spec_constants.glsl"

void main() {
    ivec2 max_coord = imageSize(specular_result).xy - ivec2(1, 1);
    ivec2 ds_coord = clamp(ivec2(gl_FragCoord.xy) / int(SPECULAR_DOWNSAMPLING), ivec2(0, 0), ivec2(max_coord));

    f_color = imageLoad(main_color, ivec2(gl_FragCoord.xy));

    float shadow_contrib = imageLoad(shadow_distance, ivec2(gl_FragCoord.xy)).y;
    float sun_transmission = imageLoad(shadow_out, ds_coord).x;
    f_color *= mix(1.0, sun_transmission, shadow_contrib * 0.9);

    // avoid optimizing out deferred_specular_ray_dir for now
    if (imageLoad(deferred_specular_ray_dir, ds_coord).a == 0xffffffff) {
        f_color.x += 0.0001;
    }

    f_color += imageLoad(specular_result, ds_coord) * imageLoad(deferred_specular_color, ivec2(gl_FragCoord.xy));
}