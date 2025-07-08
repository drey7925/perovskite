#version 460
layout (set = 0, binding = 0, rgba8) uniform restrict readonly image2D deferred_specular_color;
layout (set = 0, binding = 1, rgba32ui) uniform restrict readonly uimage2D deferred_specular_ray_dir;
layout (set = 0, binding = 2, rgba16f) uniform restrict readonly image2D specular_result;

layout(location = 0) out vec4 f_color;

#include "raytracer_spec_constants.glsl"

void main() {
    ivec2 max_coord = imageSize(specular_result).xy - ivec2(1, 1);
    ivec2 frag_coord = ivec2(gl_FragCoord.xy);
    f_color = vec4(0, 0, 0, 0);
    vec4 ds_color = imageLoad(deferred_specular_color, frag_coord);
    vec4 sum = vec4(0);
    float total_weight = 0.0001;

    ivec2 base = (ivec2(gl_FragCoord.xy) / int(SPECULAR_DOWNSAMPLING));
    uvec4 dsrd_ideal = imageLoad(deferred_specular_ray_dir, frag_coord);
    vec3 dsrd_ideal_f = uintBitsToFloat(dsrd_ideal.rgb);
    for (int i = 0; i <= 1; i++) {
        for (int j = 0; j <= 1; j++) {
            ivec2 ds_coord = clamp(base + ivec2(i, j), ivec2(0, 0), ivec2(max_coord));
            vec4 spec_color = imageLoad(specular_result, ds_coord);
            ivec2 ideal_frag_coord = ds_coord * int(SPECULAR_DOWNSAMPLING) + (int(SPECULAR_DOWNSAMPLING) / 2);
            uvec4 dsrd = imageLoad(deferred_specular_ray_dir, ideal_frag_coord);
            vec3 dsrd_f = uintBitsToFloat(dsrd.rgb);

            uint distance = abs(frag_coord.x - ideal_frag_coord.x) + abs(frag_coord.y - ideal_frag_coord.y);

            float weight = 1.0 / (length(dsrd_f - dsrd_ideal_f) + (distance / 10) + 0.0001);
            if (dsrd.a == 0) {
                weight = 0;
            }
            sum += weight * spec_color;
            total_weight += weight;
        }
    }

    f_color += (sum / total_weight) * imageLoad(deferred_specular_color, frag_coord);
}