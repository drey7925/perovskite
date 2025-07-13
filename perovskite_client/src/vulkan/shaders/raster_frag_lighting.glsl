#version 460

#include "color_mapping.glsl"

layout(location = 0) in vec2 uv_texcoord;
layout(location = 1) flat in float brightness;
layout(location = 2) flat in vec3 global_brightness;
layout(location = 3) flat in vec3 world_normal;
layout(location = 4) in vec3 world_pos;

layout(location = 0) out vec4 f_color;
#ifdef ENABLE_SPECULAR
layout(location = 1) out vec4 spec_strength;
layout(location = 2) out uvec4 spec_dir;
#endif

layout(set = 0, binding = 0) uniform sampler2D diffuse_tex;
layout(set = 0, binding = 1) uniform sampler2D emissive_tex;

layout (constant_id = 0) const bool SPARSE = true;
layout (constant_id = 1) const bool DEBUG_INVERT_RASTER_TRANSPARENT = true;

void main() {
    vec2 pix = gl_FragCoord.xy / 8.0;
    vec4 diffuse = texture(diffuse_tex, uv_texcoord);
    vec4 emissive = texture(emissive_tex, uv_texcoord);

    f_color = combine_colors(diffuse, emissive, world_normal, world_pos, brightness, global_brightness);

    if (DEBUG_INVERT_RASTER_TRANSPARENT) {
        f_color.rgb = 1.0 - f_color.rgb;
    }

    if (SPARSE) {
        if (f_color.a < 0.5) {
            discard;
        } else {
            f_color.a = 1.0;
        }
    }
}