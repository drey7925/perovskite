#version 460

#include "color_mapping.glsl"

layout(location = 0) in vec2 uv_texcoord;
layout(location = 1) flat in float brightness;
layout(location = 2) flat in vec3 global_brightness;
layout(location = 3) flat in vec3 world_normal;
layout(location = 4) in vec3 world_pos;

#ifdef ENABLE_BASIC_COLOR
// No specular, discard if SPARSE
layout(location = 0) out vec4 f_color;
layout(set = 0, binding = 0) uniform sampler2D diffuse_tex;
layout(set = 0, binding = 1) uniform sampler2D emissive_tex;
#endif

#ifdef ENABLE_UNIFIED_SPECULAR
// Unified specular, never discard
layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 spec_strength;
layout(location = 2) out uvec4 spec_dir;

layout(set = 0, binding = 0) uniform sampler2D diffuse_tex;
layout(set = 0, binding = 1) uniform sampler2D emissive_tex;
layout(set = 0, binding = 2) uniform sampler2D specular_tex;
#endif

#ifdef ENABLE_SPECULAR_ONLY
// Specular only, discard zero specular
layout(location = 0) out vec4 spec_strength;
layout(location = 1) out uvec4 spec_dir;
// Required to drive the Fresnel term
layout(set = 0, binding = 0) uniform sampler2D diffuse_tex;
layout(set = 0, binding = 1) uniform sampler2D specular_tex;
#endif

layout (constant_id = 0) const bool SPARSE = true;
layout (constant_id = 1) const bool DEBUG_INVERT_RASTER_TRANSPARENT = true;

float random (vec2 st, float f) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * f);
}

void main() {
    vec4 diffuse = texture(diffuse_tex, uv_texcoord);

    #if defined(ENABLE_BASIC_COLOR) || defined(ENABLE_UNIFIED_SPECULAR)
    vec4 emissive = texture(emissive_tex, uv_texcoord);

    f_color = combine_colors(diffuse, emissive, world_normal, world_pos, brightness, global_brightness);

    if (DEBUG_INVERT_RASTER_TRANSPARENT) {
        f_color.rgb = 1.0 - f_color.rgb;
    }

    #ifdef ENABLE_BASIC_COLOR
    if (SPARSE) {
        if (f_color.a < 0.5) {
            discard;
        } else {
            f_color.a = 1.0;
        }
    }
    #endif// ENABLE_BASIC_COLOR
    #endif// defined(ENABLE_BASIC_COLOR) || defined(ENABLE_UNIFIED_SPECULAR)

    #if defined(ENABLE_SPECULAR_ONLY) || defined(ENABLE_UNIFIED_SPECULAR)
    spec_dir = uvec4(0);
    vec4 specular = texture(specular_tex, uv_texcoord);
    if (specular.rgb != vec3(0)) {
        float rt_len = (length(world_pos) - 0.05) / 16.0;
        vec3 incident = normalize(world_pos);
        vec3 tangent = normalize(cross(world_pos, world_normal));
        vec3 bitangent = cross(world_normal, tangent);
        vec3 reflection = reflect(incident, world_normal);
        mat3 ntb = mat3(world_normal, tangent, bitangent);
        vec3 mul = vec3(
        0, 0.03 * (random(uv_texcoord, 43758.5453123) - 0.5), 0.01 * random(uv_texcoord, 43758.5453123)
        );
        //        reflection += (ntb * mul);
        float cos_theta = dot(world_normal, reflection);
        float fresnel = min((0.02 + 0.98 * pow(1 - cos_theta, 5)), 1.0);
        spec_strength = vec4(mix(fresnel, 1.0, diffuse.a) * specular.rgb, 1.0);
        reflection.y = -reflection.y;
        spec_dir = uvec4(floatBitsToUint(rt_len * reflection), 0xffffffff);
    } else {
        #ifdef ENABLE_SPECULAR_ONLY
        // Do not discard on unified (solid/translucent) materials
        discard;
        #endif// ENABLE_SPECULAR_ONLY
    }
    #endif// defined(ENABLE_SPECULAR_ONLY) || defined(ENABLE_UNIFIED_SPECULAR)
}