#version 460

#include "color_mapping.glsl"

layout(set = 1, binding = 0) uniform FarMeshUniforms {
    mat4 vp_matrix;
    vec3 global_brightness;
    vec3 global_light_dir;
};

layout(location = 0) out vec4 f_color;

layout(location = 0) flat in vec3 color;
layout(location = 1) flat in vec3 world_normal;
layout(location = 2) in vec3 world_pos;

void main() {
    // TODO: Once we have normals, use the color mapping code
    f_color = combine_colors(vec4(color, 1.0), vec4(0.0), world_normal, world_pos, 0.03125, global_brightness);
}