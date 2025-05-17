#version 460
#include "sky.glsl"

layout(location = 0) in vec4 global_coords_position;
layout(set = 0, binding = 0) uniform SkyUniformData {
// Takes an NDC position and transforms it *back* to world space
    mat4 inverse_vp_matrix;
    vec3 sun_direction;
    float supersampling;
};
layout(location = 0) out vec4 f_color;


void main() {
    vec3 ngc = normalize(global_coords_position.xyz) * vec3(1, -1, 1);
    //f_color.xyz = (ngc / 2.0) + vec3(0.5, 0.5, 0.5);
    f_color = vec4(sky_rgb(ngc, sun_direction), 1.0);
}