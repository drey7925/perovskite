#version 460

#include "raytracer_bindings.glsl"

const vec2 vertices[3] = vec2[](
vec2(-1.0, -1.0),
vec2(-1.0, 3.0),
vec2(3.0, -1.0)
);

layout(location = 0) out vec4 global_coord_facedir;

void main() {
    vec4 pos_ndc = vec4(vertices[gl_VertexIndex], 0.5, 1.0);
    gl_Position = pos_ndc;
    global_coord_facedir = inverse_vp_matrix * pos_ndc;
}