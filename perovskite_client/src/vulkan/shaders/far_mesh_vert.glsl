//  vec3 color = fract(sin(gl_VertexIndex * vec2(12.9898, 78.233)) * 43758.5453);
#version 460
layout(location = 0) in vec3 position;
layout(location = 0) flat out vec3 color;

layout(set = 1, binding = 0) uniform FarMeshUniforms {
    mat4 vp_matrix;
};
// 64 bytes of push constants :(
layout(push_constant) uniform FarMeshPushConstants {
    mat4 model_matrix;
};

void main() {
    color = fract(sin(gl_VertexIndex * vec3(12.9898, 78.233, 24.1055)) * 43758.5453);
    gl_Position = vp_matrix * model_matrix * vec4(position, 1.0);
}
