//  vec3 color = fract(sin(gl_VertexIndex * vec2(12.9898, 78.233)) * 43758.5453);
#version 460
#include "encoding.glsl"

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;
layout(location = 2) in int normal;

layout(location = 0) flat out vec3 color_out;
layout(location = 1) flat out vec3 world_normal;
layout(location = 2) out vec3 world_pos;
layout(set = 1, binding = 0) uniform FarMeshUniforms {
    mat4 vp_matrix;
    vec3 global_brightness;
    vec3 global_light_dir;
};
// 64 bytes of push constants :(
layout(push_constant) uniform FarMeshPushConstants {
    mat4 model_matrix;
};

void main() {
    // gamma correction. This is done in the shader, rather than
    // using an SRGB vulkan data type, because this color is being
    // retrieved via the vertex buffer, not a texture sampler.
    // Vulkan does not guarantee VK_FORMAT_R8G8B8A8_SRGB
    // + VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT together.
    color_out = pow(color.rgb, vec3(2.2));

    vec4 world_pos4 = model_matrix * vec4(position, 1.0);
    world_pos = world_pos4.xyz / world_pos4.w;
    gl_Position =  vp_matrix * world_pos4;
    world_normal = decode_normal_x5y5z5_pack15(normal);
}
