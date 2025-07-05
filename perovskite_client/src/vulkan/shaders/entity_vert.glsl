#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv_texcoord;
// TODO: bring back entity lighting at some later point

layout(set = 1, binding = 0) uniform EntityUniformData {
    mat4 vp_matrix;
};
// 64 bytes of push constants :(
layout(push_constant) uniform ModelMatrix {
    mat4 model_matrix;
};

layout(location = 0) out vec2 uv_texcoord_out;
layout(location = 1) flat out float brightness_out;
layout(location = 2) flat out vec3 global_brightness_out;
layout(location = 3) flat out vec3 world_normal_out;
layout(location = 4) out vec3 world_pos_out;

void main() {
    vec4 world_pos = model_matrix * vec4(position, 1.0);
    world_pos_out = world_pos.xyz / world_pos.w;
    gl_Position = vp_matrix * world_pos;
    uv_texcoord_out = uv_texcoord;
    brightness_out = 1.0;
    global_brightness_out = vec3(0.0, 0.0, 0.0);
}