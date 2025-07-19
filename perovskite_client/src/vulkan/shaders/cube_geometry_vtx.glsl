#version 460
layout(location = 0) in vec3 position;
layout(location = 2) in uvec2 uv_texcoord;
layout(location = 1) in uint normal;
layout(location = 3) in float brightness;
layout(location = 4) in float global_brightness_contribution;
layout(location = 5) in float wave_horizontal;

layout(set = 1, binding = 0) uniform UniformData {
    mat4 vp_matrix;
    vec2 plant_wave_vector;
    vec3 global_brightness_color;
    vec3 global_light_direction;
};
// 64 bytes of push constants :(
layout(push_constant) uniform ModelMatrix {
    mat4 model_matrix;
};

layout(set = 0, binding = 0) uniform sampler2D diffuse_tex;

layout(location = 0) out vec2 uv_texcoord_out;
layout(location = 1) flat out float brightness_out;
layout(location = 2) flat out vec3 global_brightness_out;
layout(location = 3) flat out vec3 world_normal_out;
layout(location = 4) out vec3 world_pos_out;


vec3 decode_normal(uint index) {
    const float sqrt_half = sqrt(0.5);
    // Matches CubeFace in BlockRenderer
    const vec3 normals[10] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(-1.0, 0.0, 0.0),
    // Warning: CubeFace Y+ then Y- is in world coords, not Vk coords
    vec3(0.0, -1.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0),
    vec3(0.0, 0.0, -1.0),
    vec3(sqrt_half, 0.0, sqrt_half),
    vec3(sqrt_half, 0.0, -sqrt_half),
    vec3(-sqrt_half, 0.0, sqrt_half),
    vec3(-sqrt_half, 0.0, -sqrt_half)
    );
    return normals[index];
}
void main() {
    float wave_x = wave_horizontal * plant_wave_vector.x;
    float wave_z = wave_horizontal * plant_wave_vector.y;
    vec3 position_with_wave = vec3(position.x + wave_x, position.y, position.z + wave_z);
    vec4 world_pos = model_matrix * vec4(position_with_wave, 1.0);
    world_pos_out = world_pos.xyz / world_pos.w;
    gl_Position = vp_matrix * world_pos;

    uv_texcoord_out = vec2(uv_texcoord) / vec2(textureSize(diffuse_tex, 0));
    brightness_out = brightness;
    // Guaranteed to be normalized
    world_normal_out = decode_normal(normal);
    float gbc_adjustment = 0.5 + 0.5 * max(-0.3, dot(global_light_direction, world_normal_out));
    global_brightness_out = global_brightness_color * global_brightness_contribution * gbc_adjustment;
}