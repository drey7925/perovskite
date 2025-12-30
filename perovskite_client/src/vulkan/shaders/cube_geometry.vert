#version 460
#extension GL_ARB_shading_language_include : enable
#include "encoding.glsl"

layout(location = 0) in vec3 position;
layout(location = 1) in uvec2 uv_texcoord;
layout(location = 2) in int normal;
layout(location = 3) in uint tangent;
layout(location = 4) in uint brightness;
layout(location = 5) in float wave_horizontal;

layout(set = 1, binding = 0) uniform UniformData {
  mat4 vp_matrix;
  vec2 plant_wave_vector;
  vec3 global_brightness_color;
  vec3 global_light_direction;
};
// 64 bytes of push constants :(
layout(push_constant) uniform ModelMatrix { mat4 model_matrix; };

layout(set = 0, binding = 0) uniform sampler2D diffuse_tex;

layout(location = 0) out vec2 uv_texcoord_out;
layout(location = 1) flat out float brightness_out;
layout(location = 2) flat out vec3 global_brightness_out;
layout(location = 3) flat out vec3 world_normal_out;
layout(location = 4) out vec3 world_pos_out;
layout(location = 5) flat out vec3 world_tangent_out;

const float global_brightness_table[] = {
    0.0,        0.015625,   0.044194173, 0.08118988, 0.125,      0.17469281,
    0.22963966, 0.28937906, 0.35355338,  0.421875,   0.49410588, 0.5700449,
    0.649519,   0.7323776,  0.8184875,   0.90773046};

const float brightness_table[] = {
    0.03125,    0.07432544, 0.123381935, 0.17677669, 0.23364824, 0.29345337,
    0.35581362, 0.4204482,  0.48713928,  0.55571234, 0.6260238,  0.69795364,
    0.77139926, 0.8462722,  0.9224952,   1.0};

const vec3 tangent_encodings[6] =
    vec3[](vec3(-1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0),
           vec3(0.0, 0.0, -1.0), vec3(0.0, -1.0, 0.0), vec3(0.0, 1.0, 0.0));

void main() {
  float wave_x = wave_horizontal * plant_wave_vector.x;
  float wave_z = wave_horizontal * plant_wave_vector.y;
  vec3 position_with_wave =
      vec3(position.x + wave_x, position.y, position.z + wave_z);
  vec4 world_pos = model_matrix * vec4(position_with_wave, 1.0);
  world_pos_out = world_pos.xyz / world_pos.w;
  gl_Position = vp_matrix * world_pos;

  uv_texcoord_out = vec2(uv_texcoord) / vec2(textureSize(diffuse_tex, 0));
  brightness_out = brightness_table[bitfieldExtract(brightness, 0, 4)];
  float global_brightness_contribution =
      global_brightness_table[bitfieldExtract(brightness, 4, 4)];
  // Guaranteed to be normalized
  world_normal_out = decode_normal_x5y5z5_pack15(normal);

  world_tangent_out = tangent_encodings[tangent];

  float gbc_adjustment =
      0.5 + 0.5 * max(-0.3, dot(global_light_direction, world_normal_out));
  global_brightness_out =
      global_brightness_color * global_brightness_contribution * gbc_adjustment;
}