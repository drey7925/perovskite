#version 460
#extension GL_ARB_shading_language_include : enable
#include "encoding.glsl"

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 top_color;
layout(location = 2) in vec4 side_color;
layout(location = 3) in int normal;
layout(location = 4) in float lod_orientation_bias;

layout(location = 0) out vec4 top_color_out;
layout(location = 1) out vec4 side_color_out;
layout(location = 2) out vec3 world_normal;
layout(location = 3) out vec3 model_pos;
layout(location = 4) out vec3 camera_relative_pos;
layout(location = 5) out float lod_orientation_bias_out;

layout(set = 1, binding = 0) uniform FarMeshUniforms {
  mat4 vp_matrix;
  vec3 global_brightness;
  vec3 global_light_dir;
};
// 64 bytes of push constants :(
layout(push_constant) uniform FarMeshPushConstants { mat4 model_matrix; };

void main() {
  // gamma correction. This is done in the shader, rather than
  // using an SRGB vulkan data type, because this color is being
  // retrieved via the vertex buffer, not a texture sampler.
  // Vulkan does not guarantee VK_FORMAT_R8G8B8A8_SRGB
  // + VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT together.
  top_color_out = vec4(pow(top_color.rgb, vec3(2.2)), top_color.a);
  side_color_out = vec4(pow(side_color.rgb, vec3(2.2)), side_color.a);
  model_pos = position;

  vec4 world_pos4 = model_matrix * vec4(position, 1.0);

  gl_Position = vp_matrix * world_pos4;

  camera_relative_pos = world_pos4.xyz / world_pos4.w;

  world_normal = decode_normal_x5y5z5_pack15(normal);

  lod_orientation_bias_out = lod_orientation_bias;
}
