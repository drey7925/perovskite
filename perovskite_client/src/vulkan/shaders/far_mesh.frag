#version 460

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
  float gbc_adjustment =
      0.5 + 0.5 * max(-0.3, dot(global_light_dir, world_normal));
  vec3 effective_color =
      (0.03125 + (global_brightness * gbc_adjustment)) * color;
  f_color = vec4(effective_color, 1.0);
}