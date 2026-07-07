#version 460
#extension GL_ARB_shading_language_include : enable
#include "color_mapping.glsl"

// Position relative to camera in world space
layout(location = 0) in vec3 world_pos;
// The actual computed normal
layout(location = 1) flat in vec3 normal;
// The diffuse color, decoded
layout(location = 2) flat in vec4 diffuse;
// The emissive color, decoded
layout(location = 3) flat in vec4 emissive;
layout(location = 4) in vec2 texcoord;

layout(set = 0, binding = 0) uniform sampler2D atlas;

layout(location = 0) out vec4 f_color;

void main() {
  float alpha = texture(atlas, texcoord).r;
  if (alpha < 0.01) {
    discard;
  }
  // TODO: Pass actual brightness through the CPU userspace and vertex shader
  f_color = combine_colors(diffuse, emissive, normal, world_pos, 0.5, vec3(0));
  f_color.a = alpha;
}