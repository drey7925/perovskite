#version 460
layout(constant_id = 0) const uint SUPERSAMPLING = 1;

layout(set = 0, binding = 0) uniform sampler2D blue_noise;
layout(set = 1, binding = 0) uniform FarMeshUniforms {
  mat4 vp_matrix;
  vec3 global_brightness;
  vec3 global_light_dir;
};

layout(location = 0) out vec4 f_color;

layout(location = 0) flat in vec4 top_color;
layout(location = 1) flat in vec4 side_color;
layout(location = 2) flat in vec3 world_normal;
layout(location = 3) in vec3 model_pos;
layout(location = 4) in vec3 camera_relative_pos;
layout(location = 5) flat in float lod_orientation_bias;

void main() {
  ivec2 texel_coords =
      ivec2(gl_FragCoord.xy / SUPERSAMPLING) % textureSize(blue_noise, 0);
  float blue_noise = texelFetch(blue_noise, texel_coords, 0).r;

  float model_pos_y_fract = fract(model_pos.y);
  vec3 effective_normal = normalize(world_normal);
  vec3 effective_view = normalize(camera_relative_pos);

  // do everything in y-up space to make the math easier, for now
  float view_y = clamp(effective_view.y, 0, 1);
  float normal_y = clamp(-effective_normal.y, 0, 1);
  // TODO: Is it valid to just map X and Z together, or do we need to do more
  // when the camera is looking at an angle not aligned with X or Z axis?
  float view_xz =
      length(effective_view.xz) + 0.0001; // epsilon to avoid div by zero
  float normal_xz = length(effective_normal.xz) + 0.0001;

  float side_part = view_xz * normal_xz;
  float top_part = view_y * normal_y;

  float side_ratio = (side_part / (side_part + top_part));

  if (side_ratio >= blue_noise) {
    effective_normal.y = 0;
  } else {
    effective_normal.x = 0;
    effective_normal.z = 0;
  }

  vec4 color;
  if (side_ratio - lod_orientation_bias >= blue_noise) {
    color = side_color;
  } else {
    color = top_color;
  }

  effective_normal = normalize(effective_normal);

  float gbc_adjustment =
      0.5 + 0.5 * max(-0.3, dot(global_light_dir, effective_normal));
  vec3 ec = (0.03125 + (global_brightness * gbc_adjustment)) * color.rgb;
  f_color = vec4(ec, color.a);
}