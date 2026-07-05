#version 460
#extension GL_ARB_shading_language_include : enable
#include "encoding.glsl"

/*
pub(crate) struct TextVertex {
    /// Position, 3d space, relative to the renderer's origin in world space
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
    /// Texture coordinate in tex space (given as texels)
    #[format(R16G16_UINT)]
    uv_texcoord: [u16; 2],
    /// Diffuse color as R5G6B5
    #[format(R16_UINT)]
    diffuse_color: u16,

    /// Emissive color as R5G6B5
    #[format(R16_UINT)]
    emissive_color: u16,
    /// Encoded normal, same as CubeGeometryVertex, using [x5y5z5_pack_vec]
    #[format(R16_UINT)]
    encoded_normal: u16,
    // Flags. Low four bits are a very coarse version of the emissive alpha.
    // 0 = tightly focused on normal, 15 = nearly isotropic.
    // High four bits are for future use.
    #[format(R8_UINT)]
    flags: u8,
    // repr(C) would pad for us, but explicit padding makes the remaining
    // space clearer to readers of the code. This pads from 23 to 24 bytes.
    #[format(R8_UINT)]
    padding: u8,
}
*/

layout(location = 0) in vec3 position;
layout(location = 1) in uvec2 uv_texcoord;
layout(location = 2) in uint diffuse_color;
layout(location = 3) in uint emissive_color;
layout(location = 4) in uint encoded_normal;
layout(location = 5) in uint flags;
// unused
layout(location = 6) in uint padding;

// Position relative to camera in world space
layout(location = 0) out vec3 world_pos_out;
// The actual computed normal
layout(location = 1) flat out vec3 normal_out;
// The diffuse color, decoded
layout(location = 2) flat out vec4 diffuse_color_out;
// The emissive color, decoded
layout(location = 3) flat out vec4 emissive_color_out;
layout(location = 4) out vec2 texcoord_out;

// 100 out of 128 bytes that Vulkan 1.3 guarantees us for push constants
layout(push_constant) uniform TextPushConstants {
  mat4 vp_matrix;   //[0-63]
  vec3 translation; //[64-75]
  //   vec3 global_brightness_color; //[76-87]
  //   vec3 global_light_direction;  //[88-99]
};

layout(set = 0, binding = 0) uniform sampler2D atlas;

void main() {
  float flags_low4 = (flags & 0xf) / 15.0;
  normal_out = decode_normal_x5y5z5_pack15(int(encoded_normal));

  diffuse_color_out = vec4(decode_color_rgb565(diffuse_color), 1.0);
  emissive_color_out = vec4(decode_color_rgb565(emissive_color), flags_low4);

  world_pos_out = position + translation;
  texcoord_out = vec2(uv_texcoord) / vec2(textureSize(atlas, 0));

  gl_Position = vp_matrix * vec4(world_pos_out, 1);
}