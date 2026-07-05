
vec3 decode_normal_x5y5z5_pack15(int index) {
  ivec3 components =
      ivec3(bitfieldExtract(index, 10, 5), bitfieldExtract(index, 5, 5),
            bitfieldExtract(index, 0, 5));
  return normalize(vec3(components));
}

vec3 decode_color_rgb565(uint rgb565) {
  uint r = (rgb565 >> 11) & 0x1Fu;
  uint g = (rgb565 >> 5) & 0x3Fu;
  uint b = rgb565 & 0x1Fu;
  return vec3(r, g, b) / vec3(31.0, 63.0, 31.0);
}
