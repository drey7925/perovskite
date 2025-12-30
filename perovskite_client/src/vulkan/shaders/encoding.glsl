
vec3 decode_normal_x5y5z5_pack15(int index) {
    ivec3 components = ivec3(bitfieldExtract(index, 10, 5), bitfieldExtract(index, 5, 5), bitfieldExtract(index, 0, 5));
    return normalize(vec3(components));
}