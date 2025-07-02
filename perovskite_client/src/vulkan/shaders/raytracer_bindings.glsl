layout(set = 1, binding = 0) uniform RaytracingPerFrameData {
// Takes an NDC position and transforms it *back* to world space
    mat4 inverse_vp_matrix;
    mat4 forward_vp_matrix;
// Player position
    // in unit of chunks
    ivec3 coarse_pos;
    // in unit of chunks
    vec3 fine_pos;
    vec3 sun_direction;
// Used for dither, only during development
    float supersampling;
// length of RaytraceControl
    uint max_cube_info_idx;
    vec3 global_brightness_color;
    uint render_distance;
    uint initial_block_id;
};
