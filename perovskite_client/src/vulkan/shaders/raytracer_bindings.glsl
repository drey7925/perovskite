struct TexRef {
    vec2 diffuse_top_left;
    vec2 diffuse_width_height;
};

const uint FLAGS_FOO = 0;

struct SimpleCubeInfo {
    uint flags;
    TexRef tex[6];
};

layout(set = 0, binding = 0) uniform sampler2D tex;
layout(set = 0, binding = 1) readonly buffer RaytraceControl {
    SimpleCubeInfo cube_info[];
};

layout(set = 1, binding = 0) uniform RaytracingPerFrameData {
// Takes an NDC position and transforms it *back* to world space
    mat4 inverse_vp_matrix;
    mat4 forward_vp_matrix;
// Player position
    // in unit of chunks
    ivec3 coarse_pos;
    // in unit of chunks
    vec3 fine_pos;
// Will be re-enabled once we have sky
    vec3 sun_direction;
// Used for dither, only during development
    float supersampling;
// length of RaytraceControl
    uint max_cube_info_idx;
    vec3 global_brightness_color;
    uint render_distance;
    uint initial_block_id;
};
layout(set = 1, binding = 1) uniform ChunkMapHeader {
    uint n_minus_one;
    uint mxc;
    uvec3 k;
    ivec3 min_chunk;
    ivec3 max_chunk;
};
layout(set = 1, binding = 2) readonly buffer chunk_map {
    uint chunks[];
};