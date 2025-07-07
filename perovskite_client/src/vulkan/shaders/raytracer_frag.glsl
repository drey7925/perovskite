#version 460
#define SKIP_MASK 4u
#include "raytracer_frag_common.glsl"

layout(input_attachment_index = 0, set = 1, binding = 3) uniform subpassInput f_depth_in;
layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 deferred_specular_strength;
layout(location = 2) out uvec4 deferred_specular_ray_dir;

void main() {
    deferred_specular_strength = vec4(0);
    deferred_specular_ray_dir = uvec4(0);

    if (RAYTRACE_DEBUG) {
        vec2 pix3 = gl_FragCoord.xy / (16.0 * supersampling);
        int ix3 = int(pix3.x);
        int iy3 = int(pix3.y);

        if (iy3 == 1) {
            if (ix3 <= 160 && ix3 >= 1) {
                int idx = ix3 - 1;
                int idx0 = idx >> 5;
                uint dbg_data;
                if (idx0 == 0) {
                    dbg_data = n_minus_one;
                }
                if (idx0 == 1) {
                    dbg_data = mxc;
                }
                if (idx0 == 2) {
                    dbg_data = k.x;
                }
                if (idx0 == 3) {
                    dbg_data = k.y;
                }
                if (idx0 == 4) {
                    dbg_data = k.z;
                }
                uint idx1 = uint(idx & 31);
                uint bit = dbg_data & (1u << idx1);
                if (bit != 0) {
                    f_color = vec4(1.0, 0.0, 0.0, 1.0);
                    return;
                } else {
                    f_color = vec4(0.0, 0.0, 0.0, 1.0);
                    return;
                }
            }
        }
        if (iy3 == 2) {
            if (ix3 <= 160 && ix3 >= 1) {
                int idx = ix3 - 1;
                int idx0 = idx >> 5;
                if (idx0 == 0) {
                    f_color = vec4(1.0, 0.0, 0.0, 1.0);
                    return;
                } else if (idx0 == 1) {
                    f_color = vec4(1.0, 1.0, 0.0, 1.0);
                    return;
                } else if (idx0 == 2) {
                    f_color = vec4(0.0, 1.0, 0.0, 1.0);
                    return;
                } else if (idx0 == 3) {
                    f_color = vec4(0.0, 0.0, 1.0, 1.0);
                    return;
                } else if (idx0 == 4) {
                    f_color = vec4(1.0, 1.0, 0.0, 1.0);
                    return;
                }
            }
        }

        if (iy3 == 3) {
            if (ix3 <= 160 && ix3 >= 1) {
                int idx = ix3 - 1;
                if ((idx % 2) == 0) {
                    f_color = vec4(0.2, 1.0, 0.4, 1.0);
                    return;
                }
                else {
                    f_color = vec4(1.0, 1.0, 0.0, 1.0);
                    return;
                }
            }
        }
        vec2 pix_fine = gl_FragCoord.xy / (2 * supersampling);
        int ixp = int(pix_fine.x);
        int iyp = int(pix_fine.y) - 64;
        uint rows = (n_minus_one + 1) / 1024;
        if (iyp >= 0 && iyp < rows) {
            if (ixp < 1024) {
                uint slot = iyp * 1024 + ixp;
                uint slot_base = slot * 4;

                if ((chunks[slot_base] & 1) == 0) {
                    f_color = vec4(0, 0, 0, 1);
                    return;
                }
                uvec3 putative = uvec3(chunks[slot_base + 1], chunks[slot_base + 2], chunks[slot_base + 3]);
                uvec3 products = putative * k;
                uint sum = products.x + products.y + products.z;
                uint try_slot = (sum % 1610612741) & n_minus_one;
                if (try_slot == slot) {
                    f_color = vec4(0, 1, 0, 1);
                    return;
                } else {
                    f_color = vec4(1, 1, 0, 1);
                    return;
                }
            }
        }
    }// if (RAYTRACE_DEBUG)
    vec3 facedir_world = normalize(facedir_world_in);
    float prev_depth = subpassLoad(f_depth_in).r;

    // All raster geometry, as well as non-rendering calcs, assume that blocks have
    // their *centers* at integer coordinates.
    // However, we have the *edges* as axis-aligned.
    // Note that this shader works in world space, with Y up throughout.
    // The Y axis orientation has been flipped in the calculation of facedir_world.
    // This is fixed on the CPU to save some registers.
    vec2 t_min_max = t_range(fine_pos, facedir_world);

    if (t_min_max.x > t_min_max.y) {
        return;
    }

    vec3 g0 = fine_pos + (t_min_max.x * facedir_world);
    vec3 g1 = fine_pos + (t_min_max.y * facedir_world);

    f_color = vec4(0);
    float strongest_specular = 0;
    uint prev_block = initial_block_id;
    for (int i = 0; i < 5; i++) {
        HitInfo info = {
        ivec3(0),
        vec3(0),
        vec3(0),
        prev_block,
        0,
        };

        if (!traverse_space(g0, g1, info)) {
            return;
        }
        // traverse_space will put valid data into info iff it returns true
        prev_block = info.block_id;
        uint info_flags = cube_info[prev_block >> 12].flags;
        vec3 hit_pos = (vec3(info.hit_block) + info.start_cc) / 16.0;
        vec4 rpos = vec4((hit_pos - fine_pos) * 16.0, 1.0) * vec4(1, -1, 1, 1);
        vec4 transformed = forward_vp_matrix * rpos;
        if (clamp(transformed.z / transformed.w, 0, 1) > prev_depth) {
            return;
        }

        // skip fallback blocks
        if (info_flags != 0) {
            if ((info_flags & 4u) != 4) {
                SampleResult result = sample_simple(info, prev_block >> 12, SPECULAR);
                float alpha_contrib = (1 - f_color.a) * result.diffuse.a;
                f_color.rgb += alpha_contrib * result.diffuse.rgb;
                f_color.a += alpha_contrib;


                // TODO proper specular from the block, for now hardcode glass/water
                if (SPECULAR) {
                    if (length(result.specular.rgb) > 0.01) {
                        vec3 new_dir;
                        vec3 normal = decode_normal(info.face_light & 7u);
                        if (FUZZY_SHADOWS) {
                            vec3 tangent = normalize(cross(facedir_world, normal));
                            vec3 bitangent = cross(normal, tangent);
                            mat3 ntb = mat3(normal, tangent, bitangent);
                            vec3 mul = vec3(
                            0, 0.03 * (random(hit_pos.xz, 43758.5453123) - 0.5), 0.01 * random(hit_pos.xz, 43758.5453123)
                            );

                            new_dir = facedir_world * face_reflectors[info.face_light & 7u] + (ntb * mul);
                        }
                        else {
                            new_dir = facedir_world * face_reflectors[info.face_light & 7u];
                        }
                        float cos_theta = dot(normal, new_dir);
                        float fresnel = min((0.02 + 0.98 * pow(1 - cos_theta, 5)), 1.0);
                        vec3 final_mix = mix(fresnel, 1.0, result.diffuse.a) * result.specular.rgb;
                        if (length(final_mix) >= strongest_specular) {
                            deferred_specular_strength = vec4(final_mix, 1.0);
                            new_dir = normalize(new_dir) * length(hit_pos - fine_pos);
                            deferred_specular_ray_dir = uvec4(floatBitsToUint(new_dir), info.block_id);
                            strongest_specular = length(final_mix);
                        }
                    }
                }

                if (f_color.a > 0.99) {
                    break;
                }
            }
        }
        info.start_cc = mix(info.start_cc, info.end_cc, 0.5);
        g0 = (vec3(info.hit_block) + info.start_cc) / 16.0;
    }

}
