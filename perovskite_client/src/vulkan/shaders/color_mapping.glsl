// Combine multiple observed colors to an apparent color
// Args:
//  diffuse: The diffuse color, unchanged from the sampler
//  emissive: The emissive color, unchanged from the sampler
//  normal: The surface normal in camera space (before projection)
//  pos: The position in camera space (before projection)
//  brightness: The local brightness (from CPU-based lighting, after lookup tablee)
//  global_brightness: The effect of sun (from CPU-based lighting, after lookup table, and time of day). RGB
//
// Returns:
//  A color, that may lie outside of [0..1] in cases of emissive lighting. Note that in any case, it is directly
//  suitable to output to a R8G8B8A8_UNORM or R16G16B16A16_SFLOAT attachment or storage image.
vec4 combine_colors(vec4 diffuse, vec4 emissive, vec3 normal, vec3 pos, float brightness, vec3 global_brightness) {
    float normal_pos_dot = abs(dot(normalize(normal), normalize(pos)));

    emissive = clamp(emissive, vec4(0), vec4(1));
    emissive.rgb = emissive.rgb / (1.001 - emissive.rgb);

    vec4 f_color = vec4((brightness + global_brightness) * diffuse.rgb, diffuse.a);
    float emissive_anisotropy_power = mix(10.0, 0.001, emissive.a);
    float anisotropic_multiplier = pow(normal_pos_dot, emissive_anisotropy_power);
    f_color.rgb += anisotropic_multiplier * emissive.rgb;
    return f_color;
}
