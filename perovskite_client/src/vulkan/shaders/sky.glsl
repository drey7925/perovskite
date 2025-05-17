const vec3 base_blue = vec3(0.25, 0.6, 1.0);
const vec3 base_orange = vec3(1.0, 0.5, 0.1);

// Takes Y-up coordinates
vec3 sky_rgb(vec3 dir, vec3 sun_direction) {
    vec3 rayleigh = vec3(5.8e-6, 13.5e-6, 33.1e-6);
    float alignment = dot(dir, sun_direction);


    // Extinction effect strongest during sunset and after
    float sun_height = max(sun_direction.y, 0.0);
    float extinction_strength = 4.0 / (4 * sun_height + 1.0);
    float extinction_unscaled = (alignment * 0.3) + 0.7;
    float extinction = pow(extinction_unscaled, extinction_strength);
    float leakage_correction = clamp(sun_direction.y * 5.0 + 1.0, 0.0, 1.0);

    float sunset_strength = 1.25 * max(0.8 - abs(sun_height), 0.0);
    float sunset_blend_factor = max(sunset_strength - max(dir.y, 0.0), 0.0);

    vec3 base_color = sunset_blend_factor * base_orange + (1.0 - sunset_blend_factor) * base_blue;
    float extra_extinction = max(1.0, -4 * dir.y + 1.0);

    vec3 color = extinction * extra_extinction * leakage_correction * base_color;

    if (abs(dir.x - sun_direction.x) < 0.1 &&
    abs(dir.y - sun_direction.y) < 0.1 &&
    abs(dir.z - sun_direction.z) < 0.1) {
        color = color * 4;
    }
    return color;
}