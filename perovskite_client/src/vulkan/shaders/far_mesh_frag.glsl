#version 460

layout(location = 0) out vec4 f_color;
layout(location = 0) flat in vec3 color;


void main() {
    // Discard half of fragments while developing to make it easier
    // to see other geometry
    int stipple = (int(gl_FragCoord.x + gl_FragCoord.y)) % 2;
    if (stipple != 0) {
        discard;
    }
    f_color = vec4(color, 1.0);
}