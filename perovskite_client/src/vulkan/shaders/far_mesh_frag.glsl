#version 460

layout(location = 0) out vec4 f_color;
layout(location = 0) flat in vec3 color;


void main() {
    // TODO: Once we have normals, use the color mapping code
    f_color = vec4(color, 1.0);
}