#version 450

layout (binding = 0) uniform VertexUniforms {
    mat4 model;
    mat4 view;
    mat4 projection;
} vu;

layout (location = 0) in vec4 position;
layout (location = 1) in vec2 a_tex_coords;

out gl_PerVertex {
    vec4 gl_Position;
};

layout (location = 0) out vec3 viewPosition;
layout (location = 1) out vec2 v_tex_coords;

void main() {
    v_tex_coords = a_tex_coords;
    vec4 view = vu.view * vu.model * position;
    viewPosition = view.xyz;
    gl_Position = vu.projection * view;
}
