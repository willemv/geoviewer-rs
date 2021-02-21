#version 450

layout (binding = 0) uniform VertexUniforms {
    mat4 model;
    mat4 view;
    mat4 projection;
} vu;

layout (location = 0) in vec4 position;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    gl_Position = vu.projection * vu.view * vu.model * position;
}
