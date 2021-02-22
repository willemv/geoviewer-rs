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

layout (location = 0) out vec3 viewPosition;

void main() {
    vec4 view = vu.view * vu.model * position;
    viewPosition = view.xyz;
    gl_Position = vu.projection * view;
}
