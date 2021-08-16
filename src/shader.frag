#version 450

layout (binding = 1) uniform Input {
    vec4 iResolution;
    float iTime;
};

layout(binding = 2) uniform texture2D t_diffuse;
layout(binding = 3) uniform sampler s_diffuse;

layout (location = 0) in vec3 viewPosition;
layout (location = 1) in vec2 v_tex_coords;

layout (location = 0) out vec4 outColor;


vec3 normal(vec3 pos) {
  vec3 fdx = dFdx(pos);
  vec3 fdy = dFdy(pos);
  return normalize(cross(fdx, fdy));
}

void mainImage( out vec4 fragColor, in vec3 fragCoord )
{
    vec3 normal = normal(viewPosition);
    float factor = abs(dot(normal, vec3(0.0, 0.0, 1.0)));

    fragColor = texture(sampler2D(t_diffuse, s_diffuse), v_tex_coords) * factor;
}

void main() {
    mainImage(outColor, gl_FragCoord.xyz);
}