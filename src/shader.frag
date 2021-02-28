#version 450

layout (binding = 1) uniform Input {
    vec4 iResolution;
    float iTime;
};

layout (location = 0) in vec3 viewPosition;

layout (location = 0) out vec4 outColor;


vec3 normal(vec3 pos) {
  vec3 fdx = dFdx(pos);
  vec3 fdy = dFdy(pos);
  return normalize(cross(fdx, fdy));
}

void mainImage( out vec4 fragColor, in vec3 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    // fragColor = vec4(-gl_FragDepth, -gl_FragDepth, -gl_FragDepth, 1.0);
    // vec3 col = vec3(1.0-fragCoord.z) ;/// 80000000.0;
    // fragColor = vec4(col, 1.0);
    vec2 uv = fragCoord.xy/iResolution.xy;

    // Time varying pixel color
    vec3 col = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(0,2,4));

    vec3 normal = normal(viewPosition);
    float factor = abs(dot(normal, vec3(0.0, 0.0, 1.0)));
    // Output to screen
    fragColor = vec4(col * factor,1.0);

}

void main() {
    mainImage(outColor, gl_FragCoord.xyz);
}