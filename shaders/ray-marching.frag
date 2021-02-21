#version 450

layout(binding = 0) uniform Input {
    vec4 iResolution;
    float iTime;
};

layout(location = 0) out vec4 outColor;

const float EPSILON = 0.000001;
const int MAX_STEPS = 255;
const float MIN_DIST = 0.0;
const float MAX_DIST = 100.0;
const float LIGHT_JITTER = 0.001;

const vec3 K_a = vec3(0.5, 0.1, 0.1);
const vec3 K_d = vec3(0.7, 0.2, 0.2);
const vec3 K_s = vec3(1.0, 1.0, 1.0);
const float K_S = 10.0; //shininess

float sdSphere(vec3 p, vec3 c, float r) {
    return length(p-c) - r;
}

// // polynomial smooth min (k = 0.1);
// float smin( float a, float b, float k )
// {
//     float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
//     return mix( b, a, h ) - k*h*(1.0-h);
// }


// // polynomial smooth min (k = 0.1);
// float smin( float a, float b, float k )
// {
//     float h = max( k-abs(a-b), 0.0 )/k;
//     return min( a, b ) - h*h*k*(1.0/4.0);
// }

// exponential smooth min (k = 32);
float smin( float a, float b, float _k )
{
    float k = 3.2/_k;
    float res = exp2( -k*a ) + exp2( -k*b );
    return -log2( res )/k;
}

// // power smooth min (k = 8);
// float smin( float a, float b, float _k )
// {
//     float k = 0.8 / _k;
//     a = pow( a, k ); b = pow( b, k );
//     return pow( (a*b)/(a+b), 1.0/k );
// }

float sdScene(vec3 p) {
    float dist = 1e20;

    dist = smin(dist, sdSphere(p, vec3(0.0, 0.0, 0.0), 1.0), 0.1);
    dist = smin(dist, sdSphere(p, vec3(1.0, 0.0, 0.0), 1.0), 0.1);
    dist = smin(dist, sdSphere(p, vec3(-1.0, 0.0, 0.0), 1.0), 0.1);

    return dist;
}

float sceneDepth(vec3 eye, float start, float end, vec3 dir) {
    float depth = start;
    for (int i = 0; i < MAX_STEPS; i++) {
        float dist = sdScene(eye + (dir * depth)	);
        if (dist < EPSILON) {
            return depth + dist;
        }
        depth += dist;

        if (depth >= end) {
            return end;
        }
    }
    return end;
}

vec3 rayDirection(float verticalFov, vec2 size, vec2 fragCoord) {
    vec2 xy = fragCoord - (size / 2.0);
    float z = (size.y / 2.0) / tan(radians(verticalFov/2.0));
    return normalize(vec3(xy, z));
}

vec3 estimateNormal(vec3 p) {
    vec3 jX = vec3(LIGHT_JITTER, 0.0, 0.0);
    vec3 jY = vec3(0.0, LIGHT_JITTER, 0.0);
    vec3 jZ = vec3(0.0, 0.0, LIGHT_JITTER);
    return normalize(vec3(
        sdScene(p + jX) - sdScene(p - jX),
        sdScene(p + jY) - sdScene(p - jY),
        sdScene(p + jZ) - sdScene(p - jZ)
    ));
}

vec3 phongContribution(vec3 eye, vec3 p, vec3 n, vec3 lightPos, vec3 lightIntensity) {
    vec3 v = normalize(eye - p);
    vec3 l = normalize(lightPos - p);
    vec3 h = normalize(n + l);
    vec3 r = normalize(reflect(-l, n));

    float LN = dot(l, n);
    float RV = dot(r, v);



    if (RV < 0.0) {
        //reflecting away, only do diffuse
        return K_d * LN * lightIntensity;
    } else {
        float t = pow(5.0, 4.0);
        return K_d * LN * lightIntensity + K_s * pow(RV, K_S) * lightIntensity;
    }
}

vec3 phongShading(vec3 eye, vec3 p) {

    vec3 dir = normalize(p - eye);
    vec3 light_1 = vec3(2.0, 2.0, -2.0 * sin(iTime));
    vec3 light_2 = vec3(-2.0, 2.0, -2.0);
    vec3 ambient = K_a;
    return ambient
        + phongContribution(eye, p, estimateNormal(p), light_1, vec3(0.4, 0.4, 0.4))
        + phongContribution(eye, p, estimateNormal(p), light_2, vec3(0.4, 0.4, 0.4));
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec3 dir = rayDirection(45.0, iResolution.xy, fragCoord);
    float t = 1.0 / sin(radians(45.0/2.0));
    vec3 eye = vec3(0.0, 0.0, -t);

    float depth = sceneDepth(eye, MIN_DIST, MAX_DIST, dir);

    if (depth < MAX_DIST) {
        fragColor = vec4(phongShading(eye, eye + dir * depth), 1.0);
    } else {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
}

void main() {
    mainImage(outColor, vec2(gl_FragCoord.x, iResolution.y - gl_FragCoord.y));
}