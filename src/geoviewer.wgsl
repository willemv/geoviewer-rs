[[block]]
struct VertexUniforms {
    model: mat4x4<f32>;
    view: mat4x4<f32>;
    projection: mat4x4<f32>;
};

struct VertexInput {
    [[location(0)]] position: vec4<f32>;
    [[location(1)]] a_tex_coords: vec2<f32>;
};

struct VertexOutput {
    [[location(0)]] viewPosition: vec3<f32>;
    [[location(1)]] v_tex_coords: vec2<f32>;
    [[builtin(position)]] gl_Position: vec4<f32>;
};

[[group(0), binding(0)]]
var<uniform> vu: VertexUniforms;


[[stage(vertex)]]
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let view = vu.view * vu.model * in.position;
    out.v_tex_coords = in.a_tex_coords;
    out.viewPosition = view.xyz;
    out.gl_Position = vu.projection * view;
    return out;
}

[[block]]
struct FragmentUniforms {
    iResolution: vec4<f32>;
    iTime: f32;
};

struct FragmentOutput {
    [[location(0)]] outColor: vec4<f32>;
};

[[group(0), binding(1)]]
var<uniform> fu: FragmentUniforms;

[[group(0), binding(2)]]
var t_diffuse: texture_2d<f32>;
[[group(0), binding(3)]]
var s_diffuse: sampler;

fn normal(pos: vec3<f32>) -> vec3<f32> {
  let fdx = dpdx(pos);
  let fdy = dpdy(pos);
  return normalize(cross(fdx, fdy));
}

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> FragmentOutput
{
    var out: FragmentOutput;

    let normal = normal(in.viewPosition);
    let factor = abs(dot(normal, vec3<f32>(0.0, 0.0, 1.0)));

    out.outColor = textureSample(t_diffuse, s_diffuse, in.v_tex_coords) * factor;

    return out;
}