extern crate bytemuck;
extern crate crossbeam;
extern crate futures;
extern crate glam;
extern crate imgui;
extern crate imgui_winit_support;
extern crate shaderc;
extern crate wgpu;
extern crate winit;

mod icosphere;

mod simple_error;
use simple_error::*;

use std::error::Error;
use std::f64::consts::PI;
use std::mem;
use std::time::SystemTime;

use bytemuck::{Pod, Zeroable};
use imgui::*;
use imgui_winit_support::*;
use wgpu::util::DeviceExt;
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

const WORLD_RADIUS: f32 = 6_371_000.0 / 2.0;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Clone, Copy)]
struct Transforms {
    projection: [f32; 4],
    view: [f32; 4],
    model: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct VertexUniforms {
    model: glam::Mat4,
    view: glam::Mat4,
    projection: glam::Mat4,
}
unsafe impl Pod for VertexUniforms {}
unsafe impl Zeroable for VertexUniforms {}

#[repr(C)]
#[derive(Clone, Copy)]
struct FragmentUniforms {
    resolution: [f32; 4],
    time: f32,
}

unsafe impl Pod for FragmentUniforms {}
unsafe impl Zeroable for FragmentUniforms {}

#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    _pos: [f32; 4],
}

unsafe impl Pod for Vertex {}
unsafe impl Zeroable for Vertex {}

fn vertex(pos: [f32; 4]) -> Vertex {
    Vertex { _pos: pos }
}

#[allow(dead_code)]
fn create_cube(half: f64) -> (Vec<Vertex>, Vec<u16>) {
    let half = half as f32;
    let vertex_data = vec![
        //front left lower
        vertex([half, -half, -half, 1.0]),
        //front right lower
        vertex([half, half, - half, 1.0]),
        //back right lower
        vertex([-half, half, - half, 1.0]),
        //back left lower
        vertex([-half, -half, - half, 1.0]),
        //front left upper
        vertex([half, -half,  half, 1.0]),
        //front right upper
        vertex([half, half,  half, 1.0]),
        //back right upper
        vertex([-half, half,  half, 1.0]),
        //back left upper
        vertex([-half, -half,  half, 1.0]),
    ];

    let index_data = vec![
        0, 2, 1, 0, 3, 2, //bottom plane
        0, 1, 5, 0, 5, 4, //front plane
        1, 2, 6, 1, 6, 5, //right plane
        2, 3, 7, 2, 7, 6, //back plane
        3, 0, 4, 3, 4, 7, //left plane
        4, 5, 6, 4, 6, 7, //top plane
    ];

    (vertex_data, index_data)
}

fn create_ico_sphere(half: f64) -> (Vec<Vertex>, Vec<u16>) {
    let half = half as f32;

    let (vertices, indices, _uvs) = icosphere::create(4, half);

    let vertex_data = vertices.into_iter().map( |v| {
        vertex([v.x, v.y, v.z, 1.0])
    }).collect();

    let index_data = indices.into_iter().map(|i| i as u16).collect();

    (vertex_data, index_data)
}

#[derive(Debug)]
struct Data {
    string: String,
}

impl Drop for Data {
    fn drop(&mut self) {
        println!("Dropping self: {}", self.string);
    }
}

#[allow(dead_code)]
struct App {
    start_time: SystemTime,
    triangle_color: [f32; 4],
    camera: Camera,
    demo_window_open: bool,
}

struct RenderContext {
    window: Window,
    surface: wgpu::Surface,
    device: wgpu::Device,
    vertex_shader: wgpu::ShaderModule,
    fragment_shader: wgpu::ShaderModule,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    render_pipeline: wgpu::RenderPipeline,
    swap_chain_descriptor: wgpu::SwapChainDescriptor,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    swap_chain: wgpu::SwapChain,
    queue: wgpu::Queue,
    depth_texture: wgpu::TextureView,
}

#[derive(Debug, Copy, Clone)]
struct Camera {
    eye: glam::Vec3,
    near: f64,
    far: f64,
    fov_y_radians: f64,
}

struct Gui {
    imgui: imgui::Context,
    imgui_platform: imgui_winit_support::WinitPlatform,
    imgui_renderer: imgui_wgpu::Renderer,
}

fn create_render_pipeline(
    device: &wgpu::Device,
    vertex_shader_module: &wgpu::ShaderModule,
    fragment_shader_module: &wgpu::ShaderModule,
    pipeline_layout: &wgpu::PipelineLayout,
    swap_chain_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("WGPU Pipeline"),
        layout: Some(pipeline_layout),
        vertex_stage: wgpu::ProgrammableStageDescriptor {
            module: vertex_shader_module,
            entry_point: "main",
        },
        fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
            module: fragment_shader_module,
            entry_point: "main",
        }),
        rasterization_state: Some(wgpu::RasterizationStateDescriptor {
            front_face: wgpu::FrontFace::Cw,
            cull_mode: wgpu::CullMode::Back,
            ..Default::default()
        }),
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        color_states: &[swap_chain_format.into()],
        depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
            format: DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Greater,
            stencil: wgpu::StencilStateDescriptor::default(),
        }),
        vertex_state: wgpu::VertexStateDescriptor {
            index_format: wgpu::IndexFormat::Uint16,
            vertex_buffers: &[wgpu::VertexBufferDescriptor {
                stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
                step_mode: wgpu::InputStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float4],
            }],
        },
        sample_count: 1,
        sample_mask: !0,
        alpha_to_coverage_enabled: false,
    })
}

fn prepare_new_shader(
    device: &wgpu::Device,
    pipeline_layout: &wgpu::PipelineLayout,
) -> Result<(wgpu::ShaderModule, wgpu::ShaderModule, wgpu::RenderPipeline), Box<dyn Error>> {
    let vs_artifact = compile_shader("src/shader.vert", shaderc::ShaderKind::Vertex)?;
    let vs_module = device.create_shader_module(wgpu::util::make_spirv(vs_artifact.as_binary_u8()));

    let fs_artifact = compile_shader("src/shader.frag", shaderc::ShaderKind::Fragment)?;
    let fs_module = device.create_shader_module(wgpu::util::make_spirv(fs_artifact.as_binary_u8()));

    let new = create_render_pipeline(
        device,
        &vs_module,
        &fs_module,
        pipeline_layout,
        wgpu::TextureFormat::Bgra8Unorm,
    );
    Ok((vs_module, fs_module, new))
}

fn compile_shader(
    path: &str,
    kind: shaderc::ShaderKind,
) -> Result<shaderc::CompilationArtifact, Box<dyn Error>> {
    let shader_text = std::fs::read_to_string(path)?;
    let mut compiler = shaderc::Compiler::new()
        .ok_or_else(|| SimpleError::new("Could not create shader compiler"))?;
    // let options = shaderc::CompileOptions::new()
    // .ok_or_else(|| SimpleError::new("Could not create compile options"))?;
    let binary = compiler.compile_into_spirv(&shader_text, kind, path, "main", None)?;
    Ok(binary)
}

async fn setup(window: Window) -> Result<(RenderContext, App, Gui), Box<dyn Error>> {
    //set up wgpu
    let window_size = window.inner_size();

    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
    let surface = unsafe { instance.create_surface(&window) };

    println!("Found these adapters:");
    for adapter in instance.enumerate_adapters(wgpu::BackendBit::PRIMARY) {
        println!("  {:?}", adapter.get_info());
    }
    println!();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
        })
        .await
        .ok_or_else(|| SimpleError::new("Could not find appropriate adapater"))?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
                shader_validation: true,
            },
            None,
        )
        .await?;

    println!("Adapter: {:?}", adapter.get_info());
    println!("Device: {:?}", device);

    //setup data
    let (vertices, indices) = create_ico_sphere(WORLD_RADIUS as f64);
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsage::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsage::INDEX,
    });
    let index_count = indices.len() as u32;

    let vertex_shader_text = std::fs::read_to_string("src/shader.vert")?;

    let mut compiler = shaderc::Compiler::new()
        .ok_or_else(|| SimpleError::new("Could not create shader compiler"))?;
    let options = shaderc::CompileOptions::new()
        .ok_or_else(|| SimpleError::new("Could not create compile options"))?;
    let binary = compiler.compile_into_spirv(
        &vertex_shader_text,
        shaderc::ShaderKind::Vertex,
        "shader_vert",
        "main",
        Some(&options),
    )?;
    let vertex_shader = device.create_shader_module(wgpu::util::make_spirv(binary.as_binary_u8()));

    let fragment_shader_text = std::fs::read_to_string("src/shader.frag")?;
    let binary = compiler.compile_into_spirv(
        &fragment_shader_text,
        shaderc::ShaderKind::Fragment,
        "shader.frag",
        "main",
        Some(&options),
    )?;
    let fragment_shader =
        device.create_shader_module(wgpu::util::make_spirv(binary.as_binary_u8()));

    let swap_chain_format = wgpu::TextureFormat::Bgra8Unorm;

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::VERTEX,
                ty: wgpu::BindingType::UniformBuffer {
                    dynamic: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<VertexUniforms>() as _
                    ),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStage::FRAGMENT,
                ty: wgpu::BindingType::UniformBuffer {
                    dynamic: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<FragmentUniforms>() as _,
                    ),
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("WGPU Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let render_pipeline = create_render_pipeline(
        &device,
        &vertex_shader,
        &fragment_shader,
        &pipeline_layout,
        swap_chain_format,
    );

    let swap_chain_descriptor = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: swap_chain_format,
        width: window_size.width,
        height: window_size.height,
        present_mode: wgpu::PresentMode::Mailbox,
    };

    let swap_chain = device.create_swap_chain(&surface, &swap_chain_descriptor);

    //set up imgui
    let hidpi_factor = window.scale_factor();
    let mut imgui = imgui::Context::create();
    let mut platform = WinitPlatform::init(&mut imgui);
    platform.attach_window(imgui.io_mut(), &window, HiDpiMode::Default);
    imgui.set_ini_filename(None);

    let font_size = (15.0 * hidpi_factor) as f32;
    imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

    imgui
        .fonts()
        .add_font(&[imgui::FontSource::DefaultFontData {
            config: Some(imgui::FontConfig {
                oversample_h: 1,
                pixel_snap_h: true,
                size_pixels: font_size,
                ..Default::default()
            }),
        }]);

    //set up imgui_wgpu
    let renderer_config = imgui_wgpu::RendererConfig {
        texture_format: swap_chain_descriptor.format,
        ..Default::default()
    };

    let renderer = imgui_wgpu::Renderer::new(&mut imgui, &device, &queue, renderer_config);

    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: swap_chain_descriptor.width,
            height: swap_chain_descriptor.height,
            depth: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        label: Some("depth"),
    });

    Ok((
        RenderContext {
            window,
            surface,
            device: device,
            vertex_shader: vertex_shader,
            fragment_shader,
            pipeline_layout: pipeline_layout,
            bind_group_layout,
            render_pipeline,
            swap_chain_descriptor,
            swap_chain,
            vertex_buffer,
            index_buffer,
            index_count,
            queue,
            depth_texture: depth_texture.create_view(&wgpu::TextureViewDescriptor::default()),
        },
        App {
            start_time: SystemTime::now(),
            triangle_color: [1.0, 0.0, 0.0, 1.0],
            camera: Camera {
                eye: glam::Vec3::new(6.0 * WORLD_RADIUS, 6.0 * WORLD_RADIUS, 0.0 * WORLD_RADIUS),
                fov_y_radians: PI / 4.0,
                near: 5.0 * WORLD_RADIUS as f64,
                far: 7.0 * WORLD_RADIUS as f64,
            },
            demo_window_open: false,
        },
        Gui {
            imgui,
            imgui_renderer: renderer,
            imgui_platform: platform,
        },
    ))
}

fn render(context: &mut RenderContext, app: &mut App, gui: &mut Gui) -> Result<(), Box<dyn Error>> {
    let frame = context.swap_chain.get_current_frame()?.output;

    let duration = SystemTime::now()
        .duration_since(app.start_time)?
        .as_millis() as f64;
    let time = duration / 1000.0;

    let device = &context.device;

    let imgui = &mut gui.imgui;
    gui.imgui_platform
        .prepare_frame(imgui.io_mut(), &context.window)?;

    let ui = imgui.frame();

    let mut reload_shaders = false;
    //draw ui
    {
        let mut t = [
            app.camera.eye.x / WORLD_RADIUS,
            app.camera.eye.y / WORLD_RADIUS,
            app.camera.eye.z / WORLD_RADIUS,
            (app.camera.near as f32 / WORLD_RADIUS),
            (app.camera.far as f32 / WORLD_RADIUS),
        ];

        let window = imgui::Window::new(im_str!("Hello world"));
        // let current_shader = context.current_shader.as_ref();
        window
            .position([0.0, 0.0], Condition::FirstUseEver)
            .size([400.0, 400.0], Condition::FirstUseEver)
            .build(&ui, || {
                ui.text(im_str!("Text"));
                reload_shaders = ui.button(im_str!("Reload shaders"), [0.0, 0.0]);
                ui.text(im_str!("Camera"));
                imgui::Drag::new(im_str!("eye_x"))
                    .range(0.0..=20.0)
                    .speed(0.05)
                    .build(&ui, &mut t[0]);
                imgui::Drag::new(im_str!("eye_y"))
                    .range(0.0..=20.0)
                    .speed(0.05)
                    .build(&ui, &mut t[1]);
                imgui::Drag::new(im_str!("eye_z"))
                    .range(0.0..=20.0)
                    .speed(0.05)
                    .build(&ui, &mut t[2]);
                imgui::Drag::new(im_str!("near"))
                    .range(0.0..=20.0)
                    .speed(0.05)
                    .build(&ui, &mut t[3]);
                imgui::Drag::new(im_str!("far"))
                    .range(0.0..=20.0)
                    .speed(0.05)
                    .build(&ui, &mut t[4]);

                ui.checkbox(im_str!("demo"), &mut app.demo_window_open);
            });

        if app.demo_window_open {
            ui.show_demo_window(&mut app.demo_window_open);
        }

        app.camera.eye.x = t[0] * WORLD_RADIUS;
        app.camera.eye.y = t[1] * WORLD_RADIUS;
        app.camera.eye.z = t[2] * WORLD_RADIUS;
        app.camera.near = t[3] as f64 * WORLD_RADIUS as f64;
        app.camera.far = t[4] as f64 * WORLD_RADIUS as f64;
    }

    if reload_shaders {
        match prepare_new_shader(&context.device, &context.pipeline_layout) {
            Ok((vs_shader, fs_shader, pipeline)) => {
                context.vertex_shader = vs_shader;
                context.fragment_shader = fs_shader;
                context.render_pipeline = pipeline;
            }
            Err(ref err) => {
                println!("Error compiling shader: {}", *err);
            }
        }
    }

    let view_width = context.swap_chain_descriptor.width;
    let view_height = context.swap_chain_descriptor.height;
    let aspect = (view_width as f32) / (view_height as f32);

    // camera position in world coordinates

    let eye = app.camera.eye;
    let center = glam::DVec3::new(0.0, 0.0, 0.0);
    let up = glam::DVec3::new(0.0, 0.0, 1.0);

    let vertex_uniforms = VertexUniforms {
        model: glam::Mat4::identity(),
        view: glam::DMat4::look_at_lh(eye.as_f64(), center, up).as_f32(),
        projection: glam::Mat4::perspective_lh(
            app.camera.fov_y_radians as f32,
            aspect,
            app.camera.near as f32,
            app.camera.far as f32,
        ),
    };

    let vertex_uniforms_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&vertex_uniforms),
        usage: wgpu::BufferUsage::UNIFORM,
    });

    let resolution = [view_width as f32, view_height as f32, 0.0, 0.0];
    let fragment_uniforms = FragmentUniforms {
        resolution,
        time: time as f32,
    };

    let fragment_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&fragment_uniforms),
        usage: wgpu::BufferUsage::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Colors bind group descriptor"),
        layout: &context.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(vertex_uniforms_buf.slice(..)),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(fragment_uniform_buf.slice(..)),
            },
        ],
    });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("WGPU Command Encoder Descriptor"),
        });
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                attachment: &frame.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: true,
                },
            }],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                attachment: &context.depth_texture,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(-1.0),
                    store: false,
                }),
                stencil_ops: None,
            }),
        });
        render_pass.set_pipeline(&context.render_pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.set_index_buffer(context.index_buffer.slice(..));
        render_pass.set_vertex_buffer(0, context.vertex_buffer.slice(..));
        render_pass.draw_indexed(0..context.index_count, 0, 0..1);
    }
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                attachment: &frame.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });

        gui.imgui_renderer
            .render(
                ui.render(),
                &context.queue,
                &context.device,
                &mut render_pass,
            )
            .map_err(|_| SimpleError::new("Error rendering imgui"))?;
    }

    context.queue.submit(Some(encoder.finish()));

    Ok(())
}

async fn run() -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("WGPU Experiments")
        .with_inner_size(winit::dpi::PhysicalSize::new(1024, 768))
        .build(&event_loop)?;

    let (mut context, mut app, mut gui) = setup(window).await?;

    event_loop.run(move |event, _target, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                context.swap_chain_descriptor.width = size.width;
                context.swap_chain_descriptor.height = size.height;
                context.swap_chain = context
                    .device
                    .create_swap_chain(&context.surface, &context.swap_chain_descriptor);

                let depth_texture = context.device.create_texture(&wgpu::TextureDescriptor {
                    size: wgpu::Extent3d {
                        width: size.width,
                        height: size.height,
                        depth: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: DEPTH_FORMAT,
                    usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
                    label: None,
                });
                context.depth_texture =
                    depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
            }
            Event::MainEventsCleared => {
                match render(&mut context, &mut app, &mut gui) {
                    Ok(()) => {} //render successful
                    Err(error) => {
                        println!("Encountered error: {}", error);
                        *control_flow = ControlFlow::Exit;
                    }
                }
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                println!("Exiting...");
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::MouseWheel { delta, .. },
                ..
            } => {
                //pos: push away
                match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => {
                        let center = glam::Vec3::splat(0.0);
                        let eye = app.camera.eye;

                        let max_dist_center_geometry = WORLD_RADIUS;
                            // glam::Vec3::new(WORLD_RADIUS, WORLD_RADIUS, WORLD_RADIUS / 2.0)
                            //     .length();

                        // delta y of 1.0 means +10% distance from 'world surface' to camera
                        let fraction = 1.0 + (y / 10.0);

                        let current_distance = eye.distance(center);
                        //TODO correct for current_distance < WORLD_RADIUS
                        let current_distance_s = current_distance - WORLD_RADIUS;
                        let new_distance_s = current_distance_s * fraction;

                        let dir = eye - center;
                        let new_eye = dir.normalize() * (new_distance_s + WORLD_RADIUS);
                        app.camera.eye = new_eye;

                        app.camera.near = ((new_distance_s + WORLD_RADIUS
                            - max_dist_center_geometry as f32
                            - 1e3) as f64)
                            .max(0.0);
                        app.camera.far =
                            (new_distance_s + WORLD_RADIUS + max_dist_center_geometry as f32 + 1e3)
                                as f64;
                    }
                    _ => {}
                }
            }
            _ => {}
        }

        gui.imgui_platform
            .handle_event(gui.imgui.io_mut(), &context.window, &event);
    });
}

fn main() -> Result<(), Box<dyn Error>> {
    futures::executor::block_on(run())
}
