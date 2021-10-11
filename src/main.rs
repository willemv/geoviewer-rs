#![deny(clippy::pedantic)]

extern crate bytemuck;
extern crate crossbeam;
extern crate futures;
extern crate glam;
extern crate image;
extern crate imgui;
extern crate imgui_winit_support;
extern crate naga;
extern crate wgpu;
extern crate winit;

mod terrain;

mod app;
use app::*;

mod camera;
use camera::*;

mod controller;
use controller::*;

mod model;
use model::*;

mod octosphere;

mod simple_error;
use simple_error::*;
use wgpu::include_wgsl;

use std::error::Error;
use std::f64::consts::PI;
use std::mem;
use std::sync::Arc;
use std::time::SystemTime;

use bytemuck::{Pod, Zeroable};
use imgui::*;
use imgui_winit_support::*;
use wgpu::util::DeviceExt;
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

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
    _tex: [f32; 2],
}

unsafe impl Pod for Vertex {}
unsafe impl Zeroable for Vertex {}

fn vertex(pos: [f32; 4], tex: [f32; 2]) -> Vertex {
    Vertex {
        _pos: pos,
        _tex: tex,
    }
}

fn create_octo_sphere(subdivisions: usize, half: f64) -> (Vec<Vertex>, Vec<u16>) {
    let half = half as f32;

    let (vertices, indices, uvs) = octosphere::create(subdivisions as u8, half);

    let vertex_data: Vec<Vertex> = vertices
        .into_iter()
        .zip(uvs.into_iter())
        .map(|(v, t)| vertex([v.x, v.y, v.z, 1.0], [t.x, t.y]))
        .collect();

    let index_data: Vec<u16> = indices.into_iter().map(|i| i as u16).collect();

    (vertex_data, index_data)
}

struct RenderContext {
    window: Window,
    surface: wgpu::Surface,
    device: Arc<wgpu::Device>,
    shader: wgpu::ShaderModule,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    render_pipeline: wgpu::RenderPipeline,
    surface_configuration: wgpu::SurfaceConfiguration,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    async_texture: terrain::AsyncTexture,
    sampler: wgpu::Sampler,
    queue: Arc<wgpu::Queue>,
    depth_texture: wgpu::TextureView,
}

struct Gui {
    imgui: imgui::Context,
    imgui_platform: imgui_winit_support::WinitPlatform,
    imgui_renderer: imgui_wgpu::Renderer,
}

fn create_render_pipeline(
    device: &wgpu::Device,
    shader_module: &wgpu::ShaderModule,
    pipeline_layout: &wgpu::PipelineLayout,
    swap_chain_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("WGPU Pipeline"),
        layout: Some(pipeline_layout),
        vertex: wgpu::VertexState {
            module: shader_module,
            entry_point: "vs_main",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x2],
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader_module,
            entry_point: "fs_main",
            targets: &[swap_chain_format.into()],
        }),
        primitive: wgpu::PrimitiveState {
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            // strip_index_format: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
    })
}

fn prepare_new_shader(
    device: &wgpu::Device,
    pipeline_layout: &wgpu::PipelineLayout,
) -> Result<(wgpu::ShaderModule, wgpu::RenderPipeline), Box<dyn Error>> {
    let source = std::fs::read_to_string("src/geoviewer.wgsl")?;
    // create_shader_module panics if there are wgsl compilation errors
    // that really kills the interactivity, so parse with error checking first
    naga::front::wgsl::parse_str(&source)?;

    let shader_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: Some("geoviewer.wgsl"),
        source: wgpu::ShaderSource::Wgsl(source.into())
    });

    let new = create_render_pipeline(
        device,
        &shader_module,
        pipeline_layout,
        wgpu::TextureFormat::Bgra8Unorm,
    );
    Ok((shader_module, new))
}

async fn setup(window: Window) -> Result<(RenderContext, App, Gui), Box<dyn Error>> {
    //set up wgpu
    let window_size = window.inner_size();

    let instance = wgpu::Instance::new(wgpu::Backends::VULKAN);
    let surface = unsafe { instance.create_surface(&window) };

    println!("Found these adapters:");
    for adapter in instance.enumerate_adapters(wgpu::Backends::VULKAN) {
        println!("  {:?}", adapter.get_info());
    }
    println!();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: Some(&surface),
        })
        .await
        .ok_or_else(|| SimpleError::new("Could not find appropriate adapater"))?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
            },
            None,
        )
        .await?;

    println!("Adapter: {:?}", adapter.get_info());
    println!("Device: {:?}", device);

    let device = std::sync::Arc::new(device);
    let queue = std::sync::Arc::new(queue);

    let subdivisions = 16;
    //setup data
    let (vertices, indices) = create_octo_sphere(subdivisions, WORLD_RADIUS as f64);
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    let index_count = indices.len() as u32;

    // let vertex_shader_text = std::fs::read_to_string("src/shader.vert")?;

    // let mut compiler = shaderc::Compiler::new()
    //     .ok_or_else(|| SimpleError::new("Could not create shader compiler"))?;
    // let options = shaderc::CompileOptions::new()
    //     .ok_or_else(|| SimpleError::new("Could not create compile options"))?;
    // let binary = compiler.compile_into_spirv(
    //     &vertex_shader_text,
    //     shaderc::ShaderKind::Vertex,
    //     "shader_vert",
    //     "main",
    //     Some(&options),
    // )?;
    // let vertex_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
    //     label: None,
    //     source: wgpu::util::make_spirv(binary.as_binary_u8()),
    // });

    // let fragment_shader_text = std::fs::read_to_string("src/shader.frag")?;
    // let binary = compiler.compile_into_spirv(
    //     &fragment_shader_text,
    //     shaderc::ShaderKind::Fragment,
    //     "shader.frag",
    //     "main",
    //     Some(&options),
    // )?;
    let shader = device.create_shader_module(&include_wgsl!("geoviewer.wgsl"));

    let swap_chain_format = wgpu::TextureFormat::Bgra8Unorm;

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<VertexUniforms>() as _
                    ),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<FragmentUniforms>() as _,
                    ),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler {
                    comparison: false,
                    filtering: true,
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
        &shader,
        &pipeline_layout,
        swap_chain_format,
    );

    let surface_configuration = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: swap_chain_format,
        width: window_size.width,
        height: window_size.height,
        present_mode: wgpu::PresentMode::Mailbox,
    };

    let aspect = window_size.width as f64 / window_size.height as f64;
    surface.configure(&device, &surface_configuration);

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
        texture_format: surface_configuration.format,
        ..Default::default()
    };

    let renderer = imgui_wgpu::Renderer::new(&mut imgui, &device, &queue, renderer_config);

    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: surface_configuration.width,
            height: surface_configuration.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        label: Some("depth"),
    });

    let async_texture = terrain::AsyncTexture::new(device.clone(), queue.clone());

    // We don't need to configure the texture view much, so let's
    // let wgpu define it.
    // let diffuse_texture_view = diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let diffuse_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    Ok((
        RenderContext {
            window,
            surface,
            device,
            shader,
            pipeline_layout,
            bind_group_layout,
            render_pipeline,
            surface_configuration,
            vertex_buffer,
            index_buffer,
            index_count,
            async_texture,
            sampler: diffuse_sampler,
            queue,
            depth_texture: depth_texture.create_view(&wgpu::TextureViewDescriptor::default()),
        },
        App {
            start_time: SystemTime::now(),
            triangle_color: [1.0, 0.0, 0.0, 1.0],
            camera: Camera {
                eye: glam::Vec3::new(6.0 * WORLD_RADIUS, 0.0 * WORLD_RADIUS, 1.2 * WORLD_RADIUS),
                fov_y_radians: PI / 4.0,
                near: 0.25 * WORLD_RADIUS as f64,
                far: 15.0 * WORLD_RADIUS as f64,
                aspect,
            },
            controller: Controller::new(),
            demo_window_open: false,
            subdivisions,
        },
        Gui {
            imgui,
            imgui_renderer: renderer,
            imgui_platform: platform,
        },
    ))
}

fn render(context: &mut RenderContext, app: &mut App, gui: &mut Gui) -> Result<(), Box<dyn Error>> {
    let frame = context.surface.get_current_frame()?.output;

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
    let mut reload_vertex_buffer = false;
    let mut reload_texture = false;
    //draw ui
    {
        let mut t = [
            app.camera.eye.x / WORLD_RADIUS,
            app.camera.eye.y / WORLD_RADIUS,
            app.camera.eye.z / WORLD_RADIUS,
            (app.camera.near as f32 / WORLD_RADIUS),
            (app.camera.far as f32 / WORLD_RADIUS),
        ];

        let window = imgui::Window::new("Hello world");
        // let current_shader = context.current_shader.as_ref();
        window
            .position([0.0, 0.0], Condition::FirstUseEver)
            .size([400.0, 400.0], Condition::FirstUseEver)
            .build(&ui, || {
                ui.text("Text");
                reload_shaders = ui.button("Reload shaders");
                reload_texture = ui.button("Hi-res texture");
                ui.text("Camera");
                imgui::Drag::new("eye_x")
                    .range(-20.0, 20.0)
                    .speed(0.05)
                    .build(&ui, &mut t[0]);
                imgui::Drag::new("eye_y")
                    .range(-20.0, 20.0)
                    .speed(0.05)
                    .build(&ui, &mut t[1]);
                imgui::Drag::new("eye_z")
                    .range(-20.0, 20.0)
                    .speed(0.05)
                    .build(&ui, &mut t[2]);
                imgui::Drag::new("near")
                    .range(0.0, 20.0)
                    .speed(0.05)
                    .build(&ui, &mut t[3]);
                imgui::Drag::new("far")
                    .range(0.0, 20.0)
                    .speed(0.05)
                    .build(&ui, &mut t[4]);

                let mut s = app.subdivisions as i32;
                use std::convert::TryInto;

                reload_vertex_buffer =
                    imgui::InputInt::new(&ui, "subdivisions", &mut s).build();
                if reload_vertex_buffer && s > 0 {
                    app.subdivisions = s.try_into().unwrap_or(app.subdivisions);
                    println!("subs changed: {}", app.subdivisions);
                }
                ui.checkbox("demo", &mut app.demo_window_open);
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

    if reload_vertex_buffer {
        let (vertex_data, index_data) = create_octo_sphere(app.subdivisions, WORLD_RADIUS as f64);
        context.vertex_buffer =
            context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&vertex_data),
                    usage: wgpu::BufferUsages::VERTEX,
                });
        context.index_buffer =
            context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&index_data),
                    usage: wgpu::BufferUsages::INDEX,
                });
        context.index_count = index_data.len() as u32;
    }

    if reload_texture {
        match context
            .async_texture
            .load_hi_res_texture(context.device.clone(), context.queue.clone())
        {
            // Ok(texture_view) => context.diffuse_texture_view = texture_view,
            Err(e) => println!("error loading texture: {}", e),
            _ => {}
        }
    }

    if reload_shaders {
        match prepare_new_shader(&context.device, &context.pipeline_layout) {
            Ok((shader, pipeline)) => {
                context.shader = shader;
                context.render_pipeline = pipeline;
            }
            Err(err) => {
                println!("Error compiling shader: {}", err);
            }
        }
    }

    let view_width = context.surface_configuration.width;
    let view_height = context.surface_configuration.height;

    // camera position in world coordinates

    let eye = app.camera.eye;
    let center = glam::DVec3::new(0.0, 0.0, 0.0);
    let up = glam::DVec3::new(0.0, 0.0, 1.0);

    let view = glam::DMat4::look_at_rh(eye.as_dvec3(), center, up);
    let vertex_uniforms = VertexUniforms {
        model: glam::Mat4::IDENTITY,
        view: view.as_mat4(),
        projection: app.camera.perspective_matrix(),
    };

    let vertex_uniforms_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&vertex_uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let resolution = [view_width as f32, view_height as f32, 0.0, 0.0];
    let fragment_uniforms = FragmentUniforms {
        resolution,
        time: time as f32,
    };

    let fragment_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&fragment_uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Colors bind group descriptor"),
        layout: &context.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: vertex_uniforms_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: fragment_uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(context.async_texture.get_texture()),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&context.sampler),
            },
        ],
    });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("WGPU Command Encoder Descriptor"),
        });
    {
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: true,
                },
            }],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &context.depth_texture,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(f32::MAX),
                    store: false,
                }),
                stencil_ops: None,
            }),
        });
        render_pass.set_pipeline(&context.render_pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.set_index_buffer(context.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.set_vertex_buffer(0, context.vertex_buffer.slice(..));
        render_pass.draw_indexed(0..context.index_count, 0, 0..1);
    }
    {
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });

        gui.imgui_renderer.render(
            ui.render(),
            &context.queue,
            &context.device,
            &mut render_pass,
        )?;
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
                println!("Resized: {:?}", size);
                context.surface_configuration.width = size.width;
                context.surface_configuration.height = size.height;
                context.surface.configure(&context.device, &context.surface_configuration);

                let depth_texture = context.device.create_texture(&wgpu::TextureDescriptor {
                    size: wgpu::Extent3d {
                        width: size.width,
                        height: size.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: DEPTH_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    label: None,
                });
                context.depth_texture =
                    depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
                app.camera.aspect = size.width as f64 / size.height as f64;
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
                event:
                    WindowEvent::MouseInput {
                        button: winit::event::MouseButton::Left,
                        ref state,
                        ..
                    },
                ..
            } => match *state {
                winit::event::ElementState::Pressed => app.controller.mouse_pressed(),
                winit::event::ElementState::Released => app.controller.mouse_released(),
            },
            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                app.controller
                    .mouse_moved(position.x, position.y, &mut app.camera);
            }
            Event::WindowEvent {
                event: WindowEvent::MouseWheel { delta, .. },
                ..
            } => {
                //pos: push away
                match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => {
                        app.controller.scroll(y, &mut app.camera);
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
