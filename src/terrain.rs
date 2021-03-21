use image;
use std;
use std::sync::Arc;
use std::error::Error;
use std::io::BufRead;
use wgpu::{self, TextureView, TextureViewDescriptor};

use futures::executor::ThreadPool;
use futures;
use futures::channel::oneshot::*;

type TextureResult = Result<wgpu::Texture, Box<dyn Error + Send + Sync>>;

pub struct AsyncTexture {
    texture_view: wgpu::TextureView,
    future_texture: Option<Receiver<TextureResult>>,
}

fn load_texture(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> TextureResult {

    println!("Opening file");
    let diffuse_file = std::fs::File::open("assets/eo_base_2020_clean_3600x1800.png")?;
    println!("Reading file");
    let mut diffuse_file = std::io::BufReader::new(diffuse_file);
    //call fill_buff without the corresponding consume, to peek the initial bytes of the file and guess the format
    let header = diffuse_file.fill_buf()?;
    let format = image::guess_format(header)?;
    let diffuse_image = image::load(diffuse_file, format)?;

    println!("Into rgba");
    let diffuse_rgba = diffuse_image.into_rgba8();

    let dimensions = diffuse_rgba.dimensions();

    let texture_size = wgpu::Extent3d {
        width: dimensions.0,
        height: dimensions.1,
        // All textures are stored as 3D, we represent our 2D texture by setting depth to 1.
        depth: 1,
    };
    println!("Creating texture");
    let diffuse_texture = device.create_texture(&wgpu::TextureDescriptor {
        size: texture_size,
        mip_level_count: 1, // We'll talk about this a little later
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        // SAMPLED tells wgpu that we want to use this texture in shaders
        // COPY_DST means that we want to copy data to this texture
        usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        label: Some("diffuse_texture"),
    });


    println!("Queuing texture write");
    queue.write_texture(
        // Tells wgpu where to copy the pixel data
        wgpu::TextureCopyView {
            texture: &diffuse_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        // The actual pixel data
        &diffuse_rgba,
        // The layout of the texture
        wgpu::TextureDataLayout {
            offset: 0,
            bytes_per_row: 4 * dimensions.0,
            rows_per_image: dimensions.1,
        },
        texture_size,
    );
    println!("Done loading texture");

    Ok(diffuse_texture)
}

impl AsyncTexture {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> AsyncTexture {
        //rgb(165, 205, 213)
        let color: u32 = 0xa5cdd5ff;
        let color_data = color.to_be_bytes();
        let dimensions = (1, 1);

        let texture_size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            // All textures are stored as 3D, we represent our 2D texture by setting depth to 1.
            depth: 1,
        };
        let diffuse_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1, // We'll talk about this a little later
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            // SAMPLED tells wgpu that we want to use this texture in shaders
            // COPY_DST means that we want to copy data to this texture
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
            label: Some("diffuse_texture"),
        });

        queue.write_texture(
            // Tells wgpu where to copy the pixel data
            wgpu::TextureCopyView {
                texture: &diffuse_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            // The actual pixel data
            &color_data,
            // The layout of the texture
            wgpu::TextureDataLayout {
                offset: 0,
                bytes_per_row: 4 * dimensions.0,
                rows_per_image: dimensions.1,
            },
            texture_size,
        );

        let (sender, receiver) = channel();
        let f = async {
            let t = load_texture(device, queue);
            if let Err(_) =  sender.send(t) {
                println!("Unable to send texture result to paint queue");

            }
        };
        let tp = ThreadPool::new().expect("msg");
        tp.spawn_ok(f);

        AsyncTexture {
            texture_view: diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default()),
            future_texture: Some(receiver)

        }
    }

    pub fn init(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), Box<dyn Error>> {
        let diffuse_file = std::fs::File::open("assets/eo_base_2020_clean_3600x1800.png")?;
        let mut diffuse_file = std::io::BufReader::new(diffuse_file);
        //call fill_buff without the corresponding consume, to peek the initial bytes of the file and guess the format
        let header = diffuse_file.fill_buf()?;
        let format = image::guess_format(header)?;
        let diffuse_image = image::load(diffuse_file, format)?;
        let diffuse_rgba = diffuse_image.into_rgba8();

        let dimensions = diffuse_rgba.dimensions();

        let texture_size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            // All textures are stored as 3D, we represent our 2D texture by setting depth to 1.
            depth: 1,
        };
        let diffuse_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1, // We'll talk about this a little later
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            // SAMPLED tells wgpu that we want to use this texture in shaders
            // COPY_DST means that we want to copy data to this texture
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
            label: Some("diffuse_texture"),
        });

        queue.write_texture(
            // Tells wgpu where to copy the pixel data
            wgpu::TextureCopyView {
                texture: &diffuse_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            // The actual pixel data
            &diffuse_rgba,
            // The layout of the texture
            wgpu::TextureDataLayout {
                offset: 0,
                bytes_per_row: 4 * dimensions.0,
                rows_per_image: dimensions.1,
            },
            texture_size,
        );

        self.texture_view = diffuse_texture.create_view(&TextureViewDescriptor::default());
        Ok(())
    }

    pub fn get_texture(&mut self) -> &TextureView {

        if let Some(ref mut receiver) = &mut self.future_texture {
            let rec = receiver.try_recv();
            match rec {
                Err(Canceled) => {
                    std::mem::take(&mut self.future_texture);
                },
                Ok(None) => {}, //still waiting for completion
                Ok(Some(ref result)) => {
                    std::mem::take(&mut self.future_texture);
                    if let Ok(texture) = result {
                        self.texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default())
                    }
                }
            }
        }

        &self.texture_view
    }
}
