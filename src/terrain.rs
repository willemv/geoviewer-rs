use image;
use std;
use std::error::Error;
use std::num::NonZeroU32;
use std::sync::Arc;
use wgpu::TextureView;

use futures;
use futures::channel::oneshot::*;
use futures::executor::ThreadPool;

type TextureResult = Result<wgpu::Texture, Box<dyn Error + Send + Sync>>;

pub struct AsyncTexture {
    texture_view: wgpu::TextureView,
    future_texture: Option<Receiver<TextureResult>>,
}

fn load_texture(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    filename: &str,
) -> TextureResult {
    println!("Reading file");
    let diffuse_image = image::io::Reader::open(filename)?.decode()?;

    println!("Into rgba");
    let diffuse_rgba = diffuse_image.into_rgba8();

    let dimensions = diffuse_rgba.dimensions();

    let texture_size = wgpu::Extent3d {
        width: dimensions.0,
        height: dimensions.1,
        // All textures are stored as 3D, we represent our 2D texture by setting depth to 1.
        depth_or_array_layers: 1,
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
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        label: Some("diffuse_texture"),
    });

    println!("Queuing texture write");
    queue.write_texture(
        // Tells wgpu where to copy the pixel data
        wgpu::ImageCopyTexture {
            texture: &diffuse_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All
        },
        // The actual pixel data
        &diffuse_rgba,
        // The layout of the texture
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(NonZeroU32::new(4 * dimensions.0).unwrap()),
            ..Default::default()
            // rows_per_image: dimensions.1,
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
            depth_or_array_layers: 1,
        };
        let diffuse_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1, // We'll talk about this a little later
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            // SAMPLED tells wgpu that we want to use this texture in shaders
            // COPY_DST means that we want to copy data to this texture
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            label: Some("diffuse_texture"),
        });

        queue.write_texture(
            // Tells wgpu where to copy the pixel data
            wgpu::ImageCopyTexture {
                texture: &diffuse_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            // The actual pixel data
            &color_data,
            // The layout of the texture
            wgpu::ImageDataLayout {
                offset: 0,
                ..Default::default()
                // bytes_per_row: 4 * dimensions.0,
                // rows_per_image: dimensions.1,
            },
            texture_size,
        );

        let mut result = AsyncTexture {
            texture_view: diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default()),
            future_texture: None,
        };

        result.start_loading(device, queue, "assets/eo_base_2020_clean_720x360.jpg");

        result
    }

    pub fn load_hi_res_texture(
        &mut self,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Result<(), Box<dyn Error>> {
        self.start_loading(device, queue, "assets/eo_base_2020_clean_3600x1800.png");
        Ok(())
    }

    fn start_loading(
        &mut self,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        filename: &'static str,
    ) {
        let (sender, receiver) = channel();
        let f = async move {
            let t = load_texture(device, queue, filename);
            if let Err(_) = sender.send(t) {
                println!("Unable to send texture result to paint queue");
            }
        };
        let tp = ThreadPool::new().expect("msg");
        tp.spawn_ok(f);

        self.future_texture = Some(receiver);
    }

    pub fn get_texture(&mut self) -> &TextureView {
        if let Some(ref mut receiver) = &mut self.future_texture {
            let rec = receiver.try_recv();
            match rec {
                Err(Canceled) => {
                    std::mem::take(&mut self.future_texture);
                }
                Ok(None) => {} //still waiting for completion
                Ok(Some(ref result)) => {
                    std::mem::take(&mut self.future_texture);
                    if let Ok(texture) = result {
                        self.texture_view =
                            texture.create_view(&wgpu::TextureViewDescriptor::default())
                    }
                }
            }
        }

        &self.texture_view
    }
}
