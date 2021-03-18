use std::time::SystemTime;

use crate::camera::*;
use crate::controller::*;

#[allow(dead_code)]
pub struct App {
    pub start_time: SystemTime,
    pub triangle_color: [f32; 4],
    pub camera: Camera,
    pub controller: Controller,
    pub demo_window_open: bool,
    pub subdivisions: usize,
}
