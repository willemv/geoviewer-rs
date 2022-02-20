use glam::Vec3;

use crate::camera::Camera;
use std::f32::consts::PI;

pub struct Controller {
    mouse_down: bool,
    last_mouse_coordinates: Option<([f64; 2], Vec3)>,
    mouse_down_coordinates: Option<([f64; 2], Vec3)>,
}

impl Controller {
    pub fn new() -> Controller {
        Controller {
            mouse_down: false,
            last_mouse_coordinates: None,
            mouse_down_coordinates: None,
        }
    }

    pub fn mouse_pressed(&mut self) {
        self.mouse_down_coordinates = self.last_mouse_coordinates;
        self.mouse_down = true;
    }

    pub fn mouse_released(&mut self) {
        self.mouse_down_coordinates = None;
        self.mouse_down = false;
    }

    pub fn mouse_moved(&mut self, x: f64, y: f64, camera: &mut Camera) {
        self.last_mouse_coordinates = Some(([x, y], camera.eye));

        if !self.mouse_down {
            return;
        }

        let (md_point, md_eye) = match self.mouse_down_coordinates {
            Some(c) => c,
            None => {
                self.mouse_down_coordinates = Some(([x, y], camera.eye));
                ([x, y], camera.eye)
            }
        };

        let dx = (x - md_point[0]) as f32;
        let dy = (y - md_point[1]) as f32;

        let original_r = md_eye.length();
        let original_rho = md_eye.y.atan2(md_eye.x);
        let original_theta = (md_eye.z / original_r).asin();

        let new_rho = original_rho + (-dx / 300.0 * 2.0 * PI);
        let new_theta = (original_theta + (dy / 300.0 * PI)).clamp(-PI / 2.0 + 0.1, PI / 2.0 - 0.1);

        camera.eye.x = original_r * new_theta.cos() * new_rho.cos();
        camera.eye.y = original_r * new_theta.cos() * new_rho.sin();
        camera.eye.z = original_r * new_theta.sin();
    }

    pub fn scroll(&mut self, y: f32, camera: &mut Camera) {

        let axis = crate::ELLIPSOID.semimajor_axis / 2.0;
        let axis = axis as f32;
        self.mouse_down_coordinates = None;
        self.last_mouse_coordinates = None;

        let center = glam::Vec3::splat(0.0);
        let eye = camera.eye;

        let max_dist_center_geometry = axis;

        // delta y of 1.0 means +10% distance from 'world surface' to camera
        let fraction = 1.0 + (y / 10.0);

        let current_distance = eye.distance(center);
        //TODO correct for current_distance < WORLD_RADIUS
        let current_distance_s = current_distance - axis;
        let new_distance_s = current_distance_s * fraction;

        let dir = eye - center;
        let new_eye = dir.normalize() * (new_distance_s + axis);
        camera.eye = new_eye;

        camera.near =
            f64::from(new_distance_s + axis - max_dist_center_geometry as f32 - 1e3)
                .max(0.0);
        camera.far =
            f64::from(new_distance_s + axis + max_dist_center_geometry as f32 + 1e3);
    }
}
