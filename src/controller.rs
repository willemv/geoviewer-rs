use glam::Vec3;

use crate::camera::*;
use crate::model::*;

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

        let (mdp, mde) = match self.mouse_down_coordinates {
            Some(c) => c,
            None => {
                self.mouse_down_coordinates = Some(([x, y], camera.eye));
                ([x, y], camera.eye)
            }
        };

        let dx = (x - mdp[0]) as f32;
        let dy = (y - mdp[1]) as f32;

        let original_r = mde.length();
        let original_rho = mde.y.atan2(mde.x);
        let original_theta = (mde.z / original_r).asin();

        use std::f32::consts::PI;
        let new_rho = original_rho + (-dx / 300.0 * 2.0 * PI);
        let new_theta = (original_theta + (dy / 300.0 * PI)).clamp(-PI / 2.0 + 0.1, PI / 2.0 - 0.1);

        camera.eye.x = original_r * new_theta.cos() * new_rho.cos();
        camera.eye.y = original_r * new_theta.cos() * new_rho.sin();
        camera.eye.z = original_r * new_theta.sin();
    }

    pub fn scroll(&mut self, y: f32, camera: &mut Camera) {
        self.mouse_down_coordinates = None;
        self.last_mouse_coordinates = None;

        let center = glam::Vec3::splat(0.0);
        let eye = camera.eye;

        let max_dist_center_geometry = WORLD_RADIUS;

        // delta y of 1.0 means +10% distance from 'world surface' to camera
        let fraction = 1.0 + (y / 10.0);

        let current_distance = eye.distance(center);
        //TODO correct for current_distance < WORLD_RADIUS
        let current_distance_s = current_distance - WORLD_RADIUS;
        let new_distance_s = current_distance_s * fraction;

        let dir = eye - center;
        let new_eye = dir.normalize() * (new_distance_s + WORLD_RADIUS);
        camera.eye = new_eye;

        camera.near = ((new_distance_s + WORLD_RADIUS - max_dist_center_geometry as f32 - 1e3)
            as f64)
            .max(0.0);
        camera.far = (new_distance_s + WORLD_RADIUS + max_dist_center_geometry as f32 + 1e3) as f64;
    }
}
