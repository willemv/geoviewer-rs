use glam;

#[derive(Debug, Copy, Clone)]
pub struct Camera {
    pub eye: glam::Vec3,
    pub near: f64,
    pub far: f64,
    pub fov_y_radians: f64,
    pub aspect: f64,
}

impl Camera {
    pub fn perspective_matrix(&self) -> glam::Mat4 {
        glam::DMat4::perspective_rh(
            self.fov_y_radians,
            self.aspect,
            self.near,
            self.far,
        ).as_f32()
    }
}