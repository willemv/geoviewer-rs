#![allow(clippy::many_single_char_names)]
use std::collections::HashMap;

use glam::{f32, Quat, Vec2, Vec4};

pub fn create(n: u8, radius: f32) -> (Vec<Vec4>, Vec<Vec2>, Vec<u16>) {
    let n = n as u16;

    let num_vertices_per_side = (2 + n * n) as u16;
    let vertex_num = num_vertices_per_side * 4;
    let mut vertices: Vec<Vec4> = Vec::with_capacity(vertex_num as usize);
    let mut indices = Vec::with_capacity(vertex_num as usize); //TODO: this initial capacity is not the actual size at the end, we can calculate that though

    let north_pole = Quat::from_xyzw(0.0, 0.0, 1.0, 0.0);
    let south_pole = Quat::from_xyzw(0.0, 0.0, -1.0, 0.0);

    let back = Quat::from_xyzw(-1.0, 0.0, 0.0, 0.0);
    let west = Quat::from_xyzw(0.0, -1.0, 0.0, 0.0);
    let front = Quat::from_xyzw(1.0, 0.0, 0.0, 0.0);
    let east = Quat::from_xyzw(0.0, 1.0, 0.0, 0.0);

    let init_vectors = [
        back, west, // 0
        west, front, // 1
        front, east, // 2
        east, back, //3
    ];

    for i in 0..4 {
        let top = north_pole;
        let left = init_vectors[2 * i];
        let right = init_vectors[2 * i + 1];
        let bottom = south_pole;

        let i = i as u16;

        //region Vertices

        //add the north pole
        vertices.push(Vec4::new(top.x, top.y, top.z, 1.0));

        let index_start = i * num_vertices_per_side;

        // ] north pole, equator ]
        for p in 0..n {
            let on_left_edge = top.lerp(left, (p + 1) as f32 / n as f32);
            let on_right_edge = top.lerp(right, (p + 1) as f32 / n as f32);

            for q in 0..=p {
                let v = on_left_edge.lerp(on_right_edge, q as f32 / (p + 1) as f32);
                vertices.push(Vec4::new(v.x, v.y, v.z, 1.0));
            }
        }

        // ] equator, south pole [
        for p in 1..n {
            let on_left_edge = left.lerp(bottom, p as f32 / n as f32);
            let on_right_edge = right.lerp(bottom, p as f32 / n as f32);

            for q in 0..(n - p) {
                let a = on_left_edge.lerp(on_right_edge, q as f32 / (n - p) as f32);
                vertices.push(Vec4::new(a.x, a.y, a.z, 1.0));
            }
        }

        //add the south pole
        vertices.push(Vec4::new(bottom.x, bottom.y, bottom.z, 1.0));

        //endregion

        //region Indices

        let idx = |face_idx: u16| (index_start + face_idx) % vertex_num;

        // the top triangle
        indices.push(idx(1)); //add the first triangle twice to break with the previous strip
        indices.push(idx(1));
        indices.push(idx(0));
        indices.push(idx(num_vertices_per_side + 1));
        //degenerate triangles to break to the next strip
        indices.push(idx(num_vertices_per_side + 1));
        indices.push(idx(num_vertices_per_side + 1)); //twice, to avoid flipping the winding

        let mut start_of_previous_line = idx(1);
        for p in 1..n {
            let start_of_current_line = start_of_previous_line + p;
            indices.push(start_of_current_line); //add the first index twice, to force a break in the triangle strip with a degenerate triangle
            indices.push(start_of_current_line);
            for q in 0..p {
                indices.push(start_of_previous_line + q);
                indices.push(start_of_current_line + q + 1);
            }

            //close the seam
            indices.push((num_vertices_per_side + start_of_previous_line) % vertex_num);
            indices.push((num_vertices_per_side + start_of_current_line) % vertex_num);

            indices.push(indices[indices.len() - 1]); //add that last index twice
            indices.push(indices[indices.len() - 1]); //add that last index trhice, to avoid flipping the winding
            start_of_previous_line = start_of_current_line;
        }

        indices.remove(indices.len() - 1); //that last flip should happen

        for p in 0..n - 1 {
            let pp = n - p;
            let start_of_current_line = start_of_previous_line + pp;
            indices.push(start_of_previous_line);
            indices.push(start_of_previous_line);
            for q in 0..pp - 1 {
                indices.push(start_of_current_line + q);
                indices.push(start_of_previous_line + q + 1);
            }

            //close the seam
            indices.push((num_vertices_per_side + start_of_current_line) % vertex_num);
            indices.push((num_vertices_per_side + start_of_previous_line) % vertex_num);

            indices.push(indices[indices.len() - 1]); //add that last index twice
            indices.push(indices[indices.len() - 1]); //add that last index thrice
            start_of_previous_line = start_of_current_line;
        }

        // the bottom triangle
        indices.push(idx(num_vertices_per_side - 2));
        indices.push(idx(num_vertices_per_side - 2));
        indices.push(idx(num_vertices_per_side - 1));
        indices.push(idx(2 * num_vertices_per_side - 2));
        //degenerate triangles to break to the next face
        indices.push(idx(2 * num_vertices_per_side - 2));

        //endregion
    }

    //region duplicate last edge vertices
    let duplicate_edge_vertex_num = n * 2 - 1;

    let mut acc = 0;
    for v in 0..n {
        let vf = (v + 1) as f32;
        let nf = n as f32;
        let v = north_pole.lerp(back, vf / nf);

        vertices.push(Vec4::new(v.x, v.y, v.z, 1.0));
        acc += 1;
    }

    for v in 1..n {
        let vf = v as f32;
        let nf = n as f32;
        let v = back.lerp(south_pole, vf / nf);

        vertices.push(Vec4::new(v.x, v.y, v.z, 1.0));
        acc += 1;
    }

    assert!(acc == duplicate_edge_vertex_num);

    let mut mapping = HashMap::new();
    let mut acc = 1u16;
    for i in 1..=n {
        mapping.insert(acc, vertex_num + i - 1);
        acc += i;
    }
    for i in 1..n {
        mapping.insert(acc, vertex_num + i + n - 1);
        acc = acc + n - i;
    }
    let mapping = mapping;

    let min_index = num_vertices_per_side * 3;
    let min_index = min_index as u16;
    let u = indices.len() / 4;
    let tofixup = &mut indices[3 * u..];
    for f in tofixup {
        if *f < min_index {
            let replacement = mapping.get(f).expect("the only low indices in this quadrant should be on the edge of the next quadrant, all of which should have gotten a mapping!");
            *f = *replacement;
        }
    }
    //endregion

    let uv = create_uv(n as usize, &vertices);

    for v in &mut vertices {
        *v *= radius;
        v.w = 1.0;
    }

    (vertices, uv, indices)
}

fn create_uv(n: usize, vertices: &[Vec4]) -> Vec<Vec2> {
    let mut uv = Vec::with_capacity(vertices.len());

    let tri = 2 + n * n;

    for v in vertices.iter() {
        let mut texture_coordinates = Vec2::splat(0.0);
        texture_coordinates.x = v.y.atan2(v.x) / (2.0 * std::f32::consts::PI) - 0.5;

        if texture_coordinates.x < 0.0 {
            texture_coordinates.x += 1.0;
        }

        texture_coordinates.y = 0.5 - v.z.asin() / std::f32::consts::PI;
        uv.push(texture_coordinates);
    }

    let tt = tri;
    uv[0].x = 0.125;
    uv[tt - 1].x = 0.125;

    uv[tt].x = 0.375;
    uv[2 * tt - 1].x = 0.375;

    uv[2 * tt].x = 0.625;
    uv[3 * tt - 1].x = 0.625;

    uv[3 * tt].x = 0.875;
    uv[4 * tt - 1].x = 0.875;

    // force u to 1.0 for all duplicated seam vertices
    let seam_start_index = vertices.len() - (2 * n - 1);
    for uv in &mut uv[seam_start_index..] {
        uv.x = 1.0;
    }

    uv
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    pub fn lod_one() {
        let (vert, uv, idx) = create(1, 2.0);
        println!("vert: {vert:?}");
        println!("idx: {idx:?}");
        println!("uv: {uv:?}");
    }
    #[test]
    pub fn lod_two() {
        let (vert, uv, idx) = create(2, 2.0);
        println!("vert: {vert:?}");
        println!("idx: {idx:?}");
        println!("uv: {uv:?}");
    }
    #[test]
    pub fn lod_three() {
        let (vert, uv, idx) = create(3, 2.0);
        println!("vert: {vert:?}");
        println!("idx: {idx:?}");
        println!("uv: {uv:?}");
    }
    #[test]
    pub fn lod_five() {
        let (vert, uv, idx) = create(5, 2.0);
        println!("vert: {vert:?}");
        println!("idx: {idx:?}");
        println!("uv: {uv:?}");
    }
}
