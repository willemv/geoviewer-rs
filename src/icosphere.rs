use glam::*;

// from https://github.com/kaiware007/IcoSphereCreator
// that one is written with Unity's coordinate system in mind:
// a left-handed coordinate system, with Y pointed up
// our world coordinate system however, is right handed
// and Z points up

pub fn create(n: u8, radius: f32) -> (Vec<Vec3>, Vec<usize>, Vec<Vec2>) {
    let n = n as usize;
    let vertex_num = n * n * 24;
    let mut vertices: Vec<Vec3> = Vec::with_capacity(vertex_num as usize);
    let mut triangles = Vec::with_capacity(vertex_num as usize);

    let init_vectors = [
        // 0
        Quat::from_xyzw(0.0, 0.0, 1.0, 0.0),
        Quat::from_xyzw(-1.0, 0.0, 0.0, 0.0),
        Quat::from_xyzw(0.0, -1.0, 0.0, 0.0),
        // 1
        Quat::from_xyzw(0.0,  0.0, -1.0, 0.0),
        Quat::from_xyzw(0.0, -1.0, 0.0, 0.0),
        Quat::from_xyzw(-1.0, 0.0, 0.0, 0.0),
        // 2
        Quat::from_xyzw(0.0, 0.0, 1.0, 0.0),
        Quat::from_xyzw(0.0, -1.0, 0.0, 0.0),
        Quat::from_xyzw(1.0, 0.0, 0.0, 0.0),
        // 3
        Quat::from_xyzw(0.0, 0.0, -1.0, 0.0),
        Quat::from_xyzw(1.0, 0.0, 0.0, 0.0),
        Quat::from_xyzw(0.0, -1.0, 0.0, 0.0),
        // 4
        Quat::from_xyzw(0.0, 0.0, 1.0, 0.0),
        Quat::from_xyzw(1.0, 0.0, 0.0, 0.0),
        Quat::from_xyzw(0.0, 1.0, 0.0, 0.0),
        // 5
        Quat::from_xyzw(0.0, 0.0, -1.0, 0.0),
        Quat::from_xyzw(0.0, 1.0, 0.0, 0.0),
        Quat::from_xyzw(1.0, 0.0, 0.0, 0.0),
        // 6
        Quat::from_xyzw(0.0, 0.0, 1.0, 0.0),
        Quat::from_xyzw(0.0, 1.0, 0.0, 0.0),
        Quat::from_xyzw(-1.0, 0.0, 0.0, 0.0),
        // 7
        Quat::from_xyzw(0.0, 0.0, -1.0, 0.0),
        Quat::from_xyzw(-1.0, 0.0, 0.0, 0.0),
        Quat::from_xyzw(0.0, 1.0, 0.0, 0.0),
    ];

    let mut j = 0; //index on vectors[]

    for i in (0..24).step_by(3) {
        /*
         *                   c _________d
         *    ^ /\           /\        /
         *   / /  \         /  \      /
         *  p /    \       /    \    /
         *   /      \     /      \  /
         *  /________\   /________\/
         *     q->       a         b
         */
        for p in 0..n {
            //edge index 1
            let edge_p1 = init_vectors[i].lerp(init_vectors[i + 2], p as f32 / n as f32);
            let edge_p2 = init_vectors[i + 1].lerp(init_vectors[i + 2], p as f32 / n as f32);
            let edge_p3 = init_vectors[i].lerp(init_vectors[i + 2], (p + 1) as f32 / n as f32);
            let edge_p4 = init_vectors[i + 1].lerp(init_vectors[i + 2], (p + 1) as f32 / n as f32);

            for q in 0..(n - p) {
                //edge index 2
                let a = edge_p1.lerp(edge_p2, q as f32 / (n - p) as f32);
                let b = edge_p1.lerp(edge_p2, (q + 1) as f32 / (n - p) as f32);
                let (c, d) = if edge_p3 == edge_p4 {
                    (edge_p3, edge_p3)
                } else {
                    (
                        edge_p3.lerp(edge_p4, q as f32 / (n - p - 1) as f32),
                        edge_p3.lerp(edge_p4, (q + 1) as f32 / (n - p - 1) as f32),
                    )
                };

                triangles.push(j);
                vertices.push(Vec3::new(a.x, a.y, a.z));
                j = j + 1;
                triangles.push(j);
                vertices.push(Vec3::new(b.x, b.y, b.z));
                j = j + 1;
                triangles.push(j);
                vertices.push(Vec3::new(c.x, c.y, c.z));
                j = j + 1;

                if q < n - p - 1 {
                    triangles.push(j);
                    vertices.push(Vec3::new(c.x, c.y, c.z));
                    j = j + 1;

                    triangles.push(j);
                    vertices.push(Vec3::new(b.x, b.y, b.z));
                    j = j + 1;

                    triangles.push(j);
                    vertices.push(Vec3::new(d.x, d.y, d.z));
                    j = j + 1;
                }
            }
        }
    }

    let uv = create_uv(n, &vertices);
    for i in 0..vertex_num {
        vertices[i] *= radius;
    }
    // mesh.RecalculateNormals();
    // CreateTangents(mesh);

    (vertices, triangles, uv)
}

fn create_uv(n: usize, vertices: &Vec<Vec3>) -> Vec<Vec2> {
    let vertex_num = n * n * 24;
    let mut uv = Vec::with_capacity(vertex_num);

    let tri = n * n; // divided triangle count (1,4,9...)
    let uv_limit = tri * 18; // range of wrap UV.x


    for i in 0..vertices.len() {
        let v = vertices[i];

        let mut texture_coordinates = Vec2::splat(0.0);
        if (v.y == 0.0) && (i > uv_limit) {
            texture_coordinates.x = 1.0;
        } else {
            texture_coordinates.x = v.y.atan2(v.x) / (2.0 * std::f32::consts::PI) - 0.5;
        }

        if texture_coordinates.x < 0.0 {
            texture_coordinates.x += 1.0;
        }

        texture_coordinates.y = 0.5 - v.z.asin() / std::f32::consts::PI;
        uv.push(texture_coordinates);
    }

    let tt = tri * 3;
    uv[0 * tt + 0].x = 0.875;
    uv[1 * tt + 0].x = 0.875;
    uv[2 * tt + 0].x = 0.125;
    uv[3 * tt + 0].x = 0.125;
    uv[4 * tt + 0].x = 0.625;
    uv[5 * tt + 0].x = 0.375;
    uv[6 * tt + 0].x = 0.375;
    uv[7 * tt + 0].x = 0.625;

    uv
}

// static void CreateTangents(Mesh mesh)
// {
//     int[] triangles = mesh.triangles;
//     Vec3[] vertices = mesh.vertices;
//     Vec2[] uv = mesh.uv;
//     Vec3[] normals = mesh.normals;

//     int triangleCount = triangles.Length;
//     int vertexCount = vertices.Length;

//     Vec3[] tan1 = Vec3::new[vertexCount];
//     Vec3[] tan2 = Vec3::new[vertexCount];

//     Vector4[] tangents = new Vector4[vertexCount];

//     for (int i = 0; i < triangleCount; i += 3)
//     {
//         int i1 = triangles[i + 0];
//         int i2 = triangles[i + 1];
//         int i3 = triangles[i + 2];

//         Vec3 v1 = vertices[i1];
//         Vec3 v2 = vertices[i2];
//         Vec3 v3 = vertices[i3];

//         Vec2 w1 = uv[i1];
//         Vec2 w2 = uv[i2];
//         Vec2 w3 = uv[i3];

//         float x1 = v2.x - v1.x;
//         float x2 = v3.x - v1.x;
//         float y1 = v2.y - v1.y;
//         float y2 = v3.y - v1.y;
//         float z1 = v2.z - v1.z;
//         float z2 = v3.z - v1.z;

//         float s1 = w2.x - w1.x;
//         float s2 = w3.x - w1.x;
//         float t1 = w2.y - w1.y;
//         float t2 = w3.y - w1.y;

//         float r = 1.0f / (s1 * t2 - s2 * t1);

//         Vec3 sdir = Vec3::new((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r);
//         Vec3 tdir = Vec3::new((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r);

//         tan1[i1] += sdir;
//         tan1[i2] += sdir;
//         tan1[i3] += sdir;

//         tan2[i1] += tdir;
//         tan2[i2] += tdir;
//         tan2[i3] += tdir;
//     }

//     for (int i = 0; i < vertexCount; ++i)
//     {
//         Vec3 n = normals[i];
//         Vec3 t = tan1[i];

//         Vec3.OrthoNormalize(ref n, ref t);
//         tangents[i].x = t.x;
//         tangents[i].y = t.y;
//         tangents[i].z = t.z;

//         tangents[i].w = ( Vec3.Dot( Vec3.Cross(n, t), tan2[i]) < 0.0f) ? -1.0f : 1.0f;
//     }

//     mesh.tangents = tangents;
// }
