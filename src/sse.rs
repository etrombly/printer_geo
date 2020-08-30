use crate::geo::*;
use rayon::prelude::*;
use std::arch::x86_64::*;

pub fn tri_bbox_sse(tris: &[Triangle3d]) -> Vec<Line3d> {
    let mut results: Vec<Line3d> = Vec::with_capacity(tris.len());
    let tri_chunks = tris.chunks_exact(4);
    let remainder = tri_chunks.remainder();
    for tri4 in tri_chunks {
        unsafe {
            let x1 = _mm_set_ps(
                tri4[3].p1.x,
                tri4[2].p1.x,
                tri4[1].p1.x,
                tri4[0].p1.x,
            );
            let x2 = _mm_set_ps(
                tri4[3].p2.x,
                tri4[2].p2.x,
                tri4[1].p2.x,
                tri4[0].p2.x,
            );
            let x3 = _mm_set_ps(
                tri4[3].p3.x,
                tri4[2].p3.x,
                tri4[1].p3.x,
                tri4[0].p3.x,
            );
            let y1 = _mm_set_ps(
                tri4[3].p1.y,
                tri4[2].p1.y,
                tri4[1].p1.y,
                tri4[0].p1.y,
            );
            let y2 = _mm_set_ps(
                tri4[3].p2.y,
                tri4[2].p2.y,
                tri4[1].p2.y,
                tri4[0].p2.y,
            );
            let y3 = _mm_set_ps(
                tri4[3].p3.y,
                tri4[2].p3.y,
                tri4[1].p3.y,
                tri4[0].p3.y,
            );
            let z1 = _mm_set_ps(
                tri4[3].p1.z,
                tri4[2].p1.z,
                tri4[1].p1.z,
                tri4[0].p1.z,
            );
            let z2 = _mm_set_ps(
                tri4[3].p2.z,
                tri4[2].p2.z,
                tri4[1].p2.z,
                tri4[0].p2.z,
            );
            let z3 = _mm_set_ps(
                tri4[3].p3.z,
                tri4[2].p3.z,
                tri4[1].p3.z,
                tri4[0].p3.z,
            );

            let x_min = _mm_min_ps(_mm_min_ps(x1, x2), x3);
            let y_min = _mm_min_ps(_mm_min_ps(y1, y2), y3);
            let z_min = _mm_min_ps(_mm_min_ps(z1, z2), z3);
            let x_max = _mm_max_ps(_mm_max_ps(x1, x2), x3);
            let y_max = _mm_max_ps(_mm_max_ps(y1, y2), y3);
            let z_max = _mm_max_ps(_mm_max_ps(z1, z2), z3);
            let x_min_dst = std::mem::transmute::<__m128, [f32; 4]>(x_min);
            let y_min_dst = std::mem::transmute::<__m128, [f32; 4]>(y_min);
            let z_min_dst = std::mem::transmute::<__m128, [f32; 4]>(z_min);
            let x_max_dst = std::mem::transmute::<__m128, [f32; 4]>(x_max);
            let y_max_dst = std::mem::transmute::<__m128, [f32; 4]>(y_max);
            let z_max_dst = std::mem::transmute::<__m128, [f32; 4]>(z_max);
            for index in 0..4 {
                results.push(Line3d {
                    p1: Point3d::new(x_min_dst[index], y_min_dst[index], z_min_dst[index]),
                    p2: Point3d::new(x_max_dst[index], y_max_dst[index], z_max_dst[index]),
                });
            }
        }
    }
    for tri in remainder {
        results.push(tri.bbox());
    }
    results
}

pub fn tri_bbox_sse_par(tris: &[Triangle3d]) -> Vec<Line3d> {
    let tri_chunks = tris.par_chunks_exact(4);
    let remainder = tri_chunks.remainder();
    let mut results: Vec<Line3d> = Vec::with_capacity(tris.len() + remainder.len());
    unsafe {
        results.set_len(tris.len());
    }
    tri_chunks
        .zip(results.par_chunks_exact_mut(4))
        .for_each(|(tri4, slice)| unsafe {
            let x1 = _mm_set_ps(
                tri4[3].p1.x,
                tri4[2].p1.x,
                tri4[1].p1.x,
                tri4[0].p1.x,
            );
            let x2 = _mm_set_ps(
                tri4[3].p2.x,
                tri4[2].p2.x,
                tri4[1].p2.x,
                tri4[0].p2.x,
            );
            let x3 = _mm_set_ps(
                tri4[3].p3.x,
                tri4[2].p3.x,
                tri4[1].p3.x,
                tri4[0].p3.x,
            );
            let y1 = _mm_set_ps(
                tri4[3].p1.y,
                tri4[2].p1.y,
                tri4[1].p1.y,
                tri4[0].p1.y,
            );
            let y2 = _mm_set_ps(
                tri4[3].p2.y,
                tri4[2].p2.y,
                tri4[1].p2.y,
                tri4[0].p2.y,
            );
            let y3 = _mm_set_ps(
                tri4[3].p3.y,
                tri4[2].p3.y,
                tri4[1].p3.y,
                tri4[0].p3.y,
            );
            let z1 = _mm_set_ps(
                tri4[3].p1.z,
                tri4[2].p1.z,
                tri4[1].p1.z,
                tri4[0].p1.z,
            );
            let z2 = _mm_set_ps(
                tri4[3].p2.z,
                tri4[2].p2.z,
                tri4[1].p2.z,
                tri4[0].p2.z,
            );
            let z3 = _mm_set_ps(
                tri4[3].p3.z,
                tri4[2].p3.z,
                tri4[1].p3.z,
                tri4[0].p3.z,
            );

            let x_min = _mm_min_ps(_mm_min_ps(x1, x2), x3);
            let y_min = _mm_min_ps(_mm_min_ps(y1, y2), y3);
            let z_min = _mm_min_ps(_mm_min_ps(z1, z2), z3);
            let x_max = _mm_max_ps(_mm_max_ps(x1, x2), x3);
            let y_max = _mm_max_ps(_mm_max_ps(y1, y2), y3);
            let z_max = _mm_max_ps(_mm_max_ps(z1, z2), z3);
            let x_min_dst = std::mem::transmute::<__m128, [f32; 4]>(x_min);
            let y_min_dst = std::mem::transmute::<__m128, [f32; 4]>(y_min);
            let z_min_dst = std::mem::transmute::<__m128, [f32; 4]>(z_min);
            let x_max_dst = std::mem::transmute::<__m128, [f32; 4]>(x_max);
            let y_max_dst = std::mem::transmute::<__m128, [f32; 4]>(y_max);
            let z_max_dst = std::mem::transmute::<__m128, [f32; 4]>(z_max);
            slice[0] = Line3d {
                p1: Point3d::new(x_min_dst[0], y_min_dst[0], z_min_dst[0]),
                p2: Point3d::new(x_max_dst[0], y_max_dst[0], z_max_dst[0]),
            };
            slice[1] = Line3d {
                p1: Point3d::new(x_min_dst[1], y_min_dst[1], z_min_dst[1]),
                p2: Point3d::new(x_max_dst[1], y_max_dst[1], z_max_dst[1]),
            };
            slice[2] = Line3d {
                p1: Point3d::new(x_min_dst[2], y_min_dst[2], z_min_dst[2]),
                p2: Point3d::new(x_max_dst[2], y_max_dst[2], z_max_dst[2]),
            };
            slice[3] = Line3d {
                p1: Point3d::new(x_min_dst[3], y_min_dst[3], z_min_dst[3]),
                p2: Point3d::new(x_max_dst[3], y_max_dst[3], z_max_dst[3]),
            };
        });
    for tri in remainder {
        results.push(tri.bbox());
    }

    results
}