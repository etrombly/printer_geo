use crate::geo::*;
use rayon::prelude::*;
use std::arch::x86_64::*;

pub struct Point3dx8 {
    pub x: __m256,
    pub y: __m256,
    pub z: __m256,
}

pub struct Line3dx8 {
    pub p1: Point3dx8,
    pub p2: Point3dx8,
}
pub struct Triangle3dx8 {
    pub p1: Point3dx8,
    pub p2: Point3dx8,
    pub p3: Point3dx8,
}

pub fn tri_bbox_simd(tris: &[Triangle3d]) -> Vec<Line3d> {
    let mut results: Vec<Line3d> = Vec::with_capacity(tris.len());
    let tri_chunks = tris.chunks_exact(8);
    let remainder = tri_chunks.remainder();
    for tri8 in tri_chunks {
        unsafe {
            let x1 = _mm256_set_ps(
                tri8[7].p1.x,
                tri8[6].p1.x,
                tri8[5].p1.x,
                tri8[4].p1.x,
                tri8[3].p1.x,
                tri8[2].p1.x,
                tri8[1].p1.x,
                tri8[0].p1.x,
            );
            let x2 = _mm256_set_ps(
                tri8[7].p2.x,
                tri8[6].p2.x,
                tri8[5].p2.x,
                tri8[4].p2.x,
                tri8[3].p2.x,
                tri8[2].p2.x,
                tri8[1].p2.x,
                tri8[0].p2.x,
            );
            let x3 = _mm256_set_ps(
                tri8[7].p3.x,
                tri8[6].p3.x,
                tri8[5].p3.x,
                tri8[4].p3.x,
                tri8[3].p3.x,
                tri8[2].p3.x,
                tri8[1].p3.x,
                tri8[0].p3.x,
            );
            let y1 = _mm256_set_ps(
                tri8[7].p1.y,
                tri8[6].p1.y,
                tri8[5].p1.y,
                tri8[4].p1.y,
                tri8[3].p1.y,
                tri8[2].p1.y,
                tri8[1].p1.y,
                tri8[0].p1.y,
            );
            let y2 = _mm256_set_ps(
                tri8[7].p2.y,
                tri8[6].p2.y,
                tri8[5].p2.y,
                tri8[4].p2.y,
                tri8[3].p2.y,
                tri8[2].p2.y,
                tri8[1].p2.y,
                tri8[7].p2.y,
            );
            let y3 = _mm256_set_ps(
                tri8[7].p3.y,
                tri8[6].p3.y,
                tri8[5].p3.y,
                tri8[4].p3.y,
                tri8[3].p3.y,
                tri8[2].p3.y,
                tri8[1].p3.y,
                tri8[0].p3.y,
            );
            let z1 = _mm256_set_ps(
                tri8[7].p1.z,
                tri8[6].p1.z,
                tri8[5].p1.z,
                tri8[4].p1.z,
                tri8[3].p1.z,
                tri8[2].p1.z,
                tri8[1].p1.z,
                tri8[0].p1.z,
            );
            let z2 = _mm256_set_ps(
                tri8[7].p2.z,
                tri8[6].p2.z,
                tri8[5].p2.z,
                tri8[4].p2.z,
                tri8[3].p2.z,
                tri8[2].p2.z,
                tri8[1].p2.z,
                tri8[0].p2.z,
            );
            let z3 = _mm256_set_ps(
                tri8[7].p3.z,
                tri8[6].p3.z,
                tri8[5].p3.z,
                tri8[4].p3.z,
                tri8[3].p3.z,
                tri8[2].p3.z,
                tri8[1].p3.z,
                tri8[0].p3.z,
            );

            let x_min = _mm256_min_ps(_mm256_min_ps(x1, x2), x3);
            let y_min = _mm256_min_ps(_mm256_min_ps(y1, y2), y3);
            let z_min = _mm256_min_ps(_mm256_min_ps(z1, z2), z3);
            let x_max = _mm256_max_ps(_mm256_max_ps(x1, x2), x3);
            let y_max = _mm256_max_ps(_mm256_max_ps(y1, y2), y3);
            let z_max = _mm256_max_ps(_mm256_max_ps(z1, z2), z3);
            let x_min_dst = std::mem::transmute::<__m256, [f32; 8]>(x_min);
            let y_min_dst = std::mem::transmute::<__m256, [f32; 8]>(y_min);
            let z_min_dst = std::mem::transmute::<__m256, [f32; 8]>(z_min);
            let x_max_dst = std::mem::transmute::<__m256, [f32; 8]>(x_max);
            let y_max_dst = std::mem::transmute::<__m256, [f32; 8]>(y_max);
            let z_max_dst = std::mem::transmute::<__m256, [f32; 8]>(z_max);
            for index in 0..8 {
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

pub fn tri_bbox_simd_par(tris: &[Triangle3d]) -> Vec<Line3d> {
    let tri_chunks = tris.par_chunks_exact(8);
    let remainder = tri_chunks.remainder();
    let mut results: Vec<Line3d> = Vec::with_capacity(tris.len() + remainder.len());
    unsafe {
        results.set_len(tris.len());
    }
    tri_chunks
        .zip(results.par_chunks_exact_mut(8))
        .for_each(|(tri8, slice)| unsafe {
            let x1 = _mm256_set_ps(
                tri8[7].p1.x,
                tri8[6].p1.x,
                tri8[5].p1.x,
                tri8[4].p1.x,
                tri8[3].p1.x,
                tri8[2].p1.x,
                tri8[1].p1.x,
                tri8[0].p1.x,
            );
            let x2 = _mm256_set_ps(
                tri8[7].p2.x,
                tri8[6].p2.x,
                tri8[5].p2.x,
                tri8[4].p2.x,
                tri8[3].p2.x,
                tri8[2].p2.x,
                tri8[1].p2.x,
                tri8[0].p2.x,
            );
            let x3 = _mm256_set_ps(
                tri8[7].p3.x,
                tri8[6].p3.x,
                tri8[5].p3.x,
                tri8[4].p3.x,
                tri8[3].p3.x,
                tri8[2].p3.x,
                tri8[1].p3.x,
                tri8[0].p3.x,
            );
            let y1 = _mm256_set_ps(
                tri8[7].p1.y,
                tri8[6].p1.y,
                tri8[5].p1.y,
                tri8[4].p1.y,
                tri8[3].p1.y,
                tri8[2].p1.y,
                tri8[1].p1.y,
                tri8[0].p1.y,
            );
            let y2 = _mm256_set_ps(
                tri8[7].p2.y,
                tri8[6].p2.y,
                tri8[5].p2.y,
                tri8[4].p2.y,
                tri8[3].p2.y,
                tri8[2].p2.y,
                tri8[1].p2.y,
                tri8[7].p2.y,
            );
            let y3 = _mm256_set_ps(
                tri8[7].p3.y,
                tri8[6].p3.y,
                tri8[5].p3.y,
                tri8[4].p3.y,
                tri8[3].p3.y,
                tri8[2].p3.y,
                tri8[1].p3.y,
                tri8[0].p3.y,
            );
            let z1 = _mm256_set_ps(
                tri8[7].p1.z,
                tri8[6].p1.z,
                tri8[5].p1.z,
                tri8[4].p1.z,
                tri8[3].p1.z,
                tri8[2].p1.z,
                tri8[1].p1.z,
                tri8[0].p1.z,
            );
            let z2 = _mm256_set_ps(
                tri8[7].p2.z,
                tri8[6].p2.z,
                tri8[5].p2.z,
                tri8[4].p2.z,
                tri8[3].p2.z,
                tri8[2].p2.z,
                tri8[1].p2.z,
                tri8[0].p2.z,
            );
            let z3 = _mm256_set_ps(
                tri8[7].p3.z,
                tri8[6].p3.z,
                tri8[5].p3.z,
                tri8[4].p3.z,
                tri8[3].p3.z,
                tri8[2].p3.z,
                tri8[1].p3.z,
                tri8[0].p3.z,
            );

            let x_min = _mm256_min_ps(_mm256_min_ps(x1, x2), x3);
            let y_min = _mm256_min_ps(_mm256_min_ps(y1, y2), y3);
            let z_min = _mm256_min_ps(_mm256_min_ps(z1, z2), z3);
            let x_max = _mm256_max_ps(_mm256_max_ps(x1, x2), x3);
            let y_max = _mm256_max_ps(_mm256_max_ps(y1, y2), y3);
            let z_max = _mm256_max_ps(_mm256_max_ps(z1, z2), z3);
            let x_min_dst = std::mem::transmute::<__m256, [f32; 8]>(x_min);
            let y_min_dst = std::mem::transmute::<__m256, [f32; 8]>(y_min);
            let z_min_dst = std::mem::transmute::<__m256, [f32; 8]>(z_min);
            let x_max_dst = std::mem::transmute::<__m256, [f32; 8]>(x_max);
            let y_max_dst = std::mem::transmute::<__m256, [f32; 8]>(y_max);
            let z_max_dst = std::mem::transmute::<__m256, [f32; 8]>(z_max);
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
            slice[4] = Line3d {
                p1: Point3d::new(x_min_dst[4], y_min_dst[4], z_min_dst[4]),
                p2: Point3d::new(x_max_dst[4], y_max_dst[4], z_max_dst[4]),
            };
            slice[5] = Line3d {
                p1: Point3d::new(x_min_dst[5], y_min_dst[5], z_min_dst[5]),
                p2: Point3d::new(x_max_dst[5], y_max_dst[5], z_max_dst[5]),
            };
            slice[6] = Line3d {
                p1: Point3d::new(x_min_dst[6], y_min_dst[6], z_min_dst[6]),
                p2: Point3d::new(x_max_dst[6], y_max_dst[6], z_max_dst[6]),
            };
            slice[7] = Line3d {
                p1: Point3d::new(x_min_dst[7], y_min_dst[7], z_min_dst[7]),
                p2: Point3d::new(x_max_dst[7], y_max_dst[7], z_max_dst[7]),
            };
        });
    for tri in remainder {
        results.push(tri.bbox());
    }

    results
}

pub fn tri_bbox_trix(tris: &[Triangle3dx8], rem: &[Triangle3d]) -> Vec<Line3d> {
    let mut results: Vec<Line3d> = Vec::with_capacity((tris.len() * 8) + rem.len());
    for tri8 in tris {
        unsafe {
            let x_min = _mm256_min_ps(_mm256_min_ps(tri8.p1.x, tri8.p2.x), tri8.p3.x);
            let y_min = _mm256_min_ps(_mm256_min_ps(tri8.p1.y, tri8.p2.y), tri8.p3.y);
            let z_min = _mm256_min_ps(_mm256_min_ps(tri8.p1.z, tri8.p2.z), tri8.p3.z);
            let x_max = _mm256_max_ps(_mm256_max_ps(tri8.p1.x, tri8.p2.x), tri8.p3.x);
            let y_max = _mm256_max_ps(_mm256_max_ps(tri8.p1.y, tri8.p2.y), tri8.p3.y);
            let z_max = _mm256_max_ps(_mm256_max_ps(tri8.p1.z, tri8.p2.z), tri8.p3.z);
            let x_min_dst = std::mem::transmute::<__m256, [f32; 8]>(x_min);
            let y_min_dst = std::mem::transmute::<__m256, [f32; 8]>(y_min);
            let z_min_dst = std::mem::transmute::<__m256, [f32; 8]>(z_min);
            let x_max_dst = std::mem::transmute::<__m256, [f32; 8]>(x_max);
            let y_max_dst = std::mem::transmute::<__m256, [f32; 8]>(y_max);
            let z_max_dst = std::mem::transmute::<__m256, [f32; 8]>(z_max);
            for index in 0..8 {
                results.push(Line3d {
                    p1: Point3d::new(x_min_dst[index], y_min_dst[index], z_min_dst[index]),
                    p2: Point3d::new(x_max_dst[index], y_max_dst[index], z_max_dst[index]),
                });
            }
        }
    }
    for tri in rem {
        results.push(tri.bbox());
    }
    results
}

pub fn tri_bbox_trix_par(tris: &[Triangle3dx8], rem: &[Triangle3d]) -> Vec<Line3d> {
    let mut results: Vec<Line3d> = Vec::with_capacity((tris.len() * 8) + rem.len());
    unsafe {
        results.set_len(tris.len() * 8);
    }
    tris.par_iter()
        .zip(results.par_chunks_exact_mut(8))
        .for_each(|(tri8, slice)| unsafe {
            let x_min = _mm256_min_ps(_mm256_min_ps(tri8.p1.x, tri8.p2.x), tri8.p3.x);
            let y_min = _mm256_min_ps(_mm256_min_ps(tri8.p1.y, tri8.p2.y), tri8.p3.y);
            let z_min = _mm256_min_ps(_mm256_min_ps(tri8.p1.z, tri8.p2.z), tri8.p3.z);
            let x_max = _mm256_max_ps(_mm256_max_ps(tri8.p1.x, tri8.p2.x), tri8.p3.x);
            let y_max = _mm256_max_ps(_mm256_max_ps(tri8.p1.y, tri8.p2.y), tri8.p3.y);
            let z_max = _mm256_max_ps(_mm256_max_ps(tri8.p1.z, tri8.p2.z), tri8.p3.z);
            let x_min_dst = std::mem::transmute::<__m256, [f32; 8]>(x_min);
            let y_min_dst = std::mem::transmute::<__m256, [f32; 8]>(y_min);
            let z_min_dst = std::mem::transmute::<__m256, [f32; 8]>(z_min);
            let x_max_dst = std::mem::transmute::<__m256, [f32; 8]>(x_max);
            let y_max_dst = std::mem::transmute::<__m256, [f32; 8]>(y_max);
            let z_max_dst = std::mem::transmute::<__m256, [f32; 8]>(z_max);
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
            slice[4] = Line3d {
                p1: Point3d::new(x_min_dst[4], y_min_dst[4], z_min_dst[4]),
                p2: Point3d::new(x_max_dst[4], y_max_dst[4], z_max_dst[4]),
            };
            slice[5] = Line3d {
                p1: Point3d::new(x_min_dst[5], y_min_dst[5], z_min_dst[5]),
                p2: Point3d::new(x_max_dst[5], y_max_dst[5], z_max_dst[5]),
            };
            slice[6] = Line3d {
                p1: Point3d::new(x_min_dst[6], y_min_dst[6], z_min_dst[6]),
                p2: Point3d::new(x_max_dst[6], y_max_dst[6], z_max_dst[6]),
            };
            slice[7] = Line3d {
                p1: Point3d::new(x_min_dst[7], y_min_dst[7], z_min_dst[7]),
                p2: Point3d::new(x_max_dst[7], y_max_dst[7], z_max_dst[7]),
            };
        });
    for tri in rem {
        results.push(tri.bbox());
    }
    results
}

pub fn tri_bbox_trix_par_line(
    tris: &[Triangle3dx8],
    rem: &[Triangle3d],
) -> (Vec<Line3dx8>, Vec<Line3d>) {
    let results: Vec<Line3dx8> = 
    tris.par_iter()
        .map(|tri8| unsafe {
            let x_min = _mm256_min_ps(_mm256_min_ps(tri8.p1.x, tri8.p2.x), tri8.p3.x);
            let y_min = _mm256_min_ps(_mm256_min_ps(tri8.p1.y, tri8.p2.y), tri8.p3.y);
            let z_min = _mm256_min_ps(_mm256_min_ps(tri8.p1.z, tri8.p2.z), tri8.p3.z);
            let x_max = _mm256_max_ps(_mm256_max_ps(tri8.p1.x, tri8.p2.x), tri8.p3.x);
            let y_max = _mm256_max_ps(_mm256_max_ps(tri8.p1.y, tri8.p2.y), tri8.p3.y);
            let z_max = _mm256_max_ps(_mm256_max_ps(tri8.p1.z, tri8.p2.z), tri8.p3.z);
            Line3dx8 {
                p1: Point3dx8 {
                    x: x_min,
                    y: y_min,
                    z: z_min,
                },
                p2: Point3dx8 {
                    x: x_max,
                    y: y_max,
                    z: z_max,
                },
            }
        }).collect();
    (results, rem.par_iter().map(|tri| tri.bbox()).collect())
}

pub fn to_trix8(tris: &[Triangle3d]) -> (Vec<Triangle3dx8>, Vec<Triangle3d>) {
    let tri_chunks = tris.chunks_exact(8);
    let remainder = tri_chunks.remainder().to_vec();
    let result = tri_chunks
        .map(|tri8| unsafe {
            let x1 = _mm256_set_ps(
                tri8[7].p1.x,
                tri8[6].p1.x,
                tri8[5].p1.x,
                tri8[4].p1.x,
                tri8[3].p1.x,
                tri8[2].p1.x,
                tri8[1].p1.x,
                tri8[0].p1.x,
            );
            let x2 = _mm256_set_ps(
                tri8[7].p2.x,
                tri8[6].p2.x,
                tri8[5].p2.x,
                tri8[4].p2.x,
                tri8[3].p2.x,
                tri8[2].p2.x,
                tri8[1].p2.x,
                tri8[0].p2.x,
            );
            let x3 = _mm256_set_ps(
                tri8[7].p3.x,
                tri8[6].p3.x,
                tri8[5].p3.x,
                tri8[4].p3.x,
                tri8[3].p3.x,
                tri8[2].p3.x,
                tri8[1].p3.x,
                tri8[0].p3.x,
            );
            let y1 = _mm256_set_ps(
                tri8[7].p1.y,
                tri8[6].p1.y,
                tri8[5].p1.y,
                tri8[4].p1.y,
                tri8[3].p1.y,
                tri8[2].p1.y,
                tri8[1].p1.y,
                tri8[0].p1.y,
            );
            let y2 = _mm256_set_ps(
                tri8[7].p2.y,
                tri8[6].p2.y,
                tri8[5].p2.y,
                tri8[4].p2.y,
                tri8[3].p2.y,
                tri8[2].p2.y,
                tri8[1].p2.y,
                tri8[7].p2.y,
            );
            let y3 = _mm256_set_ps(
                tri8[7].p3.y,
                tri8[6].p3.y,
                tri8[5].p3.y,
                tri8[4].p3.y,
                tri8[3].p3.y,
                tri8[2].p3.y,
                tri8[1].p3.y,
                tri8[0].p3.y,
            );
            let z1 = _mm256_set_ps(
                tri8[7].p1.z,
                tri8[6].p1.z,
                tri8[5].p1.z,
                tri8[4].p1.z,
                tri8[3].p1.z,
                tri8[2].p1.z,
                tri8[1].p1.z,
                tri8[0].p1.z,
            );
            let z2 = _mm256_set_ps(
                tri8[7].p2.z,
                tri8[6].p2.z,
                tri8[5].p2.z,
                tri8[4].p2.z,
                tri8[3].p2.z,
                tri8[2].p2.z,
                tri8[1].p2.z,
                tri8[0].p2.z,
            );
            let z3 = _mm256_set_ps(
                tri8[7].p3.z,
                tri8[6].p3.z,
                tri8[5].p3.z,
                tri8[4].p3.z,
                tri8[3].p3.z,
                tri8[2].p3.z,
                tri8[1].p3.z,
                tri8[0].p3.z,
            );

            Triangle3dx8 {
                p1: Point3dx8 {
                    x: x1,
                    y: y1,
                    z: z1,
                },
                p2: Point3dx8 {
                    x: x2,
                    y: y2,
                    z: z2,
                },
                p3: Point3dx8 {
                    x: x3,
                    y: y3,
                    z: z3,
                },
            }
        })
        .collect();
    (result, remainder)
}
