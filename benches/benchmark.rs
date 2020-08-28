use criterion::{criterion_group, criterion_main, Criterion};
use printer_geo::{compute::*, geo::*, simd::*, util::*};
use rayon::prelude::*;

pub fn criterion_benchmark(c: &mut Criterion) {
    let stl = load_stl("3DBenchy.stl");
    let triangles = to_triangles3d(&stl);
    let (trix8, rem) = to_trix8(&triangles);
    let tri_vk = to_tri_vk(&triangles);
    let vk = init_vk();
    c.bench_function("compute bboxes", |b| {
        b.iter(|| {
            let _bboxes: Vec<LineVk> = compute_bbox(&tri_vk, &vk);
        })
    });
    c.bench_function("iter bboxes", |b| {
        b.iter(|| {
            let _bboxes: Vec<Line3d> = triangles.iter().map(|x| x.bbox()).collect();
        })
    });
    c.bench_function("par_iter bboxes", |b| {
        b.iter(|| {
            let _bboxes: Vec<Line3d> = triangles.par_iter().map(|x| x.bbox()).collect();
        })
    });
    c.bench_function("simd bboxes", |b| {
        b.iter(|| {
            let _bboxes: Vec<Line3d> = tri_bbox_simd(&triangles);
        })
    });
    c.bench_function("simd par bboxes", |b| {
        b.iter(|| {
            let _bboxes: Vec<Line3d> = tri_bbox_simd_par(&triangles);
        })
    });
    c.bench_function("simd trix8 bboxes", |b| {
        b.iter(|| {
            let _bboxes: Vec<Line3d> = tri_bbox_trix(&trix8, &rem);
        })
    });
    c.bench_function("simd par trix8 bboxes", |b| {
        b.iter(|| {
            let _bboxes: Vec<Line3d> = tri_bbox_trix_par(&trix8, &rem);
        })
    });
    c.bench_function("simd par trix8 linex8 bboxes", |b| {
        b.iter(|| {
            let _bboxes: (Vec<Line3dx8>, Vec<Line3d>) = tri_bbox_trix_par_line(&trix8, &rem);
        })
    });
    // TODO: move this check to tests
    // let bboxes: Vec<Line3d> = triangles.iter().map(|x| x.bbox()).collect();
    // let bboxes2: Vec<Line3d> = triangles.par_iter().map(|x|
    // x.bbox()).collect(); let bboxes3: Vec<Line3d> =
    // tri_bbox_simd(&triangles); let bboxes4: Vec<Line3d> =
    // tri_bbox_simd_par(&triangles); assert_eq!(bboxes[0], bboxes2[0]);
    // println!("iter and par iter are same");
    // assert_eq!(bboxes[0], bboxes3[0]);
    // println!("iter and simd are same");
    // assert_eq!(bboxes[0], bboxes4[0]);
    // println!("iter and par simd are same");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
