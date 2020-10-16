use criterion::{criterion_group, criterion_main, Criterion};
use printer_geo::{geo::*, stl::*};
use rayon::prelude::*;

/*
// TODO: update benchmarks
pub fn criterion_benchmark(c: &mut Criterion) {
    let stl = load_stl("3DBenchy.stl");
    let triangles = to_triangles3d(&stl);
    let tri_vk = to_tri_vk(&triangles);
    let vk = init_vk();
    let mut group = c.benchmark_group("BBox");
    group.bench_function("iter bboxes", |b| {
        b.iter(|| {
            let _bboxes: Vec<Line3d> = triangles.iter().map(|x| x.bbox()).collect();
        })
    });
    group.bench_function("par_iter bboxes", |b| {
        b.iter(|| {
            let _bboxes: Vec<Line3d> = triangles.par_iter().map(|x| x.bbox()).collect();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
*/
