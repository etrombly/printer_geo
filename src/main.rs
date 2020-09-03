use printer_geo::{compute::*, geo::*, stl::*};
use rayon::prelude::*;
use std::{fs::File, io::Write};

fn main() {
    // let stl = load_stl("3DBenchy.stl");
    //
    // let triangles = to_triangles3d(&stl);
    // let tri_vk = to_tri_vk(&triangles);
    // let vk = init_vk();
    // let bboxes: Vec<LineVk> = compute_bbox(&tri_vk, &vk);
    // let bboxes2: Vec<Line3d> = triangles.par_iter().map(|x| x.bbox()).collect();
    // println!("{:?}\n{:?}", bboxes[0], bboxes2[0]);
    // println!("{:?}\n{:?}", bboxes[1], bboxes2[1]);
    //
    // for i in 0..30 {
    // let plane = Plane::new((0.0, 0.0, i as f32), (0.0, 0.0, 1.0));
    // let intersects: Vec<Option<Shape>> =
    // triangles.par_iter().map(|x| x.intersect(plane)).collect();
    // let mut lines: Vec<simplesvg::Fig> = Vec::new();
    // let mut max_x = 0.0;
    // let mut max_y = 0.0;
    //
    // for item in &intersects {
    // if let Some(Shape::Line3d(line)) = *item {
    // if line.max_x() - max_x > std::f32::EPSILON {
    // max_x = line.max_x();
    // }
    // if line.max_y() - max_y > std::f32::EPSILON {
    // max_y = line.max_y();
    // }
    // lines.push(
    // simplesvg::Fig::Line(
    // line.p1.x * 100.0,
    // line.p1.y * 100.0,
    // line.p2.x * 100.0,
    // line.p2.y * 100.0,
    // )
    // .styled(
    // simplesvg::Attr::default()
    // .stroke(simplesvg::Color(0xff, 0, 0))
    // .stroke_width(1.0),
    // ),
    // );
    // }
    // }
    //
    // println!(
    //    "total: {} intersecting plane at 1.0: {}",
    //    stl.header.num_triangles,
    //    lines.len()
    // );
    // println!("max_x: {} max_y: {}", max_x, max_y);
    // let mut f = File::create(format!("image{}.svg", i)).expect("Unable to create
    // file"); f.write_all(
    // simplesvg::Svg(
    // lines,
    // max_x.trunc() as u32 * 100,
    // max_y.trunc() as u32 * 100,
    // )
    // .to_string()
    // .as_bytes(),
    // )
    // .unwrap();
    // }
    let line = Line3d::new((0.0, 0.0, 0.0), (1.0, 4.0, 2.0));
    let plane = Plane::new((0.0, 0.0, 1.0), (0.0, 0.0, 2.0));
    let answer = line.intersect(plane);
    println!("intersecting\n    {:?}", answer);
    let line = Line3d::new((0.0, 0.0, 0.0), (2.0, 2.0, 0.0));
    let answer = line.intersect(plane);
    println!("not intersecting\n    {:?}", answer);
    let line = Line3d::new((0.0, 0.0, 1.0), (1.0, 1.0, 1.0));
    let answer = line.intersect(plane);
    println!("on plane\n    {:?}", answer);
    let triangle = Triangle3d::new((0.0, 0.0, 0.0), (2.0, 2.0, 0.0), (1.0, 1.0, 3.0));
    let answer = triangle.intersect(plane);
    println!("triangle intersecting\n    {:?}", answer);
    let triangle = Triangle3d::new((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 0.0, 0.0));
    let answer = triangle.intersect(plane);
    println!("triangle line on plane\n    {:?}", answer);
    let triangle = Triangle3d::new((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0));
    let answer = triangle.intersect(plane);
    println!("triangle not intersecting\n    {:?}", answer);
    let triangle = Triangle3d::new((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0));
    let answer = triangle.intersect(plane);
    println!("triangle on plane\n    {:?}", answer);
}
