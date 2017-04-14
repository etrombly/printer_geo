extern crate printer_geo;

use printer_geo::*;

fn main() {
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
    let triangle = Triangle3d::new((0.0, 0.0, 0.0),
                                 (2.0, 2.0, 0.0),
                                 (1.0, 1.0, 3.0));
    let answer = triangle.intersect(plane);
    println!("triangle intersecting\n    {:?}", answer);
    let triangle = Triangle3d::new((0.0, 0.0, 1.0),
                                 (1.0, 0.0, 1.0),
                                 (0.0, 0.0, 0.0));
    let answer = triangle.intersect(plane);
    println!("triangle line on plane\n    {:?}", answer);
    let triangle = Triangle3d::new((0.0, 0.0, 0.0),
                                 (1.0, 0.0, 0.0),
                                 (0.0, 1.0, 0.0));
    let answer = triangle.intersect(plane);
    println!("triangle not intersecting\n    {:?}", answer);
    let triangle = Triangle3d::new((0.0, 0.0, 1.0),
                                 (1.0, 0.0, 1.0),
                                 (0.0, 1.0, 1.0));
    let answer = triangle.intersect(plane);
    println!("triangle on plane\n    {:?}", answer);
}
