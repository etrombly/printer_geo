extern crate printer_geo;
extern crate byteorder;
extern crate simplesvg;

use std::io::{Result, Write, ErrorKind, Error};
use byteorder::{ReadBytesExt, LittleEndian, WriteBytesExt};
use printer_geo::*;
use std::fs::File;
use std::io::BufReader;

pub struct Triangle {
    normal: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
    v3: [f32; 3],
    attr_byte_count: u16
}

fn point_eq(lhs: [f32; 3], rhs: [f32; 3]) -> bool {
    lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2]
}

impl PartialEq for Triangle {
    fn eq(&self, rhs: &Triangle) -> bool {
        point_eq(self.normal, rhs.normal)
            && point_eq(self.v1, rhs.v1)
            && point_eq(self.v2, rhs.v2)
            && point_eq(self.v3, rhs.v3)
            && self.attr_byte_count == rhs.attr_byte_count
    }
}

impl Eq for Triangle {}

pub struct BinaryStlHeader {
    pub header: [u8; 80],
    pub num_triangles: u32
}

pub struct BinaryStlFile {
    pub header: BinaryStlHeader,
    pub triangles: Vec<Triangle>
}

fn read_point<T: ReadBytesExt>(input: &mut T) -> Result<[f32; 3]> {
    let x1 = try!(input.read_f32::<LittleEndian>());
    let x2 = try!(input.read_f32::<LittleEndian>());
    let x3 = try!(input.read_f32::<LittleEndian>());

    Ok([x1, x2, x3])
}

fn read_triangle<T: ReadBytesExt>(input: &mut T) -> Result<Triangle> {
    let normal = try!(read_point(input));
    let v1 = try!(read_point(input));
    let v2 = try!(read_point(input));
    let v3 = try!(read_point(input));
    let attr_count = try!(input.read_u16::<LittleEndian>());

    Ok(Triangle { normal: normal,
                  v1: v1, v2: v2, v3: v3,
                  attr_byte_count: attr_count })
}

fn read_header<T: ReadBytesExt>(input: &mut T) -> Result<BinaryStlHeader> {
    let mut header = [0u8; 80];

    match input.read(&mut header) {
        Ok(n) => if n == header.len() {
            ()
        }
        else {
            return Err(Error::new(ErrorKind::Other,
                                  "Couldn't read STL header"));
        },
        Err(e) => return Err(e)
    };

    let num_triangles = try!(input.read_u32::<LittleEndian>());

    Ok(BinaryStlHeader{ header: header, num_triangles: num_triangles })
}

pub fn read_stl<T: ReadBytesExt>(input: &mut T) -> Result<BinaryStlFile> {

    // read the header
    let header = try!(read_header(input));

    let mut triangles = Vec::new();
    for _ in 0 .. header.num_triangles {
        triangles.push(try!(read_triangle(input)));
    }

    Ok(BinaryStlFile {
        header: header,
        triangles: triangles
    })
}

fn write_point<T: WriteBytesExt>(out: &mut T, p: [f32; 3]) -> Result<()> {
    for x in &p {
        try!(out.write_f32::<LittleEndian>(*x));
    }
    Ok(())
}

pub fn write_stl<T: WriteBytesExt>(out: &mut T,
                                   stl: &BinaryStlFile) -> Result<()> {
    assert_eq!(stl.header.num_triangles as usize, stl.triangles.len());

    //write the header.
    try!(out.write_all(&stl.header.header));
    try!(out.write_u32::<LittleEndian>(stl.header.num_triangles));

    // write all the triangles
    for t in &stl.triangles {
        try!(write_point(out, t.normal));
        try!(write_point(out, t.v1));
        try!(write_point(out, t.v2));
        try!(write_point(out, t.v3));
        try!(out.write_u16::<LittleEndian>(t.attr_byte_count));
    }

    Ok(())
}

fn main() {
    let file = File::open("3DBenchy.stl").unwrap();
    let mut buf_reader = BufReader::new(file);
    let stl = read_stl(&mut buf_reader).unwrap();
    let mut triangles = Vec::new();
    for i in 0..stl.header.num_triangles - 1{
        let i = i as usize;
        triangles.push(Triangle3d::new(
            (stl.triangles[i].v1[0], stl.triangles[i].v1[1], stl.triangles[i].v1[2]),
            (stl.triangles[i].v2[0], stl.triangles[i].v2[1], stl.triangles[i].v2[2]),
            (stl.triangles[i].v3[0], stl.triangles[i].v3[1], stl.triangles[i].v3[2]),
        ));
    }

for i in 0..30{
    let plane = Plane::new((0.0, 0.0, i as f32), (0.0, 0.0, 1.0));
    let intersects: Vec<Option<Shape>> = triangles.iter().map(|x| x.intersect(plane)).collect();
    let mut lines: Vec<simplesvg::Fig> = Vec::new();
    let mut max_x = 0.0;
    let mut max_y = 0.0;

    for item in &intersects{
        if let Some(Shape::Line3d(line)) = *item {
          if line.max_x() - max_x > std::f32::EPSILON {
            max_x = line.max_x();
          }
          if line.max_y() - max_y > std::f32::EPSILON {
            max_y = line.max_y();
          }
            lines.push(simplesvg::Fig::Line(line.p1.x * 100.0, line.p1.y * 100.0,
                                            line.p2.x * 100.0, line.p2.y * 100.0)
                                           .styled(simplesvg::Attr::default()
                                           .stroke(simplesvg::Color(0xff, 0, 0))
                                           .stroke_width(1.0)));
        }
    };

    println!("total: {} intersecting plane at 1.0: {}", stl.header.num_triangles, lines.len());
    println!("max_x: {} max_y: {}", max_x, max_y);
    let mut f = File::create(format!("image{}.svg", i)).expect("Unable to create file");
    f.write_all(simplesvg::Svg(lines, max_x.trunc() as u32 * 100, max_y.trunc() as u32 * 100).to_string().as_bytes()).unwrap();
}

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
