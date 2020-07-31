use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use packed_simd::f32x4;
use printer_geo::*;
use rayon::prelude::*;
use std::fs::File;
use std::io::BufReader;
use std::io::{Error, ErrorKind, Result, Write};

pub struct Triangle {
    normal: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
    v3: [f32; 3],
    attr_byte_count: u16,
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
    pub num_triangles: u32,
}

pub struct BinaryStlFile {
    pub header: BinaryStlHeader,
    pub triangles: Vec<Triangle>,
}

fn read_point<T: ReadBytesExt>(input: &mut T) -> Result<[f32; 3]> {
    let x1 = input.read_f32::<LittleEndian>()?;
    let x2 = input.read_f32::<LittleEndian>()?;
    let x3 = input.read_f32::<LittleEndian>()?;

    Ok([x1, x2, x3])
}

fn read_triangle<T: ReadBytesExt>(input: &mut T) -> Result<Triangle> {
    let normal = read_point(input)?;
    let v1 = read_point(input)?;
    let v2 = read_point(input)?;
    let v3 = read_point(input)?;
    let attr_count = input.read_u16::<LittleEndian>()?;

    Ok(Triangle {
        normal,
        v1,
        v2,
        v3,
        attr_byte_count: attr_count,
    })
}

fn read_header<T: ReadBytesExt>(input: &mut T) -> Result<BinaryStlHeader> {
    let mut header = [0u8; 80];

    match input.read(&mut header) {
        Ok(n) => {
            if n == header.len() {
                ()
            } else {
                return Err(Error::new(ErrorKind::Other, "Couldn't read STL header"));
            }
        }
        Err(e) => return Err(e),
    };

    let num_triangles = input.read_u32::<LittleEndian>()?;

    Ok(BinaryStlHeader {
        header,
        num_triangles,
    })
}

pub fn read_stl<T: ReadBytesExt>(input: &mut T) -> Result<BinaryStlFile> {
    // read the header
    let header = read_header(input)?;

    let mut triangles = Vec::new();
    for _ in 0..header.num_triangles {
        triangles.push(read_triangle(input)?);
    }

    Ok(BinaryStlFile { header, triangles })
}

fn write_point<T: WriteBytesExt>(out: &mut T, p: [f32; 3]) -> Result<()> {
    for x in &p {
        out.write_f32::<LittleEndian>(*x)?;
    }
    Ok(())
}

pub fn write_stl<T: WriteBytesExt>(out: &mut T, stl: &BinaryStlFile) -> Result<()> {
    assert_eq!(stl.header.num_triangles as usize, stl.triangles.len());

    //write the header.
    out.write_all(&stl.header.header)?;
    out.write_u32::<LittleEndian>(stl.header.num_triangles)?;

    // write all the triangles
    for t in &stl.triangles {
        write_point(out, t.normal)?;
        write_point(out, t.v1)?;
        write_point(out, t.v2)?;
        write_point(out, t.v3)?;
        out.write_u16::<LittleEndian>(t.attr_byte_count)?;
    }

    Ok(())
}

fn main() {
    let file = File::open("3DBenchy.stl").unwrap();
    let mut buf_reader = BufReader::new(file);
    let stl = read_stl(&mut buf_reader).unwrap();

    let mut triangles = Vec::new();
    for i in 0..stl.header.num_triangles - 1 {
        let i = i as usize;
        triangles.push(Triangle3d::new(
            (
                stl.triangles[i].v1[0],
                stl.triangles[i].v1[1],
                stl.triangles[i].v1[2],
            ),
            (
                stl.triangles[i].v2[0],
                stl.triangles[i].v2[1],
                stl.triangles[i].v2[2],
            ),
            (
                stl.triangles[i].v3[0],
                stl.triangles[i].v3[1],
                stl.triangles[i].v3[2],
            ),
        ));
    }

    let mut trianglesx4: Vec<Triangle3dx4> = Vec::new();

    let chunks = stl.triangles.chunks_exact(4);
    for chunk in chunks {
        let v1 = Point3dx4::new(
            f32x4::new(
                chunk[0].v1[0],
                chunk[1].v1[0],
                chunk[2].v1[0],
                chunk[3].v1[0],
            ),
            f32x4::new(
                chunk[0].v1[1],
                chunk[1].v1[1],
                chunk[2].v1[1],
                chunk[3].v1[1],
            ),
            f32x4::new(
                chunk[0].v1[2],
                chunk[1].v1[2],
                chunk[2].v1[2],
                chunk[3].v1[2],
            ),
        );

        let v2 = Point3dx4::new(
            f32x4::new(
                chunk[0].v2[0],
                chunk[1].v2[0],
                chunk[2].v2[0],
                chunk[3].v2[0],
            ),
            f32x4::new(
                chunk[0].v2[1],
                chunk[1].v2[1],
                chunk[2].v2[1],
                chunk[3].v2[1],
            ),
            f32x4::new(
                chunk[0].v2[2],
                chunk[1].v2[2],
                chunk[2].v2[2],
                chunk[3].v2[2],
            ),
        );

        let v3 = Point3dx4::new(
            f32x4::new(
                chunk[0].v3[0],
                chunk[1].v3[0],
                chunk[2].v3[0],
                chunk[3].v3[0],
            ),
            f32x4::new(
                chunk[0].v3[1],
                chunk[1].v3[1],
                chunk[2].v3[1],
                chunk[3].v3[1],
            ),
            f32x4::new(
                chunk[0].v3[2],
                chunk[1].v3[2],
                chunk[2].v3[2],
                chunk[3].v3[2],
            ),
        );

        trianglesx4.push(Triangle3dx4 {
            p1: v1,
            p2: v2,
            p3: v3,
        });
    }

    /*
        let mut trianglesx8: Vec<Triangle3dx8> = Vec::new();

        let chunks = stl.triangles.chunks_exact(8);
        for chunk in chunks {
            let v1 = Point3dx8::new(
                    [chunk[0].v1[0],
                    chunk[1].v1[0],
                    chunk[2].v1[0],
                    chunk[3].v1[0],
                    chunk[4].v1[0],
                    chunk[5].v1[0],
                    chunk[6].v1[0],
                    chunk[7].v1[0],],
                    [chunk[0].v1[1],
                    chunk[1].v1[1],
                    chunk[2].v1[1],
                    chunk[3].v1[1],
                    chunk[4].v1[1],
                    chunk[5].v1[1],
                    chunk[6].v1[1],
                    chunk[7].v1[1],],
                    [chunk[0].v1[2],
                    chunk[1].v1[2],
                    chunk[2].v1[2],
                    chunk[3].v1[2],
                    chunk[4].v1[2],
                    chunk[5].v1[2],
                    chunk[6].v1[2],
                    chunk[7].v1[2],]
            );

            let v2 = Point3dx8::new(
                [chunk[0].v2[0],
                chunk[1].v2[0],
                chunk[2].v2[0],
                chunk[3].v2[0],
                chunk[4].v2[0],
                chunk[5].v2[0],
                chunk[6].v2[0],
                chunk[7].v2[0],],
                [chunk[0].v2[1],
                chunk[1].v2[1],
                chunk[2].v2[1],
                chunk[3].v2[1],
                chunk[4].v2[1],
                chunk[5].v2[1],
                chunk[6].v2[1],
                chunk[7].v2[1],],
                [chunk[0].v2[2],
                chunk[1].v2[2],
                chunk[2].v2[2],
                chunk[3].v2[2],
                chunk[4].v2[2],
                chunk[5].v2[2],
                chunk[6].v2[2],
                chunk[7].v2[2],]
        );

        let v3 = Point3dx8::new(
            [chunk[0].v3[0],
            chunk[1].v3[0],
            chunk[2].v3[0],
            chunk[3].v3[0],
            chunk[4].v3[0],
            chunk[5].v3[0],
            chunk[6].v3[0],
            chunk[7].v3[0],],
            [chunk[0].v3[1],
            chunk[1].v3[1],
            chunk[2].v3[1],
            chunk[3].v3[1],
            chunk[4].v3[1],
            chunk[5].v3[1],
            chunk[6].v3[1],
            chunk[7].v3[1],],
            [chunk[0].v3[2],
            chunk[1].v3[2],
            chunk[2].v3[2],
            chunk[3].v3[2],
            chunk[4].v3[2],
            chunk[5].v3[2],
            chunk[6].v3[2],
            chunk[7].v3[2],]
    );

            trianglesx8.push(Triangle3dx8 {
                p1: v1,
                p2: v2,
                p3: v3,
            });
        }

        for i in 0..30 {
            let plane = Plane::new((0.0, 0.0, i as f32), (0.0, 0.0, 1.0));
            let intersects: Vec<Option<Shape>> = trianglesx8
                .iter()
                .flat_map(|x| x.intersect(plane))
                .collect();
            let mut lines: Vec<simplesvg::Fig> = Vec::new();
            let mut max_x = 0.0;
            let mut max_y = 0.0;

            for item in &intersects {
                if let Some(Shape::Line3d(line)) = *item {
                    if line.max_x() - max_x > std::f32::EPSILON {
                        max_x = line.max_x();
                    }
                    if line.max_y() - max_y > std::f32::EPSILON {
                        max_y = line.max_y();
                    }
                    lines.push(
                        simplesvg::Fig::Line(
                            line.p1.x * 100.0,
                            line.p1.y * 100.0,
                            line.p2.x * 100.0,
                            line.p2.y * 100.0,
                        )
                        .styled(
                            simplesvg::Attr::default()
                                .stroke(simplesvg::Color(0xff, 0, 0))
                                .stroke_width(1.0),
                        ),
                    );
                }
            }

            //println!(
            //    "total: {} intersecting plane at 1.0: {}",
            //    stl.header.num_triangles,
            //    lines.len()
            //);
            //println!("max_x: {} max_y: {}", max_x, max_y);
            let mut f = File::create(format!("image_simd{}.svg", i)).expect("Unable to create file");
            f.write_all(
                simplesvg::Svg(
                    lines,
                    max_x.trunc() as u32 * 100,
                    max_y.trunc() as u32 * 100,
                )
                .to_string()
                .as_bytes(),
            )
            .unwrap();
        }
    */
    //todo: use chunks.remainder()

    for i in 0..30 {
        let plane = Plane::new((0.0, 0.0, i as f32), (0.0, 0.0, 1.0));
        let intersects: Vec<Option<Shape>> = trianglesx4
            .iter()
            .flat_map(|x| x.intersect(plane))
            .collect();
        let mut lines: Vec<simplesvg::Fig> = Vec::new();
        let mut max_x = 0.0;
        let mut max_y = 0.0;

        for item in &intersects {
            if let Some(Shape::Line3d(line)) = *item {
                if line.max_x() - max_x > std::f32::EPSILON {
                    max_x = line.max_x();
                }
                if line.max_y() - max_y > std::f32::EPSILON {
                    max_y = line.max_y();
                }
                lines.push(
                    simplesvg::Fig::Line(
                        line.p1.x * 100.0,
                        line.p1.y * 100.0,
                        line.p2.x * 100.0,
                        line.p2.y * 100.0,
                    )
                    .styled(
                        simplesvg::Attr::default()
                            .stroke(simplesvg::Color(0xff, 0, 0))
                            .stroke_width(1.0),
                    ),
                );
            }
        }

        //println!(
        //    "total: {} intersecting plane at 1.0: {}",
        //    stl.header.num_triangles,
        //    lines.len()
        //);
        //println!("max_x: {} max_y: {}", max_x, max_y);
        let mut f = File::create(format!("image_simd{}.svg", i)).expect("Unable to create file");
        f.write_all(
            simplesvg::Svg(
                lines,
                max_x.trunc() as u32 * 100,
                max_y.trunc() as u32 * 100,
            )
            .to_string()
            .as_bytes(),
        )
        .unwrap();
    }

    /*
    for i in 0..30 {
        let plane = Plane::new((0.0, 0.0, i as f32), (0.0, 0.0, 1.0));
        let intersects: Vec<Option<Shape>> =
            triangles.par_iter().map(|x| x.intersect(plane)).collect();
        let mut lines: Vec<simplesvg::Fig> = Vec::new();
        let mut max_x = 0.0;
        let mut max_y = 0.0;

        for item in &intersects {
            if let Some(Shape::Line3d(line)) = *item {
                if line.max_x() - max_x > std::f32::EPSILON {
                    max_x = line.max_x();
                }
                if line.max_y() - max_y > std::f32::EPSILON {
                    max_y = line.max_y();
                }
                lines.push(
                    simplesvg::Fig::Line(
                        line.p1.x * 100.0,
                        line.p1.y * 100.0,
                        line.p2.x * 100.0,
                        line.p2.y * 100.0,
                    )
                    .styled(
                        simplesvg::Attr::default()
                            .stroke(simplesvg::Color(0xff, 0, 0))
                            .stroke_width(1.0),
                    ),
                );
            }
        }

        //println!(
        //    "total: {} intersecting plane at 1.0: {}",
        //    stl.header.num_triangles,
        //    lines.len()
        //);
        //println!("max_x: {} max_y: {}", max_x, max_y);
        let mut f = File::create(format!("image{}.svg", i)).expect("Unable to create file");
        f.write_all(
            simplesvg::Svg(
                lines,
                max_x.trunc() as u32 * 100,
                max_y.trunc() as u32 * 100,
            )
            .to_string()
            .as_bytes(),
        )
        .unwrap();
    }
    */
    /*
    let line = Line3d::new((0.0, 0.0, 0.0), (1.0, 4.0, 2.0));
    let plane = Plane::new((0.0, 0.0, 1.0), (0.0, 0.0, 2.0));
    let answer = line.intersect(plane);
    //println!("intersecting\n    {:?}", answer);
    let line = Line3d::new((0.0, 0.0, 0.0), (2.0, 2.0, 0.0));
    let answer = line.intersect(plane);
    //println!("not intersecting\n    {:?}", answer);
    let line = Line3d::new((0.0, 0.0, 1.0), (1.0, 1.0, 1.0));
    let answer = line.intersect(plane);
    //println!("on plane\n    {:?}", answer);
    let triangle = Triangle3d::new((0.0, 0.0, 0.0), (2.0, 2.0, 0.0), (1.0, 1.0, 3.0));
    let answer = triangle.intersect(plane);
    //println!("triangle intersecting\n    {:?}", answer);
    let triangle = Triangle3d::new((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 0.0, 0.0));
    let answer = triangle.intersect(plane);
    //println!("triangle line on plane\n    {:?}", answer);
    let triangle = Triangle3d::new((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0));
    let answer = triangle.intersect(plane);
    //println!("triangle not intersecting\n    {:?}", answer);
    let triangle = Triangle3d::new((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0));
    let answer = triangle.intersect(plane);
    //println!("triangle on plane\n    {:?}", answer);
    */
}
