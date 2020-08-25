use crate::geo::Triangle3d;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::{
    fs::File,
    io::{BufReader, Error, ErrorKind, Result},
};

pub struct Triangle {
    pub normal: [f32; 3],
    pub v1: [f32; 3],
    pub v2: [f32; 3],
    pub v3: [f32; 3],
    pub attr_byte_count: u16,
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
                return Err(Error::new(
                    ErrorKind::Other,
                    "Couldn't read STL header",
                ));
            }
        },
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

pub fn write_stl<T: WriteBytesExt>(
    out: &mut T,
    stl: &BinaryStlFile,
) -> Result<()> {
    assert_eq!(stl.header.num_triangles as usize, stl.triangles.len());

    // write the header.
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

pub fn load_stl(file: &str) -> BinaryStlFile {
    let file = File::open(file).unwrap();
    let mut buf_reader = BufReader::new(file);
    read_stl(&mut buf_reader).unwrap()
}

pub fn to_triangles3d(file: &BinaryStlFile) -> Vec<Triangle3d> {
    file.triangles
        .iter()
        .map(|x| {
            Triangle3d::new(
                (x.v1[0], x.v1[1], x.v1[2]),
                (x.v2[0], x.v2[1], x.v2[2]),
                (x.v3[0], x.v3[1], x.v3[2]),
            )
        })
        .collect()
}
