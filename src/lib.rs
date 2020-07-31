use packed_simd::f32x4;
use std::cmp::Ordering;
use std::ops::{Add, Index, Mul, Sub};

const PRECISION: f32 = 0.001;

pub trait Intersect<RHS = Self> {
    type Output;
    fn intersect(self, rhs: RHS) -> Self::Output;
}

pub trait Math {
    fn powi(self, n: i32) -> f32x4;
    fn sqrt(self) -> f32x4;
}

impl Math for f32x4 {
    fn powi(self, n: i32) -> f32x4 {
        f32x4::new(
            self.extract(0).powi(n),
            self.extract(1).powi(n),
            self.extract(2).powi(n),
            self.extract(3).powi(n),
        )
    }

    fn sqrt(self) -> f32x4 {
        f32x4::new(
            self.extract(0).sqrt(),
            self.extract(1).sqrt(),
            self.extract(2).sqrt(),
            self.extract(3).sqrt(),
        )
    }
}

pub trait Support<RHS = Self> {
    type Output;
    fn dot(self, rhs: RHS) -> Self::Output;
    fn cross(&self, rhs: &RHS) -> Self;
}

pub trait Bounds {
    fn bbox(self) -> Line3d;
    fn min_x(self) -> f32;
    fn min_y(self) -> f32;
    fn min_z(self) -> f32;
    fn max_x(self) -> f32;
    fn max_y(self) -> f32;
    fn max_z(self) -> f32;
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Point3d {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Point3d {
    pub fn new(x: f32, y: f32, z: f32) -> Point3d {
        Point3d { x, y, z }
    }

    pub fn sum(&self) -> f32 {
        self.x + self.y + self.z
    }

    pub fn is_infinite(&self) -> bool {
        self.x.is_infinite() || self.y.is_infinite() || self.z.is_infinite()
    }

    pub fn is_nan(&self) -> bool {
        !self.is_infinite() && (self.x.is_nan() || self.y.is_nan() || self.z.is_nan())
    }

    pub fn distance(&self, other: &Point3d) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z - other.z).powi(2))
            .sqrt()
    }
}

impl Support<Point3d> for Point3d {
    type Output = f32;

    fn dot(self, other: Point3d) -> f32 {
        //(self * other).sum()
        self.x
            .mul_add(other.x, self.y.mul_add(other.y, self.z * other.z))
    }

    fn cross(&self, other: &Point3d) -> Point3d {
        Point3d::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }
}

impl Add<Point3d> for Point3d {
    type Output = Point3d;

    fn add(self, other: Point3d) -> Point3d {
        Point3d {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Sub<Point3d> for Point3d {
    type Output = Point3d;

    fn sub(self, other: Point3d) -> Point3d {
        Point3d {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Sub<Point3dx4> for Point3d {
    type Output = Point3dx4;

    fn sub(self, other: Point3dx4) -> Point3dx4 {
        Point3dx4 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Sub<Point3dx8> for Point3d {
    type Output = Point3dx8;

    fn sub(self, other: Point3dx8) -> Point3dx8 {
        let mut x = [0_f32; 8];
        let mut y = [0_f32; 8];
        let mut z = [0_f32; 8];
        for i in 0..8 {
            x[i] = self.x - other.x[i];
            y[i] = self.y - other.y[i];
            z[i] = self.z - other.z[i];
        }

        Point3dx8 { x, y, z }
    }
}

impl Mul<Point3d> for Point3d {
    type Output = Point3d;

    fn mul(self, other: Point3d) -> Point3d {
        Point3d {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }
}

impl Mul<f32> for Point3d {
    type Output = Point3d;

    fn mul(self, f: f32) -> Point3d {
        Point3d {
            x: self.x * f,
            y: self.y * f,
            z: self.z * f,
        }
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Line3d {
    pub p1: Point3d,
    pub p2: Point3d,
}

impl Line3d {
    pub fn new(p1: (f32, f32, f32), p2: (f32, f32, f32)) -> Line3d {
        Line3d {
            p1: Point3d::new(p1.0, p1.1, p1.2),
            p2: Point3d::new(p2.0, p2.1, p2.2),
        }
    }

    pub fn from_points(p1: &Point3d, p2: &Point3d) -> Line3d {
        Line3d { p1: *p1, p2: *p2 }
    }

    pub fn on_line(&self, point: &Point3d) -> bool {
        (self.p1.distance(point) + self.p2.distance(point)) - self.p1.distance(&self.p2) < PRECISION
    }
}

impl Intersect<Plane> for Line3d {
    type Output = Option<Shape>;

    fn intersect(self, plane: Plane) -> Option<Shape> {
        let direction = self.p2 - self.p1;
        let orthogonal = plane.n.dot(direction);
        let w = plane.p - self.p1;
        let fac = plane.n.dot(w) / orthogonal;
        let v = direction * fac;
        let answer = self.p1 + v;
        // checking for fac size handles finite lines, may want to change for ray tracing
        if answer.is_infinite() || (fac < 0.0 || fac > 1.0) {
            None
        } else if answer.is_nan() {
            Some(Shape::Line3d(self))
        } else {
            Some(Shape::Point3d(answer))
        }
    }
}

impl Bounds for Line3d {
    fn bbox(self) -> Line3d {
        Line3d {
            p1: Point3d {
                x: self.min_x(),
                y: self.min_y(),
                z: self.min_z(),
            },
            p2: Point3d {
                x: self.max_x(),
                y: self.max_y(),
                z: self.max_z(),
            },
        }
    }

    fn min_x(self) -> f32 {
        if self.p1.x - self.p2.x < std::f32::EPSILON {
            self.p1.x
        } else {
            self.p2.x
        }
    }

    fn min_y(self) -> f32 {
        if self.p1.y - self.p2.y < std::f32::EPSILON {
            self.p1.y
        } else {
            self.p2.y
        }
    }

    fn min_z(self) -> f32 {
        if self.p1.z - self.p2.z < std::f32::EPSILON {
            self.p1.z
        } else {
            self.p2.z
        }
    }

    fn max_x(self) -> f32 {
        if self.p1.x - self.p2.x > std::f32::EPSILON {
            self.p1.x
        } else {
            self.p2.x
        }
    }

    fn max_y(self) -> f32 {
        if self.p1.y - self.p2.y > std::f32::EPSILON {
            self.p1.y
        } else {
            self.p2.y
        }
    }

    fn max_z(self) -> f32 {
        if self.p1.z - self.p2.z > std::f32::EPSILON {
            self.p1.z
        } else {
            self.p2.z
        }
    }
}

impl PartialOrd for Line3d {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.min_z() - other.min_z() < PRECISION {
            Some(Ordering::Less)
        } else if self.min_z() - other.min_z() > PRECISION {
            Some(Ordering::Greater)
        } else if self.min_x() - other.min_x() < PRECISION {
            Some(Ordering::Less)
        } else if self.min_x() - other.min_x() > PRECISION {
            Some(Ordering::Greater)
        } else if self.min_y() - other.min_y() < PRECISION {
            Some(Ordering::Less)
        } else {
            Some(Ordering::Greater)
        }
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Triangle3d {
    pub p1: Point3d,
    pub p2: Point3d,
    pub p3: Point3d,
}

impl Triangle3d {
    pub fn new(p1: (f32, f32, f32), p2: (f32, f32, f32), p3: (f32, f32, f32)) -> Triangle3d {
        Triangle3d {
            p1: Point3d::new(p1.0, p1.1, p1.2),
            p2: Point3d::new(p2.0, p2.1, p2.2),
            p3: Point3d::new(p3.0, p3.1, p3.2),
        }
    }
}

impl Intersect<Plane> for Triangle3d {
    type Output = Option<Shape>;

    fn intersect(self, plane: Plane) -> Option<Shape> {
        let lines = vec![
            Line3d::from_points(&self.p1, &self.p2),
            Line3d::from_points(&self.p2, &self.p3),
            Line3d::from_points(&self.p3, &self.p1),
        ];
        let mut results: Vec<Shape> = Vec::new();
        for line in lines {
            if let Some(answer) = line.intersect(plane) {
                results.push(answer);
            }
        }
        if results.len() == 2 {
            if let (Shape::Point3d(p1), Shape::Point3d(p2)) = (results[0], results[1]) {
                return Some(Shape::Line3d(Line3d::from_points(&p1, &p2)));
            }
        }
        results.retain(|&x| x.is_line());
        match results.len() {
            0 => None,
            1 => Some(results[0]),
            _ => Some(Shape::Triangle3d(self)),
        }
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Plane {
    p: Point3d,
    n: Point3d,
}

impl Plane {
    pub fn new(p: (f32, f32, f32), n: (f32, f32, f32)) -> Plane {
        Plane {
            p: Point3d::new(p.0, p.1, p.2),
            n: Point3d::new(n.0, n.1, n.2),
        }
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum Shape {
    Point3d(Point3d),
    Line3d(Line3d),
    Triangle3d(Triangle3d),
    Plane(Plane),
}

impl Shape {
    pub fn is_line(&self) -> bool {
        match *self {
            Shape::Line3d(_) => true,
            _ => false,
        }
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Point3dx4 {
    pub x: f32x4,
    pub y: f32x4,
    pub z: f32x4,
}

impl Point3dx4 {
    pub fn new(x: f32x4, y: f32x4, z: f32x4) -> Point3dx4 {
        Point3dx4 { x, y, z }
    }

    pub fn from_points(p1: Point3d, p2: Point3d, p3: Point3d, p4: Point3d) -> Point3dx4 {
        Point3dx4 {
            x: f32x4::new(p1.x, p2.x, p3.x, p4.x),
            y: f32x4::new(p1.y, p2.y, p3.y, p4.y),
            z: f32x4::new(p1.z, p2.z, p3.z, p4.z),
        }
    }

    pub fn sum(&self) -> f32x4 {
        self.x + self.y + self.z
    }

    pub fn extract_unchecked(&self, index: usize) -> Point3d {
        unsafe {
            Point3d::new(
                self.x.extract_unchecked(index),
                self.y.extract_unchecked(index),
                self.z.extract_unchecked(index),
            )
        }
    }

    pub fn is_infinite(&self) -> (bool, bool, bool, bool) {
        let x = self.x.is_infinite();
        let y = self.y.is_infinite();
        let z = self.z.is_infinite();
        unsafe {
            (
                x.extract_unchecked(0) || y.extract_unchecked(0) || z.extract_unchecked(0),
                x.extract_unchecked(1) || y.extract_unchecked(1) || z.extract_unchecked(1),
                x.extract_unchecked(2) || y.extract_unchecked(2) || z.extract_unchecked(2),
                x.extract_unchecked(3) || y.extract_unchecked(3) || z.extract_unchecked(3),
            )
        }
    }

    pub fn is_nan(&self) -> (bool, bool, bool, bool) {
        let inf = self.is_infinite();
        let x = self.x.is_nan();
        let y = self.y.is_nan();
        let z = self.z.is_nan();
        unsafe {
            (
                !inf.0
                    && (x.extract_unchecked(0) || y.extract_unchecked(0) || z.extract_unchecked(0)),
                !inf.1
                    && (x.extract_unchecked(1) || y.extract_unchecked(1) || z.extract_unchecked(1)),
                !inf.2
                    && (x.extract_unchecked(2) || y.extract_unchecked(2) || z.extract_unchecked(2)),
                !inf.3
                    && (x.extract_unchecked(3) || y.extract_unchecked(3) || z.extract_unchecked(3)),
            )
        }
    }

    pub fn distance(&self, other: &Point3dx4) -> f32x4 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z - other.z).powi(2))
            .sqrt()
    }
}

impl Support<Point3dx4> for Point3dx4 {
    type Output = f32x4;

    fn dot(self, other: Point3dx4) -> f32x4 {
        self.x
            .mul_add(other.x, self.y.mul_add(other.y, self.z * other.z))
    }

    fn cross(&self, other: &Point3dx4) -> Point3dx4 {
        Point3dx4::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }
}

impl Support<Point3d> for Point3dx4 {
    type Output = f32x4;

    fn dot(self, other: Point3d) -> f32x4 {
        (self * other).sum()
    }

    fn cross(&self, other: &Point3d) -> Point3dx4 {
        Point3dx4::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }
}

impl Mul<Point3dx4> for Point3dx4 {
    type Output = Point3dx4;

    fn mul(self, other: Point3dx4) -> Point3dx4 {
        Point3dx4 {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }
}

impl Mul<Point3d> for Point3dx4 {
    type Output = Point3dx4;

    fn mul(self, other: Point3d) -> Point3dx4 {
        Point3dx4 {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }
}

impl Mul<f32x4> for Point3dx4 {
    type Output = Point3dx4;

    fn mul(self, other: f32x4) -> Point3dx4 {
        Point3dx4 {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
        }
    }
}

impl Sub<Point3dx4> for Point3dx4 {
    type Output = Point3dx4;

    fn sub(self, other: Point3dx4) -> Point3dx4 {
        Point3dx4 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Add<Point3dx4> for Point3dx4 {
    type Output = Point3dx4;

    fn add(self, other: Point3dx4) -> Point3dx4 {
        Point3dx4 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Line3dx4 {
    pub p1: Point3dx4,
    pub p2: Point3dx4,
}

impl Line3dx4 {
    pub fn extract_unchecked(&self, index: usize) -> Line3d {
        Line3d {
            p1: self.p1.extract_unchecked(index),
            p2: self.p2.extract_unchecked(index),
        }
    }

    pub fn from_pointsx4(p1: &Point3dx4, p2: &Point3dx4) -> Line3dx4 {
        Line3dx4 { p1: *p1, p2: *p2 }
    }
}

impl Intersect<Plane> for Line3dx4 {
    type Output = Shapex4;

    fn intersect(self, plane: Plane) -> Shapex4 {
        let direction = self.p2 - self.p1;
        let orthogonal = direction.dot(plane.n);
        let w = plane.p - self.p1;
        let fac = w.dot(plane.n) / orthogonal;
        let v = direction * fac;
        let answer = self.p1 + v;
        unsafe {
            // checking for fac size handles finite lines, may want to change for ray tracing
            let s1 = if answer.is_infinite().0
                || (fac.extract_unchecked(0) < 0.0 || fac.extract_unchecked(0) > 1.0)
            {
                None
            } else if answer.is_nan().0 {
                Some(Shape::Line3d(self.extract_unchecked(0)))
            } else {
                Some(Shape::Point3d(answer.extract_unchecked(0)))
            };
            let s2 = if answer.is_infinite().1
                || (fac.extract_unchecked(1) < 0.0 || fac.extract_unchecked(1) > 1.0)
            {
                None
            } else if answer.is_nan().1 {
                Some(Shape::Line3d(self.extract_unchecked(1)))
            } else {
                Some(Shape::Point3d(answer.extract_unchecked(1)))
            };
            let s3 = if answer.is_infinite().2
                || (fac.extract_unchecked(2) < 0.0 || fac.extract_unchecked(2) > 1.0)
            {
                None
            } else if answer.is_nan().2 {
                Some(Shape::Line3d(self.extract_unchecked(2)))
            } else {
                Some(Shape::Point3d(answer.extract_unchecked(2)))
            };
            let s4 = if answer.is_infinite().3
                || (fac.extract_unchecked(3) < 0.0 || fac.extract_unchecked(3) > 1.0)
            {
                None
            } else if answer.is_nan().3 {
                Some(Shape::Line3d(self.extract_unchecked(3)))
            } else {
                Some(Shape::Point3d(answer.extract_unchecked(3)))
            };
            Shapex4 { s1, s2, s3, s4 }
        }
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Triangle3dx4 {
    pub p1: Point3dx4,
    pub p2: Point3dx4,
    pub p3: Point3dx4,
}

impl Triangle3dx4 {
    pub fn extract_unchecked(&self, index: usize) -> Triangle3d {
        Triangle3d {
            p1: self.p1.extract_unchecked(index),
            p2: self.p2.extract_unchecked(index),
            p3: self.p3.extract_unchecked(index),
        }
    }
}

impl Intersect<Plane> for Triangle3dx4 {
    type Output = Vec<Option<Shape>>;

    fn intersect(self, plane: Plane) -> Vec<Option<Shape>> {
        let lines = vec![
            Line3dx4::from_pointsx4(&self.p1, &self.p2),
            Line3dx4::from_pointsx4(&self.p2, &self.p3),
            Line3dx4::from_pointsx4(&self.p3, &self.p1),
        ];
        let mut answer: Vec<Option<Shape>> = Vec::new();
        let intersections = [
            lines[0].intersect(plane),
            lines[1].intersect(plane),
            lines[2].intersect(plane),
        ];

        for index in 0..3 {
            let mut results: Vec<Shape> = Vec::new();
            for line in &intersections {
                if let Some(answer) = line[index] {
                    results.push(answer);
                }
            }
            if results.len() == 2 {
                if let (Shape::Point3d(p1), Shape::Point3d(p2)) = (results[0], results[1]) {
                    answer.push(Some(Shape::Line3d(Line3d::from_points(&p1, &p2))));
                    continue;
                }
            }
            results.retain(|&x| x.is_line());
            answer.push(match results.len() {
                0 => None,
                1 => Some(results[0]),
                _ => Some(Shape::Triangle3d(self.extract_unchecked(index))),
            });
        }
        answer
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Shapex4 {
    pub s1: Option<Shape>,
    pub s2: Option<Shape>,
    pub s3: Option<Shape>,
    pub s4: Option<Shape>,
}

impl Shapex4 {
    pub fn shapes(&self) -> [Option<Shape>; 4] {
        [self.s1, self.s2, self.s3, self.s4]
    }
}

impl Index<usize> for Shapex4 {
    type Output = Option<Shape>;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.s1,
            1 => &self.s2,
            2 => &self.s3,
            3 => &self.s4,
            _ => &None,
        }
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Point3dx8 {
    pub x: [f32; 8],
    pub y: [f32; 8],
    pub z: [f32; 8],
}

impl Point3dx8 {
    pub fn new(x: [f32; 8], y: [f32; 8], z: [f32; 8]) -> Point3dx8 {
        Point3dx8 { x, y, z }
    }

    pub fn extract(&self, index: usize) -> Point3d {
        Point3d {
            x: self.x[index],
            y: self.y[index],
            z: self.z[index],
        }
    }

    pub fn from_points(points: [Point3d; 8]) -> Point3dx8 {
        Point3dx8 {
            x: [
                points[0].x,
                points[1].x,
                points[2].x,
                points[3].x,
                points[4].x,
                points[5].x,
                points[6].x,
                points[7].x,
            ],
            y: [
                points[0].y,
                points[1].y,
                points[2].y,
                points[3].y,
                points[4].y,
                points[5].y,
                points[6].y,
                points[7].y,
            ],
            z: [
                points[0].z,
                points[1].z,
                points[2].z,
                points[3].z,
                points[4].z,
                points[5].z,
                points[6].z,
                points[7].z,
            ],
        }
    }

    pub fn sum(&self) -> [f32; 8] {
        let mut answer = [0_f32; 8];
        for i in 0..8 {
            answer[i] = self.x[i] + self.y[i] + self.z[i];
        }
        answer
    }

    pub fn is_infinite(&self) -> [bool; 8] {
        let mut answer = [false; 8];
        for i in 0..8 {
            answer[i] =
                self.x[i].is_infinite() || self.y[i].is_infinite() || self.z[i].is_infinite();
        }
        answer
    }

    pub fn is_nan(&self) -> [bool; 8] {
        let inf = self.is_infinite();
        let mut answer = [false; 8];
        for i in 0..8 {
            answer[i] = !inf[i] && (self.x[i].is_nan() || self.y[i].is_nan() || self.z[i].is_nan())
        }
        answer
    }

    pub fn distance(&self, other: &Point3dx8) -> [f32; 8] {
        let mut answer = [0_f32; 8];
        for i in 0..8 {
            answer[i] = ((self.x[i] - other.x[i]).powi(2)
                + (self.y[i] - other.y[i]).powi(2)
                + (self.z[i] - other.z[i]).powi(2))
            .sqrt();
        }
        answer
    }
}

impl Support<Point3dx8> for Point3dx8 {
    type Output = [f32; 8];

    fn dot(self, other: Point3dx8) -> [f32; 8] {
        let mut answer = [0_f32; 8];
        for i in 0..8 {
            answer[i] = self.x[i].mul_add(
                other.x[i],
                self.y[i].mul_add(other.y[i], self.z[i] * other.z[i]),
            );
        }
        answer
    }

    fn cross(&self, other: &Point3dx8) -> Point3dx8 {
        let mut x = [0_f32; 8];
        let mut y = [0_f32; 8];
        let mut z = [0_f32; 8];
        for i in 0..8 {
            x[i] = self.y[i] * other.z[i] - self.z[i] * other.y[i];
            y[i] = self.z[i] * other.x[i] - self.x[i] * other.z[i];
            z[i] = self.x[i] * other.y[i] - self.y[i] * other.x[i];
        }
        Point3dx8 { x, y, z }
    }
}

impl Support<Point3d> for Point3dx8 {
    type Output = [f32; 8];

    fn dot(self, other: Point3d) -> [f32; 8] {
        let mut answer = [0_f32; 8];
        for i in 0..8 {
            answer[i] = (self.extract(i) * other).sum();
        }
        answer
    }

    fn cross(&self, other: &Point3d) -> Point3dx8 {
        let mut x = [0_f32; 8];
        let mut y = [0_f32; 8];
        let mut z = [0_f32; 8];
        for i in 0..8 {
            x[i] = self.y[i] * other.z - self.z[i] * other.y;
            y[i] = self.z[i] * other.x - self.x[i] * other.z;
            z[i] = self.x[i] * other.y - self.y[i] * other.x;
        }
        Point3dx8 { x, y, z }
    }
}

impl Mul<Point3dx8> for Point3dx8 {
    type Output = Point3dx8;

    fn mul(self, other: Point3dx8) -> Point3dx8 {
        let mut x = [0_f32; 8];
        let mut y = [0_f32; 8];
        let mut z = [0_f32; 8];
        for i in 0..8 {
            x[i] = self.x[i] * other.x[i];
            y[i] = self.y[i] * other.y[i];
            z[i] = self.z[i] * other.z[i];
        }

        Point3dx8 { x, y, z }
    }
}

impl Mul<Point3d> for Point3dx8 {
    type Output = Point3dx8;

    fn mul(self, other: Point3d) -> Point3dx8 {
        let mut x = [0_f32; 8];
        let mut y = [0_f32; 8];
        let mut z = [0_f32; 8];
        for i in 0..8 {
            x[i] = self.x[i] * other.x;
            y[i] = self.y[i] * other.y;
            z[i] = self.z[i] * other.z;
        }

        Point3dx8 { x, y, z }
    }
}

impl Mul<[f32; 8]> for Point3dx8 {
    type Output = Point3dx8;

    fn mul(self, other: [f32; 8]) -> Point3dx8 {
        let mut x = [0_f32; 8];
        let mut y = [0_f32; 8];
        let mut z = [0_f32; 8];
        for i in 0..8 {
            x[i] = self.x[i] * other[i];
            y[i] = self.y[i] * other[i];
            z[i] = self.z[i] * other[i];
        }

        Point3dx8 { x, y, z }
    }
}

impl Sub<Point3dx8> for Point3dx8 {
    type Output = Point3dx8;

    fn sub(self, other: Point3dx8) -> Point3dx8 {
        let mut x = [0_f32; 8];
        let mut y = [0_f32; 8];
        let mut z = [0_f32; 8];
        for i in 0..8 {
            x[i] = self.x[i] - other.x[i];
            y[i] = self.y[i] - other.y[i];
            z[i] = self.z[i] - other.z[i];
        }

        Point3dx8 { x, y, z }
    }
}

impl Add<Point3dx8> for Point3dx8 {
    type Output = Point3dx8;

    fn add(self, other: Point3dx8) -> Point3dx8 {
        let mut x = [0_f32; 8];
        let mut y = [0_f32; 8];
        let mut z = [0_f32; 8];
        for i in 0..8 {
            x[i] = self.x[i] + other.x[i];
            y[i] = self.y[i] + other.y[i];
            z[i] = self.z[i] + other.z[i];
        }

        Point3dx8 { x, y, z }
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Line3dx8 {
    pub p1: Point3dx8,
    pub p2: Point3dx8,
}

impl Line3dx8 {
    pub fn from_pointsx8(p1: &Point3dx8, p2: &Point3dx8) -> Line3dx8 {
        Line3dx8 { p1: *p1, p2: *p2 }
    }

    pub fn extract(&self, index: usize) -> Line3d {
        Line3d {
            p1: self.p1.extract(index),
            p2: self.p2.extract(index),
        }
    }
}

impl Intersect<Plane> for Line3dx8 {
    type Output = Shapex8;

    fn intersect(self, plane: Plane) -> Shapex8 {
        let direction = self.p2 - self.p1;
        let orthogonal = direction.dot(plane.n);
        let w = plane.p - self.p1;
        let wdot = w.dot(plane.n);
        //let fac = w.dot(plane.n) / orthogonal;
        let mut fac = [0_f32; 8];
        for i in 0..8 {
            fac[i] = wdot[i] / orthogonal[i];
        }
        let v = direction * fac;
        let answer = self.p1 + v;
        let infinite = answer.is_infinite();
        let nan = answer.is_nan();
        // checking for fac size handles finite lines, may want to change for ray tracing
        let mut answers: [Option<Shape>; 8] = [None; 8];
        for i in 0..8 {
            answers[i] = if infinite[i] || (fac[i] < 0.0 || fac[i] > 1.0) {
                None
            } else if nan[i] {
                Some(Shape::Line3d(self.extract(i)))
            } else {
                Some(Shape::Point3d(answer.extract(i)))
            };
        }
        Shapex8 {
            s1: answers[0],
            s2: answers[1],
            s3: answers[2],
            s4: answers[3],
            s5: answers[4],
            s6: answers[5],
            s7: answers[6],
            s8: answers[7],
        }
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Triangle3dx8 {
    pub p1: Point3dx8,
    pub p2: Point3dx8,
    pub p3: Point3dx8,
}

impl Triangle3dx8 {
    pub fn extract(&self, index: usize) -> Triangle3d {
        Triangle3d {
            p1: self.p1.extract(index),
            p2: self.p2.extract(index),
            p3: self.p3.extract(index),
        }
    }
}

impl Intersect<Plane> for Triangle3dx8 {
    type Output = Vec<Option<Shape>>;

    fn intersect(self, plane: Plane) -> Vec<Option<Shape>> {
        let lines = vec![
            Line3dx8::from_pointsx8(&self.p1, &self.p2),
            Line3dx8::from_pointsx8(&self.p2, &self.p3),
            Line3dx8::from_pointsx8(&self.p3, &self.p1),
        ];
        let mut answer: Vec<Option<Shape>> = Vec::new();
        let intersections = [
            lines[0].intersect(plane),
            lines[1].intersect(plane),
            lines[2].intersect(plane),
        ];

        for index in 0..3 {
            let mut results: Vec<Shape> = Vec::new();
            for line in &intersections {
                if let Some(answer) = line[index] {
                    results.push(answer);
                }
            }
            if results.len() == 2 {
                if let (Shape::Point3d(p1), Shape::Point3d(p2)) = (results[0], results[1]) {
                    answer.push(Some(Shape::Line3d(Line3d::from_points(&p1, &p2))));
                    continue;
                }
            }
            results.retain(|&x| x.is_line());
            answer.push(match results.len() {
                0 => None,
                1 => Some(results[0]),
                _ => Some(Shape::Triangle3d(self.extract(index))),
            });
        }
        answer
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Shapex8 {
    pub s1: Option<Shape>,
    pub s2: Option<Shape>,
    pub s3: Option<Shape>,
    pub s4: Option<Shape>,
    pub s5: Option<Shape>,
    pub s6: Option<Shape>,
    pub s7: Option<Shape>,
    pub s8: Option<Shape>,
}

impl Shapex8 {
    pub fn shapes(&self) -> [Option<Shape>; 8] {
        [
            self.s1, self.s2, self.s3, self.s4, self.s5, self.s6, self.s7, self.s8,
        ]
    }
}

impl Index<usize> for Shapex8 {
    type Output = Option<Shape>;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.s1,
            1 => &self.s2,
            2 => &self.s3,
            3 => &self.s4,
            4 => &self.s5,
            5 => &self.s6,
            6 => &self.s7,
            7 => &self.s8,
            _ => &None,
        }
    }
}
