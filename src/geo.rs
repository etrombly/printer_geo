//! # Geo
//!
//! Collection of geometric types and functions useful for 3d models and CAM
//! mostly wraps ultraviolet types with some additional functionality

use crate::compute::{intersect_tris, Vk};
use float_cmp::approx_eq;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    cmp::{Ordering, PartialEq},
    ops::{Add, Mul, Sub},
};
use ultraviolet::{Vec2, Vec3};
use std::ops::Index;

const PRECISION: f32 = 0.001;

#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Point3d {
    pub pos: Vec3,
}

impl Point3d {
    pub fn new(x: f32, y: f32, z: f32) -> Point3d {
        Point3d {
            pos: Vec3::new(x, y, z),
        }
    }

    pub fn is_infinite(&self) -> bool { is_vec3_inf(&self.pos) }

    pub fn is_nan(&self) -> bool { is_vec3_nan(&self.pos) }

    pub fn cross(&self, other: &Point3d) -> Point3d { self.pos.cross(other.pos).into() }

    pub fn dot(&self, other: &Point3d) -> f32 { self.pos.dot(other.pos) }
}

impl Add<Point3d> for Point3d {
    type Output = Point3d;

    fn add(self, other: Point3d) -> Point3d { (self.pos + other.pos).into() }
}

impl Sub<Point3d> for Point3d {
    type Output = Point3d;

    fn sub(self, other: Point3d) -> Point3d { (self.pos - other.pos).into() }
}

impl Mul<f32> for Point3d {
    type Output = Point3d;

    fn mul(self, num: f32) -> Point3d {
        Point3d::new(self.pos.x * num, self.pos.y * num, self.pos.z * num)
    }
}

impl From<Vec3> for Point3d {
    fn from(vec3: Vec3) -> Self { Point3d { pos: vec3 } }
}

impl Eq for Point3d {}

impl PartialEq for Point3d {
    fn eq(&self, other: &Self) -> bool {
        approx_eq!(f32, self.pos.x, other.pos.x, ulps = 3)
            && approx_eq!(f32, self.pos.y, other.pos.y, ulps = 3)
            && approx_eq!(f32, self.pos.z, other.pos.z, ulps = 3)
    }
}

impl Ord for Point3d {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.pos.x > other.pos.x {
            Ordering::Greater
        } else if approx_eq!(f32, self.pos.x, other.pos.x, ulps = 3) {
            if self.pos.y > other.pos.y {
                Ordering::Greater
            } else if approx_eq!(f32, self.pos.y, other.pos.y, ulps = 3) {
                if self.pos.z > other.pos.z {
                    Ordering::Greater
                } else if approx_eq!(f32, self.pos.z, other.pos.z, ulps = 3) {
                    Ordering::Equal
                } else {
                    Ordering::Less
                }
            } else {
                Ordering::Less
            }
        } else {
            Ordering::Less
        }
    }
}

impl PartialOrd for Point3d {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

/// check if any value in point is infinite
pub fn is_vec3_inf(point: &Vec3) -> bool {
    point.x.is_infinite() || point.y.is_infinite() || point.z.is_infinite()
}

/// check if any value in point is nan
pub fn is_vec3_nan(point: &Vec3) -> bool {
    point.x.is_nan() || point.y.is_nan() || point.z.is_nan()
}

pub fn distance(left: &Vec2, right: &Vec2) -> f32 {
    ((left.x - right.x).powi(2) + (left.y - right.y).powi(2)).sqrt()
}

/// Trait for intersection of different geo types
pub trait Intersect<RHS = Self> {
    type Output;
    fn intersect(self, rhs: RHS) -> Self::Output;
}

#[cfg_attr(feature = "python", pyclass)]
#[derive(PartialEq, Clone, Copy, Debug, Default)]
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

    pub fn from_points(p1: &Point3d, p2: &Point3d) -> Line3d { Line3d { p1: *p1, p2: *p2 } }

    /// Check if point is on line segment
    ///
    /// # Examples
    ///
    /// ```
    /// use printer_geo::geo::{Line3d, Point3d};
    /// let point = Point3d::new(1., 1., 1.);
    /// let line = Line3d::new((0., 0., 1.), (2., 2., 1.));
    /// assert!(line.on_line_2d(&point));
    /// ```
    pub fn on_line_2d(&self, point: &Point3d) -> bool {
        (distance(&self.p1.pos.xy(), &point.pos.xy())
            + distance(&self.p2.pos.xy(), &point.pos.xy()))
            - distance(&self.p1.pos.xy(), &self.p2.pos.xy())
            < PRECISION
    }

    /// Check if point is within bounds defined by current line
    /// ignores Z axis
    ///
    /// # Examples
    ///
    /// ```
    /// use printer_geo::geo::{Line3d, Point3d};
    /// let point = Point3d::new(1., 1., 1.);
    /// let bounds = Line3d::new((0., 0., 0.), (2., 2., 0.));
    /// assert!(bounds.in_2d_bounds(&point));
    /// ```
    pub fn in_2d_bounds(&self, point: &Point3d) -> bool {
        point.pos.x >= self.p1.pos.x
            && point.pos.x <= self.p2.pos.x
            && point.pos.y >= self.p1.pos.y
            && point.pos.y <= self.p2.pos.y
    }

    /// Check if line segments intersect, ignoring Z axis
    ///
    /// # Examples
    ///
    /// ```
    /// use printer_geo::geo::Line3d;
    /// let line1 = Line3d::new((0., 0., 0.), (1., 1., 1.));
    /// let line2 = Line3d::new((1., 0., 1.), (0., 1., 0.));
    /// assert!(line1.intersect_2d(&line2));
    /// ```
    pub fn intersect_2d(self, other: &Line3d) -> bool {
        let a1 = self.p2.pos.y - self.p1.pos.y;
        let b1 = self.p1.pos.x - self.p2.pos.x;
        let c1 = a1 * self.p1.pos.x + b1 * self.p1.pos.x;

        let a2 = other.p2.pos.y - other.p1.pos.y;
        let b2 = other.p1.pos.x - other.p2.pos.x;
        let c2 = a2 * other.p1.pos.x + b2 * other.p1.pos.y;

        let delta = a1 * b2 - a2 * b1;
        let x = (b2 * c1 - b1 * c2) / delta;
        let y = (a1 * c2 - a2 * c1) / delta;
        delta != 0.0
            && self.p1.pos.x.min(self.p2.pos.x) <= x
            && x <= self.p1.pos.x.max(self.p2.pos.x)
            && self.p1.pos.y.min(self.p2.pos.y) <= y
            && y <= self.p1.pos.y.max(self.p2.pos.y)
    }

    /// Create bounding box from line
    pub fn bbox(self) -> Line3d {
        Line3d {
            p1: Point3d::new(self.min_x(), self.min_y(), self.min_z()),
            p2: Point3d::new(self.max_x(), self.max_y(), self.max_z()),
        }
    }

    pub fn min_x(self) -> f32 { self.p1.pos.x.min(self.p2.pos.x) }

    pub fn min_y(self) -> f32 { self.p1.pos.y.min(self.p2.pos.y) }

    pub fn min_z(self) -> f32 { self.p1.pos.z.min(self.p2.pos.z) }

    pub fn max_x(self) -> f32 { self.p1.pos.x.max(self.p2.pos.x) }

    pub fn max_y(self) -> f32 { self.p1.pos.y.max(self.p2.pos.y) }

    pub fn max_z(self) -> f32 { self.p1.pos.z.max(self.p2.pos.z) }
}

impl Intersect<Plane> for Line3d {
    type Output = Option<Shape>;

    fn intersect(self, plane: Plane) -> Option<Shape> {
        let direction = self.p2.pos - self.p1.pos;
        let n = plane.n.pos;
        let orthogonal = n.dot(direction);
        let w = plane.p.pos - self.p1.pos;
        let fac = n.dot(w) / orthogonal;
        let v = direction * fac;
        let answer = self.p1.pos + v;
        if is_vec3_inf(&answer) {
            None
        } else if is_vec3_nan(&answer) {
            Some(Shape::Line3d(self))
        } else {
            Some(Shape::Point3d(Point3d { pos: answer }))
        }
    }
}

impl Add<Line3d> for Line3d {
    type Output = Line3d;

    fn add(self, other: Line3d) -> Line3d {
        Line3d {
            p1: self.p1 + other.p1,
            p2: self.p2 + other.p2,
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

#[cfg_attr(feature = "python", pyclass)]
#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Triangle3d {
    pub p1: Point3d,
    pub p2: Point3d,
    pub p3: Point3d,
}

pub type Triangles3d = Vec<Triangle3d>;

impl Triangle3d {
    pub fn new(p1: (f32, f32, f32), p2: (f32, f32, f32), p3: (f32, f32, f32)) -> Triangle3d {
        Triangle3d {
            p1: Point3d::new(p1.0, p1.1, p1.2),
            p2: Point3d::new(p2.0, p2.1, p2.2),
            p3: Point3d::new(p3.0, p3.1, p3.2),
        }
    }

    /// Check if triangle is within bounding box,
    /// ignoring Z axis
    ///
    /// # Examples
    ///
    /// ```
    /// use printer_geo::geo::{Line3d, Triangle3d};
    /// let tri = Triangle3d::new((1., 1., 1.), (1., 2., 1.), (2., 1., 1.));
    /// let bounds = Line3d::new((0., 0., 1.), (2., 2., 1.));
    /// assert!(tri.in_2d_bounds(&bounds));
    /// ```
    pub fn in_2d_bounds(&self, bbox: &Line3d) -> bool {
        let left = Line3d {
            p1: bbox.p1,
            p2: Point3d::new(bbox.p1.pos.x, bbox.p2.pos.y, bbox.p1.pos.z),
        };
        let right = Line3d {
            p1: bbox.p2,
            p2: Point3d::new(bbox.p2.pos.x, bbox.p1.pos.y, bbox.p2.pos.z),
        };
        let line1 = Line3d {
            p1: self.p1,
            p2: self.p2,
        };
        let line2 = Line3d {
            p1: self.p2,
            p2: self.p3,
        };
        let line3 = Line3d {
            p1: self.p1,
            p2: self.p3,
        };
        bbox.in_2d_bounds(&self.p1)
            || bbox.in_2d_bounds(&self.p2)
            || bbox.in_2d_bounds(&self.p3)
            || left.intersect_2d(&line1)
            || left.intersect_2d(&line2)
            || left.intersect_2d(&line3)
            || right.intersect_2d(&line1)
            || right.intersect_2d(&line2)
            || right.intersect_2d(&line3)
    }

    /// Move triangle
    ///
    /// # Examples
    ///
    /// ```
    /// use printer_geo::geo::Triangle3d;
    /// let mut tri = Triangle3d::new((1., 1., 1.), (1., 2., 1.), (2., 1., 1.));
    /// tri.translate(1., 0., 0.);
    /// assert_eq!(tri, Triangle3d::new((2., 1., 1.), (2., 2., 1.), (3., 1., 1.)));
    /// ```
    pub fn translate(&mut self, x: f32, y: f32, z: f32) {
        let p = Point3d::new(x, y, z);
        self.p1 = self.p1 + p;
        self.p2 = self.p2 + p;
        self.p3 = self.p3 + p;
    }

    pub fn bbox(self) -> Line3d {
        Line3d {
            p1: Point3d::new(self.min_x(), self.min_y(), self.min_z()),
            p2: Point3d::new(self.max_x(), self.max_y(), self.max_z()),
        }
    }

    pub fn min_x(self) -> f32 { self.p1.pos.x.min(self.p2.pos.x).min(self.p3.pos.x) }

    pub fn min_y(self) -> f32 { self.p1.pos.y.min(self.p2.pos.y).min(self.p3.pos.y) }

    pub fn min_z(self) -> f32 { self.p1.pos.z.min(self.p2.pos.z).min(self.p3.pos.z) }

    pub fn max_x(self) -> f32 { self.p1.pos.x.max(self.p2.pos.x).max(self.p3.pos.x) }

    pub fn max_y(self) -> f32 { self.p1.pos.y.max(self.p2.pos.y).max(self.p3.pos.y) }

    pub fn max_z(self) -> f32 { self.p1.pos.z.max(self.p2.pos.z).max(self.p3.pos.z) }
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

impl Intersect<Line3d> for Triangle3d {
    type Output = Option<Point3d>;

    fn intersect(self, line: Line3d) -> Option<Point3d> {
        let p1 = self.p2 - self.p1;
        let p2 = self.p3 - self.p1;
        let lp1 = line.p1;
        let lp2 = line.p2;
        let n = p1.cross(&p2);
        let det = -lp2.dot(&n);
        let inv_det = 1.0 / det;
        let ao = lp1 - self.p1;
        let dao = ao.cross(&lp2);
        let u = p2.dot(&dao) * inv_det;
        let v = -p1.dot(&dao) * inv_det;
        let t = ao.dot(&n) * inv_det;
        if det >= 1e-6 && t >= 0.0 && u >= 0.0 && v >= 0.0 && (u + v) <= 1.0 {
            Some(lp1 + (lp2 * t))
        } else {
            None
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

#[cfg_attr(feature = "python", pyclass)]
#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Circle {
    pub center: Point3d,
    pub radius: f32,
}

impl Circle {
    pub fn new(center: Point3d, radius: f32) -> Circle { Circle { center, radius } }

    /// Check if point is in circle,
    /// ignoring Z axis
    ///
    /// # Examples
    ///
    /// ```
    /// use printer_geo::geo::{Point3d, Circle};
    /// let circle = Circle::new(Point3d::new(0., 0., 0.), 10.);
    /// let point = Point3d::new(1., 1., 1.);
    /// assert!(circle.in_2d_bounds(&point));
    /// ```
    pub fn in_2d_bounds(&self, point: &Point3d) -> bool {
        let dx = f32::abs(point.pos.x - self.center.pos.x);
        let dy = f32::abs(point.pos.y - self.center.pos.y);
        if dx > self.radius {
            false
        } else if dy > self.radius {
            false
        } else if dx + dy <= self.radius {
            true
        } else {
            dx.powi(2) + dy.powi(2) <= self.radius.powi(2)
        }
    }

    pub fn bbox(self) -> Line3d {
        Line3d {
            p1: Point3d::new(self.min_x(), self.min_y(), self.min_z()),
            p2: Point3d::new(self.max_x(), self.max_y(), self.max_z()),
        }
    }

    pub fn min_x(self) -> f32 { self.center.pos.x - self.radius }

    pub fn min_y(self) -> f32 { self.center.pos.y - self.radius }

    pub fn min_z(self) -> f32 { self.center.pos.z }

    pub fn max_x(self) -> f32 { self.center.pos.x + self.radius }

    pub fn max_y(self) -> f32 { self.center.pos.y + self.radius }

    pub fn max_z(self) -> f32 { self.center.pos.z }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum Shape {
    Point3d(Point3d),
    Line3d(Line3d),
    Triangle3d(Triangle3d),
    Plane(Plane),
}

impl Shape {
    pub fn is_line(&self) -> bool { matches!(*self, Shape::Line3d(_)) }
}

/// Get bounds for list of triangles
pub fn get_bounds(tris: &[Triangle3d]) -> Line3d {
    tris.par_iter().map(|tri| tri.bbox()).reduce(
        || Line3d {
            p1: Point3d::new(f32::MAX, f32::MAX, f32::MAX),
            p2: Point3d::new(f32::MIN, f32::MIN, f32::MIN),
        },
        |mut acc, bbox| {
            if bbox.p1.pos.x < acc.p1.pos.x {
                acc.p1.pos.x = bbox.p1.pos.x;
            }
            if bbox.p1.pos.y < acc.p1.pos.y {
                acc.p1.pos.y = bbox.p1.pos.y;
            }
            if bbox.p1.pos.z < acc.p1.pos.z {
                acc.p1.pos.z = bbox.p1.pos.z;
            }
            if bbox.p2.pos.x > acc.p2.pos.x {
                acc.p2.pos.x = bbox.p2.pos.x;
            }
            if bbox.p2.pos.y > acc.p2.pos.y {
                acc.p2.pos.y = bbox.p2.pos.y;
            }
            if bbox.p2.pos.z > acc.p2.pos.z {
                acc.p2.pos.z = bbox.p2.pos.z;
            }
            acc
        },
    )
}

/// Move triangles to x:0 y:0 and z_max: 0
pub fn move_to_zero(tris: &mut Vec<Triangle3d>) {
    // get the bounds for the model
    let bounds = get_bounds(tris);
    tris.par_iter_mut()
        .for_each(|tri| tri.translate(-bounds.p1.pos.x, -bounds.p1.pos.y, -bounds.p2.pos.z));
}

#[cfg_attr(feature = "python", pyclass)]
/// Tool for CAM operations, represented as a point cloud
#[derive(Default, Clone)]
pub struct Tool {
    pub bbox: Line3d,
    pub points: Vec<Point3d>,
}

impl Tool {
    pub fn new_endmill(radius: f32, scale: f32) -> Tool {
        let circle = Circle::new(Point3d::new(0., 0., 0.), radius);
        let points = Tool::circle_to_points(&circle, scale);
        Tool {
            bbox: circle.bbox(),
            points,
        }
    }

    pub fn new_v_bit(radius: f32, angle: f32, scale: f32) -> Tool {
        let circle = Circle::new(Point3d::new(0., 0., 0.), radius);
        let percent = (90. - (angle / 2.)).to_radians().tan();
        let points = Tool::circle_to_points(&circle, scale);
        let points = points
            .iter()
            .map(|point| {
                let distance = (point.pos.x.powi(2) + point.pos.y.powi(2)).sqrt();
                let z = distance * percent;
                Point3d::new(point.pos.x, point.pos.y, z)
            })
            .collect();
        Tool {
            bbox: circle.bbox(),
            points,
        }
    }

    // TODO: this approximates a ball end mill, probably want to generate the
    // geometry better
    pub fn new_ball(radius: f32, scale: f32) -> Tool {
        let circle = Circle::new(Point3d::new(0., 0., 0.), radius);
        let points = Tool::circle_to_points(&circle, scale);
        let points = points
            .iter()
            .map(|point| {
                let distance = (point.pos.x.powi(2) + point.pos.y.powi(2)).sqrt();
                let z = if distance > 0. {
                    // 65. is the angle
                    radius + (-radius * (65. / (radius / distance)).to_radians().cos())
                } else {
                    0.
                };
                Point3d::new(point.pos.x, point.pos.y, z)
            })
            .collect();
        Tool {
            bbox: circle.bbox(),
            points,
        }
    }

    pub fn circle_to_points(circle: &Circle, scale: f32) -> Vec<Point3d> {
        let mut points: Vec<Point3d> = (0..=(circle.radius * scale) as i32)
            .flat_map(|x| {
                (0..=(circle.radius * scale) as i32).flat_map(move |y| {
                    vec![
                        Point3d::new(-x as f32 / scale, y as f32 / scale, 0.0),
                        Point3d::new(-x as f32 / scale, -y as f32 / scale, 0.0),
                        Point3d::new(x as f32 / scale, -y as f32 / scale, 0.0),
                        Point3d::new(x as f32 / scale, y as f32 / scale, 0.0),
                    ]
                })
            })
            .filter(|x| circle.in_2d_bounds(&x))
            .collect();
        points.sort();
        points.dedup();
        points
    }
}

pub struct Grid {
    cols: usize,
    rows: usize,
    points: Vec<Point3d>
}

impl Index<usize> for Grid {
    type Output = [Point3d];
    fn index(&self, index: usize) -> &Self::Output {
        &self.points[index * self.cols .. (index+1) * self.cols]
    }
}

pub fn generate_grid(bounds: &Line3d, scale: &f32) -> Vec<Vec<Point3d>> {
    let max_x = (bounds.p2.pos.x * scale) as i32;
    let min_x = (bounds.p1.pos.x * scale) as i32;
    let max_y = (bounds.p2.pos.y * scale) as i32;
    let min_y = (bounds.p1.pos.y * scale) as i32;
    (min_x..=max_x)
        .into_par_iter()
        .map(|x| {
            (min_y..=max_y)
                .into_par_iter()
                .map(move |y| Point3d::new(x as f32 / scale, y as f32 / scale, bounds.p1.pos.z))
                .collect::<Vec<_>>()
        })
        .collect()
}

pub fn generate_columns(
    grid: &[Vec<Point3d>],
    bounds: &Line3d,
    resolution: &f32,
    scale: &f32,
) -> Vec<Line3d> {
    let max_y = (bounds.p2.pos.y * scale) as i32;
    let min_y = (bounds.p1.pos.y * scale) as i32;
    grid.iter()
        .map(|x| {
            // bounding box for this column
            Line3d {
                p1: Point3d::new(x[0].pos.x - resolution, min_y as f32 / scale, 0.),
                p2: Point3d::new(x[0].pos.x + resolution, max_y as f32 / scale, 0.),
            }
        })
        .collect()
}

pub fn generate_columns_chunks(
    grid: &[Vec<Point3d>],
    bounds: &Line3d,
    resolution: &f32,
    scale: &f32,
) -> Vec<Line3d> {
    let max_y = (bounds.p2.pos.y * scale) as i32;
    let min_y = (bounds.p1.pos.y * scale) as i32;
    grid.chunks(10)
        .map(|x| {
            let last = x.len() - 1;
            // bounding box for this column
            Line3d {
                p1: Point3d::new(x[0][0].pos.x - resolution, min_y as f32 / scale, 0.),
                p2: Point3d::new(x[last][0].pos.x + resolution, max_y as f32 / scale, 0.),
            }
        })
        .collect()
}

pub fn generate_heightmap(
    grid: &[Vec<Point3d>],
    partition: &[Vec<Triangle3d>],
    vk: &Vk,
) -> Vec<Vec<Point3d>> {
    grid.iter()
        .enumerate()
        .map(|(column, test)| {
            // ray cast on the GPU to figure out the highest point for each point in this
            // column
            intersect_tris(&partition[column], &test, &vk).unwrap()
        })
        .collect()
}

pub fn generate_heightmap_chunks(
    grid: &[Vec<Point3d>],
    partition: &[Vec<Triangle3d>],
    vk: &Vk,
) -> Vec<Vec<Point3d>> {
    let mut result = Vec::with_capacity(grid.len());
    for (column, test) in grid.chunks(10).enumerate() {
        // ray cast on the GPU to figure out the highest point for each point in this
        // column
        // TODO: there's probably a better way to process this in chunks
        let len = test[0].len();
        let tris = intersect_tris(
            &partition[column],
            &test.iter().flat_map(|x| x).copied().collect::<Vec<_>>(),
            &vk,
        )
        .unwrap();
        for chunk in tris.chunks(len) {
            result.push(chunk.to_vec());
        }
    }
    result
}

pub fn generate_toolpath(
    heightmap: &[Vec<Point3d>],
    bounds: &Line3d,
    tool: &Tool,
    radius: &f32,
    stepover: &f32,
    scale: &f32,
) -> Vec<Vec<Point3d>> {
    let columns = heightmap.len();
    let rows = heightmap[0].len();
    ((radius * scale) as usize..columns)
.into_par_iter()
// space each column based on radius and stepover
.step_by((radius * stepover * scale).ceil() as usize)
.enumerate()
.map(|(column_num, x)| {
    // alternate direction for each column, have to collect into a vec to get types to match
    let steps = if column_num % 2 == 0 {
        ((radius * scale) as usize..rows).collect::<Vec<_>>().into_par_iter()
    } else {
        ((radius * scale) as usize..rows).rev().collect::<Vec<_>>().into_par_iter()
    };
    steps
        .map(|y| {
            let max = tool
                .points
                .iter()
                .map(|tpoint| {
                    // for each point in the tool adjust it's location to the height map and calculate the intersection
                    let x_offset = (x as f32 + (tpoint.pos.x * scale)).round() as i32;
                    let y_offset = (y as f32 + (tpoint.pos.y * scale)).round() as i32;
                    if x_offset < columns as i32
                        && x_offset >= 0
                        && y_offset < rows as i32
                        && y_offset >= 0
                    {

                        heightmap[x_offset as usize][y_offset as usize].pos.z
                            - tpoint.pos.z
                    } else {
                        bounds.p1.pos.z
                    }
                })
                .fold(f32::NAN, f32::max); // same as calling max on all the values for this tool to find the heighest
            Point3d::new(x as f32 / scale, y as f32 / scale, max)
        })
        .collect()
})
.collect()
}

pub fn generate_layers(
    toolpath: &[Vec<Point3d>],
    bounds: &Line3d,
    stepdown: &f32,
) -> Vec<Vec<Vec<Point3d>>> {
    let steps = ((bounds.p2.pos.z - bounds.p1.pos.z) / stepdown) as u64;
    (1..steps + 1)
        .map(|step| {
            toolpath
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|x| match step as f32 * -stepdown {
                            z if z > x.pos.z => Point3d::new(x.pos.x, x.pos.y, z),
                            _ => *x,
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

pub fn generate_gcode(layers: &[Vec<Vec<Point3d>>], bounds: &Line3d) -> String {
    let mut last = layers[0][0][0];
    let mut output = format!("G1 Z{:.2} F300\n", bounds.p2.pos.z);

    for layer in layers {
        output.push_str(&format!(
            "G0 X{:.2} Y{:.2}\nG0 Z{:.2}\n",
            layer[0][0].pos.x, layer[0][0].pos.y, layer[0][0].pos.z
        ));
        for row in layer {
            output.push_str(&format!(
                "G0 X{:.2} Y{:.2}\nG0 Z{:.2}\n",
                row[0].pos.x, row[0].pos.y, row[0].pos.z
            ));
            for point in row {
                if !approx_eq!(f32, last.pos.x, point.pos.x, ulps = 2)
                    || !approx_eq!(f32, last.pos.z, point.pos.z, ulps = 2)
                {
                    output.push_str(&format!(
                        "G1 X{:.2} Y{:.3} Z{:.2}\n",
                        point.pos.x, point.pos.y, point.pos.z
                    ));
                }
                last = *point;
            }
            output.push_str(&format!(
                "G1 X{:?} Y{:?} Z{:?}\nG0 Z{:.2}\n",
                last.pos.x, last.pos.y, last.pos.z, bounds.p2.pos.z
            ));
        }
    }
    output
}

/*
pub fn intersect_tris_fallback(tris: &[Triangle3d], points: &[Point3d]) -> Vec<Point3d> {
    points
        .par_iter()
        .map(|point| {
            Point3d::new(
                point.pos.x,
                point.pos.y,
                tris.iter()
                    .filter_map(|tri| tri.intersect_ray(&point))
                    .fold(f32::NAN, f32::max),
            )
        })
        .collect()
}
*/
