//! # Geo
//!
//! Collection of geometric types and functions useful for 3d models and CAM
//! mostly wraps Nalgebra types with some additional functionality

use nalgebra::{distance, zero, Isometry3, Point3, Vector3};
use rayon::prelude::*;
use std::{cmp::Ordering, ops::Add};

/// Wrapper for Nalgebra Point3<f32>
pub type Point3d = Point3<f32>;

/// check if any value in point is infinite
pub fn is_point_inf(point: &Point3d) -> bool { point.iter().any(|x| x.is_infinite()) }

/// check if any value in point is nan
pub fn is_point_nan(point: &Point3d) -> bool { point.iter().any(|x| x.is_nan()) }

const PRECISION: f32 = 0.001;

/// Trait for intersection of different geo types
pub trait Intersect<RHS = Self> {
    type Output;
    fn intersect(self, rhs: RHS) -> Self::Output;
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
        (distance(&self.p1.xy(), &point.xy()) + distance(&self.p2.xy(), &point.xy())) - distance(&self.p1.xy(), &self.p2.xy())
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
        point.x >= self.p1.x && point.x <= self.p2.x && point.y >= self.p1.y && point.y <= self.p2.y
    }

    /// Check if line segments intersect
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
        // modified from https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/565282#565282
        let s1_x = self.p2[0] - self.p1[0]; 
        let s1_y = self.p2[1] - self.p1[1];
        let s2_x = other.p2[0] - other.p1[0];     
        let s2_y = other.p2[1] - other.p1[1];
    
        let denominator = -s2_x * s1_y + s1_x * s2_y;
        if denominator != 0. {
            let s = (-s1_y * (self.p1[0] - other.p1[0]) + s1_x * (self.p1[1] - other.p1[1])) / denominator;
            let t = ( s2_x * (self.p1[1] - other.p1[1]) - s2_y * (self.p1[0] - other.p1[0])) / denominator;
        
            if s >= 0. && s <= 1. && t >= 0. && t <= 1.{
                let intersect_x = self.p1[0] + (t * s1_x);
                let intersect_y = self.p1[1] + (t * s1_y);
                self.on_line_2d(&Point3d::new(intersect_x, intersect_y, 0.))
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Create bounding box from line
    pub fn bbox(self) -> Line3d {
        Line3d {
            p1: Point3d::new(self.min_x(), self.min_y(), self.min_z()),
            p2: Point3d::new(self.max_x(), self.max_y(), self.max_z()),
        }
    }

    pub fn min_x(self) -> f32 { self.p1.x.min(self.p2.x) }

    pub fn min_y(self) -> f32 { self.p1.y.min(self.p2.y) }

    pub fn min_z(self) -> f32 { self.p1.z.min(self.p2.z) }

    pub fn max_x(self) -> f32 { self.p1.x.max(self.p2.x) }

    pub fn max_y(self) -> f32 { self.p1.y.max(self.p2.y) }

    pub fn max_z(self) -> f32 { self.p1.z.max(self.p2.z) }
}

impl Intersect<Plane> for Line3d {
    type Output = Option<Shape>;

    fn intersect(self, plane: Plane) -> Option<Shape> {
        let direction = (self.p2 - self.p1).to_homogeneous();
        let n = plane.n.to_homogeneous();
        let orthogonal = n.dot(&direction);
        let w = (plane.p - self.p1).to_homogeneous();
        let fac = n.dot(&w) / orthogonal;
        let v = direction * fac;
        let answer = Point3d::from_homogeneous(self.p1.to_homogeneous() + v);
        // checking for fac size handles finite lines, may want to change for
        // ray tracing
        if answer.iter().any(|x| is_point_inf(x)) || (fac < 0.0 || fac > 1.0) {
            None
        } else if answer.iter().any(|x| is_point_nan(x)) {
            Some(Shape::Line3d(self))
        } else {
            Some(Shape::Point3d(answer.unwrap()))
        }
    }
}

impl Add<Line3d> for Line3d {
    type Output = Line3d;

    fn add(self, other: Line3d) -> Line3d {
        Line3d {
            p1: Point3d::from_homogeneous(self.p1.to_homogeneous() + other.p1.to_homogeneous())
                .unwrap(),
            p2: Point3d::from_homogeneous(self.p2.to_homogeneous() + other.p2.to_homogeneous())
                .unwrap(),
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
        let left = Line3d{p1: bbox.p1, p2: Point3d::new(bbox.p1[0], bbox.p2[1], bbox.p1[2])};
        let right = Line3d{p1: bbox.p2, p2: Point3d::new(bbox.p2[0], bbox.p1[1], bbox.p2[2])};
        let line1 = Line3d{p1: self.p1, p2: self.p2};
        let line2 = Line3d{p1: self.p2, p2: self.p3};
        let line3 = Line3d{p1: self.p1, p2: self.p3};
        bbox.in_2d_bounds(&self.p1) || bbox.in_2d_bounds(&self.p2) || bbox.in_2d_bounds(&self.p3) ||
    left.intersect_2d(&line1) || left.intersect_2d(&line2) || left.intersect_2d(&line3) || 
    right.intersect_2d(&line1) || right.intersect_2d(&line2) || right.intersect_2d(&line3)
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
        let iso = Isometry3::new(Vector3::new(x, y, z), zero());
        self.p1 = iso.transform_point(&self.p1);
        self.p2 = iso.transform_point(&self.p2);
        self.p3 = iso.transform_point(&self.p3);
    }

    pub fn bbox(self) -> Line3d {
        Line3d {
            p1: Point3d::new(self.min_x(), self.min_y(), self.min_z()),
            p2: Point3d::new(self.max_x(), self.max_y(), self.max_z()),
        }
    }

    pub fn min_x(self) -> f32 { self.p1.x.min(self.p2.x).min(self.p3.x) }

    pub fn min_y(self) -> f32 { self.p1.y.min(self.p2.y).min(self.p3.y) }

    pub fn min_z(self) -> f32 { self.p1.z.min(self.p2.z).min(self.p3.z) }

    pub fn max_x(self) -> f32 { self.p1.x.max(self.p2.x).max(self.p3.x) }

    pub fn max_y(self) -> f32 { self.p1.y.max(self.p2.y).max(self.p3.y) }

    pub fn max_z(self) -> f32 { self.p1.z.max(self.p2.z).max(self.p3.z) }
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
        let p1 = (self.p2 - self.p1).to_homogeneous();
        let p2 = (self.p3 - self.p1).to_homogeneous();
        let lp1 = line.p1.to_homogeneous();
        let lp2 = line.p2.to_homogeneous();
        let n = p1.cross(&p2);
        let det = -lp2.dot(&n);
        let inv_det = 1.0 / det;
        let ao = lp1 - self.p1.to_homogeneous();
        let dao = ao.cross(&lp2);
        let u = p2.dot(&dao) * inv_det;
        let v = -p1.dot(&dao) * inv_det;
        let t = ao.dot(&n) * inv_det;
        if det >= 1e-6 && t >= 0.0 && u >= 0.0 && v >= 0.0 && (u + v) <= 1.0 {
            Point3d::from_homogeneous(lp1 + t * lp2)
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
        let dx = f32::abs(point.x - self.center.x);
        let dy = f32::abs(point.y - self.center.y);
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

    pub fn min_x(self) -> f32 { self.center.x - self.radius }

    pub fn min_y(self) -> f32 { self.center.y - self.radius }

    pub fn min_z(self) -> f32 { self.center.z }

    pub fn max_x(self) -> f32 { self.center.x + self.radius }

    pub fn max_y(self) -> f32 { self.center.y + self.radius }

    pub fn max_z(self) -> f32 { self.center.z }
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
            if bbox.p1.x < acc.p1.x {
                acc.p1.x = bbox.p1.x;
            }
            if bbox.p1.y < acc.p1.y {
                acc.p1.y = bbox.p1.y;
            }
            if bbox.p1.z < acc.p1.z {
                acc.p1.z = bbox.p1.z;
            }
            if bbox.p2.x > acc.p2.x {
                acc.p2.x = bbox.p2.x;
            }
            if bbox.p2.y > acc.p2.y {
                acc.p2.y = bbox.p2.y;
            }
            if bbox.p2.z > acc.p2.z {
                acc.p2.z = bbox.p2.z;
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
        .for_each(|tri| tri.translate(-bounds.p1.x, -bounds.p1.y, -bounds.p2.z));
}
