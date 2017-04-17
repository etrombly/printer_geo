use std::ops::{Add, Sub, Mul};
use std::cmp::Ordering;

const  PRECISION: f32 = 0.001;

pub trait Intersect<RHS = Self> {
    type Output;
    fn intersect(self, rhs: RHS) -> Self::Output;
}

pub trait Bounds{
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
    pub fn new (x: f32, y: f32, z: f32) -> Point3d{
        Point3d{x: x, y: y, z: z}
    }

    pub fn sum (&self) -> f32 {
       self.x + self.y + self.z
    }

    pub fn dot (self, other: Point3d) -> f32 {
        (self * other).sum()
    }

    pub fn cross (&self, other: &Point3d) -> Point3d {
        Point3d::new(self.y * other.z - self.z * other.y,
                   self.z * other.x - self.x * other.z,
                   self.x * other.y - self.y * other.x)
     }

     pub fn is_infinite(&self) -> bool {
         self.x.is_infinite() || self.y.is_infinite() || self.z.is_infinite()
     }

     pub fn is_nan(&self) -> bool {
        !self.is_infinite() && (self.x.is_nan() || self.y.is_nan() || self.z.is_nan())
     }

     pub fn distance(&self, other: &Point3d) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z - other.z).powi(2)).sqrt()
     }
}

impl Add<Point3d> for Point3d {
    type Output = Point3d;

    fn add(self, other: Point3d) -> Point3d {
        Point3d {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z}
    }
}

impl Sub<Point3d> for Point3d {
    type Output = Point3d;

    fn sub(self, other: Point3d) -> Point3d {
        Point3d {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z}
    }
}

impl Mul<Point3d> for Point3d {
    type Output = Point3d;

    fn mul(self, other: Point3d) -> Point3d {
        Point3d {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z}
    }
}

impl Mul<f32> for Point3d {
    type Output = Point3d;

    fn mul(self, f: f32) -> Point3d {
        Point3d {
            x: self.x * f,
            y: self.y * f,
            z: self.z * f}
    }
}








#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Line3d {
    pub p1: Point3d,
    pub p2: Point3d,
}

impl Line3d {
    pub fn new(p1: (f32, f32, f32), p2: (f32, f32, f32)) -> Line3d {
        Line3d{
             p1: Point3d::new(p1.0, p1.1, p1.2),
             p2: Point3d::new(p2.0, p2.1, p2.2),
        }
    }

    pub fn from_points(p1: &Point3d, p2: &Point3d) -> Line3d {
        Line3d{
            p1: *p1,
            p2: *p2,
        }
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
        if answer.is_infinite() ||
          (fac < 0.0 || fac > 1.0) {
            None
        } else if answer.is_nan() {
            Some(Shape::Line3d(self))
        } else {
            Some(Shape::Point3d(answer))
        }
    }
}

impl Bounds for Line3d{
  fn bbox(self) -> Line3d {
    Line3d {
      p1: Point3d { x: self.min_x(),
                y: self.min_y(),
                z: self.min_z()},
      p2: Point3d { x: self.max_x(),
                y: self.max_y(),
                z: self.max_z()},
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
        let lines = vec![Line3d::from_points(&self.p1, &self.p2),
                         Line3d::from_points(&self.p2, &self.p3),
                         Line3d::from_points(&self.p3, &self.p1)];
        let mut results: Vec<Shape> = Vec::new();
        for line in lines {
            if let Some(answer) = line.intersect(plane) {
                results.push(answer);
            }
        }
        if results.len() == 2 {
            if let (Shape::Point3d(p1), Shape::Point3d(p2)) = (results[0], results[1]) {
                return Some(Shape::Line3d(Line3d::from_points(&p1, &p2)))
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
        Plane{
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
    pub fn is_line(&self) ->  bool {
        match *self {
            Shape::Line3d(_) => true,
            _ => false,
        }
    }
}
