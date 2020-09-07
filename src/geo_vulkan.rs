use crate::geo::*;
use float_cmp::approx_eq;
use serde::{Deserialize, Serialize};
use std::{
    cmp::{Ordering, PartialEq},
    default::Default,
    fmt,
    ops::Add,
};

// Compute buffers are 16 byte aligned
#[repr(C, align(16))]
#[derive(Default, Serialize, Deserialize, Copy, Clone)]
pub struct PointVk {
    pub position: [f32; 3],
}

impl PointVk {
    pub fn new(x: f32, y: f32, z: f32) -> PointVk {
        PointVk {
            position: [x, y, z],
        }
    }
}

impl Eq for PointVk {}

impl PartialEq for PointVk {
    fn eq(&self, other: &Self) -> bool {
        approx_eq!(f32, self.position[0], other.position[0], ulps = 2)
            && approx_eq!(f32, self.position[1], other.position[1], ulps = 2)
            && approx_eq!(f32, self.position[2], other.position[2], ulps = 2)
    }
}

impl Ord for PointVk {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.position[0] > other.position[0] {
            Ordering::Greater
        } else if approx_eq!(f32, self.position[0], other.position[0], ulps = 2) {
            if self.position[1] > other.position[1] {
                Ordering::Greater
            } else if approx_eq!(f32, self.position[1], other.position[1], ulps = 2) {
                if self.position[2] > other.position[2] {
                    Ordering::Greater
                } else if approx_eq!(f32, self.position[2], other.position[2], ulps = 2) {
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

impl PartialOrd for PointVk {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

impl fmt::Debug for PointVk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PointVk")
            .field("x", &self.position[0])
            .field("y", &self.position[1])
            .field("z", &self.position[2])
            .finish()
    }
}

impl Add<PointVk> for PointVk {
    type Output = PointVk;

    fn add(self, other: PointVk) -> PointVk {
        PointVk {
            position: [
                self.position[0] + other.position[0],
                self.position[1] + other.position[1],
                self.position[2] + other.position[2],
            ],
        }
    }
}

#[derive(Default, Debug, Copy, Clone)]
pub struct TriangleVk {
    pub p1: PointVk,
    pub p2: PointVk,
    pub p3: PointVk,
}

impl TriangleVk {
    pub fn new(p1: (f32, f32, f32), p2: (f32, f32, f32), p3: (f32, f32, f32)) -> TriangleVk {
        TriangleVk {
            p1: PointVk::new(p1.0, p1.1, p1.2),
            p2: PointVk::new(p2.0, p2.1, p2.2),
            p3: PointVk::new(p3.0, p3.1, p3.2),
        }
    }

    pub fn in_2d_bounds(&self, bbox: &LineVk) -> bool {
        bbox.in_2d_bounds(&self.p1) || bbox.in_2d_bounds(&self.p2) || bbox.in_2d_bounds(&self.p3)
    }

    pub fn filter_row(&self, bound: LineVk) -> bool {
        (self.p1.position[0] >= bound.p1.position[0] && self.p1.position[0] <= bound.p2.position[0])
            || (self.p2.position[0] >= bound.p1.position[0]
                && self.p2.position[0] <= bound.p2.position[0])
            || (self.p3.position[0] >= bound.p1.position[0]
                && self.p3.position[0] <= bound.p2.position[0])
            || (LineVk {
                p1: self.p1,
                p2: self.p2,
            })
            .intersect2d(bound)
            || (LineVk {
                p1: self.p2,
                p2: self.p3,
            })
            .intersect2d(bound)
            || (LineVk {
                p1: self.p1,
                p2: self.p3,
            })
            .intersect2d(bound)
    }
}

#[derive(Default, Debug, Copy, Clone)]
pub struct LineVk {
    pub p1: PointVk,
    pub p2: PointVk,
}

impl LineVk {
    pub fn new(p1: (f32, f32, f32), p2: (f32, f32, f32)) -> LineVk {
        LineVk {
            p1: PointVk::new(p1.0, p1.1, p1.2),
            p2: PointVk::new(p2.0, p2.1, p2.2),
        }
    }

    pub fn in_2d_bounds(&self, point: &PointVk) -> bool {
        point.position[0] >= self.p1.position[0]
            && point.position[0] <= self.p2.position[0]
            && point.position[1] >= self.p1.position[1]
            && point.position[1] <= self.p2.position[1]
    }

    pub fn intersect2d(self, other: Self) -> bool {
        let a1 = self.p2.position[1] - self.p1.position[1];
        let b1 = self.p1.position[0] - self.p2.position[0];
        let c1 = a1 * self.p1.position[0] + b1 * self.p1.position[1];

        let a2 = other.p2.position[1] - other.p1.position[1];
        let b2 = other.p1.position[0] - other.p2.position[0];
        let c2 = a2 * other.p1.position[0] + b2 * other.p1.position[1];

        let delta = a1 * b2 - a2 * b1;
        let x = (b2 * c1 - b1 * c2) / delta;
        let y = (a1 * c2 - a2 * c1) / delta;
        delta != 0.0
            && self.p1.position[0].min(self.p2.position[0]) <= x
            && x <= self.p1.position[0].max(self.p2.position[0])
            && self.p1.position[1].min(self.p2.position[1]) <= y
            && y <= self.p1.position[1].max(self.p2.position[1])
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CircleVk {
    pub center: PointVk,
    pub radius: f32,
}

impl CircleVk {
    pub fn new(center: PointVk, radius: f32) -> CircleVk { CircleVk { center, radius } }

    pub fn in_2d_bounds(&self, point: &PointVk) -> bool {
        (point.position[0] - self.center.position[0]).powi(2)
            + (point.position[1] - self.center.position[1]).powi(2)
            <= self.radius.powi(2)
    }

    pub fn bbox(self) -> LineVk {
        LineVk {
            p1: PointVk::new(
                self.min_x() - self.center.position[0] - self.radius * 2.,
                self.min_y() - self.center.position[1] - self.radius * 2.,
                self.min_z() - self.center.position[2],
            ),
            p2: PointVk::new(
                self.max_x() - self.center.position[0] + self.radius * 2.,
                self.max_y() - self.center.position[1] + self.radius * 2.,
                self.max_z() - self.center.position[2],
            ),
        }
    }

    fn min_x(self) -> f32 { self.center.position[0] - self.radius }

    fn min_y(self) -> f32 { self.center.position[1] - self.radius }

    fn min_z(self) -> f32 { self.center.position[2] }

    fn max_x(self) -> f32 { self.center.position[0] + self.radius }

    fn max_y(self) -> f32 { self.center.position[1] + self.radius }

    fn max_z(self) -> f32 { self.center.position[2] }
}

#[derive(Default, Clone)]
pub struct Tool {
    pub bbox: LineVk,
    pub points: Vec<PointVk>,
}

impl Tool {
    pub fn new_endmill(radius: f32) -> Tool {
        let circle = CircleVk::new(PointVk::new(0., 0., 0.), radius);
        let points = Tool::circle_to_points(&circle);
        Tool {
            bbox: circle.bbox(),
            points,
        }
    }

    pub fn new_v_bit(radius: f32, angle: f32) -> Tool {
        let circle = CircleVk::new(PointVk::new(0., 0., 0.), radius);
        let percent = (90. - (angle / 2.)).to_radians().tan();
        let points = Tool::circle_to_points(&circle);
        let points = points
            .iter()
            .map(|point| {
                let distance = (point.position[0].powi(2) + point.position[1].powi(2)).sqrt();
                let z = distance * percent;
                PointVk::new(point.position[0], point.position[1], z)
            })
            .collect();
        Tool {
            bbox: circle.bbox(),
            points,
        }
    }

    // TODO: this approximates a ball end mill, probably want to generate the
    // geometry better
    pub fn new_ball(radius: f32) -> Tool {
        let circle = CircleVk::new(PointVk::new(0., 0., 0.), radius);
        let points = Tool::circle_to_points(&circle);
        let points = points
            .iter()
            .map(|point| {
                let distance = (point.position[0].powi(2) + point.position[1].powi(2)).sqrt();
                let z = if distance > 0. {
                    // 65. is the angle
                    radius + (-radius * (65. / (radius / distance)).to_radians().cos())
                } else {
                    0.
                };
                PointVk::new(point.position[0], point.position[1], z)
            })
            .collect();
        Tool {
            bbox: circle.bbox(),
            points,
        }
    }

    pub fn circle_to_points(circle: &CircleVk) -> Vec<PointVk> {
        let mut points: Vec<PointVk> = (0..=(circle.radius * 20.) as i32)
            .flat_map(|x| {
                (0..=(circle.radius * 20.) as i32).flat_map(move |y| {
                    vec![
                        PointVk::new(-x as f32 / 20., y as f32 / 20., 0.0),
                        PointVk::new(-x as f32 / 20., -y as f32 / 20., 0.0),
                        PointVk::new(x as f32 / 20., -y as f32 / 20., 0.0),
                        PointVk::new(x as f32 / 20., y as f32 / 20., 0.0),
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

pub fn to_tri_vk(tris: &[Triangle3d]) -> Vec<TriangleVk> {
    tris.iter()
        .map(|tri| TriangleVk {
            p1: PointVk::new(tri.p1.x, tri.p1.y, tri.p1.z),
            p2: PointVk::new(tri.p2.x, tri.p2.y, tri.p2.z),
            p3: PointVk::new(tri.p3.x, tri.p3.y, tri.p3.z),
        })
        .collect()
}

pub fn to_line3d(line: &LineVk) -> Line3d {
    Line3d {
        p1: Point3d::new(
            line.p1.position[0],
            line.p1.position[1],
            line.p1.position[2],
        ),
        p2: Point3d::new(
            line.p2.position[0],
            line.p2.position[1],
            line.p2.position[2],
        ),
    }
}
