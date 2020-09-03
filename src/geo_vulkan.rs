use crate::geo::*;
use std::{default::Default, fmt, ops::Add};

// Compute buffers are 16 byte aligned
#[repr(C, align(16))]
#[derive(Default, Copy, Clone)]
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
}

#[derive(Clone, Copy, Debug)]
pub struct CircleVk {
    pub center: PointVk,
    pub radius: f32,
}

impl CircleVk {
    pub fn new(center: PointVk, radius: f32) -> CircleVk { CircleVk { center, radius } }

    pub fn in_2d_bounds(&self, point: &PointVk) -> bool {
        let dx = f32::abs(point.position[0] - self.center.position[0]);
        let dy = f32::abs(point.position[1] - self.center.position[1]);
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
        let circle = CircleVk::new(PointVk::new(radius, radius, 0.0), radius);
        let points: Vec<PointVk> = (0..=(radius * 20.0) as i32)
            .flat_map(|x| {
                (0..=(radius * 20.0) as i32)
                    .map(move |y| PointVk::new(x as f32 / 10.0, y as f32 / 10.0, 0.0))
            })
            .filter(|x| circle.in_2d_bounds(&x))
            .map(|x| {
                PointVk::new(
                    x.position[0] - radius,
                    x.position[1] - radius,
                    x.position[2],
                )
            })
            .collect();
        Tool {
            bbox: circle.bbox(),
            points,
        }
    }

    pub fn new_v_bit(radius: f32, angle: f32) -> Tool {
        let circle = CircleVk::new(PointVk::new(radius, radius, 0.0), radius);
        let percent = (90. - (angle / 2.)).to_radians().tan();
        let points: Vec<PointVk> = (0..=(radius * 20.0) as i32)
            .flat_map(|x| {
                (0..=(radius * 20.0) as i32).filter_map(move |y| {
                    let x = x as f32 / 10.0;
                    let y = y as f32 / 10.0;
                    if circle.in_2d_bounds(&PointVk::new(x, y, 0.)) {
                        let x = x - radius;
                        let y = y - radius;
                        let distance = (x.powi(2) + y.powi(2)).sqrt();
                        let z = distance * percent;
                        Some(PointVk::new(x, y, z))
                    } else {
                        None
                    }
                })
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
        let circle = CircleVk::new(PointVk::new(radius, radius, 0.0), radius);
        let points: Vec<PointVk> = (0..=(radius * 20.0) as i32)
            .flat_map(|x| {
                (0..=(radius * 20.0) as i32).filter_map(move |y| {
                    let x = x as f32 / 10.0;
                    let y = y as f32 / 10.0;
                    if circle.in_2d_bounds(&PointVk::new(x, y, 0.)) {
                        let x = x - radius;
                        let y = y - radius;
                        let distance = (x.powi(2) + y.powi(2)).sqrt();
                        let z = if distance > 0. {
                            // 55. is the angle
                            radius + (-radius * (55. / (radius / distance)).to_radians().cos())
                        } else {
                            0.
                        };
                        Some(PointVk::new(x, y, z))
                    } else {
                        None
                    }
                })
            })
            .collect();
        Tool {
            bbox: circle.bbox(),
            points,
        }
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