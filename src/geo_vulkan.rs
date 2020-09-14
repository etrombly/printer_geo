//! # Geo_Vulkan
//!
//! Conversions of geo types to be compatible with vulkan

use crate::{
    compute::{intersect_tris, Vk},
    geo::*,
};
use float_cmp::approx_eq;
#[cfg_attr(feature = "with_pyo3", pyclass)]
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    cmp::{Ordering, PartialEq},
    default::Default,
    fmt,
    ops::{Add, Index, Sub},
};

pub type PointsVk = Vec<PointVk>;

#[cfg_attr(feature = "python", pyclass)]
#[repr(C)]
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

    pub fn cross(&self, other: &PointVk) -> PointVk {
        PointVk::new(
            self[1] * other[2] - self[2] * other[1],
            self[2] * other[0] - self[0] * other[2],
            self[0] * other[1] - self[1] * other[0],
        )
    }

    pub fn dot(&self, other: &PointVk) -> f32 {
        self[0] * other[0] + self[1] * other[1] + self[2] * other[2]
    }
}

impl Eq for PointVk {}

impl PartialEq for PointVk {
    fn eq(&self, other: &Self) -> bool {
        approx_eq!(f32, self[0], other[0], ulps = 3)
            && approx_eq!(f32, self[1], other[1], ulps = 3)
            && approx_eq!(f32, self[2], other[2], ulps = 3)
    }
}

impl Ord for PointVk {
    fn cmp(&self, other: &Self) -> Ordering {
        if self[0] > other[0] {
            Ordering::Greater
        } else if approx_eq!(f32, self[0], other[0], ulps = 3) {
            if self[1] > other[1] {
                Ordering::Greater
            } else if approx_eq!(f32, self[1], other[1], ulps = 3) {
                if self[2] > other[2] {
                    Ordering::Greater
                } else if approx_eq!(f32, self[2], other[2], ulps = 3) {
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

impl Index<usize> for PointVk {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output { &self.position[index] }
}

impl fmt::Debug for PointVk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PointVk")
            .field("x", &self[0])
            .field("y", &self[1])
            .field("z", &self[2])
            .finish()
    }
}

impl Add<PointVk> for PointVk {
    type Output = PointVk;

    fn add(self, other: PointVk) -> PointVk {
        PointVk {
            position: [self[0] + other[0], self[1] + other[1], self[2] + other[2]],
        }
    }
}

impl Sub<PointVk> for PointVk {
    type Output = PointVk;

    fn sub(self, other: PointVk) -> PointVk {
        PointVk {
            position: [self[0] - other[0], self[1] - other[1], self[2] - other[2]],
        }
    }
}

#[cfg_attr(feature = "python", pyclass)]
#[derive(Default, Debug, Copy, Clone)]
pub struct TriangleVk {
    pub p1: PointVk,
    pub p2: PointVk,
    pub p3: PointVk,
}

pub type TrianglesVk = Vec<TriangleVk>;

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

    pub fn intersect_ray(&self, O: &PointVk) -> Option<f32> {
        let EPSILON = 0.001;
        // hard code the direction as casting up
        let D = PointVk::new(0., 0., 1.);
        // standard ray/triangle intersection, modified from a stackoverflow question
        let e1 = self.p2 - self.p1;
        let e2 = self.p3 - self.p1;
        let P = D.cross(&e2);
        let det = e1.dot(&P);
        if det > -EPSILON && det < EPSILON {
            return None;
        }
        let inv_det = 1.0 / det;
        let T = *O - self.p1;
        let u = T.dot(&P) * inv_det;
        if u < 0. || u > 1. {
            return None;
        }
        let Q = T.cross(&e1);
        let v = D.dot(&Q) * inv_det;
        if v < 0. || u + v > 1. {
            return None;
        }
        let t = e2.dot(&Q) * inv_det;
        if t > EPSILON {
            return Some(O[2] + t * D[2]);
        }
        return None;
    }

    pub fn filter_row(&self, bound: LineVk) -> bool {
        let bound1 = LineVk {
            p1: bound.p1,
            p2: PointVk {
                position: [bound.p1[0], bound.p2[1], 0.],
            },
        };
        let bound2 = LineVk {
            p1: PointVk {
                position: [bound.p2[0], bound.p1[1], 0.],
            },
            p2: bound.p2,
        };
        (self.p1[0] >= bound.p1[0] && self.p1[0] <= bound.p2[0])
            || (self.p2[0] >= bound.p1[0] && self.p2[0] <= bound.p2[0])
            || (self.p3[0] >= bound.p1[0] && self.p3[0] <= bound.p2[0])
            || (LineVk {
                p1: self.p1,
                p2: self.p2,
            })
            .intersect2d(bound1)
            || (LineVk {
                p1: self.p2,
                p2: self.p3,
            })
            .intersect2d(bound1)
            || (LineVk {
                p1: self.p1,
                p2: self.p3,
            })
            .intersect2d(bound1)
            || (LineVk {
                p1: self.p1,
                p2: self.p2,
            })
            .intersect2d(bound2)
            || (LineVk {
                p1: self.p2,
                p2: self.p3,
            })
            .intersect2d(bound2)
            || (LineVk {
                p1: self.p1,
                p2: self.p3,
            })
            .intersect2d(bound2)
    }
}

impl From<&Triangle3d> for TriangleVk {
    fn from(tri: &Triangle3d) -> Self {
        TriangleVk {
            p1: PointVk::new(tri.p1.x, tri.p1.y, tri.p1.z),
            p2: PointVk::new(tri.p2.x, tri.p2.y, tri.p2.z),
            p3: PointVk::new(tri.p3.x, tri.p3.y, tri.p3.z),
        }
    }
}

pub fn to_tri_vk(tris: &[Triangle3d]) -> Vec<TriangleVk> {
    tris.iter().map(|tri| tri.into()).collect()
}

#[cfg_attr(feature = "python", pyclass)]
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
        point[0] >= self.p1[0]
            && point[0] <= self.p2[0]
            && point[1] >= self.p1[1]
            && point[1] <= self.p2[1]
    }

    pub fn intersect2d(self, other: Self) -> bool {
        let a1 = self.p2[1] - self.p1[1];
        let b1 = self.p1[0] - self.p2[0];
        let c1 = a1 * self.p1[0] + b1 * self.p1[1];

        let a2 = other.p2[1] - other.p1[1];
        let b2 = other.p1[0] - other.p2[0];
        let c2 = a2 * other.p1[0] + b2 * other.p1[1];

        let delta = a1 * b2 - a2 * b1;
        let x = (b2 * c1 - b1 * c2) / delta;
        let y = (a1 * c2 - a2 * c1) / delta;
        delta != 0.0
            && self.p1[0].min(self.p2[0]) <= x
            && x <= self.p1[0].max(self.p2[0])
            && self.p1[1].min(self.p2[1]) <= y
            && y <= self.p1[1].max(self.p2[1])
    }
}

impl From<&Line3d> for LineVk {
    fn from(line: &Line3d) -> Self {
        LineVk {
            p1: PointVk::new(line.p1.x, line.p1.y, line.p1.z),
            p2: PointVk::new(line.p2.x, line.p2.y, line.p2.z),
        }
    }
}

#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone, Copy, Debug)]
pub struct CircleVk {
    pub center: PointVk,
    pub radius: f32,
}

impl CircleVk {
    pub fn new(center: PointVk, radius: f32) -> CircleVk { CircleVk { center, radius } }

    pub fn in_2d_bounds(&self, point: &PointVk) -> bool {
        (point[0] - self.center[0]).powi(2) + (point[1] - self.center[1]).powi(2)
            <= self.radius.powi(2)
    }

    pub fn bbox(self) -> LineVk {
        LineVk {
            p1: PointVk::new(
                self.min_x() - self.center[0] - self.radius * 2.,
                self.min_y() - self.center[1] - self.radius * 2.,
                self.min_z() - self.center[2],
            ),
            p2: PointVk::new(
                self.max_x() - self.center[0] + self.radius * 2.,
                self.max_y() - self.center[1] + self.radius * 2.,
                self.max_z() - self.center[2],
            ),
        }
    }

    fn min_x(self) -> f32 { self.center[0] - self.radius }

    fn min_y(self) -> f32 { self.center[1] - self.radius }

    fn min_z(self) -> f32 { self.center[2] }

    fn max_x(self) -> f32 { self.center[0] + self.radius }

    fn max_y(self) -> f32 { self.center[1] + self.radius }

    fn max_z(self) -> f32 { self.center[2] }
}

#[cfg_attr(feature = "python", pyclass)]
/// Tool for CAM operations, represented as a point cloud
#[derive(Default, Clone)]
pub struct Tool {
    pub bbox: LineVk,
    pub points: Vec<PointVk>,
}

impl Tool {
    pub fn new_endmill(radius: f32, scale: f32) -> Tool {
        let circle = CircleVk::new(PointVk::new(0., 0., 0.), radius);
        let points = Tool::circle_to_points(&circle, scale);
        Tool {
            bbox: circle.bbox(),
            points,
        }
    }

    pub fn new_v_bit(radius: f32, angle: f32, scale: f32) -> Tool {
        let circle = CircleVk::new(PointVk::new(0., 0., 0.), radius);
        let percent = (90. - (angle / 2.)).to_radians().tan();
        let points = Tool::circle_to_points(&circle, scale);
        let points = points
            .iter()
            .map(|point| {
                let distance = (point[0].powi(2) + point[1].powi(2)).sqrt();
                let z = distance * percent;
                PointVk::new(point[0], point[1], z)
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
        let circle = CircleVk::new(PointVk::new(0., 0., 0.), radius);
        let points = Tool::circle_to_points(&circle, scale);
        let points = points
            .iter()
            .map(|point| {
                let distance = (point[0].powi(2) + point[1].powi(2)).sqrt();
                let z = if distance > 0. {
                    // 65. is the angle
                    radius + (-radius * (65. / (radius / distance)).to_radians().cos())
                } else {
                    0.
                };
                PointVk::new(point[0], point[1], z)
            })
            .collect();
        Tool {
            bbox: circle.bbox(),
            points,
        }
    }

    pub fn circle_to_points(circle: &CircleVk, scale: f32) -> Vec<PointVk> {
        let mut points: Vec<PointVk> = (0..=(circle.radius * scale) as i32)
            .flat_map(|x| {
                (0..=(circle.radius * scale) as i32).flat_map(move |y| {
                    vec![
                        PointVk::new(-x as f32 / scale, y as f32 / scale, 0.0),
                        PointVk::new(-x as f32 / scale, -y as f32 / scale, 0.0),
                        PointVk::new(x as f32 / scale, -y as f32 / scale, 0.0),
                        PointVk::new(x as f32 / scale, y as f32 / scale, 0.0),
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

pub fn generate_grid(bounds: &Line3d, scale: &f32) -> Vec<PointsVk> {
    let max_x = (bounds.p2.x * scale) as i32;
    let min_x = (bounds.p1.x * scale) as i32;
    let max_y = (bounds.p2.y * scale) as i32;
    let min_y = (bounds.p1.y * scale) as i32;
    (min_x..=max_x)
        .map(|x| {
            (min_y..=max_y)
                .map(move |y| PointVk::new(x as f32 / scale, y as f32 / scale, bounds.p1.z))
                .collect::<Vec<_>>()
        })
        .collect()
}

pub fn generate_columns(
    grid: &[PointsVk],
    bounds: &Line3d,
    resolution: &f32,
    scale: &f32,
) -> Vec<LineVk> {
    let max_y = (bounds.p2.y * scale) as i32;
    let min_y = (bounds.p1.y * scale) as i32;
    grid.iter()
        .map(|x| {
            // bounding box for this column
            LineVk {
                p1: PointVk::new(x[0][0] - resolution, min_y as f32 / scale, 0.),
                p2: PointVk::new(x[0][0] + resolution, max_y as f32 / scale, 0.),
            }
        })
        .collect()
}

pub fn generate_columns_chunks(
    grid: &[PointsVk],
    bounds: &Line3d,
    resolution: &f32,
    scale: &f32,
) -> Vec<LineVk> {
    let max_y = (bounds.p2.y * scale) as i32;
    let min_y = (bounds.p1.y * scale) as i32;
    grid.chunks(10)
        .map(|x| {
            let last = x.len() - 1;
            // bounding box for this column
            LineVk {
                p1: PointVk::new(x[0][0][0] - resolution, min_y as f32 / scale, 0.),
                p2: PointVk::new(x[last][0][0] + resolution, max_y as f32 / scale, 0.),
            }
        })
        .collect()
}

pub fn generate_heightmap(grid: &[PointsVk], partition: &[TrianglesVk], vk: &Vk) -> Vec<PointsVk> {
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
    grid: &[Vec<PointVk>],
    partition: &[Vec<TriangleVk>],
    vk: &Vk,
) -> Vec<Vec<PointVk>> {
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
    heightmap: &[PointsVk],
    bounds: &Line3d,
    tool: &Tool,
    radius: &f32,
    stepover: &f32,
    scale: &f32,
) -> Vec<PointsVk> {
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
                    let x_offset = (x as f32 + (tpoint[0] * scale)).round() as i32;
                    let y_offset = (y as f32 + (tpoint[1] * scale)).round() as i32;
                    if x_offset < columns as i32
                        && x_offset >= 0
                        && y_offset < rows as i32
                        && y_offset >= 0
                    {

                        heightmap[x_offset as usize][y_offset as usize][2]
                            - tpoint[2]
                    } else {
                        bounds.p1.z
                    }
                })
                .fold(f32::NAN, f32::max); // same as calling max on all the values for this tool to find the heighest
            PointVk::new(x as f32 / scale, y as f32 / scale, max)
        })
        .collect()
})
.collect()
}

pub fn generate_layers(
    toolpath: &[PointsVk],
    bounds: &Line3d,
    stepdown: &f32,
) -> Vec<Vec<PointsVk>> {
    let steps = ((bounds.p2.z - bounds.p1.z) / stepdown) as u64;
    (1..steps + 1)
        .map(|step| {
            toolpath
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|x| match step as f32 * -stepdown {
                            z if z > x[2] => PointVk::new(x[0], x[1], z),
                            _ => *x,
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

pub fn generate_gcode(layers: &[Vec<PointsVk>], bounds: &Line3d) -> String {
    let mut last = layers[0][0][0];
    let mut output = format!("G1 Z{:.2} F300\n", bounds.p2.z);

    for layer in layers {
        output.push_str(&format!(
            "G0 X{:.2} Y{:.2}\nG0 Z{:.2}\n",
            layer[0][0][0], layer[0][0][1], layer[0][0][2]
        ));
        for row in layer {
            output.push_str(&format!(
                "G0 X{:.2} Y{:.2}\nG0 Z{:.2}\n",
                row[0][0], row[0][1], row[0][2]
            ));
            for point in row {
                if !approx_eq!(f32, last[0], point[0], ulps = 2)
                    || !approx_eq!(f32, last[2], point[2], ulps = 2)
                {
                    output.push_str(&format!(
                        "G1 X{:.2} Y{:.3} Z{:.2}\n",
                        point[0], point[1], point[2]
                    ));
                }
                last = *point;
            }
            output.push_str(&format!(
                "G1 X{:?} Y{:?} Z{:?}\nG0 Z{:.2}\n",
                last[0], last[1], last[2], bounds.p2.z
            ));
        }
    }
    output
}

pub fn intersect_tris_fallback(tris: &[TriangleVk], points: &[PointVk]) -> Vec<PointVk> {
    points
        .par_iter()
        .map(|point| {
            PointVk::new(
                point[0],
                point[1],
                tris.iter()
                    .filter_map(|tri| tri.intersect_ray(&point))
                    .fold(f32::NAN, f32::max),
            )
        })
        .collect()
}
