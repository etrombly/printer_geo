use crate::geo::*;
use serde::Deserialize;

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
/// Error types for vulkan compute operations
pub enum ToolError {
    #[error("Tool table not found in file")]
    Table,
    #[error("Tool number not found in table")]
    ToolIndex,
    #[error("Tool type not supported")]
    InvalidTool,
    #[error("Couldn't read tool table")]
    IO(#[from] std::io::Error),
    #[error("Error parsing tool table")]
    Json(#[from] serde_json::Error),
}

#[cfg_attr(feature = "python", pyclass)]
/// Tool for CAM operations, represented as a point cloud
#[derive(Default, Clone)]
pub struct Tool {
    pub bbox: Line3d,
    pub diameter: f32,
    pub points: Vec<Point3d>,
}

#[derive(Deserialize, Debug)]
pub struct ToolTable {
    #[serde(rename = "TableName")]
    table_name: String,
    #[serde(rename = "Tools")]
    tools: HashMap<String, ToolParams>,
}

#[derive(Deserialize, Debug)]
pub struct ToolParams {
    #[serde(rename = "CornerRadius")]
    corner_radius: Option<f32>,
    #[serde(rename = "cuttingEdgeAngle")]
    angle: Option<f32>,
    diameter: f32,
    #[serde(rename = "tooltype")]
    tool_type: String,
}

impl Tool {
    pub fn from_file<P: AsRef<Path>>(path: P, table: usize, tool_index: usize, scale: f32) -> Result<(Tool, f32), ToolError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let tool_index = tool_index.to_string();
    
        // Read the JSON contents of the file as an instance of `User`.
        let t: Vec<ToolTable> = serde_json::from_reader(reader)?;

        if t.len() < table {
            return Err(ToolError::Table);
        }
        if !t[table - 1].tools.contains_key(&tool_index){
            return Err(ToolError::ToolIndex);
        }
        let diameter = t[table - 1].tools[&tool_index].diameter;
        let tool = match t[table - 1].tools[&tool_index].tool_type.as_str() {
            "EndMill" => Tool::new_endmill(diameter, scale),
            // TODO: support angle for ball endmill
            "BallEndMill" => Tool::new_ball(diameter, scale),
            "Engraver" => Tool::new_v_bit(diameter, t[table - 1].tools[&tool_index].angle.unwrap(), scale),
            _ => return Err(ToolError::InvalidTool)
        };
        Ok((tool, diameter))
    }

    pub fn new_endmill(diameter: f32, scale: f32) -> Tool {
        let radius = diameter / 2.;
        let circle = Circle::new(Point3d::new(0., 0., 0.), radius);
        let points = Tool::circle_to_points(&circle, scale);
        Tool {
            bbox: circle.bbox(),
            diameter,
            points,
        }
    }

    pub fn new_v_bit(diameter: f32, angle: f32, scale: f32) -> Tool {
        let radius = diameter / 2.;
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
            diameter,
            points,
        }
    }

    // TODO: this approximates a ball end mill, probably want to generate the
    // geometry better
    pub fn new_ball(diameter: f32, scale: f32) -> Tool {
        let radius = diameter / 2.;
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
            diameter,
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