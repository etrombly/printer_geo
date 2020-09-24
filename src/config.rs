use crate::geo::Tool;
use clap::arg_enum;
use std::path::PathBuf;
use structopt::StructOpt;
use thiserror::Error;

#[derive(Error, Debug)]
/// Error types for vulkan devices
pub enum ConfigError {
    #[error("stepover must be between 1 and 100")]
    StepOver,
    #[error("angle must be between 1 and 180")]
    Angle,
    #[error("resolution must be between 0.001 and 1.0")]
    Resolution,
    #[error("Not a decimal number")]
    ParseFloat(#[from] std::num::ParseFloatError),
    #[error("Error parsing config")]
    Clap(#[from] clap::Error),
}

arg_enum! {
    #[derive(Debug)]
    pub enum ToolType {
        Endmill,
        VBit,
        Ball
    }
}

impl ToolType {
    pub fn create(&self, radius: f32, angle: Option<f32>, scale: f32) -> Tool {
        match self {
            ToolType::Endmill => Tool::new_endmill(radius, scale),
            ToolType::VBit => {
                Tool::new_v_bit(radius, angle.expect("V-Bit requires tool angle"), scale)
            },
            ToolType::Ball => Tool::new_ball(radius, scale),
        }
    }
}

fn parse_stepover(src: &str) -> Result<f32, ConfigError> {
    let stepover = src.parse::<f32>()?;
    if stepover < 1. || stepover > 100. {
        Err(ConfigError::StepOver)
    } else {
        Ok(stepover)
    }
}

fn parse_angle(src: &str) -> Result<f32, ConfigError> {
    let angle = src.parse::<f32>()?;
    if angle < 1. || angle > 180. {
        Err(ConfigError::Angle)
    } else {
        Ok(angle)
    }
}

fn parse_resolution(src: &str) -> Result<f32, ConfigError> {
    let resolution = src.parse::<f32>()?;
    if resolution < 0.001 || resolution > 1. {
        Err(ConfigError::Resolution)
    } else {
        Ok(resolution)
    }
}

// set up program arguments
#[derive(Debug, StructOpt)]
#[structopt(name = "printer_geo")]
pub struct Opt {
    #[structopt(short, long, parse(from_os_str))]
    pub input: PathBuf,

    #[structopt(short, long, parse(from_os_str))]
    pub output: PathBuf,

    #[structopt(long, parse(from_os_str))]
    pub heightmap: Option<PathBuf>,

    #[structopt(long, parse(from_os_str))]
    pub restmap: Option<PathBuf>,

    #[structopt(short, long)]
    pub diameter: f32,

    #[structopt(long)]
    pub debug: bool,

    #[structopt(short, long, required_if("tool", "vbit"), parse(try_from_str = parse_angle))]
    pub angle: Option<f32>,

    #[structopt(long, default_value="100", parse(try_from_str = parse_stepover))]
    pub stepover: f32,

    #[structopt(short, long, default_value="0.05", parse(try_from_str = parse_resolution))]
    pub resolution: f32,

    #[structopt(short, long)]
    pub stepdown: Option<f32>,

    #[structopt(short, long, possible_values = &ToolType::variants(), default_value="endmill", case_insensitive = true)]
    pub tool: ToolType,
}
