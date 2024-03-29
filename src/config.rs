use crate::tool::Tool;
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
            ToolType::VBit => Tool::new_v_bit(radius, angle.expect("V-Bit requires tool angle"), scale),
            ToolType::Ball => Tool::new_ball(radius, scale),
        }
    }
}

fn parse_stepover(src: &str) -> Result<f32, ConfigError> {
    let stepover = src.parse::<f32>()?;
    if (1. ..=100.).contains(&stepover) {
        Ok(stepover)
    } else {
        Err(ConfigError::StepOver)
    }
}

fn parse_angle(src: &str) -> Result<f32, ConfigError> {
    let angle = src.parse::<f32>()?;
    if (1. ..=180.).contains(&angle) {
        Ok(angle)
    } else {
        Err(ConfigError::Angle)
    }
}

fn parse_resolution(src: &str) -> Result<f32, ConfigError> {
    let resolution = src.parse::<f32>()?;
    if (0.001..=1.).contains(&resolution) {
        Ok(resolution)
    } else {
        Err(ConfigError::Resolution)
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
    pub diameter: Option<f32>,

    #[structopt(long)]
    pub debug: bool,

    #[structopt(short, long, required_if("tool", "vbit"), parse(try_from_str = parse_angle))]
    pub angle: Option<f32>,

    #[structopt(long, default_value="90", parse(try_from_str = parse_stepover))]
    pub stepover: f32,

    #[structopt(short, long, default_value="0.05", parse(try_from_str = parse_resolution))]
    pub resolution: f32,

    #[structopt(short, long, default_value = "340282300000000000000000000000000000000")]
    pub stepdown: f32,

    #[structopt(short, long, possible_values = &ToolType::variants(), default_value="endmill", case_insensitive = true)]
    pub tool: ToolType,

    #[structopt(long, parse(from_os_str))]
    pub toolfile: Option<PathBuf>,

    #[structopt(long)]
    pub tooltable: Option<usize>,

    #[structopt(long)]
    pub toolnumber: Option<usize>,

    #[structopt(long)]
    pub speed: f32,

    #[structopt(long)]
    pub feedrate: f32,
}
