#[cfg(feature = "python")]
use pyo3::{prelude::*, wrap_pymodule};

pub mod bfs;
pub mod compute;
pub mod config;
pub mod geo;
//pub mod geo_vulkan;
#[cfg(feature = "python")]
pub mod python;
pub mod stl;

#[cfg(feature = "python")]
use crate::{python::compute_py::*, python::config_py::*, python::geo_py::*, python::stl_py::*};

#[cfg(feature = "python")]
#[pymodule]
fn printer_geo(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(compute))?;
    m.add_wrapped(wrap_pymodule!(stl))?;
    m.add_wrapped(wrap_pymodule!(geo))?;
    m.add_wrapped(wrap_pymodule!(config))?;
    Ok(())
}
