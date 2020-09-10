#[cfg(feature = "python")]
use pyo3::{prelude::*, wrap_pymodule};

pub mod compute;
pub mod geo;
pub mod geo_vulkan;
pub mod stl;
#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "python")]
use crate::{python::compute_py::*, python::stl_py::*, python::geo_py::*, python::geo_vulkan_py::*};

#[cfg(feature = "python")]
#[pymodule]
fn printer_geo(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(compute))?;
    m.add_wrapped(wrap_pymodule!(stl))?;
    m.add_wrapped(wrap_pymodule!(geo))?;
    m.add_wrapped(wrap_pymodule!(geo_vulkan))?;
    Ok(())
}
