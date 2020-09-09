use pyo3::{prelude::*, wrap_pyfunction, wrap_pymodule};

pub mod compute;
pub mod geo;
pub mod geo_vulkan;
pub mod stl;

use crate::{compute::*, stl::*};

#[pymodule]
fn printer_geo(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(compute))?;
    m.add_wrapped(wrap_pymodule!(stl))?;
    Ok(())
}
