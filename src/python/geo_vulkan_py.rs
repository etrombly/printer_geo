use crate::{geo_vulkan::*, python::geo_py::*};
use pyo3::prelude::*;

#[pyclass]
pub struct TrianglesVkPy {
    pub(crate) inner: Vec<TriangleVk>,
}

impl From<Vec<TriangleVk>> for TrianglesVkPy {
    fn from(tris: Vec<TriangleVk>) -> Self { TrianglesVkPy { inner: tris } }
}

#[pyclass]
pub struct LineVkPy {
    pub(crate) inner: LineVk,
}

#[pyclass]
pub struct ToolPy {
    pub(crate) inner: Tool,
}

impl From<Tool> for ToolPy {
    fn from(tool: Tool) -> Self { ToolPy { inner: tool } }
}

#[pymodule]
pub fn geo_vulkan(py: Python, m: &PyModule) -> PyResult<()> {
    // PyO3 aware function. All of our Python interfaces could be declared in a
    // separate module. Note that the `#[pyfn()]` annotation automatically
    // converts the arguments from Python objects to Rust values, and the Rust
    // return value back into a Python object. The `_py` argument represents
    // that we're holding the GIL.
    #[pyfn(m, "to_tri_vk")]
    fn to_tri_vk_py(_py: Python, tris: TrianglesPy) -> PyResult<TrianglesVkPy> {
        Ok(to_tri_vk(&tris.inner).into())
    }

    #[pyfn(m, "new_endmill")]
    fn new_endmill_py(_py: Python, radius: f32, scale: f32) -> PyResult<ToolPy> {
        Ok(Tool::new_endmill(radius, scale).into())
    }

    #[pyfn(m, "new_v_bit")]
    fn new_v_bit_py(_py: Python, radius: f32, angle: f32, scale: f32) -> PyResult<ToolPy> {
        Ok(Tool::new_v_bit(radius, angle, scale).into())
    }

    #[pyfn(m, "new_ball")]
    fn new_ball_py(_py: Python, radius: f32, angle: f32, scale: f32) -> PyResult<ToolPy> {
        Ok(Tool::new_ball(radius, scale).into())
    }

    Ok(())
}
