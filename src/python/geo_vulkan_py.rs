use crate::{
    geo_vulkan::{to_tri_vk, LineVk, PointVk, Tool, TriangleVk},
    python::geo_py::TrianglesPy,
};
use pyo3::prelude::*;

#[pyclass]
pub struct TrianglesVkPy {
    pub(crate) inner: Vec<TriangleVk>,
}

impl From<Vec<TriangleVk>> for TrianglesVkPy {
    fn from(tris: Vec<TriangleVk>) -> Self { TrianglesVkPy { inner: tris } }
}

#[pyclass]
pub struct LinesVkPy {
    pub(crate) inner: Vec<LineVk>,
}

impl From<Vec<LineVk>> for LinesVkPy {
    fn from(lines: Vec<LineVk>) -> Self { LinesVkPy { inner: lines } }
}

#[pyclass]
pub struct LineVkPy {
    pub(crate) inner: LineVk,
}

#[pyclass]
pub struct PointVkPy {
    pub(crate) inner: PointVk,
}

#[pyclass]
pub struct PointsVkPy {
    pub(crate) inner: Vec<PointVk>,
}

impl From<Vec<PointVk>> for PointsVkPy {
    fn from(points: Vec<PointVk>) -> Self { PointsVkPy { inner: points } }
}

#[pyclass]
pub struct ToolPy {
    pub(crate) inner: Tool,
}

impl From<Tool> for ToolPy {
    fn from(tool: Tool) -> Self { ToolPy { inner: tool } }
}

#[pymodule]
pub fn geo_vulkan(_py: Python, m: &PyModule) -> PyResult<()> {
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
    fn new_ball_py(_py: Python, radius: f32, scale: f32) -> PyResult<ToolPy> {
        Ok(Tool::new_ball(radius, scale).into())
    }

    Ok(())
}
