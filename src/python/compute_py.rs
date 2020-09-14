use crate::{
    compute::{intersect_tris, partition_tris, ComputeError, Vk, VkError},
    geo_vulkan::{LineVk, PointVk, PointsVk, TriangleVk, TrianglesVk},
};
use pyo3::{exceptions::PyException, prelude::*};

impl From<VkError> for PyErr {
    fn from(err: VkError) -> Self { PyException::new_err(err.to_string()) }
}

impl From<ComputeError> for PyErr {
    fn from(err: ComputeError) -> Self { PyException::new_err(err.to_string()) }
}

#[pymodule]
pub fn compute(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "init_vk")]
    fn init_vk_py(_py: Python) -> PyResult<Vk> { Ok(Vk::new()?) }

    #[pyfn(m, "intersect_tris")]
    fn intersect_tris_py(
        _py: Python,
        tris: Vec<TriangleVk>,
        points: Vec<PointVk>,
        vk: &Vk,
    ) -> PyResult<PointsVk> {
        Ok(intersect_tris(&tris, &points, &vk)?)
    }

    #[pyfn(m, "partition_tris")]
    fn partition_tris_py(
        _py: Python,
        tris: Vec<TriangleVk>,
        columns: Vec<LineVk>,
        vk: &Vk,
    ) -> PyResult<Vec<TrianglesVk>> {
        Ok(partition_tris(&tris, &columns, vk)?)
    }

    Ok(())
}
