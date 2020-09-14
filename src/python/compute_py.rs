use crate::{
    compute::{intersect_tris, partition_tris, ComputeError, Vk, VkError},
    geo::*,
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
        tris: Vec<Triangle3d>,
        points: Vec<Point3d>,
        vk: &Vk,
    ) -> PyResult<Vec<Point3d>> {
        Ok(intersect_tris(&tris, &points, &vk)?)
    }

    #[pyfn(m, "partition_tris")]
    fn partition_tris_py(
        _py: Python,
        tris: Vec<Triangle3d>,
        columns: Vec<Line3d>,
        vk: &Vk,
    ) -> PyResult<Vec<Vec<Triangle3d>>> {
        Ok(partition_tris(&tris, &columns, vk)?)
    }

    Ok(())
}
