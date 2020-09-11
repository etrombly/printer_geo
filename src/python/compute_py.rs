use crate::{
    compute::{intersect_tris, partition_tris, ComputeError, Vk, VkError},
    python::geo_vulkan_py::{LinesVkPy, PointsVkPy, TrianglesVkPy},
};
use pyo3::{exceptions::TypeError, prelude::*};

#[pyclass]
pub struct VkPy {
    pub(crate) inner: Vk,
}

impl From<Vk> for VkPy {
    fn from(vk: Vk) -> Self { VkPy { inner: vk } }
}

impl From<VkError> for PyErr {
    fn from(err: VkError) -> Self { PyErr::new::<TypeError, _>(err.to_string()) }
}

impl From<ComputeError> for PyErr {
    fn from(err: ComputeError) -> Self { PyErr::new::<TypeError, _>(err.to_string()) }
}

#[pymodule]
pub fn compute(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "init_vk")]
    fn init_vk_py(_py: Python) -> PyResult<VkPy> {
        let vk = Vk::new()?;
        Ok(vk.into())
    }

    #[pyfn(m, "intersect_tris")]
    fn intersect_tris_py(
        _py: Python,
        tris: &TrianglesVkPy,
        points: &PointsVkPy,
        vk: &VkPy,
    ) -> PyResult<PointsVkPy> {
        let result = intersect_tris(&tris.inner, &points.inner, &vk.inner)?;
        Ok(result.into())
    }

    #[pyfn(m, "partition_tris")]
    fn partition_tris_py(
        _py: Python,
        tris: &TrianglesVkPy,
        columns: &LinesVkPy,
        vk: &VkPy,
    ) -> PyResult<Vec<TrianglesVkPy>> {
        let result = partition_tris(&tris.inner, &columns.inner, &vk.inner)?;
        let result: Vec<TrianglesVkPy> = result.into_iter().map(|x| x.into()).collect();
        Ok(result)
    }

    Ok(())
}
