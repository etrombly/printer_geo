use crate::geo::{get_bounds, move_to_zero, Line3d, Triangle3d};
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct TrianglesPy {
    pub(crate) inner: Vec<Triangle3d>,
}

impl From<Vec<Triangle3d>> for TrianglesPy {
    fn from(tris: Vec<Triangle3d>) -> Self { TrianglesPy { inner: tris } }
}

#[pyclass]
#[derive(Clone)]
pub struct Line3dPy {
    pub(crate) inner: Line3d,
}

#[pymethods]
impl Line3dPy {
    pub fn max_x(&self) -> f32 { self.inner.p2[0] }
}

#[pymodule]
pub fn geo(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "get_bounds")]
    fn get_bounds_py(_py: Python, tris: &TrianglesPy) -> Line3dPy {
        Line3dPy {
            inner: get_bounds(&tris.inner),
        }
    }

    #[pyfn(m, "move_to_zero")]
    fn move_to_zero_py(_py: Python, tris: &mut TrianglesPy) { move_to_zero(&mut tris.inner); }

    Ok(())
}
