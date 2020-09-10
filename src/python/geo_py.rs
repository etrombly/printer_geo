use crate::geo::{get_bounds, Line3d, Triangle3d};
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
pub struct Line3dPy {
    pub(crate) inner: Line3d,
}

#[pymodule]
pub fn geo(py: Python, m: &PyModule) -> PyResult<()> {
    // PyO3 aware function. All of our Python interfaces could be declared in a
    // separate module. Note that the `#[pyfn()]` annotation automatically
    // converts the arguments from Python objects to Rust values, and the Rust
    // return value back into a Python object. The `_py` argument represents
    // that we're holding the GIL.
    #[pyfn(m, "get_bounds")]
    fn get_bounds_py(_py: Python, tris: &TrianglesPy) -> PyResult<Line3dPy> {
        let bounds = Line3dPy {
            inner: get_bounds(&tris.inner),
        };
        Ok(bounds)
    }

    Ok(())
}
