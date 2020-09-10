use crate::{python::geo_py::TrianglesPy, stl::stl_to_tri};
use pyo3::prelude::*;

#[pymodule]
pub fn stl(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "load_stl")]
    fn load_stl_py(_py: Python, filename: &str) -> PyResult<TrianglesPy> {
        let triangles = stl_to_tri(filename)?;
        Ok(triangles.into())
    }

    Ok(())
}
