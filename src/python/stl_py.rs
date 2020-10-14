use crate::{geo::Triangle3d, stl::stl_to_tri};
use pyo3::prelude::*;

#[pymodule]
pub fn stl(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "stl_to_tri")]
    fn stl_to_tri_py(_py: Python, filename: &str) -> PyResult<Vec<Triangle3d>> { Ok(stl_to_tri(filename)?) }

    Ok(())
}
