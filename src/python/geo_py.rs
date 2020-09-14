use crate::geo::{get_bounds, move_to_zero, Line3d, Triangle3d};
use pyo3::prelude::*;
use pyo3::types::PyList;

#[pymodule]
pub fn geo(_py: Python, m: &PyModule) -> PyResult<()> {

    #[pyfn(m, "add_tri")]
    fn add_tri_py(
        py: Python,
        tris: &PyList,
        p1: (f32, f32, f32),
        p2: (f32, f32, f32),
        p3: (f32, f32, f32),
    ) -> PyResult<()> {
        let tri = PyCell::new(py, Triangle3d::new(p1, p2, p3))?;
        tris.append(tri)?;
        Ok(())
    }

    #[pyfn(m, "new_tris")]
    fn new_tris_py(_py: Python) -> Vec<Triangle3d> { Vec::new() }

    #[pyfn(m, "get_bounds")]
    fn get_bounds_py(_py: Python, tris: Vec<Triangle3d>) -> Line3d { get_bounds(&tris) }

    #[pyfn(m, "move_to_zero")]
    fn move_to_zero_py(_py: Python, mut tris: Vec<Triangle3d>) { move_to_zero(&mut tris); }

    Ok(())
}
