use crate::geo::{get_bounds, move_to_zero, Line3d, Triangle3d};
use pyo3::prelude::*;

#[pymodule]
pub fn geo(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "get_bounds")]
    fn get_bounds_py(_py: Python, tris: Vec<Triangle3d>) -> Line3d { get_bounds(&tris) }

    #[pyfn(m, "move_to_zero")]
    fn move_to_zero_py(_py: Python, mut tris: Vec<Triangle3d>) { move_to_zero(&mut tris); }

    Ok(())
}
