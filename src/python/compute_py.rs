use crate::compute::*;
use pyo3::prelude::*;

#[pyclass]
pub struct VkPy {
    inner: Vk,
}

#[pymodule]
pub fn compute(py: Python, m: &PyModule) -> PyResult<()> {
    // PyO3 aware function. All of our Python interfaces could be declared in a
    // separate module. Note that the `#[pyfn()]` annotation automatically
    // converts the arguments from Python objects to Rust values, and the Rust
    // return value back into a Python object. The `_py` argument represents
    // that we're holding the GIL.
    #[pyfn(m, "init_vk")]
    fn init_vk_py(_py: Python) -> PyResult<VkPy> {
        let vkpy = VkPy {
            inner: Vk::new().unwrap(),
        };
        Ok(vkpy)
    }

    Ok(())
}
