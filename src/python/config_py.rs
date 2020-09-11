use crate::config::*;
use pyo3::{exceptions::TypeError, prelude::*};
use structopt::StructOpt;

#[pyclass]
pub struct OptPy {
    inner: Opt,
}

impl From<Opt> for OptPy {
    fn from(opt: Opt) -> Self { OptPy { inner: opt } }
}

impl From<ConfigError> for PyErr {
    fn from(err: ConfigError) -> Self { PyErr::new::<TypeError, _>(err.to_string()) }
}

#[pymodule]
pub fn config(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "parse_config")]
    fn parse_config_py(_py: Python, mut config: Vec<&str>) -> PyResult<OptPy> {
        config.insert(0, "");
        let opt = Opt::from_iter_safe(&config).map_err(ConfigError::Clap)?;
        Ok(opt.into())
    }

    Ok(())
}
