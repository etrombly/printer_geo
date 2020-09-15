use crate::{compute::Vk, geo::*};

use pyo3::{prelude::*, types::PyList};

#[pymethods]
impl Point3d {
    #[getter]
    fn position(&self) -> PyResult<(f32, f32, f32)> { Ok((self.pos.x, self.pos.y, self.pos.z)) }
}

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

    #[pyfn(m, "generate_grid")]
    fn generate_grid_py(_py: Python, bounds: &Line3d, scale: f32) -> Vec<Vec<Point3d>> {
        generate_grid(bounds, &scale)
    }

    #[pyfn(m, "generate_columns")]
    fn generate_columns_py(
        _py: Python,
        grid: Vec<Vec<Point3d>>,
        bounds: &Line3d,
        resolution: f32,
        scale: f32,
    ) -> Vec<Line3d> {
        generate_columns(&grid, bounds, &resolution, &scale)
    }

    #[pyfn(m, "generate_columns_chunks")]
    fn generate_columns_chunks_py(
        _py: Python,
        grid: Vec<Vec<Point3d>>,
        bounds: &Line3d,
        resolution: f32,
        scale: f32,
    ) -> Vec<Line3d> {
        generate_columns_chunks(&grid, bounds, &resolution, &scale)
    }

    #[pyfn(m, "generate_heightmap")]
    pub fn generate_heightmap_py(
        _py: Python,
        grid: Vec<Vec<Point3d>>,
        partition: Vec<Vec<Triangle3d>>,
        vk: &Vk,
    ) -> Vec<Vec<Point3d>> {
        generate_heightmap(&grid, &partition, vk)
    }

    #[pyfn(m, "generate_heightmap_chunks")]
    pub fn generate_heightmap_chunks_py(
        _py: Python,
        grid: Vec<Vec<Point3d>>,
        partition: Vec<Vec<Triangle3d>>,
        vk: &Vk,
    ) -> Vec<Vec<Point3d>> {
        generate_heightmap_chunks(&grid, &partition, vk)
    }

    #[pyfn(m, "generate_toolpath")]
    pub fn generate_toolpath_py(
        _py: Python,
        heightmap: Vec<Vec<Point3d>>,
        bounds: &Line3d,
        tool: &Tool,
        radius: f32,
        stepover: f32,
        scale: f32,
    ) -> Vec<Vec<Point3d>> {
        generate_toolpath(&heightmap, bounds, tool, &radius, &stepover, &scale)
    }

    #[pyfn(m, "generate_layers")]
    pub fn generate_layers_py(
        toolpath: Vec<Vec<Point3d>>,
        bounds: &Line3d,
        stepdown: f32,
    ) -> Vec<Vec<Vec<Point3d>>> {
        generate_layers(&toolpath, bounds, &stepdown)
    }

    #[pyfn(m, "generate_gcode")]
    pub fn generate_gcode_py(layers: Vec<Vec<Vec<Point3d>>>, bounds: &Line3d) -> String {
        generate_gcode(&layers, &bounds)
    }

    #[pyfn(m, "new_endmill")]
    fn new_endmill_py(_py: Python, radius: f32, scale: f32) -> Tool {
        Tool::new_endmill(radius, scale)
    }

    #[pyfn(m, "new_ball")]
    fn new_ball_py(_py: Python, radius: f32, scale: f32) -> Tool { Tool::new_ball(radius, scale) }

    #[pyfn(m, "new_v_bit")]
    fn new_v_bit_py(_py: Python, radius: f32, angle: f32, scale: f32) -> Tool {
        Tool::new_v_bit(radius, angle, scale)
    }

    Ok(())
}
