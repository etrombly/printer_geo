use crate::{
    compute::Vk,
    geo::{Line3d, Triangle3d},
    geo_vulkan::{
        generate_columns, generate_gcode, generate_grid, generate_heightmap, generate_layers,
        generate_toolpath, to_tri_vk, LineVk, PointVk, PointsVk, Tool, TriangleVk, TrianglesVk,
    },
};
use pyo3::prelude::*;

#[pymodule]
pub fn geo_vulkan(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "to_tri_vk")]
    fn to_tri_vk_py(_py: Python, tris: Vec<Triangle3d>) -> TrianglesVk { to_tri_vk(&tris) }

    #[pyfn(m, "generate_grid")]
    fn generate_grid_py(_py: Python, bounds: &Line3d, scale: f32) -> Vec<PointsVk> {
        generate_grid(bounds, &scale)
    }

    #[pyfn(m, "generate_columns")]
    fn generate_columns_py(
        _py: Python,
        grid: Vec<PointsVk>,
        bounds: &Line3d,
        resolution: f32,
        scale: f32,
    ) -> Vec<LineVk> {
        generate_columns(&grid, bounds, &resolution, &scale)
    }

    #[pyfn(m, "generate_heightmap")]
    pub fn generate_heightmap_py(
        _py: Python,
        grid: Vec<PointsVk>,
        partition: Vec<TrianglesVk>,
        vk: &Vk,
    ) -> Vec<PointsVk> {
        generate_heightmap(&grid, &partition, vk)
    }

    #[pyfn(m, "generate_toolpath")]
    pub fn generate_toolpath_py(
        _py: Python,
        heightmap: Vec<PointsVk>,
        bounds: &Line3d,
        tool: &Tool,
        radius: f32,
        stepover: f32,
        scale: f32,
    ) -> Vec<PointsVk> {
        generate_toolpath(&heightmap, bounds, tool, &radius, &stepover, &scale)
    }

    #[pyfn(m, "generate_layers")]
    pub fn generate_layers_py(
        toolpath: Vec<PointsVk>,
        bounds: &Line3d,
        stepdown: f32,
    ) -> Vec<Vec<PointsVk>> {
        generate_layers(&toolpath, bounds, &stepdown)
    }

    #[pyfn(m, "generate_gcode")]
    pub fn generate_gcode_py(layers: Vec<Vec<PointsVk>>, bounds: &Line3d) -> String {
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
