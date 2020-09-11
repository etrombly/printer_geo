use crate::{
    geo_vulkan::{
        generate_columns, generate_gcode, generate_grid, generate_heightmap, generate_layers,
        generate_toolpath, to_tri_vk, LineVk, PointVk, PointsVk, Tool, TriangleVk, TrianglesVk,
    },
    python::{compute_py::VkPy, geo_py::TrianglesPy},
    Line3dPy,
};
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct TrianglesVkPy {
    pub(crate) inner: Vec<TriangleVk>,
}

impl From<Vec<TriangleVk>> for TrianglesVkPy {
    fn from(tris: Vec<TriangleVk>) -> Self { TrianglesVkPy { inner: tris } }
}

#[pyclass]
pub struct LinesVkPy {
    pub(crate) inner: Vec<LineVk>,
}

impl From<Vec<LineVk>> for LinesVkPy {
    fn from(lines: Vec<LineVk>) -> Self { LinesVkPy { inner: lines } }
}

#[pyclass]
pub struct LineVkPy {
    pub(crate) inner: LineVk,
}

impl From<LineVk> for LineVkPy {
    fn from(line: LineVk) -> Self { LineVkPy { inner: line } }
}

#[pyclass]
#[derive(Clone)]
pub struct PointVkPy {
    pub(crate) inner: PointVk,
}

impl From<PointVk> for PointVkPy {
    fn from(point: PointVk) -> Self { PointVkPy { inner: point } }
}

impl From<PointVkPy> for PointVk {
    fn from(point: PointVkPy) -> Self { point.inner }
}

#[pyclass]
#[derive(Clone)]
pub struct PointsVkPy {
    pub(crate) inner: PointsVk,
}

impl From<PointsVk> for PointsVkPy {
    fn from(points: PointsVk) -> Self { PointsVkPy { inner: points } }
}

#[pyclass]
#[derive(Clone)]
pub struct ToolPy {
    pub(crate) inner: Tool,
}

impl From<Tool> for ToolPy {
    fn from(tool: Tool) -> Self { ToolPy { inner: tool } }
}

#[pymodule]
pub fn geo_vulkan(_py: Python, m: &PyModule) -> PyResult<()> {
    //m.add_class::<PointVkPy>()?;

    #[pyfn(m, "to_tri_vk")]
    fn to_tri_vk_py(_py: Python, tris: TrianglesPy) -> TrianglesVkPy {
        to_tri_vk(&tris.inner).into()
    }

    #[pyfn(m, "generate_grid")]
    fn generate_grid_py(_py: Python, bounds: Line3dPy, scale: f32) -> Vec<PointsVkPy> {
        generate_grid(&bounds.inner, &scale)
            .into_iter()
            .map(|points| points.into())
            .collect()
    }

    #[pyfn(m, "generate_columns")]
    fn generate_columns_py(
        _py: Python,
        grid: Vec<PointsVkPy>,
        bounds: Line3dPy,
        resolution: f32,
        scale: f32,
    ) -> LinesVkPy {
        let grid: Vec<PointsVk> = grid.into_iter().map(|points| points.inner.into()).collect();
        generate_columns(&grid, &bounds.inner, &resolution, &scale)
            .into_iter()
            .map(|line| line.into())
            .collect::<Vec<_>>()
            .into()
    }

    #[pyfn(m, "generate_heightmap")]
    pub fn generate_heightmap_py(
        _py: Python,
        grid: Vec<PointsVkPy>,
        partition: Vec<TrianglesVkPy>,
        vk: &VkPy,
    ) -> Vec<PointsVkPy> {
        let grid: Vec<PointsVk> = grid.into_iter().map(|points| points.inner.into()).collect();
        let partition: Vec<TrianglesVk> = partition
            .into_iter()
            .map(|tris| tris.inner.into())
            .collect();
        generate_heightmap(&grid, &partition, &vk.inner)
            .into_iter()
            .map(|points| points.into())
            .collect()
    }

    #[pyfn(m, "generate_toolpath")]
    pub fn generate_toolpath_py(
        _py: Python,
        heightmap: Vec<PointsVkPy>,
        bounds: &Line3dPy,
        tool: ToolPy,
        radius: f32,
        stepover: f32,
        scale: f32,
    ) -> Vec<PointsVkPy> {
        let heightmap: Vec<PointsVk> = heightmap
            .into_iter()
            .map(|points| points.inner.into())
            .collect();
        generate_toolpath(
            &heightmap,
            &bounds.inner,
            &tool.inner,
            &radius,
            &stepover,
            &scale,
        )
        .into_iter()
        .map(|points| points.into())
        .collect()
    }

    #[pyfn(m, "generate_layers")]
    pub fn generate_layers_py(
        toolpath: Vec<PointsVkPy>,
        bounds: &Line3dPy,
        stepdown: f32,
    ) -> Vec<Vec<PointsVkPy>> {
        let toolpath: Vec<PointsVk> = toolpath
            .into_iter()
            .map(|points| points.inner.into())
            .collect();
        generate_layers(&toolpath, &bounds.inner, &stepdown)
            .into_iter()
            .map(|layer| layer.into_iter().map(|points| points.into()).collect())
            .collect()
    }

    #[pyfn(m, "generate_gcode")]
    pub fn generate_gcode_py(layers: Vec<Vec<PointsVkPy>>, bounds: &Line3dPy) -> String {
        let layers: Vec<Vec<PointsVk>> = layers
            .into_iter()
            .map(|layer| {
                layer
                    .into_iter()
                    .map(|points| points.inner.into())
                    .collect()
            })
            .collect();
        generate_gcode(&layers, &bounds.inner)
    }

    #[pyfn(m, "new_endmill")]
    fn new_endmill_py(_py: Python, radius: f32, scale: f32) -> ToolPy {
        Tool::new_endmill(radius, scale).into()
    }

    #[pyfn(m, "new_ball")]
    fn new_ball_py(_py: Python, radius: f32, scale: f32) -> ToolPy {
        Tool::new_ball(radius, scale).into()
    }

    #[pyfn(m, "new_v_bit")]
    fn new_v_bit_py(_py: Python, radius: f32, angle: f32, scale: f32) -> ToolPy {
        Tool::new_v_bit(radius, angle, scale).into()
    }

    Ok(())
}
