use crate::{compute::Vk, geo::*};
use pyo3::{prelude::*, types::PyList};
use rayon::prelude::*;

#[pymethods]
impl Point3d {
    #[getter]
    fn position(&self) -> (f32, f32, f32) { (self.pos.x, self.pos.y, self.pos.z) }

    #[getter]
    fn x(&self) -> f32 { self.pos.x }

    #[getter]
    fn y(&self) -> f32 { self.pos.y }

    #[getter]
    fn z(&self) -> f32 { self.pos.z }
}

#[pymethods]
impl Line3d {
    #[getter]
    fn p1(&self) -> Point3d { self.p1 }

    #[getter]
    fn p2(&self) -> Point3d { self.p2 }

    fn translate_2d(&mut self, x: f32, y: f32) {
        self.p1.pos.x = self.p1.pos.x + x;
        self.p2.pos.x = self.p2.pos.x + x;
        self.p1.pos.y = self.p1.pos.y + y;
        self.p2.pos.y = self.p2.pos.y + y;
    }
}

#[pymethods]
impl Tool {
    #[getter]
    fn diameter(&self) -> f32 { self.diameter }
}

#[pymethods]
impl DropCutter {
    #[getter]
    fn resolution(&self) -> f32 { self.resolution }

    #[getter]
    fn x_offset(&self) -> f32 { self.x_offset }

    #[getter]
    fn y_offset(&self) -> f32 { self.y_offset }

    fn generate_toolpath(&self, points: Vec<Point3d>) -> Vec<Point3d> {
        let scale = 1. / self.resolution;
        points
            .par_iter()
            .map(|point| {
                let x = (point.pos.x * scale).round() as usize;
                let y = (point.pos.y * scale).round() as usize;
                let columns = self.heightmap.len();
                let rows = self.heightmap[0].len();
                let max = self
                    .tool
                    .points
                    .iter()
                    .map(|tpoint| {
                        // for each point in the tool adjust it's location to the height map and
                        // calculate the intersection
                        let x_offset = (x as f32 + (tpoint.pos.x * scale)).round() as i32;
                        let y_offset = (y as f32 + (tpoint.pos.y * scale)).round() as i32;
                        if x_offset < columns as i32
                            && x_offset >= 0
                            && y_offset < rows as i32
                            && y_offset >= 0
                        {
                            self.heightmap[x_offset as usize][y_offset as usize].pos.z
                                - tpoint.pos.z
                        } else {
                            // TODO: this should be min z
                            0.
                        }
                    })
                    .fold(f32::NAN, f32::max); // same as calling max on all the values for this tool to find the heighest
                Point3d::new(
                    (x as f32 / scale) - self.x_offset,
                    (y as f32 / scale) - self.y_offset,
                    max,
                )
            })
            .collect()
    }
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

    #[pyfn(m, "new_point")]
    fn new_point_py(_py: Python, x: f32, y: f32, z: f32) -> Point3d { Point3d::new(x, y, z) }

    #[pyfn(m, "new_line")]
    fn new_line_py(_py: Python, x1: f32, y1: f32, z1: f32, x2: f32, y2: f32, z2: f32) -> Line3d {
        Line3d::new((x1, y1, z1), (x2, y2, z2))
    }

    #[pyfn(m, "new_dropcutter")]
    fn new_dropcutter_py(
        _py: Python,
        model: Vec<Triangle3d>,
        heightmap: Vec<Vec<Point3d>>,
        tool: Tool,
        x_offset: f32,
        y_offset: f32,
        resolution: f32,
    ) -> DropCutter {
        DropCutter {
            model,
            heightmap,
            tool,
            x_offset,
            y_offset,
            resolution,
        }
    }

    #[pyfn(m, "sample_line")]
    fn sample_line_py(_py: Python, line: Line3d, sample: f32) -> Vec<Point3d> {
        let steps = (line.length_2d() / sample) as usize + 1;
        (0..steps)
            .into_par_iter()
            .map(|i| {
                let frac = i as f32 / steps as f32;
                line.get_point(frac)
            })
            .collect()
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
