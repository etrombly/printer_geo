use crate::geo::Point3d;
use float_cmp::approx_eq;
/// modified from https://www.quora.com/What-is-an-efficient-algorithm-to-find-an-island-of-connected-1s-in-a-matrix-of-0s-and-1s

const SIBLINGS: [(i32, i32); 8] = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];

pub fn bfs(node: (usize, usize), visited: &mut Visited, data: &[Vec<Point3d>]) -> Option<Vec<(usize, usize)>> {
    if visited.visited(node.0, node.1) {
        return None;
    }
    let mut stack = Vec::new();
    let mut lands = Vec::new();

    stack.push(node);
    while let Some(cnode) = stack.pop() {
        if visited.visited(cnode.0, cnode.1) {
            continue;
        }
        visited.flip(cnode.0, cnode.1);
        lands.push(cnode);
        for sibling in &SIBLINGS {
            let x = cnode.0 as i32 + sibling.0;
            let y = cnode.1 as i32 + sibling.1;
            if x >= 0 && y >= 0 {
                let x = x as usize;
                let y = y as usize;
                if x < data.len() && y < data[0].len() && !data[x][y].pos.z.is_nan() {
                    stack.push((x, y));
                }
            }
        }
    }
    Some(lands)
}

pub fn get_islands(data: &[Vec<Point3d>], diameter: f32) -> Vec<Vec<Vec<Point3d>>> {
    let mut visited = Visited::new(data.len(), data[0].len());
    let indices: Vec<_> = data
        .iter()
        .enumerate()
        .flat_map(|(x_index, x)| {
            x.iter()
                .enumerate()
                .filter_map(|(y_index, point)| {
                    if point.pos.z.is_nan() {
                        None
                    } else {
                        bfs((x_index, y_index), &mut visited, data)
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();
    indices
        .iter()
        .map(|x| x.iter().map(|y| data[y.0][y.1]).collect::<Vec<_>>())
        .map(|island| partition_segments(&island, diameter)).collect()
}

pub struct Visited {
    columns: usize,
    mask: Vec<u32>,
}

impl Visited {
    pub fn new(columns: usize, rows: usize) -> Visited {
        let count = ((columns as f32 - 1.) + ((rows as f32 - 1.) * columns as f32) / 32.).ceil() as usize;
        let mask: Vec<u32> = (0..count).map(|_| 0_u32).collect();
        Visited { columns, mask }
    }

    pub fn flip(&mut self, column: usize, row: usize) {
        let index = (column + (row * self.columns)) / 32;
        let pos = (column + (row * self.columns)) % 32;
        self.mask[index] |= 1 << pos;
    }

    pub fn visited(&self, column: usize, row: usize) -> bool {
        let index = (column + (row * self.columns)) / 32;
        let pos = (column + (row * self.columns)) % 32;
        self.mask[index] & (1 << pos) == (1 << pos)
    }
}

pub fn partition_segments(data: &Vec<Point3d>, diameter: f32) -> Vec<Vec<Point3d>> {
    let mut output = Vec::new();
    let mut data = data.clone();
    while data.len() > 0 {
        let mut column = Vec::new();
        if let Some(current) = data.pop() {
            column.push(current);
            data = data
                .into_iter()
                .filter(|point| {
                    if approx_eq!(f32, current.pos.x, point.pos.x, ulps = 3) {
                        column.push(*point);
                        false
                    } else {
                        true
                    }
                })
                .collect();
            column.sort();
            let mut segment = Vec::new();
            let mut last = column[0];
            for point in column {
                if (point.pos.y - last.pos.y).abs() > diameter * 2. {
                    output.push(segment.clone());
                    segment.clear();
                }
                segment.push(point);
                last = point;
            }
            output.push(segment);
        }
    }
    output.sort_by(|a,b| a[0][0].partial_cmp(&b[0][0]).unwrap());
    output
}