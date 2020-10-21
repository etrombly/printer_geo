use crate::geo::{distance, Point3d};

/// Check length of tour
///
/// # Examples
///
/// ```
/// use printer_geo::geo::Point3d;
/// use printer_geo::tsp::util::*;
/// # let islands = gen_islands();
/// let path = vec![0, 1, 2, 3, 4];
/// let start_end = gen_start_end(&islands);
/// let len = tour_len(&islands, &path, &start_end, &Point3d::new(0.,0.,0.));
/// # assert_eq!(14.40493, len);
/// ```
pub fn tour_len(
    islands: &[Vec<Vec<Point3d>>],
    path: &[usize],
    start_end: &[Vec<(usize, usize, usize, usize)>],
    start: &Point3d,
) -> f32 {
    let (mut segment, mut point, mut len) = get_nearest_island_segment(&islands[path[0]], &start);
    let (current_segment, current_point) = (segment, point);
    for (start_segment, start_point, end_segment, end_point) in &start_end[0] {
        if (start_segment, start_point) == (&current_segment, &current_point) {
            segment = *end_segment;
            point = *end_point;
        }
    }
    let mut path = path.iter().peekable();
    while let Some(island) = path.next() {
        if let Some(other) = path.peek() {
            let (current_segment, current_point, current_dist) =
                get_nearest_island_segment(&islands[**other], &islands[*island][segment][point]);
            for (start_segment, start_point, end_segment, end_point) in &start_end[**other] {
                if (start_segment, start_point) == (&current_segment, &current_point) {
                    segment = *end_segment;
                    point = *end_point;
                }
            }
            len += current_dist;
        }
    }
    len
}

/// Get next line segment in island
///
/// # Examples
///
/// ```
/// use printer_geo::geo::Point3d;
/// use printer_geo::tsp::util::*;
/// # let islands = gen_islands();
/// let exclude = Vec::new();
/// let island = get_next_island(&islands, &exclude, &Point3d::new(0.,0.,0.));
/// # assert_eq!((0, 0, 0, 1.4142135), island);
/// ```
pub fn get_next_segment(segments: &[Vec<Point3d>], exclude: &[usize], last: &Point3d) -> (usize, usize, f32) {
    let mut dist = f32::MAX;
    let mut current_segment = 0;
    let mut point = 0;
    for (index, segment) in segments.iter().enumerate() {
        if exclude.contains(&index) {
            continue;
        }
        let curr_dist = distance(&segment[segment.len() - 1].pos.xy(), &last.pos.xy());
        if curr_dist < dist {
            dist = curr_dist;
            point = segment.len() - 1;
            current_segment = index;
        }
        let curr_dist = distance(&segment[0].pos.xy(), &last.pos.xy());
        if curr_dist < dist {
            dist = curr_dist;
            point = 0;
            current_segment = index;
        }
    }
    (current_segment, point, dist)
}

/// Get closest island to last point
///
/// # Examples
///
/// ```
/// use printer_geo::geo::Point3d;
/// use printer_geo::tsp::util::*;
/// # let islands = gen_islands();
/// let exclude = vec![0];
/// let island = get_next_segment(&islands[0], &exclude, &Point3d::new(1.,1.,1.));
/// # assert_eq!((1, 0, 1.0), island);
/// ```
pub fn get_next_island(islands: &[Vec<Vec<Point3d>>], exclude: &[usize], last: &Point3d) -> (usize, usize, usize, f32) {
    let mut dist = f32::MAX;
    let mut island = 0;
    let mut segment = 0;
    let mut point = 0;
    for (index, current_island) in islands.iter().enumerate() {
        if exclude.contains(&index) {
            continue;
        }
        let (current_segment, current_point, curr_dist) = get_nearest_island_segment(&current_island, &last);
        if curr_dist < dist {
            dist = curr_dist;
            segment = current_segment;
            point = current_point;
            island = index;
        }
    }
    (island, segment, point, dist)
}

pub fn get_nearest_island_segment(island: &[Vec<Point3d>], last: &Point3d) -> (usize, usize, f32) {
    let mut segment = 0;
    let mut point = 0;
    let mut dist = f32::MAX;
    for current_segment in &[0_usize, island.len() - 1] {
        // check first and last point
        for current_point in &[0_usize, island[*current_segment].len() - 1] {
            let current_dist = distance(&island[*current_segment][*current_point].pos.xy(), &last.pos.xy());
            if current_dist < dist {
                dist = current_dist;
                segment = *current_segment;
                point = *current_point;
            }
        }
    }
    (segment, point, dist)
}

pub fn gen_start_end(islands: &[Vec<Vec<Point3d>>]) -> Vec<Vec<(usize, usize, usize, usize)>> {
    let mut results = Vec::new();
    for island in islands {
        let len = island.len() - 1;
        let mut current_island = Vec::new();
        for (start_segment, start_point) in &[(0, 0), (0, island[0].len() - 1), (len, 0), (len, island[len].len() - 1)]
        {
            let mut exclude = Vec::new();
            let mut current_segment = *start_segment;
            let mut current_point = *start_point;
            for _ in 0..island.len() {
                let tmp = get_next_segment(&island, &exclude, &island[current_segment][current_point]);
                exclude.push(tmp.0);
                current_segment = tmp.0;
                current_point = tmp.1;
            }
            current_island.push((*start_segment, *start_point, current_segment, current_point));
        }
        results.push(current_island);
    }
    results
}

/// generate data for doc tests
pub fn gen_islands() -> Vec<Vec<Vec<Point3d>>> {
    let mut islands = Vec::new();

    islands.push(vec![
        vec![
            Point3d::new(1., 1., 1.),
            Point3d::new(1., 2., 1.),
            Point3d::new(1., 3., 1.),
        ],
        vec![
            Point3d::new(2., 1., 1.),
            Point3d::new(2., 2., 1.),
            Point3d::new(2., 3., 1.),
        ],
    ]);
    islands.push(vec![
        vec![
            Point3d::new(4., 6., 1.),
            Point3d::new(4., 7., 1.),
            Point3d::new(4., 8., 1.),
        ],
        vec![
            Point3d::new(5., 6., 1.),
            Point3d::new(5., 7., 1.),
            Point3d::new(5., 8., 1.),
        ],
    ]);
    islands.push(vec![
        vec![
            Point3d::new(7., 1., 1.),
            Point3d::new(7., 2., 1.),
            Point3d::new(7., 3., 1.),
        ],
        vec![
            Point3d::new(8., 1., 1.),
            Point3d::new(8., 2., 1.),
            Point3d::new(8., 3., 1.),
        ],
    ]);
    islands.push(vec![
        vec![
            Point3d::new(10., 1., 1.),
            Point3d::new(10., 2., 1.),
            Point3d::new(10., 3., 1.),
        ],
        vec![
            Point3d::new(11., 1., 1.),
            Point3d::new(11., 2., 1.),
            Point3d::new(11., 3., 1.),
        ],
    ]);
    islands.push(vec![
        vec![
            Point3d::new(13., 1., 1.),
            Point3d::new(13., 2., 1.),
            Point3d::new(13., 3., 1.),
        ],
        vec![
            Point3d::new(14., 1., 1.),
            Point3d::new(14., 2., 1.),
            Point3d::new(14., 3., 1.),
        ],
    ]);
    islands
}
