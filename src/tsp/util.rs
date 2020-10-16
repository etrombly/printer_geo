use crate::geo::{Point3d, distance};

pub fn tour_len(islands: &[Vec<Vec<Point3d>>], path: &[usize], start_end: &[Vec<(usize,usize,usize,usize)>], start: Point3d) -> f32 {
    let (mut segment, mut point, mut len) = get_nearest_island_segment(&islands[path[0]], &start);
    let mut path = path.iter().peekable();
    while let Some(island) = path.next() {
        if let Some(other) = path.peek() {
            let (current_segment, current_point, current_dist) = get_nearest_island_segment(&islands[**other], &islands[*island][segment][point]);
            for (start_segment, start_point, end_segment, end_point) in &start_end[**other] {
                if start_segment == &current_segment && start_point == &current_point {
                    segment = *end_segment;
                    point = *end_point;
                }
            }
            len += current_dist;
        }
    }
    len
}

pub fn get_next_segment(segments: &[Vec<Point3d>], exclude: &[usize], last: &Point3d) -> (usize, usize, f32) {
    let mut dist = f32::MAX;
    let mut current_segment = 0;
    let mut point = 0;
    for (index, segment) in segments.iter().enumerate() {
        if exclude.contains(&index) {continue}
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

pub fn get_next_island(islands: &[Vec<Vec<Point3d>>], exclude: &[usize], last: &Point3d) -> (usize, usize, usize, f32) {
    let mut dist = f32::MAX;
    let mut island = 0;
    let mut segment = 0;
    let mut point = 0;
    for (index, current_island) in islands.iter().enumerate() {
        if exclude.contains(&index) {continue}
        let (current_segment, current_point, curr_dist) = get_nearest_island_segment(&current_island, &last);
        if  curr_dist < dist {
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
            if  current_dist < dist {
                dist = current_dist;
                segment = *current_segment;
                point = *current_point;
            }
        }
    }
    (segment, point, dist)
}