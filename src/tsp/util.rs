use crate::geo::{Point3d, distance};

pub fn tour_len(islands: &[Vec<Vec<Point3d>>], start: Point3d) -> f32 {
    let mut len = distance(&start.pos.xy(), &islands[0][0][0].pos.xy());
    let mut islands = islands.iter().peekable();
    while let Some(island) = islands.next() {
        if let Some(other) = islands.peek() {
            let segment = island.len() - 1;
            let point = island[segment].len() - 1;
            len += distance(&island[segment][point].pos.xy(), &other[0][0].pos.xy());
        }
    }
    len
}

pub fn get_next_segment(segments: &mut Vec<Vec<Point3d>>, last: &Point3d) -> (usize, bool, f32) {
    let mut dist = f32::MAX;
    let mut current_segment = 0;
    let mut rev = true;
    for (index, segment) in segments.iter().enumerate() {
        let curr_dist = distance(&segment[segment.len() - 1].pos.xy(), &last.pos.xy());
        if curr_dist < dist {
            dist = curr_dist;
            rev = true;
            current_segment = index;
        }
        let curr_dist = distance(&segment[0].pos.xy(), &last.pos.xy());
        if curr_dist < dist {
            dist = curr_dist;
            rev = false;
            current_segment = index;
        }
    }
    (current_segment, rev, dist)
}

pub fn get_next_island(islands: &Vec<Vec<Vec<Point3d>>>, last: &Point3d) -> (usize, usize, bool, f32) {
    let mut dist = f32::MAX;
    let mut current_island = 0;
    let mut current_segment = 0;
    let mut rev = false;
    for (index, island) in islands.iter().enumerate() {
        // check first and last segment
        for segment in &[0_usize, island.len() - 1] {
            // check first and last point
            for point in &[0_usize, island[*segment].len() - 1] {
                let curr_dist = distance(&island[*segment][*point].pos.xy(), &last.pos.xy());
                if  curr_dist < dist {
                    dist = curr_dist;
                    current_segment = *segment;
                    if point == &0 {rev = false} else {rev = true}
                    current_island = index;
                }
            }
        }
    }
    (current_island, current_segment, rev, dist)
}