use crate::{geo::Point3d, tsp::util::*};

pub fn nn(islands: &[Vec<Vec<Point3d>>], start: Point3d) -> Vec<Vec<Vec<Point3d>>> {
    let mut result = Vec::new();
    let mut exclude = Vec::new();
    let mut last = start;
    let (mut current_island, mut current_segment, mut point, _) = get_next_island(&islands, &exclude, &last);
    exclude.push(current_island);
    for _ in 0..islands.len() {
        let mut island = islands[current_island].clone();
        let mut segments = Vec::new();
        let mut segment_exclude = vec![current_segment];
        for _ in 0..island.len() {
            let mut segment = island[current_segment].clone();
            if point != 0 {
                segment.reverse();
            }
            last = segment[segment.len() - 1];
            segments.push(segment);
            let tmp = get_next_segment(&mut island, &segment_exclude, &last);
            segment_exclude.push(tmp.0);
            current_segment = tmp.0;
            point = tmp.1;
        }
        result.push(segments);
        let tmp = get_next_island(&islands, &exclude, &last);
        exclude.push(tmp.0);
        current_island = tmp.0;
        current_segment = tmp.1;
        point = tmp.2;
    }
    result
}
