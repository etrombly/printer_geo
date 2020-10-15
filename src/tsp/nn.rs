use crate::geo::Point3d;
use crate::tsp::util::*;

pub fn nn(mut islands: Vec<Vec<Vec<Point3d>>>, start: Point3d) -> Vec<Vec<Vec<Point3d>>> {
    let mut result = Vec::new();
    let mut last = start;
    let (mut current_island, mut current_segment, mut rev, _) = get_next_island(&islands, &last);
    while islands.len() > 0 {
        let mut island = islands.remove(current_island);
        let mut segments = Vec::new();
        while island.len() > 0 {
            let mut segment = island.remove(current_segment);
            if rev {segment.reverse();}
            last = segment[segment.len() - 1];
            segments.push(segment);
            let tmp = get_next_segment(&mut island, &last);
            current_segment = tmp.0;
            rev = tmp.1;
        }
        result.push(segments);
        let tmp = get_next_island(&islands, &last);
        current_island = tmp.0;
        current_segment = tmp.1;
        rev = tmp.2;
    }
    result
}