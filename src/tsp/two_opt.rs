
use crate::geo::Point3d;
use crate::tsp::{util::*, nn::*};
use rayon::prelude::*;

pub fn optimize_kopt(islands: &[Vec<Vec<Point3d>>], start: Point3d) -> Vec<Vec<Vec<Point3d>>>{
    let islands = nn(islands, start);
    let size = islands.len();
    let mut path: Vec<usize> = (0..islands.len()).into_iter().collect();
    let mut improve = 0;
    if size < 5 {
        return islands;
    }

    let start_end = gen_start_end(&islands);
    let mut best_distance = tour_len(&islands, &path, &start_end, start);
    println!("start {:?}", best_distance);
    while improve < 2 {
        for i in 1..(size - 1) {
            if let Some(min) = ((i + 1)..size).into_par_iter().map(|k| {
                let new_path = gen_new_path(&path, i, k);
                (tour_len(&islands, &new_path, &start_end, start), new_path)
            }).reduce_with(|a, b| match a.0.partial_cmp(&b.0){
                Some(x) if x == std::cmp::Ordering::Less => a,
                _ => b
            }){
            if min.0 < best_distance {
                improve = 0;
                path = min.1;
                best_distance = min.0;
            }}
        }
        improve += 1;
    }
    println!("end: {:?}", best_distance);
    path.iter().map(|x| islands[*x].clone()).collect()
}

pub fn gen_new_path(path: &[usize], i: usize, k: usize) -> Vec<usize> {
    let mut new_path = path[0..i].to_vec();
    for index in (i..=k).rev() {
        new_path.push(path[index]);
    }
    if k < path.len() - 1 {
        new_path.append(&mut path[k+1..].to_vec());
    }
    new_path
}

pub fn gen_start_end(islands: &[Vec<Vec<Point3d>>]) -> Vec<Vec<(usize, usize, usize, usize)>> {
    let mut results = Vec::new();
    for island in islands {
        let len = island.len() - 1;
        let mut current_island = Vec::new();
        for (start_segment, start_point) in &[(0,0), (0,island[0].len()- 1), (len, 0), (len, island[len].len() - 1)] {
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