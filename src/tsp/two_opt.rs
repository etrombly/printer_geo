use crate::{
    geo::Point3d,
    tsp::{nn::*, util::*},
};
use rayon::prelude::*;

/// Optimize path using kopt
///
/// # Examples
///
/// ```
/// use printer_geo::tsp::two_opt::*;
/// use printer_geo::geo::Point3d;
/// # use printer_geo::tsp::{nn::*, util::*};
/// # let islands = gen_islands();
/// let islands_unordered = vec![islands[2].clone(), islands[1].clone(), islands[3].clone(), islands[0].clone(), islands[4].clone(), islands[5].clone()];
/// let islands_opt = optimize_kopt(&islands_unordered, &Point3d::new(0.,0.,0.));
/// # let islands = nn(&islands, Point3d::new(0.,0.,0.));
/// # assert_eq!(islands_opt, islands);
/// ```
pub fn optimize_kopt(islands: &[Vec<Vec<Point3d>>], start: &Point3d) -> Vec<Vec<Vec<Point3d>>> {
    let islands = nn(islands, *start);
    let size = islands.len();
    if size < 5 {
        return islands.to_vec();
    }

    let mut results = islands.to_vec();
    let mut path: Vec<usize> = (0..size).into_iter().collect();
    let mut improve = true;
    let mut start_end = gen_start_end(&results);
    let mut best_distance = tour_len(&results, &path, &start_end, start);
    println!("start {:?}", best_distance);
    while improve {
        improve = false;
        path = (0..size).into_iter().collect();
        start_end = gen_start_end(&results);
        for i in 0..(size - 1) {
            if let Some(min) = ((i + 1)..size)
                .into_par_iter()
                .map(|k| {
                    let new_path = gen_new_path(&path, i, k);
                    (tour_len(&results, &new_path, &start_end, start), new_path)
                })
                .reduce_with(|a, b| match a.0.partial_cmp(&b.0) {
                    Some(x) if x == std::cmp::Ordering::Less => a,
                    _ => b,
                })
            {
                if min.0 < best_distance {
                    improve = true;
                    path = min.1;
                    best_distance = min.0;
                }
            }
        }
        results = path.iter().map(|x| results[*x].clone()).collect();
    }
    println!("end: {:?}", best_distance);
    results
    
}

/// Create new path for kopt
///
/// # Examples
///
/// ```
/// use printer_geo::tsp::two_opt::*;
/// use printer_geo::geo::Point3d;
/// let path = vec![0,1,2,3,4,5];
/// let new_path = gen_new_path(&path, 2, 4);
/// # assert_eq!(new_path, vec![0,1,4,3,2,5]);
/// ```
pub fn gen_new_path(path: &[usize], i: usize, k: usize) -> Vec<usize> {
    let mut new_path = path[0..i].to_vec();
    for index in (i..=k).rev() {
        new_path.push(path[index]);
    }
    if k < path.len() - 1 {
        new_path.append(&mut path[k + 1..].to_vec());
    }
    new_path
}
