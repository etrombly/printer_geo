
use rand::Rng;
use std::time;
use crate::geo::Point3d;
use crate::tsp::{util::*, nn::*};

pub fn optimize_kopt(islands: Vec<Vec<Vec<Point3d>>>, start: Point3d, timeout: time::Duration) -> Vec<Vec<Vec<Point3d>>>{
    let mut islands = nn(islands, start);
    let size = islands.len();
    let mut improve = 0;
    if size < 5 {
        return islands;
    }

    let mut best_distance = tour_len(&islands, start);
    println!("start {:?}", best_distance);
    while improve < 20 {
        for i in 1..(size - 1) {
            for k in (i + 1)..size {
                let mut new_islands = islands[0..i].to_vec();
                for index in (i..=k).rev() {
                    new_islands.push(islands[index].clone());
                }
                if k < size - 1 {
                    new_islands.append(&mut islands[k+1..].to_vec());
                }
                let new_distance = tour_len(&new_islands, start);
                if new_distance < best_distance {
                    improve = 0;
                    islands = new_islands;
                    best_distance = new_distance;
                }
            }
        }
        improve += 1;
    }
    println!("end: {:?}", best_distance);
    islands
}

pub fn two_opt(islands: Vec<Vec<Vec<Point3d>>>, start: Point3d) -> (Vec<Vec<Vec<Point3d>>>, Option<f32>)
{
    let mut i = rand::thread_rng().gen_range(0, islands.len());
    let mut j = rand::thread_rng().gen_range(0, islands.len());

    if i == j {
        return (islands, None);
    }

    let mut ij = vec![i, j];
    ij.sort();
    i = ij[0];
    j = ij[1];


    let mut new_islands = Vec::from(&islands[..i]);
    let mut middle = Vec::from(&islands[i..j]);
    middle.reverse();
    new_islands.append(&mut middle);
    new_islands.append(&mut Vec::from(&islands[j..]));

    let prev_len = tour_len(&islands, start);
    let post_len = tour_len(&new_islands, start);

    if post_len < prev_len {
        (new_islands,Some(post_len - prev_len))
    } else {
        (islands, None)
    }
}

/*
pub fn optimize_kopt(islands: Vec<Vec<Vec<Point3d>>>, start: Point3d, timeout: time::Duration) -> Vec<Vec<Vec<Point3d>>>{
    let mut islands = nn(islands, start);
    let start_time = time::Instant::now();
    let max_iter_withouth_impr = islands.len() ^ 2;
    let mut iter_without_impr = 0;
    let mut best_tour_length = std::f32::MAX;
    let mut best_tour: Vec<_> = Vec::new();
    loop {
        match two_opt(islands, start) {
            (x, Some(_)) => {
                islands = x;
                iter_without_impr = 0;
            }
            (x, None) => {
                iter_without_impr += 1;
                islands = x;
                if iter_without_impr > max_iter_withouth_impr {
                    let current_tour_length = tour_len(&islands, start);
                    if current_tour_length < best_tour_length {
                        best_tour = islands.clone();
                        best_tour_length = current_tour_length;
                    }
                    //kopt::k_opt(4, self); // kick
                    //iter_without_impr = 0;
                }
            }
        }
        if start_time.elapsed() > timeout {
            break;
        }
    }
    let current_tour_length = tour_len(&islands, start);
    if current_tour_length < best_tour_length {
        best_tour = islands.clone();
    }
    best_tour
}
*/

/*
impl<T: Metrizable + Clone + Borrow<T>> Tour<T> {
    pub fn new() -> Tour<T> {
        Tour {
            path: Vec::new() as Vec<T>,
        }
    }

    pub fn from(nodes: &Vec<T>) -> Tour<T>
    where
        T: Clone,
    {
        Tour {
            path: (*nodes).clone(),
        }
    }

    pub fn tour_len(&self) -> f64 {
        if self.path.len() <= 0 {
            return 0.;
        }

        let mut sum = 0.;
        let mut prev = self.path.last().unwrap();
        for curr in &self.path {
            sum += prev.cost(&curr);
            prev = &curr;
        }
        sum
    }

    pub fn optimize_kopt(&mut self, timeout: time::Duration) {
        self.optimize_nn();
        let start_time = time::Instant::now();
        let max_iter_withouth_impr = self.path.len() ^ 2;
        let mut iter_without_impr = 0;
        let mut best_tour_length = std::f64::MAX;
        let mut best_tour: Vec<T> = Vec::new();
        loop {
            match kopt::k_opt(2, self) {
                Some(_) => {
                    iter_without_impr = 0;
                }
                None => {
                    iter_without_impr += 1;
                    if iter_without_impr > max_iter_withouth_impr {
                        let current_tour_length = self.tour_len();
                        if current_tour_length < best_tour_length {
                            best_tour = self.path.clone();
                            best_tour_length = current_tour_length;
                        }
                        kopt::k_opt(4, self); // kick
                        iter_without_impr = 0;
                    }
                }
            }
            if start_time.elapsed() > timeout {
                break;
            }
        }
        let current_tour_length = self.tour_len();
        if current_tour_length < best_tour_length {
            best_tour = self.path.clone();
        }
        self.path = best_tour;
    }
}

#[derive(Clone)]
pub(crate) struct IndexedT<T> {
    pub index: usize,
    pub value: T,
}

#[inline]
pub(crate) fn index_path<T>(path: &Vec<T>) -> Vec<IndexedT<&T>> {
    path.iter()
        .enumerate()
        .map(|(index, value)| IndexedT { index, value })
        .collect()
}
*/