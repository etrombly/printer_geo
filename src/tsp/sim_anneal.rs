use rand::{thread_rng, Rng};
use super::Metaheuristics;
use time::{Duration, PreciseTime};

pub fn solve(cities: &[(f64, f64)], runtime: Duration) -> Tour {
    let mut tsp = TravellingSalesman {
        distance_matrix: &get_distance_matrix(cities),
        rng:             &mut thread_rng(),
    };

    let best_candidate = metaheuristics::simulated_annealing::solve(&mut tsp, runtime);

    Tour {
        distance: get_route_distance(tsp.distance_matrix, &best_candidate.route),
        route:    best_candidate.route,
    }
}

struct TravellingSalesman<'a> {
    distance_matrix: &'a Vec<Vec<f64>>,
    rng:             &'a mut ThreadRng,
}

struct Candidate {
    route: Vec<usize>,
}

impl<'a> Metaheuristics<Candidate> for TravellingSalesman<'a> {
    fn clone_candidate(&mut self, candidate: &Candidate) -> Candidate {
        Candidate {
            route: candidate.route.clone(),
        }
    }

    fn generate_candidate(&mut self) -> Candidate {
        let mut route: Vec<usize> = self.distance_matrix.iter().enumerate().map(|(i,_)| i).collect();
        self.rng.shuffle(&mut route);

        let home_city = route[0];
        route.push(home_city);

        Candidate {
            route: route,
        }
    }

    fn rank_candidate(&mut self, candidate: &Candidate) -> f64 {
        0.0 - get_route_distance(self.distance_matrix, &candidate.route)
    }

    fn tweak_candidate(&mut self, candidate: &Candidate) -> Candidate {
        if candidate.route.len() <= 3 {
            return self.clone_candidate(candidate);
        }

        let mut old_route = candidate.route.clone();
        old_route.pop();

        // get two cities to work with

        let start        = self.rng.gen::<usize>() % old_route.len();
        let end          = self.rng.gen::<usize>() % old_route.len();
        let (start, end) = if start < end { (start, end) } else { (end, start) };

        // straight swap of the cities

        let mut swapped_route = old_route.clone();
        swapped_route.swap(start, end);

        // swap cities, then reverse the cities between them

        let split_route    = old_route.clone();
        let safe_offset    = if old_route.len() <= (end + 1) { old_route.len() } else { end + 1 };
        let (left, right)  = split_route.split_at(safe_offset);
        let (left, middle) = left.split_at(start);

        let mut middle = middle.to_vec();
        middle.reverse();

        let mut reordered_route = Vec::new();
        reordered_route.extend(left.into_iter());
        reordered_route.extend(middle.into_iter());
        reordered_route.extend(right.into_iter());

        // return shortest route

        let swapped_distance   = get_route_distance(self.distance_matrix, &swapped_route);
        let reordered_distance = get_route_distance(self.distance_matrix, &reordered_route);
        let mut shortest_route = if swapped_distance < reordered_distance { swapped_route } else { reordered_route };

        let home_city = shortest_route[0];
        shortest_route.push(home_city);

        return Candidate {
            route: shortest_route,
        }
    }
}

/// Represents a tour of the travelling salesman
pub struct Tour {
    /// the total distance travelled following this tour
    pub distance: f64,
    /// the ordered route for this tour
    pub route:    Vec<usize>,
}

/// Utility function to convert city coordinates to a distance matrix
///
/// `cities` is an array slice, containing `(x,y)` tuple coordinates for each city.
///
/// Returns a `Vec<Vec<f64>>`, containing the distance matrix.
///
///# Examples
///
///```
///extern crate travelling_salesman;
///
///fn main() {
///    let cities = [
///      (27.0, 78.0),
///      (18.0, 24.0),
///      (48.0, 62.0),
///      (83.0, 77.0),
///      (55.0, 56.0),
///    ];
///
///    let distance_matrix = travelling_salesman::get_distance_matrix(&cities);
///
///    println!("The distance between 1 and 2 is: {}", distance_matrix[1][2]);
///}
///```
pub fn get_distance_matrix(cities: &[(f64, f64)]) -> Vec<Vec<f64>> {
    cities.iter().map(|row| {
        cities.iter().map(|column| {
            ((column.0 - row.0).powi(2) + (column.1 - row.1).powi(2)).sqrt()
        }).collect::<Vec<f64>>()
    }).collect::<Vec<Vec<f64>>>()
}

/// Utility function to calculate the distance travelled following the specified route
///
/// `distance_matrix` is a `&Vec<Vec<f64>>` containing the distance matrix.
///
/// `route` is a `&Vec<usize>`, containing the route of the travelling salesman.
///
/// Returns an `f64`, representing the distance of the route travelled.
///
///# Examples
///
///```
///extern crate travelling_salesman;
///
///fn main() {
///    let cities = [
///      (27.0, 78.0),
///      (18.0, 24.0),
///      (48.0, 62.0),
///      (83.0, 77.0),
///      (55.0, 56.0),
///    ];
///
///    let route_distance = travelling_salesman::get_route_distance(
///      &travelling_salesman::get_distance_matrix(&cities),
///      &vec![0, 2, 3, 4, 1, 0]
///    );
///
///    println!("The route distance for the tour [0, 2, 3, 4, 1, 0] is {}", route_distance);
///}
///```
pub fn get_route_distance(distance_matrix: &Vec<Vec<f64>>, route: &Vec<usize>) -> f64 {
    let mut route_iter   = route.iter();
    let mut current_city = match route_iter.next() {
        None    => return 0.0,
        Some(v) => *v,
    };

    route_iter.fold(0.0, |mut total_distance, &next_city| {
        total_distance += distance_matrix[current_city as usize][next_city as usize];
        current_city    = next_city;
        total_distance
    })
}

pub trait Metaheuristics<T> {
    /// Clone the supplied candidate solution
    ///
    ///```
    /// let new_candidate = problem.clone_candidate(&old_candidate);
    ///```
    fn clone_candidate(&mut self, candidate: &T) -> T;

    /// Randomly generate a new candidate solution
    ///
    ///```
    /// let candidate = problem.generate_candidate();
    ///```
    fn generate_candidate(&mut self) -> T;

    /// Rank a candidate solution so that it can be compared with another (higher is better)
    ///
    ///```
    /// if problem.rank_candidate(&new_candidate) > problem.rank_candidate(&old_candidate) {
    ///     ...
    /// }
    ///```
    fn rank_candidate(&mut self, candidate: &T)  -> f64;

    /// Clone the supplied candidate solution, then make a small (but random) modification
    ///
    ///```
    /// let new_candidate = problem.tweak_candidate(&old_candidate);
    ///```
    fn tweak_candidate(&mut self, candidate: &T) -> T;
}

/// Returns an approximate solution to your optimisation problem using Simulated Annealing
///
///# Parameters
///
/// `problem` is the type that implements the `Metaheuristics` trait.
///
/// `runtime` is a `time::Duration` specifying how long to spend searching for a solution.
///
///# Examples
///
///```
///let solution = metaheuristics::simulated_annealing::solve(&mut problem, runtime);
///```
pub fn solve<T>(problem: &mut Metaheuristics<T>, runtime: Duration) -> T {
    let mut best_candidate      = problem.generate_candidate();
    let mut annealing_candidate = problem.tweak_candidate(&best_candidate);
    let start_time              = PreciseTime::now();
    let runtime_in_milliseconds = runtime.num_milliseconds() as f64;

    loop {
        let portion_elapsed = (start_time.to(PreciseTime::now()).num_milliseconds() as f64) / runtime_in_milliseconds;

        if portion_elapsed >= 1.0 {
            break;
        }

        let next_candidate        = problem.tweak_candidate(&annealing_candidate);
        let next_is_better        = problem.rank_candidate(&next_candidate) > problem.rank_candidate(&annealing_candidate);
        let replacement_threshold = 1.0f64.exp().powf(-10.0 * portion_elapsed.powf(3.0));

        if next_is_better || (thread_rng().gen_range(0.0, 1.0) < replacement_threshold) {
            annealing_candidate = next_candidate;
        }

        if problem.rank_candidate(&annealing_candidate) > problem.rank_candidate(&best_candidate) {
            best_candidate = problem.clone_candidate(&annealing_candidate);
        }
    }

    best_candidate
}
