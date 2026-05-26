use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{candidate::CandidateSet, distance::euc_2d, error::Result, problem::Problem, tour::Tour, Error};

/// Bentley's multiple-fragment greedy heuristic for the initial tour.
/// Matches LKH's default `INITIAL_TOUR_ALGORITHM = GREEDY` path: pool
/// every candidate edge, sort by cost ascending, and add edges one by
/// one while maintaining the invariants
///
/// * no node ever exceeds degree 2, and
/// * no partial fragment closes into a sub-cycle until the *final* edge
///   closes the full Hamiltonian cycle.
///
/// Reference: Bentley 1992, "Fast algorithms for geometric TSP";
/// LKH `GreedyTour.c` for the symmetric path.
///
/// Falls back to a brute-force search for the last few edges if the
/// candidate set is too sparse to reach a complete cycle — same
/// guarantee LKH's `NearestInList` branch provides.
pub fn greedy_fragment(problem: &Problem, candidates: &CandidateSet) -> Tour {
    let n = problem.n();
    let coords = problem.coords();

    let mut degree = vec![0u8; n];
    let mut adj: Vec<[i32; 2]> = vec![[-1, -1]; n];
    let mut tail: Vec<u32> = (0..n as u32).collect();
    let mut edges_added = 0usize;

    // Build the sorted edge pool — each (a, b) appears once with a < b.
    let mut edges: Vec<(i64, u32, u32)> = Vec::with_capacity(n * 16);
    for i in 0..n {
        for c in candidates.of(i as u32) {
            if (c.to as usize) > i {
                edges.push((c.cost, i as u32, c.to));
            }
        }
    }
    edges.sort_unstable_by_key(|(cost, a, b)| (*cost, *a, *b));

    for (_cost, a, b) in edges {
        if edges_added == n {
            break;
        }
        if !may_add_edge(a, b, &degree, &tail, edges_added, n) {
            continue;
        }
        add_edge(a, b, &mut degree, &mut adj, &mut tail);
        edges_added += 1;
    }

    // Patch any remaining gaps with cheapest-edge brute force across
    // degree<2 endpoints. This handles the rare case where the candidate
    // set is too thin to close the tour on its own.
    while edges_added < n {
        let (a, b) = find_cheapest_patch(&degree, &tail, edges_added, n, coords);
        add_edge(a, b, &mut degree, &mut adj, &mut tail);
        edges_added += 1;
    }

    // Walk the fragment list to materialize a tour order.
    let mut order: Vec<u32> = Vec::with_capacity(n);
    let mut prev: i32 = -1;
    let mut cur: u32 = 0;
    for _ in 0..n {
        order.push(cur);
        let next = if adj[cur as usize][0] != prev {
            adj[cur as usize][0]
        } else {
            adj[cur as usize][1]
        };
        prev = cur as i32;
        cur = next as u32;
    }

    Tour::from_order(&order)
}

fn may_add_edge(
    a: u32,
    b: u32,
    degree: &[u8],
    tail: &[u32],
    edges_added: usize,
    n: usize,
) -> bool {
    if a == b {
        return false;
    }
    if degree[a as usize] == 2 || degree[b as usize] == 2 {
        return false;
    }
    // Would adding (a, b) create a sub-cycle? It does iff the other
    // endpoint of a's fragment is b — meaning a and b are already
    // connected via a path. Allowed only on the final edge.
    if tail[a as usize] == b && edges_added + 1 != n {
        return false;
    }
    true
}

fn add_edge(
    a: u32,
    b: u32,
    degree: &mut [u8],
    adj: &mut [[i32; 2]],
    tail: &mut [u32],
) {
    let a_slot = degree[a as usize] as usize;
    let b_slot = degree[b as usize] as usize;
    adj[a as usize][a_slot] = b as i32;
    adj[b as usize][b_slot] = a as i32;
    degree[a as usize] += 1;
    degree[b as usize] += 1;

    let new_tail_a = tail[b as usize];
    let new_tail_b = tail[a as usize];
    tail[new_tail_b as usize] = new_tail_a;
    tail[new_tail_a as usize] = new_tail_b;
}

fn find_cheapest_patch(
    degree: &[u8],
    tail: &[u32],
    edges_added: usize,
    n: usize,
    coords: &[crate::coord::Point2D],
) -> (u32, u32) {
    let endpoints: Vec<u32> = (0..n as u32)
        .filter(|&i| degree[i as usize] < 2)
        .collect();
    let mut best: Option<(i64, u32, u32)> = None;
    for &a in &endpoints {
        for &b in &endpoints {
            if a >= b {
                continue;
            }
            if !may_add_edge(a, b, degree, tail, edges_added, n) {
                continue;
            }
            let d = euc_2d(coords[a as usize], coords[b as usize]);
            if best.map(|(c, _, _)| d < c).unwrap_or(true) {
                best = Some((d, a, b));
            }
        }
    }
    let (_, a, b) = best.expect("at least one valid patch edge should exist while fragments remain open");
    (a, b)
}

/// Greedy nearest-neighbor walk (LKH's `NEAREST_NEIGHBOR` initial tour).
/// Kept for reference / fallback; the default path is now
/// [`greedy_fragment`].
pub fn greedy_nn(problem: &Problem, candidates: &CandidateSet, seed: u64) -> Tour {
    let n = problem.n();
    let coords = problem.coords();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let start = rng.random_range(0..n as u64) as u32;

    let mut visited = vec![false; n];
    let mut order: Vec<u32> = Vec::with_capacity(n);
    let mut current = start;

    visited[current as usize] = true;
    order.push(current);

    for _ in 1..n {
        let next = pick_next(current, &visited, candidates, coords).unwrap_or_else(|| {
            scan_unvisited(current, &visited, coords).expect("at least one unvisited node remains")
        });
        visited[next as usize] = true;
        order.push(next);
        current = next;
    }

    Tour::from_order(&order)
}

/// Validate and adopt a caller-supplied initial tour (the LKH
/// `INITIAL_TOUR_FILE` analogue, but in-memory).
pub fn from_initial(order: &[usize], problem: &Problem) -> Result<Tour> {
    let n = problem.n();
    if order.len() != n {
        return Err(Error::invalid_input(format!(
            "initial tour length {} != problem size {}",
            order.len(),
            n
        )));
    }
    let mut seen = vec![false; n];
    let mut as_u32: Vec<u32> = Vec::with_capacity(n);
    for &v in order {
        if v >= n {
            return Err(Error::invalid_input(format!(
                "initial tour contains out-of-range node {v} for problem size {n}"
            )));
        }
        if seen[v] {
            return Err(Error::invalid_input(format!(
                "initial tour contains duplicate node {v}"
            )));
        }
        seen[v] = true;
        as_u32.push(v as u32);
    }
    Ok(Tour::from_order(&as_u32))
}

fn pick_next(
    current: u32,
    visited: &[bool],
    candidates: &CandidateSet,
    coords: &[crate::coord::Point2D],
) -> Option<u32> {
    let _ = coords;
    for c in candidates.of(current) {
        if !visited[c.to as usize] {
            return Some(c.to);
        }
    }
    None
}

fn scan_unvisited(
    current: u32,
    visited: &[bool],
    coords: &[crate::coord::Point2D],
) -> Option<u32> {
    use crate::distance::euc_2d;
    let mut best: Option<(u32, i64)> = None;
    for (i, was) in visited.iter().enumerate() {
        if *was {
            continue;
        }
        let d = euc_2d(coords[current as usize], coords[i]);
        match best {
            None => best = Some((i as u32, d)),
            Some((_, b)) if d < b => best = Some((i as u32, d)),
            _ => {}
        }
    }
    best.map(|(node, _)| node)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::Point2D;

    fn unit_square_problem() -> Problem {
        Problem::new(vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
        ])
        .expect("valid problem")
    }

    #[test]
    fn greedy_nn_visits_every_node_exactly_once() {
        let p = unit_square_problem();
        let cs = CandidateSet::build_nn(&p, 3);
        let t = greedy_nn(&p, &cs, 42);
        assert_eq!(t.n(), 4);
        let mut sorted = t.as_slice().to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0u32, 1, 2, 3]);
    }

    #[test]
    fn greedy_fragment_produces_valid_tour() {
        let p = unit_square_problem();
        let cs = CandidateSet::build_nn(&p, 3);
        let t = greedy_fragment(&p, &cs);
        assert_eq!(t.n(), 4);
        let mut sorted = t.as_slice().to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0u32, 1, 2, 3]);
    }

    #[test]
    fn greedy_fragment_finds_optimal_on_unit_square() {
        let p = unit_square_problem();
        let cs = CandidateSet::build_nn(&p, 3);
        let t = greedy_fragment(&p, &cs);
        assert_eq!(t.length(&p), 4);
    }

    #[test]
    fn from_initial_accepts_valid_order() {
        let p = unit_square_problem();
        let t = from_initial(&[2, 1, 0, 3], &p).expect("valid initial tour");
        assert_eq!(t.as_slice(), &[2u32, 1, 0, 3]);
    }

    #[test]
    fn from_initial_rejects_duplicate() {
        let p = unit_square_problem();
        let err = from_initial(&[2, 1, 0, 2], &p).expect_err("duplicate should fail");
        assert!(err.to_string().contains("duplicate"));
    }

    #[test]
    fn from_initial_rejects_wrong_length() {
        let p = unit_square_problem();
        let err = from_initial(&[0, 1, 2], &p).expect_err("length mismatch should fail");
        assert!(err.to_string().contains("length"));
    }
}
