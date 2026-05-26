//! ALPHA candidate-set construction (LKH `CreateCandidateSet.c` core).
//!
//! Computes Held-Karp Pi-values via subgradient ascent on the
//! minimum 1-tree, then uses the Pi-adjusted (Lagrangian) edge cost
//!
//! ```text
//!     c'(i, j) = c(i, j) + π[i] + π[j]
//! ```
//!
//! to rank each node's nearest neighbours. We don't compute the full
//! α-cost (which requires β-values on the MST) — instead we sort by
//! `c'(i, j)` directly. The two approximations differ by a constant for
//! edges sharing an MST node, so candidate ranking is essentially the
//! same for most instances and avoids O(n²) β storage.

use crate::{coord::Point2D, distance::euc_2d, problem::Problem};

/// Maximum subgradient ascent iterations.
const MAX_ASCENT_ITERATIONS: usize = 50;
/// Initial step size scales with the average tour edge to give Pi
/// enough leverage to actually re-rank candidate edges. Without
/// scaling, a constant `step=1.0` is invisible against integer costs
/// in the 100–1000 range typical of our scaled lat/lng coordinates.
const INITIAL_STEP_RATIO: f64 = 0.1;
/// Period (in iterations) over which the step is held constant before
/// being halved.
const ASCENT_PERIOD: usize = 5;

/// Subgradient-ascent Pi values, indexed by node id. Length == n.
pub struct PiValues(pub Vec<f64>);

impl PiValues {
    pub fn n(&self) -> usize {
        self.0.len()
    }
}

/// Compute Held-Karp Pi values for the symmetric Euclidean problem.
///
/// Returns `PiValues(vec![0.0; n])` for tiny instances where the 1-tree
/// machinery doesn't apply.
pub fn compute_pi(problem: &Problem) -> PiValues {
    let n = problem.n();
    if n < 4 {
        return PiValues(vec![0.0; n]);
    }
    let coords = problem.coords();
    let mut pi = vec![0.0f64; n];
    // Seed step magnitude from the average raw EUC_2D edge over a small
    // sample of the candidate graph. Doesn't have to be exact — we just
    // need it within an order of magnitude of typical edge cost.
    let mut sample_total: f64 = 0.0;
    let mut sample_count: usize = 0;
    let sample_stride = (n / 16).max(1);
    for i in (0..n).step_by(sample_stride) {
        for j in ((i + 1)..n).step_by(sample_stride) {
            sample_total += euc_2d(coords[i], coords[j]) as f64;
            sample_count += 1;
            if sample_count >= 64 {
                break;
            }
        }
        if sample_count >= 64 {
            break;
        }
    }
    let avg_edge = if sample_count == 0 {
        1.0
    } else {
        sample_total / sample_count as f64
    };
    let mut step = avg_edge * INITIAL_STEP_RATIO;
    let mut best_lower_bound = f64::NEG_INFINITY;
    let mut best_pi = pi.clone();

    for iter in 0..MAX_ASCENT_ITERATIONS {
        let (one_tree_cost, degrees) = build_one_tree(coords, &pi);
        // Pi-aware lower bound: L_lower = c(1-tree) - 2·sum(π).
        let pi_sum: f64 = pi.iter().sum();
        let lower_bound = one_tree_cost - 2.0 * pi_sum;
        if lower_bound > best_lower_bound {
            best_lower_bound = lower_bound;
            best_pi.copy_from_slice(&pi);
        }
        // Subgradient update: π_i += t · (V_i − 2).
        let mut all_two = true;
        for i in 0..n {
            let v = degrees[i] as i32;
            if v != 2 {
                all_two = false;
            }
            pi[i] += step * (v as f64 - 2.0);
        }
        if all_two {
            break;
        }
        if (iter + 1) % ASCENT_PERIOD == 0 {
            step *= 0.5;
            if step < 1e-6 {
                break;
            }
        }
    }

    PiValues(best_pi)
}

/// Build the minimum 1-tree under the Pi-adjusted cost
/// `c'(i, j) = c(i, j) + π[i] + π[j]`.
///
/// Returns `(one_tree_cost, degrees)` where `degrees[i]` is the degree
/// of node `i` in the 1-tree.
fn build_one_tree(coords: &[Point2D], pi: &[f64]) -> (f64, Vec<u32>) {
    let n = coords.len();
    // Pick node 0 as the "special" node. We build the MST on V \ {0}
    // then attach 0 via its two cheapest edges back to the MST.
    let special = 0usize;

    // Prim's MST over V \ {special}.
    let mut in_tree = vec![false; n];
    let mut nearest_cost = vec![f64::INFINITY; n];
    let mut nearest_from = vec![usize::MAX; n];
    let mut degrees = vec![0u32; n];

    // Seed: pick any non-special node as the first in the MST.
    let seed = if special == 0 { 1 } else { 0 };
    in_tree[seed] = true;
    in_tree[special] = true; // exclude from MST iteration
    for j in 0..n {
        if j == special || j == seed {
            continue;
        }
        let c = mod_cost(coords, pi, seed, j);
        nearest_cost[j] = c;
        nearest_from[j] = seed;
    }

    let mut mst_cost: f64 = 0.0;
    for _ in 0..(n - 2) {
        // n − 1 nodes outside `special`, already 1 in tree, so n − 2 picks.
        let mut best = f64::INFINITY;
        let mut best_node = usize::MAX;
        for j in 0..n {
            if in_tree[j] {
                continue;
            }
            if nearest_cost[j] < best {
                best = nearest_cost[j];
                best_node = j;
            }
        }
        if best_node == usize::MAX {
            break;
        }
        in_tree[best_node] = true;
        let parent = nearest_from[best_node];
        degrees[best_node] += 1;
        degrees[parent] += 1;
        mst_cost += best;
        // Relax outgoing edges from `best_node`.
        for j in 0..n {
            if in_tree[j] {
                continue;
            }
            let c = mod_cost(coords, pi, best_node, j);
            if c < nearest_cost[j] {
                nearest_cost[j] = c;
                nearest_from[j] = best_node;
            }
        }
    }
    // Re-enable special so we can find its 2 cheapest edges to the MST.
    in_tree[special] = false;
    let mut best1 = (f64::INFINITY, usize::MAX);
    let mut best2 = (f64::INFINITY, usize::MAX);
    for j in 0..n {
        if j == special {
            continue;
        }
        let c = mod_cost(coords, pi, special, j);
        if c < best1.0 {
            best2 = best1;
            best1 = (c, j);
        } else if c < best2.0 {
            best2 = (c, j);
        }
    }
    let one_tree_cost = mst_cost + best1.0 + best2.0;
    degrees[special] = 2;
    if best1.1 != usize::MAX {
        degrees[best1.1] += 1;
    }
    if best2.1 != usize::MAX {
        degrees[best2.1] += 1;
    }
    (one_tree_cost, degrees)
}

#[inline]
fn mod_cost(coords: &[Point2D], pi: &[f64], i: usize, j: usize) -> f64 {
    (euc_2d(coords[i], coords[j]) as f64) + pi[i] + pi[j]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_from(points: &[(f64, f64)]) -> Problem {
        Problem::new(points.iter().map(|&(x, y)| Point2D::new(x, y)).collect())
            .expect("valid problem")
    }

    #[test]
    fn pi_values_length_matches_problem() {
        let p = problem_from(&[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
        let pi = compute_pi(&p);
        assert_eq!(pi.n(), 4);
    }

    #[test]
    fn one_tree_cost_strictly_positive() {
        let p = problem_from(&[
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (5.0, 5.0),
        ]);
        let (cost, degrees) = build_one_tree(p.coords(), &vec![0.0; p.n()]);
        assert!(cost > 0.0);
        assert_eq!(degrees.len(), 5);
    }
}
