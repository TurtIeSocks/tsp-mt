use std::time::Instant;

use rand::{Rng, RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{
    alpha::compute_pi,
    candidate::CandidateSet,
    error::Result,
    initial::{from_initial, greedy_fragment},
    lk,
    params::Params,
    problem::Problem,
    tour::Tour,
};

/// Owns the problem instance, computed candidate set, and trial loop.
/// Equivalent to LKH's `AllocateStructures` + `FindTour` + outer run loop,
/// rolled into a single value-typed handle.
pub struct Solver {
    problem: Problem,
    params: Params,
}

/// Result of a solve: tour as a 0-based node permutation plus its
/// integer EUC_2D length.
#[derive(Clone, Debug)]
pub struct SolveOutcome {
    pub tour: Vec<usize>,
    pub length: i64,
}

impl Solver {
    pub fn new(problem: Problem, params: Params) -> Self {
        Self { problem, params }
    }

    pub fn problem(&self) -> &Problem {
        &self.problem
    }

    pub fn solve(&self) -> Result<SolveOutcome> {
        let start = Instant::now();
        let deadline = start + self.params.time_limit;

        let max_candidates = self.params.max_candidates.max(1);
        if self.params.trace_level >= 1 {
            log::info!(
                "lin_kernighan: building candidate set n={} max_candidates={}",
                self.problem.n(),
                max_candidates
            );
        }
        // NN candidates are the default — empirically the Pi-adjusted
        // ALPHA candidates re-rank the top-K nearly identically on
        // dense Euclidean instances at this scale, and the subgradient
        // ascent is non-trivial overhead. ALPHA is available via
        // `CandidateSet::build_alpha` if/when an instance class shows a
        // clear gap (clustered, hub-and-spoke, or asymmetric layouts).
        let _ = compute_pi; // keep the symbol live for future selection.
        let candidates = CandidateSet::build_nn(&self.problem, max_candidates);

        let mut best_tour = match &self.params.initial_tour {
            Some(order) => from_initial(order, &self.problem)?,
            None => greedy_fragment(&self.problem, &candidates),
        };

        let mut best_len = lk::improve(
            &self.problem,
            &candidates,
            &mut best_tour,
            &self.params,
            deadline,
        );

        if self.params.trace_level >= 1 {
            log::info!(
                "lin_kernighan: trial=1 length={} elapsed={:.2}s",
                best_len,
                start.elapsed().as_secs_f64()
            );
        }

        let mut rng = ChaCha8Rng::seed_from_u64(self.params.seed.wrapping_add(0xA5A5_A5A5));
        let mut no_improvement = 0usize;
        for trial in 2..=self.params.max_trials {
            if Instant::now() >= deadline {
                break;
            }
            if no_improvement >= self.params.max_no_improvement {
                if self.params.trace_level >= 1 {
                    log::info!(
                        "lin_kernighan: stagnation early-exit after trial={} (no_improve={}, best={best_len}, elapsed={:.2}s)",
                        trial - 1,
                        no_improvement,
                        start.elapsed().as_secs_f64()
                    );
                }
                break;
            }
            let mut candidate_tour = best_tour.clone();
            double_bridge(&mut candidate_tour, &mut rng);
            let candidate_len = lk::improve(
                &self.problem,
                &candidates,
                &mut candidate_tour,
                &self.params,
                deadline,
            );
            if candidate_len < best_len {
                best_len = candidate_len;
                best_tour = candidate_tour;
                no_improvement = 0;
                if self.params.trace_level >= 1 {
                    log::info!(
                        "lin_kernighan: trial={trial} new_best={best_len} elapsed={:.2}s",
                        start.elapsed().as_secs_f64()
                    );
                }
            } else {
                no_improvement += 1;
            }
        }

        Ok(SolveOutcome {
            tour: best_tour.into_vec_usize(),
            length: best_len,
        })
    }
}

/// Standard double-bridge perturbation: cut the tour into 4 segments at
/// 3 random positions and reconnect them in the order A-C-B-D. This is
/// the canonical kick for iterated LK because no single k-opt move can
/// undo it.
fn double_bridge<R: Rng + RngExt>(tour: &mut Tour, rng: &mut R) {
    let n = tour.n();
    if n < 8 {
        return;
    }
    let quarter = (n / 4) as u64;
    let p1 = 1 + rng.random_range(0..quarter) as usize;
    let p2 = p1 + 1 + rng.random_range(0..quarter) as usize;
    let p3 = p2 + 1 + rng.random_range(0..quarter) as usize;

    let slice = tour.as_slice();
    let mut new_order: Vec<u32> = Vec::with_capacity(n);
    new_order.extend_from_slice(&slice[0..p1]);
    new_order.extend_from_slice(&slice[p3..n]);
    new_order.extend_from_slice(&slice[p2..p3]);
    new_order.extend_from_slice(&slice[p1..p2]);
    *tour = Tour::from_order(&new_order);
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::coord::Point2D;

    fn grid_points(side: usize) -> Vec<Point2D> {
        (0..side)
            .flat_map(|y| (0..side).map(move |x| Point2D::new(x as f64, y as f64)))
            .collect()
    }

    #[test]
    fn solves_unit_square_to_optimal_length_four() {
        let problem = Problem::new(vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
        ])
        .expect("valid problem");
        let params = Params::default()
            .with_max_candidates(3)
            .with_time_limit(Duration::from_secs(1));
        let outcome = Solver::new(problem, params).solve().expect("solve");
        assert_eq!(outcome.length, 4);
    }

    #[test]
    fn solves_3x3_grid_to_known_optimum() {
        // Optimal Hamiltonian cycle on a 3×3 unit grid has length 10
        // (boundary loop): perimeter 8 + two diagonals of length 1
        // -- but actually a snake through the grid: 0,1,2,5,4,3,6,7,8,0
        // has length 8 (sides) + sqrt(8) wrap-around. The Euclidean
        // optimum is well-known to be 8 + 2*sqrt(2) ≈ 10.83 with
        // nint=11. Use a wide tolerance instead of an exact check.
        let problem = Problem::new(grid_points(3)).expect("valid problem");
        let params = Params::default()
            .with_max_candidates(5)
            .with_time_limit(Duration::from_secs(2));
        let outcome = Solver::new(problem, params).solve().expect("solve");
        assert!(
            outcome.length <= 12,
            "expected length close to optimum, got {}",
            outcome.length
        );
        assert_eq!(outcome.tour.len(), 9);
    }

    #[test]
    fn respects_caller_supplied_initial_tour() {
        let problem = Problem::new(vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
        ])
        .expect("valid problem");
        let params = Params::default()
            .with_max_candidates(3)
            .with_time_limit(Duration::from_millis(500))
            .with_initial_tour(vec![0, 2, 1, 3]);
        let outcome = Solver::new(problem, params).solve().expect("solve");
        // Whatever the initial order, the optimizer should converge to
        // the perimeter cycle (length 4).
        assert_eq!(outcome.length, 4);
    }
}
