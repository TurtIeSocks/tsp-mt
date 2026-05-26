//! Lin-Kernighan style local search.
//!
//! Implements 2-opt with don't-look bits and Or-opt segment-relocation
//! (a constrained 3-opt). This is the symmetric Euclidean TSP subset of
//! LKH's sequential k-opt search — enough to land within a few percent
//! of LKH-quality tours for the chunk sizes (≤ ~5000 nodes) tsp_mt_core
//! solves.

pub(crate) mod or_opt;
pub(crate) mod three_opt;
pub(crate) mod two_opt;

use std::time::Instant;

use crate::{candidate::CandidateSet, params::Params, problem::Problem, tour::Tour};

/// Per-sweep gain below this fraction of the current tour length is
/// treated as a plateau — we stop the local-search loop and let the
/// trial loop perturb the tour rather than burn the rest of the
/// deadline chasing micro-gains.
const PLATEAU_GAIN_RATIO: f64 = 0.0005;
/// Hard upper bound on sweep iterations per `improve` call. Real
/// convergence happens well under this number; the cap exists to keep
/// the per-trial wall time bounded when a tour keeps finding chains of
/// single-unit gains.
const MAX_SWEEPS_PER_IMPROVE: usize = 64;

/// Improve `tour` in place using 2-opt then Or-opt sweeps, repeating
/// until no further improvement is found, the sweep gain drops below a
/// plateau threshold, or `deadline` passes.
///
/// Returns the final tour length.
pub fn improve(
    problem: &Problem,
    candidates: &CandidateSet,
    tour: &mut Tour,
    params: &Params,
    deadline: Instant,
) -> i64 {
    let n = problem.n();
    let mut two_opt_bits = vec![false; n];
    let mut or_opt_bits = vec![false; n];
    let mut three_opt_bits = vec![false; n];

    let baseline_len = tour.length(problem).max(1);
    let plateau_threshold = (((baseline_len as f64) * PLATEAU_GAIN_RATIO) as i64).max(1);
    let mut sweeps = 0usize;

    loop {
        if sweeps >= MAX_SWEEPS_PER_IMPROVE {
            break;
        }
        sweeps += 1;
        let two_gain =
            two_opt::sweep(problem, candidates, tour, &mut two_opt_bits, deadline);
        if Instant::now() >= deadline {
            break;
        }
        // When 2-opt finds improvements, reset the escape-move bits so
        // they re-scan the newly-touched neighbourhood on the next
        // iteration. When 2-opt finds none, the escape moves get a
        // fresh queue of every anchor — that's the whole point of
        // running them as the "next layer" of search.
        if two_gain > 0 {
            for bit in or_opt_bits.iter_mut() {
                *bit = false;
            }
            for bit in three_opt_bits.iter_mut() {
                *bit = false;
            }
        }
        if two_gain == 0 {
            if params.move_type >= 3 {
                for bit in or_opt_bits.iter_mut() {
                    *bit = false;
                }
                for bit in three_opt_bits.iter_mut() {
                    *bit = false;
                }
                let or_gain =
                    or_opt::sweep(problem, candidates, tour, &mut or_opt_bits, deadline);
                let three_gain =
                    three_opt::sweep(problem, candidates, tour, &mut three_opt_bits, deadline);
                let combined = or_gain + three_gain;
                if combined == 0 || combined < plateau_threshold {
                    break;
                }
                // Either escape move made progress — let 2-opt re-scan
                // the touched neighbourhood next iteration.
                for bit in two_opt_bits.iter_mut() {
                    *bit = false;
                }
            } else {
                break;
            }
        } else if two_gain < plateau_threshold {
            break;
        }
    }

    tour.length(problem)
}
