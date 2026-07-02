//! # tsp_ils
//!
//! A clean-room, multi-core TSP heuristic in pure Rust.
//!
//! The solver combines well-published techniques from the Lin-Kernighan
//! family of heuristics (Lin & Kernighan 1973; Helsgaun 2000; Or 1976;
//! Bentley 1992), implemented from the algorithm descriptions:
//!
//! * **Candidate lists** - symmetrized k-nearest-neighbor lists (k-d tree)
//!   restrict which edges local search may create, making each improvement
//!   step O(k) instead of O(n).
//! * **Greedy construction** - shortest candidate edges first, then nearest
//!   -endpoint chaining of the leftover path fragments.
//! * **2-opt + Or-opt local search** with don't-look bits and an explicit
//!   work queue; 2-opt reverses the shorter arc, Or-opt relocates segments
//!   of 1..=3 nodes in either orientation (and reverses in place).
//! * **Iterated local search** - windowed double-bridge kicks with
//!   best-tour restoration.
//! * **Multi-core split/join** - the tour is cut into contiguous segments;
//!   each is optimized independently as a fixed-endpoint sub-problem (a
//!   cycle with one frozen edge), then re-joined. Segment boundaries rotate
//!   every round so the whole tour gets interior optimization. Small
//!   instances run independent ILS walkers per core instead.
//! * **Spike repair** - endpoints of unusually long edges are re-optimized
//!   with extended Or-opt segment lengths, directly targeting route
//!   outliers.
//!
//! This crate contains no code from, and no code derived from, LKH or any
//! other existing TSP solver. It works on points in any const-generic
//! dimension `D`; geographic callers typically embed lat/lng on the unit
//! sphere (D=3, chord metric), planar callers use D=2.

mod candidates;
mod construct;
mod kdtree;
mod rng;
mod solve;
mod state;

pub use kdtree::dist;
pub use solve::{Solution, SolverConfig, cycle_length, solve};

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_permutation(tour: &[u32], n: usize) {
        assert_eq!(tour.len(), n);
        let mut seen = vec![false; n];
        for &v in tour {
            assert!(!seen[v as usize]);
            seen[v as usize] = true;
        }
    }

    fn fast_cfg() -> SolverConfig {
        SolverConfig {
            time_limit: Some(std::time::Duration::from_secs(3)),
            threads: 2,
            ..SolverConfig::default()
        }
    }

    #[test]
    fn trivial_sizes() {
        for n in 0..4usize {
            let pts: Vec<[f64; 2]> = (0..n).map(|i| [i as f64, 0.0]).collect();
            let sol = solve(&pts, &fast_cfg());
            assert_permutation(&sol.tour, n);
        }
    }

    #[test]
    fn solves_a_small_ring_optimally() {
        // Points on a circle: the optimal tour is the perimeter order.
        let n = 64usize;
        let pts: Vec<[f64; 2]> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64 * std::f64::consts::TAU;
                [1000.0 * t.cos(), 1000.0 * t.sin()]
            })
            .collect();
        let sol = solve(&pts, &fast_cfg());
        assert_permutation(&sol.tour, n);
        let optimal: f64 = (0..n).map(|i| dist(&pts[i], &pts[(i + 1) % n])).sum();
        assert!(
            sol.length <= optimal * 1.0001,
            "ring should be solved optimally: got {} want {}",
            sol.length,
            optimal
        );
    }

    #[test]
    fn length_matches_reported_tour() {
        let mut rng = rng::SplitMix64::new(31);
        let pts: Vec<[f64; 2]> = (0..500)
            .map(|_| {
                [
                    rng.next_below(1_000_000) as f64 / 100.0,
                    rng.next_below(1_000_000) as f64 / 100.0,
                ]
            })
            .collect();
        let sol = solve(&pts, &fast_cfg());
        assert_permutation(&sol.tour, pts.len());
        let recomputed = cycle_length(&pts, &sol.tour);
        assert!((recomputed - sol.length).abs() < 1e-6 * recomputed.max(1.0));
    }
}
