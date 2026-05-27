//! Native Rust Lin-Kernighan symmetric Euclidean TSP solver.
//!
//! This crate is a from-scratch reimplementation of the algorithmic core of
//! Helsgaun's LKH-3 solver, scoped to the symmetric-Euclidean-TSP feature
//! subset actually used by `tsp_mt_core` (EUC_2D distance, ALPHA-default
//! candidates, sequential k-opt up to k=3, no genetic recombination, single
//! run per call, no subproblem partitioning).
//!
//! Public entry point: [`Solver::new`] + [`Solver::solve`].

pub mod alpha;
pub mod candidate;
pub mod coord;
pub mod distance;
pub mod error;
pub mod initial;
pub mod lk;
pub mod params;
pub mod problem;
pub mod recombine;
pub mod solver;
pub mod tour;

pub use candidate::CandidateSet;
pub use coord::Point2D;
pub use error::{Error, Result};
pub use params::Params;
pub use problem::Problem;
pub use solver::Solver;
