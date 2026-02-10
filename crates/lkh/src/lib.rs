//! `lkh` provides a typed Rust API for generating TSPLIB/LKH input files,
//! running LKH, and parsing TSPLIB tour output.
//!
//! It is intended for both internal and external use where consumers want to:
//! - build TSPLIB problems in Rust,
//! - write LKH parameter files,
//! - invoke LKH as a subprocess,
//! - parse returned tours into `Vec<usize>` orderings.
//!
//! # Quickstart
//!
//! ```no_run
//! use lkh::{
//!     parameters::LkhParameters,
//!     problem::TsplibProblem,
//!     solver::LkhSolver,
//! };
//!
//! fn main() -> lkh::LkhResult<()> {
//!     let problem = TsplibProblem::from_euc2d_points(vec![
//!         (0.0, 0.0),
//!         (1.0, 0.0),
//!         (0.0, 1.0),
//!     ]);
//!
//!     let params = LkhParameters::new("work/problem.tsp");
//!     let solver = LkhSolver::new(problem, params)?;
//!     let tour = solver.run_with_exe("/usr/local/bin/LKH")?;
//!     let order = tour.zero_based_tour()?;
//!     println!("{order:?}");
//!     Ok(())
//! }
//! ```
//!
//! See the crate README for feature flags, embedded-binary behavior, security
//! notes, and platform-specific build requirements.

pub mod parameters;
pub mod problem;
pub mod process;
pub mod solver;
pub mod tour;

#[cfg(feature = "fetch-lkh")]
pub mod embedded_lkh;

mod error;
mod spec_writer;

pub use error::{LkhError, LkhResult};
