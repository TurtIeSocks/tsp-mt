//! High-performance TSP solving on geographic coordinates using external LKH runs.
//! Includes direct parallel solving and H3 chunked solving for large inputs.

mod algo;
mod constants;
mod error;
pub mod file_cleanup;
mod geo;
mod io;
pub mod logging;
mod node;
pub mod runner;
mod tour;

pub(crate) use algo::{h3_chunking, stitching};
pub(crate) use geo::{geometry, projection};
pub(crate) use io::options;

pub use algo::{chunked::solve_tsp_with_lkh_h3_chunked, solver::solve_tsp_with_lkh_parallel};
pub use error::{Error, Result};
pub use io::input::SolverInput;
pub use io::options::SolverOptions;
pub use node::LKHNode;
pub use tour::Tour;
