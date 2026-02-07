//! High-performance TSP solving on geographic coordinates using external LKH runs.
//! Includes direct parallel solving and H3 chunked solving for large inputs.

mod chunked;
mod config;
mod constants;
mod geometry;
mod h3_chunking;
mod input;
mod options;
mod point;
mod problem;
mod process;
mod projection;
mod run_spec;
mod solver;
mod stitching;

pub use chunked::solve_tsp_with_lkh_h3_chunked;
pub use input::SolverInput;
pub use options::SolverOptions;
pub use point::Point;
pub use solver::solve_tsp_with_lkh_parallel;
