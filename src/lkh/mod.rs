//! High-performance TSP solving on geographic coordinates using external LKH runs.
//! Includes direct parallel solving and H3 chunked solving for large inputs.

mod chunked;
mod config;
mod constants;
mod geometry;
mod h3_chunking;
mod options;
mod point;
mod problem;
mod process;
mod projection;
mod run_spec;
mod solver;
mod stitching;

use std::{io, path::PathBuf};

pub use options::SolverOptions;
pub use point::Point;

/// Solve TSP using parallel independent LKH runs and return the best tour.
pub fn solve_tsp_with_lkh_parallel(
    lkh_exe: PathBuf,
    work_dir: PathBuf,
    input: &[Point],
) -> io::Result<Vec<Point>> {
    solve_tsp_with_lkh_parallel_with_options(lkh_exe, work_dir, input, &SolverOptions::default())
}

/// Solve TSP with explicit runtime options.
pub fn solve_tsp_with_lkh_parallel_with_options(
    lkh_exe: PathBuf,
    work_dir: PathBuf,
    input: &[Point],
    options: &SolverOptions,
) -> io::Result<Vec<Point>> {
    solver::solve_tsp_with_lkh_parallel_with_options(lkh_exe, work_dir, input, options)
}

/// Solve TSP with H3 chunking for large inputs, falling back to parallel mode for small inputs.
pub fn solve_tsp_with_lkh_h3_chunked(
    lkh_exe: PathBuf,
    work_dir: PathBuf,
    input: &[Point],
) -> io::Result<Vec<Point>> {
    solve_tsp_with_lkh_h3_chunked_with_options(lkh_exe, work_dir, input, &SolverOptions::default())
}

/// Solve TSP with H3 chunking and explicit runtime options.
pub fn solve_tsp_with_lkh_h3_chunked_with_options(
    lkh_exe: PathBuf,
    work_dir: PathBuf,
    input: &[Point],
    options: &SolverOptions,
) -> io::Result<Vec<Point>> {
    chunked::solve_tsp_with_lkh_h3_chunked_with_options(lkh_exe, work_dir, input, options)
}
