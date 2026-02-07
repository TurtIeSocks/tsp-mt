use std::{
    fs, io,
    path::{Path, PathBuf},
};

use crate::lkh::{config::LkhConfig, solver::LkhSolver};

#[derive(Clone, Debug)]
pub(crate) struct RunSpec {
    pub(crate) idx: usize,
    pub(crate) seed: u64,
    pub(crate) par_path: PathBuf,
    pub(crate) tour_path: PathBuf,
}

impl RunSpec {
    fn param_file(&self) -> String {
        format!(
            "OUTPUT_TOUR_FILE = {}\nSEED = {}",
            self.tour_path.display(),
            self.seed,
        )
    }

    pub(crate) fn write_lkh_par(&self, cfg: &LkhConfig, solver: &LkhSolver) -> io::Result<()> {
        let s = format!(
            "{}\n{}\n{}",
            self.param_file(),
            cfg.param_file(),
            solver.param_file()
        );

        fs::write(&self.par_path, s)
    }

    /// Small-problem par writer for centroid ordering.
    pub(crate) fn write_lkh_par_small(
        &self,
        problem_path: &Path,
        max_trials: usize,
        time_limit: usize,
    ) -> io::Result<()> {
        let s = format!(
            "\
PROBLEM_FILE = {}
OUTPUT_TOUR_FILE = {}
RUNS = 1
MAX_TRIALS = {}
SEED = {}
TRACE_LEVEL = 0
TIME_LIMIT = {}
MAX_CANDIDATES = 32 SYMMETRIC
",
            problem_path.display(),
            self.tour_path.display(),
            max_trials,
            self.seed,
            time_limit,
        );

        fs::write(&self.par_path, s)
    }

    pub(crate) fn parse_tsplib_tour(&self, n: usize) -> io::Result<Vec<usize>> {
        let text = fs::read_to_string(&self.tour_path)?;
        let mut in_section = false;
        let mut tour: Vec<usize> = Vec::with_capacity(n);

        for line in text.lines() {
            let line = line.trim();
            if line.eq_ignore_ascii_case("TOUR_SECTION") {
                in_section = true;
                continue;
            }
            if !in_section {
                continue;
            }
            if line == "-1" || line.eq_ignore_ascii_case("EOF") {
                break;
            }
            let id: isize = line.parse().map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Bad tour line '{line}': {e}"),
                )
            })?;
            if id <= 0 {
                continue;
            }
            tour.push((id as usize) - 1);
        }

        if tour.len() != n {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Expected {n} nodes in tour, got {}", tour.len()),
            ));
        }
        Ok(tour)
    }
}
