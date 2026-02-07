use std::{
    fs, io,
    path::{Path, PathBuf},
};

use crate::{config::LkhConfig, solver::LkhSolver};

const SMALL_RUNS: usize = 1;
const SMALL_TRACE_LEVEL: usize = 0;
const SMALL_MAX_CANDIDATES: usize = 32;

const TOUR_SECTION_HEADER: &str = "TOUR_SECTION";
const TOUR_END_MARKER: &str = "-1";
const EOF_MARKER: &str = "EOF";
const MIN_VALID_TSPLIB_NODE_ID: isize = 1;
const TSPLIB_NODE_ID_OFFSET: usize = 1;

#[derive(Clone, Debug)]
pub(crate) struct RunSpec {
    idx: usize,
    seed: u64,
    par_path: PathBuf,
    tour_path: PathBuf,
}

impl RunSpec {
    pub(crate) fn new(idx: usize, seed: u64, par_path: PathBuf, tour_path: PathBuf) -> Self {
        Self {
            idx,
            seed,
            par_path,
            tour_path,
        }
    }

    pub(crate) fn idx(&self) -> usize {
        self.idx
    }

    pub(crate) fn seed(&self) -> u64 {
        self.seed
    }

    pub(crate) fn par_path(&self) -> &Path {
        &self.par_path
    }

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
RUNS = {}
MAX_TRIALS = {}
SEED = {}
TRACE_LEVEL = {}
TIME_LIMIT = {}
MAX_CANDIDATES = {} SYMMETRIC
",
            problem_path.display(),
            self.tour_path.display(),
            SMALL_RUNS,
            max_trials,
            self.seed,
            SMALL_TRACE_LEVEL,
            time_limit,
            SMALL_MAX_CANDIDATES,
        );

        fs::write(&self.par_path, s)
    }

    pub(crate) fn parse_tsplib_tour(&self, n: usize) -> io::Result<Vec<usize>> {
        let text = fs::read_to_string(&self.tour_path)?;
        let mut in_section = false;
        let mut tour: Vec<usize> = Vec::with_capacity(n);

        for line in text.lines() {
            let line = line.trim();
            if line.eq_ignore_ascii_case(TOUR_SECTION_HEADER) {
                in_section = true;
                continue;
            }
            if !in_section {
                continue;
            }
            if line == TOUR_END_MARKER || line.eq_ignore_ascii_case(EOF_MARKER) {
                break;
            }
            let id: isize = line.parse().map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Bad tour line '{line}': {e}"),
                )
            })?;
            if id < MIN_VALID_TSPLIB_NODE_ID {
                continue;
            }
            tour.push((id as usize) - TSPLIB_NODE_ID_OFFSET);
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
