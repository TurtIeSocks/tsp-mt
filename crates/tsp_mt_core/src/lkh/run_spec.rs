use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::{Error, Result, config::LkhConfig, solver::LkhSolver};

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

    pub(crate) fn write_lkh_par(&self, cfg: &LkhConfig, solver: &LkhSolver) -> Result<()> {
        let s = format!(
            "{}\n{}\n{}",
            self.param_file(),
            cfg.param_file(),
            solver.param_file()
        );

        fs::write(&self.par_path, s)?;
        Ok(())
    }

    /// Small-problem par writer for centroid ordering.
    pub(crate) fn write_lkh_par_small(
        &self,
        problem_path: &Path,
        max_trials: usize,
        time_limit: usize,
    ) -> Result<()> {
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

        fs::write(&self.par_path, s)?;
        Ok(())
    }

    pub(crate) fn parse_tsplib_tour(&self, n: usize) -> Result<Vec<usize>> {
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
            let id: isize = line
                .parse()
                .map_err(|e| Error::invalid_data(format!("Bad tour line '{line}': {e}")))?;
            if id < MIN_VALID_TSPLIB_NODE_ID {
                continue;
            }
            tour.push((id as usize) - TSPLIB_NODE_ID_OFFSET);
        }

        if tour.len() != n {
            return Err(Error::invalid_data(format!(
                "Expected {n} nodes in tour, got {}",
                tour.len()
            )));
        }
        Ok(tour)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::RunSpec;

    fn unique_temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("tsp-mt-tests-{name}-{nanos}"))
    }

    #[test]
    fn parse_tsplib_tour_reads_tour_section_and_converts_to_zero_based() {
        let dir = unique_temp_dir("parse-ok");
        fs::create_dir_all(&dir).expect("create temp dir");

        let rs = RunSpec::new(0, 7, dir.join("run.par"), dir.join("run.tour"));
        fs::write(
            rs.tour_path.clone(),
            "NAME : test\nTOUR_SECTION\n2\n1\n3\n-1\nEOF\n",
        )
        .expect("write tour file");

        let parsed = rs.parse_tsplib_tour(3).expect("parse tsplib tour");
        assert_eq!(parsed, vec![1, 0, 2]);

        fs::remove_dir_all(&dir).expect("cleanup temp dir");
    }

    #[test]
    fn parse_tsplib_tour_errors_on_wrong_node_count() {
        let dir = unique_temp_dir("parse-count");
        fs::create_dir_all(&dir).expect("create temp dir");

        let rs = RunSpec::new(0, 7, dir.join("run.par"), dir.join("run.tour"));
        fs::write(rs.tour_path.clone(), "TOUR_SECTION\n1\n-1\nEOF\n").expect("write tour file");

        let err = rs
            .parse_tsplib_tour(2)
            .expect_err("expected node-count mismatch");
        let msg = err.to_string();
        assert!(msg.contains("Expected 2 nodes in tour, got 1"));

        fs::remove_dir_all(&dir).expect("cleanup temp dir");
    }

    #[test]
    fn write_lkh_par_small_writes_expected_fields() {
        let dir = unique_temp_dir("small-par");
        fs::create_dir_all(&dir).expect("create temp dir");

        let par_path = dir.join("run.par");
        let rs = RunSpec::new(3, 99, par_path.clone(), dir.join("run.tour"));
        rs.write_lkh_par_small(&dir.join("p.tsp"), 1234, 9)
            .expect("write lkh par small");

        let text = fs::read_to_string(par_path).expect("read par");
        assert!(text.contains("PROBLEM_FILE = "));
        assert!(text.contains("OUTPUT_TOUR_FILE = "));
        assert!(text.contains("RUNS = 1"));
        assert!(text.contains("MAX_TRIALS = 1234"));
        assert!(text.contains("SEED = 99"));
        assert!(text.contains("TIME_LIMIT = 9"));
        assert!(text.contains("MAX_CANDIDATES = 32 SYMMETRIC"));

        fs::remove_dir_all(&dir).expect("cleanup temp dir");
    }
}
