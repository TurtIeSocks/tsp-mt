use std::{
    fs,
    path::{Path, PathBuf},
    process::Output,
    thread,
};

use lkh::{
    parameters::LkhParameters, problem::TsplibProblemWriter, process::LkhProcess, tour::TsplibTour,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;

use crate::{
    Error, Result, SolverInput, Tour, constants::MIN_CYCLE_POINTS, file_cleanup, geometry,
    node::LKHNode, options::SolverOptions, projection::PlaneProjection,
};

const PREP_SEED: u64 = 1;
const THREAD_FALLBACK_PARALLELISM: usize = 2;
const THREAD_MIN_PARALLELISM: usize = 2;
const THREAD_RESERVED_CORES: usize = 1;

const PROBLEM_BASENAME: &str = "problem";
const PREP_BASENAME: &str = "prep_candidates";
const RUN_BASENAME: &str = "run";

const PAR_EXTENSION: &str = ".par";
const TOUR_EXTENSION: &str = ".tour";
const TSP_EXTENSION: &str = ".tsp";
const CANDIDATE_EXTENSION: &str = ".cand";
const PI_EXTENSION: &str = ".pi";

const ERR_LKH_PREPROCESS_FAILED: &str = "LKH preprocessing failed";
const ERR_MISSING_CANDIDATE_FILE: &str =
    "LKH finished but candidate file was not created (unexpected).";
const ERR_NO_RESULTS: &str = "No results";
const ERR_INVALID_POINT: &str = "Input contains invalid lat/lng values";
const ERR_INVALID_PROJECTION_RADIUS: &str = "projection_radius must be > 0";

const PARALLEL_RUNS: usize = 1;
const PARALLEL_TRACE_LEVEL: usize = 1;
const DEFAULT_MAX_CANDIDATES: usize = 32;
const DEFAULT_BASE_SEED: u64 = 12_345;
const MAX_TRIALS_MULTIPLIER: usize = 3;
const MIN_MAX_TRIALS: usize = 1_000;
const MAX_MAX_TRIALS: usize = 100_000;
const TIME_LIMIT_DIVISOR: usize = 512;
const MIN_TIME_LIMIT_SECONDS: usize = 2;

const PREPROCESS_RUNS: usize = 1;
const PREPROCESS_TIME_LIMIT_SECONDS: usize = 1;

fn generate_seeds(base_seed: u64, count: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(base_seed);
    (0..count).map(|_| rng.random::<u64>()).collect()
}

#[derive(Clone, Debug)]
struct LkhWorkFiles {
    work_dir: PathBuf,
}

impl LkhWorkFiles {
    fn new(work_dir: &Path) -> Self {
        Self {
            work_dir: work_dir.to_path_buf(),
        }
    }

    fn work_dir(&self) -> &Path {
        &self.work_dir
    }

    fn problem_tsp(&self) -> PathBuf {
        self.work_dir
            .join(file_name(PROBLEM_BASENAME, TSP_EXTENSION))
    }

    fn problem_candidate(&self) -> PathBuf {
        self.work_dir
            .join(file_name(PROBLEM_BASENAME, CANDIDATE_EXTENSION))
    }

    fn problem_pi(&self) -> PathBuf {
        self.work_dir
            .join(file_name(PROBLEM_BASENAME, PI_EXTENSION))
    }

    fn prep_par(&self) -> PathBuf {
        self.work_dir.join(file_name(PREP_BASENAME, PAR_EXTENSION))
    }

    fn prep_tour(&self) -> PathBuf {
        self.work_dir.join(file_name(PREP_BASENAME, TOUR_EXTENSION))
    }

    fn run_par(&self, idx: usize) -> PathBuf {
        self.work_dir
            .join(indexed_file_name(RUN_BASENAME, idx, PAR_EXTENSION))
    }

    fn run_tour(&self, idx: usize) -> PathBuf {
        self.work_dir
            .join(indexed_file_name(RUN_BASENAME, idx, TOUR_EXTENSION))
    }
}

fn file_name(stem: &str, extension: &str) -> String {
    format!("{stem}{extension}")
}

fn indexed_file_name(stem: &str, idx: usize, extension: &str) -> String {
    format!("{stem}_{idx}{extension}")
}

pub(crate) struct LkhSolver {
    executable: PathBuf,
    files: LkhWorkFiles,
}

impl LkhSolver {
    pub(crate) fn new(executable: &Path, work_dir: &Path) -> Self {
        Self {
            executable: executable.to_path_buf(),
            files: LkhWorkFiles::new(work_dir),
        }
    }

    pub(crate) fn threads() -> usize {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(THREAD_FALLBACK_PARALLELISM)
            .max(THREAD_MIN_PARALLELISM)
            - THREAD_RESERVED_CORES
    }

    pub(crate) fn work_dir(&self) -> &Path {
        self.files.work_dir()
    }

    pub(crate) fn run_par_path(&self, idx: usize) -> PathBuf {
        self.files.run_par(idx)
    }

    pub(crate) fn run_tour_path(&self, idx: usize) -> PathBuf {
        self.files.run_tour(idx)
    }

    pub(crate) fn create_work_dir(&self) -> Result<()> {
        fs::create_dir_all(self.work_dir())?;
        Ok(())
    }

    pub(crate) fn create_problem_file(&self, points: &[LKHNode]) -> Result<()> {
        TsplibProblemWriter::write_euc2d(
            &self.files.problem_tsp(),
            PROBLEM_BASENAME,
            points.iter().map(|p| (p.x, p.y)),
        )?;
        Ok(())
    }

    pub(crate) fn parallel_run_config(&self, n: usize) -> LkhParameters {
        let mut cfg = LkhParameters::new(self.files.problem_tsp());
        cfg.runs = Some(PARALLEL_RUNS);
        cfg.max_trials = Some((n * MAX_TRIALS_MULTIPLIER).clamp(MIN_MAX_TRIALS, MAX_MAX_TRIALS));
        cfg.trace_level = Some(PARALLEL_TRACE_LEVEL);
        cfg.time_limit = Some(((n / TIME_LIMIT_DIVISOR).max(MIN_TIME_LIMIT_SECONDS)) as f64);
        cfg.max_candidates = Some(lkh::parameters::CandidateLimit::new(
            DEFAULT_MAX_CANDIDATES,
            true,
        ));
        cfg.seed = Some(DEFAULT_BASE_SEED);
        cfg.candidate_files.push(self.files.problem_candidate());
        cfg.pi_file = Some(self.files.problem_pi());
        cfg
    }

    fn preprocessing_config(&self, n: usize) -> LkhParameters {
        let mut cfg = LkhParameters::new(self.files.problem_tsp());
        cfg.runs = Some(PREPROCESS_RUNS);
        cfg.max_trials = Some(n);
        cfg.trace_level = Some(PARALLEL_TRACE_LEVEL);
        cfg.time_limit = Some(PREPROCESS_TIME_LIMIT_SECONDS as f64);
        cfg.max_candidates = Some(lkh::parameters::CandidateLimit::new(
            DEFAULT_MAX_CANDIDATES,
            true,
        ));
        cfg.seed = Some(DEFAULT_BASE_SEED);
        cfg.candidate_files.push(self.files.problem_candidate());
        cfg.pi_file = Some(self.files.problem_pi());
        cfg
    }

    pub(crate) fn ensure_candidate_file(&self, n: usize) -> Result<()> {
        log::debug!("solver.preprocess: start n={n}");

        let prep_par = self.files.prep_par();
        let prep_tour = self.files.prep_tour();
        let prep_cfg = self
            .preprocessing_config(n)
            .with_seed(PREP_SEED)
            .with_output_tour_file(&prep_tour);

        prep_cfg.write_to_file(&prep_par)?;

        self.run(&prep_par, ERR_LKH_PREPROCESS_FAILED)?;

        let candidate_file = self.files.problem_candidate();
        if !candidate_file.exists() {
            return Err(Error::other(ERR_MISSING_CANDIDATE_FILE));
        }

        log::debug!("solver.preprocess: done n={n}");
        Ok(())
    }

    pub(crate) fn run(&self, par_path: &Path, context: impl ToString) -> Result<Output> {
        LkhProcess::default()
            .run(par_path, context)
            .map_err(Error::from)
    }
}

/// Solve TSP by spawning multiple LKH processes in parallel with different SEEDs.
/// Returns best tour points.
#[tsp_mt_derive::timer("solver")]
pub fn solve_tsp_with_lkh_parallel(input: SolverInput, options: SolverOptions) -> Result<Tour> {
    file_cleanup::register_workdir_for_shutdown_cleanup(&options.work_dir);

    if input.n() < MIN_CYCLE_POINTS {
        return Err(Error::invalid_input(format!(
            "Need at least {MIN_CYCLE_POINTS} points for a cycle"
        )));
    }
    if options.projection_radius <= 0.0 {
        return Err(Error::invalid_input(ERR_INVALID_PROJECTION_RADIUS));
    }
    if input.nodes.iter().any(|p| !p.is_valid()) {
        return Err(Error::invalid_input(ERR_INVALID_POINT));
    }

    let solver = LkhSolver::new(&options.lkh_exe, &options.work_dir);
    solver.create_work_dir()?;

    let points = PlaneProjection::new(&input.nodes)
        .radius(options.projection_radius)
        .project();

    solver.create_problem_file(&points)?;
    solver.ensure_candidate_file(points.len())?;

    let cfg = solver.parallel_run_config(input.n());
    let parallelism = LkhSolver::threads();

    log::info!(
        "solver: start n={} time_limit_s={:.0} threads={parallelism}",
        input.n(),
        cfg.time_limit.unwrap_or(0.0)
    );

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallelism)
        .build()
        .map_err(|e| Error::other(format!("rayon pool: {e}")))?;

    let results: Vec<(Vec<usize>, f64)> = pool.install(|| {
        generate_seeds(cfg.seed.unwrap_or(DEFAULT_BASE_SEED), parallelism)
            .into_par_iter()
            .enumerate()
            .map(|(idx, seed)| -> Result<(Vec<usize>, f64)> {
                let par_path = solver.run_par_path(idx);
                let tour_path = solver.run_tour_path(idx);
                let run_cfg = cfg
                    .clone()
                    .with_seed(seed)
                    .with_output_tour_file(&tour_path);

                run_cfg.write_to_file(&par_path)?;

                log::debug!("solver.run: start idx={idx} seed={seed}");

                solver.run(
                    &par_path,
                    &format!("LKH failed (run idx={idx}, seed={seed})"),
                )?;

                let tour = TsplibTour::parse_tsplib_tour(&tour_path, points.len())?;
                let len = geometry::tour_length(&points, &tour);

                log::debug!("solver.run: done idx={idx} seed={seed} tour_m={len:.0}");
                Ok((tour, len))
            })
            .collect::<Result<Vec<_>>>()
    })?;

    let run_count = results.len();
    let best = results
        .into_iter()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .ok_or_else(|| Error::other(ERR_NO_RESULTS))?;
    log::info!(
        "solver: complete runs={run_count} best_tour_m={:.0}",
        best.1
    );

    file_cleanup::cleanup_workdir(&options.work_dir);

    Ok(Tour::new(
        best.0.into_iter().map(|idx| input.get_point(idx)).collect(),
    ))
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::{LkhSolver, file_name, indexed_file_name};

    fn unique_temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("tsp-mt-tests-{name}-{nanos}"))
    }

    #[test]
    fn threads_is_at_least_one() {
        assert!(LkhSolver::threads() >= 1);
    }

    #[test]
    fn create_work_dir_creates_target_directory() {
        let dir = unique_temp_dir("solver-workdir");
        let solver = LkhSolver::new(Path::new("/tmp/lkh"), &dir);

        solver.create_work_dir().expect("create work dir");
        assert!(dir.exists());

        fs::remove_dir_all(&dir).expect("cleanup temp dir");
    }

    #[test]
    fn naming_helpers_create_expected_file_names() {
        assert_eq!(file_name("run", ".par"), "run.par");
        assert_eq!(indexed_file_name("run", 2, ".tour"), "run_2.tour");
    }
}
