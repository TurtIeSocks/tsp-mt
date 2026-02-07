use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    thread,
};

use rayon::prelude::*;

use crate::{
    Error, Result, SolverInput,
    config::LkhConfig,
    constants::{MIN_CYCLE_POINTS, PREP_CANDIDATES_FILE, PROBLEM_FILE, RUN_FILE},
    geometry::TourGeometry,
    node::LKHNode,
    options::SolverOptions,
    problem::TsplibProblemWriter,
    process::LkhProcess,
    projection::PlaneProjection,
    run_spec::RunSpec,
};

const PREP_RUN_INDEX: usize = 0;
const PREP_SEED: u64 = 1;
const THREAD_FALLBACK_PARALLELISM: usize = 2;
const THREAD_MIN_PARALLELISM: usize = 2;
const THREAD_RESERVED_CORES: usize = 1;

const ERR_LKH_PREPROCESS_FAILED: &str = "LKH preprocessing failed";
const ERR_MISSING_CANDIDATE_FILE: &str =
    "LKH finished but candidate file was not created (unexpected).";
const ERR_NO_RESULTS: &str = "No results";
const ERR_INVALID_POINT: &str = "Input contains invalid lat/lng values";
const ERR_INVALID_PROJECTION_RADIUS: &str = "projection_radius must be > 0";

pub(crate) struct LkhSolver<'a> {
    executable: &'a PathBuf,
    work_dir: &'a PathBuf,
    problem_file: PathBuf,
    candidate_file: PathBuf,
    pi_file: PathBuf,
}

impl<'a> LkhSolver<'a> {
    pub(crate) fn new(executable: &'a PathBuf, work_dir: &'a PathBuf) -> Self {
        let problem_file = work_dir.join(PROBLEM_FILE.tsp());
        let candidate_file = work_dir.join(PROBLEM_FILE.candidate());
        let pi_file = work_dir.join(PROBLEM_FILE.pi());

        Self {
            executable,
            work_dir,
            problem_file,
            candidate_file,
            pi_file,
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
        self.work_dir
    }

    pub(crate) fn create_work_dir(&self) -> Result<()> {
        fs::create_dir_all(self.work_dir)?;
        Ok(())
    }

    pub(crate) fn param_file(&self) -> String {
        format!(
            "\
PROBLEM_FILE = {}
CANDIDATE_FILE = {}
PI_FILE = {}
",
            self.problem_file.display(),
            self.candidate_file.display(),
            self.pi_file.display(),
        )
    }

    /// Write TSPLIB EUC_2D problem using projected XY.
    pub(crate) fn create_problem_file(&self, points: &[LKHNode]) -> Result<()> {
        TsplibProblemWriter::write_euc2d(&self.problem_file, PROBLEM_FILE.name(), points)
    }

    pub(crate) fn ensure_candidate_file(&self, n: usize) -> Result<()> {
        log::debug!("solver.preprocess: start n={n}");
        let prep_par = self.work_dir.join(PREP_CANDIDATES_FILE.par());
        let prep_tour = self.work_dir.join(PREP_CANDIDATES_FILE.tour());

        let rs = RunSpec::new(PREP_RUN_INDEX, PREP_SEED, prep_par.clone(), prep_tour);

        let prep_cfg = LkhConfig::preprocessing(n);
        rs.write_lkh_par(&prep_cfg, self)?;

        let out = self.run(&prep_par)?;

        LkhProcess::ensure_success(ERR_LKH_PREPROCESS_FAILED, &out)?;

        if !self.candidate_file.exists() {
            return Err(Error::other(ERR_MISSING_CANDIDATE_FILE));
        }

        log::debug!("solver.preprocess: done n={n}");
        Ok(())
    }

    pub(crate) fn run(&self, par_path: &Path) -> Result<Output> {
        Command::new(self.executable)
            .arg(par_path)
            .current_dir(self.work_dir)
            .output()
            .map_err(Error::from)
    }
}

impl<'a> Drop for LkhSolver<'a> {
    fn drop(&mut self) {
        rm_file(&self.candidate_file);
        rm_file(&self.pi_file);
    }
}

fn rm_file(pb: &Path) {
    if !pb.exists() {
        return;
    }
    let _ = fs::remove_file(pb);
}

/// Solve TSP by spawning multiple LKH processes in parallel with different SEEDs.
/// Returns best tour points.
#[tsp_mt_derive::timer("solver")]
pub fn solve_tsp_with_lkh_parallel(
    input: SolverInput,
    options: SolverOptions,
) -> Result<Vec<LKHNode>> {
    if input.n() < MIN_CYCLE_POINTS {
        return Err(Error::invalid_input(format!(
            "Need at least {MIN_CYCLE_POINTS} points for a cycle"
        )));
    }
    if options.projection_radius <= 0.0 {
        return Err(Error::invalid_input(ERR_INVALID_PROJECTION_RADIUS));
    }
    if input.points.iter().any(|p| !p.is_valid()) {
        return Err(Error::invalid_input(ERR_INVALID_POINT));
    }

    let cfg = LkhConfig::new(input.n());
    let solver = LkhSolver::new(&input.lkh_exe, &input.work_dir);
    solver.create_work_dir()?;

    let points = PlaneProjection::new(&input.points)
        .radius(options.projection_radius)
        .project();

    solver.create_problem_file(&points)?;
    solver.ensure_candidate_file(points.len())?;

    let parallelism = LkhSolver::threads();

    log::info!(
        "solver: start n={} time_limit_s={} threads={parallelism}",
        input.n(),
        cfg.time_limit()
    );

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallelism)
        .build()
        .map_err(|e| Error::other(format!("rayon pool: {e}")))?;

    let results: Vec<(Vec<usize>, f64)> = pool.install(|| {
        cfg.generate_seeds(parallelism)
            .into_par_iter()
            .enumerate()
            .map(|(idx, seed)| -> Result<(Vec<usize>, f64)> {
                let rs = RunSpec::new(
                    idx,
                    seed,
                    solver.work_dir().join(RUN_FILE.par_idx(idx)),
                    solver.work_dir().join(RUN_FILE.tour_idx(idx)),
                );
                rs.write_lkh_par(&cfg, &solver)?;

                log::debug!("solver.run: start idx={idx} seed={seed}");

                let out = solver.run(rs.par_path())?;

                LkhProcess::ensure_success(
                    &format!("LKH failed (run idx={}, seed={})", rs.idx(), rs.seed()),
                    &out,
                )?;

                let tour = rs.parse_tsplib_tour(points.len())?;
                let len = TourGeometry::tour_length(&points, &tour);

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

    Ok(best.0.into_iter().map(|idx| input.get_point(idx)).collect())
}
