use std::{
    fs, io,
    path::{Path, PathBuf},
    process::{Command, Output},
    thread,
    time::Instant,
};

use geo::Coord;
use rayon::prelude::*;

use crate::{
    lkh::{config::LkhConfig, geometry::TourGeometry, run_spec::RunSpec},
    project::Plane,
    utils::Point,
};

pub(crate) struct LkhSolver {
    executable: PathBuf,
    work_dir: PathBuf,
    problem_file: PathBuf,
    candidate_file: PathBuf,
    pi_file: PathBuf,
}

impl LkhSolver {
    pub(crate) fn new(executable: PathBuf, work_dir: PathBuf) -> Self {
        let problem_file = work_dir.join("problem.tsp");
        let candidate_file = work_dir.join("problem.cand");
        let pi_file = work_dir.join("problem.pi");

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
            .unwrap_or(2)
            .max(2)
            - 1
    }

    pub(crate) fn work_dir(&self) -> &Path {
        &self.work_dir
    }

    pub(crate) fn create_work_dir(&self) -> io::Result<()> {
        fs::create_dir_all(&self.work_dir)
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
    pub(crate) fn create_problem_file(&self, points: &[Coord]) -> io::Result<()> {
        let mut s = String::new();
        s.push_str("NAME: problem\n");
        s.push_str("TYPE: TSP\n");
        s.push_str(&format!("DIMENSION: {}\n", points.len()));
        s.push_str("EDGE_WEIGHT_TYPE: EUC_2D\n");
        s.push_str("NODE_COORD_SECTION\n");
        for (i, p) in points.iter().enumerate() {
            s.push_str(&format!(
                "{} {:.0} {:.0}\n",
                i + 1,
                p.x * 1000.0,
                p.y * 1000.0
            ));
        }
        s.push_str("EOF\n");
        fs::write(&self.problem_file, s)?;
        Ok(())
    }

    pub(crate) fn ensure_candidate_file(&self, n: usize) -> io::Result<()> {
        let prep_par = self.work_dir.join("prep_candidates.par");
        let prep_tour = self.work_dir.join("prep_candidates.tour");

        let rs = RunSpec {
            idx: 0,
            par_path: prep_par.clone(),
            seed: 1,
            tour_path: prep_tour,
        };

        let prep_cfg = LkhConfig::preprocessing(n);
        rs.write_lkh_par(&prep_cfg, self)?;

        let out = self.run(&prep_par)?;

        if !out.status.success() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "LKH preprocessing failed.\nSTDOUT:\n{}\nSTDERR:\n{}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr),
                ),
            ));
        }

        if !self.candidate_file.exists() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "LKH finished but candidate file was not created (unexpected).",
            ));
        }

        Ok(())
    }

    pub(crate) fn run(&self, par_path: &Path) -> io::Result<Output> {
        Command::new(&self.executable)
            .arg(par_path)
            .current_dir(&self.work_dir)
            .output()
    }
}

impl Drop for LkhSolver {
    fn drop(&mut self) {
        rm_file(&self.candidate_file);
        rm_file(&self.pi_file);
    }
}

fn rm_file(pb: &Path) {
    if !pb.exists() {
        return;
    }
    if let Err(err) = fs::remove_file(pb) {
        eprintln!("Unable to remove file {}: {}", pb.display(), err);
    }
}

/// Solve TSP by spawning multiple LKH processes in parallel with different SEEDs.
/// Returns best tour points.
pub fn solve_tsp_with_lkh_parallel(
    lkh_exe: PathBuf,
    work_dir: PathBuf,
    input: &[Point],
) -> io::Result<Vec<Point>> {
    if input.len() < 3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Need at least 3 points for a cycle",
        ));
    }

    let cfg = LkhConfig::new(input.len());
    let solver = LkhSolver::new(lkh_exe, work_dir);
    solver.create_work_dir()?;

    let points = Plane::new(&input.to_vec()).radius(70.0).project();

    solver.create_problem_file(&points)?;
    solver.ensure_candidate_file(points.len())?;

    let parallelism = LkhSolver::threads();

    eprintln!(
        "Starting LKH for {} points and will run for {}s across {parallelism} threads",
        input.len(),
        cfg.time_limit(),
    );

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallelism)
        .build()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("rayon pool: {e}")))?;

    let mut results: Vec<(Vec<usize>, f64)> = pool.install(|| {
        cfg.generate_seeds(parallelism)
            .into_par_iter()
            .enumerate()
            .map(|(idx, seed)| -> io::Result<(Vec<usize>, f64)> {
                let rs = RunSpec {
                    idx,
                    seed,
                    par_path: solver.work_dir().join(format!("run_{idx}.par")),
                    tour_path: solver.work_dir().join(format!("run_{idx}.tour")),
                };
                rs.write_lkh_par(&cfg, &solver)?;

                let now = Instant::now();
                eprintln!("Starting tour for thread {idx}");

                let out = solver.run(&rs.par_path)?;

                if !out.status.success() {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!(
                            "LKH failed (run idx={}, seed={}).\nSTDOUT:\n{}\nSTDERR:\n{}",
                            rs.idx, rs.seed, stdout, stderr
                        ),
                    ));
                }

                let tour = rs.parse_tsplib_tour(points.len())?;
                let len = TourGeometry::tour_length(&points, &tour);

                eprintln!(
                    "Finished tour for thread {idx} - took {:.2}s: {len:.0}m",
                    now.elapsed().as_secs_f32()
                );
                Ok((tour, len))
            })
            .collect::<io::Result<Vec<_>>>()
    })?;

    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let tour = results
        .into_iter()
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "No results"))?;

    Ok(tour.0.into_iter().map(|idx| input[idx]).collect())
}
