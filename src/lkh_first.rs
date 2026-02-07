use std::{
    fs, io,
    path::{Path, PathBuf},
    process::Command,
    thread,
    time::Instant,
};

use geo::Coord;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;

use crate::{project::Plane, utils::Point};

#[derive(Clone, Debug)]
pub struct LkhConfig {
    /// Trials per process. For n=5000, 10k–20k is a good start.
    max_trials: usize,
    /// LKH RUNS per process. Keep at 1 when parallelizing externally.
    runs: usize,
    /// Base seed used to generate per-run seeds.
    base_seed: u64,
    /// LKH TRACE_LEVEL (0..=3ish; 1 is nice).
    trace_level: usize,
    /// Seconds to run LKH (10..=60)
    time_limit: usize,
}

impl LkhConfig {
    pub fn new(n: usize) -> Self {
        Self {
            max_trials: (n * 3).max(1_000).min(100_000),
            time_limit: (n / 512).max(2),
            ..Default::default()
        }
    }

    fn threads(&self) -> usize {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2)
            .max(2)
            - 1
    }

    /// Deterministic seed generation; replace with your favorite scheme.
    /// Produces `count` distinct u64 seeds from `base_seed`.
    fn generate_seeds(&self, count: usize) -> Vec<u64> {
        let mut rng = StdRng::seed_from_u64(self.base_seed);
        (0..count).map(|_| rng.random::<u64>()).collect()
    }
}

impl Default for LkhConfig {
    fn default() -> Self {
        Self {
            max_trials: 10000,
            runs: 1,
            base_seed: 12345,
            trace_level: 1,
            time_limit: 60,
        }
    }
}

struct LKHSolver<'a> {
    executable: &'a Path,
    work_dir: &'a Path,
    problem_file: PathBuf,
    candidate_file: PathBuf,
    pi_file: PathBuf,
}

impl<'a> LKHSolver<'a> {
    fn new(executable: &'a Path, work_dir: &'a Path) -> Self {
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

    fn create_work_dir(&self) -> io::Result<()> {
        fs::create_dir_all(self.work_dir)
    }

    /// Write TSPLIB EUC_2D problem using projected XY.
    fn create_problem_file(&self, points: &[Coord]) -> io::Result<()> {
        let mut s = String::new();
        s.push_str("NAME: problem\n");
        s.push_str("TYPE: TSP\n");
        s.push_str(&format!("DIMENSION: {}\n", points.len()));
        s.push_str("EDGE_WEIGHT_TYPE: EUC_2D\n");
        s.push_str("NODE_COORD_SECTION\n");
        for (i, p) in points.iter().enumerate() {
            // TSPLIB is 1-based
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

    fn ensure_candidate_file(&self, n: usize) -> io::Result<()> {
        let prep_par = self.work_dir.join("prep_candidates.par");
        let prep_tour = self.work_dir.join("prep_candidates.tour");

        let rs = RunSpec {
            idx: 0,
            par_path: prep_par.clone(),
            seed: 1,
            tour_path: prep_tour,
        };

        rs.write_lkh_par(
            &self.problem_file,
            &self.candidate_file,
            &self.pi_file,
            &LkhConfig {
                max_trials: n,
                runs: 1,
                time_limit: 1,
                ..Default::default()
            },
        )?;

        eprintln!("Creating prep file");
        let out = Command::new(self.executable)
            .arg(&prep_par)
            .current_dir(self.work_dir)
            .output()?;

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

        eprintln!("Finished prep file");
        if !self.candidate_file.exists() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "LKH finished but candidate file was not created (unexpected).",
            ));
        }

        Ok(())
    }
}

fn rm_file(pb: &PathBuf) {
    if !pb.exists() {
        return;
    }
    match fs::remove_file(pb) {
        Ok(_) => return,
        Err(err) => eprintln!("Unable to remove {} file: {}", pb.display(), err),
    }
}

impl<'a> Drop for LKHSolver<'a> {
    fn drop(&mut self) {
        rm_file(&self.candidate_file);
        rm_file(&self.pi_file);
    }
}

/// Solve TSP by spawning multiple LKH processes in parallel with different SEEDs.
/// Returns (best_tour, best_length).
pub fn solve_tsp_with_lkh_parallel(
    lkh_exe: &Path,
    work_dir: &Path,
    input: &[Point],
) -> io::Result<Vec<Point>> {
    if input.len() < 3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Need at least 3 points for a cycle",
        ));
    }
    let cfg = LkhConfig::new(input.len());

    let solver = LKHSolver::new(lkh_exe, work_dir);
    solver.create_work_dir()?;
    // fs::create_dir_all(solver.work_dir)?;

    // match fs::create_dir_all(solver.work_dir) {
    //   Ok(_) => {},
    //   Err(err) => return Err(io::Error::new(
    //         io::ErrorKind::InvalidFilename,
    //         format!("Issue with workdir: {}", err),
    //     ))
    // }

    let plane = Plane::new(&input.to_vec()).radius(70.0);
    let points = plane.project();

    solver.create_problem_file(&points)?;

    let parallelism = cfg.threads();

    eprintln!(
        "Starting LKH for {} points and will run for {}s across {parallelism} threads",
        input.len(),
        cfg.time_limit,
    );

    // // Prepare per-run directories/files to avoid collisions.
    // let runs: Vec<RunSpec> = cfg
    //     .generate_seeds(parallelism)
    //     .into_iter()
    //     .enumerate()
    //     .map(|(i, seed)| RunSpec {
    //         idx: i,
    //         seed,
    //         par_path: work_dir.join(format!("run_{i}.par")),
    //         tour_path: work_dir.join(format!("run_{i}.tour")),
    //     })
    //     .collect();

    // // Write parameter files.
    // for r in &runs {}

    solver.ensure_candidate_file(points.len())?;

    // write_lkh_candidate_file_knn_alpha0(&solver.candidate_file, &points, 64)?;

    // Use a local rayon pool sized to `parallelism`.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallelism)
        .build()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("rayon pool: {e}")))?;

    // Execute LKH runs in parallel and collect results.
    let mut results: Vec<(Vec<usize>, f64)> = pool.install(|| {
        cfg.generate_seeds(parallelism)
            .into_par_iter()
            .enumerate()
            .map(|(idx, seed)| -> io::Result<(Vec<usize>, f64)> {
                let rs = RunSpec {
                    idx,
                    seed,
                    par_path: work_dir.join(format!("run_{idx}.par")),
                    tour_path: work_dir.join(format!("run_{idx}.tour")),
                };
                rs.write_lkh_par(
                    &solver.problem_file,
                    &solver.candidate_file,
                    &solver.pi_file,
                    &cfg,
                )?;

                let now = Instant::now();

                eprintln!("Starting tour for thread {idx}");
                let out = Command::new(solver.executable)
                    .arg(&rs.par_path)
                    .current_dir(work_dir)
                    .output()?;

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
                let len = tour_length(&points, &tour);

                eprintln!(
                    "Finished tour for thread {idx} - took {:.2}: {len:.0}m",
                    now.elapsed().as_secs_f32()
                );
                Ok((tour, len))
            })
            .collect::<io::Result<Vec<_>>>()
    })?;

    // Pick best (min length).
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let tour = results
        .into_iter()
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "No results"))?;
    Ok(tour.0.into_iter().map(|idx| input[idx]).collect())
}

#[derive(Clone, Debug)]
struct RunSpec {
    idx: usize,
    seed: u64,
    par_path: PathBuf,
    tour_path: PathBuf,
}

impl RunSpec {
    /// Minimal .par writer with SEED.
    /// RUNS should usually be 1 when you parallelize externally.
    fn write_lkh_par(
        &self,
        problem_path: &Path,
        candidate_path: &Path,
        pi_path: &Path,
        cfg: &LkhConfig,
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
CANDIDATE_FILE = {}
PI_FILE = {}
SUBGRADIENT = NO
MAX_CANDIDATES = 24 SYMMETRIC
",
            problem_path.display(),
            self.tour_path.display(),
            cfg.runs,
            cfg.max_trials,
            self.seed,
            cfg.trace_level,
            cfg.time_limit,
            candidate_path.display(),
            pi_path.display(),
        );

        fs::write(&self.par_path, s)
    }

    /// Parse TSPLIB TOUR file -> 0-based indices.
    fn parse_tsplib_tour(&self, n: usize) -> io::Result<Vec<usize>> {
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

/// Compute tour length (cycle) in your projected space.
/// Replace with your own distance metric if needed.
fn tour_length(points: &[Coord], tour: &[usize]) -> f64 {
    let n = tour.len();
    let mut sum = 0.0;
    for i in 0..n {
        let a = points[tour[i]];
        let b = points[tour[(i + 1) % n]];
        sum += dist(a, b);
    }
    sum
}

#[inline]
fn dist(a: Coord, b: Coord) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

// /// Writes an LKH CANDIDATE_FILE using kNN from kiddo, with Alpha=0.
// /// - Node IDs in the file are 1-based
// /// - DadId is written as 0 (unknown) for all nodes
// pub fn write_lkh_candidate_file_knn_alpha0(
//     path: &Path,
//     points: &[Coord],
//     k: usize,
// ) -> io::Result<()> {
//     const KD_TREE: usize = 2;
//     let n = points.len();
//     if n < 3 {
//         return Err(io::Error::new(
//             io::ErrorKind::InvalidInput,
//             "need >= 3 points",
//         ));
//     }
//     if k == 0 {
//         return Err(io::Error::new(io::ErrorKind::InvalidInput, "k must be > 0"));
//     }

//     // Build entries as [[f64; 2]] because kiddo's convenient conversion uses this shape. :contentReference[oaicite:2]{index=2}
//     let entries: Vec<[f64; 2]> = points.iter().map(|p| [p.x, p.y]).collect();

//     // Build the KD-tree from a reference to the entries.
//     // In kiddo 5.x, KdTree<_, 2> can be constructed via (&entries).into(). :contentReference[oaicite:3]{index=3}
//     let tree: KdTree<f64, KD_TREE> = (&entries).into();

//     // For each point i, query k+1 nearest (includes itself at distance 0), then drop self.
//     let knn_lists: Vec<Vec<u32>> = (0..n)
//         .into_par_iter()
//         .map(|i| {
//             let query = &entries[i];
//             let want = (k + 1).min(n);

//             // Returns Vec<NearestNeighbour { distance, item }>, where item is the index into `entries`. :contentReference[oaicite:4]{index=4}
//             let mut nn = tree.nearest_n::<SquaredEuclidean>(query, want);

//             // Remove self, keep first k
//             let mut out = Vec::with_capacity(k.min(n - 1));
//             for hit in nn.drain(..) {
//                 let j = hit.item as usize;
//                 if j != i {
//                     out.push((j + 1) as u32); // convert 0-based index to 1-based node id
//                     if out.len() == k || out.len() == n - 1 {
//                         break;
//                     }
//                 }
//             }
//             out
//         })
//         .collect();

//     // Write the candidate file
//     let f = File::create(path)?;
//     let mut w = BufWriter::new(f);

//     writeln!(w, "{n}")?;
//     for i in 0..n {
//         let id = (i + 1) as u32;
//         let dad_id = 0u32;

//         let cands = &knn_lists[i];
//         write!(w, "{id} {dad_id} {} ", cands.len())?;
//         for &to_id in cands {
//             write!(w, "{to_id} 0 ")?; // Alpha=0 experiment
//         }
//         writeln!(w)?;
//     }

//     w.write_all(b"-1\nEOF\n")?;
//     w.flush()?;
//     Ok(())
// }
