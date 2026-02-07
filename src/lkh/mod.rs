use std::{
    collections::HashMap,
    fs, io,
    path::{Path, PathBuf},
    process::Command,
    thread,
    time::Instant,
};

use geo::Coord;
use h3o::{CellIndex, LatLng, Resolution};
use kiddo::{KdTree, SquaredEuclidean};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;

use crate::{project::Plane, utils::Point};

#[derive(Clone, Debug)]
pub struct LkhConfig {
    /// Trials per process.
    max_trials: usize,
    /// LKH RUNS per process.
    runs: usize,
    /// Base seed used to generate per-run seeds.
    base_seed: u64,
    /// LKH TRACE_LEVEL.
    trace_level: usize,
    /// Seconds to run LKH
    time_limit: usize,
    max_candidates: usize,
}
// SUBGRADIENT = NO

impl LkhConfig {
    pub fn new(n: usize) -> Self {
        Self {
            max_trials: (n * 3).max(1_000).min(100_000),
            time_limit: (n / 512).max(2),
            ..Default::default()
        }
    }

    /// Deterministic seed generation; replace with your favorite scheme.
    /// Produces `count` distinct u64 seeds from `base_seed`.
    fn generate_seeds(&self, count: usize) -> Vec<u64> {
        let mut rng = StdRng::seed_from_u64(self.base_seed);
        (0..count).map(|_| rng.random::<u64>()).collect()
    }

    fn param_file(&self) -> String {
        format!(
            "\
RUNS = {}
MAX_TRIALS = {}
TRACE_LEVEL = {}
TIME_LIMIT = {}
MAX_CANDIDATES = {} SYMMETRIC
",
            self.runs, self.max_trials, self.trace_level, self.time_limit, self.max_candidates
        )
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
            max_candidates: 32,
        }
    }
}

struct LKHSolver {
    executable: PathBuf,
    work_dir: PathBuf,
    problem_file: PathBuf,
    candidate_file: PathBuf,
    pi_file: PathBuf,
}

impl LKHSolver {
    fn new(executable: PathBuf, work_dir: PathBuf) -> Self {
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

    fn threads() -> usize {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2)
            .max(2)
            - 1
    }

    fn create_work_dir(&self) -> io::Result<()> {
        fs::create_dir_all(&self.work_dir)
    }

    fn param_file(&self) -> String {
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
    fn create_problem_file(&self, points: &[Coord]) -> io::Result<()> {
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
            &LkhConfig {
                max_trials: n,
                runs: 1,
                time_limit: 1,
                ..Default::default()
            },
            &self,
        )?;

        let out = Command::new(&self.executable)
            .arg(&prep_par)
            .current_dir(&self.work_dir)
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
    if let Err(err) = fs::remove_file(pb) {
        eprintln!("Unable to remove file {}: {}", pb.display(), err);
    }
}

impl Drop for LKHSolver {
    fn drop(&mut self) {
        rm_file(&self.candidate_file);
        rm_file(&self.pi_file);
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
    let solver = LKHSolver::new(lkh_exe, work_dir);
    solver.create_work_dir()?;

    let points = Plane::new(&input.to_vec()).radius(70.0).project();

    solver.create_problem_file(&points)?;
    solver.ensure_candidate_file(points.len())?;

    let parallelism = LKHSolver::threads();

    eprintln!(
        "Starting LKH for {} points and will run for {}s across {parallelism} threads",
        input.len(),
        cfg.time_limit,
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
                    par_path: solver.work_dir.join(format!("run_{idx}.par")),
                    tour_path: solver.work_dir.join(format!("run_{idx}.tour")),
                };
                rs.write_lkh_par(&cfg, &solver)?;

                let now = Instant::now();
                eprintln!("Starting tour for thread {idx}");

                let out = Command::new(&solver.executable)
                    .arg(&rs.par_path)
                    .current_dir(&solver.work_dir)
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

#[derive(Clone, Debug)]
struct RunSpec {
    idx: usize,
    seed: u64,
    par_path: PathBuf,
    tour_path: PathBuf,
}

impl RunSpec {
    fn param_file(&self) -> String {
        format!(
            "OUTPUT_TOUR_FILE = {}\nSEED = {}",
            self.tour_path.display(),
            self.seed,
        )
    }

    fn write_lkh_par(&self, cfg: &LkhConfig, solver: &LKHSolver) -> io::Result<()> {
        let s = format!(
            "{}\n{}\n{}",
            self.param_file(),
            cfg.param_file(),
            solver.param_file()
        );

        fs::write(&self.par_path, s)
    }

    /// Small-problem par writer for centroid ordering.
    fn write_lkh_par_small(
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
        //     PI_FILE = 0
        // SUBGRADIENT = NO

        fs::write(&self.par_path, s)
    }

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

//
// --------------------- H3 chunking ---------------------
//

fn next_resolution(r: Resolution) -> Resolution {
    use Resolution::*;
    match r {
        Zero => One,
        One => Two,
        Two => Three,
        Three => Four,
        Four => Five,
        Five => Six,
        Six => Seven,
        Seven => Eight,
        Eight => Nine,
        Nine => Ten,
        Ten => Eleven,
        Eleven => Twelve,
        Twelve => Thirteen,
        Thirteen => Fourteen,
        Fourteen => Fifteen,
        Fifteen => Fifteen,
    }
}

fn bucket_by_res(
    input: &[Point],
    res: Resolution,
    subset: Option<&[usize]>,
) -> HashMap<CellIndex, Vec<usize>> {
    let mut map: HashMap<CellIndex, Vec<usize>> = HashMap::new();
    let iter: Box<dyn Iterator<Item = usize>> = match subset {
        Some(idxs) => Box::new(idxs.iter().copied()),
        None => Box::new((0..input.len()).into_iter()),
    };
    for i in iter {
        let p = &input[i];
        let ll = LatLng::new(p.lat, p.lng).expect("valid lat/lng");
        let cell = ll.to_cell(res);
        map.entry(cell).or_default().push(i);
    }
    map
}

fn max_bucket(map: &HashMap<CellIndex, Vec<usize>>) -> usize {
    map.values().map(|v| v.len()).max().unwrap_or(0)
}

fn h3_partition_indices(input: &[Point], max_bucket_sz: usize) -> Vec<Vec<usize>> {
    const MAX_RES: Resolution = Resolution::Fifteen;

    let mut res = Resolution::Four;
    let mut buckets = bucket_by_res(input, res, None);

    while max_bucket(&buckets) > max_bucket_sz && res != MAX_RES {
        res = next_resolution(res);
        buckets = bucket_by_res(input, res, None);
    }

    let mut out: Vec<Vec<usize>> = Vec::new();

    for (_cell, idxs) in buckets {
        if idxs.len() <= max_bucket_sz {
            out.push(idxs);
            continue;
        }

        let mut local_res = res;
        let mut frontier: Vec<Vec<usize>> = vec![idxs];

        while local_res != MAX_RES && frontier.iter().any(|b| b.len() > max_bucket_sz) {
            local_res = next_resolution(local_res);
            let mut next_frontier: Vec<Vec<usize>> = Vec::new();

            for b in frontier {
                if b.len() <= max_bucket_sz {
                    next_frontier.push(b);
                    continue;
                }
                let sub = bucket_by_res(input, local_res, Some(&b));
                for v in sub.into_values() {
                    next_frontier.push(v);
                }
            }
            frontier = next_frontier;
        }

        for mut b in frontier {
            if b.len() > max_bucket_sz {
                b.sort_unstable();
                for c in b.chunks(max_bucket_sz) {
                    out.push(c.to_vec());
                }
            } else {
                out.push(b);
            }
        }
    }

    out
}

//
// --------------------- Portal stitching + exact boundaries ---------------------
//

fn centroid_of_indices(coords: &[Coord], idxs: &[usize]) -> Coord {
    let mut sx = 0.0;
    let mut sy = 0.0;
    for &i in idxs {
        sx += coords[i].x;
        sy += coords[i].y;
    }
    let n = idxs.len().max(1) as f64;
    Coord {
        x: sx / n,
        y: sy / n,
    }
}

fn rotate_cycle(tour: &[usize], start_node: usize) -> Vec<usize> {
    let pos = tour
        .iter()
        .position(|&x| x == start_node)
        .expect("start_node not found");
    let mut out = Vec::with_capacity(tour.len());
    out.extend_from_slice(&tour[pos..]);
    out.extend_from_slice(&tour[..pos]);
    out
}

struct MergeResult {
    merged: Vec<usize>,
    /// Each index i refers to boundary edge (i -> i+1).
    boundaries: [usize; 2],
}

/// Merges two tours by finding the strictly closest points between them.
/// This prevents "random" portals from creating large outlier jumps.
fn merge_two_cycles_dense(coords: &[Coord], tour_a: &[usize], tour_b: &[usize]) -> MergeResult {
    // If bridge edges are longer than this, we penalize them extra.
    // This discourages merging chunks that are physically very far apart
    // if a slightly better local configuration exists.
    const LARGE_JUMP_PENALTY: f64 = 500.0;

    // 1. Build a KdTree for Tour B for fast querying.
    let mut tree: KdTree<f64, 2> = KdTree::new();
    for &node in tour_b {
        let c = coords[node];
        tree.add(&[c.x, c.y], node as u64);
    }

    // 2. Dense search: Find the closest pair (u \in A, v \in B)
    // We scan EVERY point in A.
    let mut candidates: Vec<(f64, usize, usize)> = Vec::with_capacity(tour_a.len());

    for &u in tour_a {
        let c = coords[u];
        // We only need the single nearest neighbor in B to find the tightest link.
        let nn = tree.nearest_one::<SquaredEuclidean>(&[c.x, c.y]);
        candidates.push((nn.distance, u, nn.item as usize));
    }

    // Sort by distance ascending: tightest kiss first.
    candidates.sort_unstable_by(|x, y| x.0.partial_cmp(&y.0).unwrap());

    // 3. Evaluate the best way to cut and splice based on the top K closest pairs.
    // We check top 40 pairs to allow for local orientation flexibility.
    let check_limit = 40.min(candidates.len());

    // Map node ID -> Index in slice for fast lookups
    let pos_a: HashMap<usize, usize> = tour_a.iter().enumerate().map(|(i, &n)| (n, i)).collect();
    let pos_b: HashMap<usize, usize> = tour_b.iter().enumerate().map(|(i, &n)| (n, i)).collect();

    let n_a = tour_a.len();
    let n_b = tour_b.len();

    let mut best: Option<(usize, usize, usize, usize, bool, f64)> = None;

    for &(_, u_node, v_node) in candidates.iter().take(check_limit) {
        let u_idx = *pos_a.get(&u_node).unwrap();
        let v_idx = *pos_b.get(&v_node).unwrap();

        // Neighbors in A
        let u_next_node = tour_a[(u_idx + 1) % n_a];
        let u_prev_node = tour_a[(u_idx + n_a - 1) % n_a];

        // Neighbors in B
        let v_next_node = tour_b[(v_idx + 1) % n_b];
        let v_prev_node = tour_b[(v_idx + n_b - 1) % n_b];

        // We can cut either edge incident to u (in A) and either edge incident to v (in B).
        // A-cuts: (u->u_next) OR (u_prev->u)
        // B-cuts: (v->v_next) OR (v_prev->v)

        let a_cuts = [(u_node, u_next_node), (u_prev_node, u_node)];
        let b_cuts = [(v_node, v_next_node), (v_prev_node, v_node)];

        for (a1, a2) in a_cuts {
            for (b1, b2) in b_cuts {
                // Cost of removing existing edges
                let removed_cost = dist(coords[a1], coords[a2]) + dist(coords[b1], coords[b2]);

                // Option 1: Forward splice (a1->b2, b1->a2)
                // Bridge edges:
                let e1 = dist(coords[a1], coords[b2]);
                let e2 = dist(coords[b1], coords[a2]);

                // Add penalty for large jumps to discourage "reaching" across gaps
                let penalty_fwd = if e1 > 1000.0 || e2 > 1000.0 {
                    LARGE_JUMP_PENALTY
                } else {
                    0.0
                };

                let score_fwd = (e1 + e2 + penalty_fwd) - removed_cost;

                // Option 2: Reverse splice (a1->b1, b2->a2) (implicitly flips B)
                let r1 = dist(coords[a1], coords[b1]);
                let r2 = dist(coords[b2], coords[a2]);

                let penalty_rev = if r1 > 1000.0 || r2 > 1000.0 {
                    LARGE_JUMP_PENALTY
                } else {
                    0.0
                };

                let score_rev = (r1 + r2 + penalty_rev) - removed_cost;

                if best.map_or(true, |x| score_fwd < x.5) {
                    best = Some((a1, a2, b1, b2, false, score_fwd));
                }
                if best.map_or(true, |x| score_rev < x.5) {
                    best = Some((a1, a2, b1, b2, true, score_rev));
                }
            }
        }
    }

    let (_a_cut_u, a_cut_v, b_cut_u, b_cut_v, flip_b, _score) =
        best.expect("tours must be non-empty and candidates found");

    // Construct the merged tour
    // A linear piece: start at a_cut_v (so it ends at a_cut_u)
    let a_lin = rotate_cycle(tour_a, a_cut_v);

    // B linear piece
    let b_lin = if !flip_b {
        // Forward: connect a_cut_u -> b_cut_v ... -> b_cut_u -> a_cut_v
        rotate_cycle(tour_b, b_cut_v)
    } else {
        // Reverse: connect a_cut_u -> b_cut_u ... -> b_cut_v -> a_cut_v
        // To do this, we reverse B and start at b_cut_u
        let mut rev = tour_b.to_vec();
        rev.reverse();
        rotate_cycle(&rev, b_cut_u)
    };

    let mut merged = Vec::with_capacity(a_lin.len() + b_lin.len());
    merged.extend_from_slice(&a_lin);
    merged.extend_from_slice(&b_lin);

    // Indices in 'merged' where the new edges exist.
    // Edge 1: merged[a_len-1] -> merged[a_len]
    // Edge 2: merged[total-1] -> merged[0]
    let boundary_mid = a_lin.len() - 1;
    let boundary_end = merged.len() - 1;

    MergeResult {
        merged,
        boundaries: [boundary_mid, boundary_end],
    }
}

fn stitch_chunk_tours_dense(
    coords: &[Coord],
    mut chunk_tours: Vec<Vec<usize>>,
) -> (Vec<usize>, Vec<usize>) {
    let mut merged = chunk_tours.remove(0);
    let mut boundaries: Vec<usize> = Vec::new();

    for t in chunk_tours {
        let res = merge_two_cycles_dense(coords, &merged, &t);
        merged = res.merged;
        boundaries.extend_from_slice(&res.boundaries);
    }

    (merged, boundaries)
}

fn boundary_two_opt(
    coords: &[Coord],
    tour: &mut Vec<usize>,
    boundaries: &[usize],
    window: usize,
    passes: usize,
) {
    let now = Instant::now();
    let n = tour.len();
    if n < 4 || boundaries.is_empty() {
        return;
    }

    // Optimization: avoid re-allocating
    // 'window' defines the range around boundary index to check.

    for _pass in 0..passes {
        let mut improved = false;

        for &b_idx in boundaries {
            // We want to check range [b_idx - window, b_idx + window]
            // wrapping around n.
            // Since implementing wrapping 2-opt iterators is complex,
            // we simply clamp indices for the "linear" scan, which captures
            // the splice point (which is exactly at b_idx).

            let start = b_idx.saturating_sub(window);
            let end = (b_idx + window).min(n - 1); // Check up to last element

            for i in start..end {
                for k in (i + 1)..end {
                    let j = k + 1; // node after k

                    // Edges are (i, i+1) and (k, k+1/j)
                    // If we reverse [i+1..=k], we replace:
                    // (i->i+1) and (k->j)
                    // with
                    // (i->k) and (i+1->j)

                    let idx_i = i;
                    let idx_i1 = i + 1;
                    let idx_k = k;
                    let idx_j = if j == n { 0 } else { j }; // Wrap for last edge check

                    let a = tour[idx_i];
                    let b = tour[idx_i1];
                    let c = tour[idx_k];
                    let d = tour[idx_j];

                    let cur_dist = dist(coords[a], coords[b]) + dist(coords[c], coords[d]);
                    let new_dist = dist(coords[a], coords[c]) + dist(coords[b], coords[d]);

                    if new_dist < cur_dist - 1e-5 {
                        // Perform reversal
                        tour[(idx_i + 1)..=idx_k].reverse();
                        improved = true;
                    }
                }
            }
        }
        if !improved {
            break;
        }
    }

    eprintln!(
        "Boundary 2-opt finished in {:.2}s",
        now.elapsed().as_secs_f32()
    );
}

//
// --------------------- Chunked entrypoint ---------------------
//

fn solve_chunk_single(
    lkh_exe: PathBuf,
    work_dir: PathBuf,
    chunk_points: &[Point],
) -> io::Result<Vec<usize>> {
    let n = chunk_points.len();
    if n < 3 {
        return Ok(chunk_points
            .into_iter()
            .enumerate()
            .map(|(idx, _)| idx)
            .collect());
    }

    let cfg = LkhConfig::new(n);
    let solver = LKHSolver::new(lkh_exe, work_dir);
    solver.create_work_dir()?;

    let pts = Plane::new(&chunk_points.to_vec()).radius(70.0).project();

    solver.create_problem_file(&pts)?;
    solver.ensure_candidate_file(pts.len())?;

    let rs = RunSpec {
        idx: 0,
        seed: cfg.base_seed,
        par_path: solver.work_dir.join("run_0.par"),
        tour_path: solver.work_dir.join("run_0.tour"),
    };
    rs.write_lkh_par(&cfg, &solver)?;

    let out = Command::new(&solver.executable)
        .arg(&rs.par_path)
        .current_dir(&solver.work_dir)
        .output()?;

    if !out.status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!(
                "LKH chunk failed.\nSTDOUT:\n{}\nSTDERR:\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr),
            ),
        ));
    }

    rs.parse_tsplib_tour(n)
}

fn order_chunks_by_centroid_tsp(
    lkh_exe: &Path,
    work_dir: &Path,
    centroids: &[Coord],
) -> io::Result<Vec<usize>> {
    if centroids.len() <= 2 {
        return Ok((0..centroids.len()).collect());
    }

    fs::create_dir_all(work_dir)?;

    let problem = work_dir.join("centroids.tsp");
    let mut s = String::new();
    s.push_str("NAME: centroids\nTYPE: TSP\n");
    s.push_str(&format!("DIMENSION: {}\n", centroids.len()));
    s.push_str("EDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n");
    for (i, p) in centroids.iter().enumerate() {
        s.push_str(&format!(
            "{} {:.0} {:.0}\n",
            i + 1,
            p.x * 1000.0,
            p.y * 1000.0
        ));
    }
    s.push_str("EOF\n");
    fs::write(&problem, s)?;

    let rs = RunSpec {
        idx: 0,
        seed: 999,
        par_path: work_dir.join("centroids.par"),
        tour_path: work_dir.join("centroids.tour"),
    };
    rs.write_lkh_par_small(&problem, 20_000, 10)?;

    let out = Command::new(lkh_exe)
        .arg(&rs.par_path)
        .current_dir(work_dir)
        .output()?;

    if !out.status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!(
                "Centroid ordering LKH failed.\nSTDOUT:\n{}\nSTDERR:\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr),
            ),
        ));
    }

    rs.parse_tsplib_tour(centroids.len())
}

pub fn solve_tsp_with_lkh_h3_chunked(
    lkh_exe: PathBuf,
    work_dir: PathBuf,
    input: &[Point],
) -> io::Result<Vec<Point>> {
    const MAX_CHUNK: usize = 5_000;

    if input.len() <= MAX_CHUNK {
        return solve_tsp_with_lkh_parallel(lkh_exe, work_dir, input);
    }

    let global_coords = Plane::new(&input.to_vec()).radius(70.0).project();

    let chunks = h3_partition_indices(input, MAX_CHUNK);
    eprintln!(
        "Chunked {} points into {} chunks (max {})",
        input.len(),
        chunks.len(),
        MAX_CHUNK
    );

    let solved_chunk_tours: Vec<Vec<usize>> = chunks
        .par_iter()
        .enumerate()
        .map(|(chunk_id, idxs)| -> io::Result<Vec<usize>> {
            let chunk_points: Vec<Point> = idxs.iter().map(|&i| input[i]).collect();
            let chunk_dir = work_dir.join(format!("chunk_{chunk_id}"));

            let now = Instant::now();
            let tour_local = solve_chunk_single(lkh_exe.clone(), chunk_dir, &chunk_points)?;
            let tour_global: Vec<usize> = tour_local.into_iter().map(|li| idxs[li]).collect();

            eprintln!(
                "chunk {chunk_id}: n={} solved in {:.2}s",
                idxs.len(),
                now.elapsed().as_secs_f32()
            );

            Ok(tour_global)
        })
        .collect::<io::Result<Vec<_>>>()?;

    let centroids: Vec<Coord> = chunks
        .iter()
        .map(|idxs| centroid_of_indices(&global_coords, idxs))
        .collect();

    let order_dir = work_dir.join("chunk_order");
    let order = order_chunks_by_centroid_tsp(&lkh_exe, &order_dir, &centroids)?;

    let mut ordered_tours: Vec<Vec<usize>> = Vec::with_capacity(solved_chunk_tours.len());
    for ci in order {
        ordered_tours.push(solved_chunk_tours[ci].clone());
    }

    // Use dense stitching instead of sparse portal stitching
    let (mut merged, boundaries) = stitch_chunk_tours_dense(&global_coords, ordered_tours);

    // Run robust boundary 2-opt
    boundary_two_opt(&global_coords, &mut merged, &boundaries, 500, 50);

    Ok(merged.into_iter().map(|i| input[i]).collect())
}
