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
    /// Trials per process. For n=5000, 10k–20k is a good start.
    max_trials: usize,
    /// LKH RUNS per process. Keep at 1 when parallelizing externally.
    runs: usize,
    /// Base seed used to generate per-run seeds.
    base_seed: u64,
    /// LKH TRACE_LEVEL (0..=3ish; 1 is nice).
    trace_level: usize,
    /// Seconds to run LKH
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

impl<'a> Drop for LKHSolver<'a> {
    fn drop(&mut self) {
        rm_file(&self.candidate_file);
        rm_file(&self.pi_file);
    }
}

/// Solve TSP by spawning multiple LKH processes in parallel with different SEEDs.
/// Returns best tour points.
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

    let plane = Plane::new(&input.to_vec()).radius(70.0);
    let points = plane.project();

    solver.create_problem_file(&points)?;
    solver.ensure_candidate_file(points.len())?;

    let parallelism = cfg.threads();

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
MAX_CANDIDATES = 32 SYMMETRIC
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
    // SUBGRADIENT = NO

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
        fs::write(&self.par_path, s)
    }

    //     PI_FILE = 0
    // SUBGRADIENT = NO

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

fn successor_map_cycle(tour: &[usize]) -> HashMap<usize, usize> {
    let mut next = HashMap::with_capacity(tour.len());
    for w in tour.windows(2) {
        next.insert(w[0], w[1]);
    }
    next.insert(*tour.last().unwrap(), tour[0]);
    next
}

struct MergeResult {
    merged: Vec<usize>,
    /// Each index i refers to boundary edge (i -> i+1).
    boundaries: [usize; 2],
}

fn merge_two_cycles_portal_exact(
    coords: &[Coord],
    tour_a: &[usize],
    tour_b: &[usize],
    portal_limit: usize,
) -> MergeResult {
    // How many nearest nodes in B to consider per portal candidate.
    const K_NEAREST_B: usize = 64;

    // Used only for the "closest-to-B" portal half.
    const PORTAL_SAMPLE: usize = 4096;

    // Penalty factor: prioritize avoiding huge bridge edges.
    // Larger => aggressively avoids monster edges (start with 50–200).
    const MAX_EDGE_PENALTY: f64 = 100.0;

    let next_a = successor_map_cycle(tour_a);
    let next_b = successor_map_cycle(tour_b);

    // Build prev maps so we can consider both incident edges for cuts.
    let mut prev_a: HashMap<usize, usize> = HashMap::with_capacity(tour_a.len());
    for &u in tour_a {
        let v = *next_a.get(&u).unwrap();
        prev_a.insert(v, u);
    }
    let mut prev_b: HashMap<usize, usize> = HashMap::with_capacity(tour_b.len());
    for &u in tour_b {
        let v = *next_b.get(&u).unwrap();
        prev_b.insert(v, u);
    }

    // KD-tree over B nodes (payload = node id)
    let mut tree: KdTree<f64, 2> = KdTree::new();
    for &node in tour_b {
        let c = coords[node];
        tree.add(&[c.x, c.y], node as u64);
    }

    // ---- Portal selection: MIXED ----
    // Half evenly spaced portals (coverage)
    let half = (portal_limit / 2).max(1);

    let step_even = (tour_a.len() / half).max(1);
    let mut portals: Vec<usize> = tour_a
        .iter()
        .step_by(step_even)
        .take(half)
        .copied()
        .collect();

    // Half "closest-ish to B" portals, but only as an additive set (not full replacement)
    let sample = PORTAL_SAMPLE.min(tour_a.len()).max(1);
    let step = (tour_a.len() / sample).max(1);

    let mut scored: Vec<(f64, usize)> = tour_a
        .iter()
        .step_by(step)
        .take(sample)
        .map(|&a| {
            let ca = coords[a];
            let nn = tree.nearest_one::<SquaredEuclidean>(&[ca.x, ca.y]);
            (nn.distance, a) // squared distance
        })
        .collect();

    scored.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    for (_, a) in scored.into_iter().take(portal_limit - portals.len()) {
        portals.push(a);
    }

    // De-dupe portals
    portals.sort_unstable();
    portals.dedup();

    // best tuple:
    // (a_cut_u, a_cut_v, b_cut_u, b_cut_v, flip_b, score)
    // each cut edge is (cut_u -> cut_v) in FORWARD direction of that tour.
    let mut best: Option<(usize, usize, usize, usize, bool, f64)> = None;

    for &a in &portals {
        let a_next = *next_a.get(&a).unwrap();
        let a_prev = *prev_a.get(&a).unwrap();

        // Consider cutting either incident edge around a:
        // (a -> a_next) OR (a_prev -> a)
        for (a_cut_u, a_cut_v) in [(a, a_next), (a_prev, a)] {
            let ca = coords[a_cut_u]; // query point can be either; a is fine too
            let knn =
                tree.nearest_n::<SquaredEuclidean>(&[ca.x, ca.y], K_NEAREST_B.min(tour_b.len()));

            for hit in knn {
                let b = hit.item as usize;
                let b_next = *next_b.get(&b).unwrap();
                let b_prev = *prev_b.get(&b).unwrap();

                // Consider cutting either incident edge around b:
                for (b_cut_u, b_cut_v) in [(b, b_next), (b_prev, b)] {
                    let removed = dist(coords[a_cut_u], coords[a_cut_v])
                        + dist(coords[b_cut_u], coords[b_cut_v]);

                    // Two orientations (same idea as before):
                    // forward: connect a_cut_u -> b_cut_v, b_cut_u -> a_cut_v
                    let e1 = dist(coords[a_cut_u], coords[b_cut_v]);
                    let e2 = dist(coords[b_cut_u], coords[a_cut_v]);
                    let score_forward = (removed + e1 + e2) + MAX_EDGE_PENALTY * e1.max(e2);

                    // reverse: connect a_cut_u -> b_cut_u, b_cut_v -> a_cut_v
                    let r1 = dist(coords[a_cut_u], coords[b_cut_u]);
                    let r2 = dist(coords[b_cut_v], coords[a_cut_v]);
                    let score_reverse = (removed + r1 + r2) + MAX_EDGE_PENALTY * r1.max(r2);

                    if best.map(|x| score_forward < x.5).unwrap_or(true) {
                        best = Some((a_cut_u, a_cut_v, b_cut_u, b_cut_v, false, score_forward));
                    }
                    if best.map(|x| score_reverse < x.5).unwrap_or(true) {
                        best = Some((a_cut_u, a_cut_v, b_cut_u, b_cut_v, true, score_reverse));
                    }
                }
            }
        }
    }

    let (a_cut_u, a_cut_v, b_cut_u, b_cut_v, flip_b, _score) =
        best.expect("tours must be non-empty");

    // Build A linear piece: start at a_cut_v so that it ends at a_cut_u
    let a_lin = rotate_cycle(tour_a, a_cut_v);

    // Build B linear piece depending on orientation and cut edge:
    // If forward: B starts at b_cut_v and ends at b_cut_u
    // If reverse: reverse tour and start at b_cut_u (ends at b_cut_v)
    let b_lin = if !flip_b {
        rotate_cycle(tour_b, b_cut_v)
    } else {
        let mut rev = tour_b.to_vec();
        rev.reverse();
        rotate_cycle(&rev, b_cut_u)
    };

    let a_len = a_lin.len();
    let b_len = b_lin.len();

    let mut merged = Vec::with_capacity(a_len + b_len);
    merged.extend_from_slice(&a_lin);
    merged.extend_from_slice(&b_lin);

    // Exact boundary indices:
    let boundary1 = a_len - 1;
    let boundary2 = a_len + b_len - 1;

    MergeResult {
        merged,
        boundaries: [boundary1, boundary2],
    }
}

fn stitch_chunk_tours_portal_exact(
    coords: &[Coord],
    mut chunk_tours: Vec<Vec<usize>>,
    portal_limit: usize,
) -> (Vec<usize>, Vec<usize>) {
    let mut merged = chunk_tours.remove(0);
    let mut boundaries: Vec<usize> = Vec::new();

    for t in chunk_tours {
        let res = merge_two_cycles_portal_exact(coords, &merged, &t, portal_limit);
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

    let delta = |t: &[usize], i: usize, k: usize| -> f64 {
        let a = t[i];
        let b = t[(i + 1) % n];
        let c = t[k];
        let d = t[(k + 1) % n];
        let before = dist(coords[a], coords[b]) + dist(coords[c], coords[d]);
        let after = dist(coords[a], coords[c]) + dist(coords[b], coords[d]);
        after - before
    };

    for _ in 0..passes {
        let mut improved = false;
        for &b in boundaries {
            let start = b.saturating_sub(window);
            let end = (b + window).min(n - 2);

            for i in start..=end {
                for k in (i + 2)..=end {
                    if i == 0 && k == n - 1 {
                        continue;
                    }
                    if delta(tour, i, k) < -1e-9 {
                        tour[(i + 1)..=k].reverse();
                        improved = true;
                    }
                }
            }
        }
        if !improved {
            break;
        }
    }

    eprintln!("2opt finished in {:.2}s", now.elapsed().as_secs_f32());
}

fn boundary_three_opt(
    coords: &[Coord],
    tour: &mut Vec<usize>,
    boundaries: &[usize],
    window: usize,
    passes: usize,
) {
    let n = tour.len();
    if n < 8 || boundaries.is_empty() {
        return;
    }

    #[inline]
    fn edge_len(coords: &[Coord], t: &[usize], i: usize) -> f64 {
        let n = t.len();
        dist(coords[t[i]], coords[t[(i + 1) % n]])
    }

    // Reverse sub-slice [l..=r] (l <= r) in-place
    #[inline]
    fn rev(t: &mut [usize], l: usize, r: usize) {
        t[l..=r].reverse();
    }

    // Apply one of a handful of useful 3-opt reconnections for a,b,c edges:
    // edges are (a,a+1), (b,b+1), (c,c+1) with a < b < c (linear, no wrap)
    //
    // We consider 4 practical cases (enough to fix most ugly boundary bridges):
    //  - 2-opt variants are implicitly included, but we keep those separate in your 2-opt pass
    //  - 3-opt “double-bridge” like moves (two segment reversals, or one reversal + swap)
    //
    // Returns (best_delta, best_case_id)
    #[inline]
    fn best_3opt_case(coords: &[Coord], t: &[usize], a: usize, b: usize, c: usize) -> (f64, u8) {
        let n = t.len();
        let a1 = (a + 1) % n;
        let b1 = (b + 1) % n;
        let c1 = (c + 1) % n;

        let A = t[a];
        let B = t[a1];
        let C = t[b];
        let D = t[b1];
        let E = t[c];
        let F = t[c1];

        let before =
            dist(coords[A], coords[B]) + dist(coords[C], coords[D]) + dist(coords[E], coords[F]);

        // Case 1: reconnect A-C, B-E, D-F  (segments: [B..C] and [D..E] swapped, no reversal)
        // This is a classic 3-opt "case" that often fixes a bad bridge.
        let after1 =
            dist(coords[A], coords[C]) + dist(coords[B], coords[E]) + dist(coords[D], coords[F]);

        // Case 2: reconnect A-C, B-D, E-F  (equivalent to reversing [B..C])
        let after2 =
            dist(coords[A], coords[C]) + dist(coords[B], coords[D]) + dist(coords[E], coords[F]);

        // Case 3: reconnect A-B, C-E, D-F  (equivalent to reversing [D..E])
        let after3 =
            dist(coords[A], coords[B]) + dist(coords[C], coords[E]) + dist(coords[D], coords[F]);

        // Case 4: reconnect A-D, E-B, C-F  (one reversal + swap; very useful)
        let after4 =
            dist(coords[A], coords[D]) + dist(coords[E], coords[B]) + dist(coords[C], coords[F]);

        let mut best = (after1 - before, 1u8);
        let d2 = after2 - before;
        if d2 < best.0 {
            best = (d2, 2);
        }
        let d3 = after3 - before;
        if d3 < best.0 {
            best = (d3, 3);
        }
        let d4 = after4 - before;
        if d4 < best.0 {
            best = (d4, 4);
        }

        best
    }

    // Actually apply a chosen case. Assumes a < b < c and no wrap-around.
    fn apply_3opt_case(t: &mut Vec<usize>, a: usize, b: usize, c: usize, case_id: u8) {
        // Segments:
        // [0..=a], [a+1..=b], [b+1..=c], [c+1..]
        let (p, q, r) = (a + 1, b + 1, c + 1);

        match case_id {
            1 => {
                // Case 1: A-C, B-E, D-F
                // reorder middle as: [p..b] + [r..c] ??? easiest: do with temp
                let mut tmp = Vec::with_capacity(t.len());
                tmp.extend_from_slice(&t[..=a]);
                tmp.extend_from_slice(&t[q..=c]); // [b+1..=c]
                tmp.extend_from_slice(&t[p..=b]); // [a+1..=b]
                tmp.extend_from_slice(&t[r..]); // [c+1..]
                *t = tmp;
            }
            2 => {
                // Case 2: A-C, B-D, E-F  => reverse [p..=b]
                t[p..=b].reverse();
            }
            3 => {
                // Case 3: A-B, C-E, D-F  => reverse [q..=c]
                t[q..=c].reverse();
            }
            4 => {
                // Case 4: A-D, E-B, C-F
                // Equivalent to: swap the two middle segments and reverse one of them.
                // We'll do with temp for clarity:
                let mut tmp = Vec::with_capacity(t.len());
                tmp.extend_from_slice(&t[..=a]);
                // take [q..=c] (segment2) then reverse [p..=b] (segment1 reversed)
                tmp.extend_from_slice(&t[q..=c]); // segment2
                let mut seg1: Vec<usize> = t[p..=b].to_vec();
                seg1.reverse();
                tmp.extend_from_slice(&seg1);
                tmp.extend_from_slice(&t[r..]);
                *t = tmp;
            }
            _ => {}
        }
    }

    // Main loop
    for _ in 0..passes {
        let mut improved_any = false;

        for &b in boundaries {
            // boundary edge index b means edge (b -> b+1) is a join edge.
            // We'll search a local window around it, avoiding wrap by working in a linear slice.
            let start = b.saturating_sub(window);
            let end = (b + window).min(n - 2); // keep linear, no wrap

            // Choose a,b,c cuts as edge indices within [start..end]
            // Keep them spaced: a < b < c, with at least 1 node between.
            let mut best_move: Option<(usize, usize, usize, u8, f64)> = None;

            for a in start..=end.saturating_sub(6) {
                for b2 in (a + 2)..=end.saturating_sub(4) {
                    for c in (b2 + 2)..=end.saturating_sub(2) {
                        let (delta, case_id) = best_3opt_case(coords, tour, a, b2, c);
                        if delta < -1e-9 {
                            if best_move.map(|m| delta < m.4).unwrap_or(true) {
                                best_move = Some((a, b2, c, case_id, delta));
                            }
                        }
                    }
                }
            }

            if let Some((a, b2, c, case_id, _delta)) = best_move {
                apply_3opt_case(tour, a, b2, c, case_id);
                improved_any = true;
            }
        }

        if !improved_any {
            break;
        }
    }
}

//
// --------------------- Chunked entrypoint ---------------------
//

fn solve_chunk_single(
    lkh_exe: &Path,
    work_dir: &Path,
    chunk_points: &[Point],
) -> io::Result<Vec<usize>> {
    let n = chunk_points.len();
    if n < 3 {
        return Ok(chunk_points
            .into_iter()
            .enumerate()
            .map(|(idx, _)| idx)
            .collect());
        // return Err(io::Error::new(io::ErrorKind::InvalidInput, "chunk < 3"));
    }

    let cfg = LkhConfig::new(n);
    let solver = LKHSolver::new(lkh_exe, work_dir);
    solver.create_work_dir()?;

    let plane = Plane::new(&chunk_points.to_vec()).radius(70.0);
    let pts = plane.project();

    solver.create_problem_file(&pts)?;
    solver.ensure_candidate_file(pts.len())?;

    let rs = RunSpec {
        idx: 0,
        seed: cfg.base_seed,
        par_path: work_dir.join("run_0.par"),
        tour_path: work_dir.join("run_0.tour"),
    };
    rs.write_lkh_par(
        &solver.problem_file,
        &solver.candidate_file,
        &solver.pi_file,
        &cfg,
    )?;

    let out = Command::new(solver.executable)
        .arg(&rs.par_path)
        .current_dir(work_dir)
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

/// Split into <=10k chunks using H3, solve chunks in parallel, order chunks by centroid TSP,
/// stitch with portals (exact boundaries), then boundary-only 2-opt.
///
/// For <=10k, falls back to your existing multi-seed solver.
pub fn solve_tsp_with_lkh_h3_chunked(
    lkh_exe: &Path,
    work_dir: &Path,
    input: &[Point],
) -> io::Result<Vec<Point>> {
    const MAX_CHUNK: usize = 5_000;
    const PORTALS: usize = 512;

    if input.len() <= MAX_CHUNK {
        return solve_tsp_with_lkh_parallel(lkh_exe, work_dir, input);
    }

    // Global projection for centroid computation + stitching distances
    let global_plane = Plane::new(&input.to_vec()).radius(70.0);
    let global_coords = global_plane.project();

    let chunks = h3_partition_indices(input, MAX_CHUNK);
    eprintln!(
        "Chunked {} points into {} chunks (max {})",
        input.len(),
        chunks.len(),
        MAX_CHUNK
    );

    // Solve chunks in parallel (1 LKH run per chunk to avoid oversubscription)
    let solved_chunk_tours: Vec<Vec<usize>> = chunks
        .par_iter()
        .enumerate()
        .map(|(chunk_id, idxs)| -> io::Result<Vec<usize>> {
            let chunk_points: Vec<Point> = idxs.iter().map(|&i| input[i]).collect();
            let chunk_dir = work_dir.join(format!("chunk_{chunk_id}"));

            let now = Instant::now();
            let tour_local = solve_chunk_single(lkh_exe, &chunk_dir, &chunk_points)?;
            let tour_global: Vec<usize> = tour_local.into_iter().map(|li| idxs[li]).collect();

            eprintln!(
                "chunk {chunk_id}: n={} solved in {:.2}s",
                idxs.len(),
                now.elapsed().as_secs_f32()
            );

            Ok(tour_global)
        })
        .collect::<io::Result<Vec<_>>>()?;

    // Order chunks via centroid TSP
    let centroids: Vec<Coord> = chunks
        .iter()
        .map(|idxs| centroid_of_indices(&global_coords, idxs))
        .collect();

    let order_dir = work_dir.join("chunk_order");
    let order = order_chunks_by_centroid_tsp(lkh_exe, &order_dir, &centroids)?;

    let mut ordered_tours: Vec<Vec<usize>> = Vec::with_capacity(solved_chunk_tours.len());
    for ci in order {
        ordered_tours.push(solved_chunk_tours[ci].clone());
    }

    // Stitch + exact boundaries
    let (mut merged, boundaries) =
        stitch_chunk_tours_portal_exact(&global_coords, ordered_tours, PORTALS);

    // Boundary-only polish
    boundary_two_opt(&global_coords, &mut merged, &boundaries, 400, 4);
    // boundary_three_opt(&global_coords, &mut merged, &boundaries, 250, 2);

    Ok(merged.into_iter().map(|i| input[i]).collect())
}
