//! recursive.rs
//!
//! Recursive parallel TSP wrapper around an external `tsp` process.
//!
//! This version treats the solution as a **CLOSED TOUR** (a cycle):
//! - We represent a tour as `Vec<Point>` **without** repeating the first point at the end.
//! - The tour is interpreted cyclically: edge i -> (i+1)%n.
//!
//! External `tsp` contract (your current parser behavior):
//! - stdin : "lat,lng lat,lng lat,lng ..."
//! - stdout: same format, typically includes a repeated start; we parse into a Vec<Point> cycle
//!   (your existing parser skips the first token and drops the last).
//!
//! Parallelism:
//! - Recursion uses `rayon::join`
//! - Splitting uses `par_sort_unstable_by`
//! - Quadtree child recursion avoids clones (moves child vectors into tasks)
//!
//! Crates (Cargo.toml):
//!   rayon = "1.10"
//!   num_cpus = "1.16"
//!   rand = { version = "0.8", features = ["small_rng"] }
//!   ryu = "1.0"

use rand::{Rng, SeedableRng, rngs::SmallRng};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::time::Instant;

use crate::outliers::{
    BroadOptions, SniperOptions, outlier_splice_repair_v6_par, outlier_splice_repair_v6_par_sniper,
};
use crate::utils::{Point, measure_distance_closed};

/// Unique key for a point, based on exact IEEE-754 bits.
/// Your inputs have no exact duplicates, so this is safe and fast.
#[derive(Clone, Copy, Debug, Eq)]
struct Key(u64, u64);

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}
impl Hash for Key {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.0);
        state.write_u64(self.1);
    }
}
fn key(p: Point) -> Key {
    Key(p.lat.to_bits(), p.lng.to_bits())
}

#[derive(Clone, Debug)]
pub struct Options {
    pub tsp_path: String,

    /// Call external tsp at or below this size (not counting halos).
    pub leaf_size: usize,

    /// Hard cap: if a subproblem exceeds this, keep splitting.
    pub max_leaf_size: usize,

    /// Enable 4-way splits when subproblem is large enough.
    pub enable_quadtree_split: bool,

    /// Halo size near split boundaries. Keep small (8..64 typical).
    /// This is "per boundary" budget: we select up to halo points near lat median and halo near lng median.
    pub halo: usize,

    /// Merge quality knob: number of candidate cut edges per cycle.
    /// 0 will still merge, but using a small fallback set of cuts.
    pub portals: usize,

    /// Optional small seam/window refinement after each merge.
    pub seam_refine: bool,
    pub seam_window: usize,

    /// Final global improvement: random 2-opt iterations (open 2-opt on a few rotations).
    pub final_2opt_iters: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            tsp_path: "tsp".to_string(),
            leaf_size: 1200,
            max_leaf_size: 3000,
            enable_quadtree_split: true,
            halo: 0, // consider 16..64 for higher quality boundaries
            portals: 96,
            seam_refine: true,
            seam_window: 128,
            final_2opt_iters: 20_000,
            // rng_seed: 0xC0FFEE,
        }
    }
}

pub struct RecursiveSolver<'a> {
    opts: &'a Options,
    points: Vec<Point>,
    tour: Vec<Point>,
}

impl<'a> RecursiveSolver<'a> {
    pub fn new(opts: &'a Options, points: Vec<Point>) -> Self {
        Self {
            opts,
            points,
            tour: Vec::new(),
        }
    }

    pub fn tour(&self) -> &[Point] {
        &self.tour
    }

    /// Solve full problem: returns a CLOSED TOUR visiting each original point exactly once.
    /// Representation: Vec<Point> (no repeated start point at the end).
    ///
    /// Notes:
    /// - Because halos introduce duplicates into subproblems, we enforce the original set at the end.
    /// - Inputs have no exact duplicates => enforcing via Key-set is safe.
    pub fn solve(&mut self) -> std::io::Result<Vec<Point>> {
        let now = Instant::now();

        if self.points.len() <= 2 {
            self.tour = self.points.clone();
            return Ok(self.tour.clone());
        }

        let required: HashSet<Key> = self.points.iter().copied().map(key).collect();

        // Recurse on a *possibly duplicated* problem due to halos, then filter back to required.
        let mut tour = self.solve_rec(self.points.clone())?;

        // Enforce uniqueness and membership in original set.
        tour = Self::enforce_required_set(tour, &required);

        // if tour.len() != required.len() {
        //     return Err(std::io::Error::new(
        //         std::io::ErrorKind::InvalidData,
        //         format!(
        //             "final tour missing points: expected {}, got {}",
        //             required.len(),
        //             tour.len()
        //         ),
        //     ));
        // }

        eprintln!("Finished TSP: {:.2}", now.elapsed().as_secs_f32());
        measure_distance_closed(&tour);

        // Final global pass (light): do a few windowed cycle 2-opt passes via rotations + long-edge cleanup.
        if self.opts.final_2opt_iters > 0 {
            // A few “cycle-aware-ish” passes: rotate, do open window 2-opt, rotate back.
            Self::cycle_window_2opt(&mut tour, 256, 2);
            Self::improve_long_edges_cycle(&mut tour, 200, 512);
            Self::cycle_window_2opt(&mut tour, 128, 2);
            // random_cycle_2opt(&mut tour, opts.final_2opt_iters, opts.rng_seed);
        }

        eprintln!("Finished First Pass: {:.2}", now.elapsed().as_secs_f32());
        measure_distance_closed(&tour);

        let mut broad = BroadOptions::default();
        broad.cycle_passes = 2;
        broad.hot_edges = 64;
        outlier_splice_repair_v6_par(&mut tour, &broad);
        eprintln!("Finished Second Pass: {:.2}", now.elapsed().as_secs_f32());
        measure_distance_closed(&tour);

        let mut sniper = SniperOptions::default();
        sniper.cycle_passes = 2;
        sniper.hot_edges = 64;
        sniper.ratio_gate = 8.0;
        outlier_splice_repair_v6_par_sniper(&mut tour, &sniper);
        eprintln!("Finished Third Pass: {:.2}", now.elapsed().as_secs_f32());
        measure_distance_closed(&tour);

        // repair_cycle_by_rotations(
        //     &mut tour,
        //     2,  // passes of the wrapper
        //     32, // hot_edges per pass (try 16..64)
        //     |r| {
        //         outlier_splice_repair_v3_par(
        //             r, 1,    // internal passes per rotation (keep small!)
        //             512,  // max_outliers
        //             8000, // window
        //             10,   // seg_len_small
        //             64,   // seg_len_big
        //             24,   // big_len_for_top_k
        //             512,  // global_samples
        //             16,   // recompute_every
        //             true, // enable_endpoint_repair (endpoint doesn't exist; but fine)
        //             300,  // endpoint_band
        //             0xC0FFEE,
        //         );
        //     },
        // );

        // repair_cycle_by_rotations(&mut tour, 2, 32, |r| {
        //     outlier_splice_repair_v3_par_sniper(
        //         r,
        //         1,     // passes
        //         64,    // max_spikes_to_consider
        //         64,    // local_w
        //         25,    // sample_cap
        //         0.2,   // trim_frac
        //         8.0,   // ratio_gate
        //         20000, // window
        //         1200,  // max_local_candidates
        //         512,   // global_samples
        //         64,    // seg_len_big
        //         8,     // recompute_every
        //         0xC0FFEE ^ 0x9E3779B9,
        //     );
        // });

        // // Your existing outlier passes (currently operate on Vec order; if they assume OPEN,
        // // consider updating them to include the wrap edge n-1 -> 0 when computing edge lengths).
        // outlier_splice_repair_v3_par(
        //     &mut tour, 4, 512, 8000, 10, 64, 24, 512, 16, true, 300, 0xC0FFEE,
        // );
        // outlier_splice_repair_v3_par_sniper(
        //     &mut tour,
        //     2,     // passes
        //     64,    // max_spikes_to_consider
        //     64,    // local_w baseline
        //     25,    // sample_cap
        //     0.2,   // trim_frac
        //     8.0,   // ratio_gate
        //     20000, // window
        //     1200,  // max_local_candidates
        //     8192,  // global_samples
        //     64,    // seg_len_big
        //     8,     // recompute_every
        //     0xC0FFEE ^ 0x9E3779B9,
        // );

        self.tour = tour;
        Ok(self.tour.clone())
    }

    fn solve_rec(&self, points: Vec<Point>) -> std::io::Result<Vec<Point>> {
        let n = points.len();

        // Leaf: call external tsp.
        // Note: halos can inflate n. That's OK. The strict runner ensures tsp returns expected count.
        if n <= self.opts.max_leaf_size {
            return self.run_external_tsp_strict(&points);
        }

        let use_quadtree = self.opts.enable_quadtree_split && n >= 4 * self.opts.leaf_size;

        if use_quadtree {
            let (children, med_lat, med_lng) = Self::split_quadtree_median(points);
            let children = Self::add_halos_quadtree(children, med_lat, med_lng, self.opts.halo);

            // MOVE children into tasks (no clones).
            let [c0, c1, c2, c3] = children;

            let (r0, r1) = rayon::join(
                || {
                    if c0.is_empty() {
                        Ok(Vec::new())
                    } else {
                        self.solve_rec(c0)
                    }
                },
                || {
                    if c1.is_empty() {
                        Ok(Vec::new())
                    } else {
                        self.solve_rec(c1)
                    }
                },
            );
            let (r2, r3) = rayon::join(
                || {
                    if c2.is_empty() {
                        Ok(Vec::new())
                    } else {
                        self.solve_rec(c2)
                    }
                },
                || {
                    if c3.is_empty() {
                        Ok(Vec::new())
                    } else {
                        self.solve_rec(c3)
                    }
                },
            );

            let solved: [Vec<Point>; 4] = [r0?, r1?, r2?, r3?];

            // Order children by centroid mini-tour (size <= 4) and merge sequentially.
            let order = Self::best_order_by_centroids(&solved);
            let mut merged: Vec<Point> = Vec::new();

            for &idx in &order {
                let child_tour = &solved[idx];
                if child_tour.is_empty() {
                    continue;
                }
                merged = if merged.is_empty() {
                    child_tour.clone()
                } else {
                    Self::merge_cycles(&merged, child_tour, self.opts.portals)
                };
                if self.opts.seam_refine {
                    Self::refine_cycle_seams(&mut merged, self.opts.seam_window);
                }
            }

            return Ok(merged);
        }

        // Binary split fallback (still supports halos).
        let (left, right, boundary, split_axis_lat) = Self::split_long_axis(points);
        let (left, right) =
            Self::add_halos_binary(left, right, boundary, split_axis_lat, self.opts.halo);

        let (a_res, b_res) = rayon::join(|| self.solve_rec(left), || self.solve_rec(right));
        let a = a_res?;
        let b = b_res?;

        let mut merged = Self::merge_cycles(&a, &b, self.opts.portals);
        if self.opts.seam_refine {
            Self::refine_cycle_seams(&mut merged, self.opts.seam_window);
        }
        Ok(merged)
    }

    fn run_external_tsp_strict(&self, points: &[Point]) -> std::io::Result<Vec<Point>> {
        // Use Ryu for shortest-roundtrip formatting => stable parsing back to same bits.
        let mut input = String::with_capacity(points.len() * 32);
        for (i, p) in points.iter().enumerate() {
            if i > 0 {
                input.push(' ');
            }
            input.push_str(&Self::format_point_ryu(*p));
        }

        let mut child = Command::new(&self.opts.tsp_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        {
            let mut stdin = child.stdin.take().expect("piped stdin");
            stdin.write_all(input.as_bytes())?;
        }

        let mut stdout_s = String::new();
        let mut stderr_s = String::new();
        if let Some(mut stdout) = child.stdout.take() {
            stdout.read_to_string(&mut stdout_s)?;
        }
        if let Some(mut stderr) = child.stderr.take() {
            stderr.read_to_string(&mut stderr_s)?;
        }

        let status = child.wait()?;
        if !status.success() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("tsp failed: status={status}, stderr={stderr_s}"),
            ));
        }

        // if !stderr_s.is_empty() {
        //     eprintln!("tsp err: {stderr_s}");
        // }

        let out = Self::parse_points_from_stdout(&stdout_s)?;
        if out.len() != points.len() {
            let first = out.first().unwrap();
            let last = out.last().unwrap();
            if first.lat == last.lat && first.lng == last.lng {
                eprintln!("First and last match");
            } else {
                eprintln!("First and last do not match")
            }

            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "tsp returned wrong number of points: in={}, out={}, stderr={}",
                    points.len(),
                    out.len(),
                    stderr_s
                ),
            ));
        }
        Ok(Self::exclude_outliers(out))
    }
}

/// Solve full problem: returns a CLOSED TOUR visiting each original point exactly once.
/// Representation: Vec<Point> (no repeated start point at the end).
///
/// Notes:
/// - Because halos introduce duplicates into subproblems, we enforce the original set at the end.
/// - Inputs have no exact duplicates => enforcing via Key-set is safe.
pub fn solve_tsp_parallel(points: &[Point], opts: &Options) -> std::io::Result<Vec<Point>> {
    RecursiveSolver::new(opts, points.to_vec()).solve()
}

/* ================================ Recursion ================================ */

impl<'a> RecursiveSolver<'a> {

/* ================================ Splitting ================================ */

/// Quadtree by median lat and median lng.
/// Returns: children[4], med_lat, med_lng
fn split_quadtree_median(pts: Vec<Point>) -> ([Vec<Point>; 4], f64, f64) {
    let n = pts.len();
    let mut lats: Vec<f64> = pts.iter().map(|p| p.lat).collect();
    let mut lngs: Vec<f64> = pts.iter().map(|p| p.lng).collect();

    lats.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    lngs.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let med_lat = lats[n / 2];
    let med_lng = lngs[n / 2];

    let mut q0 = Vec::new();
    let mut q1 = Vec::new();
    let mut q2 = Vec::new();
    let mut q3 = Vec::new();

    for p in pts {
        let lat_hi = p.lat >= med_lat;
        let lng_hi = p.lng >= med_lng;
        match (lat_hi, lng_hi) {
            (false, false) => q0.push(p),
            (false, true) => q1.push(p),
            (true, false) => q2.push(p),
            (true, true) => q3.push(p),
        }
    }

    ([q0, q1, q2, q3], med_lat, med_lng)
}

/// Add halos near the median boundaries in a quadtree split.
fn add_halos_quadtree(
    mut children: [Vec<Point>; 4],
    med_lat: f64,
    med_lng: f64,
    halo: usize,
) -> [Vec<Point>; 4] {
    if halo == 0 {
        return children;
    }

    let mut bucket_keys: [HashSet<Key>; 4] = [
        children[0].iter().copied().map(key).collect(),
        children[1].iter().copied().map(key).collect(),
        children[2].iter().copied().map(key).collect(),
        children[3].iter().copied().map(key).collect(),
    ];

    // Collect all points once.
    let mut all: Vec<Point> = Vec::new();
    for c in &children {
        all.extend_from_slice(c);
    }

    all.par_sort_unstable_by(|a, b| {
        a.lat
            .partial_cmp(&b.lat)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.lng.partial_cmp(&b.lng).unwrap_or(Ordering::Equal))
    });
    all.dedup_by(|a, b| key(*a) == key(*b));

    // Near lat boundary:
    let mut near_lat: Vec<(f64, Point)> = all
        .iter()
        .copied()
        .map(|p| ((p.lat - med_lat).abs(), p))
        .collect();
    near_lat.par_sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    for &(_, p) in near_lat.iter().take(halo) {
        let lat_hi = p.lat >= med_lat;
        let lng_hi = p.lng >= med_lng;
        let dst = Self::bucket_index(!lat_hi, lng_hi);
        let k = key(p);
        if !bucket_keys[dst].contains(&k) {
            children[dst].push(p);
            bucket_keys[dst].insert(k);
        }
    }

    // Near lng boundary:
    let mut near_lng: Vec<(f64, Point)> = all
        .iter()
        .copied()
        .map(|p| ((p.lng - med_lng).abs(), p))
        .collect();
    near_lng.par_sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    for &(_, p) in near_lng.iter().take(halo) {
        let lat_hi = p.lat >= med_lat;
        let lng_hi = p.lng >= med_lng;
        let dst = Self::bucket_index(lat_hi, !lng_hi);
        let k = key(p);
        if !bucket_keys[dst].contains(&k) {
            children[dst].push(p);
            bucket_keys[dst].insert(k);
        }
    }

    children
}

fn bucket_index(lat_hi: bool, lng_hi: bool) -> usize {
    match (lat_hi, lng_hi) {
        (false, false) => 0,
        (false, true) => 1,
        (true, false) => 2,
        (true, true) => 3,
    }
}

/// Binary split by longest axis (lat or lng).
/// Returns (left, right, boundary_value, split_axis_is_lat).
fn split_long_axis(mut pts: Vec<Point>) -> (Vec<Point>, Vec<Point>, f64, bool) {
    let mut min_lat = f64::INFINITY;
    let mut max_lat = f64::NEG_INFINITY;
    let mut min_lng = f64::INFINITY;
    let mut max_lng = f64::NEG_INFINITY;

    for p in &pts {
        min_lat = min_lat.min(p.lat);
        max_lat = max_lat.max(p.lat);
        min_lng = min_lng.min(p.lng);
        max_lng = max_lng.max(p.lng);
    }

    let lat_span = max_lat - min_lat;
    let lng_span = max_lng - min_lng;
    let split_axis_is_lat = lat_span >= lng_span;

    if split_axis_is_lat {
        pts.par_sort_unstable_by(|a, b| a.lat.partial_cmp(&b.lat).unwrap_or(Ordering::Equal));
    } else {
        pts.par_sort_unstable_by(|a, b| a.lng.partial_cmp(&b.lng).unwrap_or(Ordering::Equal));
    }

    let mid = pts.len() / 2;
    let boundary = if split_axis_is_lat {
        pts[mid].lat
    } else {
        pts[mid].lng
    };
    let right = pts.split_off(mid);
    (pts, right, boundary, split_axis_is_lat)
}

/// Add halos near a binary boundary.
fn add_halos_binary(
    mut left: Vec<Point>,
    mut right: Vec<Point>,
    boundary: f64,
    axis_lat: bool,
    halo: usize,
) -> (Vec<Point>, Vec<Point>) {
    if halo == 0 {
        return (left, right);
    }

    let mut left_keys: HashSet<Key> = left.iter().copied().map(key).collect();
    let mut right_keys: HashSet<Key> = right.iter().copied().map(key).collect();

    let mut candidates: Vec<(f64, Point)> = left
        .iter()
        .chain(right.iter())
        .copied()
        .map(|p| {
            let d = if axis_lat {
                (p.lat - boundary).abs()
            } else {
                (p.lng - boundary).abs()
            };
            (d, p)
        })
        .collect();

    candidates.par_sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    for &(_, p) in candidates.iter().take(halo) {
        let k = key(p);
        if !left_keys.contains(&k) {
            left.push(p);
            left_keys.insert(k);
        }
        if !right_keys.contains(&k) {
            right.push(p);
            right_keys.insert(k);
        }
    }

    (left, right)
}

/* ============================= External TSP I/O ============================= */

fn exclude_outliers(points: Vec<Point>) -> Vec<Point> {
    let mut out = vec![];

    let n = points.len();
    let mut total = 0.0;
    let mut longest = 0.0;

    for i in 0..n {
        let j = (i + 1) % n;
        let d = points[i].dist(&points[j]);
        total += d;
        if d > longest {
            longest = d;
        }
    }
    let avg_edge = total / (n as f64);
    for i in 0..n {
        let j = (i + 1) % n;
        let d = points[i].dist(&points[j]);
        if d > 10.0 * avg_edge {
            // spikes += 1;
        } else {
            eprintln!("Excluding a point");
            out.push(points[i]);
        }
    }

    out
}
/// Keep your current parsing semantics (as in your uploaded file):
/// - skip first token
/// - parse the rest
/// - pop last point
///
/// This usually corresponds to a solver that prints a repeated start.
fn parse_points_from_stdout(s: &str) -> std::io::Result<Vec<Point>> {
    let mut out = Vec::new();
    for tok in s.split_whitespace() {
        // if i == 0 {
        //     continue;
        // }
        // println!("{tok}");
        let (a, b) = tok
            .split_once(',')
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad token"))?;
        let lat: f64 = a
            .parse()
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad lat"))?;
        let lng: f64 = b
            .parse()
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad lng"))?;
        out.push(Point { lat, lng });
    }
    eprintln!("Finished with TSP: {}", out.len());
    measure_distance_closed(&out);
    eprintln!();

    // drop repeated start (common)
    out.pop();

    Ok(out)
}

fn format_point_ryu(p: Point) -> String {
    let mut b1 = ryu::Buffer::new();
    let mut b2 = ryu::Buffer::new();
    format!("{},{}", b1.format(p.lat), b2.format(p.lng))
}

/* ============================== Merging (Closed Tours) ============================== */

/// Merge two CLOSED tours (cycles) into one CLOSED tour (cycle), via best edge exchange.
///
/// Idea:
/// - Pick an edge (ai -> a(i+1)) to cut in A
/// - Pick an edge (bj -> b(j+1)) to cut in B
/// - Reconnect either with B forward or B reversed
///
/// Construction (forward B):
///   A view: start at a(i+1) ... a(i)
///   B view: start at b(j+1) ... b(j)
///   Output: A_view + B_view
///
/// The cycle edges then include:
///   a(i) -> b(j+1)
///   b(j) -> a(i+1)
fn merge_cycles(a: &[Point], b: &[Point], portals: usize) -> Vec<Point> {
    if a.is_empty() {
        return b.to_vec();
    }
    if b.is_empty() {
        return a.to_vec();
    }

    // Merge by best edge exchange (and try symmetric direction just in case).
    let (ab, ab_delta) = Self::merge_cycles_edge_exchange(a, b, portals);
    let (ba, ba_delta) = Self::merge_cycles_edge_exchange(b, a, portals);
    if ba_delta < ab_delta { ba } else { ab }
}

fn merge_cycles_edge_exchange(a: &[Point], b: &[Point], portals: usize) -> (Vec<Point>, f64) {
    let na = a.len();
    let nb = b.len();
    if na < 2 {
        return (b.to_vec(), 0.0);
    }
    if nb < 2 {
        return (a.to_vec(), 0.0);
    }

    let a_edges = Self::portal_edge_indices(na, portals);
    let b_edges = Self::portal_edge_indices(nb, portals);

    let mut best_delta = f64::INFINITY;
    let mut best_ai = 0usize;
    let mut best_bj = 0usize;
    let mut best_rev_b = false;

    for &ai in &a_edges {
        let a0 = a[ai];
        let a1 = a[(ai + 1) % na];
        let old_a = a0.dist(&a1);

        for &bj in &b_edges {
            let b0 = b[bj];
            let b1 = b[(bj + 1) % nb];
            let old_b = b0.dist(&b1);

            // B forward: connect a0->b1 and b0->a1
            let new_fwd = a0.dist(&b1) + b0.dist(&a1);
            let delta_fwd = new_fwd - (old_a + old_b);

            if delta_fwd < best_delta {
                best_delta = delta_fwd;
                best_ai = ai;
                best_bj = bj;
                best_rev_b = false;
            }

            // B reversed: connect a0->b0 and b1->a1 (equivalent to reversing B view)
            let new_rev = a0.dist(&b0) + b1.dist(&a1);
            let delta_rev = new_rev - (old_a + old_b);

            if delta_rev < best_delta {
                best_delta = delta_rev;
                best_ai = ai;
                best_bj = bj;
                best_rev_b = true;
            }
        }
    }

    // Build output = A_view + B_view
    let mut out = Vec::with_capacity(na + nb);

    // A view starts at a(ai+1) and ends at a(ai)
    out.extend(Self::cycle_view_forward(a, (best_ai + 1) % na));

    // B view starts at b(bj+1) and ends at b(bj)
    let mut b_view: Vec<Point> = Self::cycle_view_forward(b, (best_bj + 1) % nb);
    if best_rev_b {
        b_view.reverse(); // now starts at b(bj) and ends at b(bj+1)
    }
    out.extend_from_slice(&b_view);
    Self::seam_reinsert_repair_cycle(&mut out, na - 1, 24, 12);
    (out, best_delta)
}

/// Total length of a CLOSED tour (includes wrap).
// pub fn cycle_length_m(route: &[Point]) -> f64 {
//     let n = route.len();
//     if n < 2 {
//         return 0.0;
//     }
//     (0..n).map(|i| route[i].dist(&route[(i + 1) % n])).sum()
// }

fn portal_edge_indices(n: usize, portals: usize) -> Vec<usize> {
    if n <= 2 {
        return vec![0];
    }
    let k = portals.min(n).max(8); // ensure some diversity
    let step = (n as f64) / (k as f64);
    let mut v: Vec<usize> = (0..k)
        .map(|t| ((t as f64) * step).floor() as usize % n)
        .collect();
    v.par_sort_unstable();
    v.dedup();
    if v.is_empty() {
        v.push(0);
    }
    v
}

/// Returns a full cycle traversal starting at `start` (length = n), forward direction.
fn cycle_view_forward(c: &[Point], start: usize) -> Vec<Point> {
    let n = c.len();
    let mut out = Vec::with_capacity(n);
    for t in 0..n {
        out.push(c[(start + t) % n]);
    }
    out
}

fn seam_reinsert_repair_cycle(
    route: &mut Vec<Point>,
    seam_a_end: usize,
    band: usize,
    max_moves: usize,
) {
    // seam_a_end is the index of the last point of A_view in the concatenated route:
    // seam edge 1: seam_a_end -> seam_a_end+1
    // seam edge 2: (n-1) -> 0
    //
    // band: how wide around each seam to consider "suspect" points (e.g. 8..64)
    // max_moves: cap number of successful reinsertions (e.g. 8..32)

    let n = route.len();
    if n < 8 {
        return;
    }
    let band = band.min(n / 4).max(4);
    let mut moved = 0usize;

    // Helper: cycle index wrap
    #[inline]
    fn idx(i: isize, n: usize) -> usize {
        let mut x = i % (n as isize);
        if x < 0 {
            x += n as isize;
        }
        x as usize
    }

    // Cheapest insertion delta for inserting p between u->v
    // #[inline]
    // fn insert_delta(u: Point, p: Point, v: Point) -> f64 {
    //     u.dist(&p) + p.dist(&v) - u.dist(&v)
    // }

    // Remove a node at k and reinsert after edge (j -> j+1) in cycle sense.
    // Returns improvement (negative is better).
    fn delta_relocate_cycle(route: &[Point], k: usize, j: usize) -> Option<f64> {
        let n = route.len();
        if n < 6 {
            return None;
        }

        // can't insert into edges adjacent to the removed node in the current cycle
        let k_prev = (k + n - 1) % n;
        let k_next = (k + 1) % n;

        // edges are (j -> j_next)
        let j_next = (j + 1) % n;

        // Disallow inserting into edges that touch k or its neighbors (would be degenerate)
        if j == k
            || j_next == k
            || j == k_prev
            || j_next == k_prev
            || j == k_next
            || j_next == k_next
        {
            return None;
        }

        let prev = route[k_prev];
        let p = route[k];
        let next = route[k_next];

        let u = route[j];
        let v = route[j_next];

        // remove p: replace prev->p->next with prev->next
        let remove_gain = prev.dist(&next) - prev.dist(&p) - p.dist(&next);

        // insert p into u->v
        let insert_cost = u.dist(&p) + p.dist(&v) - u.dist(&v);

        Some(remove_gain + insert_cost)
    }

    // Apply relocate: remove node at k, then insert it between j and j_next (after j)
    fn apply_relocate_cycle(route: &mut Vec<Point>, k: usize, j: usize) {
        let n = route.len();
        let p = route.remove(k);

        // After removal, indices >= k shift left by 1.
        let n2 = n - 1;

        // We want to insert after edge j->j+1 in the *new* array.
        // If j was after k, it shifted by -1.
        let mut j2 = j;
        if j > k {
            j2 -= 1;
        }

        // Insert after j2 (i.e., at position j2+1)
        let insert_pos = (j2 + 1).min(n2);
        route.insert(insert_pos, p);
    }

    // Build suspect indices around seam endpoints:
    // seam1 endpoints: seam_a_end and seam_a_end+1
    // seam2 endpoints: n-1 and 0
    let seam1_l = seam_a_end % n;
    let seam1_r = (seam_a_end + 1) % n;

    let seam2_l = (n - 1) % n;
    let seam2_r = 0usize;

    let mut suspects: Vec<usize> = Vec::with_capacity(8 * band);

    for d in -(band as isize)..=(band as isize) {
        suspects.push(idx(seam1_l as isize + d, n));
        suspects.push(idx(seam1_r as isize + d, n));
        suspects.push(idx(seam2_l as isize + d, n));
        suspects.push(idx(seam2_r as isize + d, n));
    }

    suspects.sort_unstable();
    suspects.dedup();

    // Candidate insertion edges: avoid a small forbidden region around seam endpoints,
    // because inserting right back into the seam is pointless.
    // We'll just consider edges outside +/- band around both seams.
    let mut candidate_edges: Vec<usize> = Vec::with_capacity(n);

    let forbid = |i: usize| -> bool {
        let near = |a: usize, b: usize| -> bool {
            let d = a.abs_diff(b);
            d <= band || d >= n.saturating_sub(band)
        };
        near(i, seam1_l) || near(i, seam1_r) || near(i, seam2_l) || near(i, seam2_r)
    };

    for j in 0..n {
        if !forbid(j) {
            candidate_edges.push(j);
        }
    }

    // Main loop: try to relocate suspect points if it improves.
    // Greedy is fine; we’re doing this at every merge so small improvements compound.
    while moved < max_moves {
        let mut best: Option<(usize, usize, f64)> = None; // (k, j, delta)

        for &k in &suspects {
            for &j in &candidate_edges {
                if let Some(d) = delta_relocate_cycle(route, k, j) {
                    if d < -1e-9 {
                        match best {
                            None => best = Some((k, j, d)),
                            Some((_, _, bd)) if d < bd => best = Some((k, j, d)),
                            _ => {}
                        }
                    }
                }
            }
        }

        let Some((k, j, _d)) = best else {
            break;
        };
        apply_relocate_cycle(route, k, j);

        moved += 1;
    }
}

/* ====================== A) Centroid mini-tour (size <= 4) ====================== */

fn centroid(points: &[Point]) -> Point {
    let mut s_lat = 0.0;
    let mut s_lng = 0.0;
    for p in points {
        s_lat += p.lat;
        s_lng += p.lng;
    }
    let n = points.len() as f64;
    Point {
        lat: s_lat / n,
        lng: s_lng / n,
    }
}

/// Returns an order of indices for non-empty chunks by brute-forcing permutations (<= 4! = 24).
fn best_order_by_centroids(chunks: &[Vec<Point>; 4]) -> Vec<usize> {
    let mut idxs: Vec<usize> = (0..4).filter(|&i| !chunks[i].is_empty()).collect();
    if idxs.len() <= 1 {
        return idxs;
    }

    let cents: [Point; 4] = [
        if chunks[0].is_empty() {
            Point { lat: 0.0, lng: 0.0 }
        } else {
            Self::centroid(&chunks[0])
        },
        if chunks[1].is_empty() {
            Point { lat: 0.0, lng: 0.0 }
        } else {
            Self::centroid(&chunks[1])
        },
        if chunks[2].is_empty() {
            Point { lat: 0.0, lng: 0.0 }
        } else {
            Self::centroid(&chunks[2])
        },
        if chunks[3].is_empty() {
            Point { lat: 0.0, lng: 0.0 }
        } else {
            Self::centroid(&chunks[3])
        },
    ];

    let mut best = idxs.clone();
    let mut best_cost = f64::INFINITY;

    Self::permute(&mut idxs, 0, &mut |perm| {
        let mut cost = 0.0;
        for w in perm.windows(2) {
            cost += cents[w[0]].dist(&cents[w[1]]);
        }
        if cost < best_cost {
            best_cost = cost;
            best = perm.to_vec();
        }
    });

    best
}

fn permute<F: FnMut(&[usize])>(a: &mut [usize], k: usize, f: &mut F) {
    if k == a.len() {
        f(a);
        return;
    }
    for i in k..a.len() {
        a.swap(k, i);
        Self::permute(a, k + 1, f);
        a.swap(k, i);
    }
}

/* ====================== B) Enforce original unique set (dedupe halos) ====================== */

fn enforce_required_set(route: Vec<Point>, required: &HashSet<Key>) -> Vec<Point> {
    let mut seen: HashSet<Key> = HashSet::with_capacity(required.len());
    let mut out = Vec::with_capacity(required.len());

    for p in route {
        let k = key(p);
        if required.contains(&k) && !seen.contains(&k) {
            seen.insert(k);
            out.push(p);
        }
    }
    out
}

/* ====================== C) Refinement passes (Cycle-ish) ====================== */

/// A small seam improvement for cycles.
/// Implementation: rotate to a few offsets, run the existing open seam refinement, rotate back.
fn refine_cycle_seams(route: &mut Vec<Point>, window: usize) {
    let n = route.len();
    if n < 8 {
        return;
    }
    let offsets = [0usize, n / 3, (2 * n) / 3];

    for &off in &offsets {
        route.rotate_left(off);
        Self::refine_seams_small_open(route, window);
        route.rotate_right(off);
    }
}

/// Your original seam refinement logic (open). Kept as a helper.
fn refine_seams_small_open(route: &mut [Point], window: usize) {
    let n = route.len();
    if n < 6 {
        return;
    }
    let w = window.min(n / 2).max(8);

    for i in 1..(n - 3) {
        let a = route[i - 1];
        let b = route[i];
        let j_max = (i + w).min(n - 2);

        for j in (i + 1)..=j_max {
            let c = route[j];
            let d = route[j + 1];
            let before = a.dist(&b) + c.dist(&d);
            let after = a.dist(&c) + b.dist(&d);
            if after + 1e-9 < before {
                route[i..=j].reverse();
            }
        }
    }
}

/// Windowed 2-opt on cycles via rotations (open 2-opt on a few breaks).
fn cycle_window_2opt(route: &mut Vec<Point>, window: usize, passes: usize) {
    let n = route.len();
    if n < 8 {
        return;
    }
    let offsets = [0usize, n / 4, n / 2, (3 * n) / 4];

    for &off in &offsets {
        route.rotate_left(off);
        Self::global_window_2opt_open(route, window, passes);
        route.rotate_right(off);
    }
}

/// Open windowed 2-opt (your original global_window_2opt adapted).
fn global_window_2opt_open(route: &mut [Point], window: usize, passes: usize) {
    let n = route.len();
    if n < 6 {
        return;
    }
    let w = window.min(n / 2).max(16);

    for _ in 0..passes {
        let mut improved = false;
        for i in 1..(n - 3) {
            let a = route[i - 1];
            let b = route[i];
            let j_max = (i + w).min(n - 2);

            for j in (i + 1)..=j_max {
                let c = route[j];
                let d = route[j + 1];
                let before = a.dist(&b) + c.dist(&d);
                let after = a.dist(&c) + b.dist(&d);
                if after + 1e-9 < before {
                    route[i..=j].reverse();
                    improved = true;
                }
            }
        }
        if !improved {
            break;
        }
    }
}

/// Long-edge cleanup for cycles, including the wrap edge (n-1 -> 0).
/// Uses rotations to handle wrap interactions safely.
fn improve_long_edges_cycle(route: &mut Vec<Point>, k: usize, window: usize) {
    let n = route.len();
    if n < 8 {
        return;
    }

    // Evaluate edges including wrap.
    let mut edges: Vec<(f64, usize)> = (0..n)
        .map(|i| (route[i].dist(&route[(i + 1) % n]), i))
        .collect();
    edges.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    edges.truncate(k.min(edges.len()));

    // For each long edge, rotate so that edge becomes (i -> i+1) in open sense, then do local open repair.
    for &(_, ei) in &edges {
        // rotate so that ei is at position 0 edge (0->1), making wrap disappear for the target edge
        route.rotate_left(ei);

        // Now the bad edge is between 0 and 1. Try local open 2-opt around it.
        // Use a couple quick passes.
        Self::global_window_2opt_open(route, window.min(n / 2), 1);

        // rotate back
        route.rotate_right(ei);
    }
}

/// Optional: random 2-opt on cycles via rotations (kept for completeness).
#[allow(dead_code)]
fn random_cycle_2opt(route: &mut Vec<Point>, iterations: usize, seed: u64) {
    let n = route.len();
    if n < 8 {
        return;
    }
    let mut rng = SmallRng::seed_from_u64(seed);

    for _ in 0..iterations {
        let off = (rng.random::<u32>() as usize) % n;
        route.rotate_left(off);
        Self::random_2opt_open(route, 32, rng.random::<u64>());
        route.rotate_right(off);
    }
}

fn random_2opt_open(route: &mut [Point], iterations: usize, seed: u64) {
    let n = route.len();
    if n < 6 {
        return;
    }
    let mut rng = SmallRng::seed_from_u64(seed);

    for _ in 0..iterations {
        let i = rng.random_range(1..n - 3);
        let j = rng.random_range(i + 1..n - 2);
        if j <= i + 1 {
            continue;
        }

        let a = route[i - 1];
        let b = route[i];
        let c = route[j];
        let d = route[j + 1];

        let before = a.dist(&b) + c.dist(&d);
        let after = a.dist(&c) + b.dist(&d);
        if after + 1e-9 < before {
            route[i..=j].reverse();
        }
    }
}

}
