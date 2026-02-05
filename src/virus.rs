//! tsp_wrapper.rs
//!
//! High-quality parallel TSP wrapper around an external single-threaded `tsp` binary.
//!
//! Key features:
//! - **Recursive region-growing (virus) splitting** for strong spatial locality.
//! - **Parallel recursion** via `rayon::join`.
//! - External `tsp` invoked per-leaf: stdin "lat,lng lat,lng ...", stdout same.
//! - Treats tours as **closed cycles** (external solver repeats the start at the end).
//! - **Proximity-based cycle merge** (candidate edges driven by spatial nearest neighbors).
//! - **Targeted kNN-guided 2-opt postprocessing** to eliminate long edges globally.
//!
//! Dependencies (Cargo.toml):
//! ```toml
//! [dependencies]
//! rayon = "1.10"
//! num_cpus = "1.16"
//! kiddo = "5.2.4"
//! ```
//!
//! Notes:
//! - Internally, tours are stored WITHOUT the repeated start point.
//! - Final output can optionally re-append the start point if you want `p0 ... pN p0`.

use kiddo::{KdTree, NearestNeighbour, SquaredEuclidean};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::io::{Read, Write};

use crate::utils::Point;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Node {
    pub id: u32,
    pub lat: f64,
    pub lng: f64,
}

impl ToString for Node {
    fn to_string(&self) -> String {
        let mut b1 = ryu::Buffer::new();
        let mut b2 = ryu::Buffer::new();
        format!("{},{}", b1.format(self.lat), b2.format(self.lng))
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub tsp_path: String,

    /// Use all CPUs by default; recursion uses rayon’s global pool.
    pub _workers: usize,

    /// Recursion leaf size: <= leaf_size -> call tsp.
    pub leaf_size: usize,

    /// Hard cap: if leaf would exceed this, keep splitting.
    pub max_leaf_size: usize,

    /// Region-growing: number of nearest neighbors to push per frontier step.
    pub grow_k: usize,

    /// Merge: how many A sample points to drive candidate B edges.
    pub merge_sample: usize,

    /// Merge: for each sampled A point, how many nearest B points to consider.
    pub merge_per_sample: usize,

    /// Postprocess: kNN candidate size for targeted 2-opt.
    pub post_k: usize,

    /// Postprocess: passes over thresholds schedule.
    pub post_passes: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            tsp_path: "tsp".to_string(),
            _workers: num_cpus::get().max(1),
            leaf_size: 1500,
            max_leaf_size: 3000,
            grow_k: 24,
            merge_sample: 2048,
            merge_per_sample: 16,
            post_k: 80,
            post_passes: 6,
        }
    }
}

/* ============================== Public API ============================== */

/// Solve TSP approximately (fast) for many points.
/// Returns a closed cycle in visitation order WITHOUT repeating the start at the end.
/// If you want `start` repeated at end, call `close_cycle(&mut out)`.
pub fn solve(points: &[Point], opts: &Options) -> std::io::Result<Vec<Point>> {
    if points.len() <= 2 {
        return Ok(points.to_vec());
    }

    // Assign stable IDs.
    let nodes: Vec<Node> = points
        .iter()
        .enumerate()
        .map(|(i, p)| Node {
            id: i as u32,
            lat: p.lat,
            lng: p.lng,
        })
        .collect();

    // Recursively solve to get an internal cycle (no repeated start).
    let mut route = solve_rec(nodes, opts)?;

    // Ensure internal route is normalized (no repeated start).
    normalize_cycle_nodes_in_place(&mut route);

    // Postprocessing: targeted kNN-guided 2-opt using thresholds schedule.
    let mut pos_of_id = vec![0usize; route.len()];
    rebuild_pos_of_id(&route, &mut pos_of_id);

    let avg = avg_edge_m_nodes(&route);
    // Schedule thresholds: start aggressive on worst tail, then tighten.
    // These multipliers work well for your "10x avg still too many" situation.
    let multipliers = match opts.post_passes {
        0 => vec![],
        1 => vec![10.0],
        2 => vec![10.0, 7.0],
        _ => vec![10.0, 7.0, 5.0],
    };

    for m in multipliers {
        let thr = m * avg;
        targeted_knn_2opt(&mut route, &mut pos_of_id, opts.post_k, 1, thr);
    }

    // Return as Points
    Ok(route
        .into_iter()
        .map(|n| Point {
            lat: n.lat,
            lng: n.lng,
        })
        .collect())
}

/// If you want the solver style `p0 ... pN p0`, call this on the output.
pub fn close_cycle(out: &mut Vec<Point>) {
    if out.len() >= 2 {
        let first = out[0];
        if *out.last().unwrap() != first {
            out.push(first);
        }
    }
}

/* ============================ Recursive Solve ============================ */

fn solve_rec(points: Vec<Node>, opts: &Options) -> std::io::Result<Vec<Node>> {
    let n = points.len();

    // Leaf.
    if n <= opts.leaf_size {
        let mut tour = run_external_tsp_nodes(&opts.tsp_path, &points)?;
        normalize_cycle_nodes_in_place(&mut tour);
        return Ok(tour);
    }

    // Hard cap safety: if <= max_leaf_size, you can call tsp directly (usually best for quality).
    if n <= opts.max_leaf_size {
        let mut tour = run_external_tsp_nodes(&opts.tsp_path, &points)?;
        normalize_cycle_nodes_in_place(&mut tour);
        return Ok(tour);
    }

    // Split by balanced region-growing (virus).
    let (left, right) = split_region_growing_balanced(&points, n / 2, opts.grow_k);

    // Recurse in parallel.
    let (a_res, b_res) = rayon::join(|| solve_rec(left, opts), || solve_rec(right, opts));

    let mut tour_a = a_res?;
    let mut tour_b = b_res?;
    normalize_cycle_nodes_in_place(&mut tour_a);
    normalize_cycle_nodes_in_place(&mut tour_b);

    // Proximity-based merge (much better than index portals for virus pools).
    let mut merged =
        merge_cycles_proximity_symmetric(&tour_a, &tour_b, opts.merge_sample, opts.merge_per_sample);

    normalize_cycle_nodes_in_place(&mut merged);
    Ok(merged)
}

/* ========================= Region-Growing Split ========================= */

#[derive(Clone, Copy, Debug)]
struct Cand {
    neg_d2: f64, // max-heap => negate to pop smallest
    idx: usize,  // local index into points slice
}
impl Eq for Cand {}
impl PartialEq for Cand {
    fn eq(&self, other: &Self) -> bool {
        self.neg_d2 == other.neg_d2 && self.idx == other.idx
    }
}
impl Ord for Cand {
    fn cmp(&self, other: &Self) -> Ordering {
        self.neg_d2
            .partial_cmp(&other.neg_d2)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.idx.cmp(&other.idx))
    }
}
impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Balanced two-seed region-growing split.
/// Operates on local indices; returns two Vec<Node>.
fn split_region_growing_balanced(
    all: &[Node],
    target_left: usize,
    grow_k: usize,
) -> (Vec<Node>, Vec<Node>) {
    let n = all.len();
    assert!(n >= 2);
    let grow_k = grow_k.max(4).min(n);

    // Local planar projection (equirectangular around mean lat).
    let mean_lat = all.iter().map(|p| p.lat).sum::<f64>() / (n as f64);
    let cos_phi = mean_lat.to_radians().cos().max(1e-12);
    let proj = |p: &Node| -> [f64; 2] { [p.lng.to_radians() * cos_phi, p.lat.to_radians()] };

    let mut coords: Vec<[f64; 2]> = Vec::with_capacity(n);
    for p in all {
        coords.push(proj(p));
    }

    // KD-tree over local indices.
    let mut tree: KdTree<f64, 2> = KdTree::with_capacity(n);
    for (i, c) in coords.iter().enumerate() {
        tree.add(c, i as u64);
    }

    // Two far-ish seeds.
    let seed0 = 0usize;
    let seed1 = farthest_from(seed0, &coords);
    let seed0b = farthest_from(seed1, &coords);

    let left_seed = seed0b;
    let right_seed = seed1;

    let mut assigned = vec![false; n];
    let mut left_idx: Vec<usize> = Vec::with_capacity(target_left);
    let mut right_idx: Vec<usize> = Vec::with_capacity(n.saturating_sub(target_left));

    let mut left_heap: BinaryHeap<Cand> = BinaryHeap::new();
    let mut right_heap: BinaryHeap<Cand> = BinaryHeap::new();

    assign_idx(left_seed, &mut assigned, &mut left_idx);
    assign_idx(right_seed, &mut assigned, &mut right_idx);

    push_neighbors_local(&tree, &coords, left_seed, &assigned, &mut left_heap, grow_k);
    push_neighbors_local(
        &tree,
        &coords,
        right_seed,
        &assigned,
        &mut right_heap,
        grow_k,
    );

    while left_idx.len() + right_idx.len() < n {
        let grow_left = left_idx.len() < target_left;

        if grow_left {
            if let Some(next) = pop_valid_local(&mut left_heap, &assigned) {
                assign_idx(next, &mut assigned, &mut left_idx);
                push_neighbors_local(&tree, &coords, next, &assigned, &mut left_heap, grow_k);
            } else if let Some(u) = find_unassigned_local(&assigned) {
                assign_idx(u, &mut assigned, &mut left_idx);
                push_neighbors_local(&tree, &coords, u, &assigned, &mut left_heap, grow_k);
            } else {
                break;
            }
        } else {
            if let Some(next) = pop_valid_local(&mut right_heap, &assigned) {
                assign_idx(next, &mut assigned, &mut right_idx);
                push_neighbors_local(&tree, &coords, next, &assigned, &mut right_heap, grow_k);
            } else if let Some(u) = find_unassigned_local(&assigned) {
                assign_idx(u, &mut assigned, &mut right_idx);
                push_neighbors_local(&tree, &coords, u, &assigned, &mut right_heap, grow_k);
            } else {
                break;
            }
        }
    }

    // Exact sizing guard.
    if left_idx.len() > target_left {
        while left_idx.len() > target_left {
            right_idx.push(left_idx.pop().unwrap());
        }
    } else if left_idx.len() < target_left {
        while left_idx.len() < target_left && !right_idx.is_empty() {
            left_idx.push(right_idx.pop().unwrap());
        }
    }

    // Materialize.
    let left: Vec<Node> = left_idx.into_iter().map(|i| all[i]).collect();
    let right: Vec<Node> = right_idx.into_iter().map(|i| all[i]).collect();

    debug_assert_eq!(left.len() + right.len(), n);
    (left, right)
}

fn farthest_from(seed: usize, coords: &[[f64; 2]]) -> usize {
    let s = coords[seed];
    let mut best = seed;
    let mut best_d2 = -1.0;
    for (i, c) in coords.iter().enumerate() {
        let dx = c[0] - s[0];
        let dy = c[1] - s[1];
        let d2 = dx * dx + dy * dy;
        if d2 > best_d2 {
            best_d2 = d2;
            best = i;
        }
    }
    best
}

fn assign_idx(idx: usize, assigned: &mut [bool], out: &mut Vec<usize>) {
    if !assigned[idx] {
        assigned[idx] = true;
        out.push(idx);
    }
}

fn pop_valid_local(heap: &mut BinaryHeap<Cand>, assigned: &[bool]) -> Option<usize> {
    while let Some(c) = heap.pop() {
        if !assigned[c.idx] {
            return Some(c.idx);
        }
    }
    None
}

fn find_unassigned_local(assigned: &[bool]) -> Option<usize> {
    assigned.iter().position(|&x| !x)
}

fn push_neighbors_local(
    tree: &KdTree<f64, 2>,
    coords: &[[f64; 2]],
    from_idx: usize,
    assigned: &[bool],
    heap: &mut BinaryHeap<Cand>,
    k: usize,
) {
    let q = coords[from_idx];
    let nn: Vec<NearestNeighbour<f64, u64>> = tree.nearest_n::<SquaredEuclidean>(&q, k);
    for r in nn {
        let idx = r.item as usize;
        if assigned[idx] {
            continue;
        }
        heap.push(Cand {
            neg_d2: -r.distance,
            idx,
        });
    }
}

/* =============================== Merging =============================== */

/// Proximity-driven cycle merge:
/// - Sample points from A (by index).
/// - For each sampled point, query nearest neighbors in B.
/// - Candidate cut edges in B are edges adjacent to those nearest indices.
/// - Candidate cut edges in A are edges adjacent to sampled indices.
/// - Evaluate best 2-edge splice, then construct merged by safe rotate + optional reverse.
/// Assumes tours have NO repeated start.
fn merge_cycles_proximity_symmetric(
    a: &[Node],
    b: &[Node],
    max_sample: usize,
    per_sample: usize,
) -> Vec<Node> {
    let na = a.len();
    let nb = b.len();
    assert!(na >= 3 && nb >= 3);

    // scale sample with size: ~min(max_sample, na) but also keep density ~ 1/32
    let s_a = ((na / 32).max(256)).min(max_sample).min(na);
    let s_b = ((nb / 32).max(256)).min(max_sample).min(nb);
    let k_b = per_sample.min(nb).max(1);
    let k_a = per_sample.min(na).max(1);

    let mean_lat = (a.iter().map(|p| p.lat).sum::<f64>() + b.iter().map(|p| p.lat).sum::<f64>())
        / ((na + nb) as f64);
    let cos_phi = mean_lat.to_radians().cos().max(1e-12);
    let proj = |p: &Node| -> [f64; 2] { [p.lng.to_radians() * cos_phi, p.lat.to_radians()] };

    // KD for A and B
    let mut tree_a: KdTree<f64, 2> = KdTree::with_capacity(na);
    let mut tree_b: KdTree<f64, 2> = KdTree::with_capacity(nb);

    for (i, p) in a.iter().enumerate() {
        tree_a.add(&proj(p), i as u64);
    }
    for (j, p) in b.iter().enumerate() {
        tree_b.add(&proj(p), j as u64);
    }

    let mut cand_a: Vec<usize> = Vec::new();
    let mut cand_b: Vec<usize> = Vec::new();

    // sample A -> nearest in B
    for t in 0..s_a {
        let i = (t * na) / s_a;
        cand_a.push((i + na - 1) % na);
        cand_a.push(i);

        let q = proj(&a[i]);
        let nn: Vec<NearestNeighbour<f64, u64>> = tree_b.nearest_n::<SquaredEuclidean>(&q, k_b);
        for r in nn {
            let j = r.item as usize;
            cand_b.push((j + nb - 1) % nb);
            cand_b.push(j);
        }
    }

    // sample B -> nearest in A  (the missing half!)
    for t in 0..s_b {
        let j = (t * nb) / s_b;
        cand_b.push((j + nb - 1) % nb);
        cand_b.push(j);

        let q = proj(&b[j]);
        let nn: Vec<NearestNeighbour<f64, u64>> = tree_a.nearest_n::<SquaredEuclidean>(&q, k_a);
        for r in nn {
            let i = r.item as usize;
            cand_a.push((i + na - 1) % na);
            cand_a.push(i);
        }
    }

    cand_a.sort_unstable();
    cand_a.dedup();
    cand_b.sort_unstable();
    cand_b.dedup();

    // Evaluate best 2-edge splice
    let mut best: Option<(usize, usize, bool, f64)> = None;
    for &i in &cand_a {
        let i1 = (i + 1) % na;
        let cut_a = haversine_m_node(&a[i], &a[i1]);

        for &j in &cand_b {
            let j1 = (j + 1) % nb;
            let cut_b = haversine_m_node(&b[j], &b[j1]);

            let delta1 = (haversine_m_node(&a[i], &b[j]) + haversine_m_node(&a[i1], &b[j1]))
                - (cut_a + cut_b);
            let delta2 = (haversine_m_node(&a[i], &b[j1]) + haversine_m_node(&a[i1], &b[j]))
                - (cut_a + cut_b);

            let cand1 = (i, j, false, delta1);
            let cand2 = (i, j, true, delta2);

            best = match best {
                None => Some(if delta2 < delta1 { cand2 } else { cand1 }),
                Some(cur) => {
                    let mut cur = cur;
                    if cand1.3 < cur.3 {
                        cur = cand1;
                    }
                    if cand2.3 < cur.3 {
                        cur = cand2;
                    }
                    Some(cur)
                }
            };
        }
    }

    let (i, j, rev_b, _) = best.expect("candidates non-empty");
    let i1 = (i + 1) % na;
    let j1 = (j + 1) % nb;

    let a_rot = rotate_cycle_nodes(a, i1);
    let mut b_rot = if !rev_b {
        rotate_cycle_nodes(b, j1)
    } else {
        let mut tmp = rotate_cycle_nodes(b, j);
        tmp.reverse();
        tmp
    };

    let mut merged = Vec::with_capacity(na + nb);
    merged.extend_from_slice(&a_rot);
    merged.append(&mut b_rot);
    debug_assert_eq!(merged.len(), na + nb);
    merged
}

fn rotate_cycle_nodes(tour: &[Node], start: usize) -> Vec<Node> {
    let n = tour.len();
    let mut out = Vec::with_capacity(n);
    out.extend_from_slice(&tour[start..]);
    out.extend_from_slice(&tour[..start]);
    out
}
/* ========================== External TSP Runner ========================== */

fn run_external_tsp_nodes(tsp_path: &str, points: &[Node]) -> std::io::Result<Vec<Node>> {
    // Map coordinates -> IDs (for sanity and to preserve identity).
    // Uses rounding to avoid float-string mismatches.
    let buckets = make_buckets(points);

    // Build input: "lat,lng lat,lng ..."
    let mut input = String::with_capacity(points.len() * 24);
    for (i, p) in points.iter().enumerate() {
        if i > 0 {
            input.push(' ');
        }
        input.push_str(&p.to_string());
    }

    let mut child = std::process::Command::new(tsp_path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;

    {
        let mut stdin = child.stdin.take().expect("piped stdin");
        stdin.write_all(input.as_bytes())?;
    }

    let mut out_s = String::new();
    let mut err_s = String::new();
    if let Some(mut stdout) = child.stdout.take() {
        stdout.read_to_string(&mut out_s)?;
    }
    if let Some(mut stderr) = child.stderr.take() {
        stderr.read_to_string(&mut err_s)?;
    }

    let status = child.wait()?;
    if !status.success() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("tsp failed: status={} stderr={}", status, err_s),
        ));
    }

    let pts = parse_points_from_stdout(&out_s)?;
    // Map back to IDs.
    let mapped = map_points_to_nodes(&pts, &buckets).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("tsp output mapping failed: {e}. stderr={err_s}"),
        )
    })?;

    // The solver repeats the start at the end; normalize later, but keep mapping correct.
    // If it doesn't repeat, normalize won't pop.
    if mapped.len() < 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "tsp returned too few points",
        ));
    }

    Ok(mapped)
}

fn parse_points_from_stdout(s: &str) -> std::io::Result<Vec<Point>> {
    let mut out = Vec::new();
    for tok in s.split_whitespace() {
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
    if out.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "tsp returned no points",
        ));
    }
    out.pop();

    Ok(out)
}

/* ====================== Identity Buckets / Mapping ====================== */

fn key_point(lat: f64, lng: f64) -> (i64, i64) {
    // 1e-7 deg is a good compromise; tweak if needed.
    let scale = 1e7;
    ((lat * scale).round() as i64, (lng * scale).round() as i64)
}

fn make_buckets(points: &[Node]) -> HashMap<(i64, i64), Vec<u32>> {
    let mut m: HashMap<(i64, i64), Vec<u32>> = HashMap::new();
    for p in points {
        m.entry(key_point(p.lat, p.lng)).or_default().push(p.id);
    }
    m
}

fn map_points_to_nodes(
    pts: &[Point],
    buckets: &HashMap<(i64, i64), Vec<u32>>,
) -> Result<Vec<Node>, String> {
    // We need a mutable copy of buckets to pop IDs in order.
    let mut work: HashMap<(i64, i64), Vec<u32>> = buckets.clone();
    // We'll pop from the end for efficiency; preserve a stable order by reversing once.
    for v in work.values_mut() {
        v.reverse();
    }

    let mut out = Vec::with_capacity(pts.len());
    for p in pts {
        let k = key_point(p.lat, p.lng);
        let ids = work
            .get_mut(&k)
            .ok_or_else(|| format!("point not found in buckets: {:?}", k))?;
        let id = ids
            .pop()
            .ok_or_else(|| format!("too many occurrences of point: {:?}", k))?;
        out.push(Node {
            id,
            lat: p.lat,
            lng: p.lng,
        });
    }
    Ok(out)
}

/* ============================== Normalization ============================== */

fn normalize_cycle_nodes_in_place(tour: &mut Vec<Node>) {
    // If solver repeats start at end, remove it.
    if tour.len() >= 2 && tour[0].id == tour[tour.len() - 1].id {
        tour.pop();
    }
    // If still has duplicates, that's a deeper issue; ignore here.
}

/* ============================ Postprocessing ============================ */

fn rebuild_pos_of_id(route: &[Node], pos_of_id: &mut [usize]) {
    for (i, p) in route.iter().enumerate() {
        pos_of_id[p.id as usize] = i;
    }
}

fn avg_edge_m_nodes(route: &[Node]) -> f64 {
    let n = route.len();
    if n < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..n {
        sum += haversine_m_node(&route[i], &route[(i + 1) % n]);
    }
    sum / (n as f64)
}

/// Targeted kNN-guided 2-opt:
/// - Find edges longer than `threshold_m`.
/// - For each such edge (worst-first), look up k nearest points to its endpoint in space.
/// - Try 2-opt swaps against edges adjacent to those nearby points.
/// This dramatically reduces "too many long edges" without O(n^2).
fn targeted_knn_2opt(
    route: &mut [Node],
    pos_of_id: &mut [usize],
    k: usize,
    passes: usize,
    threshold_m: f64,
) {
    let n = route.len();
    if n < 10 {
        return;
    }

    // Build KD-tree over *IDs* (stable) and store coords_by_id.
    let mean_lat = route.iter().map(|p| p.lat).sum::<f64>() / (n as f64);
    let cos_phi = mean_lat.to_radians().cos().max(1e-12);
    let proj = |p: &Node| -> [f64; 2] { [p.lng.to_radians() * cos_phi, p.lat.to_radians()] };

    let mut tree: KdTree<f64, 2> = KdTree::with_capacity(n);
    let mut coords_by_id: Vec<[f64; 2]> = vec![[0.0; 2]; n];
    for p in route.iter() {
        let c = proj(p);
        coords_by_id[p.id as usize] = c;
        tree.add(&c, p.id as u64);
    }

    for _ in 0..passes {
        // Collect long edges
        let mut long_edges: Vec<(f64, usize)> = (0..n)
            .map(|i| (haversine_m_node(&route[i], &route[(i + 1) % n]), i))
            .filter(|(d, _)| *d > threshold_m)
            .collect();

        if long_edges.is_empty() {
            break;
        }
        long_edges.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        let mut improved_any = false;

        for &(_len, i) in &long_edges {
            let i2 = (i + 1) % n;
            let a = route[i];
            let b = route[i2];

            let q = coords_by_id[a.id as usize];
            let nn: Vec<NearestNeighbour<f64, u64>> =
                tree.nearest_n::<SquaredEuclidean>(&q, k.min(n));

            let mut improved = false;

            'cand: for cand in nn {
                let cid = cand.item as usize;
                let j = pos_of_id[cid];

                // Try both edges adjacent to j
                for &e1 in &[(j + n - 1) % n, j] {
                    let e2 = (e1 + 1) % n;

                    // Skip overlaps
                    if e1 == i || e1 == i2 || e2 == i || e2 == i2 {
                        continue;
                    }

                    let c1 = route[e1];
                    let c2 = route[e2];

                    let before = haversine_m_node(&a, &b) + haversine_m_node(&c1, &c2);
                    let after = haversine_m_node(&a, &c1) + haversine_m_node(&b, &c2);

                    if after + 1e-9 < before {
                        // Apply 2-opt reversal of segment (i2 .. e1) in cyclic order.
                        reverse_cyclic_segment_nodes(route, pos_of_id, i2, e1);
                        improved = true;
                        improved_any = true;
                        break 'cand;
                    }
                }
            }

            if improved {
                // continue to next long edge
            }
        }

        if !improved_any {
            break;
        }
    }
}

fn reverse_cyclic_segment_nodes(
    route: &mut [Node],
    pos_of_id: &mut [usize],
    start: usize,
    end: usize,
) {
    let n = route.len();
    if n == 0 || start == end {
        return;
    }

    // Collect indices in segment start..=end with wrap.
    let idxs: Vec<usize> = if start < end {
        (start..=end).collect()
    } else {
        (start..n).chain(0..=end).collect()
    };

    // Reverse by swapping
    let mut l = 0usize;
    let mut r = idxs.len() - 1;
    while l < r {
        route.swap(idxs[l], idxs[r]);
        l += 1;
        r -= 1;
    }

    // Update positions for affected indices
    for &i in &idxs {
        pos_of_id[route[i].id as usize] = i;
    }
}

/* =============================== Distance =============================== */

fn haversine_m_node(a: &Node, b: &Node) -> f64 {
    let r = 6_371_000.0_f64;
    let (lat1, lat2) = (a.lat.to_radians(), b.lat.to_radians());
    let dlat = (b.lat - a.lat).to_radians();
    let dlng = (b.lng - a.lng).to_radians();
    let s1 = (dlat / 2.0).sin();
    let s2 = (dlng / 2.0).sin();
    let h = s1 * s1 + lat1.cos() * lat2.cos() * s2 * s2;
    2.0 * r * h.sqrt().asin()
}

/* =============================== Example =============================== */
/*
fn main() -> std::io::Result<()> {
    let pts: Vec<Point> = load_points_somehow();

    let opts = SolverOptions {
        tsp_path: "tsp".into(),
        leaf_size: 700,
        max_leaf_size: 2200,
        grow_k: 24,
        merge_sample: 1024,
        merge_per_sample: 8,
        post_k: 40,
        post_passes: 3,
        ..Default::default()
    };

    let mut tour = solve(&pts, &opts)?;
    // If you want closed output:
    close_cycle(&mut tour);

    println!("tour len = {}", tour.len());
    Ok(())
}
*/
