use std::{
    cmp::Ordering,
    collections::HashSet,
    hash::{Hash, Hasher},
    time::Instant,
};

use rayon::slice::ParallelSliceMut;

use crate::{
    outliers::{
        BroadOptions, SniperOptions, outlier_splice_repair_v6_par,
        outlier_splice_repair_v6_par_sniper,
    },
    processing,
    utils::{self, Point},
};

#[derive(Clone, Debug)]
pub struct Options {
    pub tsp_path: String,

    /// Call tsp directly at or below this size.
    pub leaf_size: usize,

    /// Safety valve: if leaf_size is higher, still split above this.
    pub max_leaf_size: usize,

    /// Halo size near split boundaries. Keep small (8..64 typical).
    /// This is "per boundary" budget: we select up to halo points near lat median and halo near lng median.
    pub halo: usize,

    /// Merge quality knob: number of candidate cut edges per cycle.
    /// 0 will still merge, but using a small fallback set of cuts.
    pub portals: usize,

    /// Optional small seam/window refinement after each merge.
    pub seam_refine: bool,
    pub seam_window: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            tsp_path: "tsp".to_string(),
            leaf_size: 100,
            max_leaf_size: 500,
            portals: 96, // set 0 to disable portal insertion
            seam_refine: true,
            halo: 16, // consider 16..64 for higher quality boundaries
            seam_window: 64,
        }
    }
}

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

pub fn solve(points: &[Point], opts: &Options) -> std::io::Result<Vec<Point>> {
    let now = Instant::now();
    if points.len() <= 2 {
        return Ok(points.to_vec());
    }
    let mut solution = solve_rec(points.to_vec(), opts)?;

    eprintln!("Finished TSP: {:.2}", now.elapsed().as_secs_f32());
    utils::measure_distance_open(&solution);
    eprintln!();
   
    let now = Instant::now();
    processing::cycle_window_2opt(&mut solution, 256, 2);
    eprintln!("Finished First Pass: {:.2}", now.elapsed().as_secs_f32());
    utils::measure_distance_open(&solution);
    eprintln!();

    let now = Instant::now();
    processing::improve_long_edges_cycle(&mut solution, 200, 512);
    eprintln!("Finished Second Pass: {:.2}", now.elapsed().as_secs_f32());
    utils::measure_distance_open(&solution);
    eprintln!();

    let now = Instant::now();
    processing::cycle_window_2opt(&mut solution, 128, 2);
    eprintln!("Finished Third Pass: {:.2}", now.elapsed().as_secs_f32());
    utils::measure_distance_open(&solution);
    eprintln!();

    let now = Instant::now();
    outlier_splice_repair_v6_par(
        &mut solution,
        &BroadOptions {
            cycle_passes: 2,
            hot_edges: 64,
            ..Default::default()
        },
    );
    eprintln!("Finished Fourth Pass: {:.2}", now.elapsed().as_secs_f32());
    utils::measure_distance_open(&solution);
    eprintln!();

    let now = Instant::now();
    outlier_splice_repair_v6_par_sniper(
        &mut solution,
        &SniperOptions {
            cycle_passes: 6,
            hot_edges: 64,
            ratio_gate: 8.0,
            ..Default::default()
        },
    );
    eprintln!("Finished Fifth Pass: {:.2}", now.elapsed().as_secs_f32());
    utils::measure_distance_open(&solution);
    eprintln!();

    Ok(solution)
}

fn solve_rec(points: Vec<Point>, opts: &Options) -> std::io::Result<Vec<Point>> {
    let n = points.len();

    // Leaf: call external tsp.
    // Note: halos can inflate n. That's OK. The strict runner ensures tsp returns expected count.
    if n <= opts.max_leaf_size {
        return utils::run_external_tsp_strict(&opts.tsp_path, &points);
    }

    let use_quadtree = n >= 4 * opts.leaf_size;

    if use_quadtree {
        let (children, med_lat, med_lng) = split_quadtree_median(points);
        let children = add_halos_quadtree(children, med_lat, med_lng, opts.halo);

        // MOVE children into tasks (no clones).
        let [c0, c1, c2, c3] = children;

        let (r0, r1) = rayon::join(
            || {
                if c0.is_empty() {
                    Ok(Vec::new())
                } else {
                    solve_rec(c0, opts)
                }
            },
            || {
                if c1.is_empty() {
                    Ok(Vec::new())
                } else {
                    solve_rec(c1, opts)
                }
            },
        );
        let (r2, r3) = rayon::join(
            || {
                if c2.is_empty() {
                    Ok(Vec::new())
                } else {
                    solve_rec(c2, opts)
                }
            },
            || {
                if c3.is_empty() {
                    Ok(Vec::new())
                } else {
                    solve_rec(c3, opts)
                }
            },
        );

        let solved: [Vec<Point>; 4] = [r0?, r1?, r2?, r3?];

        // Order children by centroid mini-tour (size <= 4) and merge sequentially.
        let order = best_order_by_centroids(&solved);
        let mut merged: Vec<Point> = Vec::new();

        for &idx in &order {
            let child_tour = &solved[idx];
            if child_tour.is_empty() {
                continue;
            }
            merged = if merged.is_empty() {
                child_tour.clone()
            } else {
                merge_cycles(&merged, child_tour, opts.portals)
            };
            if opts.seam_refine {
                refine_cycle_seams(&mut merged, opts.seam_window);
            }
        }

        return Ok(merged);
    }

    // Binary split fallback (still supports halos).
    let (left, right, boundary, split_axis_lat) = split_long_axis(points);
    let (left, right) = add_halos_binary(left, right, boundary, split_axis_lat, opts.halo);

    let (a_res, b_res) = rayon::join(|| solve_rec(left, opts), || solve_rec(right, opts));
    let a = a_res?;
    let b = b_res?;

    let mut merged = merge_cycles(&a, &b, opts.portals);
    if opts.seam_refine {
        refine_cycle_seams(&mut merged, opts.seam_window);
    }
    Ok(merged)
}

/* ------------------------------ Splitting ------------------------------ */

// fn split_long_axis(mut pts: Vec<Point>, opts: &Options) -> (Vec<Point>, Vec<Point>) {
//     let mut min_lat = f64::INFINITY;
//     let mut max_lat = f64::NEG_INFINITY;
//     let mut min_lng = f64::INFINITY;
//     let mut max_lng = f64::NEG_INFINITY;

//     for p in &pts {
//         min_lat = min_lat.min(p.lat);
//         max_lat = max_lat.max(p.lat);
//         min_lng = min_lng.min(p.lng);
//         max_lng = max_lng.max(p.lng);
//     }

//     let lat_span = max_lat - min_lat;
//     let lng_span = max_lng - min_lng;

//     if lat_span >= lng_span {
//         pts.par_sort_unstable_by(|a, b| a.lat.partial_cmp(&b.lat).unwrap_or(Ordering::Equal));
//     } else {
//         pts.par_sort_unstable_by(|a, b| a.lng.partial_cmp(&b.lng).unwrap_or(Ordering::Equal));
//     }
//     let mid = pts.len() / 2;
//     let right = pts.split_off(mid);
//     let left = pts;

//     // for p in left.iter() {
//     //   println!("{}", p.to_string());
//     // }
//     return (left, right);

// let mut new_left = Vec::with_capacity(left.len());
// let mut new_right = Vec::with_capacity(right.len());
// let mut remaining = vec![];

// // let l_avg = utils::get_avg_dist(&left);
// // let r_avg = utils::get_avg_dist(&right);

// const FACTOR: f64 = 2.0;
// // let (left, l_excl) = utils::get_outliers(&left, FACTOR);
// // let (right, r_excl) = utils::get_outliers(&right, FACTOR);
// // remaining.extend(l_excl);
// // remaining.extend(r_excl);

// let l_centroid = utils::centroid(&left).unwrap();
// let mut total = 0.0;
// let mut worst = 0.0;
// for p in left.iter() {
//     let d_own = p.dist(&l_centroid);
//     total += d_own;
// }
// let avg = total / (left.len() as f64);
// let threshold = avg * FACTOR;
// for p in left.into_iter() {
//     let d_own = p.dist(&l_centroid);
//     if d_own > threshold {
//         remaining.push(p);
//     } else {
//         new_left.push(p);
//     }
//     if d_own > worst {
//         worst = d_own;
//     }
// }
// eprintln!("L avg: {avg} thresh: {threshold} worst: {worst}");

// let r_centroid = utils::centroid(&right).unwrap();
// let mut total = 0.0;
// let mut worst = 0.0;
// for p in right.iter() {
//     let d_own = p.dist(&r_centroid);
//     total += d_own;
// }
// let avg = total / (right.len() as f64);
// let threshold = avg * FACTOR;
// for p in right.into_iter() {
//     let d_own = p.dist(&r_centroid);
//     if d_own > avg {
//         remaining.push(p);
//     } else {
//         new_right.push(p);
//     }
//     if d_own > worst {
//         worst = d_own;
//     }
// }
// eprintln!("R avg: {avg} thresh: {threshold} worst: {worst}");

// eprintln!("Remaining: {}", remaining.len());

// (new_left, new_right)
// }

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
        let dst = bucket_index(!lat_hi, lng_hi);
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
        let dst = bucket_index(lat_hi, !lng_hi);
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
            centroid(&chunks[0])
        },
        if chunks[1].is_empty() {
            Point { lat: 0.0, lng: 0.0 }
        } else {
            centroid(&chunks[1])
        },
        if chunks[2].is_empty() {
            Point { lat: 0.0, lng: 0.0 }
        } else {
            centroid(&chunks[2])
        },
        if chunks[3].is_empty() {
            Point { lat: 0.0, lng: 0.0 }
        } else {
            centroid(&chunks[3])
        },
    ];

    let mut best = idxs.clone();
    let mut best_cost = f64::INFINITY;

    permute(&mut idxs, 0, &mut |perm| {
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
        permute(a, k + 1, f);
        a.swap(k, i);
    }
}

/* ------------------------------ Open Merge ----------------------------- */

fn merge_cycles(a: &[Point], b: &[Point], portals: usize) -> Vec<Point> {
    if a.is_empty() {
        return b.to_vec();
    }
    if b.is_empty() {
        return a.to_vec();
    }

    // Merge by best edge exchange (and try symmetric direction just in case).
    let (ab, ab_delta) = merge_cycles_edge_exchange(a, b, portals);
    let (ba, ba_delta) = merge_cycles_edge_exchange(b, a, portals);
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

    let a_edges = portal_edge_indices(na, portals);
    let b_edges = portal_edge_indices(nb, portals);

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
    out.extend(cycle_view_forward(a, (best_ai + 1) % na));

    // B view starts at b(bj+1) and ends at b(bj)
    let mut b_view: Vec<Point> = cycle_view_forward(b, (best_bj + 1) % nb);
    if best_rev_b {
        b_view.reverse(); // now starts at b(bj) and ends at b(bj+1)
    }
    out.extend_from_slice(&b_view);
    seam_reinsert_repair_cycle(&mut out, na - 1, 24, 12);
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

/* --------------------------- Local Refinement -------------------------- */

// fn refine_seams_small(route: &mut [Point], window: usize) {
//     let n = route.len();
//     if n < 6 {
//         return;
//     }
//     let w = window.min(n / 2).max(8);

//     for i in 1..(n - 3) {
//         let a = route[i - 1];
//         let b = route[i];
//         let j_max = (i + w).min(n - 2);

//         for j in (i + 1)..=j_max {
//             let c = route[j];
//             let d = route[j + 1];
//             let before = a.dist(&b) + c.dist(&d);
//             let after = a.dist(&c) + b.dist(&d);
//             if after + 1e-9 < before {
//                 route[i..=j].reverse();
//             }
//         }
//     }
// }

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
        refine_seams_small_open(route, window);
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
