use std::{cmp::Ordering, time::Instant};

use rayon::slice::ParallelSliceMut;

use crate::{
    outliers::{
        BroadOptions, SniperOptions, outlier_splice_repair_v6_par,
        outlier_splice_repair_v6_par_sniper,
    }, processing, utils::{self, Point, run_external_tsp_strict}
};

#[derive(Clone, Debug)]
pub struct Options {
    pub tsp_path: String,

    /// Call tsp directly at or below this size.
    pub leaf_size: usize,

    /// Safety valve: if leaf_size is higher, still split above this.
    pub max_leaf_size: usize,

    /// Portal count for optional better merge (0 disables).
    pub portals: usize,

    /// Tiny local refinement around seams
    pub seam_refine: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            tsp_path: "tsp".to_string(),
            leaf_size: 400,
            max_leaf_size: 800,
            portals: 96, // set 0 to disable portal insertion
            seam_refine: true,
        }
    }
}

pub fn solve(points: &[Point], opts: &Options) -> std::io::Result<Vec<Point>> {
    let now = Instant::now();
    if points.len() <= 2 {
        return Ok(points.to_vec());
    }
    let mut solution = solve_rec(points.to_vec(), opts)?;

    eprintln!("Finished TSP: {:.2}", now.elapsed().as_secs_f32());
    utils::measure_distance_open(&solution);

    // Final global pass (light): do a few windowed cycle 2-opt passes via rotations + long-edge cleanup.
    // A few “cycle-aware-ish” passes: rotate, do open window 2-opt, rotate back.
    processing::cycle_window_2opt(&mut solution, 256, 2);
    processing::improve_long_edges_cycle(&mut solution, 200, 512);
    processing::cycle_window_2opt(&mut solution, 128, 2);
    // random_cycle_2opt(&mut tour, opts.final_2opt_iters, opts.rng_seed);

    eprintln!("Finished First Pass: {:.2}", now.elapsed().as_secs_f32());
    utils::measure_distance_open(&solution);

    let mut broad = BroadOptions::default();
    broad.cycle_passes = 2;
    broad.hot_edges = 64;
    outlier_splice_repair_v6_par(&mut solution, &broad);
    eprintln!("Finished Second Pass: {:.2}", now.elapsed().as_secs_f32());
    utils::measure_distance_open(&solution);

    let mut sniper = SniperOptions::default();
    sniper.cycle_passes = 2;
    sniper.hot_edges = 64;
    sniper.ratio_gate = 8.0;
    outlier_splice_repair_v6_par_sniper(&mut solution, &sniper);
    eprintln!("Finished Third Pass: {:.2}", now.elapsed().as_secs_f32());
    utils::measure_distance_open(&solution);

    Ok(solution)
}

fn solve_rec(points: Vec<Point>, opts: &Options) -> std::io::Result<Vec<Point>> {
    let n = points.len();

    if n <= opts.leaf_size || n <= opts.max_leaf_size {
        return run_external_tsp_strict(&opts.tsp_path, &points);
    }
    // if n <= opts.leaf_size {
    //     return run_external_tsp_strict(&opts.tsp_path, &points);
    // }

    let (left, right) = split_long_axis(points);

    let (a_res, b_res) = rayon::join(|| solve_rec(left, opts), || solve_rec(right, opts));

    let a = a_res?;
    let b = b_res?;

    let (incl, excl) = utils::find_outliers(&a, 10.0);
    eprintln!(
        "A: Incl: {} | Excl: {}",
        incl.len(),
        excl.iter().map(|p| p.to_string()).collect::<String>()
    );

    let (incl, excl) = utils::find_outliers(&b, 10.0);
    eprintln!(
        "B: Incl: {} | Excl: {}",
        incl.len(),
        excl.iter().map(|p| p.to_string()).collect::<String>()
    );

    debug_assert!(!a.is_empty() && !b.is_empty()); 

    // Merge open tours properly.
    let mut merged = if opts.portals > 0 {
        // Higher quality merge: try a small set of insertion points (portal edges) in A.
        merge_open_paths_with_portal_insertion(&a, &b, opts.portals)
    } else {
        // Fast merge: just pick best endpoint orientation.
        best_endpoint_concat(a, b)
    };

    if opts.seam_refine {
        refine_seams_small(&mut merged, 64);
    }

    Ok(merged)
}

/* ------------------------------ Splitting ------------------------------ */

fn split_long_axis(mut pts: Vec<Point>) -> (Vec<Point>, Vec<Point>) {
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

    if lat_span >= lng_span {
        pts.par_sort_unstable_by(|a, b| a.lat.partial_cmp(&b.lat).unwrap_or(Ordering::Equal));
    } else {
        pts.par_sort_unstable_by(|a, b| a.lng.partial_cmp(&b.lng).unwrap_or(Ordering::Equal));
    }

    let mid = pts.len() / 2;
    let right = pts.split_off(mid);


    (pts, right)
}

/* ------------------------------ Open Merge ----------------------------- */

fn best_endpoint_concat(mut a: Vec<Point>, mut b: Vec<Point>) -> Vec<Point> {
    // Choose orientation among 4 combinations to minimize bridge.
    // We'll keep 'a' as primary; maybe reverse a, maybe reverse b.
    let (a0, a1) = (a[0], *a.last().unwrap());
    let (b0, b1) = (b[0], *b.last().unwrap());

    // costs for connecting end_of_first -> start_of_second
    let cost_a_b = a1.dist(&b0);
    let cost_a_rb = a1.dist(&b1);
    let cost_ra_b = a0.dist(&b0);
    let cost_ra_rb = a0.dist(&b1);

    // Pick smallest and orient accordingly.
    let mut best = cost_a_b;
    let mut mode = 0; // 0: a + b
    if cost_a_rb < best {
        best = cost_a_rb;
        mode = 1;
    } // a + rev(b)
    if cost_ra_b < best {
        best = cost_ra_b;
        mode = 2;
    } // rev(a) + b
    if cost_ra_rb < best {
        best = cost_ra_rb;
        mode = 3;
    } // rev(a) + rev(b)

    if mode == 2 || mode == 3 {
        a.reverse();
    }
    if mode == 1 || mode == 3 {
        b.reverse();
    }

    a.extend(b);
    a
}

/// Higher-quality open-path merge:
/// Insert B into A by cutting A at a "portal edge" and reconnecting through B's endpoints.
/// We try a limited set of cut positions for speed and pick the best.
fn merge_open_paths_with_portal_insertion(a: &[Point], b: &[Point], portals: usize) -> Vec<Point> {
    // Candidates are cut points i meaning split A into prefix A[0..=i] and suffix A[i+1..]
    // Then join: prefix + (best-oriented B) + suffix, with two bridges:
    //   bridge1 = a[i].dist(&b_start)
    //   bridge2 = b_end.dist(&a[i+1])
    //
    // We try also reversed B.
    let n = a.len();
    if n < 2 {
        return best_endpoint_concat(a.to_vec(), b.to_vec());
    }

    let idxs = portal_cut_indices(n, portals);

    let mut best_cost = f64::INFINITY;
    let mut best_i = 0usize;
    let mut best_rev_b = false;

    for &i in &idxs {
        if i + 1 >= n {
            continue;
        }
        let left_end = a[i];
        let right_start = a[i + 1];

        // B normal
        let b_start = b[0];
        let b_end = *b.last().unwrap();
        let c1 = left_end.dist(&b_start) + b_end.dist(&right_start);

        // B reversed
        let rb_start = b_end;
        let rb_end = b_start;
        let c2 = left_end.dist(&rb_start) + rb_end.dist(&right_start);

        if c1 < best_cost {
            best_cost = c1;
            best_i = i;
            best_rev_b = false;
        }
        if c2 < best_cost {
            best_cost = c2;
            best_i = i;
            best_rev_b = true;
        }
    }

    // Build: A[0..=best_i] + (B oriented) + A[best_i+1..]
    let mut out = Vec::with_capacity(a.len() + b.len());
    out.extend_from_slice(&a[..=best_i]);

    if !best_rev_b {
        out.extend_from_slice(b);
    } else {
        out.extend(b.iter().rev().copied());
    }

    out.extend_from_slice(&a[(best_i + 1)..]);
    out
}

fn portal_cut_indices(n: usize, portals: usize) -> Vec<usize> {
    let k = portals.min(n.saturating_sub(1)).max(4);
    let step = (n as f64) / (k as f64);
    let mut v: Vec<usize> = (0..k)
        .map(|t| ((t as f64) * step).floor() as usize)
        .filter(|&i| i + 1 < n)
        .collect();
    v.sort_unstable();
    v.dedup();
    if !v.contains(&0) {
        v.push(0);
    }
    v.sort_unstable();
    v
}

/* --------------------------- Local Refinement -------------------------- */

fn refine_seams_small(route: &mut [Point], window: usize) {
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
