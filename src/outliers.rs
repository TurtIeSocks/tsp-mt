// outliers.rs
//
// One-stop module for *all* outlier repair work.
// - Works on CLOSED tours (cycles) via targeted rotations.
// - Contains the OPEN-path move logic internally (so you can delete older outlier modules).
//
// Public API (closed-tour):
//   - BroadOptions / SniperOptions
//   - outlier_splice_repair_v6_par(route, &BroadOptions)
//   - outlier_splice_repair_v6_par_sniper(route, &SniperOptions)
//
// Notes:
// - The v6 functions operate on CLOSED tours (cycles) stored as Vec<Point> WITHOUT repeating the start.
// - The internal OPEN implementations are applied under rotations so every cycle edge can be optimized.

use rayon::prelude::*;

use crate::utils::Point;

/// Where we want the targeted cycle edge to land after rotation (OPEN view edge index).
#[derive(Clone, Copy, Debug)]
pub enum TargetEdgePos {
    // Front,
    Middle,
    // Back,
}

#[inline]
fn desired_edge_index(n: usize, pos: TargetEdgePos) -> usize {
    if n < 2 {
        return 0;
    }
    match pos {
        // TargetEdgePos::Front => 0,
        TargetEdgePos::Middle => (n / 2).saturating_sub(1).min(n - 2),
        // TargetEdgePos::Back => n - 2,
    }
}

/// Options for the CLOSED-tour wrapper around the broad pass.
#[derive(Clone, Debug)]
pub struct BroadOptions {
    pub cycle_passes: usize,
    pub hot_edges: usize,
    pub target_edge_pos: TargetEdgePos,
    pub early_exit: bool,
    pub early_exit_ratio: f64,

    // Underlying OPEN broad-pass knobs
    pub passes: usize,
    pub max_outliers: usize,
    pub window: usize,
    pub seg_len_small: usize,
    pub seg_len_big: usize,
    pub big_len_for_top_k: usize,
    pub global_samples: usize,
    pub recompute_every: usize,
    pub enable_endpoint_repair: bool,
    pub endpoint_band: usize,
    pub seed: u64,
}

impl Default for BroadOptions {
    fn default() -> Self {
        Self {
            cycle_passes: 2,
            hot_edges: 64,
            target_edge_pos: TargetEdgePos::Middle,
            early_exit: true,
            early_exit_ratio: 4.0,

            passes: 1,
            max_outliers: 512,
            window: 8000,
            seg_len_small: 10,
            seg_len_big: 64,
            big_len_for_top_k: 24,
            global_samples: 512,
            recompute_every: 16,
            enable_endpoint_repair: true,
            endpoint_band: 300,
            seed: 0xC0FFEE,
        }
    }
}

/// Options for the CLOSED-tour wrapper around the sniper pass.
#[derive(Clone, Debug)]
pub struct SniperOptions {
    pub cycle_passes: usize,
    pub hot_edges: usize,
    pub target_edge_pos: TargetEdgePos,
    pub early_exit: bool,
    pub early_exit_ratio: f64,

    // Underlying OPEN sniper knobs
    pub passes: usize,
    pub max_spikes_to_consider: usize,
    pub local_w: usize,
    pub sample_cap: usize,
    pub trim_frac: f64,
    pub ratio_gate: f64,
    pub window: usize,
    pub max_local_candidates: usize,
    pub global_samples: usize,
    pub seg_len_big: usize,
    pub recompute_every: usize,
    pub seed: u64,
}

impl Default for SniperOptions {
    fn default() -> Self {
        Self {
            cycle_passes: 2,
            hot_edges: 64,
            target_edge_pos: TargetEdgePos::Middle,
            early_exit: true,
            early_exit_ratio: 5.0,

            passes: 1,
            max_spikes_to_consider: 64,
            local_w: 64,
            sample_cap: 25,
            trim_frac: 0.2,
            ratio_gate: 8.0,
            window: 20000,
            max_local_candidates: 1200,
            global_samples: 8192,
            seg_len_big: 64,
            recompute_every: 8,
            seed: 0xC0FFEE ^ 0x9E3779B9,
        }
    }
}

/// Compute CLOSED-cycle edge lengths: edge i is route[i] -> route[(i+1)%n]
fn cycle_edge_lengths(route: &[Point]) -> Vec<f64> {
    let n = route.len();
    (0..n)
        .into_par_iter()
        .map(|i| route[i].dist(&route[(i + 1) % n]))
        .collect()
}

fn mean_cycle_edge(edge_len: &[f64]) -> f64 {
    if edge_len.is_empty() {
        return 0.0;
    }
    edge_len.iter().copied().sum::<f64>() / (edge_len.len() as f64)
}

fn top_k_long_edges(edge_len: &[f64], k: usize) -> Vec<usize> {
    let mut idxs: Vec<usize> = (0..edge_len.len()).collect();
    idxs.sort_by(|&i, &j| edge_len[j].partial_cmp(&edge_len[i]).unwrap());
    idxs.truncate(k.min(idxs.len()));
    idxs
}

#[inline]
fn rotate_left(route: &mut Vec<Point>, k: usize) {
    let n = route.len();
    if n > 0 {
        route.rotate_left(k % n);
    }
}
#[inline]
fn rotate_right(route: &mut Vec<Point>, k: usize) {
    let n = route.len();
    if n > 0 {
        route.rotate_right(k % n);
    }
}

/// Rotation so that cycle edge `edge_idx` lands at `desired_edge` in the OPEN view.
#[inline]
fn rotation_for_edge(n: usize, edge_idx: usize, desired_edge: usize) -> usize {
    if n == 0 {
        0
    } else {
        (edge_idx + n - (desired_edge % n)) % n
    }
}

/// CLOSED-tour broad outlier repair (v6).
pub fn outlier_splice_repair_v6_par(route: &mut Vec<Point>, opt: &BroadOptions) {
    let n = route.len();
    if n < 6 || opt.cycle_passes == 0 {
        return;
    }

    let desired_edge = desired_edge_index(n, opt.target_edge_pos);

    let mut seed = opt.seed ^ (n as u64).wrapping_mul(0x9E3779B97F4A7C15);

    for _ in 0..opt.cycle_passes {
        let edge_len = cycle_edge_lengths(route);

        if opt.early_exit {
            let mean = mean_cycle_edge(&edge_len).max(1e-9);
            let top = edge_len
                .iter()
                .copied()
                .fold(0.0_f64, |a, b| if b > a { b } else { a });
            if top / mean < opt.early_exit_ratio {
                break;
            }
        }

        // Prefer threshold-based selection so we don't do 64 full passes when there are only ~10 spikes.
        let mut hot = spike_edges_by_threshold(&edge_len, 7.5, opt.hot_edges.max(1));
        if hot.is_empty() {
            // Fallback: if nothing crosses threshold, just take a small top-k
            hot = top_k_long_edges(&edge_len, opt.hot_edges.min(12).max(1));
        }

        for (t, &edge_idx) in hot.iter().enumerate() {
            let shift = rotation_for_edge(n, edge_idx, desired_edge);
            rotate_left(route, shift);

            open_broad::outlier_splice_repair_open_broad(
                route,
                opt.passes,
                opt.max_outliers,
                opt.window,
                opt.seg_len_small,
                opt.seg_len_big,
                opt.big_len_for_top_k,
                opt.global_samples,
                opt.recompute_every,
                opt.enable_endpoint_repair,
                opt.endpoint_band,
                seed ^ (t as u64).wrapping_mul(0xD1B54A32D192ED03),
            );

            rotate_right(route, shift);
        }

        seed = seed.wrapping_add(0x9E3779B97F4A7C15);
    }
}

/// CLOSED-tour sniper outlier repair (v6).
pub fn outlier_splice_repair_v6_par_sniper(route: &mut Vec<Point>, opt: &SniperOptions) {
    let n = route.len();
    if n < 6 || opt.cycle_passes == 0 {
        return;
    }

    let desired_edge = desired_edge_index(n, opt.target_edge_pos);

    let mut seed = opt.seed ^ (n as u64).wrapping_mul(0xBF58476D1CE4E5B9);

    for _ in 0..opt.cycle_passes {
        let edge_len = cycle_edge_lengths(route);

        if opt.early_exit {
            let mean = mean_cycle_edge(&edge_len).max(1e-9);
            let top = edge_len
                .iter()
                .copied()
                .fold(0.0_f64, |a, b| if b > a { b } else { a });
            if top / mean < opt.early_exit_ratio {
                break;
            }
        }

        // Prefer threshold-based selection so we don't do 64 full passes when there are only ~10 spikes.
        let mut hot = spike_edges_by_threshold(&edge_len, 7.5, opt.hot_edges.max(1));
        if hot.is_empty() {
            // Fallback: if nothing crosses threshold, just take a small top-k
            hot = top_k_long_edges(&edge_len, opt.hot_edges.min(12).max(1));
        }

        for (t, &edge_idx) in hot.iter().enumerate() {
            let shift = rotation_for_edge(n, edge_idx, desired_edge);
            rotate_left(route, shift);

            open_sniper::outlier_splice_repair_open_sniper(
                route,
                opt.passes,
                opt.max_spikes_to_consider,
                opt.local_w,
                opt.sample_cap,
                opt.trim_frac,
                opt.ratio_gate,
                opt.window,
                opt.max_local_candidates,
                opt.global_samples,
                opt.seg_len_big,
                opt.recompute_every,
                seed ^ (t as u64).wrapping_mul(0x94D049BB133111EB),
            );

            rotate_right(route, shift);
        }

        seed = seed.wrapping_add(0x9E3779B97F4A7C15);
    }
}

/* ===========================================================================================
OPEN repair implementations (embedded)
=========================================================================================== */

mod open_broad {
    use rayon::prelude::*;
    use std::cmp::Ordering;

    use crate::utils::Point;

    /// Drop-in parallelized v3 repair for OPEN tours.
    /// - Keeps the same overall behavior as v3 (teleport scoring + adaptive segment lengths + optional endpoint repair)
    /// - Parallelizes the expensive parts:
    ///   * edge-length computation + teleport scoring
    ///   * per-teleport candidate evaluation (node relocate, 2-opt, segments by length and candidate edges)
    ///
    /// Still applies moves SEQUENTIALLY (required for correctness), but “find best move” work is parallel.
    ///
    /// Recommended for ~47k on mixed hardware:
    /// outlier_splice_repair_v3_par(&mut route, 3, 256, 7000, 10, 64, 12, 512, 16, true, 300, 0xC0FFEE);
    pub fn outlier_splice_repair_open_broad(
        route: &mut Vec<Point>,
        passes: usize,
        max_outliers: usize,
        window: usize,
        seg_len_small: usize,
        seg_len_big: usize,
        big_len_for_top_k: usize,
        global_samples: usize,
        recompute_every: usize,
        enable_endpoint_repair: bool,
        endpoint_band: usize,
        seed: u64,
    ) {
        if route.len() < 6 || passes == 0 {
            return;
        }

        #[derive(Clone, Debug)]
        enum Move {
            RelocateNode {
                k: usize,
                j: usize,
            },
            RelocateSeg {
                k: usize,
                len: usize,
                j: usize,
                reverse: bool,
            },
            TwoOpt {
                i: usize,
                j: usize,
            }, // reverse [i..=j]
        }

        // ---------- deterministic PRNG (splitmix64) ----------
        #[inline]
        fn splitmix64_next(state: &mut u64) -> u64 {
            let mut z = state.wrapping_add(0x9E3779B97F4A7C15);
            *state = z;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }

        // ---------- helpers ----------
        // #[inline]
        // fn edge_len(route: &[Point], i: usize) -> f64 {
        //     route[i].dist(&route[i + 1])
        // }

        #[inline]
        fn window_bounds(n: usize, center_edge: usize, window: usize) -> (usize, usize) {
            let left = center_edge.saturating_sub(window);
            let right = (center_edge + 1 + window).min(n - 1);
            (left, right)
        }

        fn apply_move(route: &mut Vec<Point>, mv: &Move) {
            match *mv {
                Move::RelocateNode { k, j } => {
                    let node = route.remove(k);
                    let jj = if k <= j { j.saturating_sub(1) } else { j };
                    let insert_pos = (jj + 1).min(route.len());
                    route.insert(insert_pos, node);
                }
                Move::RelocateSeg { k, len, j, reverse } => {
                    let end = k + len;
                    let mut seg: Vec<Point> = route.drain(k..end).collect();
                    if reverse {
                        seg.reverse();
                    }
                    let jj = if k <= j { j.saturating_sub(len) } else { j };
                    let insert_pos = (jj + 1).min(route.len());
                    route.splice(insert_pos..insert_pos, seg);
                }
                Move::TwoOpt { i, j } => {
                    if i < j && j < route.len() {
                        route[i..=j].reverse();
                    }
                }
            }
        }

        // ---- delta calculations (negative => improvement) ----
        #[inline]
        fn delta_relocate_node(route: &[Point], k: usize, j: usize) -> Option<f64> {
            let n = route.len();
            if n < 4 {
                return None;
            }
            if k == 0 || k + 1 >= n || j + 1 >= n {
                return None;
            }
            if j == k || j + 1 == k || j == k - 1 {
                return None;
            }

            let prev_k = route[k - 1];
            let node = route[k];
            let next_k = route[k + 1];

            let a = route[j];
            let b = route[j + 1];

            let remove_gain = prev_k.dist(&next_k) - prev_k.dist(&node) - node.dist(&next_k);
            let insert_cost = a.dist(&node) + node.dist(&b) - a.dist(&b);

            Some(remove_gain + insert_cost)
        }

        #[inline]
        fn delta_relocate_seg(
            route: &[Point],
            k: usize,
            len: usize,
            j: usize,
            reverse: bool,
        ) -> Option<f64> {
            let n = route.len();
            if len < 2 || k == 0 || k + len >= n || j + 1 >= n {
                return None;
            }

            // disallow inserting into/near segment region
            let seg_lo = k.saturating_sub(1);
            let seg_hi = (k + len - 1).min(n - 2);
            if j >= seg_lo && j <= seg_hi {
                return None;
            }

            let prev = route[k - 1];
            let next = route[k + len];

            let s0 = route[k];
            let s1 = route[k + len - 1];
            let (seg_start, seg_end) = if !reverse { (s0, s1) } else { (s1, s0) };

            let a = route[j];
            let b = route[j + 1];

            let remove_gain = prev.dist(&next) - prev.dist(&s0) - s1.dist(&next);
            let insert_cost = a.dist(&seg_start) + seg_end.dist(&b) - a.dist(&b);

            Some(remove_gain + insert_cost)
        }

        #[inline]
        fn delta_2opt(route: &[Point], i: usize, j: usize) -> Option<f64> {
            let n = route.len();
            if i == 0 || i >= j || j + 1 >= n {
                return None;
            }
            let a = route[i - 1];
            let b = route[i];
            let c = route[j];
            let d = route[j + 1];
            let before = a.dist(&b) + c.dist(&d);
            let after = a.dist(&c) + b.dist(&d);
            Some(after - before)
        }

        // Candidate edges = local downsample (cap) + global deterministic samples
        fn build_candidates(
            n_edges: usize,
            left_edge: usize,
            right_edge: usize,
            max_local: usize,
            global_samples: usize,
            rng: &mut u64,
        ) -> Vec<usize> {
            let mut cand = Vec::new();

            let span = right_edge.saturating_sub(left_edge);
            if span <= max_local {
                cand.extend(left_edge..=right_edge);
            } else {
                let step = (span / max_local).max(1);
                let mut j = left_edge;
                while j <= right_edge {
                    cand.push(j);
                    j = j.saturating_add(step);
                    if j == 0 {
                        break;
                    }
                }
            }

            if global_samples > 0 && n_edges > 4 {
                for _ in 0..global_samples {
                    let r = (splitmix64_next(rng) as usize) % n_edges;
                    cand.push(r);
                }
            }

            cand.sort_unstable();
            cand.dedup();
            cand
        }

        // ---- teleport scoring (parallel) ----
        //
        // For speed and to avoid per-edge allocations, v3_par uses a *mean-of-samples* baseline
        // instead of median/MAD:
        // score(i) = d[i] / (mean(sampled local window) + eps)
        //
        // This still strongly prioritizes “spike edges” while being much cheaper at 47k+.
        fn teleport_edges_par(
            route: &[Point],
            max_outliers: usize,
            local_w: usize,
        ) -> Vec<(usize, f64)> {
            let n = route.len();
            let n_edges = n - 1;

            // Edge lengths in parallel
            let d: Vec<f64> = (0..n_edges)
                .into_par_iter()
                .map(|i| route[i].dist(&route[i + 1]))
                .collect();

            let eps = 1e-9;
            let sample_cap = 25usize; // small, fixed work per edge

            let mut scored: Vec<(usize, f64)> = (0..n_edges)
                .into_par_iter()
                .map(|i| {
                    let l = i.saturating_sub(local_w);
                    let r = (i + local_w).min(n_edges - 1);

                    // evenly spaced samples, capped
                    let span = r.saturating_sub(l).max(1);
                    let take = sample_cap.min(span + 1).max(3);
                    let step = (span / (take - 1)).max(1);

                    let mut sum = 0.0;
                    let mut cnt = 0usize;
                    let mut j = l;
                    while j <= r && cnt < take {
                        sum += d[j];
                        cnt += 1;
                        j = j.saturating_add(step);
                        if j == 0 {
                            break;
                        }
                    }
                    let mean = sum / (cnt.max(1) as f64);

                    let score = d[i] / (mean + eps);
                    (i, score)
                })
                .collect();

            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            scored.truncate(max_outliers.min(scored.len()));
            scored
        }

        // ---- find best move for one teleport edge (parallel candidate evaluation) ----
        fn best_move_for_teleport_par(
            route: &[Point],
            edge_i: usize,
            window: usize,
            seg_len_max: usize,
            global_samples: usize,
            rng: &mut u64,
        ) -> Option<(Move, f64)> {
            let n = route.len();
            if edge_i + 1 >= n {
                return None;
            }

            let n_edges = n - 1;
            let (left_bound, right_bound) = window_bounds(n, edge_i, window);
            let left_edge = left_bound.min(n_edges - 1);
            let right_edge = right_bound.min(n_edges - 1);

            // candidates (built sequentially; cheap relative to evaluation)
            let cands = build_candidates(n_edges, left_edge, right_edge, 2500, global_samples, rng);
            if cands.is_empty() {
                return None;
            }

            let seam_nodes = [edge_i, edge_i + 1];

            let mut best: Option<(Move, f64)> = None;

            // --- Node relocate: evaluate in parallel over candidates, for k in {i,i+1} ---
            for &k in &seam_nodes {
                if k == 0 || k + 1 >= n {
                    continue;
                }
                let local_best = cands
                    .par_iter()
                    .filter_map(|&j| delta_relocate_node(route, k, j).map(|d| (j, d)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

                if let Some((j, delta)) = local_best {
                    if delta < -1e-9 {
                        let mv = Move::RelocateNode { k, j };
                        match best {
                            None => best = Some((mv, delta)),
                            Some((_, bd)) if delta < bd => best = Some((mv, delta)),
                            _ => {}
                        }
                    }
                }
            }

            // --- Seg relocate: parallel over lengths; inside each, parallel over candidates ---
            let seg_len_max = seg_len_max.max(2);
            let best_seg = (2..=seg_len_max)
                .into_par_iter()
                .filter_map(|len| {
                    // We’ll consider starts at seam nodes and ends at seam nodes
                    let mut best_for_len: Option<(Move, f64)> = None;

                    // start at seam
                    for &k0 in &seam_nodes {
                        if k0 == 0 || k0 + len >= n {
                            continue;
                        }

                        // forward
                        let bf = cands
                            .par_iter()
                            .filter_map(|&j| {
                                delta_relocate_seg(route, k0, len, j, false).map(|d| (j, d))
                            })
                            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                        if let Some((j, d)) = bf {
                            if d < -1e-9 {
                                let mv = Move::RelocateSeg {
                                    k: k0,
                                    len,
                                    j,
                                    reverse: false,
                                };
                                match best_for_len {
                                    None => best_for_len = Some((mv, d)),
                                    Some((_, bd)) if d < bd => best_for_len = Some((mv, d)),
                                    _ => {}
                                }
                            }
                        }

                        // reverse
                        let br = cands
                            .par_iter()
                            .filter_map(|&j| {
                                delta_relocate_seg(route, k0, len, j, true).map(|d| (j, d))
                            })
                            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                        if let Some((j, d)) = br {
                            if d < -1e-9 {
                                let mv = Move::RelocateSeg {
                                    k: k0,
                                    len,
                                    j,
                                    reverse: true,
                                };
                                match best_for_len {
                                    None => best_for_len = Some((mv, d)),
                                    Some((_, bd)) if d < bd => best_for_len = Some((mv, d)),
                                    _ => {}
                                }
                            }
                        }
                    }

                    // end at seam (compute start = end+1-len)
                    for &end_idx in &seam_nodes {
                        if end_idx + 1 < len {
                            continue;
                        }
                        let k = end_idx + 1 - len;
                        if k == 0 || k + len >= n {
                            continue;
                        }

                        let bf = cands
                            .par_iter()
                            .filter_map(|&j| {
                                delta_relocate_seg(route, k, len, j, false).map(|d| (j, d))
                            })
                            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                        if let Some((j, d)) = bf {
                            if d < -1e-9 {
                                let mv = Move::RelocateSeg {
                                    k,
                                    len,
                                    j,
                                    reverse: false,
                                };
                                match best_for_len {
                                    None => best_for_len = Some((mv, d)),
                                    Some((_, bd)) if d < bd => best_for_len = Some((mv, d)),
                                    _ => {}
                                }
                            }
                        }

                        let br = cands
                            .par_iter()
                            .filter_map(|&j| {
                                delta_relocate_seg(route, k, len, j, true).map(|d| (j, d))
                            })
                            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                        if let Some((j, d)) = br {
                            if d < -1e-9 {
                                let mv = Move::RelocateSeg {
                                    k,
                                    len,
                                    j,
                                    reverse: true,
                                };
                                match best_for_len {
                                    None => best_for_len = Some((mv, d)),
                                    Some((_, bd)) if d < bd => best_for_len = Some((mv, d)),
                                    _ => {}
                                }
                            }
                        }
                    }

                    best_for_len
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            if let Some((mv, delta)) = best_seg {
                match best {
                    None => best = Some((mv, delta)),
                    Some((_, bd)) if delta < bd => best = Some((mv, delta)),
                    _ => {}
                }
            }

            // --- centered 2-opt: parallel over candidates (j) ---
            let i = edge_i;
            let rev_start = i + 1;
            if rev_start < n - 2 {
                let best_2opt = cands
                    .par_iter()
                    .filter_map(|&j| {
                        if j < i + 2 {
                            return None;
                        }
                        delta_2opt(route, rev_start, j).map(|d| (j, d))
                    })
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

                if let Some((j, delta)) = best_2opt {
                    if delta < -1e-9 {
                        let mv = Move::TwoOpt { i: rev_start, j };
                        match best {
                            None => best = Some((mv, delta)),
                            Some((_, bd)) if delta < bd => best = Some((mv, delta)),
                            _ => {}
                        }
                    }
                }
            }

            best
        }

        // ---- endpoint repair (parallel candidate evaluation) ----
        fn endpoint_repair_par(
            route: &mut Vec<Point>,
            endpoint_band: usize,
            global_samples: usize,
            rng: &mut u64,
        ) -> bool {
            let n = route.len();
            if n < 6 {
                return false;
            }
            let band = endpoint_band.max(50).min(n / 2);
            let mut did = false;

            for which in 0..2 {
                let idx = if which == 0 { 0 } else { n - 1 };
                let removed = if idx == 0 {
                    route.remove(0)
                } else {
                    route.pop().unwrap()
                };

                let n2 = route.len();
                if n2 < 2 {
                    if idx == 0 {
                        route.insert(0, removed);
                    } else {
                        route.push(removed);
                    }
                    continue;
                }

                let n_edges = n2 - 1;
                let mut cand: Vec<usize> = Vec::new();
                let first_r = band.min(n_edges.saturating_sub(1));
                cand.extend(0..=first_r);

                let last_l = n_edges.saturating_sub(band);
                cand.extend(last_l..=n_edges.saturating_sub(1));

                for _ in 0..global_samples {
                    cand.push((splitmix64_next(rng) as usize) % n_edges);
                }

                cand.sort_unstable();
                cand.dedup();

                let best = cand
                    .par_iter()
                    .filter_map(|&j| {
                        if j + 1 >= route.len() {
                            return None;
                        }
                        let a = route[j];
                        let b = route[j + 1];
                        let d = a.dist(&removed) + removed.dist(&b) - a.dist(&b);
                        Some((j, d))
                    })
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

                let ins = best.map(|(j, _)| j + 1).unwrap_or(0);
                route.insert(ins.min(route.len()), removed);
                did = true;
            }

            did
        }

        let n0 = route.len();
        let mut rng_state = seed ^ (n0 as u64).wrapping_mul(0x9E3779B97F4A7C15);

        // Local neighborhood width for teleport scoring. Smaller = focuses on sharp spikes.
        let local_w = 64usize;

        for _pass in 0..passes {
            let mut applied_moves = 0usize;

            loop {
                if enable_endpoint_repair {
                    let _ = endpoint_repair_par(
                        route,
                        endpoint_band,
                        (global_samples / 4).max(64),
                        &mut rng_state,
                    );
                }

                // 1) pick teleports (parallel)
                let tele = teleport_edges_par(route, max_outliers, local_w);

                let mut any_applied = false;

                for (rank, (edge_i, _score)) in tele.iter().enumerate() {
                    if route.len() < 6 || edge_i + 1 >= route.len() {
                        continue;
                    }

                    let seg_len = if rank < big_len_for_top_k {
                        seg_len_big.max(seg_len_small).max(2)
                    } else {
                        seg_len_small.max(2)
                    };

                    // 2) find best move (parallel evaluation), then apply sequentially
                    if let Some((mv, _delta)) = best_move_for_teleport_par(
                        route,
                        *edge_i,
                        window,
                        seg_len,
                        global_samples,
                        &mut rng_state,
                    ) {
                        apply_move(route, &mv);
                        applied_moves += 1;
                        any_applied = true;

                        if recompute_every > 0 && applied_moves % recompute_every == 0 {
                            break;
                        }
                    }
                }

                if !any_applied {
                    break;
                }
                if recompute_every > 0 && applied_moves % recompute_every == 0 {
                    continue;
                } else {
                    break;
                }
            }
        }

        debug_assert_eq!(
            route.len(),
            n0,
            "outlier_splice_repair_v3_par must not change point count"
        );
    }
}

mod open_sniper {
    use rayon::prelude::*;
    use std::cmp::Ordering;

    use crate::utils::Point;

    /// Sniper pass: only attacks **true teleports** where an edge is >= `ratio_gate` times the
    /// local baseline (trimmed-mean-of-samples). Designed to clean up the last stubborn spikes
    /// after your main v3 pass.
    ///
    /// Key ideas:
    /// - Compute edge lengths in parallel
    /// - For each edge, compute a robust local baseline via trimmed mean of sampled neighbors
    /// - Keep only edges with ratio >= ratio_gate (e.g. 5.0 or 8.0 or 10.0)
    /// - For those edges only, run aggressive move search:
    ///     * node relocate (both endpoints)
    ///     * centered 2-opt
    ///     * segment relocate up to seg_len_big (full range), both orientations
    /// - Parallelize evaluation over candidates and segment lengths
    /// - Apply moves sequentially (required)
    ///
    /// Typical usage:
    /// ```rs
    /// // After your broad pass:
    /// outlier_splice_repair_v3_par_sniper(
    ///     &mut route,
    ///     2,        // passes
    ///     64,       // max_spikes_to_consider (cap)
    ///     64,       // local_w for baseline
    ///     25,       // samples in window
    ///     0.2,      // trim_frac (20% top & bottom)
    ///     8.0,      // ratio_gate
    ///     16000,    // window (candidate index window)
    ///     1000,     // max_local_candidates
    ///     8192,     // global_samples (aggressive)
    ///     64,       // seg_len_big
    ///     8,        // recompute_every
    ///     0xC0FFEE,
    /// );
    /// ```
    ///
    /// Notes:
    /// - Open tour: endpoints (index 0 and n-1) are not moved by node/segment relocate.
    /// - This pass is intentionally “sharp”: it will often do nothing if no ratios exceed the gate.
    pub fn outlier_splice_repair_open_sniper(
        route: &mut Vec<Point>,
        passes: usize,
        max_spikes_to_consider: usize,
        local_w: usize,
        sample_cap: usize,
        trim_frac: f64,
        ratio_gate: f64,
        window: usize,
        max_local_candidates: usize,
        global_samples: usize,
        seg_len_big: usize,
        recompute_every: usize,
        seed: u64,
    ) {
        if route.len() < 6 || passes == 0 {
            return;
        }

        #[derive(Clone, Debug)]
        enum Move {
            RelocateNode {
                k: usize,
                j: usize,
            },
            RelocateSeg {
                k: usize,
                len: usize,
                j: usize,
                reverse: bool,
            },
            TwoOpt {
                i: usize,
                j: usize,
            }, // reverse [i..=j]
        }

        // ---------- deterministic PRNG (splitmix64) ----------
        #[inline]
        fn splitmix64_next(state: &mut u64) -> u64 {
            let mut z = state.wrapping_add(0x9E3779B97F4A7C15);
            *state = z;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }

        // ---------- helpers ----------
        // #[inline]
        // fn edge_len(route: &[Point], i: usize) -> f64 {
        //     route[i].dist(&route[i + 1])
        // }

        #[inline]
        fn window_bounds(n_nodes: usize, center_edge: usize, window: usize) -> (usize, usize) {
            // edges are 0..n_nodes-2
            let left = center_edge.saturating_sub(window);
            let right = (center_edge + 1 + window).min(n_nodes - 2);
            (left, right)
        }

        fn build_candidates(
            n_edges: usize,
            left_edge: usize,
            right_edge: usize,
            max_local: usize,
            global_samples: usize,
            rng: &mut u64,
        ) -> Vec<usize> {
            let mut cand = Vec::new();

            let span = right_edge.saturating_sub(left_edge);
            if span <= max_local {
                cand.extend(left_edge..=right_edge);
            } else {
                let step = (span / max_local).max(1);
                let mut j = left_edge;
                while j <= right_edge {
                    cand.push(j);
                    j = j.saturating_add(step);
                    if j == 0 {
                        break;
                    }
                }
            }

            if global_samples > 0 && n_edges > 4 {
                for _ in 0..global_samples {
                    cand.push((splitmix64_next(rng) as usize) % n_edges);
                }
            }

            cand.sort_unstable();
            cand.dedup();
            cand
        }

        fn apply_move(route: &mut Vec<Point>, mv: &Move) {
            match *mv {
                Move::RelocateNode { k, j } => {
                    let node = route.remove(k);
                    let jj = if k <= j { j.saturating_sub(1) } else { j };
                    let insert_pos = (jj + 1).min(route.len());
                    route.insert(insert_pos, node);
                }
                Move::RelocateSeg { k, len, j, reverse } => {
                    let end = k + len;
                    let mut seg: Vec<Point> = route.drain(k..end).collect();
                    if reverse {
                        seg.reverse();
                    }
                    let jj = if k <= j { j.saturating_sub(len) } else { j };
                    let insert_pos = (jj + 1).min(route.len());
                    route.splice(insert_pos..insert_pos, seg);
                }
                Move::TwoOpt { i, j } => {
                    if i < j && j < route.len() {
                        route[i..=j].reverse();
                    }
                }
            }
        }

        // ---- delta calculations (negative => improvement) ----
        #[inline]
        fn delta_relocate_node(route: &[Point], k: usize, j: usize) -> Option<f64> {
            let n = route.len();
            if n < 4 {
                return None;
            }
            if k == 0 || k + 1 >= n || j + 1 >= n {
                return None;
            }
            // disallow adjacent/no-op inserts
            if j == k || j + 1 == k || j == k - 1 {
                return None;
            }

            let prev_k = route[k - 1];
            let node = route[k];
            let next_k = route[k + 1];

            let a = route[j];
            let b = route[j + 1];

            let remove_gain = prev_k.dist(&next_k) - prev_k.dist(&node) - node.dist(&next_k);
            let insert_cost = a.dist(&node) + node.dist(&b) - a.dist(&b);

            Some(remove_gain + insert_cost)
        }

        #[inline]
        fn delta_relocate_seg(
            route: &[Point],
            k: usize,
            len: usize,
            j: usize,
            reverse: bool,
        ) -> Option<f64> {
            let n = route.len();
            if len < 2 || k == 0 || k + len >= n || j + 1 >= n {
                return None;
            }

            // disallow inserting into/near segment region
            let seg_lo = k.saturating_sub(1);
            let seg_hi = (k + len - 1).min(n - 2);
            if j >= seg_lo && j <= seg_hi {
                return None;
            }

            let prev = route[k - 1];
            let next = route[k + len];

            let s0 = route[k];
            let s1 = route[k + len - 1];
            let (seg_start, seg_end) = if !reverse { (s0, s1) } else { (s1, s0) };

            let a = route[j];
            let b = route[j + 1];

            let remove_gain = prev.dist(&next) - prev.dist(&s0) - s1.dist(&next);
            let insert_cost = a.dist(&seg_start) + seg_end.dist(&b) - a.dist(&b);

            Some(remove_gain + insert_cost)
        }

        #[inline]
        fn delta_2opt(route: &[Point], i: usize, j: usize) -> Option<f64> {
            let n = route.len();
            if i == 0 || i >= j || j + 1 >= n {
                return None;
            }
            let a = route[i - 1];
            let b = route[i];
            let c = route[j];
            let d = route[j + 1];
            let before = a.dist(&b) + c.dist(&d);
            let after = a.dist(&c) + b.dist(&d);
            Some(after - before)
        }

        // ---------- robust local baseline: trimmed mean of sampled window ----------
        // For edge i, we sample up to sample_cap edges in [l..=r], sort, trim ends, average middle.
        fn trimmed_mean_of_samples(
            d: &[f64],
            l: usize,
            r: usize,
            sample_cap: usize,
            trim_frac: f64,
        ) -> f64 {
            let span = r.saturating_sub(l).max(1);
            let take = sample_cap.min(span + 1).max(5);
            let step = (span / (take - 1)).max(1);

            let mut v: Vec<f64> = Vec::with_capacity(take);
            let mut j = l;
            while j <= r && v.len() < take {
                v.push(d[j]);
                j = j.saturating_add(step);
                if j == 0 {
                    break;
                }
            }

            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let n = v.len();

            let t = ((n as f64) * trim_frac).floor() as usize;
            let lo = t.min(n.saturating_sub(1));
            let hi = (n - t).max(lo + 1);

            let slice = &v[lo..hi];
            let sum: f64 = slice.iter().copied().sum();
            sum / (slice.len().max(1) as f64)
        }

        // Compute spike list: edges where ratio >= gate, sorted by ratio desc, capped.
        fn gated_spikes_par(
            route: &[Point],
            max_spikes: usize,
            local_w: usize,
            sample_cap: usize,
            trim_frac: f64,
            ratio_gate: f64,
        ) -> Vec<(usize, f64)> {
            let n = route.len();
            let n_edges = n - 1;

            // edge lengths parallel
            let d: Vec<f64> = (0..n_edges)
                .into_par_iter()
                .map(|i| route[i].dist(&route[i + 1]))
                .collect();

            let eps = 1e-9;

            let mut spikes: Vec<(usize, f64)> = (0..n_edges)
                .into_par_iter()
                .filter_map(|i| {
                    let l = i.saturating_sub(local_w);
                    let r = (i + local_w).min(n_edges - 1);

                    let base = trimmed_mean_of_samples(&d, l, r, sample_cap, trim_frac);
                    let ratio = d[i] / (base + eps);

                    if ratio >= ratio_gate {
                        Some((i, ratio))
                    } else {
                        None
                    }
                })
                .collect();

            spikes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            spikes.truncate(max_spikes.min(spikes.len()));
            spikes
        }

        // Find best move for one spike edge (aggressive).
        fn best_move_for_spike_par(
            route: &[Point],
            edge_i: usize,
            window: usize,
            max_local_candidates: usize,
            global_samples: usize,
            seg_len_big: usize,
            rng: &mut u64,
        ) -> Option<(Move, f64)> {
            let n = route.len();
            if edge_i + 1 >= n {
                return None;
            }
            let n_edges = n - 1;

            let (left_edge, right_edge) = window_bounds(n, edge_i, window);

            let cands = build_candidates(
                n_edges,
                left_edge.min(n_edges - 1),
                right_edge.min(n_edges - 1),
                max_local_candidates,
                global_samples,
                rng,
            );
            if cands.is_empty() {
                return None;
            }

            let seam_nodes = [edge_i, edge_i + 1];
            let mut best: Option<(Move, f64)> = None;

            // --- Node relocate (both endpoints) ---
            for &k in &seam_nodes {
                if k == 0 || k + 1 >= n {
                    continue;
                }
                let local_best = cands
                    .par_iter()
                    .filter_map(|&j| delta_relocate_node(route, k, j).map(|d| (j, d)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

                if let Some((j, delta)) = local_best {
                    if delta < -1e-9 {
                        let mv = Move::RelocateNode { k, j };
                        match best {
                            None => best = Some((mv, delta)),
                            Some((_, bd)) if delta < bd => best = Some((mv, delta)),
                            _ => {}
                        }
                    }
                }
            }

            // --- Segment relocate: FULL 2..=seg_len_big ---
            let seg_len_big = seg_len_big.max(2).min(n.saturating_sub(3));

            let best_seg = (2..=seg_len_big)
                .into_par_iter()
                .filter_map(|len| {
                    let mut best_for_len: Option<(Move, f64)> = None;

                    // start at seam nodes
                    for &k0 in &seam_nodes {
                        if k0 == 0 || k0 + len >= n {
                            continue;
                        }

                        // forward
                        let bf = cands
                            .par_iter()
                            .filter_map(|&j| {
                                delta_relocate_seg(route, k0, len, j, false).map(|d| (j, d))
                            })
                            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                        if let Some((j, d)) = bf {
                            if d < -1e-9 {
                                let mv = Move::RelocateSeg {
                                    k: k0,
                                    len,
                                    j,
                                    reverse: false,
                                };
                                match best_for_len {
                                    None => best_for_len = Some((mv, d)),
                                    Some((_, bd)) if d < bd => best_for_len = Some((mv, d)),
                                    _ => {}
                                }
                            }
                        }

                        // reverse
                        let br = cands
                            .par_iter()
                            .filter_map(|&j| {
                                delta_relocate_seg(route, k0, len, j, true).map(|d| (j, d))
                            })
                            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                        if let Some((j, d)) = br {
                            if d < -1e-9 {
                                let mv = Move::RelocateSeg {
                                    k: k0,
                                    len,
                                    j,
                                    reverse: true,
                                };
                                match best_for_len {
                                    None => best_for_len = Some((mv, d)),
                                    Some((_, bd)) if d < bd => best_for_len = Some((mv, d)),
                                    _ => {}
                                }
                            }
                        }
                    }

                    // end at seam nodes (k = end+1-len)
                    for &end_idx in &seam_nodes {
                        if end_idx + 1 < len {
                            continue;
                        }
                        let k = end_idx + 1 - len;
                        if k == 0 || k + len >= n {
                            continue;
                        }

                        let bf = cands
                            .par_iter()
                            .filter_map(|&j| {
                                delta_relocate_seg(route, k, len, j, false).map(|d| (j, d))
                            })
                            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                        if let Some((j, d)) = bf {
                            if d < -1e-9 {
                                let mv = Move::RelocateSeg {
                                    k,
                                    len,
                                    j,
                                    reverse: false,
                                };
                                match best_for_len {
                                    None => best_for_len = Some((mv, d)),
                                    Some((_, bd)) if d < bd => best_for_len = Some((mv, d)),
                                    _ => {}
                                }
                            }
                        }

                        let br = cands
                            .par_iter()
                            .filter_map(|&j| {
                                delta_relocate_seg(route, k, len, j, true).map(|d| (j, d))
                            })
                            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                        if let Some((j, d)) = br {
                            if d < -1e-9 {
                                let mv = Move::RelocateSeg {
                                    k,
                                    len,
                                    j,
                                    reverse: true,
                                };
                                match best_for_len {
                                    None => best_for_len = Some((mv, d)),
                                    Some((_, bd)) if d < bd => best_for_len = Some((mv, d)),
                                    _ => {}
                                }
                            }
                        }
                    }

                    best_for_len
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            if let Some((mv, delta)) = best_seg {
                match best {
                    None => best = Some((mv, delta)),
                    Some((_, bd)) if delta < bd => best = Some((mv, delta)),
                    _ => {}
                }
            }

            // --- Centered 2-opt: reverse [edge_i+1 ..= j] ---
            let rev_start = edge_i + 1;
            if rev_start < n - 2 {
                let best_2opt = cands
                    .par_iter()
                    .filter_map(|&j| {
                        if j < edge_i + 2 {
                            return None;
                        }
                        delta_2opt(route, rev_start, j).map(|d| (j, d))
                    })
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

                if let Some((j, delta)) = best_2opt {
                    if delta < -1e-9 {
                        let mv = Move::TwoOpt { i: rev_start, j };
                        match best {
                            None => best = Some((mv, delta)),
                            Some((_, bd)) if delta < bd => best = Some((mv, delta)),
                            _ => {}
                        }
                    }
                }
            }

            best
        }

        let n0 = route.len();
        let mut rng_state = seed ^ (n0 as u64).wrapping_mul(0x9E3779B97F4A7C15);

        for _pass in 0..passes {
            let mut applied = 0usize;

            loop {
                let spikes = gated_spikes_par(
                    route,
                    max_spikes_to_consider,
                    local_w,
                    sample_cap,
                    trim_frac,
                    ratio_gate,
                );

                if spikes.is_empty() {
                    break; // nothing exceeds gate
                }

                let mut any_applied = false;

                for (edge_i, _ratio) in spikes.iter().map(|(i, r)| (*i, *r)) {
                    if edge_i + 1 >= route.len() {
                        continue;
                    }

                    if let Some((mv, _delta)) = best_move_for_spike_par(
                        route,
                        edge_i,
                        window,
                        max_local_candidates,
                        global_samples,
                        seg_len_big,
                        &mut rng_state,
                    ) {
                        apply_move(route, &mv);
                        applied += 1;
                        any_applied = true;

                        if recompute_every > 0 && applied % recompute_every == 0 {
                            break; // recompute spikes list
                        }
                    }
                }

                if !any_applied {
                    break;
                }
                if recompute_every > 0 && applied % recompute_every == 0 {
                    continue;
                } else {
                    break;
                }
            }
        }

        debug_assert_eq!(route.len(), n0, "sniper must not change point count");
    }
}

fn spike_edges_by_threshold(edge_len: &[f64], ratio_gate: f64, cap: usize) -> Vec<usize> {
    if edge_len.is_empty() {
        return Vec::new();
    }
    let mean = mean_cycle_edge(edge_len).max(1e-9);

    let mut idxs: Vec<usize> = edge_len
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| if d > ratio_gate * mean { Some(i) } else { None })
        .collect();

    // Sort by length descending, keep cap
    idxs.sort_by(|&i, &j| edge_len[j].partial_cmp(&edge_len[i]).unwrap());
    idxs.truncate(cap.min(idxs.len()));
    idxs
}
