//! Solve pipeline and multi-core orchestration.
//!
//! Large instances are optimized with *segment rounds*: the tour is split
//! into contiguous segments, each segment is extracted as an independent
//! sub-problem (an open path with fixed endpoints) and optimized on its own
//! core, then the improved paths are joined back in place. The split offset
//! rotates every round so former segment boundaries become segment interiors
//! and get optimized too. Small instances instead run one independent
//! iterated-local-search walker per core and keep the best result.

use alloc::vec;
use alloc::vec::Vec;
use core::time::Duration;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::candidates::Candidates;
use crate::construct::greedy_tour;
use crate::kdtree::{KdTree, dist};
use crate::platform::{self, Instant};
use crate::rng::SplitMix64;
use crate::state::TourState;

#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SolverConfig {
    /// Wall-clock budget. `None` derives one from the instance size.
    /// Construction (candidate lists + greedy tour) always runs to
    /// completion, so the effective floor is the construction time; the
    /// budget bounds the improvement phases.
    pub time_limit: Option<Duration>,
    /// RNG seed for perturbation; fixed seed gives reproducible search paths.
    pub seed: u64,
    /// Worker threads. `0` uses all available cores.
    pub threads: usize,
    /// k for nearest-neighbor candidate generation.
    pub max_neighbors: usize,
    /// Per-node candidate cap after symmetrization.
    pub max_candidates: usize,
    /// Longest segment considered by Or-opt during normal search.
    pub or_opt_max_len: usize,
    /// Instances up to this size use parallel multi-start instead of
    /// segment rounds.
    pub multi_start_max: usize,
    /// Segment size at the start of segment rounds. Segments grow (their
    /// count halves) whenever a round stops improving: small segments give
    /// parallel speed early, large segments give quality late, because every
    /// segment boundary is a pair of frozen tour edges.
    pub min_segment_len: usize,
    /// Upper bound on segment size at the coarse end of the schedule.
    pub max_segment_len: usize,
    /// Cap on segments per thread at the fine end of the schedule.
    pub segments_per_thread: usize,
    /// When a round has fewer segments than threads, each segment is solved
    /// up to this many times in parallel with different perturbation seeds
    /// and the best result wins ("best-of-k"). Uses otherwise-idle cores at
    /// the coarse end of the schedule; can only improve a round. `1` = off.
    pub max_segment_replicas: usize,
    /// Double-bridge kick span in tour positions. `0` = auto.
    pub kick_window: usize,
    /// Longest Or-opt segment during spike repair.
    pub spike_or_opt_len: usize,
    /// Edges longer than `spike_factor * avg_edge` get targeted repair.
    pub spike_factor: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            time_limit: None,
            seed: 12_345,
            threads: 0,
            max_neighbors: 10,
            max_candidates: 16,
            or_opt_max_len: 3,
            multi_start_max: 10_000,
            min_segment_len: 3_000,
            max_segment_len: 50_000,
            segments_per_thread: 4,
            max_segment_replicas: 16,
            kick_window: 0,
            spike_or_opt_len: 12,
            spike_factor: 3.0,
        }
    }
}

impl SolverConfig {
    fn budget(&self, n: usize) -> Duration {
        self.time_limit
            .unwrap_or_else(|| Duration::from_secs_f64((n as f64 / 1000.0).clamp(2.0, 120.0)))
    }

    fn kick_window_for(&self, n: usize) -> usize {
        if self.kick_window > 0 {
            self.kick_window
        } else {
            (n / 6).clamp(8, 400)
        }
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub struct Solution {
    /// Visiting order as indices into the input point slice.
    pub tour: Vec<u32>,
    /// Total cycle length (same metric as the input coordinates).
    pub length: f64,
}

pub fn solve<const D: usize>(pts: &[[f64; D]], cfg: &SolverConfig) -> Solution {
    let n = pts.len();
    if n <= 3 {
        let tour: Vec<u32> = (0..n as u32).collect();
        let length = cycle_length(pts, &tour);
        return Solution { tour, length };
    }
    // Custom pools are unsupported on wasm (wasm-bindgen-rayon initializes
    // the global pool instead), and pool creation can fail under OS resource
    // pressure; in both cases fall back to the current/global pool rather
    // than panicking.
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    match rayon::ThreadPoolBuilder::new()
        .num_threads(cfg.threads)
        .build()
    {
        Ok(pool) => return pool.install(|| solve_inner(pts, cfg)),
        Err(err) => log::warn!("solver: thread pool build failed ({err}); using global pool"),
    }
    solve_inner(pts, cfg)
}

/// Worker count visible to the orchestration heuristics.
fn worker_count() -> usize {
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads().max(1)
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

fn solve_inner<const D: usize>(pts: &[[f64; D]], cfg: &SolverConfig) -> Solution {
    let n = pts.len();
    let start = Instant::now();
    let deadline = start + cfg.budget(n);

    let tree = KdTree::build(pts);
    let cand = Candidates::build(pts, &tree, cfg.max_neighbors, cfg.max_candidates);
    log::info!(
        "solver: candidates built n={n} k={} in {:.2}s",
        cfg.max_neighbors,
        start.elapsed().as_secs_f64()
    );

    let initial = greedy_tour(pts, &cand, &tree);
    log::info!(
        "solver: greedy tour len={:.0} in {:.2}s",
        cycle_length(pts, &initial),
        start.elapsed().as_secs_f64()
    );

    let seg_capable = n / cfg.min_segment_len.max(4) >= 2;

    let mut order = if n <= cfg.multi_start_max || !seg_capable {
        multi_start(pts, &cand, initial, cfg, deadline)
    } else {
        segment_rounds(pts, &cand, initial, cfg, deadline)
    };

    // Final sequential polish + spike repair on the whole tour.
    let rng = SplitMix64::derive(cfg.seed, 0xF17A1, 0);
    let mut st = TourState::new(
        pts,
        &cand,
        core::mem::take(&mut order),
        None,
        cfg.or_opt_max_len,
        cfg.kick_window_for(n),
        rng,
    );
    st.activate_all();
    st.run(deadline);
    spike_pass(&mut st, cfg, deadline);
    log::info!(
        "solver: finished len={:.0} in {:.2}s",
        st.cur_len,
        start.elapsed().as_secs_f64()
    );

    let length = st.tour_length();
    Solution {
        tour: st.order,
        length,
    }
}

/// One independent iterated-local-search walker per core; best tour wins.
fn multi_start<const D: usize>(
    pts: &[[f64; D]],
    cand: &Candidates,
    initial: Vec<u32>,
    cfg: &SolverConfig,
    deadline: Instant,
) -> Vec<u32> {
    let n = pts.len();
    let runs = worker_count();
    // Without a clock the kick loop needs a finite budget; with one, the
    // deadline is the budget.
    let kicks = if cfg!(feature = "std") {
        usize::MAX
    } else {
        (n * 8).max(1024)
    };
    let walker = |run: usize| {
        let rng = SplitMix64::derive(cfg.seed, 0x5EED, run as u64);
        let mut st = TourState::new(
            pts,
            cand,
            initial.clone(),
            None,
            cfg.or_opt_max_len,
            cfg.kick_window_for(n),
            rng,
        );
        st.optimize(deadline, kicks);
        (st.cur_len, st.order)
    };
    #[cfg(feature = "parallel")]
    let results: Vec<(f64, Vec<u32>)> = (0..runs).into_par_iter().map(walker).collect();
    #[cfg(not(feature = "parallel"))]
    let results: Vec<(f64, Vec<u32>)> = (0..runs).map(walker).collect();
    let best = results
        .into_iter()
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .expect("at least one run");
    log::info!(
        "solver: multi-start best len={:.0} over {runs} runs",
        best.0
    );
    best.1
}

/// Parallel split/optimize/join rounds with rotating segment boundaries.
fn segment_rounds<const D: usize>(
    pts: &[[f64; D]],
    cand: &Candidates,
    initial: Vec<u32>,
    cfg: &SolverConfig,
    deadline: Instant,
) -> Vec<u32> {
    let n = pts.len();
    let threads = worker_count();
    // Coarse-to-fine schedule: start with enough segments to keep every core
    // busy for the initial descent, then grow segments (halve their count)
    // as rounds stop improving — benchmarks show segment boundaries are the
    // main quality cost, so the endgame wants few, large segments.
    let count_cap = (threads * cfg.segments_per_thread).max(2);
    let floor_count = (n / cfg.max_segment_len.max(cfg.min_segment_len).max(4)).max(2);
    // The max_segment_len cost guard takes precedence over the thread cap:
    // with few threads and huge n, count_cap alone would produce oversized
    // segments (and starve the coarsening branch).
    let mut seg_count = (n / cfg.min_segment_len.max(4))
        .clamp(2, count_cap)
        .max(floor_count);
    // Reserve tail time for the final global polish + spike repair.
    let now = Instant::now();
    let rounds_deadline = now + deadline.saturating_duration_since(now).mul_f64(0.85);

    let mut order = initial;
    let mut len = cycle_length(pts, &order);
    let mut stall = 0usize;
    let mut round = 0u64;
    // Once at the coarsest segmentation, spend the remaining budget on
    // progressively heavier perturbation instead of giving up early.
    let mut kick_scale = 1usize;
    const MAX_KICK_SCALE: usize = 64;
    loop {
        let bounds = segment_bounds(n, seg_count, round);
        // Idle cores run redundant solves of the same segments (different
        // kick seeds, best result wins) instead of sitting out the round.
        let replicas = (threads / bounds.len()).clamp(1, cfg.max_segment_replicas.max(1));
        let mut pos = vec![0u32; n];
        for (i, &v) in order.iter().enumerate() {
            pos[v as usize] = i as u32;
        }

        let order_ref = &order;
        let pos_ref = &pos;
        let solve_one = |(seg_idx, &(lo, seg_len)): (usize, &(usize, usize))| {
            let path = solve_segment(
                pts,
                cand,
                order_ref,
                pos_ref,
                lo,
                seg_len,
                cfg,
                round,
                seg_idx as u64,
                kick_scale,
                replicas,
                platform::earlier(rounds_deadline, deadline),
            );
            (lo, path)
        };
        #[cfg(feature = "parallel")]
        let paths: Vec<(usize, Vec<u32>)> = bounds.par_iter().enumerate().map(solve_one).collect();
        #[cfg(not(feature = "parallel"))]
        let paths: Vec<(usize, Vec<u32>)> = bounds.iter().enumerate().map(solve_one).collect();

        let mut new_order = vec![0u32; n];
        for (lo, path) in paths {
            for (i, &v) in path.iter().enumerate() {
                new_order[(lo + i) % n] = v;
            }
        }
        order = new_order;

        let new_len = cycle_length(pts, &order);
        let improved = len - new_len;
        log::debug!(
            "solver: round={round} segments={} kick_scale={kick_scale} len={new_len:.0} improved={improved:.0}",
            bounds.len()
        );
        if improved < 1e-6 * len.max(1.0) {
            if seg_count > floor_count {
                seg_count = (seg_count / 2).max(floor_count);
            } else {
                kick_scale = (kick_scale * 2).min(MAX_KICK_SCALE);
                stall += 1;
            }
        } else {
            stall = 0;
        }
        len = new_len;
        round += 1;
        if platform::expired(rounds_deadline) || (stall >= 6 && kick_scale >= MAX_KICK_SCALE) {
            break;
        }
    }
    log::info!("solver: segment rounds done rounds={round} len={len:.0}");
    order
}

/// Extract the tour segment starting at global position `lo`, optimize it as
/// an open path with fixed endpoints (up to `replicas` independent attempts
/// in parallel, shortest wins), and return the improved path in global node
/// ids.
#[allow(clippy::too_many_arguments)]
fn solve_segment<const D: usize>(
    pts: &[[f64; D]],
    cand: &Candidates,
    order: &[u32],
    pos: &[u32],
    lo: usize,
    seg_len: usize,
    cfg: &SolverConfig,
    round: u64,
    seg_idx: u64,
    kick_scale: usize,
    replicas: usize,
    deadline: Instant,
) -> Vec<u32> {
    let n = order.len();
    let globals: Vec<u32> = (0..seg_len).map(|i| order[(lo + i) % n]).collect();
    if seg_len < 8 {
        return globals;
    }
    let local_pts: Vec<[f64; D]> = globals.iter().map(|&g| pts[g as usize]).collect();
    let lists: Vec<Vec<(f64, u32)>> = globals
        .iter()
        .map(|&g| {
            cand.neighbors(g)
                .filter_map(|(t, d)| {
                    let rel = (pos[t as usize] as usize + n - lo) % n;
                    (rel < seg_len).then_some((d, rel as u32))
                })
                .collect()
        })
        .collect();
    let local_cand = Candidates::from_lists(lists);
    let frozen = Some((seg_len as u32 - 1, 0));
    let kicks = (seg_len / 4).max(32).saturating_mul(kick_scale);

    // Replica 0 uses the same RNG stream as a replica-free run, so best-of-k
    // is never worse than solving the segment once.
    let attempt = |rep: u64| {
        let rng = SplitMix64::derive(
            cfg.seed,
            round.wrapping_mul(0x9E37) ^ 0xA11CE,
            seg_idx | (rep << 32),
        );
        let mut st = TourState::new(
            &local_pts,
            &local_cand,
            (0..seg_len as u32).collect(),
            frozen,
            cfg.or_opt_max_len,
            cfg.kick_window_for(seg_len),
            rng,
        );
        st.optimize(deadline, kicks);
        (st.cur_len, rep, st.path_from(0, seg_len as u32 - 1))
    };
    let range = 0..replicas.max(1) as u64;
    #[cfg(feature = "parallel")]
    let best = range
        .into_par_iter()
        .map(attempt)
        .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    #[cfg(not(feature = "parallel"))]
    let best = range
        .map(attempt)
        .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let (_, _, path) = best.expect("at least one replica");

    path.into_iter()
        .map(|local| globals[local as usize])
        .collect()
}

/// Targeted repair of outlier edges: re-activate the endpoints of unusually
/// long edges and allow longer Or-opt relocations, which unhooks "spikes"
/// (points or short chains visited far out of the way).
fn spike_pass<const D: usize>(st: &mut TourState<'_, D>, cfg: &SolverConfig, deadline: Instant) {
    let n = st.order.len();
    if n < 8 || cfg.spike_factor <= 0.0 {
        return;
    }
    let avg = st.cur_len / n as f64;
    let threshold = avg * cfg.spike_factor;
    let mut hot: Vec<u32> = Vec::new();
    for i in 0..n {
        let a = st.order[i];
        let b = st.order[(i + 1) % n];
        if st.dist_between(a, b) > threshold {
            hot.push(a);
            hot.push(b);
        }
    }
    if hot.is_empty() {
        return;
    }
    log::debug!("solver: spike pass on {} endpoints", hot.len());
    st.set_or_opt_max(cfg.spike_or_opt_len);
    st.activate(hot.iter().copied());
    st.run(deadline);
    st.set_or_opt_max(cfg.or_opt_max_len);
    // Spike moves can open new 2-opt opportunities nearby; settle them.
    st.activate(hot);
    st.run(deadline);
}

pub fn cycle_length<const D: usize>(pts: &[[f64; D]], tour: &[u32]) -> f64 {
    let n = tour.len();
    if n < 2 {
        return 0.0;
    }
    (0..n)
        .map(|i| dist(&pts[tour[i] as usize], &pts[tour[(i + 1) % n] as usize]))
        .sum()
}

fn segment_bounds(n: usize, seg_count: usize, round: u64) -> Vec<(usize, usize)> {
    let seg_count = seg_count.min(n / 4).max(1);
    if seg_count < 2 {
        return vec![(0, n)];
    }
    let base = n / seg_count;
    // Rotate boundaries by an irregular stride so they sweep the whole tour
    // instead of oscillating between two alignments.
    let stride = base / 2 + base / 5 + 1;
    let offset = ((round as usize) * stride) % n;
    let mut bounds = Vec::with_capacity(seg_count);
    let mut start = offset;
    let mut remaining = n;
    for i in 0..seg_count {
        let len = if i + 1 == seg_count { remaining } else { base };
        bounds.push((start, len));
        start = (start + len) % n;
        remaining -= len;
    }
    bounds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_bounds_partition_the_cycle() {
        for (n, seg_count, round) in [(100, 4, 0u64), (1003, 7, 3), (50_000, 128, 9)] {
            let bounds = segment_bounds(n, seg_count, round);
            let total: usize = bounds.iter().map(|&(_, l)| l).sum();
            assert_eq!(total, n);
            let mut covered = vec![false; n];
            for &(lo, len) in &bounds {
                for i in 0..len {
                    let p = (lo + i) % n;
                    assert!(!covered[p], "position {p} covered twice");
                    covered[p] = true;
                }
            }
            assert!(covered.iter().all(|&c| c));
        }
    }

    #[test]
    fn segment_bounds_rotate_between_rounds() {
        let a = segment_bounds(10_000, 8, 0);
        let b = segment_bounds(10_000, 8, 1);
        assert_ne!(a[0].0, b[0].0);
    }
}
