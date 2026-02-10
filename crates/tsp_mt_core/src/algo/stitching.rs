use std::collections::{HashMap, HashSet};

use h3o::{CellIndex, Resolution};
use kiddo::{KdTree, SquaredEuclidean};

use crate::{geometry, node::LKHNode};

const MIN_TOUR_SIZE_FOR_2OPT: usize = 4;
const TWO_OPT_IMPROVEMENT_EPSILON: f64 = 1e-5;

#[derive(Clone, Copy, Debug)]
pub(crate) struct StitchingTuning {
    pub(crate) candidate_pair_check_limit: usize,
    pub(crate) portal_node_floor: usize,
    pub(crate) portal_node_cap: usize,
    pub(crate) portal_pair_check_limit: usize,
    pub(crate) large_jump_distance_threshold: f64,
    pub(crate) large_jump_penalty: f64,
    pub(crate) very_large_jump_distance_threshold: f64,
    pub(crate) very_large_jump_penalty: f64,
    pub(crate) non_neighbor_bridge_penalty: f64,
    pub(crate) non_neighbor_selection_penalty_multiplier: f64,
    pub(crate) long_edge_boundary_multiplier: f64,
    pub(crate) long_edge_boundary_limit: usize,
    pub(crate) spike_repair_top_n: usize,
    pub(crate) spike_repair_window: usize,
    pub(crate) spike_repair_passes: usize,
}

impl Default for StitchingTuning {
    fn default() -> Self {
        Self {
            candidate_pair_check_limit: 120,
            portal_node_floor: 8,
            portal_node_cap: 32,
            portal_pair_check_limit: 96,
            large_jump_distance_threshold: 1_000.0,
            large_jump_penalty: 500.0,
            very_large_jump_distance_threshold: 2_500.0,
            very_large_jump_penalty: 1_200.0,
            non_neighbor_bridge_penalty: 250.0,
            non_neighbor_selection_penalty_multiplier: 2.0,
            long_edge_boundary_multiplier: 1.75,
            long_edge_boundary_limit: 24,
            spike_repair_top_n: 24,
            spike_repair_window: 500,
            spike_repair_passes: 3,
        }
    }
}

struct MergeResult {
    merged: Vec<usize>,
    /// Each index i refers to boundary edge (i -> i+1).
    boundaries: [usize; 2],
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ChunkStitchState {
    pub(crate) cell: CellIndex,
    pub(crate) resolution: Resolution,
    pub(crate) centroid: LKHNode,
    pub(crate) size: usize,
}

struct PendingChunk {
    tour: Vec<usize>,
    state: ChunkStitchState,
}

#[derive(Clone, Copy)]
struct MergePortalHints {
    from_a_target: LKHNode,
    from_b_target: LKHNode,
    non_neighbor: bool,
}

#[allow(dead_code)]
pub(crate) fn stitch_chunk_tours_dense(
    coords: &[LKHNode],
    chunk_tours: Vec<Vec<usize>>,
) -> (Vec<usize>, Vec<usize>) {
    stitch_chunk_tours_dense_impl(coords, chunk_tours, None, StitchingTuning::default())
}

pub(crate) fn stitch_chunk_tours_dense_with_state(
    coords: &[LKHNode],
    chunk_tours: Vec<Vec<usize>>,
    chunk_states: Vec<ChunkStitchState>,
    tuning: StitchingTuning,
) -> (Vec<usize>, Vec<usize>) {
    stitch_chunk_tours_dense_impl(coords, chunk_tours, Some(chunk_states), tuning)
}

fn stitch_chunk_tours_dense_impl(
    coords: &[LKHNode],
    mut chunk_tours: Vec<Vec<usize>>,
    chunk_states: Option<Vec<ChunkStitchState>>,
    tuning: StitchingTuning,
) -> (Vec<usize>, Vec<usize>) {
    let chunk_count = chunk_tours.len();
    if chunk_count == 0 {
        log::warn!("stitcher: no chunk tours to merge");
        return (Vec::new(), Vec::new());
    }

    log::info!("stitcher: start merge chunk_tours={chunk_count}");

    if let Some(chunk_states) = chunk_states {
        if chunk_states.len() != chunk_count {
            log::warn!(
                "stitcher: chunk state count mismatch chunk_tours={} chunk_states={} fallback=plain",
                chunk_count,
                chunk_states.len()
            );
        } else {
            let mut pending: Vec<PendingChunk> = chunk_tours
                .into_iter()
                .zip(chunk_states)
                .map(|(tour, state)| PendingChunk { tour, state })
                .collect();
            let first = pending.remove(0);
            let mut merged = first.tour;
            let mut merged_states = vec![first.state];
            let mut boundaries: Vec<usize> = Vec::new();
            let mut merge_idx = 0usize;

            while !pending.is_empty() {
                let next_idx = choose_next_pending_chunk(&pending, &merged_states, tuning);
                let next = pending.swap_remove(next_idx);
                let is_neighbor = merged_states
                    .iter()
                    .any(|state| chunks_are_neighbors(*state, next.state));
                let merged_centroid = weighted_state_centroid(&merged_states);
                let portal_hints = MergePortalHints {
                    from_a_target: next.state.centroid,
                    from_b_target: merged_centroid,
                    non_neighbor: !is_neighbor,
                };
                let res = merge_two_cycles_dense(coords, &merged, &next.tour, portal_hints, tuning);
                merged = res.merged;
                boundaries.extend_from_slice(&res.boundaries);
                merged_states.push(next.state);
                merge_idx += 1;

                log::debug!(
                    "stitcher.merge: done step={} merged_n={} boundaries={} pending={} neighbor={}",
                    merge_idx,
                    merged.len(),
                    boundaries.len(),
                    pending.len(),
                    is_neighbor
                );
            }

            log::info!(
                "stitcher: complete merged_n={} boundaries={}",
                merged.len(),
                boundaries.len()
            );
            return (merged, boundaries);
        }
    }

    let mut merged = chunk_tours.remove(0);
    let mut boundaries: Vec<usize> = Vec::new();

    for (merge_idx, tour) in chunk_tours.into_iter().enumerate() {
        let portal_hints = fallback_portal_hints(coords, &merged, &tour);
        let res = merge_two_cycles_dense(coords, &merged, &tour, portal_hints, tuning);
        merged = res.merged;
        boundaries.extend_from_slice(&res.boundaries);
        log::debug!(
            "stitcher.merge: done step={} merged_n={} boundaries={}",
            merge_idx + 1,
            merged.len(),
            boundaries.len()
        );
    }

    log::info!(
        "stitcher: complete merged_n={} boundaries={}",
        merged.len(),
        boundaries.len()
    );
    (merged, boundaries)
}

fn weighted_state_centroid(states: &[ChunkStitchState]) -> LKHNode {
    let (sx, sy, sw) = states.iter().fold((0.0, 0.0, 0.0), |acc, state| {
        let weight = state.size.max(1) as f64;
        (
            acc.0 + state.centroid.x * weight,
            acc.1 + state.centroid.y * weight,
            acc.2 + weight,
        )
    });
    if sw > 0.0 {
        LKHNode::new(sx / sw, sy / sw)
    } else {
        LKHNode::new(0.0, 0.0)
    }
}

fn fallback_portal_hints(
    coords: &[LKHNode],
    tour_a: &[usize],
    tour_b: &[usize],
) -> MergePortalHints {
    MergePortalHints {
        from_a_target: tour_centroid(coords, tour_b),
        from_b_target: tour_centroid(coords, tour_a),
        non_neighbor: false,
    }
}

fn choose_next_pending_chunk(
    pending: &[PendingChunk],
    merged_states: &[ChunkStitchState],
    tuning: StitchingTuning,
) -> usize {
    pending
        .iter()
        .enumerate()
        .min_by(|(idx_l, lhs), (idx_r, rhs)| {
            let lhs_score = chunk_selection_score(lhs.state, merged_states, tuning);
            let rhs_score = chunk_selection_score(rhs.state, merged_states, tuning);
            lhs_score
                .total_cmp(&rhs_score)
                .then_with(|| idx_l.cmp(idx_r))
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn chunk_selection_score(
    candidate: ChunkStitchState,
    merged_states: &[ChunkStitchState],
    tuning: StitchingTuning,
) -> f64 {
    let min_distance = merged_states
        .iter()
        .map(|state| geometry::dist(candidate.centroid, state.centroid))
        .fold(f64::INFINITY, f64::min);
    let min_distance = if min_distance.is_finite() {
        min_distance
    } else {
        0.0
    };
    let has_neighbor = merged_states
        .iter()
        .any(|state| chunks_are_neighbors(candidate, *state));
    if has_neighbor {
        min_distance
    } else {
        min_distance * tuning.non_neighbor_selection_penalty_multiplier
    }
}

fn chunks_are_neighbors(a: ChunkStitchState, b: ChunkStitchState) -> bool {
    if a.cell == b.cell {
        return true;
    }

    let target_resolution = if a.resolution <= b.resolution {
        a.resolution
    } else {
        b.resolution
    };

    let a_cell = normalize_cell_resolution(a.cell, a.resolution, target_resolution);
    let b_cell = normalize_cell_resolution(b.cell, b.resolution, target_resolution);
    let (Some(a_cell), Some(b_cell)) = (a_cell, b_cell) else {
        return false;
    };

    a_cell == b_cell || a_cell.is_neighbor_with(b_cell).unwrap_or(false)
}

fn normalize_cell_resolution(
    cell: CellIndex,
    source_resolution: Resolution,
    target_resolution: Resolution,
) -> Option<CellIndex> {
    if source_resolution == target_resolution {
        Some(cell)
    } else {
        cell.parent(target_resolution)
    }
}

pub(crate) fn augment_boundaries_with_long_edges(
    coords: &[LKHNode],
    tour: &[usize],
    boundaries: &[usize],
    tuning: StitchingTuning,
) -> Vec<usize> {
    let n = tour.len();
    if n < MIN_TOUR_SIZE_FOR_2OPT {
        return boundaries.to_vec();
    }

    let edge_lengths: Vec<(usize, f64)> = (0..n)
        .map(|idx| {
            let a = tour[idx];
            let b = tour[(idx + 1) % n];
            (idx, geometry::dist(coords[a], coords[b]))
        })
        .collect();
    let total: f64 = edge_lengths.iter().map(|(_, len)| *len).sum();
    let average = total / (n as f64);
    let threshold = average * tuning.long_edge_boundary_multiplier;

    let mut long_edges: Vec<(usize, f64)> = edge_lengths
        .into_iter()
        .filter(|(_, len)| *len > threshold)
        .collect();
    long_edges.sort_unstable_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));

    let mut focused = Vec::with_capacity(boundaries.len() + tuning.long_edge_boundary_limit);
    let mut seen: HashSet<usize> = HashSet::new();

    for &boundary in boundaries {
        if seen.insert(boundary) {
            focused.push(boundary);
        }
    }
    for (edge_idx, _) in long_edges.into_iter().take(tuning.long_edge_boundary_limit) {
        if seen.insert(edge_idx) {
            focused.push(edge_idx);
        }
    }

    focused
}

pub(crate) fn spike_repair_two_opt(
    coords: &[LKHNode],
    tour: &mut [usize],
    tuning: StitchingTuning,
) {
    if tuning.spike_repair_top_n == 0
        || tuning.spike_repair_window == 0
        || tuning.spike_repair_passes == 0
    {
        log::debug!("stitcher.spike_repair: skip reason=disabled");
        return;
    }
    let n = tour.len();
    if n < MIN_TOUR_SIZE_FOR_2OPT {
        log::debug!(
            "stitcher.spike_repair: skip n={} reason=insufficient_size",
            n
        );
        return;
    }

    let tracked_edges = tuning.spike_repair_top_n.min(n);
    if tracked_edges == 0 {
        log::debug!("stitcher.spike_repair: skip reason=no_boundaries");
        return;
    }

    log::info!(
        "stitcher.spike_repair: start boundaries={} window={} passes={}",
        tracked_edges,
        tuning.spike_repair_window,
        tuning.spike_repair_passes
    );

    let mut total_swaps = 0usize;
    let mut passes_executed = 0usize;
    for pass_idx in 0..tuning.spike_repair_passes {
        passes_executed = pass_idx + 1;
        let spike_edges = top_spike_edges(coords, tour, tracked_edges);
        let mut pass_swaps = 0usize;

        for edge_idx in spike_edges {
            if try_spike_edge_repair(coords, tour, edge_idx, tuning.spike_repair_window) {
                pass_swaps += 1;
            }
        }

        total_swaps += pass_swaps;
        log::debug!(
            "stitcher.spike_repair: pass={} swaps={}",
            pass_idx + 1,
            pass_swaps
        );
        if pass_swaps == 0 {
            break;
        }
    }
    log::info!(
        "stitcher.spike_repair: complete passes={} swaps={}",
        passes_executed,
        total_swaps
    );
}

fn top_spike_edges(coords: &[LKHNode], tour: &[usize], top_n: usize) -> Vec<usize> {
    let mut edges: Vec<(usize, f64)> = (0..tour.len())
        .map(|idx| {
            let a = tour[idx];
            let b = tour[(idx + 1) % tour.len()];
            (idx, geometry::dist(coords[a], coords[b]))
        })
        .collect();
    edges.sort_unstable_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));
    edges.into_iter().take(top_n).map(|(idx, _)| idx).collect()
}

fn try_spike_edge_repair(
    coords: &[LKHNode],
    tour: &mut [usize],
    edge_idx: usize,
    window: usize,
) -> bool {
    let n = tour.len();
    if n < MIN_TOUR_SIZE_FOR_2OPT || window < 2 {
        return false;
    }

    let i = edge_idx % n;
    let i_next = (i + 1) % n;
    let a = tour[i];
    let b = tour[i_next];

    let mut best_k: Option<usize> = None;
    let mut best_gain = 0.0;
    let max_offset = window.min(n.saturating_sub(2));

    for offset in 2..=max_offset {
        let k = (i + offset) % n;
        let k_next = (k + 1) % n;
        if k == i || k == i_next {
            continue;
        }
        if k_next == i {
            continue;
        }

        let c = tour[k];
        let d = tour[k_next];
        let current = geometry::dist(coords[a], coords[b]) + geometry::dist(coords[c], coords[d]);
        let proposal = geometry::dist(coords[a], coords[c]) + geometry::dist(coords[b], coords[d]);
        let gain = current - proposal;
        if gain > TWO_OPT_IMPROVEMENT_EPSILON && gain > best_gain {
            best_gain = gain;
            best_k = Some(k);
        }
    }

    let Some(best_k) = best_k else {
        return false;
    };
    reverse_cyclic_segment(tour, i_next, best_k);
    true
}

fn reverse_cyclic_segment(tour: &mut [usize], start: usize, end: usize) {
    let n = tour.len();
    if n == 0 {
        return;
    }
    let mut idxs = Vec::new();
    let mut cur = start % n;
    loop {
        idxs.push(cur);
        if cur == end % n {
            break;
        }
        cur = (cur + 1) % n;
    }

    let mut left = 0usize;
    let mut right = idxs.len().saturating_sub(1);
    while left < right {
        tour.swap(idxs[left], idxs[right]);
        left += 1;
        right = right.saturating_sub(1);
    }
}

#[tsp_mt_derive::timer("stitcher.2opt")]
pub(crate) fn boundary_two_opt(
    coords: &[LKHNode],
    tour: &mut [usize],
    boundaries: &[usize],
    window: usize,
    passes: usize,
) {
    let n = tour.len();
    if n < MIN_TOUR_SIZE_FOR_2OPT || boundaries.is_empty() {
        log::debug!(
            "stitcher.2opt: skip n={} boundaries={} reason=insufficient_input",
            n,
            boundaries.len()
        );
        return;
    }

    let mut passes_executed = 0usize;
    let mut total_swaps = 0usize;
    for pass_idx in 0..passes {
        passes_executed = pass_idx + 1;
        let mut pass_swaps = 0usize;

        for &b_idx in boundaries {
            let start = b_idx.saturating_sub(window);
            let end = (b_idx + window).min(n - 1);

            for i in start..end {
                for k in (i + 1)..end {
                    let j = k + 1;

                    let idx_i = i;
                    let idx_i1 = i + 1;
                    let idx_k = k;
                    let idx_j = j;

                    let a = tour[idx_i];
                    let b = tour[idx_i1];
                    let c = tour[idx_k];
                    let d = tour[idx_j];

                    let cur_dist =
                        geometry::dist(coords[a], coords[b]) + geometry::dist(coords[c], coords[d]);
                    let new_dist =
                        geometry::dist(coords[a], coords[c]) + geometry::dist(coords[b], coords[d]);

                    if new_dist < cur_dist - TWO_OPT_IMPROVEMENT_EPSILON {
                        tour[(idx_i + 1)..=idx_k].reverse();
                        pass_swaps += 1;
                    }
                }
            }
        }

        total_swaps += pass_swaps;
        log::debug!("stitcher.2opt: pass={} swaps={}", pass_idx + 1, pass_swaps);

        if pass_swaps == 0 {
            break;
        }
    }
    log::info!(
        "stitcher.2opt: complete n={} boundaries={} passes={} swaps={}",
        n,
        boundaries.len(),
        passes_executed,
        total_swaps
    );
}

/// Merges two tours by finding the strictly closest points between them.
/// This prevents random portals from creating large outlier jumps.
fn merge_two_cycles_dense(
    coords: &[LKHNode],
    tour_a: &[usize],
    tour_b: &[usize],
    portal_hints: MergePortalHints,
    tuning: StitchingTuning,
) -> MergeResult {
    let mut tree: KdTree<f64, 2> = KdTree::new();
    for &node in tour_b {
        let c = coords[node];
        tree.add(&[c.x, c.y], node as u64);
    }

    let mut candidates: Vec<(f64, usize, usize)> = Vec::with_capacity(tour_a.len());

    for &u in tour_a {
        let c = coords[u];
        let nn = tree.nearest_one::<SquaredEuclidean>(&[c.x, c.y]);
        candidates.push((nn.distance, u, nn.item as usize));
    }

    // Portal candidates bias the seam search toward boundary-facing nodes.
    let portal_a = select_portal_nodes(coords, tour_a, portal_hints.from_a_target, tuning);
    let portal_b = select_portal_nodes(coords, tour_b, portal_hints.from_b_target, tuning);
    candidates.extend(portal_candidate_pairs(coords, &portal_a, &portal_b, tuning));
    candidates.sort_unstable_by(|lhs, rhs| lhs.0.total_cmp(&rhs.0));
    candidates = dedupe_candidates(candidates, tuning);

    let check_limit = tuning.candidate_pair_check_limit.min(candidates.len());

    let pos_a: HashMap<usize, usize> = tour_a.iter().enumerate().map(|(i, &n)| (n, i)).collect();
    let pos_b: HashMap<usize, usize> = tour_b.iter().enumerate().map(|(i, &n)| (n, i)).collect();

    let n_a = tour_a.len();
    let n_b = tour_b.len();

    let mut best: Option<(usize, usize, usize, usize, bool, f64)> = None;

    for &(_, u_node, v_node) in candidates.iter().take(check_limit) {
        let Some(&u_idx) = pos_a.get(&u_node) else {
            continue;
        };
        let Some(&v_idx) = pos_b.get(&v_node) else {
            continue;
        };

        let u_next_node = tour_a[(u_idx + 1) % n_a];
        let u_prev_node = tour_a[(u_idx + n_a - 1) % n_a];

        let v_next_node = tour_b[(v_idx + 1) % n_b];
        let v_prev_node = tour_b[(v_idx + n_b - 1) % n_b];

        let a_cuts = [(u_node, u_next_node), (u_prev_node, u_node)];
        let b_cuts = [(v_node, v_next_node), (v_prev_node, v_node)];

        for (a1, a2) in a_cuts {
            for (b1, b2) in b_cuts {
                let removed_cost =
                    geometry::dist(coords[a1], coords[a2]) + geometry::dist(coords[b1], coords[b2]);

                let e1 = geometry::dist(coords[a1], coords[b2]);
                let e2 = geometry::dist(coords[b1], coords[a2]);

                let penalty_fwd = bridge_penalty(e1, e2, portal_hints.non_neighbor, tuning);

                let score_fwd = (e1 + e2 + penalty_fwd) - removed_cost;

                let r1 = geometry::dist(coords[a1], coords[b1]);
                let r2 = geometry::dist(coords[b2], coords[a2]);

                let penalty_rev = bridge_penalty(r1, r2, portal_hints.non_neighbor, tuning);

                let score_rev = (r1 + r2 + penalty_rev) - removed_cost;

                if best.is_none_or(|x| score_fwd < x.5) {
                    best = Some((a1, a2, b1, b2, false, score_fwd));
                }
                if best.is_none_or(|x| score_rev < x.5) {
                    best = Some((a1, a2, b1, b2, true, score_rev));
                }
            }
        }
    }

    let Some((_a_cut_u, a_cut_v, b_cut_u, b_cut_v, flip_b, _score)) = best else {
        log::warn!("stitcher.merge: no merge candidate found, appending tours");
        let mut merged = Vec::with_capacity(tour_a.len() + tour_b.len());
        merged.extend_from_slice(tour_a);
        merged.extend_from_slice(tour_b);
        let boundary_mid = tour_a.len().saturating_sub(1);
        let boundary_end = merged.len().saturating_sub(1);
        return MergeResult {
            merged,
            boundaries: [boundary_mid, boundary_end],
        };
    };

    let a_lin = geometry::rotate_cycle(tour_a, a_cut_v);

    let b_lin = if !flip_b {
        geometry::rotate_cycle(tour_b, b_cut_v)
    } else {
        let mut rev = tour_b.to_vec();
        rev.reverse();
        geometry::rotate_cycle(&rev, b_cut_u)
    };

    let mut merged = Vec::with_capacity(a_lin.len() + b_lin.len());
    merged.extend_from_slice(&a_lin);
    merged.extend_from_slice(&b_lin);

    let boundary_mid = a_lin.len() - 1;
    let boundary_end = merged.len() - 1;

    MergeResult {
        merged,
        boundaries: [boundary_mid, boundary_end],
    }
}

fn bridge_penalty(
    bridge_a: f64,
    bridge_b: f64,
    non_neighbor: bool,
    tuning: StitchingTuning,
) -> f64 {
    let mut penalty = 0.0;
    if bridge_a > tuning.large_jump_distance_threshold
        || bridge_b > tuning.large_jump_distance_threshold
    {
        penalty += tuning.large_jump_penalty;
    }
    if bridge_a > tuning.very_large_jump_distance_threshold
        || bridge_b > tuning.very_large_jump_distance_threshold
    {
        penalty += tuning.very_large_jump_penalty;
    }
    if non_neighbor {
        penalty += tuning.non_neighbor_bridge_penalty;
    }
    penalty
}

fn portal_node_budget(tour_len: usize, tuning: StitchingTuning) -> usize {
    let dynamic = (tour_len as f64).sqrt() as usize;
    dynamic.clamp(tuning.portal_node_floor, tuning.portal_node_cap)
}

fn select_portal_nodes(
    coords: &[LKHNode],
    tour: &[usize],
    target: LKHNode,
    tuning: StitchingTuning,
) -> Vec<usize> {
    let mut ranked: Vec<(f64, usize)> = tour
        .iter()
        .map(|&node| (geometry::dist(coords[node], target), node))
        .collect();
    ranked.sort_unstable_by(|lhs, rhs| lhs.0.total_cmp(&rhs.0));
    ranked
        .into_iter()
        .take(portal_node_budget(tour.len(), tuning))
        .map(|(_, node)| node)
        .collect()
}

fn portal_candidate_pairs(
    coords: &[LKHNode],
    portal_a: &[usize],
    portal_b: &[usize],
    tuning: StitchingTuning,
) -> Vec<(f64, usize, usize)> {
    let mut pairs: Vec<(f64, usize, usize)> = Vec::with_capacity(portal_a.len() * portal_b.len());
    for &u in portal_a {
        for &v in portal_b {
            pairs.push((geometry::dist(coords[u], coords[v]), u, v));
        }
    }

    pairs.sort_unstable_by(|lhs, rhs| lhs.0.total_cmp(&rhs.0));
    pairs.truncate(tuning.portal_pair_check_limit.min(pairs.len()));
    pairs
}

fn dedupe_candidates(
    candidates: Vec<(f64, usize, usize)>,
    tuning: StitchingTuning,
) -> Vec<(f64, usize, usize)> {
    let mut out = Vec::with_capacity(candidates.len());
    let mut seen: HashSet<(usize, usize)> = HashSet::with_capacity(candidates.len());
    for candidate in candidates {
        let pair = (candidate.1, candidate.2);
        if seen.insert(pair) {
            out.push(candidate);
            if out.len() >= tuning.candidate_pair_check_limit {
                break;
            }
        }
    }
    out
}

fn tour_centroid(coords: &[LKHNode], tour: &[usize]) -> LKHNode {
    let mut sx = 0.0;
    let mut sy = 0.0;
    for &node in tour {
        let point = coords[node];
        sx += point.x;
        sy += point.y;
    }
    let n = tour.len().max(1) as f64;
    LKHNode::new(sx / n, sy / n)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use h3o::{LatLng, Resolution};

    use super::*;
    use crate::node::LKHNode;

    fn sorted(mut values: Vec<usize>) -> Vec<usize> {
        values.sort_unstable();
        values
    }

    #[test]
    fn stitch_chunk_tours_dense_empty_input_returns_empty_outputs() {
        let (merged, boundaries) = stitch_chunk_tours_dense(&[], vec![]);
        assert!(merged.is_empty());
        assert!(boundaries.is_empty());
    }

    #[test]
    fn merge_two_cycles_dense_keeps_all_nodes_exactly_once() {
        let coords = vec![
            LKHNode::new(0.0, 0.0),
            LKHNode::new(0.0, 1.0),
            LKHNode::new(1.0, 0.0),
            LKHNode::new(10.0, 10.0),
            LKHNode::new(10.0, 11.0),
            LKHNode::new(11.0, 10.0),
        ];
        let tour_a = vec![0, 1, 2];
        let tour_b = vec![3, 4, 5];

        let result = merge_two_cycles_dense(
            &coords,
            &tour_a,
            &tour_b,
            MergePortalHints {
                from_a_target: LKHNode::new(10.0, 10.0),
                from_b_target: LKHNode::new(0.0, 0.0),
                non_neighbor: false,
            },
            StitchingTuning::default(),
        );
        let merged_set: HashSet<usize> = result.merged.iter().copied().collect();

        assert_eq!(result.merged.len(), tour_a.len() + tour_b.len());
        assert_eq!(merged_set.len(), result.merged.len());
        assert_eq!(sorted(result.merged), vec![0, 1, 2, 3, 4, 5]);
        assert!(result.boundaries[0] < 6);
        assert!(result.boundaries[1] < 6);
    }

    #[test]
    fn stitch_chunk_tours_dense_records_boundary_pairs_per_merge() {
        let coords = vec![
            LKHNode::new(0.0, 0.0),
            LKHNode::new(0.0, 1.0),
            LKHNode::new(1.0, 0.0),
            LKHNode::new(10.0, 10.0),
            LKHNode::new(10.0, 11.0),
            LKHNode::new(11.0, 10.0),
            LKHNode::new(20.0, 20.0),
            LKHNode::new(20.0, 21.0),
            LKHNode::new(21.0, 20.0),
        ];

        let (merged, boundaries) =
            stitch_chunk_tours_dense(&coords, vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]]);

        assert_eq!(merged.len(), 9);
        assert_eq!(boundaries.len(), 4);
    }

    #[test]
    fn stitch_chunk_tours_dense_with_state_keeps_all_nodes_exactly_once() {
        let coords = vec![
            LKHNode::new(0.0, 0.0),
            LKHNode::new(0.0, 1.0),
            LKHNode::new(1.0, 0.0),
            LKHNode::new(10.0, 10.0),
            LKHNode::new(10.0, 11.0),
            LKHNode::new(11.0, 10.0),
        ];
        let tours = vec![vec![0, 1, 2], vec![3, 4, 5]];

        let sf = LatLng::new(37.7749, -122.4194)
            .expect("valid lat/lng")
            .to_cell(Resolution::Six);
        let la = LatLng::new(34.0522, -118.2437)
            .expect("valid lat/lng")
            .to_cell(Resolution::Six);
        let states = vec![
            ChunkStitchState {
                cell: sf,
                resolution: Resolution::Six,
                centroid: LKHNode::new(0.0, 0.0),
                size: 3,
            },
            ChunkStitchState {
                cell: la,
                resolution: Resolution::Six,
                centroid: LKHNode::new(10.0, 10.0),
                size: 3,
            },
        ];

        let (merged, boundaries) =
            stitch_chunk_tours_dense_with_state(&coords, tours, states, StitchingTuning::default());
        let merged_set: HashSet<usize> = merged.iter().copied().collect();

        assert_eq!(merged.len(), 6);
        assert_eq!(merged_set.len(), merged.len());
        assert_eq!(sorted(merged), vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(boundaries.len(), 2);
    }

    #[test]
    fn chunk_selection_score_penalizes_non_neighbors() {
        let sf = LatLng::new(37.7749, -122.4194)
            .expect("valid lat/lng")
            .to_cell(Resolution::Six);
        let nyc = LatLng::new(40.7128, -74.0060)
            .expect("valid lat/lng")
            .to_cell(Resolution::Six);

        let merged = vec![ChunkStitchState {
            cell: sf,
            resolution: Resolution::Six,
            centroid: LKHNode::new(0.0, 0.0),
            size: 3,
        }];
        let neighbor = ChunkStitchState {
            cell: sf,
            resolution: Resolution::Six,
            centroid: LKHNode::new(1.0, 0.0),
            size: 3,
        };
        let non_neighbor = ChunkStitchState {
            cell: nyc,
            resolution: Resolution::Six,
            centroid: LKHNode::new(1.0, 0.0),
            size: 3,
        };

        let neighbor_score = chunk_selection_score(neighbor, &merged, StitchingTuning::default());
        let non_neighbor_score =
            chunk_selection_score(non_neighbor, &merged, StitchingTuning::default());

        assert!(neighbor_score < non_neighbor_score);
    }

    #[test]
    fn augment_boundaries_with_long_edges_adds_spike_edges() {
        let coords = vec![
            LKHNode::new(0.0, 0.0),
            LKHNode::new(1.0, 0.0),
            LKHNode::new(2.0, 0.0),
            LKHNode::new(30.0, 0.0),
        ];
        let tour = vec![0, 1, 2, 3];
        let focused =
            augment_boundaries_with_long_edges(&coords, &tour, &[1], StitchingTuning::default());

        assert!(focused.contains(&1));
        assert!(focused.contains(&2) || focused.contains(&3));
        assert!(focused.len() >= 2);
    }

    #[test]
    fn boundary_two_opt_skips_when_boundaries_are_empty() {
        let coords = vec![
            LKHNode::new(0.0, 0.0),
            LKHNode::new(2.0, 2.0),
            LKHNode::new(0.0, 2.0),
            LKHNode::new(2.0, 0.0),
        ];
        let mut tour = vec![0, 1, 2, 3];
        let original = tour.clone();

        boundary_two_opt(&coords, &mut tour, &[], 4, 2);

        assert_eq!(tour, original);
    }

    #[test]
    fn boundary_two_opt_reduces_length_for_crossing_edges() {
        let coords = vec![
            LKHNode::new(0.0, 0.0),
            LKHNode::new(2.0, 2.0),
            LKHNode::new(0.0, 2.0),
            LKHNode::new(2.0, 0.0),
        ];
        let mut tour = vec![0, 1, 2, 3];
        let before = geometry::tour_length(&coords, &tour);

        boundary_two_opt(&coords, &mut tour, &[1], 4, 3);
        let after = geometry::tour_length(&coords, &tour);

        assert!(after < before);
    }

    #[test]
    fn spike_repair_two_opt_reduces_length_on_spike_edge() {
        let coords = vec![
            LKHNode::new(0.0, 0.0),
            LKHNode::new(1.0, 0.0),
            LKHNode::new(2.0, 0.0),
            LKHNode::new(10.0, 10.0),
            LKHNode::new(3.0, 0.0),
        ];
        let mut tour = vec![0, 1, 3, 2, 4];
        let before = geometry::tour_length(&coords, &tour);

        let tuning = StitchingTuning {
            spike_repair_top_n: 3,
            spike_repair_window: 4,
            spike_repair_passes: 3,
            ..StitchingTuning::default()
        };
        spike_repair_two_opt(&coords, &mut tour, tuning);
        let after = geometry::tour_length(&coords, &tour);

        assert!(after < before);
    }
}
