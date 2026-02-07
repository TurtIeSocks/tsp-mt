use std::collections::HashMap;
use std::time::Instant;

use kiddo::{KdTree, SquaredEuclidean};

use crate::{geometry::TourGeometry, node::LKHNode};

const LARGE_JUMP_PENALTY: f64 = 500.0;
const LARGE_JUMP_DISTANCE_THRESHOLD: f64 = 1_000.0;
const CANDIDATE_PAIR_CHECK_LIMIT: usize = 40;
const MIN_TOUR_SIZE_FOR_2OPT: usize = 4;
const TWO_OPT_IMPROVEMENT_EPSILON: f64 = 1e-5;

struct MergeResult {
    merged: Vec<usize>,
    /// Each index i refers to boundary edge (i -> i+1).
    boundaries: [usize; 2],
}

pub(crate) struct TourStitcher;

impl TourStitcher {
    pub(crate) fn stitch_chunk_tours_dense(
        coords: &[LKHNode],
        mut chunk_tours: Vec<Vec<usize>>,
    ) -> (Vec<usize>, Vec<usize>) {
        let chunk_count = chunk_tours.len();
        if chunk_count == 0 {
            log::warn!("stitcher: no chunk tours to merge");
            return (Vec::new(), Vec::new());
        }

        log::info!("stitcher: start merge chunk_tours={chunk_count}");
        let mut merged = chunk_tours.remove(0);
        let mut boundaries: Vec<usize> = Vec::new();

        for (merge_idx, t) in chunk_tours.into_iter().enumerate() {
            let res = Self::merge_two_cycles_dense(coords, &merged, &t);
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

    pub(crate) fn boundary_two_opt(
        coords: &[LKHNode],
        tour: &mut [usize],
        boundaries: &[usize],
        window: usize,
        passes: usize,
    ) {
        let now = Instant::now();
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

                        let cur_dist = TourGeometry::dist(coords[a], coords[b])
                            + TourGeometry::dist(coords[c], coords[d]);
                        let new_dist = TourGeometry::dist(coords[a], coords[c])
                            + TourGeometry::dist(coords[b], coords[d]);

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
            "stitcher.2opt: complete n={} boundaries={} passes={} swaps={} secs={:.2}",
            n,
            boundaries.len(),
            passes_executed,
            total_swaps,
            now.elapsed().as_secs_f32()
        );
    }

    /// Merges two tours by finding the strictly closest points between them.
    /// This prevents random portals from creating large outlier jumps.
    fn merge_two_cycles_dense(
        coords: &[LKHNode],
        tour_a: &[usize],
        tour_b: &[usize],
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

        candidates.sort_unstable_by(|x, y| x.0.total_cmp(&y.0));

        let check_limit = CANDIDATE_PAIR_CHECK_LIMIT.min(candidates.len());

        let pos_a: HashMap<usize, usize> =
            tour_a.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        let pos_b: HashMap<usize, usize> =
            tour_b.iter().enumerate().map(|(i, &n)| (n, i)).collect();

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
                    let removed_cost = TourGeometry::dist(coords[a1], coords[a2])
                        + TourGeometry::dist(coords[b1], coords[b2]);

                    let e1 = TourGeometry::dist(coords[a1], coords[b2]);
                    let e2 = TourGeometry::dist(coords[b1], coords[a2]);

                    let penalty_fwd = if e1 > LARGE_JUMP_DISTANCE_THRESHOLD
                        || e2 > LARGE_JUMP_DISTANCE_THRESHOLD
                    {
                        LARGE_JUMP_PENALTY
                    } else {
                        0.0
                    };

                    let score_fwd = (e1 + e2 + penalty_fwd) - removed_cost;

                    let r1 = TourGeometry::dist(coords[a1], coords[b1]);
                    let r2 = TourGeometry::dist(coords[b2], coords[a2]);

                    let penalty_rev = if r1 > LARGE_JUMP_DISTANCE_THRESHOLD
                        || r2 > LARGE_JUMP_DISTANCE_THRESHOLD
                    {
                        LARGE_JUMP_PENALTY
                    } else {
                        0.0
                    };

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

        let a_lin = TourGeometry::rotate_cycle(tour_a, a_cut_v);

        let b_lin = if !flip_b {
            TourGeometry::rotate_cycle(tour_b, b_cut_v)
        } else {
            let mut rev = tour_b.to_vec();
            rev.reverse();
            TourGeometry::rotate_cycle(&rev, b_cut_u)
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
}
