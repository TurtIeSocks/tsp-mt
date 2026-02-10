use crate::{LKHNode, h3_chunking};

use super::common::{MIN_CYCLE_POINTS, build_node_to_chunk_map};

const METRIC_SPIKE_TARGET_REMAINING: usize = 2;
const METRIC_SPIKE_REPAIR_MAX_PASSES: usize = 12;
const METRIC_SPIKE_REPAIR_EDGE_LIMIT: usize = 160;
const METRIC_SPIKE_REPAIR_SWAPS_PER_PASS: usize = 64;
const METRIC_SPIKE_REPAIR_IMPROVEMENT_EPSILON: f64 = 1e-6;
const METRIC_SPIKE_REPAIR_SPIKE_BONUS_M: f64 = 6_000.0;
const METRIC_SPIKE_REPAIR_OVERFLOW_WEIGHT: f64 = 1.0;
const METRIC_SPIKE_REPAIR_MAX_DISTANCE_DEGRADATION_M: f64 = 220.0;
const METRIC_SPIKE_REPAIR_MIN_OVERFLOW_GAIN_M: f64 = 120.0;

fn metric_spike_edges(
    points: &[LKHNode],
    tour: &[usize],
    outlier_factor: f64,
) -> (Vec<(usize, f64)>, f64, f64) {
    let n = tour.len();
    if n == 0 {
        return (Vec::new(), 0.0, 0.0);
    }

    let edge_lengths: Vec<(usize, f64)> = (0..n)
        .map(|idx| {
            let a = points[tour[idx]];
            let b = points[tour[(idx + 1) % n]];
            (idx, a.dist(&b))
        })
        .collect();
    let total = edge_lengths.iter().map(|(_, len)| *len).sum::<f64>();
    let average = total / (n as f64);
    let threshold = average * outlier_factor;
    let mut spikes: Vec<(usize, f64)> = edge_lengths
        .into_iter()
        .filter(|(_, len)| *len > threshold)
        .collect();
    spikes.sort_unstable_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0)));

    (spikes, average, threshold)
}

#[inline]
fn overflow_above_threshold(edge_len: f64, threshold: f64) -> f64 {
    (edge_len - threshold).max(0.0)
}

pub(super) fn log_metric_spike_breakdown(
    points: &[LKHNode],
    tour: &[usize],
    outlier_factor: f64,
    chunk_partitions: &[h3_chunking::ChunkPartition],
    label: &str,
) {
    let n = tour.len();
    if n == 0 {
        return;
    }

    let (spikes, avg, threshold) = metric_spike_edges(points, tour, outlier_factor);
    if spikes.is_empty() {
        return;
    }

    let node_to_chunk = build_node_to_chunk_map(points.len(), chunk_partitions);
    let mut seam_count = 0usize;
    let mut intra_count = 0usize;

    for (rank, (edge_idx, edge_len)) in spikes.iter().enumerate() {
        let a = tour[*edge_idx % n];
        let b = tour[(*edge_idx + 1) % n];
        let chunk_a = node_to_chunk.get(a).copied().unwrap_or(usize::MAX);
        let chunk_b = node_to_chunk.get(b).copied().unwrap_or(usize::MAX);
        let seam = chunk_a != usize::MAX && chunk_b != usize::MAX && chunk_a != chunk_b;
        if seam {
            seam_count += 1;
        } else {
            intra_count += 1;
        }

        log::info!(
            "spikes.{label}: rank={} edge_pos={} len_m={:.0} seam={} from_node={} from_chunk={} to_node={} to_chunk={}",
            rank + 1,
            edge_idx,
            edge_len,
            seam,
            a,
            chunk_a,
            b,
            chunk_b
        );
    }

    log::info!(
        "spikes.{label}: count={} seam={} intra={} avg_m={:.0} threshold_m={:.0}",
        spikes.len(),
        seam_count,
        intra_count,
        avg,
        threshold
    );
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

fn try_global_metric_spike_two_opt(
    points: &[LKHNode],
    tour: &mut [usize],
    edge_idx: usize,
    threshold: f64,
) -> bool {
    let n = tour.len();
    if n < MIN_CYCLE_POINTS + 1 {
        return false;
    }

    let i = edge_idx % n;
    let i_next = (i + 1) % n;
    let a = tour[i];
    let b = tour[i_next];
    let ab = points[a].dist(&points[b]);

    let mut best_k: Option<usize> = None;
    let mut best_composite_gain = 0.0;
    let mut best_gain = 0.0;
    let mut best_worst_new_edge = f64::INFINITY;

    for k in 0..n {
        let k_next = (k + 1) % n;
        if k == i || k == i_next || k_next == i || k_next == i_next {
            continue;
        }

        let c = tour[k];
        let d = tour[k_next];
        let cd = points[c].dist(&points[d]);
        let ac = points[a].dist(&points[c]);
        let bd = points[b].dist(&points[d]);
        let distance_gain = (ab + cd) - (ac + bd);
        let before_spikes = usize::from(ab > threshold) + usize::from(cd > threshold);
        let after_spikes = usize::from(ac > threshold) + usize::from(bd > threshold);
        let spike_gain = before_spikes.saturating_sub(after_spikes) as f64;
        let before_overflow =
            overflow_above_threshold(ab, threshold) + overflow_above_threshold(cd, threshold);
        let after_overflow =
            overflow_above_threshold(ac, threshold) + overflow_above_threshold(bd, threshold);
        let overflow_gain = before_overflow - after_overflow;

        let composite_gain = if spike_gain > 0.0 {
            distance_gain
                + (spike_gain * METRIC_SPIKE_REPAIR_SPIKE_BONUS_M)
                + (overflow_gain * METRIC_SPIKE_REPAIR_OVERFLOW_WEIGHT)
        } else if distance_gain > METRIC_SPIKE_REPAIR_IMPROVEMENT_EPSILON {
            distance_gain + (overflow_gain * METRIC_SPIKE_REPAIR_OVERFLOW_WEIGHT)
        } else if overflow_gain >= METRIC_SPIKE_REPAIR_MIN_OVERFLOW_GAIN_M
            && distance_gain >= -METRIC_SPIKE_REPAIR_MAX_DISTANCE_DEGRADATION_M
        {
            (overflow_gain * METRIC_SPIKE_REPAIR_OVERFLOW_WEIGHT) + distance_gain
        } else {
            f64::NEG_INFINITY
        };

        if composite_gain > METRIC_SPIKE_REPAIR_IMPROVEMENT_EPSILON {
            let worst_new_edge = ac.max(bd);
            let is_better = if composite_gain > best_composite_gain {
                true
            } else if (composite_gain - best_composite_gain).abs()
                <= METRIC_SPIKE_REPAIR_IMPROVEMENT_EPSILON
            {
                if distance_gain > best_gain {
                    true
                } else if (distance_gain - best_gain).abs()
                    <= METRIC_SPIKE_REPAIR_IMPROVEMENT_EPSILON
                {
                    if worst_new_edge < best_worst_new_edge {
                        true
                    } else if (worst_new_edge - best_worst_new_edge).abs()
                        <= METRIC_SPIKE_REPAIR_IMPROVEMENT_EPSILON
                    {
                        best_k.is_none_or(|best| k < best)
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            };
            if is_better {
                best_composite_gain = composite_gain;
                best_gain = distance_gain;
                best_worst_new_edge = worst_new_edge;
                best_k = Some(k);
            }
        }
    }

    let Some(best_k) = best_k else {
        return false;
    };

    reverse_cyclic_segment(tour, i_next, best_k);
    true
}

fn move_node_within_cycle(tour: &mut [usize], from_pos: usize, to_pos_after: usize) {
    if tour.is_empty() {
        return;
    }
    if from_pos == to_pos_after || (from_pos + tour.len() - 1) % tour.len() == to_pos_after {
        return;
    }

    if from_pos < to_pos_after {
        tour[from_pos..=to_pos_after].rotate_left(1);
    } else {
        tour[(to_pos_after + 1)..=from_pos].rotate_right(1);
    }
}

fn try_metric_spike_node_relocation(
    points: &[LKHNode],
    tour: &mut [usize],
    edge_idx: usize,
    threshold: f64,
) -> bool {
    let n = tour.len();
    if n < MIN_CYCLE_POINTS + 2 {
        return false;
    }

    let i = edge_idx % n;
    let candidates = [i, (i + 1) % n];
    let mut best_move: Option<(usize, usize, f64, f64, f64)> = None;

    for from_pos in candidates {
        let prev_idx = (from_pos + n - 1) % n;
        let next_idx = (from_pos + 1) % n;
        let node = tour[from_pos];
        let prev_node = tour[prev_idx];
        let next_node = tour[next_idx];
        let prev_to_node = points[prev_node].dist(&points[node]);
        let node_to_next = points[node].dist(&points[next_node]);
        let prev_to_next = points[prev_node].dist(&points[next_node]);
        let removal_gain = (prev_to_node + node_to_next) - prev_to_next;

        for to_pos_after in 0..n {
            let to_next = (to_pos_after + 1) % n;
            if to_pos_after == from_pos
                || to_next == from_pos
                || to_pos_after == prev_idx
                || to_next == prev_idx
            {
                continue;
            }

            let left = tour[to_pos_after];
            let right = tour[to_next];
            let left_node = points[left];
            let right_node = points[right];
            let node_value = points[node];

            let left_to_node = left_node.dist(&node_value);
            let node_to_right = node_value.dist(&right_node);
            let left_to_right = left_node.dist(&right_node);
            let insertion_cost = (left_to_node + node_to_right) - left_to_right;
            let distance_gain = removal_gain - insertion_cost;
            let before_spikes = usize::from(prev_to_node > threshold)
                + usize::from(node_to_next > threshold)
                + usize::from(left_to_right > threshold);
            let after_spikes = usize::from(prev_to_next > threshold)
                + usize::from(left_to_node > threshold)
                + usize::from(node_to_right > threshold);
            let spike_gain = before_spikes.saturating_sub(after_spikes) as f64;
            let before_overflow = overflow_above_threshold(prev_to_node, threshold)
                + overflow_above_threshold(node_to_next, threshold)
                + overflow_above_threshold(left_to_right, threshold);
            let after_overflow = overflow_above_threshold(prev_to_next, threshold)
                + overflow_above_threshold(left_to_node, threshold)
                + overflow_above_threshold(node_to_right, threshold);
            let overflow_gain = before_overflow - after_overflow;
            let composite_gain = if spike_gain > 0.0 {
                distance_gain
                    + (spike_gain * METRIC_SPIKE_REPAIR_SPIKE_BONUS_M)
                    + (overflow_gain * METRIC_SPIKE_REPAIR_OVERFLOW_WEIGHT)
            } else if distance_gain > METRIC_SPIKE_REPAIR_IMPROVEMENT_EPSILON {
                distance_gain + (overflow_gain * METRIC_SPIKE_REPAIR_OVERFLOW_WEIGHT)
            } else if overflow_gain >= METRIC_SPIKE_REPAIR_MIN_OVERFLOW_GAIN_M
                && distance_gain >= -METRIC_SPIKE_REPAIR_MAX_DISTANCE_DEGRADATION_M
            {
                (overflow_gain * METRIC_SPIKE_REPAIR_OVERFLOW_WEIGHT) + distance_gain
            } else {
                f64::NEG_INFINITY
            };
            if composite_gain <= METRIC_SPIKE_REPAIR_IMPROVEMENT_EPSILON {
                continue;
            }

            let worst_new_edge = left_to_node.max(node_to_right);
            let is_better =
                if let Some((best_from, best_to, best_composite, best_distance, best_worst_edge)) =
                    best_move
                {
                    if composite_gain > best_composite {
                        true
                    } else if (composite_gain - best_composite).abs()
                        <= METRIC_SPIKE_REPAIR_IMPROVEMENT_EPSILON
                    {
                        if distance_gain > best_distance {
                            true
                        } else if (distance_gain - best_distance).abs()
                            <= METRIC_SPIKE_REPAIR_IMPROVEMENT_EPSILON
                        {
                            if worst_new_edge < best_worst_edge {
                                true
                            } else if (worst_new_edge - best_worst_edge).abs()
                                <= METRIC_SPIKE_REPAIR_IMPROVEMENT_EPSILON
                            {
                                from_pos < best_from
                                    || (from_pos == best_from && to_pos_after < best_to)
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    true
                };

            if is_better {
                best_move = Some((
                    from_pos,
                    to_pos_after,
                    composite_gain,
                    distance_gain,
                    worst_new_edge,
                ));
            }
        }
    }

    let Some((from_pos, to_pos_after, _, _, _)) = best_move else {
        return false;
    };
    move_node_within_cycle(tour, from_pos, to_pos_after);
    true
}

pub(super) fn repair_metric_spikes_with_global_two_opt(
    points: &[LKHNode],
    tour: &mut [usize],
    outlier_factor: f64,
) {
    let n = tour.len();
    if n < MIN_CYCLE_POINTS + 1 {
        return;
    }

    let (initial_spikes, initial_avg, initial_threshold) =
        metric_spike_edges(points, tour, outlier_factor);
    if initial_spikes.len() <= METRIC_SPIKE_TARGET_REMAINING {
        log::info!(
            "stitcher.metric_spike_repair: skip spikes={} target={} threshold_m={:.0}",
            initial_spikes.len(),
            METRIC_SPIKE_TARGET_REMAINING,
            initial_threshold
        );
        return;
    }

    log::info!(
        "stitcher.metric_spike_repair: start spikes={} avg_m={:.0} threshold_m={:.0}",
        initial_spikes.len(),
        initial_avg,
        initial_threshold
    );

    let mut total_swaps = 0usize;
    let mut passes_executed = 0usize;
    for pass_idx in 0..METRIC_SPIKE_REPAIR_MAX_PASSES {
        passes_executed = pass_idx + 1;
        let mut pass_swaps = 0usize;

        loop {
            let (spikes, _, threshold) = metric_spike_edges(points, tour, outlier_factor);
            if spikes.len() <= METRIC_SPIKE_TARGET_REMAINING {
                break;
            }

            let mut improved = false;
            for (edge_idx, _) in spikes.into_iter().take(METRIC_SPIKE_REPAIR_EDGE_LIMIT) {
                if try_global_metric_spike_two_opt(points, tour, edge_idx, threshold)
                    || try_metric_spike_node_relocation(points, tour, edge_idx, threshold)
                {
                    improved = true;
                    pass_swaps += 1;
                    total_swaps += 1;
                    break;
                }
            }

            if !improved || pass_swaps >= METRIC_SPIKE_REPAIR_SWAPS_PER_PASS {
                break;
            }
        }

        log::debug!(
            "stitcher.metric_spike_repair: pass={} swaps={}",
            pass_idx + 1,
            pass_swaps
        );
        if pass_swaps == 0 {
            break;
        }
    }

    let (final_spikes, final_avg, final_threshold) =
        metric_spike_edges(points, tour, outlier_factor);
    log::info!(
        "stitcher.metric_spike_repair: complete passes={} swaps={} spikes={} avg_m={:.0} threshold_m={:.0}",
        passes_executed,
        total_swaps,
        final_spikes.len(),
        final_avg,
        final_threshold
    );
}

#[cfg(test)]
mod tests {
    use super::super::common::cycle_length;
    use super::{try_global_metric_spike_two_opt, try_metric_spike_node_relocation};
    use crate::node::LKHNode;

    #[test]
    fn try_global_metric_spike_two_opt_improves_crossing_square() {
        let points = vec![
            LKHNode::from_lat_lng(0.0, 0.0),
            LKHNode::from_lat_lng(0.0, 1.0),
            LKHNode::from_lat_lng(1.0, 0.0),
            LKHNode::from_lat_lng(1.0, 1.0),
        ];
        let mut tour = vec![0, 1, 2, 3];
        let before = cycle_length(&points, &tour);

        let improved = try_global_metric_spike_two_opt(&points, &mut tour, 1, 130_000.0);
        let after = cycle_length(&points, &tour);

        assert!(improved);
        assert!(after < before);
    }

    #[test]
    fn try_metric_spike_node_relocation_moves_misplaced_endpoint() {
        let points = vec![
            LKHNode::from_lat_lng(0.0, 0.000),
            LKHNode::from_lat_lng(0.0, 0.001),
            LKHNode::from_lat_lng(0.0, 1.000),
            LKHNode::from_lat_lng(0.0, 0.002),
            LKHNode::from_lat_lng(0.0, 1.001),
        ];
        let mut tour = vec![0, 1, 2, 3, 4];
        let before = cycle_length(&points, &tour);

        let improved = try_metric_spike_node_relocation(&points, &mut tour, 1, 120_000.0);
        let after = cycle_length(&points, &tour);

        assert!(improved);
        assert!(after < before);
    }
}
