use std::time::Instant;

use crate::{candidate::CandidateSet, distance::euc_2d, problem::Problem, tour::Tour};

/// Or-opt sweep — relocate segments of length 1, 2, or 3 to a better
/// position. This is the constrained 3-opt move popularized by Or
/// (1976); LKH covers the same ground under its general `BestKOptMove`
/// search but Or-opt alone is sufficient to escape most 2-opt local
/// optima.
///
/// Returns the total integer gain applied during the sweep.
pub fn sweep(
    problem: &Problem,
    candidates: &CandidateSet,
    tour: &mut Tour,
    dont_look: &mut [bool],
    deadline: Instant,
) -> i64 {
    let n = problem.n();
    if n < 5 {
        return 0;
    }
    let coords = problem.coords();
    let mut total_gain: i64 = 0;

    const DEADLINE_CHECK_STRIDE: usize = 256;
    for seg_len in [1usize, 2, 3] {
        let mut idx: usize = 0;
        let mut visited = 0usize;
        while visited < n {
            if visited % DEADLINE_CHECK_STRIDE == 0 && Instant::now() >= deadline {
                return total_gain;
            }
            let s = idx as u32;
            idx = (idx + 1) % n;
            visited += 1;

            if dont_look[s as usize] {
                continue;
            }

            if let Some((gain, s_node, t_node, prev_s, next_t, ins_left, ins_right)) =
                try_relocate(tour, candidates, coords, s, seg_len, n)
            {
                for node in [s_node, t_node, prev_s, next_t, ins_left, ins_right] {
                    dont_look[node as usize] = false;
                }
                total_gain += gain;
            }
        }
    }

    total_gain
}

fn try_relocate(
    tour: &mut Tour,
    candidates: &CandidateSet,
    coords: &[crate::coord::Point2D],
    s: u32,
    seg_len: usize,
    n: usize,
) -> Option<(i64, u32, u32, u32, u32, u32, u32)> {
    let p_s = tour.position_of(s);
    let p_t = (p_s + seg_len - 1) % n;
    let t = tour.node_at(p_t);
    let prev_s = tour.prev(s);
    let next_t = tour.next(t);
    if prev_s == t || next_t == s {
        return None;
    }

    let d_remove = euc_2d(coords[prev_s as usize], coords[s as usize])
        + euc_2d(coords[t as usize], coords[next_t as usize])
        - euc_2d(coords[prev_s as usize], coords[next_t as usize]);

    // Insertion targets: pick from candidates of both endpoints — common
    // LKH-style optimization that catches good positions either side of
    // the segment finds reachable.
    let mut best: Option<(usize, i64, bool)> = None;
    for endpoint_candidates in [candidates.of(s), candidates.of(t)] {
        for c in endpoint_candidates {
            if c.to == s || c.to == t || c.to == prev_s || c.to == next_t {
                continue;
            }
            // Skip nodes that are inside the segment being relocated.
            let pc = tour.position_of(c.to);
            if pos_in_segment(pc, p_s, seg_len, n) {
                continue;
            }
            // Try inserting after c (segment goes c -> s ... t -> next(c)).
            for reversed in [false, true] {
                let (seg_head, seg_tail) = if reversed { (t, s) } else { (s, t) };
                let nc = tour.next(c.to);
                if nc == s || nc == t {
                    continue;
                }
                let d_insert = euc_2d(coords[c.to as usize], coords[seg_head as usize])
                    + euc_2d(coords[seg_tail as usize], coords[nc as usize])
                    - euc_2d(coords[c.to as usize], coords[nc as usize]);
                let gain = d_remove - d_insert;
                if gain <= 0 {
                    continue;
                }
                let is_better = match best {
                    None => true,
                    Some((_, best_gain, _)) => gain > best_gain,
                };
                if is_better {
                    best = Some((pc, gain, reversed));
                }
            }
        }
    }

    let (insert_after_pos, gain, reversed) = best?;
    let ins_left = tour.node_at(insert_after_pos);
    let ins_right = tour.node_at((insert_after_pos + 1) % n);
    apply_relocate(tour, p_s, seg_len, insert_after_pos, reversed, n);
    Some((gain, s, t, prev_s, next_t, ins_left, ins_right))
}

fn pos_in_segment(pos: usize, start: usize, len: usize, n: usize) -> bool {
    for k in 0..len {
        if (start + k) % n == pos {
            return true;
        }
    }
    false
}

fn apply_relocate(
    tour: &mut Tour,
    seg_start: usize,
    seg_len: usize,
    insert_after_pos: usize,
    reversed: bool,
    n: usize,
) {
    // Extract segment nodes, build a fresh order with the segment removed
    // then re-inserted after `insert_after_pos`. O(n) per move — fine for
    // the chunk sizes we target.
    let segment: Vec<u32> = (0..seg_len)
        .map(|k| tour.node_at((seg_start + k) % n))
        .collect();
    let segment_iter: Vec<u32> = if reversed {
        segment.into_iter().rev().collect()
    } else {
        segment
    };

    let insert_after_node = tour.node_at(insert_after_pos);
    let old_order: Vec<u32> = tour.as_slice().to_vec();
    let mut new_order: Vec<u32> = Vec::with_capacity(n);
    let mut in_segment = vec![false; n];
    for &node in &segment_iter {
        in_segment[node as usize] = true;
    }

    for &node in &old_order {
        if in_segment[node as usize] {
            continue;
        }
        new_order.push(node);
        if node == insert_after_node {
            new_order.extend_from_slice(&segment_iter);
        }
    }

    debug_assert_eq!(new_order.len(), n);
    *tour = Tour::from_order(&new_order);
}
