//! Sequential 3-opt move (LKH `Best3OptMove` Case 1).
//!
//! Implements the canonical Lin-Kernighan k=3 sequential exchange.
//! Picks an anchor `t1`, breaks the edge to its tour neighbour `t2`,
//! and attempts to extend a chain
//!
//! ```text
//!     t1 — t2  →  t2 — t3  →  t3 — t4  →  t4 — t5  →  t5 — t6  →  t6 — t1
//! ```
//!
//! with `t3 ∈ candidates(t2)`, `t4 = prev(t3)`, `t5 ∈ candidates(t4)`,
//! `t6 = next(t5)`, and the feasibility constraint that `t5` lies
//! between `t2` and `t4` in the current tour orientation. This is the
//! LKH "Case 1" sub-case of `Best3OptMove`; the remaining cases (2, 5,
//! 6) extend the same machinery to the other choices of `t4` / `t6`.
//!
//! Applying the move on a position-indexed tour is:
//!
//! * keep `tour[..=p1]` (segment S0, ending at `t1`)
//! * splice `tour[p6..=p4]` next (segment B, from `t6` to `t4`)
//! * then `tour[p2..=p5]` *reversed* (segment A, originally `t2..t5`)
//! * then `tour[p3..]` (segment S_C, starting at `t3`)
//!
//! Gain criterion (LKH `GainCriterionUsed`): each partial sum
//! `G_i = G_{i-1} + cost_removed - cost_added` must stay positive.
//! Otherwise we prune; this keeps the search bounded.

use std::time::Instant;

use crate::{candidate::CandidateSet, distance::euc_2d, problem::Problem, tour::Tour};

/// 3-opt sweep. Returns total gain applied during the sweep.
pub fn sweep(
    problem: &Problem,
    candidates: &CandidateSet,
    tour: &mut Tour,
    dont_look: &mut [bool],
    deadline: Instant,
) -> i64 {
    let n = problem.n();
    if n < 6 {
        return 0;
    }
    let coords = problem.coords();
    let mut total_gain: i64 = 0;

    const DEADLINE_CHECK_STRIDE: usize = 256;
    let mut idx: usize = 0;
    let mut visited = 0usize;
    let mut moves_applied = 0usize;
    while visited < n {
        if visited % DEADLINE_CHECK_STRIDE == 0 && Instant::now() >= deadline {
            break;
        }
        let t1 = idx as u32;
        idx = (idx + 1) % n;
        visited += 1;

        if dont_look[t1 as usize] {
            continue;
        }

        if let Some((gain, touched)) = try_at(tour, candidates, coords, t1) {
            for node in touched {
                dont_look[node as usize] = false;
            }
            total_gain += gain;
            moves_applied += 1;
        }
    }

    if moves_applied > 0 {
        log::debug!(
            "lin_kernighan.three_opt: applied {} move(s), gain={}",
            moves_applied,
            total_gain
        );
    }

    total_gain
}

fn try_at(
    tour: &mut Tour,
    candidates: &CandidateSet,
    coords: &[crate::coord::Point2D],
    t1: u32,
) -> Option<(i64, [u32; 6])> {
    let n = tour.n();
    // Try both rotational directions: we treat (t1, next(t1)) and
    // (t1, prev(t1)) as the candidate broken edge by temporarily
    // reversing the tour-side function callers see. In LKH this is the
    // `Reversed` flag; here we just inline both sides.
    for &forward in &[true, false] {
        if let Some(result) = try_at_side(tour, candidates, coords, t1, forward, n) {
            return Some(result);
        }
    }
    None
}

fn try_at_side(
    tour: &mut Tour,
    candidates: &CandidateSet,
    coords: &[crate::coord::Point2D],
    t1: u32,
    forward: bool,
    n: usize,
) -> Option<(i64, [u32; 6])> {
    let t2 = if forward { tour.next(t1) } else { tour.prev(t1) };
    let c_t1 = coords[t1 as usize];
    let c_t2 = coords[t2 as usize];
    let d_t1_t2 = euc_2d(c_t1, c_t2);

    let mut best: Option<(i64, [u32; 6])> = None;
    for cand_t3 in candidates.of(t2) {
        let t3 = cand_t3.to;
        if t3 == t1 || t3 == t2 {
            continue;
        }
        let d_t2_t3 = cand_t3.cost;
        // 3-opt explores beyond the 2-opt frontier: don't gate on a
        // positive partial gain at depth 1. Only the closing gain at
        // depth 3 has to be net-positive. (LKH calls this
        // `GainCriterionUsed = NO` at this level.)
        let g1 = d_t1_t2 - d_t2_t3;

        let t4 = if forward { tour.prev(t3) } else { tour.next(t3) };
        if t4 == t1 || t4 == t2 {
            continue;
        }
        let d_t3_t4 = euc_2d(coords[t3 as usize], coords[t4 as usize]);
        let g2 = g1 + d_t3_t4;

        for cand_t5 in candidates.of(t4) {
            let t5 = cand_t5.to;
            if t5 == t1 || t5 == t2 || t5 == t3 || t5 == t4 {
                continue;
            }
            let d_t4_t5 = cand_t5.cost;
            let g3 = g2 - d_t4_t5;
            if g3 <= 0 {
                continue;
            }
            if !between(tour, t2, t5, t4, forward) {
                continue;
            }
            let t6 = if forward { tour.next(t5) } else { tour.prev(t5) };
            if t6 == t1 || t6 == t2 || t6 == t3 || t6 == t4 {
                continue;
            }
            let d_t5_t6 = euc_2d(coords[t5 as usize], coords[t6 as usize]);
            let d_t6_t1 = euc_2d(coords[t6 as usize], coords[t1 as usize]);
            let gain = g3 + d_t5_t6 - d_t6_t1;
            if gain <= 0 {
                continue;
            }

            if best.map(|(g, _)| gain > g).unwrap_or(true) {
                best = Some((gain, [t1, t2, t3, t4, t5, t6]));
            }
        }
    }

    let (gain, nodes) = best?;
    if apply_case1(tour, nodes, forward, n) {
        Some((gain, nodes))
    } else {
        None
    }
}

/// Tour-direction-aware `BETWEEN`: returns true iff `mid` lies strictly
/// between `from` and `to` along the directed tour, where direction is
/// "forward" = follow `next` from `from`, "backward" = follow `prev`.
fn between(tour: &Tour, from: u32, mid: u32, to: u32, forward: bool) -> bool {
    let n = tour.n();
    let p_from = tour.position_of(from);
    let p_mid = tour.position_of(mid);
    let p_to = tour.position_of(to);
    if p_mid == p_from || p_mid == p_to {
        return false;
    }
    if forward {
        let span_to = (p_to + n - p_from) % n;
        let span_mid = (p_mid + n - p_from) % n;
        span_mid < span_to
    } else {
        let span_to = (p_from + n - p_to) % n;
        let span_mid = (p_from + n - p_mid) % n;
        span_mid < span_to
    }
}

/// Apply LKH Case 1: keep S0, splice B (`t6..t4`) followed by reversed
/// A (`t2..t5`), then continue with S_C from `t3`.
///
/// Returns `true` iff the tour was rewritten. Returns `false` when the
/// positional invariant doesn't hold (e.g. the tour orientation chosen
/// at the search phase didn't survive into the apply phase).
fn apply_case1(tour: &mut Tour, nodes: [u32; 6], forward: bool, n: usize) -> bool {
    let [t1, t2, t3, t4, t5, t6] = nodes;
    let mut order: Vec<u32> = if forward {
        tour.as_slice().to_vec()
    } else {
        let mut v = tour.as_slice().to_vec();
        v.reverse();
        v
    };
    let position_of = |order: &[u32], node: u32| -> usize {
        order.iter().position(|&x| x == node).expect("node in tour")
    };
    let p1 = position_of(&order, t1);

    order.rotate_left(p1);
    let p1 = 0usize;
    let p2 = 1usize;
    let p_t3 = position_of(&order, t3);
    let p_t4 = (p_t3 + n - 1) % n;
    let p_t5 = position_of(&order, t5);
    let p_t6 = (p_t5 + 1) % n;

    if order[p2] != t2 || order[p_t4] != t4 || order[p_t6] != t6 {
        return false;
    }
    if !(p2 <= p_t5 && p_t5 < p_t6 && p_t6 <= p_t4 && p_t4 < p_t3) {
        return false;
    }

    let mut new_order: Vec<u32> = Vec::with_capacity(n);
    new_order.push(order[p1]);
    new_order.extend_from_slice(&order[p_t6..=p_t4]);
    new_order.extend(order[p2..=p_t5].iter().rev().copied());
    new_order.extend_from_slice(&order[p_t3..]);
    debug_assert_eq!(new_order.len(), n);

    *tour = Tour::from_order(&new_order);
    true
}

