//! Sequential 3-opt move (LKH `Best3OptMove`).
//!
//! Implements all four sub-cases LKH enumerates via `(X4, Case6)`:
//!
//! | X4 | Case6 | t4         | t6         | Constraint                     |
//! |----|-------|------------|------------|--------------------------------|
//! | 1  | 1     | PRED(t3)   | SUC(t5)    | BETWEEN(t2, t5, t4) is true    |
//! | 1  | 2     | PRED(t3)   | PRED(t5)   | BETWEEN(t2, t5, t4) is false   |
//! | 2  | 5     | SUC(t3)    | SUC(t5)    | BETWEEN(t2, t5, t3) is true    |
//! | 2  | 6     | SUC(t3)    | PRED(t5)   | BETWEEN(t2, t5, t3) is true    |
//!
//! All four remove the edges `(t1,t2)`, `(t3,t4)`, `(t5,t6)` and add
//! `(t2,t3)`, `(t4,t5)`, `(t6,t1)`. The tour layout for the apply
//! step differs per case because the relative arc each `t_i` sits on
//! changes.
//!
//! The gain criterion is only enforced at the deepest level: a 3-opt
//! is accepted iff the closing gain after all three edge swaps is
//! strictly positive. This matches LKH's `GainCriterionUsed = NO` at
//! the intermediate depths — important because the 2-opt sweep has
//! already mopped up every move with a positive depth-1 partial gain.

use std::time::Instant;

use crate::{candidate::CandidateSet, distance::euc_2d, problem::Problem, tour::Tour};

#[derive(Clone, Copy, Debug)]
enum Case {
    /// X4=1, Case6=1
    One,
    /// X4=1, Case6=2
    Two,
    /// X4=2, Case6=5
    Five,
    /// X4=2, Case6=6
    Six,
}

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
    for &forward in &[true, false] {
        if let Some(result) = try_at_side(tour, candidates, coords, t1, forward) {
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
) -> Option<(i64, [u32; 6])> {
    let t2 = if forward { tour.next(t1) } else { tour.prev(t1) };
    let c_t1 = coords[t1 as usize];
    let c_t2 = coords[t2 as usize];
    let d_t1_t2 = euc_2d(c_t1, c_t2);

    let mut best: Option<(i64, [u32; 6], Case)> = None;
    for cand_t3 in candidates.of(t2) {
        let t3 = cand_t3.to;
        if t3 == t1 || t3 == t2 {
            continue;
        }
        let d_t2_t3 = cand_t3.cost;
        let g1 = d_t1_t2 - d_t2_t3;

        // Enumerate (X4, Case6) pairs. X4=1 picks t4 = PRED(t3),
        // X4=2 picks t4 = SUC(t3). Both choices are considered
        // independently; LKH's enumeration order is X4=1 first.
        for &x4 in &[1u8, 2u8] {
            let t4 = match x4 {
                1 => {
                    if forward {
                        tour.prev(t3)
                    } else {
                        tour.next(t3)
                    }
                }
                _ => {
                    if forward {
                        tour.next(t3)
                    } else {
                        tour.prev(t3)
                    }
                }
            };
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

                // Per-case feasibility + t6 selection.
                let case_options: [(Case, bool); 2] = match x4 {
                    1 => {
                        let in_arc = between(tour, t2, t5, t4, forward);
                        [(Case::One, in_arc), (Case::Two, !in_arc)]
                    }
                    _ => {
                        let in_arc = between(tour, t2, t5, t3, forward);
                        [(Case::Five, in_arc), (Case::Six, in_arc)]
                    }
                };

                for (case, feasible) in case_options {
                    if !feasible {
                        continue;
                    }
                    let t6 = match case {
                        Case::One | Case::Five => {
                            if forward {
                                tour.next(t5)
                            } else {
                                tour.prev(t5)
                            }
                        }
                        Case::Two | Case::Six => {
                            if forward {
                                tour.prev(t5)
                            } else {
                                tour.next(t5)
                            }
                        }
                    };
                    if t6 == t1 || t6 == t2 || t6 == t3 || t6 == t4 || t6 == t5 {
                        continue;
                    }
                    let d_t5_t6 = euc_2d(coords[t5 as usize], coords[t6 as usize]);
                    let d_t6_t1 = euc_2d(coords[t6 as usize], coords[t1 as usize]);
                    let gain = g3 + d_t5_t6 - d_t6_t1;
                    if gain <= 0 {
                        continue;
                    }
                    if best.map(|(g, _, _)| gain > g).unwrap_or(true) {
                        best = Some((gain, [t1, t2, t3, t4, t5, t6], case));
                    }
                }
            }
        }
    }

    let (gain, nodes, case) = best?;
    if apply_case(tour, nodes, case, forward) {
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

fn apply_case(tour: &mut Tour, nodes: [u32; 6], case: Case, forward: bool) -> bool {
    let n = tour.n();
    // Materialise the tour in the chosen direction so the case-specific
    // slice arithmetic can assume "forward = follow indices ascending"
    // unconditionally. We then rotate so `t1` sits at index 0.
    let mut order: Vec<u32> = if forward {
        tour.as_slice().to_vec()
    } else {
        let mut v = tour.as_slice().to_vec();
        v.reverse();
        v
    };

    let [t1, t2, t3, t4, t5, t6] = nodes;
    let position_of = |order: &[u32], node: u32| -> usize {
        order.iter().position(|&x| x == node).expect("node in tour")
    };
    let p1 = position_of(&order, t1);
    order.rotate_left(p1);
    // After the rotate, t1 is at index 0 and the tour is read
    // left-to-right in the chosen direction.

    let p2 = 1usize;
    let p_t3 = position_of(&order, t3);
    let p_t4 = position_of(&order, t4);
    let p_t5 = position_of(&order, t5);
    let p_t6 = position_of(&order, t6);

    // Sanity: t2 must always be at index 1 (it's the successor of t1
    // in the rotation direction). If not, something invalidated our
    // search assumptions — bail.
    if order[p2] != t2 {
        return false;
    }

    let new_order = match case {
        Case::One => apply_case_one(&order, p2, p_t3, p_t4, p_t5, p_t6, n),
        Case::Two => apply_case_two(&order, p2, p_t3, p_t4, p_t5, p_t6, n),
        Case::Five => apply_case_five(&order, p2, p_t3, p_t4, p_t5, p_t6, n),
        Case::Six => apply_case_six(&order, p2, p_t3, p_t4, p_t5, p_t6, n),
    };
    let Some(new_order) = new_order else {
        return false;
    };
    if new_order.len() != n {
        return false;
    }
    // Validate Hamiltonian-ness — the slice arithmetic per case is
    // fiddly, and a miscount silently produces a permutation with
    // duplicates / missing nodes that Tour::from_order accepts but
    // later sweeps compute garbage from. Cheap O(n) sanity guard.
    let mut seen = vec![false; n];
    for &node in &new_order {
        let id = node as usize;
        if id >= n || seen[id] {
            return false;
        }
        seen[id] = true;
    }
    *tour = Tour::from_order(&new_order);
    true
}

/// Case 1: X4=1 (t4=PRED(t3)), t6=SUC(t5), t5 between t2 and t4.
/// Layout: `[t1] + tour[p6..=p4] + reverse(tour[p2..=p5]) + tour[p3..]`.
fn apply_case_one(
    order: &[u32],
    p2: usize,
    p_t3: usize,
    p_t4: usize,
    p_t5: usize,
    p_t6: usize,
    n: usize,
) -> Option<Vec<u32>> {
    if !(p2 <= p_t5 && p_t5 < p_t6 && p_t6 <= p_t4 && p_t4 < p_t3) {
        return None;
    }
    let mut out = Vec::with_capacity(n);
    out.push(order[0]);
    out.extend_from_slice(&order[p_t6..=p_t4]);
    out.extend(order[p2..=p_t5].iter().rev().copied());
    out.extend_from_slice(&order[p_t3..]);
    Some(out)
}

/// Case 2: X4=1 (t4=PRED(t3)), t6=PRED(t5), t5 NOT between t2 and t4.
/// Tour shape: `t1, t2, ..., t4, t3, ..., t6, t5, ..., t_last, [t1]`.
/// Layout: `[t1] + reverse(tour[p3..=p6]) + tour[p2..=p4] + tour[p5..]`.
fn apply_case_two(
    order: &[u32],
    p2: usize,
    p_t3: usize,
    p_t4: usize,
    p_t5: usize,
    p_t6: usize,
    n: usize,
) -> Option<Vec<u32>> {
    if !(p2 <= p_t4 && p_t4 < p_t3 && p_t3 <= p_t6 && p_t6 < p_t5) {
        return None;
    }
    let mut out = Vec::with_capacity(n);
    out.push(order[0]);
    out.extend(order[p_t3..=p_t6].iter().rev().copied());
    out.extend_from_slice(&order[p2..=p_t4]);
    out.extend_from_slice(&order[p_t5..]);
    Some(out)
}

/// Case 5: X4=2 (t4=SUC(t3)), t6=SUC(t5), t5 between t2 and t3.
/// Tour shape: `t1, t2, ..., t5, t6, ..., t3, t4, ..., t_last, [t1]`.
/// Layout: `[t1] + tour[p6..=p3] + tour[p2..=p5] + tour[p4..]`.
fn apply_case_five(
    order: &[u32],
    p2: usize,
    p_t3: usize,
    p_t4: usize,
    p_t5: usize,
    p_t6: usize,
    n: usize,
) -> Option<Vec<u32>> {
    if !(p2 <= p_t5 && p_t5 < p_t6 && p_t6 <= p_t3 && p_t3 < p_t4) {
        return None;
    }
    let mut out = Vec::with_capacity(n);
    out.push(order[0]);
    out.extend_from_slice(&order[p_t6..=p_t3]);
    out.extend_from_slice(&order[p2..=p_t5]);
    out.extend_from_slice(&order[p_t4..]);
    Some(out)
}

/// Case 6: X4=2 (t4=SUC(t3)), t6=PRED(t5), t5 between t2 and t3.
/// Tour shape: `t1, t2, ..., t6, t5, ..., t3, t4, ..., t_last, [t1]`.
/// Layout: `[t1] + reverse(tour[p2..=p6]) + reverse(tour[p5..=p3]) + tour[p4..]`.
fn apply_case_six(
    order: &[u32],
    p2: usize,
    p_t3: usize,
    p_t4: usize,
    p_t5: usize,
    p_t6: usize,
    n: usize,
) -> Option<Vec<u32>> {
    if !(p2 <= p_t6 && p_t6 < p_t5 && p_t5 <= p_t3 && p_t3 < p_t4) {
        return None;
    }
    let mut out = Vec::with_capacity(n);
    out.push(order[0]);
    out.extend(order[p2..=p_t6].iter().rev().copied());
    out.extend(order[p_t5..=p_t3].iter().rev().copied());
    out.extend_from_slice(&order[p_t4..]);
    Some(out)
}
