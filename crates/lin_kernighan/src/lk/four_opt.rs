//! Sequential 4-opt extension of the LK chain.
//!
//! Reaches the LKH "MOVE_TYPE=4" search depth by extending the 3-opt
//! chain one more level: pick `t7 ∈ candidates(t6)` and `t8` as a
//! tour neighbour of `t7`, then close with edge `(t8, t1)`. The 4-opt
//! move removes four edges and adds four:
//!
//! * **Remove**: `(t1,t2)`, `(t3,t4)`, `(t5,t6)`, `(t7,t8)`
//! * **Add**: `(t2,t3)`, `(t4,t5)`, `(t6,t7)`, `(t8,t1)`
//!
//! Rather than enumerate the ≥8 LKH "case" sub-categories with their
//! per-case position invariants, this implementation rebuilds the
//! tour generically: form the new adjacency, walk the cycle from
//! `t1`, and accept only if the walk visits all `n` nodes exactly
//! once (a valid Hamiltonian). Each move attempt costs O(n) — the
//! same as our 3-opt apply — and the case-free apply makes adding
//! the move much cheaper to maintain than per-case slice arithmetic.

use std::time::Instant;

use crate::{candidate::CandidateSet, distance::euc_2d, problem::Problem, tour::Tour};

pub fn sweep(
    problem: &Problem,
    candidates: &CandidateSet,
    tour: &mut Tour,
    dont_look: &mut [bool],
    deadline: Instant,
) -> i64 {
    let n = problem.n();
    if n < 8 {
        return 0;
    }
    let coords = problem.coords();
    let mut total_gain: i64 = 0;
    let mut moves_applied = 0usize;

    const DEADLINE_CHECK_STRIDE: usize = 64;
    let mut idx: usize = 0;
    let mut visited = 0usize;
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
            "lin_kernighan.four_opt: applied {} move(s), gain={}",
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
) -> Option<(i64, [u32; 8])> {
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
) -> Option<(i64, [u32; 8])> {
    let t2 = if forward { tour.next(t1) } else { tour.prev(t1) };
    let c_t1 = coords[t1 as usize];
    let c_t2 = coords[t2 as usize];
    let d_t1_t2 = euc_2d(c_t1, c_t2);

    // Best move tracking — we keep the highest-gain feasible 4-opt
    // across all candidate combinations at the current anchor.
    let mut best: Option<(i64, [u32; 8])> = None;

    // The four-deep nested candidate enumeration is expensive
    // (O(k^3) per anchor); limit the inner candidate breadth so this
    // sweep doesn't dominate the deadline. LKH uses MaxBreadth for
    // the same purpose.
    const MAX_BREADTH: usize = 8;

    for cand_t3 in candidates.of(t2).iter().take(MAX_BREADTH) {
        let t3 = cand_t3.to;
        if t3 == t1 || t3 == t2 {
            continue;
        }
        let d_t2_t3 = cand_t3.cost;
        let g1 = d_t1_t2 - d_t2_t3;

        for &t4_side in &[true, false] {
            let t4 = if t4_side {
                if forward { tour.prev(t3) } else { tour.next(t3) }
            } else if forward {
                tour.next(t3)
            } else {
                tour.prev(t3)
            };
            if t4 == t1 || t4 == t2 || t4 == t3 {
                continue;
            }
            let d_t3_t4 = euc_2d(coords[t3 as usize], coords[t4 as usize]);
            let g2 = g1 + d_t3_t4;

            for cand_t5 in candidates.of(t4).iter().take(MAX_BREADTH) {
                let t5 = cand_t5.to;
                if t5 == t1 || t5 == t2 || t5 == t3 || t5 == t4 {
                    continue;
                }
                let d_t4_t5 = cand_t5.cost;
                let g3 = g2 - d_t4_t5;

                for &t6_side in &[true, false] {
                    let t6 = if t6_side {
                        if forward { tour.next(t5) } else { tour.prev(t5) }
                    } else if forward {
                        tour.prev(t5)
                    } else {
                        tour.next(t5)
                    };
                    if t6 == t1 || t6 == t2 || t6 == t3 || t6 == t4 || t6 == t5 {
                        continue;
                    }
                    let d_t5_t6 = euc_2d(coords[t5 as usize], coords[t6 as usize]);
                    let g4 = g3 + d_t5_t6;

                    for cand_t7 in candidates.of(t6).iter().take(MAX_BREADTH) {
                        let t7 = cand_t7.to;
                        if t7 == t1
                            || t7 == t2
                            || t7 == t3
                            || t7 == t4
                            || t7 == t5
                            || t7 == t6
                        {
                            continue;
                        }
                        let d_t6_t7 = cand_t7.cost;
                        let g5 = g4 - d_t6_t7;

                        for &t8_side in &[true, false] {
                            let t8 = if t8_side {
                                if forward { tour.next(t7) } else { tour.prev(t7) }
                            } else if forward {
                                tour.prev(t7)
                            } else {
                                tour.next(t7)
                            };
                            if t8 == t1
                                || t8 == t2
                                || t8 == t3
                                || t8 == t4
                                || t8 == t5
                                || t8 == t6
                                || t8 == t7
                            {
                                continue;
                            }
                            let d_t7_t8 = euc_2d(coords[t7 as usize], coords[t8 as usize]);
                            let d_t8_t1 = euc_2d(coords[t8 as usize], coords[t1 as usize]);
                            let gain = g5 + d_t7_t8 - d_t8_t1;
                            if gain <= 0 {
                                continue;
                            }
                            if best.map(|(g, _)| gain > g).unwrap_or(true) {
                                best = Some((gain, [t1, t2, t3, t4, t5, t6, t7, t8]));
                            }
                        }
                    }
                }
            }
        }
    }

    let (gain, nodes) = best?;
    if apply_generic(tour, nodes) {
        Some((gain, nodes))
    } else {
        None
    }
}

/// Generic apply: rebuild the tour from the new edge adjacency. Each
/// node's two new neighbours are determined by the edge set
/// `removed = {(t1,t2), (t3,t4), (t5,t6), (t7,t8)}`,
/// `added   = {(t2,t3), (t4,t5), (t6,t7), (t8,t1)}`. If the resulting
/// adjacency walks all `n` nodes in a single cycle, accept and write
/// back the order; otherwise the move would split the tour into
/// disjoint sub-cycles and we bail.
fn apply_generic(tour: &mut Tour, nodes: [u32; 8]) -> bool {
    let n = tour.n();
    let [t1, t2, t3, t4, t5, t6, t7, t8] = nodes;

    // Build the current adjacency from the tour, then apply the
    // removed/added edge changes.
    let mut adj: Vec<[u32; 2]> = Vec::with_capacity(n);
    for i in 0..n {
        let node = tour.node_at(i);
        let prev = tour.prev(node);
        let next = tour.next(node);
        adj.push([prev, next]);
    }
    let position_in_adj = |adj: &Vec<[u32; 2]>, node: u32, neighbour: u32| -> Option<usize> {
        if adj[node as usize][0] == neighbour {
            Some(0)
        } else if adj[node as usize][1] == neighbour {
            Some(1)
        } else {
            None
        }
    };

    let removals: [(u32, u32); 4] = [(t1, t2), (t3, t4), (t5, t6), (t7, t8)];
    let additions: [(u32, u32); 4] = [(t2, t3), (t4, t5), (t6, t7), (t8, t1)];

    // Remove edges by clearing the slot to a sentinel (u32::MAX).
    for (a, b) in removals {
        let Some(slot_a) = position_in_adj(&adj, a, b) else {
            return false;
        };
        let Some(slot_b) = position_in_adj(&adj, b, a) else {
            return false;
        };
        adj[a as usize][slot_a] = u32::MAX;
        adj[b as usize][slot_b] = u32::MAX;
    }

    // Add edges by placing into the first empty slot. If both slots
    // are full, the move is invalid (would push a node above degree 2).
    for (a, b) in additions {
        let slot_a = if adj[a as usize][0] == u32::MAX {
            0
        } else if adj[a as usize][1] == u32::MAX {
            1
        } else {
            return false;
        };
        let slot_b = if adj[b as usize][0] == u32::MAX {
            0
        } else if adj[b as usize][1] == u32::MAX {
            1
        } else {
            return false;
        };
        adj[a as usize][slot_a] = b;
        adj[b as usize][slot_b] = a;
    }

    // Walk the cycle from t1.
    let mut order: Vec<u32> = Vec::with_capacity(n);
    let mut current = t1;
    let mut prev_node: u32 = u32::MAX;
    for _ in 0..n {
        order.push(current);
        let next = if adj[current as usize][0] != prev_node {
            adj[current as usize][0]
        } else {
            adj[current as usize][1]
        };
        if next == u32::MAX {
            return false;
        }
        prev_node = current;
        current = next;
    }
    if current != t1 {
        return false;
    }

    // Verify Hamiltonian (no repeats).
    let mut seen = vec![false; n];
    for &node in &order {
        let id = node as usize;
        if id >= n || seen[id] {
            return false;
        }
        seen[id] = true;
    }

    *tour = Tour::from_order(&order);
    true
}
