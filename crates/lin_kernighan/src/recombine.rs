//! Iterative Partial Transcription (IPT) recombination — one-shot
//! variant.
//!
//! Given two Hamiltonian cycles `T1` and `T2` on the same node set,
//! looks for the single best improving boundary-segment swap and
//! returns the resulting tour, or `None` if no improvement exists.
//!
//! Algorithm:
//! 1. Boundary detection: a node is *boundary* iff its `T1` neighbours
//!    don't match its `T2` neighbours (as an unordered pair). All
//!    other nodes are *pass-throughs* — they sit between the same
//!    two neighbours in both tours and so are forced into the same
//!    rigid chain anchored between two boundary nodes.
//! 2. Shrunken tours: extract the boundary subsequence of `T1` (and
//!    of `T2` forward and `T2` backward). Each is a Hamiltonian
//!    cycle on the boundary node set.
//! 3. For each pair of boundary nodes `(n1, n2)` and a direction in
//!    `T2`, the segment between them in shrunken `T1` and in
//!    shrunken `T2` covers the same boundary set iff the rank delta
//!    along shrunken `T1` equals the step count along shrunken `T2`
//!    (both measured from `n1` to `n2`). When they match the
//!    boundary segments are interchangeable.
//! 4. Pick the highest-gain matching pair (cost computed in projected
//!    `euc_2d` over the full macro-edges, pass-throughs included).
//! 5. Materialise the new tour by walking shrunken `T1`, switching to
//!    shrunken `T2`'s segment at `n1`, returning to shrunken `T1` at
//!    `n2`. Expand each shrunken edge by inserting the rigid chain
//!    of pass-throughs anchored to that boundary pair. Validate
//!    Hamiltonicity before returning.
//!
//! Reference: Möbius et al., "Combinatorial Optimization by
//! Iterative Partial Transcription", Phys. Rev. E 59 (1999).

use std::collections::HashMap;

use crate::{coord::Point2D, distance::euc_2d};

pub fn merge_with_tour_ipt(t1: &[u32], t2: &[u32], coords: &[Point2D]) -> Option<Vec<u32>> {
    let n = t1.len();
    if t2.len() != n || n < 6 {
        return None;
    }
    if !is_hamiltonian(t1, n) || !is_hamiltonian(t2, n) {
        return None;
    }

    let (t1_suc, t1_pred) = build_neighbour_maps(t1, n);
    let (t2_suc, t2_pred) = build_neighbour_maps(t2, n);

    // Boundary set.
    let mut is_boundary = vec![false; n];
    for i in 0..n {
        let s1 = t1_suc[i];
        let p1 = t1_pred[i];
        let s2 = t2_suc[i];
        let p2 = t2_pred[i];
        let suc_matches = s1 == s2 || s1 == p2;
        let pred_matches = p1 == s2 || p1 == p2;
        if !(suc_matches && pred_matches) {
            is_boundary[i] = true;
        }
    }
    let boundary_count = is_boundary.iter().filter(|&&b| b).count();
    if boundary_count < 2 {
        return None;
    }

    // Shrunken successors: next boundary node in tour's suc direction
    // for each boundary node. Pass-throughs map to themselves.
    let shrunken_t1_suc = build_shrunken_suc(&t1_suc, &is_boundary, n);
    let shrunken_t2_suc = build_shrunken_suc(&t2_suc, &is_boundary, n);
    let shrunken_t2_pred = build_shrunken_suc(&t2_pred, &is_boundary, n);

    // Pass-through chains for each macro-edge. By rigidity (a
    // pass-through has the same neighbour pair in both tours), the
    // chain anchored to a boundary-pair `(a, b)` is unique — equal
    // in T1 and T2 when both contain that macro-edge. Discover via
    // T1's suc walk; cross-fill from T2 to cover edges T1 doesn't
    // have but T2 does (used when the new tour borrows T2 macro-
    // edges that don't exist in T1).
    let mut chains: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    insert_chains_from_tour(&t1_suc, &is_boundary, n, &mut chains);
    insert_chains_from_tour(&t2_suc, &is_boundary, n, &mut chains);

    // Pick the lowest-id boundary node as walk anchor; rank tables
    // place this at rank 0 along T1 / along T2-fwd / along T2-bwd.
    let start = is_boundary.iter().position(|&b| b).unwrap() as u32;
    let rank_t1 = boundary_ranks(&t1_suc, &is_boundary, start, n);
    let rank_t2_fwd = boundary_ranks(&t2_suc, &is_boundary, start, n);
    let rank_t2_bwd = boundary_ranks(&t2_pred, &is_boundary, start, n);

    // Macro-edge costs along each shrunken successor map. Cost of
    // macro-edge (a, b) = sum of euc_2d edges in the rigid chain
    // a → ... → b in the original (full) tour.
    let macro_cost_t1 = build_macro_costs(&shrunken_t1_suc, &chains, coords, &is_boundary, n);
    let macro_cost_t2_fwd =
        build_macro_costs(&shrunken_t2_suc, &chains, coords, &is_boundary, n);
    let macro_cost_t2_bwd =
        build_macro_costs(&shrunken_t2_pred, &chains, coords, &is_boundary, n);

    // Find highest-gain (n1, n2, direction) where shrunken-T2-step-
    // count from n1 to n2 equals shrunken-T1-rank-delta. Skip n2=n1
    // (zero-length segment) and skip the case where T2's first step
    // from n1 lands on the same node T1's first step does (no swap
    // needed — segment of length 1 with identical endpoints).
    let mut best: Option<(u32, u32, bool, i64)> = None;
    let mut n1 = start;
    let mut outer_steps = 0usize;
    loop {
        outer_steps += 1;
        if outer_steps > n + 1 {
            return None;
        }
        if is_boundary[n1 as usize] {
            let n1_rank_t1 = rank_t1[n1 as usize];
            for direction_fwd in [true, false] {
                let shrunken = if direction_fwd {
                    &shrunken_t2_suc
                } else {
                    &shrunken_t2_pred
                };
                let ranks_t2 = if direction_fwd {
                    &rank_t2_fwd
                } else {
                    &rank_t2_bwd
                };
                if ranks_t2[n1 as usize] == u32::MAX {
                    continue;
                }
                let mut n2 = shrunken[n1 as usize];
                let mut sub_count: u32 = 0;
                let mut walked = 0usize;
                // Maximum T1 rank delta seen so far along the T2 walk.
                // A match `sub_t1 == sub_count` is valid only when
                // `sub_t1 > max_sub_t1`, i.e. we've reached a new
                // T1 rank further than any previously visited
                // boundary. This invariant — borrowed from LKH's
                // MergeWithTourIPT — ensures the T2 walk covered
                // exactly the T1 ranks `[1..sub_count]` (the same
                // boundary set as T1's segment).
                let mut max_sub_t1: u32 = 0;
                while n2 != n1 {
                    walked += 1;
                    if walked > boundary_count {
                        break;
                    }
                    sub_count += 1;
                    let n2_rank_t1 = rank_t1[n2 as usize];
                    let sub_t1 = if n2_rank_t1 >= n1_rank_t1 {
                        n2_rank_t1 - n1_rank_t1
                    } else {
                        boundary_count as u32 - n1_rank_t1 + n2_rank_t1
                    };
                    if sub_t1 <= max_sub_t1 {
                        // T2 walk re-visits a boundary at lower
                        // T1 rank than already covered — skip.
                        n2 = shrunken[n2 as usize];
                        continue;
                    }
                    if sub_t1 == sub_count {
                        // Equal-cardinality boundary segment.
                        let cost_t1 = shrunken_segment_cost(
                            &shrunken_t1_suc,
                            &macro_cost_t1,
                            n1,
                            n2,
                            boundary_count,
                        );
                        let cost_t2 = shrunken_segment_cost(
                            shrunken,
                            if direction_fwd { &macro_cost_t2_fwd } else { &macro_cost_t2_bwd },
                            n1,
                            n2,
                            boundary_count,
                        );
                        let (Some(cost_t1), Some(cost_t2)) = (cost_t1, cost_t2) else {
                            break;
                        };
                        let gain = cost_t1 - cost_t2;
                        if gain > 0 && best.as_ref().map(|s| gain > s.3).unwrap_or(true) {
                            best = Some((n1, n2, direction_fwd, gain));
                        }
                        break;
                    }
                    // Not a match yet; record this T1 rank as the
                    // new ceiling so subsequent same-or-lower ranks
                    // are skipped.
                    max_sub_t1 = sub_t1;
                    n2 = shrunken[n2 as usize];
                }
            }
        }
        n1 = t1_suc[n1 as usize];
        if n1 == start {
            break;
        }
    }

    let (n1, n2, fwd, _gain) = best?;

    // Materialise: start the output at `n1`. First walk T2's
    // shrunken successors from `n1` to `n2` (the splice). Then walk
    // T1's shrunken successors from `n2` back to `n1` (the
    // complement). Each shrunken edge expands its rigid pass-through
    // chain via the global chain lookup (chains are anchored to a
    // boundary-pair and identical across tours that contain them).
    // Beginning the walk at `n1` avoids the failure mode where `n1`
    // appears later than `n2` in T1's shrunken cycle: a walk from a
    // generic `start` would otherwise hit `n2` first and end the
    // splice before it began.
    let shrunken_t2_chosen = if fwd {
        &shrunken_t2_suc
    } else {
        &shrunken_t2_pred
    };

    let mut out: Vec<u32> = Vec::with_capacity(n);
    let mut cur = n1;
    let mut steps_used = 0usize;
    // Phase 1: T2 splice from n1 to n2.
    loop {
        steps_used += 1;
        if steps_used > boundary_count + 1 {
            return None;
        }
        out.push(cur);
        let next = shrunken_t2_chosen[cur as usize];
        if let Some(chain) = chains.get(&(cur, next)) {
            for &p in chain {
                out.push(p);
            }
        }
        if next == n2 {
            cur = next;
            break;
        }
        cur = next;
        if out.len() > n {
            return None;
        }
    }
    // Phase 2: T1 complement from n2 back to n1 (exclusive of n1
    // since we already pushed it at the start of phase 1).
    let mut comp_steps = 0usize;
    loop {
        comp_steps += 1;
        if comp_steps > boundary_count + 1 {
            return None;
        }
        out.push(cur);
        let next = shrunken_t1_suc[cur as usize];
        if let Some(chain) = chains.get(&(cur, next)) {
            for &p in chain {
                out.push(p);
            }
        }
        if next == n1 {
            break;
        }
        cur = next;
        if out.len() > n {
            return None;
        }
    }

    if out.len() != n {
        return None;
    }
    // Validate Hamiltonicity.
    let mut seen = vec![false; n];
    for &v in &out {
        let i = v as usize;
        if i >= n || seen[i] {
            return None;
        }
        seen[i] = true;
    }
    Some(out)
}

fn is_hamiltonian(tour: &[u32], n: usize) -> bool {
    if tour.len() != n {
        return false;
    }
    let mut seen = vec![false; n];
    for &v in tour {
        let i = v as usize;
        if i >= n || seen[i] {
            return false;
        }
        seen[i] = true;
    }
    true
}

fn build_neighbour_maps(tour: &[u32], n: usize) -> (Vec<u32>, Vec<u32>) {
    let mut suc = vec![0u32; n];
    let mut pred = vec![0u32; n];
    for i in 0..n {
        let cur = tour[i] as usize;
        let nxt = tour[(i + 1) % n];
        suc[cur] = nxt;
        pred[nxt as usize] = cur as u32;
    }
    (suc, pred)
}

/// For each boundary node, returns the next boundary node along the
/// given `suc` walk. Pass-throughs map to themselves (never queried).
fn build_shrunken_suc(suc: &[u32], is_boundary: &[bool], n: usize) -> Vec<u32> {
    let mut shrunken = (0..n).map(|i| i as u32).collect::<Vec<_>>();
    for i in 0..n {
        if !is_boundary[i] {
            continue;
        }
        let mut cur = suc[i];
        let mut steps = 0usize;
        while !is_boundary[cur as usize] {
            cur = suc[cur as usize];
            steps += 1;
            if steps > n {
                // Defensive: malformed input.
                return shrunken;
            }
        }
        shrunken[i] = cur;
    }
    shrunken
}

/// Discovers pass-through chains by walking `suc` from each boundary
/// node, recording pass-throughs until the next boundary. Inserts
/// the chain at key `(boundary, next_boundary)` and the reversed
/// chain at key `(next_boundary, boundary)`. Skips inserts when the
/// key already has an entry (rigidity guarantees existing entries
/// match anyway).
fn insert_chains_from_tour(
    suc: &[u32],
    is_boundary: &[bool],
    n: usize,
    chains: &mut HashMap<(u32, u32), Vec<u32>>,
) {
    for i in 0..n {
        if !is_boundary[i] {
            continue;
        }
        let a = i as u32;
        let mut chain: Vec<u32> = Vec::new();
        let mut cur = suc[i];
        let mut steps = 0usize;
        while !is_boundary[cur as usize] {
            chain.push(cur);
            cur = suc[cur as usize];
            steps += 1;
            if steps > n {
                return;
            }
        }
        let b = cur;
        chains.entry((a, b)).or_insert_with(|| chain.clone());
        let mut rev = chain;
        rev.reverse();
        chains.entry((b, a)).or_insert(rev);
    }
}

fn boundary_ranks(walk: &[u32], is_boundary: &[bool], start: u32, n: usize) -> Vec<u32> {
    let mut ranks = vec![u32::MAX; n];
    let mut node = start;
    let mut r: u32 = 0;
    let mut steps = 0usize;
    loop {
        steps += 1;
        if steps > n + 1 {
            break;
        }
        if is_boundary[node as usize] {
            ranks[node as usize] = r;
            r += 1;
        }
        node = walk[node as usize];
        if node == start {
            break;
        }
    }
    ranks
}

/// Cost of each macro-edge: `cost[boundary] = sum of euc_2d edges in
/// the chain from `boundary` to `shrunken_suc[boundary]`.
fn build_macro_costs(
    shrunken_suc: &[u32],
    chains: &HashMap<(u32, u32), Vec<u32>>,
    coords: &[Point2D],
    is_boundary: &[bool],
    n: usize,
) -> Vec<i64> {
    let mut cost = vec![0i64; n];
    for i in 0..n {
        if !is_boundary[i] {
            continue;
        }
        let a = i as u32;
        let b = shrunken_suc[i];
        if a == b {
            // Boundary node maps to itself (no successor). Shouldn't
            // happen for boundary count >= 2.
            continue;
        }
        let chain = chains.get(&(a, b));
        let mut total: i64 = 0;
        let mut prev = a;
        if let Some(chain) = chain {
            for &p in chain {
                total += euc_2d(coords[prev as usize], coords[p as usize]);
                prev = p;
            }
        }
        total += euc_2d(coords[prev as usize], coords[b as usize]);
        cost[i] = total;
    }
    cost
}

/// Sum of macro-edge costs from `n1` to `n2` (exclusive of `n2`)
/// along `shrunken_suc`. Returns None if `n2` is not reachable
/// within `boundary_count` steps from `n1`.
fn shrunken_segment_cost(
    shrunken_suc: &[u32],
    macro_cost: &[i64],
    n1: u32,
    n2: u32,
    boundary_count: usize,
) -> Option<i64> {
    let mut total: i64 = 0;
    let mut cur = n1;
    let mut steps = 0usize;
    while cur != n2 {
        steps += 1;
        if steps > boundary_count {
            return None;
        }
        total += macro_cost[cur as usize];
        cur = shrunken_suc[cur as usize];
    }
    Some(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::Point2D;

    fn tour_len(tour: &[u32], coords: &[Point2D]) -> i64 {
        let n = tour.len();
        let mut total = 0;
        for i in 0..n {
            total += euc_2d(coords[tour[i] as usize], coords[tour[(i + 1) % n] as usize]);
        }
        total
    }

    #[test]
    fn no_op_when_tours_identical() {
        let coords: Vec<Point2D> = (0..10).map(|i| Point2D::new(i as f64, 0.0)).collect();
        let t1: Vec<u32> = (0..10).collect();
        assert!(merge_with_tour_ipt(&t1, &t1, &coords).is_none());
    }

    #[test]
    fn finds_improvement_when_t2_has_shorter_segment() {
        let coords: Vec<Point2D> = (0..6).map(|i| Point2D::new(i as f64, 0.0)).collect();
        let t1 = vec![0u32, 2, 1, 3, 4, 5];
        let t2 = vec![0u32, 1, 2, 3, 4, 5];
        let len_t1 = tour_len(&t1, &coords);
        let child = merge_with_tour_ipt(&t1, &t2, &coords)
            .expect("IPT should find an improving swap");
        let len_child = tour_len(&child, &coords);
        assert!(len_child < len_t1, "child must beat T1");
        // Hamiltonicity sanity.
        let mut sorted = child.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0u32, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn returns_none_on_size_mismatch() {
        let coords: Vec<Point2D> = (0..6).map(|i| Point2D::new(i as f64, 0.0)).collect();
        let t1: Vec<u32> = (0..6).collect();
        let t2: Vec<u32> = (0..5).collect();
        assert!(merge_with_tour_ipt(&t1, &t2, &coords).is_none());
    }

    /// Tour pair with pass-through chains so the splice has to
    /// expand non-trivial chains correctly.
    #[test]
    fn handles_pass_through_chains() {
        // 8 nodes on a line.
        let coords: Vec<Point2D> = (0..8).map(|i| Point2D::new(i as f64, 0.0)).collect();
        // T1: 0,1,2,3,4,5,6,7 (length 7 forward + 7 back = 14).
        // T2: 0,2,1,3,4,5,7,6 (twice-flipped).
        let t1: Vec<u32> = (0..8).collect();
        let t2 = vec![0u32, 2, 1, 3, 4, 5, 7, 6];
        if let Some(child) = merge_with_tour_ipt(&t1, &t2, &coords) {
            // Validate Hamiltonicity.
            let mut sorted = child.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..8u32).collect::<Vec<_>>());
            // Length should not exceed T1's.
            assert!(tour_len(&child, &coords) <= tour_len(&t1, &coords));
        }
    }
}
