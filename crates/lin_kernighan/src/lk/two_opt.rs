use std::time::Instant;

use crate::{candidate::CandidateSet, distance::euc_2d, problem::Problem, tour::Tour};

#[derive(Clone, Copy, Debug)]
pub(super) struct AppliedMove {
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub d: u32,
    pub gain: i64,
}

/// 2-opt sweep with don't-look bits.
///
/// For each anchor node `a` whose don't-look bit is clear, look at the
/// two tour edges incident on `a` and try to exchange one of them with
/// an edge incident on a candidate neighbour. On improvement the bits
/// for the 4 nodes involved are cleared so the next pass re-examines
/// them; on no improvement `a`'s bit is set so we skip it until a
/// neighbour move re-activates it.
///
/// Returns the total integer gain applied during the sweep — 0 means
/// no improvement was found (a true 2-optimal plateau under the
/// candidate set).
pub fn sweep(
    problem: &Problem,
    candidates: &CandidateSet,
    tour: &mut Tour,
    dont_look: &mut [bool],
    deadline: Instant,
) -> i64 {
    let n = problem.n();
    let coords = problem.coords();
    let mut total_gain: i64 = 0;

    let mut idx: usize = 0;
    let mut visited = 0usize;
    // Check the deadline lazily — calling `Instant::now()` per anchor
    // tanks throughput on tight loops. Once per `DEADLINE_CHECK_STRIDE`
    // is enough; the per-trial deadline is rarely tighter than a few
    // milliseconds.
    const DEADLINE_CHECK_STRIDE: usize = 256;
    while visited < n {
        if visited % DEADLINE_CHECK_STRIDE == 0 && Instant::now() >= deadline {
            break;
        }
        let a = idx as u32;
        idx = (idx + 1) % n;
        visited += 1;

        if dont_look[a as usize] {
            continue;
        }

        if let Some(applied) = try_move_around(problem, candidates, tour, a, coords) {
            dont_look[applied.a as usize] = false;
            dont_look[applied.b as usize] = false;
            dont_look[applied.c as usize] = false;
            dont_look[applied.d as usize] = false;
            total_gain += applied.gain;
        } else {
            dont_look[a as usize] = true;
        }
    }

    total_gain
}

fn try_move_around(
    problem: &Problem,
    candidates: &CandidateSet,
    tour: &mut Tour,
    a: u32,
    coords: &[crate::coord::Point2D],
) -> Option<AppliedMove> {
    if let Some(m) = try_side(problem, candidates, tour, a, true, coords) {
        return Some(m);
    }
    if let Some(m) = try_side(problem, candidates, tour, a, false, coords) {
        return Some(m);
    }
    None
}

fn try_side(
    problem: &Problem,
    candidates: &CandidateSet,
    tour: &mut Tour,
    a: u32,
    forward: bool,
    coords: &[crate::coord::Point2D],
) -> Option<AppliedMove> {
    let _ = problem;
    let pa = tour.position_of(a);
    let b = if forward { tour.next(a) } else { tour.prev(a) };
    let d_ab = euc_2d(coords[a as usize], coords[b as usize]);

    if let Some(m) = try_with_anchor(tour, candidates, coords, a, b, pa, d_ab, forward, true) {
        return Some(m);
    }
    if let Some(m) = try_with_anchor(tour, candidates, coords, a, b, pa, d_ab, forward, false) {
        return Some(m);
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn try_with_anchor(
    tour: &mut Tour,
    candidates: &CandidateSet,
    coords: &[crate::coord::Point2D],
    a: u32,
    b: u32,
    pa: usize,
    d_ab: i64,
    forward: bool,
    iterate_a_candidates: bool,
) -> Option<AppliedMove> {
    let n = tour.n();
    let anchor_node = if iterate_a_candidates { a } else { b };

    let mut best: Option<(usize, usize, i64, u32, u32)> = None;
    for cand in candidates.of(anchor_node) {
        if cand.cost >= d_ab {
            break;
        }
        if cand.to == a || cand.to == b {
            continue;
        }
        let (c, d) = if iterate_a_candidates {
            let c = cand.to;
            let d = if forward { tour.next(c) } else { tour.prev(c) };
            (c, d)
        } else {
            let d = cand.to;
            let c = if forward { tour.prev(d) } else { tour.next(d) };
            (c, d)
        };
        if d == a || c == a || c == b || d == b {
            continue;
        }
        let pc = tour.position_of(c);
        let d_ac = euc_2d(coords[a as usize], coords[c as usize]);
        let d_cd = euc_2d(coords[c as usize], coords[d as usize]);
        let d_bd = euc_2d(coords[b as usize], coords[d as usize]);
        let gain = d_ab + d_cd - d_ac - d_bd;
        if gain <= 0 {
            continue;
        }

        let (lo, hi) = if forward { (pa, pc) } else { (pc, pa) };
        let segment_len = if hi >= lo {
            hi - lo
        } else {
            n - lo + hi
        };
        if segment_len < 1 || segment_len >= n {
            continue;
        }

        if best.map(|(_, _, g, _, _)| gain > g).unwrap_or(true) {
            best = Some((lo, hi, gain, c, d));
        }
    }

    if let Some((lo, hi, gain, c, d)) = best {
        tour.flip_positions(lo, hi);
        Some(AppliedMove { a, b, c, d, gain })
    } else {
        None
    }
}
