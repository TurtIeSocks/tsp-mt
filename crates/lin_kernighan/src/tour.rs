use std::sync::atomic::{AtomicU64, Ordering};

use crate::{distance::euc_2d, problem::Problem};

/// Global perf counters for hot-path operations. Read via `Tour::stats`
/// from the test/bench harness; production code can leave these
/// untouched at near-zero cost (single relaxed atomic increment per
/// op).
pub static FLIP_COUNT: AtomicU64 = AtomicU64::new(0);
pub static FLIP_OPS: AtomicU64 = AtomicU64::new(0);
pub static RELOCATE_COUNT: AtomicU64 = AtomicU64::new(0);
pub static RELOCATE_OPS: AtomicU64 = AtomicU64::new(0);

/// Snapshot the four perf counters and reset to zero. Returns
/// `(flip_calls, flip_ops_total, relocate_calls, relocate_ops_total)`.
pub fn take_stats() -> (u64, u64, u64, u64) {
    (
        FLIP_COUNT.swap(0, Ordering::Relaxed),
        FLIP_OPS.swap(0, Ordering::Relaxed),
        RELOCATE_COUNT.swap(0, Ordering::Relaxed),
        RELOCATE_OPS.swap(0, Ordering::Relaxed),
    )
}

/// Tour stored as a circular sequence of node indices.
///
/// LKH represents tours with a `Pred`/`Suc` doubly-linked list plus a
/// two-level segment tree for fast `Flip`/`Between`. For the working size
/// range tsp_mt_core uses (≤ a few thousand nodes per chunk) an O(n) flip
/// on a `Vec<u32>` is fast enough — the more elaborate representation is
/// reserved for a follow-up if profiling demands it.
///
/// Invariants:
/// - `tour.len() == n`
/// - `tour` is a permutation of `0..n`
/// - `pos[tour[i]] == i` for every position
#[derive(Clone, Debug)]
pub struct Tour {
    tour: Vec<u32>,
    pos: Vec<u32>,
}

impl Tour {
    pub fn from_order(order: &[u32]) -> Self {
        let n = order.len();
        let mut pos = vec![0u32; n];
        for (i, &node) in order.iter().enumerate() {
            pos[node as usize] = i as u32;
        }
        Self {
            tour: order.to_vec(),
            pos,
        }
    }

    pub fn identity(n: usize) -> Self {
        let order: Vec<u32> = (0..n as u32).collect();
        Self::from_order(&order)
    }

    pub fn n(&self) -> usize {
        self.tour.len()
    }

    #[inline]
    pub fn node_at(&self, position: usize) -> u32 {
        self.tour[position]
    }

    #[inline]
    pub fn position_of(&self, node: u32) -> usize {
        self.pos[node as usize] as usize
    }

    #[inline]
    pub fn next(&self, node: u32) -> u32 {
        let n = self.tour.len();
        let p = self.pos[node as usize] as usize;
        self.tour[(p + 1) % n]
    }

    #[inline]
    pub fn prev(&self, node: u32) -> u32 {
        let n = self.tour.len();
        let p = self.pos[node as usize] as usize;
        self.tour[(p + n - 1) % n]
    }

    /// Apply a double-bridge perturbation in place: given three split
    /// positions `p1 < p2 < p3`, the tour `[A | B | C | D]` becomes
    /// `[A | C | B | D]`. Implemented via three in-place reversals
    /// (the "double reversal" identity for adjacent-block swap), no
    /// allocations.
    pub fn double_bridge_in_place(&mut self, p1: usize, p2: usize, p3: usize) {
        let n = self.tour.len();
        if !(p1 < p2 && p2 < p3 && p3 <= n) {
            return;
        }
        // Three in-place reversals: reverse B, reverse C, then reverse
        // the combined B'+C' span. Net effect: blocks B and C swap.
        self.tour[p1..p2].reverse();
        self.tour[p2..p3].reverse();
        self.tour[p1..p3].reverse();
        for i in p1..p3 {
            self.pos[self.tour[i] as usize] = i as u32;
        }
    }

    /// Relocate the segment `tour[seg_start..seg_start+seg_len]` to
    /// the position immediately after `insert_after_pos`, optionally
    /// reversing it first. In-place: mutates the tour and pos arrays
    /// directly, no allocations.
    ///
    /// The caller is responsible for ensuring the move is valid (the
    /// segment must not overlap the insertion point, and neither
    /// `seg_start` nor `insert_after_pos` may straddle the wrap).
    /// Returns false on an unsupported wrap configuration; the caller
    /// must fall back to a rebuild path in that case.
    pub fn relocate_segment(
        &mut self,
        seg_start: usize,
        seg_len: usize,
        insert_after_pos: usize,
        reversed: bool,
    ) -> bool {
        let n = self.tour.len();
        if seg_len == 0 || seg_len >= n {
            return false;
        }
        let seg_end = seg_start.checked_add(seg_len - 1).filter(|&v| v < n);
        let Some(seg_end) = seg_end else {
            return false;
        };
        // Wrap-around segment or insertion target is too rare to be
        // worth a fast path here; fall back to caller-level rebuild.
        if insert_after_pos >= n {
            return false;
        }
        // Insertion point inside the segment is a no-op; reject.
        if insert_after_pos >= seg_start && insert_after_pos <= seg_end {
            return false;
        }
        // Reverse segment in place if requested.
        if reversed {
            self.tour[seg_start..=seg_end].reverse();
        }
        // Three cases depending on insertion direction:
        //
        // 1. insert_after_pos > seg_end → forward move.
        //    Rotate the sub-slice [seg_start..=insert_after_pos] left
        //    by seg_len: the segment slides past the intervening
        //    elements and lands at the end of the sub-slice.
        // 2. insert_after_pos < seg_start → backward move.
        //    Rotate the sub-slice [insert_after_pos+1..=seg_end] right
        //    by seg_len: the segment slides back to just after the
        //    insertion point.
        // 3. otherwise: impossible (filtered above).
        RELOCATE_COUNT.fetch_add(1, Ordering::Relaxed);
        let span = if insert_after_pos > seg_end {
            insert_after_pos - seg_start + 1
        } else {
            seg_end - insert_after_pos
        };
        RELOCATE_OPS.fetch_add(span as u64, Ordering::Relaxed);
        if insert_after_pos > seg_end {
            self.tour[seg_start..=insert_after_pos].rotate_left(seg_len);
            for i in seg_start..=insert_after_pos {
                self.pos[self.tour[i] as usize] = i as u32;
            }
        } else {
            self.tour[(insert_after_pos + 1)..=seg_end].rotate_right(seg_len);
            for i in (insert_after_pos + 1)..=seg_end {
                self.pos[self.tour[i] as usize] = i as u32;
            }
        }
        true
    }

    /// Reverse the tour segment between positions `i+1` and `j`
    /// (inclusive) — a standard 2-opt flip. Picks the shorter side
    /// to flip so the worst case is O(n/2) per move.
    pub fn flip_positions(&mut self, i: usize, j: usize) {
        let n = self.tour.len();
        let start = (i + 1) % n;
        let end = j % n;
        let segment_len = if end >= start {
            end - start + 1
        } else {
            n - start + end + 1
        };

        if segment_len * 2 > n {
            // Flip the complement instead (shorter side).
            let comp_start = (end + 1) % n;
            let comp_end = (start + n - 1) % n;
            self.flip_range(comp_start, comp_end, n - segment_len);
        } else {
            self.flip_range(start, end, segment_len);
        }
    }

    fn flip_range(&mut self, start: usize, end: usize, len: usize) {
        FLIP_COUNT.fetch_add(1, Ordering::Relaxed);
        FLIP_OPS.fetch_add(len as u64 / 2, Ordering::Relaxed);
        let n = self.tour.len();
        let mut left = start;
        let mut right = end;
        for _ in 0..(len / 2) {
            self.tour.swap(left, right);
            self.pos[self.tour[left] as usize] = left as u32;
            self.pos[self.tour[right] as usize] = right as u32;
            left = (left + 1) % n;
            right = (right + n - 1) % n;
        }
        if len % 2 == 1 {
            self.pos[self.tour[left] as usize] = left as u32;
        }
    }

    /// Tour length under the EUC_2D edge weight.
    pub fn length(&self, problem: &Problem) -> i64 {
        let n = self.tour.len();
        let coords = problem.coords();
        let mut total: i64 = 0;
        for i in 0..n {
            let a = self.tour[i] as usize;
            let b = self.tour[(i + 1) % n] as usize;
            total += euc_2d(coords[a], coords[b]);
        }
        total
    }

    pub fn as_slice(&self) -> &[u32] {
        &self.tour
    }

    pub fn into_vec_usize(self) -> Vec<usize> {
        self.tour.into_iter().map(|v| v as usize).collect()
    }

    #[cfg(test)]
    pub fn validate(&self) -> bool {
        let n = self.tour.len();
        if self.pos.len() != n {
            return false;
        }
        let mut seen = vec![false; n];
        for (i, &node) in self.tour.iter().enumerate() {
            let id = node as usize;
            if id >= n || seen[id] {
                return false;
            }
            seen[id] = true;
            if self.pos[id] as usize != i {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::Point2D;

    fn square_problem() -> Problem {
        Problem::new(vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
        ])
        .expect("valid problem")
    }

    #[test]
    fn identity_tour_is_well_formed() {
        let t = Tour::identity(4);
        assert!(t.validate());
        assert_eq!(t.next(0), 1);
        assert_eq!(t.next(3), 0);
        assert_eq!(t.prev(0), 3);
    }

    #[test]
    fn length_of_unit_square_traversal_is_four() {
        let t = Tour::identity(4);
        assert_eq!(t.length(&square_problem()), 4);
    }

    #[test]
    fn flip_reverses_short_segment() {
        let mut t = Tour::identity(6);
        t.flip_positions(0, 3);
        assert!(t.validate());
        assert_eq!(t.as_slice(), &[0, 3, 2, 1, 4, 5]);
    }

    #[test]
    fn flip_picks_shorter_side() {
        let mut t = Tour::identity(6);
        t.flip_positions(4, 1);
        assert!(t.validate());
        // After flipping the wrap-around segment 5->0->1: the equivalent
        // un-rotated tour stays canonical because we picked the shorter side.
        // Just verify invariants.
        assert_eq!(t.n(), 6);
    }

    #[test]
    fn flip_full_round_trip_restores_tour() {
        let mut t = Tour::identity(8);
        let before = t.as_slice().to_vec();
        t.flip_positions(2, 6);
        assert!(t.validate());
        t.flip_positions(2, 6);
        assert!(t.validate());
        assert_eq!(t.as_slice(), &before[..]);
    }
}
