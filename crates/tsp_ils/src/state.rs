//! Tour state and local-search moves.
//!
//! The tour is an array `order` (position -> node) plus its inverse `pos`
//! (node -> position). Improvement moves are the classic candidate-restricted
//! neighborhood used by Lin-Kernighan-family solvers:
//!
//! * 2-opt: replace edges (a,b),(c,d) with (a,c),(b,d), applied by reversing
//!   the shorter of the two arcs.
//! * Or-opt: relocate a short segment (1..=max_len nodes) elsewhere in the
//!   tour, in either orientation, next to a candidate neighbor. Also covers
//!   in-place segment reversal.
//!
//! Both are driven by a work queue with don't-look bits: a node is examined
//! only when one of its incident edges changed. Perturbation is the classic
//! double-bridge kick, windowed so re-optimization stays local.
//!
//! A tour segment extracted from a bigger problem is optimized as a cycle
//! with one *frozen* edge joining its two boundary nodes: any move that
//! would remove the frozen edge is rejected, which is exactly equivalent to
//! optimizing an open path with fixed endpoints.

use std::collections::VecDeque;
use std::time::Instant;

use crate::candidates::Candidates;
use crate::kdtree::dist;
use crate::rng::SplitMix64;

/// Minimum meaningful gain. Distances are meters-scale f64; this sits far
/// above accumulated floating-point noise but below any real improvement.
pub const EPS: f64 = 1e-7;

pub struct TourState<'a, const D: usize> {
    pts: &'a [[f64; D]],
    cand: &'a Candidates,
    pub order: Vec<u32>,
    pub pos: Vec<u32>,
    /// Undirected node pair that must remain tour-adjacent (path endpoints).
    frozen: Option<(u32, u32)>,
    dont_look: Vec<bool>,
    in_queue: Vec<bool>,
    queue: VecDeque<u32>,
    /// Incrementally maintained cycle length (includes the frozen edge).
    pub cur_len: f64,
    or_opt_max: usize,
    kick_window: usize,
    rng: SplitMix64,
    buf: Vec<u32>,
}

impl<'a, const D: usize> TourState<'a, D> {
    pub fn new(
        pts: &'a [[f64; D]],
        cand: &'a Candidates,
        order: Vec<u32>,
        frozen: Option<(u32, u32)>,
        or_opt_max: usize,
        kick_window: usize,
        rng: SplitMix64,
    ) -> Self {
        let n = order.len();
        let mut pos = vec![0u32; n];
        for (i, &v) in order.iter().enumerate() {
            pos[v as usize] = i as u32;
        }
        let mut state = Self {
            pts,
            cand,
            order,
            pos,
            frozen,
            dont_look: vec![false; n],
            in_queue: vec![false; n],
            queue: VecDeque::with_capacity(n),
            cur_len: 0.0,
            or_opt_max,
            kick_window,
            rng,
            buf: Vec::with_capacity(64),
        };
        state.cur_len = state.tour_length();
        state
    }

    #[inline(always)]
    fn n(&self) -> usize {
        self.order.len()
    }

    #[inline(always)]
    fn succ_pos(&self, p: u32) -> u32 {
        if p as usize + 1 == self.n() { 0 } else { p + 1 }
    }

    #[inline(always)]
    fn pred_pos(&self, p: u32) -> u32 {
        if p == 0 { self.n() as u32 - 1 } else { p - 1 }
    }

    #[inline(always)]
    fn succ(&self, v: u32) -> u32 {
        self.order[self.succ_pos(self.pos[v as usize]) as usize]
    }

    #[inline(always)]
    fn pred(&self, v: u32) -> u32 {
        self.order[self.pred_pos(self.pos[v as usize]) as usize]
    }

    #[inline(always)]
    fn d(&self, a: u32, b: u32) -> f64 {
        dist(&self.pts[a as usize], &self.pts[b as usize])
    }

    /// Distance between two nodes (exposed for spike detection).
    #[inline]
    pub fn dist_between(&self, a: u32, b: u32) -> f64 {
        self.d(a, b)
    }

    #[inline(always)]
    fn is_frozen(&self, a: u32, b: u32) -> bool {
        match self.frozen {
            Some((x, y)) => (a == x && b == y) || (a == y && b == x),
            None => false,
        }
    }

    /// Forward arc length from position `i` to position `j`, inclusive.
    #[inline(always)]
    fn arc_len(&self, i: u32, j: u32) -> usize {
        let n = self.n();
        ((j as usize + n - i as usize) % n) + 1
    }

    pub fn tour_length(&self) -> f64 {
        let n = self.n();
        (0..n)
            .map(|i| self.d(self.order[i], self.order[(i + 1) % n]))
            .sum()
    }

    #[inline]
    fn mark_dirty(&mut self, v: u32) {
        self.dont_look[v as usize] = false;
        if !self.in_queue[v as usize] {
            self.in_queue[v as usize] = true;
            self.queue.push_back(v);
        }
    }

    /// Queue every node for examination (tour order).
    pub fn activate_all(&mut self) {
        self.queue.clear();
        for i in 0..self.n() {
            let v = self.order[i];
            self.dont_look[v as usize] = false;
            self.in_queue[v as usize] = true;
            self.queue.push_back(v);
        }
    }

    /// Queue a specific set of nodes (e.g. spike-edge endpoints).
    pub fn activate(&mut self, nodes: impl IntoIterator<Item = u32>) {
        for v in nodes {
            self.mark_dirty(v);
        }
    }

    pub fn set_or_opt_max(&mut self, max_len: usize) {
        self.or_opt_max = max_len;
    }

    /// Reverse the forward arc from position `i` to position `j`, inclusive.
    fn reverse_arc(&mut self, i: u32, j: u32) {
        let len = self.arc_len(i, j);
        let mut a = i;
        let mut b = j;
        for _ in 0..len / 2 {
            let x = self.order[a as usize];
            let y = self.order[b as usize];
            self.order[a as usize] = y;
            self.pos[y as usize] = a;
            self.order[b as usize] = x;
            self.pos[x as usize] = b;
            a = self.succ_pos(a);
            b = self.pred_pos(b);
        }
    }

    /// Reverse whichever of the two complementary arcs is shorter. The arcs
    /// are `(i1..j1)` and `(i2..j2)`; reversing either yields the same
    /// undirected cycle.
    fn reverse_shorter(&mut self, i1: u32, j1: u32, i2: u32, j2: u32) {
        if self.arc_len(i1, j1) * 2 <= self.n() {
            self.reverse_arc(i1, j1);
        } else {
            self.reverse_arc(i2, j2);
        }
    }

    /// Try to improve the tour with a 2-opt move on an edge incident to `a`.
    fn try_two_opt(&mut self, a: u32) -> bool {
        let cand = self.cand;
        // Direction 1: remove successor edges (a,b) and (c,succ(c)).
        let b = self.succ(a);
        if !self.is_frozen(a, b) {
            let d_ab = self.d(a, b);
            for (c, d_ac) in cand.neighbors(a) {
                if d_ac >= d_ab - EPS {
                    break;
                }
                if c == b {
                    continue;
                }
                let cd = self.succ(c);
                if cd == a || self.is_frozen(c, cd) {
                    continue;
                }
                let gain = d_ab + self.d(c, cd) - d_ac - self.d(b, cd);
                if gain > EPS {
                    // ... a b ... c cd ...  ->  ... a c ... b cd ...
                    self.reverse_shorter(
                        self.pos[b as usize],
                        self.pos[c as usize],
                        self.pos[cd as usize],
                        self.pos[a as usize],
                    );
                    self.cur_len -= gain;
                    for v in [a, b, c, cd] {
                        self.mark_dirty(v);
                    }
                    return true;
                }
            }
        }

        // Direction 2: remove predecessor edges (b,a) and (pred(c),c).
        let b = self.pred(a);
        if !self.is_frozen(b, a) {
            let d_ab = self.d(a, b);
            for (c, d_ac) in cand.neighbors(a) {
                if d_ac >= d_ab - EPS {
                    break;
                }
                if c == b {
                    continue;
                }
                let cd = self.pred(c);
                if cd == a || self.is_frozen(cd, c) {
                    continue;
                }
                let gain = d_ab + self.d(cd, c) - d_ac - self.d(b, cd);
                if gain > EPS {
                    // ... b a ... cd c ...  ->  ... b cd ... a c ...
                    self.reverse_shorter(
                        self.pos[a as usize],
                        self.pos[cd as usize],
                        self.pos[c as usize],
                        self.pos[b as usize],
                    );
                    self.cur_len -= gain;
                    for v in [a, b, c, cd] {
                        self.mark_dirty(v);
                    }
                    return true;
                }
            }
        }
        false
    }

    /// Is node `x` inside the segment of `len` nodes starting at position `ps`?
    #[inline(always)]
    fn in_segment(&self, x: u32, ps: u32, len: usize) -> bool {
        let n = self.n();
        ((self.pos[x as usize] as usize + n - ps as usize) % n) < len
    }

    /// Try to improve the tour by relocating (Or-opt) or reversing in place
    /// a short segment with `a` at one end.
    fn try_or_opt(&mut self, a: u32) -> bool {
        let cand = self.cand;
        let n = self.n();
        // seg_dir 0: segment grows forward from a (a is the first node).
        // seg_dir 1: segment grows backward from a (a is the last node).
        for seg_dir in 0..2u8 {
            let mut s = a; // first segment node (tour order)
            let mut e = a; // last segment node (tour order)
            for seg_len in 1..=self.or_opt_max {
                if seg_len + 2 > n {
                    break;
                }
                if seg_len > 1 {
                    if seg_dir == 0 {
                        e = self.succ(e);
                    } else {
                        s = self.pred(s);
                    }
                } else if seg_dir == 1 {
                    // Length-1 segments are direction-independent; skip dup.
                    continue;
                }
                let p = self.pred(s);
                let q = self.succ(e);
                if self.is_frozen(p, s) || self.is_frozen(e, q) {
                    continue;
                }
                let d_ps = self.d(p, s);
                let d_eq = self.d(e, q);
                let d_pq = self.d(p, q);
                let remove_gain = d_ps + d_eq - d_pq;
                if remove_gain <= EPS {
                    continue;
                }

                // In-place reversal (a pure 3-opt "reverse segment" move).
                if seg_len > 1 {
                    let delta = self.d(p, e) + self.d(s, q) - d_pq;
                    if remove_gain - delta > EPS {
                        self.reverse_arc(self.pos[s as usize], self.pos[e as usize]);
                        self.cur_len -= remove_gain - delta;
                        for v in [p, s, e, q] {
                            self.mark_dirty(v);
                        }
                        return true;
                    }
                }

                let ps_pos = self.pos[s as usize];
                // Relocation: try inserting the segment next to a candidate
                // neighbor of either segment end (both ends coincide at len 1).
                let ends: &[(u32, u32)] = if seg_len == 1 {
                    &[(s, e)]
                } else {
                    &[(s, e), (e, s)]
                };
                for &(end, other) in ends {
                    for (c, d_c) in cand.neighbors(end) {
                        // Heuristic prune: the new edge (end,c) alone should
                        // not consume the removal gain.
                        if d_c >= remove_gain - EPS {
                            break;
                        }
                        if self.in_segment(c, ps_pos, seg_len) {
                            continue;
                        }
                        // Slot A: between c and succ(c), `end` adjacent to c.
                        let v = self.succ(c);
                        if !self.in_segment(v, ps_pos, seg_len) && !self.is_frozen(c, v) {
                            let delta = d_c + self.d(other, v) - self.d(c, v);
                            if remove_gain - delta > EPS {
                                self.apply_or_opt(s, e, seg_len, c, end == e);
                                self.cur_len -= remove_gain - delta;
                                for x in [p, q, s, e, c, v] {
                                    self.mark_dirty(x);
                                }
                                return true;
                            }
                        }
                        // Slot B: between pred(c) and c, `end` adjacent to c.
                        let u = self.pred(c);
                        if !self.in_segment(u, ps_pos, seg_len) && !self.is_frozen(u, c) {
                            let delta = self.d(u, other) + d_c - self.d(u, c);
                            if remove_gain - delta > EPS {
                                self.apply_or_opt(s, e, seg_len, u, end == s);
                                self.cur_len -= remove_gain - delta;
                                for x in [p, q, s, e, u, c] {
                                    self.mark_dirty(x);
                                }
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    }

    /// Move the segment `s..=e` (`len` nodes, tour-forward order) so it sits
    /// between `u` and `succ(u)`. If `reversed`, the segment is flipped.
    /// `u` must not be inside the segment and `succ(u)` must not be `s`.
    fn apply_or_opt(&mut self, s: u32, e: u32, len: usize, u: u32, reversed: bool) {
        let n = self.n();
        self.buf.clear();
        let mut node = s;
        for _ in 0..len {
            self.buf.push(node);
            node = self.succ(node);
        }
        if reversed {
            self.buf.reverse();
        }

        let ps = self.pos[s as usize];
        let pe = self.pos[e as usize];
        let v = self.succ(u);
        let fspan = (self.pos[u as usize] as usize + n - pe as usize) % n;
        let bspan = (ps as usize + n - self.pos[v as usize] as usize) % n;
        debug_assert!(fspan >= 1 && bspan >= 1);

        if fspan <= bspan {
            // Shift the nodes between the segment and u backward by `len`,
            // then place the segment right after u.
            let mut write = ps;
            let mut read = self.succ_pos(pe);
            for _ in 0..fspan {
                let x = self.order[read as usize];
                self.order[write as usize] = x;
                self.pos[x as usize] = write;
                write = self.succ_pos(write);
                read = self.succ_pos(read);
            }
            for i in 0..len {
                let x = self.buf[i];
                self.order[write as usize] = x;
                self.pos[x as usize] = write;
                write = self.succ_pos(write);
            }
        } else {
            // Shift the nodes between v and the segment forward by `len`,
            // then place the segment right before v's new location.
            let mut write = pe;
            let mut read = self.pred_pos(ps);
            for _ in 0..bspan {
                let x = self.order[read as usize];
                self.order[write as usize] = x;
                self.pos[x as usize] = write;
                write = self.pred_pos(write);
                read = self.pred_pos(read);
            }
            for i in (0..len).rev() {
                let x = self.buf[i];
                self.order[write as usize] = x;
                self.pos[x as usize] = write;
                write = self.pred_pos(write);
            }
        }
    }

    /// Classic double-bridge perturbation, windowed. Returns true if applied.
    fn double_bridge(&mut self) -> bool {
        let n = self.n();
        if n < 8 {
            return false;
        }
        let w = self.kick_window.clamp(2, (n - 2) / 3);
        'attempt: for _ in 0..10 {
            let base = self.rng.next_below(n) as u32;
            let g1 = 1 + self.rng.next_below(w);
            let g2 = 1 + self.rng.next_below(w);
            let g3 = 1 + self.rng.next_below(w);
            if g1 + g2 + g3 > n - 1 {
                continue;
            }
            let i = (base as usize + g1) % n;
            let j = (i + g2) % n;
            let k = (j + g3) % n;
            // Removed edges: (i-1,i), (j-1,j), (k-1,k).
            let ni = self.order[i];
            let nj = self.order[j];
            let nk = self.order[k];
            let pi = self.order[(i + n - 1) % n];
            let pj = self.order[(j + n - 1) % n];
            let pk = self.order[(k + n - 1) % n];
            for (x, y) in [(pi, ni), (pj, nj), (pk, nk)] {
                if self.is_frozen(x, y) {
                    continue 'attempt;
                }
            }
            let delta = self.d(pi, nj) + self.d(pk, ni) + self.d(pj, nk)
                - self.d(pi, ni)
                - self.d(pj, nj)
                - self.d(pk, nk);

            // Rewrite positions i..k as segment C (j..k) then B (i..j).
            self.buf.clear();
            let mut r = j;
            for _ in 0..g3 {
                self.buf.push(self.order[r]);
                r = (r + 1) % n;
            }
            let mut r = i;
            for _ in 0..g2 {
                self.buf.push(self.order[r]);
                r = (r + 1) % n;
            }
            let mut write = i;
            for idx in 0..self.buf.len() {
                let x = self.buf[idx];
                self.order[write] = x;
                self.pos[x as usize] = write as u32;
                write = (write + 1) % n;
            }
            self.cur_len += delta;
            for x in [pi, ni, pj, nj, pk, nk] {
                self.mark_dirty(x);
            }
            return true;
        }
        false
    }

    /// Drain the work queue, applying improving moves until none remain.
    /// Returns false if interrupted by the deadline.
    pub fn run(&mut self, deadline: Instant) -> bool {
        let mut steps: usize = 0;
        while let Some(v) = self.queue.pop_front() {
            self.in_queue[v as usize] = false;
            if self.dont_look[v as usize] {
                continue;
            }
            loop {
                if self.try_two_opt(v) {
                    continue;
                }
                if self.try_or_opt(v) {
                    continue;
                }
                break;
            }
            self.dont_look[v as usize] = true;
            steps += 1;
            if steps & 0x3FF == 0 && Instant::now() > deadline {
                return false;
            }
        }
        true
    }

    /// Local search to convergence, then iterated local search: double-bridge
    /// kicks with reopt, keeping the best tour found.
    pub fn optimize(&mut self, deadline: Instant, max_kicks: usize) {
        self.activate_all();
        self.run(deadline);
        if max_kicks == 0 || self.n() < 8 {
            return;
        }
        let mut best = self.order.clone();
        let mut best_len = self.cur_len;
        let mut kicks = 0;
        while kicks < max_kicks && Instant::now() < deadline {
            if !self.double_bridge() {
                break;
            }
            self.run(deadline);
            if self.cur_len < best_len - EPS {
                best_len = self.cur_len;
                best.copy_from_slice(&self.order);
            } else if self.cur_len > best_len + EPS {
                self.restore(&best, best_len);
            }
            kicks += 1;
        }
        if self.cur_len > best_len + EPS {
            self.restore(&best, best_len);
        }
    }

    fn restore(&mut self, saved: &[u32], saved_len: f64) {
        self.order.copy_from_slice(saved);
        for (i, &v) in self.order.iter().enumerate() {
            self.pos[v as usize] = i as u32;
        }
        self.cur_len = saved_len;
        self.queue.clear();
        self.in_queue.fill(false);
        self.dont_look.fill(true);
    }

    /// Walk the cycle from `s` to `t` without crossing the frozen edge,
    /// returning the path (used to write an optimized segment back into the
    /// global tour). `s` and `t` must be tour-adjacent.
    pub fn path_from(&self, s: u32, t: u32) -> Vec<u32> {
        let n = self.n();
        debug_assert!(self.succ(s) == t || self.pred(s) == t);
        let forward = self.succ(s) != t;
        let mut out = Vec::with_capacity(n);
        let mut p = self.pos[s as usize];
        for _ in 0..n {
            out.push(self.order[p as usize]);
            p = if forward {
                self.succ_pos(p)
            } else {
                self.pred_pos(p)
            };
        }
        debug_assert_eq!(*out.last().unwrap(), t);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidates::Candidates;
    use crate::construct::greedy_tour;
    use crate::kdtree::KdTree;

    fn random_points(n: usize, seed: u64) -> Vec<[f64; 2]> {
        let mut rng = SplitMix64::new(seed);
        (0..n)
            .map(|_| {
                [
                    rng.next_below(100_000) as f64 / 100.0,
                    rng.next_below(100_000) as f64 / 100.0,
                ]
            })
            .collect()
    }

    fn assert_valid(state: &TourState<'_, 2>) {
        let n = state.order.len();
        let mut seen = vec![false; n];
        for (i, &v) in state.order.iter().enumerate() {
            assert!(!seen[v as usize], "node {v} twice");
            seen[v as usize] = true;
            assert_eq!(state.pos[v as usize] as usize, i, "pos inverse broken");
        }
        let real = state.tour_length();
        assert!(
            (real - state.cur_len).abs() < 1e-4 * (1.0 + real),
            "incremental length drifted: {} vs {}",
            state.cur_len,
            real
        );
    }

    fn make_state<'a>(
        pts: &'a [[f64; 2]],
        cand: &'a Candidates,
        seed: u64,
        frozen: Option<(u32, u32)>,
    ) -> TourState<'a, 2> {
        let order: Vec<u32> = (0..pts.len() as u32).collect();
        TourState::new(pts, cand, order, frozen, 3, 50, SplitMix64::new(seed))
    }

    #[test]
    fn or_opt_move_mechanics_match_naive() {
        // Randomized cross-check of apply_or_opt against a naive Vec rebuild.
        let pts = random_points(60, 21);
        let tree = KdTree::build(&pts);
        let cand = Candidates::build(&pts, &tree, 6, 12);
        let mut rng = SplitMix64::new(77);
        for trial in 0..2000 {
            let mut state = make_state(&pts, &cand, trial, None);
            // Shuffle the tour so segments land everywhere incl. wraparound.
            for i in (1..state.order.len()).rev() {
                let j = rng.next_below(i + 1);
                state.order.swap(i, j);
            }
            for (i, &v) in state.order.iter().enumerate() {
                state.pos[v as usize] = i as u32;
            }
            state.cur_len = state.tour_length();
            let n = state.order.len();
            let len = 1 + rng.next_below(3);
            let ps = rng.next_below(n);
            let s = state.order[ps];
            let e = state.order[(ps + len - 1) % n];
            // Pick u outside the segment with succ(u) also outside.
            let mut u_pos = (ps + len + rng.next_below(n - len - 1)) % n;
            if (u_pos + 1) % n == ps {
                u_pos = (ps + len) % n;
                if u_pos == ps || (u_pos + 1) % n == ps {
                    continue;
                }
            }
            let u = state.order[u_pos];
            let reversed = rng.next_below(2) == 1;

            // Naive expectation.
            let mut seg = Vec::new();
            for k in 0..len {
                seg.push(state.order[(ps + k) % n]);
            }
            if reversed {
                seg.reverse();
            }
            let mut rest: Vec<u32> = Vec::new();
            for k in 0..n {
                let idx = (ps + len + k) % n;
                if idx == ps {
                    break;
                }
                rest.push(state.order[idx]);
            }
            let upos_in_rest = rest.iter().position(|&x| x == u).unwrap();
            let mut expect = rest[..=upos_in_rest].to_vec();
            expect.extend_from_slice(&seg);
            expect.extend_from_slice(&rest[upos_in_rest + 1..]);

            state.apply_or_opt(s, e, len, u, reversed);
            state.cur_len = state.tour_length();
            assert_valid(&state);

            // Compare as rotations of the same cycle.
            let got = &state.order;
            let start = got.iter().position(|&x| x == expect[0]).unwrap();
            let rotated: Vec<u32> = (0..n).map(|k| got[(start + k) % n]).collect();
            assert_eq!(rotated, expect, "trial {trial}");
        }
    }

    #[test]
    fn local_search_reduces_length_and_stays_valid() {
        let pts = random_points(600, 5);
        let tree = KdTree::build(&pts);
        let cand = Candidates::build(&pts, &tree, 8, 16);
        let mut state = make_state(&pts, &cand, 1, None);
        let before = state.cur_len;
        state.activate_all();
        state.run(Instant::now() + std::time::Duration::from_secs(30));
        assert_valid(&state);
        assert!(state.cur_len < before * 0.7, "should improve substantially");
    }

    #[test]
    fn optimize_with_kicks_improves_over_plain_descent() {
        let pts = random_points(300, 6);
        let tree = KdTree::build(&pts);
        let cand = Candidates::build(&pts, &tree, 8, 16);

        let mut plain = make_state(&pts, &cand, 2, None);
        plain.activate_all();
        plain.run(Instant::now() + std::time::Duration::from_secs(30));

        let mut kicked = make_state(&pts, &cand, 2, None);
        kicked.optimize(Instant::now() + std::time::Duration::from_secs(30), 400);
        assert_valid(&kicked);
        assert!(kicked.cur_len <= plain.cur_len + EPS);
    }

    #[test]
    fn frozen_edge_survives_optimization() {
        let pts = random_points(200, 9);
        let tree = KdTree::build(&pts);
        let cand = Candidates::build(&pts, &tree, 8, 16);
        let order: Vec<u32> = greedy_tour(&pts, &cand, &tree);
        let s = order[0];
        let t = *order.last().unwrap();
        let mut state = TourState::new(&pts, &cand, order, Some((t, s)), 3, 50, SplitMix64::new(3));
        state.optimize(Instant::now() + std::time::Duration::from_secs(10), 200);
        assert_valid(&state);
        assert!(
            state.succ(t) == s || state.pred(t) == s,
            "frozen edge must remain adjacent"
        );
        let path = state.path_from(s, t);
        assert_eq!(path.len(), pts.len());
        assert_eq!(path[0], s);
        assert_eq!(*path.last().unwrap(), t);
    }

    #[test]
    fn double_bridge_keeps_permutation_and_length_tracking() {
        let pts = random_points(120, 4);
        let tree = KdTree::build(&pts);
        let cand = Candidates::build(&pts, &tree, 6, 12);
        let mut state = make_state(&pts, &cand, 8, None);
        for _ in 0..200 {
            state.double_bridge();
        }
        assert_valid(&state);
    }
}
