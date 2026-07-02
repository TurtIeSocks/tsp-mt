//! Static k-d tree over a point slice, used for k-nearest-neighbor candidate
//! generation and nearest-endpoint queries during tour construction.
//!
//! Layout: `idx` is a permutation of point indices arranged as an implicit
//! balanced tree. For a range `[lo, hi)` the median position `lo + (hi-lo)/2`
//! holds the splitting point; everything left of it is <= the split value on
//! the split dimension, everything right is >=.

use alloc::vec;
use alloc::vec::Vec;

#[cfg(feature = "parallel")]
use rayon::join;

pub const LEAF_SIZE: usize = 8;
#[cfg(feature = "parallel")]
const PARALLEL_BUILD_MIN: usize = 4096;

/// `f64::sqrt` is a std intrinsic; route through libm without std.
#[inline(always)]
pub(crate) fn sqrt(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sqrt(x)
    }
}

pub struct KdTree<'a, const D: usize> {
    pts: &'a [[f64; D]],
    idx: Vec<u32>,
    split_dim: Vec<u8>,
}

#[inline(always)]
pub fn dist_sq<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    let mut acc = 0.0;
    for d in 0..D {
        let diff = a[d] - b[d];
        acc += diff * diff;
    }
    acc
}

#[inline(always)]
pub fn dist<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    sqrt(dist_sq(a, b))
}

impl<'a, const D: usize> KdTree<'a, D> {
    pub fn build(pts: &'a [[f64; D]]) -> Self {
        let mut idx: Vec<u32> = (0..pts.len() as u32).collect();
        let mut split_dim = vec![0u8; pts.len()];
        build_range(pts, &mut idx, &mut split_dim);
        Self {
            pts,
            idx,
            split_dim,
        }
    }

    /// K nearest neighbors of `query`, excluding the point index `skip`.
    /// Results are appended to `out` sorted by ascending distance as
    /// `(distance, point_index)` pairs. `out` is cleared first.
    pub fn knn(&self, query: &[f64; D], k: usize, skip: u32, out: &mut Vec<(f64, u32)>) {
        out.clear();
        if k == 0 || self.pts.is_empty() {
            return;
        }
        self.knn_range(0, self.idx.len(), query, k, skip, out);
        for entry in out.iter_mut() {
            entry.0 = sqrt(entry.0);
        }
    }

    fn knn_range(
        &self,
        lo: usize,
        hi: usize,
        query: &[f64; D],
        k: usize,
        skip: u32,
        best: &mut Vec<(f64, u32)>,
    ) {
        if hi - lo <= LEAF_SIZE {
            for &node in &self.idx[lo..hi] {
                if node != skip {
                    push_candidate(best, k, dist_sq(query, &self.pts[node as usize]), node);
                }
            }
            return;
        }
        let mid = lo + (hi - lo) / 2;
        let node = self.idx[mid];
        if node != skip {
            push_candidate(best, k, dist_sq(query, &self.pts[node as usize]), node);
        }
        let dim = self.split_dim[mid] as usize;
        let delta = query[dim] - self.pts[node as usize][dim];
        let (near_lo, near_hi, far_lo, far_hi) = if delta < 0.0 {
            (lo, mid, mid + 1, hi)
        } else {
            (mid + 1, hi, lo, mid)
        };
        self.knn_range(near_lo, near_hi, query, k, skip, best);
        if best.len() < k || delta * delta < best[best.len() - 1].0 {
            self.knn_range(far_lo, far_hi, query, k, skip, best);
        }
    }

    /// Exact nearest point satisfying `accept`, or None if no point does.
    pub fn nearest_filtered<F: Fn(u32) -> bool>(
        &self,
        query: &[f64; D],
        accept: F,
    ) -> Option<(f64, u32)> {
        let mut best: Option<(f64, u32)> = None;
        self.nearest_filtered_range(0, self.idx.len(), query, &accept, &mut best);
        best.map(|(d2, i)| (sqrt(d2), i))
    }

    fn nearest_filtered_range<F: Fn(u32) -> bool>(
        &self,
        lo: usize,
        hi: usize,
        query: &[f64; D],
        accept: &F,
        best: &mut Option<(f64, u32)>,
    ) {
        if hi - lo <= LEAF_SIZE {
            for &node in &self.idx[lo..hi] {
                if accept(node) {
                    let d2 = dist_sq(query, &self.pts[node as usize]);
                    if best.is_none_or(|(bd, _)| d2 < bd) {
                        *best = Some((d2, node));
                    }
                }
            }
            return;
        }
        let mid = lo + (hi - lo) / 2;
        let node = self.idx[mid];
        if accept(node) {
            let d2 = dist_sq(query, &self.pts[node as usize]);
            if best.is_none_or(|(bd, _)| d2 < bd) {
                *best = Some((d2, node));
            }
        }
        let dim = self.split_dim[mid] as usize;
        let delta = query[dim] - self.pts[node as usize][dim];
        let (near_lo, near_hi, far_lo, far_hi) = if delta < 0.0 {
            (lo, mid, mid + 1, hi)
        } else {
            (mid + 1, hi, lo, mid)
        };
        self.nearest_filtered_range(near_lo, near_hi, query, accept, best);
        if best.is_none_or(|(bd, _)| delta * delta < bd) {
            self.nearest_filtered_range(far_lo, far_hi, query, accept, best);
        }
    }
}

/// Insert `(d2, node)` into `best` (sorted ascending, at most `k` entries).
#[inline]
fn push_candidate(best: &mut Vec<(f64, u32)>, k: usize, d2: f64, node: u32) {
    if best.len() == k && d2 >= best[k - 1].0 {
        return;
    }
    let at = best.partition_point(|&(bd, _)| bd <= d2);
    if best.len() == k {
        best.pop();
    }
    best.insert(at, (d2, node));
}

fn build_range<const D: usize>(pts: &[[f64; D]], idx: &mut [u32], split_dim: &mut [u8]) {
    let len = idx.len();
    if len <= LEAF_SIZE {
        return;
    }
    // Split on the dimension with the largest spread in this range.
    let mut min = [f64::INFINITY; D];
    let mut max = [f64::NEG_INFINITY; D];
    for &i in idx.iter() {
        let p = &pts[i as usize];
        for d in 0..D {
            min[d] = min[d].min(p[d]);
            max[d] = max[d].max(p[d]);
        }
    }
    let mut dim = 0;
    let mut spread = max[0] - min[0];
    for d in 1..D {
        let s = max[d] - min[d];
        if s > spread {
            spread = s;
            dim = d;
        }
    }
    let mid = len / 2;
    idx.select_nth_unstable_by(mid, |&a, &b| {
        pts[a as usize][dim].total_cmp(&pts[b as usize][dim])
    });
    split_dim[mid] = dim as u8;

    let (idx_left, idx_rest) = idx.split_at_mut(mid);
    let (_, idx_right) = idx_rest.split_at_mut(1);
    let (dim_left, dim_rest) = split_dim.split_at_mut(mid);
    let (_, dim_right) = dim_rest.split_at_mut(1);
    #[cfg(feature = "parallel")]
    if len >= PARALLEL_BUILD_MIN {
        join(
            || build_range(pts, idx_left, dim_left),
            || build_range(pts, idx_right, dim_right),
        );
        return;
    }
    build_range(pts, idx_left, dim_left);
    build_range(pts, idx_right, dim_right);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    fn random_points(n: usize, seed: u64) -> Vec<[f64; 2]> {
        let mut rng = SplitMix64::new(seed);
        (0..n)
            .map(|_| {
                [
                    rng.next_below(1_000_000) as f64 / 1000.0,
                    rng.next_below(1_000_000) as f64 / 1000.0,
                ]
            })
            .collect()
    }

    fn brute_knn(pts: &[[f64; 2]], q: &[f64; 2], k: usize, skip: u32) -> Vec<(f64, u32)> {
        let mut all: Vec<(f64, u32)> = pts
            .iter()
            .enumerate()
            .filter(|&(i, _)| i as u32 != skip)
            .map(|(i, p)| (dist(q, p), i as u32))
            .collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        all.truncate(k);
        all
    }

    #[test]
    fn knn_matches_brute_force() {
        let pts = random_points(500, 99);
        let tree = KdTree::build(&pts);
        let mut out = Vec::new();
        for qi in (0..pts.len()).step_by(17) {
            tree.knn(&pts[qi], 8, qi as u32, &mut out);
            let brute = brute_knn(&pts, &pts[qi], 8, qi as u32);
            assert_eq!(out.len(), brute.len());
            for (got, want) in out.iter().zip(brute.iter()) {
                // Distances must match exactly; indices may differ on ties.
                assert!(
                    (got.0 - want.0).abs() < 1e-9,
                    "query {qi}: got {got:?} want {want:?}"
                );
            }
        }
    }

    #[test]
    fn knn_handles_k_larger_than_n() {
        let pts = random_points(5, 3);
        let tree = KdTree::build(&pts);
        let mut out = Vec::new();
        tree.knn(&pts[0], 10, 0, &mut out);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn nearest_filtered_respects_predicate() {
        let pts = random_points(300, 5);
        let tree = KdTree::build(&pts);
        let q = pts[0];
        let allowed = |i: u32| i % 3 == 1;
        let got = tree.nearest_filtered(&q, allowed).unwrap();
        let want = pts
            .iter()
            .enumerate()
            .filter(|&(i, _)| allowed(i as u32))
            .map(|(i, p)| (dist(&q, p), i as u32))
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .unwrap();
        assert!((got.0 - want.0).abs() < 1e-9);
    }

    #[test]
    fn nearest_filtered_returns_none_when_all_rejected() {
        let pts = random_points(50, 6);
        let tree = KdTree::build(&pts);
        assert!(tree.nearest_filtered(&pts[0], |_| false).is_none());
    }

    #[test]
    fn works_with_duplicate_points() {
        let mut pts = random_points(64, 8);
        for i in 0..32 {
            pts[i + 32] = pts[i];
        }
        let tree = KdTree::build(&pts);
        let mut out = Vec::new();
        tree.knn(&pts[0], 4, 0, &mut out);
        assert_eq!(out.len(), 4);
        assert!(out[0].0 < 1e-12, "duplicate should be nearest");
    }
}
