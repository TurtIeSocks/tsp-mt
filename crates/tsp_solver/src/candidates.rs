//! Candidate edge lists: for each node, a short list of nearby nodes that
//! local-search moves are allowed to create edges to. This is the central
//! LKH idea that makes local search near-linear: instead of considering all
//! O(n^2) edges, only geometrically promising ones are examined.
//!
//! Lists are built from k-nearest-neighbor queries and then symmetrized
//! (if j is a candidate of i, i becomes a candidate of j), which improves
//! move coverage for points on cluster boundaries.

use rayon::prelude::*;

use crate::kdtree::KdTree;

pub struct Candidates {
    offsets: Vec<u32>,
    targets: Vec<u32>,
    dists: Vec<f64>,
}

impl Candidates {
    /// Neighbors of `v`, sorted by ascending distance.
    #[inline]
    pub fn neighbors(&self, v: u32) -> impl Iterator<Item = (u32, f64)> + '_ {
        let lo = self.offsets[v as usize] as usize;
        let hi = self.offsets[v as usize + 1] as usize;
        self.targets[lo..hi]
            .iter()
            .copied()
            .zip(self.dists[lo..hi].iter().copied())
    }

    pub fn node_count(&self) -> usize {
        self.offsets.len() - 1
    }

    /// All undirected candidate edges `(dist, a, b)` with `a < b`.
    pub fn undirected_edges(&self) -> Vec<(f64, u32, u32)> {
        let n = self.node_count();
        let mut edges = Vec::with_capacity(self.targets.len() / 2 + n);
        for a in 0..n as u32 {
            for (b, d) in self.neighbors(a) {
                if a < b {
                    edges.push((d, a, b));
                }
            }
        }
        edges
    }

    pub fn build<const D: usize>(
        pts: &[[f64; D]],
        tree: &KdTree<'_, D>,
        k: usize,
        max_per_node: usize,
    ) -> Self {
        let n = pts.len();
        let k = k.min(n.saturating_sub(1));

        // Parallel kNN queries: raw (asymmetric) neighbor lists.
        let raw: Vec<Vec<(f64, u32)>> = pts
            .par_iter()
            .enumerate()
            .map_init(Vec::new, |scratch, (i, p)| {
                tree.knn(p, k, i as u32, scratch);
                scratch.clone()
            })
            .collect();

        // Symmetrize into a CSR structure: each undirected edge contributes
        // to both endpoints, duplicates removed per node. The intermediate
        // buffers (raw lists + unsymmetrized CSR) are scoped so they drop
        // before the final arrays are allocated, roughly halving the
        // transient memory peak on large instances.
        let node_lists: Vec<Vec<(f64, u32)>> = {
            let mut degree = vec![0u32; n];
            for (i, list) in raw.iter().enumerate() {
                degree[i] += list.len() as u32;
                for &(_, j) in list {
                    degree[j as usize] += 1;
                }
            }
            let mut offsets = vec![0u32; n + 1];
            for i in 0..n {
                offsets[i + 1] = offsets[i] + degree[i];
            }
            drop(degree);
            let total = offsets[n] as usize;
            let mut targets = vec![0u32; total];
            let mut dists = vec![0.0f64; total];
            let mut cursor: Vec<u32> = offsets[..n].to_vec();
            for (i, list) in raw.iter().enumerate() {
                for &(d, j) in list {
                    let ci = cursor[i] as usize;
                    targets[ci] = j;
                    dists[ci] = d;
                    cursor[i] += 1;
                    let cj = cursor[j as usize] as usize;
                    targets[cj] = i as u32;
                    dists[cj] = d;
                    cursor[j as usize] += 1;
                }
            }
            drop(cursor);
            drop(raw);

            // Per node: sort by distance, drop duplicate targets, cap the list.
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let lo = offsets[i] as usize;
                    let hi = offsets[i + 1] as usize;
                    let mut list: Vec<(f64, u32)> = targets[lo..hi]
                        .iter()
                        .copied()
                        .zip(dists[lo..hi].iter().copied())
                        .map(|(t, d)| (d, t))
                        .collect();
                    list.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                    list.dedup_by_key(|e| e.1);
                    list.truncate(max_per_node);
                    list.shrink_to_fit();
                    list
                })
                .collect()
        };

        let mut offsets = vec![0u32; n + 1];
        for i in 0..n {
            offsets[i + 1] = offsets[i] + node_lists[i].len() as u32;
        }
        let total = offsets[n] as usize;
        let mut targets = vec![0u32; total];
        let mut dists = vec![0.0f64; total];
        for (i, list) in node_lists.iter().enumerate() {
            let lo = offsets[i] as usize;
            for (slot, &(d, t)) in list.iter().enumerate() {
                targets[lo + slot] = t;
                dists[lo + slot] = d;
            }
        }
        Self {
            offsets,
            targets,
            dists,
        }
    }

    /// Build a candidate structure directly from per-node lists
    /// (used for sub-problem extraction). Lists must be sorted by distance.
    pub fn from_lists(lists: Vec<Vec<(f64, u32)>>) -> Self {
        let n = lists.len();
        let mut offsets = vec![0u32; n + 1];
        for i in 0..n {
            offsets[i + 1] = offsets[i] + lists[i].len() as u32;
        }
        let total = offsets[n] as usize;
        let mut targets = vec![0u32; total];
        let mut dists = vec![0.0f64; total];
        for (i, list) in lists.iter().enumerate() {
            let lo = offsets[i] as usize;
            for (slot, &(d, t)) in list.iter().enumerate() {
                targets[lo + slot] = t;
                dists[lo + slot] = d;
            }
        }
        Self {
            offsets,
            targets,
            dists,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kdtree::dist;
    use crate::rng::SplitMix64;

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

    #[test]
    fn lists_are_symmetric_and_sorted() {
        let pts = random_points(400, 11);
        let tree = KdTree::build(&pts);
        let cand = Candidates::build(&pts, &tree, 6, 12);

        for a in 0..pts.len() as u32 {
            let list: Vec<(u32, f64)> = cand.neighbors(a).collect();
            assert!(!list.is_empty());
            for w in list.windows(2) {
                assert!(w[0].1 <= w[1].1, "list must be sorted by distance");
            }
            for &(b, d) in &list {
                assert_ne!(a, b, "no self-loops");
                assert!((d - dist(&pts[a as usize], &pts[b as usize])).abs() < 1e-9);
            }
        }
        // Symmetry: for most edges both directions exist (capping may drop a
        // few on hub nodes, but with cap 2*k none should be dropped here).
        for a in 0..pts.len() as u32 {
            for (b, _) in cand.neighbors(a) {
                assert!(
                    cand.neighbors(b).any(|(t, _)| t == a),
                    "edge {a}->{b} missing reverse"
                );
            }
        }
    }

    #[test]
    fn no_duplicate_targets() {
        let pts = random_points(200, 12);
        let tree = KdTree::build(&pts);
        let cand = Candidates::build(&pts, &tree, 8, 16);
        for a in 0..pts.len() as u32 {
            let mut seen: Vec<u32> = cand.neighbors(a).map(|(t, _)| t).collect();
            seen.sort();
            let before = seen.len();
            seen.dedup();
            assert_eq!(before, seen.len());
        }
    }

    #[test]
    fn undirected_edges_covers_all_pairs_once() {
        let pts = random_points(100, 13);
        let tree = KdTree::build(&pts);
        let cand = Candidates::build(&pts, &tree, 5, 10);
        let edges = cand.undirected_edges();
        let mut keys: Vec<(u32, u32)> = edges.iter().map(|&(_, a, b)| (a, b)).collect();
        keys.sort();
        let before = keys.len();
        keys.dedup();
        assert_eq!(before, keys.len());
        for &(_, a, b) in &edges {
            assert!(a < b);
        }
    }

    #[test]
    fn tiny_instances_work() {
        let pts = random_points(3, 14);
        let tree = KdTree::build(&pts);
        let cand = Candidates::build(&pts, &tree, 8, 16);
        assert_eq!(cand.node_count(), 3);
        for a in 0..3u32 {
            assert_eq!(cand.neighbors(a).count(), 2);
        }
    }
}
