//! Greedy tour construction: add the shortest candidate edges first (as long
//! as they keep degree <= 2 and don't close a premature cycle), then chain
//! the leftover path fragments end-to-end by nearest endpoint.
//!
//! Greedy matching typically starts 10-20% above the optimal tour length,
//! substantially better than nearest-neighbor, and gives local search a
//! spatially coherent tour to refine.

use rayon::prelude::*;

use crate::candidates::Candidates;
use crate::kdtree::KdTree;

pub fn greedy_tour<const D: usize>(
    pts: &[[f64; D]],
    cand: &Candidates,
    tree: &KdTree<'_, D>,
) -> Vec<u32> {
    let n = pts.len();
    if n <= 3 {
        return (0..n as u32).collect();
    }

    let mut edges = cand.undirected_edges();
    edges.par_sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then((a.1, a.2).cmp(&(b.1, b.2))));

    let mut uf = UnionFind::new(n);
    let mut degree = vec![0u8; n];
    // Up to two tour neighbors per node; u32::MAX = empty slot.
    let mut adj = vec![[u32::MAX; 2]; n];

    let link = |adj: &mut Vec<[u32; 2]>, a: u32, b: u32| {
        let slots = &mut adj[a as usize];
        if slots[0] == u32::MAX {
            slots[0] = b;
        } else {
            slots[1] = b;
        }
    };

    for &(_, a, b) in &edges {
        if degree[a as usize] >= 2 || degree[b as usize] >= 2 {
            continue;
        }
        if uf.find(a) == uf.find(b) {
            continue;
        }
        uf.union(a, b);
        degree[a as usize] += 1;
        degree[b as usize] += 1;
        link(&mut adj, a, b);
        link(&mut adj, b, a);
    }

    // Collect path fragments (isolated nodes count as length-1 fragments).
    let mut visited = vec![false; n];
    let mut fragments: Vec<Vec<u32>> = Vec::new();
    for start in 0..n as u32 {
        if visited[start as usize] || degree[start as usize] == 2 {
            continue;
        }
        // Walk from an endpoint (degree 0 or 1) to the other end.
        let mut path = Vec::new();
        let mut prev = u32::MAX;
        let mut cur = start;
        loop {
            visited[cur as usize] = true;
            path.push(cur);
            let [x, y] = adj[cur as usize];
            let next = if x != prev && x != u32::MAX {
                x
            } else if y != prev && y != u32::MAX {
                y
            } else {
                break;
            };
            prev = cur;
            cur = next;
        }
        fragments.push(path);
    }
    debug_assert_eq!(
        fragments.iter().map(|f| f.len()).sum::<usize>(),
        n,
        "greedy edges must form open paths covering all nodes"
    );

    if fragments.len() == 1 {
        return fragments.pop().unwrap();
    }

    // Chain fragments: from the current chain end, repeatedly jump to the
    // nearest endpoint of any remaining fragment.
    let mut frag_of = vec![u32::MAX; n];
    let mut endpoint_active = vec![false; n];
    for (fi, frag) in fragments.iter().enumerate() {
        let (first, last) = (frag[0], *frag.last().unwrap());
        frag_of[first as usize] = fi as u32;
        frag_of[last as usize] = fi as u32;
        endpoint_active[first as usize] = true;
        endpoint_active[last as usize] = true;
    }

    let mut fragments: Vec<Option<Vec<u32>>> = fragments.into_iter().map(Some).collect();
    let mut chain = fragments[0].take().unwrap();
    endpoint_active[chain[0] as usize] = false;
    endpoint_active[*chain.last().unwrap() as usize] = false;
    let mut remaining = fragments.len() - 1;

    while remaining > 0 {
        let end = *chain.last().unwrap();
        let end_pt = &pts[end as usize];
        let found = tree.nearest_filtered(end_pt, |v| endpoint_active[v as usize]);
        let Some((_, hit)) = found else {
            unreachable!("active endpoints remain but none found");
        };
        let fi = frag_of[hit as usize] as usize;
        let mut frag = fragments[fi].take().expect("fragment consumed twice");
        endpoint_active[frag[0] as usize] = false;
        endpoint_active[*frag.last().unwrap() as usize] = false;
        if frag[0] != hit {
            frag.reverse();
        }
        debug_assert_eq!(frag[0], hit);
        chain.append(&mut frag);
        remaining -= 1;
    }

    debug_assert_eq!(chain.len(), n);
    chain
}

struct UnionFind {
    parent: Vec<u32>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
        }
    }

    fn find(&mut self, mut v: u32) -> u32 {
        while self.parent[v as usize] != v {
            let grand = self.parent[self.parent[v as usize] as usize];
            self.parent[v as usize] = grand;
            v = grand;
        }
        v
    }

    fn union(&mut self, a: u32, b: u32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent[ra as usize] = rb;
        }
    }
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
                    rng.next_below(100_000) as f64 / 100.0,
                    rng.next_below(100_000) as f64 / 100.0,
                ]
            })
            .collect()
    }

    fn is_permutation(tour: &[u32], n: usize) -> bool {
        let mut seen = vec![false; n];
        for &v in tour {
            if seen[v as usize] {
                return false;
            }
            seen[v as usize] = true;
        }
        tour.len() == n
    }

    #[test]
    fn produces_valid_permutation() {
        for seed in [1, 2, 3] {
            let pts = random_points(777, seed);
            let tree = KdTree::build(&pts);
            let cand = Candidates::build(&pts, &tree, 8, 16);
            let tour = greedy_tour(&pts, &cand, &tree);
            assert!(is_permutation(&tour, pts.len()));
        }
    }

    #[test]
    fn beats_random_order_substantially() {
        let pts = random_points(2000, 42);
        let tree = KdTree::build(&pts);
        let cand = Candidates::build(&pts, &tree, 8, 16);
        let tour = greedy_tour(&pts, &cand, &tree);

        let len = |order: &[u32]| -> f64 {
            (0..order.len())
                .map(|i| {
                    let a = &pts[order[i] as usize];
                    let b = &pts[order[(i + 1) % order.len()] as usize];
                    crate::kdtree::dist(a, b)
                })
                .sum()
        };
        let identity: Vec<u32> = (0..pts.len() as u32).collect();
        assert!(len(&tour) < 0.5 * len(&identity));
    }

    #[test]
    fn tiny_inputs_pass_through() {
        let pts = random_points(3, 9);
        let tree = KdTree::build(&pts);
        let cand = Candidates::build(&pts, &tree, 8, 16);
        let tour = greedy_tour(&pts, &cand, &tree);
        assert!(is_permutation(&tour, 3));
    }
}
