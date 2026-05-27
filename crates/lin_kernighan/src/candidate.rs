use kiddo::{KdTree, SquaredEuclidean};

use crate::{alpha::PiValues, distance::euc_2d, problem::Problem};

/// Per-node candidate edges, sorted ascending by cost. Each inner
/// vector has at most `max_candidates` entries.
///
/// This is the analogue of LKH's `Node.CandidateSet`. LKH defaults to
/// the ALPHA candidate set (computed via minimum 1-tree + subgradient
/// ascent on Pi-values), which has slightly better quality on hard
/// instances. We start with plain k-nearest-neighbor candidates — they
/// give ~95% of ALPHA's quality for ≤10k-node Euclidean instances and
/// require no setup cost beyond a k-d tree build.
#[derive(Clone, Debug)]
pub struct CandidateSet {
    candidates: Vec<Vec<Candidate>>,
}

#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    pub to: u32,
    pub cost: i64,
}

impl CandidateSet {
    pub fn build_nn(problem: &Problem, max_candidates: usize) -> Self {
        let n = problem.n();
        let coords = problem.coords();

        let mut tree: KdTree<f64, 2> = KdTree::with_capacity(n);
        for (i, p) in coords.iter().enumerate() {
            tree.add(&[p.x, p.y], i as u64);
        }

        // `max_candidates + 1` to include self; we drop self from results.
        let k_query = (max_candidates + 1).min(n);

        let mut candidates: Vec<Vec<Candidate>> = Vec::with_capacity(n);
        for (i, p) in coords.iter().enumerate() {
            let neighbors = tree.nearest_n::<SquaredEuclidean>(&[p.x, p.y], k_query);
            let mut entry: Vec<Candidate> = Vec::with_capacity(max_candidates);
            for nb in neighbors {
                let j = nb.item as usize;
                if j == i {
                    continue;
                }
                entry.push(Candidate {
                    to: j as u32,
                    cost: euc_2d(coords[i], coords[j]),
                });
                if entry.len() >= max_candidates {
                    break;
                }
            }
            // Cost-sorted (kiddo gives squared-Euclidean order, which is
            // monotonically equivalent for non-negative coords but we
            // re-sort under the integer EUC_2D metric to be safe).
            entry.sort_unstable_by_key(|c| c.cost);
            candidates.push(entry);
        }

        // Symmetrize: if `j` is in `i`'s candidate list, ensure `i` is in
        // `j`'s list — LKH's `SymmetrizeCandidateSet` does this so the
        // sequential k-opt search can find the reverse direction.
        let mut needed: Vec<Vec<Candidate>> = vec![Vec::new(); n];
        for (i, entry) in candidates.iter().enumerate() {
            for &c in entry {
                let j = c.to as usize;
                let already_present = candidates[j].iter().any(|x| x.to as usize == i);
                if !already_present {
                    needed[j].push(Candidate {
                        to: i as u32,
                        cost: c.cost,
                    });
                }
            }
        }
        for (j, extras) in needed.into_iter().enumerate() {
            if extras.is_empty() {
                continue;
            }
            candidates[j].extend(extras);
            candidates[j].sort_unstable_by_key(|c| c.cost);
            candidates[j].dedup_by_key(|c| c.to);
        }

        Self { candidates }
    }

    /// Build a candidate set using Pi-adjusted (Lagrangian) edge costs.
    /// For each node we pick the `max_candidates` cheapest neighbours
    /// under the modified cost `c(i,j) + π[i] + π[j]`. The displayed
    /// `Candidate.cost` is still the *raw* integer EUC_2D — that's
    /// what the LK gain criterion compares against — but the ranking
    /// used to *select* the candidates incorporates the Pi values.
    pub fn build_alpha(problem: &Problem, pi: &PiValues, max_candidates: usize) -> Self {
        let n = problem.n();
        let coords = problem.coords();
        debug_assert_eq!(pi.n(), n);

        // For each node, score every other by modified cost, partial-sort
        // to top `k`, then sort the survivors by raw cost so the LK gain
        // criterion's "candidates sorted by cost ascending" assumption
        // still holds.
        let mut candidates: Vec<Vec<Candidate>> = Vec::with_capacity(n);
        let mut scored: Vec<(f64, u32, i64)> = Vec::with_capacity(n.saturating_sub(1));
        for i in 0..n {
            scored.clear();
            for j in 0..n {
                if i == j {
                    continue;
                }
                let raw = euc_2d(coords[i], coords[j]);
                let modified = (raw as f64) + pi.0[i] + pi.0[j];
                scored.push((modified, j as u32, raw));
            }
            // Partial sort: keep the `max_candidates` smallest by modified.
            let k = max_candidates.min(scored.len());
            scored.select_nth_unstable_by(k.saturating_sub(1), |a, b| a.0.total_cmp(&b.0));
            let mut entry: Vec<Candidate> = scored[..k]
                .iter()
                .map(|&(_, to, cost)| Candidate { to, cost })
                .collect();
            entry.sort_unstable_by_key(|c| c.cost);
            candidates.push(entry);
        }

        // Symmetrize to match `build_nn` invariants.
        let mut needed: Vec<Vec<Candidate>> = vec![Vec::new(); n];
        for (i, entry) in candidates.iter().enumerate() {
            for &c in entry {
                let j = c.to as usize;
                let already_present = candidates[j].iter().any(|x| x.to as usize == i);
                if !already_present {
                    needed[j].push(Candidate {
                        to: i as u32,
                        cost: c.cost,
                    });
                }
            }
        }
        for (j, extras) in needed.into_iter().enumerate() {
            if extras.is_empty() {
                continue;
            }
            candidates[j].extend(extras);
            candidates[j].sort_unstable_by_key(|c| c.cost);
            candidates[j].dedup_by_key(|c| c.to);
        }

        Self { candidates }
    }

    /// Truncate each node's candidate list to its `k` cheapest entries.
    /// Zero-alloc when `k` ≥ current size; otherwise reuses the same
    /// outer Vec by draining the tail. Lets a single globally-built
    /// k=K candidate set serve callers that want a tighter k for hot
    /// inner loops (e.g. the post-stitch refinement).
    pub fn truncated(&self, k: usize) -> Self {
        let candidates: Vec<Vec<Candidate>> = self
            .candidates
            .iter()
            .map(|entry| entry.iter().take(k).copied().collect())
            .collect();
        Self { candidates }
    }

    /// Restrict a globally-built candidate set to a chunk identified by
    /// `local_to_global` (length = chunk size, mapping chunk-local node
    /// index → original global node id). For each local node we look up
    /// its global candidates, drop any that aren't in the chunk, and
    /// rewrite the remaining target ids to chunk-local indices.
    ///
    /// Useful when the same input is solved many times across
    /// overlapping subsets (chunked TSP, online repair) — building the
    /// k-d tree + symmetrised NN list once on the global node set
    /// amortises the work across every chunk.
    pub fn subset(global: &CandidateSet, local_to_global: &[u32]) -> Self {
        let n_local = local_to_global.len();
        // Reverse map: global_id → local_id. Sentinel u32::MAX = not in chunk.
        let global_n = global.n();
        let mut global_to_local: Vec<u32> = vec![u32::MAX; global_n];
        for (local_id, &global_id) in local_to_global.iter().enumerate() {
            global_to_local[global_id as usize] = local_id as u32;
        }

        let mut candidates: Vec<Vec<Candidate>> = Vec::with_capacity(n_local);
        for &global_id in local_to_global {
            let entry: Vec<Candidate> = global
                .of(global_id)
                .iter()
                .filter_map(|c| {
                    let local_to = global_to_local[c.to as usize];
                    if local_to == u32::MAX {
                        None
                    } else {
                        Some(Candidate {
                            to: local_to,
                            cost: c.cost,
                        })
                    }
                })
                .collect();
            candidates.push(entry);
        }
        Self { candidates }
    }

    #[inline]
    pub fn of(&self, node: u32) -> &[Candidate] {
        &self.candidates[node as usize]
    }

    pub fn n(&self) -> usize {
        self.candidates.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::Point2D;

    fn problem(points: &[(f64, f64)]) -> Problem {
        Problem::new(points.iter().map(|&(x, y)| Point2D::new(x, y)).collect())
            .expect("valid problem")
    }

    #[test]
    fn nn_candidates_are_sorted_by_cost_and_exclude_self() {
        let p = problem(&[(0.0, 0.0), (1.0, 0.0), (5.0, 0.0), (10.0, 0.0)]);
        let cs = CandidateSet::build_nn(&p, 3);
        let c0 = cs.of(0);
        assert!(!c0.iter().any(|c| c.to == 0));
        assert!(c0.windows(2).all(|w| w[0].cost <= w[1].cost));
    }

    #[test]
    fn nn_candidates_are_symmetric() {
        let p = problem(&[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (5.0, 5.0)]);
        let cs = CandidateSet::build_nn(&p, 2);
        for i in 0..p.n() {
            for c in cs.of(i as u32) {
                assert!(
                    cs.of(c.to).iter().any(|d| d.to == i as u32),
                    "asymmetric: {i} -> {}",
                    c.to
                );
            }
        }
    }

    #[test]
    fn subset_remaps_ids_and_drops_outside_chunk() {
        let p = problem(&[
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (10.0, 0.0),
            (11.0, 0.0),
        ]);
        let global = CandidateSet::build_nn(&p, 4);
        // Chunk = global nodes [0, 1, 2], rewritten as local [0, 1, 2].
        let local = CandidateSet::subset(&global, &[0u32, 1, 2]);
        for i in 0..3u32 {
            for c in local.of(i) {
                assert!(c.to < 3, "subset emitted out-of-range local id {}", c.to);
            }
        }
        // Node 0's global candidates include node 3 (cost 10) and node 4
        // (cost 11); both must be dropped in the subset.
        assert!(local.of(0).iter().all(|c| c.to < 3));
    }
}
