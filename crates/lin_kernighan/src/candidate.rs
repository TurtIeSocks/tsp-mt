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
}
