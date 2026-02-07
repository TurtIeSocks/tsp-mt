use rayon::slice::ParallelSliceMut;

use crate::utils::Point;

pub(crate) fn merge_cycles(a: &[Point], b: &[Point], portals: usize) -> Vec<Point> {
    if a.is_empty() {
        return b.to_vec();
    }
    if b.is_empty() {
        return a.to_vec();
    }

    // Merge by best edge exchange (and try symmetric direction just in case).
    let (ab, ab_delta) = merge_cycles_edge_exchange(a, b, portals);
    let (ba, ba_delta) = merge_cycles_edge_exchange(b, a, portals);
    if ba_delta < ab_delta { ba } else { ab }
}

fn merge_cycles_edge_exchange(a: &[Point], b: &[Point], portals: usize) -> (Vec<Point>, f64) {
    let na = a.len();
    let nb = b.len();
    if na < 2 {
        return (b.to_vec(), 0.0);
    }
    if nb < 2 {
        return (a.to_vec(), 0.0);
    }

    let a_edges = portal_edge_indices(na, portals);
    let b_edges = portal_edge_indices(nb, portals);

    let mut best_delta = f64::INFINITY;
    let mut best_ai = 0usize;
    let mut best_bj = 0usize;
    let mut best_rev_b = false;

    for &ai in &a_edges {
        let a0 = a[ai];
        let a1 = a[(ai + 1) % na];
        let old_a = a0.dist(&a1);

        for &bj in &b_edges {
            let b0 = b[bj];
            let b1 = b[(bj + 1) % nb];
            let old_b = b0.dist(&b1);

            // B forward: connect a0->b1 and b0->a1
            let new_fwd = a0.dist(&b1) + b0.dist(&a1);
            let delta_fwd = new_fwd - (old_a + old_b);

            if delta_fwd < best_delta {
                best_delta = delta_fwd;
                best_ai = ai;
                best_bj = bj;
                best_rev_b = false;
            }

            // B reversed: connect a0->b0 and b1->a1 (equivalent to reversing B view)
            let new_rev = a0.dist(&b0) + b1.dist(&a1);
            let delta_rev = new_rev - (old_a + old_b);

            if delta_rev < best_delta {
                best_delta = delta_rev;
                best_ai = ai;
                best_bj = bj;
                best_rev_b = true;
            }
        }
    }

    // Build output = A_view + B_view
    let mut out = Vec::with_capacity(na + nb);

    // A view starts at a(ai+1) and ends at a(ai)
    out.extend(cycle_view_forward(a, (best_ai + 1) % na));

    // B view starts at b(bj+1) and ends at b(bj)
    let mut b_view: Vec<Point> = cycle_view_forward(b, (best_bj + 1) % nb);
    if best_rev_b {
        b_view.reverse(); // now starts at b(bj) and ends at b(bj+1)
    }
    out.extend_from_slice(&b_view);
    seam_reinsert_repair_cycle(&mut out, na - 1, 24, 12);
    (out, best_delta)
}

fn portal_edge_indices(n: usize, portals: usize) -> Vec<usize> {
    if n <= 2 {
        return vec![0];
    }
    let k = portals.min(n).max(8); // ensure some diversity
    let step = (n as f64) / (k as f64);
    let mut v: Vec<usize> = (0..k)
        .map(|t| ((t as f64) * step).floor() as usize % n)
        .collect();
    v.par_sort_unstable();
    v.dedup();
    if v.is_empty() {
        v.push(0);
    }
    v
}

/// Returns a full cycle traversal starting at `start` (length = n), forward direction.
fn cycle_view_forward(c: &[Point], start: usize) -> Vec<Point> {
    let n = c.len();
    let mut out = Vec::with_capacity(n);
    for t in 0..n {
        out.push(c[(start + t) % n]);
    }
    out
}

fn seam_reinsert_repair_cycle(
    route: &mut Vec<Point>,
    seam_a_end: usize,
    band: usize,
    max_moves: usize,
) {
    // seam_a_end is the index of the last point of A_view in the concatenated route:
    // seam edge 1: seam_a_end -> seam_a_end+1
    // seam edge 2: (n-1) -> 0
    //
    // band: how wide around each seam to consider "suspect" points (e.g. 8..64)
    // max_moves: cap number of successful reinsertions (e.g. 8..32)

    let n = route.len();
    if n < 8 {
        return;
    }
    let band = band.min(n / 4).max(4);
    let mut moved = 0usize;

    // Helper: cycle index wrap
    #[inline]
    fn idx(i: isize, n: usize) -> usize {
        let mut x = i % (n as isize);
        if x < 0 {
            x += n as isize;
        }
        x as usize
    }

    // Remove a node at k and reinsert after edge (j -> j+1) in cycle sense.
    // Returns improvement (negative is better).
    fn delta_relocate_cycle(route: &[Point], k: usize, j: usize) -> Option<f64> {
        let n = route.len();
        if n < 6 {
            return None;
        }

        // can't insert into edges adjacent to the removed node in the current cycle
        let k_prev = (k + n - 1) % n;
        let k_next = (k + 1) % n;

        // edges are (j -> j_next)
        let j_next = (j + 1) % n;

        // Disallow inserting into edges that touch k or its neighbors (would be degenerate)
        if j == k
            || j_next == k
            || j == k_prev
            || j_next == k_prev
            || j == k_next
            || j_next == k_next
        {
            return None;
        }

        let prev = route[k_prev];
        let p = route[k];
        let next = route[k_next];

        let u = route[j];
        let v = route[j_next];

        // remove p: replace prev->p->next with prev->next
        let remove_gain = prev.dist(&next) - prev.dist(&p) - p.dist(&next);

        // insert p into u->v
        let insert_cost = u.dist(&p) + p.dist(&v) - u.dist(&v);

        Some(remove_gain + insert_cost)
    }

    // Apply relocate: remove node at k, then insert it between j and j_next (after j)
    fn apply_relocate_cycle(route: &mut Vec<Point>, k: usize, j: usize) {
        let n = route.len();
        let p = route.remove(k);

        // After removal, indices >= k shift left by 1.
        let n2 = n - 1;

        // We want to insert after edge j->j+1 in the *new* array.
        // If j was after k, it shifted by -1.
        let mut j2 = j;
        if j > k {
            j2 -= 1;
        }

        // Insert after j2 (i.e., at position j2+1)
        let insert_pos = (j2 + 1).min(n2);
        route.insert(insert_pos, p);
    }

    // Build suspect indices around seam endpoints:
    // seam1 endpoints: seam_a_end and seam_a_end+1
    // seam2 endpoints: n-1 and 0
    let seam1_l = seam_a_end % n;
    let seam1_r = (seam_a_end + 1) % n;

    let seam2_l = (n - 1) % n;
    let seam2_r = 0usize;

    let mut suspects: Vec<usize> = Vec::with_capacity(8 * band);

    for d in -(band as isize)..=(band as isize) {
        suspects.push(idx(seam1_l as isize + d, n));
        suspects.push(idx(seam1_r as isize + d, n));
        suspects.push(idx(seam2_l as isize + d, n));
        suspects.push(idx(seam2_r as isize + d, n));
    }

    suspects.sort_unstable();
    suspects.dedup();

    // Candidate insertion edges: avoid a small forbidden region around seam endpoints,
    // because inserting right back into the seam is pointless.
    // We'll just consider edges outside +/- band around both seams.
    let mut candidate_edges: Vec<usize> = Vec::with_capacity(n);

    let forbid = |i: usize| -> bool {
        let near = |a: usize, b: usize| -> bool {
            let d = a.abs_diff(b);
            d <= band || d >= n.saturating_sub(band)
        };
        near(i, seam1_l) || near(i, seam1_r) || near(i, seam2_l) || near(i, seam2_r)
    };

    for j in 0..n {
        if !forbid(j) {
            candidate_edges.push(j);
        }
    }

    // Main loop: try to relocate suspect points if it improves.
    // Greedy is fine; we’re doing this at every merge so small improvements compound.
    while moved < max_moves {
        let mut best: Option<(usize, usize, f64)> = None; // (k, j, delta)

        for &k in &suspects {
            for &j in &candidate_edges {
                if let Some(d) = delta_relocate_cycle(route, k, j) {
                    if d < -1e-9 {
                        match best {
                            None => best = Some((k, j, d)),
                            Some((_, _, bd)) if d < bd => best = Some((k, j, d)),
                            _ => {}
                        }
                    }
                }
            }
        }

        let Some((k, j, _d)) = best else {
            break;
        };
        apply_relocate_cycle(route, k, j);

        moved += 1;
    }
}
