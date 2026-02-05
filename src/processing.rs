use std::cmp::Ordering;

use rand::{Rng, SeedableRng, rngs::SmallRng};

use crate::utils::Point;

/// Windowed 2-opt on cycles via rotations (open 2-opt on a few breaks).
pub fn cycle_window_2opt(route: &mut Vec<Point>, window: usize, passes: usize) {
    let n = route.len();
    if n < 8 {
        return;
    }
    let offsets = [0usize, n / 4, n / 2, (3 * n) / 4];

    for &off in &offsets {
        route.rotate_left(off);
        global_window_2opt_open(route, window, passes);
        route.rotate_right(off);
    }
}

/// Open windowed 2-opt (your original global_window_2opt adapted).
fn global_window_2opt_open(route: &mut [Point], window: usize, passes: usize) {
    let n = route.len();
    if n < 6 {
        return;
    }
    let w = window.min(n / 2).max(16);

    for _ in 0..passes {
        let mut improved = false;
        for i in 1..(n - 3) {
            let a = route[i - 1];
            let b = route[i];
            let j_max = (i + w).min(n - 2);

            for j in (i + 1)..=j_max {
                let c = route[j];
                let d = route[j + 1];
                let before = a.dist(&b) + c.dist(&d);
                let after = a.dist(&c) + b.dist(&d);
                if after + 1e-9 < before {
                    route[i..=j].reverse();
                    improved = true;
                }
            }
        }
        if !improved {
            break;
        }
    }
}

/// Long-edge cleanup for cycles, including the wrap edge (n-1 -> 0).
/// Uses rotations to handle wrap interactions safely.
pub fn improve_long_edges_cycle(route: &mut Vec<Point>, k: usize, window: usize) {
    let n = route.len();
    if n < 8 {
        return;
    }

    // Evaluate edges including wrap.
    let mut edges: Vec<(f64, usize)> = (0..n)
        .map(|i| (route[i].dist(&route[(i + 1) % n]), i))
        .collect();
    edges.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    edges.truncate(k.min(edges.len()));

    // For each long edge, rotate so that edge becomes (i -> i+1) in open sense, then do local open repair.
    for &(_, ei) in &edges {
        // rotate so that ei is at position 0 edge (0->1), making wrap disappear for the target edge
        route.rotate_left(ei);

        // Now the bad edge is between 0 and 1. Try local open 2-opt around it.
        // Use a couple quick passes.
        global_window_2opt_open(route, window.min(n / 2), 1);

        // rotate back
        route.rotate_right(ei);
    }
}

/// Optional: random 2-opt on cycles via rotations (kept for completeness).
#[allow(dead_code)]
fn random_cycle_2opt(route: &mut Vec<Point>, iterations: usize, seed: u64) {
    let n = route.len();
    if n < 8 {
        return;
    }
    let mut rng = SmallRng::seed_from_u64(seed);

    for _ in 0..iterations {
        let off = (rng.random::<u32>() as usize) % n;
        route.rotate_left(off);
        random_2opt_open(route, 32, rng.random::<u64>());
        route.rotate_right(off);
    }
}

fn random_2opt_open(route: &mut [Point], iterations: usize, seed: u64) {
    let n = route.len();
    if n < 6 {
        return;
    }
    let mut rng = SmallRng::seed_from_u64(seed);

    for _ in 0..iterations {
        let i = rng.random_range(1..n - 3);
        let j = rng.random_range(i + 1..n - 2);
        if j <= i + 1 {
            continue;
        }

        let a = route[i - 1];
        let b = route[i];
        let c = route[j];
        let d = route[j + 1];

        let before = a.dist(&b) + c.dist(&d);
        let after = a.dist(&c) + b.dist(&d);
        if after + 1e-9 < before {
            route[i..=j].reverse();
        }
    }
}
