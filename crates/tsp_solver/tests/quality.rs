//! Quality anchors: instances where the optimal tour length is known
//! exactly (tiny instances via Held-Karp DP, structured instances by
//! construction), so solver regressions show up as hard failures.

use std::time::Duration;

use tsp_solver::{SolverConfig, cycle_length, solve};

fn cfg(seconds: f64) -> SolverConfig {
    SolverConfig {
        time_limit: Some(Duration::from_secs_f64(seconds)),
        threads: 4,
        ..SolverConfig::default()
    }
}

/// Exact optimal cycle length via Held-Karp dynamic programming (n <= ~15).
fn held_karp(pts: &[[f64; 2]]) -> f64 {
    let n = pts.len();
    assert!((2..=15).contains(&n));
    let d = |a: usize, b: usize| tsp_solver::dist(&pts[a], &pts[b]);
    let full = 1usize << (n - 1);
    // dp[mask][j]: shortest path from node 0 visiting exactly the nodes in
    // mask (over nodes 1..n), ending at node j+1.
    let mut dp = vec![vec![f64::INFINITY; n - 1]; full];
    for j in 0..n - 1 {
        dp[1 << j][j] = d(0, j + 1);
    }
    for mask in 1..full {
        for j in 0..n - 1 {
            if mask & (1 << j) == 0 || dp[mask][j].is_infinite() {
                continue;
            }
            let base = dp[mask][j];
            #[allow(clippy::needless_range_loop)]
            for k in 0..n - 1 {
                if mask & (1 << k) != 0 {
                    continue;
                }
                let next = mask | (1 << k);
                let cost = base + d(j + 1, k + 1);
                if cost < dp[next][k] {
                    dp[next][k] = cost;
                }
            }
        }
    }
    (0..n - 1)
        .map(|j| dp[full - 1][j] + d(j + 1, 0))
        .fold(f64::INFINITY, f64::min)
}

fn random_points(n: usize, seed: u64, side: f64) -> Vec<[f64; 2]> {
    let mut state = seed;
    let mut next = move || {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    };
    (0..n)
        .map(|_| {
            let x = (next() >> 11) as f64 / (1u64 << 53) as f64 * side;
            let y = (next() >> 11) as f64 / (1u64 << 53) as f64 * side;
            [x, y]
        })
        .collect()
}

#[test]
fn matches_held_karp_optimum_on_tiny_instances() {
    for seed in 1..=6u64 {
        let pts = random_points(11, seed, 1000.0);
        let optimal = held_karp(&pts);
        let sol = solve(&pts, &cfg(1.0));
        assert!(
            sol.length <= optimal * (1.0 + 1e-9),
            "seed {seed}: got {} but optimum is {optimal}",
            sol.length
        );
    }
}

#[test]
fn solves_unit_grid_near_optimally() {
    // k x k unit grid with even k: the optimal tour has length exactly n
    // (a boustrophedon path), and it has zero long edges.
    let k = 40usize;
    let n = k * k;
    let pts: Vec<[f64; 2]> = (0..n).map(|i| [(i % k) as f64, (i / k) as f64]).collect();
    let sol = solve(&pts, &cfg(20.0));
    let optimal = n as f64;
    let gap = sol.length / optimal - 1.0;
    assert!(
        gap < 0.02,
        "grid gap too large: {:.3}% (len {})",
        gap * 100.0,
        sol.length
    );
    // Spikes: no edge should be more than ~3 grid units on a near-optimal
    // grid tour (allows a few diagonal stitches).
    let long_edges = (0..n)
        .filter(|&i| {
            tsp_solver::dist(
                &pts[sol.tour[i] as usize],
                &pts[sol.tour[(i + 1) % n] as usize],
            ) > 3.0
        })
        .count();
    assert!(long_edges <= 2, "grid tour has {long_edges} long edges");
}

#[test]
fn handles_many_duplicate_points() {
    let mut pts = random_points(200, 9, 100.0);
    let dups: Vec<[f64; 2]> = pts.iter().take(100).copied().collect();
    pts.extend(dups);
    let sol = solve(&pts, &cfg(2.0));
    assert_eq!(sol.tour.len(), pts.len());
    let mut seen = vec![false; pts.len()];
    for &v in &sol.tour {
        assert!(!seen[v as usize]);
        seen[v as usize] = true;
    }
    // Duplicates must sit adjacent to their twin (zero-length edge), so the
    // tour must not pay more than the deduplicated instance would.
    let dedup_sol = solve(&pts[..200], &cfg(2.0));
    assert!(sol.length <= dedup_sol.length * 1.05);
}

#[test]
fn collinear_points_are_fine() {
    let pts: Vec<[f64; 2]> = (0..300).map(|i| [i as f64, 0.0]).collect();
    let sol = solve(&pts, &cfg(2.0));
    assert_eq!(sol.tour.len(), pts.len());
    // Optimal: sweep right then come back = 2 * 299.
    assert!((sol.length - 598.0).abs() < 1e-6, "got {}", sol.length);
}

#[test]
fn segment_rounds_path_solves_grid_near_optimally() {
    // Force the parallel split/join path (multi_start_max = 0) with small
    // initial segments so several rounds of the coarse-to-fine schedule run.
    let k = 40usize;
    let n = k * k;
    let pts: Vec<[f64; 2]> = (0..n).map(|i| [(i % k) as f64, (i / k) as f64]).collect();
    let sol = solve(
        &pts,
        &SolverConfig {
            time_limit: Some(Duration::from_secs(10)),
            threads: 4,
            multi_start_max: 0,
            min_segment_len: 200,
            ..SolverConfig::default()
        },
    );
    let mut seen = vec![false; n];
    for &v in &sol.tour {
        assert!(!seen[v as usize]);
        seen[v as usize] = true;
    }
    let gap = sol.length / n as f64 - 1.0;
    assert!(
        gap < 0.03,
        "segmented grid gap too large: {:.3}%",
        gap * 100.0
    );
}

#[test]
fn respects_thread_count_and_stays_consistent() {
    let pts = random_points(4000, 33, 100_000.0);
    for threads in [1, 8] {
        let sol = solve(
            &pts,
            &SolverConfig {
                time_limit: Some(Duration::from_secs(3)),
                threads,
                ..SolverConfig::default()
            },
        );
        assert_eq!(sol.tour.len(), pts.len());
        let recomputed = cycle_length(&pts, &sol.tour);
        assert!((recomputed - sol.length).abs() < 1e-6 * recomputed);
    }
}
