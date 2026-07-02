//! Quality/scaling benchmark for the solver.
//!
//! Usage:
//!   cargo run --release -p tsp_ils --example bench -- [n] [seconds] [threads] [k] [oropt] [min_seg]
//!   cargo run --release -p tsp_ils --example bench -- <file.tsp> [seconds] [threads] [k] [oropt] [min_seg]
//!
//! With a numeric first argument, benchmarks `n` uniform random points in a
//! square and reports the gap against the Beardwood-Halton-Hammersley
//! estimate (~0.7124 * sqrt(n * area), asymptotically the optimal tour
//! length for uniform points). With a TSPLIB path (NODE_COORD_SECTION,
//! EUC_2D), benchmarks that instance.

use std::time::{Duration, Instant};

use tsp_ils::{SolverConfig, cycle_length, solve};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let spec = args.first().cloned().unwrap_or_else(|| "10000".into());
    let seconds: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10.0);
    let threads: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);

    let (pts, bhh): (Vec<[f64; 2]>, Option<f64>) = if let Ok(n) = spec.parse::<usize>() {
        let side = 1_000_000.0f64;
        let pts = random_points(n, side, 20260701);
        let est = 0.7124 * (n as f64 * side * side).sqrt();
        (pts, Some(est))
    } else {
        (read_tsplib(&spec), None)
    };

    let defaults = SolverConfig::default();
    let max_neighbors: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(defaults.max_neighbors);
    let or_opt_max_len: usize = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(defaults.or_opt_max_len);
    let min_segment_len: usize = args
        .get(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(defaults.min_segment_len);
    let multi_start_max: usize = args
        .get(6)
        .and_then(|s| s.parse().ok())
        .unwrap_or(defaults.multi_start_max);
    let mut cfg = defaults;
    cfg.time_limit = Some(Duration::from_secs_f64(seconds));
    cfg.threads = threads;
    cfg.max_neighbors = max_neighbors;
    cfg.max_candidates = max_neighbors + 6;
    cfg.or_opt_max_len = or_opt_max_len;
    cfg.min_segment_len = min_segment_len;
    cfg.multi_start_max = multi_start_max;
    let cfg = cfg;

    let start = Instant::now();
    let sol = solve(&pts, &cfg);
    let elapsed = start.elapsed().as_secs_f64();

    let n = pts.len();
    let recomputed = cycle_length(&pts, &sol.tour);
    assert!((recomputed - sol.length).abs() < 1e-6 * recomputed.max(1.0));
    let avg = sol.length / n as f64;
    let spikes_3x = count_spikes(&pts, &sol.tour, avg * 3.0);
    let spikes_10x = count_spikes(&pts, &sol.tour, avg * 10.0);

    println!(
        "n={n} threads={threads} time={elapsed:.2}s length={:.0}",
        sol.length
    );
    if let Some(est) = bhh {
        println!(
            "BHH-estimate={est:.0} gap={:+.2}%",
            (sol.length / est - 1.0) * 100.0
        );
    }
    println!("avg-edge={avg:.1} spikes>3x={spikes_3x} spikes>10x={spikes_10x}");
}

fn count_spikes(pts: &[[f64; 2]], tour: &[u32], threshold: f64) -> usize {
    let n = tour.len();
    (0..n)
        .filter(|&i| {
            tsp_ils::dist(&pts[tour[i] as usize], &pts[tour[(i + 1) % n] as usize]) > threshold
        })
        .count()
}

fn random_points(n: usize, side: f64, seed: u64) -> Vec<[f64; 2]> {
    // SplitMix64, inlined to keep the example self-contained.
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

/// Minimal TSPLIB NODE_COORD_SECTION reader (EUC_2D-style coordinates).
fn read_tsplib(path: &str) -> Vec<[f64; 2]> {
    let text = std::fs::read_to_string(path).expect("read TSPLIB file");
    let mut pts = Vec::new();
    let mut in_coords = false;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with("NODE_COORD_SECTION") {
            in_coords = true;
            continue;
        }
        if !in_coords {
            continue;
        }
        if line == "EOF" || line.is_empty() {
            break;
        }
        let mut it = line.split_whitespace();
        let _id = it.next();
        let x: f64 = it.next().expect("x coord").parse().expect("x parse");
        let y: f64 = it.next().expect("y coord").parse().expect("y parse");
        pts.push([x, y]);
    }
    assert!(!pts.is_empty(), "no coordinates found in {path}");
    pts
}
