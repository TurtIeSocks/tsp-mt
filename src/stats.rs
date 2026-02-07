use crate::{
    outliers::{BroadOptions, SniperOptions},
    utils::Point,
};

#[derive(Clone, Debug)]
pub struct AutoPlan {
    // light global passes
    pub do_2opt_256: bool,
    pub do_long_edges: bool,
    pub do_2opt_128: bool,

    // outlier passes
    pub broad: Option<BroadOptions>,
    pub sniper: Option<SniperOptions>,
}

pub fn build_auto_plan(n: usize, stats_after_global: &EdgeStats) -> AutoPlan {
    // Basic size buckets
    let big = n >= 20_000;
    let huge = n >= 50_000;

    // Severity
    let ratio = if stats_after_global.mean_m > 0.0 {
        stats_after_global.max_m / stats_after_global.mean_m
    } else {
        0.0
    };

    let spikes = stats_after_global
        .spikes_10x
        .max(stats_after_global.spikes_8x);

    // ---- Always keep the cheap improvements, but scale down for tiny inputs ----
    let do_2opt_256 = n >= 300;
    let do_long_edges = n >= 800;
    let do_2opt_128 = n >= 300;

    // ---- Decide if we should run outlier logic at all ----
    // “smooth enough”: no meaningful spikes and ratio not insane
    let needs_outliers = spikes > 0 || ratio >= 7.0;

    if !needs_outliers {
        return AutoPlan {
            do_2opt_256,
            do_long_edges,
            do_2opt_128,
            broad: None,
            sniper: None,
        };
    }

    // ---- Broad pass tuning ----
    // Broad is good at removing many medium-bad seams quickly.
    // Keep cycle_passes small; increase hot_edges if spikes are numerous.
    let hot_edges = match spikes {
        0..=4 => 32,
        5..=12 => 64,
        13..=40 => 96,
        _ => 128,
    };

    // Broad cycle_passes: more passes if ratio is huge, but cap.
    let mut cycle_passes = if ratio >= 12.0 {
        4
    } else if ratio >= 9.0 {
        3
    } else {
        2
    };
    if huge {
        cycle_passes = cycle_passes.min(3);
    } // keep runtime bounded

    // ---- Sniper tuning ----
    // Sniper is expensive; only crank if big spikes remain.
    let mut sniper_passes = if spikes >= 20 || ratio >= 10.0 { 6 } else { 4 };
    if big {
        sniper_passes = sniper_passes.min(6);
    }
    if huge {
        sniper_passes = sniper_passes.min(5);
    }

    // Make sniper more selective for huge n: higher ratio gate
    let ratio_gate = if huge { 9.0 } else { 8.0 };

    // Windows: for huge n, make window smaller but rely on rotations.
    // (Rotations already give you coverage.)
    let window = if huge {
        12_000
    } else if big {
        16_000
    } else {
        20_000
    };

    // global_samples: cap hard for huge n (this is where time blows up)
    let global_samples = if huge {
        256
    } else if big {
        384
    } else {
        512
    };

    // ---- Construct options ----
    let broad = BroadOptions {
        cycle_passes,
        hot_edges,
        // keep defaults for internals unless you want to expose them
        ..Default::default()
    };

    let sniper = SniperOptions {
        cycle_passes: sniper_passes,
        hot_edges: hot_edges.min(96),
        ratio_gate,
        window,
        global_samples,
        ..Default::default()
    };

    AutoPlan {
        do_2opt_256,
        do_long_edges,
        do_2opt_128,
        broad: Some(broad),
        sniper: Some(sniper),
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EdgeStats {
    pub n: usize,
    pub edges: usize, // n-1 for open
    pub total_m: f64,
    pub mean_m: f64,
    pub median_m: f64,
    pub p95_m: f64,
    pub max_m: f64,
    pub spikes_8x: usize,  // > 8x mean
    pub spikes_10x: usize, // > 10x mean
}

pub fn edge_stats_open(route: &[Point]) -> EdgeStats {
    let n = route.len();
    if n < 2 {
        return EdgeStats {
            n,
            edges: 0,
            total_m: 0.0,
            mean_m: 0.0,
            median_m: 0.0,
            p95_m: 0.0,
            max_m: 0.0,
            spikes_8x: 0,
            spikes_10x: 0,
        };
    }

    let edges = n - 1;
    let mut lens = Vec::with_capacity(edges);

    let mut total = 0.0;
    let mut max = 0.0;
    for i in 0..edges {
        let d = route[i].dist(&route[i + 1]);
        total += d;
        if d > max {
            max = d;
        }
        lens.push(d);
    }

    lens.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = lens[edges / 2];
    let p95 = (lens[((edges as f64) * 0.95) as usize] as usize).min(edges - 1);

    let mean = total / (edges as f64);
    let spikes_8x = lens.iter().filter(|&&d| d > 8.0 * mean).count();
    let spikes_10x = lens.iter().filter(|&&d| d > 10.0 * mean).count();

    EdgeStats {
        n,
        edges,
        total_m: total,
        mean_m: mean,
        median_m: median,
        p95_m: lens[p95],
        max_m: max,
        spikes_8x,
        spikes_10x,
    }
}
