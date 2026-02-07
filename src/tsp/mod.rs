mod merge;
mod recursion;
mod refine;
mod split;
#[macro_use]
mod timed;

use crate::utils::Point;

#[derive(Clone, Debug)]
pub struct Options {
    pub tsp_path: String,

    /// Call tsp directly at or below this size.
    pub leaf_size: usize,

    /// Safety valve: if leaf_size is higher, still split above this.
    pub max_leaf_size: usize,

    /// Halo size near split boundaries. Keep small (8..64 typical).
    /// This is "per boundary" budget: we select up to halo points near lat median and halo near lng median.
    pub halo: usize,

    /// Merge quality knob: number of candidate cut edges per cycle.
    /// 0 will still merge, but using a small fallback set of cuts.
    pub portals: usize,

    /// Optional small seam/window refinement after each merge.
    pub seam_refine: bool,
    pub seam_window: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            tsp_path: "tsp".to_string(),
            leaf_size: 100,
            max_leaf_size: 500,
            portals: 96, // set 0 to disable portal insertion
            seam_refine: true,
            halo: 0, // consider 16..64 for higher quality boundaries
            seam_window: 64,
        }
    }
}

pub fn solve(points: &[Point], opts: &Options) -> std::io::Result<Vec<Point>> {
    if points.len() <= 2 {
        return Ok(points.to_vec());
    }

    let mut solution = vec![];

    timed_pass!("TSP", &solution, {
        solution = recursion::solve_rec(points.to_vec(), opts)?;
    });

    // // Light global improvements (always cheap enough)
    // timed_pass!("First Pass", &solution, {
    //     processing::cycle_window_2opt(&mut solution, 256, 2);
    // });

    // timed_pass!("Second Pass", &solution, {
    //     processing::improve_long_edges_cycle(&mut solution, 200, 512);
    // });

    // timed_pass!("Third Pass", &solution, {
    //     processing::cycle_window_2opt(&mut solution, 128, 2);
    // });

    // // Build plan based on current route stats
    // let stats = stats::edge_stats_open(&solution);
    // let ratio = stats.max_m / stats.mean_m.max(1e-9);
    // eprintln!(
    //     "AutoStats: n={} mean={:.1}m median={:.1}m p95={:.1}m max={:.1}m spikes10x={}",
    //     stats.n, stats.mean_m, stats.median_m, stats.p95_m, stats.max_m, stats.spikes_10x
    // );
    // let run_broad = ratio >= 8.0 || stats.spikes_10x >= 6;
    // let plan = stats::build_auto_plan(solution.len(), &stats);
    // if run_broad {
    //     let n = solution.len();
    //     let (window, global_samples, max_outliers, hot_edges) = if n <= 12_000 {
    //         (3000, 128, 128, 16)
    //     } else if n <= 30_000 {
    //         (5000, 128, 128, 12)
    //     } else {
    //         (8000, 64, 96, 8)
    //     };

    //     outlier_splice_repair_v6_par(
    //         &mut solution,
    //         &BroadOptions {
    //             cycle_passes: 1,
    //             hot_edges,
    //             window,
    //             global_samples,
    //             max_outliers,
    //             recompute_every: 64,
    //             early_exit: true,
    //             early_exit_ratio: 6.0,
    //             ..Default::default()
    //         },
    //     );
    // }
    // // Broad outlier
    // if let Some(broad) = plan.broad.clone() {
    //     timed_pass!("Fourth Pass (Broad)", &solution, {
    //         outlier_splice_repair_v6_par(&mut solution, &broad);
    //     });
    // } else {
    //     eprintln!("Skipping Broad outlier pass");
    // }

    // // Recompute stats after broad (so sniper scales to what’s left)
    // let stats2 = stats::edge_stats_open(&solution);
    // let plan2 = stats::build_auto_plan(solution.len(), &stats2);

    // if let Some(mut sniper) = plan2.sniper.clone() {
    //     // If broad already fixed most spikes, reduce sniper work further:
    //     if stats2.spikes_10x <= 4 && stats2.max_m / stats2.mean_m < 8.0 {
    //         sniper.cycle_passes = sniper.cycle_passes.min(3);
    //         sniper.hot_edges = sniper.hot_edges.min(32);
    //     }

    //     timed_pass!("Fifth Pass (Sniper)", &solution, {
    //         outlier_splice_repair_v6_par_sniper(&mut solution, &sniper);
    //     });
    // } else {
    //     eprintln!("Skipping Sniper outlier pass");
    // }

    Ok(solution)
}
