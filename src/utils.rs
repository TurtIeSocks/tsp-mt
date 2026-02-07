use crate::LKHNode;

pub fn measure_distance_open(points: &[LKHNode]) -> (f64, f64, i32) {
    if points.len() < 2 {
        log::info!(
            "metrics.open_distance: n={} total_m=0 longest_m=0 avg_m=0 spike_threshold_m=0 spikes=0",
            points.len()
        );
        return (0.0, 0.0, 0);
    }

    let mut total = 0.0;
    let mut longest = 0.0;
    let n = points.len();

    // OPEN: only edges i -> i+1
    for i in 0..(points.len() - 1) {
        let d = points[i].dist(&points[i + 1]);
        total += d;
        if d > longest {
            longest = d;
        }
    }
    let avg_edge = total / ((points.len() - 1) as f64);
    let threshold = avg_edge * 10.0;

    // Spike threshold: 10× average edge length (OPEN edges count = n-1)
    let mut spikes = 0;
    for i in 0..(points.len() - 1) {
        let d_l = points[i].dist(&points[(i + 1) % n]);
        if d_l > threshold {
            spikes += 1;
        }
        // let d_r = points[i].dist(&points[(i - 1) % n]);
        // if d_l > threshold && d_r > threshold {
        //     spikes += 1;
        // }
    }

    log::info!(
        "metrics.open_distance: n={} total_m={total:.0} longest_m={longest:.0} avg_m={avg_edge:.0} spike_threshold_m={threshold:.0} spikes={spikes}",
        points.len()
    );

    (total, longest, spikes)
}
