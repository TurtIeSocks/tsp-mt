use crate::LKHNode;

pub fn measure_distance_open(points: &[LKHNode]) -> (f64, f64, i32) {
    if points.len() < 2 {
        eprintln!("Total m: 0");
        eprintln!("Longest m: 0");
        eprintln!("Total Spikes: 0");
        eprintln!();
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

    eprintln!("Total dist: {total:.0}m");
    eprintln!("Longest dist: {longest:.0}m");
    eprintln!("Average: {avg_edge:.0}m");

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

    eprintln!("Total Over {threshold:.0}m: {spikes}");

    (total, longest, spikes)
}
