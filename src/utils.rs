use std::io::Read;

use crate::LKHNode;

pub fn read_points_from_stdin() -> Result<Vec<LKHNode>, String> {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| format!("stdin read failed: {e}"))?;

    // Tokens look like: "40.780374,-73.969161"
    let mut points = Vec::new();

    for (idx, tok) in input.split_whitespace().enumerate() {
        let mut it = tok.split(',');

        let lat_s = it
            .next()
            .ok_or_else(|| format!("Token {}: missing latitude", idx + 1))?;
        let lon_s = it
            .next()
            .ok_or_else(|| format!("Token {}: missing longitude", idx + 1))?;

        if it.next().is_some() {
            return Err(format!(
                "Token {}: expected 'lat,lng' but got extra comma fields: {tok}",
                idx + 1
            ));
        }

        let lat: f64 = lat_s
            .parse()
            .map_err(|_| format!("Token {}: invalid latitude: {}", idx + 1, lat_s))?;
        let lon: f64 = lon_s
            .parse()
            .map_err(|_| format!("Token {}: invalid longitude: {}", idx + 1, lon_s))?;

        points.push(LKHNode::new(lat, lon));
    }

    if points.is_empty() {
        return Err("No points provided on stdin.".to_string());
    }

    Ok(points)
}

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
