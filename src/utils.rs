use std::io::Read;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub lat: f64,
    pub lng: f64,
}

const R: f64 = 6_371_000.0;

impl ToString for Point {
    fn to_string(&self) -> String {
        let mut b1 = ryu::Buffer::new();
        let mut b2 = ryu::Buffer::new();
        format!("{},{}", b1.format(self.lat), b2.format(self.lng))
    }
}

impl Point {
    pub fn dist(self, rhs: &Point) -> f64 {
        // Haversine meters
        let (lat1, lat2) = (self.lat.to_radians(), rhs.lat.to_radians());
        let dlat = (rhs.lat - self.lat).to_radians();
        let dlng = (rhs.lng - self.lng).to_radians();
        let s1 = (dlat / 2.0).sin();
        let s2 = (dlng / 2.0).sin();
        let h = s1 * s1 + lat1.cos() * lat2.cos() * s2 * s2;
        2.0 * R * h.sqrt().asin()
    }
}

// pub fn get_outliers(points: &[Point], factor: f64) -> (Vec<Point>, Vec<Point>) {
//     let mut included = Vec::new();
//     let mut excluded = Vec::new();

//     let n = points.len();
//     if n == 0 {
//         return (included, excluded);
//     }
//     if n < 3 {
//         return (points.to_vec(), excluded);
//     }

//     let total: f64 = (0..n).map(|i| points[i].dist(&points[(i + 1) % n])).sum();
//     let avg = total / (n as f64);
//     let threshold = avg * factor;

//     eprintln!("total={total}, avg={avg}, threshold={threshold}");

//     for i in 0..n {
//         let prev = (i + n - 1) % n;
//         let next = (i + 1) % n;
//         let d_prev = points[prev].dist(&points[i]);
//         let d_next = points[i].dist(&points[next]);
//         if d_prev > threshold && d_next > threshold {
//             excluded.push(points[i]);
//         } else {
//             included.push(points[i]);
//         }
//     }

//     (included, excluded)
// }

// pub fn centroid(points: &[Point]) -> Result<Point, String> {
//     if points.is_empty() {
//         return Err("No Points".to_string());
//     }
//     let mut sum_lat = 0.0;
//     let mut sum_lng = 0.0;
//     for p in points {
//         sum_lat += p.lat;
//         sum_lng += p.lng;
//     }
//     let n = points.len() as f64;
//     Ok(Point {
//         lat: sum_lat / n,
//         lng: sum_lng / n,
//     })
// }

pub fn read_points_from_stdin() -> Result<Vec<Point>, String> {
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

        points.push(Point { lat, lng: lon });
    }

    if points.is_empty() {
        return Err("No points provided on stdin.".to_string());
    }

    // if let Some(first) = points.first() {
    //     if let Some(last) = points.last() {
    //         if first.lat != last.lat || first.lng != last.lng {
    //             eprintln!("Adding last point");
    //             // out_points.pop();
    //             points.push(first.clone())
    //         }
    //     }
    // }

    // points.pop();

    Ok(points)
}

// pub fn run_external_tsp_strict(tsp_path: &str, points: &[Point]) -> std::io::Result<Vec<Point>> {
//     // eprintln!("Sending to TSP");
//     // Use default formatting (shortest-roundtrip). Avoid fixed decimals that can collapse distinct points.
//     let mut input = String::with_capacity(points.len() * 32);
//     for (i, p) in points.iter().enumerate() {
//         if i > 0 {
//             input.push(' ');
//         }
//         input.push_str(&format!("{},{}", p.lat, p.lng));
//     }

//     let mut child = Command::new(tsp_path)
//         .stdin(Stdio::piped())
//         .stdout(Stdio::piped())
//         .stderr(Stdio::piped())
//         .spawn()?;

//     {
//         let mut stdin = child.stdin.take().expect("piped stdin");
//         stdin.write_all(input.as_bytes())?;
//     }

//     let mut stdout_s = String::new();
//     let mut stderr_s = String::new();
//     if let Some(mut stdout) = child.stdout.take() {
//         stdout.read_to_string(&mut stdout_s)?;
//     }
//     if let Some(mut stderr) = child.stderr.take() {
//         stderr.read_to_string(&mut stderr_s)?;
//     }

//     let status = child.wait()?;
//     if !status.success() {
//         return Err(std::io::Error::new(
//             std::io::ErrorKind::Other,
//             format!("tsp failed: status={status}, stderr={stderr_s}"),
//         ));
//     }

//     // eprintln!("Single TSP finished");
//     let out = parse_points_from_stdout(&stdout_s)?;
//     if out.len() != points.len() {
//         return Err(std::io::Error::new(
//             std::io::ErrorKind::InvalidData,
//             format!(
//                 "tsp returned wrong number of points: in={}, out={}, stderr={}",
//                 points.len(),
//                 out.len(),
//                 stderr_s
//             ),
//         ));
//     }
//     Ok(out)
// }

// pub fn parse_points_from_stdout(s: &str) -> std::io::Result<Vec<Point>> {
//     let mut out = Vec::new();
//     // let mut seen = HashSet::new();
//     for tok in s.split_whitespace() {
//         // if seen.contains(tok) {
//         //     continue;
//         // }
//         // seen.insert(tok);
//         // println!("{tok}");
//         let (a, b) = tok
//             .split_once(',')
//             .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad token"))?;
//         let lat: f64 = a
//             .parse()
//             .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad lat"))?;
//         let lng: f64 = b
//             .parse()
//             .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad lng"))?;
//         out.push(Point { lat, lng });
//     }

//     // Some solvers may output trailing whitespace or empty output on edge cases.
//     if out.is_empty() {
//         return Err(std::io::Error::new(
//             std::io::ErrorKind::InvalidData,
//             "tsp returned no points",
//         ));
//     }

//     // out.push(out.first().unwrap().clone());
//     // out.pop();
//     out.pop();

//     Ok(out)
// }

// pub fn measure_distance_closed(points: &[Point]) -> (f64, f64, f64, i32) {
//     if points.len() < 2 {
//         eprintln!("Total km: 0");
//         eprintln!("Longest km: 0");
//         eprintln!("Total Spikes: 0");
//         eprintln!();
//         return (0.0, 0.0, 0.0, 0);
//     }

//     let n = points.len();
//     let mut total = 0.0;
//     let mut longest = 0.0;

//     // CLOSED: edges i -> (i+1)%n, includes wrap edge (n-1 -> 0)
//     for i in 0..n {
//         let j = (i + 1) % n;
//         let d = points[i].dist(&points[j]);
//         total += d;
//         if d > longest {
//             longest = d;
//         }
//     }

//     eprintln!("Total km: {total:.0}");
//     eprintln!("Longest km: {longest:.0}");

//     // Spike threshold: 10× average edge length (CLOSED edges count = n)
//     let avg_edge = total / (n as f64);
//     let mut spikes = 0;
//     for i in 0..n {
//         let j = (i + 1) % n;
//         let d = points[i].dist(&points[j]);
//         if d > 10.0 * avg_edge {
//             spikes += 1;
//         }
//     }

//     eprintln!("Total Spikes: {spikes}");
//     eprintln!();

//     (total, longest, avg_edge, spikes)
// }

pub fn measure_distance_open(points: &[Point]) -> (f64, f64, i32) {
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

// pub fn get_avg_dist(points: &[Point]) -> f64 {
//     if points.len() < 2 {
//         // eprintln!("Total m: 0");
//         // eprintln!("Longest m: 0");
//         // eprintln!("Total Spikes: 0");
//         // eprintln!();
//         return 0.0;
//     }

//     let mut total = 0.0;

//     // OPEN: only edges i -> i+1
//     for i in 0..(points.len() - 1) {
//         let d = points[i].dist(&points[i + 1]);
//         total += d;
//     }

//     total / ((points.len() - 1) as f64)
// }
