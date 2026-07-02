//! Bridge between geographic input and the planar/spherical core solver.
//!
//! Points are embedded on an Earth-radius sphere in 3D and the core solver
//! minimizes Euclidean (chord) distance. Chord and great-circle distance
//! agree to within parts-per-million at routing scales, so the heuristic
//! effectively minimizes the haversine meters the metrics report, without
//! the distortion a tangent-plane projection would add for large extents.

use std::time::Duration;

use crate::{Error, GeoPoint, Result, SolverInput, SolverOptions};

/// Anything above this is treated as "no limit was intended" rather than fed
/// to `Duration::from_secs_f64`, which panics on huge or non-finite values.
const MAX_TIME_LIMIT_SECS: f64 = 1e9;

#[tsp_mt_derive::timer()]
pub fn solve(input: SolverInput, options: &SolverOptions) -> Result<Vec<GeoPoint>> {
    if !options.time_limit.is_finite()
        || options.time_limit < 0.0
        || options.time_limit > MAX_TIME_LIMIT_SECS
    {
        return Err(Error::invalid_input(format!(
            "--time-limit must be a finite number of seconds in [0, {MAX_TIME_LIMIT_SECS:.0}], got {}",
            options.time_limit
        )));
    }

    let nodes = input.nodes;
    let pts: Vec<[f64; 3]> = nodes.iter().map(GeoPoint::unit_sphere_meters).collect();

    let defaults = tsp_ils::SolverConfig::default();
    let max_neighbors = options.max_neighbors.max(4);
    let cfg = tsp_ils::SolverConfig {
        time_limit: (options.time_limit > 0.0).then(|| Duration::from_secs_f64(options.time_limit)),
        seed: options.seed,
        threads: options.threads,
        max_neighbors,
        max_candidates: defaults.max_candidates.max(max_neighbors + 6),
        ..defaults
    };

    let solution = tsp_ils::solve(&pts, &cfg);
    log::info!(
        "solver: n={} chord_length_m={:.0}",
        nodes.len(),
        solution.length
    );

    Ok(solution
        .tour
        .into_iter()
        .map(|idx| nodes[idx as usize])
        .collect())
}

#[cfg(test)]
mod tests {
    use super::solve;
    use crate::{GeoPoint, SolverInput, SolverOptions};

    fn fast_options() -> SolverOptions {
        SolverOptions {
            time_limit: 2.0,
            threads: 2,
            ..SolverOptions::default()
        }
    }

    #[test]
    fn solves_a_geographic_ring_optimally() {
        // Points on a small geographic circle around Berlin.
        let n = 48;
        let center = (52.52, 13.405);
        let nodes: Vec<GeoPoint> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64 * std::f64::consts::TAU;
                GeoPoint::from_lat_lng(center.0 + 0.01 * t.sin(), center.1 + 0.016 * t.cos())
            })
            .collect();
        let input = SolverInput::new(&nodes);
        let route = solve(input, &fast_options()).expect("solve");
        assert_eq!(route.len(), n);

        let optimal: f64 = (0..n).map(|i| nodes[i].dist(&nodes[(i + 1) % n])).sum();
        let got: f64 = (0..n).map(|i| route[i].dist(&route[(i + 1) % n])).sum();
        assert!(
            got <= optimal * 1.0001,
            "ring should be recovered: got {got} want {optimal}"
        );
    }

    #[test]
    fn returns_all_points_exactly_once() {
        let nodes: Vec<GeoPoint> = (0..200)
            .map(|i| {
                let f = i as f64;
                GeoPoint::from_lat_lng(
                    40.0 + (f * 0.7).sin() * 0.05,
                    -74.0 + (f * 1.3).cos() * 0.05,
                )
            })
            .collect();
        let input = SolverInput::new(&nodes);
        let route = solve(input, &fast_options()).expect("solve");
        assert_eq!(route.len(), nodes.len());
        let mut used = vec![false; nodes.len()];
        for p in &route {
            let idx = nodes
                .iter()
                .position(|n| n == p)
                .expect("output point must come from input");
            // Duplicate coordinates would match the same index; tolerate by
            // scanning for the first unused match.
            let idx = if used[idx] {
                nodes
                    .iter()
                    .enumerate()
                    .position(|(i, n)| !used[i] && n == p)
                    .expect("unused duplicate")
            } else {
                idx
            };
            used[idx] = true;
        }
        assert!(used.iter().all(|&u| u));
    }

    #[test]
    fn rejects_non_finite_or_absurd_time_limits() {
        let nodes: Vec<GeoPoint> = (0..5)
            .map(|i| GeoPoint::from_lat_lng(10.0 + i as f64, 20.0))
            .collect();
        for bad in [f64::INFINITY, f64::NAN, -1.0, 1e300] {
            let options = SolverOptions {
                time_limit: bad,
                ..SolverOptions::default()
            };
            let err = solve(SolverInput::new(&nodes), &options)
                .expect_err("bad time limit should be rejected");
            assert!(err.to_string().contains("--time-limit"), "{err}");
        }
    }

    #[test]
    fn tiny_inputs_pass_through() {
        for n in 1..4usize {
            let nodes: Vec<GeoPoint> = (0..n)
                .map(|i| GeoPoint::from_lat_lng(10.0 + i as f64, 20.0))
                .collect();
            let input = SolverInput::new(&nodes);
            let route = solve(input, &fast_options()).expect("solve");
            assert_eq!(route.len(), n);
        }
    }
}
