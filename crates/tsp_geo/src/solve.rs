//! Geographic solve API: lat/lng points in, visiting order out.
//!
//! Points are embedded on an Earth-radius sphere in 3D and the core solver
//! minimizes Euclidean (chord) distance. Chord and great-circle distance
//! agree to within parts-per-million at routing scales, so the heuristic
//! effectively minimizes haversine meters, without the distortion a
//! tangent-plane projection would add for large extents.

use alloc::vec::Vec;

use crate::{Error, GeoPoint, Result};

pub use tsp_ils::SolverConfig;

/// Solve the route order for `points`, returning indices into `points`.
///
/// Fails if any point has out-of-range or non-finite coordinates.
pub fn solve_order(points: &[GeoPoint], cfg: &SolverConfig) -> Result<Vec<u32>> {
    if let Some(bad) = points.iter().position(|p| !p.is_valid()) {
        return Err(Error::invalid_input(alloc::format!(
            "point {bad} has invalid lat/lng values: {}",
            points[bad]
        )));
    }
    let pts: Vec<[f64; 3]> = points.iter().map(GeoPoint::unit_sphere_meters).collect();
    let solution = tsp_ils::solve(&pts, cfg);
    log::info!(
        "solver: n={} chord_length_m={:.0}",
        points.len(),
        solution.length
    );
    Ok(solution.tour)
}

/// Solve the route order for `points`, returning the points in visiting
/// order. See [`solve_order`] to get indices instead.
pub fn solve(points: &[GeoPoint], cfg: &SolverConfig) -> Result<Vec<GeoPoint>> {
    Ok(solve_order(points, cfg)?
        .into_iter()
        .map(|idx| points[idx as usize])
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{SolverConfig, solve, solve_order};
    use crate::GeoPoint;
    use std::time::Duration;

    fn fast_config() -> SolverConfig {
        let mut cfg = SolverConfig::default();
        cfg.time_limit = Some(Duration::from_secs(2));
        cfg.threads = 2;
        cfg
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
        let route = solve(&nodes, &fast_config()).expect("solve");
        assert_eq!(route.len(), n);

        let optimal: f64 = (0..n).map(|i| nodes[i].dist(&nodes[(i + 1) % n])).sum();
        let got: f64 = (0..n).map(|i| route[i].dist(&route[(i + 1) % n])).sum();
        assert!(
            got <= optimal * 1.0001,
            "ring should be recovered: got {got} want {optimal}"
        );
    }

    #[test]
    fn solve_order_returns_a_permutation() {
        let nodes: Vec<GeoPoint> = (0..200)
            .map(|i| {
                let f = i as f64;
                GeoPoint::from_lat_lng(
                    40.0 + (f * 0.7).sin() * 0.05,
                    -74.0 + (f * 1.3).cos() * 0.05,
                )
            })
            .collect();
        let order = solve_order(&nodes, &fast_config()).expect("solve");
        assert_eq!(order.len(), nodes.len());
        let mut seen = vec![false; nodes.len()];
        for &idx in &order {
            assert!(!seen[idx as usize], "index {idx} repeated");
            seen[idx as usize] = true;
        }
    }

    #[test]
    fn rejects_invalid_points() {
        let nodes = vec![
            GeoPoint::from_lat_lng(10.0, 20.0),
            GeoPoint::from_lat_lng(91.0, 20.0),
            GeoPoint::from_lat_lng(11.0, 21.0),
        ];
        let err = solve(&nodes, &fast_config()).expect_err("invalid point should fail");
        assert!(err.to_string().contains("point 1"), "{err}");
    }

    #[test]
    fn tiny_inputs_pass_through() {
        for n in 1..4usize {
            let nodes: Vec<GeoPoint> = (0..n)
                .map(|i| GeoPoint::from_lat_lng(10.0 + i as f64, 20.0))
                .collect();
            let route = solve(&nodes, &fast_config()).expect("solve");
            assert_eq!(route.len(), n);
        }
    }
}
