//! Conversions to/from the georust [`geo-types`](https://docs.rs/geo-types)
//! primitives, enabled by the `rustgeo` feature.
//!
//! Both libraries use the same axis convention (`x` = longitude,
//! `y` = latitude), so conversions are direct field copies.
//!
//! ```
//! use geo_types::{Coord, LineString, Point};
//! use tsp_geo::{GeoPoint, SolverConfig, solve_order_of};
//!
//! let stops = vec![
//!     Point::new(13.4050, 52.5200), // Berlin (x = lng, y = lat)
//!     Point::new(2.3522, 48.8566),  // Paris
//!     Point::new(-0.1278, 51.5074), // London
//! ];
//! // geo-types values feed straight into the solver...
//! let order = solve_order_of(stops.iter().copied(), &SolverConfig::default()).unwrap();
//! // ...and results convert straight back.
//! let route: LineString = order
//!     .iter()
//!     .map(|&i| Coord::from(GeoPoint::from(stops[i as usize])))
//!     .collect();
//! # assert_eq!(route.0.len(), 3);
//! ```

use geo_types::{Coord, Point};

use crate::GeoPoint;

impl From<Coord<f64>> for GeoPoint {
    fn from(c: Coord<f64>) -> Self {
        GeoPoint::new(c.x, c.y)
    }
}

impl From<Point<f64>> for GeoPoint {
    fn from(p: Point<f64>) -> Self {
        GeoPoint::new(p.x(), p.y())
    }
}

impl From<&Coord<f64>> for GeoPoint {
    fn from(c: &Coord<f64>) -> Self {
        GeoPoint::new(c.x, c.y)
    }
}

impl From<&Point<f64>> for GeoPoint {
    fn from(p: &Point<f64>) -> Self {
        GeoPoint::new(p.x(), p.y())
    }
}

impl From<GeoPoint> for Coord<f64> {
    fn from(p: GeoPoint) -> Self {
        Coord { x: p.x, y: p.y }
    }
}

impl From<GeoPoint> for Point<f64> {
    fn from(p: GeoPoint) -> Self {
        Point::new(p.x, p.y)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use geo_types::{Coord, LineString, MultiPoint, Point};

    use crate::{GeoPoint, SolverConfig, solve_order_of};

    fn fast_config() -> SolverConfig {
        let mut cfg = SolverConfig::default();
        cfg.time_limit = Some(core::time::Duration::from_secs(2));
        cfg.threads = 2;
        cfg
    }

    #[test]
    fn conversions_preserve_lat_lng() {
        let coord = Coord {
            x: 13.405,
            y: 52.52,
        };
        let gp = GeoPoint::from(coord);
        assert_eq!(gp.x, 13.405); // lng
        assert_eq!(gp.y, 52.52); // lat
        assert_eq!(Coord::from(gp), coord);

        let point = Point::new(-0.1278, 51.5074);
        let gp = GeoPoint::from(point);
        assert_eq!(Point::from(gp), point);
        assert_eq!(GeoPoint::from(&point), gp);
        assert_eq!(GeoPoint::from(&coord), GeoPoint::from(coord));
    }

    #[test]
    fn solves_from_geo_types_inputs() {
        // A ring of points; the optimal order is the ring order.
        let n = 32;
        let points: Vec<Point<f64>> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64 * core::f64::consts::TAU;
                Point::new(13.4 + 0.02 * t.cos(), 52.5 + 0.0125 * t.sin())
            })
            .collect();

        let order = solve_order_of(points.iter(), &fast_config()).expect("solve");
        assert_eq!(order.len(), n);
        let mut seen = vec![false; n];
        for &i in &order {
            assert!(!seen[i as usize]);
            seen[i as usize] = true;
        }

        // MultiPoint and LineString point iterators work as inputs too.
        let mp: MultiPoint<f64> = points.clone().into();
        let from_mp = solve_order_of(mp.iter(), &fast_config()).expect("solve");
        assert_eq!(from_mp.len(), n);

        let ls: LineString<f64> = points
            .iter()
            .map(|p| Coord::from(GeoPoint::from(p)))
            .collect();
        let from_ls = solve_order_of(ls.points(), &fast_config()).expect("solve");
        assert_eq!(from_ls.len(), n);
    }

    #[test]
    fn invalid_geo_types_inputs_are_rejected() {
        let points = [Point::new(0.0, 0.0), Point::new(0.0, 91.0)];
        let err = solve_order_of(points.iter(), &fast_config()).expect_err("invalid latitude");
        assert!(err.to_string().contains("point 1"), "{err}");
    }
}
