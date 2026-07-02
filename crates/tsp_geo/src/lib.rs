//! # tsp-geo
//!
//! TSP route ordering for geographic coordinates (`lat,lng`), built on the
//! [`tsp_ils`] solver.
//!
//! Points are embedded on an Earth-radius sphere in 3D, so the solver
//! optimizes chord distance — indistinguishable from great-circle distance
//! at routing scales, with no tangent-plane distortion at poles, the date
//! line, or continental extents.
//!
//! ```no_run
//! use tsp_geo::{GeoPoint, SolverConfig, solve};
//!
//! let points = vec![
//!     GeoPoint::from_lat_lng(52.5200, 13.4050),
//!     GeoPoint::from_lat_lng(48.8566, 2.3522),
//!     GeoPoint::from_lat_lng(51.5074, -0.1278),
//! ];
//! let route = solve(&points, &SolverConfig::default()).unwrap();
//! ```
//!
//! With the `rustgeo` feature, georust [`geo-types`](https://docs.rs/geo-types)
//! values (`Coord`, `Point`) convert to and from [`GeoPoint`], and
//! [`solve_order_of`] accepts iterators of them directly.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod error;
mod fmath;
mod node;
#[cfg(feature = "rustgeo")]
mod rustgeo;
mod solve;
mod tour;

pub use error::{Error, Result};
pub use node::GeoPoint;
pub use solve::{SolverConfig, solve, solve_order, solve_order_of};
pub use tour::{Tour, TourMetrics};

/// Re-export of the underlying solver crate for advanced configuration.
pub use tsp_ils;
