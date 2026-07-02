# tsp-geo

TSP route ordering for geographic coordinates (`lat,lng`), built on the
[`tsp-ils`](https://crates.io/crates/tsp-ils) heuristic.

Points are embedded on an Earth-radius sphere in 3D, so the solver
optimizes chord distance — indistinguishable from great-circle distance at
routing scales, with **no tangent-plane distortion** at poles, the date
line, or continental extents.

## Usage

```rust
use tsp_geo::{GeoPoint, SolverConfig, solve, solve_order};

let points = vec![
    GeoPoint::from_lat_lng(52.5200, 13.4050), // Berlin
    GeoPoint::from_lat_lng(48.8566, 2.3522),  // Paris
    GeoPoint::from_lat_lng(51.5074, -0.1278), // London
];

// Points in visiting order...
let route = solve(&points, &SolverConfig::default()).unwrap();
// ...or indices into `points` (handy for wasm/JS interop)
let order = solve_order(&points, &SolverConfig::default()).unwrap();
```

`Tour::tour_metrics` reports haversine totals, average edge, and
outlier-spike counts for a solved route.

### georust interop (`rustgeo` feature)

With the `rustgeo` feature, [`geo-types`](https://docs.rs/geo-types)
primitives convert to and from `GeoPoint`, and `solve_order_of` accepts any
iterator of convertible items — `Point`s, `Coord`s, references to either,
or a `LineString`'s `.points()`:

```rust
use geo_types::Point;
use tsp_geo::{SolverConfig, solve_order_of};

let stops = vec![
    Point::new(13.4050, 52.5200), // x = lng, y = lat
    Point::new(2.3522, 48.8566),
    Point::new(-0.1278, 51.5074),
];
let order = solve_order_of(stops.iter().copied(), &SolverConfig::default()).unwrap();
```

## Features

- `std` *(default)* — OS clock for wall-time budgets.
- `parallel` *(default, implies `std`)* — multi-core via rayon.
- `libm` — pure-Rust float math for `no_std` builds (requires `alloc`).
- `rustgeo` — `From` conversions for georust `geo-types` (`Coord`, `Point`);
  composes with every other feature, including `no_std`.

For WebAssembly notes (sequential builds vs. wasm-bindgen-rayon threads),
see the [`tsp-ils` README](https://crates.io/crates/tsp-ils).

## License

Apache-2.0. The underlying solver is a clean-room implementation — no code
from LKH or any other solver.
