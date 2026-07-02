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

## Features

- `std` *(default)* — OS clock for wall-time budgets.
- `parallel` *(default, implies `std`)* — multi-core via rayon.
- `libm` — pure-Rust float math for `no_std` builds (requires `alloc`).

For WebAssembly notes (sequential builds vs. wasm-bindgen-rayon threads),
see the [`tsp-ils` README](https://crates.io/crates/tsp-ils).

## License

Apache-2.0. The underlying solver is a clean-room implementation — no code
from LKH or any other solver.
