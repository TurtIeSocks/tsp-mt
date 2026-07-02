# tsp-ils

A clean-room TSP heuristic in pure Rust: candidate-list local search
(2-opt + Or-opt) over greedy construction, driven by iterated local search
with double-bridge kicks — optionally multi-core via segment-based
split/join rounds with best-of-k replicas.

Implements well-published techniques from the Lin–Kernighan family
(Lin & Kernighan 1973; Or 1976; Bentley 1992; Helsgaun 2000) from the
algorithm descriptions. **No code from, or derived from, LKH or any other
solver** — everything is original and Apache-2.0.

## Quality

On uniform random instances, gap vs. the Beardwood–Halton–Hammersley
asymptotic optimal estimate at default settings (32 cores):

| n | budget | gap | edges >10× avg |
|---|---|---|---|
| 20k | 15s | +1.1% | 0 |
| 100k | 30s | +0.7% | 0 |
| 1M | 120s | +1.8% | 27 |

Tiny instances (n ≤ 12) reach Held-Karp-verified exact optima in the test
suite. Extra cores buy either wall time (~3× faster to equal quality) or a
lower gap at equal time.

## Usage

```rust
use tsp_ils::{SolverConfig, solve};

// Points in any const-generic dimension (D=2 planar, D=3 sphere-embedded).
let pts: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
let mut cfg = SolverConfig::default();
cfg.time_limit = Some(std::time::Duration::from_secs(5));
let solution = solve(&pts, &cfg);
// solution.tour: visiting order as indices; solution.length: cycle length
```

For geographic (lat/lng) input, use the companion crate
[`tsp-geo`](https://crates.io/crates/tsp-geo).

## Features

- `std` *(default)* — OS clock for wall-time budgets.
- `parallel` *(default, implies `std`)* — multi-core via rayon.
- `libm` — pure-Rust float math for `no_std` builds.

### WebAssembly

- Simplest: `--no-default-features --features std` gives a fully sequential
  build that runs in any browser (wall-clock via `web-time`), no
  cross-origin isolation needed.
- Threads: keep `parallel` and initialize rayon's global pool with
  [wasm-bindgen-rayon](https://github.com/RReverser/wasm-bindgen-rayon)
  (requires COOP/COEP headers). The `threads` config field is ignored on
  wasm — pool size is set at initialization.

### `no_std`

`--no-default-features --features libm` (requires `alloc`). Without a
clock, `time_limit` is ignored; the solver stops on convergence and finite
kick budgets, which also makes results bit-reproducible.

## Benchmarking

```bash
# n points, seconds, threads — reports gap vs the BHH estimate
cargo run --release --example bench -- 100000 30 0
# or a TSPLIB NODE_COORD_SECTION file
cargo run --release --example bench -- path/to/instance.tsp 30 0
```
