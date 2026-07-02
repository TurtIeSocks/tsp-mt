# Benchmark Workflow

This project uses a single benchmark root directory with dedicated subfolders:

- `benchmark/inputs/`: benchmark input `.txt` files (`lat,lng` per line).
- `benchmark/outputs/`: benchmark run route outputs.
- `benchmark/logs/`: benchmark run stderr logs.
- `benchmark/results/`: generated summary artifacts (`.tsv`).

## End-to-end geographic benchmark

```bash
./scripts/benchmark.sh [time_limit_seconds]
```

- Discovers inputs from `benchmark/inputs/*.txt`; if none exist, generates
  clustered instances of 1k/5k/20k/100k points first.
- Runs the release binary on each input and records the metrics line
  (total distance, longest edge, average edge, spike count) plus wall time
  into `benchmark/results/benchmark-<timestamp>.tsv`.
- `time_limit_seconds` defaults to `0` (auto: scales with input size).

Optional environment variables:

- `TSP_MT_BENCH_TIMESTAMP=<timestamp>`: override the timestamp used in
  generated file names.

## Planar quality benchmark (optimality gap)

The core solver crate has a standalone benchmark that reports the gap against
the Beardwood–Halton–Hammersley estimate on uniform random instances, or
solves a TSPLIB `NODE_COORD_SECTION` file:

```bash
cargo run --release -p tsp_solver --example bench -- 100000 30 0
cargo run --release -p tsp_solver --example bench -- path/to/instance.tsp 30 0
```
