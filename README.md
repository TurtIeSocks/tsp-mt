# Multi Threaded Traveling Salesperson

High-performance, multi-core TSP solver for geographic coordinates (`lat,lng`),
fully self-contained in Rust — no external solver binary required.

## What It Does

- Reads points from `stdin` or `--input <path>`
- Solves a route order with a built-in clean-room heuristic
- Writes ordered points to `stdout` (one `lat,lng` per line) or `--output <path>`
- Writes logs/metrics to `stderr` (default) or `--log-output <path>`

## How It Works

The solver (crate `tsp_ils`) is an original implementation of well-published
techniques from the Lin–Kernighan family of TSP heuristics. It contains no code
from LKH or any other solver — only the ideas, implemented from the literature
(Lin & Kernighan 1973; Or 1976; Bentley 1992; Helsgaun 2000):

1. **Spherical embedding** — points are embedded on an Earth-radius sphere in
   3D; the solver optimizes Euclidean (chord) distance, which is monotone in
   great-circle distance. No tangent-plane projection, so there is no
   distortion for large extents, poles, or the date line.
2. **Candidate lists** — a k-d tree provides symmetrized k-nearest-neighbor
   lists; local search only creates geometrically promising edges.
3. **Greedy construction** — shortest candidate edges first, then nearest-
   endpoint chaining of leftover path fragments.
4. **2-opt + Or-opt local search** — with don't-look bits and a work queue;
   2-opt reverses the shorter arc, Or-opt relocates 1–3 node segments in
   either orientation.
5. **Iterated local search** — windowed double-bridge kicks with best-tour
   restoration, escalating perturbation while budget remains.
6. **Multi-core split/join** — the tour is split into contiguous segments,
   each optimized independently as a fixed-endpoint sub-problem on its own
   core, then re-joined; boundaries rotate every round so the whole tour gets
   interior-quality optimization. Small inputs run one independent ILS walker
   per core instead.
7. **Spike repair** — endpoints of unusually long edges get a targeted pass
   with extended Or-opt lengths, directly reducing route outliers.

On uniform random instances the solver lands within ~0.7–1.8% of the
asymptotic optimal-tour estimate (Beardwood–Halton–Hammersley) at default
settings (100k points, 30s, 32 cores: +0.72%; 1M points, 120s: +1.79%), with
near-zero outlier spikes. Extra cores buy either wall time (32 threads reach
1-thread quality ~3× sooner) or quality (lower gap at equal time).

## Prerequisites

- Rust toolchain with 2024 edition support (Rust 1.90+ recommended)

That's it — there is no external solver to download and no licensing caveat:
everything is Apache-2.0.

## Setup

```bash
git clone https://github.com/TurtIeSocks/tsp-mt.git
cd tsp-mt
cargo build --release
```

### Use With [Koji](https://github.com/TurtIeSocks/Koji)

Build, then copy the final binary to Koji's routing plugins folder:

```bash
cp target/release/tsp-mt ~/{your_koji_directory}/server/algorithms/src/routing/plugins/
```

## Input Format

### `stdin` (default)

- Whitespace-separated tokens, each exactly: `lat,lng`
- Newlines and spaces are both valid separators and can be mixed
- Latitude must be in `[-90, 90]`
- Longitude must be in `[-180, 180]`

Example valid input:

```text
37.7749,-122.4194
34.0522,-118.2437 36.1699,-115.1398
```

### `--input <path>` (file mode)

- File must be UTF-8 raw text
- Exactly one `lat,lng` row per line
- Blank lines are rejected
- Space-separated tokens on the same line are rejected in file mode

## Output Format

- `stdout`: ordered route, one `lat,lng` per line (default)
- `--output <path>`: ordered route written to the specified file
- `stderr`: progress, timing, and distance metrics (default)
- `--log-output <path>`: logs/metrics written to the specified file instead

## Run Examples

```bash
# stdin -> stdout
target/release/tsp-mt < points.txt > route.txt

# files, with a 30 second budget
target/release/tsp-mt --input points.txt --output route.txt --time-limit 30

# reproducible run on 8 threads with metrics
target/release/tsp-mt --threads 8 --seed 42 --log-level=info < points.txt
```

## CLI Arguments

All arguments are long-form flags. Both `--flag value` and `--flag=value` work.

| Argument                    | Type                |      Default | Notes                                                                |
| --------------------------- | ------------------- | -----------: | -------------------------------------------------------------------- |
| `--input <path>`            | path                |      `stdin` | Read points from this file instead of stdin; UTF-8 `lat,lng` rows     |
| `--output <path>`           | path                |     `stdout` | Write ordered route points to this file instead of stdout             |
| `--time-limit <seconds>`    | float               | `0` (auto)   | Wall-clock budget; `0` derives one from the input size (2s–120s)      |
| `--seed <u64>`              | int                 |      `12345` | RNG seed for the perturbation phase                                   |
| `--threads <usize>`         | int                 |    `0` (all) | Worker threads; `0` uses all available cores                          |
| `--max-neighbors <usize>`   | int                 |         `10` | Candidate edges per point; higher = better quality, slower            |
| `--outlier-threshold <f64>` | float               |       `10.0` | Spike threshold as a multiple of the average edge, for metrics logs   |
| `--log-level <value>`       | enum                |       `warn` | One of: `error`, `warn`, `warning`, `info`, `debug`, `trace`, `off`   |
| `--log-format <value>`      | enum                |    `compact` | One of: `compact`, `pretty`                                           |
| `--log-timestamp[=<bool>]`  | bool (optional val) |       `true` | If provided without value, sets to `true`                             |
| `--no-log-timestamp`        | flag                |        `n/a` | Forces `log_timestamp=false`                                          |
| `--log-output <path>`       | path                |     `stderr` | Write logs/metrics to this file instead of stderr                     |
| `--help`, `-h`              | flag                |        `n/a` | Prints usage and exits                                                |

Accepted boolean values for `--log-timestamp=<bool>`:
`1/0`, `true/false`, `yes/no`, `on/off` (common case variants supported).

## Benchmarking

Planar quality benchmark against the Beardwood–Halton–Hammersley estimate
(asymptotic optimal length for uniform random points):

```bash
# n points, seconds, threads
cargo run --release -p tsp-ils --example bench -- 100000 30 0

# or a TSPLIB file (NODE_COORD_SECTION / EUC_2D)
cargo run --release -p tsp-ils --example bench -- path/to/instance.tsp 30 0
```

End-to-end geographic benchmark:

```bash
./scripts/benchmark.sh
```

See [benchmark/README.md](benchmark/README.md).

## Common Errors

- `No points provided on stdin.`
  You did not pipe or redirect any input.

- `Input file must be UTF-8 raw text: ...`
  The `--input` file is not valid UTF-8 text.

- `Input file ... line N: expected exactly one 'lat,lng' row`
  The `--input` file has invalid row formatting (for example multiple tokens on one line).

- `... invalid lat/lng values: ...`
  At least one point is out of bounds or non-finite; the message names the
  offending token or file line.

## License & Attribution

Apache-2.0 for the entire workspace. The solver implements algorithms
*described* in the published TSP literature; it is not a port of, and shares
no code with, LKH or any other solver implementation.
