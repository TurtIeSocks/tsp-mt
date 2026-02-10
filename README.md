# Multi Threaded Traveling Salesperson

High-performance TSP solver for geographic coordinates (`lat,lng`) using LKH, parallel runs, and H3 chunking for large inputs.

## What It Does

- Reads points from `stdin`
- Solves a route order
- Writes ordered points to `stdout` (one `lat,lng` per line) or `--output <path>`
- Writes logs/metrics to `stderr` (default) or `--log-output <path>`

## Prerequisites

- Rust toolchain with 2024 edition support (Rust 1.90+ recommended)
- Windows users: download `LKH.exe` from `https://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.exe` and place it in the repository root (`./LKH.exe`) before building.
- Linux/macOS users: system tools:
  - `make`
  - `tar`
  - `curl` or `wget` (for downloading LKH source during build)

## Setup

1. Clone the repo:

```bash
git clone https://github.com/TurtIeSocks/tsp-mt.git
cd tsp-mt
```

2. Build release binary:

```bash
cargo build --release
```

### Use With [Koji](https://github.com/TurtIeSocks/Koji)

Build, then copy the final binary to Koji's routing plugins folder:

```bash
cargo build --release
cp target/release/tsp-mt ~/{your_koji_directory}/server/algorithms/src/routing/plugins/
```

### How LKH Is Provided

During build, `build.rs`:

1. On Linux/macOS:
   - Downloads `LKH-3.0.13.tgz` (default URL: `https://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.13.tgz`)
   - Builds `LKH` with `make`
   - Embeds the resulting executable bytes into this binary
2. On Windows:
   - Reads `./LKH.exe` from the repository root
   - Embeds it into this binary

Runtime then extracts the embedded LKH binary into your OS temp directory (unless you pass `--lkh-exe`).

You can override the download URL:

```bash
TSP_MT_LKH_URL='<your-url-to-LKH-3.0.13.tgz>' cargo build --release
```

If download fails, build also looks for a vendored archive at `lkh/lkh.tgz`.

## Input Format (`stdin`)

- Whitespace-separated tokens, each exactly: `lat,lng`
- Newlines and spaces are both fine separators and can be mixed
- Latitude must be in `[-90, 90]`
- Longitude must be in `[-180, 180]`
- Provide at least 3 points for normal TSP cycle solving

Example valid input:

```text
37.7749,-122.4194
34.0522,-118.2437 36.1699,-115.1398
```

## Output Format

- `stdout`: ordered route, one `lat,lng` per line (default)
- `--output <path>`: ordered route written to the specified file
- `stderr`: progress, timing, and distance metrics (default)
- `--log-output <path>`: logs/metrics written to the specified file instead of stderr

## Run

Use either:

```bash
cargo run --release -- [args] < points.txt
```

or (cleaner output, no Cargo compile logs):

```bash
target/release/tsp-mt [args] < points.txt
```

or to save output to file:

```bash
target/release/tsp-mt [args] --output output.txt < points.txt
```

or to save logs to file:

```bash
target/release/tsp-mt [args] --log-output run.log < points.txt
```

## CLI Arguments

All arguments are long-form flags.
Both `--flag value` and `--flag=value` work.

| Argument                              | Type                |                  Default | Notes                                                                         |
| ------------------------------------- | ------------------- | -----------------------: | ----------------------------------------------------------------------------- |
| `--lkh-exe <path>`                    | path                |                   `auto` | Use this LKH executable instead of extracted embedded one                     |
| `--work-dir <path>`                   | path                | `<os-temp>/tsp-mt-<pid>` | Temp/output workspace for run artifacts                                       |
| `--output <path>`                     | path                |                 `stdout` | Write ordered route points to this file instead of stdout                     |
| `--projection-radius <f64>`           | float               |                   `70.0` | Must be `> 0`                                                                 |
| `--max-chunk-size <usize>`            | int                 |                   `5000` | Must be `> 0`; above this input size, H3 chunked solver is used               |
| `--centroid-order-seed <u64>`         | int                 |                    `999` | Seed for chunk-centroid ordering run                                          |
| `--centroid-order-max-trials <usize>` | int                 |                  `20000` | LKH `MAX_TRIALS` for centroid ordering                                        |
| `--centroid-order-time-limit <usize>` | int                 |                     `10` | LKH `TIME_LIMIT` seconds for centroid ordering                                |
| `--boundary-2opt-window <usize>`      | int                 |                    `500` | Boundary-local 2-opt window during chunk stitching                            |
| `--boundary-2opt-passes <usize>`      | int                 |                     `50` | Boundary-local 2-opt passes during chunk stitching                            |
| `--outlier-threshold <f64>`           | float               |                   `10.0` | Distance threshold (meters) for counting spike/outlier jumps in route metrics |
| `--log-level <value>`                 | enum                |                   `warn` | One of: `error`, `warn`, `warning`, `info`, `debug`, `trace`, `off`           |
| `--log-format <value>`                | enum                |                `compact` | One of: `compact`, `pretty`                                                   |
| `--log-timestamp[=<bool>]`            | bool (optional val) |                   `true` | If provided without value, sets to `true`                                     |
| `--no-log-timestamp`                  | flag                |                    `n/a` | Forces `log_timestamp=false`                                                  |
| `--log-output <path>`                 | path                |                 `stderr` | Write logs/metrics to this file instead of stderr                             |
| `--help`, `-h`                        | flag                |                    `n/a` | Prints usage and exits                                                        |

Accepted boolean values for `--log-timestamp=<bool>`:  
`1/0`, `true/false`, `yes/no`, `on/off` (common case variants supported).

## Common Errors

- `No points provided on stdin.`  
  You did not pipe or redirect any input.

- `Need at least 3 points for a cycle`  
  Add at least 3 points.

- `Input contains invalid lat/lng values`  
  At least one point is out of bounds or non-finite.

- Build error downloading LKH archive  
  Ensure `curl`/`wget` and network access, set `TSP_MT_LKH_URL`, or place `lkh/lkh.tgz`.
