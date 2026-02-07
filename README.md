# Multi Threaded Traveling Salesperson

High-performance TSP solver for geographic coordinates (`lat,lng`) using LKH, parallel runs, and H3 chunking for large inputs.

## What It Does

- Reads points from `stdin`
- Solves a route order
- Writes ordered points to `stdout` (one `lat,lng` per line)
- Writes logs/metrics to `stderr`

## Prerequisites

- Rust toolchain with 2024 edition support (Rust 1.85+ recommended)
- Windows users: download `LKH.exe` from `http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.exe` and place it in the repository root (`./LKH.exe`) before building.
- Linux/macOS users: system tools:
  - `make`
  - `tar`
  - `curl` or `wget` (for downloading LKH source during build)

## Setup

1. Clone the repo:

```bash
git clone <your-repo-url>
cd tsp-mt
```

2. Build release binary:

```bash
cargo build --release
```

### Build Binary To A Specific File

Build, then copy the final binary to the exact output path you want:

```bash
cargo build --release
cp target/release/tsp-mt /absolute/path/to/output/my-tsp-mt

# to use as a plugin with koji
# cp target/release/tsp-mt ~/{your_koji_directory}/server/algorithms/src/routing/plugins/
```

Windows PowerShell equivalent:

```powershell
cargo build --release
Copy-Item target\release\tsp-mt.exe C:\absolute\path\to\output\my-tsp-mt.exe
```

### How LKH Is Provided

During build, `build.rs`:

1. On Linux/macOS:
   - Downloads `LKH-3.0.13.tgz` (default URL: `http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.13.tgz`)
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

- `stdout`: ordered route, one `lat,lng` per line
- `stderr`: progress, timing, and distance metrics

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
target/release/tsp-mt [args] < points.txt > output.txt
```

## CLI Arguments

All arguments are long-form flags.
Both `--flag value` and `--flag=value` work.

| Argument                              | Type  |   Default | Notes                                                                  |
| ------------------------------------- | ----- | --------: | ---------------------------------------------------------------------- |
| `--lkh-exe <path>`                    | path  |      auto | Use this LKH executable instead of extracted embedded one              |
| `--work-dir <path>`                   | path  | `./.temp` | Temp/output workspace for run artifacts                                |
| `--projection-radius <f64>`           | float |    `70.0` | Must be `> 0`                                                          |
| `--max-chunk-size <usize>`            | int   |    `5000` | Must be `> 0`; above this input size, H3 chunked solver is used        |
| `--centroid-order-seed <u64>`         | int   |     `999` | Seed for chunk-centroid ordering run                                   |
| `--centroid-order-max-trials <usize>` | int   |   `20000` | LKH `MAX_TRIALS` for centroid ordering                                 |
| `--centroid-order-time-limit <usize>` | int   |      `10` | LKH `TIME_LIMIT` seconds for centroid ordering                         |
| `--boundary-2opt-window <usize>`      | int   |     `500` | Boundary-local 2-opt window during chunk stitching                     |
| `--boundary-2opt-passes <usize>`      | int   |      `50` | Boundary-local 2-opt passes during chunk stitching                     |
| `--verbose` / `--verbose=<bool>`      | bool  |    `true` | Enables progress logs to `stderr`                                      |
| `--no-verbose`                        | flag  |       n/a | Forces `verbose=false`                                                 |
| `--help`, `-h`                        | flag  |       n/a | Prints usage and exits (currently shows input-option usage text first) |

Accepted boolean values for `--verbose=<bool>`:  
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
