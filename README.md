# Multi Threaded Traveling Salesperson

High-performance TSP solver for geographic coordinates (`lat,lng`) utilizing parallel runs, and H3 chunking for large inputs.

## What It Does

- Reads points from `stdin`
- Or reads points from `--input <path>`
- Solves a route order
- Writes ordered points to `stdout` (one `lat,lng` per line) or `--output <path>`
- Writes logs/metrics to `stderr` (default) or `--log-output <path>`

## Prerequisites

- Rust toolchain with 2024 edition support (Rust 1.90+ recommended)

### Using an external LKH binary (default)

If you do **not** enable the `fetch-lkh` feature, you must provide an LKH executable at runtime via `--lkh-exe <path>`.

### Embedding LKH at build time (`fetch-lkh` feature)

- Windows:
  - Download `LKH.exe` from:
    http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.exe
  - Provide it to the build either by:
    - setting `LKH_EXE_PATH` to the full path of `LKH.exe` (recommended), or
    - placing `LKH.exe` in the repository root (do **not** commit it)
- Linux/macOS:
  - system tools:
    - `make`
    - `tar`
    - `curl` or `wget` (to download LKH source during build)

## Setup

1. Clone the repo:

```bash
git clone https://github.com/TurtIeSocks/tsp-mt.git
cd tsp-mt
```

2. Build the release binary.

### Default build (external LKH at runtime)

```bash
cargo build --release
```

At runtime, pass your LKH executable:

```bash
target/release/tsp-mt --lkh-exe /path/to/LKH [args] < points.txt
```

### Build with embedded LKH (`fetch-lkh`)

```bash
# Linux/macOS: downloads + builds LKH during `cargo build`
cargo build --release --features fetch-lkh

# Windows: you must provide LKH.exe yourself
# PowerShell:
$env:LKH_EXE_PATH="C:\path\to\LKH.exe"; cargo build --release --features fetch-lkh
```

### Use With [Koji](https://github.com/TurtIeSocks/Koji)

Build, then copy the final binary to Koji's routing plugins folder:

```bash
# build with script above ^
cp target/release/tsp-mt ~/{your_koji_directory}/server/algorithms/src/routing/plugins/
```

### How LKH Is Provided

During build, `build.rs` only runs when the `fetch-lkh` feature is enabled.

- Linux/macOS (`fetch-lkh`):
  - Downloads the LKH source archive (default is a pinned GitHub source mirror URL)
  - Verifies the archive SHA-256
  - Builds `LKH` with `make`
  - Embeds the resulting executable bytes into this binary

- Windows (`fetch-lkh`):
  - Uses `LKH_EXE_PATH` if set; otherwise falls back to `./LKH.exe` in the repository root
  - (Optional) Verifies SHA-256 if you set `TSP_MT_LKH_WINDOWS_EXE_SHA256`
  - Embeds the executable bytes into this binary

At runtime, the embedded LKH binary is extracted into your OS temp directory (unless you pass `--lkh-exe`).

Overrides:

- `TSP_MT_LKH_URL`: override the source archive URL (Linux/macOS only)
- `TSP_MT_LKH_SHA256`: override the expected SHA-256 for the archive
- `LKH_EXE_PATH`: provide a path to `LKH.exe` on Windows
- `TSP_MT_LKH_WINDOWS_EXE_SHA256`: verify the Windows `LKH.exe` by SHA-256

## Input Format

### `stdin` (default)

- Whitespace-separated tokens, each exactly: `lat,lng`
- Newlines and spaces are both valid separators and can be mixed
- Latitude must be in `[-90, 90]`
- Longitude must be in `[-180, 180]`
- Provide at least 3 points for normal TSP cycle solving

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
- Latitude must be in `[-90, 90]`
- Longitude must be in `[-180, 180]`

## Output Format

- `stdout`: ordered route, one `lat,lng` per line (default)
- `--output <path>`: ordered route written to the specified file
- `stderr`: progress, timing, and distance metrics (default)
- `--log-output <path>`: logs/metrics written to the specified file instead of stderr

## Run Examples

Basic usage. Reads points via stdin and saves them via stdout.

```bash
cargo run --release -- [args] < points.txt > output.txt
```

Cleaner output, no Cargo compile logs.

```bash
target/release/tsp-mt [args] < points.txt > output.txt
```

Read points from a file directly (no stdin redirect):

```bash
target/release/tsp-mt [args] --input points.txt --output output.txt
```

To save output to a file rather than capturing it via `stdout`.

```bash
target/release/tsp-mt [args] --output output.txt < points.txt
```

To save logs to file:

```bash
target/release/tsp-mt [args] --output output.txt --log-output run.log < points.txt
```

## CLI Arguments

All arguments are long-form flags.
Both `--flag value` and `--flag=value` work.

| Argument                              | Type                |                  Default | Notes                                                                         |
| ------------------------------------- | ------------------- | -----------------------: | ----------------------------------------------------------------------------- |
| `--lkh-exe <path>`                    | path                |                   `auto` | Use this LKH executable instead of extracted embedded one                     |
| `--work-dir <path>`                   | path                | `<os-temp>/tsp-mt-<pid>` | Temp/output workspace for run artifacts                                       |
| `--input <path>`                      | path                |                  `stdin` | Read points from this file instead of stdin; requires UTF-8 `lat,lng` rows    |
| `--output <path>`                     | path                |                 `stdout` | Write ordered route points to this file instead of stdout                     |
| `--projection-radius <f64>`           | float               |                   `70.0` | Must be `> 0`                                                                 |
| `--max-chunk-size <usize>`            | int                 |                   `5000` | Must be `> 0`; above this input size, H3 chunked solver is used               |
| `--centroid-order-seed <u64>`         | int                 |                    `999` | Seed for chunk-centroid ordering run                                          |
| `--centroid-order-max-trials <usize>` | int                 |                  `20000` | LKH `MAX_TRIALS` for centroid ordering                                        |
| `--centroid-order-time-limit <usize>` | int                 |                     `10` | LKH `TIME_LIMIT` seconds for centroid ordering                                |
| `--solver-mode <value>`               | enum                |         `multi-parallel` | One of: `single`, `multi-seed`, `multi-parallel`                              |
| `--boundary-2opt-window <usize>`      | int                 |                    `500` | Boundary-local 2-opt window during chunk stitching                            |
| `--boundary-2opt-passes <usize>`      | int                 |                     `50` | Boundary-local 2-opt passes during chunk stitching                            |
| `--spike-repair-top-n <usize>`        | int                 |                     `48` | Number of longest edges targeted by post-stitch spike-repair                  |
| `--spike-repair-window <usize>`       | int                 |                    `700` | 2-opt window used in post-stitch spike-repair                                 |
| `--spike-repair-passes <usize>`       | int                 |                      `5` | 2-opt passes used in post-stitch spike-repair                                 |
| `--outlier-threshold <f64>`           | float               |                   `10.0` | Distance threshold (meters) for counting spike/outlier jumps in route metrics |
| `--cleanup[=<bool>]`                  | bool (optional val) |                   `true` | If provided without value, sets to `true`                                     |
| `--no-cleanup`                        | flag                |                    `n/a` | Forces `cleanup=false`                                                        |
| `--log-level <value>`                 | enum                |                   `warn` | One of: `error`, `warn`, `warning`, `info`, `debug`, `trace`, `off`           |
| `--log-format <value>`                | enum                |                `compact` | One of: `compact`, `pretty`                                                   |
| `--log-timestamp[=<bool>]`            | bool (optional val) |                   `true` | If provided without value, sets to `true`                                     |
| `--no-log-timestamp`                  | flag                |                    `n/a` | Forces `log_timestamp=false`                                                  |
| `--log-output <path>`                 | path                |                 `stderr` | Write logs/metrics to this file instead of stderr                             |
| `--help`, `-h`                        | flag                |                    `n/a` | Prints usage and exits                                                        |

Accepted boolean values for `--log-timestamp=<bool>` and `--cleanup=<bool>`:  
`1/0`, `true/false`, `yes/no`, `on/off` (common case variants supported).

## Common Errors

- `No points provided on stdin.`  
  You did not pipe or redirect any input.

- `Input file must be UTF-8 raw text: ...`  
  The `--input` file is not valid UTF-8 text.

- `Input file ... line N: expected exactly one 'lat,lng' row`  
  The `--input` file has invalid row formatting (for example multiple tokens on one line).

- `Need at least 3 points for a cycle`  
  Add at least 3 points.

- `Input contains invalid lat/lng values`  
  At least one point is out of bounds or non-finite.

- Build error downloading LKH archive  
  Ensure `curl`/`wget` and network access, set `TSP_MT_LKH_URL`.

## LKH dependency notice

This project can optionally use the LKH-3 solver by Keld Helsgaun.

LKH is **not open source software** and is distributed under a
research-only license by its author. It is **not included** in this
repository.

If you enable LKH support, the build process will download LKH source
code from a public source mirror and compile it locally on your
machine. By enabling this functionality, **you explicitly agree to
comply with the LKH license terms**, and you are responsible for any
use or redistribution.

LKH license and homepage:
http://webhotel4.ruc.dk/~keld/research/LKH-3/

To enable LKH support, you must enable the `fetch-lkh` cargo feature. By enabling the `fetch-lkh` feature, you acknowledge and agree to comply
with the LKH license terms.

```bash
cargo build --release --features fetch-lkh
```
