# `lkh` crate

Typed Rust helpers for working with [LKH](https://github.com/blaulan/LKH-3):

- TSPLIB problem modeling/writing
- LKH parameter file modeling/writing
- LKH process execution
- TSPLIB tour parsing

This crate can be used directly from other Rust projects, or as part of this
workspace.

## File Format Specs

- LKH parameter file spec: [`docs/PAR_SPEC.md`](docs/PAR_SPEC.md)
  - Includes modeled LKH-3 keys, value shapes, upstream defaults, and abbreviation notes.
- TSPLIB problem/tour spec: [`docs/TSPLIB_SPEC.md`](docs/TSPLIB_SPEC.md)
  - Includes modeled LKH-3 TSPLIB extensions (`TYPE` variants, additional headers, and sections).

## Features

- `embedded-lkh` (default):
  - Build-time embeds an LKH executable into the Rust artifact.
  - Runtime extracts that embedded executable to the OS temp directory.
  - `LkhSolver::run()` is available only with this feature.
- Without `embedded-lkh`:
  - Use `LkhSolver::run_with_exe(path)` and manage the LKH binary yourself.

## Quickstart

```rust,no_run
use lkh::{
    parameters::LkhParameters,
    problem::TsplibProblem,
    solver::LkhSolver,
};

fn main() -> lkh::LkhResult<()> {
    let problem = TsplibProblem::from_euc2d_points(vec![
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
    ]);

    let params = LkhParameters::new("work/problem.tsp");
    let solver = LkhSolver::new(problem, params)?;
    let tour = solver.run_with_exe("/usr/local/bin/LKH")?;
    let order = tour.zero_based_tour()?;
    println!("{order:?}");
    Ok(())
}
```

## Embedded LKH Security Model

Build script behavior (`embedded-lkh` enabled):

1. Resolve archive from:
   - existing build output cache, then
   - `lkh/lkh.tgz` vendored archive, then
   - download URL(s)
2. Verify SHA-256 of archive before extraction/build.
3. Build LKH (`make` on Unix-like platforms) and embed resulting bytes.

Default pinned archive digest:

- `TSP_MT_LKH_SHA256` fallback value:
  `c2bb3974bbeb016d2e45c56eae34bcb35617e28a6d8b1356de159256bc18ecbf`

Override knobs:

- `TSP_MT_LKH_URL` - custom archive URL.
- `TSP_MT_LKH_SHA256` - expected SHA-256 for custom archive.
- `TSP_MT_ALLOW_INSECURE_HTTP_LKH=1` - allow HTTP for custom URL.
- `TSP_MT_LKH_WINDOWS_EXE_SHA256` - optional checksum pin for repository-root
  `LKH.exe` on Windows.

At runtime, extracted embedded executable bytes are validated against the
embedded payload and written via temporary-file + rename flow.

## Platform Notes

- Linux/macOS (`embedded-lkh`):
  - Requires `tar`, `make`, and `curl` or `wget` available in PATH.
- Windows (`embedded-lkh`):
  - Requires `LKH.exe` present at repository root during build.
  - Build script embeds this file directly.
- Any platform (`no-default-features`):
  - No embedded LKH build.
  - Provide an LKH path via `run_with_exe`.

## Release Checklist

Run the release check script from the workspace root:

```bash
scripts/release-check-lkh.sh
```

For full package verification (requires crates.io network access):

```bash
LKH_RELEASE_FULL_PACKAGE=1 scripts/release-check-lkh.sh
```
