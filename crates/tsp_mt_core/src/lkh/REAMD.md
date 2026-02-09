# LKH Module

## File Overview

- `config.rs`
  - Lives in the separate `lkh` crate.
  - Defines `LkhConfig`, the full LKH-3.0 parameter model.
  - Contains typed enums/structs for constrained parameter values.
  - Renders/writes `.par` files via `render()` and `write_to_file()`.
- `problem.rs`
  - Defines `TsplibProblemWriter`.
  - Writes projected points to TSPLIB EUC_2D format (`problem.tsp` style files).
- `process.rs`
  - Defines `LkhProcess`.
  - `ensure_success(...)` wraps process status + stdout/stderr into `Error::ProcessFailed`.
- `tour.rs` (in the `lkh` crate)
  - Defines `TsplibTour`.
  - `parse_tsplib_tour(...)` parses TSPLIB `TOUR_SECTION` into zero-based indices.
- `solver.rs`
  - Defines `LkhSolver` and `solve_tsp_with_lkh_parallel(...)`.
  - Owns file naming/path conventions under a work directory.
  - Runs LKH preprocessing and parallel multi-seed runs.
- `embedded_lkh.rs`
  - Writes embedded LKH binary to temp location and sets executable permissions.
- `mod.rs`
  - Re-exports module parts for crate-internal use.

## Main Parallel Solve Flow

Entry point: `solve_tsp_with_lkh_parallel(input, options)` in `solver.rs`.

1. Validate input and runtime options.
1. Build `LkhSolver` with `lkh_exe` + `work_dir`.
1. Project lat/lng points to local plane (`PlaneProjection`).
1. Write TSPLIB problem file (`problem.tsp`) using `TsplibProblemWriter`.
1. Ensure candidate/PI files exist:
   - Build preprocessing config in `solver.rs` from `LkhConfig::new(...)`.
   - Write `prep_candidates.par`.
   - Execute LKH once.
   - Verify `problem.cand` was produced.
1. Build base run config in `solver.rs` from `LkhConfig::new(...)`.
1. Generate deterministic per-run seeds from config base seed.
1. For each parallel run:
   - Clone config and set run-specific `SEED` + `OUTPUT_TOUR_FILE`.
   - Write `run_{idx}.par`.
   - Execute LKH.
   - Parse `run_{idx}.tour` via `TsplibTour::parse_tsplib_tour(...)`.
   - Score tour length.
1. Pick best run by total distance and return ordered `Tour`.
1. Cleanup work dir.

## Chunked Mode Interaction

Chunked solve lives in `src/algo/chunked.rs`, but uses this module directly:

- Per chunk:
  - Reuses `LkhSolver` + `parallel_run_config(...)`.
  - Executes a single run (`run_0.par`/`run_0.tour`) for local chunk ordering.
- Centroid ordering:
  - Builds standalone config in `chunked.rs` from `LkhConfig::new(...)`.
  - Writes `centroids.par` + reads `centroids.tour`.

So chunked and non-chunked modes share the same config rendering and tour parsing behavior.

## File Convention (Inside Work Dir)

- Problem artifacts:
  - `problem.tsp`
  - `problem.cand`
  - `problem.pi`
- Preprocessing artifacts:
  - `prep_candidates.par`
  - `prep_candidates.tour`
- Main run artifacts:
  - `run_{idx}.par`
  - `run_{idx}.tour`

(Chunked mode additionally creates per-chunk directories and centroid files.)

## Design Notes

- `LkhConfig` is intentionally broad: it models the full LKH-3.0 parameter surface.
- Runtime flows set only a subset of parameters needed for this crate.
- Removed legacy split between run-specific and shared parameter writers (`RunSpec`); one config model now owns parameter emission.
- Error boundaries:
  - Process errors become `Error::ProcessFailed` with captured stdout/stderr.
  - Tour parse/shape mismatches become `Error::InvalidData`.

## When Extending

- Add or adjust parameter behavior in `LkhConfig` first.
- Keep solver/chunked code focused on orchestration (file paths, process execution, selection).
- Reuse `TsplibTour::parse_tsplib_tour(...)` for any new tour read path.
