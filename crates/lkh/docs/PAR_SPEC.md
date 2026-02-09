# LKH `.par` File Specification (as modeled by this crate)

This document describes the LKH-3 parameter-file surface supported by
`lkh::parameters::LkhParameters`.

Reference baseline:

- LKH-3.0.13 parameter behavior (`SRC/ReadParameters.c` in the upstream tarball)
- LKH parameter guide docs shipped with LKH

## Line Syntax

- Assignment lines:
  - `KEY = VALUE`
- Comment lines:
  - `COMMENT text...`
  - `# text...`
- Standalone macro keyword:
  - `SPECIAL`
- Optional terminator:
  - `EOF`

Upstream parsing semantics:

- Keywords are case-insensitive.
- Parameter order is arbitrary.

## Writer Ordering Rules

When serialized via `Display` / `write_to_file`:

1. `PROBLEM_FILE` is always written first.
2. Remaining assignment keys are emitted in alphabetical order.
3. Repeated keys (`CANDIDATE_FILE`, `EDGE_FILE`, `MERGE_TOUR_FILE`) are emitted once per entry.
4. `COMMENT ...` / `# ...` lines are emitted where `comment_lines` appears.
5. `SPECIAL` is emitted when `special == true`.
6. `EOF` is emitted only when `emit_eof == true`.

## Supported Keys

The crate models and writes the following LKH-3 parameter keywords:

- `PROBLEM_FILE` (required)
- `ASCENT_CANDIDATES`
- `BACKBONE_TRIALS`
- `BACKTRACKING`
- `BWTSP`
- `CANDIDATE_FILE` (repeatable)
- `CANDIDATE_SET_TYPE`
- `COMMENT` / `#`
- `DEPOT`
- `DISTANCE`
- `EDGE_FILE` (repeatable)
- `EXCESS`
- `EXTERNAL_SALESMEN`
- `EXTRA_CANDIDATES`
- `EXTRA_CANDIDATE_SET_TYPE`
- `GAIN23`
- `GAIN_CRITERION`
- `INITIAL_PERIOD`
- `INITIAL_STEP_SIZE`
- `INITIAL_TOUR_ALGORITHM`
- `INITIAL_TOUR_FILE`
- `INITIAL_TOUR_FRACTION`
- `INPUT_TOUR_FILE`
- `K`
- `KICKS`
- `KICK_TYPE`
- `MAKESPAN`
- `MAX_BREADTH`
- `MAX_CANDIDATES`
- `MAX_SWAPS`
- `MAX_TRIALS`
- `MERGE_TOUR_FILE` (repeatable)
- `MOVE_TYPE`
- `MTSP_MAX_SIZE`
- `MTSP_MIN_SIZE`
- `MTSP_OBJECTIVE`
- `MTSP_SOLUTION_FILE`
- `NONSEQUENTIAL_MOVE_TYPE`
- `OPTIMUM`
- `OUTPUT_TOUR_FILE`
- `PATCHING_A`
- `PATCHING_C`
- `PI_FILE`
- `POPMUSIC_INITIAL_TOUR`
- `POPMUSIC_MAX_NEIGHBORS`
- `POPMUSIC_SAMPLE_SIZE`
- `POPMUSIC_SOLUTIONS`
- `POPMUSIC_TRIALS`
- `POPULATION_SIZE`
- `PRECISION`
- `PROBABILITY`
- `RECOMBINATION`
- `RESTRICTED_SEARCH`
- `RUNS`
- `SALESMEN`
- `SCALE`
- `SEED`
- `SINTEF_SOLUTION_FILE`
- `SPECIAL` (standalone macro line)
- `STOP_AT_OPTIMUM`
- `SUBGRADIENT`
- `SUBPROBLEM_SIZE`
- `SUBPROBLEM_TOUR_FILE`
- `SUBSEQUENT_MOVE_TYPE`
- `SUBSEQUENT_PATCHING`
- `TIME_LIMIT`
- `TOTAL_TIME_LIMIT`
- `TOUR_FILE`
- `TRACE_LEVEL`
- `VEHICLES`
- `EOF` (optional trailing marker)

## Value Types

- Integer-like keys use `usize`, `u64` (`SEED`), or `i64` (`MTSP_MIN_SIZE`).
- Floating-point keys use `f64`.
- Path-like keys use `PathBuf`.
- Boolean switches use `YES` / `NO` via `YesNo`.

## Enumerated Value Domains

### `YES | NO`

- `YES`
- `NO`

### `CANDIDATE_SET_TYPE`

- `ALPHA`
- `DELAUNAY`
- `DELAUNAY PURE`
- `NEAREST-NEIGHBOR`
- `POPMUSIC`
- `QUADRANT`

### `EXTRA_CANDIDATE_SET_TYPE`

- `NEAREST-NEIGHBOR`
- `POPMUSIC`
- `QUADRANT`

### `INITIAL_TOUR_ALGORITHM`

- `BORUVKA`
- `CTSP`
- `CVRP`
- `GCTSP`
- `GREEDY`
- `MOORE`
- `MTSP`
- `NEAREST-NEIGHBOR`
- `PCTSP`
- `QUICK-BORUVKA`
- `SIERPINSKI`
- `SOP`
- `TSPDL`
- `WALK`

### `MTSP_OBJECTIVE`

- `MINMAX`
- `MINMAX_SIZE`
- `MINSUM`

### `RECOMBINATION`

- `IPT`
- `GPX2`
- `CLARIST`

### `PATCHING_A` / `PATCHING_C` mode suffix

- `RESTRICTED`
- `EXTENDED`

### `SUBPROBLEM_SIZE` partitioning suffix

- `DELAUNAY`
- `KARP`
- `K-CENTER`
- `K-MEANS`
- `MOORE`
- `ROHE`
- `SIERPINSKI`

## Composite Value Shapes

### `BWTSP`

Modeled by `BwtspSpec`:

- `<B> <Q>`
- `<B> <Q> <L>`

### `MAX_CANDIDATES` / `EXTRA_CANDIDATES`

Modeled by `CandidateLimit`:

- `<N>`
- `<N> SYMMETRIC`

### `MOVE_TYPE` / `SUBSEQUENT_MOVE_TYPE`

Modeled by integer field + optional `*_special` boolean:

- `<K>`
- `<K> SPECIAL`

### `PATCHING_A` / `PATCHING_C`

Modeled by `PatchingRule`:

- `<MAX_CYCLES>`
- `<MAX_CYCLES> RESTRICTED`
- `<MAX_CYCLES> EXTENDED`

### `SUBPROBLEM_SIZE`

Modeled by `SubproblemSpec`:

- `<SIZE>`
- `<SIZE> <PARTITIONING>`
- Optional suffix flags:
  - `BORDERS`
  - `COMPRESSED`

## Upstream Defaults (modeled keys)

LKH applies these defaults internally when keys are omitted. This crate writes
only fields you set.

| Key | Upstream default |
| --- | --- |
| `PROBLEM_FILE` | required |
| `ASCENT_CANDIDATES` | `50` |
| `BACKBONE_TRIALS` | `0` |
| `BACKTRACKING` | `NO` |
| `BWTSP` | `0 0` (`L` internally `INT_MAX`) |
| `CANDIDATE_SET_TYPE` | `ALPHA` |
| `DEPOT` | `1` |
| `DISTANCE` | `DBL_MAX` |
| `EXCESS` | `1.0 / DIMENSION` (derived) |
| `EXTERNAL_SALESMEN` | `0` |
| `EXTRA_CANDIDATES` | `0` |
| `EXTRA_CANDIDATE_SET_TYPE` | `QUADRANT` |
| `GAIN23` | `YES` |
| `GAIN_CRITERION` | `YES` |
| `INITIAL_PERIOD` | `max(DIMENSION / 2, 100)` |
| `INITIAL_STEP_SIZE` | `1` |
| `INITIAL_TOUR_ALGORITHM` | `WALK` |
| `INITIAL_TOUR_FRACTION` | `1.0` |
| `KICKS` | `1` |
| `KICK_TYPE` | `0` |
| `MAKESPAN` | `NO` |
| `MAX_BREADTH` | `INT_MAX` |
| `MAX_CANDIDATES` | `5` |
| `MAX_SWAPS` | `DIMENSION` |
| `MAX_TRIALS` | `DIMENSION` |
| `MOVE_TYPE` | `5` |
| `MTSP_MAX_SIZE` | `DIMENSION - 1` (derived if unset) |
| `MTSP_MIN_SIZE` | `1` |
| `NONSEQUENTIAL_MOVE_TYPE` | `MOVE_TYPE + PATCHING_A + PATCHING_C - 1` (derived) |
| `OPTIMUM` | `MINUS_INFINITY` |
| `PATCHING_A` | `1` |
| `PATCHING_C` | `0` |
| `POPMUSIC_INITIAL_TOUR` | `NO` |
| `POPMUSIC_MAX_NEIGHBORS` | `5` |
| `POPMUSIC_SAMPLE_SIZE` | `10` |
| `POPMUSIC_SOLUTIONS` | `50` |
| `POPMUSIC_TRIALS` | `1` |
| `POPULATION_SIZE` | `0` |
| `PRECISION` | `100` |
| `PROBABILITY` | `100` |
| `RECOMBINATION` | `IPT` |
| `RESTRICTED_SEARCH` | `YES` |
| `RUNS` | `10` |
| `SALESMEN` / `VEHICLES` | `1` |
| `SCALE` | `1` (effective) |
| `SEED` | `1` |
| `STOP_AT_OPTIMUM` | `YES` |
| `SUBGRADIENT` | `YES` |
| `SUBPROBLEM_SIZE` | `0` |
| `SUBSEQUENT_MOVE_TYPE` | `0` |
| `SUBSEQUENT_PATCHING` | `YES` |
| `TIME_LIMIT` | `DBL_MAX` |
| `TOTAL_TIME_LIMIT` | `DBL_MAX` |
| `TRACE_LEVEL` | `1` |

## Behavior Notes

- `INITIAL_TOUR_FILE`, `INPUT_TOUR_FILE`, and `SUBPROBLEM_TOUR_FILE` tour lists
  are expected by LKH to end with `-1`.
- In `OUTPUT_TOUR_FILE`, `TOUR_FILE`, `MTSP_SOLUTION_FILE`, and
  `SINTEF_SOLUTION_FILE`, `$` is replaced by tour cost by LKH.
- `SPECIAL` (standalone keyword) is an LKH macro for a predefined parameter
  bundle. This crate writes it as-is when `special == true`.

## Abbreviations

LKH accepts unambiguous abbreviations for many string values. This crate writes
canonical full tokens, but these are commonly seen in existing `.par` files:

| Canonical | Common abbreviation |
| --- | --- |
| `ALPHA` | `A` |
| `BORDERS` | `B` |
| `BORUVKA` | `B` |
| `CLARIST` | `C` |
| `COMPRESSED` | `C` |
| `CVRP` | `C` |
| `DELAUNAY` | `D` |
| `EXTENDED` | `E` |
| `GPX2` | `G` |
| `GREEDY` | `G` |
| `IPT` | `I` |
| `KARP` | `KA` |
| `K-CENTER` | `K-C` |
| `K-MEANS` | `K-M` |
| `MOORE` | `MO` |
| `MTSP` | `MT` |
| `NEAREST-NEIGHBOR` | `N` |
| `NO` | `N` |
| `POPMUSIC` | `P` |
| `PURE` | `P` |
| `QUADRANT` | `Q` |
| `QUICK-BORUVKA` | `Q` |
| `RESTRICTED` | `R` |
| `ROHE` | `R` |
| `SIERPINSKI` | `SI` |
| `SOP` | `SO` |
| `SPECIAL` | `S` |
| `SYMMETRIC` | `S` |
| `TSPDL` | `T` |
| `WALK` | `W` |
| `YES` | `Y` |

## Notes on Validation

- This model is intentionally permissive and primarily provides typed
  formatting and ergonomic construction.
- It does not enforce all LKH cross-field constraints or numeric bounds.
