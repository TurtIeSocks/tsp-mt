# LKH `.par` File Specification (as modeled by this crate)

This document describes the `.par` format supported by `lkh::parameters::LkhParameters`.
It is aligned with the LKH 2.0 parameter guide (`LKH-2.0_PARAMETERS.pdf`) and the
crate's current writer implementation.

## Line Syntax

- Assignment lines:
  - `KEY = VALUE`
- Comment lines (both supported):
  - `COMMENT text...`
  - `# text...`
- Optional terminator:
  - `EOF`

## Writer Ordering Rules

When serialized via `Display` / `write_to_file`:

1. `PROBLEM_FILE` is always written first.
2. Remaining assignment keys are emitted in alphabetical order.
3. Repeated keys (`CANDIDATE_FILE`, `MERGE_TOUR_FILE`) are emitted once per entry.
4. `COMMENT ...` / `# ...` lines are emitted where `comment_lines` appears.
5. `EOF` is emitted only when `emit_eof == true`.

## Supported Keys

The crate currently models and writes the following LKH keys:

- `PROBLEM_FILE` (required by constructor)
- `ASCENT_CANDIDATES`
- `BACKBONE_TRIALS`
- `BACKTRACKING`
- `CANDIDATE_FILE` (repeatable)
- `CANDIDATE_SET_TYPE`
- `EXCESS`
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
- `KICKS`
- `KICK_TYPE`
- `MAX_BREADTH`
- `MAX_CANDIDATES`
- `MAX_SWAPS`
- `MAX_TRIALS`
- `MERGE_TOUR_FILE` (repeatable)
- `MOVE_TYPE`
- `NONSEQUENTIAL_MOVE_TYPE`
- `OPTIMUM`
- `OUTPUT_TOUR_FILE`
- `PATCHING_A`
- `PATCHING_C`
- `PI_FILE`
- `PRECISION`
- `RESTRICTED_SEARCH`
- `RUNS`
- `SEED`
- `STOP_AT_OPTIMUM`
- `SUBGRADIENT`
- `SUBPROBLEM_SIZE`
- `SUBPROBLEM_TOUR_FILE`
- `SUBSEQUENT_MOVE_TYPE`
- `SUBSEQUENT_PATCHING`
- `TIME_LIMIT`
- `TOUR_FILE`
- `TRACE_LEVEL`
- `EOF` (optional trailing marker)

## Value Types

- Integer-like keys use `usize`/`u64` (`SEED`) or `i64` where applicable.
- Floating-point keys use `f64` (for example `EXCESS`, `OPTIMUM`, `TIME_LIMIT`).
- Path keys use `PathBuf`, serialized using path display output.
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
- `QUADRANT`

### `EXTRA_CANDIDATE_SET_TYPE`

- `NEAREST-NEIGHBOR`
- `QUADRANT`

### `INITIAL_TOUR_ALGORITHM`

- `BORUVKA`
- `GREEDY`
- `NEAREST-NEIGHBOR`
- `QUICK-BORUVKA`
- `SIERPINSKI`
- `WALK`

### `PATCHING_A` / `PATCHING_C` mode suffix

- `RESTRICTED`
- `EXTENDED`

### `SUBPROBLEM_SIZE` partitioning suffix

- `DELAUNAY`
- `KARP`
- `K-MEANS`
- `ROHE`
- `SIERPINSKI`

## Composite Value Shapes

### `MAX_CANDIDATES` / `EXTRA_CANDIDATES`

Modeled by `CandidateLimit`:

- `<N>`
- `<N> SYMMETRIC`

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

Examples:

- `SUBPROBLEM_SIZE = 1000`
- `SUBPROBLEM_SIZE = 1000 K-MEANS`
- `SUBPROBLEM_SIZE = 1000 DELAUNAY BORDERS COMPRESSED`

## Notes on Validation

- This model is intentionally permissive and mainly provides typed formatting.
- It does not enforce all semantic constraints from the LKH manual
  (for example cross-field dependencies and legal ranges for every key).
- Unsupported keys can still be represented externally by writing your own
  `.par` file if needed.
