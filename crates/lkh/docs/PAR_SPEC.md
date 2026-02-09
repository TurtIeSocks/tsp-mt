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

Upstream LKH parsing semantics (manual behavior):

- Keywords are case-insensitive.
- Parameter order is arbitrary.

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

## Upstream Defaults (for modeled keys)

Unless explicitly set in your `.par`, LKH applies its own defaults. This crate
does not auto-fill these defaults; it only writes fields you set.

| Key                        | Upstream default                                       |
| -------------------------- | ------------------------------------------------------ |
| `PROBLEM_FILE`             | required                                               |
| `ASCENT_CANDIDATES`        | `50`                                                   |
| `BACKBONE_TRIALS`          | `0`                                                    |
| `BACKTRACKING`             | `NO`                                                   |
| `CANDIDATE_SET_TYPE`       | `ALPHA`                                                |
| `EXCESS`                   | `1.0 / DIMENSION`                                      |
| `EXTRA_CANDIDATES`         | `0`                                                    |
| `EXTRA_CANDIDATE_SET_TYPE` | `QUADRANT`                                             |
| `GAIN23`                   | `YES`                                                  |
| `GAIN_CRITERION`           | `YES`                                                  |
| `INITIAL_PERIOD`           | `max(DIMENSION / 2, 100)`                              |
| `INITIAL_STEP_SIZE`        | `1`                                                    |
| `INITIAL_TOUR_ALGORITHM`   | `WALK`                                                 |
| `INITIAL_TOUR_FRACTION`    | `1.0`                                                  |
| `KICKS`                    | `1`                                                    |
| `KICK_TYPE`                | `0`                                                    |
| `MAX_BREADTH`              | `INT_MAX`                                              |
| `MAX_CANDIDATES`           | `5`                                                    |
| `MAX_SWAPS`                | `DIMENSION`                                            |
| `MAX_TRIALS`               | `DIMENSION`                                            |
| `MOVE_TYPE`                | `5`                                                    |
| `NONSEQUENTIAL_MOVE_TYPE`  | derived from move/patching settings                    |
| `OPTIMUM`                  | very negative sentinel (manual notation: `-LLONG_MIN`) |
| `PATCHING_A`               | `1`                                                    |
| `PATCHING_C`               | `0`                                                    |
| `PRECISION`                | `100`                                                  |
| `RESTRICTED_SEARCH`        | `YES`                                                  |
| `RUNS`                     | `10`                                                   |
| `SEED`                     | `1`                                                    |
| `STOP_AT_OPTIMUM`          | `YES`                                                  |
| `SUBGRADIENT`              | `YES`                                                  |
| `SUBPROBLEM_SIZE`          | `0`                                                    |
| `SUBSEQUENT_MOVE_TYPE`     | `0`                                                    |
| `SUBSEQUENT_PATCHING`      | `YES`                                                  |
| `TIME_LIMIT`               | `DBL_MAX`                                              |
| `TRACE_LEVEL`              | `1`                                                    |
| `CANDIDATE_FILE`           | no default (repeatable)                                |
| `INITIAL_TOUR_FILE`        | no default                                             |
| `INPUT_TOUR_FILE`          | no default                                             |
| `MERGE_TOUR_FILE`          | no default (repeatable)                                |
| `OUTPUT_TOUR_FILE`         | no default                                             |
| `PI_FILE`                  | no default                                             |
| `SUBPROBLEM_TOUR_FILE`     | no default                                             |
| `TOUR_FILE`                | no default                                             |

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

## Behavioral Notes from the LKH Parameter Guide

- `INITIAL_TOUR_FILE` content is expected to end with `-1`.
- `INPUT_TOUR_FILE` can both constrain search and force alpha values to zero
  for listed tour edges.
- In `OUTPUT_TOUR_FILE`, `$` is replaced by the tour cost by LKH.

## Abbreviations Accepted by LKH

LKH accepts several shorthand tokens in parameter values. This crate writes
canonical full tokens, but these abbreviations are useful when reading legacy
`.par` files or manual examples.

| Canonical          | Abbrev |
| ------------------ | ------ |
| `ALPHA`            | `A`    |
| `BORDERS`          | `B`    |
| `BORUVKA`          | `B`    |
| `COMPRESSED`       | `C`    |
| `DELAUNAY`         | `D`    |
| `EXTENDED`         | `E`    |
| `GREEDY`           | `G`    |
| `KARP`             | `KA`   |
| `K-MEANS`          | `K`    |
| `NEAREST-NEIGHBOR` | `N`    |
| `NO`               | `N`    |
| `PURE`             | `P`    |
| `QUADRANT`         | `Q`    |
| `QUICK-BORUVKA`    | `Q`    |
| `RESTRICTED`       | `R`    |
| `ROHE`             | `R`    |
| `SIERPINSKI`       | `S`    |
| `SYMMETRIC`        | `S`    |
| `WALK`             | `W`    |
| `YES`              | `Y`    |

## Notes on Validation

- This model is intentionally permissive and mainly provides typed formatting.
- It does not enforce all semantic constraints from the LKH manual
  (for example cross-field dependencies and legal ranges for every key).
- Unsupported keys can still be represented externally by writing your own
  `.par` file if needed.
