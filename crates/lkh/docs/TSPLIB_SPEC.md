# TSPLIB Problem and Tour Specifications (as modeled by this crate)

This document describes the TSPLIB-related formats supported by:

- `lkh::problem::TsplibProblem` (`.tsp`/problem files)
- `lkh::tour::TsplibTour` (`.tour` files)

It focuses on what the crate currently reads/writes.

## `TsplibProblem` (`.tsp`) Format

### Header Style

`TsplibProblem` writes header lines using `KEY: VALUE`.

Always written:

- `NAME: ...`
- `TYPE: ...`

Optionally written when populated:

- `COMMENT: ...` (repeatable)
- `DIMENSION: ...`
- `CAPACITY: ...`
- `EDGE_WEIGHT_TYPE: ...`
- `EDGE_WEIGHT_FORMAT: ...`
- `EDGE_DATA_FORMAT: ...`
- `NODE_COORD_TYPE: ...`
- `DISPLAY_DATA_TYPE: ...`

### Supported Header Enums

#### `TYPE`

- `TSP`
- `ATSP`
- `SOP`
- `HCP`
- `CVRP`
- `TOUR`

#### `EDGE_WEIGHT_TYPE`

- `EXPLICIT`
- `EUC_2D`
- `EUC_3D`
- `MAX_2D`
- `MAX_3D`
- `MAN_2D`
- `MAN_3D`
- `CEIL_2D`
- `GEO`
- `ATT`
- `XRAY1`
- `XRAY2`
- `SPECIAL`

#### `EDGE_WEIGHT_FORMAT`

- `FUNCTION`
- `FULL_MATRIX`
- `UPPER_ROW`
- `LOWER_ROW`
- `UPPER_DIAG_ROW`
- `LOWER_DIAG_ROW`
- `UPPER_COL`
- `LOWER_COL`
- `UPPER_DIAG_COL`
- `LOWER_DIAG_COL`

#### `EDGE_DATA_FORMAT`

- `EDGE_LIST`
- `ADJ_LIST`

#### `NODE_COORD_TYPE`

- `TWOD_COORDS`
- `THREED_COORDS`
- `NO_COORDS`

#### `DISPLAY_DATA_TYPE`

- `COORD_DISPLAY`
- `TWOD_DISPLAY`
- `NO_DISPLAY`

### Supported Sections

Section emission is conditional: empty vectors are omitted.

- `NODE_COORD_SECTION`
  - rows: `<id> <x> <y> [z]`
- `DEPOT_SECTION`
  - one depot id per line
  - terminated with `-1`
- `DEMAND_SECTION`
  - rows: `<id> <demand>`
- `EDGE_DATA_SECTION`
  - for edge-list: rows `<from> <to>`, then terminating `-1`
  - for adjacency-list: rows `<node> <neighbor1> ... -1`
- `FIXED_EDGES_SECTION`
  - rows: `<from> <to>`
- `DISPLAY_DATA_SECTION`
  - rows: `<id> <x> <y>`
- `TOUR_SECTION`
  - one id per line
  - terminated with `-1`
- `EDGE_WEIGHT_SECTION`
  - one matrix row per line, space-separated

Optional final marker:

- `EOF` (controlled by `emit_eof`, default `true` in `TsplibProblem::new`).

### Convenience Constructor

`TsplibProblem::from_euc2d_points(...)` initializes:

- `TYPE = TSP`
- `EDGE_WEIGHT_TYPE = EUC_2D`
- `NODE_COORD_TYPE = TWOD_COORDS`
- `DIMENSION = points.len()`
- `NODE_COORD_SECTION` with 1-based node ids

## `TsplibTour` (`.tour`) Format

### Parsed Headers

The parser recognizes:

- `NAME: ...`
- `COMMENT: ...` (repeatable)
- `TYPE: TOUR`
- `DIMENSION: ...`
- `TOUR_SECTION`
- `EOF`

Unknown headers are ignored.

### Tour Section Rules

- `TOUR_SECTION` is required.
- Node ids are expected to be TSPLIB 1-based ids.
- `-1` terminates the section.
- `EOF` also terminates parsing.

### Permissive Parser Behavior

The parser is intentionally permissive:

- Non-positive node ids in `TOUR_SECTION` other than the `-1` terminator are skipped.
- If `DIMENSION` is present, it must match the parsed node count.
- If `DIMENSION` is absent, no count check is enforced.

### Conversion to Internal Ordering

`TsplibTour::zero_based_tour()` converts parsed ids from 1-based to 0-based:

- input id `1` -> output `0`
- input id `N` -> output `N-1`

## Practical Interop Notes

- `TsplibProblem` is write-oriented in this crate (no parser implemented here).
- `TsplibTour` is both readable and writable.
- For end-to-end solving, `LkhSolver` writes `.tsp` + `.par`, executes LKH,
  and reads back `.tour`.
